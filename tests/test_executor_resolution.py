"""Unit tests for the unified executor resolution chain.

Covers:
- ``get_executor`` SAF lookup + static fallback.
- ``list_executors`` returns the registered set.
- Full chain layering (CLI > recipe > runtime > per-executor adjustments
  > SparkrunConfig > per-executor defaults).
- Backwards-compat: ``EXECUTOR_DEFAULTS`` re-export.  All
  ``executor_*.py`` legacy module paths have been removed; callers
  must import from ``orchestration.executors.*``.
- Plugin model: each concrete executor exposes ``executor_name``,
  ``default_config``, ``apply_runtime_adjustments``.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from sparkrun.orchestration.executor import (
    EXECUTOR_DEFAULTS,
    EXT_EXECUTOR,
    DOCKER_DEFAULTS,
    DockerExecutor,
    Executor,
    ExecutorConfig,
    ExecutorUnavailableError,
    get_executor,
    list_executors,
    resolve_executor,
)
from sparkrun.orchestration.executors.k8s import K8sExecutor
from sparkrun.orchestration.executors.local import LocalExecutor


# ---------------------------------------------------------------------------
# Plugin model
# ---------------------------------------------------------------------------


class TestExecutorPluginModel:
    """Each concrete executor must expose the plugin surface."""

    @pytest.mark.parametrize(
        "cls,expected_name",
        [(DockerExecutor, "docker"), (LocalExecutor, "local"), (K8sExecutor, "k8s")],
    )
    def test_executor_name(self, cls, expected_name):
        assert cls.executor_name == expected_name

    @pytest.mark.parametrize("cls", [DockerExecutor, LocalExecutor, K8sExecutor])
    def test_default_config_returns_dict(self, cls):
        cfg = cls.default_config()
        assert isinstance(cfg, dict)

    def test_docker_default_config_is_docker_defaults(self):
        # Returns a fresh copy (not the module-level dict).
        d = DockerExecutor.default_config()
        assert d == DOCKER_DEFAULTS
        assert d is not DOCKER_DEFAULTS

    def test_local_default_config_is_empty(self):
        assert LocalExecutor.default_config() == {}

    def test_k8s_default_config_is_empty(self):
        assert K8sExecutor.default_config() == {}

    @pytest.mark.parametrize("cls", [LocalExecutor, K8sExecutor])
    def test_non_docker_adjustments_are_empty(self, cls):
        assert cls.apply_runtime_adjustments(rootless=True, auto_user=True) == {}
        assert cls.apply_runtime_adjustments(rootless=False, auto_user=False) == {}

    def test_docker_adjustments_react_to_rootless(self):
        rootless_on = DockerExecutor.apply_runtime_adjustments(rootless=True, auto_user=False)
        rootless_off = DockerExecutor.apply_runtime_adjustments(rootless=False, auto_user=False)
        assert rootless_on.get("privileged") is False
        assert "security_opt" in rootless_on
        assert "ulimit" in rootless_on
        assert rootless_off == {}

    def test_docker_adjustments_react_to_auto_user(self):
        adj = DockerExecutor.apply_runtime_adjustments(rootless=False, auto_user=True)
        assert adj.get("user") == "$SHELL_USER"
        adj_off = DockerExecutor.apply_runtime_adjustments(rootless=False, auto_user=False)
        assert "user" not in adj_off

    def test_docker_adjustments_accept_unknown_kwargs(self):
        # Forward-compat: future signals must not break old executors.
        out = DockerExecutor.apply_runtime_adjustments(rootless=True, auto_user=True, cluster="x", future="y")
        assert "privileged" in out


# ---------------------------------------------------------------------------
# get_executor / list_executors
# ---------------------------------------------------------------------------


class TestExecutorLookup:
    def test_get_executor_static_fallback_docker(self):
        # Passing v=None forces the static fallback path.
        assert get_executor("docker", v=None) is DockerExecutor

    def test_get_executor_static_fallback_local(self):
        assert get_executor("local", v=None) is LocalExecutor

    def test_get_executor_static_fallback_k8s(self):
        assert get_executor("k8s", v=None) is K8sExecutor

    def test_get_executor_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown executor"):
            get_executor("wasm", v=None)

    def test_saf_discovery_registers_all(self):
        from sparkrun.core.bootstrap import init_sparkrun

        v = init_sparkrun()
        registered = list_executors(v)
        # We need at least the three built-in executors.
        assert "docker" in registered
        assert "local" in registered
        assert "k8s" in registered


# ---------------------------------------------------------------------------
# resolve_executor — public surface tests
# ---------------------------------------------------------------------------


class TestResolveExecutor:
    """Direct unit tests for the public ``resolve_executor`` API.

    The :class:`TestResolutionChain` and :class:`TestConfigChainOrdering`
    classes below also exercise ``resolve_executor`` but framed around
    chain-priority behaviour.  This class focuses on parameter
    semantics: every kwarg, return-type guarantees, fresh-instance
    behaviour, and the v= override.
    """

    def test_all_kwargs_optional(self):
        """Zero-arg call must succeed and return a DockerExecutor."""
        ex = resolve_executor()
        assert isinstance(ex, DockerExecutor)
        assert isinstance(ex.config, ExecutorConfig)

    def test_return_value_is_fresh_instance_each_call(self):
        """Each call constructs a new Executor — no shared mutable state."""
        a = resolve_executor()
        b = resolve_executor()
        assert a is not b
        assert a.config is not b.config

    def test_return_type_matches_selected_executor(self):
        from sparkrun.orchestration.executors.local import LocalExecutor as LE
        from sparkrun.orchestration.executors.k8s import K8sExecutor as KE

        assert isinstance(resolve_executor(recipe=_FakeRecipe(executor="docker")), DockerExecutor)
        assert isinstance(resolve_executor(recipe=_FakeRecipe(executor="local")), LE)
        assert isinstance(resolve_executor(recipe=_FakeRecipe(executor="k8s")), KE)

    def test_v_param_routes_through_saf_registry(self):
        """Passing v= uses the SAF plugin registry path explicitly."""
        from sparkrun.core.bootstrap import init_sparkrun

        v = init_sparkrun()
        ex = resolve_executor(recipe=_FakeRecipe(executor="local"), v=v)
        assert ex.config.executor_type == "local"

    def test_v_none_uses_singleton_or_static_fallback(self):
        """v=None must still resolve correctly via the static fallback."""
        ex = resolve_executor(recipe=_FakeRecipe(executor="k8s"), v=None)
        assert ex.config.executor_type == "k8s"

    def test_rootless_auto_user_defaults_apply_to_docker(self):
        """Default kwargs (rootless=True, auto_user=True) trigger Docker adjustments."""
        ex = resolve_executor()
        assert isinstance(ex, DockerExecutor)
        assert ex.config.privileged is False
        assert ex.config.user == "$SHELL_USER"

    def test_rootless_false_skips_security_adjustments(self):
        """Lifecycle paths use rootless=False — Docker config is neutral."""
        ex = resolve_executor(rootless=False, auto_user=False)
        assert isinstance(ex, DockerExecutor)
        assert ex.config.privileged is True
        assert ex.config.user is None
        assert ex.config.security_opt is None

    def test_cli_overrides_doubles_as_lifecycle_override_dict(self):
        """Lifecycle paths reuse cli_overrides to pass metadata-derived config.

        ``_stop_logs.py`` flattens job-metadata into a dict and passes
        it as ``cli_overrides``.  Verify the dict is honoured at the
        top of the chain.
        """
        ex = resolve_executor(
            cli_overrides={"executor": "local", "log_dir": "/var/log/x", "pid_dir": "/run/x"},
            rootless=False,
            auto_user=False,
        )
        assert ex.config.executor_type == "local"
        assert ex.config.log_dir == "/var/log/x"
        assert ex.config.pid_dir == "/run/x"

    def test_none_inputs_are_safe(self):
        """Every param accepts None without raising."""
        ex = resolve_executor(recipe=None, runtime=None, config=None, cli_overrides=None, v=None)
        assert isinstance(ex, DockerExecutor)

    def test_kwarg_only_signature(self):
        """All resolve_executor arguments must be keyword-only.

        Protects against accidental positional-call drift in callers.
        """
        with pytest.raises(TypeError):
            resolve_executor(_FakeRecipe(executor="local"))  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Resolution chain layering
# ---------------------------------------------------------------------------


class _FakeRecipe:
    def __init__(self, executor: str = "", executor_config: dict | None = None, builder: str = ""):
        self.executor = executor
        self.executor_config = dict(executor_config or {})
        self.builder = builder


class _FakeRuntime:
    def __init__(self, default_executor: str | None = None, default_executor_config: dict | None = None):
        self._default = default_executor
        self._default_config = dict(default_executor_config or {})

    def default_executor(self):
        return self._default

    def default_executor_config(self):
        return dict(self._default_config)


class _FakeConfig:
    def __init__(self, default_executor: str | None = None, executor_config: dict | None = None):
        self.default_executor = default_executor
        self.executor_config = dict(executor_config or {})


class TestResolutionChain:
    """Verify CLI > recipe > runtime > config > docker layering."""

    def test_baseline_no_inputs_returns_docker(self):
        ex = resolve_executor()
        assert isinstance(ex, DockerExecutor)

    def test_runtime_default_only(self):
        ex = resolve_executor(runtime=_FakeRuntime(default_executor="local"))
        assert isinstance(ex, LocalExecutor)

    def test_config_default_only(self):
        ex = resolve_executor(config=_FakeConfig(default_executor="k8s"))
        assert isinstance(ex, K8sExecutor)

    def test_recipe_overrides_runtime(self):
        ex = resolve_executor(
            recipe=_FakeRecipe(executor="k8s"),
            runtime=_FakeRuntime(default_executor="local"),
        )
        assert isinstance(ex, K8sExecutor)

    def test_recipe_overrides_config(self):
        ex = resolve_executor(
            recipe=_FakeRecipe(executor="k8s"),
            config=_FakeConfig(default_executor="local"),
        )
        assert isinstance(ex, K8sExecutor)

    def test_runtime_overrides_config(self):
        ex = resolve_executor(
            runtime=_FakeRuntime(default_executor="k8s"),
            config=_FakeConfig(default_executor="local"),
        )
        assert isinstance(ex, K8sExecutor)

    def test_cli_overrides_all(self):
        ex = resolve_executor(
            cli_overrides={"executor": "docker"},
            recipe=_FakeRecipe(executor="k8s"),
            runtime=_FakeRuntime(default_executor="local"),
            config=_FakeConfig(default_executor="k8s"),
        )
        assert isinstance(ex, DockerExecutor)

    def test_unknown_explicit_request_raises(self):
        # An explicitly-requested but unknown executor must fail loudly
        # rather than silently downgrade to docker.
        with pytest.raises(ExecutorUnavailableError, match="Unknown executor"):
            resolve_executor(cli_overrides={"executor": "wasm"})


class TestDefaultExecutorFallback:
    """B5: the baseline default (no layer names an executor) honors a disabled
    docker instead of always returning it."""

    def test_docker_enabled_returns_docker(self):
        from sparkrun.orchestration.executor import _default_executor_name

        assert _default_executor_name({"docker", "local"}, None, None) == "docker"

    def test_docker_disabled_single_alternative_is_picked(self):
        from sparkrun.orchestration.executor import _default_executor_name

        # A host that disabled docker and enabled exactly one alternative gets it.
        assert _default_executor_name({"local"}, None, None) == "local"

    def test_docker_disabled_ambiguous_raises(self):
        from sparkrun.orchestration.executor import _default_executor_name

        with pytest.raises(ExecutorUnavailableError, match="disabled and no executor is named"):
            _default_executor_name({"local", "k8s"}, None, None)

    def test_resolve_name_falls_back_to_sole_enabled_when_docker_off(self, monkeypatch):
        import sparkrun.orchestration.executor as ex_mod

        # SAF reports only local enabled (docker gated off); no layer names one.
        monkeypatch.setattr(ex_mod, "_known_executor_names", lambda v=None: {"local"})
        assert ex_mod.resolve_executor_name() == "local"

    def test_resolve_name_still_docker_when_enabled(self, monkeypatch):
        import sparkrun.orchestration.executor as ex_mod

        monkeypatch.setattr(ex_mod, "_known_executor_names", lambda v=None: {"docker", "local"})
        assert ex_mod.resolve_executor_name() == "docker"


class TestDockerAdjustmentsApplyOnlyToDocker:
    """rootless/auto_user must not affect non-Docker configs."""

    def test_docker_path_picks_up_rootless(self):
        ex = resolve_executor(rootless=True, auto_user=True)
        assert isinstance(ex, DockerExecutor)
        assert ex.config.privileged is False
        assert ex.config.user == "$SHELL_USER"
        assert ex.config.security_opt == ["no-new-privileges"]

    def test_docker_default_config_propagates(self):
        # DockerExecutor's default_config provides shm_size=32gb; appears
        # only when DockerExecutor is selected.
        ex = resolve_executor(rootless=False, auto_user=False)
        assert ex.config.shm_size == "32gb"
        assert ex.config.ipc == "shareable"
        assert ex.config.network == "host"

    def test_default_ipc_survives_both_privilege_modes(self):
        # `ipc` is orthogonal to privilege: it is not part of the rootless
        # adjustment layer, so --rootful does not restore the host namespace.
        # Keeping them coupled would make `shm_size` inert in one mode only
        # (Docker ignores --shm-size under --ipc=host).
        assert resolve_executor(rootless=True, auto_user=True).config.ipc == "shareable"
        assert resolve_executor(rootless=False, auto_user=False).config.ipc == "shareable"

    def test_recipe_can_still_pin_host_ipc(self):
        ex = resolve_executor(recipe=_FakeRecipe(executor_config={"ipc": "host"}), rootless=True, auto_user=True)
        assert ex.config.ipc == "host"
        assert "--ipc=host" in ex.run_cmd("img:latest")

    def test_local_path_does_not_pick_up_docker_adjustments(self):
        ex = resolve_executor(
            recipe=_FakeRecipe(executor="local"),
            rootless=True,
            auto_user=True,
        )
        assert isinstance(ex, LocalExecutor)
        # Local doesn't ship security_opt / ulimit / shm_size adjustments.
        assert ex.config.security_opt is None
        assert ex.config.ulimit is None
        assert ex.config.user is None
        # Dataclass default for shm_size (25gb) — DockerExecutor's 32gb
        # does NOT leak in.
        assert ex.config.shm_size == "25gb"

    def test_k8s_path_does_not_pick_up_docker_adjustments(self):
        ex = resolve_executor(
            recipe=_FakeRecipe(executor="k8s"),
            rootless=True,
            auto_user=True,
        )
        assert isinstance(ex, K8sExecutor)
        assert ex.config.security_opt is None
        assert ex.config.user is None


class TestConfigChainOrdering:
    """Lower-priority layers must NOT clobber higher-priority values."""

    def test_recipe_executor_config_overrides_docker_default(self):
        ex = resolve_executor(
            recipe=_FakeRecipe(executor="docker", executor_config={"shm_size": "1gb"}),
            rootless=False,
            auto_user=False,
        )
        assert ex.config.shm_size == "1gb"

    def test_cli_executor_config_overrides_recipe(self):
        ex = resolve_executor(
            cli_overrides={"shm_size": "999gb"},
            recipe=_FakeRecipe(executor="docker", executor_config={"shm_size": "1gb"}),
            rootless=False,
            auto_user=False,
        )
        assert ex.config.shm_size == "999gb"

    def test_sparkrunconfig_executor_config_below_recipe(self):
        ex = resolve_executor(
            recipe=_FakeRecipe(executor_config={"k8s_namespace": "recipe-ns"}),
            config=_FakeConfig(
                default_executor="k8s",
                executor_config={"k8s_namespace": "global-ns"},
            ),
        )
        assert isinstance(ex, K8sExecutor)
        # Recipe wins over SparkrunConfig.
        assert ex.config.k8s_namespace == "recipe-ns"

    def test_sparkrunconfig_fills_when_recipe_silent(self):
        ex = resolve_executor(
            config=_FakeConfig(
                default_executor="k8s",
                executor_config={"k8s_namespace": "global-ns"},
            ),
        )
        assert isinstance(ex, K8sExecutor)
        assert ex.config.k8s_namespace == "global-ns"

    def test_runtime_executor_config_default_fills_entrypoint(self):
        ex = resolve_executor(
            runtime=_FakeRuntime(default_executor_config={"entrypoint": ""}),
            rootless=False,
            auto_user=False,
        )
        assert ex.config.entrypoint == ""

    def test_recipe_entrypoint_overrides_runtime_default(self):
        ex = resolve_executor(
            recipe=_FakeRecipe(executor_config={"entrypoint": "bash"}),
            runtime=_FakeRuntime(default_executor_config={"entrypoint": ""}),
            rootless=False,
            auto_user=False,
        )
        assert ex.config.entrypoint == "bash"

    def test_recipe_entrypoint_null_clears_runtime_default(self):
        ex = resolve_executor(
            recipe=_FakeRecipe(executor_config={"entrypoint": None}),
            runtime=_FakeRuntime(default_executor_config={"entrypoint": ""}),
            rootless=False,
            auto_user=False,
        )
        assert ex.config.entrypoint is None


# ---------------------------------------------------------------------------
# Backwards compat (only the bits we keep)
# ---------------------------------------------------------------------------


class TestBackwardsCompat:
    def test_executor_defaults_is_docker_defaults(self):
        # Legacy callers import EXECUTOR_DEFAULTS expecting Docker
        # semantics.  Kept as a re-export of ``DOCKER_DEFAULTS`` (which
        # in turn comes from ``DockerExecutor.default_config()``).
        assert EXECUTOR_DEFAULTS is DOCKER_DEFAULTS


class TestClusterLayer:
    """Cluster row in the resolution chain — Task 4.

    The cluster row sits between recipe (workload-specific) and the
    runtime/SparkrunConfig generic defaults, in both name-selection
    and config chains.  Verified precedence:

      CLI > recipe > **cluster** > runtime.default_executor() > SparkrunConfig > library defaults
    """

    @staticmethod
    def _cluster(executor=None, executor_config=None):
        from sparkrun.core.cluster_manager import ClusterDefinition

        return ClusterDefinition(
            name="test-cluster",
            hosts=["h1"],
            executor=executor,
            executor_config=executor_config,
        )

    # ---- name selection ----

    def test_cluster_executor_wins_over_runtime_default(self):
        ex = resolve_executor(
            cluster=self._cluster(executor="local"),
            runtime=_FakeRuntime(default_executor="k8s"),
        )
        assert isinstance(ex, LocalExecutor)

    def test_recipe_executor_wins_over_cluster(self):
        ex = resolve_executor(
            recipe=_FakeRecipe(executor="k8s"),
            cluster=self._cluster(executor="docker"),
            runtime=_FakeRuntime(default_executor="local"),
        )
        assert isinstance(ex, K8sExecutor)

    def test_cli_wins_over_recipe_and_cluster(self):
        ex = resolve_executor(
            cli_overrides={"executor": "local"},
            recipe=_FakeRecipe(executor="k8s"),
            cluster=self._cluster(executor="docker"),
            runtime=_FakeRuntime(default_executor="k8s"),
        )
        assert isinstance(ex, LocalExecutor)

    def test_cluster_wins_over_sparkrun_config(self):
        ex = resolve_executor(
            cluster=self._cluster(executor="local"),
            config=_FakeConfig(default_executor="docker"),
        )
        assert isinstance(ex, LocalExecutor)

    def test_cluster_config_only_no_selector_falls_through(self):
        """A cluster with executor_config but no executor selector
        contributes config but does not name an executor — name falls
        through to the next layer."""
        ex = resolve_executor(
            cluster=self._cluster(executor_config={"shm_size": "16g"}),
            runtime=_FakeRuntime(default_executor="local"),
        )
        assert isinstance(ex, LocalExecutor)

    def test_cluster_executor_config_pid_dir_reaches_local(self):
        """A cluster's ``executor_config.pid_dir`` flows into the local
        executor config.  This is the mechanism the unified status path relies
        on (``api.status`` resolves the local executor cluster-aware), which
        replaced the manual ``resolve_local_pid_dir`` threading."""
        ex = resolve_executor(
            cli_overrides={"executor": "local"},
            cluster=self._cluster(executor_config={"pid_dir": "/var/run/sparkrun/pids"}),
        )
        assert isinstance(ex, LocalExecutor)
        assert ex.config.pid_dir == "/var/run/sparkrun/pids"

    def test_cluster_none_preserves_pre_task4_behavior(self):
        """cluster=None must reproduce the pre-Task-4 chain exactly."""
        ex = resolve_executor(
            recipe=_FakeRecipe(executor="local"),
            cluster=None,
            runtime=_FakeRuntime(default_executor="k8s"),
        )
        assert isinstance(ex, LocalExecutor)

    def test_cluster_unknown_executor_raises(self):
        with pytest.raises(ExecutorUnavailableError, match="Unknown executor"):
            resolve_executor(cluster=self._cluster(executor="some-future-thing"))

    # ---- config chain ----

    def test_cluster_executor_config_provides_baseline(self):
        ex = resolve_executor(
            cluster=self._cluster(executor="docker", executor_config={"shm_size": "16g"}),
            rootless=False,
            auto_user=False,
        )
        assert ex.config.shm_size == "16g"

    def test_recipe_config_overrides_cluster_config(self):
        ex = resolve_executor(
            recipe=_FakeRecipe(executor="docker", executor_config={"shm_size": "64g"}),
            cluster=self._cluster(executor="docker", executor_config={"shm_size": "16g"}),
            rootless=False,
            auto_user=False,
        )
        assert ex.config.shm_size == "64g"

    def test_cli_overrides_cluster_config(self):
        ex = resolve_executor(
            cli_overrides={"shm_size": "256g"},
            cluster=self._cluster(executor="docker", executor_config={"shm_size": "16g"}),
            rootless=False,
            auto_user=False,
        )
        assert ex.config.shm_size == "256g"

    def test_cluster_config_overrides_sparkrun_config(self):
        ex = resolve_executor(
            cluster=self._cluster(executor="docker", executor_config={"shm_size": "16g"}),
            config=_FakeConfig(
                default_executor="docker",
                executor_config={"shm_size": "4g"},
            ),
            rootless=False,
            auto_user=False,
        )
        assert ex.config.shm_size == "16g"

    def test_cluster_config_merges_independent_fields(self):
        """Each layer contributes its own fields; the merge is by key."""
        ex = resolve_executor(
            recipe=_FakeRecipe(executor="docker", executor_config={"shm_size": "64g"}),
            cluster=self._cluster(executor="docker", executor_config={"memory_limit": "100g"}),
            rootless=False,
            auto_user=False,
        )
        # shm_size from recipe; memory_limit from cluster.
        assert ex.config.shm_size == "64g"
        assert ex.config.memory_limit == "100g"

    def test_cluster_executor_config_not_mutated(self):
        """Resolution does not mutate the cluster's input dict."""
        original = {"shm_size": "16g"}
        cluster = self._cluster(executor="docker", executor_config=original)
        resolve_executor(cluster=cluster, rootless=False, auto_user=False)
        assert original == {"shm_size": "16g"}

    def test_realistic_full_chain(self):
        """Realistic scenario: cluster declares k8s + privileged=False,
        recipe overrides shm_size, CLI tightens privileged back on."""
        ex = resolve_executor(
            cli_overrides={"privileged": True},
            recipe=_FakeRecipe(executor_config={"shm_size": "100g"}),
            cluster=self._cluster(executor="k8s", executor_config={"privileged": False, "shm_size": "16g"}),
            runtime=_FakeRuntime(default_executor="docker"),
        )
        # Name: cluster wins (recipe doesn't pin an executor).
        assert isinstance(ex, K8sExecutor)
        # Config: CLI > recipe for shm_size; CLI overrides cluster's privileged.
        assert ex.config.privileged is True
        assert ex.config.shm_size == "100g"


class TestPlatformLayer:
    """``platform.default_executor_config(name)`` sits above the executor's own
    defaults and below every user-facing layer.  Threaded via ``host_hardware``.
    """

    @staticmethod
    def _gb10():
        from sparkrun.core.hardware import default_dgx_spark_hardware

        return default_dgx_spark_hardware()

    @staticmethod
    def _h100():
        from sparkrun.core.hardware import AcceleratorSpec, HostHardware

        return HostHardware(accelerators=[AcceleratorSpec(vendor="nvidia", model="h100", memory_gb=80.0)])

    def test_no_hardware_keeps_executor_default(self):
        """Naming / teardown paths pass no hardware — the layer drops out."""
        ex = resolve_executor(rootless=False, auto_user=False)
        assert ex.config.gpu_access_mode == "cdi"

    def test_dgx_spark_pins_gpus_mode(self):
        ex = resolve_executor(host_hardware=self._gb10(), rootless=False, auto_user=False)
        assert ex.config.gpu_access_mode == "gpus"

    def test_generic_nvidia_inherits_cdi(self):
        """Only DGX Spark opts out of CDI; other NVIDIA hosts keep the default."""
        ex = resolve_executor(host_hardware=self._h100(), rootless=False, auto_user=False)
        assert ex.config.gpu_access_mode == "cdi"

    def test_recipe_overrides_platform(self):
        ex = resolve_executor(
            recipe=_FakeRecipe(executor_config={"gpu_access_mode": "cdi"}),
            host_hardware=self._gb10(),
            rootless=False,
            auto_user=False,
        )
        assert ex.config.gpu_access_mode == "cdi"

    def test_config_overrides_platform(self):
        class _Cfg:
            default_executor = None
            executor_config = {"gpu_access_mode": "cdi"}

        ex = resolve_executor(config=_Cfg(), host_hardware=self._gb10(), rootless=False, auto_user=False)
        assert ex.config.gpu_access_mode == "cdi"

    def test_platform_layer_is_executor_scoped(self):
        """DGX Spark publishes docker defaults only — a different executor is untouched."""
        ex = resolve_executor(
            recipe=_FakeRecipe(executor="local"),
            host_hardware=self._gb10(),
            rootless=False,
            auto_user=False,
        )
        assert ex.config.gpu_access_mode == "cdi"

    def test_platform_error_is_swallowed(self, monkeypatch):
        import sparkrun.platforms as platforms

        def _raise(hw):
            raise ValueError("boom")

        monkeypatch.setattr(platforms, "resolve_platform", _raise)
        ex = resolve_executor(host_hardware=self._gb10(), rootless=False, auto_user=False)
        assert ex.config.gpu_access_mode == "cdi"

    def test_unclaimed_hardware_contributes_nothing(self):
        """A host no platform claims (AMD today) falls through to the executor default."""
        from sparkrun.core.hardware import AcceleratorSpec, HostHardware

        hw = HostHardware(accelerators=[AcceleratorSpec(vendor="amd", model="mi300x")])
        ex = resolve_executor(host_hardware=hw, rootless=False, auto_user=False)
        assert ex.config.gpu_access_mode == "cdi"


class TestBuilderEnvFileLayer:
    """A builder that produces a shell env_file auto-populates the executor's
    ``env_file`` — below an explicit recipe/CLI ``executor_config.env_file``
    but above runtime/config/dataclass defaults.
    """

    class _StubBuilder:
        def __init__(self, env_file):
            self._env_file = env_file

        def default_env_file(self, recipe):
            return self._env_file

    def _patch_builder(self, monkeypatch, env_file):
        stub = self._StubBuilder(env_file)
        import sparkrun.core.bootstrap as bootstrap

        monkeypatch.setattr(bootstrap, "get_builder", lambda name, v=None: stub)
        return stub

    def test_builder_env_file_populates_when_recipe_silent(self, monkeypatch):
        self._patch_builder(monkeypatch, "/x/env.sh")
        ex = resolve_executor(
            recipe=_FakeRecipe(executor="local", builder="venv"),
            rootless=False,
            auto_user=False,
        )
        assert ex.config.env_file == "/x/env.sh"

    def test_recipe_executor_config_env_file_wins(self, monkeypatch):
        self._patch_builder(monkeypatch, "/x/env.sh")
        ex = resolve_executor(
            recipe=_FakeRecipe(
                executor="local",
                executor_config={"env_file": "/explicit.sh"},
                builder="venv",
            ),
            rootless=False,
            auto_user=False,
        )
        assert ex.config.env_file == "/explicit.sh"

    def test_no_builder_contributes_nothing(self, monkeypatch):
        # get_builder must not even be consulted when recipe.builder is empty.
        import sparkrun.core.bootstrap as bootstrap

        def _boom(name, v=None):
            raise AssertionError("get_builder should not be called")

        monkeypatch.setattr(bootstrap, "get_builder", _boom)
        ex = resolve_executor(recipe=_FakeRecipe(executor="local"), rootless=False, auto_user=False)
        assert ex.config.env_file is None

    def test_builder_error_is_swallowed(self, monkeypatch):
        import sparkrun.core.bootstrap as bootstrap

        def _raise(name, v=None):
            raise ValueError("unknown builder")

        monkeypatch.setattr(bootstrap, "get_builder", _raise)
        ex = resolve_executor(recipe=_FakeRecipe(executor="local", builder="nope"), rootless=False, auto_user=False)
        assert ex.config.env_file is None

    def test_empty_env_file_contributes_nothing(self, monkeypatch):
        self._patch_builder(monkeypatch, "")
        ex = resolve_executor(recipe=_FakeRecipe(executor="local", builder="venv"), rootless=False, auto_user=False)
        assert ex.config.env_file is None

    def test_builder_env_file_beats_cluster_env_file(self, monkeypatch):
        """An environment builder's essential env_file (e.g. venv activation)
        must outrank a cluster's generic executor_config.env_file — otherwise
        the native serve command runs under the wrong interpreter."""
        from sparkrun.core.cluster_manager import ClusterDefinition

        self._patch_builder(monkeypatch, "/venv/activate.sh")
        ex = resolve_executor(
            recipe=_FakeRecipe(executor="local", builder="venv"),
            cluster=ClusterDefinition(
                name="c",
                hosts=["h1"],
                executor_config={"env_file": "/cluster-generic.sh"},
            ),
            rootless=False,
            auto_user=False,
        )
        assert ex.config.env_file == "/venv/activate.sh"


class TestRemovedShims:
    """The legacy ``orchestration.executor_*`` module paths are gone.

    Callers must import from ``orchestration.executors.<name>`` (the
    package) or from the public facade ``orchestration.executor`` (the
    module).  These tests guard against accidental reintroduction.
    """

    def test_executor_docker_module_removed(self):
        with pytest.raises(ModuleNotFoundError):
            __import__("sparkrun.orchestration.executor_docker")

    def test_executor_local_module_removed(self):
        with pytest.raises(ModuleNotFoundError):
            __import__("sparkrun.orchestration.executor_local")

    def test_executor_k8s_module_removed(self):
        with pytest.raises(ModuleNotFoundError):
            __import__("sparkrun.orchestration.executor_k8s")


# ---------------------------------------------------------------------------
# ExecutorConfig.from_chain: dataclass-default preservation
# ---------------------------------------------------------------------------


class TestFromChainDefaultPreservation:
    """Bare chains must preserve dataclass field defaults.

    Per-executor defaults (e.g. ``DOCKER_DEFAULTS``) are applied by
    ``resolve_executor`` as the bottom chain layer — NOT inside
    ``from_chain`` itself.  This keeps ``ExecutorConfig`` agnostic of
    which executor will receive it.
    """

    def test_empty_chain_keeps_dataclass_defaults(self):
        cfg = ExecutorConfig.from_chain({})
        assert cfg.auto_remove is True
        assert cfg.privileged is True  # dataclass default
        assert cfg.gpus == "all"
        assert cfg.executor_type == "docker"

    def test_chain_overrides_only_present_fields(self):
        cfg = ExecutorConfig.from_chain({"auto_remove": False, "gpus": "0"})
        assert cfg.auto_remove is False
        assert cfg.gpus == "0"
        # Untouched fields keep dataclass defaults.
        assert cfg.privileged is True
        assert cfg.ipc == "shareable"


# ---------------------------------------------------------------------------
# resolve_executor robustness against mocked / partial objects
# ---------------------------------------------------------------------------


class TestResolveExecutorRobustness:
    """Lifecycle and test paths often pass MagicMock — chain must cope."""

    def test_mock_runtime_default_executor_ignored(self):
        from unittest.mock import MagicMock

        rt = MagicMock()
        # Default behaviour: rt.default_executor() returns MagicMock —
        # non-string return value should be treated as "no opinion".
        ex = resolve_executor(runtime=rt)
        assert isinstance(ex, DockerExecutor)

    def test_mock_config_executor_attrs_ignored(self):
        from unittest.mock import MagicMock

        cfg = MagicMock()
        ex = resolve_executor(config=cfg)
        assert isinstance(ex, DockerExecutor)

    def test_recipe_with_no_executor_field_defaults(self):
        from unittest.mock import MagicMock

        recipe = MagicMock()
        recipe.executor = ""
        recipe.executor_config = {}
        ex = resolve_executor(recipe=recipe)
        assert isinstance(ex, DockerExecutor)


# ---------------------------------------------------------------------------
# Golden equivalence: default Docker config matches pre-refactor.
# ---------------------------------------------------------------------------


class TestGoldenEquivalence:
    """Critical: default Docker launch path must produce identical config.

    The snapshot ``/tmp/exec-golden-pre.json`` was captured before the
    refactor.  We re-run the *default Docker* case (which is the
    regression-risk path) and assert byte-identical output.  The Local/
    K8s cases in the snapshot used the pre-refactor (broken) behaviour
    of leaking Docker adjustments into non-Docker configs and are not
    re-asserted here.

    One field has deliberately diverged from that snapshot since: ``ipc``
    moved from ``host`` to ``shareable`` (issue #285 — a host IPC namespace
    lets systemd-logind's ``RemoveIPC`` reap the workload's semaphores once
    the launching SSH session closes).  ``shm_size`` is unchanged but only
    takes effect now, since Docker ignores ``--shm-size`` under ``ipc=host``.
    """

    def test_default_docker_byte_identical(self):
        # The launcher passes rootless=True, auto_user=True for Docker.
        ex = resolve_executor(rootless=True, auto_user=True)
        cfg = asdict(ex.config)
        # Spot-check the canonical fields from the snapshot.
        expected = {
            "auto_remove": True,
            "privileged": False,  # rootless adjustment
            "gpus": "all",
            "ipc": "shareable",
            "shm_size": "32gb",
            "network": "host",
            "user": "$SHELL_USER",
            "security_opt": ["no-new-privileges"],
            "cap_add": None,  # rootless sets cap_add=[] → coerced to None
            "ulimit": ["memlock=-1:-1", "stack=67108864", "nofile=65535:65535"],
            "devices": ["/dev/infiniband"],
            "executor_type": "docker",
        }
        for key, want in expected.items():
            assert cfg[key] == want, "Mismatch on %s: got %r, want %r" % (key, cfg[key], want)


# ---------------------------------------------------------------------------
# Extension-point identity
# ---------------------------------------------------------------------------


class TestExtensionPointIdentity:
    """The EXT_EXECUTOR extension point must match what bootstrap uses."""

    def test_ext_constant(self):
        assert EXT_EXECUTOR == "sparkrun.executor"

    def test_each_executor_advertises_ext(self):
        for cls in (DockerExecutor, LocalExecutor, K8sExecutor):
            instance = cls()
            assert instance.extension_point_name(None) == EXT_EXECUTOR
            # name() is namespaced by executor_name.
            assert instance.name() == "sparkrun.executor.%s" % cls.executor_name

    def test_executors_are_plugin_subclasses(self):
        for cls in (DockerExecutor, LocalExecutor, K8sExecutor):
            assert issubclass(cls, Executor)
