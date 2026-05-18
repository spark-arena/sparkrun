"""Unit tests for the experimental K8sExecutor draft + executor chain layering.

Covers:
- ``ExecutorConfig`` accepts ``executor: k8s`` and the new k8s_* fields.
- ``build_executor()`` factory dispatches to :class:`K8sExecutor`.
- ``K8sExecutor`` generates well-formed ``kubectl`` command strings and
  bash-syntax-valid scripts.
- Ray script methods raise ``NotImplementedError`` (draft does not
  implement Ray).
- The full executor chain layers CLI > recipe > runtime > global
  defaults correctly.
"""

from __future__ import annotations

import subprocess

import pytest

from scitrera_app_framework.api import EnvPlacement, Variables

from sparkrun.orchestration.executor import (
    EXECUTOR_DEFAULTS,
    ExecutorConfig,
    build_executor,
)
from sparkrun.orchestration.executor_docker import DockerExecutor
from sparkrun.orchestration.executor_k8s import K8sExecutor
from sparkrun.orchestration.executor_local import LocalExecutor


# ---------------------------------------------------------------------------
# ExecutorConfig: k8s field handling
# ---------------------------------------------------------------------------


class TestExecutorConfigK8sFields:
    def test_k8s_selector_via_executor_key(self):
        cfg = ExecutorConfig.from_chain({"executor": "k8s"})
        assert cfg.executor_type == "k8s"

    def test_k8s_fields_round_trip(self):
        chain = {
            "executor": "k8s",
            "k8s_namespace": "inference",
            "k8s_context": "prod-cluster",
            "k8s_node_selector": "gpu=true,zone=us-east",
            "k8s_image_pull_policy": "Always",
            "kubeconfig": "/etc/sparkrun/kubeconfig",
        }
        cfg = ExecutorConfig.from_chain(chain)
        assert cfg.executor_type == "k8s"
        assert cfg.k8s_namespace == "inference"
        assert cfg.k8s_context == "prod-cluster"
        assert cfg.k8s_node_selector == "gpu=true,zone=us-east"
        assert cfg.k8s_image_pull_policy == "Always"
        assert cfg.kubeconfig == "/etc/sparkrun/kubeconfig"

    def test_k8s_fields_default_to_none(self):
        cfg = ExecutorConfig()
        assert cfg.k8s_namespace is None
        assert cfg.k8s_context is None
        assert cfg.k8s_node_selector is None
        assert cfg.k8s_image_pull_policy is None
        assert cfg.kubeconfig is None


# ---------------------------------------------------------------------------
# build_executor factory: k8s dispatch
# ---------------------------------------------------------------------------


class TestBuildExecutorK8s:
    def test_factory_returns_k8s_executor(self):
        ex = build_executor("k8s", {"k8s_namespace": "inf"})
        assert isinstance(ex, K8sExecutor)
        assert ex.config.k8s_namespace == "inf"

    def test_factory_selector_in_cfg(self):
        ex = build_executor(None, {"executor": "k8s"})
        assert isinstance(ex, K8sExecutor)


# ---------------------------------------------------------------------------
# K8sExecutor: command generators
# ---------------------------------------------------------------------------


def _k8s(**cfg_kwargs) -> K8sExecutor:
    cfg_kwargs.setdefault("executor_type", "k8s")
    return K8sExecutor(ExecutorConfig(**cfg_kwargs))


class TestK8sExecutorBasics:
    def test_inspect_exists_and_pull_are_noops(self):
        e = _k8s()
        assert e.inspect_exists_cmd("img:tag") == "true"
        assert e.pull_cmd("img:tag") == "true"

    def test_run_cmd_requires_container_name(self):
        with pytest.raises(ValueError, match="container_name"):
            _k8s().run_cmd(image="img", command="cmd", container_name=None)

    def test_run_cmd_requires_image(self):
        with pytest.raises(ValueError, match="image"):
            _k8s().run_cmd(image="", command="cmd", container_name="foo")

    def test_run_cmd_basic_shape(self):
        cmd = _k8s().run_cmd(image="img:tag", command="echo hi", container_name="pod1")
        assert cmd.startswith("kubectl ")
        assert " run " in cmd
        assert "pod1" in cmd
        assert "--image=img:tag" in cmd
        assert "--restart=Never" in cmd
        assert "bash -c" in cmd

    def test_run_cmd_includes_namespace_context_kubeconfig(self):
        cmd = _k8s(
            k8s_namespace="inf",
            k8s_context="prod",
            kubeconfig="/etc/k.cfg",
        ).run_cmd(image="img", command="echo", container_name="pod1")
        assert "--kubeconfig /etc/k.cfg" in cmd
        assert "--context prod" in cmd
        assert "-n inf" in cmd

    def test_run_cmd_gpus_device_maps_to_count(self):
        cmd = _k8s(gpus="device=0,2,3").run_cmd(image="img", command="echo", container_name="pod1")
        assert "--limits=nvidia.com/gpu=3" in cmd

    def test_run_cmd_gpus_all_maps_to_one(self):
        cmd = _k8s(gpus="all").run_cmd(image="img", command="echo", container_name="pod1")
        assert "--limits=nvidia.com/gpu=1" in cmd

    def test_run_cmd_gpus_none_omits_limit(self):
        cmd = _k8s(gpus="none").run_cmd(image="img", command="echo", container_name="pod1")
        assert "nvidia.com/gpu" not in cmd

    def test_run_cmd_passes_env_vars(self):
        cmd = _k8s().run_cmd(
            image="img",
            command="echo",
            container_name="pod1",
            env={"HF_TOKEN": "secret", "OTHER": "val"},
        )
        assert "--env=HF_TOKEN=secret" in cmd
        assert "--env=OTHER=val" in cmd

    def test_run_cmd_node_selector_via_overrides(self):
        cmd = _k8s(k8s_node_selector="gpu=true,zone=us-east").run_cmd(image="img", command="echo", container_name="pod1")
        assert "--overrides=" in cmd
        # JSON in --overrides must include both nodeSelector entries.
        assert "nodeSelector" in cmd
        assert "gpu" in cmd and "true" in cmd
        assert "us-east" in cmd

    def test_stop_cmd_uses_delete_pod(self):
        cmd = _k8s().stop_cmd("pod1")
        assert "kubectl " in cmd
        assert "delete pod pod1" in cmd
        assert "--ignore-not-found" in cmd
        assert "--grace-period=0" in cmd
        assert "--force" in cmd

    def test_stop_cmd_non_force_skips_grace_period(self):
        cmd = _k8s().stop_cmd("pod1", force=False)
        assert "--grace-period=0" not in cmd
        assert "--force" not in cmd

    def test_status_cmd_checks_running_phase(self):
        cmd = _k8s().status_cmd("pod1")
        assert "get pod pod1" in cmd
        assert "{.status.phase}" in cmd
        assert "= 'Running'" in cmd or "='Running'" in cmd

    def test_logs_cmd_follow_with_tail(self):
        cmd = _k8s().logs_cmd("pod1", follow=True, tail=100)
        assert cmd.startswith("kubectl ")
        assert " logs " in cmd
        assert "-f" in cmd
        assert "--tail=100" in cmd
        assert cmd.endswith("pod1")

    def test_logs_cmd_no_follow_no_tail(self):
        cmd = _k8s().logs_cmd("pod1", follow=False, tail=None)
        assert "-f" not in cmd
        assert "--tail" not in cmd


class TestK8sExecutorScripts:
    def test_launch_script_is_preflight(self):
        script = _k8s().generate_launch_script(image="img", container_name="pod1", command="ignored")
        assert "delete pod" in script
        # The actual `kubectl run` for the workload happens in
        # generate_exec_serve_script, not the preflight.
        assert "kubectl run" not in script.replace("\n", " ")
        assert "preflight complete" in script

    def test_exec_serve_script_creates_pod_with_image_env(self):
        script = _k8s().generate_exec_serve_script(
            container_name="pod1",
            serve_command="vllm serve foo",
            env={"SPARKRUN_K8S_IMAGE": "vllm/vllm:latest", "EXTRA": "x"},
        )
        assert "kubectl " in script
        assert "run pod1" in script
        assert "--image=vllm/vllm:latest" in script
        # SPARKRUN_K8S_IMAGE itself must NOT leak into the Pod env —
        # it's a wiring sentinel, not a workload variable.
        assert "SPARKRUN_K8S_IMAGE" not in script
        assert "--env=EXTRA=x" in script

    def test_exec_serve_script_missing_image_uses_placeholder(self):
        script = _k8s().generate_exec_serve_script(container_name="pod1", serve_command="cmd")
        # When no image is wired, we emit a loud placeholder so the
        # operator sees a clear failure rather than silent pod errors.
        assert "sparkrun-k8s-image-not-configured" in script

    def test_node_script_per_rank(self):
        ex = _k8s()
        s0 = ex.generate_node_script(image="img", container_name="sparkrun_abc_node_0", serve_command="rank0")
        s1 = ex.generate_node_script(image="img", container_name="sparkrun_abc_node_1", serve_command="rank1")
        assert "sparkrun_abc_node_0" in s0 and "sparkrun_abc_node_0" not in s1
        assert "sparkrun_abc_node_1" in s1 and "sparkrun_abc_node_1" not in s0

    def test_ray_head_raises(self):
        with pytest.raises(NotImplementedError, match="Ray"):
            _k8s().generate_ray_head_script(image="x", container_name="y")

    def test_ray_worker_raises(self):
        with pytest.raises(NotImplementedError, match="Ray"):
            _k8s().generate_ray_worker_script(image="x", container_name="y", head_ip="1.2.3.4")


class TestK8sExecutorBashSyntax:
    """Generated scripts must be valid bash."""

    def _check(self, script: str) -> None:
        result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        assert result.returncode == 0, "bash -n failed:\n%s\n---\n%s" % (script, result.stderr)

    def test_launch_script(self):
        self._check(_k8s().generate_launch_script(image="img", container_name="pod1", command=""))

    def test_exec_serve_script(self):
        self._check(
            _k8s().generate_exec_serve_script(
                container_name="pod1",
                serve_command="echo hi",
                env={"SPARKRUN_K8S_IMAGE": "img"},
            )
        )

    def test_node_script(self):
        self._check(
            _k8s().generate_node_script(
                image="img",
                container_name="sparkrun_abc_node_0",
                serve_command="echo hi",
            )
        )

    def test_lifecycle_snippets(self):
        ex = _k8s()
        for snippet in (
            ex.stop_cmd("pod1"),
            ex.status_cmd("pod1"),
            ex.logs_cmd("pod1", follow=True, tail=10),
        ):
            self._check("#!/bin/bash\n" + snippet + "\n")


# ---------------------------------------------------------------------------
# Executor resolution chain: CLI > recipe > runtime > defaults
# ---------------------------------------------------------------------------


def _resolve(*sources: dict) -> str:
    """Reproduce the launcher's chain build and return the resolved executor_type."""
    chain = Variables(sources=(*sources, EXECUTOR_DEFAULTS), env_placement=EnvPlacement.IGNORED)
    return ExecutorConfig.from_chain(chain).executor_type


class TestExecutorResolutionChain:
    """Validates the precedence layering done in ``core.launcher.launch_inference``.

    Order (highest → lowest):
        1. CLI executor_config
        2. Recipe executor_config (with recipe.executor merged in)
        3. Runtime.default_executor()
        4. exec_adjustments (rootless / auto_user — never set executor)
        5. EXECUTOR_DEFAULTS (no executor key → falls back to "docker")
    """

    def test_baseline_defaults_to_docker(self):
        assert _resolve({}, {}, {}, {}) == "docker"

    def test_runtime_default_local(self):
        assert _resolve({}, {}, {"executor": "local"}, {}) == "local"

    def test_runtime_default_k8s(self):
        assert _resolve({}, {}, {"executor": "k8s"}, {}) == "k8s"

    def test_recipe_overrides_runtime(self):
        assert _resolve({}, {"executor": "k8s"}, {"executor": "local"}, {}) == "k8s"

    def test_cli_overrides_recipe_and_runtime(self):
        assert _resolve({"executor": "docker"}, {"executor": "k8s"}, {"executor": "local"}, {}) == "docker"

    def test_empty_recipe_falls_through_to_runtime(self):
        # Recipe layer present but with no ``executor`` key — runtime
        # default still wins because the recipe didn't actually set one.
        assert _resolve({}, {"unrelated": "value"}, {"executor": "local"}, {}) == "local"


# ---------------------------------------------------------------------------
# RuntimePlugin.default_executor() base contract
# ---------------------------------------------------------------------------


class TestRuntimePluginDefaultExecutor:
    def test_base_class_returns_none(self):
        # Import after sparkrun bootstrap so plugin discovery doesn't
        # spin up. We construct a bare subclass to exercise the
        # default_executor() default.
        from sparkrun.runtimes.base import RuntimePlugin

        class _Stub(RuntimePlugin):
            runtime_name = "stub"

            def generate_command(self, recipe, overrides=None, **kwargs):
                return "echo stub"

            def resolve_container(self, recipe, overrides=None):
                return "img:tag"

        # ``RuntimePlugin`` itself extends SAF's ``Plugin``; we don't
        # need to call ``__init__`` — ``default_executor`` is a plain
        # method that reads no instance state.
        stub = _Stub.__new__(_Stub)
        assert stub.default_executor() is None


# ---------------------------------------------------------------------------
# Cross-executor: factory still dispatches Docker by default
# ---------------------------------------------------------------------------


class TestFactoryFullMatrix:
    def test_docker_default(self):
        assert isinstance(build_executor(None, None), DockerExecutor)

    def test_local(self):
        assert isinstance(build_executor("local", None), LocalExecutor)

    def test_k8s(self):
        assert isinstance(build_executor("k8s", None), K8sExecutor)

    def test_unknown_falls_back_to_docker(self):
        assert isinstance(build_executor("wasm", None), DockerExecutor)
