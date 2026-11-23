"""Tests for channel-aware feature flags and plugin gating.

Covers:

- :func:`sparkrun.core.features.is_feature_enabled` precedence
  (env > config > channel-default > baseline > fail-closed).
- Per-channel defaults on the built-in executor flags.
- Plugin self-gate: a gated-off executor hides itself from the SAF registry
  via ``is_multi_extension``; channel / config / env enabling brings it back.
- ``resolve_executor`` fails loudly for an explicitly-requested but gated
  executor.
- A guard that every plugin ``required_feature_flag`` maps to a registered flag.
"""

from __future__ import annotations

import yaml

from sparkrun.core.config import SparkrunConfig
from sparkrun.core.features import (
    FeatureFlag,
    all_features,
    get_feature,
    is_feature_enabled,
)


def _run_gated_snippet(tmp_path, config_data: dict, snippet: str, env_extra: dict | None = None) -> str:
    """Run *snippet* in a clean subprocess with an isolated sparkrun config.

    Bootstrap gating is observed end-to-end here because SAF registers
    plugins in a process-global registry — once an in-process test enables
    ``local``/``k8s`` they never un-register. A fresh subprocess with
    ``HOME`` redirected exercises the real production path
    (reading ``$HOME/.config/sparkrun/config.yaml``) with a clean registry.
    """
    import os
    import subprocess
    import sys

    home = tmp_path / "home"
    cfg_dir = home / ".config" / "sparkrun"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(config_data))

    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARKRUN_FEATURE_")}
    env.pop("STATEFUL_ROOT", None)
    env["HOME"] = str(home)
    env["SPARKRUN_NO_TELEMETRY"] = "1"
    if env_extra:
        env.update(env_extra)

    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, "subprocess failed: %s\n%s" % (proc.stdout, proc.stderr)
    return proc.stdout.strip()


def _executor_names(tmp_path, config_data: dict, env_extra: dict | None = None) -> list[str]:
    """Return the SAF-registered executor names for a gated config."""
    import json

    snippet = (
        "import json\n"
        "from sparkrun.core.bootstrap import init_sparkrun\n"
        "from sparkrun.orchestration.executor import list_executors\n"
        "v = init_sparkrun()\n"
        "print(json.dumps(list_executors(v)))\n"
    )
    out = _run_gated_snippet(tmp_path, config_data, snippet, env_extra)
    return json.loads(out.splitlines()[-1])


# ---------------------------------------------------------------------------
# FeatureFlag.default_for_channel
# ---------------------------------------------------------------------------


class TestFeatureFlagDefaults:
    def test_default_for_listed_channel(self):
        flag = FeatureFlag("x", "d", channel_defaults={"alpha": True}, default=False)
        assert flag.default_for_channel("alpha") is True
        assert flag.default_for_channel("stable") is False

    def test_unlisted_channel_uses_baseline(self):
        flag = FeatureFlag("x", "d", channel_defaults={"alpha": True}, default=True)
        assert flag.default_for_channel("beta") is True

    def test_channel_normalized(self):
        flag = FeatureFlag("x", "d", channel_defaults={"alpha": True})
        # "yolo" normalizes to alpha.
        assert flag.default_for_channel("yolo") is True


# ---------------------------------------------------------------------------
# cli.setup.features — visibility-only flag (group always functional)
# ---------------------------------------------------------------------------


class TestCliSetupFeaturesFlag:
    def test_channel_defaults(self):
        # On for beta/alpha, off for stable — no env/config overrides.
        assert is_feature_enabled("cli.setup.features", channel="stable", env={}) is False
        assert is_feature_enabled("cli.setup.features", channel="beta", env={}) is True
        assert is_feature_enabled("cli.setup.features", channel="alpha", env={}) is True

    def test_hidden_on_stable_but_still_functional(self, tmp_path):
        # Clean subprocess (the group's `hidden=` freezes at import from config).
        snippet = (
            "from click.testing import CliRunner\n"
            "from sparkrun.cli import main\n"
            "h = CliRunner().invoke(main, ['setup', '--help']).output\n"
            "listed = any(l.strip().startswith('features') for l in h.splitlines())\n"
            "r = CliRunner().invoke(main, ['setup', 'features', 'list'])\n"
            "print('LISTED', listed)\n"
            "print('LISTEXIT', r.exit_code)\n"
        )
        out = _run_gated_snippet(tmp_path, {"self_update": {"channel": "stable"}}, snippet)
        assert "LISTED False" in out  # hidden from --help on stable
        assert "LISTEXIT 0" in out  # ...but the command still works

    def test_visible_on_beta(self, tmp_path):
        snippet = (
            "from click.testing import CliRunner\n"
            "from sparkrun.cli import main\n"
            "h = CliRunner().invoke(main, ['setup', '--help']).output\n"
            "print('LISTED', any(l.strip().startswith('features') for l in h.splitlines()))\n"
        )
        out = _run_gated_snippet(tmp_path, {"self_update": {"channel": "beta"}}, snippet)
        assert "LISTED True" in out


# ---------------------------------------------------------------------------
# Resolution precedence
# ---------------------------------------------------------------------------


class TestResolutionPrecedence:
    def test_unknown_flag_fails_closed(self):
        assert is_feature_enabled("does.not.exist", env={}) is False

    def test_local_executor_off_on_all_channels(self):
        # Local remains opt-in; Kubernetes is owned by its integration gate.
        for channel in ("stable", "beta", "alpha"):
            assert is_feature_enabled("executor.local", channel=channel, env={}) is False

    def test_docker_flag_on_all_channels(self):
        # The default docker executor ships ENABLED on every channel (it carries
        # a flag only for uniformity + a future opt-out).
        for channel in ("stable", "beta", "alpha"):
            assert is_feature_enabled("executor.docker", channel=channel, env={}) is True

    def test_docker_flag_can_be_disabled_by_config(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"features": {"executor.docker": False}}))
        config = SparkrunConfig(cfg_path)
        assert is_feature_enabled("executor.docker", config=config, env={}) is False

    def test_env_overrides_channel(self):
        assert is_feature_enabled("executor.k8s", channel="stable", env={"SPARKRUN_FEATURE_EXECUTOR_K8S": "1"}) is True
        assert is_feature_enabled("executor.k8s", channel="alpha", env={"SPARKRUN_FEATURE_EXECUTOR_K8S": "0"}) is False

    def test_config_override_beats_channel(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"features": {"channel": "stable", "integration.k8s": True}}))
        config = SparkrunConfig(cfg_path)
        # env empty so config wins over the stable channel default.
        assert is_feature_enabled("integration.k8s", config=config, env={}) is True

    def test_env_beats_config(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"features": {"executor.k8s": True}}))
        config = SparkrunConfig(cfg_path)
        assert is_feature_enabled("executor.k8s", config=config, env={"SPARKRUN_FEATURE_EXECUTOR_K8S": "0"}) is False


# ---------------------------------------------------------------------------
# SparkrunConfig surface
# ---------------------------------------------------------------------------


class TestConfigSurface:
    def test_feature_channel_defaults_to_self_update(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"self_update": {"channel": "beta"}}))
        config = SparkrunConfig(cfg_path)
        assert config.feature_channel == "beta"

    def test_feature_channel_explicit_override(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"self_update": {"channel": "stable"}, "features": {"channel": "alpha"}}))
        config = SparkrunConfig(cfg_path)
        assert config.feature_channel == "alpha"

    def test_feature_override_unset_is_none(self, tmp_path):
        config = SparkrunConfig(tmp_path / "missing.yaml")
        assert config.feature_override("executor.k8s") is None

    def test_channel_key_not_treated_as_flag(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"features": {"channel": "alpha"}}))
        config = SparkrunConfig(cfg_path)
        assert config.feature_override("channel") is None


# ---------------------------------------------------------------------------
# Plugin self-gate — is_multi_extension hides gated-off executors
# ---------------------------------------------------------------------------


class TestGatingDecision:
    """Unit-level tests of the plugin self-gate (``is_multi_extension``).

    The gate resolves via env first, so these are deterministic regardless
    of the developer's real config file.
    """

    def test_hidden_when_flag_disabled(self, monkeypatch):
        from sparkrun.plugins.k8s.executor import K8sExecutor

        monkeypatch.setenv("SPARKRUN_FEATURE_EXECUTOR_K8S", "0")
        assert K8sExecutor().is_multi_extension(None) is False

    def test_exposed_when_flag_enabled(self, monkeypatch):
        from sparkrun.plugins.k8s.executor import K8sExecutor

        monkeypatch.setenv("SPARKRUN_FEATURE_EXECUTOR_K8S", "1")
        assert K8sExecutor().is_multi_extension(None) is True

    def test_local_hidden_when_flag_disabled(self, monkeypatch):
        from sparkrun.orchestration.executors.local import LocalExecutor

        monkeypatch.setenv("SPARKRUN_FEATURE_EXECUTOR_LOCAL", "0")
        assert LocalExecutor().is_multi_extension(None) is False

    def test_docker_exposed_when_flag_enabled(self, monkeypatch):
        # Docker carries a flag (uniformity) that defaults on; an explicit
        # enable keeps it exposed.
        from sparkrun.orchestration.executors.docker import DockerExecutor

        monkeypatch.setenv("SPARKRUN_FEATURE_EXECUTOR_DOCKER", "1")
        assert DockerExecutor().is_multi_extension(None) is True

    def test_docker_hidden_when_flag_disabled(self, monkeypatch):
        # Objective: the default executor can be turned off on hosts that don't
        # need it — the self-gate honors an explicit disable just like the
        # experimental executors.
        from sparkrun.orchestration.executors.docker import DockerExecutor

        monkeypatch.setenv("SPARKRUN_FEATURE_EXECUTOR_DOCKER", "0")
        assert DockerExecutor().is_multi_extension(None) is False


class TestBootstrapGatingEndToEnd:
    """End-to-end gating in a clean subprocess (real production config path)."""

    def test_stable_excludes_experimental_executors(self, tmp_path):
        names = _executor_names(tmp_path, {"features": {"channel": "stable"}})
        assert "docker" in names
        assert "local" not in names
        assert "k8s" not in names

    def test_alpha_enables_k8s_but_keeps_local_opt_in(self, tmp_path):
        # The Kubernetes plugin is available by default on alpha.
        names = _executor_names(tmp_path, {"features": {"channel": "alpha"}})
        assert "local" not in names
        assert "k8s" in names

    def test_docker_included_by_default(self, tmp_path):
        # Docker's flag defaults on, so it registers without any opt-in.
        names = _executor_names(tmp_path, {"features": {"channel": "stable"}})
        assert "docker" in names

    def test_docker_can_be_disabled(self, tmp_path):
        # Objective: drop the docker executor where it isn't needed — an
        # explicit disable removes it from the registered set.
        names = _executor_names(tmp_path, {"features": {"executor.docker": False}})
        assert "docker" not in names

    def test_config_override_includes_experimental_executors(self, tmp_path):
        names = _executor_names(tmp_path, {"features": {"executor.local": True, "integration.k8s": True}})
        assert {"docker", "local", "k8s"} <= set(names)

    def test_explicit_override_on_stable(self, tmp_path):
        names = _executor_names(tmp_path, {"features": {"channel": "stable", "integration.k8s": True}})
        assert "k8s" in names
        assert "local" not in names

    def test_env_override_on_stable(self, tmp_path):
        names = _executor_names(
            tmp_path,
            {"features": {"channel": "stable"}},
            env_extra={"SPARKRUN_FEATURE_EXECUTOR_LOCAL": "1"},
        )
        assert "local" in names
        assert "k8s" not in names


# ---------------------------------------------------------------------------
# resolve_executor fails loudly for a gated explicit request
# ---------------------------------------------------------------------------


class TestGatedExplicitRequestFails:
    def test_gated_executor_request_raises_with_guidance(self, tmp_path):
        snippet = (
            "from sparkrun.core.bootstrap import init_sparkrun\n"
            "from sparkrun.core.config import SparkrunConfig, get_config_root\n"
            "from sparkrun.orchestration.executor import ExecutorUnavailableError, resolve_executor\n"
            "v = init_sparkrun()\n"
            "config = SparkrunConfig(get_config_root(v) / 'config.yaml')\n"
            "try:\n"
            "    resolve_executor(cli_overrides={'executor': 'k8s'}, config=config, v=v)\n"
            "    print('NO_RAISE')\n"
            "except ExecutorUnavailableError as e:\n"
            "    print('RAISED:' + str(e))\n"
        )
        out = _run_gated_snippet(tmp_path, {"features": {"channel": "stable"}}, snippet)
        line = out.splitlines()[-1]
        assert line.startswith("RAISED:"), out
        assert "integration.k8s" in line
        assert "setup features enable" in line


# ---------------------------------------------------------------------------
# Guard: every plugin required_feature_flag maps to a registered flag
# ---------------------------------------------------------------------------


class TestRequiredFeatureGuard:
    def test_executor_required_features_are_registered(self):
        from scitrera_app_framework.util import find_types_in_modules
        from sparkrun.orchestration.executors._base import Executor

        for cls in find_types_in_modules("sparkrun.orchestration.executors", Executor):
            feature = getattr(cls, "required_feature_flag", None)
            if feature:
                assert get_feature(feature) is not None, "%s gates on unregistered flag %r" % (cls.__name__, feature)

    def test_transport_required_features_are_registered(self):
        from scitrera_app_framework.util import find_types_in_modules
        from sparkrun.transports.base import Transport

        for cls in find_types_in_modules("sparkrun.transports", Transport):
            feature = getattr(cls, "required_feature_flag", None)
            if feature:
                assert get_feature(feature) is not None, "%s gates on unregistered flag %r" % (cls.__name__, feature)

    def test_builtin_executor_flags_present(self):
        names = {f.name for f in all_features()}
        assert {"executor.local", "gateway"} <= names
        assert "integration.k8s" not in names  # Legacy loading aliases live in plugin controls.


class TestCliTuneFlag:
    def test_enabled_on_every_core_channel(self):
        for channel in ("stable", "beta", "alpha"):
            assert is_feature_enabled("cli.tune", channel=channel, env={})

    def test_disabled_hides_help_completion_and_blocks_execution(self, tmp_path):
        snippet = """
from click.testing import CliRunner
from sparkrun.cli import main
runner = CliRunner()
help_result = runner.invoke(main, ['--help'])
assert help_result.exit_code == 0, help_result.output
assert not any(line.strip().startswith('tune ') for line in help_result.output.splitlines())
completion = runner.invoke(main, [], prog_name='sparkrun', env={
    '_SPARKRUN_COMPLETE': 'bash_complete', 'COMP_WORDS': 'sparkrun tu', 'COMP_CWORD': '1',
})
assert completion.exit_code == 0, completion.output
assert 'plain,tune' not in completion.output
result = runner.invoke(main, ['tune', 'vllm', '--help'])
assert result.exit_code == 1, result.output
assert 'setup features enable cli.tune' in result.output
print('gated')
"""
        assert _run_gated_snippet(tmp_path, {"features": {"cli.tune": False}}, snippet) == "gated"

    def test_environment_can_override_disabled_config(self, tmp_path):
        snippet = """
from click.testing import CliRunner
from sparkrun.cli import main
result = CliRunner().invoke(main, ['tune', 'vllm', '--help'])
assert result.exit_code == 0, result.output
assert '--tp' in result.output
print('enabled')
"""
        assert _run_gated_snippet(tmp_path, {"features": {"cli.tune": False}}, snippet, {"SPARKRUN_FEATURE_CLI_TUNE": "1"}) == "enabled"


class TestFeatureListSetupSteps:
    def test_default_and_all_filter_text_and_json(self, tmp_path, monkeypatch):
        import json
        from types import SimpleNamespace
        from click.testing import CliRunner
        from sparkrun.cli._setup import _commands
        from sparkrun.core.features import FEATURE_FLAGS

        config = SparkrunConfig(tmp_path / "config.yaml")
        monkeypatch.setattr(_commands, "_get_context", lambda ctx: SimpleNamespace(config=config))
        # The prefix applies equally to built-in and plugin-contributed steps.
        monkeypatch.setitem(FEATURE_FLAGS, "setup.steps.fixture", FeatureFlag("setup.steps.fixture", "Plugin setup step", default=True))
        monkeypatch.setitem(FEATURE_FLAGS, "test.off.fixture", FeatureFlag("test.off.fixture", "Off by default", default=False))
        runner = CliRunner()
        expected = {flag.name for flag in all_features()}
        enabled = {name for name in expected if config.is_feature_enabled(name) and not name.startswith("setup.steps.")}
        hidden_off = sum(1 for name in expected if not config.is_feature_enabled(name) and not name.startswith("setup.steps."))
        assert hidden_off >= 1
        for include_all in (False, True):
            args = ["--all"] if include_all else []
            result = runner.invoke(_commands.setup_features_list, [*args, "--json"])
            assert result.exit_code == 0, result.output
            rows = json.loads(result.output)
            assert {row["name"] for row in rows} == (expected if include_all else enabled)
            assert include_all or all(row["enabled"] for row in rows)
            result = runner.invoke(_commands.setup_features_list, args)
            assert result.exit_code == 0, result.output
            assert ("setup.steps.sudoers" in result.output) is include_all
            assert ("setup.steps.fixture" in result.output) is include_all
            assert ("test.off.fixture" in result.output) is include_all
            assert ("%d disabled feature(s) hidden; use --all" % hidden_off in result.output) is not include_all
            assert "cli.tune" in result.output

    def test_all_option_is_advertised(self):
        from click.testing import CliRunner
        from sparkrun.cli._setup import _commands

        result = CliRunner().invoke(_commands.setup_features_list, ["--help"])
        assert result.exit_code == 0
        assert "--json" in result.output
        assert "--all" in result.output

    def test_omitted_steps_still_accept_overrides(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from sparkrun.cli._setup import _commands

        config = SparkrunConfig(tmp_path / "config.yaml")
        monkeypatch.setattr("sparkrun.cli._plugins.management_config", lambda ctx: config)
        runner = CliRunner()
        for command, expected in (
            (_commands.setup_features_disable, False),
            (_commands.setup_features_enable, True),
            (_commands.setup_features_reset, None),
        ):
            result = runner.invoke(command, ["setup.steps.sudoers"])
            assert result.exit_code == 0, result.output
            assert SparkrunConfig(config.config_path).feature_override("setup.steps.sudoers") is expected


_RUNTIME_SNIPPET = """
import json
from sparkrun.core.bootstrap import get_runtime, init_sparkrun, list_runtimes
from sparkrun.runtimes.base import RuntimeUnavailableError

v = init_sparkrun()
out = {"runtimes": list_runtimes(v)}
for name in ("tokenary", "sglang", "no-such-runtime"):
    try:
        get_runtime(name, v)
        out[name] = "ok"
    except RuntimeUnavailableError as exc:
        out[name] = "disabled: " + str(exc)
    except ValueError as exc:
        out[name] = "unknown"
print(json.dumps(out))
"""


def _runtime_gates(tmp_path, config_data: dict, env_extra: dict | None = None) -> dict:
    import json

    return json.loads(_run_gated_snippet(tmp_path, config_data, _RUNTIME_SNIPPET, env_extra).splitlines()[-1])


class TestRuntimeGates:
    """Every in-tree runtime has a ``runtime.*`` gate; tokenary is opt-in."""

    def test_defaults(self):
        from sparkrun.core.features import FEATURE_FLAGS

        defaults = {name: flag.default for name, flag in FEATURE_FLAGS.items() if name.startswith("runtime.")}
        assert defaults == {
            "runtime.vllm": True,
            "runtime.sglang": True,
            "runtime.llama_cpp": True,
            "runtime.trtllm": True,
            "runtime.atlas": True,
            "runtime.modular_max": True,
            "runtime.tokenary": False,
        }
        assert not any(FEATURE_FLAGS[name].channel_defaults for name in defaults)  # same on every channel

    def test_every_in_tree_runtime_declares_a_registered_gate(self):
        from scitrera_app_framework.util import find_types_in_modules

        from sparkrun.core.features import FEATURE_FLAGS
        from sparkrun.runtimes.base import RuntimePlugin

        runtimes = [cls for cls in find_types_in_modules("sparkrun.runtimes", RuntimePlugin) if cls.runtime_name]
        assert runtimes
        for cls in runtimes:
            assert cls.required_feature_flag in FEATURE_FLAGS, cls.runtime_name

    def test_vllm_family_shares_one_gate(self):
        from sparkrun.runtimes.eugr_vllm_ray import EugrVllmRayRuntime
        from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
        from sparkrun.runtimes.vllm_ray import VllmRayRuntime

        assert {c.required_feature_flag for c in (VllmRayRuntime, VllmDistributedRuntime, EugrVllmRayRuntime)} == {"runtime.vllm"}

    def test_tokenary_off_by_default_and_named_when_requested(self, tmp_path):
        out = _runtime_gates(tmp_path, {"features": {"channel": "alpha"}})
        assert "tokenary" not in out["runtimes"]
        assert {"vllm-distributed", "vllm-ray", "sglang", "llama-cpp"} <= set(out["runtimes"])
        assert out["sglang"] == "ok"
        assert out["tokenary"].startswith("disabled: ") and "sparkrun setup features enable runtime.tokenary" in out["tokenary"]
        assert out["no-such-runtime"] == "unknown"  # an unknown name is still "unknown", not "disabled"

    def test_tokenary_opt_in(self, tmp_path):
        out = _runtime_gates(tmp_path, {"features": {"runtime.tokenary": True}})
        assert "tokenary" in out["runtimes"] and out["tokenary"] == "ok"

    def test_default_on_runtime_can_be_disabled(self, tmp_path):
        out = _runtime_gates(tmp_path, {}, env_extra={"SPARKRUN_FEATURE_RUNTIME_VLLM": "0"})
        assert not {"vllm-ray", "vllm-distributed", "eugr-vllm"} & set(out["runtimes"])
        assert "sglang" in out["runtimes"]
