"""Tests for `sparkrun proxy start` CLI persistence and --restart behavior.

Covers:
- Change A: explicit CLI flags persist to proxy.yaml before any restart decision.
- Change B: --restart flag stops the running proxy and starts it again, and the
  bare "already running" path is now a hard error.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from click.testing import CliRunner


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def proxy_yaml(tmp_path: Path) -> Path:
    """Path that the patched ProxyConfig will read/write."""
    return tmp_path / "proxy.yaml"


@pytest.fixture
def real_proxy_cfg(proxy_yaml: Path):
    """A real ProxyConfig pointed at a tmp file, returned together with its path."""
    from sparkrun.proxy.config import ProxyConfig

    return ProxyConfig(proxy_yaml)


@pytest.fixture
def patch_proxy_config(proxy_yaml: Path):
    """Make ``sctx.proxy_config`` resolve to a real ProxyConfig at *proxy_yaml*.

    Yields the ProxyConfig instance so tests can pre-seed values.
    """
    from sparkrun.proxy.config import ProxyConfig

    cfg = ProxyConfig(proxy_yaml)

    def fake_get_proxy_config(self):
        return cfg

    with patch("sparkrun.core.config.SparkrunConfig.get_proxy_config", fake_get_proxy_config):
        yield cfg


@pytest.fixture
def patch_discovery():
    """Stub out endpoint discovery — return one healthy endpoint."""
    from sparkrun.proxy.discovery import DiscoveredEndpoint

    endpoints = [
        DiscoveredEndpoint(
            cluster_id="sparkrun_test",
            model="test/model",
            served_model_name=None,
            runtime="vllm",
            host="10.0.0.1",
            port=8000,
            healthy=True,
            actual_models=["test/model"],
            recipe_name="test-recipe",
        ),
    ]
    with patch("sparkrun.proxy.discovery.discover_endpoints", return_value=endpoints):
        yield endpoints


# =====================================================================
# _persist_cli_overrides helper
# =====================================================================


class TestPersistCliOverrides:
    """Direct unit tests for the persistence helper."""

    def test_no_flags_supplied_returns_empty(self, real_proxy_cfg):
        """None values are skipped — nothing changes, nothing written."""
        from sparkrun.cli._proxy import _persist_cli_overrides

        changed = _persist_cli_overrides(
            real_proxy_cfg,
            port=None,
            bind_host=None,
            master_key=None,
            enable_ui=None,
            discover_interval=None,
        )
        assert changed == []
        # File should not have been created by save()
        assert not real_proxy_cfg.config_path.exists()

    def test_supplied_value_matching_current_is_skipped(self, real_proxy_cfg):
        """A flag whose value equals the saved value is a no-op (no write, no echo)."""
        from sparkrun.cli._proxy import _persist_cli_overrides

        real_proxy_cfg.set_proxy(master_key="sk-existing")
        real_proxy_cfg.save()
        mtime_before = real_proxy_cfg.config_path.stat().st_mtime_ns

        changed = _persist_cli_overrides(
            real_proxy_cfg,
            port=None,
            bind_host=None,
            master_key="sk-existing",
            enable_ui=None,
            discover_interval=None,
        )
        assert changed == []
        assert real_proxy_cfg.config_path.stat().st_mtime_ns == mtime_before

    def test_supplied_value_persists_and_returns_key(self, real_proxy_cfg):
        from sparkrun.cli._proxy import _persist_cli_overrides

        changed = _persist_cli_overrides(
            real_proxy_cfg,
            port=5000,
            bind_host=None,
            master_key="sk-new",
            enable_ui=True,
            discover_interval=None,
        )
        assert set(changed) == {"port", "master_key", "enable_ui"}

        # Re-read fresh to confirm persisted to disk.
        from sparkrun.proxy.config import ProxyConfig

        fresh = ProxyConfig(real_proxy_cfg.config_path)
        assert fresh.port == 5000
        assert fresh.master_key == "sk-new"
        assert fresh.enable_ui is True

    def test_enable_ui_false_is_explicit_and_persists(self, real_proxy_cfg):
        """``enable_ui=False`` must be treated as an explicit user choice."""
        from sparkrun.cli._proxy import _persist_cli_overrides

        real_proxy_cfg.set_proxy(enable_ui=True)
        real_proxy_cfg.save()

        changed = _persist_cli_overrides(
            real_proxy_cfg,
            port=None,
            bind_host=None,
            master_key=None,
            enable_ui=False,
            discover_interval=None,
        )
        assert changed == ["enable_ui"]

        from sparkrun.proxy.config import ProxyConfig

        fresh = ProxyConfig(real_proxy_cfg.config_path)
        assert fresh.enable_ui is False


# =====================================================================
# CLI integration — `sparkrun proxy start`
# =====================================================================


class _RunningState:
    """Tiny container the patched ProxyEngine reads/writes."""

    def __init__(self, running: bool, pid: int = 12345, port: int = 4000):
        self.running = running
        self.pid = pid
        self.port = port
        self.start_called = False
        self.stop_called = False
        # Number of is_running() calls observed before flip; used for timeout test.
        self.is_running_calls = 0


@pytest.fixture
def patch_engine():
    """Patch ProxyEngine.is_running/stop/start with introspectable side effects.

    Tests configure the returned ``state`` object first, then yield to run
    the CLI; assertions consult the state afterwards.
    """
    state = _RunningState(running=False)

    def fake_is_running(self):
        state.is_running_calls += 1
        return state.running

    def fake_read_pid(self):
        return state.pid if state.running else None

    def fake_start(self, **kwargs):
        state.start_called = True
        state.running = True
        return 0

    def fake_stop(self, dry_run: bool = False):
        state.stop_called = True
        state.running = False
        return True

    with (
        patch("sparkrun.proxy.engine.ProxyEngine.is_running", fake_is_running),
        patch("sparkrun.proxy.engine.ProxyEngine._read_pid", fake_read_pid),
        patch("sparkrun.proxy.engine.ProxyEngine.start", fake_start),
        patch("sparkrun.proxy.engine.ProxyEngine.stop", fake_stop),
        patch("sparkrun.proxy.engine.write_config", return_value=Path("/tmp/cfg.yaml")),
    ):
        yield state


class TestStartCli:
    """End-to-end CLI tests of `sparkrun proxy start` with new logic."""

    def test_start_clean_no_flags_calls_start(self, patch_proxy_config, patch_discovery, patch_engine):
        """Baseline: no flags, proxy not running -> engine.start called."""
        from sparkrun.cli._proxy import proxy

        result = CliRunner().invoke(proxy, ["start"])
        assert result.exit_code == 0, result.output
        assert patch_engine.start_called is True
        assert patch_engine.stop_called is False
        # No keys changed -> no "Saved proxy.yaml" line.
        assert "Saved proxy.yaml" not in result.output

    def test_start_with_master_key_when_not_running_persists_and_starts(
        self, patch_proxy_config, patch_discovery, patch_engine, proxy_yaml: Path
    ):
        """Supplied --master-key persists to YAML and engine.start runs."""
        from sparkrun.cli._proxy import proxy

        result = CliRunner().invoke(proxy, ["start", "--master-key", "sk-NEW"])
        assert result.exit_code == 0, result.output
        assert "Saved proxy.yaml" in result.output
        assert "master_key" in result.output
        assert patch_engine.start_called is True

        # YAML really was written.
        data = yaml.safe_load(proxy_yaml.read_text())
        assert data["proxy"]["master_key"] == "sk-NEW"

    def test_start_with_master_key_when_running_no_restart_persists_then_exits_1(
        self, patch_proxy_config, patch_discovery, patch_engine, proxy_yaml: Path
    ):
        """Already-running -> persist new flags, then exit 1 with --restart hint."""
        from sparkrun.cli._proxy import proxy

        patch_engine.running = True

        result = CliRunner().invoke(proxy, ["start", "--master-key", "sk-NEW"])
        assert result.exit_code == 1, result.output
        # Persistence happened.
        assert "Saved proxy.yaml" in result.output
        data = yaml.safe_load(proxy_yaml.read_text())
        assert data["proxy"]["master_key"] == "sk-NEW"
        # Hint points to --restart.
        assert "--restart" in result.output
        # Did not stop/start.
        assert patch_engine.stop_called is False
        assert patch_engine.start_called is False

    def test_restart_when_running_stops_then_starts(self, patch_proxy_config, patch_discovery, patch_engine):
        """--restart on a running proxy: stop, then start."""
        from sparkrun.cli._proxy import proxy

        patch_engine.running = True

        result = CliRunner().invoke(proxy, ["start", "--restart"])
        assert result.exit_code == 0, result.output
        assert "Restarting proxy" in result.output
        assert patch_engine.stop_called is True
        assert patch_engine.start_called is True

    def test_restart_when_not_running_does_not_stop(self, patch_proxy_config, patch_discovery, patch_engine):
        """--restart with no running proxy: skip stop, normal start."""
        from sparkrun.cli._proxy import proxy

        result = CliRunner().invoke(proxy, ["start", "--restart"])
        assert result.exit_code == 0, result.output
        assert patch_engine.stop_called is False
        assert patch_engine.start_called is True

    def test_restart_with_flags_persists_and_restarts(self, patch_proxy_config, patch_discovery, patch_engine, proxy_yaml: Path):
        """--restart --enable-ui --master-key: persist both, stop, then start."""
        from sparkrun.cli._proxy import proxy

        patch_engine.running = True

        result = CliRunner().invoke(
            proxy,
            ["start", "--restart", "--enable-ui", "--master-key", "sk-NEW"],
        )
        assert result.exit_code == 0, result.output
        assert "Saved proxy.yaml" in result.output

        data = yaml.safe_load(proxy_yaml.read_text())
        assert data["proxy"]["master_key"] == "sk-NEW"
        assert data["proxy"]["enable_ui"] is True
        assert patch_engine.stop_called is True
        assert patch_engine.start_called is True

    def test_start_no_op_master_key_does_not_save(self, patch_proxy_config, patch_discovery, patch_engine, proxy_yaml: Path):
        """Supplied --master-key matching current value: no save, no echo."""
        from sparkrun.cli._proxy import proxy

        # Pre-seed proxy.yaml with the same master_key.
        patch_proxy_config.set_proxy(master_key="sk-SAME")
        patch_proxy_config.save()
        mtime_before = proxy_yaml.stat().st_mtime_ns

        result = CliRunner().invoke(proxy, ["start", "--master-key", "sk-SAME"])
        assert result.exit_code == 0, result.output
        assert "Saved proxy.yaml" not in result.output
        assert proxy_yaml.stat().st_mtime_ns == mtime_before

    def test_restart_stop_timeout_aborts(self, patch_proxy_config, patch_discovery, monkeypatch):
        """If engine.stop() does not flip is_running, exit 1 after the 10s budget."""
        from sparkrun.cli._proxy import proxy

        # Custom engine patches: stop() does NOT clear running state.
        state = {"running": True, "stop_called": False, "start_called": False}

        def fake_is_running(self):
            return state["running"]

        def fake_read_pid(self):
            return 999

        def fake_stop(self, dry_run: bool = False):
            state["stop_called"] = True
            # Intentionally leave running=True to simulate stuck process.
            return True

        def fake_start(self, **kwargs):
            state["start_called"] = True
            return 0

        # Make time.sleep instantaneous so the polling loop exits the 10s budget fast.
        import sparkrun.cli._proxy as proxy_mod  # noqa: F401  (anchor for time import scope)

        sleep_calls = []

        def fake_sleep(secs):
            sleep_calls.append(secs)

        monkeypatch.setattr("time.sleep", fake_sleep)

        with (
            patch("sparkrun.proxy.engine.ProxyEngine.is_running", fake_is_running),
            patch("sparkrun.proxy.engine.ProxyEngine._read_pid", fake_read_pid),
            patch("sparkrun.proxy.engine.ProxyEngine.stop", fake_stop),
            patch("sparkrun.proxy.engine.ProxyEngine.start", fake_start),
            patch("sparkrun.proxy.engine.write_config", return_value=Path("/tmp/cfg.yaml")),
        ):
            result = CliRunner().invoke(proxy, ["start", "--restart"])

        assert result.exit_code == 1, result.output
        assert "did not stop cleanly" in result.output
        assert state["stop_called"] is True
        assert state["start_called"] is False
        # Polling actually ran (>= 1 sleep call).
        assert len(sleep_calls) >= 1
