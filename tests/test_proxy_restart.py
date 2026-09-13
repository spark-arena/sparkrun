"""Tests for `sparkrun proxy start` CLI persistence and --restart behavior.

Covers:
- Change A: explicit CLI flags persist to proxy.yaml before any restart decision.
- Change B: --restart flag stops the running proxy and starts it again, and the
  bare "already running" path is now a hard error.
"""

from __future__ import annotations

from pathlib import Path
import sys
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
# _persist_overrides helper (api.proxy)
# =====================================================================


class TestPersistCliOverrides:
    """Direct unit tests for the persistence helper (now in api.proxy)."""

    def test_no_flags_supplied_returns_empty(self, real_proxy_cfg):
        """None values are skipped — nothing changes, nothing written."""
        from sparkrun.api.proxy._ops import ProxyStartOptions, _persist_overrides

        changed = _persist_overrides(real_proxy_cfg, ProxyStartOptions())
        assert changed == []
        # File should not have been created by save()
        assert not real_proxy_cfg.config_path.exists()

    def test_supplied_value_matching_current_is_skipped(self, real_proxy_cfg):
        """A flag whose value equals the saved value is a no-op (no write, no echo)."""
        from sparkrun.api.proxy._ops import ProxyStartOptions, _persist_overrides

        real_proxy_cfg.set_proxy(master_key="sk-existing")
        real_proxy_cfg.save()
        mtime_before = real_proxy_cfg.config_path.stat().st_mtime_ns

        changed = _persist_overrides(real_proxy_cfg, ProxyStartOptions(master_key="sk-existing"))
        assert changed == []
        assert real_proxy_cfg.config_path.stat().st_mtime_ns == mtime_before

    def test_supplied_value_persists_and_returns_key(self, real_proxy_cfg):
        from sparkrun.api.proxy._ops import ProxyStartOptions, _persist_overrides

        changed = _persist_overrides(real_proxy_cfg, ProxyStartOptions(port=5000, master_key="sk-new"))
        assert set(changed) == {"port", "master_key"}

        # Re-read fresh to confirm persisted to disk.
        from sparkrun.proxy.config import ProxyConfig

        fresh = ProxyConfig(real_proxy_cfg.config_path)
        assert fresh.port == 5000
        assert fresh.master_key == "sk-new"

    def test_bind_host_false_value_still_persists(self, real_proxy_cfg):
        """A falsy-but-explicit value is an explicit user choice, not "unset"."""
        from sparkrun.api.proxy._ops import ProxyStartOptions, _persist_overrides

        real_proxy_cfg.set_proxy(host="0.0.0.0")
        real_proxy_cfg.save()

        changed = _persist_overrides(real_proxy_cfg, ProxyStartOptions(host="127.0.0.1"))
        assert changed == ["host"]

        from sparkrun.proxy.config import ProxyConfig

        fresh = ProxyConfig(real_proxy_cfg.config_path)
        assert fresh.host == "127.0.0.1"


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
        patch("sparkrun.proxy.engine.ProxyEngine._await_exit", return_value=True),
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
        """--restart --port --master-key: persist both, stop, then start."""
        from sparkrun.cli._proxy import proxy

        patch_engine.running = True

        result = CliRunner().invoke(
            proxy,
            ["start", "--restart", "--port", "4321", "--master-key", "sk-NEW"],
        )
        assert result.exit_code == 0, result.output
        assert "Saved proxy.yaml" in result.output

        data = yaml.safe_load(proxy_yaml.read_text())
        assert data["proxy"]["master_key"] == "sk-NEW"
        assert data["proxy"]["port"] == 4321
        assert patch_engine.stop_called is True
        assert patch_engine.start_called is True

    def test_stale_enable_ui_warns_but_still_starts(self, patch_proxy_config, patch_discovery, patch_engine):
        """A leftover ``enable_ui: true`` warns and is ignored, never blocks.

        The UI is unsupported (LiteLLM's /ui needs PostgreSQL), but a key
        persisted by an older sparkrun must not make the proxy unstartable —
        the user would be stuck until they hand-edited proxy.yaml.
        """
        from sparkrun.cli._proxy import proxy

        patch_proxy_config.set_proxy(enable_ui=True)
        patch_proxy_config.save()

        result = CliRunner().invoke(proxy, ["start"])

        assert result.exit_code == 0, result.output
        assert "obsolete and ignored" in result.output
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

        with (
            patch("sparkrun.proxy.engine.ProxyEngine.is_running", fake_is_running),
            patch("sparkrun.proxy.engine.ProxyEngine._read_pid", fake_read_pid),
            patch("sparkrun.proxy.engine.ProxyEngine.stop", fake_stop),
            patch("sparkrun.proxy.engine.ProxyEngine._await_exit", return_value=False) as wait_exit,
            patch("sparkrun.proxy.engine.ProxyEngine.start", fake_start),
            patch("sparkrun.proxy.engine.write_config", return_value=Path("/tmp/cfg.yaml")),
        ):
            result = CliRunner().invoke(proxy, ["start", "--restart"])

        assert result.exit_code == 1, result.output
        assert "did not stop cleanly" in result.output
        assert state["stop_called"] is True
        assert state["start_called"] is False
        wait_exit.assert_called_once_with(999, 10.0)


@pytest.mark.skipif(sys.platform == "win32", reason="fixture uses a delayed POSIX SIGTERM handler")
@pytest.mark.parametrize("clear_on_signal", [False, True])
def test_restart_waits_for_database_lock_release(tmp_path, clear_on_signal):
    """A different CLI invocation must wait for exit, even if state disappears."""
    import sqlite3
    import subprocess
    import time

    from sparkrun.api.proxy._ops import _stop_and_wait
    from sparkrun.proxy._supervisor import GatewaySupervisor

    database = tmp_path / "credential-store.lock"
    ready = tmp_path / "ready"
    code = """
import signal, sqlite3, sys, time
from pathlib import Path
connection = sqlite3.connect(sys.argv[1])
connection.execute('BEGIN EXCLUSIVE')
def stop(*_):
    time.sleep(0.4)
    connection.close()
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
Path(sys.argv[2]).touch()
while True: time.sleep(0.02)
"""
    process = subprocess.Popen([sys.executable, "-c", code, str(database), str(ready)])
    try:
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert process.poll() is None, "fixture stopped before acquiring the lock"
            assert time.monotonic() < deadline, "fixture did not acquire the lock"
            time.sleep(0.01)

        class Engine(GatewaySupervisor):
            gateway_name = "fixture"

            def stop(self, dry_run=False):
                result = super().stop(dry_run=dry_run)
                if clear_on_signal:
                    self._clear_state()  # also cover asynchronous third-party engines
                return result

        engine = Engine(state_dir=tmp_path / "proxy")
        engine._save_state(process.pid)
        assert engine._proc is None  # daemon belongs to a previous CLI invocation
        assert _stop_and_wait(engine)
        assert not engine.state_file.exists()
        with sqlite3.connect(database, timeout=0) as replacement:
            replacement.execute("BEGIN EXCLUSIVE")
            replacement.rollback()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_restart_timeout_keeps_original_pid_for_retry(tmp_path):
    from sparkrun.api.proxy._ops import _stop_and_wait
    from sparkrun.proxy._supervisor import GatewaySupervisor

    engine = GatewaySupervisor(state_dir=tmp_path)
    engine._save_state(98765)
    with patch("os.kill"), patch.object(engine, "_await_exit", return_value=False) as wait_exit:
        assert _stop_and_wait(engine) is False
    assert engine.current_pid() == 98765
    assert [call.args for call in wait_exit.call_args_list] == [(98765, 0.0), (98765, 10.0)]
