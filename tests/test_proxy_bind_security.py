"""Tests for proxy bind-host security hardening (A3) and secret-file perms (A4)."""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch


def _start_engine_capturing(engine, state_dir: Path):
    """Run engine.start() with a mocked litellm subprocess.

    Returns (rc, captured_cmds).
    """
    captured_cmds: list[list[str]] = []

    def fake_popen(cmd, *args, **kwargs):
        captured_cmds.append(list(cmd))
        mock_proc = MagicMock()
        mock_proc.pid = 4321
        mock_proc.poll.return_value = None  # survived startup
        mock_proc.wait.return_value = 0
        return mock_proc

    state_dir.mkdir(parents=True, exist_ok=True)
    config_path = state_dir / "litellm_config.yaml"
    config_path.write_text("model_list: []\n")

    with (
        patch("sparkrun.proxy.engine.shutil.which", return_value="/usr/bin/uvx"),
        patch("sparkrun.proxy.engine.subprocess.Popen", side_effect=fake_popen),
        patch("time.sleep", lambda *_a, **_k: None),
    ):
        rc = engine.start(config_path=config_path, foreground=False)
    return rc, captured_cmds


# ---------------------------------------------------------------------------
# A4: secret-file permissions
# ---------------------------------------------------------------------------


class TestSecretFilePermissions:
    def test_state_file_is_owner_only(self, tmp_path: Path):
        """_save_state writes state.yaml with 0o600 perms and 0o700 parent dir."""
        from sparkrun.proxy.engine import ProxyEngine

        state_dir = tmp_path / "proxy"
        engine = ProxyEngine(master_key="secret", state_dir=state_dir)
        engine._save_state(pid=111)

        mode = stat.S_IMODE(os.stat(engine.state_file).st_mode)
        assert mode == 0o600, "state.yaml should be owner-only, got %o" % mode

        dir_mode = stat.S_IMODE(os.stat(state_dir).st_mode)
        assert dir_mode == 0o700, "state dir should be owner-only, got %o" % dir_mode

    def test_autodiscover_file_is_owner_only(self, tmp_path: Path):
        """start_autodiscover writes autodiscover.yaml with 0o600 perms."""
        from sparkrun.proxy.engine import ProxyEngine

        state_dir = tmp_path / "proxy"
        engine = ProxyEngine(master_key="secret", state_dir=state_dir)

        with patch("sparkrun.proxy.engine.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(pid=222)
            engine.start_autodiscover(proxy_pid=111, interval=5)

        cfg_path = engine._autodiscover_config_path
        assert cfg_path.exists()
        mode = stat.S_IMODE(os.stat(cfg_path).st_mode)
        assert mode == 0o600, "autodiscover.yaml should be owner-only, got %o" % mode

        dir_mode = stat.S_IMODE(os.stat(state_dir).st_mode)
        assert dir_mode == 0o700

    def test_litellm_config_is_owner_only(self, tmp_path: Path):
        """write_config writes litellm_config.yaml 0o600 with a 0o700 parent.

        The litellm config carries general_settings.master_key plus every
        upstream endpoint api_key, so it must be owner-only at rest.
        """
        from sparkrun.proxy.engine import write_config

        config_path = tmp_path / "proxy" / "litellm_config.yaml"
        config = {
            "model_list": [{"model_name": "m", "litellm_params": {"api_key": "upstream-secret"}}],
            "general_settings": {"master_key": "bearer-secret"},
        }
        write_config(config, config_path=config_path)

        mode = stat.S_IMODE(os.stat(config_path).st_mode)
        assert mode == 0o600, "litellm_config.yaml should be owner-only, got %o" % mode
        dir_mode = stat.S_IMODE(os.stat(config_path.parent).st_mode)
        assert dir_mode == 0o700, "proxy config dir should be owner-only, got %o" % dir_mode

    def test_litellm_config_repairs_preexisting_world_readable_file(self, tmp_path: Path):
        """A pre-existing 0o644 config (older version) is tightened to 0o600."""
        from sparkrun.proxy.engine import write_config

        config_dir = tmp_path / "proxy"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "litellm_config.yaml"
        config_path.write_text("model_list: []\n")
        os.chmod(config_path, 0o644)

        write_config({"model_list": []}, config_path=config_path)

        mode = stat.S_IMODE(os.stat(config_path).st_mode)
        assert mode == 0o600, "stale world-readable config should be tightened, got %o" % mode


# ---------------------------------------------------------------------------
# A3: persisted bind host + loud warning
# ---------------------------------------------------------------------------


class TestBindHostConfig:
    def test_host_configured_false_when_unset(self, tmp_path: Path):
        """ProxyConfig.host_configured is False with no persisted host."""
        from sparkrun.proxy.config import ProxyConfig

        cfg = ProxyConfig(config_path=tmp_path / "proxy.yaml")
        assert cfg.host_configured is False
        assert cfg.host == "0.0.0.0"  # legacy default preserved

    def test_host_configured_true_when_persisted(self, tmp_path: Path):
        """Explicit bind host is persisted and reloaded as configured."""
        from sparkrun.proxy.config import ProxyConfig

        path = tmp_path / "proxy.yaml"
        cfg = ProxyConfig(config_path=path)
        cfg.set_proxy(host="127.0.0.1")
        cfg.save()

        # Reload from disk to prove persistence.
        reloaded = ProxyConfig(config_path=path)
        assert reloaded.host_configured is True
        assert reloaded.host == "127.0.0.1"


class TestLoudWarning:
    def test_warns_when_unconfigured_and_no_master_key(self, tmp_path: Path, caplog):
        """Loud unauthenticated warning fires on start when nothing configured."""
        from sparkrun.proxy.engine import ProxyEngine

        state_dir = tmp_path / "proxy"
        engine = ProxyEngine(host="0.0.0.0", master_key=None, state_dir=state_dir, host_configured=False)

        with caplog.at_level(logging.WARNING, logger="sparkrun.proxy.engine"):
            rc, _cmds = _start_engine_capturing(engine, state_dir)

        assert rc == 0
        text = caplog.text
        assert "SECURITY WARNING" in text
        assert "0.0.0.0" in text
        assert "NO authentication" in text

    def test_softer_warning_when_master_key_set(self, tmp_path: Path, caplog):
        """With a master key, the warning notes auth is enabled, not unauthenticated."""
        from sparkrun.proxy.engine import ProxyEngine

        state_dir = tmp_path / "proxy"
        engine = ProxyEngine(host="0.0.0.0", master_key="secret", state_dir=state_dir, host_configured=False)

        with caplog.at_level(logging.WARNING, logger="sparkrun.proxy.engine"):
            rc, _cmds = _start_engine_capturing(engine, state_dir)

        assert rc == 0
        text = caplog.text
        assert "Authentication IS enabled" in text
        assert "NO authentication" not in text

    def test_no_warning_when_host_explicitly_configured(self, tmp_path: Path, caplog):
        """Explicit bind host overrides the legacy default and silences the warning."""
        from sparkrun.proxy.engine import ProxyEngine

        state_dir = tmp_path / "proxy"
        engine = ProxyEngine(host="127.0.0.1", master_key=None, state_dir=state_dir, host_configured=True)

        with caplog.at_level(logging.WARNING, logger="sparkrun.proxy.engine"):
            rc, cmds = _start_engine_capturing(engine, state_dir)

        assert rc == 0
        assert "SECURITY WARNING" not in caplog.text
        # Explicit bind host is wired into the litellm launch command.
        assert cmds, "Popen was not called"
        assert "127.0.0.1" in cmds[0]
        assert "--host" in cmds[0]


class TestPersistedBindHostReused:
    def test_explicit_bind_host_persisted_and_reused_in_cmd(self, tmp_path: Path):
        """A persisted bind host flows into the litellm launch command on restart."""
        from sparkrun.proxy.config import ProxyConfig
        from sparkrun.proxy.engine import ProxyEngine

        cfg_path = tmp_path / "proxy.yaml"
        cfg = ProxyConfig(config_path=cfg_path)
        cfg.set_proxy(host="127.0.0.1")
        cfg.save()

        reloaded = ProxyConfig(config_path=cfg_path)
        state_dir = tmp_path / "proxy"
        engine = ProxyEngine(
            host=reloaded.host,
            master_key=None,
            state_dir=state_dir,
            host_configured=reloaded.host_configured,
        )

        rc, cmds = _start_engine_capturing(engine, state_dir)
        assert rc == 0
        assert "127.0.0.1" in cmds[0]
        assert "0.0.0.0" not in cmds[0]
