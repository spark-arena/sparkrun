"""Tests for stateless master-key auth (no DB env-vars)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestBuildLitellmConfigMasterKey:
    """Test build_litellm_config emits master_key without DB fields."""

    def test_master_key_set_emits_general_settings(self):
        """With master_key='secret', config has general_settings.master_key='secret'."""
        from sparkrun.proxy.engine import build_litellm_config

        config = build_litellm_config([], master_key="secret")

        assert "general_settings" in config
        assert config["general_settings"]["master_key"] == "secret"

    def test_master_key_set_no_database_keys(self):
        """With master_key, config has NO database_url / store_model_in_db keys."""
        from sparkrun.proxy.engine import build_litellm_config

        config = build_litellm_config([], master_key="secret")

        gen = config.get("general_settings", {})
        assert "database_url" not in gen
        assert "store_model_in_db" not in gen
        # Also check the top-level dict
        assert "database_url" not in config
        assert "store_model_in_db" not in config

    def test_master_key_none_no_general_settings(self):
        """With master_key=None, config has no general_settings.master_key."""
        from sparkrun.proxy.engine import build_litellm_config

        config = build_litellm_config([], master_key=None)

        # general_settings should be absent entirely (or contain no master_key)
        assert "master_key" not in config.get("general_settings", {})


class TestStartupEnvironmentNoDatabaseUrl:
    """Test that the litellm subprocess env never contains a database env var."""

    @pytest.fixture
    def state_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "proxy_state"
        d.mkdir()
        return d

    def _run_engine_start_capturing_env(
        self,
        state_dir: Path,
        master_key: str | None,
        enable_ui: bool = False,
        ui_username: str | None = None,
        ui_password: str | None = None,
    ):
        """Invoke ProxyEngine.start() and capture the env passed to Popen."""
        from sparkrun.proxy.engine import ProxyEngine

        captured_envs: list[dict] = []

        def fake_popen(*args, **kwargs):
            captured_envs.append(dict(kwargs.get("env") or {}))
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_proc.poll.return_value = None  # still running
            mock_proc.wait.return_value = 0
            return mock_proc

        engine = ProxyEngine(
            host="127.0.0.1",
            port=14123,
            master_key=master_key,
            state_dir=state_dir,
            enable_ui=enable_ui,
            ui_username=ui_username,
            ui_password=ui_password,
        )

        # Write a minimal litellm config to avoid path errors
        config_path = state_dir / "litellm_config.yaml"
        config_path.write_text("model_list: []\n")

        with (
            patch("sparkrun.proxy.engine.shutil.which", return_value="/usr/bin/uvx"),
            patch("sparkrun.proxy.engine.subprocess.Popen", side_effect=fake_popen),
            patch("time.sleep", lambda *_a, **_k: None),
        ):
            rc = engine.start(config_path=config_path, foreground=False)
        assert rc == 0
        assert captured_envs, "Popen was not called"
        return captured_envs[0]

    def test_no_database_url_when_master_key_set_ui_disabled(self, state_dir: Path):
        """ProxyEngine.start() never sets DATABASE_URL with master_key when enable_ui=False."""
        env = self._run_engine_start_capturing_env(state_dir, master_key="secret", enable_ui=False)
        # No env var name should contain DATABASE or DB in a setting role.
        # The narrow check the spec requires: DATABASE_URL is absent.
        assert "DATABASE_URL" not in env

    def test_no_database_url_when_master_key_none_ui_disabled(self, state_dir: Path):
        """ProxyEngine.start() never sets DATABASE_URL when master_key=None and enable_ui=False."""
        env = self._run_engine_start_capturing_env(state_dir, master_key=None, enable_ui=False)
        assert "DATABASE_URL" not in env


class TestEnableUiEnvironment:
    """Test that enable_ui re-enables DATABASE_URL and UI credentials in env."""

    @pytest.fixture
    def state_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "proxy_state"
        d.mkdir()
        return d

    def _run_engine_start_capturing_env(
        self,
        state_dir: Path,
        master_key: str | None,
        enable_ui: bool = False,
        ui_username: str | None = None,
        ui_password: str | None = None,
    ):
        """Invoke ProxyEngine.start() and capture the env passed to Popen."""
        from sparkrun.proxy.engine import ProxyEngine

        captured_envs: list[dict] = []

        def fake_popen(*args, **kwargs):
            captured_envs.append(dict(kwargs.get("env") or {}))
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_proc.poll.return_value = None
            mock_proc.wait.return_value = 0
            return mock_proc

        engine = ProxyEngine(
            host="127.0.0.1",
            port=14123,
            master_key=master_key,
            state_dir=state_dir,
            enable_ui=enable_ui,
            ui_username=ui_username,
            ui_password=ui_password,
        )

        config_path = state_dir / "litellm_config.yaml"
        config_path.write_text("model_list: []\n")

        with (
            patch("sparkrun.proxy.engine.shutil.which", return_value="/usr/bin/uvx"),
            patch("sparkrun.proxy.engine.subprocess.Popen", side_effect=fake_popen),
            patch("time.sleep", lambda *_a, **_k: None),
        ):
            rc = engine.start(config_path=config_path, foreground=False)
        assert rc == 0
        assert captured_envs, "Popen was not called"
        return captured_envs[0]

    def test_enable_ui_with_master_key_sets_database_and_ui_defaults(self, state_dir: Path):
        """enable_ui=True + master_key='secret' sets DATABASE_URL, UI_USERNAME='admin', UI_PASSWORD='secret'."""
        env = self._run_engine_start_capturing_env(state_dir, master_key="secret", enable_ui=True)
        expected_db = "sqlite:///%s" % (state_dir / "litellm.db")
        assert env.get("DATABASE_URL") == expected_db
        assert env.get("UI_USERNAME") == "admin"
        assert env.get("UI_PASSWORD") == "secret"

    def test_enable_ui_with_custom_ui_creds_overrides_master_key(self, state_dir: Path):
        """Custom ui_username and ui_password are used, NOT the master_key."""
        env = self._run_engine_start_capturing_env(
            state_dir,
            master_key="secret",
            enable_ui=True,
            ui_username="ops",
            ui_password="opspw",
        )
        assert env.get("UI_USERNAME") == "ops"
        assert env.get("UI_PASSWORD") == "opspw"
        # Master key should NOT leak into UI_PASSWORD when explicit value given.
        assert env.get("UI_PASSWORD") != "secret"

    def test_enable_ui_without_master_key_raises(self, state_dir: Path):
        """enable_ui=True with master_key=None raises ValueError at construction time."""
        from sparkrun.proxy.engine import ProxyEngine

        with pytest.raises(ValueError, match="enable_ui requires master_key"):
            ProxyEngine(
                host="127.0.0.1",
                port=14123,
                master_key=None,
                state_dir=state_dir,
                enable_ui=True,
            )


class TestEnableUiBuildLitellmConfig:
    """Test that build_litellm_config still emits master_key and never adds store_model_in_db."""

    def test_master_key_still_emitted_when_ui_mode(self):
        """LiteLLM config YAML still has general_settings.master_key when UI mode is on."""
        from sparkrun.proxy.engine import build_litellm_config

        # build_litellm_config takes master_key directly — UI mode does not change YAML.
        config = build_litellm_config([], master_key="secret")
        assert config["general_settings"]["master_key"] == "secret"

    def test_no_store_model_in_db_regardless_of_ui(self):
        """build_litellm_config never emits store_model_in_db, with or without UI."""
        from sparkrun.proxy.engine import build_litellm_config

        for mk in (None, "secret"):
            config = build_litellm_config([], master_key=mk)
            assert "store_model_in_db" not in config
            assert "store_model_in_db" not in config.get("general_settings", {})
            assert "store_model_in_db" not in config.get("litellm_settings", {})
