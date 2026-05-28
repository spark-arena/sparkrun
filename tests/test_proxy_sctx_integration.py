"""Tests for ProxyConfig integration with SparkrunContext."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch):
    """Point SparkrunConfig at a temp config path so no real ~/.config writes happen."""
    config_path = tmp_path / "config.yaml"
    # Also redirect proxy.yaml default location
    proxy_path = tmp_path / "proxy.yaml"
    monkeypatch.setattr(
        "sparkrun.core.config.DEFAULT_CONFIG_DIR",
        tmp_path,
    )
    yield config_path, proxy_path


class TestSparkrunConfigProxyFactory:
    """Test SparkrunConfig.get_proxy_config() factory."""

    def test_returns_proxy_config_instance(self, isolated_config):
        """Factory returns a ProxyConfig instance."""
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.proxy.config import ProxyConfig

        config = SparkrunConfig()
        result = config.get_proxy_config()
        assert isinstance(result, ProxyConfig)

    def test_returns_cached_instance(self, isolated_config):
        """Factory returns the same instance on repeated access."""
        from sparkrun.core.config import SparkrunConfig

        config = SparkrunConfig()
        a = config.get_proxy_config()
        b = config.get_proxy_config()
        assert a is b


class TestSparkrunContextProxyConfig:
    """Test SparkrunContext.proxy_config cached property."""

    def test_proxy_config_is_cached(self, isolated_config):
        """sctx.proxy_config returns the same instance on repeated access."""
        from sparkrun.api import default_sctx

        sctx = default_sctx()
        a = sctx.proxy_config
        b = sctx.proxy_config
        assert a is b

    def test_proxy_config_delegates_to_config(self, isolated_config):
        """sctx.proxy_config returns the same instance as sctx.config.get_proxy_config()."""
        from sparkrun.api import default_sctx

        sctx = default_sctx()
        from_sctx = sctx.proxy_config
        from_config = sctx.config.get_proxy_config()
        assert from_sctx is from_config


class TestCLIProxyConfigViaContext:
    """Test that CLI proxy commands resolve config via _get_context."""

    def test_alias_list_uses_sctx_proxy_config(self, isolated_config):
        """`sparkrun proxy alias list` reads from sctx.proxy_config."""
        from sparkrun.cli import main
        from sparkrun.proxy.config import ProxyConfig

        # Build a stub proxy config with a known alias
        stub_proxy_cfg = ProxyConfig(isolated_config[1])
        stub_proxy_cfg.add_alias("stub-alias", "stub-target/model")
        stub_proxy_cfg.save()

        # Patch _get_context to return a sctx whose proxy_config is our stub
        from sparkrun.cli._common import _get_context as real_get_context

        def fake_get_context(ctx):
            sctx = real_get_context(ctx)
            # Override the cached property by assigning directly.  Because
            # SparkrunContext is a dataclass, this works for the cached_property
            # via the underlying __dict__.
            object.__setattr__(sctx, "proxy_config", stub_proxy_cfg)
            return sctx

        with patch("sparkrun.cli._proxy._get_context", side_effect=fake_get_context):
            runner = CliRunner()
            result = runner.invoke(main, ["proxy", "alias", "list"])

        assert result.exit_code == 0, result.output
        assert "stub-alias" in result.output
        assert "stub-target/model" in result.output
