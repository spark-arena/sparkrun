"""Exercise gateway defaults through real bootstrap in a fresh process."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import yaml


@pytest.mark.parametrize(
    "channel,overrides,env_flags,expected",
    [
        ("stable", {}, {}, "litellm"),
        ("beta", {}, {}, "litellm"),
        ("alpha", {}, {}, "sparkroute"),
        ("alpha", {"gateway.sparkroute": False, "gateway.litellm": True}, {}, "litellm"),
        ("stable", {"gateway.sparkroute": True, "gateway.litellm": False}, {}, "sparkroute"),
        ("alpha", {}, {"SPARKRUN_FEATURE_GATEWAY_SPARKROUTE": "0", "SPARKRUN_FEATURE_GATEWAY_LITELLM": "1"}, "litellm"),
    ],
)
def test_channel_defaults_and_explicit_overrides(tmp_path, channel, overrides, env_flags, expected):
    # SAF's desktop bootstrap appends .config/sparkrun to STATEFUL_ROOT.
    config_dir = tmp_path / ".config" / "sparkrun"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump({"self_update": {"channel": channel}, "features": overrides}))
    snippet = """
import json, sys
from pathlib import Path
import sparkrun.core.config as config
config.DEFAULT_CONFIG_DIR = Path(sys.argv[1])
config.DEFAULT_CACHE_DIR = Path(sys.argv[2])
import sparkrun.core.registry as registry
registry.BOOTSTRAP_REGISTRY_URLS = []
from sparkrun.core.bootstrap import init_sparkrun
from sparkrun.core.cli_registry import registered_cli_commands
from sparkrun.proxy.gateway import list_gateways, resolve_gateway
init_sparkrun()
print(json.dumps({
    "available": list_gateways(), "selected": resolve_gateway(),
    "plugin_loaded": "sparkrun.plugins.sparkroute" in sys.modules,
    "bridge_registered": any(spec.name == "gateway-bridge" for spec in registered_cli_commands()),
}))
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARKRUN_FEATURE_") and k != "STATEFUL_ROOT"}
    env.update(
        STATEFUL_ROOT=str(tmp_path),
        RUN_ID=".config",
        RUN_SERIAL="",
        SPARKRUN_NO_TELEMETRY="1",
        SPARKRUN_NO_EXTERNAL_PLUGINS="1",
        **env_flags,
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet, str(config_dir), str(tmp_path / "cache")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout.splitlines()[-1])
    assert observed == {
        "available": [expected],
        "selected": expected,
        "plugin_loaded": expected == "sparkroute",
        "bridge_registered": expected == "sparkroute",
    }
