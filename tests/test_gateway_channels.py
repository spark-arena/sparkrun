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
        ("alpha", {}, {}, "litellm"),
        ("alpha", {"gateway.sparkroute": False, "gateway.litellm": True}, {}, "litellm"),
        ("stable", {"gateway.sparkroute": True, "plugins.coldsnap": True}, {}, "litellm"),
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
        "plugin_loaded": False,
        "bridge_registered": False,
    }


@pytest.mark.parametrize("first_call", ["list", "resolve"])
@pytest.mark.parametrize(
    "channel,pin,expected", [("alpha", None, "litellm"), ("stable", "litellm", "litellm"), ("stable", None, "litellm")]
)
def test_public_selection_initializes_and_uses_saved_pin(tmp_path, channel, pin, expected, first_call):
    config_dir = tmp_path / ".config" / "sparkrun"
    config_dir.mkdir(parents=True)
    features = {"gateway.sparkroute": True, "gateway.litellm": True} if pin else {}
    (config_dir / "config.yaml").write_text(yaml.safe_dump({"self_update": {"channel": channel}, "features": features}))
    (config_dir / "proxy.yaml").write_text(yaml.safe_dump({"proxy": {"gateway": pin}}))
    code = """
import json, sys
from pathlib import Path
import sparkrun.core.config as config
config.DEFAULT_CONFIG_DIR = Path(sys.argv[1])
config.DEFAULT_CACHE_DIR = Path(sys.argv[2])
import sparkrun.core.registry as registry
registry.BOOTSTRAP_REGISTRY_URLS = []
from sparkrun import api
from sparkrun.application import initialize
first = api.proxy.list_gateways() if sys.argv[3] == 'list' else api.proxy.resolve_gateway()
context = initialize()
assert api.proxy.list_gateways() == api.proxy.list_gateways(sctx=context)
assert api.proxy.resolve_gateway() == api.proxy.resolve_gateway(sctx=context)
if context.config.is_feature_enabled('gateway.litellm'):
    assert api.proxy.resolve_gateway('litellm') == 'litellm'
else:
    try:
        api.proxy.resolve_gateway('litellm')
    except api.proxy.GatewayUnavailable:
        pass
    else:
        raise AssertionError('Disabled provider accepted')
try:
    api.proxy.resolve_gateway('unknown-provider')
except api.proxy.GatewayUnavailable:
    pass
else:
    raise AssertionError('Unknown provider accepted')
context.proxy_config.set_proxy(gateway='sparkroute')
context.proxy_config.save()
try:
    api.proxy.resolve_gateway(sctx=context)
except api.proxy.GatewayUnavailable as error:
    assert error.gateway == 'sparkroute'
else:
    raise AssertionError('Missing pinned provider silently replaced')
context.proxy_config.set_proxy(gateway='litellm')
context.proxy_config.save()
assert 'click' not in sys.modules and 'sparkrun.cli' not in sys.modules
print(json.dumps({'first': first, 'selected': api.proxy.resolve_gateway()}))
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARKRUN_") and k not in {"STATEFUL_ROOT", "RUN_ID", "RUN_SERIAL"}}
    env.update(STATEFUL_ROOT=str(tmp_path), RUN_ID=".config", RUN_SERIAL="", SPARKRUN_NO_TELEMETRY="1", SPARKRUN_NO_EXTERNAL_PLUGINS="1")
    result = subprocess.run(
        [sys.executable, "-c", code, str(config_dir), str(tmp_path / "cache"), first_call],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout)
    assert observed["selected"] == expected
    assert expected in observed["first"] if first_call == "list" else observed["first"] == expected


@pytest.mark.parametrize("operation", ["list_gateways", "resolve_gateway"])
def test_public_gateway_selection_wraps_bootstrap_failures(monkeypatch, operation):
    from sparkrun import api

    cause = RuntimeError("bootstrap failed")

    def fail():
        raise cause

    monkeypatch.setattr("sparkrun.application.initialize", fail)
    with pytest.raises(api.SparkrunError, match="Application initialization failed") as caught:
        getattr(api.proxy, operation)()
    assert caught.value.__cause__ is cause
