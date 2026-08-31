"""Tests for out-of-tree plugin loading (``plugins.paths``).

Concrete plugins here subclass :class:`Transport` — the lightest SAF plugin
base (no abstract methods) — so the tests exercise the discovery/registration
machinery without standing up a full runtime/executor.  Module names carry a
per-test token because ``sys.modules`` caches by name process-globally.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from sparkrun.core.bootstrap import init_sparkrun
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.external_plugins import load_external_plugins
from sparkrun.transports import list_transports


def _write(dir_path: Path, name: str, body: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / ("%s.py" % name)).write_text(textwrap.dedent(body))


def _transport_module(transport_name: str) -> str:
    return (
        """
        from sparkrun.transports.base import Transport

        class _Ext(Transport):
            transport_name = %r
    """
        % transport_name
    )


@pytest.fixture
def clean_sys():
    """Isolate temp-plugin side effects from the rest of the suite.

    SAF's plugin registry lives on the process-global Variables instance
    (see the ``saf-registry-global-test-gotcha`` note), so transports these
    tests register would otherwise leak into other tests (e.g.
    ``test_transports.test_registry_lists_builtins``).  We snapshot the
    transport extension dict — the exact structure ``get_extensions`` reads —
    plus ``sys.path`` / ``sys.modules``, and prune anything added on teardown.
    """
    from scitrera_app_framework.core.plugins import _multi_ext_options

    from sparkrun.core.bootstrap import get_variables
    from sparkrun.transports.base import EXT_TRANSPORT

    v = get_variables()
    transport_eor = _multi_ext_options(EXT_TRANSPORT, v)
    saved_transports = set(transport_eor)
    saved_path = list(sys.path)
    saved_mods = set(sys.modules)
    yield
    for name in set(transport_eor) - saved_transports:
        transport_eor.pop(name, None)
    sys.path[:] = saved_path
    for mod in set(sys.modules) - saved_mods:
        sys.modules.pop(mod, None)


def test_explicit_empty_paths_is_noop():
    v = init_sparkrun()
    assert load_external_plugins(v, paths=[]) == []


def test_kill_switch_disables_auto_load(monkeypatch):
    # conftest sets SPARKRUN_NO_EXTERNAL_PLUGINS=1; even with the feature flag
    # forced on, the hard kill-switch wins and the config-driven path is inert.
    monkeypatch.setenv("SPARKRUN_FEATURE_CORE_EXTERNAL_PLUGINS", "1")
    v = init_sparkrun()
    assert load_external_plugins(v) == []


def test_auto_load_disabled_when_flag_off(monkeypatch):
    # With the kill-switch cleared, the config-driven path is still inert while
    # core.external_plugins resolves off (its default on every channel).
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS", raising=False)
    monkeypatch.setenv("SPARKRUN_FEATURE_CORE_EXTERNAL_PLUGINS", "0")
    v = init_sparkrun()
    assert load_external_plugins(v) == []


def test_gate_off_does_not_read_configured_paths(tmp_path, clean_sys, monkeypatch):
    # Flag off must short-circuit BEFORE reading plugins.paths: a configured
    # (and valid) plugin dir is ignored entirely.
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS", raising=False)
    monkeypatch.setenv("SPARKRUN_FEATURE_CORE_EXTERNAL_PLUGINS", "0")
    v = init_sparkrun()
    cfg_root = tmp_path / "cfg"
    cfg_root.mkdir()
    plug = tmp_path / "plugins"
    _write(plug, "ext_gated_z", _transport_module("extprov_gated_z"))
    (cfg_root / "config.yaml").write_text("plugins:\n  paths:\n    - %s\n" % plug)
    monkeypatch.setattr("sparkrun.core.config.get_config_root", lambda v=None: cfg_root)

    assert load_external_plugins(v) == []
    assert "extprov_gated_z" not in list_transports(v)


def test_single_file_plugin_registered(tmp_path, clean_sys):
    v = init_sparkrun()
    plug = tmp_path / "plugins"
    _write(plug, "ext_single_a", _transport_module("extprov_single_a"))

    loaded = load_external_plugins(v, paths=[plug])

    assert "ext_single_a" in loaded
    assert "extprov_single_a" in list_transports(v)


def test_package_plugin_recursively_scanned(tmp_path, clean_sys):
    v = init_sparkrun()
    plug = tmp_path / "plugins"
    pkg = plug / "ext_pkg_b"
    _write(pkg, "__init__", "")
    # Class lives in a submodule, not __init__ — exercises package recursion.
    _write(pkg, "provider", _transport_module("extprov_pkg_b"))

    loaded = load_external_plugins(v, paths=[plug])

    assert "ext_pkg_b" in loaded
    assert "extprov_pkg_b" in list_transports(v)


def test_register_hook_invoked_with_variables(tmp_path, clean_sys):
    v = init_sparkrun()
    plug = tmp_path / "plugins"
    _write(
        plug,
        "ext_hook_c",
        """
        RECEIVED = []

        def register(v):
            RECEIVED.append(v)
        """,
    )

    load_external_plugins(v, paths=[plug])

    mod = importlib.import_module("ext_hook_c")
    assert mod.RECEIVED == [v]


def test_broken_module_skipped_others_still_load(tmp_path, clean_sys):
    v = init_sparkrun()
    plug = tmp_path / "plugins"
    _write(plug, "ext_broken_d", "raise RuntimeError('boom at import')")
    _write(plug, "ext_good_d", _transport_module("extprov_good_d"))

    loaded = load_external_plugins(v, paths=[plug])

    assert "ext_broken_d" not in loaded
    assert "ext_good_d" in loaded
    assert "extprov_good_d" in list_transports(v)


def test_underscore_and_missing_paths_ignored(tmp_path, clean_sys):
    v = init_sparkrun()
    plug = tmp_path / "plugins"
    _write(plug, "_private_e", _transport_module("extprov_private_e"))
    missing = tmp_path / "does_not_exist"

    loaded = load_external_plugins(v, paths=[plug, missing])

    assert loaded == []  # underscore module skipped, missing dir skipped
    assert "extprov_private_e" not in list_transports(v)


def test_config_driven_auto_load(tmp_path, clean_sys, monkeypatch):
    v = init_sparkrun()
    cfg_root = tmp_path / "cfg"
    cfg_root.mkdir()
    plug = tmp_path / "myplugins"
    _write(plug, "ext_cfg_f", _transport_module("extprov_cfg_f"))
    (cfg_root / "config.yaml").write_text("plugins:\n  paths:\n    - %s\n" % plug)

    monkeypatch.setattr("sparkrun.core.config.get_config_root", lambda v=None: cfg_root)
    monkeypatch.delenv("SPARKRUN_NO_EXTERNAL_PLUGINS", raising=False)
    monkeypatch.setenv("SPARKRUN_FEATURE_CORE_EXTERNAL_PLUGINS", "1")

    loaded = load_external_plugins(v)  # paths=None -> config-driven

    assert "ext_cfg_f" in loaded
    assert "extprov_cfg_f" in list_transports(v)


def test_external_plugin_paths_config_parsing(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("plugins:\n  paths:\n    - ~/p1\n    - /abs/p2\n")
    config = SparkrunConfig(config_path=cfg)

    paths = config.external_plugin_paths

    assert paths == [Path.home() / "p1", Path("/abs/p2")]


def test_external_plugin_paths_defaults_empty(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("defaults:\n  executor: docker\n")
    assert SparkrunConfig(config_path=cfg).external_plugin_paths == []


def test_plugin_settings_returns_isolated_plugin_mapping(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("plugins:\n  paths: /plugins\n  coldsnap:\n    artifact_generations: 2\n")
    config = SparkrunConfig(config_path=cfg)

    settings = config.plugin_settings("coldsnap")
    settings["artifact_generations"] = 99

    assert config.plugin_settings("coldsnap") == {"artifact_generations": 2}
    assert config.plugin_settings("missing") == {}
    assert config.external_plugin_paths == [Path("/plugins")]


def test_feature_flag_registered_and_off_by_default():
    from sparkrun.core.channels import CHANNEL_ALPHA, CHANNEL_BETA, CHANNEL_STABLE
    from sparkrun.core.features import get_feature

    flag = get_feature("core.external_plugins")
    assert flag is not None
    for channel in (CHANNEL_STABLE, CHANNEL_BETA, CHANNEL_ALPHA):
        assert flag.default_for_channel(channel) is False
