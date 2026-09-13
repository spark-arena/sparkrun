"""Tests for the plugin inventory behind ``sparkrun setup plugins list``.

Two properties carry the weight here:

* The inventory and the loaders agree on **what a plugin is** — a listing that
  names something the loader would skip (or omits one it loads) is read as an
  answer and is worse than no listing.
* A version is either **declared or unknown**, never inferred. Specifically an
  in-tree plugin that declares nothing must not inherit sparkrun's own version
  via the distribution fallback.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from sparkrun.core import in_tree_plugins as in_tree_module
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.external_plugins import (
    clear_loaded_plugin_modules,
    iter_plugin_module_names,
    load_external_plugins,
)
from sparkrun.core.features import FEATURE_FLAGS, FeatureFlag, register_feature
from sparkrun.core.in_tree_plugins import (
    IN_TREE_PLUGIN_FEATURES,
    iter_in_tree_plugin_names,
    load_in_tree_plugins,
)
from sparkrun.core.plugin_inventory import (
    SOURCE_EXTERNAL,
    SOURCE_IN_TREE,
    VERSION_FROM_MODULE,
    list_plugins,
)


@pytest.fixture(autouse=True)
def _forget_loaded_plugins():
    """The loaded-module record is process-global; keep it per-test."""
    clear_loaded_plugin_modules()
    yield
    clear_loaded_plugin_modules()


@pytest.fixture
def fake_in_tree(tmp_path, monkeypatch):
    """An importable throwaway in-tree plugin package, wired as the real one."""
    root = tmp_path / "inv_plugins"
    root.mkdir()
    (root / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(in_tree_module, "IN_TREE_PLUGIN_PACKAGE", "inv_plugins")

    bound: list[str] = []
    flags: list[str] = []

    def add(name: str, body: str = "", *, flag: bool | None = True) -> None:
        pkg = root / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text(textwrap.dedent(body))
        if flag is None:
            return
        flag_name = "test.inv.%s" % name
        IN_TREE_PLUGIN_FEATURES[name] = flag_name
        register_feature(FeatureFlag(name=flag_name, description="test", default=flag))
        bound.append(name)
        flags.append(flag_name)

    yield add

    for flag_name in flags:
        FEATURE_FLAGS.pop(flag_name, None)
    for name in bound:
        IN_TREE_PLUGIN_FEATURES.pop(name, None)
    for mod in [m for m in sys.modules if m.startswith("inv_plugins")]:
        sys.modules.pop(mod, None)


def _external_dir(tmp_path: Path, name: str, body: str) -> Path:
    plugin_dir = tmp_path / "ext"
    plugin_dir.mkdir(exist_ok=True)
    (plugin_dir / ("%s.py" % name)).write_text(textwrap.dedent(body))
    return plugin_dir


def _config_with_paths(tmp_path: Path, *paths: Path) -> SparkrunConfig:
    config_path = tmp_path / "config.yaml"
    config = SparkrunConfig(config_path=config_path)
    config.set("plugins", {"paths": [str(p) for p in paths]})
    return config


def _by_name(plugins):
    return {p.name: p for p in plugins}


# ---------------------------------------------------------------------------
# Agreement with the loaders
# ---------------------------------------------------------------------------


def test_inventory_lists_exactly_what_the_in_tree_loader_would_load(fake_in_tree):
    fake_in_tree("alpha", "def register(v): pass")
    fake_in_tree("beta", "def register(v): pass")
    fake_in_tree("_private", "raise AssertionError('never imported')", flag=None)

    assert iter_in_tree_plugin_names() == ["alpha", "beta"]
    assert sorted(load_in_tree_plugins(None)) == ["alpha", "beta"]
    assert sorted(p.name for p in list_plugins()) == ["alpha", "beta"]


def test_external_enumeration_matches_what_the_loader_imports(tmp_path):
    plugin_dir = _external_dir(tmp_path, "inv_ext_match", "__version__ = '1.0'")
    (plugin_dir / "_skipme.py").write_text("raise AssertionError('never imported')")

    assert iter_plugin_module_names(plugin_dir) == ["inv_ext_match"]
    assert load_external_plugins(None, paths=[plugin_dir]) == ["inv_ext_match"]

    config = _config_with_paths(tmp_path, plugin_dir)
    assert [p.name for p in list_plugins(config=config) if p.source == SOURCE_EXTERNAL] == ["inv_ext_match"]


def test_a_plugin_with_no_binding_is_listed_off_rather_than_omitted(fake_in_tree):
    """The defect must be visible; omitting it is how it stays silent."""
    fake_in_tree("unbound", "def register(v): pass", flag=None)

    info = _by_name(list_plugins())["unbound"]
    assert info.feature_flag is None
    assert info.enabled is False
    assert info.loaded is False


# ---------------------------------------------------------------------------
# Versions: declared or unknown, never inferred
# ---------------------------------------------------------------------------


def test_version_is_read_from_the_loaded_module(fake_in_tree):
    fake_in_tree("versioned", "__version__ = '2.5.1'\n\ndef register(v): pass")
    load_in_tree_plugins(None)

    info = _by_name(list_plugins())["versioned"]
    assert (info.version, info.version_source) == ("2.5.1", VERSION_FROM_MODULE)
    assert info.version_display == "2.5.1"
    assert info.loaded is True


def test_an_in_tree_plugin_never_inherits_sparkruns_own_version(fake_in_tree):
    """The distribution fallback is out-of-tree only.

    Every in-tree plugin's top-level package maps to the ``sparkrun``
    distribution, so applying the fallback here would report sparkrun's version
    as the plugin's — a fabricated answer where the honest one is "unknown".
    """
    import sparkrun

    fake_in_tree("undeclared", "def register(v): pass")
    load_in_tree_plugins(None)

    info = _by_name(list_plugins())["undeclared"]
    assert info.loaded is True
    assert info.version is None
    assert info.version_source is None
    assert info.version_display == "unknown"
    assert getattr(sparkrun, "__version__", None) != info.version_display


def test_a_blank_version_attribute_reads_as_unknown(fake_in_tree):
    fake_in_tree("blank", "__version__ = '   '\n\ndef register(v): pass")
    load_in_tree_plugins(None)

    assert _by_name(list_plugins())["blank"].version is None


def test_an_unloaded_plugin_reports_unknown_without_importing(fake_in_tree):
    """Listing must never import a plugin the user has switched off."""
    fake_in_tree("disabled", "__version__ = '9.9'\n\nraise AssertionError('imported')", flag=False)

    info = _by_name(list_plugins())["disabled"]
    assert info.enabled is False
    assert info.loaded is False
    assert info.version is None
    assert "inv_plugins.disabled" not in sys.modules


def test_a_version_is_reported_only_for_a_module_we_loaded_as_a_plugin(tmp_path, monkeypatch):
    """``sys.modules`` is not the record.

    An external plugin's top-level name may be importable for unrelated
    reasons; attributing that module's version to a plugin sparkrun never
    loaded is a wrong answer, not a missing one.
    """
    import types

    stranger = types.ModuleType("inv_ext_stranger")
    stranger.__version__ = "6.6.6"
    monkeypatch.setitem(sys.modules, "inv_ext_stranger", stranger)

    plugin_dir = _external_dir(tmp_path, "inv_ext_stranger", "__version__ = '1.2.3'")
    config = _config_with_paths(tmp_path, plugin_dir)

    info = _by_name(list_plugins(config=config))["inv_ext_stranger"]
    assert info.loaded is False
    assert info.version is None


# ---------------------------------------------------------------------------
# Sources and shape
# ---------------------------------------------------------------------------


def test_external_plugins_are_omitted_without_a_config(tmp_path):
    plugin_dir = _external_dir(tmp_path, "inv_ext_nocfg", "__version__ = '1.0'")
    load_external_plugins(None, paths=[plugin_dir])

    assert all(p.source == SOURCE_IN_TREE for p in list_plugins())


def test_external_rows_carry_their_source_path_and_gate(tmp_path):
    plugin_dir = _external_dir(tmp_path, "inv_ext_pathed", "__version__ = '0.4'")
    load_external_plugins(None, paths=[plugin_dir])
    config = _config_with_paths(tmp_path, plugin_dir)

    info = _by_name(list_plugins(config=config))["inv_ext_pathed"]
    assert info.source == SOURCE_EXTERNAL
    assert info.module == "inv_ext_pathed"
    assert info.path == plugin_dir
    assert info.feature_flag == "core.external_plugins"
    assert (info.version, info.version_source) == ("0.4", VERSION_FROM_MODULE)


def test_a_missing_plugins_path_contributes_no_rows(tmp_path):
    config = _config_with_paths(tmp_path, tmp_path / "does-not-exist")
    assert [p for p in list_plugins(config=config) if p.source == SOURCE_EXTERNAL] == []


def test_in_tree_plugins_sort_before_external_ones(tmp_path, fake_in_tree):
    fake_in_tree("zzz_in_tree", "def register(v): pass")
    plugin_dir = _external_dir(tmp_path, "aaa_external", "pass")
    config = _config_with_paths(tmp_path, plugin_dir)

    assert [p.name for p in list_plugins(config=config)] == ["zzz_in_tree", "aaa_external"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    from click.testing import CliRunner

    return CliRunner()


def test_bare_group_defaults_to_list(runner, fake_in_tree):
    from sparkrun.cli import main

    fake_in_tree("cli_default", "__version__ = '3.1'\n\ndef register(v): pass")
    load_in_tree_plugins(None)

    bare = runner.invoke(main, ["setup", "plugins"])
    listed = runner.invoke(main, ["setup", "plugins", "list"])

    assert bare.exit_code == 0, bare.output
    assert listed.exit_code == 0, listed.output
    assert bare.output == listed.output
    assert "cli_default" in bare.output
    assert "3.1" in bare.output


def test_the_group_is_functional_even_when_hidden(runner, monkeypatch, fake_in_tree):
    """Visibility-only gate: ``SPARKRUN_ADVANCED`` decides ``--help``, not access.

    Someone debugging a plugin that will not load needs this command whether or
    not they run with advanced options on.
    """
    import sparkrun.cli._setup._plugins as plugins_cli

    from sparkrun.cli import main

    monkeypatch.setattr(plugins_cli.setup_plugins, "hidden", True)
    fake_in_tree("cli_hidden", "def register(v): pass")

    assert "plugins" not in runner.invoke(main, ["setup", "--help"]).output
    result = runner.invoke(main, ["setup", "plugins", "list"])
    assert result.exit_code == 0, result.output
    assert "cli_hidden" in result.output


def test_json_output_is_a_bare_array_of_plugin_objects(runner, tmp_path, fake_in_tree, monkeypatch):
    import json

    from sparkrun.cli import main

    fake_in_tree("cli_json", "__version__ = '1.4'\n\ndef register(v): pass")
    load_in_tree_plugins(None)

    result = runner.invoke(main, ["setup", "plugins", "list", "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    assert isinstance(payload, list)
    row = next(r for r in payload if r["name"] == "cli_json")
    assert row == {
        "name": "cli_json",
        "source": SOURCE_IN_TREE,
        "module": "inv_plugins.cli_json",
        "enabled": True,
        "loaded": True,
        "feature_flag": "test.inv.cli_json",
        "version": "1.4",
        "version_source": VERSION_FROM_MODULE,
        "path": None,
    }


def test_json_spells_an_unknown_version_as_null(runner, fake_in_tree):
    """'unknown' is a display rendering.

    Emitting it here would be indistinguishable from a plugin that declared
    the literal string "unknown" as its version.
    """
    import json

    from sparkrun.cli import main

    fake_in_tree("cli_json_undeclared", "def register(v): pass")
    load_in_tree_plugins(None)

    result = runner.invoke(main, ["setup", "plugins", "list", "--json"])
    row = next(r for r in json.loads(result.output) if r["name"] == "cli_json_undeclared")
    assert row["version"] is None
    assert row["version_source"] is None


def test_json_serializes_an_external_plugins_path(runner, tmp_path, monkeypatch):
    """``Path`` is why ``PluginInfo`` hand-writes ``to_dict``.

    The JSON encoder's dataclass fallback would emit a ``Path`` it cannot
    serialize.
    """
    import json

    from sparkrun.cli import main

    plugin_dir = _external_dir(tmp_path, "inv_ext_json", "__version__ = '7.0'")
    load_external_plugins(None, paths=[plugin_dir])
    monkeypatch.setattr(
        SparkrunConfig,
        "external_plugin_paths",
        property(lambda self: [plugin_dir]),
    )

    result = runner.invoke(main, ["setup", "plugins", "list", "--json"])
    assert result.exit_code == 0, result.output
    row = next(r for r in json.loads(result.output) if r["name"] == "inv_ext_json")
    assert row["path"] == str(plugin_dir)
    assert row["version"] == "7.0"


def test_features_json_is_a_bare_array_carrying_the_channel(runner):
    import json

    from sparkrun.cli import main

    from sparkrun.core.features import all_features

    result = runner.invoke(main, ["setup", "features", "list", "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    assert isinstance(payload, list)
    assert [r["name"] for r in payload] == [f.name for f in all_features()]
    assert set(payload[0]) == {"name", "description", "enabled", "source", "override", "channel"}
    # The channel is what every flag resolved under, so it is per-row rather
    # than an envelope — the array shape is the convention for list commands.
    assert len({r["channel"] for r in payload}) == 1


def test_an_enabled_plugin_that_failed_to_import_is_distinguishable(runner, fake_in_tree):
    """``on`` and ``on (load failed)`` are the diagnostic; collapsing them hides
    exactly the case someone runs this to find."""
    from sparkrun.cli import main

    fake_in_tree("cli_broken", "raise RuntimeError('boom')")
    assert load_in_tree_plugins(None) == []

    result = runner.invoke(main, ["setup", "plugins", "list"])
    assert result.exit_code == 0, result.output
    assert "on (load failed)" in result.output
