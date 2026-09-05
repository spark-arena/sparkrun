"""Tests for plugin-declared default registries (``core.registry_defaults``)."""

from __future__ import annotations

import pytest

from sparkrun.core.registry import (
    FALLBACK_DEFAULT_REGISTRIES,
    SUPPRESSED_REGISTRIES_KEY,
    RegistryEntry,
    RegistryError,
    RegistryManager,
)
from sparkrun.core.registry_defaults import (
    DeclarationTier,
    declaring_tier,
    iter_declared_registries,
    register_default_registry,
    reset_declared_registries,
)

import yaml

PLUGIN_URL = "https://github.com/sparksq/sparkrun-recipes.git"


def _entry(name: str = "coldsnap", **kw) -> RegistryEntry:
    fields = {"name": name, "url": PLUGIN_URL, "subpath": "coldsnap-recipes"}
    fields.update(kw)
    return RegistryEntry(**fields)


@pytest.fixture
def mgr(tmp_path) -> RegistryManager:
    return RegistryManager(config_root=tmp_path / "config", cache_root=tmp_path / "cache")


def _read_yaml(mgr: RegistryManager) -> dict:
    return yaml.safe_load(mgr._registries_path.read_text()) or {}


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def test_declaration_requires_a_non_blank_owner():
    with pytest.raises(ValueError):
        register_default_registry(_entry(), owner="   ")


def test_declaration_is_idempotent_for_the_same_owner_and_entry():
    register_default_registry(_entry(), owner="coldsnap")
    register_default_registry(_entry(), owner="coldsnap")
    assert len(iter_declared_registries()) == 1


def test_a_second_owner_cannot_claim_an_existing_name():
    register_default_registry(_entry(), owner="coldsnap")
    with pytest.raises(RegistryError, match="already declared by plugin 'coldsnap'"):
        register_default_registry(_entry(), owner="impostor")


def test_the_same_owner_may_revise_its_own_declaration():
    register_default_registry(_entry(), owner="coldsnap")
    register_default_registry(_entry(description="revised"), owner="coldsnap")
    (declared,) = iter_declared_registries()
    assert declared.entry.description == "revised"


def test_an_unsafe_name_is_rejected():
    with pytest.raises(RegistryError):
        register_default_registry(_entry(name="../escape"), owner="coldsnap")


def test_an_unsafe_subpath_is_rejected():
    with pytest.raises(RegistryError):
        register_default_registry(_entry(subpath="../../etc"), owner="coldsnap")


def test_a_reserved_name_from_the_wrong_org_is_rejected():
    """`coldsnap` is reserved to sparksq; another org may not declare it."""
    with pytest.raises(RegistryError, match="reserved"):
        register_default_registry(
            _entry(url="https://github.com/somebody-else/recipes.git"),
            owner="impostor",
        )


def test_the_reserved_name_is_accepted_from_its_own_org():
    register_default_registry(_entry(), owner="coldsnap")
    assert [d.entry.name for d in iter_declared_registries()] == ["coldsnap"]


def test_reserved_names_admit_the_urls_coldsnap_declares():
    """Guards the §4 failure mode: a mismatch makes the plugin raise at bootstrap."""
    from sparkrun.core.registry import EXTERNAL_RESERVED_NAMES, _get_git_org

    org = _get_git_org(PLUGIN_URL)
    assert org == "sparksq"
    for name in ("coldsnap", "coldsnap-vanilla"):
        assert org in EXTERNAL_RESERVED_NAMES[name]


def test_declarations_are_ordered_deterministically():
    register_default_registry(_entry("coldsnap-vanilla", subpath="vanilla-recipes"), owner="coldsnap")
    register_default_registry(_entry("coldsnap"), owner="coldsnap")
    assert [d.entry.name for d in iter_declared_registries()] == ["coldsnap", "coldsnap-vanilla"]


# --------------------------------------------------------------------------
# Tier / trust
# --------------------------------------------------------------------------


def test_in_tree_declarations_may_ship_trusted():
    with declaring_tier(DeclarationTier.IN_TREE):
        register_default_registry(_entry(trusted=True), owner="coldsnap")
    (declared,) = iter_declared_registries()
    assert declared.effective_entry().trusted is True


def test_out_of_tree_declarations_are_forced_untrusted():
    with declaring_tier(DeclarationTier.OUT_OF_TREE):
        register_default_registry(_entry(trusted=True), owner="coldsnap")
    (declared,) = iter_declared_registries()
    assert declared.effective_entry().trusted is False


def test_the_default_tier_is_the_unprivileged_one():
    """A declaration made outside any loader must not acquire in-tree trust."""
    register_default_registry(_entry(trusted=True), owner="coldsnap")
    (declared,) = iter_declared_registries()
    assert declared.tier is DeclarationTier.OUT_OF_TREE
    assert declared.effective_entry().trusted is False


def test_effective_entry_does_not_alias_the_declaration():
    with declaring_tier(DeclarationTier.IN_TREE):
        register_default_registry(_entry(), owner="coldsnap")
    (declared,) = iter_declared_registries()
    first = declared.effective_entry()
    first.enabled = False
    assert declared.effective_entry().enabled is True


def test_the_loader_supplies_the_tier(monkeypatch):
    """load_plugin_module wraps registration; the plugin never states its tier."""
    import types

    from sparkrun.core.external_plugins import load_plugin_module

    module = types.ModuleType("fake_in_tree_plugin")
    module.register = lambda v: register_default_registry(_entry(trusted=True), owner="coldsnap")
    monkeypatch.setattr("sparkrun.core.external_plugins._plugin_base_types", lambda: [])

    load_plugin_module(module, None, tier=DeclarationTier.IN_TREE)
    (declared,) = iter_declared_registries()
    assert declared.tier is DeclarationTier.IN_TREE
    assert declared.effective_entry().trusted is True


def test_the_tier_does_not_leak_past_a_load(monkeypatch):
    import types

    from sparkrun.core.external_plugins import load_plugin_module

    module = types.ModuleType("fake_plugin")
    module.register = lambda v: None
    monkeypatch.setattr("sparkrun.core.external_plugins._plugin_base_types", lambda: [])
    load_plugin_module(module, None, tier=DeclarationTier.IN_TREE)

    register_default_registry(_entry(trusted=True), owner="later")
    (declared,) = iter_declared_registries()
    assert declared.tier is DeclarationTier.OUT_OF_TREE


# --------------------------------------------------------------------------
# Overlay
# --------------------------------------------------------------------------


def test_the_overlay_appears_on_a_fresh_install(mgr):
    register_default_registry(_entry(), owner="coldsnap")
    assert "coldsnap" in {r.name for r in mgr.list_registries()}


def _materialize_config(mgr: RegistryManager) -> None:
    """Force registries.yaml into existence.

    ``_default_registries`` only persists when manifest discovery found
    something, and the hermetic suite empties ``BOOTSTRAP_REGISTRY_URLS`` — so a
    plain ``list_registries()`` leaves no file. Any mutation writes one.
    """
    mgr.add_registry(RegistryEntry(name="seed", url="https://github.com/me/seed.git", subpath="r"))
    assert mgr._registries_path.exists()


def test_the_overlay_appears_on_an_existing_install(mgr):
    _materialize_config(mgr)

    register_default_registry(_entry(), owner="coldsnap")
    entry = next(r for r in mgr.list_registries() if r.name == "coldsnap")
    assert entry.declared_by == "coldsnap"


def test_the_overlay_is_never_persisted(mgr):
    register_default_registry(_entry(), owner="coldsnap")
    mgr.list_registries()
    mgr.disable_registry("official")  # force a save of an unrelated entry

    names = {r["name"] for r in _read_yaml(mgr)["registries"]}
    assert "coldsnap" not in names


def test_withdrawing_a_declaration_removes_the_entry(mgr):
    _materialize_config(mgr)
    register_default_registry(_entry(), owner="coldsnap")
    assert "coldsnap" in {r.name for r in mgr.list_registries()}

    reset_declared_registries()
    assert "coldsnap" not in {r.name for r in mgr.list_registries()}
    assert "coldsnap" not in {r["name"] for r in _read_yaml(mgr)["registries"]}


def test_a_file_entry_of_the_same_name_wins(mgr):
    mgr.add_registry(_entry(url="https://github.com/sparksq/sparkrun-recipes.git", subpath="mine"))
    register_default_registry(_entry(subpath="coldsnap-recipes"), owner="coldsnap")

    entry = next(r for r in mgr.list_registries() if r.name == "coldsnap")
    assert entry.subpath == "mine"
    assert entry.declared_by == ""


def test_a_declaration_cannot_shadow_a_shipped_default(mgr, caplog):
    shipped = FALLBACK_DEFAULT_REGISTRIES[0]
    register_default_registry(
        RegistryEntry(name=shipped.name, url="https://github.com/scitrera/other.git", subpath="r"),
        owner="impostor",
    )
    with caplog.at_level("WARNING"):
        entries = mgr.list_registries()

    entry = next(r for r in entries if r.name == shipped.name)
    assert entry.declared_by == ""
    assert entry.url == shipped.url
    assert "shipped default" in caplog.text


# --------------------------------------------------------------------------
# Materialize on mutation / tombstones
# --------------------------------------------------------------------------


def test_disable_materializes_the_entry(mgr):
    register_default_registry(_entry(), owner="coldsnap")
    mgr.disable_registry("coldsnap")

    saved = {r["name"]: r for r in _read_yaml(mgr)["registries"]}
    assert saved["coldsnap"]["enabled"] is False
    assert mgr.get_registry("coldsnap").declared_by == ""


def test_a_materialized_entry_survives_the_declaration_being_withdrawn(mgr):
    register_default_registry(_entry(), owner="coldsnap")
    mgr.disable_registry("coldsnap")
    reset_declared_registries()

    entry = mgr.get_registry("coldsnap")
    assert entry.enabled is False
    assert entry.declared_by == ""


def test_trust_materializes_and_sticks(mgr):
    with declaring_tier(DeclarationTier.OUT_OF_TREE):
        register_default_registry(_entry(trusted=True), owner="coldsnap")
    assert mgr.get_registry("coldsnap").trusted is False  # forced untrusted by tier

    mgr.trust_registry("coldsnap")
    assert mgr.get_registry("coldsnap").trusted is True
    assert {r["name"]: r for r in _read_yaml(mgr)["registries"]}["coldsnap"]["trusted"] is True


def test_remove_tombstones_a_declared_registry(mgr):
    register_default_registry(_entry(), owner="coldsnap")
    mgr.remove_registry("coldsnap")

    assert _read_yaml(mgr)[SUPPRESSED_REGISTRIES_KEY] == ["coldsnap"]
    assert "coldsnap" not in {r.name for r in mgr.list_registries()}


def test_a_tombstone_survives_a_reload_with_the_declaration_present(mgr):
    register_default_registry(_entry(), owner="coldsnap")
    mgr.remove_registry("coldsnap")

    fresh = RegistryManager(config_root=mgr.config_root, cache_root=mgr.cache_root)
    assert "coldsnap" not in {r.name for r in fresh.list_registries()}


def test_re_adding_clears_the_tombstone(mgr):
    register_default_registry(_entry(), owner="coldsnap")
    mgr.remove_registry("coldsnap")
    mgr.add_registry(_entry(subpath="mine"))

    assert SUPPRESSED_REGISTRIES_KEY not in _read_yaml(mgr)
    assert mgr.get_registry("coldsnap").subpath == "mine"


def test_removing_an_ordinary_registry_writes_no_tombstone(mgr):
    mgr.add_registry(RegistryEntry(name="mine", url="https://github.com/me/r.git", subpath="r"))
    mgr.remove_registry("mine")

    assert SUPPRESSED_REGISTRIES_KEY not in _read_yaml(mgr)


def test_removing_an_absent_registry_still_raises(mgr):
    with pytest.raises(RegistryError, match="not found"):
        mgr.remove_registry("nope")


# --------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------


def test_a_declared_registry_is_not_counted_as_third_party():
    from sparkrun.telemetry.util import registry_summary

    declared = RegistryEntry(name="coldsnap", url=PLUGIN_URL, subpath="r", declared_by="coldsnap")
    user_added = RegistryEntry(name="mine", url="https://github.com/me/r.git", subpath="r")

    summary = registry_summary([declared, user_added])
    assert summary["plugin_registry_count"] == 1
    assert summary["non_default_registry_count"] == 1  # only the user's own
    assert summary["has_non_default_registries"] is True

    only_declared = registry_summary([declared])
    assert only_declared["has_non_default_registries"] is False
    assert only_declared["plugin_registry_count"] == 1
