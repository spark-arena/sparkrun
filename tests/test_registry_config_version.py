"""``config_version`` migration marker + explicit two-directional ``trusted``.

Closes F2/F3 from the #257 verification pass. Both came from one ambiguity:
``_save_registries`` omitted ``trusted: false``, so a file with nothing trusted
was byte-identical to one written before the trust model existed. That made the
"one-time" migration re-fire on every load, and silently reverted a user who had
untrusted every registry.

The coverage gap that let them through is the reason this file exists: *every*
pre-existing migration test has at least one entry that ends up trusted, so the
all-untrusted case was never exercised.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import sparkrun.core.registry as reg
from sparkrun.core.registry import (
    CONFIG_VERSION,
    FALLBACK_DEFAULT_REGISTRIES,
    RegistryEntry,
    RegistryManager,
)

OFFICIAL_URL = "https://github.com/spark-arena/recipe-registry.git"


@pytest.fixture(autouse=True)
def _restore_shipped_defaults():
    before = [dataclasses.replace(e) for e in FALLBACK_DEFAULT_REGISTRIES]
    yield
    for live, orig in zip(FALLBACK_DEFAULT_REGISTRIES, before, strict=True):
        live.trusted, live.enabled, live.visible = orig.trusted, orig.enabled, orig.visible


def _hermetic():
    return (
        patch.object(reg, "BOOTSTRAP_REGISTRY_URLS", []),
        patch.object(RegistryManager, "_clone_or_pull", lambda self, *a, **k: None),
    )


@pytest.fixture
def mgr():
    d = Path(tempfile.mkdtemp())
    a, b = _hermetic()
    with a, b:
        yield RegistryManager(config_root=d, cache_root=d / "cache")


def _raw(mgr) -> dict:
    return yaml.safe_load(mgr._registries_path.read_text())


def _reload(mgr) -> dict[str, bool]:
    a, b = _hermetic()
    with a, b:
        fresh = RegistryManager(config_root=mgr.config_root, cache_root=mgr.config_root / "cache")
        return {e.name: e.trusted for e in fresh._load_registries()}


# ---------------------------------------------------------------------------
# Marker basics
# ---------------------------------------------------------------------------


def test_save_stamps_the_marker(mgr):
    mgr._save_registries([RegistryEntry(name="r", url="https://example.com", subpath="r")])
    assert _raw(mgr)["config_version"] == CONFIG_VERSION


def test_absent_marker_and_no_trusted_key_is_version_zero(mgr):
    mgr._registries_path.write_text(yaml.safe_dump({"registries": [{"name": "r", "url": "https://example.com", "subpath": "r"}]}))
    assert mgr._read_config_version() == 0


def test_absent_marker_but_trusted_key_present_is_version_one(mgr):
    """A file written after the trust model landed but before the marker did."""
    mgr._registries_path.write_text(
        yaml.safe_dump({"registries": [{"name": "r", "url": "https://example.com", "subpath": "r", "trusted": False}]})
    )
    assert mgr._read_config_version() == 1


def test_empty_registry_list_is_treated_as_current(mgr):
    """Nothing to migrate — must not re-fire forever on an empty file."""
    mgr._registries_path.write_text(yaml.safe_dump({"registries": []}))
    assert mgr._read_config_version() == CONFIG_VERSION


def test_unreadable_file_is_treated_as_current(mgr):
    mgr._registries_path.write_text("{{{ not yaml")
    assert mgr._read_config_version() == CONFIG_VERSION


def test_bool_is_not_accepted_as_a_version(mgr):
    """`config_version: true` is bool, and bool is an int in Python."""
    mgr._registries_path.write_text(
        yaml.safe_dump({"config_version": True, "registries": [{"name": "r", "url": "https://example.com", "subpath": "r"}]})
    )
    assert mgr._read_config_version() == 0


# ---------------------------------------------------------------------------
# F2 — the migration terminates even when it trusts nothing
# ---------------------------------------------------------------------------


def test_migration_terminates_when_no_entry_ends_up_trusted(mgr):
    """The case every pre-existing migration test missed.

    With no trusted entry the old writer emitted no ``trusted`` key at all, so
    the file still looked pre-trust and the "one-time" migration re-ran (and
    rewrote the file) on every single load.
    """
    mgr._registries_path.write_text(
        yaml.safe_dump(
            {
                "registries": [
                    {"name": "mine", "url": "https://github.com/someone/private", "subpath": "recipes"},
                    {"name": "other", "url": "https://gitlab.com/someone/else", "subpath": "recipes"},
                ]
            }
        )
    )
    assert mgr._read_config_version() == 0
    entries = mgr._load_registries()
    assert all(e.trusted is False for e in entries)

    assert mgr._read_config_version() == CONFIG_VERSION
    first = mgr._registries_path.read_text()
    mgr._load_registries()
    assert mgr._registries_path.read_text() == first, "migration rewrote the file a second time"


# ---------------------------------------------------------------------------
# F3 — an explicit untrust of every registry survives a reload
# ---------------------------------------------------------------------------


def test_untrusting_every_registry_is_not_reverted(mgr):
    entries = mgr._load_registries()
    assert any(e.trusted for e in entries), "fixture precondition: defaults ship some trust"

    for e in entries:
        e.trusted = False
    mgr._save_registries(entries)

    assert all(r["trusted"] is False for r in _raw(mgr)["registries"])
    assert _reload(mgr)["official"] is False


def test_untrusting_one_registry_is_not_reverted(mgr):
    mgr._load_registries()
    mgr.untrust_registry("official")
    assert _reload(mgr)["official"] is False


def test_trust_survives_a_reload_too(mgr):
    mgr._load_registries()
    mgr.trust_registry("community")
    assert _reload(mgr)["community"] is True


# ---------------------------------------------------------------------------
# Never migrate backwards
# ---------------------------------------------------------------------------


def test_file_from_a_newer_sparkrun_is_left_alone(mgr):
    mgr._registries_path.write_text(
        yaml.safe_dump(
            {
                "config_version": CONFIG_VERSION + 5,
                "registries": [{"name": "official", "url": OFFICIAL_URL, "subpath": "official-recipes", "trusted": False}],
            }
        )
    )
    before = mgr._registries_path.read_text()
    entries = {e.name: e for e in mgr._load_registries()}
    # No backfill, no rewrite — an unknown future revision is not an error.
    assert entries["official"].trusted is False
    assert mgr._registries_path.read_text() == before


# ---------------------------------------------------------------------------
# Mutating commands must not stamp an unmigrated file
# ---------------------------------------------------------------------------


def _write_legacy(mgr, entries: list[dict]):
    mgr._registries_path.write_text(yaml.safe_dump({"registries": entries}))


def test_restore_missing_defaults_migrates_before_stamping(mgr):
    """It ends in a save, and a save stamps — so it must migrate first."""
    _write_legacy(mgr, [{"name": "official", "url": OFFICIAL_URL, "subpath": "official-recipes"}])
    mgr.restore_missing_defaults()
    entries = {e.name: e.trusted for e in mgr._load_registries()}
    assert entries["official"] is True, "trust backfill was skipped by the stamp"


def test_cleanup_deprecated_migrates_before_stamping(mgr):
    _write_legacy(mgr, [{"name": "official", "url": OFFICIAL_URL, "subpath": "official-recipes"}])
    with patch.object(reg, "DEPRECATED_REGISTRIES", ["https://example.com/gone"]):
        mgr.cleanup_deprecated()
    entries = {e.name: e.trusted for e in mgr._load_registries()}
    assert entries["official"] is True


def test_cleanup_deprecated_still_sees_deprecated_entries(mgr):
    """Regression guard: routing this through ``_load_registries`` would have
    filtered them out before the sweep, so nothing was reported or cache-cleaned."""
    mgr.add_registry(RegistryEntry(name="old-reg", url="https://example.com/old/repo", subpath="r"))
    with patch.object(reg, "DEPRECATED_REGISTRIES", ["https://example.com/old/repo"]):
        assert "old-reg" in mgr.cleanup_deprecated()
