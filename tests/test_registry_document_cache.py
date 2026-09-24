"""``registries.yaml`` is parsed once per version of the file, not once per lookup."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

import sparkrun.core.registry as reg
from sparkrun.core.registry import RegistryEntry, RegistryManager


@pytest.fixture
def manager(tmp_path: Path):
    (tmp_path / "config").mkdir()
    (tmp_path / "cache").mkdir()
    mgr = RegistryManager(tmp_path / "config", tmp_path / "cache")
    mgr._manifest_discovery_attempted = True
    mgr._save_registries([RegistryEntry(name="a", url="https://example.invalid/a.git", subpath="recipes")])
    return mgr


@pytest.fixture
def reads():
    calls: list[str] = []
    real = reg.read_yaml

    def _counting(path):
        calls.append(str(path))
        return real(path)

    with mock.patch.object(reg, "read_yaml", _counting):
        yield calls


def test_repeated_lookups_do_not_reparse(manager, reads):
    for _ in range(25):
        manager._load_registries()
        manager.get_registry("a")
    assert reads == []  # our own save already recorded the document


def test_a_new_manager_in_the_same_process_shares_the_parse(manager, reads):
    """SparkrunConfig.get_registry_manager() builds a new manager per call."""
    RegistryManager(manager.config_root, manager.cache_root)._load_registries()
    assert reads == []


def test_an_external_write_is_seen_with_one_reread(manager, reads):
    path = Path(manager._registries_path)
    path.write_text(path.read_text().replace("subpath: recipes", "subpath: moved"))
    assert manager.get_registry("a").subpath == "moved"
    manager.get_registry("a")
    assert len(reads) == 1


def test_callers_get_private_copies(manager):
    first = manager._load_registries()
    first[0].subpath = "mutated-in-memory"
    assert manager._load_registries()[0].subpath == "recipes"


def test_reset_to_defaults_drops_the_cached_document(manager, monkeypatch):
    """After the file is deleted, a stale cached copy must not answer for it."""
    monkeypatch.setattr(RegistryManager, "update", lambda self, *a, **kw: None)  # no git
    manager.reset_to_defaults()
    assert "a" not in {e.name for e in manager._load_registries()}
