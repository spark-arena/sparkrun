"""Registry URL canonicalization and default-entry aliasing (issue #257 follow-up).

F1: `_normalize_registry_url` stripped only a trailing `/` and `.git`, so
`official` spelled as an SSH remote / `http://` / with different capitalisation
did not match the shipped default. The trust backfill then marked it untrusted
and its recipes prompted for hook confirmation forever — unanswerable on a
non-TTY.

F4: `_default_registries()` handed out the module-level `FALLBACK_DEFAULT_REGISTRIES`
objects by reference, so a public mutation rewrote the shipped defaults
process-wide.
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
    FALLBACK_DEFAULT_REGISTRIES,
    RegistryManager,
    _default_trusted_urls,
    _normalize_registry_url,
)

CANON = "github.com/spark-arena/recipe-registry"


# ---------------------------------------------------------------------------
# F1 — canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    [
        "https://github.com/spark-arena/recipe-registry.git",
        "https://github.com/spark-arena/recipe-registry",
        "https://github.com/spark-arena/recipe-registry/",
        "https://github.com/spark-arena/recipe-registry.git/",
        "http://github.com/spark-arena/recipe-registry",
        "ssh://git@github.com/spark-arena/recipe-registry.git",
        "git://github.com/spark-arena/recipe-registry.git",
        "git@github.com:spark-arena/recipe-registry.git",
        "git@github.com:spark-arena/recipe-registry",
        "https://GitHub.com/Spark-Arena/Recipe-Registry",
        "HTTPS://GITHUB.COM/SPARK-ARENA/RECIPE-REGISTRY.GIT",
        "https://token@github.com/spark-arena/recipe-registry.git",
        "  https://github.com/spark-arena/recipe-registry.git  ",
    ],
)
def test_spellings_of_one_repo_canonicalize_together(spelling):
    assert _normalize_registry_url(spelling) == CANON


@pytest.mark.parametrize(
    "other",
    [
        "https://github.com/spark-arena/community-recipe-registry",
        "https://github.com/someone-else/recipe-registry",
        "https://gitlab.com/spark-arena/recipe-registry",
        # A lookalike host must not fold into the real one.
        "https://github.com.evil.example/spark-arena/recipe-registry",
        # Not a git URL shape; failing to match is the safe outcome.
        "https://github.com/spark-arena/recipe-registry?ref=evil",
    ],
)
def test_different_repos_stay_distinct(other):
    assert _normalize_registry_url(other) != CANON


def test_empty_and_none_are_safe():
    assert _normalize_registry_url("") == ""
    assert _normalize_registry_url(None) == ""


def test_at_sign_in_path_is_preserved():
    """Only authority credentials are dropped; a later '@' is part of the repo."""
    assert _normalize_registry_url("https://host/org/repo@v2") == "host/org/repo@v2"


def test_official_default_is_in_the_trusted_url_set():
    assert CANON in _default_trusted_urls()


# ---------------------------------------------------------------------------
# F1 end-to-end — the trust backfill on a legacy registries.yaml
# ---------------------------------------------------------------------------


def _hermetic():
    """No bootstrap manifest discovery, no git."""
    return (
        patch.object(reg, "BOOTSTRAP_REGISTRY_URLS", []),
        patch.object(RegistryManager, "_clone_or_pull", lambda self, *a, **k: None),
    )


def _migrate_legacy(url: str) -> dict[str, bool]:
    """Write a pre-trust registries.yaml naming official by *url*, then load."""
    d = Path(tempfile.mkdtemp())
    (d / "registries.yaml").write_text(
        yaml.safe_dump(
            {
                "registries": [
                    {"name": "official", "url": url, "subpath": "official-recipes"},
                    {"name": "mine", "url": "https://github.com/someone/private", "subpath": "recipes"},
                ]
            }
        )
    )
    a, b = _hermetic()
    with a, b:
        entries = RegistryManager(config_root=d, cache_root=d / "cache")._load_registries()
    return {e.name: e.trusted for e in entries}


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/spark-arena/recipe-registry.git",
        "https://github.com/spark-arena/recipe-registry",
        "git@github.com:spark-arena/recipe-registry.git",
        "http://github.com/spark-arena/recipe-registry",
        "https://GitHub.com/Spark-Arena/Recipe-Registry",
    ],
)
def test_legacy_official_migrates_trusted_for_every_spelling(url):
    trusted = _migrate_legacy(url)
    assert trusted["official"] is True, "official spelled %r was backfilled untrusted" % url
    # A genuinely third-party registry must stay untrusted regardless.
    assert trusted["mine"] is False


def test_migration_of_a_recognized_url_terminates():
    """The migration stamps the file, so it cannot re-fire."""
    from sparkrun.core.registry import CONFIG_VERSION

    d = Path(tempfile.mkdtemp())
    (d / "registries.yaml").write_text(
        yaml.safe_dump(
            {"registries": [{"name": "official", "url": "git@github.com:spark-arena/recipe-registry.git", "subpath": "official-recipes"}]}
        )
    )
    a, b = _hermetic()
    with a, b:
        mgr = RegistryManager(config_root=d, cache_root=d / "cache")
        assert mgr._read_config_version() == 0
        mgr._load_registries()
        assert mgr._read_config_version() == CONFIG_VERSION


# ---------------------------------------------------------------------------
# F1 — deprecated-URL matching shares the same normalizer
# ---------------------------------------------------------------------------


def test_deprecated_url_matching_handles_ssh_and_case():
    dep = reg.DEPRECATED_REGISTRIES[0]
    canon = _normalize_registry_url(dep)
    host, _, path = canon.partition("/")
    assert RegistryManager._is_deprecated_url(dep) is True
    assert RegistryManager._is_deprecated_url("git@%s:%s.git" % (host, path)) is True
    assert RegistryManager._is_deprecated_url(canon.upper()) is True
    assert RegistryManager._is_deprecated_url("https://github.com/spark-arena/recipe-registry") is False


# ---------------------------------------------------------------------------
# F4 — module-level defaults must not be aliased
# ---------------------------------------------------------------------------


@pytest.fixture
def defaults_guard():
    """Fail loudly if a test mutates the shipped defaults, and restore them."""
    before = [dataclasses.replace(e) for e in FALLBACK_DEFAULT_REGISTRIES]
    yield
    for live, orig in zip(FALLBACK_DEFAULT_REGISTRIES, before, strict=True):
        live.trusted, live.enabled, live.visible = orig.trusted, orig.enabled, orig.visible


def _snapshot():
    return {e.name: (e.trusted, e.enabled, e.visible) for e in FALLBACK_DEFAULT_REGISTRIES}


def test_untrust_on_fresh_install_does_not_rewrite_shipped_defaults(defaults_guard):
    """`untrust_registry` mutates the entry it loads; on a fresh install that
    entry used to *be* the module-level default."""
    before = _snapshot()
    d = Path(tempfile.mkdtemp())
    a, b = _hermetic()
    with a, b:
        RegistryManager(config_root=d, cache_root=d / "cache").untrust_registry("official")
    assert _snapshot() == before
    assert before["official"][0] is True


def test_disable_on_fresh_install_does_not_rewrite_shipped_defaults(defaults_guard):
    before = _snapshot()
    d = Path(tempfile.mkdtemp())
    a, b = _hermetic()
    with a, b:
        RegistryManager(config_root=d, cache_root=d / "cache").disable_registry("official")
    assert _snapshot() == before


def test_default_registries_returns_distinct_objects(defaults_guard):
    d = Path(tempfile.mkdtemp())
    a, b = _hermetic()
    with a, b:
        entries = RegistryManager(config_root=d, cache_root=d / "cache")._default_registries()
    by_name = {e.name: e for e in entries}
    for shipped in FALLBACK_DEFAULT_REGISTRIES:
        assert by_name[shipped.name] is not shipped
        assert by_name[shipped.name] == shipped  # same values, different object


def test_restore_missing_defaults_restores_trust_and_leaves_globals_alone(defaults_guard):
    """`restore_missing_defaults` also appended the shipped objects by reference.

    Unlike the fresh-install path that aliasing is currently *latent* — the
    appended entries are local and the next read reconstructs them from YAML —
    so this asserts the functional outcome plus an unmutated global, not the
    identity. The copy there is defensive: it stops the next caller who decides
    to mutate that list from reaching the module-level defaults.
    """
    before = _snapshot()
    d = Path(tempfile.mkdtemp())
    (d / "registries.yaml").write_text(
        yaml.safe_dump(
            {"registries": [{"name": "mine", "url": "https://github.com/someone/private", "subpath": "recipes", "trusted": False}]}
        )
    )
    a, b = _hermetic()
    with a, b:
        mgr = RegistryManager(config_root=d, cache_root=d / "cache")
        added = mgr.restore_missing_defaults()
        entries = {e.name: e for e in mgr._load_registries_from_file()}
    assert "official" in added
    assert entries["official"].trusted is True
    assert entries["mine"].trusted is False
    assert _snapshot() == before
