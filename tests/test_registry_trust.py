"""Tests for the per-registry ``trusted`` field and related plumbing."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import yaml

from sparkrun.core.registry import (
    BOOTSTRAP_REGISTRY_URLS,
    FALLBACK_DEFAULT_REGISTRIES,
    RegistryEntry,
    RegistryError,
    RegistryManager,
)


@pytest.fixture
def reg_dirs(tmp_path: Path):
    """Create config and cache directories for RegistryManager."""
    config = tmp_path / "config"
    cache = tmp_path / "cache"
    config.mkdir()
    cache.mkdir()
    return config, cache


@pytest.fixture
def mgr(reg_dirs):
    """Create a RegistryManager with temp dirs and manifest discovery suppressed."""
    config, cache = reg_dirs
    m = RegistryManager(config, cache)
    m._manifest_discovery_attempted = True  # skip network calls in tests
    return m


# ---------------------------------------------------------------------------
# RegistryEntry round-trips trusted through save/load
# ---------------------------------------------------------------------------


class TestTrustedRoundTrip:
    def test_default_trusted_is_false(self):
        entry = RegistryEntry(name="x", url="https://example.com", subpath="recipes")
        assert entry.trusted is False

    def test_trusted_field_persists(self, mgr):
        entry = RegistryEntry(
            name="trusted-reg",
            url="https://example.com/repo",
            subpath="recipes",
            trusted=True,
        )
        mgr._save_registries([entry])
        loaded = mgr._load_registries_from_file()
        assert len(loaded) == 1
        assert loaded[0].trusted is True

    def test_untrusted_field_round_trips(self, mgr):
        entry = RegistryEntry(
            name="untrusted-reg",
            url="https://example.com/repo",
            subpath="recipes",
            trusted=False,
        )
        mgr._save_registries([entry])
        loaded = mgr._load_registries_from_file()
        assert loaded[0].trusted is False


class TestSaveOmitsDefaultTrust:
    """``trusted: false`` should not be emitted (mirrors enabled/visible behavior)."""

    def test_save_omits_trusted_false(self, mgr):
        entry = RegistryEntry(name="r", url="https://example.com", subpath="recipes", trusted=False)
        mgr._save_registries([entry])
        with open(mgr._registries_path) as f:
            raw = yaml.safe_load(f)
        assert "trusted" not in raw["registries"][0]

    def test_save_emits_trusted_true(self, mgr):
        entry = RegistryEntry(name="r", url="https://example.com", subpath="recipes", trusted=True)
        mgr._save_registries([entry])
        with open(mgr._registries_path) as f:
            raw = yaml.safe_load(f)
        assert raw["registries"][0]["trusted"] is True


# ---------------------------------------------------------------------------
# FALLBACK_DEFAULT_REGISTRIES alignment with BOOTSTRAP_REGISTRY_URLS
# ---------------------------------------------------------------------------


class TestFallbackDefaultsTrustAlignment:
    """Defaults whose URL is in BOOTSTRAP_REGISTRY_URLS are trusted; others are not."""

    def test_bootstrap_url_entries_are_trusted(self):
        for entry in FALLBACK_DEFAULT_REGISTRIES:
            if entry.url in BOOTSTRAP_REGISTRY_URLS:
                assert entry.trusted is True, "Expected %s (url=%s) to be trusted because its URL is in BOOTSTRAP_REGISTRY_URLS" % (
                    entry.name,
                    entry.url,
                )

    def test_non_bootstrap_entries_are_not_trusted(self):
        for entry in FALLBACK_DEFAULT_REGISTRIES:
            if entry.url not in BOOTSTRAP_REGISTRY_URLS:
                assert entry.trusted is False, "Expected %s (url=%s) to be untrusted because its URL is NOT in BOOTSTRAP_REGISTRY_URLS" % (
                    entry.name,
                    entry.url,
                )

    def test_eugr_and_atlas_not_trusted(self):
        """Concrete sanity check: eugr and atlas ship untrusted."""
        by_name = {e.name: e for e in FALLBACK_DEFAULT_REGISTRIES}
        assert by_name["eugr"].trusted is False
        assert by_name["atlas"].trusted is False

    def test_official_and_sparkrun_testing_trusted(self):
        """Concrete sanity check: official + sparkrun-testing ship trusted."""
        by_name = {e.name: e for e in FALLBACK_DEFAULT_REGISTRIES}
        assert by_name["official"].trusted is True
        assert by_name["sparkrun-testing"].trusted is True


# ---------------------------------------------------------------------------
# Bootstrap manifest discovery marks entries trusted
# ---------------------------------------------------------------------------


class TestBootstrapManifestDiscoveryTrust:
    def test_init_defaults_marks_manifest_entries_trusted(self, mgr):
        """Entries discovered via the bootstrap path get trusted=True."""
        from sparkrun.core import registry as reg_module

        manifest_entries = [
            RegistryEntry(name="m1", url="https://example.com/r1", subpath="r"),
            RegistryEntry(name="m2", url="https://example.com/r1", subpath="r2"),
        ]
        original = reg_module.BOOTSTRAP_REGISTRY_URLS
        try:
            reg_module.BOOTSTRAP_REGISTRY_URLS = ["https://example.com/r1"]
            with mock.patch.object(mgr, "_discover_manifest_entries", return_value=manifest_entries):
                result = mgr._init_defaults_from_manifests()
        finally:
            reg_module.BOOTSTRAP_REGISTRY_URLS = original
        assert len(result) == 2
        # Bootstrap-discovered entries are sparkrun-trusted regardless of
        # what the manifest set (manifest_entries here defaults trusted=False).
        assert all(e.trusted is True for e in result)

    def test_discover_manifest_entries_does_not_auto_trust(self, mgr, tmp_path):
        """Standalone manifest parsing (used by add_registry_from_url) keeps trusted=False."""
        tmp_repo = tmp_path / "repo"
        # _discover_manifest_entries uses tempfile.TemporaryDirectory + git clone,
        # so it's easier to assert the *parsing* layer by mocking subprocess.run
        # to write the manifest into the cloned temp dir.

        def _fake_run(*args, **kwargs):
            # The first positional arg is the argv list.
            argv = args[0]
            # When git clone is invoked, write a manifest into the destination path.
            if "clone" in argv:
                dest = Path(argv[-1])
                (dest / ".sparkrun").mkdir(parents=True, exist_ok=True)
                (dest / ".sparkrun" / "registry.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "registries": [
                                {
                                    "name": "from-manifest",
                                    "recipes": "recipes",
                                    "description": "manifest entry",
                                    # Manifest tries to set trusted; standalone path must ignore it.
                                    "trusted": True,
                                }
                            ]
                        }
                    )
                )
            return mock.Mock(returncode=0, stderr="", stdout="")

        _ = tmp_repo  # silence "unused" — the fake_run writes into mgr-allocated tempdirs
        with mock.patch("sparkrun.core.registry.subprocess.run", side_effect=_fake_run):
            entries = mgr._discover_manifest_entries("https://example.com/r1")
        assert len(entries) == 1
        # _discover_manifest_entries does NOT read 'trusted' from the manifest —
        # standalone discovery keeps the dataclass default (False) regardless of
        # what the manifest YAML claims.  This is the design: the manifest cannot
        # grant itself trust.
        assert entries[0].trusted is False


# ---------------------------------------------------------------------------
# trust_registry / untrust_registry
# ---------------------------------------------------------------------------


class TestTrustUntrustMethods:
    def test_trust_registry_flips_bit_and_persists(self, mgr):
        entry = RegistryEntry(name="r", url="https://example.com", subpath="recipes", trusted=False)
        mgr._save_registries([entry])
        mgr.trust_registry("r")
        loaded = mgr._load_registries_from_file()
        assert loaded[0].trusted is True

    def test_untrust_registry_flips_bit_and_persists(self, mgr):
        entry = RegistryEntry(name="r", url="https://example.com", subpath="recipes", trusted=True)
        mgr._save_registries([entry])
        mgr.untrust_registry("r")
        loaded = mgr._load_registries_from_file()
        assert loaded[0].trusted is False

    def test_trust_unknown_raises(self, mgr):
        with pytest.raises(RegistryError, match="not found"):
            mgr.trust_registry("nope")

    def test_untrust_unknown_raises(self, mgr):
        with pytest.raises(RegistryError, match="not found"):
            mgr.untrust_registry("nope")


# ---------------------------------------------------------------------------
# add_registry_from_url(..., trust=True)
# ---------------------------------------------------------------------------


class TestAddRegistryFromUrlTrustFlag:
    def test_add_with_trust_flag_marks_trusted(self, mgr):
        added_entries = [
            RegistryEntry(name="new1", url="https://example.com/x", subpath="r"),
            RegistryEntry(name="new2", url="https://example.com/x", subpath="r2"),
        ]
        with mock.patch.object(mgr, "_discover_manifest_entries", return_value=added_entries):
            result = mgr.add_registry_from_url("https://example.com/x", trust=True)
        assert len(result) == 2
        assert all(e.trusted is True for e in result)
        # And persisted: re-load
        loaded = {e.name: e for e in mgr._load_registries_from_file()}
        assert loaded["new1"].trusted is True
        assert loaded["new2"].trusted is True

    def test_add_default_is_untrusted(self, mgr):
        added_entries = [
            RegistryEntry(name="new1", url="https://example.com/y", subpath="r"),
        ]
        with mock.patch.object(mgr, "_discover_manifest_entries", return_value=added_entries):
            result = mgr.add_registry_from_url("https://example.com/y")
        assert result[0].trusted is False
        loaded = mgr._load_registries_from_file()
        # Find the new entry
        match = [e for e in loaded if e.name == "new1"][0]
        assert match.trusted is False


# ---------------------------------------------------------------------------
# One-time migration of legacy registries.yaml
# ---------------------------------------------------------------------------


class TestTrustMigration:
    def _write_legacy(self, mgr, urls: list[tuple[str, str]]):
        """Write a registries.yaml lacking any 'trusted' fields."""
        data = {"registries": [{"name": name, "url": url, "subpath": "recipes"} for name, url in urls]}
        with open(mgr._registries_path, "w") as f:
            yaml.safe_dump(data, f)

    def test_migration_marks_bootstrap_urls_trusted(self, mgr):
        bootstrap_url = BOOTSTRAP_REGISTRY_URLS[0]
        self._write_legacy(
            mgr,
            [
                ("from-bootstrap", bootstrap_url),
                ("third-party", "https://example.com/third"),
            ],
        )
        entries = mgr._load_registries()
        by_name = {e.name: e for e in entries}
        assert by_name["from-bootstrap"].trusted is True
        assert by_name["third-party"].trusted is False

    def test_migration_persists_to_disk(self, mgr):
        bootstrap_url = BOOTSTRAP_REGISTRY_URLS[0]
        self._write_legacy(mgr, [("from-bootstrap", bootstrap_url)])
        mgr._load_registries()
        with open(mgr._registries_path) as f:
            raw = yaml.safe_load(f)
        # After migration, the saved YAML carries the trust flag explicitly.
        assert raw["registries"][0].get("trusted") is True

    def test_migration_is_idempotent(self, mgr):
        bootstrap_url = BOOTSTRAP_REGISTRY_URLS[0]
        self._write_legacy(mgr, [("from-bootstrap", bootstrap_url)])
        # Trigger once
        mgr._load_registries()
        # Capture file mtime + content
        post_first = mgr._registries_path.read_text()
        # Re-loading must NOT re-run the migration / churn the file.
        mgr._load_registries()
        post_second = mgr._registries_path.read_text()
        assert post_first == post_second
        # And _needs_trust_migration returns False post-migration.
        assert mgr._needs_trust_migration() is False

    def test_no_migration_when_field_present(self, mgr):
        """If every entry already has 'trusted', migration must NOT fire."""
        data = {
            "registries": [
                {
                    "name": "explicit",
                    "url": "https://example.com",
                    "subpath": "r",
                    "trusted": False,
                }
            ]
        }
        with open(mgr._registries_path, "w") as f:
            yaml.safe_dump(data, f)
        assert mgr._needs_trust_migration() is False


# ---------------------------------------------------------------------------
# resolve_recipe_trust — name-based registry lookup
# ---------------------------------------------------------------------------


class _StubRecipe:
    def __init__(self, source_registry: str | None):
        self.source_registry = source_registry


class TestResolveRecipeTrust:
    def _patch_config(self, monkeypatch, mgr):
        """Make ``SparkrunConfig().get_registry_manager()`` return *mgr*."""
        from sparkrun.core import launcher as launcher_module

        # resolve_recipe_trust imports lazily, so we patch the source module.
        import sparkrun.core.config as config_module

        class _StubConfig:
            def get_registry_manager(self_inner):
                return mgr

        monkeypatch.setattr(config_module, "SparkrunConfig", _StubConfig)
        return launcher_module

    def test_trust_cli_flag_wins(self, mgr, monkeypatch):
        from sparkrun.core.launcher import resolve_recipe_trust

        # Even with no recipe registry, trust_cli forces trust.
        assert resolve_recipe_trust(_StubRecipe(source_registry="anything"), trust_cli=True) is True

    def test_local_recipe_trusted(self, mgr, monkeypatch):
        from sparkrun.core.launcher import resolve_recipe_trust

        assert resolve_recipe_trust(_StubRecipe(source_registry=None), trust_cli=False) is True

    def test_trusted_registry_recipe_trusted(self, mgr, monkeypatch):
        self._patch_config(monkeypatch, mgr)
        from sparkrun.core.launcher import resolve_recipe_trust

        mgr._save_registries([RegistryEntry(name="r", url="https://example.com", subpath="recipes", trusted=True)])
        assert resolve_recipe_trust(_StubRecipe(source_registry="r"), trust_cli=False) is True

    def test_untrusted_registry_recipe_untrusted(self, mgr, monkeypatch):
        self._patch_config(monkeypatch, mgr)
        from sparkrun.core.launcher import resolve_recipe_trust

        mgr._save_registries([RegistryEntry(name="r", url="https://example.com", subpath="recipes", trusted=False)])
        assert resolve_recipe_trust(_StubRecipe(source_registry="r"), trust_cli=False) is False

    def test_unknown_registry_recipe_untrusted(self, mgr, monkeypatch):
        self._patch_config(monkeypatch, mgr)
        from sparkrun.core.launcher import resolve_recipe_trust

        # No entries saved — get_registry will raise RegistryError → untrusted.
        mgr._save_registries([])
        assert resolve_recipe_trust(_StubRecipe(source_registry="missing"), trust_cli=False) is False
