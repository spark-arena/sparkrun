"""Filesystem-safety validation for registry names and asset subpaths.

Registry names and subpaths arrive from ``.sparkrun/registry.yaml`` manifests in
**remote repositories** (``sparkrun registry add <url>``, bootstrap discovery)
and are then turned into real paths:

* ``RegistryManager._cache_dir(name)`` is ``cache_root / name``, and
  ``_link_registry_to_shared`` ``rmtree``s a non-link cache dir — so an
  escaping name is a delete primitive.
* ``RegistryManager.asset_dir()`` is ``_cache_dir(name) / subpath``, whose
  contents ``iter_asset_files`` ``rglob``s and ``find_recipe`` offers as
  runnable recipes — so an escaping subpath is a read primitive that feeds the
  recipe loader.

These tests pin the guards at every point a name or subpath can enter.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import yaml

from sparkrun.core.registry import (
    FALLBACK_DEFAULT_REGISTRIES,
    SUBPATH_FIELDS,
    RegistryEntry,
    RegistryError,
    RegistryManager,
    assert_safe_registry_entry,
    assert_safe_registry_name,
    assert_safe_registry_subpath,
    validate_registry_name,
)


@pytest.fixture
def mgr(tmp_path: Path) -> RegistryManager:
    """A RegistryManager on temp dirs with manifest discovery disabled."""
    config = tmp_path / "config"
    cache = tmp_path / "cache"
    config.mkdir()
    cache.mkdir()
    m = RegistryManager(config, cache)
    m._manifest_discovery_attempted = True  # no network
    return m


# ---------------------------------------------------------------------------
# assert_safe_registry_name
# ---------------------------------------------------------------------------


class TestSafeRegistryName:
    @pytest.mark.parametrize(
        "name",
        [
            "official",
            "atlas",
            "sparkrun-testing",
            "my.registry",
            "reg_1",
            "A1",
            "9lives",
        ],
    )
    def test_accepts_ordinary_names(self, name):
        assert_safe_registry_name(name)

    @pytest.mark.parametrize("name", [e.name for e in FALLBACK_DEFAULT_REGISTRIES])
    def test_accepts_every_shipped_default(self, name):
        """A shipped default that failed validation would be silently dropped."""
        assert_safe_registry_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "../evil",
            "..",
            ".",
            "a/b",
            "a\\b",
            "/abs",
            "./rel",
            ".hidden",
            "-foo",  # would reach git as an option
            "_url_deadbeef",  # collides with a shared-clone directory
            "has space",
            "has:colon",  # 'C:' style drive prefixes
            "semi;colon",
            "$(whoami)",
            "",
        ],
    )
    def test_rejects_unsafe_names(self, name):
        with pytest.raises(RegistryError):
            assert_safe_registry_name(name)

    def test_rejects_overlong_name(self):
        with pytest.raises(RegistryError, match="too long"):
            assert_safe_registry_name("a" * 101)

    def test_traversal_name_would_have_escaped_cache_root(self, mgr):
        """The property the charset protects, stated directly."""
        escaped = (mgr.cache_root / "../evil").resolve()
        assert not escaped.is_relative_to(mgr.cache_root.resolve())
        with pytest.raises(RegistryError):
            assert_safe_registry_name("../evil")

    def test_shared_clone_prefix_would_have_aliased_a_shared_clone(self, mgr):
        """A ``_url_<hash>`` name resolves onto the shared checkout of some URL.

        ``_link_registry_to_shared`` rmtree's a non-link cache dir, so such a
        registry would delete the checkout its siblings on that URL share.
        """
        url = "https://github.com/example/repo"
        shared = mgr._clone_dir_for_url(url)
        assert mgr._cache_dir(shared.name) == shared
        with pytest.raises(RegistryError):
            assert_safe_registry_name(shared.name)


# ---------------------------------------------------------------------------
# assert_safe_registry_subpath
# ---------------------------------------------------------------------------


class TestSafeRegistrySubpath:
    @pytest.mark.parametrize(
        "subpath",
        [
            "",  # undeclared — asset_dir already returns None
            "recipes",
            "testing/recipes",
            "official-recipes",
            "a/b/c",
            "v0.2/recipes",
            "recipes/",  # tolerated trailing slash
        ],
    )
    def test_accepts_ordinary_subpaths(self, subpath):
        assert_safe_registry_subpath(subpath)

    def test_accepts_the_repository_root_marker(self):
        """``/`` alone is the repo-root spelling; it resolves to the checkout, never the filesystem root."""
        assert_safe_registry_subpath("/")

    @pytest.mark.parametrize(
        "field",
        SUBPATH_FIELDS,
    )
    def test_accepts_every_shipped_default_subpath(self, field):
        """A shipped subpath that failed validation would drop that asset kind."""
        for entry in FALLBACK_DEFAULT_REGISTRIES:
            assert_safe_registry_subpath(getattr(entry, field), field=field)

    @pytest.mark.parametrize(
        "subpath",
        [
            "../../../.ssh",
            "..",
            "recipes/../../..",
            "/etc",
            "//etc",
            "~/secrets",
            "a\\b",  # Windows separator
            "C:/Windows",  # absolute on Windows
            ".git",  # dotfile segment
            "recipes/.git/config",
            "has space/recipes",
            "$(whoami)",
        ],
    )
    def test_rejects_unsafe_subpaths(self, subpath):
        with pytest.raises(RegistryError):
            assert_safe_registry_subpath(subpath)

    def test_error_names_the_offending_field(self):
        with pytest.raises(RegistryError, match="benchmark_subpath"):
            assert_safe_registry_subpath("../../x", field="benchmark_subpath")

    def test_traversal_subpath_would_have_escaped_the_clone(self, mgr):
        entry = RegistryEntry(name="r", url="https://example.com/r", subpath="../../../..")
        escaped = (mgr._cache_dir(entry.name) / entry.subpath).resolve()
        assert not escaped.is_relative_to(mgr.cache_root.resolve())
        with pytest.raises(RegistryError):
            assert_safe_registry_subpath(entry.subpath)


# ---------------------------------------------------------------------------
# assert_safe_registry_entry
# ---------------------------------------------------------------------------


class TestSafeRegistryEntry:
    def test_accepts_ordinary_entry(self):
        assert_safe_registry_entry(
            RegistryEntry(
                name="reg",
                url="https://example.com/r",
                subpath="recipes",
                tuning_subpath="tuning",
                benchmark_subpath="benchmarking",
                mods_subpath="mods",
            )
        )

    @pytest.mark.parametrize("field", SUBPATH_FIELDS)
    def test_rejects_traversal_in_any_subpath_field(self, field):
        """Every path-forming field is covered — not just the recipe subpath."""
        entry = RegistryEntry(name="reg", url="https://example.com/r", subpath="recipes")
        setattr(entry, field, "../../escape")
        with pytest.raises(RegistryError):
            assert_safe_registry_entry(entry)

    @pytest.mark.parametrize("entry", FALLBACK_DEFAULT_REGISTRIES, ids=lambda e: e.name)
    def test_every_shipped_default_entry_is_safe(self, entry):
        assert_safe_registry_entry(entry)


# ---------------------------------------------------------------------------
# validate_registry_name folds the safety check in
# ---------------------------------------------------------------------------


class TestValidateRegistryNameSafety:
    def test_traversal_name_rejected_regardless_of_org(self):
        """Even an allowed org may not claim a path-escaping name."""
        with pytest.raises(RegistryError):
            validate_registry_name("../evil", "https://github.com/spark-arena/repo")

    def test_safety_check_precedes_reserved_prefix_check(self):
        """An unsafe name is rejected on safety grounds, not left to the prefix rule.

        ``../sparkrun-x`` does not *start with* a reserved prefix, so the
        namespace check alone would wave it through.
        """
        with pytest.raises(RegistryError, match="valid directory name"):
            validate_registry_name("../sparkrun-x", "https://github.com/random/repo")

    def test_ordinary_name_still_passes(self):
        validate_registry_name("my-custom-recipes", "https://github.com/random-user/repo")


# ---------------------------------------------------------------------------
# add_registry
# ---------------------------------------------------------------------------


class TestAddRegistryRejectsUnsafe:
    def test_traversal_name_rejected_and_nothing_persisted(self, mgr):
        with pytest.raises(RegistryError):
            mgr.add_registry(RegistryEntry(name="../evil", url="https://example.com/r", subpath="recipes"))
        assert not mgr._registries_path.exists()

    def test_traversal_subpath_rejected(self, mgr):
        """validate_registry_name only sees the name; the subpaths need their own gate."""
        with pytest.raises(RegistryError, match="subpath"):
            mgr.add_registry(RegistryEntry(name="ok-name", url="https://example.com/r", subpath="../../../.ssh"))

    def test_traversal_benchmark_subpath_rejected(self, mgr):
        with pytest.raises(RegistryError, match="benchmark_subpath"):
            mgr.add_registry(
                RegistryEntry(
                    name="ok-name",
                    url="https://example.com/r",
                    subpath="recipes",
                    benchmark_subpath="../../elsewhere",
                )
            )

    def test_safe_entry_still_adds(self, mgr):
        # An empty config seeds FALLBACK_DEFAULT_REGISTRIES first, so assert on
        # membership rather than on the whole list.
        mgr.add_registry(RegistryEntry(name="fine", url="https://example.com/r", subpath="recipes"))
        assert "fine" in {e.name for e in mgr.list_registries()}


# ---------------------------------------------------------------------------
# _load_registries_from_file
# ---------------------------------------------------------------------------


def _write_registries(mgr: RegistryManager, entries: list[dict]) -> None:
    mgr._registries_path.write_text(yaml.safe_dump({"config_version": 1, "registries": entries}))


class TestLoadRegistriesSkipsUnsafe:
    def test_unsafe_entry_skipped_and_others_kept(self, mgr, caplog):
        """One bad entry must not take the user's other registries with it."""
        _write_registries(
            mgr,
            [
                {"name": "good", "url": "https://example.com/a", "subpath": "recipes", "trusted": True},
                {"name": "../evil", "url": "https://example.com/b", "subpath": "recipes", "trusted": True},
                {"name": "also-good", "url": "https://example.com/c", "subpath": "recipes", "trusted": False},
            ],
        )
        with caplog.at_level("WARNING"):
            entries = mgr._load_registries_from_file()
        assert [e.name for e in entries] == ["good", "also-good"]
        assert "Skipping unusable registry entry" in caplog.text

    def test_unsafe_subpath_entry_skipped(self, mgr):
        _write_registries(
            mgr,
            [
                {"name": "good", "url": "https://example.com/a", "subpath": "recipes"},
                {"name": "escapes", "url": "https://example.com/b", "subpath": "../../../.."},
            ],
        )
        assert [e.name for e in mgr._load_registries_from_file()] == ["good"]

    def test_load_registries_does_not_fall_back_to_defaults(self, mgr):
        """The skip is narrower than the enclosing except, which reverts to defaults."""
        _write_registries(
            mgr,
            [
                {"name": "mine", "url": "https://example.com/a", "subpath": "recipes"},
                {"name": "../evil", "url": "https://example.com/b", "subpath": "recipes"},
            ],
        )
        names = [e.name for e in mgr._load_registries()]
        assert names == ["mine"]
        assert "official" not in names  # i.e. we did not revert to FALLBACK_DEFAULT_REGISTRIES

    def test_unsafe_entry_never_reaches_a_cache_dir(self, mgr):
        """End-to-end: the escaping name cannot be resolved to a path by any caller."""
        _write_registries(
            mgr,
            [{"name": "../evil", "url": "https://example.com/b", "subpath": "recipes"}],
        )
        for entry in mgr._load_registries_from_file():
            assert mgr._cache_dir(entry.name).resolve().is_relative_to(mgr.cache_root.resolve())


# ---------------------------------------------------------------------------
# _discover_manifest_entries
# ---------------------------------------------------------------------------


def _fake_clone_writing_manifest(manifest: dict, calls: list[list[str]] | None = None):
    """subprocess.run stub: the clone materializes ``.sparkrun/registry.yaml``."""

    def _run(cmd, **kwargs):
        if calls is not None:
            calls.append(list(cmd))
        if "clone" in cmd:
            dest = Path(cmd[-1])
            (dest / ".sparkrun").mkdir(parents=True, exist_ok=True)
            (dest / ".sparkrun" / "registry.yaml").write_text(yaml.safe_dump(manifest))
        return mock.Mock(returncode=0, stderr="", stdout="")

    return _run


class TestDiscoverManifestEntriesSafety:
    def test_unsafe_entry_dropped_safe_entry_kept(self, mgr, caplog):
        manifest = {
            "registries": [
                {"name": "../evil", "recipes": "recipes"},
                {"name": "legit", "recipes": "recipes"},
            ]
        }
        with mock.patch("sparkrun.core.registry.subprocess.run", side_effect=_fake_clone_writing_manifest(manifest)):
            with caplog.at_level("WARNING"):
                entries = mgr._discover_manifest_entries("https://example.com/repo")
        assert [e.name for e in entries] == ["legit"]
        assert "Ignoring unsafe entry" in caplog.text

    def test_unsafe_subpath_entry_dropped(self, mgr):
        manifest = {
            "registries": [
                {"name": "sneaky", "recipes": "../../../../etc"},
                {"name": "legit", "recipes": "recipes"},
            ]
        }
        with mock.patch("sparkrun.core.registry.subprocess.run", side_effect=_fake_clone_writing_manifest(manifest)):
            entries = mgr._discover_manifest_entries("https://example.com/repo")
        assert [e.name for e in entries] == ["legit"]

    def test_unsafe_optional_subpath_entry_dropped(self, mgr):
        manifest = {"registries": [{"name": "sneaky", "recipes": "recipes", "benchmarks": "../../elsewhere"}]}
        with mock.patch("sparkrun.core.registry.subprocess.run", side_effect=_fake_clone_writing_manifest(manifest)):
            with pytest.raises(RegistryError, match="no usable registries"):
                mgr._discover_manifest_entries("https://example.com/repo")

    def test_all_entries_unsafe_raises(self, mgr):
        """A wholly hostile manifest must not be reported as a successful no-op add."""
        manifest = {"registries": [{"name": "../a", "recipes": "recipes"}, {"name": "b/c", "recipes": "recipes"}]}
        with mock.patch("sparkrun.core.registry.subprocess.run", side_effect=_fake_clone_writing_manifest(manifest)):
            with pytest.raises(RegistryError, match="no usable registries"):
                mgr._discover_manifest_entries("https://example.com/repo")

    def test_add_registry_from_url_drops_the_unsafe_entry(self, mgr):
        """The user-facing path: the hostile entry never lands in registries.yaml."""
        manifest = {
            "registries": [
                {"name": "../evil", "recipes": "recipes"},
                {"name": "legit", "recipes": "recipes"},
            ]
        }
        with mock.patch("sparkrun.core.registry.subprocess.run", side_effect=_fake_clone_writing_manifest(manifest)):
            added = mgr.add_registry_from_url("https://example.com/repo")
        assert [e.name for e in added] == ["legit"]
        persisted = {e.name for e in mgr._load_registries_from_file()}
        assert "legit" in persisted
        assert "../evil" not in persisted


class TestDiscoverManifestEntriesSparseClone:
    """The manifest clone fetches only ``.sparkrun``, not the recipe trees."""

    def test_clone_is_blob_filtered_and_sparse(self, mgr):
        calls: list[list[str]] = []
        manifest = {"registries": [{"name": "legit", "recipes": "recipes"}]}
        with mock.patch(
            "sparkrun.core.registry.subprocess.run",
            side_effect=_fake_clone_writing_manifest(manifest, calls),
        ):
            mgr._discover_manifest_entries("https://example.com/repo")

        clone = next(c for c in calls if "clone" in c)
        assert "--filter=blob:none" in clone
        assert "--sparse" in clone
        # The URL stays behind `--`, so a dash-leading URL can't become an option.
        assert clone.index("--") < clone.index("https://example.com/repo")

    def test_sparse_checkout_requests_the_manifest_dir(self, mgr):
        """Without this, ``--sparse`` leaves .sparkrun absent and discovery fails."""
        calls: list[list[str]] = []
        manifest = {"registries": [{"name": "legit", "recipes": "recipes"}]}
        with mock.patch(
            "sparkrun.core.registry.subprocess.run",
            side_effect=_fake_clone_writing_manifest(manifest, calls),
        ):
            mgr._discover_manifest_entries("https://example.com/repo")

        sparse = next((c for c in calls if "sparse-checkout" in c), None)
        assert sparse is not None, "expected a sparse-checkout call"
        assert sparse[-1] == ".sparkrun"

    def test_sparse_checkout_failure_raises(self, mgr):
        """A clone whose manifest dir was never checked out must not look empty."""

        def _run(cmd, **kwargs):
            if "sparse-checkout" in cmd:
                return mock.Mock(returncode=1, stderr="fatal: nope", stdout="")
            return mock.Mock(returncode=0, stderr="", stdout="")

        with mock.patch("sparkrun.core.registry.subprocess.run", side_effect=_run):
            with pytest.raises(RegistryError, match=r"Failed to check out \.sparkrun/"):
                mgr._discover_manifest_entries("https://example.com/repo")
