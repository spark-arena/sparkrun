"""Tests for EugrBuilder.resolve_long_term_image() and BuilderPlugin base."""

from __future__ import annotations

from unittest import mock


from sparkrun.builders.base import BuilderPlugin
from sparkrun.builders.eugr import (
    EugrBuilder,
    GHCR_EUGR_NIGHTLY,
    GHCR_EUGR_NIGHTLY_TF5,
    GHCR_EUGR_PKG,
    GHCR_EUGR_PKG_TF5,
)
from sparkrun.core.recipe import Recipe


def _make_recipe(**kwargs):
    base = {"name": "test", "model": "some/model", "runtime": "vllm", "builder": "eugr"}
    base.update(kwargs)
    return Recipe.from_dict(base)


# ---------------------------------------------------------------------------
# BuilderPlugin base — default returns (image, False)
# ---------------------------------------------------------------------------


class TestBuilderPluginResolveLongTermImage:
    def test_base_returns_unchanged(self):
        plugin = BuilderPlugin()
        recipe = _make_recipe()
        image, pinned = plugin.resolve_long_term_image("my-image:latest", {}, recipe)
        assert image == "my-image:latest"
        assert pinned is False


# ---------------------------------------------------------------------------
# EugrBuilder._resolve_ghcr_target
# ---------------------------------------------------------------------------


class TestResolveGhcrTarget:
    def test_local_nightly_maps_to_ghcr(self):
        builder = EugrBuilder()
        recipe = _make_recipe()
        ghcr, pkg = builder._resolve_ghcr_target("sparkrun-eugr-vllm", recipe)
        assert ghcr == GHCR_EUGR_NIGHTLY
        assert pkg == GHCR_EUGR_PKG

    def test_local_nightly_tf5_maps_to_ghcr_tf5(self):
        builder = EugrBuilder()
        recipe = _make_recipe(runtime_config={"build_args": ["--tf5"]})
        ghcr, pkg = builder._resolve_ghcr_target("sparkrun-eugr-vllm-tf5", recipe)
        assert ghcr == GHCR_EUGR_NIGHTLY_TF5
        assert pkg == GHCR_EUGR_PKG_TF5

    def test_ghcr_latest_maps(self):
        builder = EugrBuilder()
        recipe = _make_recipe()
        ghcr, pkg = builder._resolve_ghcr_target(
            "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest",
            recipe,
        )
        assert ghcr == GHCR_EUGR_NIGHTLY

    def test_ghcr_tf5_latest_maps(self):
        builder = EugrBuilder()
        recipe = _make_recipe(runtime_config={"build_args": ["--tf5"]})
        ghcr, pkg = builder._resolve_ghcr_target(
            "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:latest",
            recipe,
        )
        assert ghcr == GHCR_EUGR_NIGHTLY_TF5

    def test_custom_build_args_returns_none(self):
        builder = EugrBuilder()
        recipe = _make_recipe(runtime_config={"build_args": ["--custom-flag"]})
        ghcr, pkg = builder._resolve_ghcr_target("sparkrun-eugr-vllm", recipe)
        assert ghcr is None
        assert pkg is None

    def test_unknown_image_returns_none(self):
        builder = EugrBuilder()
        recipe = _make_recipe()
        ghcr, pkg = builder._resolve_ghcr_target("totally-custom-image", recipe)
        assert ghcr is None
        assert pkg is None

    def test_tf5_build_args_with_plain_image_resolves_tf5(self):
        """--tf5 build_args should resolve to tf5 variant regardless of image name."""
        builder = EugrBuilder()
        recipe = _make_recipe(runtime_config={"build_args": ["--tf5"]})
        ghcr, pkg = builder._resolve_ghcr_target("sparkrun-eugr-vllm", recipe)
        assert ghcr == GHCR_EUGR_NIGHTLY_TF5


# ---------------------------------------------------------------------------
# EugrBuilder.resolve_long_term_image — full flow
# ---------------------------------------------------------------------------


class TestResolveLongTermImage:
    def test_no_git_commit_returns_unpinned(self):
        builder = EugrBuilder()
        recipe = _make_recipe()
        image, pinned = builder.resolve_long_term_image(
            "sparkrun-eugr-vllm",
            {},
            recipe,
        )
        assert image == "sparkrun-eugr-vllm"
        assert pinned is False

    def test_custom_build_args_returns_unpinned(self):
        builder = EugrBuilder()
        recipe = _make_recipe(runtime_config={"build_args": ["--custom"]})
        runtime_info = {"build_build_script_commit": "abc123"}
        image, pinned = builder.resolve_long_term_image(
            "sparkrun-eugr-vllm",
            runtime_info,
            recipe,
        )
        assert pinned is False

    @mock.patch("sparkrun.builders.eugr.EugrBuilder._match_via_ghcr_api", return_value=None)
    @mock.patch("sparkrun.builders._ghcr.fetch_build_index")
    def test_match_via_build_index(self, mock_fetch, mock_api):
        mock_fetch.return_value = [
            {
                "tag": "2025032501",
                "variant": "nightly",
                "repo_commit": "abc123def456",
                "vllm_hash": "",
                "flashinfer_hash": "",
            },
        ]
        builder = EugrBuilder()
        recipe = _make_recipe()
        runtime_info = {"build_build_script_commit": "abc123def456"}
        image, pinned = builder.resolve_long_term_image(
            "sparkrun-eugr-vllm",
            runtime_info,
            recipe,
        )
        assert pinned is True
        assert image == "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:2025032501"
        mock_api.assert_not_called()

    @mock.patch("sparkrun.builders.eugr.EugrBuilder._match_via_ghcr_api", return_value=None)
    @mock.patch("sparkrun.builders._ghcr.fetch_build_index")
    def test_match_via_build_index_tf5(self, mock_fetch, mock_api):
        mock_fetch.return_value = [
            {"tag": "2025032501", "variant": "nightly-tf5", "repo_commit": "abc123def456"},
        ]
        builder = EugrBuilder()
        recipe = _make_recipe(runtime_config={"build_args": ["--tf5"]})
        runtime_info = {"build_build_script_commit": "abc123def456"}
        image, pinned = builder.resolve_long_term_image(
            "sparkrun-eugr-vllm-tf5",
            runtime_info,
            recipe,
        )
        assert pinned is True
        assert image == "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:2025032501"

    @mock.patch("sparkrun.builders.eugr.EugrBuilder._match_via_ghcr_api", return_value=None)
    @mock.patch("sparkrun.builders._ghcr.fetch_build_index")
    def test_wrong_variant_not_matched(self, mock_fetch, mock_api):
        """Index entry for 'nightly-tf5' should not match plain 'nightly' target."""
        mock_fetch.return_value = [
            {"tag": "2025032501", "variant": "nightly-tf5", "repo_commit": "abc123def456"},
        ]
        builder = EugrBuilder()
        recipe = _make_recipe()  # no --tf5
        runtime_info = {"build_build_script_commit": "abc123def456"}
        image, pinned = builder.resolve_long_term_image(
            "sparkrun-eugr-vllm",
            runtime_info,
            recipe,
        )
        assert pinned is False

    @mock.patch("sparkrun.builders._ghcr.fetch_build_index", return_value=[])
    @mock.patch("sparkrun.builders._ghcr.ghcr_list_tags")
    @mock.patch("sparkrun.builders._ghcr.ghcr_get_labels")
    def test_fallback_to_ghcr_api(self, mock_labels, mock_tags, mock_fetch):
        mock_tags.return_value = ["2025032501"]
        mock_labels.return_value = {
            "dev.sparkrun.repo-commit": "abc123def456",
        }
        builder = EugrBuilder()
        recipe = _make_recipe()
        runtime_info = {"build_build_script_commit": "abc123def456"}
        image, pinned = builder.resolve_long_term_image(
            "sparkrun-eugr-vllm",
            runtime_info,
            recipe,
        )
        assert pinned is True
        assert image == "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:2025032501"

    @mock.patch("sparkrun.builders._ghcr.fetch_build_index", return_value=[])
    @mock.patch("sparkrun.builders._ghcr.ghcr_list_tags", return_value=[])
    def test_no_match_returns_unpinned(self, mock_tags, mock_fetch):
        builder = EugrBuilder()
        recipe = _make_recipe()
        runtime_info = {"build_build_script_commit": "abc123def456"}
        image, pinned = builder.resolve_long_term_image(
            "sparkrun-eugr-vllm",
            runtime_info,
            recipe,
        )
        assert pinned is False
        assert image == "sparkrun-eugr-vllm"

    @mock.patch("sparkrun.builders.eugr.EugrBuilder._match_via_ghcr_api", return_value=None)
    @mock.patch("sparkrun.builders._ghcr.fetch_build_index")
    def test_secondary_hash_mismatch_skips_entry(self, mock_fetch, mock_api):
        """If vllm_hash is present but doesn't match, entry is skipped."""
        mock_fetch.return_value = [
            {
                "tag": "2025032501",
                "variant": "nightly",
                "repo_commit": "abc123def456",
                "vllm_hash": "wrong_hash_value",
            },
        ]
        builder = EugrBuilder()
        recipe = _make_recipe()
        runtime_info = {
            "build_build_script_commit": "abc123def456",
            "build_vllm_commit": "correct_hash_val",
        }
        image, pinned = builder.resolve_long_term_image(
            "sparkrun-eugr-vllm",
            runtime_info,
            recipe,
        )
        assert pinned is False

    @mock.patch("sparkrun.builders._ghcr.fetch_build_index", return_value=[])
    @mock.patch("sparkrun.builders._ghcr.ghcr_list_tags")
    @mock.patch("sparkrun.builders._ghcr.ghcr_get_labels")
    def test_ghcr_api_secondary_hash_mismatch(self, mock_labels, mock_tags, mock_fetch):
        """GHCR API fallback also checks secondary hashes."""
        mock_tags.return_value = ["2025032501"]
        mock_labels.return_value = {
            "dev.sparkrun.repo-commit": "abc123def456",
            "dev.sparkrun.vllm-hash": "wrong_hash_value",
        }
        builder = EugrBuilder()
        recipe = _make_recipe()
        runtime_info = {
            "build_build_script_commit": "abc123def456",
            "build_vllm_commit": "correct_hash_val",
        }
        image, pinned = builder.resolve_long_term_image(
            "sparkrun-eugr-vllm",
            runtime_info,
            recipe,
        )
        assert pinned is False

    @mock.patch("sparkrun.builders.eugr.EugrBuilder._match_via_ghcr_api", return_value=None)
    @mock.patch("sparkrun.builders._ghcr.fetch_build_index")
    def test_prefix_matching_on_commit(self, mock_fetch, mock_api):
        """Commit matching uses 12-char prefix comparison."""
        mock_fetch.return_value = [
            {
                "tag": "2025032501",
                "variant": "nightly",
                "repo_commit": "abc123def456full_sha",
            },
        ]
        builder = EugrBuilder()
        recipe = _make_recipe()
        runtime_info = {"build_build_script_commit": "abc123def456different_suffix"}
        image, pinned = builder.resolve_long_term_image(
            "sparkrun-eugr-vllm",
            runtime_info,
            recipe,
        )
        assert pinned is True
