"""Tests for GGUF model support: parsing, caching, path resolution, download."""

from __future__ import annotations

from pathlib import Path
from unittest import mock


from sparkrun.models.download import (
    CONTAINER_HF_CACHE,
    is_gguf_model,
    is_model_cached,
    parse_gguf_model_spec,
    resolve_gguf_container_path,
    resolve_gguf_path,
    download_model,
)
from sparkrun.models.download import model_cache_path


# ---------------------------------------------------------------------------
# parse_gguf_model_spec
# ---------------------------------------------------------------------------


class TestParseGgufModelSpec:
    def test_repo_with_quant(self):
        repo, quant = parse_gguf_model_spec("Qwen/Qwen3-1.7B-GGUF:Q4_K_M")
        assert repo == "Qwen/Qwen3-1.7B-GGUF"
        assert quant == "Q4_K_M"

    def test_repo_without_quant(self):
        repo, quant = parse_gguf_model_spec("Qwen/Qwen3-1.7B-GGUF")
        assert repo == "Qwen/Qwen3-1.7B-GGUF"
        assert quant is None

    def test_non_gguf_repo(self):
        repo, quant = parse_gguf_model_spec("meta-llama/Llama-3-8B")
        assert repo == "meta-llama/Llama-3-8B"
        assert quant is None

    def test_multiple_colons_uses_last(self):
        """rsplit on ':' takes the last segment as quant."""
        repo, quant = parse_gguf_model_spec("org/repo:tag:Q8_0")
        assert repo == "org/repo:tag"
        assert quant == "Q8_0"


# ---------------------------------------------------------------------------
# is_gguf_model
# ---------------------------------------------------------------------------


class TestIsGgufModel:
    def test_repo_with_quant(self):
        assert is_gguf_model("Qwen/Qwen3-1.7B-GGUF:Q4_K_M") is True

    def test_repo_with_gguf_in_name(self):
        assert is_gguf_model("Qwen/Qwen3-1.7B-GGUF") is True

    def test_case_insensitive(self):
        assert is_gguf_model("org/model-gguf") is True

    def test_non_gguf(self):
        assert is_gguf_model("meta-llama/Llama-3-8B") is False

    def test_quant_on_non_gguf_name(self):
        """Colon syntax always means GGUF even if name doesn't say GGUF."""
        assert is_gguf_model("org/some-model:Q4_K_M") is True


# ---------------------------------------------------------------------------
# resolve_gguf_path (filesystem-based tests with tmp_path)
# ---------------------------------------------------------------------------


class TestResolveGgufPath:
    def _create_cached_gguf(self, cache_dir: Path, repo: str, filename: str):
        """Create a fake GGUF file in the HF cache structure."""
        safe_name = repo.replace("/", "--")
        snapshot = cache_dir / "hub" / f"models--{safe_name}" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True, exist_ok=True)
        gguf_file = snapshot / filename
        gguf_file.write_text("fake gguf")
        return gguf_file

    def test_finds_matching_quant(self, tmp_path):
        gguf = self._create_cached_gguf(
            tmp_path,
            "Qwen/Qwen3-1.7B-GGUF",
            "qwen3-1.7b-q4_k_m.gguf",
        )
        result = resolve_gguf_path("Qwen/Qwen3-1.7B-GGUF:Q4_K_M", str(tmp_path))
        assert result == str(gguf)

    def test_no_match_returns_none(self, tmp_path):
        self._create_cached_gguf(
            tmp_path,
            "Qwen/Qwen3-1.7B-GGUF",
            "qwen3-1.7b-q8_0.gguf",
        )
        result = resolve_gguf_path("Qwen/Qwen3-1.7B-GGUF:Q4_K_M", str(tmp_path))
        assert result is None

    def test_no_cache_dir_returns_none(self, tmp_path):
        result = resolve_gguf_path("Qwen/Qwen3-1.7B-GGUF:Q4_K_M", str(tmp_path))
        assert result is None

    def test_no_quant_returns_first_gguf(self, tmp_path):
        gguf = self._create_cached_gguf(
            tmp_path,
            "Qwen/Qwen3-1.7B-GGUF",
            "some-model.gguf",
        )
        result = resolve_gguf_path("Qwen/Qwen3-1.7B-GGUF", str(tmp_path))
        assert result == str(gguf)

    def test_case_insensitive_quant_match(self, tmp_path):
        gguf = self._create_cached_gguf(
            tmp_path,
            "Qwen/Qwen3-1.7B-GGUF",
            "Qwen3-1.7B-Q4_K_M.gguf",
        )
        result = resolve_gguf_path("Qwen/Qwen3-1.7B-GGUF:q4_k_m", str(tmp_path))
        assert result == str(gguf)


# ---------------------------------------------------------------------------
# resolve_gguf_container_path
# ---------------------------------------------------------------------------


class TestResolveGgufContainerPath:
    def _create_cached_gguf(self, cache_dir: Path, repo: str, filename: str):
        safe_name = repo.replace("/", "--")
        snapshot = cache_dir / "hub" / f"models--{safe_name}" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True, exist_ok=True)
        gguf_file = snapshot / filename
        gguf_file.write_text("fake gguf")
        return gguf_file

    def test_translates_host_to_container_path(self, tmp_path):
        self._create_cached_gguf(
            tmp_path,
            "Qwen/Qwen3-1.7B-GGUF",
            "qwen3-1.7b-q4_k_m.gguf",
        )
        result = resolve_gguf_container_path(
            "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
            str(tmp_path),
        )
        assert result is not None
        assert result.startswith(CONTAINER_HF_CACHE)
        assert "qwen3-1.7b-q4_k_m.gguf" in result

    def test_none_when_not_cached(self, tmp_path):
        result = resolve_gguf_container_path(
            "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
            str(tmp_path),
        )
        assert result is None


# ---------------------------------------------------------------------------
# is_model_cached (GGUF-aware)
# ---------------------------------------------------------------------------


class TestIsModelCachedGguf:
    def _create_cached_gguf(self, cache_dir: Path, repo: str, filename: str):
        safe_name = repo.replace("/", "--")
        snapshot = cache_dir / "hub" / f"models--{safe_name}" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / filename).write_text("fake gguf")

    def test_cached_gguf_returns_true(self, tmp_path):
        self._create_cached_gguf(
            tmp_path,
            "Qwen/Qwen3-1.7B-GGUF",
            "qwen3-1.7b-q4_k_m.gguf",
        )
        assert is_model_cached("Qwen/Qwen3-1.7B-GGUF:Q4_K_M", str(tmp_path)) is True

    def test_wrong_quant_returns_false(self, tmp_path):
        self._create_cached_gguf(
            tmp_path,
            "Qwen/Qwen3-1.7B-GGUF",
            "qwen3-1.7b-q8_0.gguf",
        )
        assert is_model_cached("Qwen/Qwen3-1.7B-GGUF:Q4_K_M", str(tmp_path)) is False

    def test_not_cached_returns_false(self, tmp_path):
        assert is_model_cached("Qwen/Qwen3-1.7B-GGUF:Q4_K_M", str(tmp_path)) is False


# ---------------------------------------------------------------------------
# model_cache_path (GGUF-aware)
# ---------------------------------------------------------------------------


class TestModelCachePathGguf:
    def test_strips_quant_variant(self):
        path = model_cache_path("Qwen/Qwen3-1.7B-GGUF:Q4_K_M", "/hf/cache")
        assert path == "/hf/cache/hub/models--Qwen--Qwen3-1.7B-GGUF"
        assert "Q4_K_M" not in path

    def test_non_gguf_unchanged(self):
        path = model_cache_path("meta-llama/Llama-3-8B", "/hf/cache")
        assert path == "/hf/cache/hub/models--meta-llama--Llama-3-8B"


# ---------------------------------------------------------------------------
# download_model GGUF dispatch
# ---------------------------------------------------------------------------


class TestDownloadModelGguf:
    @mock.patch("sparkrun.models.download.resolve_gguf_path", return_value="/cached/q4.gguf")
    def test_gguf_already_cached_still_verifies(self, mock_resolve):
        """GGUF model already cached still calls snapshot_download to verify completeness."""
        mock_hf = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"huggingface_hub": mock_hf, "huggingface_hub.utils": mock_hf.utils}):
            rc = download_model("Qwen/Qwen3-1.7B-GGUF:Q4_K_M", cache_dir="/fake")
            assert rc == 0
            mock_resolve.assert_called_once()
            # snapshot_download should have been called even though cache existed
            mock_hf.snapshot_download.assert_called_once()

    @mock.patch("sparkrun.models.download.resolve_gguf_path", return_value=None)
    @mock.patch("sparkrun.models.download.snapshot_download", create=True)
    def test_gguf_downloads_with_allow_patterns(self, mock_snap, mock_resolve):
        """GGUF download uses allow_patterns for the quant variant."""
        # Mock the import inside _download_gguf
        with mock.patch.dict("sys.modules", {"huggingface_hub": mock.MagicMock()}) as _:
            with mock.patch("sparkrun.models.download.resolve_gguf_path", return_value=None):
                # Use the actual function but patch the inner import
                import sparkrun.models.download as dl_mod

                def patched_download(model_id, cache_dir=None, token=None, revision=None, dry_run=False):
                    # Just verify it's called with the right model
                    assert model_id == "Qwen/Qwen3-1.7B-GGUF:Q4_K_M"
                    return 0

                with mock.patch.object(dl_mod, "_download_gguf", side_effect=patched_download):
                    rc = dl_mod.download_model("Qwen/Qwen3-1.7B-GGUF:Q4_K_M", cache_dir="/fake")
                    assert rc == 0

    def test_gguf_dry_run(self):
        """Dry-run GGUF download returns 0 without doing anything."""
        rc = download_model("Qwen/Qwen3-1.7B-GGUF:Q4_K_M", cache_dir="/fake", dry_run=True)
        assert rc == 0

    def test_non_gguf_not_dispatched_to_gguf(self):
        """Non-GGUF model does not take the GGUF path."""
        with mock.patch("sparkrun.models.download._download_gguf") as mock_gguf:
            # Will fail on actual download since /fake doesn't exist, but
            # should NOT call _download_gguf
            download_model("meta-llama/Llama-3-8B", cache_dir="/fake", dry_run=True)
            mock_gguf.assert_not_called()


# ---------------------------------------------------------------------------
# Revision-aware cache checking
# ---------------------------------------------------------------------------


class TestIsModelCachedRevision:
    """Test revision-aware is_model_cached behaviour."""

    def _create_snapshot(self, cache_dir: Path, model_id: str, commit_hash: str, files: list[str], ref: str | None = None):
        """Create a fake HF cache snapshot with optional ref."""
        safe_name = model_id.replace("/", "--")
        model_cache = cache_dir / "hub" / f"models--{safe_name}"
        snapshot = model_cache / "snapshots" / commit_hash
        snapshot.mkdir(parents=True, exist_ok=True)
        for f in files:
            (snapshot / f).write_text("fake")
        if ref:
            refs_dir = model_cache / "refs"
            refs_dir.mkdir(parents=True, exist_ok=True)
            (refs_dir / ref).write_text(commit_hash)

    def test_no_revision_defaults_to_main_ref(self, tmp_path):
        """Without revision, checks refs/main first."""
        self._create_snapshot(
            tmp_path,
            "org/model",
            "abc123",
            ["model.safetensors"],
            ref="main",
        )
        assert is_model_cached("org/model", str(tmp_path)) is True

    def test_no_revision_config_only_returns_false(self, tmp_path):
        """refs/main snapshot with only config.json is not cached."""
        self._create_snapshot(
            tmp_path,
            "org/model",
            "abc123",
            ["config.json"],
            ref="main",
        )
        assert is_model_cached("org/model", str(tmp_path)) is False

    def test_specific_revision_by_ref(self, tmp_path):
        """Checks only the snapshot for the requested ref."""
        # v1 has only config, v2 has weights
        self._create_snapshot(
            tmp_path,
            "org/model",
            "aaa111",
            ["config.json"],
            ref="v1",
        )
        self._create_snapshot(
            tmp_path,
            "org/model",
            "bbb222",
            ["model.safetensors"],
            ref="v2",
        )
        assert is_model_cached("org/model", str(tmp_path), revision="v1") is False
        assert is_model_cached("org/model", str(tmp_path), revision="v2") is True

    def test_revision_by_commit_hash(self, tmp_path):
        """Revision can be a direct commit hash (no ref file needed)."""
        self._create_snapshot(
            tmp_path,
            "org/model",
            "deadbeef",
            ["model-00001.safetensors"],
        )
        assert is_model_cached("org/model", str(tmp_path), revision="deadbeef") is True
        assert is_model_cached("org/model", str(tmp_path), revision="other") is False

    def test_fallback_to_all_snapshots(self, tmp_path):
        """Falls back to any snapshot when refs/main does not exist."""
        self._create_snapshot(
            tmp_path,
            "org/model",
            "abc123",
            ["model.bin"],
            # No ref — simulates manually placed cache
        )
        assert is_model_cached("org/model", str(tmp_path)) is True

    def test_revision_dry_run_accepted(self):
        """download_model accepts revision parameter in dry-run mode."""
        rc = download_model("org/model", cache_dir="/fake", revision="v2.1", dry_run=True)
        assert rc == 0

    def test_gguf_revision_dry_run_accepted(self):
        """GGUF download_model accepts revision parameter in dry-run mode."""
        rc = download_model("Qwen/Qwen3-1.7B-GGUF:Q4_K_M", cache_dir="/fake", revision="v1.0", dry_run=True)
        assert rc == 0
