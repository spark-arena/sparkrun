"""Tests for eugr build cache — skip redundant builds via wheel hash detection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from sparkrun.builders.eugr import (
    EugrBuilder,
    EUGR_BUILD_CACHE_NAME,
    _load_build_cache,
    _save_build_cache,
    _fetch_upstream_wheel_hashes,
    _CACHEABLE_BUILD_ARGS,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class TestLoadSaveBuildCache:

    def test_round_trip(self, tmp_path):
        data = {"sparkrun-eugr-vllm": {"repo_commit": "abc123"}}
        _save_build_cache(tmp_path, data)
        loaded = _load_build_cache(tmp_path)
        assert loaded == data

    def test_load_missing_file(self, tmp_path):
        assert _load_build_cache(tmp_path) == {}

    def test_load_corrupt_json(self, tmp_path):
        (tmp_path / EUGR_BUILD_CACHE_NAME).write_text("not json{")
        assert _load_build_cache(tmp_path) == {}

    def test_save_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b"
        _save_build_cache(nested, {"key": "val"})
        assert _load_build_cache(nested) == {"key": "val"}


class TestFetchUpstreamWheelHashes:

    def _mock_release(self, name: str):
        m = mock.MagicMock()
        m.read.return_value = json.dumps({"name": name}).encode()
        m.__enter__ = mock.Mock(return_value=m)
        m.__exit__ = mock.Mock(return_value=False)
        return m

    def test_parses_both_hashes(self):
        vllm_resp = self._mock_release(
            "Prebuilt vLLM Wheels (0.18.1rc1.dev121+gcd7643015.d20260325.cu132)"
        )
        fi_resp = self._mock_release(
            "Prebuilt FlashInfer Wheels (0.6.7-ede7a275-d20260325)"
        )
        with mock.patch("urllib.request.urlopen", side_effect=[vllm_resp, fi_resp]):
            result = _fetch_upstream_wheel_hashes()
        assert result == {"vllm_commit": "cd7643015", "flashinfer_commit": "ede7a275"}

    def test_returns_empty_on_network_error(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            assert _fetch_upstream_wheel_hashes() == {}

    def test_returns_empty_on_unparseable_name(self):
        resp = self._mock_release("Something unexpected")
        with mock.patch("urllib.request.urlopen", return_value=resp):
            assert _fetch_upstream_wheel_hashes() == {}


# ---------------------------------------------------------------------------
# EugrBuilder._cache_key
# ---------------------------------------------------------------------------

class TestCacheKey:

    def test_local_key(self):
        assert EugrBuilder._cache_key("sparkrun-eugr-vllm") == "sparkrun-eugr-vllm"

    def test_local_key_explicit_none(self):
        assert EugrBuilder._cache_key("sparkrun-eugr-vllm", host=None) == "sparkrun-eugr-vllm"

    def test_delegated_key(self):
        assert EugrBuilder._cache_key("sparkrun-eugr-vllm", host="head1") == "head1:sparkrun-eugr-vllm"


# ---------------------------------------------------------------------------
# EugrBuilder._get_image_id_on_host / _get_repo_head_on_host
# ---------------------------------------------------------------------------

def _remote_result(success: bool, stdout: str = "", stderr: str = ""):
    from sparkrun.orchestration.ssh import RemoteResult
    return RemoteResult(host="h", returncode=0 if success else 1, stdout=stdout, stderr=stderr)


class TestGetImageIdOnHost:

    def test_returns_id_on_success(self):
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host",
                        return_value=_remote_result(True, stdout="sha256:abc123\n")):
            assert EugrBuilder._get_image_id_on_host("img", "host1") == "sha256:abc123"

    def test_returns_none_on_failure(self):
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host",
                        return_value=_remote_result(False)):
            assert EugrBuilder._get_image_id_on_host("img", "host1") is None

    def test_returns_none_on_empty_stdout(self):
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host",
                        return_value=_remote_result(True, stdout="  \n")):
            assert EugrBuilder._get_image_id_on_host("img", "host1") is None


class TestGetRepoHeadOnHost:

    def test_returns_commit_on_success(self):
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host",
                        return_value=_remote_result(True, stdout="abc123def456\n")):
            assert EugrBuilder._get_repo_head_on_host("/repo", "host1") == "abc123def456"

    def test_returns_none_on_failure(self):
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host",
                        return_value=_remote_result(False)):
            assert EugrBuilder._get_repo_head_on_host("/repo", "host1") is None


# ---------------------------------------------------------------------------
# EugrBuilder._can_skip_build
# ---------------------------------------------------------------------------

_CACHE_ENTRY = {
    "build_args": [],
    "repo_commit": "abc123def456",
    "vllm_commit": "cd7643015",
    "flashinfer_commit": "ede7a275",
    "image_id": "sha256:deadbeef",
    "built_at": "2026-03-25T00:00:00+00:00",
}


def _make_builder(tmp_path, repo_dir=None) -> EugrBuilder:
    builder = EugrBuilder()
    builder._repo_dir = repo_dir or tmp_path / "repo"
    return builder


def _write_cache(tmp_path, image="sparkrun-eugr-vllm", entry=None):
    _save_build_cache(tmp_path, {image: entry or _CACHE_ENTRY})


def _mock_config(tmp_path):
    cfg = mock.MagicMock()
    cfg.cache_dir = str(tmp_path)
    return cfg


class TestCanSkipBuild:

    def test_cache_hit_skips_build(self, tmp_path):
        _write_cache(tmp_path)
        builder = _make_builder(tmp_path)
        config = _mock_config(tmp_path)

        with mock.patch("sparkrun.containers.registry.get_image_id", return_value="sha256:deadbeef"), \
             mock.patch("subprocess.run") as mock_run, \
             mock.patch("sparkrun.builders.eugr._fetch_upstream_wheel_hashes",
                        return_value={"vllm_commit": "cd7643015", "flashinfer_commit": "ede7a275"}):
            mock_run.return_value = mock.MagicMock(returncode=0, stdout="abc123def456\n")
            assert builder._can_skip_build("sparkrun-eugr-vllm", [], config) is True

    def test_cache_miss_new_vllm_wheel(self, tmp_path):
        _write_cache(tmp_path)
        builder = _make_builder(tmp_path)
        config = _mock_config(tmp_path)

        with mock.patch("sparkrun.containers.registry.get_image_id", return_value="sha256:deadbeef"), \
             mock.patch("subprocess.run") as mock_run, \
             mock.patch("sparkrun.builders.eugr._fetch_upstream_wheel_hashes",
                        return_value={"vllm_commit": "newcommit99", "flashinfer_commit": "ede7a275"}):
            mock_run.return_value = mock.MagicMock(returncode=0, stdout="abc123def456\n")
            assert builder._can_skip_build("sparkrun-eugr-vllm", [], config) is False

    def test_cache_miss_new_flashinfer_wheel(self, tmp_path):
        _write_cache(tmp_path)
        builder = _make_builder(tmp_path)
        config = _mock_config(tmp_path)

        with mock.patch("sparkrun.containers.registry.get_image_id", return_value="sha256:deadbeef"), \
             mock.patch("subprocess.run") as mock_run, \
             mock.patch("sparkrun.builders.eugr._fetch_upstream_wheel_hashes",
                        return_value={"vllm_commit": "cd7643015", "flashinfer_commit": "newflash99"}):
            mock_run.return_value = mock.MagicMock(returncode=0, stdout="abc123def456\n")
            assert builder._can_skip_build("sparkrun-eugr-vllm", [], config) is False

    def test_cache_miss_repo_commit_changed(self, tmp_path):
        _write_cache(tmp_path)
        builder = _make_builder(tmp_path)
        config = _mock_config(tmp_path)

        with mock.patch("sparkrun.containers.registry.get_image_id", return_value="sha256:deadbeef"), \
             mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0, stdout="different_commit\n")
            assert builder._can_skip_build("sparkrun-eugr-vllm", [], config) is False

    def test_cache_miss_image_deleted(self, tmp_path):
        _write_cache(tmp_path)
        builder = _make_builder(tmp_path)
        config = _mock_config(tmp_path)

        with mock.patch("sparkrun.containers.registry.get_image_id", return_value=None):
            assert builder._can_skip_build("sparkrun-eugr-vllm", [], config) is False

    def test_cache_miss_image_id_changed(self, tmp_path):
        _write_cache(tmp_path)
        builder = _make_builder(tmp_path)
        config = _mock_config(tmp_path)

        with mock.patch("sparkrun.containers.registry.get_image_id", return_value="sha256:different"):
            assert builder._can_skip_build("sparkrun-eugr-vllm", [], config) is False

    def test_cache_miss_api_failure(self, tmp_path):
        _write_cache(tmp_path)
        builder = _make_builder(tmp_path)
        config = _mock_config(tmp_path)

        with mock.patch("sparkrun.containers.registry.get_image_id", return_value="sha256:deadbeef"), \
             mock.patch("subprocess.run") as mock_run, \
             mock.patch("sparkrun.builders.eugr._fetch_upstream_wheel_hashes", return_value={}):
            mock_run.return_value = mock.MagicMock(returncode=0, stdout="abc123def456\n")
            assert builder._can_skip_build("sparkrun-eugr-vllm", [], config) is False

    def test_exotic_build_args_bypass(self, tmp_path):
        """Non-cacheable build_args always return False without checking anything."""
        builder = _make_builder(tmp_path)
        config = _mock_config(tmp_path)
        assert builder._can_skip_build("sparkrun-eugr-vllm", ["--custom-flag"], config) is False

    def test_tf5_build_args_are_cacheable(self, tmp_path):
        entry = {**_CACHE_ENTRY, "build_args": ["--tf5"]}
        _write_cache(tmp_path, image="sparkrun-eugr-vllm-tf5", entry=entry)
        builder = _make_builder(tmp_path)
        config = _mock_config(tmp_path)

        with mock.patch("sparkrun.containers.registry.get_image_id", return_value="sha256:deadbeef"), \
             mock.patch("subprocess.run") as mock_run, \
             mock.patch("sparkrun.builders.eugr._fetch_upstream_wheel_hashes",
                        return_value={"vllm_commit": "cd7643015", "flashinfer_commit": "ede7a275"}):
            mock_run.return_value = mock.MagicMock(returncode=0, stdout="abc123def456\n")
            assert builder._can_skip_build("sparkrun-eugr-vllm-tf5", ["--tf5"], config) is True

    def test_no_cache_entry(self, tmp_path):
        builder = _make_builder(tmp_path)
        config = _mock_config(tmp_path)
        assert builder._can_skip_build("sparkrun-eugr-vllm", [], config) is False

    def test_build_args_mismatch(self, tmp_path):
        entry = {**_CACHE_ENTRY, "build_args": ["--tf5"]}
        _write_cache(tmp_path, entry=entry)
        builder = _make_builder(tmp_path)
        config = _mock_config(tmp_path)

        with mock.patch("sparkrun.containers.registry.get_image_id", return_value="sha256:deadbeef"):
            assert builder._can_skip_build("sparkrun-eugr-vllm", [], config) is False


# ---------------------------------------------------------------------------
# EugrBuilder._can_skip_build — delegated mode
# ---------------------------------------------------------------------------

class TestCanSkipBuildDelegated:

    def test_delegated_cache_hit(self, tmp_path):
        """Cache hit with host-qualified key, remote Docker/git checks."""
        entry = {**_CACHE_ENTRY}
        _save_build_cache(tmp_path, {"head1:sparkrun-eugr-vllm": entry})
        builder = _make_builder(tmp_path)
        config = _mock_config(tmp_path)

        with mock.patch.object(EugrBuilder, "_get_image_id_on_host", return_value="sha256:deadbeef"), \
             mock.patch.object(EugrBuilder, "_get_repo_head_on_host", return_value="abc123def456"), \
             mock.patch("sparkrun.builders.eugr._fetch_upstream_wheel_hashes",
                        return_value={"vllm_commit": "cd7643015", "flashinfer_commit": "ede7a275"}):
            assert builder._can_skip_build(
                "sparkrun-eugr-vllm", [], config, host="head1", ssh_kwargs={}
            ) is True

    def test_delegated_cache_miss_no_entry(self, tmp_path):
        """Local-only cache entry does not match delegated lookup key."""
        _write_cache(tmp_path)  # keyed as plain "sparkrun-eugr-vllm"
        builder = _make_builder(tmp_path)
        config = _mock_config(tmp_path)

        assert builder._can_skip_build(
            "sparkrun-eugr-vllm", [], config, host="head1", ssh_kwargs={}
        ) is False

    def test_delegated_cache_miss_image_gone(self, tmp_path):
        _save_build_cache(tmp_path, {"head1:sparkrun-eugr-vllm": _CACHE_ENTRY})
        builder = _make_builder(tmp_path)
        config = _mock_config(tmp_path)

        with mock.patch.object(EugrBuilder, "_get_image_id_on_host", return_value=None):
            assert builder._can_skip_build(
                "sparkrun-eugr-vllm", [], config, host="head1", ssh_kwargs={}
            ) is False

    def test_delegated_cache_miss_repo_changed(self, tmp_path):
        _save_build_cache(tmp_path, {"head1:sparkrun-eugr-vllm": _CACHE_ENTRY})
        builder = _make_builder(tmp_path)
        config = _mock_config(tmp_path)

        with mock.patch.object(EugrBuilder, "_get_image_id_on_host", return_value="sha256:deadbeef"), \
             mock.patch.object(EugrBuilder, "_get_repo_head_on_host", return_value="different_commit"):
            assert builder._can_skip_build(
                "sparkrun-eugr-vllm", [], config, host="head1", ssh_kwargs={}
            ) is False

    def test_delegated_cache_miss_upstream_changed(self, tmp_path):
        _save_build_cache(tmp_path, {"head1:sparkrun-eugr-vllm": _CACHE_ENTRY})
        builder = _make_builder(tmp_path)
        config = _mock_config(tmp_path)

        with mock.patch.object(EugrBuilder, "_get_image_id_on_host", return_value="sha256:deadbeef"), \
             mock.patch.object(EugrBuilder, "_get_repo_head_on_host", return_value="abc123def456"), \
             mock.patch("sparkrun.builders.eugr._fetch_upstream_wheel_hashes",
                        return_value={"vllm_commit": "newcommit99", "flashinfer_commit": "ede7a275"}):
            assert builder._can_skip_build(
                "sparkrun-eugr-vllm", [], config, host="head1", ssh_kwargs={}
            ) is False


# ---------------------------------------------------------------------------
# EugrBuilder._save_build_metadata
# ---------------------------------------------------------------------------

class TestSaveBuildMetadata:

    def test_saves_full_metadata(self, tmp_path):
        builder = _make_builder(tmp_path, repo_dir=tmp_path / "repo")
        config = _mock_config(tmp_path)

        metadata_yaml = "vllm_commit: cd7643015\nflinfer_commit: ede7a275\n"

        git_result = mock.MagicMock(returncode=0, stdout="abc123def456\n")
        docker_result = mock.MagicMock(returncode=0, stdout=metadata_yaml)

        with mock.patch("subprocess.run", side_effect=[git_result, docker_result]), \
             mock.patch("sparkrun.containers.registry.get_image_id", return_value="sha256:newimage"), \
             mock.patch.object(builder, "process_version_info", return_value={
                 "build_vllm_commit": "cd7643015",
                 "build_flashinfer_commit": "ede7a275",
             }):
            builder._save_build_metadata("sparkrun-eugr-vllm", [], config)

        cache = _load_build_cache(tmp_path)
        entry = cache["sparkrun-eugr-vllm"]
        assert entry["repo_commit"] == "abc123def456"
        assert entry["vllm_commit"] == "cd7643015"
        assert entry["flashinfer_commit"] == "ede7a275"
        assert entry["image_id"] == "sha256:newimage"
        assert entry["build_args"] == []
        assert "built_at" in entry

    def test_handles_git_failure_gracefully(self, tmp_path):
        builder = _make_builder(tmp_path, repo_dir=tmp_path / "repo")
        config = _mock_config(tmp_path)

        git_result = mock.MagicMock(returncode=1, stdout="")
        docker_result = mock.MagicMock(returncode=0, stdout="vllm_commit: abc\n")

        with mock.patch("subprocess.run", side_effect=[git_result, docker_result]), \
             mock.patch("sparkrun.containers.registry.get_image_id", return_value="sha256:img"), \
             mock.patch.object(builder, "process_version_info", return_value={}):
            builder._save_build_metadata("sparkrun-eugr-vllm", [], config)

        cache = _load_build_cache(tmp_path)
        assert cache["sparkrun-eugr-vllm"]["repo_commit"] is None

    def test_handles_docker_failure_gracefully(self, tmp_path):
        builder = _make_builder(tmp_path, repo_dir=tmp_path / "repo")
        config = _mock_config(tmp_path)

        git_result = mock.MagicMock(returncode=0, stdout="abc123\n")
        docker_result = mock.MagicMock(returncode=1, stdout="")

        with mock.patch("subprocess.run", side_effect=[git_result, docker_result]), \
             mock.patch("sparkrun.containers.registry.get_image_id", return_value="sha256:img"):
            builder._save_build_metadata("sparkrun-eugr-vllm", [], config)

        cache = _load_build_cache(tmp_path)
        entry = cache["sparkrun-eugr-vllm"]
        assert entry["vllm_commit"] is None
        assert entry["flashinfer_commit"] is None


# ---------------------------------------------------------------------------
# EugrBuilder._save_build_metadata — delegated mode
# ---------------------------------------------------------------------------

class TestSaveBuildMetadataDelegated:

    def test_saves_with_host_key(self, tmp_path):
        builder = _make_builder(tmp_path, repo_dir=tmp_path / "repo")
        config = _mock_config(tmp_path)

        with mock.patch.object(EugrBuilder, "_get_repo_head_on_host", return_value="abc123def456"), \
             mock.patch("sparkrun.orchestration.primitives.run_script_on_host",
                        return_value=_remote_result(True, stdout="vllm_commit: cd7643015\n")), \
             mock.patch.object(builder, "process_version_info", return_value={
                 "build_vllm_commit": "cd7643015",
                 "build_flashinfer_commit": "ede7a275",
             }), \
             mock.patch.object(EugrBuilder, "_get_image_id_on_host", return_value="sha256:remoteimg"):
            builder._save_build_metadata("sparkrun-eugr-vllm", [], config, host="head1", ssh_kwargs={})

        cache = _load_build_cache(tmp_path)
        assert "head1:sparkrun-eugr-vllm" in cache
        assert "sparkrun-eugr-vllm" not in cache
        entry = cache["head1:sparkrun-eugr-vllm"]
        assert entry["repo_commit"] == "abc123def456"
        assert entry["image_id"] == "sha256:remoteimg"
        assert entry["vllm_commit"] == "cd7643015"

    def test_handles_ssh_failure_gracefully(self, tmp_path):
        builder = _make_builder(tmp_path, repo_dir=tmp_path / "repo")
        config = _mock_config(tmp_path)

        with mock.patch.object(EugrBuilder, "_get_repo_head_on_host", return_value=None), \
             mock.patch("sparkrun.orchestration.primitives.run_script_on_host",
                        return_value=_remote_result(False)), \
             mock.patch.object(EugrBuilder, "_get_image_id_on_host", return_value=None):
            builder._save_build_metadata("sparkrun-eugr-vllm", [], config, host="head1", ssh_kwargs={})

        cache = _load_build_cache(tmp_path)
        entry = cache["head1:sparkrun-eugr-vllm"]
        assert entry["repo_commit"] is None
        assert entry["image_id"] is None
        assert entry["vllm_commit"] is None


# ---------------------------------------------------------------------------
# Integration: prepare_image skip path
# ---------------------------------------------------------------------------

class TestPrepareImageCacheIntegration:

    def test_skip_build_when_cache_hit(self, tmp_path):
        """When _can_skip_build returns True, _build_image should NOT be called."""
        builder = EugrBuilder()
        recipe = mock.MagicMock()
        recipe.runtime_config = {"build_args": [], "mods": []}
        recipe.pre_exec = []
        config = _mock_config(tmp_path)

        with mock.patch.object(builder, "ensure_repo", return_value=tmp_path / "repo"), \
             mock.patch.object(builder, "_can_skip_build", return_value=True), \
             mock.patch.object(builder, "_build_image") as mock_build, \
             mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            result = builder.prepare_image(
                "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest",
                recipe, ["host1"], config=config, dry_run=False,
            )

        mock_build.assert_not_called()
        assert result == "sparkrun-eugr-vllm"

    def test_build_when_cache_miss(self, tmp_path):
        """When _can_skip_build returns False, _build_image should be called."""
        builder = EugrBuilder()
        recipe = mock.MagicMock()
        recipe.runtime_config = {"build_args": [], "mods": []}
        recipe.pre_exec = []
        config = _mock_config(tmp_path)

        with mock.patch.object(builder, "ensure_repo", return_value=tmp_path / "repo"), \
             mock.patch.object(builder, "_can_skip_build", return_value=False), \
             mock.patch.object(builder, "_build_image") as mock_build, \
             mock.patch.object(builder, "_save_build_metadata") as mock_save, \
             mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            result = builder.prepare_image(
                "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest",
                recipe, ["host1"], config=config, dry_run=False,
            )

        mock_build.assert_called_once()
        mock_save.assert_called_once()
        assert result == "sparkrun-eugr-vllm"

    def test_dry_run_skips_cache_check(self, tmp_path):
        """In dry_run mode, cache check is not performed."""
        builder = EugrBuilder()
        recipe = mock.MagicMock()
        recipe.runtime_config = {"build_args": [], "mods": []}
        recipe.pre_exec = []
        config = _mock_config(tmp_path)

        with mock.patch.object(builder, "ensure_repo", return_value=tmp_path / "repo"), \
             mock.patch.object(builder, "_can_skip_build") as mock_skip, \
             mock.patch.object(builder, "_build_image") as mock_build, \
             mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            builder.prepare_image(
                "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest",
                recipe, ["host1"], config=config, dry_run=True,
            )

        mock_skip.assert_not_called()

    def test_delegated_mode_checks_cache(self, tmp_path):
        """In delegated mode, cache check IS performed with host param."""
        builder = EugrBuilder()
        recipe = mock.MagicMock()
        recipe.runtime_config = {"build_args": ["--tf5"], "mods": []}
        recipe.pre_exec = []
        config = _mock_config(tmp_path)

        with mock.patch.object(builder, "_ensure_repo_remote", return_value="/remote/path"), \
             mock.patch.object(builder, "_can_skip_build", return_value=True) as mock_skip, \
             mock.patch.object(builder, "_build_image_remote") as mock_build:
            builder.prepare_image(
                "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:latest",
                recipe, ["host1"], config=config, dry_run=False,
                transfer_mode="delegated", ssh_kwargs={},
            )

        mock_skip.assert_called_once_with(
            "sparkrun-eugr-vllm-tf5", ["--tf5"], config,
            host="host1", ssh_kwargs={},
        )
        mock_build.assert_not_called()

    def test_delegated_build_saves_metadata(self, tmp_path):
        """After a delegated build, metadata is saved with host param."""
        builder = EugrBuilder()
        recipe = mock.MagicMock()
        recipe.runtime_config = {"build_args": [], "mods": []}
        recipe.pre_exec = []
        config = _mock_config(tmp_path)

        with mock.patch.object(builder, "_ensure_repo_remote", return_value="/remote/path"), \
             mock.patch.object(builder, "_can_skip_build", return_value=False), \
             mock.patch.object(builder, "_build_image_remote") as mock_build, \
             mock.patch.object(builder, "_save_build_metadata") as mock_save, \
             mock.patch.object(builder, "_image_exists_on_host", return_value=False):
            builder.prepare_image(
                "my-image", recipe, ["head1"], config=config, dry_run=False,
                transfer_mode="delegated", ssh_kwargs={"user": "u"},
            )

        mock_build.assert_called_once()
        mock_save.assert_called_once_with(
            "my-image", [], config, host="head1", ssh_kwargs={"user": "u"},
        )
