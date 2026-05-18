"""Tests for sparkrun builder plugin system."""

from __future__ import annotations

from unittest import mock

import pytest

from sparkrun.core.recipe import Recipe


# ---------------------------------------------------------------------------
# BuilderPlugin base class
# ---------------------------------------------------------------------------


class TestBuilderPluginBase:
    """Tests for BuilderPlugin abstract base class."""

    def _make_plugin(self):
        """Return an instance of the concrete base class."""
        from sparkrun.builders.base import BuilderPlugin

        return BuilderPlugin()

    def test_builder_plugin_is_enabled_false(self):
        """is_enabled returns False — required for multi-extension registration."""
        from scitrera_app_framework import Variables

        plugin = self._make_plugin()
        v = Variables()
        assert plugin.is_enabled(v) is False

    def test_builder_plugin_is_multi_extension_true(self):
        """is_multi_extension returns True."""
        from scitrera_app_framework import Variables

        plugin = self._make_plugin()
        v = Variables()
        assert plugin.is_multi_extension(v) is True

    def test_builder_plugin_prepare_image_returns_unchanged(self):
        """Default prepare_image returns the image arg unchanged."""
        plugin = self._make_plugin()
        recipe = Recipe.from_dict({"name": "test", "model": "some/model", "runtime": "vllm"})
        result = plugin.prepare_image("my-image:latest", recipe, ["10.0.0.1"])
        assert result == "my-image:latest"

    def test_builder_plugin_validate_recipe_returns_empty(self):
        """Default validate_recipe returns an empty list."""
        plugin = self._make_plugin()
        recipe = Recipe.from_dict({"name": "test", "model": "some/model", "runtime": "vllm"})
        assert plugin.validate_recipe(recipe) == []

    def test_builder_plugin_repr(self):
        """repr includes builder_name."""
        from sparkrun.builders.base import BuilderPlugin

        plugin = BuilderPlugin()
        plugin.builder_name = "my-builder"
        r = repr(plugin)
        assert "BuilderPlugin" in r
        assert "my-builder" in r


# ---------------------------------------------------------------------------
# DockerPullBuilder
# ---------------------------------------------------------------------------


class TestDockerPullBuilder:
    """Tests for DockerPullBuilder."""

    def _make_builder(self):
        from sparkrun.builders.docker_pull import DockerPullBuilder

        return DockerPullBuilder()

    def test_docker_pull_builder_name(self):
        """builder_name == 'docker-pull'."""
        assert self._make_builder().builder_name == "docker-pull"

    def test_docker_pull_prepare_image_noop(self):
        """prepare_image returns image unchanged (no subprocess calls)."""
        builder = self._make_builder()
        recipe = Recipe.from_dict({"name": "test", "model": "some/model", "runtime": "vllm"})
        with mock.patch("subprocess.run") as mock_run:
            result = builder.prepare_image("docker-image:latest", recipe, ["10.0.0.1"])
        assert result == "docker-image:latest"
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# EugrBuilder
# ---------------------------------------------------------------------------


@pytest.fixture
def eugr_builder_with_repo(tmp_path):
    """Create EugrBuilder with a fake repo containing build-and-copy.sh and a sample mod."""
    from sparkrun.builders.eugr import EugrBuilder

    builder = EugrBuilder()
    repo_dir = tmp_path / "eugr-repo"
    repo_dir.mkdir()
    build_script = repo_dir / "build-and-copy.sh"
    build_script.write_text("#!/bin/bash\nexit 0\n")
    build_script.chmod(0o755)
    mod_dir = repo_dir / "mods" / "flash-attn"
    mod_dir.mkdir(parents=True)
    (mod_dir / "run.sh").write_text("#!/bin/bash\necho applied\n")
    return builder, repo_dir


class TestEugrBuilderName:
    """Test EugrBuilder identity."""

    def test_eugr_builder_name(self):
        """builder_name == 'eugr'."""
        from sparkrun.builders.eugr import EugrBuilder

        assert EugrBuilder().builder_name == "eugr"


class TestEugrPrepareImage:
    """Test EugrBuilder.prepare_image() — build and mod injection."""

    def test_eugr_prepare_with_build_args(self, eugr_builder_with_repo):
        """prepare_image() calls build-and-copy.sh when build_args are present and image missing."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "eugr-vllm",
                "container": "my-image",
                "runtime_config": {"build_args": ["--some-flag"]},
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                with mock.patch("sparkrun.builders.eugr._run_build_capturing", return_value=(0, "")) as mock_build:
                    with mock.patch.object(builder, "_verify_image_imports"):
                        with mock.patch.object(builder, "_save_build_metadata"):
                            result = builder.prepare_image("my-image", recipe, ["10.0.0.1"])

        assert result == "my-image"
        mock_build.assert_called_once()
        cmd = mock_build.call_args[0][0]
        assert str(repo_dir / "build-and-copy.sh") in cmd[0]
        assert "-t" in cmd
        assert "my-image" in cmd
        assert "--some-flag" in cmd
        # --cleanup is always appended to the effective build args (issue #164)
        assert "--cleanup" in cmd

    def test_eugr_prepare_without_build_args_image_exists(self, eugr_builder_with_repo):
        """prepare_image() is a no-op when no build_args/mods and image exists locally."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "eugr-vllm",
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=True):
            with mock.patch.object(builder, "ensure_repo") as mock_ensure:
                result = builder.prepare_image("vllm-node", recipe, ["10.0.0.1"])
                mock_ensure.assert_not_called()

        assert result == "vllm-node"

    def test_eugr_prepare_builds_when_image_missing(self, eugr_builder_with_repo):
        """prepare_image() triggers a build when image is not found locally."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "eugr-vllm",
                "container": "my-image",
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                with mock.patch("sparkrun.builders.eugr._run_build_capturing", return_value=(0, "")) as mock_build:
                    with mock.patch.object(builder, "_verify_image_imports"):
                        with mock.patch.object(builder, "_save_build_metadata"):
                            result = builder.prepare_image("my-image", recipe, ["10.0.0.1"])

        mock_build.assert_called_once()
        cmd = mock_build.call_args[0][0]
        assert str(repo_dir / "build-and-copy.sh") in cmd[0]
        assert "-t" in cmd
        assert "my-image" in cmd
        assert result == "my-image"

    def test_eugr_prepare_dry_run(self, eugr_builder_with_repo):
        """prepare_image() in dry-run does not execute subprocess (build script)."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "eugr-vllm",
                "runtime_config": {"build_args": ["--flag"]},
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                with mock.patch("sparkrun.builders.eugr._run_build_capturing") as mock_build:
                    with mock.patch.object(builder, "_verify_image_imports") as mock_verify:
                        builder.prepare_image("vllm-node", recipe, ["10.0.0.1"], dry_run=True)

        mock_build.assert_not_called()
        mock_verify.assert_not_called()

    # Note: mod -> pre_exec injection moved out of the eugr builder into the
    # generic core/mods.py resolver. See tests/test_mods.py for that coverage.

    def test_eugr_prepare_build_failure_raises(self, eugr_builder_with_repo):
        """prepare_image() raises RuntimeError when the build exits non-zero."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "eugr-vllm",
                "runtime_config": {"build_args": ["--flag"]},
            }
        )
        # Simulate FlashInfer download exhausting all options.
        captured_output = (
            "Phase 1\n"
            "Could not fetch release metadata for 'prebuilt-flashinfer-current' — skipping download.\n"
            "No FlashInfer wheels available (download failed) — building...\n"
            "FlashInfer build failed — restoring previous wheels...\n"
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                with mock.patch(
                    "sparkrun.builders.eugr._run_build_capturing",
                    return_value=(1, captured_output),
                ):
                    with pytest.raises(RuntimeError) as excinfo:
                        builder.prepare_image("vllm-node", recipe, ["10.0.0.1"])

        msg = str(excinfo.value)
        assert "eugr container build failed" in msg
        # Phase identification should pick up the explicit failure marker.
        assert "flashinfer-build (failed)" in msg

    def test_eugr_prepare_appends_cleanup_without_duplicates(self, eugr_builder_with_repo):
        """`--cleanup` is not duplicated if already present in build_args."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "eugr-vllm",
                "container": "my-image",
                "runtime_config": {"build_args": ["--cleanup", "--flag"]},
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                with mock.patch("sparkrun.builders.eugr._run_build_capturing", return_value=(0, "")) as mock_build:
                    with mock.patch.object(builder, "_verify_image_imports"):
                        with mock.patch.object(builder, "_save_build_metadata"):
                            builder.prepare_image("my-image", recipe, ["10.0.0.1"])

        cmd = mock_build.call_args[0][0]
        # Exactly one --cleanup, regardless of recipe ordering.
        assert cmd.count("--cleanup") == 1

    def test_eugr_prepare_cleanup_not_persisted_to_cache(self, eugr_builder_with_repo):
        """`--cleanup` is appended to the build invocation but NOT to the cached build_args.

        This keeps the build cache identity tied to the recipe's canonical args so
        subsequent cache lookups still hit when nothing else changed (commit 110aca7).
        """
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "eugr-vllm",
                "container": "my-image",
                "runtime_config": {"build_args": ["--tf5"]},
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                with mock.patch("sparkrun.builders.eugr._run_build_capturing", return_value=(0, "")):
                    with mock.patch.object(builder, "_verify_image_imports"):
                        with mock.patch.object(builder, "_save_build_metadata") as mock_save:
                            builder.prepare_image("my-image", recipe, ["10.0.0.1"])

        # _save_build_metadata is called with the recipe's original build_args (no --cleanup).
        args, kwargs = mock_save.call_args
        # Signature: _save_build_metadata(image, build_args, config, host=..., ssh_kwargs=...)
        saved_build_args = args[1]
        assert saved_build_args == ["--tf5"]

    def test_eugr_prepare_smoke_failure_removes_tag_and_skips_cache(self, eugr_builder_with_repo):
        """If the post-build flashinfer smoke test fails, the tag is removed and cache is not updated."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "eugr-vllm",
                "container": "my-image",
                "runtime_config": {"build_args": ["--flag"]},
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                with mock.patch("sparkrun.builders.eugr._run_build_capturing", return_value=(0, "")):
                    with mock.patch.object(builder, "_remove_image") as mock_rmi:
                        # Make the smoke test report missing flashinfer_jit_cache.
                        smoke_proc = mock.Mock(returncode=1, stderr="ModuleNotFoundError: flashinfer_jit_cache\n")
                        with mock.patch("subprocess.run", return_value=smoke_proc):
                            with mock.patch.object(builder, "_save_build_metadata") as mock_save:
                                with pytest.raises(RuntimeError, match="without flashinfer"):
                                    builder.prepare_image("my-image", recipe, ["10.0.0.1"])

        mock_rmi.assert_called_once_with("my-image", host=None, ssh_kwargs=None)
        mock_save.assert_not_called()

    def test_eugr_prepare_respects_use_sentinel_image_false(self, eugr_builder_with_repo):
        """When `defaults.builders.eugr.use_sentinel_image=false`, `:latest` is NOT remapped.

        Power-user escape hatch: the user opts out of the historical force-rebuild
        behavior so `:latest` is treated as a regular pullable GHCR image.
        """
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "eugr-vllm",
                "container": "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest",
            }
        )
        config = mock.Mock()
        config.get_defaults_builder = mock.Mock(return_value={"use_sentinel_image": False})

        with mock.patch.object(builder, "ensure_repo") as mock_ensure:
            with mock.patch("sparkrun.builders.eugr._run_build_capturing") as mock_build:
                with mock.patch.object(builder, "_verify_image_imports") as mock_verify:
                    result = builder.prepare_image(
                        "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest",
                        recipe,
                        ["10.0.0.1"],
                        config=config,
                    )

        # No remap, no build, no smoke test: image flows through as pullable.
        assert result == "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest"
        mock_ensure.assert_not_called()
        mock_build.assert_not_called()
        mock_verify.assert_not_called()
        config.get_defaults_builder.assert_called_with("eugr")

    def test_eugr_prepare_sentinel_default_is_on(self, eugr_builder_with_repo, tmp_path):
        """With config returning empty builder defaults, sentinel behavior stays ON."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "eugr-vllm",
                "container": "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest",
            }
        )
        config = mock.Mock()
        config.cache_dir = tmp_path  # real path so Path(config.cache_dir) works
        config.get_defaults_builder = mock.Mock(return_value={})

        with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
            with mock.patch("sparkrun.builders.eugr._run_build_capturing", return_value=(0, "")) as mock_build:
                with mock.patch.object(builder, "_verify_image_imports"):
                    with mock.patch.object(builder, "_save_build_metadata"):
                        result = builder.prepare_image(
                            "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest",
                            recipe,
                            ["10.0.0.1"],
                            config=config,
                        )

        # Sentinel remap fired: image renamed to local tag, build happened.
        assert result == "sparkrun-eugr-vllm"
        mock_build.assert_called_once()


# ---------------------------------------------------------------------------
# Bootstrap integration — get_builder / list_builders
# ---------------------------------------------------------------------------


class TestBootstrapBuilders:
    """Test builder lookup and listing via sparkrun.core.bootstrap."""

    def test_get_builder_docker_pull(self, v):
        """get_builder('docker-pull') returns a DockerPullBuilder."""
        from sparkrun.builders.docker_pull import DockerPullBuilder
        from sparkrun.core.bootstrap import get_builder

        builder = get_builder("docker-pull", v=v)
        assert isinstance(builder, DockerPullBuilder)

    def test_get_builder_eugr(self, v):
        """get_builder('eugr') returns an EugrBuilder."""
        from sparkrun.builders.eugr import EugrBuilder
        from sparkrun.core.bootstrap import get_builder

        builder = get_builder("eugr", v=v)
        assert isinstance(builder, EugrBuilder)

    def test_get_builder_unknown_raises(self, v):
        """get_builder('nonexistent') raises ValueError."""
        from sparkrun.core.bootstrap import get_builder

        with pytest.raises(ValueError, match="nonexistent"):
            get_builder("nonexistent", v=v)

    def test_list_builders(self, v):
        """list_builders() includes both 'docker-pull' and 'eugr'."""
        from sparkrun.core.bootstrap import list_builders

        names = list_builders(v=v)
        assert "docker-pull" in names
        assert "eugr" in names


# ---------------------------------------------------------------------------
# Recipe builder field handling
# ---------------------------------------------------------------------------


class TestRecipeBuilderField:
    """Test that Recipe correctly parses builder / builder_config fields."""

    def test_recipe_builder_field(self):
        """Recipe.from_dict with builder field stores the value."""
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "vllm",
                "builder": "eugr",
            }
        )
        assert recipe.builder == "eugr"

    def test_recipe_builder_config_field(self):
        """Recipe.from_dict with builder_config stores the mapping."""
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "vllm",
                "builder_config": {"key": "value"},
            }
        )
        assert recipe.builder_config == {"key": "value"}

    def test_recipe_builder_not_in_runtime_config(self):
        """builder field is NOT swept into runtime_config."""
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "vllm",
                "builder": "eugr",
            }
        )
        assert "builder" not in recipe.runtime_config

    def test_recipe_builder_config_not_in_runtime_config(self):
        """builder_config field is NOT swept into runtime_config."""
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "vllm",
                "builder_config": {"opt": "val"},
            }
        )
        assert "builder_config" not in recipe.runtime_config

    def test_recipe_builder_in_export(self):
        """builder appears in the export dict when set."""
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "vllm",
                "builder": "eugr",
                "builder_config": {"base": "nvidia/cuda"},
            }
        )
        exported = recipe._build_export_dict()
        assert exported.get("builder") == "eugr"
        assert exported.get("builder_config") == {"base": "nvidia/cuda"}

    def test_recipe_builder_empty_not_in_export(self):
        """Empty builder does NOT appear in the export dict."""
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "vllm",
            }
        )
        exported = recipe._build_export_dict()
        assert "builder" not in exported
        assert "builder_config" not in exported


# ---------------------------------------------------------------------------
# EugrBuilder — delegated transfer mode
# ---------------------------------------------------------------------------


class TestEugrDelegatedMode:
    """Tests for EugrBuilder in delegated transfer mode."""

    def test_eugr_prepare_delegated_checks_image_on_head(self, eugr_builder_with_repo):
        """In delegated mode, image existence is checked on the head node."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "eugr-vllm",
            }
        )
        with mock.patch.object(builder, "_image_exists_on_host", return_value=True) as mock_check:
            with mock.patch.object(builder, "_ensure_repo_remote"):
                result = builder.prepare_image(
                    "vllm-node",
                    recipe,
                    ["head-host", "worker-host"],
                    transfer_mode="delegated",
                    ssh_kwargs={"ssh_user": "user"},
                )
        mock_check.assert_called_once_with("vllm-node", "head-host", {"ssh_user": "user"})
        # No mods or build_args, image exists => no-op
        assert result == "vllm-node"

    def test_eugr_prepare_delegated_builds_remotely(self, eugr_builder_with_repo):
        """In delegated mode, build is dispatched to head via _build_image_remote."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "eugr-vllm",
                "container": "my-image",
                "runtime_config": {"build_args": ["--flag"]},
            }
        )
        with mock.patch.object(builder, "_ensure_repo_remote", return_value="/remote/repo"):
            with mock.patch.object(builder, "_build_image_remote") as mock_remote_build:
                with mock.patch.object(builder, "_verify_image_imports"):
                    with mock.patch.object(builder, "_save_build_metadata"):
                        result = builder.prepare_image(
                            "my-image",
                            recipe,
                            ["head-host"],
                            transfer_mode="delegated",
                            ssh_kwargs={"ssh_user": "u"},
                        )
        mock_remote_build.assert_called_once_with(
            "my-image",
            [
                "--flag",
                "--cleanup",  # cleanup flag is injected (issue #164)
            ],
            "head-host",
            {"ssh_user": "u"},
            False,
            config=None,
        )
        assert result == "my-image"

    # Note: mod-specific delegated-mode behavior moved to core/mods.py;
    # covered by tests/test_mods.py.

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_image_exists_on_host_true(self, mock_run):
        """_image_exists_on_host returns True when docker inspect succeeds."""
        from sparkrun.builders.eugr import EugrBuilder
        from sparkrun.orchestration.ssh import RemoteResult

        mock_run.return_value = RemoteResult(host="h1", returncode=0, stdout="", stderr="")
        result = EugrBuilder._image_exists_on_host("my-image", "h1", ssh_kwargs={})
        assert result is True
        mock_run.assert_called_once()
        assert "docker image inspect" in mock_run.call_args[0][1]

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_image_exists_on_host_false(self, mock_run):
        """_image_exists_on_host returns False when docker inspect fails."""
        from sparkrun.builders.eugr import EugrBuilder
        from sparkrun.orchestration.ssh import RemoteResult

        mock_run.return_value = RemoteResult(host="h1", returncode=1, stdout="", stderr="not found")
        result = EugrBuilder._image_exists_on_host("my-image", "h1", ssh_kwargs={})
        assert result is False

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_ensure_repo_remote_calls_ssh(self, mock_run):
        """_ensure_repo_remote pipes a clone/pull script to the head node."""
        from sparkrun.builders.eugr import EugrBuilder
        from sparkrun.orchestration.ssh import RemoteResult

        mock_run.return_value = RemoteResult(host="head", returncode=0, stdout="/path\n", stderr="")
        builder = EugrBuilder()
        result = builder._ensure_repo_remote("head", ssh_kwargs={"ssh_user": "user"})
        assert result == "~/.cache/sparkrun/eugr-spark-vllm-docker"
        mock_run.assert_called_once()
        script = mock_run.call_args[0][1]
        assert "git clone" in script
        assert "git" in script

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_ensure_repo_remote_failure_raises(self, mock_run):
        """_ensure_repo_remote raises RuntimeError on SSH failure."""
        from sparkrun.builders.eugr import EugrBuilder
        from sparkrun.orchestration.ssh import RemoteResult

        mock_run.return_value = RemoteResult(host="head", returncode=1, stdout="", stderr="network error")
        builder = EugrBuilder()
        with pytest.raises(RuntimeError, match="Failed to ensure eugr repo"):
            builder._ensure_repo_remote("head", ssh_kwargs={})

    @mock.patch("sparkrun.orchestration.ssh.run_remote_script_streaming")
    def test_build_image_remote_calls_ssh(self, mock_run):
        """_build_image_remote pipes a build script to the head node."""
        from sparkrun.builders.eugr import EugrBuilder
        from sparkrun.orchestration.ssh import RemoteResult

        mock_run.return_value = RemoteResult(host="head", returncode=0, stdout="", stderr="")
        builder = EugrBuilder()
        builder._build_image_remote("my-image", ["--flag"], "head", ssh_kwargs={"ssh_user": "test1"})
        mock_run.assert_called_once()
        _args, kwargs = mock_run.call_args
        script = _args[1]
        assert "build-and-copy.sh" in script
        assert "-t my-image" in script
        assert "--flag" in script
        assert kwargs["ssh_user"] == "test1"

    @mock.patch("sparkrun.orchestration.ssh.run_remote_script_streaming")
    def test_build_image_remote_failure_raises(self, mock_run):
        """_build_image_remote raises RuntimeError on SSH failure."""
        from sparkrun.builders.eugr import EugrBuilder
        from sparkrun.orchestration.ssh import RemoteResult

        mock_run.return_value = RemoteResult(host="head", returncode=1, stdout="", stderr="build error")
        builder = EugrBuilder()
        with pytest.raises(RuntimeError, match="eugr remote container build failed"):
            builder._build_image_remote("my-image", [], "head", ssh_kwargs={})

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_build_image_remote_dry_run(self, mock_run):
        """_build_image_remote in dry_run mode does not call SSH."""
        from sparkrun.builders.eugr import EugrBuilder

        builder = EugrBuilder()
        builder._build_image_remote("my-image", ["--flag"], "head", ssh_kwargs={}, dry_run=True)
        mock_run.assert_not_called()

    def test_ensure_repo_remote_dry_run(self):
        """_ensure_repo_remote in dry_run returns path without SSH calls."""
        from sparkrun.builders.eugr import EugrBuilder

        builder = EugrBuilder()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host") as mock_run:
            result = builder._ensure_repo_remote("head", ssh_kwargs={}, dry_run=True)
        mock_run.assert_not_called()
        assert result == "~/.cache/sparkrun/eugr-spark-vllm-docker"

    def test_eugr_prepare_delegated_skips_build_when_image_exists_with_build_args(self, eugr_builder_with_repo):
        """When build_args are present but the image already exists on head, skip the build."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "eugr-vllm",
                "container": "my-image",
                "runtime_config": {"build_args": ["--tf5"]},
            }
        )
        with mock.patch.object(builder, "_image_exists_on_host", return_value=True) as mock_check:
            with mock.patch.object(builder, "_build_image_remote") as mock_build:
                result = builder.prepare_image(
                    "my-image",
                    recipe,
                    ["head-host"],
                    transfer_mode="delegated",
                    ssh_kwargs={"ssh_user": "user"},
                )
        mock_check.assert_called_once_with("my-image", "head-host", {"ssh_user": "user"})
        mock_build.assert_not_called()
        assert result == "my-image"

    def test_eugr_prepare_local_skips_build_when_image_exists_with_build_args(self, eugr_builder_with_repo):
        """When build_args are present but the image already exists locally, skip the build."""
        builder, repo_dir = eugr_builder_with_repo
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some/model",
                "runtime": "eugr-vllm",
                "container": "my-image",
                "runtime_config": {"build_args": ["--tf5"]},
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=True):
            with mock.patch("subprocess.run") as mock_run:
                result = builder.prepare_image("my-image", recipe, ["10.0.0.1"])
        mock_run.assert_not_called()
        assert result == "my-image"


# ---------------------------------------------------------------------------
# EugrBuilder — build logging and phase identification helpers (issue #164)
# ---------------------------------------------------------------------------


class TestEugrBuildLogging:
    """Tests for build-log capture and phase-aware error context."""

    def test_safe_image_name_replaces_unsafe_chars(self):
        from sparkrun.builders.eugr import _safe_image_name

        assert _safe_image_name("ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest") == ("ghcr.io-spark-arena-dgx-vllm-eugr-nightly-latest")
        # Already-safe input is preserved (modulo trailing strip).
        assert _safe_image_name("sparkrun-eugr-vllm") == "sparkrun-eugr-vllm"
        assert _safe_image_name("///") == "image"

    def test_identify_phase_flashinfer_build_failure(self):
        from sparkrun.builders.eugr import _identify_build_phase

        output = (
            "Cleaning up wheels directory...\n"
            "FlashInfer build command: docker build --target flashinfer-export ...\n"
            "FlashInfer build failed — restoring previous wheels...\n"
        )
        assert _identify_build_phase(output) == "flashinfer-build (failed)"

    def test_identify_phase_vllm_build_failure(self):
        from sparkrun.builders.eugr import _identify_build_phase

        output = (
            "FlashInfer wheels ready.\n"
            "vLLM build command: docker build --target vllm-export ...\n"
            "vLLM build failed — restoring previous wheels...\n"
        )
        assert _identify_build_phase(output) == "vllm-build (failed)"

    def test_identify_phase_no_assets(self):
        from sparkrun.builders.eugr import _identify_build_phase

        output = "Could not fetch release metadata for 'prebuilt-flashinfer-current' — skipping download.\n"
        assert _identify_build_phase(output) == "wheel-download (release metadata)"

    def test_identify_phase_runner_no_wheels(self):
        from sparkrun.builders.eugr import _identify_build_phase

        output = "FlashInfer wheels ready.\nvLLM wheels ready.\nError: No wheel files found in ./wheels/ — cannot build runner image.\n"
        assert _identify_build_phase(output) == "runner-build (no wheels)"

    def test_identify_phase_latest_marker_wins_when_no_explicit_failure(self):
        from sparkrun.builders.eugr import _identify_build_phase

        # No failure marker -> falls back to the last phase reached.
        output = "Cleaning up wheels directory...\nFlashInfer wheels ready.\nvLLM build command: docker build ...\n"
        assert _identify_build_phase(output) == "vllm-build"

    def test_build_error_tail_returns_last_n_nonempty_lines(self):
        from sparkrun.builders.eugr import _build_error_tail

        output = "first\n\nsecond\nthird\n\nfourth\n"
        assert _build_error_tail(output, n=2) == "third\nfourth"
        assert _build_error_tail(output, n=10) == "first\nsecond\nthird\nfourth"

    def test_run_build_capturing_writes_log_file(self, tmp_path):
        """_run_build_capturing captures combined output to memory and to disk."""
        from sparkrun.builders.eugr import _run_build_capturing

        log = tmp_path / "build.log"
        rc, out = _run_build_capturing(
            ["bash", "-c", "echo hello; echo error >&2; exit 0"],
            log_path=log,
            stream=False,
        )
        assert rc == 0
        assert "hello" in out
        assert "error" in out  # stderr merged into stdout
        # log file contains the same content
        assert "hello" in log.read_text()
        assert "error" in log.read_text()

    def test_run_build_capturing_handles_nonzero_exit(self, tmp_path):
        from sparkrun.builders.eugr import _run_build_capturing

        rc, out = _run_build_capturing(
            ["bash", "-c", "echo phase-marker; exit 5"],
            log_path=tmp_path / "x.log",
            stream=False,
        )
        assert rc == 5
        assert "phase-marker" in out

    def test_run_build_capturing_tolerates_unwritable_log(self, tmp_path):
        """If the log file can't be opened, the build still runs and returns output."""
        from sparkrun.builders.eugr import _run_build_capturing

        # Point at a directory that doesn't exist and can't be created (read-only parent).
        # Use a file *as* a parent so the mkdir fails.
        not_a_dir = tmp_path / "not-a-dir"
        not_a_dir.write_text("blocker")
        bad_log = not_a_dir / "child" / "build.log"

        rc, out = _run_build_capturing(
            ["bash", "-c", "echo ok"],
            log_path=bad_log,
            stream=False,
        )
        assert rc == 0
        assert "ok" in out

    def test_build_log_path_uses_cache_dir(self, tmp_path):
        from sparkrun.builders.eugr import EugrBuilder

        builder = EugrBuilder()
        with mock.patch.object(builder, "_resolve_cache_dir", return_value=tmp_path):
            p = builder._build_log_path("ghcr.io/foo:bar")
        assert p is not None
        assert p.parent == tmp_path / "eugr-builds"
        assert p.suffix == ".log"
        assert "ghcr.io-foo-bar" in p.name

    def test_build_log_path_returns_none_when_no_cache(self):
        from sparkrun.builders.eugr import EugrBuilder

        builder = EugrBuilder()
        with mock.patch.object(builder, "_resolve_cache_dir", return_value=None):
            assert builder._build_log_path("img") is None

    def test_build_log_path_includes_host_for_remote(self, tmp_path):
        from sparkrun.builders.eugr import EugrBuilder

        builder = EugrBuilder()
        with mock.patch.object(builder, "_resolve_cache_dir", return_value=tmp_path):
            p = builder._build_log_path("img", host="head.example.com")
        assert p is not None
        assert "remote-head.example.com" in p.name


# ---------------------------------------------------------------------------
# EugrBuilder — flashinfer smoke test (_verify_image_imports)
# ---------------------------------------------------------------------------


class TestEugrVerifyImageImports:
    """Tests for the post-build flashinfer importability check."""

    def test_verify_image_imports_local_success(self):
        from sparkrun.builders.eugr import EugrBuilder

        builder = EugrBuilder()
        with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0, stderr="")) as mock_run:
            builder._verify_image_imports("my-image")
        # Verify the command shape: docker run --rm --entrypoint= <image> python3 -c "<imports>"
        cmd = mock_run.call_args[0][0]
        assert cmd[:4] == ["docker", "run", "--rm", "--entrypoint="]
        assert cmd[4] == "my-image"
        assert cmd[5:7] == ["python3", "-c"]
        check = cmd[7]
        assert "import flashinfer" in check
        assert "import flashinfer_cubin" in check
        assert "import flashinfer_jit_cache" in check

    def test_verify_image_imports_local_failure_raises_and_removes_image(self):
        from sparkrun.builders.eugr import EugrBuilder

        builder = EugrBuilder()
        bad = mock.Mock(returncode=1, stderr="ModuleNotFoundError: flashinfer_jit_cache")
        with mock.patch("subprocess.run", return_value=bad):
            with mock.patch.object(builder, "_remove_image") as mock_rmi:
                with pytest.raises(RuntimeError, match="without flashinfer"):
                    builder._verify_image_imports("my-image")
        mock_rmi.assert_called_once_with("my-image", host=None, ssh_kwargs=None)

    def test_verify_image_imports_local_timeout_treated_as_failure(self):
        import subprocess as _sp

        from sparkrun.builders.eugr import EugrBuilder

        builder = EugrBuilder()
        with mock.patch("subprocess.run", side_effect=_sp.TimeoutExpired(cmd="docker", timeout=60)):
            with mock.patch.object(builder, "_remove_image"):
                with pytest.raises(RuntimeError, match="without flashinfer"):
                    builder._verify_image_imports("my-image")

    def test_verify_image_imports_remote_success(self):
        from sparkrun.builders.eugr import EugrBuilder
        from sparkrun.orchestration.ssh import RemoteResult

        builder = EugrBuilder()
        ok = RemoteResult(host="head", returncode=0, stdout="", stderr="")
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=ok) as mock_ssh:
            builder._verify_image_imports("my-image", host="head", ssh_kwargs={"ssh_user": "u"})
        script = mock_ssh.call_args[0][1]
        assert "docker run --rm --entrypoint=" in script
        assert "python3 -c" in script
        assert "import flashinfer" in script

    def test_verify_image_imports_remote_failure_removes_image(self):
        from sparkrun.builders.eugr import EugrBuilder
        from sparkrun.orchestration.ssh import RemoteResult

        builder = EugrBuilder()
        bad = RemoteResult(host="head", returncode=1, stdout="", stderr="ModuleNotFoundError: flashinfer")
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=bad):
            with mock.patch.object(builder, "_remove_image") as mock_rmi:
                with pytest.raises(RuntimeError, match="without flashinfer"):
                    builder._verify_image_imports("my-image", host="head", ssh_kwargs={"ssh_user": "u"})
        mock_rmi.assert_called_once_with("my-image", host="head", ssh_kwargs={"ssh_user": "u"})
