"""Tests for runtime version detection and metadata persistence."""

from unittest import mock

from sparkrun.builders.base import _flatten_dict
from sparkrun.builders.eugr import EugrBuilder
from sparkrun.core.recipe import Recipe
from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata
from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.runtimes.base import RuntimePlugin
from sparkrun.runtimes.vllm_ray import VllmRayRuntime
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
from sparkrun.runtimes.sglang import SglangRuntime
from sparkrun.runtimes.llama_cpp import LlamaCppRuntime
from sparkrun.runtimes.trtllm import TrtllmRuntime


# --- version_commands() tests ---


class TestVersionCommands:
    """Test version_commands() returns expected keys for base and subclasses."""

    def test_base_has_common_keys(self):
        cmds = RuntimePlugin.version_commands(RuntimePlugin())
        assert "cuda" in cmds
        assert "python" in cmds
        assert "torch" in cmds
        assert "nccl" in cmds

    def test_vllm_ray_has_vllm_key(self):
        cmds = VllmRayRuntime().version_commands()
        assert "vllm" in cmds
        assert "cuda" in cmds

    def test_vllm_distributed_has_vllm_key(self):
        cmds = VllmDistributedRuntime().version_commands()
        assert "vllm" in cmds
        assert "cuda" in cmds

    def test_sglang_has_sglang_key(self):
        cmds = SglangRuntime().version_commands()
        assert "sglang" in cmds
        assert "cuda" in cmds

    def test_llama_cpp_has_llama_cpp_key(self):
        cmds = LlamaCppRuntime().version_commands()
        assert "llama_cpp" in cmds
        assert "cuda" in cmds

    def test_trtllm_has_trtllm_key(self):
        cmds = TrtllmRuntime().version_commands()
        assert "trtllm" in cmds
        assert "cuda" in cmds


# --- _collect_runtime_info() tests ---


class TestCollectRuntimeInfo:
    """Test _collect_runtime_info parses output and handles failures."""

    def test_parses_key_value_output(self):
        stdout = "SPARKRUN_VER_CUDA=12.9\nSPARKRUN_VER_PYTHON=3.12.3\nSPARKRUN_VER_TORCH=2.7.0\nSPARKRUN_VER_VLLM=0.8.5\n"
        fake_result = RemoteResult(
            host="host1",
            stdout=stdout,
            stderr="",
            returncode=0,
        )

        runtime = VllmRayRuntime()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=fake_result):
            info = runtime._collect_runtime_info("host1", "container1", {}, dry_run=False)

        assert info == {
            "cuda": "12.9",
            "python": "3.12.3",
            "torch": "2.7.0",
            "vllm": "0.8.5",
        }

    def test_skips_unknown_values(self):
        stdout = "SPARKRUN_VER_CUDA=12.9\nSPARKRUN_VER_TORCH=unknown\n"
        fake_result = RemoteResult(
            host="host1",
            stdout=stdout,
            stderr="",
            returncode=0,
        )

        runtime = VllmRayRuntime()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=fake_result):
            info = runtime._collect_runtime_info("host1", "container1", {}, dry_run=False)

        assert info == {"cuda": "12.9"}
        assert "torch" not in info

    def test_returns_empty_on_failure(self):
        fake_result = RemoteResult(
            host="host1",
            stdout="",
            stderr="error",
            returncode=1,
        )

        runtime = VllmRayRuntime()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=fake_result):
            info = runtime._collect_runtime_info("host1", "container1", {}, dry_run=False)

        assert info == {}

    def test_returns_empty_on_exception(self):
        runtime = VllmRayRuntime()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", side_effect=RuntimeError("boom")):
            info = runtime._collect_runtime_info("host1", "container1", {}, dry_run=False)

        assert info == {}

    def test_returns_empty_on_dry_run(self):
        runtime = VllmRayRuntime()
        info = runtime._collect_runtime_info("host1", "container1", {}, dry_run=True)
        assert info == {}

    def test_handles_empty_lines_and_noise(self):
        stdout = "some noise before\nSPARKRUN_VER_CUDA=12.9\n\nSPARKRUN_VER_PYTHON=\nmore noise\nSPARKRUN_VER_SGLANG=0.4.6.post1\n"
        fake_result = RemoteResult(
            host="host1",
            stdout=stdout,
            stderr="",
            returncode=0,
        )

        runtime = SglangRuntime()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=fake_result):
            info = runtime._collect_runtime_info("host1", "container1", {}, dry_run=False)

        assert info == {"cuda": "12.9", "sglang": "0.4.6.post1"}
        assert "python" not in info  # empty value skipped


# --- job_metadata runtime_info persistence tests ---


class TestJobMetadataRuntimeInfo:
    """Test save/load of runtime_info in job metadata."""

    def _make_recipe(self):
        return Recipe.from_dict(
            {
                "name": "test",
                "runtime": "vllm",
                "model": "test/model",
            }
        )

    def test_save_and_load_with_runtime_info(self, tmp_path):
        recipe = self._make_recipe()
        runtime_info = {"vllm": "0.8.5", "cuda": "12.9", "torch": "2.7.0"}

        save_job_metadata(
            "sparkrun_abc123",
            recipe,
            ["host1"],
            cache_dir=str(tmp_path),
            runtime_info=runtime_info,
        )

        meta = load_job_metadata("sparkrun_abc123", cache_dir=str(tmp_path))
        assert meta is not None
        assert meta["runtime_info"] == runtime_info

    def test_save_without_runtime_info(self, tmp_path):
        recipe = self._make_recipe()

        save_job_metadata(
            "sparkrun_def456",
            recipe,
            ["host1"],
            cache_dir=str(tmp_path),
        )

        meta = load_job_metadata("sparkrun_def456", cache_dir=str(tmp_path))
        assert meta is not None
        assert "runtime_info" not in meta

    def test_save_with_empty_runtime_info(self, tmp_path):
        recipe = self._make_recipe()

        save_job_metadata(
            "sparkrun_ghi789",
            recipe,
            ["host1"],
            cache_dir=str(tmp_path),
            runtime_info={},
        )

        meta = load_job_metadata("sparkrun_ghi789", cache_dir=str(tmp_path))
        assert meta is not None
        assert "runtime_info" not in meta  # empty dict not persisted


# --- _flatten_dict tests ---


class TestFlattenDict:
    """Test the _flatten_dict helper used by builders."""

    def test_flat_dict_with_prefix(self):
        result = _flatten_dict({"version": "1.0", "name": "foo"}, prefix="build")
        assert result == {"build_version": "1.0", "build_name": "foo"}

    def test_nested_dict(self):
        result = _flatten_dict(
            {"version": "1.0", "git": {"commit": "abc", "branch": "main"}},
            prefix="build",
        )
        assert result == {
            "build_version": "1.0",
            "build_git_commit": "abc",
            "build_git_branch": "main",
        }

    def test_deeply_nested(self):
        result = _flatten_dict({"a": {"b": {"c": "deep"}}}, prefix="x")
        assert result == {"x_a_b_c": "deep"}

    def test_empty_dict(self):
        assert _flatten_dict({}) == {}
        assert _flatten_dict({}, prefix="build") == {}

    def test_non_string_values_stringified(self):
        result = _flatten_dict({"count": 42, "flag": True}, prefix="build")
        assert result == {"build_count": "42", "build_flag": "True"}

    def test_no_prefix(self):
        result = _flatten_dict({"a": "1", "b": "2"})
        assert result == {"a": "1", "b": "2"}

    def test_normalize_keys(self):
        """Dots, slashes, and dashes in keys are replaced with separator."""
        result = _flatten_dict(
            {"org.opencontainers.image.version": "1.0", "com/example/name": "test", "my-key": "val"},
            prefix="container",
            normalize=True,
        )
        assert result == {
            "container_org_opencontainers_image_version": "1.0",
            "container_com_example_name": "test",
            "container_my_key": "val",
        }

    def test_normalize_nested(self):
        result = _flatten_dict(
            {"org.example": {"sub.key": "v"}},
            prefix="container",
            normalize=True,
        )
        assert result == {"container_org_example_sub_key": "v"}

    def test_normalize_false_preserves_dots(self):
        result = _flatten_dict(
            {"org.example": "v"},
            prefix="build",
            normalize=False,
        )
        assert result == {"build_org.example": "v"}


# --- EugrBuilder version info tests ---


class TestEugrBuilderVersionInfo:
    """Test EugrBuilder.version_info_commands and process_version_info."""

    def test_version_info_commands_returns_expected(self):
        builder = EugrBuilder()
        cmds = builder.version_info_commands()
        assert "build_metadata" in cmds
        assert "cat" in cmds["build_metadata"]

    def test_process_version_info_valid_yaml(self):
        builder = EugrBuilder()
        raw = {
            "build_metadata": "version: '1.0'\ngit:\n  commit: abc123\n  branch: main\n",
        }
        result = builder.process_version_info(raw)
        assert result == {
            "build_version": "1.0",
            "build_git_commit": "abc123",
            "build_git_branch": "main",
        }

    def test_process_version_info_empty_content(self):
        builder = EugrBuilder()
        assert builder.process_version_info({"build_metadata": ""}) == {}
        assert builder.process_version_info({"build_metadata": "   "}) == {}
        assert builder.process_version_info({}) == {}

    def test_process_version_info_invalid_yaml(self):
        builder = EugrBuilder()
        result = builder.process_version_info({"build_metadata": "{{invalid yaml"})
        assert result == {}

    def test_process_version_info_non_dict_yaml(self):
        builder = EugrBuilder()
        result = builder.process_version_info({"build_metadata": "just a string"})
        assert result == {}


# --- collect_container_labels tests ---


class TestCollectContainerLabels:
    """Test BuilderPlugin.collect_container_labels default docker inspect implementation."""

    def test_parses_json_labels(self):
        """Docker inspect JSON labels are parsed and prefixed with container_."""
        fake_result = RemoteResult(
            host="host1",
            stdout='{"org.opencontainers.image.version":"1.0","maintainer":"test@example.com"}\n',
            stderr="",
            returncode=0,
        )
        from sparkrun.builders.base import BuilderPlugin

        builder = BuilderPlugin()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=fake_result):
            labels = builder.collect_container_labels("container1", "host1", {})

        assert labels == {
            "container_org_opencontainers_image_version": "1.0",
            "container_maintainer": "test@example.com",
        }

    def test_empty_labels_returns_empty(self):
        fake_result = RemoteResult(
            host="host1",
            stdout="{}\n",
            stderr="",
            returncode=0,
        )
        from sparkrun.builders.base import BuilderPlugin

        builder = BuilderPlugin()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=fake_result):
            labels = builder.collect_container_labels("container1", "host1", {})

        assert labels == {}

    def test_inspect_failure_returns_empty(self):
        fake_result = RemoteResult(
            host="host1",
            stdout="",
            stderr="Error",
            returncode=1,
        )
        from sparkrun.builders.base import BuilderPlugin

        builder = BuilderPlugin()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=fake_result):
            labels = builder.collect_container_labels("container1", "host1", {})

        assert labels == {}

    def test_invalid_json_returns_empty(self):
        fake_result = RemoteResult(
            host="host1",
            stdout="not json at all\n",
            stderr="",
            returncode=0,
        )
        from sparkrun.builders.base import BuilderPlugin

        builder = BuilderPlugin()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=fake_result):
            labels = builder.collect_container_labels("container1", "host1", {})

        assert labels == {}

    def test_exception_returns_empty(self):
        from sparkrun.builders.base import BuilderPlugin

        builder = BuilderPlugin()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", side_effect=OSError("ssh failed")):
            labels = builder.collect_container_labels("container1", "host1", {})

        assert labels == {}

    def test_null_json_returns_empty(self):
        """docker inspect returns 'null' when no labels are set."""
        fake_result = RemoteResult(
            host="host1",
            stdout="null\n",
            stderr="",
            returncode=0,
        )
        from sparkrun.builders.base import BuilderPlugin

        builder = BuilderPlugin()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=fake_result):
            labels = builder.collect_container_labels("container1", "host1", {})

        assert labels == {}

    def test_labels_with_complex_oci_keys(self):
        """OCI label keys with dots/slashes/dashes are normalized."""
        fake_result = RemoteResult(
            host="host1",
            stdout='{"org.opencontainers.image.source":"https://github.com/example","vcs-ref":"abc123"}\n',
            stderr="",
            returncode=0,
        )
        from sparkrun.builders.base import BuilderPlugin

        builder = BuilderPlugin()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=fake_result):
            labels = builder.collect_container_labels("container1", "host1", {})

        assert "container_org_opencontainers_image_source" in labels
        assert "container_vcs_ref" in labels


# --- _collect_runtime_info with builder tests ---


class TestCollectRuntimeInfoWithBuilder:
    """Test _collect_runtime_info merges builder version data."""

    def test_builder_commands_included_in_script(self):
        """Builder commands produce delimited blocks in the output."""
        yaml_content = "version: '2.0'\ngit:\n  commit: def456\n"
        stdout = (
            "SPARKRUN_VER_CUDA=12.9\n"
            "SPARKRUN_VER_VLLM=0.8.5\n"
            "SPARKRUN_BUILDER_START_build_metadata\n"
            "%s\n"
            "SPARKRUN_BUILDER_END_build_metadata\n"
        ) % yaml_content
        fake_result = RemoteResult(
            host="host1",
            stdout=stdout,
            stderr="",
            returncode=0,
        )

        runtime = VllmRayRuntime()
        builder = EugrBuilder()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=fake_result):
            info = runtime._collect_runtime_info(
                "host1",
                "container1",
                {},
                dry_run=False,
                builder=builder,
            )

        assert info["cuda"] == "12.9"
        assert info["vllm"] == "0.8.5"
        assert info["build_version"] == "2.0"
        assert info["build_git_commit"] == "def456"

    def test_builder_keys_dont_overwrite_runtime_keys(self):
        """If builder and runtime produce the same key, runtime wins."""
        # Contrived: builder YAML has a "cuda" key at top level
        stdout = "SPARKRUN_VER_CUDA=12.9\nSPARKRUN_BUILDER_START_build_metadata\ncuda: '11.0'\nSPARKRUN_BUILDER_END_build_metadata\n"
        fake_result = RemoteResult(
            host="host1",
            stdout=stdout,
            stderr="",
            returncode=0,
        )

        runtime = VllmRayRuntime()
        builder = EugrBuilder()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=fake_result):
            info = runtime._collect_runtime_info(
                "host1",
                "container1",
                {},
                dry_run=False,
                builder=builder,
            )

        # Runtime's cuda wins; builder's gets prefixed as build_cuda
        assert info["cuda"] == "12.9"
        assert info["build_cuda"] == "11.0"

    def test_builder_fails_silently_on_empty_file(self):
        """Empty builder block produces no builder keys."""
        stdout = "SPARKRUN_VER_CUDA=12.9\nSPARKRUN_BUILDER_START_build_metadata\n\nSPARKRUN_BUILDER_END_build_metadata\n"
        fake_result = RemoteResult(
            host="host1",
            stdout=stdout,
            stderr="",
            returncode=0,
        )

        runtime = VllmRayRuntime()
        builder = EugrBuilder()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=fake_result):
            info = runtime._collect_runtime_info(
                "host1",
                "container1",
                {},
                dry_run=False,
                builder=builder,
            )

        assert info == {"cuda": "12.9"}

    def test_builder_fails_silently_on_parse_error(self):
        """Invalid YAML in builder block produces no builder keys."""
        stdout = "SPARKRUN_VER_CUDA=12.9\nSPARKRUN_BUILDER_START_build_metadata\n{{not valid yaml\nSPARKRUN_BUILDER_END_build_metadata\n"
        fake_result = RemoteResult(
            host="host1",
            stdout=stdout,
            stderr="",
            returncode=0,
        )

        runtime = VllmRayRuntime()
        builder = EugrBuilder()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=fake_result):
            info = runtime._collect_runtime_info(
                "host1",
                "container1",
                {},
                dry_run=False,
                builder=builder,
            )

        assert info == {"cuda": "12.9"}

    def test_no_builder_backward_compatible(self):
        """Without a builder, behavior is unchanged."""
        stdout = "SPARKRUN_VER_CUDA=12.9\nSPARKRUN_VER_VLLM=0.8.5\n"
        fake_result = RemoteResult(
            host="host1",
            stdout=stdout,
            stderr="",
            returncode=0,
        )

        runtime = VllmRayRuntime()
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", return_value=fake_result):
            info = runtime._collect_runtime_info(
                "host1",
                "container1",
                {},
                dry_run=False,
            )

        assert info == {"cuda": "12.9", "vllm": "0.8.5"}


# --- Full launcher flow: labels + build-metadata + runtime versions ---


class TestLauncherLabelIntegration:
    """Test that launcher merges labels, build-metadata, and runtime versions correctly."""

    def test_all_three_sources_merge(self):
        """Runtime versions + builder build-metadata + container labels all merge into runtime_info."""
        # Simulate runtime version collection output with builder block
        yaml_content = "version: '2.0'\ngit:\n  commit: def456\n"
        version_stdout = (
            "SPARKRUN_VER_CUDA=12.9\n"
            "SPARKRUN_VER_VLLM=0.8.5\n"
            "SPARKRUN_BUILDER_START_build_metadata\n"
            "%s\n"
            "SPARKRUN_BUILDER_END_build_metadata\n"
        ) % yaml_content
        version_result = RemoteResult(
            host="host1",
            stdout=version_stdout,
            stderr="",
            returncode=0,
        )

        # Simulate docker inspect label output
        label_result = RemoteResult(
            host="host1",
            stdout='{"org.opencontainers.image.version":"3.0","maintainer":"dev@example.com"}\n',
            stderr="",
            returncode=0,
        )

        runtime = VllmRayRuntime()
        builder = EugrBuilder()

        call_count = {"n": 0}
        original_results = [version_result, label_result]

        def mock_run_script(host, script, **kwargs):
            idx = call_count["n"]
            call_count["n"] += 1
            return original_results[idx]

        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", side_effect=mock_run_script):
            # Step 1: runtime version collection (includes builder blocks)
            runtime_info = runtime._collect_runtime_info(
                "host1",
                "container1",
                {},
                dry_run=False,
                builder=builder,
            )
            # Step 2: container label collection
            label_info = builder.collect_container_labels("container1", "host1", {})
            for k, v in label_info.items():
                if k not in runtime_info:
                    runtime_info[k] = v

        # Runtime versions (no prefix)
        assert runtime_info["cuda"] == "12.9"
        assert runtime_info["vllm"] == "0.8.5"
        # Build metadata (build_ prefix)
        assert runtime_info["build_version"] == "2.0"
        assert runtime_info["build_git_commit"] == "def456"
        # Container labels (container_ prefix, normalized keys)
        assert runtime_info["container_org_opencontainers_image_version"] == "3.0"
        assert runtime_info["container_maintainer"] == "dev@example.com"
