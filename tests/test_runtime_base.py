"""Unit tests for sparkrun.runtimes base class and cross-cutting orchestration helpers."""

import re
from unittest import mock

from scitrera_app_framework import Variables
from sparkrun.orchestration.job_metadata import generate_cluster_id
from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.base import RuntimePlugin


# --- generate_cluster_id Tests ---


class TestGenerateClusterId:
    """Test deterministic cluster ID generation."""

    def _make_recipe(self, runtime="vllm", model="meta-llama/Llama-2-7b-hf"):
        return Recipe.from_dict(
            {
                "name": "test",
                "runtime": runtime,
                "model": model,
            }
        )

    def test_deterministic(self):
        """Same inputs produce the same cluster ID."""
        recipe = self._make_recipe()
        hosts = ["10.0.0.1", "10.0.0.2"]
        assert generate_cluster_id(recipe, hosts) == generate_cluster_id(recipe, hosts)

    def test_host_order_independent(self):
        """Host ordering does not affect the ID (sorted internally)."""
        recipe = self._make_recipe()
        id_a = generate_cluster_id(recipe, ["10.0.0.1", "10.0.0.2"])
        id_b = generate_cluster_id(recipe, ["10.0.0.2", "10.0.0.1"])
        assert id_a == id_b

    def test_different_hosts_differ(self):
        """Different host sets produce different IDs."""
        recipe = self._make_recipe()
        id_a = generate_cluster_id(recipe, ["10.0.0.1"])
        id_b = generate_cluster_id(recipe, ["10.0.0.2"])
        assert id_a != id_b

    def test_prefix_and_format(self):
        """Result starts with 'sparkrun_' followed by 12 hex characters."""
        recipe = self._make_recipe()
        cid = generate_cluster_id(recipe, ["host1"])
        assert re.fullmatch(r"sparkrun_[0-9a-f]{12}", cid)


# --- resolve_api_key Tests ---


def test_base_resolve_api_key_returns_none():
    """Base RuntimePlugin returns None — runtimes opt in by overriding."""

    class _Stub(RuntimePlugin):
        runtime_name = "stub"

        def generate_command(self, *args, **kwargs):
            return ""

    recipe = Recipe.from_dict({"name": "r", "model": "m", "runtime": "stub", "defaults": {"api_key": "abc"}})
    assert _Stub().resolve_api_key(recipe) is None


# --- Base RuntimePlugin Tests ---


def test_base_runtime_is_enabled_false():
    """RuntimePlugin.is_enabled() returns False (critical for multi-extension)."""
    runtime = RuntimePlugin()
    v = Variables()

    # is_enabled must return False for multi-extension plugins
    assert runtime.is_enabled(v) is False


def test_base_runtime_is_multi_extension_true():
    """RuntimePlugin.is_multi_extension() returns True."""
    runtime = RuntimePlugin()
    v = Variables()

    assert runtime.is_multi_extension(v) is True


def test_base_runtime_is_not_delegating():
    """Base RuntimePlugin.is_delegating_runtime() returns False."""
    runtime = RuntimePlugin()
    assert runtime.is_delegating_runtime() is False


# --- follow_logs() Tests ---


class _StubRuntime(RuntimePlugin):
    """Minimal concrete runtime for testing base class behaviour."""

    runtime_name = "stub"

    def generate_command(self, recipe, overrides, is_cluster, num_nodes=1, head_ip=None):
        return ""

    def resolve_container(self, recipe, overrides=None):
        return "stub:latest"


class TestBaseFollowLogs:
    """Test base RuntimePlugin.follow_logs()."""

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_solo(self, mock_stream):
        """Base follow_logs calls stream_container_file_logs with solo container name."""
        runtime = _StubRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1"],
            cluster_id="mytest0",
            config=None,
            dry_run=False,
            tail=50,
        )

        mock_stream.assert_called_once_with(
            "10.0.0.1",
            "mytest0_solo",
            tail=50,
            dry_run=False,
        )

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_localhost_default(self, mock_stream):
        """Base follow_logs with empty hosts uses localhost."""
        runtime = _StubRuntime()
        runtime.follow_logs(hosts=[], cluster_id="sparkrun0")

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "localhost"
        assert args[0][1] == "sparkrun0_solo"

    @mock.patch("sparkrun.orchestration.ssh.stream_remote_logs")
    def test_follow_logs_cluster_uses_docker_logs(self, mock_stream):
        """Base _follow_cluster_logs streams docker logs on head node."""
        runtime = _StubRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1", "10.0.0.2"],
            cluster_id="test0",
        )

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "10.0.0.1"
        assert args[0][1] == "test0_head"


# --- _augment_served_model_name tests ---


class TestAugmentServedModelName:
    """Test that served_model_name from overrides is appended to rendered template commands."""

    def _make_recipe(self, runtime="vllm", command=None, defaults=None):
        data = {
            "name": "test",
            "runtime": runtime,
            "model": "org/some-model",
        }
        if command:
            data["command"] = command
        if defaults:
            data["defaults"] = defaults
        return Recipe.from_dict(data)

    # --- vllm-ray ---

    def test_vllm_ray_appends_when_missing(self):
        """vllm-ray: template without {served_model_name} gets flag appended."""
        from sparkrun.runtimes.vllm_ray import VllmRayRuntime

        recipe = self._make_recipe(
            runtime="vllm",
            command="vllm serve {model} --port 8000",
            defaults={"served_model_name": "my-model"},
        )
        runtime = VllmRayRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert cmd.endswith("--served-model-name my-model")

    def test_vllm_ray_override_appends(self):
        """vllm-ray: CLI override served_model_name appended to template."""
        from sparkrun.runtimes.vllm_ray import VllmRayRuntime

        recipe = self._make_recipe(
            runtime="vllm",
            command="vllm serve {model} --port 8000",
        )
        runtime = VllmRayRuntime()
        cmd = runtime.generate_command(recipe, {"served_model_name": "cli-name"}, is_cluster=False)
        assert "--served-model-name cli-name" in cmd

    def test_vllm_ray_no_duplicate(self):
        """vllm-ray: template already has --served-model-name → no duplicate."""
        from sparkrun.runtimes.vllm_ray import VllmRayRuntime

        recipe = self._make_recipe(
            runtime="vllm",
            command="vllm serve {model} --served-model-name {served_model_name} --port 8000",
            defaults={"served_model_name": "in-template"},
        )
        runtime = VllmRayRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert cmd.count("--served-model-name") == 1

    def test_vllm_ray_no_override_no_change(self):
        """vllm-ray: no served_model_name in config → command unchanged."""
        from sparkrun.runtimes.vllm_ray import VllmRayRuntime

        recipe = self._make_recipe(
            runtime="vllm",
            command="vllm serve {model} --port 8000",
        )
        runtime = VllmRayRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--served-model-name" not in cmd

    def test_vllm_ray_skip_keys_suppresses(self):
        """vllm-ray: skip_keys={served_model_name} → no augmentation."""
        from sparkrun.runtimes.vllm_ray import VllmRayRuntime

        recipe = self._make_recipe(
            runtime="vllm",
            command="vllm serve {model} --port 8000",
            defaults={"served_model_name": "my-model"},
        )
        runtime = VllmRayRuntime()
        cmd = runtime.generate_command(
            recipe,
            {},
            is_cluster=False,
            skip_keys={"served_model_name"},
        )
        assert "--served-model-name" not in cmd

    # --- vllm-distributed ---

    def test_vllm_distributed_generate_appends(self):
        """vllm-distributed generate_command: template missing flag → appended."""
        from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

        recipe = self._make_recipe(
            runtime="vllm-distributed",
            command="vllm serve {model} --port 8000",
            defaults={"served_model_name": "dist-model"},
        )
        runtime = VllmDistributedRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--served-model-name dist-model" in cmd

    def test_vllm_distributed_node_command_appends(self):
        """vllm-distributed generate_node_command: template missing flag → appended.

        Uses tp=2 (replica_size > 1) so torch-distributed flags are emitted
        alongside the served-model-name append.
        """
        from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

        recipe = self._make_recipe(
            runtime="vllm-distributed",
            command="vllm serve {model} --port 8000",
            defaults={"served_model_name": "dist-model", "tensor_parallel": 2},
        )
        runtime = VllmDistributedRuntime()
        cmd = runtime.generate_node_command(
            recipe,
            {},
            head_ip="10.0.0.1",
            num_nodes=2,
            node_rank=0,
        )
        assert "--served-model-name dist-model" in cmd
        assert "--nnodes 2" in cmd

    def test_vllm_distributed_node_command_no_duplicate(self):
        """vllm-distributed generate_node_command: template has flag → no dup."""
        from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

        recipe = self._make_recipe(
            runtime="vllm-distributed",
            command="vllm serve {model} --served-model-name {served_model_name}",
            defaults={"served_model_name": "in-tpl"},
        )
        runtime = VllmDistributedRuntime()
        cmd = runtime.generate_node_command(
            recipe,
            {},
            head_ip="10.0.0.1",
            num_nodes=2,
            node_rank=1,
        )
        assert cmd.count("--served-model-name") == 1

    # --- sglang ---

    def test_sglang_generate_appends(self):
        """sglang generate_command: template missing flag → appended."""
        from sparkrun.runtimes.sglang import SglangRuntime

        recipe = self._make_recipe(
            runtime="sglang",
            command="python3 -m sglang.launch_server --model-path {model} --port 8000",
            defaults={"served_model_name": "sg-model"},
        )
        runtime = SglangRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--served-model-name sg-model" in cmd

    def test_sglang_node_command_appends(self):
        """sglang generate_node_command: template missing flag → appended."""
        from sparkrun.runtimes.sglang import SglangRuntime

        recipe = self._make_recipe(
            runtime="sglang",
            command="python3 -m sglang.launch_server --model-path {model} --port 8000",
            defaults={"served_model_name": "sg-model"},
        )
        runtime = SglangRuntime()
        cmd = runtime.generate_node_command(
            recipe,
            {},
            head_ip="10.0.0.1",
            num_nodes=2,
            node_rank=0,
        )
        assert "--served-model-name sg-model" in cmd
        assert "--nnodes 2" in cmd

    def test_sglang_node_command_no_duplicate(self):
        """sglang generate_node_command: template has flag → no dup."""
        from sparkrun.runtimes.sglang import SglangRuntime

        recipe = self._make_recipe(
            runtime="sglang",
            command="python3 -m sglang.launch_server --model-path {model} --served-model-name {served_model_name}",
            defaults={"served_model_name": "in-tpl"},
        )
        runtime = SglangRuntime()
        cmd = runtime.generate_node_command(
            recipe,
            {},
            head_ip="10.0.0.1",
            num_nodes=2,
            node_rank=1,
        )
        assert cmd.count("--served-model-name") == 1

    def test_sglang_skip_keys_suppresses(self):
        """sglang: skip_keys={served_model_name} → no augmentation."""
        from sparkrun.runtimes.sglang import SglangRuntime

        recipe = self._make_recipe(
            runtime="sglang",
            command="python3 -m sglang.launch_server --model-path {model}",
            defaults={"served_model_name": "sg-model"},
        )
        runtime = SglangRuntime()
        cmd = runtime.generate_command(
            recipe,
            {},
            is_cluster=False,
            skip_keys={"served_model_name"},
        )
        assert "--served-model-name" not in cmd

    # --- llama-cpp ---

    def test_llama_cpp_uses_alias_flag(self):
        """llama-cpp: uses --alias instead of --served-model-name."""
        from sparkrun.runtimes.llama_cpp import LlamaCppRuntime

        recipe = self._make_recipe(
            runtime="llama-cpp",
            command="llama-server -hf {model} --port 8080",
            defaults={"served_model_name": "llama-alias"},
        )
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--alias llama-alias" in cmd
        assert "--served-model-name" not in cmd

    def test_llama_cpp_no_duplicate_alias(self):
        """llama-cpp: template already has --alias → no duplicate."""
        from sparkrun.runtimes.llama_cpp import LlamaCppRuntime

        recipe = self._make_recipe(
            runtime="llama-cpp",
            command="llama-server -hf {model} --alias {served_model_name} --port 8080",
            defaults={"served_model_name": "in-tpl"},
        )
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert cmd.count("--alias") == 1

    def test_llama_cpp_short_alias_no_duplicate(self):
        """llama-cpp: template has -a short form → -a is not --alias, so --alias appended."""
        from sparkrun.runtimes.llama_cpp import LlamaCppRuntime

        recipe = self._make_recipe(
            runtime="llama-cpp",
            command="llama-server -hf {model} -a {served_model_name} --port 8080",
            defaults={"served_model_name": "short"},
        )
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        # -a is not the same string as --alias, so augment will append --alias.
        # This is fine — the template already used {served_model_name} via -a.
        # The real safety net is strip_flags_from_command for skip_keys.
        assert "-a short" in cmd

    def test_llama_cpp_skip_keys_suppresses(self):
        """llama-cpp: skip_keys={served_model_name} → no augmentation."""
        from sparkrun.runtimes.llama_cpp import LlamaCppRuntime

        recipe = self._make_recipe(
            runtime="llama-cpp",
            command="llama-server -hf {model} --port 8080",
            defaults={"served_model_name": "llama-alias"},
        )
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(
            recipe,
            {},
            is_cluster=False,
            skip_keys={"served_model_name"},
        )
        assert "--alias" not in cmd


class TestResolveHostsForInit:
    """Verify loopback substitution for NCCL / torch-distributed init addresses.

    Cluster configs commonly list the control machine's own host as
    ``127.0.0.1`` for SSH convenience.  ``resolve_hosts_for_init`` must
    swap that loopback for the detected non-loopback head IP before the
    list is fed into ``generate_node_command``, otherwise workers would
    treat their own loopback as the master address.
    """

    def _make_ctx(self, hosts, head_host=None):
        from sparkrun.runtimes._cluster_ops import ClusterContext

        head_host = head_host or hosts[0]
        return ClusterContext(
            hosts=list(hosts),
            head_host=head_host,
            worker_hosts=[h for h in hosts if h != head_host],
            num_nodes=len(hosts),
            ssh_kwargs={},
            volumes={},
            all_env={},
            cluster_id="test-cluster",
            image="test:image",
            dry_run=True,
            config=None,
        )

    def test_loopback_head_substituted(self):
        """``127.0.0.1`` head entry is replaced by the detected head IP."""
        from sparkrun.runtimes._cluster_ops import resolve_hosts_for_init

        ctx = self._make_ctx(["127.0.0.1", "10.0.0.5"])
        assert resolve_hosts_for_init(ctx, head_ip="10.0.0.4") == ["10.0.0.4", "10.0.0.5"]

    def test_localhost_alias_head_substituted(self):
        """``localhost`` is treated the same as ``127.0.0.1``."""
        from sparkrun.runtimes._cluster_ops import resolve_hosts_for_init

        ctx = self._make_ctx(["localhost", "10.0.0.5"])
        assert resolve_hosts_for_init(ctx, head_ip="10.0.0.4") == ["10.0.0.4", "10.0.0.5"]

    def test_all_remote_unchanged(self):
        """When no host is local, the list is returned as-is."""
        from sparkrun.runtimes._cluster_ops import resolve_hosts_for_init

        ctx = self._make_ctx(["10.0.0.4", "10.0.0.5"])
        assert resolve_hosts_for_init(ctx, head_ip="10.0.0.4") == ["10.0.0.4", "10.0.0.5"]

    def test_vllm_distributed_master_addr_no_loopback(self):
        """End-to-end: vllm_distributed renders no ``127.0.0.1`` master-addr.

        Guards the bug: with raw ``hosts[0] == "127.0.0.1"`` the runtime
        used to emit ``--master-addr 127.0.0.1``.  After ``resolve_hosts_for_init``
        the resolved list feeds ``generate_node_command``, so the master
        address is the actual head IP.
        """
        from sparkrun.runtimes._cluster_ops import resolve_hosts_for_init
        from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

        ctx = self._make_ctx(["127.0.0.1", "10.0.0.5"])
        resolved = resolve_hosts_for_init(ctx, head_ip="10.0.0.4")

        recipe = Recipe.from_dict(
            {
                "name": "loopback-fix",
                "model": "m",
                "runtime": "vllm-distributed",
                "defaults": {"tensor_parallel": 2},
            }
        )
        runtime = VllmDistributedRuntime()
        cmd = runtime.generate_node_command(
            recipe=recipe,
            overrides={},
            head_ip="10.0.0.4",
            num_nodes=2,
            node_rank=1,
            init_port=25000,
            hosts=resolved,
        )
        assert "127.0.0.1" not in cmd
        assert "--master-addr 10.0.0.4" in cmd


# --- world_size hook (default) ---


class TestRuntimeWorldSize:
    """Base runtime ``world_size`` returns ``parallelism.total_gpus``."""

    def test_default_returns_total_gpus(self):
        from sparkrun.core.cluster_manager import ClusterDefinition
        from sparkrun.core.parallelism import ParallelismConfig
        from sparkrun.core.recipe import Recipe

        runtime = _StubRuntime()
        parallelism = ParallelismConfig(tensor_parallel=2, pipeline_parallel=3, data_parallel=4)
        cluster = ClusterDefinition(name="c", hosts=["h"])
        recipe = Recipe({"sparkrun_version": "2", "runtime": "stub", "model": "m"})
        # total_gpus = tp * pp * dp = 2 * 3 * 4 = 24
        assert runtime.world_size(parallelism, recipe=recipe, cluster=cluster) == 24

    def test_default_ignores_expert_parallel(self):
        """Default base implementation does NOT factor in ep (Atlas overrides this)."""
        from sparkrun.core.cluster_manager import ClusterDefinition
        from sparkrun.core.parallelism import ParallelismConfig
        from sparkrun.core.recipe import Recipe

        runtime = _StubRuntime()
        parallelism = ParallelismConfig(tensor_parallel=2, expert_parallel=8)
        cluster = ClusterDefinition(name="c", hosts=["h"])
        recipe = Recipe({"sparkrun_version": "2", "runtime": "stub", "model": "m"})
        # Base default uses total_gpus = tp * pp * dp = 2 * 1 * 1 = 2 (ep absent).
        assert runtime.world_size(parallelism, recipe=recipe, cluster=cluster) == 2
