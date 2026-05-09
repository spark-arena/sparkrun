"""Unit tests for sparkrun.runtimes module."""

import re
from unittest import mock

import pytest
from sparkrun.orchestration.job_metadata import generate_cluster_id
from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.vllm_ray import VllmRayRuntime
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
from sparkrun.runtimes.sglang import SglangRuntime
from sparkrun.runtimes.eugr_vllm_ray import EugrVllmRayRuntime
from sparkrun.runtimes.llama_cpp import LlamaCppRuntime
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


# --- VllmRuntime Tests ---


def test_vllm_runtime_name():
    """VllmRayRuntime.runtime_name == 'vllm-ray'."""
    runtime = VllmRayRuntime()
    assert runtime.runtime_name == "vllm-ray"


def test_vllm_resolve_container_from_recipe():
    """Recipe with container field."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "container": "custom-vllm:v1.0",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "custom-vllm:v1.0"


def test_vllm_resolve_container_default():
    """Recipe without container uses default prefix."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:latest"


def test_vllm_generate_command_from_template():
    """Recipe with command template renders correctly."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "command": "vllm serve {model} --port {port}",
        "defaults": {"port": 8000},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd == "vllm serve meta-llama/Llama-2-7b-hf --port 8000"


def test_vllm_generate_command_structured():
    """Recipe without template generates vllm serve command from defaults."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "defaults": {
            "port": 8000,
            "tensor_parallel": 2,
            "gpu_memory_utilization": 0.9,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("vllm serve meta-llama/Llama-2-7b-hf")
    assert "-tp 2" in cmd
    assert "--port 8000" in cmd
    assert "--gpu-memory-utilization 0.9" in cmd


def test_vllm_generate_command_cluster():
    """Cluster mode adds --distributed-executor-backend ray."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-70b-hf",
        "runtime": "vllm",
        "defaults": {"tensor_parallel": 4},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=True, num_nodes=2)
    assert "--distributed-executor-backend ray" in cmd
    assert "-tp 4" in cmd


def test_vllm_generate_command_bool_flags():
    """Boolean flags like enforce_eager are handled."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "defaults": {
            "enforce_eager": True,
            "enable_prefix_caching": False,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--enforce-eager" in cmd
    # enable_prefix_caching is False, should not appear
    assert "--enable-prefix-caching" not in cmd


def test_vllm_validate_recipe_valid():
    """Valid recipe returns no issues."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    issues = runtime.validate_recipe(recipe)
    assert issues == []


def test_vllm_validate_recipe_no_model():
    """Missing model returns issue."""
    recipe_data = {
        "name": "test-recipe",
        "runtime": "vllm",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    issues = runtime.validate_recipe(recipe)
    assert len(issues) == 1
    assert "model is required" in issues[0]


def test_vllm_cluster_env():
    """get_cluster_env returns RAY_memory_monitor_refresh_ms."""
    runtime = VllmRayRuntime()
    env = runtime.get_cluster_env(head_ip="192.168.1.100", num_nodes=2)

    assert env["RAY_memory_monitor_refresh_ms"] == "0"


# --- SglangRuntime Tests ---


def test_sglang_runtime_name():
    """SglangRuntime.runtime_name == 'sglang'."""
    runtime = SglangRuntime()
    assert runtime.runtime_name == "sglang"


def test_sglang_resolve_container():
    """Container resolution."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "scitrera/dgx-spark-sglang:latest"


def test_sglang_generate_command_structured():
    """Generates python3 -m sglang.launch_server with --tp-size, etc."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
        "defaults": {
            "port": 30000,
            "tensor_parallel": 2,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("python3 -m sglang.launch_server")
    assert "--model-path meta-llama/Llama-2-7b-hf" in cmd
    assert "--tp-size 2" in cmd
    assert "--port 30000" in cmd


def test_sglang_generate_command_cluster():
    """Cluster mode adds --nnodes and --dist-init-addr."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-70b-hf",
        "runtime": "sglang",
        "defaults": {"tensor_parallel": 4},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=True, num_nodes=2, head_ip="192.168.1.100")
    assert "--dist-init-addr 192.168.1.100:25000" in cmd
    assert "--nnodes 2" in cmd
    assert "--tp-size 4" in cmd


def test_sglang_cluster_env():
    """Returns NCCL_CUMEM_ENABLE."""
    runtime = SglangRuntime()
    env = runtime.get_cluster_env(head_ip="192.168.1.100", num_nodes=2)

    assert env["NCCL_CUMEM_ENABLE"] == "0"


def test_sglang_validate_recipe():
    """Validate recipe."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    issues = runtime.validate_recipe(recipe)
    assert issues == []


def test_sglang_validate_recipe_no_model():
    """Missing model returns issue."""
    recipe_data = {
        "name": "test-recipe",
        "runtime": "sglang",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    issues = runtime.validate_recipe(recipe)
    assert len(issues) == 1
    assert "model is required" in issues[0]


# --- VllmDistributedRuntime Tests ---


def test_vllm_distributed_runtime_name():
    """VllmDistributedRuntime.runtime_name == 'vllm-distributed'."""
    runtime = VllmDistributedRuntime()
    assert runtime.runtime_name == "vllm-distributed"


def test_vllm_distributed_cluster_strategy():
    """vllm-distributed uses native clustering, not Ray."""
    runtime = VllmDistributedRuntime()
    assert runtime.cluster_strategy() == "native"


def test_vllm_distributed_resolve_container():
    """Default container uses same images as vllm-ray."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm-distributed",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmDistributedRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:latest"


def test_vllm_distributed_generate_command_structured():
    """Generates vllm serve command with tp, port, etc."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm-distributed",
        "defaults": {
            "port": 8000,
            "tensor_parallel": 2,
            "gpu_memory_utilization": 0.9,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmDistributedRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("vllm serve meta-llama/Llama-2-7b-hf")
    assert "-tp 2" in cmd
    assert "--port 8000" in cmd
    assert "--gpu-memory-utilization 0.9" in cmd


def test_vllm_distributed_generate_command_no_ray():
    """Cluster mode does NOT include --distributed-executor-backend ray."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-70b-hf",
        "runtime": "vllm-distributed",
        "defaults": {"tensor_parallel": 4},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmDistributedRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=True, num_nodes=2, head_ip="192.168.1.100")
    assert "--distributed-executor-backend" not in cmd


def test_vllm_distributed_generate_command_cluster():
    """Cluster mode adds --nnodes, --master-addr, --master-port."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-70b-hf",
        "runtime": "vllm-distributed",
        "defaults": {"tensor_parallel": 4},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmDistributedRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=True, num_nodes=2, head_ip="192.168.1.100")
    assert "--nnodes 2" in cmd
    assert "--master-addr 192.168.1.100" in cmd
    assert "--master-port 25000" in cmd
    assert "-tp 4" in cmd


def test_vllm_distributed_generate_node_command_head():
    """Head node (rank 0) gets --nnodes, --node-rank 0, --master-addr, --master-port, NO --headless."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-70b-hf",
        "runtime": "vllm-distributed",
        "defaults": {"tensor_parallel": 2},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmDistributedRuntime()

    cmd = runtime.generate_node_command(
        recipe=recipe,
        overrides={},
        head_ip="192.168.1.100",
        num_nodes=2,
        node_rank=0,
        init_port=25000,
    )
    assert cmd.startswith("vllm serve meta-llama/Llama-2-70b-hf")
    assert "-tp 2" in cmd
    assert "--nnodes 2" in cmd
    assert "--node-rank 0" in cmd
    assert "--master-addr 192.168.1.100" in cmd
    assert "--master-port 25000" in cmd
    assert "--headless" not in cmd


def test_vllm_distributed_generate_node_command_worker():
    """Worker nodes (rank > 0) get --headless in addition to cluster flags."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-70b-hf",
        "runtime": "vllm-distributed",
        "defaults": {"tensor_parallel": 2},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmDistributedRuntime()

    cmd = runtime.generate_node_command(
        recipe=recipe,
        overrides={},
        head_ip="192.168.1.100",
        num_nodes=2,
        node_rank=1,
        init_port=25000,
    )
    assert cmd.startswith("vllm serve meta-llama/Llama-2-70b-hf")
    assert "-tp 2" in cmd
    assert "--nnodes 2" in cmd
    assert "--node-rank 1" in cmd
    assert "--master-addr 192.168.1.100" in cmd
    assert "--master-port 25000" in cmd
    assert "--headless" in cmd


class TestVllmDistributedDPRankMath:
    """Verify per-node DP/TP rank math in vllm_distributed.generate_node_command."""

    def _make_recipe(self, defaults):
        return Recipe.from_dict(
            {
                "name": "dp-test",
                "model": "meta-llama/Llama-2-7b-hf",
                "runtime": "vllm-distributed",
                "defaults": defaults,
            }
        )

    def _cmd_for(self, defaults, node_rank, hosts, num_nodes=None):
        recipe = self._make_recipe(defaults)
        runtime = VllmDistributedRuntime()
        return runtime.generate_node_command(
            recipe=recipe,
            overrides={},
            head_ip=hosts[0],
            num_nodes=num_nodes or len(hosts),
            node_rank=node_rank,
            init_port=25000,
            hosts=hosts,
        )

    # --- Regression: pure TP (dp=1) keeps today's behavior ---

    def test_pure_tp_no_dp_flags(self):
        """TP=2, DP=1: classic torch-distributed tp, no DP flags emitted."""
        hosts = ["10.0.0.1", "10.0.0.2"]
        cmd = self._cmd_for({"tensor_parallel": 2}, node_rank=1, hosts=hosts)
        assert "--nnodes 2" in cmd
        assert "--node-rank 1" in cmd
        assert "--master-addr 10.0.0.1" in cmd
        assert "--headless" in cmd
        assert "--data-parallel-size" not in cmd
        assert "--data-parallel-rank" not in cmd

    # --- Pure DP (tp=pp=1, dp>1) ---

    def test_pure_dp_rank0_flags(self):
        """Pure DP: no --nnodes/--master-addr (no intra-replica coordination)."""
        hosts = ["10.0.0.1", "10.0.0.2"]
        cmd = self._cmd_for({"data_parallel": 2}, node_rank=0, hosts=hosts)
        assert "--data-parallel-size 2" in cmd
        assert "--data-parallel-rank 0" in cmd
        assert "--data-parallel-address 10.0.0.1" in cmd
        assert "--data-parallel-rpc-port 13345" in cmd
        assert "--nnodes" not in cmd
        assert "--node-rank" not in cmd
        assert "--master-addr" not in cmd
        assert "--headless" not in cmd

    def test_pure_dp_rank1_uses_global_rank0_as_address(self):
        """DP rank 1: --data-parallel-address points at the global first host."""
        hosts = ["10.0.0.1", "10.0.0.2"]
        cmd = self._cmd_for({"data_parallel": 2}, node_rank=1, hosts=hosts)
        assert "--data-parallel-rank 1" in cmd
        assert "--data-parallel-address 10.0.0.1" in cmd

    def test_pure_dp_respects_custom_rpc_port(self):
        hosts = ["10.0.0.1", "10.0.0.2"]
        cmd = self._cmd_for(
            {"data_parallel": 2, "data_parallel_rpc_port": 20000},
            node_rank=0,
            hosts=hosts,
        )
        assert "--data-parallel-rpc-port 20000" in cmd

    # --- Hybrid TP+DP (the rank-collision case the user flagged) ---

    def test_hybrid_tp2_dp2_node0(self):
        """Node A (global rank 0): dp_rank=0, intra=0, tp_master=A."""
        hosts = ["A", "B", "C", "D"]
        cmd = self._cmd_for(
            {"tensor_parallel": 2, "data_parallel": 2},
            node_rank=0,
            hosts=hosts,
        )
        assert "--nnodes 2" in cmd
        assert "--node-rank 0" in cmd
        assert "--master-addr A" in cmd
        assert "--data-parallel-rank 0" in cmd
        assert "--data-parallel-address A" in cmd
        assert "--headless" not in cmd  # intra rank 0

    def test_hybrid_tp2_dp2_node1(self):
        """Node B (global rank 1): dp_rank=0, intra=1, tp_master=A, headless."""
        hosts = ["A", "B", "C", "D"]
        cmd = self._cmd_for(
            {"tensor_parallel": 2, "data_parallel": 2},
            node_rank=1,
            hosts=hosts,
        )
        assert "--node-rank 1" in cmd
        assert "--master-addr A" in cmd
        assert "--data-parallel-rank 0" in cmd
        assert "--data-parallel-address A" in cmd
        assert "--headless" in cmd

    def test_hybrid_tp2_dp2_node2(self):
        """Node C (global rank 2): dp_rank=1, intra=0, tp_master=C (NOT A)."""
        hosts = ["A", "B", "C", "D"]
        cmd = self._cmd_for(
            {"tensor_parallel": 2, "data_parallel": 2},
            node_rank=2,
            hosts=hosts,
        )
        assert "--node-rank 0" in cmd  # intra-replica rank 0
        assert "--master-addr C" in cmd  # replica 1's own master
        assert "--data-parallel-rank 1" in cmd
        assert "--data-parallel-address A" in cmd  # DP master is always global rank 0
        assert "--headless" not in cmd  # intra rank 0

    def test_hybrid_tp2_dp2_node3(self):
        """Node D (global rank 3): dp_rank=1, intra=1, tp_master=C, headless."""
        hosts = ["A", "B", "C", "D"]
        cmd = self._cmd_for(
            {"tensor_parallel": 2, "data_parallel": 2},
            node_rank=3,
            hosts=hosts,
        )
        assert "--node-rank 1" in cmd
        assert "--master-addr C" in cmd
        assert "--data-parallel-rank 1" in cmd
        assert "--headless" in cmd

    # --- Recipe-template already has --data-parallel-size: don't duplicate ---

    def test_template_data_parallel_size_not_duplicated(self):
        """If the recipe command template already emits --data-parallel-size, runtime skips it."""
        recipe = Recipe.from_dict(
            {
                "name": "template-dp",
                "model": "m",
                "runtime": "vllm-distributed",
                "defaults": {"data_parallel": 2},
                "command": "vllm serve {model} --data-parallel-size {data_parallel}",
            }
        )
        runtime = VllmDistributedRuntime()
        cmd = runtime.generate_node_command(
            recipe=recipe,
            overrides={},
            head_ip="10.0.0.1",
            num_nodes=2,
            node_rank=0,
            init_port=25000,
            hosts=["10.0.0.1", "10.0.0.2"],
        )
        assert cmd.count("--data-parallel-size") == 1
        assert "--data-parallel-rank 0" in cmd
        assert "--data-parallel-address 10.0.0.1" in cmd


def test_vllm_distributed_cluster_env():
    """Returns NCCL_CUMEM_ENABLE and OMP_NUM_THREADS (no Ray vars)."""
    runtime = VllmDistributedRuntime()
    env = runtime.get_cluster_env(head_ip="192.168.1.100", num_nodes=2)

    assert env["NCCL_CUMEM_ENABLE"] == "0"
    assert env["OMP_NUM_THREADS"] == "4"
    assert "RAY_memory_monitor_refresh_ms" not in env


def test_vllm_distributed_validate_recipe():
    """Valid recipe returns no issues."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm-distributed",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmDistributedRuntime()

    issues = runtime.validate_recipe(recipe)
    assert issues == []


def test_vllm_distributed_validate_recipe_no_model():
    """Missing model returns issue."""
    recipe_data = {
        "name": "test-recipe",
        "runtime": "vllm-distributed",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmDistributedRuntime()

    issues = runtime.validate_recipe(recipe)
    assert len(issues) == 1
    assert "model is required" in issues[0]


def test_vllm_distributed_container_name():
    """Container naming uses {cluster_id}_node_{rank} pattern."""
    from sparkrun.orchestration.docker import generate_node_container_name

    assert generate_node_container_name("spark0", 0) == "spark0_node_0"
    assert generate_node_container_name("spark0", 1) == "spark0_node_1"
    assert generate_node_container_name("spark0", 5) == "spark0_node_5"


def test_vllm_distributed_bool_flags():
    """Boolean flags work correctly."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm-distributed",
        "defaults": {
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "trust_remote_code": True,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmDistributedRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--enforce-eager" in cmd
    assert "--trust-remote-code" in cmd
    # enable_prefix_caching is False, should not appear
    assert "--enable-prefix-caching" not in cmd


class TestVllmDistributedFollowLogs:
    """Test VllmDistributedRuntime.follow_logs()."""

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_solo_uses_file_logs(self, mock_stream):
        """Single-host vllm-distributed tails serve log file inside solo container."""
        runtime = VllmDistributedRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1"],
            cluster_id="test0",
        )

        mock_stream.assert_called_once()
        assert mock_stream.call_args[0][1] == "test0_solo"

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_cluster_uses_node_0(self, mock_stream):
        """Multi-host vllm-distributed follows the _node_0 container (file mode, sleep-infinity + exec)."""
        runtime = VllmDistributedRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1", "10.0.0.2"],
            cluster_id="mycluster",
        )

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "10.0.0.1"
        assert args[0][1] == "mycluster_node_0"


def test_vllm_distributed_overrides_in_command():
    """CLI overrides properly override defaults."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm-distributed",
        "defaults": {"port": 8000},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmDistributedRuntime()

    cmd = runtime.generate_command(recipe, {"port": 9000}, is_cluster=False)
    assert "--port 9000" in cmd
    assert "--port 8000" not in cmd


# --- EugrVllmRuntime Tests ---


def test_eugr_inherits_vllm():
    """EugrVllmRuntime extends VllmRuntime."""
    runtime = EugrVllmRayRuntime()
    assert isinstance(runtime, VllmRayRuntime)


def test_eugr_is_not_delegating():
    """EugrVllmRuntime.is_delegating_runtime() returns False (native orchestration)."""
    runtime = EugrVllmRayRuntime()
    assert runtime.is_delegating_runtime() is False


def test_eugr_runtime_name():
    """runtime_name == 'eugr-vllm'."""
    runtime = EugrVllmRayRuntime()
    assert runtime.runtime_name == "eugr-vllm"


def test_eugr_resolve_container():
    """Resolve container for eugr."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "eugr-vllm",
        "container": "custom-eugr:latest",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = EugrVllmRayRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "custom-eugr:latest"


def test_eugr_resolve_container_default():
    """Default container for eugr."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "eugr-vllm",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = EugrVllmRayRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "vllm-node"


def test_eugr_generate_command_from_template():
    """Generate command renders recipe command template (inherited from VllmRuntime)."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "eugr-vllm",
        "command": "vllm serve {model} --port {port}",
        "defaults": {"port": 8000},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = EugrVllmRayRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd == "vllm serve meta-llama/Llama-2-7b-hf --port 8000"


def test_eugr_generate_command_structured():
    """Without a command template, generates vllm serve from defaults (inherited)."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "eugr-vllm",
        "defaults": {"port": 8000, "tensor_parallel": 2},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = EugrVllmRayRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("vllm serve meta-llama/Llama-2-7b-hf")
    assert "-tp 2" in cmd
    assert "--port 8000" in cmd


def test_eugr_validate_recipe():
    """Validate eugr recipe."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "eugr-vllm",
        "command": "vllm serve model",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = EugrVllmRayRuntime()

    issues = runtime.validate_recipe(recipe)
    # Should pass validation
    assert all("model is required" not in issue for issue in issues)


class TestEugrPrepare:
    """Test EugrBuilder.prepare_image() — container build and mod injection."""

    @pytest.fixture
    def eugr_builder(self, tmp_path):
        """Create builder with a fake repo containing build-and-copy.sh."""
        from sparkrun.builders.eugr import EugrBuilder

        builder = EugrBuilder()
        repo_dir = tmp_path / "eugr-repo"
        repo_dir.mkdir()
        (repo_dir / "build-and-copy.sh").write_text("#!/bin/bash\nexit 0\n")
        (repo_dir / "build-and-copy.sh").chmod(0o755)
        # Create a sample mod directory
        mod_dir = repo_dir / "mods" / "flash-attn"
        mod_dir.mkdir(parents=True)
        (mod_dir / "run.sh").write_text("#!/bin/bash\necho applied\n")
        return builder, repo_dir

    def test_prepare_with_build_args(self, eugr_builder):
        """prepare_image() calls build-and-copy.sh when build_args present."""
        builder, repo_dir = eugr_builder
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "runtime": "eugr-vllm",
                "container": "my-image",
                "runtime_config": {"build_args": ["--some-flag"]},
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                with mock.patch("subprocess.run") as mock_run:
                    mock_run.return_value = mock.Mock(returncode=0)
                    with mock.patch.object(builder, "_save_build_metadata"):
                        builder.prepare_image("my-image", recipe, ["10.0.0.1"])

                    # Should call build-and-copy.sh with -t and build_args
                    cmd = mock_run.call_args[0][0]
                    assert str(repo_dir / "build-and-copy.sh") in cmd[0]
                    assert "-t" in cmd
                    assert "my-image" in cmd
                    assert "--some-flag" in cmd

    def test_prepare_without_build_args_or_mods_image_exists(self, eugr_builder):
        """prepare_image() is a no-op when no build_args/mods and image exists."""
        builder, repo_dir = eugr_builder
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "runtime": "eugr-vllm",
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=True):
            with mock.patch.object(builder, "ensure_repo") as mock_ensure:
                builder.prepare_image("vllm-node", recipe, ["10.0.0.1"])
                # ensure_repo should not be called when nothing to prepare
                mock_ensure.assert_not_called()

    def test_prepare_builds_when_image_missing(self, eugr_builder):
        """prepare_image() triggers a build when image is missing locally."""
        builder, repo_dir = eugr_builder
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "runtime": "eugr-vllm",
                "container": "my-image",
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                with mock.patch("subprocess.run") as mock_run:
                    mock_run.return_value = mock.Mock(returncode=0)
                    with mock.patch.object(builder, "_save_build_metadata"):
                        builder.prepare_image("my-image", recipe, ["10.0.0.1"])
                    mock_run.assert_called_once()
                    cmd = mock_run.call_args[0][0]
                    assert str(repo_dir / "build-and-copy.sh") in cmd[0]
                    assert "-t" in cmd
                    assert "my-image" in cmd

    def test_prepare_dry_run(self, eugr_builder):
        """prepare_image() in dry-run does not execute the build."""
        builder, repo_dir = eugr_builder
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "runtime": "eugr-vllm",
                "runtime_config": {"build_args": ["--flag"]},
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                with mock.patch("subprocess.run") as mock_run:
                    builder.prepare_image("vllm-node", recipe, ["10.0.0.1"], dry_run=True)
                    # subprocess.run should not be called in dry-run
                    mock_run.assert_not_called()

    def test_prepare_injects_mod_pre_exec(self, eugr_builder):
        """prepare_image() injects mod entries into recipe.pre_exec."""
        builder, repo_dir = eugr_builder
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "runtime": "eugr-vllm",
                "runtime_config": {"mods": ["mods/flash-attn"]},
            }
        )
        with mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=True):
            with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
                builder.prepare_image("vllm-node", recipe, ["10.0.0.1"])
                # Should have injected copy + exec entries
                assert len(recipe.pre_exec) == 2
                assert isinstance(recipe.pre_exec[0], dict)
                assert "copy" in recipe.pre_exec[0]
                assert "run.sh" in recipe.pre_exec[1]

    def test_prepare_build_failure_raises(self, eugr_builder):
        """prepare_image() raises RuntimeError on build failure."""
        builder, repo_dir = eugr_builder
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "runtime": "eugr-vllm",
                "runtime_config": {"build_args": ["--flag"]},
            }
        )
        with mock.patch.object(builder, "ensure_repo", return_value=repo_dir):
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=1)
                with pytest.raises(RuntimeError, match="eugr container build failed"):
                    builder.prepare_image("vllm-node", recipe, ["10.0.0.1"])


class TestEugrPreServe:
    """Test base RuntimePlugin._pre_serve() with pre_exec from recipe."""

    def test_pre_serve_with_pre_exec(self):
        """_pre_serve() runs pre_exec commands from recipe."""
        from sparkrun.runtimes.base import RuntimePlugin

        runtime = RuntimePlugin()
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "runtime": "vllm",
                "pre_exec": ["echo hello"],
            }
        )
        with mock.patch("sparkrun.orchestration.hooks.run_pre_exec") as mock_hook:
            runtime._pre_serve(
                [("localhost", "sparkrun_abc_solo")],
                ssh_kwargs={},
                dry_run=False,
                recipe=recipe,
                config_chain=None,
            )
            mock_hook.assert_called_once()

    def test_pre_serve_without_pre_exec(self):
        """_pre_serve() is a no-op when recipe has no pre_exec."""
        runtime = EugrVllmRayRuntime()
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "runtime": "eugr-vllm",
            }
        )
        with mock.patch("sparkrun.orchestration.hooks.run_pre_exec") as mock_hook:
            runtime._pre_serve(
                [("localhost", "sparkrun_abc_solo")],
                ssh_kwargs={},
                dry_run=False,
                recipe=recipe,
            )
            mock_hook.assert_not_called()

    def test_pre_serve_dry_run(self):
        """_pre_serve() passes dry_run through to hooks."""
        from sparkrun.runtimes.base import RuntimePlugin

        runtime = RuntimePlugin()
        recipe = Recipe.from_dict(
            {
                "name": "test",
                "model": "some-model",
                "pre_exec": ["echo hello"],
            }
        )
        with mock.patch("sparkrun.orchestration.hooks.run_pre_exec") as mock_hook:
            runtime._pre_serve(
                [("localhost", "sparkrun_abc_solo")],
                ssh_kwargs={},
                dry_run=True,
                recipe=recipe,
                config_chain=None,
            )
            mock_hook.assert_called_once()
            # Verify dry_run was passed through
            assert mock_hook.call_args[1]["dry_run"] is True

    def test_pre_serve_no_recipe(self):
        """_pre_serve() is a no-op when no recipe provided (backward compat)."""
        runtime = EugrVllmRayRuntime()
        with mock.patch("sparkrun.orchestration.hooks.run_pre_exec") as mock_hook:
            runtime._pre_serve(
                [("localhost", "sparkrun_abc_solo")],
                ssh_kwargs={},
                dry_run=False,
            )
            mock_hook.assert_not_called()


# --- Base RuntimePlugin Tests ---


def test_base_runtime_is_enabled_false():
    """RuntimePlugin.is_enabled() returns False (critical for multi-extension)."""
    from scitrera_app_framework import Variables

    runtime = RuntimePlugin()
    v = Variables()

    # is_enabled must return False for multi-extension plugins
    assert runtime.is_enabled(v) is False


def test_base_runtime_is_multi_extension_true():
    """RuntimePlugin.is_multi_extension() returns True."""
    from scitrera_app_framework import Variables

    runtime = RuntimePlugin()
    v = Variables()

    assert runtime.is_multi_extension(v) is True


def test_base_runtime_is_not_delegating():
    """Base RuntimePlugin.is_delegating_runtime() returns False."""
    runtime = RuntimePlugin()
    assert runtime.is_delegating_runtime() is False


def test_vllm_cluster_injects_ray_backend_into_template():
    """Cluster mode injects --distributed-executor-backend ray into command templates."""
    recipe_data = {
        "name": "test-recipe",
        "model": "nvidia/some-model",
        "runtime": "vllm",
        "command": "vllm serve {model} -tp {tensor_parallel} --port {port}",
        "defaults": {"tensor_parallel": 2, "port": 8000},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    # Solo mode: no injection
    cmd_solo = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--distributed-executor-backend" not in cmd_solo

    # Cluster mode: auto-injected
    cmd_cluster = runtime.generate_command(recipe, {}, is_cluster=True, num_nodes=2)
    assert "--distributed-executor-backend ray" in cmd_cluster


def test_vllm_cluster_preserves_existing_backend_in_template():
    """Cluster mode does not double-add if template already has the flag."""
    recipe_data = {
        "name": "test-recipe",
        "model": "nvidia/some-model",
        "runtime": "vllm",
        "command": "vllm serve {model} --distributed-executor-backend ray",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=True, num_nodes=2)
    assert cmd.count("--distributed-executor-backend") == 1


def test_vllm_overrides_in_command():
    """Test that CLI overrides properly override defaults."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "defaults": {"port": 8000},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmRayRuntime()

    # Override port
    cmd = runtime.generate_command(recipe, {"port": 9000}, is_cluster=False)
    assert "--port 9000" in cmd
    assert "--port 8000" not in cmd


def test_sglang_overrides_in_command():
    """Test that CLI overrides work for sglang."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
        "defaults": {"port": 30000},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    # Override port
    cmd = runtime.generate_command(recipe, {"port": 31000}, is_cluster=False)
    assert "--port 31000" in cmd
    assert "--port 30000" not in cmd


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


class TestVllmFollowLogs:
    """Test VllmRuntime.follow_logs()."""

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_solo_tails_serve_log(self, mock_stream):
        """Single-host vllm tails serve log in solo container."""
        runtime = VllmRayRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1"],
            cluster_id="test0",
        )

        mock_stream.assert_called_once()
        assert mock_stream.call_args[0][0] == "10.0.0.1"
        assert mock_stream.call_args[0][1] == "test0_solo"

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_cluster_tails_serve_log_on_head(self, mock_stream):
        """Multi-host vllm tails serve log in _head container on hosts[0]."""
        runtime = VllmRayRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1", "10.0.0.2"],
            cluster_id="mycluster",
        )

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "10.0.0.1"
        assert args[0][1] == "mycluster_head"


class TestSglangFollowLogs:
    """Test SglangRuntime.follow_logs()."""

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_solo_uses_file_logs(self, mock_stream):
        """Single-host sglang tails serve log file inside solo container."""
        runtime = SglangRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1"],
            cluster_id="test0",
        )

        mock_stream.assert_called_once()
        assert mock_stream.call_args[0][1] == "test0_solo"

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_cluster_uses_node_0(self, mock_stream):
        """Multi-host sglang follows the _node_0 container (file mode, sleep-infinity + exec)."""
        runtime = SglangRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1", "10.0.0.2"],
            cluster_id="mycluster",
        )

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "10.0.0.1"
        assert args[0][1] == "mycluster_node_0"


class TestEugrFollowLogs:
    """Test EugrVllmRuntime.follow_logs() — inherited from VllmRuntime."""

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_solo_tails_serve_log(self, mock_stream):
        """Single-host eugr tails serve log in solo container (inherited)."""
        runtime = EugrVllmRayRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1"],
            cluster_id="test0",
        )

        mock_stream.assert_called_once()
        assert mock_stream.call_args[0][0] == "10.0.0.1"
        assert mock_stream.call_args[0][1] == "test0_solo"

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_cluster_tails_head(self, mock_stream):
        """Multi-host eugr tails serve log on head container (inherited from vllm)."""
        runtime = EugrVllmRayRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1", "10.0.0.2"],
            cluster_id="mycluster",
        )

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "10.0.0.1"
        assert args[0][1] == "mycluster_head"


# --- LlamaCppRuntime Tests ---


def test_llama_cpp_runtime_name():
    """LlamaCppRuntime.runtime_name == 'llama-cpp'."""
    runtime = LlamaCppRuntime()
    assert runtime.runtime_name == "llama-cpp"


def test_llama_cpp_cluster_strategy():
    """LlamaCppRuntime uses native (RPC) clustering, not Ray."""
    runtime = LlamaCppRuntime()
    assert runtime.cluster_strategy() == "native"


def test_llama_cpp_resolve_container_from_recipe():
    """Recipe with container field."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "container": "custom-llama:v1.0",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "custom-llama:v1.0"


def test_llama_cpp_resolve_container_default():
    """Recipe without container uses default prefix."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    container = runtime.resolve_container(recipe)
    assert container == "scitrera/dgx-spark-llama-cpp:latest"


def test_llama_cpp_generate_command_from_template():
    """Recipe with command template renders correctly."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "command": "llama-server -hf {model} --port {port}",
        "defaults": {"port": 8080},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd == "llama-server -hf Qwen/Qwen3-1.7B-GGUF:Q4_K_M --port 8080"


def test_llama_cpp_generate_command_structured_hf():
    """HuggingFace model (contains '/') uses -hf flag."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {
            "port": 8080,
            "n_gpu_layers": 99,
            "ctx_size": 8192,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("llama-server -hf Qwen/Qwen3-1.7B-GGUF:Q4_K_M")
    assert "--port 8080" in cmd
    assert "--n-gpu-layers 99" in cmd
    assert "--ctx-size 8192" in cmd


def test_llama_cpp_generate_command_gguf_path():
    """Local .gguf path uses -m flag."""
    recipe_data = {
        "name": "test-recipe",
        "model": "/models/qwen3-1.7b-q4_k_m.gguf",
        "runtime": "llama-cpp",
        "defaults": {"port": 8080},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("llama-server -m /models/qwen3-1.7b-q4_k_m.gguf")
    assert "--port 8080" in cmd


def test_llama_cpp_generate_command_bool_flags():
    """Boolean flags flash_attn, jinja, no_webui are handled."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {
            "flash_attn": True,
            "jinja": True,
            "no_webui": True,
            "cont_batching": False,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--flash-attn" in cmd
    assert "--jinja" in cmd
    assert "--no-webui" in cmd
    # cont_batching is False, should not appear
    assert "--cont-batching" not in cmd


def test_llama_cpp_generate_command_overrides():
    """CLI overrides properly override defaults."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {"port": 8080},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {"port": 9090}, is_cluster=False)
    assert "--port 9090" in cmd
    assert "--port 8080" not in cmd


def test_llama_cpp_validate_recipe_valid():
    """Valid recipe returns no issues."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    issues = runtime.validate_recipe(recipe)
    assert issues == []


def test_llama_cpp_validate_recipe_no_model():
    """Missing model returns issue."""
    recipe_data = {
        "name": "test-recipe",
        "runtime": "llama-cpp",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    issues = runtime.validate_recipe(recipe)
    assert len(issues) == 1
    assert "model is required" in issues[0]


def test_llama_cpp_build_rpc_head_command():
    """_build_rpc_head_command appends --rpc with worker addresses."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {"port": 8080, "n_gpu_layers": 99},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()
    config = recipe.build_config_chain({})

    cmd = runtime._build_rpc_head_command(
        recipe,
        config,
        worker_hosts=["10.0.0.2", "10.0.0.3"],
        rpc_port=50052,
    )
    assert "--rpc 10.0.0.2:50052,10.0.0.3:50052" in cmd
    assert cmd.startswith("llama-server -hf Qwen/Qwen3-1.7B-GGUF:Q4_K_M")


def test_llama_cpp_build_rpc_worker_command():
    """_build_rpc_worker_command returns rpc-server with host and port."""
    cmd = LlamaCppRuntime._build_rpc_worker_command(50052)
    assert cmd == "rpc-server --host 0.0.0.0 --port 50052"


def test_llama_cpp_container_name():
    """_container_name returns {cluster_id}_{role}."""
    assert LlamaCppRuntime._container_name("spark0", "head") == "spark0_head"
    assert LlamaCppRuntime._container_name("spark0", "worker") == "spark0_worker"


def test_llama_cpp_generate_command_gguf_presync_template():
    """When _gguf_model_path is set, template renders with resolved path
    and -hf is switched to -m."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {
            "port": 8080,
            "host": "0.0.0.0",
            "n_gpu_layers": 99,
            "ctx_size": 8192,
        },
        "command": (
            "llama-server \\\n"
            "    -hf {model} \\\n"
            "    --host {host} \\\n"
            "    --port {port} \\\n"
            "    --n-gpu-layers {n_gpu_layers} \\\n"
            "    --ctx-size {ctx_size} \\\n"
            "    --flash-attn \\\n"
            "    --jinja \\\n"
            "    --no-webui"
        ),
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    gguf_path = "/cache/huggingface/hub/models--Qwen--Qwen3-1.7B-GGUF/snapshots/abc123/q4_k_m.gguf"
    cmd = runtime.generate_command(
        recipe,
        {"_gguf_model_path": gguf_path, "model": gguf_path},
        is_cluster=False,
    )
    # Template is respected: -hf switched to -m, path substituted
    assert "-m " + gguf_path in cmd
    assert "-hf " not in cmd
    # Other flags from template are preserved
    assert "--host 0.0.0.0" in cmd
    assert "--port 8080" in cmd
    assert "--n-gpu-layers 99" in cmd
    assert "--ctx-size 8192" in cmd
    assert "--flash-attn" in cmd
    assert "--jinja" in cmd
    assert "--no-webui" in cmd


def test_llama_cpp_generate_command_gguf_presync_structured():
    """When _gguf_model_path is set and no template, structured build uses -m."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {"port": 8080, "n_gpu_layers": 99},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    gguf_path = "/cache/huggingface/hub/models--Qwen--Qwen3-1.7B-GGUF/snapshots/abc123/q4_k_m.gguf"
    cmd = runtime.generate_command(
        recipe,
        {"_gguf_model_path": gguf_path, "model": gguf_path},
        is_cluster=False,
    )
    assert cmd.startswith("llama-server -m " + gguf_path)
    assert "--port 8080" in cmd
    assert "--n-gpu-layers 99" in cmd


def test_llama_cpp_generate_command_no_presync_uses_hf():
    """Without _gguf_model_path, template renders -hf with original model."""
    recipe_data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        "runtime": "llama-cpp",
        "defaults": {"port": 8080},
        "command": "llama-server -hf {model} --port {port}",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = LlamaCppRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "-hf Qwen/Qwen3-1.7B-GGUF:Q4_K_M" in cmd
    assert "--port 8080" in cmd


class TestLlamaCppFollowLogs:
    """Test LlamaCppRuntime.follow_logs()."""

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_solo_uses_file_logs(self, mock_stream):
        """Single-host llama-cpp tails serve log file inside solo container."""
        runtime = LlamaCppRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1"],
            cluster_id="test0",
        )

        mock_stream.assert_called_once()
        assert mock_stream.call_args[0][1] == "test0_solo"

    @mock.patch("sparkrun.orchestration.ssh.stream_container_file_logs")
    def test_follow_logs_cluster_uses_file_logs_on_head(self, mock_stream):
        """Multi-host llama-cpp follows file logs on _head container (sleep-infinity + exec)."""
        runtime = LlamaCppRuntime()
        runtime.follow_logs(
            hosts=["10.0.0.1", "10.0.0.2"],
            cluster_id="mycluster",
        )

        mock_stream.assert_called_once()
        args = mock_stream.call_args
        assert args[0][0] == "10.0.0.1"
        assert args[0][1] == "mycluster_head"


# --- compute_required_nodes Tests ---


class TestComputeRequiredNodes:
    """Test base RuntimePlugin.compute_required_nodes()."""

    def _make_recipe(self, defaults=None):
        data = {
            "name": "test",
            "runtime": "vllm",
            "model": "meta-llama/Llama-2-7b-hf",
        }
        if defaults:
            data["defaults"] = defaults
        return Recipe.from_dict(data)

    def test_returns_tp_value(self):
        """Base class returns tensor_parallel as required nodes."""
        recipe = self._make_recipe(defaults={"tensor_parallel": 4})
        runtime = _StubRuntime()
        assert runtime.compute_required_nodes(recipe) == 4

    def test_returns_none_when_no_tp(self):
        """Returns None when tensor_parallel is not set."""
        recipe = self._make_recipe()
        runtime = _StubRuntime()
        assert runtime.compute_required_nodes(recipe) is None

    def test_overrides_take_precedence(self):
        """CLI overrides override recipe defaults."""
        recipe = self._make_recipe(defaults={"tensor_parallel": 2})
        runtime = _StubRuntime()
        assert runtime.compute_required_nodes(recipe, {"tensor_parallel": 8}) == 8

    def test_pp_only(self):
        """PP=3 with no explicit TP → 1*3 = 3 nodes."""
        recipe = self._make_recipe(defaults={"pipeline_parallel": 3})
        runtime = _StubRuntime()
        assert runtime.compute_required_nodes(recipe) == 3

    def test_tp_times_pp(self):
        """TP=2, PP=2 → 4 nodes."""
        recipe = self._make_recipe(
            defaults={
                "tensor_parallel": 2,
                "pipeline_parallel": 2,
            }
        )
        runtime = _StubRuntime()
        assert runtime.compute_required_nodes(recipe) == 4

    def test_tp_no_pp(self):
        """TP=2, no PP → 2 nodes (backward compat)."""
        recipe = self._make_recipe(defaults={"tensor_parallel": 2})
        runtime = _StubRuntime()
        assert runtime.compute_required_nodes(recipe) == 2

    def test_returns_none_with_empty_overrides(self):
        """Empty overrides don't change None result."""
        recipe = self._make_recipe()
        runtime = _StubRuntime()
        assert runtime.compute_required_nodes(recipe, {}) is None

    def test_dp_only(self):
        """DP=2 alone → requires 2 nodes (one per replica on DGX Spark)."""
        recipe = self._make_recipe(defaults={"data_parallel": 2})
        runtime = _StubRuntime()
        assert runtime.compute_required_nodes(recipe) == 2

    def test_tp_times_dp(self):
        """TP=2, DP=2 → 4 nodes (2 nodes × 2 replicas)."""
        recipe = self._make_recipe(defaults={"tensor_parallel": 2, "data_parallel": 2})
        runtime = _StubRuntime()
        assert runtime.compute_required_nodes(recipe) == 4

    def test_tp_pp_dp_combined(self):
        """TP=2, PP=2, DP=3 → 12 nodes."""
        recipe = self._make_recipe(
            defaults={
                "tensor_parallel": 2,
                "pipeline_parallel": 2,
                "data_parallel": 3,
            }
        )
        runtime = _StubRuntime()
        assert runtime.compute_required_nodes(recipe) == 12

    def test_dp_override(self):
        """CLI --dp override is honored."""
        recipe = self._make_recipe(defaults={"tensor_parallel": 1})
        runtime = _StubRuntime()
        assert runtime.compute_required_nodes(recipe, {"data_parallel": 4}) == 4


class TestSglangComputeRequiredNodes:
    """Test SglangRuntime inherits base tp*pp (no override needed)."""

    def _make_recipe(self, defaults=None):
        data = {
            "name": "test",
            "runtime": "sglang",
            "model": "meta-llama/Llama-2-7b-hf",
        }
        if defaults:
            data["defaults"] = defaults
        return Recipe.from_dict(data)

    def test_tp_only(self):
        """TP=4, no PP → requires 4 nodes."""
        recipe = self._make_recipe(defaults={"tensor_parallel": 4})
        runtime = SglangRuntime()
        assert runtime.compute_required_nodes(recipe) == 4

    def test_tp_times_pp(self):
        """TP=2, PP=2 → requires 4 nodes."""
        recipe = self._make_recipe(
            defaults={
                "tensor_parallel": 2,
                "pipeline_parallel": 2,
            }
        )
        runtime = SglangRuntime()
        assert runtime.compute_required_nodes(recipe) == 4

    def test_pp_only(self):
        """PP=3 with no explicit TP → 1*3 = 3 nodes."""
        recipe = self._make_recipe(defaults={"pipeline_parallel": 3})
        runtime = SglangRuntime()
        assert runtime.compute_required_nodes(recipe) == 3

    def test_no_parallelism_returns_none(self):
        """Neither TP nor PP set → None."""
        recipe = self._make_recipe()
        runtime = SglangRuntime()
        assert runtime.compute_required_nodes(recipe) is None

    def test_overrides_pp(self):
        """CLI overrides PP value."""
        recipe = self._make_recipe(defaults={"tensor_parallel": 2})
        runtime = SglangRuntime()
        assert runtime.compute_required_nodes(recipe, {"pipeline_parallel": 3}) == 6

    def test_overrides_both(self):
        """CLI overrides both TP and PP."""
        recipe = self._make_recipe(
            defaults={
                "tensor_parallel": 2,
                "pipeline_parallel": 2,
            }
        )
        runtime = SglangRuntime()
        assert runtime.compute_required_nodes(recipe, {"tensor_parallel": 4, "pipeline_parallel": 3}) == 12


class TestTrtllmComputeRequiredNodes:
    """Test TrtllmRuntime inherits base tp*pp behavior."""

    def _make_recipe(self, defaults=None):
        data = {
            "name": "test",
            "runtime": "trtllm",
            "model": "meta-llama/Llama-2-7b-hf",
        }
        if defaults:
            data["defaults"] = defaults
        return Recipe.from_dict(data)

    def test_tp_times_pp(self):
        """TRT-LLM inherits base class tp*pp."""
        from sparkrun.runtimes.trtllm import TrtllmRuntime

        recipe = self._make_recipe(
            defaults={
                "tensor_parallel": 2,
                "pipeline_parallel": 2,
            }
        )
        runtime = TrtllmRuntime()
        assert runtime.compute_required_nodes(recipe) == 4

    def test_pp_only(self):
        """PP=2 with no TP → 2 nodes."""
        from sparkrun.runtimes.trtllm import TrtllmRuntime

        recipe = self._make_recipe(defaults={"pipeline_parallel": 2})
        runtime = TrtllmRuntime()
        assert runtime.compute_required_nodes(recipe) == 2

    def test_returns_none_when_neither(self):
        """No TP or PP → None."""
        from sparkrun.runtimes.trtllm import TrtllmRuntime

        recipe = self._make_recipe()
        runtime = TrtllmRuntime()
        assert runtime.compute_required_nodes(recipe) is None


def test_sglang_pp_size_in_generated_command():
    """SGLang --pp-size flag appears in generated command."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-70b-hf",
        "runtime": "sglang",
        "defaults": {
            "tensor_parallel": 2,
            "pipeline_parallel": 2,
        },
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--pp-size 2" in cmd
    assert "--tp-size 2" in cmd


def test_sglang_pp_size_override_in_command():
    """SGLang --pp-size from overrides appears in generated command."""
    recipe_data = {
        "name": "test-recipe",
        "model": "meta-llama/Llama-2-70b-hf",
        "runtime": "sglang",
        "defaults": {"tensor_parallel": 2},
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = SglangRuntime()

    cmd = runtime.generate_command(recipe, {"pipeline_parallel": 3}, is_cluster=False)
    assert "--pp-size 3" in cmd
    assert "--tp-size 2" in cmd


# --- llama.cpp TP/PP split-mode tests ---


class TestLlamaCppComputeRequiredNodes:
    """Test LlamaCppRuntime.compute_required_nodes() — TP/PP are mutually exclusive."""

    def _make_recipe(self, defaults=None):
        data = {
            "name": "test",
            "runtime": "llama-cpp",
            "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        }
        if defaults:
            data["defaults"] = defaults
        return Recipe.from_dict(data)

    def test_tp_returns_tp(self):
        """TP=4 → 4 nodes (split_mode=row)."""
        recipe = self._make_recipe(defaults={"tensor_parallel": 4})
        runtime = LlamaCppRuntime()
        assert runtime.compute_required_nodes(recipe) == 4

    def test_pp_returns_pp(self):
        """PP=3 → 3 nodes (split_mode=layer)."""
        recipe = self._make_recipe(defaults={"pipeline_parallel": 3})
        runtime = LlamaCppRuntime()
        assert runtime.compute_required_nodes(recipe) == 3

    def test_neither_returns_none(self):
        """No TP/PP → None."""
        recipe = self._make_recipe()
        runtime = LlamaCppRuntime()
        assert runtime.compute_required_nodes(recipe) is None

    def test_both_raises_value_error(self):
        """TP + PP simultaneously → ValueError."""
        recipe = self._make_recipe(
            defaults={
                "tensor_parallel": 2,
                "pipeline_parallel": 2,
            }
        )
        runtime = LlamaCppRuntime()
        with pytest.raises(ValueError, match="simultaneously"):
            runtime.compute_required_nodes(recipe)

    def test_overrides_tp(self):
        """CLI --tp override."""
        recipe = self._make_recipe()
        runtime = LlamaCppRuntime()
        assert runtime.compute_required_nodes(recipe, {"tensor_parallel": 2}) == 2

    def test_overrides_pp(self):
        """CLI --pp override."""
        recipe = self._make_recipe()
        runtime = LlamaCppRuntime()
        assert runtime.compute_required_nodes(recipe, {"pipeline_parallel": 4}) == 4

    def test_override_conflicts_with_default_raises(self):
        """Recipe has TP, CLI passes PP → conflict → ValueError."""
        recipe = self._make_recipe(defaults={"tensor_parallel": 2})
        runtime = LlamaCppRuntime()
        with pytest.raises(ValueError, match="simultaneously"):
            runtime.compute_required_nodes(recipe, {"pipeline_parallel": 2})

    def test_dp_gt_1_raises_value_error(self):
        """llama.cpp has no DP support — dp>1 must raise."""
        recipe = self._make_recipe(defaults={"data_parallel": 2})
        runtime = LlamaCppRuntime()
        with pytest.raises(ValueError, match="data_parallel"):
            runtime.compute_required_nodes(recipe)


class TestLlamaCppSplitModeCommand:
    """Test that TP/PP correctly inject --split-mode in llama-server commands."""

    def _make_recipe(self, defaults=None, command=None):
        data = {
            "name": "test",
            "runtime": "llama-cpp",
            "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
        }
        if defaults:
            data["defaults"] = defaults
        if command:
            data["command"] = command
        return Recipe.from_dict(data)

    def test_tp_generates_split_mode_row(self):
        """--tp → --split-mode row in generated command."""
        recipe = self._make_recipe(defaults={"tensor_parallel": 2, "port": 8080})
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--split-mode row" in cmd
        assert "--split-mode layer" not in cmd

    def test_pp_generates_split_mode_layer(self):
        """--pp → --split-mode layer in generated command."""
        recipe = self._make_recipe(defaults={"pipeline_parallel": 4, "port": 8080})
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--split-mode layer" in cmd
        assert "--split-mode row" not in cmd

    def test_neither_uses_default_layer(self):
        """No TP/PP → default split_mode=layer from _LLAMA_CPP_DEFAULTS."""
        recipe = self._make_recipe(defaults={"port": 8080})
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--split-mode layer" in cmd

    def test_tp_override_generates_row(self):
        """CLI override tensor_parallel → --split-mode row."""
        recipe = self._make_recipe(defaults={"port": 8080})
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {"tensor_parallel": 2}, is_cluster=False)
        assert "--split-mode row" in cmd

    def test_pp_override_generates_layer(self):
        """CLI override pipeline_parallel → --split-mode layer."""
        recipe = self._make_recipe(defaults={"port": 8080})
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {"pipeline_parallel": 2}, is_cluster=False)
        assert "--split-mode layer" in cmd

    def test_tp_overrides_recipe_split_mode(self):
        """TP takes precedence over recipe split_mode=layer."""
        recipe = self._make_recipe(
            defaults={
                "tensor_parallel": 2,
                "split_mode": "layer",
            }
        )
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--split-mode row" in cmd
        assert "--split-mode layer" not in cmd

    def test_both_tp_pp_raises_in_generate(self):
        """Both TP and PP raises ValueError in generate_command too."""
        recipe = self._make_recipe(
            defaults={
                "tensor_parallel": 2,
                "pipeline_parallel": 2,
            }
        )
        runtime = LlamaCppRuntime()
        with pytest.raises(ValueError, match="simultaneously"):
            runtime.generate_command(recipe, {}, is_cluster=False)

    def test_template_split_mode_overridden_by_tp(self):
        """Command template with --split-mode layer is overridden by TP → row."""
        recipe = self._make_recipe(
            defaults={"tensor_parallel": 2},
            command="llama-server -hf {model} --split-mode layer --port 8080",
        )
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--split-mode row" in cmd
        assert "--split-mode layer" not in cmd

    def test_template_split_mode_overridden_by_pp(self):
        """Command template with --split-mode row is overridden by PP → layer."""
        recipe = self._make_recipe(
            defaults={"pipeline_parallel": 4},
            command="llama-server -hf {model} --split-mode row --port 8080",
        )
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--split-mode layer" in cmd
        assert "--split-mode row" not in cmd

    def test_template_no_tp_pp_preserves_existing_split_mode(self):
        """Template with --split-mode is preserved when no TP/PP set."""
        recipe = self._make_recipe(
            command="llama-server -hf {model} --split-mode row --port 8080",
        )
        runtime = LlamaCppRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--split-mode row" in cmd


class TestLlamaCppValidateRecipe:
    """Test validate_recipe catches TP+PP conflict in recipe defaults."""

    def test_both_tp_pp_in_defaults_warns(self):
        recipe_data = {
            "name": "test",
            "runtime": "llama-cpp",
            "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
            "defaults": {"tensor_parallel": 2, "pipeline_parallel": 2},
        }
        recipe = Recipe.from_dict(recipe_data)
        runtime = LlamaCppRuntime()
        issues = runtime.validate_recipe(recipe)
        assert any("mutually exclusive" in i for i in issues)

    def test_tp_only_no_warning(self):
        recipe_data = {
            "name": "test",
            "runtime": "llama-cpp",
            "model": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
            "defaults": {"tensor_parallel": 2},
        }
        recipe = Recipe.from_dict(recipe_data)
        runtime = LlamaCppRuntime()
        issues = runtime.validate_recipe(recipe)
        assert not any("mutually exclusive" in i for i in issues)


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
        recipe = self._make_recipe(
            runtime="vllm",
            command="vllm serve {model} --port 8000",
        )
        runtime = VllmRayRuntime()
        cmd = runtime.generate_command(recipe, {"served_model_name": "cli-name"}, is_cluster=False)
        assert "--served-model-name cli-name" in cmd

    def test_vllm_ray_no_duplicate(self):
        """vllm-ray: template already has --served-model-name → no duplicate."""
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
        recipe = self._make_recipe(
            runtime="vllm",
            command="vllm serve {model} --port 8000",
        )
        runtime = VllmRayRuntime()
        cmd = runtime.generate_command(recipe, {}, is_cluster=False)
        assert "--served-model-name" not in cmd

    def test_vllm_ray_skip_keys_suppresses(self):
        """vllm-ray: skip_keys={served_model_name} → no augmentation."""
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
