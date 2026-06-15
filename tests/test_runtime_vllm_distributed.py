"""Unit tests for sparkrun.runtimes.vllm_distributed (VllmDistributedRuntime)."""

from unittest import mock

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime


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


def test_vllm_distributed_command_literal_ray_overridden_to_mp():
    """Legacy command hardcodes ray; -o distributed_executor_backend=mp rewrites it."""
    recipe_data = {
        "name": "legacy",
        "model": "org/model",
        "runtime": "vllm-distributed",
        "command": "vllm serve {model} -tp 2 --distributed-executor-backend ray",
    }
    recipe = Recipe.from_dict(recipe_data)
    runtime = VllmDistributedRuntime()

    cmd = runtime.generate_command(recipe, {"distributed_executor_backend": "mp"}, is_cluster=True, num_nodes=2, head_ip="10.0.0.1")
    assert "--distributed-executor-backend mp" in cmd
    assert "--distributed-executor-backend ray" not in cmd
    assert cmd.count("--distributed-executor-backend") == 1

    # No override: the literal command value is left untouched.
    cmd_none = runtime.generate_command(recipe, {}, is_cluster=True, num_nodes=2, head_ip="10.0.0.1")
    assert "--distributed-executor-backend ray" in cmd_none

    # The node command honors the override too.
    node_cmd = runtime.generate_node_command(recipe, {"distributed_executor_backend": "mp"}, head_ip="10.0.0.1", num_nodes=2, node_rank=1)
    assert "--distributed-executor-backend mp" in node_cmd
    assert "--distributed-executor-backend ray" not in node_cmd


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
