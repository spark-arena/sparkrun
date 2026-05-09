"""Unit tests for the Atlas runtime plugin."""

from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.atlas import AtlasRuntime


def _recipe(**overrides) -> Recipe:
    base = {
        "name": "test-recipe",
        "model": "Sehyo/Qwen3.5-35B-A3B-NVFP4",
        "runtime": "atlas",
    }
    base.update(overrides)
    return Recipe.from_dict(base)


# --- Identity / container ---


def test_atlas_runtime_name():
    runtime = AtlasRuntime()
    assert runtime.runtime_name == "atlas"
    assert runtime.cluster_strategy() == "native"


def test_atlas_resolve_container_default():
    """No container field → public Docker Hub image."""
    runtime = AtlasRuntime()
    assert runtime.resolve_container(_recipe()) == "avarok/atlas-gb10:latest"


def test_atlas_resolve_container_from_recipe():
    """Recipe container field wins."""
    runtime = AtlasRuntime()
    recipe = _recipe(container="avarok/atlas-gb10:custom-tag")
    assert runtime.resolve_container(recipe) == "avarok/atlas-gb10:custom-tag"


# --- Solo command generation ---


def test_atlas_generate_command_structured():
    """Generates `atlas serve <model>` with mapped flags."""
    runtime = AtlasRuntime()
    recipe = _recipe(
        defaults={
            "port": 8888,
            "max_model_len": 8192,
            "kv_cache_dtype": "nvfp4",
            "gpu_memory_utilization": 0.88,
            "scheduling_policy": "slai",
        },
    )

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd.startswith("atlas serve Sehyo/Qwen3.5-35B-A3B-NVFP4")
    assert "--port 8888" in cmd
    assert "--max-seq-len 8192" in cmd
    assert "--kv-cache-dtype nvfp4" in cmd
    assert "--gpu-memory-utilization 0.88" in cmd
    assert "--scheduling-policy slai" in cmd


def test_atlas_generate_command_bool_flags():
    """Boolean flags are present when truthy and absent when false."""
    runtime = AtlasRuntime()
    recipe = _recipe(
        defaults={
            "speculative": True,
            "enable_prefix_caching": True,
            "high_speed_swap": False,
        },
    )

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--speculative" in cmd
    assert "--enable-prefix-caching" in cmd
    assert "--high-speed-swap" not in cmd


def test_atlas_generate_command_from_template():
    """Recipe with explicit command template renders it verbatim."""
    runtime = AtlasRuntime()
    recipe = _recipe(
        command="atlas serve {model} --port {port}",
        defaults={"port": 9000},
    )

    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert cmd == "atlas serve Sehyo/Qwen3.5-35B-A3B-NVFP4 --port 9000"


def test_atlas_cli_overrides_defaults():
    """CLI overrides take priority over recipe defaults."""
    runtime = AtlasRuntime()
    recipe = _recipe(defaults={"port": 8888})
    cmd = runtime.generate_command(recipe, {"port": 9000}, is_cluster=False)
    assert "--port 9000" in cmd
    assert "--port 8888" not in cmd


# --- Atlas-specific flags ---


def test_atlas_mtp_and_speculative_flags():
    """MTP speculative-decoding flags pass through correctly."""
    runtime = AtlasRuntime()
    recipe = _recipe(
        defaults={
            "speculative": True,
            "mtp_quantization": "nvfp4",
            "num_drafts": 2,
        },
    )
    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--speculative" in cmd
    assert "--mtp-quantization nvfp4" in cmd
    assert "--num-drafts 2" in cmd


def test_atlas_ep_size_first_class():
    """`ep_size` maps to --ep-size (Atlas-specific but standardized in this runtime)."""
    runtime = AtlasRuntime()
    recipe = _recipe(defaults={"ep_size": 2})
    cmd = runtime.generate_command(recipe, {}, is_cluster=False)
    assert "--ep-size 2" in cmd


def test_atlas_skip_served_model_name():
    """`skip_keys` suppresses --model-name (used by benchmark flow)."""
    runtime = AtlasRuntime()
    recipe = _recipe(defaults={"served_model_name": "my-alias", "port": 8888})
    cmd = runtime.generate_command(recipe, {}, is_cluster=False, skip_keys={"served_model_name"})
    assert "--model-name" not in cmd


# --- Cluster command generation ---


def test_atlas_generate_command_cluster_head():
    """Cluster mode head command includes --rank 0 / --master-addr / --master-port."""
    runtime = AtlasRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 1, "ep_size": 2, "port": 8888})

    cmd = runtime.generate_command(recipe, {}, is_cluster=True, num_nodes=2, head_ip="10.0.0.1")
    assert "--rank 0" in cmd
    assert "--world-size 2" in cmd
    assert "--master-addr 10.0.0.1" in cmd
    assert "--master-port 29500" in cmd
    assert "--ep-size 2" in cmd
    assert "--port 8888" in cmd


def test_atlas_generate_node_command_worker_binds_port_zero():
    """Workers must bind --port 0 — only rank 0 exposes the OpenAI API."""
    runtime = AtlasRuntime()
    recipe = _recipe(defaults={"ep_size": 2, "port": 8888})

    head = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=0, init_port=29500)
    worker = runtime.generate_node_command(recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=1, init_port=29500)

    assert "--port 8888" in head
    assert "--rank 0" in head
    assert "--master-addr 10.0.0.1" in head
    assert "--master-port 29500" in head

    assert "--port 0" in worker
    assert "--port 8888" not in worker
    assert "--rank 1" in worker


# --- compute_required_nodes ---


def test_atlas_compute_required_nodes_solo():
    """No parallelism → None (= use whatever hosts the user provided)."""
    runtime = AtlasRuntime()
    assert runtime.compute_required_nodes(_recipe()) is None


def test_atlas_compute_required_nodes_pure_ep():
    """ep_size=2, tp=1 → 2 nodes."""
    runtime = AtlasRuntime()
    recipe = _recipe(defaults={"ep_size": 2})
    assert runtime.compute_required_nodes(recipe) == 2


def test_atlas_compute_required_nodes_overlapping_tp_ep():
    """tp=2, ep=2 → 2 nodes (overlapping groups, not 4)."""
    runtime = AtlasRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2, "ep_size": 2})
    assert runtime.compute_required_nodes(recipe) == 2


def test_atlas_compute_required_nodes_orthogonal_tp_ep():
    """tp=2, ep=4 → 8 nodes (orthogonal mesh)."""
    runtime = AtlasRuntime()
    recipe = _recipe(defaults={"tensor_parallel": 2, "ep_size": 4})
    assert runtime.compute_required_nodes(recipe) == 8


# --- Cluster env / docker opts ---


def test_atlas_cluster_env_includes_rdma():
    """Cluster env mirrors the validated GB10 RoCEv2 NCCL settings."""
    runtime = AtlasRuntime()
    env = runtime.get_cluster_env(head_ip="10.0.0.1", num_nodes=2)
    assert env["NCCL_IB_HCA"] == "rocep1s0f0"
    assert env["NCCL_IB_ROCE_VERSION_NUM"] == "2"
    assert env["NCCL_PROTO"] == "Simple"


def test_atlas_extra_docker_opts_include_infiniband():
    """RDMA + io_uring + IPC_LOCK opts required by Atlas's storage and NCCL paths."""
    runtime = AtlasRuntime()
    opts = runtime.get_extra_docker_opts()
    joined = " ".join(opts)
    assert "/dev/infiniband" in joined
    assert "IPC_LOCK" in joined
    assert "memlock=-1" in joined
    assert "seccomp=unconfined" in joined


# --- validate_recipe ---


def test_atlas_validate_recipe_valid():
    runtime = AtlasRuntime()
    assert runtime.validate_recipe(_recipe()) == []


def test_atlas_validate_recipe_no_model():
    runtime = AtlasRuntime()
    recipe = Recipe.from_dict({"name": "test", "runtime": "atlas"})
    issues = runtime.validate_recipe(recipe)
    assert len(issues) == 1
    assert "model is required" in issues[0]
