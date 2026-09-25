"""DGX Spark hardware platform.

Identifies NVIDIA GB10 hosts and publishes the Spark Arena container
defaults that have been validated for the GB10 RoCEv2 fabric.
"""

from __future__ import annotations


from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.orchestration.collectives import CollectiveBackend, NcclBackend
from sparkrun.platforms.base import HardwarePlatformPlugin
from sparkrun.core.setup_plans import SetupPlan


# Per-runtime defaults curated for GB10 / Spark Arena.  ``None`` means
# "no default image — recipe.container must be set explicitly".
DGX_SPARK_MEMORY_GB = 121.0
DGX_SPARK_COMPUTE_CAPABILITY = "12.1"
DGX_SPARK_SCHEDULING_FRACTION = 0.90


_DGX_SPARK_DEFAULTS: dict[str, str | None] = {
    "vllm-distributed": "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:latest",
    "vllm-ray": "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:latest",
    "sglang": "scitrera/dgx-spark-sglang:latest",
    "llama-cpp": "scitrera/dgx-spark-llama-cpp:latest",
    "trtllm": "nvcr.io/nvidia/tensorrt-llm/release:latest",
    "modular-max": "modular/max-nvidia-full:latest",
    "atlas": "azeezish/atlas-gb10:latest",
    "tokenary": "scitrera/tokenary:latest",
}


# GB10 is a unified-memory system: the 121 GB "available for inference" figure
# is shared with the CPU/OS and runtime overhead, so scheduling/fit should not
# assume the full amount is usable.  0.90 leaves headroom; users can override
# per-cluster.  See sparkrun.core.limits.
_DGX_SPARK_MAX_GPU_MEMORY_UTILIZATION = DGX_SPARK_SCHEDULING_FRACTION


# ``gpu_memory_utilization`` for vLLM / SGLang on GB10 when the recipe sets
# none. The engines' own defaults (0.9 upstream, 0.92 in the B12X image) are
# discrete-card numbers, and GB10's 121 GiB pool is shared with the OS, so they
# can fail at startup with too little free memory. A recipe, override or CLI
# value always wins: this only fills a gap, it never lowers what a recipe says.
DGX_SPARK_DEFAULT_SERVE_GPU_MEMORY_UTILIZATION = 0.8

_SERVE_MEMORY_DEFAULT = {"gpu_memory_utilization": DGX_SPARK_DEFAULT_SERVE_GPU_MEMORY_UTILIZATION}

# Per-runtime recipe-flag defaults for GB10.  Applied at the recipe-default
# tier (only when the recipe/CLI are silent).  Memory-mapped GGUF loading
# performs poorly on GB10's unified memory, so llama.cpp defaults to
# ``--no-mmap`` (mmap off) unless a recipe opts back in with ``mmap: true``.
_DGX_SPARK_RUNTIME_FLAGS: dict[str, dict[str, object]] = {
    "llama-cpp": {"mmap": False},
    "vllm-distributed": _SERVE_MEMORY_DEFAULT,
    "vllm-ray": _SERVE_MEMORY_DEFAULT,
    "eugr-vllm": _SERVE_MEMORY_DEFAULT,
    "sglang": _SERVE_MEMORY_DEFAULT,
}


# Per-runtime container env defaults for GB10, keyed by runtime **family**
# first (``get_family()``: ``"vllm"`` covers vllm-ray / vllm-distributed /
# eugr-vllm) with exact ``runtime_name`` keys taking precedence.  Applied as the
# lowest env tier, so a cluster / recipe / ``-e`` override always wins.
#
# ``expandable_segments`` lets PyTorch's caching allocator grow a segment in
# place instead of reserving fixed-size blocks, which cuts the fragmentation
# that GB10's unified memory makes expensive — the GPU's pool is shared with the
# CPU/OS, so reserved-but-unused blocks are not free.  vLLM and SGLang are the
# torch-allocator-bound runtimes; llama.cpp / TRT-LLM manage their own memory
# and are deliberately left alone.
_TORCH_EXPANDABLE_SEGMENTS = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

# ``CUTE_DSL_ARCH`` names the target the CuTe DSL (nvidia-cutlass-dsl) JIT
# compiles for. B12X and FlashInfer's CuTe kernels need the arch-specific
# ``sm_121a`` feature set on GB10, and every B12X recipe used to restate it in
# its own ``env:``. It is a fact about the hardware, not the model, so it lives
# here. A recipe that sets it still wins, since this tier is the lowest.
_CUTE_DSL_ARCH = {"CUTE_DSL_ARCH": "sm_121a"}

_DGX_SPARK_RUNTIME_ENV: dict[str, dict[str, str]] = {
    "vllm": {**_TORCH_EXPANDABLE_SEGMENTS, **_CUTE_DSL_ARCH},
    "sglang": {**_TORCH_EXPANDABLE_SEGMENTS, **_CUTE_DSL_ARCH},
}


# Per-executor config defaults for GB10.  DGX Spark pins the docker executor
# back to the classic ``--gpus`` flag: CDI (``--device nvidia.com/gpu=all``) is
# the portable default elsewhere, but on GB10 it depends on an
# ``/etc/cdi/nvidia.yaml`` that goes stale across driver upgrades (the spec pins
# versioned absolute paths), and a stale spec fails ``docker run`` outright.
# ``--gpus`` resolves through the container runtime at launch instead, so it
# survives driver churn.  Override per cluster/recipe/config with
# ``executor_config.gpu_access_mode: cdi``.
_DGX_SPARK_EXECUTOR_CONFIG: dict[str, dict[str, object]] = {
    "docker": {"gpu_access_mode": "gpus"},
}


class DgxSparkPlatform(HardwarePlatformPlugin):
    """NVIDIA DGX Spark (GB10, ConnectX-7 RoCEv2 fabric)."""

    platform_name = "dgx-spark"
    display_name = "DGX Spark"
    vendors = frozenset({"nvidia"})

    setup_plans = (
        SetupPlan(
            "docker",
            ("docker", "docker_group", "nvidia_container", "nvidia_cdi", "host_ipc", "earlyoom", "sudoers", "ssh_mesh", "cx7", "rdma"),
        ),
        SetupPlan("local", ("host_ipc", "earlyoom", "sudoers", "ssh_mesh", "cx7", "rdma")),
    )

    def matches(self, host_hardware: HostHardware) -> bool:
        return any(a.vendor == "nvidia" and a.model == "gb10" for a in host_hardware.accelerators)

    def assumed_hardware(self) -> HostHardware:
        """Legacy application policy, with no inference about attached fabric."""
        return HostHardware(
            accelerators=[
                AcceleratorSpec(
                    vendor="nvidia",
                    model="gb10",
                    memory_gb=DGX_SPARK_MEMORY_GB,
                    capabilities=frozenset({"cuda", "unified-memory"}),
                    memory_capacity_source="assumed platform default",
                )
            ],
            source="assumed",
            notes="assumed by application policy (no hardware inventory)",
        )

    def accelerator_vendor(self) -> str:
        return "nvidia"

    def collective_backend(self) -> CollectiveBackend:
        return NcclBackend()

    def default_image(self, runtime_name: str) -> str | None:
        return _DGX_SPARK_DEFAULTS.get(runtime_name)

    def default_executor_config(self, executor_name: str) -> dict[str, object]:
        """GB10 executor defaults (docker requests GPUs via ``--gpus``)."""
        return dict(_DGX_SPARK_EXECUTOR_CONFIG.get(executor_name, {}))

    def default_runtime_flags(self, runtime_name: str, accelerator: AcceleratorSpec) -> dict[str, object]:
        """GB10 recipe-flag defaults (``mmap: False`` for llama.cpp, serve memory for vLLM/SGLang)."""
        if accelerator.vendor == "nvidia" and accelerator.model == "gb10":
            return dict(_DGX_SPARK_RUNTIME_FLAGS.get(runtime_name, {}))
        return {}

    def default_env(
        self,
        runtime_name: str,
        accelerator: AcceleratorSpec,
        *,
        runtime_family: str | None = None,
    ) -> dict[str, str]:
        """GB10 container env defaults (torch expandable segments for vLLM/SGLang).

        An exact ``runtime_name`` entry wins over the family entry, so a single
        variant can opt out or differ without disturbing the rest of its family.
        """
        if not (accelerator.vendor == "nvidia" and accelerator.model == "gb10"):
            return {}
        for key in (runtime_name, runtime_family):
            if key and key in _DGX_SPARK_RUNTIME_ENV:
                return dict(_DGX_SPARK_RUNTIME_ENV[key])
        return {}

    def default_accelerator_memory_gb(self, accelerator: AcceleratorSpec) -> float | None:
        if accelerator.vendor == "nvidia" and accelerator.model == "gb10":
            return DGX_SPARK_MEMORY_GB
        return None

    def default_compute_capability(self, accelerator: AcceleratorSpec) -> str | None:
        if accelerator.vendor == "nvidia" and accelerator.model == "gb10":
            return DGX_SPARK_COMPUTE_CAPABILITY
        return None

    def declared_capabilities(self, accelerator: AcceleratorSpec) -> frozenset[str]:
        if accelerator.vendor == "nvidia" and accelerator.model == "gb10":
            return frozenset({"unified-memory"})
        return frozenset()

    def default_max_gpu_memory_utilization(self, accelerator: AcceleratorSpec) -> float | None:
        """GB10 unified memory → cap usable memory at 0.90 for scheduling/fit."""
        if accelerator.vendor == "nvidia" and accelerator.model == "gb10":
            return _DGX_SPARK_MAX_GPU_MEMORY_UTILIZATION
        return None

    def validate_host(self, host_hardware: HostHardware) -> list[str]:
        """Validate that a host looks like a healthy DGX Spark (GB10 + RoCEv2).

        Checks performed:

        * At least one accelerator is NVIDIA with model ``"gb10"`` — warns if
          the host matched via :meth:`matches` but carries a different model
          name (unlikely in practice, guards against fingerprint drift).
        * RoCEv2 RDMA capability (``"rdma:roce-v2"``) is present on the GB10
          accelerator — warns when missing from inventory or detection. An
          assumed profile has no fabric evidence and cannot establish absence.

        Returns warning strings for recorded hardware; empty does not prove fabric connectivity.
        """
        warnings: list[str] = []

        gb10_accels = [a for a in host_hardware.accelerators if a.vendor == "nvidia" and a.model == "gb10"]
        if not gb10_accels:
            # matches() returned True but no GB10 found — should not happen in
            # normal usage, but guard against stale fingerprint data.
            nvidia_models = [a.model for a in host_hardware.accelerators if a.vendor == "nvidia"]
            warnings.append(
                "Expected NVIDIA GB10 accelerator but found: %s — hardware may not be a DGX Spark"
                % (", ".join(nvidia_models) if nvidia_models else "none")
            )
            return warnings

        warnings.extend(self.compute_capability_warnings(host_hardware))

        # Assumed hardware intentionally omits fabric capabilities. The caller
        # reports that hardware is unverified; absence here is not detection.
        has_roce = any("rdma:roce-v2" in a.capabilities for a in gb10_accels)
        if not has_roce and host_hardware.source != "assumed":
            warnings.append(
                "DGX Spark GB10 accelerator is missing 'rdma:roce-v2' capability — "
                "multi-node collective communication over ConnectX-7 fabric may fail"
            )

        return warnings
