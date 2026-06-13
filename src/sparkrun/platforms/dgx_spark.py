"""DGX Spark hardware platform.

Identifies NVIDIA GB10 hosts and publishes the Spark Arena container
defaults that have been validated for the GB10 RoCEv2 fabric.
"""

from __future__ import annotations

from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.orchestration.collectives import CollectiveBackend, NcclBackend
from sparkrun.platforms.base import HardwarePlatformPlugin


# Per-runtime defaults curated for GB10 / Spark Arena.  ``None`` means
# "no default image — recipe.container must be set explicitly".
_DGX_SPARK_DEFAULTS: dict[str, str | None] = {
    "vllm-distributed": "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest",
    "vllm-ray": "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest",
    "sglang": "scitrera/dgx-spark-sglang:latest",
    "llama-cpp": "scitrera/dgx-spark-llama-cpp:latest",
    "trtllm": "nvcr.io/nvidia/tensorrt-llm/release:latest",
    "atlas": "avarok/atlas-gb10:latest",
}


# GB10 is a unified-memory system: the 121 GB "available for inference" figure
# is shared with the CPU/OS and runtime overhead, so scheduling/fit should not
# assume the full amount is usable.  0.85 leaves headroom; users can override
# per-cluster.  See sparkrun.core.limits.
_DGX_SPARK_MAX_GPU_MEMORY_UTILIZATION = 0.85


# Per-runtime recipe-flag defaults for GB10.  Applied at the recipe-default
# tier (only when the recipe/CLI are silent).  Memory-mapped GGUF loading
# performs poorly on GB10's unified memory, so llama.cpp defaults to
# ``--no-mmap`` (mmap off) unless a recipe opts back in with ``mmap: true``.
_DGX_SPARK_RUNTIME_FLAGS: dict[str, dict[str, object]] = {
    "llama-cpp": {"mmap": False},
}


class DgxSparkPlatform(HardwarePlatformPlugin):
    """NVIDIA DGX Spark (GB10, ConnectX-7 RoCEv2 fabric)."""

    platform_name = "dgx-spark"
    display_name = "DGX Spark"
    vendors = frozenset({"nvidia"})

    def matches(self, host_hardware: HostHardware) -> bool:
        return any(a.vendor == "nvidia" and a.model == "gb10" for a in host_hardware.accelerators)

    def accelerator_vendor(self) -> str:
        return "nvidia"

    def collective_backend(self) -> CollectiveBackend:
        return NcclBackend()

    def default_image(self, runtime_name: str) -> str | None:
        return _DGX_SPARK_DEFAULTS.get(runtime_name)

    def default_runtime_flags(self, runtime_name: str, accelerator: AcceleratorSpec) -> dict[str, object]:
        """GB10 recipe-flag defaults (e.g. ``mmap: False`` for llama.cpp)."""
        if accelerator.vendor == "nvidia" and accelerator.model == "gb10":
            return dict(_DGX_SPARK_RUNTIME_FLAGS.get(runtime_name, {}))
        return {}

    def default_max_gpu_memory_utilization(self, accelerator: AcceleratorSpec) -> float | None:
        """GB10 unified memory → cap usable memory at 0.85 for scheduling/fit."""
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
          accelerator — warns when missing because multi-node collectives over
          the ConnectX-7 fabric require RoCEv2.

        Returns a list of human-readable warning strings; empty means healthy.
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

        # Check for RoCEv2 on at least one GB10 entry
        has_roce = any("rdma:roce-v2" in a.capabilities for a in gb10_accels)
        if not has_roce:
            warnings.append(
                "DGX Spark GB10 accelerator is missing 'rdma:roce-v2' capability — "
                "multi-node collective communication over ConnectX-7 fabric may fail"
            )

        return warnings
