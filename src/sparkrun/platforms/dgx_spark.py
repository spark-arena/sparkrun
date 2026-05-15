"""DGX Spark hardware platform.

Identifies NVIDIA GB10 hosts and publishes the Spark Arena container
defaults that have been validated for the GB10 RoCEv2 fabric.
"""

from __future__ import annotations

from sparkrun.core.hardware import HostHardware
from sparkrun.orchestration.collectives import CollectiveBackend, NcclBackend
from sparkrun.platforms.base import HardwarePlatformPlugin


# Per-runtime defaults curated for GB10 / Spark Arena.  ``None`` means
# "no default image — recipe.container must be set explicitly".
_DGX_SPARK_DEFAULTS: dict[str, str | None] = {
    "vllm-distributed": "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:latest",
    "vllm-ray": "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:latest",
    "sglang": "scitrera/dgx-spark-sglang:latest",
    "llama-cpp": "scitrera/dgx-spark-llama-cpp:latest",
    "trtllm": "nvcr.io/nvidia/tensorrt-llm/release:latest",
    "atlas": "avarok/atlas-gb10:latest",
}


class DgxSparkPlatform(HardwarePlatformPlugin):
    """NVIDIA DGX Spark (GB10, ConnectX-7 RoCEv2 fabric)."""

    platform_name = "dgx-spark"
    vendors = frozenset({"nvidia"})

    def matches(self, host_hardware: HostHardware) -> bool:
        return any(a.vendor == "nvidia" and a.model == "gb10" for a in host_hardware.accelerators)

    def accelerator_vendor(self) -> str:
        return "nvidia"

    def collective_backend(self) -> CollectiveBackend:
        return NcclBackend()

    def default_image(self, runtime_name: str) -> str | None:
        return _DGX_SPARK_DEFAULTS.get(runtime_name)
