"""Generic NVIDIA hardware platform.

Catch-all for any NVIDIA host that doesn't match a more specific
platform (e.g. DGX Spark).  Publishes upstream images where they exist
and ``None`` elsewhere so the recipe must declare ``container:``
explicitly for niche runtimes.
"""

from __future__ import annotations

from sparkrun.core.hardware import HostHardware
from sparkrun.orchestration.collectives import CollectiveBackend, NcclBackend
from sparkrun.platforms.base import HardwarePlatformPlugin


# Upstream defaults for generic NVIDIA hosts (H100, H200, B200, RTX
# workstations, etc.).  ``None`` means "no default — set
# recipe.container explicitly".  Atlas / eugr stay DGX-Spark-only via
# their own ``requires_capability`` gates on the runtime plugin.
_NVIDIA_GENERIC_DEFAULTS: dict[str, str | None] = {
    "vllm-distributed": "vllm/vllm-openai:latest",
    "vllm-ray": "vllm/vllm-openai:latest",
    "sglang": "lmsysorg/sglang:latest",
    "llama-cpp": "ghcr.io/ggerganov/llama.cpp:server-cuda",
    "trtllm": "nvcr.io/nvidia/tensorrt-llm/release:latest",
}


class GenericNvidiaPlatform(HardwarePlatformPlugin):
    """Any NVIDIA host (H100/H200/B200, RTX workstations, generic CUDA boxes)."""

    platform_name = "nvidia-generic"
    vendors = frozenset({"nvidia"})

    def matches(self, host_hardware: HostHardware) -> bool:
        return any(a.vendor == "nvidia" for a in host_hardware.accelerators)

    def accelerator_vendor(self) -> str:
        return "nvidia"

    def collective_backend(self) -> CollectiveBackend:
        return NcclBackend()

    def default_image(self, runtime_name: str) -> str | None:
        return _NVIDIA_GENERIC_DEFAULTS.get(runtime_name)
