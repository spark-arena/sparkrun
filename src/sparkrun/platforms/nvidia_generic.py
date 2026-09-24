"""Generic NVIDIA hardware platform.

Catch-all for any NVIDIA host that doesn't match a more specific
platform (e.g. DGX Spark).  Publishes upstream images where they exist
and ``None`` elsewhere so the recipe must declare ``container:``
explicitly for niche runtimes.
"""

from __future__ import annotations

from sparkrun.core.hardware import AcceleratorSpec, HostHardware, normalize_compute_capability
from sparkrun.orchestration.collectives import CollectiveBackend, NcclBackend
from sparkrun.platforms.base import HardwarePlatformPlugin
from sparkrun.core.setup_plans import SetupPlan


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
    "modular-max": "modular/max-nvidia-full:latest",
}


# Declared compute capability by normalized model slug
# (``fingerprint.normalize_nvidia_model``: ``"NVIDIA H100 80GB HBM3"`` →
# ``"h100-80gb-hbm3"``). A key matches the slug exactly or as a leading
# ``-``-delimited prefix, so memory / form-factor suffixes need no entries of
# their own. Only models whose capability is fixed across every SKU sharing the
# prefix belong here; anything else stays ``None`` and falls back to inventory.
_NVIDIA_COMPUTE_CAPABILITY: dict[str, str] = {
    "a100": "8.0",
    "l4": "8.9",
    "l40s": "8.9",
    "h100": "9.0",
    "h200": "9.0",
    "gh200": "9.0",
    "b200": "10.0",
    "gb200": "10.0",
    "b300": "10.3",
    "gb300": "10.3",
    # "RTX PRO" is Blackwell-only branding (the Ada part is "RTX 6000 Ada
    # Generation" → "rtx-6000-ada-generation", which does not match).
    "rtx-pro-6000": "12.0",
    "rtx-pro-5000": "12.0",
    "rtx-pro-4500": "12.0",
    "rtx-pro-4000": "12.0",
    "geforce-rtx-5090": "12.0",
}


def _declared_compute_capability(model: str) -> str | None:
    """Longest table key that is *model* or a ``-``-delimited prefix of it."""
    best: tuple[int, str] | None = None
    for key, capability in _NVIDIA_COMPUTE_CAPABILITY.items():
        if model == key or model.startswith(key + "-"):
            if best is None or len(key) > best[0]:
                best = (len(key), capability)
    return best[1] if best is not None else None


# Runtime families whose images JIT CuTe DSL kernels (FlashInfer, B12X).
_CUTE_DSL_RUNTIME_FAMILIES = frozenset({"vllm", "sglang"})


def _cute_dsl_arch(capability: str | None) -> str | None:
    """Arch-specific CuTe DSL target (``"12.0"`` → ``"sm_120a"``).

    ``None`` below 9.0: the ``a`` feature sets start at ``sm_90a``, so a lower
    capability has no such target, and naming one would be wrong, not merely
    unused.
    """
    canonical = normalize_compute_capability(capability)
    if canonical is None:
        return None
    major, minor = canonical.split(".")
    if int(major) < 9:
        return None
    return "sm_%s%sa" % (major, minor)


class GenericNvidiaPlatform(HardwarePlatformPlugin):
    """Any NVIDIA host (H100/H200/B200, RTX workstations, generic CUDA boxes)."""

    platform_name = "nvidia-generic"
    display_name = "Generic NVIDIA"
    vendors = frozenset({"nvidia"})

    # Vendor recognition alone does not qualify driver/CDI, OS tuning, or
    # fabric changes. A more specific platform can opt into those steps.
    setup_plans = (
        SetupPlan("docker", ("docker", "docker_group", "host_ipc", "ssh_mesh")),
        SetupPlan("local", ("host_ipc", "ssh_mesh")),
    )

    def matches(self, host_hardware: HostHardware) -> bool:
        return any(a.vendor == "nvidia" for a in host_hardware.accelerators)

    def accelerator_vendor(self) -> str:
        return "nvidia"

    def collective_backend(self) -> CollectiveBackend:
        return NcclBackend()

    def default_image(self, runtime_name: str) -> str | None:
        return _NVIDIA_GENERIC_DEFAULTS.get(runtime_name)

    def default_compute_capability(self, accelerator: AcceleratorSpec) -> str | None:
        if accelerator.vendor != "nvidia":
            return None
        return _declared_compute_capability(accelerator.model)

    def default_env(
        self,
        runtime_name: str,
        accelerator: AcceleratorSpec,
        *,
        runtime_family: str | None = None,
    ) -> dict[str, str]:
        """``CUTE_DSL_ARCH`` for CuTe-DSL runtimes when the capability is known.

        Declared capability first, then inventory, which is the
        :func:`~sparkrun.platforms.resolve_compute_capability` order. An image
        that never JITs CuTe kernels ignores the variable.
        """
        if accelerator.vendor != "nvidia" or not ({runtime_name, runtime_family} & _CUTE_DSL_RUNTIME_FAMILIES):
            return {}
        arch = _cute_dsl_arch(self.default_compute_capability(accelerator) or accelerator.compute_capability)
        return {"CUTE_DSL_ARCH": arch} if arch else {}

    def validate_host(self, host_hardware: HostHardware) -> list[str]:
        """Validate that a host carries at least one NVIDIA accelerator.

        The generic platform is intentionally lenient — it accepts any NVIDIA
        GPU, so the only check worth making is confirming the vendor is
        actually NVIDIA (guards against :meth:`matches` being called on a
        host that slipped through without accelerator metadata).

        Returns a list of human-readable warning strings; empty means healthy.
        """
        warnings: list[str] = []

        nvidia_accels = [a for a in host_hardware.accelerators if a.vendor == "nvidia"]
        if not nvidia_accels:
            non_nvidia = sorted({a.vendor for a in host_hardware.accelerators})
            warnings.append(
                "GenericNvidiaPlatform selected but no NVIDIA accelerator found — "
                "detected vendor(s): %s" % (", ".join(non_nvidia) if non_nvidia else "none")
            )

        warnings.extend(self.compute_capability_warnings(host_hardware))
        return warnings
