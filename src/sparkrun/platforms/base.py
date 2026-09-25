"""Hardware platform plugin abstraction.

A :class:`HardwarePlatformPlugin` binds an accelerator vendor to a
concrete :class:`CollectiveBackend`, an executor accelerator flag, and
a per-runtime default image.  Recipes and clusters can identify a
platform by name; the registry in :mod:`sparkrun.platforms` resolves
the right plugin from per-host :class:`HostHardware`.

Methods delegate straight to the underlying primitives
(``accelerator_vendor_for``, ``get_backend``).  Keeping the surface
this thin means platform-aware features (entry-point discovery,
per-host executor construction, platform-aware container selection)
have a single integration point to grow from.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from logging import Logger

from scitrera_app_framework import Plugin, Variables

from sparkrun.core.hardware import AcceleratorSpec, HostHardware, normalize_compute_capability
from sparkrun.core.setup_plans import SetupPlan
from sparkrun.orchestration.collectives import CollectiveBackend

logger = logging.getLogger(__name__)

# Reserved for future SAF entry-point discovery of platform plugins.
# Today platform resolution uses the ordered in-process registry in
# ``sparkrun.platforms.__init__`` (see that module's docstring for why);
# nothing scans this extension point yet.  The ``Plugin`` hooks below
# wire it up so flipping to SAF discovery later is a drop-in change.
EXT_PLATFORM = "sparkrun.platform"


class HardwarePlatformPlugin(Plugin):
    """Abstract hardware platform plugin.

    Mirrors :class:`~sparkrun.runtimes.base.RuntimePlugin`'s SAF
    lifecycle: lazy initialisation, multi-extension registration.
    """

    eager = False

    # --- Subclass must define ---
    platform_name: str = ""
    """Stable identifier, e.g. ``"dgx-spark"``, ``"nvidia-generic"``."""

    display_name: str = "Unknown"
    """Human-readable platform name shown in ``sparkrun run`` output (e.g. ``"DGX Spark"``)."""

    vendors: frozenset[str] = frozenset()
    """Accelerator vendors this platform serves (``"nvidia"`` / ``"amd"`` / ``"intel"``)."""

    setup_plans: tuple[SetupPlan, ...] = ()
    """Explicit setup steps per executor. Empty means discovery only."""

    # --- SAF Plugin interface ---

    def name(self) -> str:
        return "sparkrun.platform.%s" % self.platform_name

    def extension_point_name(self, v: Variables) -> str:
        return EXT_PLATFORM

    def is_enabled(self, v: Variables) -> bool:
        # Mirrors RuntimePlugin: returning False prevents SAF's
        # single-extension cache from short-circuiting subsequent
        # registrations under the same extension point.
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True

    def initialize(self, v: Variables, logger_arg: Logger) -> "HardwarePlatformPlugin":
        return self

    # --- Platform interface ---

    @abstractmethod
    def matches(self, host_hardware: HostHardware) -> bool:
        """True iff this platform claims *host_hardware*.

        Implementations typically inspect accelerator vendor + model
        (e.g. ``DgxSparkPlatform`` matches ``model="gb10"``,
        ``GenericNvidiaPlatform`` matches any NVIDIA accelerator).
        """
        ...

    @abstractmethod
    def accelerator_vendor(self) -> str:
        """Vendor string for :class:`~sparkrun.orchestration.executor.ExecutorConfig`."""
        ...

    @abstractmethod
    def collective_backend(self) -> CollectiveBackend:
        """Collective backend (NCCL/RCCL/HCCL) appropriate for this platform."""
        ...

    def assumed_hardware(self) -> HostHardware | None:
        """Optional legacy inventory assumption, used only by application policy.

        A platform name does not generally identify capacity or device count.
        Leave this unset unless the platform can qualify an explicit assumption;
        do not advertise unobserved network capabilities.
        """
        return None

    def default_image(self, runtime_name: str) -> str | None:
        """Default container image for *runtime_name* on this platform.

        Base implementation returns ``None``; subclasses override to
        publish per-runtime defaults (e.g. Spark Arena images for
        DGX Spark, upstream vLLM/SGLang images for generic NVIDIA).
        """
        return None

    def default_executor_config(self, executor_name: str) -> dict[str, object]:
        """Platform-specific executor-config defaults for *executor_name*.

        Returns a mapping of :class:`~sparkrun.orchestration.executor.ExecutorConfig`
        keys that should apply on this platform.  Folded into the executor
        resolution chain **below** ``config.executor_config`` and **above** the
        executor's own ``default_config()``, so a recipe / cluster / user config
        always wins while the platform still outranks the generic per-executor
        default.

        This is the platform tier for hardware-conditional container plumbing —
        e.g. DGX Spark returns ``{"gpu_access_mode": "gpus"}`` for ``docker``
        because CDI is fragile on GB10 hosts whose ``/etc/cdi/nvidia.yaml`` goes
        stale across driver upgrades.

        Base implementation returns ``{}`` (no platform-specific defaults).
        """
        return {}

    def default_runtime_flags(self, runtime_name: str, accelerator: AcceleratorSpec) -> dict[str, object]:
        """Platform/runtime/accelerator-specific recipe-flag defaults.

        Returns a mapping of recipe ``defaults`` keys to values that should
        apply on this platform for *runtime_name* + *accelerator* **when the
        recipe (and CLI) do not specify them**.  The launcher folds the result
        in at the recipe-default tier (``setdefault``), so explicit recipe
        defaults and CLI overrides always win.

        This is the platform tier for hardware-conditional runtime tuning —
        e.g. DGX Spark returns ``{"mmap": False}`` for ``llama-cpp`` on GB10
        because memory-mapped GGUF loading performs poorly on its unified
        memory.

        Base implementation returns ``{}`` (no platform-specific defaults).
        """
        return {}

    def default_env(
        self,
        runtime_name: str,
        accelerator: AcceleratorSpec,
        *,
        runtime_family: str | None = None,
    ) -> dict[str, str]:
        """Platform/runtime/accelerator-specific container env defaults.

        The env-var peer of :meth:`default_runtime_flags`: same
        platform × runtime × accelerator targeting, but the result is merged
        into the container environment instead of the recipe's serve flags.
        Use it for hardware tuning that a runtime reads from the environment
        rather than the command line — e.g. DGX Spark sets
        ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` for vLLM/SGLang on
        GB10.

        *runtime_family* is :meth:`~sparkrun.runtimes.base.RuntimePlugin.get_family`
        (``"vllm"`` for ``vllm-ray`` / ``vllm-distributed`` / ``eugr-vllm``), so a
        platform can target a whole family without enumerating every variant.
        Implementations should let an exact *runtime_name* match win over the
        family match.

        The launcher folds the result in as the **lowest** env tier — under the
        cluster's ``env_file``, the recipe's ``env``, and any ``-e`` CLI
        override — so a platform default can always be overridden, including
        with an empty value.

        Base implementation returns ``{}`` (no platform-specific env).
        """
        return {}

    def default_accelerator_memory_gb(self, accelerator: AcceleratorSpec) -> float | None:
        """Known capacity when hardware inventory has no memory measurement.

        Return a value only when this accelerator has a qualified, fixed
        capacity. Explicit inventory values always win. Unknown or variable
        capacity devices keep ``None``; never infer capacity from vendor alone.
        """
        return None

    def default_max_gpu_memory_utilization(self, accelerator: AcceleratorSpec) -> float | None:
        """Default usable-memory cap for *accelerator* on this platform.

        This is the **platform tier** of
        :func:`sparkrun.core.limits.resolve_max_gpu_memory_utilization` — the
        fraction of nominal ``memory_gb`` treated as usable for scheduling /
        fit when neither the accelerator nor the cluster config supplies one.

        Base implementation returns ``None`` (no platform default → the hard
        ``1.0`` fallback applies).  Subclasses override to publish a default
        for their accelerators — e.g. DGX Spark returns ``0.90`` for GB10
        because its unified memory is shared with the CPU/OS.
        """
        return None

    def declared_accelerators(self) -> list[AcceleratorSpec]:
        """Accelerators this platform declares facts for, without any host to look at.

        Lets hardware-free callers ask "what would this platform publish?"
        (``recipe validate`` uses it to notice a recipe restating a platform
        fact). Base: whatever :meth:`assumed_hardware` describes.
        """
        assumed = self.assumed_hardware()
        return list(assumed.accelerators) if assumed is not None else []

    def declared_capabilities(self, accelerator: AcceleratorSpec) -> frozenset[str]:
        """Capability tags this platform knows *accelerator* has, whether or not a probe reported them.

        Detection records what it can see (``cuda``, an RDMA fabric). A fact
        fixed by the model, such as GB10's ``unified-memory``, is the
        platform's to declare, the same rule as
        :meth:`default_compute_capability`. Consumers that match on
        capabilities (recipe ``overrides:``) take the union.
        """
        return frozenset()

    def default_compute_capability(self, accelerator: AcceleratorSpec) -> str | None:
        """Declared ``"<major>.<minor>"`` compute capability for *accelerator*.

        Compute capability is fixed per accelerator model, so the platform that
        recognizes the model is the authority, and its answer needs no probe.
        That keeps ``arch`` known for assumed hardware, under ``--dry-run``,
        and before placement. It **outranks** inventory, which is the reverse
        of :meth:`default_accelerator_memory_gb`: memory is a measurement, while
        a probed compute capability that disagrees with the model means
        misdetection. :meth:`compute_capability_warnings` reports that case.

        Return a value only for models this platform qualifies. Unknown models
        keep ``None``; never infer from the vendor alone.
        """
        return None

    def compute_capability_warnings(self, host_hardware: HostHardware) -> list[str]:
        """Warn where inventory disagrees with this platform's declaration.

        The probe is a cross-check on :meth:`default_compute_capability`, not
        a source. Shared so every platform's ``validate_host`` reports the
        disagreement the same way.
        """
        warnings: list[str] = []
        for accel in host_hardware.accelerators:
            declared = normalize_compute_capability(self.default_compute_capability(accel))
            recorded = normalize_compute_capability(accel.compute_capability)
            if declared is None or recorded is None or declared == recorded:
                continue
            warnings.append(
                "%s declares compute capability %s for %s %s, but the host reports %s — "
                "the host may not be the hardware this platform describes"
                % (self.display_name or self.platform_name, declared, accel.vendor, accel.model, recorded)
            )
        return warnings

    def validate_host(self, host_hardware: HostHardware) -> list[str]:
        """Return a list of warning strings about this host's hardware.

        Empty list means the host looks healthy for this platform.  Non-empty
        means there are concerns the user should be aware of (e.g. missing
        RoCEv2 capability on DGX Spark, mismatched accelerator family).

        Default implementation reports only
        :meth:`compute_capability_warnings`; subclasses that override should
        include it.
        """
        return self.compute_capability_warnings(host_hardware)
