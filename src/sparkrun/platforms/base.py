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

from sparkrun.core.hardware import HostHardware
from sparkrun.orchestration.collectives import CollectiveBackend

logger = logging.getLogger(__name__)

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

    def default_image(self, runtime_name: str) -> str | None:
        """Default container image for *runtime_name* on this platform.

        Base implementation returns ``None``; subclasses override to
        publish per-runtime defaults (e.g. Spark Arena images for
        DGX Spark, upstream vLLM/SGLang images for generic NVIDIA).
        """
        return None

    def validate_host(self, host_hardware: HostHardware) -> list[str]:
        """Return a list of warning strings about this host's hardware.

        Empty list means the host looks healthy for this platform.  Non-empty
        means there are concerns the user should be aware of (e.g. missing
        RoCEv2 capability on DGX Spark, mismatched accelerator family).

        Default implementation returns an empty list — subclasses may
        override to add platform-specific validation.
        """
        return []
