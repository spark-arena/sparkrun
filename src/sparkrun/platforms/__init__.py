"""Hardware platform plugins.

A :class:`HardwarePlatformPlugin` binds an accelerator vendor to a
collective backend, executor accelerator flag, and per-runtime default
container images.  Built-in platforms are ordered most-specific first
(``DgxSparkPlatform`` before ``GenericNvidiaPlatform``) so
:func:`resolve_platform` matches the DGX path even though both claim
NVIDIA hosts.

Why an in-process ordered registry instead of SAF discovery
------------------------------------------------------------
Platforms (and likewise the :mod:`sparkrun.orchestration.collectives`
backends) deliberately use the ordered ``_REGISTRY`` list below rather
than SAF extension-point discovery.  Two properties make that the right
call today: (1) **resolution is order-sensitive** — multiple platforms
legitimately claim the same host (``DgxSparkPlatform`` and
``GenericNvidiaPlatform`` both ``matches()`` a GB10 box), and
``resolve_platform`` must return the *most-specific* match, which a
deterministic registration order expresses directly but SAF's
unordered multi-extension set does not; and (2) the set is **tiny and
closed** (two built-ins plus the occasional external ``register_platform``
call), so the discovery / lazy-init machinery SAF brings would add
indirection without buying anything.  ``register_platform(prepend=...)``
gives external packages explicit control over where they sit in the
match order — the semantics callers actually need.

``EXT_PLATFORM`` (defined in :mod:`sparkrun.platforms.base` and
re-exported here) is **reserved for future SAF entry-point discovery**.
:class:`HardwarePlatformPlugin` already implements the SAF ``Plugin``
hooks (``extension_point_name`` returns ``EXT_PLATFORM``,
``is_multi_extension`` is True) so that wiring it into
``core.bootstrap`` later is a drop-in change.  Until then nothing scans
that extension point: the ordered registry above is the single source
of truth for platform resolution.
"""

from __future__ import annotations

import logging

from sparkrun.core.hardware import HostHardware
from sparkrun.platforms.base import EXT_PLATFORM, HardwarePlatformPlugin
from sparkrun.platforms.dgx_spark import DgxSparkPlatform
from sparkrun.platforms.nvidia_generic import GenericNvidiaPlatform

logger = logging.getLogger(__name__)


# Ordered most-specific first: DGX Spark wins over generic NVIDIA on
# GB10 hosts because both call ``matches()`` True and the first wins.
_REGISTRY: list[HardwarePlatformPlugin] = [
    DgxSparkPlatform(),
    GenericNvidiaPlatform(),
]


def register_platform(platform: HardwarePlatformPlugin, *, prepend: bool = False) -> None:
    """Register a platform plugin instance.

    Args:
        platform: An instantiated :class:`HardwarePlatformPlugin`.
        prepend: When ``True``, insert at the front so this platform
            wins ties.  Default appends, so the built-in NVIDIA
            platforms keep their specificity ordering.
    """
    if prepend:
        _REGISTRY.insert(0, platform)
    else:
        _REGISTRY.append(platform)
    logger.debug("Registered platform %r (prepend=%s)", platform.platform_name, prepend)


def iter_platforms() -> list[HardwarePlatformPlugin]:
    """Return the registered platforms in resolution order (fresh list)."""
    return list(_REGISTRY)


def get_platform_by_name(platform_name: str) -> HardwarePlatformPlugin | None:
    """Look up a platform by its ``platform_name`` (e.g. ``"dgx-spark"``)."""
    for p in _REGISTRY:
        if p.platform_name == platform_name:
            return p
    return None


def resolve_platform(host_hardware: HostHardware) -> HardwarePlatformPlugin | None:
    """Return the first platform that claims *host_hardware*, or ``None``.

    Order is determined by registration order — built-ins are
    registered most-specific first so the DGX Spark plugin pre-empts
    the generic NVIDIA plugin on GB10 hosts.
    """
    for p in _REGISTRY:
        try:
            if p.matches(host_hardware):
                return p
        except Exception as e:
            logger.warning("Platform %r matcher raised: %s", p.platform_name, e)
    return None


__all__ = [
    "EXT_PLATFORM",
    "DgxSparkPlatform",
    "GenericNvidiaPlatform",
    "HardwarePlatformPlugin",
    "get_platform_by_name",
    "iter_platforms",
    "register_platform",
    "resolve_platform",
]
