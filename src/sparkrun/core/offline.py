"""Offline launches: use only what already exists, with no internet egress.

**Offline means no internet egress.** Images resident on the hosts, models in
their HuggingFace caches and recipes in the local registry clones are used as
they are. Copies inside the cluster are still allowed, but only of copies that
already exist: an image on the control machine or head may be ``docker save |
docker load``-ed to a worker, a cached model rsynced. Nothing is pulled,
downloaded, cloned, fetched or built.

It is always *chosen*, never inferred: switching modes because the network
looked down would turn a transient outage into a launch that behaves
differently. And it is not workload identity: it changes where bytes come
from, not what runs, so it stays out of the intent id and the fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OfflineMode:
    """Whether a launch is offline, and which layer decided it."""

    offline: bool
    source: str
    """``"cli"``, ``"cluster '<name>'"`` or ``"default"``."""

    def describe(self) -> str:
        """Banner text, e.g. ``yes (cluster 'lab')``."""
        return "%s (%s)" % ("yes" if self.offline else "no", self.source)


def resolve_offline(cli: bool | None, cluster=None) -> OfflineMode:
    """CLI (``--offline`` / ``--online``) > the cluster's ``offline:`` > online."""
    if cli is not None:
        return OfflineMode(bool(cli), "cli")
    value = getattr(cluster, "offline", None)
    if value is not None:
        return OfflineMode(bool(value), "cluster '%s'" % getattr(cluster, "name", "?"))
    return OfflineMode(False, "default")


# ---------------------------------------------------------------------------
# Preflight: is everything this launch needs already here?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OfflineGap:
    """One thing an offline launch needs that no allowed source has."""

    kind: str
    """``"image"`` or ``"model"``."""
    ref: str
    hosts: tuple[str, ...]
    """Hosts that lack it and cannot be supplied from a copy source."""
    checked: str
    """Which copy sources were checked, for the message."""


class OfflineUnavailableError(RuntimeError):
    """An offline launch would have to fetch something. Raised before any side effect."""

    def __init__(self, gaps: list[OfflineGap], *, reason: str | None = None):
        self.gaps = gaps
        if reason is not None:
            super().__init__("Offline launch cannot proceed: %s" % reason)
            return
        lines = ["Offline launch cannot proceed; not present anywhere it could be copied from:"]
        for gap in gaps:
            lines.append("  %s %s: missing on %s (checked: %s)" % (gap.kind, gap.ref, ", ".join(gap.hosts), gap.checked))
        lines.append(
            "Fix: run once online (or with --online), sync from a connected machine (`sparkrun setup sync`), "
            "or pre-place the weights (cluster cache_dir)."
        )
        super().__init__("\n".join(lines))


def copy_sources(transfer_mode: str, *, auto_delegated: bool, heterogeneous: bool = False) -> tuple[str, ...]:
    """Where distribution copies from in *transfer_mode*: ``"control"`` and/or ``"head"``.

    Mirrors ``orchestration.distribution``: ``local`` / ``push`` send the
    control machine's copy; ``delegated`` fans out from the head; ``pull``
    copies from nowhere. An inferred mode (auto → delegated) falls back to the
    control machine, and a heterogeneous delegated launch is distributed as a
    per-node pull, never through the head.
    """
    if transfer_mode in ("local", "push"):
        return ("control",)
    if transfer_mode == "delegated" and not heterogeneous:
        return ("head", "control") if auto_delegated else ("head",)
    return ("control",) if auto_delegated else ()


def _covered(missing: list[str], sources: tuple[str, ...], *, head: str, control_has: bool, present_on: set[str]) -> bool:
    if not missing:
        return True
    if "control" in sources and control_has:
        return True
    return "head" in sources and head in present_on


def find_offline_gaps(
    *,
    images: dict[str, list[str]],
    model: str | None,
    model_hosts: list[str],
    head: str,
    transfer_mode: str,
    auto_delegated: bool,
    heterogeneous: bool = False,
    image_present_on,
    image_on_control,
    model_present_on,
    model_on_control,
) -> list[OfflineGap]:
    """Every image or model the launch would have to fetch.

    *images* maps each distinct image to the hosts that run it. The four
    callables are presence probes (injected so the rules are testable without
    SSH): ``image_present_on(image, hosts) -> set``, ``image_on_control(image)
    -> bool``, and the model equivalents. A host missing something is fine
    when a copy source for this transfer mode has it; otherwise it is a gap.
    """

    def _checked(sources: tuple[str, ...]) -> str:
        names = {"control": "this machine", "head": "the head (%s)" % head}
        return ", ".join(names[s] for s in sources) or "none: this transfer mode copies nothing"

    gaps: list[OfflineGap] = []
    # Images follow the heterogeneous redirect (per-node pull); models never do.
    image_sources = copy_sources(transfer_mode, auto_delegated=auto_delegated, heterogeneous=heterogeneous)
    for image, targets in images.items():
        probe_hosts = sorted(set(targets) | ({head} if "head" in image_sources else set()))
        present = image_present_on(image, probe_hosts)
        missing = [h for h in targets if h not in present]
        control_has = bool(missing) and "control" in image_sources and image_on_control(image)
        if not _covered(missing, image_sources, head=head, control_has=control_has, present_on=present):
            gaps.append(OfflineGap("image", image, tuple(missing), _checked(image_sources)))
    if model and model_hosts:
        model_sources = copy_sources(transfer_mode, auto_delegated=auto_delegated)
        probe_hosts = sorted(set(model_hosts) | ({head} if "head" in model_sources else set()))
        present = model_present_on(probe_hosts)
        missing = [h for h in model_hosts if h not in present]
        control_has = bool(missing) and "control" in model_sources and model_on_control()
        if not _covered(missing, model_sources, head=head, control_has=control_has, present_on=present):
            gaps.append(OfflineGap("model", model, tuple(missing), _checked(model_sources)))
    return gaps
