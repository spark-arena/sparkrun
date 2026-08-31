"""``sparkrun setup check`` — non-destructive setup-readiness evaluation.

This inspects the *current* state of every host in a cluster against the
things ``sparkrun setup wizard`` would configure and reports gaps with
remediation guidance. It only reads state; it never changes a host.

Design note — future direction
------------------------------
Today each readiness signal is a :class:`SetupCheck` in the ordered
:data:`SETUP_CHECKS` registry: a stable ``key`` and an
``evaluate(state, ctx)`` callable that turns a host's :class:`HostState`
(shallow probe facts + richer signals like CX7 detection) into a
:class:`CheckItem`. This is deliberately the *check half* of a larger
"setup step" abstraction we want to grow into: a registry of ordered,
possibly dependency-driven steps assignable per hardware/platform or per
cluster, each carrying both a ``check`` and an ``apply`` stage so that
``setup check`` (check-only) and ``setup wizard`` (check-then-apply) drive
off one source of truth. For now the ``guidance`` string names the command
that would apply the fix — the eventual ``apply`` stage's stand-in.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

import click

from sparkrun.orchestration.executors.docker import GPU_ACCESS_CDI

from .._common import host_options, json_option

if TYPE_CHECKING:
    from sparkrun.orchestration.networking import CX7HostDetection

logger = logging.getLogger(__name__)


# --- Status levels ----------------------------------------------------------

OK = "ok"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"

_STATUS_LABEL = {OK: "OK", WARN: "WARN", FAIL: "FAIL", SKIP: "SKIP"}
_STATUS_MARK = {OK: "[OK]  ", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[--]  "}


@dataclass
class CheckItem:
    """The outcome of one :class:`SetupCheck` for one host."""

    key: str
    label: str
    status: str
    detail: str = ""
    guidance: str = ""


@dataclass
class CheckContext:
    """Cluster-wide facts a check may need beyond a host's own probe output."""

    cluster_name: str | None
    multi_host: bool

    gpu_access_modes: dict[str, str] = field(default_factory=dict)
    """host -> the ``gpu_access_mode`` a launch on that host would resolve to.

    Empty (or missing a host) when resolution failed; see :meth:`cdi_required`.
    """

    @property
    def cluster_flag(self) -> str:
        """`` --cluster <name>`` suffix for guidance commands (empty if unknown)."""
        return " --cluster %s" % self.cluster_name if self.cluster_name else ""

    def cdi_required(self, host: str) -> bool:
        """Would a launch on *host* actually need a CDI spec?

        Only when the resolved ``gpu_access_mode`` is ``cdi`` — a cluster that
        requests GPUs with ``--gpus`` never reads ``/etc/cdi/nvidia.yaml``, so a
        missing or stale spec is not a gap for it.

        Fails **safe**: an unresolvable mode is treated as requiring CDI, which
        is the historical behavior (a missing spec is a hard failure).
        """
        return self.gpu_access_modes.get(host, GPU_ACCESS_CDI) == GPU_ACCESS_CDI


@dataclass
class HostState:
    """Everything a check needs about one host.

    ``facts`` is the parsed key=value output of ``setup_check.sh``. Richer
    signals that reuse existing sparkrun machinery hang off dedicated fields
    (e.g. ``cx7`` from :func:`detect_cx7_for_hosts`) rather than being
    re-derived from a shallow probe — this is what lets a check vet the
    *effective* config, not just a file's presence.
    """

    host: str
    facts: dict[str, str] = field(default_factory=dict)
    cx7: "CX7HostDetection | None" = None


def _truthy(facts: dict[str, str], key: str) -> bool:
    return facts.get(key, "").strip() == "1"


def _int_fact(facts: dict[str, str], key: str) -> int:
    """Read a numeric probe fact, treating anything unparseable as 0.

    A missing or malformed value means "the probe could not tell us", which
    must read as *no finding* rather than as a finding of zero severity.
    """
    try:
        return int(facts.get(key, "").strip())
    except (TypeError, ValueError):
        return 0


#: Shared remedy for both CDI failure modes (absent spec, stale spec). One
#: string so the manual command can never drift from the wizard step that
#: runs it.
_CDI_REGENERATE = "sparkrun setup wizard%s (NVIDIA CDI step) — or: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"

#: Why a CDI finding is reported at SKIP rather than FAIL/WARN. Shown verbatim
#: in the detail so the reader knows it is a mode question, not a broken host.
_CDI_UNUSED = "this cluster requests GPUs with --gpus (executor_config.gpu_access_mode); regenerate before switching to cdi"


def _as_int(value: str | None) -> int:
    try:
        return int((value or "0").strip())
    except (TypeError, ValueError):
        return 0


# --- Individual checks ------------------------------------------------------
#
# Each takes the host's HostState + the cluster CheckContext and returns a
# CheckItem, or None when the check does not apply to this host (omitted).


def _check_docker_installed(state: HostState, ctx: CheckContext) -> CheckItem:
    if _truthy(state.facts, "CHECK_DOCKER_INSTALLED"):
        return CheckItem("docker_installed", "Docker installed", OK)
    return CheckItem(
        "docker_installed",
        "Docker installed",
        FAIL,
        "docker not found on PATH",
        "Install Docker Engine on this host (https://docs.docker.com/engine/install/)",
    )


def _check_docker_group(state: HostState, ctx: CheckContext) -> CheckItem:
    if _truthy(state.facts, "CHECK_DOCKER_GROUP"):
        return CheckItem("docker_group", "Docker group membership", OK)
    user = state.facts.get("CHECK_USER", "the ssh user")
    return CheckItem(
        "docker_group",
        "Docker group membership",
        WARN,
        "user '%s' is not in the docker group" % user,
        "sparkrun setup docker-group%s" % ctx.cluster_flag,
    )


def _check_docker_usable(state: HostState, ctx: CheckContext) -> CheckItem | None:
    facts = state.facts
    if not _truthy(facts, "CHECK_DOCKER_INSTALLED"):
        return None  # already reported as a failure by docker_installed
    if _truthy(facts, "CHECK_DOCKER_USABLE"):
        return CheckItem("docker_usable", "Docker usable without sudo", OK)
    if _truthy(facts, "CHECK_DOCKER_GROUP"):
        detail = "in the docker group but the daemon is not usable yet"
        guidance = "Log out and back in (or run 'newgrp docker'); ensure dockerd is running"
    else:
        detail = "cannot run docker without sudo"
        guidance = "sparkrun setup docker-group%s, then re-login" % ctx.cluster_flag
    return CheckItem("docker_usable", "Docker usable without sudo", FAIL, detail, guidance)


def _check_nvidia_ctk(state: HostState, ctx: CheckContext) -> CheckItem:
    facts = state.facts
    if not _truthy(facts, "CHECK_GPU_PRESENT"):
        return CheckItem("nvidia_ctk", "NVIDIA Container Toolkit", SKIP, "no NVIDIA GPU detected")
    if _truthy(facts, "CHECK_NVIDIA_CTK"):
        return CheckItem("nvidia_ctk", "NVIDIA Container Toolkit", OK)
    return CheckItem(
        "nvidia_ctk",
        "NVIDIA Container Toolkit",
        FAIL,
        "nvidia-ctk not found",
        "Install the NVIDIA Container Toolkit (provides nvidia-ctk)",
    )


def _check_cdi_spec(state: HostState, ctx: CheckContext) -> CheckItem:
    facts = state.facts
    if not _truthy(facts, "CHECK_GPU_PRESENT"):
        return CheckItem("cdi_spec", "NVIDIA CDI spec (/etc/cdi/nvidia.yaml)", SKIP, "no NVIDIA GPU detected")
    if not _truthy(facts, "CHECK_NVIDIA_CTK"):
        return CheckItem("cdi_spec", "NVIDIA CDI spec (/etc/cdi/nvidia.yaml)", SKIP, "requires nvidia-ctk (see above)")

    # Severity is conditional on how this cluster actually asks for GPUs. With
    # ``gpu_access_mode: gpus`` (the DGX Spark default) nothing reads the CDI
    # spec, so its absence is not a gap — reporting FAIL would send the user to
    # fix something that cannot affect their launches. The finding is still
    # surfaced, at SKIP, because it becomes real the moment they switch modes.
    required = ctx.cdi_required(state.host)

    missing = _int_fact(facts, "CHECK_CDI_PATHS_MISSING")
    checked = _int_fact(facts, "CHECK_CDI_PATHS_CHECKED")
    stale = bool(checked and missing)

    if _truthy(facts, "CHECK_CDI_SPEC"):
        # Present, but possibly stale: the spec pins absolute driver-library
        # and device-node paths, and a driver upgrade moves them. Reported as
        # WARN rather than FAIL because a spec may legitimately reference an
        # optional path, and a false hard-failure on a working host is worse
        # than a prompt to regenerate.
        if not stale:
            return CheckItem("cdi_spec", "NVIDIA CDI spec (/etc/cdi/nvidia.yaml)", OK)
        detail = "%d of %d referenced paths missing — spec looks stale (driver upgraded?)" % (missing, checked)
        if required:
            return CheckItem("cdi_spec", "NVIDIA CDI spec (/etc/cdi/nvidia.yaml)", WARN, detail, _CDI_REGENERATE % ctx.cluster_flag)
        return CheckItem("cdi_spec", "NVIDIA CDI spec (/etc/cdi/nvidia.yaml)", SKIP, "%s — %s" % (detail, _CDI_UNUSED))

    if required:
        return CheckItem(
            "cdi_spec",
            "NVIDIA CDI spec (/etc/cdi/nvidia.yaml)",
            FAIL,
            "/etc/cdi/nvidia.yaml missing or empty",
            _CDI_REGENERATE % ctx.cluster_flag,
        )
    return CheckItem("cdi_spec", "NVIDIA CDI spec (/etc/cdi/nvidia.yaml)", SKIP, "not needed — %s" % _CDI_UNUSED)


def _check_earlyoom(state: HostState, ctx: CheckContext) -> CheckItem:
    facts = state.facts
    if _truthy(facts, "CHECK_EARLYOOM_ACTIVE"):
        return CheckItem("earlyoom", "earlyoom OOM protection", OK)
    if _truthy(facts, "CHECK_EARLYOOM_INSTALLED"):
        detail = "earlyoom installed but not active"
    else:
        detail = "earlyoom not installed"
    return CheckItem(
        "earlyoom",
        "earlyoom OOM protection",
        WARN,
        detail,
        "sparkrun setup earlyoom%s" % ctx.cluster_flag,
    )


def _check_sudoers(state: HostState, ctx: CheckContext) -> CheckItem:
    chown = state.facts.get("CHECK_SUDOERS_CHOWN", "unknown")
    dropcaches = state.facts.get("CHECK_SUDOERS_DROPCACHES", "unknown")
    if chown == "unknown" or dropcaches == "unknown":
        return CheckItem(
            "sudoers",
            "Sudoers entries (fix-permissions, clear-cache)",
            SKIP,
            "cannot verify without passwordless sudo",
        )
    if chown == "1" and dropcaches == "1":
        return CheckItem("sudoers", "Sudoers entries (fix-permissions, clear-cache)", OK)
    missing = []
    if chown != "1":
        missing.append("fix-permissions")
    if dropcaches != "1":
        missing.append("clear-cache")
    return CheckItem(
        "sudoers",
        "Sudoers entries (fix-permissions, clear-cache)",
        WARN,
        "missing: %s" % ", ".join(missing),
        "sparkrun setup wizard%s — or setup fix-permissions/clear-cache with --save-sudo" % ctx.cluster_flag,
    )


def _check_ssh_mesh(state: HostState, ctx: CheckContext) -> CheckItem | None:
    if not ctx.multi_host:
        return None  # SSH mesh is only meaningful with peers
    total = _as_int(state.facts.get("CHECK_MESH_TOTAL"))
    reachable = _as_int(state.facts.get("CHECK_MESH_OK"))
    if total == 0:
        return None
    if reachable >= total:
        return CheckItem("ssh_mesh", "SSH mesh to peers", OK, "all %d peer(s) reachable" % total)
    return CheckItem(
        "ssh_mesh",
        "SSH mesh to peers",
        WARN,
        "%d/%d peer(s) reachable over SSH" % (reachable, total),
        "sparkrun setup ssh%s" % ctx.cluster_flag,
    )


def _check_cx7(state: HostState, ctx: CheckContext) -> CheckItem | None:
    """Vet *effective* CX7 state via :func:`detect_cx7_for_hosts` output.

    Reuses the same detection the real CX7 flow uses — per-interface link
    state, assigned IP, subnet, and *who persists the address* — so this
    reflects working networking rather than the presence of one config file.

    Persistence is attributed to whatever owns the interface (any netplan
    file, a NetworkManager profile, a ``.network`` unit, ifupdown), because
    the question is "will this address come back after a reboot", not "did
    sparkrun write it". Checking for sparkrun's own ``40-cx7.yaml`` flagged
    every hand-configured cluster — including one set up with ``nmcli``,
    which on Ubuntu 24.04 writes its own ``90-NM-<uuid>.yaml``.

    :attr:`CX7Persistence.UNKNOWN` (no probe available on the host) is
    reported but is **not** a warning: "couldn't tell" is not "won't persist".

    CX7 is inter-node, so it is only evaluated for multi-host clusters.
    """
    from sparkrun.orchestration.networking import CX7Persistence

    if not ctx.multi_host:
        return None
    det = state.cx7
    if det is None:
        return CheckItem("cx7", "CX7 high-speed networking", SKIP, "could not probe CX7 on this host")
    if not det.detected or not det.interfaces:
        return CheckItem("cx7", "CX7 high-speed networking", SKIP, "no CX7/InfiniBand interfaces detected")

    up = [i for i in det.interfaces if i.state.lower() == "up" and i.ip]
    not_ready = [i for i in det.interfaces if i.state.lower() != "up" or not i.ip]

    if not up:
        names = ", ".join(i.name for i in det.interfaces)
        return CheckItem(
            "cx7",
            "CX7 high-speed networking",
            WARN,
            "CX7 interface(s) present but none up with an IP (%s)" % names,
            "sparkrun setup cx7%s" % ctx.cluster_flag,
        )

    subnets = ", ".join(sorted({i.subnet for i in up if i.subnet}))
    detail = "%d interface(s) up%s" % (len(up), " on %s" % subnets if subnets else "")

    if not_ready:
        bad = ", ".join("%s(%s)" % (i.name, i.state or "no-ip") for i in not_ready)
        return CheckItem(
            "cx7",
            "CX7 high-speed networking",
            WARN,
            "%s; not ready: %s" % (detail, bad),
            "sparkrun setup cx7%s" % ctx.cluster_flag,
        )
    ephemeral = [i for i in up if i.persistence is CX7Persistence.EPHEMERAL]
    if ephemeral:
        # Nothing on the host declares these addresses — they were added by
        # hand (or by a profile that won't auto-connect) and are gone on boot.
        named = ", ".join("%s (%s)" % (i.name, i.describe_persistence()) if i.persistence_source else i.name for i in ephemeral)
        return CheckItem(
            "cx7",
            "CX7 high-speed networking",
            WARN,
            "%s but no network config declares %s (won't survive reboot)" % (detail, named),
            "sparkrun setup cx7%s" % ctx.cluster_flag,
        )

    leased = [i for i in up if i.dhcp]
    if leased:
        # Persistent, but not pinned: NCCL peer addressing needs these stable.
        return CheckItem(
            "cx7",
            "CX7 high-speed networking",
            WARN,
            "%s but %s hold DHCP leases, so the addresses may change" % (detail, ", ".join(i.name for i in leased)),
            "sparkrun setup cx7%s" % ctx.cluster_flag,
        )

    unknown = [i for i in up if i.persistence is CX7Persistence.UNKNOWN]
    if unknown:
        return CheckItem(
            "cx7",
            "CX7 high-speed networking",
            OK,
            "%s; could not verify persistence of %s (no netplan/nmcli/networkctl on host)" % (detail, ", ".join(i.name for i in unknown)),
        )

    owners = sorted({i.describe_persistence() for i in up})
    return CheckItem("cx7", "CX7 high-speed networking", OK, "%s; persisted by %s" % (detail, ", ".join(owners)))


@dataclass
class SetupCheck:
    """One readiness check — the check half of a future setup "step"."""

    key: str
    evaluate: Callable[["HostState", CheckContext], "CheckItem | None"]


#: Ordered registry of readiness checks. Order is the display/evaluation
#: order and is the seed of the future ordered/dependency-driven step system.
SETUP_CHECKS: tuple[SetupCheck, ...] = (
    SetupCheck("docker_installed", _check_docker_installed),
    SetupCheck("docker_group", _check_docker_group),
    SetupCheck("docker_usable", _check_docker_usable),
    SetupCheck("nvidia_ctk", _check_nvidia_ctk),
    SetupCheck("cdi_spec", _check_cdi_spec),
    SetupCheck("earlyoom", _check_earlyoom),
    SetupCheck("sudoers", _check_sudoers),
    SetupCheck("ssh_mesh", _check_ssh_mesh),
    SetupCheck("cx7", _check_cx7),
)


def evaluate_host(state: HostState, ctx: CheckContext) -> list[CheckItem]:
    """Run every applicable check for one host's *state*."""
    items: list[CheckItem] = []
    for check in SETUP_CHECKS:
        item = check.evaluate(state, ctx)
        if item is not None:
            items.append(item)
    return items


# --- CLI command ------------------------------------------------------------


def _render_host(host: str, items: list[CheckItem]) -> None:
    click.echo(host)
    for item in items:
        line = "  %s %s" % (_STATUS_MARK[item.status], item.label)
        if item.detail:
            line += " — %s" % item.detail
        click.echo(line)
        if item.guidance and item.status in (WARN, FAIL):
            click.echo("         → %s" % item.guidance)
    click.echo()


def register(setup_group) -> None:
    """Attach the ``check`` command to the ``setup`` group.

    Called from ``cli/_setup/__init__.py`` to avoid an import cycle (the
    group is defined there).
    """

    def _resolve_gpu_access_modes(host_list, *, cluster_name, hosts, hosts_file, config) -> dict[str, str]:
        """Map each host to the ``gpu_access_mode`` a launch on it would resolve to.

        Runs the *real* executor resolution chain per host (the platform tier is
        keyed off that host's hardware), so the check reports on the same value
        the launch would use rather than re-deriving the rule. Mirrors the
        launcher's fallback: with no cluster definition, hardware defaults to
        DGX Spark.

        Best-effort — any failure yields an empty/partial map and
        :meth:`CheckContext.cdi_required` falls back to "CDI is required".
        """
        from sparkrun.core.hardware import default_dgx_spark_hardware
        from sparkrun.orchestration.executor import resolve_executor

        from .._common import _get_cluster_manager

        modes: dict[str, str] = {}
        cluster_def = None
        try:
            cluster_mgr = _get_cluster_manager()
            # Same "which cluster" rule as resolve_cluster_config: an explicit
            # --hosts/--hosts-file means the named cluster isn't the host source.
            resolved = cluster_name or (cluster_mgr.get_default() if not (hosts or hosts_file) else None)
            if resolved:
                cluster_def = cluster_mgr.get(resolved)
        except Exception:
            logger.debug("Could not resolve cluster for gpu_access_mode", exc_info=True)

        for host in host_list:
            try:
                hw = cluster_def.hardware_for(host) if cluster_def is not None else default_dgx_spark_hardware()
                executor = resolve_executor(cluster=cluster_def, config=config, host_hardware=hw, rootless=False, auto_user=False)
                modes[host] = executor.config.gpu_access_mode
            except Exception:
                logger.debug("Could not resolve gpu_access_mode for %s", host, exc_info=True)
        return modes

    @setup_group.command("check")
    @host_options
    @json_option(help="Emit the full results as JSON")
    @click.pass_context
    def setup_check(ctx, hosts, hosts_file, cluster_name, output_json):
        """Check cluster hosts against the setup steps, without changing anything.

        Probes each host for the things ``sparkrun setup wizard`` configures
        (Docker access, NVIDIA CDI, earlyoom, sudoers, SSH mesh, CX7) and
        reports gaps with the command that fixes each. Uses the default
        cluster unless ``--cluster``/``--hosts`` is given. Read-only.

        Examples:

          sparkrun setup check

          sparkrun setup check --cluster mylab

          sparkrun setup check --hosts 10.0.0.1,10.0.0.2 --json
        """
        from concurrent.futures import ThreadPoolExecutor

        from sparkrun.core.config import SparkrunConfig
        from sparkrun.orchestration.ssh import run_remote_script
        from sparkrun.scripts import read_script
        from sparkrun.utils.text import parse_kv_output

        from .._common import _resolve_setup_context, print_json

        config = SparkrunConfig()
        host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user=None)

        multi_host = len(host_list) > 1
        check_ctx = CheckContext(
            cluster_name=cluster_name,
            multi_host=multi_host,
            gpu_access_modes=_resolve_gpu_access_modes(
                host_list,
                cluster_name=cluster_name,
                hosts=hosts,
                hosts_file=hosts_file,
                config=config,
            ),
        )

        target = "cluster '%s'" % cluster_name if cluster_name else "%d host(s)" % len(host_list)
        click.echo("Setup check for %s (%d host(s))" % (target, len(host_list)))
        click.echo("=" * 56)
        click.echo()

        # Probe every host concurrently. Each host gets its own peer list
        # (all other hosts) injected into the script for the SSH-mesh probe.
        results_by_host: dict[str, list[CheckItem]] = {}
        unreachable: list[str] = []
        json_hosts: dict[str, object] = {}

        def _probe(host: str):
            peers = " ".join(h for h in host_list if h != host)
            script = read_script("setup_check.sh").format(peers=peers)
            return run_remote_script(host, script, timeout=60, quiet=True, **ssh_kwargs)

        with ThreadPoolExecutor(max_workers=min(len(host_list), 16)) as pool:
            raw_results = dict(zip(host_list, pool.map(_probe, host_list)))

        # CX7 readiness reuses the real detection machinery (per-interface link
        # state + IP + netplan), so it reflects effective networking rather than
        # a config file's presence. Only meaningful multi-host; best-effort.
        cx7_detections = {}
        if multi_host:
            from sparkrun.orchestration.networking import detect_cx7_for_hosts

            try:
                cx7_detections = detect_cx7_for_hosts(host_list, ssh_kwargs=ssh_kwargs)
            except Exception:
                logger.debug("CX7 detection failed during setup check", exc_info=True)
                cx7_detections = {}

        # Render in a deterministic host order.
        for host in host_list:
            r = raw_results.get(host)
            if r is None or not r.success or "CHECK_COMPLETE=1" not in (r.stdout or ""):
                unreachable.append(host)
                click.echo(host)
                detail = (r.stderr.strip()[:160] if r and r.stderr else "no response") if r else "no response"
                click.echo("  %s SSH connectivity — %s" % (_STATUS_MARK[FAIL], detail))
                click.echo("         → sparkrun setup ssh%s (verify SSH access first)" % check_ctx.cluster_flag)
                click.echo()
                json_hosts[host] = {"reachable": False, "checks": []}
                continue

            state = HostState(host=host, facts=parse_kv_output(r.stdout), cx7=cx7_detections.get(host))
            items = evaluate_host(state, check_ctx)
            results_by_host[host] = items
            _render_host(host, items)
            json_hosts[host] = {
                "reachable": True,
                "checks": [{"key": i.key, "label": i.label, "status": i.status, "detail": i.detail, "guidance": i.guidance} for i in items],
            }

        # Aggregate.
        fail_count = sum(1 for items in results_by_host.values() for i in items if i.status == FAIL)
        warn_count = sum(1 for items in results_by_host.values() for i in items if i.status == WARN)
        gaps = fail_count + warn_count + len(unreachable)

        click.echo("=" * 56)
        if gaps == 0:
            click.echo("All checks passed across %d host(s). No setup gaps found." % len(host_list))
        else:
            parts = []
            if unreachable:
                parts.append("%d unreachable host(s)" % len(unreachable))
            if fail_count:
                parts.append("%d critical gap(s)" % fail_count)
            if warn_count:
                parts.append("%d advisory(s)" % warn_count)
            click.echo("Found %s." % ", ".join(parts))
            click.echo("Fix per the '→' guidance above, or run 'sparkrun setup wizard%s'." % check_ctx.cluster_flag)

        if output_json:
            print_json(
                {
                    "cluster": cluster_name,
                    "hosts": host_list,
                    "unreachable": unreachable,
                    "critical_gaps": fail_count,
                    "advisories": warn_count,
                    "results": json_hosts,
                }
            )

        # Critical gaps (or unreachable hosts) fail the command; advisories don't.
        if fail_count or unreachable:
            sys.exit(1)
