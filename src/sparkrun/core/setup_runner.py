"""Console-free sequencing, manifest recording, and reprobes for host setup."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from sparkrun.core.setup_actions import SetupActionContext, SetupActionResult, aggregate_action_status
from sparkrun.core.setup_models import CheckContext, HostState, FAIL, WARN, OK, SKIP
from sparkrun.core.setup_steps import PlannedStep, SetupStep, all_setup_steps, apply_setup_step, build_setup_plan

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.setup_manifest import ManifestManager


@dataclass(frozen=True)
class SetupEvent:
    kind: str
    step: str
    label: str
    hosts: tuple[str, ...]
    status: str = ""
    detail: str = ""


@dataclass
class SetupRunResult:
    steps: dict[str, str]
    states: dict[str, HostState]
    context: CheckContext
    plans: dict[str, list[PlannedStep]]
    findings: int
    outcomes: dict[str, dict[str, SetupActionResult]] = field(default_factory=dict)
    """Step ID -> host -> action/no-op outcome, including details and changes."""


def _cluster_name(context: CheckContext, cluster: ClusterDefinition | None, explicit: str | None) -> str | None:
    names = {name for name in (context.cluster_name, cluster.name if cluster else None, explicit) if name}
    if len(names) > 1:
        raise ValueError("Conflicting setup cluster names: " + ", ".join(sorted(names)))
    return next(iter(names), None)


def _validated_states(states: Mapping[str, HostState], *, expected_hosts: list[str] | None = None) -> dict[str, HostState]:
    """Keep approval, action targets, and recording on the same host identity."""
    if not isinstance(states, Mapping):
        raise TypeError("Setup states must be a mapping of host names to HostState")
    checked: dict[str, HostState] = {}
    for host, state in states.items():
        if not isinstance(host, str) or not host.strip() or not isinstance(state, HostState):
            raise ValueError("Setup states require nonempty host names and HostState values")
        if state.host != host:
            raise ValueError("Setup state host %r does not match mapping key %r" % (state.host, host))
        if expected_hosts is not None and host not in expected_hosts:
            raise ValueError("Setup reprobe returned an unexpected host: %r" % host)
        checked[host] = replace(state)
    return checked


def run_setup_steps(
    states: dict[str, HostState],
    context: CheckContext,
    action_context: SetupActionContext,
    *,
    cluster: ClusterDefinition | None = None,
    cluster_name: str | None = None,
    manifest_mgr: ManifestManager | None = None,
    approve: Callable[[SetupStep, tuple[str, ...]], bool] | None = None,
    credentials: Callable[[], str | None] | None = None,
    host_credentials: Callable[[str], str | None] | None = None,
    progress_callback: Callable[[SetupEvent], None] | None = None,
    only_steps: set[str] | None = None,
) -> SetupRunResult:
    """Apply eligible steps with deterministic, per-host action results.

    None for approve means apply. No manifest manager explicitly disables
    recording; otherwise a consistent cluster name is required. Real recording
    holds a validated manifest lock across changes and reprobes. Dry runs invoke
    no actions, credentials, approvals, manifest writes/locks, or reprobes.
    SSH-mesh/CX7 topology interaction remains frontend-owned and reports skip.
    host_credentials(host) is asked, once per host per action, when sudo
    rejects the shared password there; a password it supplies is reused for
    that host's later actions. Omit it for non-interactive callers.
    """
    from sparkrun.core.setup_probe import probe_setup_hosts

    states = _validated_states(states)
    config = context.config
    if not action_context.dry_run and config is None:
        raise ValueError("Setup execution requires a configuration for reprobing changed hosts")
    cluster_name = _cluster_name(context, cluster, cluster_name)
    if manifest_mgr is not None and cluster_name is None:
        raise ValueError("Setup manifest recording requires a cluster name")
    context = replace(context, cluster_name=cluster_name)
    steps = all_setup_steps()
    if only_steps is not None and (unknown := only_steps - {step.key for step in steps}):
        raise ValueError("Unknown setup steps: " + ", ".join(sorted(unknown)))
    results: dict[str, str] = {}
    outcomes: dict[str, dict[str, SetupActionResult]] = {}
    passwords = action_context.passwords
    if passwords is None and host_credentials is not None:
        from sparkrun.orchestration.sudo import SudoPasswords

        passwords = SudoPasswords(prompt_host=host_credentials)

    def emit(event: SetupEvent) -> None:
        if progress_callback is not None:
            progress_callback(event)

    recording = nullcontext()
    if manifest_mgr is not None and not action_context.dry_run:
        assert cluster_name is not None
        recording = manifest_mgr.recording(cluster_name)
    with recording:

        def plan_hosts():
            return {host: {entry.step.key: entry for entry in build_setup_plan(state, context)} for host, state in states.items()}

        host_plans = plan_hosts()
        for step in steps:
            if only_steps is not None and step.key not in only_steps:
                continue
            entries = {host: plan[step.key] for host, plan in host_plans.items()}
            per_host = outcomes[step.key] = {}
            candidates = []
            for host, entry in entries.items():
                if not entry.selected or entry.blocked_by:
                    detail = entry.reason or "blocked by " + ", ".join(entry.blocked_by)
                    outcome = SetupActionResult(host, FAIL if only_steps is not None else SKIP, detail)
                elif step.key in {"ssh_mesh", "cx7"}:
                    outcome = SetupActionResult(host, SKIP, "requires the frontend topology adapter")
                elif entry.needs_action:
                    if not action_context.dry_run:
                        candidates.append(host)
                    outcome = SetupActionResult(host, SKIP, "would apply " + step.label)
                else:
                    status = next((status for status in (FAIL, WARN) if any(item.status == status for item in entry.checks)), OK)
                    detail = "already satisfied" if status == OK else "check only; no automatic action available"
                    outcome = SetupActionResult(host, status, detail)
                per_host[host] = outcome
                if action_context.dry_run:
                    emit(SetupEvent("preview", step.key, step.label, (host,), outcome.status, outcome.detail))
            results[step.key] = aggregate_action_status(per_host)
            if not candidates:
                continue
            candidates = tuple(candidates)
            emit(SetupEvent("selected", step.key, step.label, candidates))
            if approve is not None and not approve(step, candidates):
                for host in candidates:
                    per_host[host] = SetupActionResult(host, SKIP, "declined by caller")
                results[step.key] = aggregate_action_status(per_host)
                continue
            action = replace(
                action_context,
                sudo_password=(credentials() if credentials else action_context.sudo_password) if step.requires_sudo else None,
                passwords=passwords if step.requires_sudo else None,
            )
            changed = False
            for host in candidates:
                try:
                    outcome = apply_setup_step(step.key, states[host], context, action)
                except Exception as exc:
                    per_host[host] = SetupActionResult(host, FAIL, str(exc))
                    emit(SetupEvent("error", step.key, step.label, (host,), FAIL, str(exc)))
                    continue
                per_host[host] = outcome
                if outcome.changed:
                    changed = True
                    if manifest_mgr is not None:
                        legacy = {
                            key: value
                            for key, value in outcome.extra.items()
                            if key not in {"cluster_name", "user", "hosts", "phase", "host_details"}
                        }
                        assert cluster_name is not None
                        manifest_mgr.record_phase(cluster_name, action.user, [host], step.key, host_details={host: outcome.extra}, **legacy)
                # Persist before notifying the frontend: a rendering failure
                # cannot hide completed changes from subsequent teardown.
                emit(SetupEvent("result", step.key, step.label, (host,), outcome.status, outcome.detail))
            results[step.key] = aggregate_action_status(per_host)
            if changed:
                assert config is not None
                reprobe_hosts = [host for host, state in states.items() if state.reachable]
                refreshed, new_context = probe_setup_hosts(
                    reprobe_hosts,
                    ssh_kwargs=action_context.ssh_kwargs,
                    config=config,
                    cluster=cluster,
                    cluster_name=cluster_name,
                )
                states.update(_validated_states(refreshed, expected_hosts=reprobe_hosts))
                context = replace(new_context, multi_host=context.multi_host, cluster_name=cluster_name)
                host_plans = plan_hosts()

        plans = {host: list(plan.values()) for host, plan in host_plans.items()}
        findings = sum(
            item.status in {FAIL, WARN} for entries in plans.values() for entry in entries if entry.selected for item in entry.checks
        )
        return SetupRunResult(results, states, context, plans, findings, outcomes)
