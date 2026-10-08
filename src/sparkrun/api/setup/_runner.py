"""Public error boundary for the shared setup runner."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING
from sparkrun.api._errors import SparkrunError
from sparkrun.core.setup_actions import SetupActionContext
from sparkrun.core.setup_models import CheckContext, HostState
from sparkrun.core.setup_runner import SetupEvent, SetupRunResult, run_setup_steps as _run_setup_steps
from sparkrun.core.setup_steps import SetupStep
from sparkrun.core.setup_undo import SetupUndoResult, run_setup_undo as _run_setup_undo
from ._errors import SetupFailed

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.setup_manifest import ManifestManager, SetupManifest


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
    """Sequence checked actions; orchestration failures raise SetupFailed.

    Per-host action failures are recorded in the returned result and do not
    abort other eligible actions. Callbacks run synchronously; interrupts pass
    through. See core.setup_runner.run_setup_steps for recording/preview policy.
    """
    try:
        return _run_setup_steps(
            states,
            context,
            action_context,
            cluster=cluster,
            cluster_name=cluster_name,
            manifest_mgr=manifest_mgr,
            approve=approve,
            credentials=credentials,
            host_credentials=host_credentials,
            progress_callback=progress_callback,
            only_steps=only_steps,
        )
    except SparkrunError:
        raise
    except Exception as exc:
        raise SetupFailed("Setup failed: %s" % exc) from exc


def run_setup_undo(
    manifest: SetupManifest,
    action_context: SetupActionContext,
    *,
    manifest_mgr: ManifestManager | None = None,
    steps: Mapping[str, SetupStep] | None = None,
    only_steps: set[str] | None = None,
    approve: Callable[[SetupStep, tuple[str, ...]], bool] | None = None,
    credentials: Callable[[], str | None] | None = None,
    progress_callback: Callable[[SetupEvent], None] | None = None,
) -> SetupUndoResult:
    """Undo recorded changes; incomplete targets remain available for retry.

    Per-host failures are returned outcomes. Orchestration/recording/callback
    failures raise SetupFailed; interrupts propagate unchanged.
    """
    try:
        return _run_setup_undo(
            manifest,
            action_context,
            manifest_mgr=manifest_mgr,
            steps=steps,
            only_steps=only_steps,
            approve=approve,
            credentials=credentials,
            progress_callback=progress_callback,
        )
    except SparkrunError:
        raise
    except Exception as exc:
        raise SetupFailed("Setup undo failed: %s" % exc) from exc
