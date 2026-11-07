"""Public library API for cluster setup operations.

Console-free surface for the parts of ``sparkrun setup`` that a GUI needs to
drive itself.  Mirrors the :mod:`sparkrun.plugins.k8s.api` / :mod:`sparkrun.api.tailscale`
conventions: dataclass returns, typed
:class:`~sparkrun.api._errors.SparkrunError` subclasses, no writes to
stdout/stderr.

Includes SSH bootstrap/mesh, RDMA diagnostics, and the shared host setup runner.

Stable: the SSH bootstrap (``probe_ssh_access``, ``ensure_local_key``,
``install_public_key_interactive``, ``mesh_ssh_keys_native``), ``rdma_test``,
``check_ssh_host_identities`` / ``rotate_ssh_host_keys`` and the
``run_setup_steps`` / ``run_setup_undo`` runner with their result types.

.. warning::
   **Experimental:** the planning internals re-exported for custom planning
   interfaces (``ManifestManager``, ``SetupManifest``, ``PhaseRecord``,
   ``SetupPlan``, ``PlannedStep``, ``SetupStep``, ``CheckContext``,
   ``CheckItem``, ``HostState``, ``SetupActionContext``, ``SetupActionResult``,
   ``probe_setup_hosts``, ``resolve_setup_context``, ``build_setup_plan``,
   ``evaluate_host``, ``apply_setup_step``) are core types aliased here and
   may change in a minor release. So may the CX7 fabric configuration
   (``plan_cx7``, ``apply_cx7``, ``save_cx7_topology``,
   ``validate_cx7_options`` and the ``Cx7*`` types) and the SSH diagnosis
   (``diagnose_ssh``, ``find_local_public_key``, ``SshDiagnosticReport`` and
   ``SshHostDiagnosis``), which hold the decisions
   behind ``setup cx7`` / the wizard's CX7 phase and ``setup ssh --diagnose``.
"""

from __future__ import annotations

from ._runner import SetupEvent, SetupRunResult, SetupUndoResult, run_setup_steps, run_setup_undo

# Caller-facing planning/action models are aliases of the core types. Plugin
# registration remains in core.setup_steps; frontends need only this facade.
from sparkrun.core.setup_actions import SetupActionContext, SetupActionResult
from sparkrun.orchestration.sudo import SudoPasswords
from sparkrun.core.setup_manifest import ManifestManager, SetupManifest, PhaseRecord
from sparkrun.core.setup_plans import SetupPlan
from sparkrun.core.setup_models import CheckContext, CheckItem, HostState, OK, WARN, FAIL, SKIP
from sparkrun.core.setup_probe import probe_setup_hosts, resolve_setup_context
from sparkrun.core.setup_steps import PlannedStep, SetupStep, build_setup_plan, apply_setup_step, evaluate_host

from ._errors import SetupFailed, OpenSshUnavailable, RdmaTestError, SshAccessError, SshKeyError
from ._mesh import MeshResult, mesh_ssh_keys_native
from ._cx7 import (
    CX7_EVENT_DETECTED,
    CX7_EVENT_DETECTED_TOPOLOGY,
    CX7_EVENT_DETECTING_TOPOLOGY,
    CX7_EVENT_INTERFACES,
    CX7_EVENT_RING_DETECTING,
    CX7_EVENT_SAVED_TOPOLOGY,
    CX7_EVENT_SUBNETS,
    CX7_EVENT_TOPOLOGY,
    CX7_TOPOLOGY_CHOICES,
    Cx7ApplyResult,
    Cx7InterfacesSave,
    Cx7Plan,
    Cx7PlanError,
    Cx7TopologySave,
    apply_cx7,
    plan_cx7,
    save_cx7_topology,
    validate_cx7_options,
)
from ._rdma_plan import CoverageIssue
from ._rdma import (
    DEFAULT_RDMA_TEST_IMAGE,
    STATUS_FAIL,
    STATUS_OK,
    STATUS_SKIP,
    STATUS_WARN,
    SUITE_ALL,
    SUITE_NCCL,
    SUITE_PERFTEST,
    LinkTestResult,
    AggregateTestResult,
    RdmaCoverage,
    NcclTestResult,
    PairTestResult,
    RdmaTestReport,
    rdma_test,
)
from ._ssh_access import (
    SPARKRUN_KEY_NAME,
    ControlKey,
    LocalSshKey,
    SshProbe,
    ensure_local_key,
    install_public_key_interactive,
    load_identity,
    probe_ssh_access,
    resolve_control_key,
    validate_public_key,
)

from ._ssh_diagnose import (
    SSH_DIAG_COLLECT_START,
    SSH_DIAG_CONNECT_FAILED,
    SSH_DIAG_CONNECT_START,
    SSH_DIAG_HOST_DONE,
    SSH_DIAG_KEY_CHECK_DONE,
    SSH_DIAG_KEY_CHECK_SKIPPED,
    SSH_DIAG_KEY_CHECK_START,
    SSH_DIAG_LOCAL_KEY,
    SSH_DIAG_PUBKEY_DONE,
    SSH_DIAG_PUBKEY_START,
    SshDiagnosticReport,
    SshHostDiagnosis,
    diagnose_ssh,
    find_local_public_key,
)
from ._ssh_identity import SshHostIdentity, DuplicateHostIdentity, SshIdentityReport, check_ssh_host_identities
from ._ssh_rotation import SshRotationResult, rotate_ssh_host_keys

__all__ = [
    "SshHostIdentity",
    "DuplicateHostIdentity",
    "SshIdentityReport",
    "check_ssh_host_identities",
    "SshRotationResult",
    "rotate_ssh_host_keys",
    "SetupActionContext",
    "SudoPasswords",
    "SetupActionResult",
    "ManifestManager",
    "SetupManifest",
    "PhaseRecord",
    "CheckContext",
    "CheckItem",
    "HostState",
    "OK",
    "WARN",
    "FAIL",
    "SKIP",
    "probe_setup_hosts",
    "resolve_setup_context",
    "PlannedStep",
    "SetupStep",
    "SetupPlan",
    "build_setup_plan",
    "apply_setup_step",
    "evaluate_host",
    "SetupFailed",
    "SetupEvent",
    "SetupRunResult",
    "SetupUndoResult",
    "run_setup_undo",
    "run_setup_steps",
    # Functions
    "probe_ssh_access",
    "ensure_local_key",
    "install_public_key_interactive",
    "validate_public_key",
    "load_identity",
    "resolve_control_key",
    "mesh_ssh_keys_native",
    "rdma_test",
    # CX7 fabric configuration (experimental)
    "plan_cx7",
    "apply_cx7",
    "save_cx7_topology",
    "validate_cx7_options",
    "Cx7Plan",
    "Cx7ApplyResult",
    "Cx7TopologySave",
    "Cx7InterfacesSave",
    "Cx7PlanError",
    "CX7_TOPOLOGY_CHOICES",
    "CX7_EVENT_DETECTED",
    "CX7_EVENT_INTERFACES",
    "CX7_EVENT_SAVED_TOPOLOGY",
    "CX7_EVENT_DETECTING_TOPOLOGY",
    "CX7_EVENT_DETECTED_TOPOLOGY",
    "CX7_EVENT_RING_DETECTING",
    "CX7_EVENT_TOPOLOGY",
    "CX7_EVENT_SUBNETS",
    # SSH diagnosis (experimental)
    "diagnose_ssh",
    "find_local_public_key",
    "SshDiagnosticReport",
    "SshHostDiagnosis",
    "SSH_DIAG_LOCAL_KEY",
    "SSH_DIAG_PUBKEY_START",
    "SSH_DIAG_PUBKEY_DONE",
    "SSH_DIAG_CONNECT_START",
    "SSH_DIAG_CONNECT_FAILED",
    "SSH_DIAG_COLLECT_START",
    "SSH_DIAG_KEY_CHECK_START",
    "SSH_DIAG_KEY_CHECK_DONE",
    "SSH_DIAG_KEY_CHECK_SKIPPED",
    "SSH_DIAG_HOST_DONE",
    # Data models
    "SshProbe",
    "LocalSshKey",
    "ControlKey",
    "MeshResult",
    "RdmaTestReport",
    "PairTestResult",
    "LinkTestResult",
    "AggregateTestResult",
    "RdmaCoverage",
    "CoverageIssue",
    "NcclTestResult",
    # Errors
    "SshAccessError",
    "SshKeyError",
    "OpenSshUnavailable",
    "RdmaTestError",
    # Constants
    "SPARKRUN_KEY_NAME",
    "DEFAULT_RDMA_TEST_IMAGE",
    "SUITE_PERFTEST",
    "SUITE_NCCL",
    "SUITE_ALL",
    "STATUS_OK",
    "STATUS_WARN",
    "STATUS_FAIL",
    "STATUS_SKIP",
]
