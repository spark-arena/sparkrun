"""Public library API for cluster setup operations.

Console-free surface for the parts of ``sparkrun setup`` that a GUI needs to
drive itself.  Mirrors the :mod:`sparkrun.plugins.k8s.api` / :mod:`sparkrun.api.tailscale`
conventions: dataclass returns, typed
:class:`~sparkrun.api._errors.SparkrunError` subclasses, no writes to
stdout/stderr.

Includes SSH bootstrap/mesh, RDMA diagnostics, and the shared host setup runner.
Low-level setup checks/actions remain available for custom planning interfaces.
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
from ._mesh import (
    MeshResult,
    build_collect_key_script,
    build_install_keys_script,
    mesh_ssh_keys_native,
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
    LocalSshKey,
    SshProbe,
    build_authorized_key_script,
    ensure_local_key,
    install_public_key_interactive,
    probe_ssh_access,
)

__all__ = [
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
    "mesh_ssh_keys_native",
    "rdma_test",
    # Script builders (exposed for testing and for callers that pipe them
    # through their own transport)
    "build_authorized_key_script",
    "build_collect_key_script",
    "build_install_keys_script",
    # Data models
    "SshProbe",
    "LocalSshKey",
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
