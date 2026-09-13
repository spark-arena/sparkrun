"""sparkrun setup group and subcommands."""

from __future__ import annotations

import click

from .._common import _get_cluster_manager


@click.group(invoke_without_command=True)
@click.pass_context
def setup(ctx):
    """Setup and configuration commands."""
    if ctx.invoked_subcommand is not None:
        return
    # Smart routing: auto-launch wizard when no default cluster is set
    mgr = _get_cluster_manager()
    if mgr.get_default() is None:
        from ._wizard import setup_wizard

        ctx.invoke(setup_wizard)
    else:
        click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# Register subcommands
# ---------------------------------------------------------------------------

from . import _commands as _commands  # noqa: E402, F401  — registers @setup.command() decorators
from . import _fe_update as _fe_update  # noqa: E402, F401  — registers @setup.command("fe-system-update")
from . import _gpu_clock as _gpu_clock  # noqa: E402, F401  — registers @setup.command("throttle-gpu-clock")
from . import _k8s as _k8s  # noqa: E402, F401  — registers @setup.group("k8s")
from . import _plugins as _plugins  # noqa: E402, F401  — registers @setup.group("plugins")
from . import _rdma as _rdma  # noqa: E402, F401  — registers @setup.command("rdma-test")
from . import _tailscale as _tailscale  # noqa: E402, F401  — registers @setup.group("tailscale")
from ._wizard import setup_wizard  # noqa: E402
from ._uninstall import setup_uninstall  # noqa: E402
from . import _check as _check  # noqa: E402  — registers @setup.command("check")

setup.add_command(setup_wizard)
setup.add_command(setup_uninstall)
_check.register(setup)

# ---------------------------------------------------------------------------
# Re-export symbols used by external code (tests, etc.)
# ---------------------------------------------------------------------------

from ._phases import (  # noqa: E402, F401
    EARLYOOM_PREFER_PATTERNS,
    EARLYOOM_AVOID_PATTERNS,
    _build_earlyoom_regex,
    _DOCKER_GROUP_SCRIPT,
    _DOCKER_GROUP_FALLBACK_SCRIPT,
    _docker_group_summary,
)
from ._ssh import _run_ssh_mesh, _detect_and_update_mgmt_ips  # noqa: E402, F401
from ._sudo import _record_setup_phase  # noqa: E402, F401
