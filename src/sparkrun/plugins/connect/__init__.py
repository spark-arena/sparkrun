"""``sparkrun ssh`` and ``sparkrun exec``: interactive access to cluster nodes and workloads.

``sparkrun ssh [N | CLUSTER[/N]]`` opens SSH on a node of a saved cluster
(node 0 of the default cluster, which follows ``SPARKRUN_CLUSTER``, when no
target is given). ``sparkrun exec [N | WORKLOAD[/N]]`` opens a shell inside a
running workload's container (or, for the ``local`` executor, in the native
process's environment). Anything after ``--`` runs instead of a shell.
``sparkrun connect`` is a hidden alias of ``ssh``.

Node numbers index the cluster's saved host list for ``ssh`` (not the head of
a running job, which a load-aware scheduler may have placed elsewhere), and
the workload's containers, head first, for ``exec``.

Pieces:

* :mod:`.target`: target parsing and remote-command quoting, console-free
* :mod:`.cli`: the Click commands, built lazily so ``api`` never imports Click

The plugin is on by default; each command has its own flag (``cli.ssh``,
``cli.exec``), on for the beta and alpha channels. Container entry itself is
core: :meth:`~sparkrun.orchestration.executors._base.Executor.interactive_exec_argv`
behind :func:`sparkrun.api.open_exec_target`.
"""

from __future__ import annotations

from sparkrun import __version__ as __version__

SPARKRUN_PLUGIN_API_VERSION = 1

SSH_FEATURE_FLAG = "cli.ssh"
EXEC_FEATURE_FLAG = "cli.exec"

# Read as a literal without importing this module, so the flag names are spelled out.
SPARKRUN_PLUGIN_MANIFEST = {
    "api": 1,
    "features": {
        "cli.ssh": {
            "description": "'sparkrun ssh': SSH into a node of a saved cluster",
            "default": False,
            "channel_defaults": {"beta": True, "alpha": True},
        },
        "cli.exec": {
            "description": "'sparkrun exec': open a shell in a running workload's container",
            "default": False,
            "channel_defaults": {"beta": True, "alpha": True},
        },
    },
}


def _load_ssh():
    from .cli import build_ssh_command

    return build_ssh_command("ssh")


def _load_connect():
    from .cli import build_ssh_command

    return build_ssh_command("connect", hidden=True)


def _load_exec():
    from .cli import build_exec_command

    return build_exec_command()


def register(v):
    from sparkrun.core.cli_registry import register_cli_command
    from sparkrun.core.features import feature_gate_enabled

    if feature_gate_enabled(SSH_FEATURE_FLAG, v):
        register_cli_command(_load_ssh, name="ssh")
        register_cli_command(_load_connect, name="connect")
    if feature_gate_enabled(EXEC_FEATURE_FLAG, v):
        register_cli_command(_load_exec, name="exec")
