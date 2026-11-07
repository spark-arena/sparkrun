"""Lazy Click presentation for ``sparkrun ssh`` / ``sparkrun exec``.

Plugin scanning imports every submodule, and that happens on the console-free
``api`` path too, so Click is imported inside the builders and never at module
level. The argv itself always comes from core (``interactive_ssh_argv`` for
``ssh``, the executor's ``interactive_exec_argv`` via
``api.open_exec_target`` for ``exec``); this module parses, resolves and hands
the terminal over.
"""

from __future__ import annotations

_COMMAND_KEY = "connect.command"
_SSH_HINT = " Install the OpenSSH client (Windows: Settings > Apps > Optional Features > OpenSSH Client)."


def _wants_tty(command: str | None) -> bool:
    """A PTY for a shell whenever stdin is a terminal; for a command only when both ends are.

    A login shell needs one to be usable. A command forced onto a PTY with its
    output redirected (``-- cat f > out``) would write CRLF line endings.
    """
    import sys

    if not sys.stdin.isatty():
        return False
    return command is None or sys.stdout.isatty()


def _hand_over(argv: list[str]) -> None:
    """Give the terminal to *argv*, exiting with its status.

    POSIX replaces this process, so the tool owns the terminal, its signals
    and the exit code outright. Windows has no exec that keeps the console, so
    it runs as a child and its status is passed through.

    Raises:
        FileNotFoundError: *argv[0]* is not on PATH (checked before anything is replaced).
    """
    import os
    import shutil
    import signal
    import subprocess
    import sys

    if shutil.which(argv[0]) is None:
        raise FileNotFoundError(argv[0])
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "posix":
        os.execvp(argv[0], argv)
    else:
        # Ctrl-C reaches every process on the console. Ignoring it here leaves
        # the child to decide, instead of Popen killing it as the parent unwinds.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        sys.exit(subprocess.call(argv))


def _shared():
    """Click pieces both commands use, built on first call so Click stays lazy."""
    import click

    class DashDashCommand(click.Command):
        """Splits off everything after ``--`` before Click parses the rest.

        With an optional ``TARGET`` and a variadic command, Click alone cannot
        tell ``ssh -- uptime`` from ``ssh uptime``: it would bind the remote
        command as the target. Recording that ``--`` was seen (even with
        nothing after it) also tells completion to stop offering targets.
        """

        def parse_args(self, ctx, args):
            if "--" in args:
                split = args.index("--")
                ctx.meta[_COMMAND_KEY] = args[split + 1 :]
                args = args[:split]
            return super().parse_args(ctx, args)

    def run_or_print(argv: list[str], *, dry_run: bool, banner: str | None) -> None:
        import shlex

        if dry_run:
            click.echo(shlex.join(argv))
            return
        if banner:
            click.echo(banner, err=True)
        try:
            _hand_over(argv)
        except FileNotFoundError:
            raise click.ClickException("%s not found on PATH.%s" % (argv[0], _SSH_HINT if argv[0] == "ssh" else "")) from None

    return DashDashCommand, run_or_print


def build_ssh_command(name: str = "ssh", *, hidden: bool = False):
    import click

    from sparkrun.cli._common import CLUSTER_NAME, _get_context, _load_cluster_or_exit, _require_cluster, dry_run_option

    from .target import ConnectTargetError, parse_cluster_target, remote_command

    DashDashCommand, run_or_print = _shared()

    class _ClusterTargetType(click.ParamType):
        name = "target"

        def shell_complete(self, ctx, param, incomplete):
            from click.shell_completion import CompletionItem

            if _COMMAND_KEY in ctx.meta:
                return []
            try:
                mgr = _get_context(ctx).cluster_manager
                if "/" in incomplete:
                    cluster = incomplete.partition("/")[0]
                    items = [("%s/%d" % (cluster, i), host) for i, host in enumerate(mgr.get(cluster).hosts)]
                else:
                    items = [("%s/" % c.name, c.description or None) for c in mgr.list_clusters()]
                return [CompletionItem(value, help=help) for value, help in items if value.startswith(incomplete)]
            except Exception:
                return []

    @click.command(
        name,
        cls=DashDashCommand,
        hidden=hidden,
        epilog="Node numbers index the cluster's saved host list, not the head of a running job.",
    )
    @click.argument("target", required=False, type=_ClusterTargetType())
    @click.option("--cluster", "cluster_name", type=CLUSTER_NAME, default=None, help="Cluster to connect to (default: the default cluster)")
    @dry_run_option
    @click.pass_context
    def ssh(ctx, target, cluster_name, dry_run):
        """SSH into a node of a saved cluster.

        \b
        TARGET is a node number, a cluster, or both as CLUSTER/NODE:
          {app_command} ssh                  node 0 of the default cluster
          {app_command} ssh 2                node 2 of the default cluster
          {app_command} ssh lab              node 0 of cluster 'lab'
          {app_command} ssh lab/1            node 1 of cluster 'lab'
          {app_command} ssh lab/1 -- uptime  run a command instead of a shell
        A single word after -- reaches the remote shell as written ('ps | grep x');
        several words are passed as separate arguments.
        """
        from sparkrun.api import SparkrunError
        from sparkrun.api._resolve import scope_operation
        from sparkrun.core.cluster_manager import DEFAULT_CLUSTER_ENV
        from sparkrun.orchestration.ssh import interactive_ssh_argv

        try:
            parsed = parse_cluster_target(target, cluster_name)
        except ConnectTargetError as e:
            raise click.UsageError(str(e)) from None

        sctx = _get_context(ctx)
        mgr = sctx.cluster_manager
        name = parsed.ref or mgr.get_default()
        if not name:
            raise click.ClickException(
                "No cluster given and no default cluster set. Name one (CLUSTER[/NODE] or --cluster), "
                "set a default (cluster set-default <cluster>) or export %s." % DEFAULT_CLUSTER_ENV
            )
        _require_cluster(mgr, name)
        cluster = _load_cluster_or_exit(mgr, name)
        if not cluster.hosts:
            raise click.ClickException("Cluster '%s' has no hosts." % name)
        if parsed.node >= len(cluster.hosts):
            listing = ", ".join("%d=%s" % (i, h) for i, h in enumerate(cluster.hosts))
            raise click.ClickException("Cluster '%s' has no node %d (nodes: %s)." % (name, parsed.node, listing))
        host = cluster.hosts[parsed.node]

        try:
            _scoped, ssh_kwargs = scope_operation(cluster, sctx=sctx, dry_run=dry_run)
        except SparkrunError as e:
            raise click.ClickException(str(e)) from None
        command = remote_command(ctx.meta.get(_COMMAND_KEY) or [])
        # A login shell gets a PTY from ssh unasked; only a command needs -t.
        tty = command is not None and _wants_tty(command)
        try:
            argv = interactive_ssh_argv(host, command, tty=tty, **ssh_kwargs)
        except ValueError as e:
            raise click.ClickException(str(e)) from None
        banner = None if command is not None else "Connecting to %s node %d (%s)..." % (name, parsed.node, host)
        run_or_print(argv, dry_run=dry_run, banner=banner)

    return ssh


def build_exec_command():
    import click

    import sparkrun.api as api
    from sparkrun.cli._common import (
        _get_context,
        _is_cluster_id,
        _load_recipe,
        dry_run_option,
        host_options,
        resolve_host_context,
        resolve_hosts_with_metadata_fallback,
    )
    from sparkrun.orchestration.job_metadata import load_job_metadata

    from .target import remote_command, split_node_suffix

    DashDashCommand, run_or_print = _shared()

    def _sole_running_workload(sctx, hctx) -> str:
        status = api.status(hctx.host_list, cluster=hctx.cluster_def or hctx.cluster_name, sctx=sctx)
        running: dict[str, str | None] = {}
        for occupancy in status.hosts:
            for workload in occupancy.workloads:
                running.setdefault(workload.cluster_id, workload.recipe_name)
        unreachable = ", ".join(sorted(status.observation_errors))
        where = hctx.describe()
        if len(running) == 1:
            return next(iter(running))
        if not running:
            raise click.ClickException("No running workloads (%s)%s." % (where, "; could not reach " + unreachable if unreachable else ""))
        listing = "\n".join("  %s  %s" % (cid, name or "") for cid, name in sorted(running.items()))
        raise click.ClickException("Several workloads are running (%s); name one:\n%s" % (where, listing))

    @click.command("exec", cls=DashDashCommand)
    @click.argument("target", required=False)
    @host_options
    @click.option(
        "--tp", "--tensor-parallel", "tp_override", type=int, default=None, help="Tensor parallel (to match host trimming from run)"
    )
    @click.option("--port", type=int, default=None, help="Override port (to match run-time override)")
    @click.option("--served-model-name", default=None, help="Override served model name (to match run-time override)")
    @dry_run_option
    @click.pass_context
    def exec_(ctx, target, hosts, hosts_file, cluster_name, tp_override, port, served_model_name, dry_run):
        """Open a shell inside a running workload.

        \b
        TARGET is a workload (recipe name or workload ID from status), a node
        number, or both as WORKLOAD/NODE; nodes are the workload's containers,
        head first. With no workload, the only one running on the cluster:
          {app_command} exec                       head of the running workload
          {app_command} exec 1                     its node 1
          {app_command} exec qwen3-1.7b-vllm/1     node 1 of that recipe's workload
          {app_command} exec 1 -- nvidia-smi       run a command instead of a shell
        Docker and Kubernetes enter the container; the local executor enters
        the native process's environment (its venv, env and working directory).
        A single word after -- reaches the shell as written; several words are
        passed as separate arguments.
        """
        from sparkrun.cli._common import build_cluster_id_overrides

        parsed = split_node_suffix(target)
        sctx = _get_context(ctx)
        overrides = build_cluster_id_overrides(port=port, served_model_name=served_model_name, tp_override=tp_override)
        workload_id = _is_cluster_id(parsed.ref) if parsed.ref else None
        try:
            if workload_id is not None or parsed.ref is None:
                if workload_id is None:
                    hctx = resolve_host_context(hosts, hosts_file, cluster_name, sctx.config, sctx=sctx)
                    workload_id = _sole_running_workload(sctx, hctx)
                # Containers are addressed by rank over the hosts the job was
                # *placed* on, which its metadata records; a cluster's full host
                # list is only right when the job spans all of it. So hosts are
                # passed only when typed, and --cluster names the connection.
                meta = load_job_metadata(workload_id, cache_dir=str(sctx.config.cache_dir))
                if hosts or hosts_file or not (meta and meta.get("hosts")):
                    host_list, cluster = resolve_hosts_with_metadata_fallback(
                        hosts, hosts_file, cluster_name, sctx.config, meta, parsed.ref or workload_id, sctx=sctx
                    )
                else:
                    host_list, cluster = None, cluster_name
                kwargs = dict(workload_id=workload_id, hosts=host_list, cluster=cluster)
            else:
                recipe, _path, _reg = _load_recipe(sctx.config, parsed.ref)
                hctx = resolve_host_context(hosts, hosts_file, cluster_name, sctx.config, sctx=sctx)
                kwargs = dict(recipe=recipe, hosts=hctx.host_list, cluster=hctx.cluster_name, overrides=overrides)
            target_ = api.open_exec_target(node=parsed.node, cache_dir=str(sctx.config.cache_dir), sctx=sctx, **kwargs)
            command = remote_command(ctx.meta.get(_COMMAND_KEY) or [])
            argv = target_.argv(command, tty=_wants_tty(command))
        except (api.JobNotFound, api.AmbiguousWorkload, api.SparkrunError, ValueError) as e:
            raise click.ClickException(str(e)) from None
        banner = None
        if command is None:
            banner = "Entering %s node %d (%s on %s)..." % (target_.workload_id, target_.node, target_.container, target_.host)
        run_or_print(argv, dry_run=dry_run, banner=banner)

    return exec_
