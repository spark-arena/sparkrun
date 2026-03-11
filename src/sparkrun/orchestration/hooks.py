"""Pre/post lifecycle hook execution helpers.

Provides functions for running hook commands defined in recipes:
- ``pre_exec``: commands run inside containers before serve command
- ``post_exec``: commands run inside the head container after server is healthy
- ``post_commands``: commands run on the control machine after server is healthy
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from vpd.legacy.arguments import arg_substitute

logger = logging.getLogger(__name__)


def build_hook_context(
        config_chain,
        *,
        head_host: str | None = None,
        head_ip: str | None = None,
        port: int | str | None = None,
        cluster_id: str | None = None,
        container_name: str | None = None,
        cache_dir: str | None = None,
) -> dict[str, str]:
    """Build an extended variable dict for post-hook template substitution.

    Starts with all values from *config_chain* and adds host/port/URL
    variables that are only known after the server is running.

    Args:
        config_chain: VPD config chain (CLI overrides -> recipe defaults).
        head_host: Head node hostname.
        head_ip: Detected IP of head node.
        port: Serve port.
        cluster_id: Cluster identifier.
        container_name: Head container name.
        cache_dir: Effective HuggingFace cache directory.

    Returns:
        Flat dict suitable for ``{key}`` substitution.
    """
    ctx: dict[str, str] = {}

    # Pull all values from the config chain into a flat dict
    if hasattr(config_chain, 'keys'):
        for key in config_chain.keys():
            val = config_chain.get(key)
            if val is not None:
                ctx[key] = str(val)
    elif hasattr(config_chain, 'get'):
        # Minimal interface: only pull known keys
        pass

    # Add post-hook specific variables
    if head_host is not None:
        ctx["head_host"] = head_host
    if head_ip is not None:
        ctx["head_ip"] = head_ip
    if port is not None:
        ctx["port"] = str(port)
    if cluster_id is not None:
        ctx["cluster_id"] = cluster_id
    if container_name is not None:
        ctx["container_name"] = container_name
    if cache_dir is not None:
        ctx["cache_dir"] = cache_dir

    # Derive base_url if we have enough info
    if head_ip and port:
        ctx["base_url"] = "http://%s:%s/v1" % (head_ip, port)

    return ctx


def render_hook_command(cmd: str, context: dict[str, str]) -> str:
    """Render ``{key}`` placeholders in a hook command string.

    Uses the same ``arg_substitute`` used for recipe command rendering.

    Args:
        cmd: Command string with ``{key}`` placeholders.
        context: Variable dict for substitution.

    Returns:
        Rendered command string.
    """
    rendered = cmd
    last = None
    while last != rendered:
        last = rendered
        rendered = arg_substitute(rendered, context)
    return rendered


def render_hook_commands(
        commands: list[str | dict[str, str]],
        context: dict[str, str],
) -> list[str | dict[str, str]]:
    """Render ``{key}`` placeholders in a list of hook commands.

    String entries are rendered directly.  Dict entries (e.g. copy
    commands) have their string values rendered.

    Args:
        commands: List of command strings or dicts.
        context: Variable dict for substitution.

    Returns:
        New list with rendered commands.
    """
    rendered: list[str | dict[str, str]] = []
    for cmd in commands:
        if isinstance(cmd, str):
            rendered.append(render_hook_command(cmd, context))
        elif isinstance(cmd, dict):
            rendered.append({
                k: render_hook_command(v, context) if isinstance(v, str) else v
                for k, v in cmd.items()
            })
        else:
            rendered.append(cmd)
    return rendered


def run_pre_exec(
        hosts_containers: list[tuple[str, str]],
        commands: list[str | dict[str, str]],
        config_chain,
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> None:
    """Execute pre_exec commands inside containers.

    Runs on ALL containers (solo: 1, cluster: all nodes).
    Processes commands sequentially with fail-fast semantics.

    Each command entry can be:
    - A string: executed as ``docker exec <container> bash -c '<cmd>'``
    - A dict with ``copy`` key: file injection via ``docker cp``

    Args:
        hosts_containers: List of (host, container_name) pairs.
        commands: Pre_exec command list from recipe.
        config_chain: Config chain for template substitution.
        ssh_kwargs: SSH connection kwargs.
        dry_run: Show what would be done without executing.

    Raises:
        RuntimeError: If any command fails (fail-fast).
    """
    if not commands:
        return

    # Build context from config chain for rendering
    ctx: dict[str, str] = {}
    if hasattr(config_chain, 'keys'):
        for key in config_chain.keys():
            val = config_chain.get(key)
            if val is not None:
                ctx[key] = str(val)

    rendered = render_hook_commands(commands, ctx)

    logger.info("Running %d pre_exec command(s) on %d container(s)...", len(rendered), len(hosts_containers))

    for host, container_name in hosts_containers:
        for i, cmd in enumerate(rendered, 1):
            if isinstance(cmd, dict) and "copy" in cmd:
                _run_copy_command(host, container_name, cmd, ssh_kwargs, dry_run, label="pre_exec[%d]" % i)
            elif isinstance(cmd, str):
                _run_exec_command(host, container_name, cmd, ssh_kwargs, dry_run, label="pre_exec[%d]" % i)
            else:
                logger.warning("Skipping unrecognized pre_exec entry: %r", cmd)


def run_post_exec(
        head_host: str,
        container_name: str,
        commands: list[str],
        context: dict[str, str],
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
) -> None:
    """Execute post_exec commands inside the head container.

    Runs after server is confirmed healthy.  Sequential, fail-fast.

    Args:
        head_host: Head node hostname.
        container_name: Head container name.
        commands: Post_exec command list from recipe.
        context: Extended variable dict for substitution.
        ssh_kwargs: SSH connection kwargs.
        dry_run: Show what would be done without executing.

    Raises:
        RuntimeError: If any command fails (fail-fast).
    """
    if not commands:
        return

    rendered = render_hook_commands(commands, context)

    logger.info("Running %d post_exec command(s) on %s...", len(rendered), container_name)

    for i, cmd in enumerate(rendered, 1):
        if isinstance(cmd, str):
            _run_exec_command(head_host, container_name, cmd, ssh_kwargs, dry_run, label="post_exec[%d]" % i)
        else:
            logger.warning("Skipping non-string post_exec entry: %r", cmd)


def run_post_commands(
        commands: list[str],
        context: dict[str, str],
        dry_run: bool = False,
) -> None:
    """Execute post_commands on the control machine.

    Runs via ``subprocess`` on the machine where sparkrun is running.
    Sequential, fail-fast.  Stdout/stderr are streamed to terminal.

    Args:
        commands: Post_commands list from recipe.
        context: Extended variable dict for substitution.
        dry_run: Show what would be done without executing.

    Raises:
        RuntimeError: If any command fails (fail-fast).
    """
    if not commands:
        return

    rendered = render_hook_commands(commands, context)

    logger.info("Running %d post_command(s) on control machine...", len(rendered))

    for i, cmd in enumerate(rendered, 1):
        if not isinstance(cmd, str):
            logger.warning("Skipping non-string post_commands entry: %r", cmd)
            continue

        logger.info("  post_commands[%d]: %s", i, cmd)

        if dry_run:
            logger.info("  [dry-run] Would execute: %s", cmd)
            continue

        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Stream output
        if result.stdout:
            for line in result.stdout.rstrip().splitlines():
                logger.info("  | %s", line)

        if result.returncode != 0:
            raise RuntimeError(
                "post_commands[%d] failed (exit %d): %s"
                % (i, result.returncode, cmd)
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_exec_command(
        host: str,
        container_name: str,
        cmd: str,
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
        label: str = "hook",
) -> None:
    """Execute a single command inside a container via docker exec.

    Args:
        host: Target host.
        container_name: Container name.
        cmd: Command to execute.
        ssh_kwargs: SSH connection kwargs.
        dry_run: Show what would be done without executing.
        label: Human-readable label for log messages.

    Raises:
        RuntimeError: If the command exits with non-zero status.
    """
    from sparkrun.orchestration.primitives import run_script_on_host

    # Escape single quotes in command for bash -c
    escaped = cmd.replace("'", "'\\''")
    script = "docker exec %s bash -c '%s'" % (container_name, escaped)

    logger.info("  %s on %s/%s: %s", label, host, container_name, cmd)

    result = run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=600, dry_run=dry_run)

    if not dry_run and not result.success:
        error_msg = result.stderr or result.stdout or "(no output)"
        raise RuntimeError(
            "%s failed (exit %d) on %s/%s: %s"
            % (label, result.returncode, host, container_name, error_msg[:500])
        )


def _run_copy_command(
        host: str,
        container_name: str,
        cmd: dict[str, str],
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
        label: str = "hook",
) -> None:
    """Execute a file copy into a container via docker cp.

    The *cmd* dict must have a ``copy`` key with the source path.
    An optional ``dest`` key specifies the container destination
    (defaults to ``/workspace/mods/<basename>``).

    When ``source_host`` is present in *cmd*, the source files live
    on that remote host (delegated mode) rather than the control
    machine:

    - If *host* == *source_host*: files are already local on that
      host; run ``docker cp`` directly via SSH.
    - If *host* != *source_host*: rsync FROM source_host to a temp
      dir on *host*, then ``docker cp`` into the container.

    Args:
        host: Target host where the container is running.
        container_name: Container name.
        cmd: Dict with ``copy``, optional ``dest``, and optional
            ``source_host`` keys.
        ssh_kwargs: SSH connection kwargs.
        dry_run: Show what would be done without executing.
        label: Human-readable label for log messages.

    Raises:
        RuntimeError: If the copy fails.
    """
    from sparkrun.orchestration.primitives import run_script_on_host
    from sparkrun.core.hosts import is_local_host

    source = cmd["copy"]
    source_path = Path(source)
    basename = source_path.name
    dest = cmd.get("dest", "/workspace/mods/%s" % basename)
    source_host = cmd.get("source_host")

    logger.info("  %s copy %s -> %s:%s on %s", label, source, container_name, dest, host)

    if dry_run:
        return

    if source_host is not None:
        # Delegated mode: source files live on source_host, not locally.
        result = _run_delegated_copy(
            host, container_name, source, dest, source_host,
            ssh_kwargs=ssh_kwargs, label=label,
        )
    elif is_local_host(host):
        # Local: docker cp directly
        script = (
            "docker exec %(c)s mkdir -p %(dest)s\n"
            "docker cp %(src)s/. %(c)s:%(dest)s/\n"
        ) % {"c": container_name, "src": source, "dest": dest}
        result = run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=120)
    else:
        # Remote: rsync source to temp dir, then docker cp
        from sparkrun.orchestration.ssh import run_rsync_parallel
        kw = ssh_kwargs or {}

        remote_tmp = "/tmp/sparkrun_hook_%s" % basename
        run_script_on_host(host, "mkdir -p %s" % remote_tmp, ssh_kwargs=ssh_kwargs, timeout=30)

        run_rsync_parallel(
            str(source_path) + "/", [host], remote_tmp + "/",
            ssh_user=kw.get("ssh_user"),
            ssh_key=kw.get("ssh_key"),
            ssh_options=kw.get("ssh_options"),
        )

        script = (
            "docker exec %(c)s mkdir -p %(dest)s\n"
            "docker cp %(tmp)s/. %(c)s:%(dest)s/\n"
            "rm -rf %(tmp)s\n"
        ) % {"c": container_name, "dest": dest, "tmp": remote_tmp}
        result = run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=120)

    if not result.success:
        raise RuntimeError(
            "%s copy failed on %s/%s: %s"
            % (label, host, container_name, result.stderr[:500] if result.stderr else "(no output)")
        )


def _run_delegated_copy(
        host: str,
        container_name: str,
        source: str,
        dest: str,
        source_host: str,
        ssh_kwargs: dict | None = None,
        label: str = "hook",
):
    """Copy files from *source_host* into a container on *host*.

    When the target host IS the source host, the files are already
    local — run ``docker cp`` directly.  Otherwise rsync from the
    source host to a temp dir on the target, then ``docker cp``.

    Returns:
        The :class:`~sparkrun.orchestration.ssh.RemoteResult` of the
        final ``docker cp`` script.
    """
    from sparkrun.orchestration.primitives import run_script_on_host

    basename = Path(source).name
    kw = ssh_kwargs or {}

    if host == source_host:
        # Files already on this host — docker cp directly
        script = (
            "docker exec %(c)s mkdir -p %(dest)s\n"
            "docker cp %(src)s/. %(c)s:%(dest)s/\n"
        ) % {"c": container_name, "src": source, "dest": dest}
        logger.info("  %s delegated copy (local to %s): %s -> %s:%s", label, host, source, container_name, dest)
        return run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=120)
    else:
        # rsync FROM source_host to target host, then docker cp
        remote_tmp = "/tmp/sparkrun_hook_%s" % basename
        ssh_user = kw.get("ssh_user", "")
        ssh_user_prefix = "%s@" % ssh_user if ssh_user else ""

        # Build SSH options for rsync between cluster nodes
        ssh_opts_parts = []
        if kw.get("ssh_key"):
            ssh_opts_parts.append("-i %s" % kw["ssh_key"])
        if kw.get("ssh_options"):
            ssh_opts_parts.extend(kw["ssh_options"])
        ssh_opts_str = " ".join(ssh_opts_parts) if ssh_opts_parts else ""
        rsync_ssh = "-e 'ssh %s'" % ssh_opts_str if ssh_opts_str else ""

        script = (
            "set -e\n"
            "mkdir -p %(tmp)s\n"
            "rsync -a %(rsync_ssh)s %(user)s%(src_host)s:%(src)s/ %(tmp)s/\n"
            "docker exec %(c)s mkdir -p %(dest)s\n"
            "docker cp %(tmp)s/. %(c)s:%(dest)s/\n"
            "rm -rf %(tmp)s\n"
        ) % {
            "c": container_name, "dest": dest, "tmp": remote_tmp,
            "src": source, "src_host": source_host,
            "user": ssh_user_prefix, "rsync_ssh": rsync_ssh,
        }
        logger.info(
            "  %s delegated copy (rsync %s -> %s): %s -> %s:%s",
            label, source_host, host, source, container_name, dest,
        )
        return run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=300)
