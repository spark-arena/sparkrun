"""Pre/post lifecycle hook execution helpers.

Provides functions for running hook commands defined in recipes:
- ``pre_exec``: commands run inside containers before serve command
- ``post_exec``: commands run inside the head container after server is healthy
- ``post_commands``: commands run on the control machine after server is healthy
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from sparkrun.utils.text import coerce_command_list, render_template, sanitize_line_continuations

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.runtime_cache import RuntimeCacheMounts


HookPhase = Literal["pre_exec", "post_exec", "post_commands"]


@dataclass(frozen=True)
class HookLaunchContext:
    """Resolved launch metadata, separate from recipe template substitutions.

    ``hosts`` is the full selected node order, even for a hook target subset.
    Cache paths come from the mount plan used for this launch. ``volumes``
    optionally records the final mapping so overridden mounts aren't advertised.
    """

    hosts: tuple[str, ...] = ()
    cluster_id: str = ""
    recipe_name: str = ""
    runtime: str = ""
    model: str = ""
    model_revision: str = ""
    model_path: str = ""
    port: str = ""
    head_ip: str = ""
    base_url: str = ""
    runtime_cache: RuntimeCacheMounts | None = None
    volumes: Mapping[str, str] | None = None


def build_hook_launch_context(
    hosts: Sequence[str],
    config_chain=None,
    *,
    recipe: Recipe | None = None,
    runtime: str = "",
    cluster_id: str = "",
    runtime_cache: RuntimeCacheMounts | None = None,
    volumes: Mapping[str, str] | None = None,
) -> HookLaunchContext:
    """Snapshot launch metadata without probing hosts or resolving settings."""
    config = config_chain if config_chain is not None else {}

    def value(key: str) -> str:
        val = config.get(key)
        return str(val) if val is not None else ""

    return HookLaunchContext(
        hosts=tuple(dict.fromkeys(hosts)),
        cluster_id=cluster_id,
        recipe_name=(getattr(recipe, "qualified_name", None) or getattr(recipe, "name", "") or ""),
        runtime=runtime,
        model=value("model"),
        model_revision=value("model_revision"),
        model_path=value("_prepared_model_path"),
        port=value("port"),
        runtime_cache=runtime_cache,
        volumes=dict(volumes) if volumes is not None else None,
    )


def build_hook_env(
    context: HookLaunchContext,
    phase: HookPhase,
    *,
    host: str = "",
    container_name: str = "",
) -> dict[str, str]:
    """Build a fresh hook-only environment; node ranks are not GPU ranks."""
    hosts = tuple(dict.fromkeys(context.hosts))
    control = phase == "post_commands"
    rank = None if control else hosts.index(host)
    role = "control" if control else ("solo" if len(hosts) == 1 else ("head" if rank == 0 else "worker"))
    head_host = hosts[0] if hosts else ""
    cache_host = head_host if control else host
    cache = context.runtime_cache
    cache_dir = ""
    cache_leaf = ""
    if cache is not None and cache_host:
        volumes = context.volumes if context.volumes is not None else cache.volumes
        destination = cache.volumes.get(cache.leaf)
        # Only advertise the cache if its leaf still maps unambiguously to its
        # intended target after runtime-specific extra volumes have been added.
        if destination and volumes.get(cache.leaf) == destination and sum(v == destination for v in volumes.values()) == 1:
            cache_dir, cache_leaf = destination, cache.leaf

    post = phase != "pre_exec"
    return {
        "SPARKRUN_HOOK": phase,
        "SPARKRUN_CLUSTER_ID": context.cluster_id,
        "SPARKRUN_RECIPE_NAME": context.recipe_name,
        "SPARKRUN_RUNTIME": context.runtime,
        "SPARKRUN_MODEL": context.model,
        "SPARKRUN_MODEL_REVISION": context.model_revision,
        "SPARKRUN_MODEL_PATH": "" if control else context.model_path,
        "SPARKRUN_NUM_NODES": str(len(hosts)) if hosts else "",
        "SPARKRUN_NODE_RANK": str(rank) if rank is not None else "",
        "SPARKRUN_NODE_HOST": "" if control else host,
        "SPARKRUN_NODE_ROLE": role,
        "SPARKRUN_IS_HEAD": "1" if rank == 0 else "0",
        "SPARKRUN_HEAD_HOST": head_host,
        "SPARKRUN_CONTAINER_NAME": "" if control else container_name,
        "SPARKRUN_PORT": context.port,
        "SPARKRUN_HEAD_IP": context.head_ip if post and context.head_ip != "<HEAD_IP>" else "",
        "SPARKRUN_BASE_URL": context.base_url if post and "<HEAD_IP>" not in context.base_url else "",
        "SPARKRUN_RUNTIME_CACHE_ENABLED": "1" if cache_leaf else "0",
        "SPARKRUN_RUNTIME_CACHE_DIR": "" if control else cache_dir,
        "SPARKRUN_RUNTIME_CACHE_HOST_DIR": cache_leaf,
        "SPARKRUN_RUNTIME_CACHE_HOST": cache_host if cache_leaf else "",
    }


HOOK_ENV_KEYS = frozenset(build_hook_env(HookLaunchContext(), "post_commands"))
_PRE_SERVE_CONTEXT: ContextVar[HookLaunchContext | None] = ContextVar("sparkrun_pre_serve_context", default=None)
_LAUNCH_CONTEXTS: ContextVar[list[HookLaunchContext] | None] = ContextVar("sparkrun_hook_launches", default=None)


@contextmanager
def capture_hook_launch_contexts() -> Iterator[list[HookLaunchContext]]:
    """Retain actual runtime mount snapshots for post hooks, without plugin state."""
    contexts: list[HookLaunchContext] = []
    token = _LAUNCH_CONTEXTS.set(contexts)
    try:
        yield contexts
    finally:
        _LAUNCH_CONTEXTS.reset(token)


@contextmanager
def hook_launch_scope(context: HookLaunchContext) -> Iterator[None]:
    """Carry context across legacy overrides without changing their signature."""
    contexts = _LAUNCH_CONTEXTS.get()
    if contexts is not None:
        contexts.append(context)
    token = _PRE_SERVE_CONTEXT.set(context)
    try:
        yield
    finally:
        _PRE_SERVE_CONTEXT.reset(token)


def current_hook_launch_context() -> HookLaunchContext | None:
    """Context of the current runtime pre-serve invocation, if any."""
    return _PRE_SERVE_CONTEXT.get()


def _log_hook_target(env: Mapping[str, str]) -> None:
    logger.info(
        "  %s context: host=%s rank=%s/%s role=%s runtime_cache=%s (host=%s:%s)",
        env["SPARKRUN_HOOK"],
        env["SPARKRUN_NODE_HOST"] or "<control>",
        env["SPARKRUN_NODE_RANK"],
        env["SPARKRUN_NUM_NODES"],
        env["SPARKRUN_NODE_ROLE"],
        env["SPARKRUN_RUNTIME_CACHE_DIR"] or "<none>",
        env["SPARKRUN_RUNTIME_CACHE_HOST"],
        env["SPARKRUN_RUNTIME_CACHE_HOST_DIR"],
    )


# Hide the opening brace of metadata shell expansions during substitution.
# This also allows ordinary {key} placeholders inside parameter defaults.
# YAML forbids the sentinel in scalar content, so it cannot collide with a
# recipe's text. Protect values too: nested config references can introduce
# shell expressions on later template passes.
_HOOK_ENV_OPEN = re.compile(r"\$\{(?=[#!]?(?:%s)(?![A-Z0-9_]))" % "|".join(sorted(HOOK_ENV_KEYS)))
_SHELL_LBRACE = "\x02"


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
    if hasattr(config_chain, "keys"):
        for key in config_chain.keys():
            val = config_chain.get(key)
            if val is not None:
                ctx[key] = str(val)
    elif hasattr(config_chain, "get"):
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

    Uses the same brace-masking renderer as recipe command rendering, so a
    placeholder inside a JSON payload resolves here too — a hook like
    ``curl -d '{"model":"{model}"}'`` otherwise reached the shell with a
    literal ``{model}``, because vpd's regex matched from the opening brace
    of the JSON object through the placeholder's closing brace and restored
    the whole span verbatim.

    Unlike a recipe command, ``{{`` is left as two literal braces rather than
    collapsed to one: a hook body is arbitrary shell, where a doubled brace is
    plausibly meant literally (bash brace expansion, an ``awk`` program), and
    nothing here needs the escape — a JSON payload can be written with plain
    braces because the mask already keeps it away from vpd's regex.

    Broken line-continuations *are* repaired the same way as in a recipe
    command, though.  The "arbitrary shell" argument that spares ``{{`` does
    not extend to ``\\<blank><newline>``: a doubled brace has legitimate
    meanings, an escaped blank at end of line has none, and a multi-line
    ``pre_exec`` breaks on one exactly as a ``command:`` template does.

    Args:
        cmd: Command string with ``{key}`` placeholders.
        context: Variable dict for substitution.

    Returns:
        Rendered command string.
    """

    def protect(value: str) -> str:
        return _HOOK_ENV_OPEN.sub("$" + _SHELL_LBRACE, value)

    template_context = {key: protect(val) if isinstance(val, str) else val for key, val in context.items()}
    rendered = render_template(protect(cmd), template_context).replace(_SHELL_LBRACE, "{")
    return sanitize_line_continuations(rendered)


def render_hook_commands(
    commands: Sequence[str | dict[str, str]],
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
    commands = coerce_command_list(commands)

    rendered: list[str | dict[str, str]] = []
    for cmd in commands:
        if isinstance(cmd, str):
            rendered.append(render_hook_command(cmd, context))
        elif isinstance(cmd, dict):
            rendered.append({k: render_hook_command(v, context) if isinstance(v, str) else v for k, v in cmd.items()})
        else:
            rendered.append(cmd)
    return rendered


def _confirm_hook_execution(hook_label: str, commands: Sequence[str | dict[str, str]], trust: bool) -> None:
    """Shared trust-gating prompt for hook commands.

    When *trust* is True, returns immediately (no prompt).
    When *trust* is False:
    - Logs the commands as a warning,
    - Refuses to proceed if stdin is not a TTY (raises RuntimeError),
    - Otherwise asks the user to confirm via click.confirm.

    Args:
        hook_label: Human-readable hook name (e.g. ``"pre_exec"``).
        commands: Hook command list (string entries are logged; dict
            entries are summarized).
        trust: When True, bypass confirmation entirely.

    Raises:
        RuntimeError: If *trust* is False and either stdin is not a TTY
            or the user declines.
    """
    import sys

    if trust:
        return

    logger.warning("Recipe %s will execute inside containers:", hook_label)
    for i, cmd in enumerate(commands, 1):
        if isinstance(cmd, str):
            logger.warning("  [%d] %s", i, cmd)
        elif isinstance(cmd, dict) and "copy" in cmd:
            logger.warning("  [%d] copy %s -> %s", i, cmd.get("copy"), cmd.get("dest", "<default>"))
        else:
            logger.warning("  [%d] %r", i, cmd)

    if not sys.stdin.isatty():
        raise RuntimeError(
            "%s require confirmation but stdin is not a TTY. Use --trust to allow %s from third-party registries."
            % (hook_label, hook_label)
        )
    import click

    if not click.confirm("Allow these %s to run inside containers?" % hook_label, default=False):
        raise RuntimeError("%s execution cancelled by user." % hook_label)


def run_pre_exec(
    hosts_containers: list[tuple[str, str]],
    commands: Sequence[str | dict[str, str]],
    config_chain,
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    trust: bool = False,
    cache_dir: str | None = None,
    *,
    launch_context: HookLaunchContext | None = None,
) -> None:
    """Execute pre_exec commands inside containers.

    Runs on ALL containers (solo: 1, cluster: all nodes).
    Processes commands sequentially with fail-fast semantics.

    Each command entry can be:
    - A string: executed as ``docker exec <container> bash -c '<cmd>'``
    - A dict with ``copy`` key: file injection via ``docker cp``

    When *trust* is False (the default), commands from third-party
    registries require explicit user confirmation before executing.
    If stdin is not a TTY, execution is refused and a RuntimeError is
    raised directing the user to pass ``--trust``.

    Args:
        hosts_containers: List of (host, container_name) pairs.
        commands: Pre_exec command list from recipe. A bare string is
            accepted and treated as a single command.
        config_chain: Config chain for template substitution.
        ssh_kwargs: SSH connection kwargs.
        dry_run: Show what would be done without executing.
        trust: Skip confirmation prompt (auto-trust the commands).
        cache_dir: Effective HuggingFace cache directory on remote hosts.
            Threaded from the launcher so disk-space failure messages
            show the correct path rather than the ``$HOME/.cache/huggingface``
            fallback.
        launch_context: Full resolved launch metadata for the hook environment.

    Raises:
        RuntimeError: If any command fails (fail-fast), or if *trust*
            is False and stdin is not a TTY.
    """
    if not commands:
        return

    commands = coerce_command_list(commands)

    _confirm_hook_execution("pre_exec", commands, trust)

    # Build context from config chain for rendering
    ctx: dict[str, str] = {}
    if hasattr(config_chain, "keys"):
        for key in config_chain.keys():
            val = config_chain.get(key)
            if val is not None:
                ctx[key] = str(val)

    rendered = render_hook_commands(commands, ctx)
    launch_context = launch_context or build_hook_launch_context([host for host, _ in hosts_containers], config_chain)

    logger.info("Running %d pre_exec command(s) on %d container(s)...", len(rendered), len(hosts_containers))

    for host, container_name in hosts_containers:
        hook_env = build_hook_env(launch_context, "pre_exec", host=host, container_name=container_name)
        _log_hook_target(hook_env)
        for i, cmd in enumerate(rendered, 1):
            if isinstance(cmd, dict) and "copy" in cmd:
                _run_copy_command(host, container_name, cmd, ssh_kwargs, dry_run, label="pre_exec[%d]" % i, cache_dir=cache_dir)
            elif isinstance(cmd, str):
                _run_exec_command(host, container_name, cmd, ssh_kwargs, dry_run, label="pre_exec[%d]" % i, env=hook_env)
            else:
                logger.warning("Skipping unrecognized pre_exec entry: %r", cmd)


def run_post_exec(
    head_host: str,
    container_name: str,
    commands: str | Sequence[str],
    context: dict[str, str],
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    trust: bool = False,
    cache_dir: str | None = None,  # noqa: ARG001 — reserved for future copy-type post_exec entries
    *,
    launch_context: HookLaunchContext | None = None,
) -> None:
    """Execute post_exec commands inside the head container.

    Runs after server is confirmed healthy.  Sequential, fail-fast.

    When *trust* is False (the default), commands from third-party
    registries require explicit user confirmation before executing.
    If stdin is not a TTY, execution is refused and a RuntimeError is
    raised directing the user to pass ``--trust``.

    Args:
        head_host: Head node hostname.
        container_name: Head container name.
        commands: Post_exec command list from recipe. A bare string is
            accepted and treated as a single command.
        context: Extended variable dict for substitution.
        ssh_kwargs: SSH connection kwargs.
        dry_run: Show what would be done without executing.
        trust: Skip confirmation prompt (auto-trust the commands).
        launch_context: Full launch metadata, including all participating nodes.

    Raises:
        RuntimeError: If any command fails (fail-fast), or if *trust*
            is False and stdin is not a TTY.
    """
    if not commands:
        return

    commands = coerce_command_list(commands)

    _confirm_hook_execution("post_exec", commands, trust)

    rendered = render_hook_commands(commands, context)
    launch_context = launch_context or _post_launch_context(context, head_host)
    hook_env = build_hook_env(launch_context, "post_exec", host=head_host, container_name=container_name)
    _log_hook_target(hook_env)

    logger.info("Running %d post_exec command(s) on %s...", len(rendered), container_name)

    for i, cmd in enumerate(rendered, 1):
        if isinstance(cmd, str):
            _run_exec_command(head_host, container_name, cmd, ssh_kwargs, dry_run, label="post_exec[%d]" % i, env=hook_env)
        else:
            logger.warning("Skipping non-string post_exec entry: %r", cmd)


def run_post_commands(
    commands: str | Sequence[str],
    context: dict[str, str],
    dry_run: bool = False,
    trust: bool = False,
    *,
    launch_context: HookLaunchContext | None = None,
) -> None:
    """Execute post_commands on the control machine.

    Runs via ``subprocess`` on the machine where sparkrun is running.
    Sequential, fail-fast.  Stdout/stderr are streamed to terminal.

    When *trust* is False (the default), commands from third-party
    registries require explicit user confirmation before executing.
    If stdin is not a TTY, execution is refused and a RuntimeError is
    raised directing the user to pass ``--trust``.

    Args:
        commands: Post_commands list from recipe. A bare string is
            accepted and treated as a single command.
        context: Extended variable dict for substitution.
        dry_run: Show what would be done without executing.
        trust: Skip confirmation prompt (auto-trust the commands).
        launch_context: Full launch metadata; this shell has no workload rank.

    Raises:
        RuntimeError: If any command fails (fail-fast), or if *trust*
            is False and stdin is not a TTY.
    """
    import sys

    if not commands:
        return

    commands = coerce_command_list(commands)

    if not trust:
        logger.warning("Recipe post_commands will execute on this machine:")
        for i, cmd in enumerate(commands, 1):
            if isinstance(cmd, str):
                logger.warning("  [%d] %s", i, cmd)
        if not sys.stdin.isatty():
            raise RuntimeError(
                "post_commands require confirmation but stdin is not a TTY. Use --trust to allow post_commands from third-party registries."
            )
        import click

        if not click.confirm("Allow these post_commands to run on this machine?", default=False):
            raise RuntimeError("post_commands execution cancelled by user.")

    rendered = render_hook_commands(commands, context)
    hook_env = build_hook_env(launch_context or _post_launch_context(context), "post_commands")
    _log_hook_target(hook_env)

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
            env={**os.environ, **hook_env},
        )

        # Stream output
        if result.stdout:
            for line in result.stdout.rstrip().splitlines():
                logger.info("  | %s", line)

        if result.returncode != 0:
            raise RuntimeError("post_commands[%d] failed (exit %d): %s" % (i, result.returncode, cmd))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _post_launch_context(context: dict[str, str], head_host: str = "") -> HookLaunchContext:
    """Compatibility for direct post-hook callers without launch metadata."""
    head_host = head_host or context.get("head_host", "")
    return replace(
        build_hook_launch_context([head_host] if head_host else [], context, cluster_id=context.get("cluster_id", "")),
        head_ip=context.get("head_ip", ""),
        base_url=context.get("base_url", ""),
    )


def _run_exec_command(
    host: str,
    container_name: str,
    cmd: str,
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    label: str = "hook",
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    """Execute a single command inside a container via docker exec.

    Args:
        host: Target host.
        container_name: Container name.
        cmd: Command to execute.
        ssh_kwargs: SSH connection kwargs.
        dry_run: Show what would be done without executing.
        label: Human-readable label for log messages.
        env: Hook-only metadata exported to this exec and its child processes.

    Raises:
        RuntimeError: If the command exits with non-zero status.
    """
    # Use base64 encoding to safely pass the command through bash -c
    from sparkrun.orchestration.primitives import run_script_on_host
    from sparkrun.utils.shell import b64_wrap_bash, quote

    # TODO: this should delegate via executor implementation; this should allow control over user
    env_args = "".join(" --env %s" % quote("%s=%s" % (key, value)) for key, value in sorted((env or {}).items()))
    script = "docker exec --user root%s %s bash -c %s" % (env_args, quote(container_name), b64_wrap_bash(cmd))

    logger.info("  %s on %s/%s: %s", label, host, container_name, cmd)

    result = run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=600, dry_run=dry_run)

    if not dry_run and not result.success:
        error_msg = result.stderr or result.stdout or "(no output)"
        raise RuntimeError("%s failed (exit %d) on %s/%s: %s" % (label, result.returncode, host, container_name, error_msg[:500]))


def _run_copy_command(
    host: str,
    container_name: str,
    cmd: dict[str, str],
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    label: str = "hook",
    cache_dir: str | None = None,
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
        cache_dir: Effective HuggingFace cache directory on remote hosts.
            Used when reporting disk-space failures so the cache-status
            table probes the correct path.  Falls back to
            ``"$HOME/.cache/huggingface"`` when ``None``.

    Raises:
        RuntimeError: If the copy fails.
    """
    from sparkrun.orchestration.primitives import run_script_on_host, should_run_locally
    from sparkrun.orchestration.transfer import (
        TransferError,
        TransferFailure,
        classify_rsync_failure,
        map_transfer_failures_detailed,
        present_and_raise_transfer_failure,
    )

    source = cmd["copy"]
    source_path = Path(source).expanduser()
    basename = source_path.name
    dest = cmd.get("dest", "/workspace/mods/%s" % basename)
    source_host = cmd.get("source_host")
    kw = ssh_kwargs or {}

    logger.info("  %s copy %s -> %s:%s on %s", label, source, container_name, dest, host)

    if dry_run:
        return

    if source_host is not None:
        # Delegated mode: source files live on source_host, not locally.
        # Validate against injection, then convert ~/… to $HOME/… so the
        # path expands correctly inside double-quoted remote shell scripts.
        from sparkrun.utils.shell import safe_remote_path

        remote_source = safe_remote_path(source)
        result = _run_delegated_copy(
            host,
            container_name,
            remote_source,
            dest,
            source_host,
            ssh_kwargs=ssh_kwargs,
            label=label,
        )
        if not result.success:
            raise TransferError(
                "%s copy failed on %s/%s: %s" % (label, host, container_name, result.stderr[:500] if result.stderr else "(no output)")
            )
        return
    elif should_run_locally(host, kw.get("ssh_user")):
        # Local: docker cp directly
        script = ("set -e\ndocker exec --user root %(c)s mkdir -p %(dest)s\ndocker cp %(src)s/. %(c)s:%(dest)s/\n") % {
            "c": container_name,
            "src": source_path,
            "dest": dest,
        }
        result = run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=120)
        if not result.success:
            raise TransferError(
                "%s copy failed on %s/%s: %s" % (label, host, container_name, result.stderr[:500] if result.stderr else "(no output)")
            )
    else:
        # Remote: rsync source to temp dir, then docker cp.
        # Check steps in order so the FIRST failure (root cause) is reported,
        # not a downstream artifact (e.g. lstat on a dir that never got created).
        from sparkrun.orchestration.ssh import run_rsync_parallel

        from sparkrun.core.application_profile import resource_name

        remote_tmp = "/tmp/" + resource_name("_hook_%s" % basename)
        mkdir_result = run_script_on_host(host, "mkdir -p %s" % remote_tmp, ssh_kwargs=ssh_kwargs, timeout=30)
        _effective_cache_dir = cache_dir if cache_dir is not None else "$HOME/.cache/huggingface"

        if not mkdir_result.success:
            # Synthesise a RemoteResult-like failure record so classify_rsync_failure
            # can pattern-match the stderr (e.g. "No space left on device").
            failure = TransferFailure(
                host=host,
                reason=classify_rsync_failure(mkdir_result),
                detail=(mkdir_result.stderr or mkdir_result.stdout or "")[:400],
            )
            present_and_raise_transfer_failure(
                [failure],
                operation="%s copy failed" % label,
                cache_status_hosts=[host],
                cache_dir=_effective_cache_dir,
                ssh_kwargs=ssh_kwargs,
                label="copy",
            )

        rsync_results = run_rsync_parallel(
            str(source_path) + "/",
            [host],
            remote_tmp + "/",
            ssh_user=kw.get("ssh_user"),
            ssh_key=kw.get("ssh_key"),
            ssh_options=kw.get("ssh_options"),
        )
        rsync_failures = map_transfer_failures_detailed(rsync_results, [host], [host])
        if rsync_failures:
            present_and_raise_transfer_failure(
                rsync_failures,
                operation="%s copy failed" % label,
                cache_status_hosts=[host],
                cache_dir=_effective_cache_dir,
                ssh_kwargs=ssh_kwargs,
                label="copy",
            )

        script = ("set -e\ndocker exec --user root %(c)s mkdir -p %(dest)s\ndocker cp %(tmp)s/. %(c)s:%(dest)s/\nrm -rf %(tmp)s\n") % {
            "c": container_name,
            "dest": dest,
            "tmp": remote_tmp,
        }
        result = run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=120)
        if not result.success:
            raise TransferError(
                "%s copy failed on %s/%s: %s" % (label, host, container_name, result.stderr[:500] if result.stderr else "(no output)")
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
    from sparkrun.utils.shell import safe_remote_path, validate_hostname

    validate_hostname(source_host)
    dest = safe_remote_path(dest)

    basename = Path(source).name
    kw = ssh_kwargs or {}

    if host == source_host:
        # Files already on this host — docker cp directly
        script = ('set -e\ndocker exec --user root %(c)s mkdir -p %(dest)s\ndocker cp "%(src)s"/. %(c)s:%(dest)s/\n') % {
            "c": container_name,
            "src": source,
            "dest": dest,
        }
        logger.info("  %s delegated copy (local to %s): %s -> %s:%s", label, host, source, container_name, dest)
        return run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=120)
    else:
        # rsync FROM source_host to target host, then docker cp
        from sparkrun.core.application_profile import resource_name

        remote_tmp = "/tmp/" + resource_name("_hook_%s" % basename)
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
            'rsync -a %(rsync_ssh)s %(user)s%(src_host)s:"%(src)s"/ %(tmp)s/\n'
            "docker exec --user root %(c)s mkdir -p %(dest)s\n"
            "docker cp %(tmp)s/. %(c)s:%(dest)s/\n"
            "rm -rf %(tmp)s\n"
        ) % {
            "c": container_name,
            "dest": dest,
            "tmp": remote_tmp,
            "src": source,
            "src_host": source_host,
            "user": ssh_user_prefix,
            "rsync_ssh": rsync_ssh,
        }
        logger.info(
            "  %s delegated copy (rsync %s -> %s): %s -> %s:%s",
            label,
            source_host,
            host,
            source,
            container_name,
            dest,
        )
        return run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=300)
