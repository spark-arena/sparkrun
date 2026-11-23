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
from typing import TYPE_CHECKING, Literal, cast

from sparkrun.utils.text import coerce_command_list, excerpt, render_template, sanitize_line_continuations

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.runtime_cache import RuntimeCacheMounts


HookPhase = Literal["pre_exec", "post_exec", "post_commands"]
#: A recipe hook entry: a command string, ``{command, timeout?}``, or (pre_exec
#: only) ``{copy, dest?, source_host?}``.
HookEntry = str | Mapping[str, object]


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


# ---------------------------------------------------------------------------
# Hook timeouts and entry shapes
# ---------------------------------------------------------------------------

#: Wall-clock ceiling for one hook command when neither its entry
#: (``{command: ..., timeout: N}``) nor ``hooks.timeout_s`` in config.yaml
#: names one.  Was a hardcoded 600s, which killed legitimate first-run
#: downloads (issue #326).  The default has no "unbounded" spelling: a hook
#: that hangs would otherwise hold the launch (and, under systemd, the unit's
#: start) forever.  A hook that needs longer says so per entry, where
#: ``timeout: 0`` opts that one command out of any limit.
DEFAULT_HOOK_TIMEOUT_S = 3600.0
#: Grace a timed-out hook gets between TERM and KILL.
_HOOK_KILL_AFTER_S = 10
#: How much longer the control side waits than the hook's own deadline, so the
#: in-container watchdog fires first and takes the hook's process group with
#: it.  Killing only the ssh / ``docker exec`` client leaves the hook running.
_HOOK_TIMEOUT_GRACE_S = 30

COMMAND_ENTRY_KEYS = frozenset({"command", "timeout"})
COPY_ENTRY_KEYS = frozenset({"copy", "dest", "source_host"})

_HOOK_TIMEOUT: ContextVar[float | None] = ContextVar("sparkrun_hook_timeout", default=None)


@contextmanager
def hook_timeout_scope(seconds: float | None) -> Iterator[None]:
    """Set the default hook timeout for hooks run within this scope.

    The launcher sets it from ``SparkrunConfig.hook_timeout_s``; a contextvar
    because ``pre_exec`` runs from inside runtime plugins whose ``_pre_serve``
    signature carries no config.
    """
    token = _HOOK_TIMEOUT.set(seconds)
    try:
        yield
    finally:
        _HOOK_TIMEOUT.reset(token)


def resolve_hook_timeout(default_timeout_s: float | None = None) -> float:
    """Default hook timeout: explicit argument → launch scope → built-in default.

    Always finite: ``0`` is accepted only on an entry (``timeout: 0``), as a
    deliberate per-command opt-out, never as the default for every hook.
    """
    for candidate in (default_timeout_s, _HOOK_TIMEOUT.get()):
        try:
            return parse_hook_timeout(candidate)
        except ValueError:
            continue
    return DEFAULT_HOOK_TIMEOUT_S


def parse_hook_timeout(value: object, *, allow_unbounded: bool = False) -> float:
    """Validate a ``timeout`` (seconds); raises ``ValueError``.

    Numbers only: a positive int or float.  Booleans are rejected (YAML
    ``timeout: yes`` is not a duration), and so are strings, which would
    otherwise read differently before and after placeholder rendering.  With
    *allow_unbounded* (an entry's own ``timeout:``), ``0`` means no timeout
    and returns ``math.inf``; ``.inf`` itself is not a spelling of it.
    """
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout must be a number of seconds, got %r" % (value,))
    if allow_unbounded and value == 0:
        return math.inf
    if not math.isfinite(value) or value <= 0:
        expected = "a positive, finite number of seconds" + (", or 0 for no timeout" if allow_unbounded else "")
        raise ValueError("timeout must be %s, got %r" % (expected, value))
    return float(value)


def hook_entry_problems(entry: object, *, allow_copy: bool) -> list[str]:
    """Why *entry* is not a runnable hook entry; empty when it is.

    One answer shared by ``recipe validate`` and execution, so validation
    refuses exactly what a launch would.  Shapes: a string, a
    ``{command, timeout?}`` mapping, or (``pre_exec`` only) a
    ``{copy, dest?, source_host?}`` mapping.
    """
    if isinstance(entry, str):
        return []
    if not isinstance(entry, Mapping):
        return ["entry must be a string or a mapping, got %s" % type(entry).__name__]
    keys = set(entry)
    if "copy" in keys and "command" in keys:
        return ["entry sets both 'copy' and 'command'; split it into two entries"]
    if "copy" in keys:
        if not allow_copy:
            return ["'copy' entries are only supported in pre_exec"]
        unknown = keys - COPY_ENTRY_KEYS
        problems = ["unknown key(s) %s for a copy entry (allowed: %s)" % (sorted(unknown), sorted(COPY_ENTRY_KEYS))] if unknown else []
        for key in keys & COPY_ENTRY_KEYS:
            if not isinstance(entry[key], str) or not entry[key].strip():
                problems.append("'%s' must be a non-empty string" % key)
        return problems
    if "command" not in keys:
        expected = "'command' or 'copy'" if allow_copy else "'command'"
        return ["mapping entry has no %s key (keys: %s)" % (expected, sorted(keys))]
    problems: list[str] = []
    if not isinstance(entry["command"], str) or not entry["command"].strip():
        problems.append("'command' must be a non-empty string")
    if keys - COMMAND_ENTRY_KEYS:
        problems.append(
            "unknown key(s) %s for a command entry (allowed: %s)" % (sorted(keys - COMMAND_ENTRY_KEYS), sorted(COMMAND_ENTRY_KEYS))
        )
    if "timeout" in keys:
        try:
            parse_hook_timeout(entry["timeout"], allow_unbounded=True)
        except ValueError as e:
            problems.append(str(e))
    return problems


def describe_hook_entry(entry: object) -> str:
    """One-line text of a hook entry, for trust prompts and listings."""
    if isinstance(entry, Mapping):
        if "copy" in entry:
            return "copy %s -> %s" % (entry.get("copy"), entry.get("dest", "<default>"))
        if "command" in entry:
            text = str(entry.get("command"))
            if "timeout" not in entry:
                return text
            return "%s  (%s)" % (text, "no timeout" if entry["timeout"] == 0 else "timeout %ss" % entry["timeout"])
    return str(entry)


def _hook_steps(entries: Sequence[object], phase: HookPhase) -> list[Mapping[str, str] | tuple[str, float | None]]:
    """Validate rendered entries up front: a copy mapping, or ``(command, timeout_s)``.

    Raises ``RuntimeError`` for a malformed entry before anything runs.  Hooks
    are fail-fast, and silently skipping a step the recipe asked for (what
    happened to these before) launches something other than the recipe.
    """
    steps: list[Mapping[str, str] | tuple[str, float | None]] = []
    for i, entry in enumerate(entries, 1):
        problems = hook_entry_problems(entry, allow_copy=phase == "pre_exec")
        if problems:
            raise RuntimeError("%s[%d] is not a valid hook entry: %s" % (phase, i, "; ".join(problems)))
        if isinstance(entry, str):
            steps.append((entry, None))
        else:
            assert isinstance(entry, Mapping)  # validated above
            if "copy" in entry:
                steps.append(cast(Mapping[str, str], entry))  # copy values were validated as strings
            else:
                command = entry["command"]
                assert isinstance(command, str)
                timeout = entry.get("timeout")
                steps.append((command, parse_hook_timeout(timeout, allow_unbounded=True) if timeout is not None else None))
    return steps


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
    commands: Sequence[HookEntry],
    context: dict[str, str],
) -> list[HookEntry]:
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

    rendered: list[HookEntry] = []
    for cmd in commands:
        if isinstance(cmd, str):
            rendered.append(render_hook_command(cmd, context))
        elif isinstance(cmd, dict):
            rendered.append({k: render_hook_command(v, context) if isinstance(v, str) else v for k, v in cmd.items()})
        else:
            rendered.append(cmd)
    return rendered


def _confirm_hook_execution(
    hook_label: str,
    commands: Sequence[HookEntry],
    trust: bool,
    *,
    dry_run: bool = False,
    where: str = "inside containers",
) -> None:
    """Shared trust-gating prompt for hook commands.

    When *trust* is True, returns immediately (no prompt).
    When *trust* is False:
    - Logs the commands as a warning,
    - Under *dry_run* returns there: nothing executes, so nothing needs
      authorizing — the log says what a real run would ask about,
    - Refuses to proceed if stdin is not a TTY (raises RuntimeError),
    - Otherwise asks the user to confirm via click.confirm.

    Args:
        hook_label: Human-readable hook name (e.g. ``"pre_exec"``).
        commands: Hook command list (string entries are logged; dict
            entries are summarized).
        trust: When True, bypass confirmation entirely.
        dry_run: Report instead of prompting.
        where: Where the commands run, for the prompt ("on this machine").

    Raises:
        RuntimeError: If *trust* is False, this is not a dry run, and either
            stdin is not a TTY or the user declines.
    """
    import sys

    if trust:
        return

    if dry_run:
        logger.warning("Recipe %s would require authorization (--trust) to run:", hook_label)
        for i, cmd in enumerate(commands, 1):
            logger.warning("  [%d] %s", i, describe_hook_entry(cmd))
        return

    logger.warning("Recipe %s will execute %s:", hook_label, where)
    for i, cmd in enumerate(commands, 1):
        logger.warning("  [%d] %s", i, describe_hook_entry(cmd))

    if not sys.stdin.isatty():
        raise RuntimeError(
            "%s require confirmation but stdin is not a TTY. Use --trust to allow %s from third-party registries."
            % (hook_label, hook_label)
        )
    import click

    if not click.confirm("Allow these %s to run %s?" % (hook_label, where), default=False):
        raise RuntimeError("%s execution cancelled by user." % hook_label)


def run_pre_exec(
    hosts_containers: list[tuple[str, str]],
    commands: Sequence[HookEntry],
    config_chain,
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    trust: bool = False,
    cache_dir: str | None = None,
    *,
    launch_context: HookLaunchContext | None = None,
    default_timeout_s: float | None = None,
    executor=None,
) -> None:
    """Execute pre_exec commands inside containers.

    Runs on ALL containers (solo: 1, cluster: all nodes).
    Processes commands sequentially with fail-fast semantics.

    Each command entry can be:
    - A string: executed in the workload through its executor
      (``docker exec`` for a container, the native environment for ``local``)
    - A dict with ``command`` and optional ``timeout`` (seconds)
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
        default_timeout_s: Per-command timeout for entries that name none;
            ``None`` uses the launch's ``hooks.timeout_s`` (see
            :func:`resolve_hook_timeout`).
        executor: The workload's executor (per host via ``for_host``);
            ``None`` means docker.

    Raises:
        RuntimeError: If any command fails (fail-fast), or if *trust*
            is False and stdin is not a TTY.
    """
    if not commands:
        return

    commands = coerce_command_list(commands)
    _hook_steps(commands, "pre_exec")  # refuse a malformed entry before asking about it

    _confirm_hook_execution("pre_exec", commands, trust, dry_run=dry_run)

    # Build context from config chain for rendering
    ctx: dict[str, str] = {}
    if hasattr(config_chain, "keys"):
        for key in config_chain.keys():
            val = config_chain.get(key)
            if val is not None:
                ctx[key] = str(val)

    rendered = render_hook_commands(commands, ctx)
    launch_context = launch_context or build_hook_launch_context([host for host, _ in hosts_containers], config_chain)
    default_timeout = resolve_hook_timeout(default_timeout_s)
    steps = _hook_steps(rendered, "pre_exec")
    _require_copy_support(steps, hosts_containers, executor)

    logger.info("Running %d pre_exec command(s) on %d container(s)...", len(rendered), len(hosts_containers))

    for host, container_name in hosts_containers:
        hook_env = build_hook_env(launch_context, "pre_exec", host=host, container_name=container_name)
        _log_hook_target(hook_env)
        for i, step in enumerate(steps, 1):
            if isinstance(step, Mapping):
                _run_copy_command(host, container_name, step, ssh_kwargs, dry_run, label="pre_exec[%d]" % i, cache_dir=cache_dir)
            else:
                cmd, timeout_s = step
                _run_exec_command(
                    host,
                    container_name,
                    cmd,
                    ssh_kwargs,
                    dry_run,
                    label="pre_exec[%d]" % i,
                    env=hook_env,
                    timeout_s=timeout_s or default_timeout,
                    executor=executor,
                )


def run_post_exec(
    head_host: str,
    container_name: str,
    commands: str | Sequence[HookEntry],
    context: dict[str, str],
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    trust: bool = False,
    cache_dir: str | None = None,  # noqa: ARG001 — reserved for future copy-type post_exec entries
    *,
    launch_context: HookLaunchContext | None = None,
    default_timeout_s: float | None = None,
    executor=None,
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
        default_timeout_s: Per-command timeout for entries that name none.
        executor: The workload's executor; ``None`` means docker.

    Raises:
        RuntimeError: If any command fails (fail-fast), or if *trust*
            is False and stdin is not a TTY.
    """
    if not commands:
        return

    commands = coerce_command_list(commands)
    _hook_steps(commands, "post_exec")

    _confirm_hook_execution("post_exec", commands, trust, dry_run=dry_run)

    rendered = render_hook_commands(commands, context)
    launch_context = launch_context or _post_launch_context(context, head_host)
    hook_env = build_hook_env(launch_context, "post_exec", host=head_host, container_name=container_name)
    _log_hook_target(hook_env)

    default_timeout = resolve_hook_timeout(default_timeout_s)
    steps = _hook_steps(rendered, "post_exec")

    logger.info("Running %d post_exec command(s) on %s...", len(rendered), container_name)

    for i, step in enumerate(steps, 1):
        assert isinstance(step, tuple)  # post_exec rejects copy entries
        cmd, timeout_s = step
        _run_exec_command(
            head_host,
            container_name,
            cmd,
            ssh_kwargs,
            dry_run,
            label="post_exec[%d]" % i,
            env=hook_env,
            timeout_s=timeout_s or default_timeout,
            executor=executor,
        )


def run_post_commands(
    commands: str | Sequence[HookEntry],
    context: dict[str, str],
    dry_run: bool = False,
    trust: bool = False,
    *,
    launch_context: HookLaunchContext | None = None,
    default_timeout_s: float | None = None,
) -> None:
    """Execute post_commands on the control machine.

    Runs via ``subprocess`` on the machine where sparkrun is running.
    Sequential, fail-fast.  Output is logged once each command finishes.
    Each command runs in its own process group, which a timeout kills whole.

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
        default_timeout_s: Per-command timeout for entries that name none.

    Raises:
        RuntimeError: If any command fails (fail-fast), or if *trust*
            is False and stdin is not a TTY.
    """
    if not commands:
        return

    commands = coerce_command_list(commands)
    _hook_steps(commands, "post_commands")

    _confirm_hook_execution("post_commands", commands, trust, dry_run=dry_run, where="on this machine")

    rendered = render_hook_commands(commands, context)
    hook_env = build_hook_env(launch_context or _post_launch_context(context), "post_commands")
    _log_hook_target(hook_env)
    default_timeout = resolve_hook_timeout(default_timeout_s)
    steps = _hook_steps(rendered, "post_commands")

    logger.info("Running %d post_command(s) on control machine...", len(rendered))

    for i, step in enumerate(steps, 1):
        assert isinstance(step, tuple)  # post_commands rejects copy entries
        cmd, timeout_s = step
        timeout_s = timeout_s or default_timeout
        logger.info("  post_commands[%d] (%s): %s", i, _describe_limit(None if timeout_s == float("inf") else timeout_s), cmd)

        if dry_run:
            logger.info("  [dry-run] Would execute: %s", cmd)
            continue

        returncode, stdout = _run_control_command(cmd, env={**os.environ, **hook_env}, timeout_s=timeout_s)

        if stdout:
            for line in stdout.rstrip().splitlines():
                logger.info("  | %s", line)

        if returncode is None:
            raise RuntimeError(
                _hook_failure_message(
                    "post_commands[%d] timed out after %s: %s" % (i, _format_timeout(timeout_s), cmd),
                    cmd=cmd,
                    stdout=stdout,
                    stderr="",
                    log_key="%s-control" % (hook_env.get("SPARKRUN_CLUSTER_ID") or "unscoped"),
                )
            )
        if returncode != 0:
            raise RuntimeError(
                _hook_failure_message(
                    "post_commands[%d] failed (exit %d): %s" % (i, returncode, cmd),
                    cmd=cmd,
                    stdout=stdout,
                    stderr="",
                    log_key="%s-control" % (hook_env.get("SPARKRUN_CLUSTER_ID") or "unscoped"),
                )
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Inline budget for a failed hook's output. The message reaches the terminal
# and, for an exported systemd unit, the journal; the full output goes to a
# file (see _save_failure_output).
HOOK_FAILURE_EXCERPT_LIMIT = 3000
_COPY_FAILURE_EXCERPT_LIMIT = 1000
# A failure log is kept per workload and host, overwritten by the next failure
# there; this bounds the ones whose workload never fails (or runs) again.
FAILURE_LOG_MAX_AGE_S = 30 * 24 * 3600
_FAILURE_LOG_KEY = re.compile(r"[^A-Za-z0-9_.-]+")


def _redactor():
    """Redactor for hook output, including the HF token when one can be read.

    The token lookup reads a file and can raise (an unreadable token file); a
    failure message must never be lost to that, so it degrades to the
    pattern- and environment-based redaction.
    """
    from sparkrun.core.config import resolve_hf_token
    from sparkrun.diagnostics.redaction import Redactor

    try:
        token = resolve_hf_token()
    except Exception as exc:
        logger.debug("HF token unavailable for redaction: %s", exc)
        token = None
    return Redactor([token])


def _failure_log_dir() -> Path:
    from sparkrun.core.config import resolve_sparkrun_cache_dir

    return resolve_sparkrun_cache_dir() / "failures"


def _save_failure_output(log_key: str, content: str) -> Path | None:
    """Write a failed hook's complete output where the user can read it later.

    One file per workload and host, overwritten by the next failure, so it
    always describes the latest attempt without accumulating. Best effort: a
    failure to write is logged and never replaces the hook's own error.
    """
    from sparkrun.utils.fs import atomic_private_write

    try:
        directory = _failure_log_dir()
        path = directory / ("%s.log" % _FAILURE_LOG_KEY.sub("_", log_key).strip("._"))
    except Exception as exc:
        logger.warning("Could not resolve where to save full hook output: %s", exc)
        return None
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        atomic_private_write(path, content)
    except OSError as exc:
        logger.warning("Could not save full hook output to %s: %s", path, exc)
        return None
    _prune_failure_logs(directory, keep=path)
    return path


def _prune_failure_logs(directory: Path, *, keep: Path) -> None:
    import time

    cutoff = time.time() - FAILURE_LOG_MAX_AGE_S
    for entry in directory.glob("*.log"):
        try:
            if entry != keep and entry.stat().st_mtime < cutoff:
                entry.unlink()
        except OSError as exc:
            logger.debug("Could not prune failure log %s: %s", entry, exc)


def _hook_failure_message(summary: str, *, cmd: str, stdout: str, stderr: str, log_key: str) -> str:
    """Failure message for a hook: a head+tail excerpt plus where the rest is.

    The excerpt keeps the end of the output, where a traceback names its
    exception (issue #325). Both the message and the saved file are redacted,
    since hook output can echo tokens and the message reaches the journal.
    """
    from datetime import datetime, timezone

    redact = _redactor()
    output = stderr if (stderr or "").strip() else stdout
    # Redact before cutting: a cut through a token leaves a fragment that
    # neither the literal match nor the KEY=value pattern recognizes.
    detail = excerpt(redact(output or ""), HOOK_FAILURE_EXCERPT_LIMIT) or "(no output)"
    sections = ["%s\n" % summary, "time: %s\n" % datetime.now(timezone.utc).isoformat(), "command:\n%s\n" % cmd]
    for name, stream in (("stderr", stderr), ("stdout", stdout)):
        if (stream or "").strip():
            sections.append("--- %s ---\n%s\n" % (name, stream.rstrip()))
    path = _save_failure_output(log_key, redact("\n".join(sections)))
    message = "%s:\n%s" % (summary, detail)
    if path is not None:
        message += "\n(full output: %s)" % path
    return message


def _copy_failure_message(label: str, host: str, container_name: str, result) -> str:
    detail = excerpt(result.stderr or result.stdout, _COPY_FAILURE_EXCERPT_LIMIT) or "(no output)"
    return "%s copy failed on %s/%s: %s" % (label, host, container_name, detail)


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
    timeout_s: float = DEFAULT_HOOK_TIMEOUT_S,
    executor=None,
) -> None:
    """Execute a single command inside a workload, through its executor.

    The deadline is enforced twice.  Inside the container a watchdog kills
    the hook's whole process group (:func:`timed_hook_script`); killing only
    the ``docker exec`` (or ssh) client would leave the hook running in the
    container.  The control side waits a grace period longer, as the
    backstop for a hung connection.

    The command comes from :meth:`Executor.exec_cmd` of *host*'s executor,
    as root where the substrate has users, so a ``local`` workload's hook
    runs on the host instead of a ``docker exec`` into a container that does
    not exist. There it gets the executor's setup (working_dir, env_file,
    GPU visibility) and the hook env, but not the launch env, and container
    paths in the hook env (``SPARKRUN_RUNTIME_CACHE_DIR``) are not mapped
    to the host (``SPARKRUN_RUNTIME_CACHE_HOST_DIR`` is the host one).
    Without *executor* it is docker, as it always was.

    The hook script is base64'd once by :func:`timed_hook_script` and again
    by ``exec_cmd`` and travels as one argument, which Linux caps at 128 KiB:
    a single inline hook can be about 73 KB (mods ship as copied files).

    Args:
        host: Target host.
        container_name: Container name.
        cmd: Command to execute.
        ssh_kwargs: SSH connection kwargs.
        dry_run: Show what would be done without executing.
        label: Human-readable label for log messages.
        env: Hook-only metadata exported to this exec and its child processes.
        timeout_s: Wall-clock limit for the command; ``math.inf`` for none.
        executor: The workload's executor; ``None`` means docker.

    Raises:
        RuntimeError: If the command exits with non-zero status or times out.
    """
    import math

    from sparkrun.orchestration.primitives import run_script_on_host

    if math.isinf(timeout_s):
        # ``timeout: 0`` on the entry: no watchdog, and no control-side limit.
        seconds = None
        payload = cmd
    else:
        seconds = _whole_seconds(timeout_s)
        payload = timed_hook_script(cmd, seconds)
    script = _hook_executor(executor, host).exec_cmd(container_name, payload, env=dict(env or {}), user="root")

    logger.info("  %s on %s/%s (%s): %s", label, host, container_name, _describe_limit(seconds), cmd)

    result = run_script_on_host(
        host,
        script,
        ssh_kwargs=ssh_kwargs,
        timeout=None if seconds is None else seconds + _HOOK_KILL_AFTER_S + _HOOK_TIMEOUT_GRACE_S,
        dry_run=dry_run,
    )

    if not dry_run and not result.success:
        if result.returncode == 124 and seconds is not None:
            summary = "%s timed out after %s on %s/%s (raise it with a per-entry 'timeout:' or hooks.timeout_s in config.yaml)" % (
                label,
                _format_timeout(seconds),
                host,
                container_name,
            )
        elif result.returncode == -1 and seconds is not None and "timed out" in (result.stderr or ""):
            summary = "%s on %s/%s did not return within %ds of its %s deadline; the connection or docker exec hung" % (
                label,
                host,
                container_name,
                _HOOK_KILL_AFTER_S + _HOOK_TIMEOUT_GRACE_S,
                _format_timeout(seconds),
            )
        else:
            summary = "%s failed (exit %d) on %s/%s" % (label, result.returncode, host, container_name)
        raise RuntimeError(
            _hook_failure_message(
                summary,
                cmd=cmd,
                stdout=result.stdout,
                stderr=result.stderr,
                log_key="%s-%s" % ((env or {}).get("SPARKRUN_CLUSTER_ID") or "unscoped", host),
            )
        )


def _require_copy_support(steps, hosts_containers, executor) -> None:
    """Refuse ``copy`` entries (and mods, which become them) before any hook runs.

    A copy is a ``docker cp`` into the container; a substrate without one
    (``local`` runs a native process) would fail it halfway through the
    hooks, after earlier entries already changed the workload.
    """
    if executor is None or not any(isinstance(step, Mapping) for step in steps):
        return
    from sparkrun.orchestration.executors.docker import DockerExecutor

    for host, _container in hosts_containers:
        target = executor.for_host(host)
        if not isinstance(target, DockerExecutor):
            raise RuntimeError(
                "pre_exec copy entries (and recipe mods) need a container; the %s executor on %s has none. "
                "Bake the files into the environment instead, or run this recipe with the docker executor." % (target.executor_name, host)
            )


def _hook_executor(executor, host: str):
    """*host*'s executor for a hook (docker when the caller has none, as before)."""
    if executor is None:
        from sparkrun.orchestration.executors.docker import DockerExecutor

        return DockerExecutor()
    return executor.for_host(host)


def timed_hook_script(cmd: str, seconds: int) -> str:
    """Bash text that runs *cmd* with a deadline enforced where it runs.

    The hook runs as a job in its own process group (``set -m``, turned off
    again inside so jobs it backgrounds stay in that group) and a watchdog,
    also in its own group, TERMs the hook's group at the deadline and KILLs
    whatever is left after ``_HOOK_KILL_AFTER_S``.  The watchdog tells the
    shell it fired (USR1, which also interrupts ``wait``, hence the loop), so
    the verdict is not inferred from the exit code; on a timeout the shell
    waits for the escalation, since TERM usually ends the job's subshell
    while a child ignoring it lives on.  A timed-out hook exits 124, like
    ``timeout(1)``.  Pure bash rather than
    ``timeout(1)`` because busybox's (Alpine images) signals only its direct
    child, which leaves a hook's children running; the hook already requires
    bash.  A hook that finishes in time stops the watchdog, so a daemon it
    deliberately started keeps running, as before.  Brace-free.
    """
    from sparkrun.utils.shell import b64_wrap_bash

    return _HOOK_DEADLINE_SCRIPT % {
        "payload": b64_wrap_bash(cmd, quoted=False),
        "seconds": seconds,
        "kill_after": _HOOK_KILL_AFTER_S,
    }


_HOOK_DEADLINE_SCRIPT = """__sr_timed_out=0
trap '__sr_timed_out=1' USR1
set -m
( set +m
%(payload)s ) &
__sr_hook=$!
( sleep %(seconds)d
  kill -USR1 $$
  kill -TERM -$__sr_hook 2>/dev/null || exit 0
  __sr_i=0
  while [ $__sr_i -lt %(kill_after)d ] && kill -0 -$__sr_hook 2>/dev/null; do sleep 1; __sr_i=$((__sr_i + 1)); done
  kill -KILL -$__sr_hook 2>/dev/null ) </dev/null >/dev/null 2>&1 &
__sr_watchdog=$!
set +m
while :; do
  wait $__sr_hook
  __sr_rc=$?
  kill -0 $__sr_hook 2>/dev/null || break
done
if [ $__sr_timed_out = 1 ]; then
  wait $__sr_watchdog
  exit 124
fi
kill -KILL -$__sr_watchdog 2>/dev/null
exit $__sr_rc
"""


def _whole_seconds(timeout_s: float) -> int:
    import math

    return max(1, math.ceil(timeout_s))


def _describe_limit(seconds: float | None) -> str:
    return "no timeout" if seconds is None else "timeout %s" % _format_timeout(seconds)


def _format_timeout(seconds: float) -> str:
    seconds = _whole_seconds(seconds)
    if seconds % 3600 == 0:
        return "%dh" % (seconds // 3600)
    if seconds % 60 == 0:
        return "%dm" % (seconds // 60)
    return "%ds" % seconds


def _run_control_command(cmd: str, *, env: Mapping[str, str], timeout_s: float) -> tuple[int | None, str]:
    """Run a ``post_commands`` shell command; ``(returncode, output)``.

    ``returncode`` is ``None`` when *timeout_s* expired; ``math.inf`` waits
    for the command however long it takes (an entry's ``timeout: 0``).  The command runs in
    its own process group so expiry (or an exception here, Ctrl-C included)
    kills the whole group: ``subprocess.run(timeout=)`` kills only the shell
    and leaves a backgrounded child running.  A group, not a session: with
    a controlling terminal the group is made the foreground one for the
    command's lifetime, as a shell does, so a ``sudo`` password prompt keeps
    working (``setsid`` would detach it from the terminal entirely).
    """
    import signal

    posix = hasattr(os, "killpg")
    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(env),
        process_group=0 if posix else None,
    )
    terminal = _hand_terminal_to(proc) if posix else None
    try:
        stdout, _ = proc.communicate(timeout=None if timeout_s == float("inf") else timeout_s)
        return proc.returncode, stdout or ""
    except subprocess.TimeoutExpired:
        _kill_process_group(proc, signal.SIGTERM)
        try:
            stdout, _ = proc.communicate(timeout=_HOOK_KILL_AFTER_S)
        except subprocess.TimeoutExpired:
            stdout = None
        finally:
            # Reaping the shell and reaching EOF do not prove its children
            # exited: a TERM-ignoring child can redirect both output streams.
            _kill_process_group(proc, kill_signal)
        if stdout is None:
            stdout = _drain_after_kill(proc)
        return None, stdout or ""
    except BaseException:
        _kill_process_group(proc, kill_signal)
        _drain_after_kill(proc)
        raise
    finally:
        if terminal is not None:
            _reclaim_terminal(terminal)


def _hand_terminal_to(proc: subprocess.Popen) -> int | None:
    """Make *proc*'s group the terminal's foreground group; the tty fd, or None.

    Only when this process is itself in the foreground (a backgrounded
    sparkrun has no business taking the terminal).  The group is SIGCONTed
    afterwards in case the command reached the tty first and was stopped.
    """
    import signal
    import sys

    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd) or os.tcgetpgrp(fd) != os.getpgrp():
            return None
        _without_sigttou(lambda: os.tcsetpgrp(fd, proc.pid))
        os.killpg(proc.pid, signal.SIGCONT)
    except (OSError, ValueError, AttributeError) as e:
        logger.debug("post_commands keep the terminal in sparkrun's process group: %s", e)
        return None
    return fd


def _reclaim_terminal(fd: int) -> None:
    try:
        _without_sigttou(lambda: os.tcsetpgrp(fd, os.getpgrp()))
    except OSError as e:
        logger.warning("Could not take the terminal back after post_commands: %s", e)


def _without_sigttou(action) -> None:
    """Run *action* with SIGTTOU ignored (``tcsetpgrp`` from a background group raises it)."""
    import signal

    previous = signal.signal(signal.SIGTTOU, signal.SIG_IGN)
    try:
        action()
    finally:
        signal.signal(signal.SIGTTOU, previous)


def _drain_after_kill(proc: subprocess.Popen) -> str:
    """Collect output after a KILL without waiting on an escaped holder of the pipe.

    A descendant outside the group (``setsid``, or any child on Windows where
    there is no group) can keep stdout open forever; give it a bounded
    moment, then close our end and stop waiting.
    """
    try:
        stdout, _ = proc.communicate(timeout=_HOOK_KILL_AFTER_S)
        return stdout or ""
    except subprocess.TimeoutExpired:
        logger.warning("A killed post_command left a process holding its output open; not waiting for it")
        if proc.stdout is not None:
            proc.stdout.close()
        try:
            proc.wait(timeout=_HOOK_KILL_AFTER_S)
        except subprocess.TimeoutExpired:
            logger.warning("post_command shell (pid %s) did not exit after SIGKILL", proc.pid)
        return ""


def _kill_process_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal *proc*'s process group (POSIX) or its process tree (Windows)."""
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, sig)
            return
        except ProcessLookupError:
            return  # the whole group is already gone
        except OSError as e:
            logger.debug("Signalling post_command group %s failed (%s); killing the shell only", proc.pid, e)
    elif os.name == "nt":
        result = subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
        if result.returncode == 0:
            return
        logger.debug("taskkill of post_command tree %s failed (rc %s)", proc.pid, result.returncode)
    try:
        proc.kill()
    except OSError as e:
        logger.debug("post_command shell %s already gone: %s", proc.pid, e)


def _run_copy_command(
    host: str,
    container_name: str,
    cmd: Mapping[str, str],
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
            raise TransferError(_copy_failure_message(label, host, container_name, result))
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
            raise TransferError(_copy_failure_message(label, host, container_name, result))
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
                detail=excerpt(mkdir_result.stderr or mkdir_result.stdout, 400),
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
            raise TransferError(_copy_failure_message(label, host, container_name, result))


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
