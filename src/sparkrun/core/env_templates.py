"""Explicit recipe environment templates resolved from a single launch context.

Only reserved {config.NAME} / {launch.NAME} tokens in recipe env are expanded.
There is no shell evaluation, host environment lookup or recursive interpolation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

LAUNCH_FIELDS = frozenset({"model_path", "model_revision", "num_nodes", "node_rank", "node_host", "cluster_id", "runtime_cache_dir"})
# Match only our reserved namespaces, not JSON, shell ${VAR}, or other braces.
_OPEN = re.compile(r"(?<![$\{])(?P<braces>\{+)(?:config|launch)\.")
_FIELD = re.compile(r"(config|launch)\.([A-Za-z_][A-Za-z0-9_]*)\Z")


def _tokens(value: str):
    for match in _OPEN.finditer(value):
        start, width = match.start(), len(match["braces"])
        end = value.find("}", match.end())
        if end < 0 or "{" in value[match.end() : end]:
            yield start, match.end(), width, "", 0
            continue
        stop = end
        while stop < len(value) and value[stop] == "}":
            stop += 1
        count = min(width, stop - end)
        yield start, end + count, width, value[start + width : end], count


def _parts(template: str) -> list[tuple[str, str | None]]:
    parts: list[tuple[str, str | None]] = []
    cursor = 0
    for start, stop, width, name, closing in _tokens(template):
        parts.append((template[cursor:start], None))
        if width > 1:
            # Escaped reserved tokens are literal, including unknown names.
            # An incomplete escaped token also remains literal; this lets a
            # verbatim CLI value survive export/reload without interpretation.
            literal = template[start + 1 : stop - 1] if closing == width else template[start + 1 : stop]
            parts.append((literal, None))
        else:
            match = _FIELD.fullmatch(name)
            if match is None:
                raise ValueError("env accepts only {config.NAME} or {launch.NAME} without formatting")
            if match[1] == "launch" and match[2] not in LAUNCH_FIELDS:
                raise ValueError("unknown env template launch field: " + match[2])
            parts.append(("", name))
        cursor = stop
    parts.append((template[cursor:], None))
    return parts


def validate_env_template(template: str) -> set[str]:
    """Validate reserved tokens and return their config references."""
    return {name[7:] for _, name in _parts(template) if name and name.startswith("config.")}


def select_env_templates(env: Mapping[str, str], literal_keys=()) -> dict[str, str]:
    """Derive the values requiring rendering from one recipe env mapping."""
    result = {}
    for key, value in env.items():
        if key in literal_keys:
            continue
        try:
            parts = _parts(value)
        except ValueError as error:
            raise ValueError("env.%s: %s" % (key, error)) from error
        if any(name is not None for _, name in parts) or "".join(literal for literal, _ in parts) != value:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError("env: templated keys must be valid environment variable names")
            result[key] = value
    return result


def escape_env_literal(value: str) -> str:
    """Encode a verbatim CLI value as literal recipe source for export/hash."""
    chunks, cursor = [], 0
    for start, stop, width, _name, closing in _tokens(value):
        chunks.extend((value[cursor:start], "{", value[start:stop]))
        if closing == width:
            chunks.append("}")
        cursor = stop
    chunks.append(value[cursor:])
    return "".join(chunks)


def canonical_env(env: Mapping[str, str], literal_keys=()) -> dict[str, str]:
    return {key: escape_env_literal(value) if key in literal_keys else value for key, value in env.items()}


def render_env_template(template: str, config: Mapping[str, Any], launch: Mapping[str, Any]) -> str:
    chunks = []
    for literal, name in _parts(template):
        chunks.append(literal)
        if name is not None:
            namespace, key = name.split(".", 1)
            source = config if namespace == "config" else launch
            value = source.get(key)
            if value is None or (namespace == "launch" and value == ""):
                raise ValueError("unavailable env template field: " + name)
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError("env template field must be a scalar: " + name)
            chunks.append(str(value))
    result = "".join(chunks)
    if "\x00" in result:
        raise ValueError("env template values cannot contain NUL")
    return result


def env_template_launch_fields(templates: Mapping[str, str]) -> set[str]:
    """Return launch facts a strategy must provide, ignoring escaped tokens."""
    return {name[7:] for value in templates.values() for _, name in _parts(value) if name and name.startswith("launch.")}


@dataclass(frozen=True)
class PreparedModelPath:
    host_path: str
    revision: str


def probe_model_path(
    host: str, model: str, revision: str, cache_dir: str, ssh_kwargs: dict, *, expected_revision: str = ""
) -> PreparedModelPath:
    """Resolve cached assets on the target, never on the controller or network."""
    from sparkrun.orchestration.primitives import run_script_on_host
    from sparkrun.scripts import read_script
    from sparkrun.utils.shell import quote

    payload = json.dumps([model, revision, cache_dir, expected_revision])
    script = "python3 - %s <<'SPARKRUN_MODEL_PROBE'\n%s\nSPARKRUN_MODEL_PROBE\n" % (quote(payload), read_script("resolve_model_path.py"))
    result = run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=60)
    if not result.success:
        raise ValueError("cannot resolve prepared model on %s: %s" % (host, (result.stderr or "cache probe failed").strip()[:500]))
    lines = [line.removeprefix("SPARKRUN_MODEL_PATH=") for line in result.stdout.splitlines() if line.startswith("SPARKRUN_MODEL_PATH=")]
    try:
        data = json.loads(lines[-1])
        path, commit = data["path"], data["revision"]
        if not isinstance(path, str) or not path.startswith("/") or not isinstance(commit, str):
            raise ValueError
    except (IndexError, KeyError, TypeError, ValueError):
        raise ValueError("invalid model path probe response from %s" % host) from None
    if expected_revision and commit != expected_revision:
        raise ValueError("model revision differs between launch nodes: " + host)
    return PreparedModelPath(path, commit)


def workload_model_path(host_path: str, volumes: Mapping[str, str], executor_name: str) -> str:
    """Map a host asset through the longest covering mount; refuse shadowing."""
    if executor_name == "local":
        return host_path
    if executor_name != "docker":
        raise ValueError("launch.model_path is unsupported by executor " + executor_name)
    path = PurePosixPath(host_path)
    candidates = []
    for source, destination in volumes.items():
        if not source.startswith("/"):
            continue
        try:
            relative = path.relative_to(source)
        except ValueError:
            continue
        target = str(PurePosixPath(destination.split(":", 1)[0]) / relative)
        candidates.append((len(PurePosixPath(source).parts), source, target))
    if not candidates:
        raise ValueError("prepared model is not visible through the workload mounts")
    _, source, target = max(candidates)
    destination = volumes[source].split(":", 1)[0]
    for other_source, other_destination in volumes.items():
        other_destination = other_destination.split(":", 1)[0]
        if (
            other_source != source
            and (PurePosixPath(target).is_relative_to(other_destination) or PurePosixPath(other_destination).is_relative_to(target))
            and PurePosixPath(other_destination).is_relative_to(destination)
        ):
            raise ValueError("prepared model mount is shadowed by another workload mount")
    return target


def _mounts(volumes: Mapping[str, str], executor) -> dict[str, str]:
    """Include explicit executor mounts when checking source visibility/conflicts."""
    result = {}
    for spec in getattr(getattr(executor, "config", None), "volumes", None) or []:
        parts = str(spec).split(":")
        source, target = parts[0], parts[1] if len(parts) > 1 else parts[0]
        previous = {**result, **volumes}.get(source)
        if previous is not None and previous.split(":", 1)[0] != target:
            raise ValueError("env templates cannot resolve a source mounted at multiple destinations")
        result[source] = target
    result.update(volumes)
    return result


@dataclass
class EnvTemplatePlan:
    templates: dict[str, str]
    config: Mapping[str, Any]
    hosts: tuple[str, ...]
    cluster_id: str
    models: dict[str, PreparedModelPath] = field(default_factory=dict)
    dry_run: bool = False
    rendered: dict[str, dict[str, str]] = field(default_factory=dict)
    contexts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def render(self, host: str, volumes: Mapping[str, str], runtime_cache, executor) -> dict[str, str]:
        required = env_template_launch_fields(self.templates)
        launch: dict[str, Any] = {
            "num_nodes": len(self.hosts),
            "node_rank": self.hosts.index(host),
            "node_host": host,
            "cluster_id": self.cluster_id,
        }
        mounts = _mounts(volumes, executor) if required & {"model_path", "runtime_cache_dir"} else dict(volumes)
        name = executor.executor_name
        if "model_path" in required:
            launch["model_path"] = (
                "<unresolved:launch.model_path>" if self.dry_run else workload_model_path(self.models[host].host_path, mounts, name)
            )
        if "model_revision" in required:
            launch["model_revision"] = "<unresolved:launch.model_revision>" if self.dry_run else self.models[host].revision
        if "runtime_cache_dir" in required:
            if runtime_cache is None:
                raise ValueError("launch.runtime_cache_dir requires an enabled runtime cache")
            destination = runtime_cache.volumes.get(runtime_cache.leaf)
            if (
                not destination
                or mounts.get(runtime_cache.leaf) != destination
                or sum(v.split(":", 1)[0] == destination for v in mounts.values()) != 1
            ):
                raise ValueError("runtime cache mount is missing or overridden")
            if any(
                source != runtime_cache.leaf and PurePosixPath(target.split(":", 1)[0]).is_relative_to(destination)
                for source, target in mounts.items()
            ):
                raise ValueError("runtime cache contains an overriding workload mount")
            if name not in ("docker", "local"):
                raise ValueError("launch.runtime_cache_dir is unsupported by executor " + name)
            launch["runtime_cache_dir"] = runtime_cache.leaf if name == "local" else destination
        if host in self.rendered:
            if self.contexts[host] != launch:
                raise ValueError("launch environment context changed after preparation for " + host)
            return dict(self.rendered[host])
        result = {}
        for key, template in self.templates.items():
            try:
                result[key] = render_env_template(template, self.config, launch)
            except ValueError as exc:
                raise ValueError("env.%s on %s: %s" % (key, host, exc)) from exc
        self.contexts[host], self.rendered[host] = launch, result
        return dict(result)


def prepare_env_templates(
    recipe,
    overrides,
    hosts: Sequence[str],
    cluster_id: str,
    cache_dir: str,
    ssh_kwargs: dict,
    *,
    dry_run: bool = False,
    extra_docker_opts: Sequence[str] = (),
) -> EnvTemplatePlan | None:
    templates = getattr(recipe, "env_templates", None) or {}
    if not templates:
        return None
    config = dict(recipe.build_config_chain(overrides))
    plan = EnvTemplatePlan(templates, config, tuple(dict.fromkeys(hosts)), cluster_id, dry_run=dry_run)
    required = env_template_launch_fields(templates)
    if required & {"model_path", "runtime_cache_dir"}:
        # Raw options bypass the structured mount map. Do not claim a path is
        # visible when these options can replace it or add anonymous mounts.
        from sparkrun.orchestration.executors._seccomp import split_options

        _, tokens, _ = split_options(None, list(extra_docker_opts))
        if any(
            token in {"-v", "--volume", "--mount", "--tmpfs", "--volumes-from"}
            or token.startswith(("--volume=", "--mount=", "--tmpfs=", "--volumes-from=", "-v"))
            for token in tokens
        ):
            raise ValueError("path env templates require structured executor_config.volumes instead of raw Docker mount options")
    if required & {"model_path", "model_revision"} and not dry_run:
        # Preparation distributes recipe.model. The command config may contain
        # an already translated GGUF container path; never probe that as a host
        # path or silently select a different model from a command override.
        model = str(recipe.model)
        revision = str(recipe.model_revision or "")
        configured_model = config.get("model", model)
        if configured_model != model and configured_model != (overrides or {}).get("_gguf_model_path"):
            raise ValueError("launch model fields require model to match the prepared recipe.model")
        if str(config.get("model_revision") or "") != revision:
            raise ValueError("launch model fields require model_revision to match the prepared recipe.model_revision")
        expected = ""
        for host in plan.hosts:
            record = probe_model_path(host, model, revision, cache_dir, ssh_kwargs, expected_revision=expected)
            plan.models[host] = record
            expected = expected or record.revision
    return plan


_ACTIVE_PLAN: ContextVar[EnvTemplatePlan | None] = ContextVar("sparkrun_env_templates", default=None)


@contextmanager
def env_template_scope(plan: EnvTemplatePlan | None) -> Iterator[None]:
    token = _ACTIVE_PLAN.set(plan)
    try:
        yield
    finally:
        _ACTIVE_PLAN.reset(token)


def render_launch_env(recipe, host: str, volumes: Mapping[str, str], runtime_cache, executor) -> dict[str, str]:
    """Shared solo/cluster boundary, before container creation and hooks."""
    templates = getattr(recipe, "env_templates", None) or {}
    if not templates:
        return {}
    plan = _ACTIVE_PLAN.get()
    if plan is None:
        raise ValueError("templated env requires the standard launch pipeline's prepared context")
    return plan.render(host, volumes, runtime_cache, executor)
