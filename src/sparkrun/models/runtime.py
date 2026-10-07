"""Consume prepared artifacts without resolving refs or scanning the cache again."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import PurePosixPath
import shlex

from sparkrun.core.env_templates import _mounts, workload_model_path
from sparkrun.core.model_distribution import prepared_model, prepared_models
from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.models.observation import observation_script, parse_observations
from sparkrun.utils.shell import quote


def bind_runtime_models(recipe, overrides: dict, runtime, hosts, executors, volumes) -> None:
    """Pin normal HF loaders; use selected files for GGUF and directory loaders.

    Recipe identity and served names remain declarative. Only operation-local
    overrides carry prepared paths and immutable revisions.
    """
    bindings = [prepared_model(recipe.model, host) for host in hosts]
    if not any(bindings):
        return
    if not all(bindings):
        raise ModelArtifactError("served model was not validated on every launch host")
    binding = bindings[0]
    assert binding is not None
    if any(b is not None and b.manifest.identity != binding.manifest.identity for b in bindings):
        raise ModelArtifactError("launch hosts have different model artifacts")
    paths = {
        workload_model_path(b.model_path, _mounts(volumes, executors[b.host]), executors[b.host].executor_name)
        for b in bindings
        if b is not None
    }
    if len(paths) != 1:
        raise ModelArtifactError("this runtime requires the same workload model path on every host")
    path = paths.pop()
    overrides["model_revision"] = binding.manifest.revision
    overrides["_prepared_model_path"] = path
    overrides["_prepared_model_revision"] = binding.manifest.revision
    # Templated commands explicitly referring to {model} use the prepared path.
    # A literal model ID is accepted only with the verified revision (below).
    if recipe.command or binding.manifest.weight_path:
        overrides["model"] = path
    if binding.manifest.weight_path:
        overrides["_gguf_model_path"] = path
    if binding.manifest.projector_path:
        projector = str(PurePosixPath(binding.snapshot_path) / binding.manifest.projector_path)
        overrides["mmproj"] = overrides["_mmproj_path"] = workload_model_path(
            projector, _mounts(volumes, executors[hosts[0]]), executors[hosts[0]].executor_name
        )
    if runtime.get_family() in {"vllm", "sglang"}:
        overrides["revision"] = binding.manifest.revision
    config = recipe.build_config_chain(overrides)

    def draft_binding(model):
        matches = [b for b in prepared_models() if b.model == model]
        if len({b.manifest.identity for b in matches}) > 1:
            raise ModelArtifactError("draft model differs between launch hosts")
        return matches[0] if matches else None

    # Draft artifacts have independent identities, never the served model SHA.
    if runtime.get_family() == "sglang":
        draft = config.get("speculative_draft_model_path") or config.get("speculative_draft_model")
        prepared = draft_binding(str(draft)) if draft else None
        if prepared:
            overrides["speculative_draft_model_revision"] = prepared.manifest.revision
    elif runtime.get_family() == "vllm":
        spec = config.get("speculative_config")
        if spec:
            values = json.loads(spec) if isinstance(spec, str) else dict(spec)
            prepared = draft_binding(str(values.get("model", "")))
            if prepared:
                values["revision"] = prepared.manifest.revision
                overrides["speculative_config"] = json.dumps(values)


def validate_runtime_command(recipe, overrides, runtime, command: str) -> None:
    """Refuse a literal custom command that would load a different artifact."""
    path = overrides.get("_prepared_model_path")
    if not path:
        return
    tokens = shlex.split(command)
    if path in tokens or any(token.endswith("=" + path) for token in tokens):
        return
    revision = overrides["_prepared_model_revision"]
    if recipe.model not in tokens and not any(t.endswith("=" + recipe.model) for t in tokens):
        raise ModelArtifactError("serve command does not identify the validated model")
    flags = runtime.model_revision_flags()
    for i, token in enumerate(tokens):
        if token in flags and i + 1 < len(tokens) and tokens[i + 1] == revision:
            return
        if any(token == flag + "=" + revision for flag in flags):
            return
    raise ModelArtifactError(
        "serve command does not consume the validated model; use {model} or a supported revision flag with {model_revision}"
    )


def verify_workload_access(hosts, executors, images, volumes, ssh_kwargs, *, extra_opts=()) -> None:
    """Observe the same inventory through workload mounts and the effective UID.

    Probes run before eviction, without GPUs or the image entrypoint. Third-party
    executors can supply model_access_command(image, script, volumes, extra_opts).
    They must preserve the final workload's filesystem and identity semantics.
    """
    from sparkrun.models.host_io import ModelHostIO
    from sparkrun.orchestration.executors._seccomp import split_options

    bindings = prepared_models()
    if not bindings:
        return
    _, tokens, _ = split_options(None, list(extra_opts))
    if any(
        t in {"-v", "--volume", "--mount", "--tmpfs", "--volumes-from"}
        or t.startswith(("-v", "--volume=", "--mount=", "--tmpfs=", "--volumes-from="))
        for t in tokens
    ):
        raise ModelArtifactError("validated model access requires structured executor_config.volumes instead of raw mount options")
    with ModelHostIO(ssh_kwargs) as io:
        for binding in bindings:
            if binding.host not in hosts:
                continue
            executor = executors[binding.host]
            mounts = _mounts(volumes, executor)
            snapshot = workload_model_path(binding.snapshot_path, mounts, executor.executor_name)
            script = observation_script(binding.manifest, snapshot)
            hook = getattr(executor, "model_access_command", None)
            if callable(hook):
                command = hook(images[hosts.index(binding.host)], script, volumes, list(extra_opts))
            elif executor.executor_name == "local":
                command = "printf %s " + quote(script) + " | bash -s"
            else:
                raise ModelArtifactError("executor does not support model workload access validation: " + executor.executor_name)
            if not isinstance(command, str) or not command:
                raise ModelArtifactError("executor returned an invalid model access command")
            result = io.execute(binding.host, command, timeout=90)
            report = parse_observations(
                binding.host, binding.manifest, result.stdout.decode(errors="replace"), returncode=result.returncode
            )
            report = replace(report, access_context=executor.executor_name)
            if not report.complete:
                details = "; ".join(f.path + ": " + f.reason for f in report.failures[:5])
                raise ModelArtifactError("model is inaccessible to workload on %s: %s" % (binding.host, details or report.detail))
