"""Operator-visible manifest and cache migration commands."""

from dataclasses import asdict
from functools import wraps
import json
from pathlib import Path

import click

from sparkrun.api import model_cache as api
from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactManifest, ModelArtifactRequest
from ._common import _get_context


@click.group("model-cache")
def model_cache():
    """Inspect, import and repair manifest-based model caches."""


def options(function):
    @wraps(function)
    def wrapped(ctx, model, revision, file_selection, projector, endpoint, cache_dir, hosts, checksums, **kwargs):
        from sparkrun.core.config import resolve_hf_cache_home
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        sctx = _get_context(ctx)
        try:
            request = ModelArtifactRequest(model, revision, endpoint, projector, file_selection)
            result = function(
                request,
                cache_dir=resolve_hf_cache_home(cache_dir),
                hosts=hosts or ("localhost",),
                checksums=checksums,
                ssh_kwargs=build_ssh_kwargs(sctx.config),
                config=sctx.config,
                **kwargs,
            )
        except (ModelArtifactError, OSError) as error:
            raise click.ClickException(str(error)) from error
        if result is not None:
            click.echo(json.dumps([asdict(report) for report in result], indent=2))
            if any(not report.complete for report in result):
                ctx.exit(1)

    for decorator in (
        click.option("--checksum", "checksums", is_flag=True, help="Hash selected files; normal inspection checks size and readability."),
        click.option("--host", "hosts", multiple=True, help="Target host (repeatable; default localhost)."),
        click.option("--cache-dir", default=None, help="HF_HOME on the target hosts."),
        click.option("--endpoint", default="https://huggingface.co", envvar="HF_ENDPOINT"),
        click.option("--projector", default=None),
        click.option("--file-selection", type=click.Choice(["auto", "all", "safetensors"]), default="auto", show_default=True),
        click.option("--revision", default=None),
        click.argument("model"),
    ):
        wrapped = decorator(wrapped)
    return click.pass_context(wrapped)


@model_cache.command("inspect")
@options
def inspect_cache(request, *, config, **kwargs):
    """Observe saved inventory without network access or modifying the cache."""
    return api.inspect_model(request, **kwargs)


@model_cache.command("resolve")
@click.option("--output", required=True, type=click.Path(path_type=Path), help="Write the portable manifest here.")
@options
def resolve(request, *, output, cache_dir, **kwargs):
    """Resolve origin metadata online; download only the small weight index."""
    from sparkrun.core.config import resolve_hf_token

    manifest = api.resolve_manifest(request, token=resolve_hf_token(), cache_dir=cache_dir)
    output.write_text(manifest.to_json() + "\n")
    click.echo("Wrote %d files at revision %s to %s" % (len(manifest.files), manifest.revision, output))


@model_cache.command("import")
@click.option("--manifest", "manifest_file", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@options
def import_cache(request, *, manifest_file, config, **kwargs):
    """Import trusted expected metadata and report current completeness."""
    return api.import_manifest(request, ModelArtifactManifest.from_json(manifest_file.read_text()), **kwargs)


@model_cache.command("repair")
@click.option("--offline", is_flag=True, help="Permit cluster copies only; no origin calls or tool installation.")
@click.option("--dry-run", is_flag=True)
@click.option("--local-cache-dir", default=None)
@click.option("--transfer-mode", type=click.Choice(["local", "push", "delegated", "pull"]), default="local")
@options
def repair(request, **kwargs):
    """Repair selected paths, then validate every target against the manifest."""
    from sparkrun.core.config import resolve_hf_token

    prepared = api.repair_model(request, token=resolve_hf_token(), **kwargs)
    if prepared is None:
        click.echo("Dry run: no model metadata acquired or caches changed.")
        return None
    return tuple(binding.report for binding in prepared.bindings)
