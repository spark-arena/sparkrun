"""Explicit model cache inspection, inventory import and targeted repair.

Inspection is read-only and offline. Resolution contacts the model origin;
repair is the only operation here that downloads model payloads. These helpers
never start/evict workloads and do not change api.materialize's contract.
"""

from __future__ import annotations

from sparkrun.models.artifacts import CacheValidationReport, ModelArtifactManifest, ModelArtifactRequest
from sparkrun.models.host_io import ModelHostIO
from sparkrun.models.manifest import check_request, fetch_inventory, manifest_path, select_manifest
from sparkrun.models.preparation import PreparedModel, prepare_model


def resolve_manifest(request: ModelArtifactRequest, *, token: str | None = None, cache_dir: str | None = None) -> ModelArtifactManifest:
    """Resolve authoritative metadata, including a small weight index if present."""
    return select_manifest(request, fetch_inventory(request, token=token, cache=cache_dir))


def inspect_model(
    request: ModelArtifactRequest, *, cache_dir: str, hosts=("localhost",), checksums=False, ssh_kwargs=None, session=None
) -> tuple[CacheValidationReport, ...]:
    reports = []
    with ModelHostIO(ssh_kwargs, session=session) as io:
        for host in hosts:
            root = io.root(host, cache_dir)
            stored = io.read(host, str(manifest_path(root, request)))
            if stored is None:
                reports.append(
                    CacheValidationReport(
                        host,
                        "",
                        "checksum" if checksums else "metadata",
                        "unverified",
                        detail="no saved manifest; resolve or import authoritative inventory first",
                    )
                )
                continue
            manifest = ModelArtifactManifest.from_json(stored)
            check_request(manifest, request)
            reports.append(io.observe(host, manifest, root, checksums=checksums))
    return tuple(reports)


def import_manifest(
    request: ModelArtifactRequest,
    manifest: ModelArtifactManifest,
    *,
    cache_dir: str,
    hosts=("localhost",),
    checksums=False,
    ssh_kwargs=None,
    session=None,
) -> tuple[CacheValidationReport, ...]:
    """Import operator-supplied expected metadata, never infer it by listing files.

    Import can record an incomplete inventory for later repair. The returned
    observations explicitly distinguish that from a complete cache.
    """
    check_request(manifest, request)
    reports = []
    with ModelHostIO(ssh_kwargs, session=session) as io:
        for host in hosts:
            root = io.root(host, cache_dir)
            io.lock(host, root, request.repo)
            reports.append(io.observe(host, manifest, root, checksums=checksums))
            io.write(host, str(manifest_path(root, request)), manifest.to_json())
    return tuple(reports)


def repair_model(
    request: ModelArtifactRequest,
    *,
    cache_dir: str,
    hosts=("localhost",),
    local_cache_dir=None,
    transfer_mode="local",
    checksums=False,
    offline=False,
    dry_run=False,
    token=None,
    ssh_kwargs=None,
    config=None,
    session=None,
    manifest=None,
) -> PreparedModel | None:
    """Repair selected missing/corrupt paths and revalidate all requested hosts."""
    return prepare_model(
        request.model,
        list(hosts),
        cache_dir=cache_dir,
        local_cache_dir=local_cache_dir,
        revision=request.revision,
        projector=request.projector,
        file_selection=request.file_selection,
        endpoint=request.endpoint,
        transfer_mode=transfer_mode,
        checksums=checksums,
        offline=offline,
        dry_run=dry_run,
        token=token,
        ssh_kwargs=ssh_kwargs,
        config=config,
        session=session,
        manifest=manifest,
    )
