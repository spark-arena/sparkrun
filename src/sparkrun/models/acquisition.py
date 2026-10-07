"""Pinned, required-metadata and file-repair adapter for Hugging Face."""

from __future__ import annotations

from dataclasses import asdict
import json
import inspect
from textwrap import indent

from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactManifest, ModelArtifactRequest, ModelInventoryUnavailable
from sparkrun.models.host_io import ModelHostIO
from sparkrun.models.manifest import HUB_ADAPTER_VERSION, read_weight_index
from sparkrun.utils.shell import quote


def repair_snapshot_files(repo, revision, cache, endpoint, token, files):
    """Publish only selected snapshot entries; never unlink shared HF blobs.

    Hub force_download updates a blob but leaves an existing regular snapshot
    entry untouched. A private local_dir followed by rename handles both that
    representation and classic/shared symlinks without a second weight copy.
    This function is also shipped verbatim to the pinned remote adapter.
    """
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import os
    from huggingface_hub import hf_hub_download

    snapshot = Path(cache) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in files:
        with TemporaryDirectory(prefix=".sparkrun-repair-", dir=snapshot.parent) as staging:
            downloaded = hf_hub_download(
                repo,
                name,
                revision=revision,
                endpoint=endpoint,
                token=token,
                local_dir=staging,
            )
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(downloaded, target)


def hub_script(
    request: ModelArtifactRequest,
    cache: str,
    *,
    token: str | None,
    manifest: ModelArtifactManifest | None = None,
    files: tuple[str, ...] = (),
) -> str:
    """One pinned adapter for controller/head acquisition; tokens travel on stdin."""
    from sparkrun.core.tooling import UV_INSTALL_BIN_DIR, UV_INSTALL_URL

    payload = {
        "request": asdict(request),
        "cache": cache,
        "token": token,
        "revision": manifest.revision if manifest else None,
        "files": files,
    }
    program = (
        inspect.getsource(read_weight_index)
        + "\n"
        + inspect.getsource(repair_snapshot_files)
        + "\n"
        + """import json, os
from huggingface_hub import HfApi, hf_hub_download
payload = json.loads(PAYLOAD)
request = payload['request']
repo = request['model'].split(':', 1)[0]
if payload['revision'] is None:
    info = HfApi(endpoint=request['endpoint'], token=payload['token']).model_info(repo, revision=request['revision'], files_metadata=True, timeout=30)
    files = []
    for f in info.siblings or ():
        algorithm = 'sha256' if f.lfs else 'git-sha1' if f.blob_id else None
        checksum = f.lfs.sha256 if f.lfs else f.blob_id
        files.append(dict(path=f.rfilename, size=f.size, checksum_algorithm=algorithm, checksum=checksum))
    result = dict(revision=info.sha, files=files)
    index = next((f for f in info.siblings or () if f.rfilename == 'model.safetensors.index.json'), None)
    if index is not None and request['file_selection'] != 'all':
        if index.size is None or index.size > 16 * 1024 * 1024:
            raise ValueError('safetensors index exceeds the 16 MiB inventory limit; use file selection all')
        entry = next(f for f in files if f['path'] == index.rfilename)
        result['weight_map'] = read_weight_index(repo, info.sha, request['endpoint'], payload['token'], entry)
else:
    repair_snapshot_files(repo, payload['revision'], payload['cache'], request['endpoint'], payload['token'], payload['files'])
    result = dict(revision=payload['revision'], repaired=len(payload['files']))
print('SPARKRUN_MODEL_RESULT=' + json.dumps(result, separators=(',', ':')))
""".replace("PAYLOAD", repr(json.dumps(payload)))
    )
    # Keep a network/tool outage distinct from an invalid authoritative index.
    # Neither payload acquisition nor a malformed inventory may use the
    # compatibility fallback merely because the subprocess exited nonzero.
    program = (
        "import sys, httpx\ntry:\n" + indent(program, "    ") + "\nexcept (httpx.HTTPError, OSError) as error:\n"
        "    print('SPARKRUN_MODEL_ERROR=unavailable')\n"
        "    print(str(error), file=sys.stderr)\n    sys.exit(1)\n"
        "except Exception as error:\n"
        "    print('SPARKRUN_MODEL_ERROR=invalid')\n"
        "    print(str(error), file=sys.stderr)\n    sys.exit(1)\n"
    )
    # Invoke the pinned installed client, otherwise a pinned isolated uv env.
    # This path is never reached for offline or already-complete observations.
    version_check = "import huggingface_hub; assert huggingface_hub.__version__ == %r" % HUB_ADAPTER_VERSION
    script = "set -euo pipefail\nexport HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30\n"
    script += f"if command -v python3 >/dev/null && python3 -c {quote(version_check)} 2>/dev/null; then\n  runner=(python3)\nelse\n"
    script += f'  export PATH="{UV_INSTALL_BIN_DIR}:$PATH"\n'
    script += f"  if ! command -v uv >/dev/null; then curl -fLsS {quote(UV_INSTALL_URL)} | sh >&2; fi\n"
    script += f"  runner=(uv run --no-project --with huggingface-hub=={HUB_ADAPTER_VERSION} python)\nfi\n"
    # Bootstrap failure is eligible for compatibility fallback; an unexpected
    # adapter failure after this point is not an unavailable-inventory result.
    script += "\"${runner[@]}\" -c 'import huggingface_hub, httpx'\n"
    script += "printf '%s\\n' SPARKRUN_MODEL_ADAPTER_READY\n"
    script += "\"${runner[@]}\" - <<'SPARKRUN_HUB_ADAPTER'\n" + program + "\nSPARKRUN_HUB_ADAPTER\n"
    return script


def run_adapter(
    io: ModelHostIO,
    host: str | None,
    request: ModelArtifactRequest,
    cache: str,
    *,
    token: str | None,
    manifest: ModelArtifactManifest | None = None,
    files: tuple[str, ...] = (),
) -> dict:
    result = io.execute(host, hub_script(request, cache, token=token, manifest=manifest, files=files), timeout=7200 if manifest else 90)
    if result.returncode:
        # Avoid printing a shell payload (which contains an optional HF token).
        detail = result.stderr.decode(errors="replace")[-1500:]
        if token:
            detail = detail.replace(token, "<redacted>")
        lines = result.stdout.splitlines()
        unavailable = b"SPARKRUN_MODEL_ERROR=unavailable" in lines or b"SPARKRUN_MODEL_ADAPTER_READY" not in lines
        error_type = (
            ModelInventoryUnavailable
            if manifest is None and unavailable and b"SPARKRUN_MODEL_ERROR=invalid" not in lines
            else ModelArtifactError
        )
        raise error_type(
            "Hugging Face %s on %s failed (rc=%d): %s"
            % ("repair" if manifest else "inventory", host or "control", result.returncode, detail)
        )
    records = [
        line.removeprefix("SPARKRUN_MODEL_RESULT=")
        for line in result.stdout.decode().splitlines()
        if line.startswith("SPARKRUN_MODEL_RESULT=")
    ]
    if len(records) != 1:
        raise ModelArtifactError("Hub adapter returned malformed metadata")
    try:
        return json.loads(records[0])
    except json.JSONDecodeError as exc:
        raise ModelArtifactError("Hub adapter returned malformed metadata") from exc
