"""Resolve authoritative HF inventories once, then persist portable manifests."""

from __future__ import annotations

from fnmatch import fnmatchcase
import json
from pathlib import Path
import re

from sparkrun.models.artifacts import (
    ModelArtifactError,
    ModelArtifactFile,
    ModelArtifactManifest,
    ModelArtifactRequest,
    ModelInventoryUnavailable,
)

HUB_ADAPTER_VERSION = "1.8.0"
STATE_DIRECTORY = ".sparkrun-model-artifacts"


def select_manifest(request: ModelArtifactRequest, inventory: dict) -> ModelArtifactManifest:
    from sparkrun.models.download import is_gguf_model, parse_gguf_model_spec

    revision = inventory["revision"]
    if request.revision and re.fullmatch(r"[0-9a-f]{40}", request.revision) and revision != request.revision:
        raise ModelArtifactError("Hub returned a different revision than the requested commit")
    files = [ModelArtifactFile(**entry) for entry in inventory["files"]]
    weight, projector = None, None
    weight_files: tuple[str, ...] = ()
    selection = "full-snapshot-v1"
    if not is_gguf_model(request.model) and request.file_selection != "all":
        index = next((f for f in files if f.path == "model.safetensors.index.json"), None)
        if index is not None:
            mapping = inventory.get("weight_map")
            if not isinstance(mapping, dict) or not mapping or any(not isinstance(p, str) for p in mapping.values()):
                raise ModelArtifactError("safetensors index must provide a nonempty weight_map")
            weight_files = tuple(sorted(set(mapping.values())))
            available = {f.path for f in files}
            if set(weight_files) - available or any(not p.endswith(".safetensors") for p in weight_files):
                raise ModelArtifactError("safetensors index references missing or invalid shards")
            files = [f for f in files if f.path in weight_files or not is_alternative_weight(f.path)]
            selection = "safetensors-index-v1"
        elif request.file_selection == "safetensors":
            raise ModelArtifactError("safetensors selection requires model.safetensors.index.json")
    if is_gguf_model(request.model):
        _, quant = parse_gguf_model_spec(request.model)
        if quant:
            files = [f for f in files if fnmatchcase(f.path.lower(), "*%s*" % quant.lower()) or "mmproj" in f.path.lower()]
        weights = [f.path for f in files if f.path.lower().endswith(".gguf") and "mmproj" not in f.path.lower()]
        if not weights:
            raise ModelArtifactError("requested GGUF selection contains no weight files")
        parts: dict[str, list[tuple[int, int, str]]] = {}
        for path in weights:
            match = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf", path, re.IGNORECASE)
            key = match[1] if match else path
            parts.setdefault(key, []).append((int(match[2]), int(match[3]), path) if match else (1, 1, path))
        if len(parts) != 1:
            raise ModelArtifactError("ambiguous GGUF selection; specify a quantization identifying one model")
        shards = sorted(next(iter(parts.values())))
        total = shards[0][1]
        if [part[0] for part in shards] != list(range(1, total + 1)) or any(part[1] != total for part in shards):
            raise ModelArtifactError("GGUF multipart inventory is incomplete or inconsistent")
        weight = shards[0][2]
        projectors = sorted(f.path for f in files if "mmproj" in f.path.lower() and f.path.lower().endswith(".gguf"))
        selector = str(request.projector) if request.projector is not None else None
        disabled = (selector or "").lower() in {"false", "none", "off", "no", "0", "disable", "disabled"}
        if not disabled:
            if selector and selector.lower() not in {"auto", "true", "1", "yes"}:
                projectors = [p for p in projectors if selector.lower() in p.lower()]
                if len(projectors) != 1:
                    raise ModelArtifactError("projector selector must identify exactly one file")
            if projectors:
                projector = min(
                    projectors,
                    key=lambda p: (
                        next(
                            (
                                i
                                for i, v in enumerate(("f16", "bf16", "f32", "f8"))
                                if re.search(r"(?<![a-z0-9])" + v + r"(?![a-z0-9])", p.lower())
                            ),
                            4,
                        ),
                        p,
                    ),
                )
        # Preserve the existing download scope (selected quant + all projectors)
        # while binding the runtime to exactly one validated projector.
        selection = "gguf-v1:%s:%s" % (quant or "", selector or "auto")
    return ModelArtifactManifest(
        request.repo, revision, request.endpoint, selection, tuple(sorted(files, key=lambda f: f.path)), weight, projector, weight_files
    )


def is_alternative_weight(path: str) -> bool:
    """Known checkpoint payloads, never a blanket exclusion of ancillary .bin."""
    name = path.rsplit("/", 1)[-1].lower()
    if "/" in path and path.split("/", 1)[0].lower() not in {"original", "metal"}:
        # Independent components (e.g. a vision tower) may be required despite
        # not appearing in the root language-model index. Keep unfamiliar trees.
        return False
    if name.endswith(".safetensors"):
        return True
    return name.startswith(("model", "pytorch_model", "consolidated", "weights", "params", "tf_model", "flax_model")) and name.endswith(
        (".bin", ".pt", ".pth", ".gguf", ".h5", ".msgpack")
    )


def fetch_inventory(request: ModelArtifactRequest, *, token: str | None = None, cache: str | None = None) -> dict:
    """Required metadata: failures propagate, independent of advisory budgets."""
    from huggingface_hub import HfApi
    from httpx import HTTPError
    from sparkrun.models.hub import configure_hub_client

    configure_hub_client()
    try:
        info = HfApi(endpoint=request.endpoint, token=token).model_info(
            request.repo, revision=request.revision, files_metadata=True, timeout=30
        )
    except (HTTPError, OSError) as exc:
        raise ModelInventoryUnavailable("Hugging Face inventory is unavailable for " + request.repo) from exc
    result = inventory_from_info(info)
    index = next((f for f in info.siblings or () if f.rfilename == "model.safetensors.index.json"), None)
    if index is not None and request.file_selection != "all":
        if index.size is None or index.size > 16 * 1024 * 1024:
            raise ModelArtifactError("safetensors index metadata exceeds the 16 MiB inventory limit; use file selection all")
        entry = next(f for f in result["files"] if f["path"] == index.rfilename)
        try:
            result["weight_map"] = read_weight_index(request.repo, info.sha, request.endpoint, token, entry)
        except (HTTPError, OSError) as exc:
            raise ModelInventoryUnavailable("Hugging Face weight index is unavailable for " + request.repo) from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise ModelArtifactError("invalid authoritative weight index for " + request.repo + ": " + str(exc)) from exc
    return result


def read_weight_index(repo, revision, endpoint, token, entry):
    """Read and authenticate the small index, independent of stale cache files.

    Also embedded verbatim in the pinned remote adapter.
    """
    import hashlib
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from huggingface_hub import hf_hub_download

    with TemporaryDirectory(prefix="sparkrun-index-") as directory:
        path = hf_hub_download(repo, entry["path"], revision=revision, endpoint=endpoint, token=token, local_dir=directory)
        raw = Path(path).read_bytes()
    if len(raw) != entry["size"]:
        raise ValueError("safetensors index size does not match origin inventory")
    algorithm = entry["checksum_algorithm"]
    digest = (
        hashlib.sha256(raw).hexdigest()
        if algorithm == "sha256"
        else hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    )
    if algorithm not in {"sha256", "git-sha1"} or digest != entry["checksum"]:
        raise ValueError("safetensors index checksum does not match origin inventory")
    return json.loads(raw)["weight_map"]


def inventory_from_info(info) -> dict:
    files = []
    for entry in info.siblings or ():
        lfs = entry.lfs
        algorithm, checksum = None, None
        if lfs is not None:
            algorithm, checksum = "sha256", lfs.sha256
        elif entry.blob_id:
            algorithm, checksum = "git-sha1", entry.blob_id
        files.append({"path": entry.rfilename, "size": entry.size, "checksum_algorithm": algorithm, "checksum": checksum})
    return {"revision": info.sha, "files": files}


def manifest_path(cache_root: str, request: ModelArtifactRequest) -> Path:
    return Path(cache_root).expanduser() / STATE_DIRECTORY / (request.key + ".json")


def check_request(manifest: ModelArtifactManifest, request: ModelArtifactRequest) -> None:
    if manifest.repo != request.repo or manifest.endpoint != request.endpoint:
        raise ModelArtifactError("saved manifest does not match the requested repository/endpoint")
    if request.revision and re.fullmatch(r"[0-9a-f]{40}", request.revision) and manifest.revision != request.revision:
        raise ModelArtifactError("saved manifest does not match the requested commit")
    # Re-run selection against the recorded inventory. This catches wrong quant
    # or projector imports without treating observed cache contents as truth.
    selected = select_manifest(
        request,
        {
            "revision": manifest.revision,
            "files": json.loads(manifest.to_json())["files"],
            "weight_map": {p: p for p in manifest.weight_files},
        },
    )
    if (selected.selection, selected.weight_path, selected.projector_path) != (
        manifest.selection,
        manifest.weight_path,
        manifest.projector_path,
    ):
        raise ModelArtifactError("saved manifest does not match the requested selection")
