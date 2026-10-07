"""Read a prepared HF snapshot on the target. Stdlib-only; no downloads."""

import json
import re
import sys
from pathlib import Path


def resolve_model_path(model, revision, cache_dir, expected_revision=""):
    """Return an exact host-side model path and concrete HF commit."""
    if Path(model).is_absolute():
        path = Path(model)
        if not path.exists():
            raise ValueError("local model path does not exist")
        return {"path": str(path), "revision": ""}
    repo, separator, quant = model.partition(":")
    if any(part in ("", ".", "..") for part in repo.split("/")) or len(repo.split("/")) != 2:
        raise ValueError("model must be a local absolute path or org/repository")
    cache = Path(cache_dir).expanduser().absolute() / "hub" / ("models--" + repo.replace("/", "--"))
    snapshots = cache / "snapshots"
    requested = revision or "main"
    if requested.startswith("/") or ".." in requested.split("/"):
        raise ValueError("invalid model revision")
    if expected_revision:
        commit = expected_revision
    elif re.fullmatch(r"[0-9a-fA-F]{40}", requested):
        commit = requested
    else:
        ref = cache / "refs" / requested
        if not ref.is_file():
            raise ValueError("requested revision has no cached ref; pin a cached commit explicitly")
        commit = ref.read_text().strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("cached revision is not a concrete commit")
    snapshot = snapshots / commit
    if not snapshot.is_dir():
        raise ValueError("required checkpoint revision is not present on this node")
    if separator or "gguf" in repo.lower():
        candidates = sorted(
            p for p in snapshot.rglob("*.gguf") if "mmproj" not in p.name.lower() and (not quant or quant.lower() in p.name.lower())
        )
        if not candidates or any(not p.is_file() for p in candidates):
            raise ValueError("GGUF checkpoint files are absent or incomplete")
        path = candidates[0]
        match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name)
        if match:
            total = int(match[3])
            expected = [path.with_name(f"{match[1]}-{i:05}-of-{total:05}.gguf") for i in range(1, total + 1)]
            if total < 1 or candidates != expected:
                raise ValueError("GGUF checkpoint is ambiguous or has missing shards")
        elif len(candidates) != 1:
            raise ValueError("GGUF checkpoint is ambiguous; select a single quantization")
    else:
        path = snapshot
        if not (path / "config.json").is_file():
            raise ValueError("checkpoint config.json is absent or incomplete")
        indexes = sorted(path.glob("*.index.json"))
        for index in indexes:
            data = json.loads(index.read_text())
            weights = data.get("weight_map") if isinstance(data, dict) else None
            if not isinstance(weights, dict) or not weights or any(not isinstance(name, str) for name in weights.values()):
                raise ValueError("checkpoint index has no valid weight map")
            for name in set(weights.values()):
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts or not (path / relative).is_file():
                    raise ValueError("checkpoint index references a missing or invalid weight file")
    return {"path": str(path), "revision": commit}


if __name__ == "__main__":
    try:
        print("SPARKRUN_MODEL_PATH=" + json.dumps(resolve_model_path(*json.loads(sys.argv[1]))))
    except (ValueError, OSError, KeyError) as exc:
        print("Model path resolution failed: " + str(exc), file=sys.stderr)
        sys.exit(1)
