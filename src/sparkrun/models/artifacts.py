"""Immutable model identities and validation evidence, independent of transport."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import PurePosixPath
import re

MANIFEST_VERSION = 1
SELECTION_VERSION = 1
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST_LENGTHS = {"sha256": 64, "git-sha1": 40}


class ModelArtifactError(ValueError):
    """An artifact cannot be resolved, prepared, or verified as requested."""


class ModelInventoryUnavailable(ModelArtifactError):
    """Expected inventory is unavailable before payload mutation or validation.

    Compatibility mode may fall back only for this condition. Malformed
    manifests, failed validation, repair and transfer errors are not eligible.
    """


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


@dataclass(frozen=True)
class ModelArtifactRequest:
    model: str
    revision: str | None = None
    endpoint: str = "https://huggingface.co"
    projector: str | None = None
    file_selection: str = "auto"

    def __post_init__(self):
        from sparkrun.models.download import parse_gguf_model_spec

        if self.file_selection not in {"auto", "all", "safetensors"}:
            raise ModelArtifactError("model file selection must be auto, all or safetensors")
        repo, _ = parse_gguf_model_spec(self.model)
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)?", repo):
            raise ModelArtifactError("invalid Hugging Face repository: " + repo)
        if self.revision and ("\x00" in self.revision or "\n" in self.revision):
            raise ModelArtifactError("invalid model revision")
        from urllib.parse import urlsplit

        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ModelArtifactError("model endpoint must be an HTTP(S) URL without credentials, query or fragment")

    @property
    def key(self) -> str:
        return _digest({**asdict(self), "revision": self.revision or "main", "selection_version": SELECTION_VERSION})

    @property
    def repo(self) -> str:
        from sparkrun.models.download import parse_gguf_model_spec

        return parse_gguf_model_spec(self.model)[0]


@dataclass(frozen=True)
class ModelArtifactFile:
    path: str
    size: int
    checksum_algorithm: str | None = None
    checksum: str | None = None

    def __post_init__(self):
        path = PurePosixPath(self.path)
        if (
            not self.path
            or self.path == "."
            or path.is_absolute()
            or str(path) != self.path
            or any(p in {".", ".."} for p in path.parts)
            or "\x00" in self.path
        ):
            raise ModelArtifactError("manifest file path must be canonical and relative: " + repr(self.path))
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise ModelArtifactError("manifest file size must be a nonnegative integer")
        if self.checksum_algorithm is None and self.checksum is None:
            return
        length = _DIGEST_LENGTHS.get(self.checksum_algorithm or "")
        if length is None or not isinstance(self.checksum, str) or not re.fullmatch(r"[0-9a-f]{%d}" % length, self.checksum):
            raise ModelArtifactError("invalid or unsupported typed checksum for " + self.path)


@dataclass(frozen=True)
class ModelArtifactManifest:
    repo: str
    revision: str
    endpoint: str
    selection: str
    files: tuple[ModelArtifactFile, ...]
    weight_path: str | None = None
    projector_path: str | None = None
    weight_files: tuple[str, ...] = ()
    provenance: str = "huggingface-model-info"
    schema_version: int = MANIFEST_VERSION
    selection_version: int = SELECTION_VERSION

    def __post_init__(self):
        ModelArtifactRequest(self.repo, endpoint=self.endpoint)
        if self.schema_version != MANIFEST_VERSION or self.selection_version != SELECTION_VERSION:
            raise ModelArtifactError("unsupported model manifest version")
        if not _COMMIT.fullmatch(self.revision):
            raise ModelArtifactError("manifest revision must be an immutable full commit SHA")
        if not isinstance(self.files, tuple) or not self.files or any(not isinstance(f, ModelArtifactFile) for f in self.files):
            raise ModelArtifactError("manifest must contain an immutable nonempty file inventory")
        paths = [f.path for f in self.files]
        if paths != sorted(set(paths)):
            raise ModelArtifactError("manifest paths must be unique and sorted")
        names = set(paths)
        if (
            not isinstance(self.weight_files, tuple)
            or list(self.weight_files) != sorted(set(self.weight_files))
            or set(self.weight_files) - names
        ):
            raise ModelArtifactError("manifest weight selection must be unique, sorted and present")
        if any(str(parent) in names for name in paths for parent in PurePosixPath(name).parents if str(parent) != "."):
            raise ModelArtifactError("manifest file conflicts with a parent directory")
        for selected in (self.weight_path, self.projector_path):
            if selected is not None and selected not in names:
                raise ModelArtifactError("selected runtime file is missing from the manifest")
        if self.provenance not in {"huggingface-model-info", "operator-import"}:
            raise ModelArtifactError("unsupported manifest provenance")

    @property
    def identity(self) -> str:
        return _digest(asdict(self))

    @property
    def relative_snapshot(self) -> str:
        return "hub/models--%s/snapshots/%s" % (self.repo.replace("/", "--"), self.revision)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_json(cls, text: str) -> ModelArtifactManifest:
        try:
            data = json.loads(text)
            data["files"] = tuple(ModelArtifactFile(**entry) for entry in data["files"])
            data["weight_files"] = tuple(data.get("weight_files", ()))
            return cls(**data)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ModelArtifactError("invalid model manifest") from exc


@dataclass(frozen=True)
class ModelFileFailure:
    path: str
    reason: str


@dataclass(frozen=True)
class CacheValidationReport:
    host: str
    manifest_id: str
    level: str
    status: str
    checked_files: int = 0
    failures: tuple[ModelFileFailure, ...] = ()
    detail: str = ""
    access_context: str = "host-session"

    @property
    def complete(self) -> bool:
        return self.status == "complete"


@dataclass(frozen=True)
class ValidatedModelBinding:
    model: str
    host: str
    cache_root: str
    manifest: ModelArtifactManifest
    report: CacheValidationReport
    snapshot_override: str | None = None

    @property
    def snapshot_path(self) -> str:
        if self.snapshot_override is not None:
            return self.snapshot_override
        return str(PurePosixPath(self.cache_root) / self.manifest.relative_snapshot)

    @property
    def model_path(self) -> str:
        if self.manifest.weight_path:
            return str(PurePosixPath(self.snapshot_path) / self.manifest.weight_path)
        return self.snapshot_path
