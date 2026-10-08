# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/coldsnap/LICENSE_EXCEPTION.

"""Schema owned by the ``coldsnap:`` top-level recipe item."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
import re
from typing import Any


DEFAULT_CACHE_PATHS = ("/var/cache/coldsnap/runtime",)

_SAFE_OWNER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

SPARKRUN_OWNED_COMM_ENV = frozenset(
    {
        "GLOO_SOCKET_IFNAME",
        "MN_IF_NAME",
        "NCCL_CROSS_NIC",
        "NCCL_IB_DISABLE",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_HCA",
        "NCCL_IB_MERGE_NICS",
        "NCCL_IB_SUBNET_AWARE_ROUTING",
        "NCCL_NET",
        "NCCL_NET_PLUGIN",
        "NCCL_SOCKET_IFNAME",
        "NODE_IP",
        "OMPI_MCA_btl_tcp_if_include",
        "TP_SOCKET_IFNAME",
        "UCX_NET_DEVICES",
        "VLLM_HOST_IP",
    }
)

SPARKRUN_OWNED_RUNTIME_ENV = frozenset(
    {
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    }
)

COLDSNAP_OWNED_RUNTIME_ENV = frozenset(
    {
        "COLDSNAP_RECOVERY_LOADER_BACKEND",
        "CUTE_DSL_CACHE_DIR",
        "CUDA_CACHE_DISABLE",
        "CUDA_CACHE_MAXSIZE",
        "CUDA_CACHE_PATH",
        "FLASHINFER_CACHE_DIR",
        "FLASHINFER_WORKSPACE_BASE",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
        "NCCL_CHECKPOINT_TERMINATION",
        "NCCL_CUMEM_ENABLE",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "TORCH_HOME",
        "TRITON_CACHE_DIR",
        "TVM_FFI_CACHE_DIR",
        "VLLM_CACHE_ROOT",
        "XDG_CACHE_HOME",
    }
)


@dataclass(frozen=True)
class NativeWeights:
    repository: str = ""
    revision: str = ""
    requires_model_files: bool = False


@dataclass(frozen=True)
class RecoveryWeights:
    loader_backend: str | None = None


@dataclass(frozen=True)
class OCICapsule:
    repository: str = ""


@dataclass(frozen=True)
class OCIArtifact:
    reference: str = ""


@dataclass(frozen=True)
class ColdSnapRecipe:
    format: int = 1
    weight_mode: str | None = None
    native: NativeWeights = field(default_factory=NativeWeights)
    recovery: RecoveryWeights = field(default_factory=RecoveryWeights)
    capsule: OCICapsule = field(default_factory=OCICapsule)
    artifact: OCIArtifact = field(default_factory=OCIArtifact)
    process_backend: str | None = None
    kv_discard: bool | None = None
    async_graphs: bool | None = None
    graph_policy: str | None = None
    shape_calibration: str | None = None
    enforce_captured_driver_floor: bool = False
    cache_seed: bool = True
    cache_paths: tuple[str, ...] = DEFAULT_CACHE_PATHS
    health_path: str | None = None
    prompt: str | None = None
    expected: str | None = None


class ColdSnapRecipeHandler:
    def parse(self, value: Any, recipe) -> ColdSnapRecipe:
        if not isinstance(value, dict):
            raise ValueError("must be a mapping")
        allowed = {"format", "weights", "process", "cache", "capsule", "artifact", "validation", "compatibility"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError("unknown field(s): %s" % ", ".join(unknown))
        weights = _mapping(value.get("weights"), "weights")
        process = _mapping(value.get("process"), "process")
        cache = _mapping(value.get("cache"), "cache")
        validation = _mapping(value.get("validation"), "validation")
        compatibility = _mapping(value.get("compatibility"), "compatibility")
        capsule = _mapping(value.get("capsule"), "capsule")
        artifact = _mapping(value.get("artifact"), "artifact")
        _reject_unknown(weights, {"mode", "native", "recovery"}, "weights")
        native = _mapping(weights.get("native"), "weights.native")
        recovery = _mapping(weights.get("recovery"), "weights.recovery")
        _reject_unknown(native, {"repository", "revision", "requires_model_files"}, "weights.native")
        _reject_unknown(recovery, {"loader_backend"}, "weights.recovery")
        _reject_unknown(process, {"backend", "kv_discard", "async_graphs", "graph_policy", "shape_calibration"}, "process")
        _reject_unknown(cache, {"seed", "paths"}, "cache")
        _reject_unknown(validation, {"health_path", "prompt", "expected"}, "validation")
        _reject_unknown(compatibility, {"enforce_captured_driver_floor"}, "compatibility")
        _reject_unknown(capsule, {"repository"}, "capsule")
        _reject_unknown(artifact, {"reference"}, "artifact")
        cache_seed = _boolean(cache.get("seed", True), "cache.seed")
        cache_paths = _string_tuple(
            cache.get("paths", DEFAULT_CACHE_PATHS if cache_seed else ()),
            "cache.paths",
        )
        return ColdSnapRecipe(
            format=_integer(value.get("format", 1), "format"),
            weight_mode=(_optional_string(weights["mode"], "", "weights.mode") if "mode" in weights else None),
            native=NativeWeights(
                repository=_optional_string(native.get("repository"), "", "weights.native.repository"),
                revision=_optional_string(native.get("revision"), "", "weights.native.revision"),
                requires_model_files=_boolean(native.get("requires_model_files", False), "weights.native.requires_model_files"),
            ),
            recovery=RecoveryWeights(
                loader_backend=(
                    _optional_string(recovery["loader_backend"], "", "weights.recovery.loader_backend")
                    if "loader_backend" in recovery
                    else None
                ),
            ),
            capsule=OCICapsule(
                repository=_optional_string(capsule.get("repository"), "", "capsule.repository"),
            ),
            artifact=OCIArtifact(
                reference=_optional_string(artifact.get("reference"), "", "artifact.reference"),
            ),
            process_backend=(_optional_string(process["backend"], "", "process.backend") if "backend" in process else None),
            kv_discard=(_boolean(process["kv_discard"], "process.kv_discard") if "kv_discard" in process else None),
            async_graphs=(_boolean(process["async_graphs"], "process.async_graphs") if "async_graphs" in process else None),
            graph_policy=(_optional_string(process["graph_policy"], "", "process.graph_policy") if "graph_policy" in process else None),
            shape_calibration=(
                _optional_string(process["shape_calibration"], "", "process.shape_calibration") if "shape_calibration" in process else None
            ),
            enforce_captured_driver_floor=_boolean(
                compatibility.get("enforce_captured_driver_floor", False),
                "compatibility.enforce_captured_driver_floor",
            ),
            cache_seed=cache_seed,
            cache_paths=cache_paths,
            health_path=(
                _optional_string(validation["health_path"], "", "validation.health_path") if "health_path" in validation else None
            ),
            prompt=(_optional_string(validation["prompt"], "", "validation.prompt") if "prompt" in validation else None),
            expected=(_optional_string(validation["expected"], "", "validation.expected") if "expected" in validation else None),
        )

    def validate(self, value: ColdSnapRecipe, recipe) -> list[str]:
        issues: list[str] = []
        recipe_env = set(recipe.env or {})
        cluster_owned = sorted(recipe_env & SPARKRUN_OWNED_COMM_ENV)
        if cluster_owned:
            issues.append("env must not set sparkrun-owned communication variables: %s" % ", ".join(cluster_owned))
        sparkrun_runtime_owned = sorted(recipe_env & SPARKRUN_OWNED_RUNTIME_ENV)
        if sparkrun_runtime_owned:
            issues.append("env must not set sparkrun-owned runtime variables: %s" % ", ".join(sparkrun_runtime_owned))
        runtime_owned = sorted(recipe_env & COLDSNAP_OWNED_RUNTIME_ENV)
        if runtime_owned:
            issues.append("env must not set ColdSnap-owned runtime variables: %s" % ", ".join(runtime_owned))
        if value.format != 1:
            issues.append("format must be 1")
        if value.weight_mode is not None and value.weight_mode not in {"auto", "native", "recovery", "cache-only-auto"}:
            issues.append("weights.mode is unsupported")
        if value.recovery.loader_backend is not None and value.recovery.loader_backend not in {
            "direct",
            "buffered",
            "auto",
            "mmap",
            "torch",
        }:
            issues.append("weights.recovery.loader_backend is unsupported")
        if value.process_backend is not None and value.process_backend != "cuda-criu":
            issues.append("process.backend is unsupported")
        if value.shape_calibration is not None and value.shape_calibration not in {"auto", "enabled", "disabled"}:
            issues.append("process.shape_calibration is unsupported")
        if value.graph_policy is not None and value.graph_policy not in {
            "disabled",
            "preserve-exec",
            "preserve-nccl-exec",
            "recreate-from-plan",
        }:
            issues.append("process.graph_policy is unsupported")
        if value.native.repository and not value.native.revision:
            issues.append("weights.native.revision is required with repository")
        if not recipe.model_revision:
            issues.append("requires top-level model_revision for safetensors recovery")
        if not _valid_capsule_repository(value.capsule.repository):
            issues.append("capsule.repository is invalid")
        if not _valid_oci_artifact_reference(value.artifact.reference):
            issues.append("artifact.reference is invalid")
        if not value.cache_seed and value.cache_paths:
            issues.append("cache.paths requires cache.seed")
        if (
            len(set(value.cache_paths)) != len(value.cache_paths)
            or any(not _safe_cache_path(path) for path in value.cache_paths)
            or _has_overlapping_paths(value.cache_paths)
        ):
            issues.append("cache.paths is invalid")
        if value.health_path is not None and not value.health_path.startswith("/"):
            issues.append("validation.health_path must be absolute")
        return issues

    def export(self, value: ColdSnapRecipe, recipe) -> dict[str, Any]:
        # This normalized form is part of sparkrun's recipe fingerprint and
        # therefore the artifact lookup key. Keep its pre-driver defaults
        # stable. build_request() uses the typed optional fields directly and
        # omits values the recipe did not declare, so this compatibility export
        # does not impose these values on ColdSnap at runtime.
        result: dict[str, Any] = {
            "format": value.format,
            "process": {
                "backend": value.process_backend if value.process_backend is not None else "cuda-criu",
                "kv_discard": value.kv_discard if value.kv_discard is not None else True,
                "async_graphs": value.async_graphs if value.async_graphs is not None else True,
            },
            "weights": {
                "mode": value.weight_mode if value.weight_mode is not None else "auto",
                "recovery": {"loader_backend": value.recovery.loader_backend if value.recovery.loader_backend is not None else "auto"},
            },
            "cache": {"seed": value.cache_seed, "paths": list(value.cache_paths)},
            "capsule": {
                "repository": value.capsule.repository,
            },
            "validation": {
                "health_path": value.health_path if value.health_path is not None else "/health",
                "prompt": (value.prompt if value.prompt is not None else "Reply with exactly: coldsnap-cuda-snapshot-ok"),
                "expected": value.expected if value.expected is not None else "coldsnap-cuda-snapshot-ok",
            },
        }
        if value.native.repository:
            result["weights"]["native"] = {
                "repository": value.native.repository,
                "revision": value.native.revision,
            }
        if value.native.requires_model_files:
            result["weights"].setdefault("native", {})["requires_model_files"] = True
        if value.artifact.reference:
            result["artifact"] = {"reference": value.artifact.reference}
        if value.shape_calibration is not None:
            result["process"]["shape_calibration"] = value.shape_calibration
        if value.graph_policy is not None:
            result["process"]["graph_policy"] = value.graph_policy
        if value.enforce_captured_driver_floor:
            result["compatibility"] = {"enforce_captured_driver_floor": True}
        return result


def _mapping(value: Any, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("%s must be a mapping" % name)
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError("%s must be a boolean" % name)
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s must be an integer" % name)
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise ValueError("%s must be a list of strings" % name)
    result: list[str] = []
    for item in value:
        result.append(_string(item, "%s item" % name))
    return tuple(result)


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("%s must be a non-empty string" % name)
    return value


def _optional_string(value: Any, fallback: str, name: str) -> str:
    if value is None:
        return fallback
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("%s must be a string" % name)
    return value or fallback


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and value == path.as_posix()
        and path.name not in {"", ".", ".."}
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in value
    )


def _safe_cache_path(value: str) -> bool:
    path = PurePosixPath(value)
    if value != path.as_posix() or not path.is_absolute():
        return False
    roots = (
        "/root/.cache/flashinfer",
        "/root/.cache/torch",
        "/root/.cache/torch_extensions",
        "/root/.cache/vllm",
        "/root/.triton/cache",
        "/tmp/coldsnap-derived-cache",
        "/tmp/torchinductor_root",
        "/var/cache/coldsnap",
    )
    return any(value == root or value.startswith(root + "/") for root in roots)


def _valid_capsule_repository(value: str) -> bool:
    if not value:
        return True
    if value != value.lower() or any(character.isspace() for character in value) or "@" in value:
        return False
    if value.startswith("/") or value.endswith("/") or "//" in value:
        return False
    return value.rfind(":") <= value.rfind("/")


def _valid_oci_artifact_reference(value: str) -> bool:
    if not value:
        return True
    if not value.startswith("oci://"):
        return False
    reference = value.removeprefix("oci://")
    if (
        not reference
        or reference != reference.lower()
        or any(character.isspace() for character in reference)
        or reference.startswith("/")
        or reference.endswith("/")
        or "//" in reference
    ):
        return False
    if "@" in reference:
        repository, separator, digest = reference.rpartition("@")
        return bool(repository and separator and re.fullmatch(r"sha256:[0-9a-f]{64}", digest))
    return reference.rfind(":") > reference.rfind("/")


def _has_overlapping_paths(values: tuple[str, ...]) -> bool:
    for index, first in enumerate(values):
        for second in values[index + 1 :]:
            if first == second or first.startswith(second + "/") or second.startswith(first + "/"):
                return True
    return False


def _reject_unknown(value: dict, allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError("%s has unknown field(s): %s" % (name, ", ".join(unknown)))


__all__ = ["ColdSnapRecipe", "ColdSnapRecipeHandler", "NativeWeights", "RecoveryWeights", "OCICapsule", "OCIArtifact"]
