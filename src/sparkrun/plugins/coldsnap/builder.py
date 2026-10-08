# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/coldsnap/LICENSE_EXCEPTION.

"""Plugin-owned builder for deriving a ColdSnap inference runtime image.

This is deliberately homed with the ColdSnap integration rather than under
``sparkrun.builders``: disabling the first-party plugin removes its recipe
item, CLI, execution strategy, and builder as one capability.

The builder runs on the controller for ordinary transfer modes or on the
delegated head. It stages verified sources from the controller without forwarding credentials,
observes the base image's NCCL identity, selects the newest capability-admitted
same-major provider, reuses its published multiarch payload when available,
and invokes ColdSnap's canonical engine Dockerfile against the input image.
Missing payload platforms fall back to a target-architecture source build.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import logging
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from sparkrun.builders.base import BuilderPlugin
from sparkrun.core.progress import PROGRESS, progress_heartbeat
from sparkrun.orchestration.primitives import run_script_on_host, run_script_on_host_streaming
from sparkrun.orchestration.ssh import run_rsync, should_run_locally
from sparkrun.plugins.coldsnap.git_sources import clone_pinned_source
from sparkrun.plugins.coldsnap.tool import DEFAULT_CONTROLLER_COMMIT, DEFAULT_CONTROLLER_VERSION
from sparkrun.utils.shell import quote


logger = logging.getLogger(__name__)

_BUILDER_SCHEMA = "coldsnap-inference-provider-v9"
_DEFAULT_CRIU_IMAGE = "ghcr.io/sparksq/criu@sha256:2ff53a61af48e7e676bd4d64747394ca7c7622840ef0c740e719e6ecadb0d07c"
_COLDSNAP_RUNTIME_LABEL = "io.sparksq.coldsnap.runtime"
_COLDSNAP_DRIVERS_LABEL = "io.sparksq.coldsnap.snapshot-drivers"
_PINNED_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CUDA_ARCH = re.compile(r"^[0-9]{2,3}$")
_NCCL_POLICIES = {"exact", "match-or-latest-qualified"}
_NCCL_LATEST_MESSAGE = re.compile(
    r"\[coldsnap-builder\] base NCCL ([0-9.]+); "
    r"using latest qualified same-major provider ([0-9.]+)"
)


@dataclass(frozen=True)
class GitSource:
    name: str
    url: str
    ref: str
    revision: str


_DEFAULT_SOURCES = (
    GitSource(
        "coldsnap",
        "https://github.com/sparksq/coldsnap.git",
        "refs/tags/v%s" % DEFAULT_CONTROLLER_VERSION,
        DEFAULT_CONTROLLER_COMMIT,
    ),
    GitSource(
        "go_criu",
        "https://github.com/sparksq/go-criu.git",
        "29a4f2f8e8374d38319a9851d9c1ef880dd0a0e8",
        "29a4f2f8e8374d38319a9851d9c1ef880dd0a0e8",
    ),
    GitSource(
        "cuda_checkpoint",
        "https://github.com/sparksq/cuda-checkpoint.git",
        "00d5cce84c628088d6caa203fc4af40c1538b6f7",
        "00d5cce84c628088d6caa203fc4af40c1538b6f7",
    ),
)


@dataclass(frozen=True)
class BuildSettings:
    output_repository: str
    docker_command: str
    cuda_arch: str
    build_jobs: int
    rebuild: bool
    nccl_policy: str
    criu_image: str
    sources: tuple[GitSource, ...]


@dataclass(frozen=True)
class ColdSnapBuildPlan:
    engine: str
    input_image: str
    output_image: str
    nccl_image: str
    fingerprint: str
    cuda_arch: str
    build_jobs: int
    docker_command: str
    nccl_policy: str
    criu_image: str
    snapshot_driver: str
    sources: tuple[GitSource, ...]
    docker_platform: str


def _string_setting(values: Mapping[str, Any], name: str, default: str) -> str:
    value = values.get(name, default)
    if not isinstance(value, str) or not value.strip() or any(character in value for character in "\r\n\x00"):
        raise ValueError("defaults.builders.coldsnap.%s must be a non-empty string" % name)
    return value.strip()


def _normalize_cuda_arch(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"", "auto"}:
        return ""
    normalized = normalized.removeprefix("compute_").removeprefix("sm_").replace(".", "")
    if not _CUDA_ARCH.fullmatch(normalized):
        raise ValueError("defaults.builders.coldsnap.cuda_arch must be auto or a CUDA architecture such as 121")
    return normalized


def _validate_output_repository(value: str) -> str:
    if any(character.isspace() for character in value) or "@" in value or value.endswith("/"):
        raise ValueError("defaults.builders.coldsnap.output_repository must be an untagged OCI repository")
    # A colon in the final path component is an image tag. A registry port in
    # an earlier component remains valid (for example localhost:5500/repo).
    if ":" in value.rsplit("/", 1)[-1]:
        raise ValueError("defaults.builders.coldsnap.output_repository must not include an image tag")
    return value


def _normalize_nccl_policy(value: object) -> str:
    if not isinstance(value, str) or value not in _NCCL_POLICIES:
        raise ValueError("defaults.builders.coldsnap.nccl_policy must be exact or match-or-latest-qualified")
    return value


def _resolve_settings(recipe: Any, config: Any | None, engine: str = "vllm") -> BuildSettings:
    if engine not in {"vllm", "sglang"}:
        raise ValueError("ColdSnap builder engine must be vllm or sglang")
    values: dict[str, Any] = {}
    if config is not None:
        configured = config.get_defaults_builder("coldsnap")
        if isinstance(configured, dict):
            values.update(configured)
    recipe_values = getattr(recipe, "builder_config", None)
    if isinstance(recipe_values, dict):
        values.update(recipe_values)

    output_repository = _validate_output_repository(_string_setting(values, "output_repository", "sparkrun/coldsnap-%s" % engine))
    docker_command = _string_setting(values, "docker_command", "docker")
    if any(character.isspace() for character in docker_command):
        raise ValueError("defaults.builders.coldsnap.docker_command must name one executable")
    cuda_arch = _normalize_cuda_arch(str(values.get("cuda_arch", "auto")))
    build_jobs = values.get("build_jobs", 8)
    if isinstance(build_jobs, bool) or not isinstance(build_jobs, int) or not 1 <= build_jobs <= 256:
        raise ValueError("defaults.builders.coldsnap.build_jobs must be an integer in [1, 256]")
    rebuild = values.get("rebuild", False)
    if not isinstance(rebuild, bool):
        raise ValueError("builder_config.rebuild must be a boolean")
    nccl_policy = _normalize_nccl_policy(values.get("nccl_policy", "match-or-latest-qualified"))
    criu_image = _string_setting(values, "criu_image", _DEFAULT_CRIU_IMAGE)
    if not _PINNED_IMAGE.fullmatch(criu_image):
        raise ValueError("defaults.builders.coldsnap.criu_image must be digest-pinned")

    sources: list[GitSource] = []
    for default in _DEFAULT_SOURCES:
        prefix = default.name
        source = GitSource(
            name=default.name,
            url=_string_setting(values, "%s_url" % prefix, default.url),
            ref=_string_setting(values, "%s_ref" % prefix, default.ref),
            revision=_string_setting(values, "%s_revision" % prefix, default.revision).lower(),
        )
        if not _COMMIT.fullmatch(source.revision):
            raise ValueError("defaults.builders.coldsnap.%s_revision must be a full Git commit" % prefix)
        sources.append(source)
    return BuildSettings(
        output_repository=output_repository,
        docker_command=docker_command,
        cuda_arch=cuda_arch,
        build_jobs=build_jobs,
        rebuild=rebuild,
        nccl_policy=nccl_policy,
        criu_image=criu_image,
        sources=tuple(sources),
    )


def _build_plan(
    image: str,
    settings: BuildSettings,
    cuda_arch: str,
    snapshot_driver: str = "n610",
    engine: str = "vllm",
    *,
    docker_platform: str,
) -> ColdSnapBuildPlan:
    if not _PINNED_IMAGE.fullmatch(image):
        raise ValueError("builder: coldsnap requires a digest-pinned container: name@sha256:<64 lowercase hex characters>")
    if not _CUDA_ARCH.fullmatch(cuda_arch):
        raise ValueError("ColdSnap builder requires a resolved CUDA architecture")
    if snapshot_driver not in {"n580", "n610"}:
        raise ValueError("ColdSnap builder snapshot driver must be n580 or n610")
    if engine not in {"vllm", "sglang"}:
        raise ValueError("ColdSnap builder engine must be vllm or sglang")
    if docker_platform not in {"linux/arm64", "linux/amd64", "linux/target"}:
        raise ValueError("ColdSnap builder requires a supported Linux Docker platform")
    identity = {
        "schema": _BUILDER_SCHEMA,
        "engine": engine,
        "input_image": image,
        "cuda_arch": cuda_arch,
        "nccl_policy": settings.nccl_policy,
        "criu_image": settings.criu_image,
        "snapshot_driver": snapshot_driver,
        "docker_platform": docker_platform,
        "sources": [source.__dict__ for source in settings.sources],
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    short = fingerprint[:20]
    return ColdSnapBuildPlan(
        engine=engine,
        input_image=image,
        output_image="%s:%s" % (settings.output_repository, short),
        nccl_image="sparkrun/coldsnap-nccl:%s" % short,
        fingerprint=fingerprint,
        cuda_arch=cuda_arch,
        build_jobs=settings.build_jobs,
        docker_command=settings.docker_command,
        nccl_policy=settings.nccl_policy,
        criu_image=settings.criu_image,
        snapshot_driver=snapshot_driver,
        sources=settings.sources,
        docker_platform=docker_platform,
    )


def _dry_run_plan(image: str, settings: BuildSettings, engine: str = "vllm") -> ColdSnapBuildPlan:
    # Dry runs cannot query a GPU. Keep the output stable and visibly distinct
    # from a real architecture-specific build.
    return _build_plan(image, settings, settings.cuda_arch or "000", "n610", engine, docker_platform="linux/target")


def _source_variables(sources: tuple[GitSource, ...]) -> str:
    lines: list[str] = []
    for source in sources:
        prefix = "COLDSNAP_SOURCE_%s" % source.name.upper()
        lines.extend(
            (
                "%s_URL=%s" % (prefix, quote(source.url)),
                "%s_REF=%s" % (prefix, quote(source.ref)),
                "%s_REVISION=%s" % (prefix, quote(source.revision)),
            )
        )
    return "\n".join(lines)


def render_build_script(plan: ColdSnapBuildPlan, *, source_root: str = "") -> str:
    """Render the delegated/local BuildKit conversion script."""
    return """#!/usr/bin/env bash
set -euo pipefail

COLDSNAP_DOCKER=%(docker)s
COLDSNAP_DOCKER_PLATFORM=%(docker_platform)s
COLDSNAP_ENGINE=%(engine)s
COLDSNAP_INPUT_IMAGE=%(input)s
COLDSNAP_OUTPUT_IMAGE=%(output)s
COLDSNAP_NCCL_IMAGE=%(nccl_image)s
COLDSNAP_FINGERPRINT=%(fingerprint)s
COLDSNAP_CUDA_ARCH=%(cuda_arch)s
COLDSNAP_BUILD_JOBS=%(build_jobs)s
COLDSNAP_NCCL_POLICY=%(nccl_policy)s
COLDSNAP_CRIU_IMAGE=%(criu_image)s
COLDSNAP_SNAPSHOT_DRIVER=%(snapshot_driver)s
COLDSNAP_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/sparkrun/coldsnap-builder"
COLDSNAP_SOURCE_ROOT="$COLDSNAP_CACHE_ROOT/sources"
%(prepared_sources)s
COLDSNAP_BUILD_ROOT="$COLDSNAP_CACHE_ROOT/builds/$COLDSNAP_FINGERPRINT"
%(source_variables)s

mkdir -p "$COLDSNAP_SOURCE_ROOT" "$COLDSNAP_BUILD_ROOT"

sync_source() {
  local coldsnap_source_name="$1"
  local coldsnap_source_url="$2"
  local coldsnap_source_ref="$3"
  local coldsnap_source_revision="$4"
  local coldsnap_source_destination="$COLDSNAP_SOURCE_ROOT/${coldsnap_source_name}-${coldsnap_source_revision}"
  local coldsnap_source_actual=""

  if [ -d "$coldsnap_source_destination/.git" ]; then
    coldsnap_source_actual="$(git -C "$coldsnap_source_destination" rev-parse HEAD 2>/dev/null || true)"
    if [ "$coldsnap_source_actual" = "$coldsnap_source_revision" ] \
       && git -C "$coldsnap_source_destination" diff --quiet \
       && git -C "$coldsnap_source_destination" diff --cached --quiet \
       && [ -z "$(git -C "$coldsnap_source_destination" ls-files --others --exclude-standard)" ]; then
      echo "[coldsnap-builder] source cache hit: $coldsnap_source_name@$coldsnap_source_revision" >&2
      printf '%%s\n' "$coldsnap_source_destination"
      return
    fi
  fi
  case "$coldsnap_source_destination" in
    "$COLDSNAP_SOURCE_ROOT"/*) rm -rf -- "$coldsnap_source_destination" ;;
    *) echo "[coldsnap-builder] refusing unsafe source path: $coldsnap_source_destination" >&2; exit 1 ;;
  esac
  echo "[coldsnap-builder] fetching $coldsnap_source_name from $coldsnap_source_url ($coldsnap_source_ref)" >&2
  # Bash disables errexit inside command substitution: check every operation.
  git init -q "$coldsnap_source_destination" || return 1
  git -C "$coldsnap_source_destination" remote add origin "$coldsnap_source_url" || return 1
  if ! GIT_TERMINAL_PROMPT=0 git -C "$coldsnap_source_destination" fetch -q --depth=1 origin "$coldsnap_source_ref"; then
    echo "[coldsnap-builder] failed to fetch pinned $coldsnap_source_name source; build stopped" >&2
    return 1
  fi
  git -C "$coldsnap_source_destination" checkout -q --detach FETCH_HEAD || return 1
  coldsnap_source_actual="$(git -C "$coldsnap_source_destination" rev-parse --verify 'HEAD^{commit}')" || return 1
  if [ "$coldsnap_source_actual" != "$coldsnap_source_revision" ]; then
    echo "[coldsnap-builder] $coldsnap_source_name resolved to $coldsnap_source_actual, expected $coldsnap_source_revision" >&2
    exit 1
  fi
  printf '%%s\n' "$coldsnap_source_destination"
}

COLDSNAP_SOURCE_COLDSNAP_DIR="$(sync_source coldsnap "$COLDSNAP_SOURCE_COLDSNAP_URL" "$COLDSNAP_SOURCE_COLDSNAP_REF" "$COLDSNAP_SOURCE_COLDSNAP_REVISION")"
COLDSNAP_SOURCE_GO_CRIU_DIR="$(sync_source go-criu "$COLDSNAP_SOURCE_GO_CRIU_URL" "$COLDSNAP_SOURCE_GO_CRIU_REF" "$COLDSNAP_SOURCE_GO_CRIU_REVISION")"
COLDSNAP_SOURCE_CUDA_CHECKPOINT_DIR="$(sync_source cuda-checkpoint "$COLDSNAP_SOURCE_CUDA_CHECKPOINT_URL" "$COLDSNAP_SOURCE_CUDA_CHECKPOINT_REF" "$COLDSNAP_SOURCE_CUDA_CHECKPOINT_REVISION")"
COLDSNAP_BASE_NCCL_RELEASE="$("$COLDSNAP_DOCKER" run --rm --network none --platform "$COLDSNAP_DOCKER_PLATFORM" --gpus all --entrypoint python3 "$COLDSNAP_INPUT_IMAGE" -c \
  'import ctypes; v=ctypes.c_int(); n=ctypes.CDLL("libnccl.so.2"); r=n.ncclGetVersion(ctypes.byref(v)); assert r == 0 and v.value > 0; print(f"{v.value // 10000}.{(v.value // 100) %% 100}.{v.value %% 100}")')"
COLDSNAP_NCCL_SELECTION_FILE="$COLDSNAP_BUILD_ROOT/nccl-selection.txt"
python3 - "$COLDSNAP_SOURCE_COLDSNAP_DIR/native/nccl/releases" "$COLDSNAP_BASE_NCCL_RELEASE" "$COLDSNAP_NCCL_POLICY" >"$COLDSNAP_NCCL_SELECTION_FILE" <<'PY'
import json
from pathlib import Path
import sys

release_root = Path(sys.argv[1])
base_release = sys.argv[2]
policy = sys.argv[3]
base = tuple(int(part) for part in base_release.split("."))
if len(base) != 3 or policy not in {"exact", "match-or-latest-qualified"}:
    raise SystemExit("[coldsnap-builder] invalid NCCL selection input")

candidates = []
for directory in sorted(release_root.iterdir()):
    try:
        recipe = json.loads((directory / "recipe.json").read_text(encoding="utf-8"))
        qualification = json.loads((directory / "qualification.json").read_text(encoding="utf-8"))
        release = tuple(int(part) for part in recipe["nccl_release"].split("."))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        continue
    if (
        len(release) != 3
        or recipe.get("format") != 2
        or qualification.get("format") != 2
        or qualification.get("provider_id") != recipe.get("provider_id")
        or qualification.get("state") != "accepted"
        or qualification.get("policy") != "production"
        or qualification.get("capabilities") != recipe.get("capabilities")
        or not qualification.get("checks")
        or "ib-roce" not in qualification.get("transports", [])
    ):
        continue
    if release == base or (
        policy == "match-or-latest-qualified"
        and release[0] == base[0]
        and release > base
    ):
        candidates.append((release, int(recipe.get("provider_revision", 0)), directory))

exact = [candidate for candidate in candidates if candidate[0] == base]
if policy == "exact" and len(exact) > 1:
    raise SystemExit(f"[coldsnap-builder] multiple qualified exact providers for NCCL {base_release}")
if policy == "exact" and exact:
    selected = exact[0]
    mode = "exact"
else:
    if not candidates:
        raise SystemExit(
            f"[coldsnap-builder] no qualified NCCL provider for base {base_release} under policy {policy}"
        )
    selected = max(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    mode = "latest-qualified-same-major"

print(selected[2])
print(".".join(str(part) for part in selected[0]))
print(mode)
PY
mapfile -t COLDSNAP_NCCL_SELECTION < "$COLDSNAP_NCCL_SELECTION_FILE"
if [ "${#COLDSNAP_NCCL_SELECTION[@]}" -ne 3 ]; then
  echo "[coldsnap-builder] NCCL selector returned an invalid result" >&2
  exit 1
fi
COLDSNAP_NCCL_RELEASE_DIR="${COLDSNAP_NCCL_SELECTION[0]}"
COLDSNAP_NCCL_RELEASE="${COLDSNAP_NCCL_SELECTION[1]}"
COLDSNAP_NCCL_SELECTION_MODE="${COLDSNAP_NCCL_SELECTION[2]}"
if [ "$COLDSNAP_NCCL_SELECTION_MODE" = "latest-qualified-same-major" ]; then
  echo "[coldsnap-builder] base NCCL $COLDSNAP_BASE_NCCL_RELEASE; using latest qualified same-major provider $COLDSNAP_NCCL_RELEASE" >&2
else
  echo "[coldsnap-builder] using exact qualified NCCL provider $COLDSNAP_NCCL_RELEASE" >&2
fi
COLDSNAP_NCCL_RELEASE_TAG="$(basename "$COLDSNAP_NCCL_RELEASE_DIR")"
COLDSNAP_NCCL_SOURCE_LOCK="$COLDSNAP_NCCL_RELEASE_DIR/source.lock"
COLDSNAP_NCCL_RECIPE="$COLDSNAP_NCCL_RELEASE_DIR/recipe.json"
COLDSNAP_NCCL_SOURCE_URL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["repository"])' "$COLDSNAP_NCCL_SOURCE_LOCK")"
COLDSNAP_NCCL_SOURCE_TAG="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["signed_tag"])' "$COLDSNAP_NCCL_SOURCE_LOCK")"
COLDSNAP_NCCL_SOURCE_REVISION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["commit"])' "$COLDSNAP_NCCL_SOURCE_LOCK")"
COLDSNAP_NCCL_PROVIDER_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["provider_id"])' "$COLDSNAP_NCCL_RECIPE")"
COLDSNAP_NCCL_PROVIDER_REVISION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["provider_revision"])' "$COLDSNAP_NCCL_RECIPE")"
COLDSNAP_NCCL_PAYLOAD_REPOSITORY="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["builder"]["payload_repository"])' "$COLDSNAP_NCCL_RECIPE")"
COLDSNAP_NCCL_PAYLOAD_REPODIGEST_REPOSITORY="${COLDSNAP_NCCL_PAYLOAD_REPOSITORY#docker.io/}"
COLDSNAP_NCCL_PAYLOAD_TAG="${COLDSNAP_NCCL_RELEASE_TAG}.coldsnap.${COLDSNAP_NCCL_PROVIDER_REVISION}"
COLDSNAP_NCCL_PUBLISHED_PAYLOAD="${COLDSNAP_NCCL_PAYLOAD_REPOSITORY}:${COLDSNAP_NCCL_PAYLOAD_TAG}"
COLDSNAP_NCCL_LOCAL_PAYLOAD="${COLDSNAP_NCCL_IMAGE}-payload"
COLDSNAP_NCCL_BUILD_IMAGE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["builder"]["payload_build_image"])' "$COLDSNAP_NCCL_RECIPE")"
case "$COLDSNAP_NCCL_BUILD_IMAGE" in
  *@sha256:????????????????????????????????????????????????????????????????) ;;
  *) echo "[coldsnap-builder] NCCL recipe payload_build_image must be digest-pinned" >&2; exit 2 ;;
esac
COLDSNAP_NCCL_VERSION_CODE="$(python3 -c 'import sys; a,b,c=(int(v) for v in sys.argv[1].split(".")); print(a*10000+b*100+c)' "$COLDSNAP_NCCL_RELEASE")"
COLDSNAP_NCCL_REPRODUCIBLE_NVCC="$(python3 -c 'import json,sys; v=json.load(open(sys.argv[1], encoding="utf-8"))["build"]["use_reproducible_nvcc"]; assert isinstance(v, bool); print(int(v))' "$COLDSNAP_NCCL_RECIPE")"
COLDSNAP_NCCL_STRIP_OUTPUTS="$(python3 -c 'import json,sys; v=json.load(open(sys.argv[1], encoding="utf-8"))["build"]["strip_unneeded"]; assert isinstance(v, bool); print(int(v))' "$COLDSNAP_NCCL_RECIPE")"
COLDSNAP_NCCL_TLS_BACKEND="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["build"].get("tls_backend", ""))' "$COLDSNAP_NCCL_RECIPE")"
COLDSNAP_NCCL_PAYLOAD_IMAGE=""
COLDSNAP_NCCL_PAYLOAD_FALLBACK_REASON="published NCCL payload unavailable"
COLDSNAP_ALLOW_LOCAL_NCCL_PAYLOAD=0
COLDSNAP_NCCL_HOST_ARCH="${COLDSNAP_DOCKER_PLATFORM#linux/}"

echo "[coldsnap-builder] resolving published NCCL payload $COLDSNAP_NCCL_PUBLISHED_PAYLOAD" >&2
if "$COLDSNAP_DOCKER" pull --platform "$COLDSNAP_DOCKER_PLATFORM" "$COLDSNAP_NCCL_PUBLISHED_PAYLOAD"; then
  COLDSNAP_NCCL_PAYLOAD_ARCH="$("$COLDSNAP_DOCKER" image inspect \
    --format '{{.Architecture}}' "$COLDSNAP_NCCL_PUBLISHED_PAYLOAD" 2>/dev/null || true)"
  COLDSNAP_NCCL_PAYLOAD_COMPATIBLE=0
  if [ "$COLDSNAP_NCCL_PAYLOAD_ARCH" != "$COLDSNAP_NCCL_HOST_ARCH" ]; then
    COLDSNAP_NCCL_PAYLOAD_FALLBACK_REASON="published NCCL payload architecture $COLDSNAP_NCCL_PAYLOAD_ARCH does not match $COLDSNAP_NCCL_HOST_ARCH"
  else
    COLDSNAP_NCCL_PAYLOAD_GENCODE="$("$COLDSNAP_DOCKER" image inspect \
      --format '{{index .Config.Labels "io.sparksq.coldsnap.cuda.gencode"}}' \
      "$COLDSNAP_NCCL_PUBLISHED_PAYLOAD" 2>/dev/null || true)"
    for coldsnap_gencode_token in $COLDSNAP_NCCL_PAYLOAD_GENCODE; do
      case "$coldsnap_gencode_token" in
        *,code=sm_"$COLDSNAP_CUDA_ARCH") COLDSNAP_NCCL_PAYLOAD_COMPATIBLE=1; break ;;
      esac
    done
    if [ "$COLDSNAP_NCCL_PAYLOAD_COMPATIBLE" != 1 ]; then
      COLDSNAP_NCCL_PAYLOAD_FALLBACK_REASON="published NCCL payload lacks sm_$COLDSNAP_CUDA_ARCH"
    fi
  fi
  if [ "$COLDSNAP_NCCL_PAYLOAD_COMPATIBLE" = 1 ]; then
    while IFS= read -r coldsnap_payload_digest; do
      case "$coldsnap_payload_digest" in
        "$COLDSNAP_NCCL_PAYLOAD_REPOSITORY"@sha256:????????????????????????????????????????????????????????????????|"$COLDSNAP_NCCL_PAYLOAD_REPODIGEST_REPOSITORY"@sha256:????????????????????????????????????????????????????????????????)
          COLDSNAP_NCCL_PAYLOAD_IMAGE="$coldsnap_payload_digest"
          break
          ;;
      esac
    done < <("$COLDSNAP_DOCKER" image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$COLDSNAP_NCCL_PUBLISHED_PAYLOAD")
    if [ -z "$COLDSNAP_NCCL_PAYLOAD_IMAGE" ]; then
      echo "[coldsnap-builder] published NCCL payload did not resolve to a repository digest" >&2
      exit 2
    fi
    echo "[coldsnap-builder] using published NCCL payload $COLDSNAP_NCCL_PAYLOAD_IMAGE" >&2
  fi
fi
if [ -z "$COLDSNAP_NCCL_PAYLOAD_IMAGE" ]; then
  echo "[coldsnap-builder] $COLDSNAP_NCCL_PAYLOAD_FALLBACK_REASON; building this CUDA architecture locally" >&2
  COLDSNAP_SOURCE_NCCL_DIR="$(sync_source nccl "$COLDSNAP_NCCL_SOURCE_URL" "refs/tags/$COLDSNAP_NCCL_SOURCE_TAG" "$COLDSNAP_NCCL_SOURCE_REVISION")"
  if ! "$COLDSNAP_DOCKER" image inspect "$COLDSNAP_NCCL_LOCAL_PAYLOAD" >/dev/null 2>&1; then
    DOCKER_BUILDKIT=1 "$COLDSNAP_DOCKER" build \
      --platform "$COLDSNAP_DOCKER_PLATFORM" \
      --pull=false \
      --file "$COLDSNAP_SOURCE_COLDSNAP_DIR/deploy/nccl/Dockerfile.payload" \
      --target provider_payload_oci \
      --build-arg "NCCL_BUILD_IMAGE=$COLDSNAP_NCCL_BUILD_IMAGE" \
      --build-arg "NCCL_LIBRARY_RELEASE=$COLDSNAP_NCCL_RELEASE" \
      --build-arg "NCCL_VERSION_CODE=$COLDSNAP_NCCL_VERSION_CODE" \
      --build-arg "NCCL_TLS_BACKEND=$COLDSNAP_NCCL_TLS_BACKEND" \
      --build-arg "NCCL_REPRODUCIBLE_NVCC=$COLDSNAP_NCCL_REPRODUCIBLE_NVCC" \
      --build-arg "NCCL_STRIP_OUTPUTS=$COLDSNAP_NCCL_STRIP_OUTPUTS" \
      --build-arg "NCCL_BUILD_JOBS=$COLDSNAP_BUILD_JOBS" \
      --build-arg "NCCL_NVCC_GENCODE=-gencode=arch=compute_${COLDSNAP_CUDA_ARCH},code=sm_${COLDSNAP_CUDA_ARCH}" \
      --build-arg "COLDSNAP_NCCL_PROVIDER_ID=$COLDSNAP_NCCL_PROVIDER_ID" \
      --build-arg "COLDSNAP_SOURCE_REVISION=$COLDSNAP_SOURCE_COLDSNAP_REVISION" \
      --build-context "nccl_source=$COLDSNAP_SOURCE_NCCL_DIR" \
      --build-context "nccl_release=$COLDSNAP_NCCL_RELEASE_DIR" \
      --tag "$COLDSNAP_NCCL_LOCAL_PAYLOAD" \
      "$COLDSNAP_SOURCE_COLDSNAP_DIR"
  fi
  COLDSNAP_NCCL_PAYLOAD_IMAGE="$COLDSNAP_NCCL_LOCAL_PAYLOAD"
  COLDSNAP_ALLOW_LOCAL_NCCL_PAYLOAD=1
fi

if ! "$COLDSNAP_DOCKER" image inspect "$COLDSNAP_NCCL_IMAGE" >/dev/null 2>&1; then
  echo "[coldsnap-builder] assembling qualified NCCL provider for the inference image"
  DOCKER_BUILDKIT=1 "$COLDSNAP_DOCKER" build \
    --platform "$COLDSNAP_DOCKER_PLATFORM" \
    --pull=false \
    --file "$COLDSNAP_SOURCE_COLDSNAP_DIR/deploy/nccl/Dockerfile.provider" \
    --build-arg "TARGET_IMAGE=$COLDSNAP_INPUT_IMAGE" \
    --build-arg "NCCL_PAYLOAD_IMAGE=$COLDSNAP_NCCL_PAYLOAD_IMAGE" \
    --build-arg "COLDSNAP_ALLOW_LOCAL_NCCL_PAYLOAD=$COLDSNAP_ALLOW_LOCAL_NCCL_PAYLOAD" \
    --build-arg "NCCL_RELEASE_TAG=$COLDSNAP_NCCL_RELEASE_TAG" \
    --build-arg "COLDSNAP_NCCL_POLICY=$COLDSNAP_NCCL_POLICY" \
    --tag "$COLDSNAP_NCCL_IMAGE" \
    "$COLDSNAP_SOURCE_COLDSNAP_DIR"
fi

echo "[coldsnap-builder] deriving $COLDSNAP_OUTPUT_IMAGE from $COLDSNAP_INPUT_IMAGE"
DOCKER_BUILDKIT=1 "$COLDSNAP_DOCKER" build \
  --platform "$COLDSNAP_DOCKER_PLATFORM" \
  --pull=false \
  --file "$COLDSNAP_SOURCE_COLDSNAP_DIR/deploy/%(engine_path)s/Dockerfile" \
  --build-arg "%(base_image_arg)s=$COLDSNAP_INPUT_IMAGE" \
  --build-arg "CRIU_IMAGE=$COLDSNAP_CRIU_IMAGE" \
  --build-context "criu_image=docker-image://$COLDSNAP_CRIU_IMAGE" \
  --build-context "go_criu_source=$COLDSNAP_SOURCE_GO_CRIU_DIR" \
  --build-context "cuda_checkpoint_source=$COLDSNAP_SOURCE_CUDA_CHECKPOINT_DIR" \
  --build-context "nccl_provider=docker-image://$COLDSNAP_NCCL_IMAGE" \
  --label "io.sparksq.coldsnap.builder.fingerprint=$COLDSNAP_FINGERPRINT" \
  --label "io.sparksq.coldsnap.nccl.base=$COLDSNAP_BASE_NCCL_RELEASE" \
  --label "io.sparksq.coldsnap.nccl.provider=$COLDSNAP_NCCL_RELEASE" \
  --label "io.sparksq.coldsnap.nccl.selection=$COLDSNAP_NCCL_SELECTION_MODE" \
  --label "io.sparksq.coldsnap.snapshot-drivers=%(snapshot_drivers)s" \
  --label "io.sparksq.coldsnap.builder.snapshot-driver=$COLDSNAP_SNAPSHOT_DRIVER" \
  --tag "$COLDSNAP_OUTPUT_IMAGE" \
  "$COLDSNAP_SOURCE_COLDSNAP_DIR"

COLDSNAP_RUNTIME_LABEL="$("$COLDSNAP_DOCKER" image inspect --format '{{ index .Config.Labels "io.sparksq.coldsnap.runtime" }}' "$COLDSNAP_OUTPUT_IMAGE")"
if [ "$COLDSNAP_RUNTIME_LABEL" != %(runtime_label)s ]; then
  echo "[coldsnap-builder] derived image has incompatible ColdSnap runtime label: $COLDSNAP_RUNTIME_LABEL" >&2
  exit 1
fi
"$COLDSNAP_DOCKER" run --rm --network none --platform "$COLDSNAP_DOCKER_PLATFORM" %(validation_gpu)s--entrypoint python3 "$COLDSNAP_OUTPUT_IMAGE" -c \
  %(plugin_validation)s
echo "[coldsnap-builder] ready: $COLDSNAP_OUTPUT_IMAGE"
""" % {
        "docker": quote(plan.docker_command),
        "docker_platform": quote(plan.docker_platform),
        "prepared_sources": "COLDSNAP_SOURCE_ROOT=%s" % quote(source_root) if source_root else "",
        "engine": quote(plan.engine),
        "engine_path": plan.engine,
        "base_image_arg": "SGLANG_IMAGE" if plan.engine == "sglang" else "VLLM_IMAGE",
        "snapshot_drivers": "n580,n610",
        "runtime_label": quote("%s-cuda-criu-v1" % plan.engine),
        "validation_gpu": "--gpus all -e EP_SUPPRESS_NCCL_CHECK=0 " if plan.engine == "sglang" else "",
        "plugin_validation": quote(
            "import coldsnap_sglang; from importlib.metadata import entry_points; "
            'assert any(ep.name == "coldsnap" for ep in entry_points(group="sglang.srt.plugins"))'
            "; "
            'import deep_ep; from pathlib import Path; import json; active=json.load(open("/opt/coldsnap/nccl/active.json", encoding="utf-8")); assert Path(active["files"]["nccl-runtime"]["path"]).samefile(next(Path(deep_ep.find_nccl_root(), "lib").glob("libnccl.so*")))'
            if plan.engine == "sglang"
            else "import coldsnap_plugin; from importlib.metadata import entry_points; "
            'assert any(ep.name == "coldsnap" for ep in entry_points(group="vllm.general_plugins"))'
        ),
        "input": quote(plan.input_image),
        "output": quote(plan.output_image),
        "nccl_image": quote(plan.nccl_image),
        "fingerprint": quote(plan.fingerprint),
        "cuda_arch": quote(plan.cuda_arch),
        "build_jobs": quote(str(plan.build_jobs)),
        "nccl_policy": quote(plan.nccl_policy),
        "criu_image": quote(plan.criu_image),
        "snapshot_driver": quote(plan.snapshot_driver),
        "source_variables": _source_variables(plan.sources),
    }


def _result_detail(result: Any) -> str:
    detail = str(getattr(result, "stderr", "") or getattr(result, "stdout", "") or "").strip()
    return detail[-4000:] if detail else "no command output"


def _report_nccl_selection(result: Any) -> None:
    output = "%s\n%s" % (
        str(getattr(result, "stdout", "") or ""),
        str(getattr(result, "stderr", "") or ""),
    )
    match = _NCCL_LATEST_MESSAGE.search(output)
    if match:
        logger.log(
            PROGRESS,
            "ColdSnap builder: base NCCL %s; used latest qualified same-major provider %s",
            match.group(1),
            match.group(2),
        )


class ColdSnapBuilder(BuilderPlugin):
    """Convert a digest-pinned inference image into a ColdSnap capture base."""

    builder_name = "coldsnap"

    def prepare(
        self,
        image: str,
        recipe: Any,
        hosts: list[str],
        config: Any | None = None,
        dry_run: bool = False,
        transfer_mode: str = "local",
        ssh_kwargs: dict | None = None,
        builder_context: Mapping[str, Any] | None = None,
    ) -> str:
        snapshot_driver = None
        engine = "vllm"
        if builder_context is not None:
            unknown = set(builder_context) - {"snapshot_driver", "engine"}
            if unknown:
                raise ValueError("ColdSnap builder received unknown context: %s" % ", ".join(sorted(unknown)))
            snapshot_driver = builder_context.get("snapshot_driver")
            engine = str(builder_context.get("engine") or "vllm")
        return self.prepare_image(
            image,
            recipe,
            hosts,
            config=config,
            dry_run=dry_run,
            transfer_mode=transfer_mode,
            ssh_kwargs=ssh_kwargs,
            snapshot_driver=snapshot_driver,
            engine=engine,
        )

    def _run(
        self,
        host: str,
        script: str,
        *,
        ssh_kwargs: dict | None,
        timeout: int,
        dry_run: bool = False,
    ) -> Any:
        return run_script_on_host(
            host,
            script,
            ssh_kwargs=ssh_kwargs,
            timeout=timeout,
            dry_run=dry_run,
        )

    def _run_streaming(
        self,
        host: str,
        script: str,
        *,
        ssh_kwargs: dict | None,
        timeout: int,
        progress_label: str,
    ) -> Any:
        # Default verbosity gets structured PROGRESS messages and heartbeats.
        # -v and above additionally inherit the command's native build output;
        # command/script diagnostics remain DEBUG-only.
        stream_output = logger.isEnabledFor(logging.INFO)
        logger.debug(
            "ColdSnap builder command on %s: %d bytes, timeout=%ds, live_output=%s",
            host,
            len(script),
            timeout,
            stream_output,
        )
        with progress_heartbeat(logger, progress_label):
            return run_script_on_host_streaming(
                host,
                script,
                ssh_kwargs=ssh_kwargs,
                timeout=timeout,
                quiet=not stream_output,
                session_guard=True,
            )

    def _ensure_input_image(
        self,
        image: str,
        host: str,
        settings: BuildSettings,
        ssh_kwargs: dict | None,
        docker_platform: str,
    ) -> None:
        docker = quote(settings.docker_command)
        image_value = quote(image)
        script = """set -e
if [ "$(%(docker)s image inspect --format '{{.Os}}/{{.Architecture}}' %(image)s 2>/dev/null || true)" = %(platform)s ]; then
  echo "[coldsnap-builder] input image already present: %(image)s"
else
  echo "[coldsnap-builder] pulling input image: %(image)s"
  %(docker)s pull --platform %(platform)s %(image)s
fi
test "$(%(docker)s image inspect --format '{{.Os}}/{{.Architecture}}' %(image)s)" = %(platform)s
""" % {"docker": docker, "image": image_value, "platform": quote(docker_platform)}
        result = self._run_streaming(
            host,
            script,
            ssh_kwargs=ssh_kwargs,
            timeout=1800,
            progress_label="ColdSnap builder: preparing input image",
        )
        if not getattr(result, "success", False):
            raise RuntimeError("ColdSnap builder could not prepare input image %s: %s" % (image, _result_detail(result)))

    def _input_runtime_label(
        self,
        image: str,
        host: str,
        settings: BuildSettings,
        ssh_kwargs: dict | None,
    ) -> str:
        docker = quote(settings.docker_command)
        image_value = quote(image)
        script = """set -e
%(docker)s image inspect --format '{{ index .Config.Labels "%(label)s" }}' %(image)s
""" % {"docker": docker, "image": image_value, "label": _COLDSNAP_RUNTIME_LABEL}
        result = self._run(host, script, ssh_kwargs=ssh_kwargs, timeout=30)
        if not getattr(result, "success", False):
            raise RuntimeError("ColdSnap builder could not resolve input image %s: %s" % (image, _result_detail(result)))
        label = str(getattr(result, "stdout", "") or "").strip().splitlines()[-1:]
        return "" if not label or label[0] == "<no value>" else label[0]

    def _input_supports_driver(
        self,
        image: str,
        host: str,
        snapshot_driver: str,
        settings: BuildSettings,
        ssh_kwargs: dict | None,
    ) -> bool:
        docker = quote(settings.docker_command)
        script = "%(docker)s image inspect --format '{{ index .Config.Labels \"%(label)s\" }}' %(image)s" % {
            "docker": docker,
            "image": quote(image),
            "label": _COLDSNAP_DRIVERS_LABEL,
        }
        result = self._run(host, script, ssh_kwargs=ssh_kwargs, timeout=30)
        if not getattr(result, "success", False):
            return False
        values = str(getattr(result, "stdout", "") or "").strip().splitlines()
        supported = {value.strip() for value in (values[-1].split(",") if values else ())}
        return snapshot_driver in supported

    def _detect_snapshot_driver(
        self,
        host: str,
        settings: BuildSettings,
        ssh_kwargs: dict | None,
    ) -> str:
        result = self._run(
            host,
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1",
            ssh_kwargs=ssh_kwargs,
            timeout=30,
        )
        if not getattr(result, "success", False):
            raise RuntimeError("ColdSnap builder could not detect NVIDIA driver: %s" % _result_detail(result))
        output = str(getattr(result, "stdout", "") or "").strip().splitlines()
        match = re.match(r"^(\d+)(?:\.|$)", output[-1] if output else "")
        if match is None or int(match.group(1)) < 580:
            raise RuntimeError("ColdSnap builder requires NVIDIA driver 580 or newer")
        return "n610" if int(match.group(1)) >= 610 else "n580"

    def _detect_cuda_arch(
        self,
        image: str,
        host: str,
        settings: BuildSettings,
        ssh_kwargs: dict | None,
        docker_platform: str,
    ) -> str:
        if settings.cuda_arch:
            return settings.cuda_arch
        script = "%s run --rm --network none --platform %s --gpus all --entrypoint python3 %s -c %s" % (
            quote(settings.docker_command),
            quote(docker_platform),
            quote(image),
            quote("import torch; major, minor = torch.cuda.get_device_capability(); print(f'{major}{minor}')"),
        )
        result = self._run(host, script, ssh_kwargs=ssh_kwargs, timeout=120)
        if not getattr(result, "success", False):
            raise RuntimeError(
                "ColdSnap builder could not detect the target CUDA architecture; "
                "set defaults.builders.coldsnap.cuda_arch: %s" % _result_detail(result)
            )
        output = str(getattr(result, "stdout", "") or "").strip().splitlines()
        if not output:
            raise RuntimeError("ColdSnap builder CUDA architecture probe returned no output")
        return _normalize_cuda_arch(output[-1])

    def _image_exists(
        self,
        image: str,
        host: str,
        settings: BuildSettings,
        ssh_kwargs: dict | None,
        docker_platform: str,
    ) -> bool:
        script = 'test "$(%s image inspect --format %s %s 2>/dev/null)" = %s' % (
            quote(settings.docker_command),
            quote("{{.Os}}/{{.Architecture}}"),
            quote(image),
            quote(docker_platform),
        )
        result = self._run(host, script, ssh_kwargs=ssh_kwargs, timeout=30)
        return bool(getattr(result, "success", False))

    def _detect_docker_platform(self, host, settings, ssh_kwargs) -> str:
        result = self._run(
            host,
            "%s info --format '{{.OSType}}/{{.Architecture}}'" % quote(settings.docker_command),
            ssh_kwargs=ssh_kwargs,
            timeout=30,
        )
        value = str(getattr(result, "stdout", "") or "").strip()
        aliases = {"linux/aarch64": "linux/arm64", "linux/x86_64": "linux/amd64"}
        value = aliases.get(value, value)
        if not getattr(result, "success", False) or value not in {"linux/arm64", "linux/amd64"}:
            raise RuntimeError("ColdSnap builder could not determine Docker platform on %s: %s" % (host, _result_detail(result)))
        return value

    @contextmanager
    def _prepared_sources(self, plan, host, ssh_kwargs):
        """Send only source checkouts to the build host, never credentials."""
        remote = not should_run_locally(host, (ssh_kwargs or {}).get("ssh_user"))
        with TemporaryDirectory(prefix="sparkrun-coldsnap-sources-") as temporary:
            root = Path(temporary)
            logger.log(PROGRESS, "ColdSnap builder: fetching pinned sources on the control node")
            with progress_heartbeat(logger, "ColdSnap builder: fetching pinned sources"):
                for source in plan.sources:
                    path = root / (source.name.replace("_", "-") + "-" + source.revision)
                    clone_pinned_source(path, source.url, source.ref, source.revision)
            if not remote:
                yield str(root)
                return
            created = self._run(
                host,
                "mktemp -d /tmp/sparkrun-coldsnap-sources.XXXXXXXXXX",
                ssh_kwargs=ssh_kwargs,
                timeout=30,
            )
            destination = str(getattr(created, "stdout", "") or "").strip()
            if not getattr(created, "success", False) or not re.fullmatch(r"/tmp/sparkrun-coldsnap-sources\.[A-Za-z0-9]{10}", destination):
                raise RuntimeError("ColdSnap builder could not create private source staging directory: %s" % _result_detail(created))
            try:
                logger.log(PROGRESS, "ColdSnap builder: staging verified sources on %s", host)
                result = run_rsync(str(root), host, destination, **(ssh_kwargs or {}), timeout=600)
                if not result.success:
                    raise RuntimeError("ColdSnap builder could not stage verified sources on %s: %s" % (host, _result_detail(result)))
                yield destination
            finally:
                removed = self._run(host, "rm -rf -- %s" % quote(destination), ssh_kwargs=ssh_kwargs, timeout=30)
                if not getattr(removed, "success", False):
                    logger.warning("ColdSnap builder could not remove temporary source checkouts on %s: %s", host, destination)

    def prepare_image(
        self,
        image: str,
        recipe: Any,
        hosts: list[str],
        config: Any | None = None,
        dry_run: bool = False,
        transfer_mode: str = "local",
        ssh_kwargs: dict | None = None,
        snapshot_driver: str | None = None,
        engine: str = "vllm",
    ) -> str:
        settings = _resolve_settings(recipe, config, engine)
        # Validate immutable input identity before a dry run or any remote
        # probe. Capture artifacts must never be rooted in a mutable tag.
        if not _PINNED_IMAGE.fullmatch(image):
            raise ValueError("builder: coldsnap requires a digest-pinned container: name@sha256:<64 lowercase hex characters>")
        if not hosts:
            raise ValueError("ColdSnap builder requires at least one target host")
        build_host = hosts[0] if transfer_mode == "delegated" else "localhost"

        if snapshot_driver is not None and snapshot_driver not in {"n580", "n610"}:
            raise ValueError("ColdSnap builder snapshot driver must be n580 or n610")
        if engine not in {"vllm", "sglang"}:
            raise ValueError("ColdSnap builder engine must be vllm or sglang")
        if dry_run:
            plan = (
                _build_plan(image, settings, settings.cuda_arch or "000", snapshot_driver, engine, docker_platform="linux/target")
                if snapshot_driver is not None
                else _dry_run_plan(image, settings, engine)
            )
            logger.info(
                "[dry-run] Would derive ColdSnap image %s from %s on %s (CUDA arch %s)",
                plan.output_image,
                image,
                build_host,
                settings.cuda_arch or "auto",
            )
            return plan.output_image

        logger.log(PROGRESS, "ColdSnap builder: preparing pinned input image on %s", build_host)
        logger.info("ColdSnap builder input: %s", image)
        docker_platform = self._detect_docker_platform(hosts[0], settings, ssh_kwargs)
        if build_host != hosts[0] and self._detect_docker_platform(build_host, settings, ssh_kwargs) != docker_platform:
            raise RuntimeError(
                "ColdSnap runtime builds require the target Docker platform %s; "
                "use delegated transfer mode to build on the cluster head instead of the control node" % docker_platform
            )
        self._ensure_input_image(image, build_host, settings, ssh_kwargs, docker_platform)
        snapshot_driver = snapshot_driver or self._detect_snapshot_driver(build_host, settings, ssh_kwargs)
        logger.info("ColdSnap builder snapshot driver: %s", snapshot_driver)
        if self._input_runtime_label(image, build_host, settings, ssh_kwargs) == "%s-cuda-criu-v1" % engine and self._input_supports_driver(
            image, build_host, snapshot_driver, settings, ssh_kwargs
        ):
            logger.log(PROGRESS, "ColdSnap builder: input image is already enabled; conversion skipped")
            logger.info("ColdSnap-enabled input image: %s", image)
            return image

        logger.log(PROGRESS, "ColdSnap builder: detecting target CUDA architecture")
        cuda_arch = self._detect_cuda_arch(image, build_host, settings, ssh_kwargs, docker_platform)
        logger.info("ColdSnap builder CUDA architecture: sm_%s", cuda_arch)
        plan = _build_plan(image, settings, cuda_arch, snapshot_driver, engine, docker_platform=docker_platform)
        logger.debug(
            "ColdSnap builder plan: fingerprint=%s output=%s sources=%s",
            plan.fingerprint,
            plan.output_image,
            ", ".join("%s@%s" % (source.name, source.revision) for source in plan.sources),
        )
        if not settings.rebuild and self._image_exists(plan.output_image, build_host, settings, ssh_kwargs, docker_platform):
            logger.log(PROGRESS, "ColdSnap builder: reusing cached image %s", plan.output_image)
            return plan.output_image

        logger.log(
            PROGRESS,
            "ColdSnap builder: building runtime image on %s (first build may take several minutes)",
            build_host,
        )
        logger.info("ColdSnap builder output: %s", plan.output_image)
        with self._prepared_sources(plan, build_host, ssh_kwargs) as source_root:
            result = self._run_streaming(
                build_host,
                render_build_script(plan, source_root=source_root),
                ssh_kwargs=ssh_kwargs,
                timeout=3 * 60 * 60,
                progress_label="ColdSnap builder: building runtime image",
            )
        if not getattr(result, "success", False):
            raise RuntimeError("ColdSnap image conversion failed on %s: %s" % (build_host, _result_detail(result)))
        _report_nccl_selection(result)
        logger.log(PROGRESS, "ColdSnap builder: ready %s", plan.output_image)
        return plan.output_image


__all__ = [
    "ColdSnapBuildPlan",
    "ColdSnapBuilder",
    "GitSource",
    "render_build_script",
]
