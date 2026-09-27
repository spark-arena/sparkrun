# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/coldsnap/LICENSE_EXCEPTION.

"""Versioned acquisition of the ColdSnap controller tool set.

ColdSnap execution needs the engine-neutral ``coldsnap`` controller and its
matching vLLM and SGLang adapters. Runtime
images contain their own copies, but sparkrun must not depend on a developer
checkout being present on the controller's ``PATH``.

Release assets are fetched from one pinned GitHub release, verified against its
``checksums.txt``, and installed atomically under sparkrun's cache directory.
If that transport is unavailable, sparkrun extracts the same release from a
multi-architecture OCI binary bundle before falling back to building the exact
pinned tag over HTTPS in a pinned Go container. Every path verifies the
release version, source commit, platform, and binary hashes before use.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, TypedDict

from sparkrun.core.progress import PROGRESS, progress_heartbeat
from sparkrun.plugins.coldsnap._controller_version import __version__ as DEFAULT_CONTROLLER_VERSION
from sparkrun.plugins.coldsnap.git_sources import clone_pinned_source

logger = logging.getLogger(__name__)

DEFAULT_CONTROLLER_COMMIT = "b9533f980852a56d071c7b1596b13d54e7285a69"
DEFAULT_RELEASE_REPOSITORY = "sparksq/coldsnap"
DEFAULT_BINARY_OCI_REPOSITORY = "docker.io/scitrera/coldsnap-binaries"
_DOWNLOAD_TIMEOUT = 60
_OCI_TIMEOUT = 10 * 60
_VERSION = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_OCI_REPOSITORY = re.compile(r"^[a-z0-9]+(?:[._:-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OCI_BUNDLE_KIND = "coldsnap-controller-binary-bundle"
_OCI_BUNDLE_ROOT = "/opt/coldsnap/bin"
_BINARIES = (
    "coldsnap",
    "coldsnap-vllm-adapter",
    "coldsnap-sglang-adapter",
    "coldsnap-criu-rpc",
)


def _binaries(os_name: str) -> tuple[str, ...]:
    if os_name == "linux":
        return _BINARIES
    if os_name == "darwin":
        return _BINARIES[:-1]  # CRIU is executed only on Linux targets.
    raise ColdSnapToolError("ColdSnap controllers support Linux and macOS, not %s" % os_name)


_GO_CRIU_REPOSITORY = "https://github.com/sparksq/go-criu.git"
_GO_CRIU_COMMIT = "29a4f2f8e8374d38319a9851d9c1ef880dd0a0e8"
_SOURCE_BUILD_TIMEOUT = 15 * 60
_GO_DIRECTIVE = re.compile(r"^go[ \t]+(\d+\.\d+\.\d+)[ \t]*$", re.MULTILINE)
_COLDSNAP_BUILDER = re.compile(
    r"^FROM[ \t]+"
    r"((?:docker\.io/library/)?golang:(\d+\.\d+\.\d+)@sha256:[0-9a-f]{64})"
    r"[ \t]+AS[ \t]+coldsnap_builder[ \t]*$",
    re.MULTILINE,
)


class _InstallOptions(TypedDict, total=False):
    target_tools: bool


class ColdSnapToolError(RuntimeError):
    """A managed controller could not be resolved or verified."""


class ColdSnapReleaseAccessError(ColdSnapToolError):
    """The pinned GitHub release could not be read with API credentials."""


@dataclass(frozen=True)
class ControllerTool:
    path: Path
    adapter_path: Path
    version: str
    source: str
    sglang_adapter_path: Path | None = None
    criu_rpc_path: Path | None = None

    def payload_verifier(self, engine: str) -> Path:
        """Return the architecture-matched adapter exposing payload-verify."""
        if engine == "vllm":
            return self.adapter_path
        if engine == "sglang" and self.sglang_adapter_path is not None:
            return self.sglang_adapter_path
        raise ColdSnapToolError("ColdSnap has no payload verifier for engine %s" % engine)

    @property
    def environment(self) -> dict[str, str]:
        environment = {"COLDSNAP_VLLM_ADAPTER": str(self.adapter_path)}
        if self.sglang_adapter_path is not None:
            environment["COLDSNAP_SGLANG_ADAPTER"] = str(self.sglang_adapter_path)
        if self.criu_rpc_path is not None:
            environment["COLDSNAP_CRIU_RPC"] = str(self.criu_rpc_path)
        return environment


def _platform() -> tuple[str, str]:
    os_name = platform.system().lower()
    _binaries(os_name)
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return os_name, "amd64"
    if machine in {"aarch64", "arm64"}:
        return os_name, "arm64"
    raise ColdSnapToolError("ColdSnap controller releases do not support architecture %s" % machine)


def _settings(config: Any) -> tuple[str, str, str, str, str, bool]:
    plugin = config.plugin_settings("coldsnap")
    raw = plugin.get("controller", {}) if isinstance(plugin, Mapping) else {}
    values = raw if isinstance(raw, Mapping) else {}
    version = str(values.get("version") or DEFAULT_CONTROLLER_VERSION).removeprefix("v")
    commit = str(values.get("commit") or DEFAULT_CONTROLLER_COMMIT)
    repository = str(values.get("repository") or DEFAULT_RELEASE_REPOSITORY)
    oci_repository = str(values.get("oci_repository") or DEFAULT_BINARY_OCI_REPOSITORY)
    path = str(values.get("path") or "")
    download = values.get("download", True)
    if not _VERSION.fullmatch(version):
        raise ColdSnapToolError("plugins.coldsnap.controller.version must be a semantic release version")
    if not _COMMIT.fullmatch(commit):
        raise ColdSnapToolError("plugins.coldsnap.controller.commit must be a full lowercase Git commit")
    if not _REPOSITORY.fullmatch(repository):
        raise ColdSnapToolError("plugins.coldsnap.controller.repository must be an owner/repository GitHub name")
    if not _OCI_REPOSITORY.fullmatch(oci_repository):
        raise ColdSnapToolError("plugins.coldsnap.controller.oci_repository must be an untagged OCI repository")
    if not isinstance(download, bool):
        raise ColdSnapToolError("plugins.coldsnap.controller.download must be a boolean")
    return version, commit, repository, oci_repository, path, download


def _resolve_explicit(path: str, version: str) -> ControllerTool:
    resolved = shutil.which(path) if os.path.sep not in path else path
    controller = Path(resolved or path).expanduser()
    if not controller.is_file() or not os.access(controller, os.X_OK):
        raise ColdSnapToolError("Configured ColdSnap controller does not exist: %s" % controller)
    adapter = controller.with_name("coldsnap-vllm-adapter")
    if not adapter.is_file() or not os.access(adapter, os.X_OK):
        on_path = shutil.which("coldsnap-vllm-adapter")
        if on_path:
            adapter = Path(on_path)
    if not adapter.is_file() or not os.access(adapter, os.X_OK):
        raise ColdSnapToolError("ColdSnap vLLM adapter is not beside %s and is not on PATH" % controller)
    sglang_adapter = controller.with_name("coldsnap-sglang-adapter")
    if not sglang_adapter.is_file() or not os.access(sglang_adapter, os.X_OK):
        on_path = shutil.which("coldsnap-sglang-adapter")
        if on_path:
            sglang_adapter = Path(on_path)
    if not sglang_adapter.is_file() or not os.access(sglang_adapter, os.X_OK):
        raise ColdSnapToolError("ColdSnap SGLang adapter is not beside %s and is not on PATH" % controller)
    if _platform()[0] == "darwin":
        return ControllerTool(controller, adapter, version, "config", sglang_adapter)
    criu_rpc = controller.with_name("coldsnap-criu-rpc")
    if not criu_rpc.is_file() or not os.access(criu_rpc, os.X_OK):
        on_path = shutil.which("coldsnap-criu-rpc")
        if on_path:
            criu_rpc = Path(on_path)
    if not criu_rpc.is_file() or not os.access(criu_rpc, os.X_OK):
        raise ColdSnapToolError("ColdSnap CRIU RPC helper is not beside %s and is not on PATH" % controller)
    return ControllerTool(controller, adapter, version, "config", sglang_adapter, criu_rpc)


def explicit_controller_environment(path: str) -> dict[str, str]:
    """Bind an explicit controller to its sibling development adapters."""
    resolved = shutil.which(path) if os.path.sep not in path else path
    if not resolved:
        return {}
    controller = Path(resolved).expanduser()
    environment: dict[str, str] = {}
    for engine in ("vllm", "sglang"):
        adapter = controller.with_name("coldsnap-%s-adapter" % engine)
        if adapter.is_file() and os.access(adapter, os.X_OK):
            environment["COLDSNAP_%s_ADAPTER" % engine.upper()] = str(adapter)
    criu_rpc = controller.with_name("coldsnap-criu-rpc")
    if _platform()[0] == "linux" and criu_rpc.is_file() and os.access(criu_rpc, os.X_OK):
        environment["COLDSNAP_CRIU_RPC"] = str(criu_rpc)
    return environment


def _cache_path(cache_dir: str | Path, version: str, os_name: str, arch: str) -> Path:
    return Path(cache_dir) / "tools" / "coldsnap" / version / ("%s-%s" % (os_name, arch))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _github_token() -> str:
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    gh = shutil.which("gh")
    if not gh:
        return ""
    try:
        result = subprocess.run(
            [gh, "auth", "token", "--hostname", "github.com"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _request_bytes(url: str, *, token: str, accept: str) -> bytes:
    headers = {"Accept": accept, "User-Agent": "sparkrun-coldsnap"}
    if token:
        headers["Authorization"] = "Bearer %s" % token
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT) as response:
        return response.read()


def _fetch_release(repository: str, version: str, *, token: str) -> dict[str, str]:
    url = "https://api.github.com/repos/%s/releases/tags/v%s" % (repository, version)
    try:
        payload = _request_bytes(url, token=token, accept="application/vnd.github+json")
        document = json.loads(payload)
    except urllib.error.HTTPError as error:
        authentication = " Authenticate with `gh auth login` or GH_TOKEN." if not token else ""
        error_type = ColdSnapReleaseAccessError if error.code in {401, 403, 404} else ColdSnapToolError
        raise error_type(
            "Could not resolve ColdSnap release v%s from %s: HTTP %s.%s" % (version, repository, error.code, authentication)
        ) from error
    except Exception as error:
        authentication = " Authenticate with `gh auth login` or GH_TOKEN." if not token else ""
        raise ColdSnapToolError(
            "Could not resolve ColdSnap release v%s from %s: %s.%s" % (version, repository, error, authentication)
        ) from error
    assets = document.get("assets") if isinstance(document, dict) else None
    if not isinstance(assets, list):
        raise ColdSnapToolError("ColdSnap release v%s has no asset inventory" % version)
    result: dict[str, str] = {}
    for asset in assets:
        if isinstance(asset, dict) and isinstance(asset.get("name"), str) and isinstance(asset.get("url"), str):
            result[asset["name"]] = asset["url"]
    return result


def _checksum_for(checksums: bytes, name: str) -> str:
    try:
        text = checksums.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ColdSnapToolError("ColdSnap release checksums are not UTF-8") from error
    matches: list[str] = []
    for line in text.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            continue
        filename = PurePosixPath(fields[1].lstrip("*"))
        safe_parts = tuple(part for part in filename.parts if part != ".")
        if safe_parts == (name,) and _SHA256.fullmatch(fields[0]):
            matches.append(fields[0])
    if len(matches) != 1:
        raise ColdSnapToolError("ColdSnap release checksums do not contain exactly one entry for %s" % name)
    return matches[0]


def _extract_binary(archive: bytes, name: str) -> bytes:
    try:
        with tarfile.open(fileobj=BytesIO(archive), mode="r:gz") as bundle:
            candidates = []
            for member in bundle.getmembers():
                path = PurePosixPath(member.name)
                safe_parts = tuple(part for part in path.parts if part != ".")
                if member.isfile() and safe_parts == (name,):
                    candidates.append(member)
            if len(candidates) != 1:
                raise ColdSnapToolError("ColdSnap archive must contain exactly one %s binary" % name)
            stream = bundle.extractfile(candidates[0])
            if stream is None:
                raise ColdSnapToolError("ColdSnap archive member %s is unreadable" % name)
            return stream.read()
    except (tarfile.TarError, OSError) as error:
        raise ColdSnapToolError("ColdSnap release archive for %s is invalid: %s" % (name, error)) from error


def _atomic_write(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".%s-" % path.name, dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_cached(
    root: Path,
    version: str,
    commit: str,
    *,
    target_arch: str = "",
    controller_platform: tuple[str, str] | None = None,
) -> ControllerTool | None:
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != version
        or manifest.get("commit") != commit
        or manifest.get("format") != 1
    ):
        return None
    os_name, arch = ("linux", target_arch) if target_arch else (controller_platform or _platform())
    if manifest.get("platform") != "%s-%s" % (os_name, arch):
        return None
    binaries = _binaries(os_name)
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(binaries):
        return None
    for name in binaries:
        path = root / name
        expected = hashes.get(name)
        if path.is_symlink() or not path.is_file() or not isinstance(expected, str) or _file_sha256(path) != expected:
            return None
    controller = root / "coldsnap"
    adapter = root / "coldsnap-vllm-adapter"
    sglang_adapter = root / "coldsnap-sglang-adapter"
    criu_rpc = root / "coldsnap-criu-rpc" if os_name == "linux" else None
    if target_arch:
        try:
            for name in _BINARIES:
                verify_target_elf(root / name, target_arch)
        except ColdSnapToolError:
            return None
        # This bundle is data on the controller. Its release/hash/ELF identity
        # is checked here; the adapter verifies its running identity on target.
        return ControllerTool(controller, adapter, version, "cache", sglang_adapter, criu_rpc)
    if os_name == "darwin":
        try:
            for name in binaries:
                verify_macos_executable(root / name, arch)
        except ColdSnapToolError:
            return None
    selected = ControllerTool(controller, adapter, version, "cache", sglang_adapter, criu_rpc)
    try:
        result = subprocess.run(
            [str(controller), "version", "--json"],
            env={**os.environ, **selected.environment},
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        identity = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    if result.returncode != 0 or not isinstance(identity, dict) or identity.get("version") != version or identity.get("commit") != commit:
        return None
    return selected


def verify_macos_executable(path: Path, arch: str) -> None:
    """Admit only a native 64-bit Mach-O executable for the requested CPU."""
    expected = {"amd64": 0x01000007, "arm64": 0x0100000C}.get(arch)
    with path.open("rb") as stream:
        header = stream.read(32)
    if (
        expected is None
        or len(header) != 32
        or header[:4] != b"\xcf\xfa\xed\xfe"
        or int.from_bytes(header[4:8], "little") != expected
        or int.from_bytes(header[12:16], "little") != 2
        or not os.access(path, os.X_OK)
    ):
        raise ColdSnapToolError("Controller %s is not an executable macOS/%s Mach-O binary" % (path, arch))


def verify_target_elf(path: Path, arch: str) -> None:
    """Reject wrong-architecture target executables without running them."""
    expected = {"amd64": 62, "arm64": 183}.get(arch)
    if expected is None:
        raise ColdSnapToolError("Unsupported target architecture: %s" % arch)
    try:
        with path.open("rb") as stream:
            header = stream.read(64)
    except OSError as error:
        raise ColdSnapToolError("Cannot read target helper %s: %s" % (path, error)) from error
    if (
        len(header) != 64
        or header[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(header[16:18], "little") not in {2, 3}
        or int.from_bytes(header[18:20], "little") != expected
        or not os.access(path, os.X_OK)
    ):
        raise ColdSnapToolError("Target helper %s is not an executable Linux/%s ELF binary" % (path, arch))


def _command_detail(result: subprocess.CompletedProcess[str]) -> str:
    detail = (result.stderr or result.stdout or "").strip()
    return detail[-2000:] if detail else "exit status %s" % result.returncode


def _clone_pinned_source(destination: Path, repository: str, version: str, commit: str) -> str:
    remote = "https://github.com/%s.git" % repository
    logger.log(PROGRESS, "ColdSnap: cloning pinned controller source over HTTPS")
    try:
        with progress_heartbeat(logger, "ColdSnap: cloning pinned controller source"):
            clone_pinned_source(destination, remote, "refs/tags/v%s" % version, commit)
    except RuntimeError as error:
        raise ColdSnapToolError(str(error)) from error
    return remote


def _clone_pinned_go_criu(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.log(PROGRESS, "ColdSnap: fetching pinned public go-criu source")
    try:
        with progress_heartbeat(logger, "ColdSnap: fetching pinned go-criu source"):
            clone_pinned_source(destination, _GO_CRIU_REPOSITORY, _GO_CRIU_COMMIT, _GO_CRIU_COMMIT)
    except RuntimeError as error:
        raise ColdSnapToolError(str(error)) from error


def _source_build_toolchain(source: Path) -> tuple[str, str]:
    """Resolve the release's own Go version and immutable builder image."""
    try:
        go_mod = (source / "go.mod").read_text(encoding="utf-8")
        dockerfile = (source / "deploy" / "vllm" / "Dockerfile").read_text(encoding="utf-8")
    except OSError as error:
        raise ColdSnapToolError("Pinned ColdSnap source does not expose its Go build contract: %s" % error) from error
    go_match = _GO_DIRECTIVE.search(go_mod)
    builder_match = _COLDSNAP_BUILDER.search(dockerfile)
    if go_match is None:
        raise ColdSnapToolError("Pinned ColdSnap go.mod requires a full Go patch version")
    if builder_match is None:
        raise ColdSnapToolError("Pinned ColdSnap source has no digest-pinned coldsnap_builder image")
    go_version = go_match.group(1)
    builder_image = builder_match.group(1)
    builder_version = builder_match.group(2)
    if builder_version != go_version:
        raise ColdSnapToolError(
            "Pinned ColdSnap Go build contract is inconsistent: go.mod requires %s, "
            "builder image provides %s" % (go_version, builder_version)
        )
    return go_version, builder_image


def _build_controller_binaries_with_docker(
    source: Path,
    output: Path,
    version: str,
    commit: str,
    os_name: str,
    arch: str,
) -> tuple[str, str]:
    go_version, builder_image = _source_build_toolchain(source)
    docker = shutil.which("docker")
    if not docker:
        raise ColdSnapToolError("Temporary ColdSnap source-build fallback requires docker")
    output.mkdir(parents=True, exist_ok=True)
    workspace = source.parent
    build_script = (
        "set -eu\n"
        "mkdir -p /work/out /work/go-cache /work/go-mod\n"
        'ldflags="-s -w -buildid= -X github.com/sparksq/coldsnap/internal/buildinfo.Version=$COLDSNAP_BUILD_VERSION '
        '-X github.com/sparksq/coldsnap/internal/buildinfo.Commit=$COLDSNAP_BUILD_COMMIT"\n'
        'go build -trimpath -ldflags "$ldflags" -o /work/out/coldsnap ./cmd/coldsnap\n'
        'go build -trimpath -ldflags "$ldflags" -o /work/out/coldsnap-vllm-adapter ./cmd/coldsnap-vllm-adapter\n'
        'go build -trimpath -ldflags "$ldflags" -o /work/out/coldsnap-sglang-adapter ./cmd/coldsnap-sglang-adapter\n'
    )
    if os_name == "linux":
        build_script += '(cd cmd/coldsnap-criu-rpc && go build -trimpath -ldflags "$ldflags" -o /work/out/coldsnap-criu-rpc .)\n'
    command = [
        docker,
        "run",
        "--rm",
        "--platform",
        "linux/%s" % _platform()[1],
        "--pull",
        "missing",
        "--user",
        "%s:%s" % (os.getuid(), os.getgid()),
        "--mount",
        "type=bind,src=%s,dst=/work" % workspace,
        "--workdir",
        "/work/source",
        "--env",
        "CGO_ENABLED=0",
        "--env",
        "GOOS=%s" % os_name,
        "--env",
        "GOARCH=%s" % arch,
        "--env",
        "GOCACHE=/work/go-cache",
        "--env",
        "GOMODCACHE=/work/go-mod",
        "--env",
        "COLDSNAP_BUILD_VERSION=%s" % version,
        "--env",
        "COLDSNAP_BUILD_COMMIT=%s" % commit,
        builder_image,
        "sh",
        "-c",
        build_script,
    ]
    logger.log(
        PROGRESS,
        "ColdSnap: building controller with source-pinned Go %s container",
        go_version,
    )
    try:
        with progress_heartbeat(logger, "ColdSnap: building controller from pinned source"):
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=_SOURCE_BUILD_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ColdSnapToolError("Dockerized ColdSnap controller build failed: %s" % error) from error
    if result.returncode != 0:
        raise ColdSnapToolError("Dockerized ColdSnap controller build failed: %s" % _command_detail(result))
    missing = [name for name in _binaries(os_name) if not (output / name).is_file()]
    if missing:
        raise ColdSnapToolError("Dockerized ColdSnap controller build did not produce: %s" % ", ".join(missing))
    return go_version, builder_image


def install_controller_tool_from_ssh(
    cache_dir: str | Path,
    version: str,
    repository: str,
    os_name: str,
    arch: str,
    *,
    commit: str = DEFAULT_CONTROLLER_COMMIT,
    target_tools: bool = False,
) -> ControllerTool:
    """Build a pinned release using controller-side Git and Docker.

    The historical function name is retained for compatibility. Retrieval now
    prefers HTTPS, with controller-local credentials and SSH as fallbacks.
    """
    with tempfile.TemporaryDirectory(prefix="sparkrun-coldsnap-source-") as temporary:
        workspace = Path(temporary)
        source = workspace / "source"
        output = workspace / "out"
        remote = _clone_pinned_source(source, repository, version, commit)
        if os_name == "linux":
            _clone_pinned_go_criu(source / "build" / "sources" / "go-criu")
        go_version, builder_image = _build_controller_binaries_with_docker(source, output, version, commit, os_name, arch)
        root = _cache_path(cache_dir, version, os_name, arch)
        binary_digests: dict[str, str] = {}
        for name in _binaries(os_name):
            payload = (output / name).read_bytes()
            binary_digests[name] = hashlib.sha256(payload).hexdigest()
            _atomic_write(
                root / name,
                payload,
                stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH,
            )
        manifest = {
            "format": 1,
            "version": version,
            "commit": commit,
            "repository": repository,
            "platform": "%s-%s" % (os_name, arch),
            "sha256": binary_digests,
            "archives": {},
            "source_build": {
                "remote": remote,
                "tag": "v%s" % version,
                "go_version": go_version,
                "builder_image": builder_image,
            },
        }
        _atomic_write(
            root / "manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH,
        )
    resolved = _verify_cached(
        root, version, commit, target_arch=arch if target_tools else "", controller_platform=None if target_tools else (os_name, arch)
    )
    if resolved is None:
        raise ColdSnapToolError("Source-built ColdSnap controller v%s failed identity verification" % version)
    return ControllerTool(
        resolved.path,
        resolved.adapter_path,
        resolved.version,
        "git-build",
        resolved.sglang_adapter_path,
        resolved.criu_rpc_path,
    )


def install_controller_tool(
    cache_dir: str | Path,
    version: str,
    repository: str,
    os_name: str,
    arch: str,
    *,
    commit: str = DEFAULT_CONTROLLER_COMMIT,
    token: str = "",
    release_assets: Callable[[str, str], Mapping[str, str]] | None = None,
    fetch_bytes: Callable[[str], bytes] | None = None,
    target_tools: bool = False,
) -> ControllerTool:
    if not token and (release_assets is None or fetch_bytes is None):
        token = _github_token()
    assets = dict(release_assets(repository, version)) if release_assets is not None else _fetch_release(repository, version, token=token)
    fetch = fetch_bytes or (lambda url: _request_bytes(url, token=token, accept="application/octet-stream"))
    checksum_url = assets.get("checksums.txt")
    if not checksum_url:
        raise ColdSnapToolError("ColdSnap release v%s is missing checksums.txt" % version)
    checksums = fetch(checksum_url)
    payloads: dict[str, bytes] = {}
    archive_digests: dict[str, str] = {}
    for name in _binaries(os_name):
        archive_name = "%s_%s_%s_%s.tar.gz" % (name, version, os_name, arch)
        url = assets.get(archive_name)
        if not url:
            raise ColdSnapToolError("ColdSnap release v%s is missing %s" % (version, archive_name))
        archive = fetch(url)
        expected = _checksum_for(checksums, archive_name)
        actual = hashlib.sha256(archive).hexdigest()
        if actual != expected:
            raise ColdSnapToolError("ColdSnap release checksum mismatch for %s: expected %s, got %s" % (archive_name, expected, actual))
        payloads[name] = _extract_binary(archive, name)
        archive_digests[archive_name] = actual

    root = _cache_path(cache_dir, version, os_name, arch)
    binary_digests: dict[str, str] = {}
    for name, payload in payloads.items():
        binary_digests[name] = hashlib.sha256(payload).hexdigest()
        _atomic_write(
            root / name, payload, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
        )
    manifest = {
        "format": 1,
        "version": version,
        "commit": commit,
        "repository": repository,
        "platform": "%s-%s" % (os_name, arch),
        "sha256": binary_digests,
        "archives": archive_digests,
    }
    _atomic_write(
        root / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH,
    )
    resolved = _verify_cached(
        root, version, commit, target_arch=arch if target_tools else "", controller_platform=None if target_tools else (os_name, arch)
    )
    if resolved is None:
        raise ColdSnapToolError("Installed ColdSnap controller v%s failed identity verification" % version)
    return ControllerTool(
        resolved.path,
        resolved.adapter_path,
        resolved.version,
        "download",
        resolved.sglang_adapter_path,
        resolved.criu_rpc_path,
    )


def _docker_command(arguments: list[str], *, timeout: int = _OCI_TIMEOUT) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(arguments, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ColdSnapToolError("ColdSnap OCI binary-bundle command failed: %s" % error) from error
    if result.returncode != 0:
        raise ColdSnapToolError("ColdSnap OCI binary-bundle command failed: %s" % _command_detail(result))
    return result


def _canonical_oci_repository(repository: str) -> str:
    """Normalize Docker Hub's optional registry prefix for digest comparison."""
    return repository.removeprefix("docker.io/").removeprefix("index.docker.io/")


def _read_oci_bundle(root: Path, version: str, commit: str, os_name: str, arch: str) -> dict[str, bytes]:
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ColdSnapToolError("ColdSnap OCI binary-bundle manifest must be a regular file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ColdSnapToolError("ColdSnap OCI binary-bundle manifest is invalid: %s" % error) from error
    expected_platform = "%s-%s" % (os_name, arch)
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != 1
        or manifest.get("kind") != _OCI_BUNDLE_KIND
        or manifest.get("version") != version
        or manifest.get("commit") != commit
        or manifest.get("platform") != expected_platform
    ):
        raise ColdSnapToolError("ColdSnap OCI binary bundle does not match release v%s at %s for %s" % (version, commit, expected_platform))
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(_binaries(os_name)):
        raise ColdSnapToolError("ColdSnap OCI binary-bundle hash inventory is invalid")
    payloads: dict[str, bytes] = {}
    for name in _binaries(os_name):
        path = root / name
        expected = hashes.get(name)
        if path.is_symlink() or not path.is_file() or not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            raise ColdSnapToolError("ColdSnap OCI binary bundle has an invalid %s entry" % name)
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ColdSnapToolError("ColdSnap OCI binary-bundle checksum mismatch for %s: expected %s, got %s" % (name, expected, actual))
        payloads[name] = payload
    return payloads


def _extract_oci_with_docker(oci_repository: str, version: str, arch: str, bundle: Path) -> str:
    docker = shutil.which("docker")
    if not docker:
        raise ColdSnapToolError("ColdSnap OCI binary-bundle fallback requires docker")
    image = "%s:%s" % (oci_repository, version)
    platform_name = "linux/%s" % arch
    logger.log(PROGRESS, "ColdSnap: pulling controller binary bundle %s for %s", image, platform_name)
    with progress_heartbeat(logger, "ColdSnap: pulling controller binary bundle"):
        _docker_command([docker, "pull", "--platform", platform_name, image])
    inspect = _docker_command([docker, "image", "inspect", "--format", "{{json .RepoDigests}}", image])
    try:
        repo_digests = json.loads(inspect.stdout)
    except json.JSONDecodeError as error:
        raise ColdSnapToolError("ColdSnap OCI binary bundle has no readable repository digest") from error
    expected_repository = _canonical_oci_repository(oci_repository)
    resolved = ""
    if isinstance(repo_digests, list):
        for value in repo_digests:
            if not isinstance(value, str):
                continue
            candidate_repository, separator, digest = value.rpartition("@sha256:")
            if separator and _canonical_oci_repository(candidate_repository) == expected_repository and _SHA256.fullmatch(digest):
                resolved = value
                break
    if not resolved:
        raise ColdSnapToolError("ColdSnap OCI binary bundle has no immutable repository digest")

    created = _docker_command([docker, "create", "--platform", platform_name, resolved])
    container = created.stdout.strip()
    if not container or any(character.isspace() for character in container):
        raise ColdSnapToolError("ColdSnap OCI binary bundle returned an invalid container ID")
    try:
        _docker_command([docker, "cp", "%s:%s/." % (container, _OCI_BUNDLE_ROOT), str(bundle)])
    finally:
        try:
            _docker_command([docker, "rm", "-f", container], timeout=30)
        except ColdSnapToolError:
            logger.warning("ColdSnap: could not remove temporary binary-bundle container %s", container)
    return resolved


def install_controller_tool_from_oci(
    cache_dir: str | Path,
    version: str,
    repository: str,
    oci_repository: str,
    os_name: str,
    arch: str,
    *,
    commit: str = DEFAULT_CONTROLLER_COMMIT,
    target_tools: bool = False,
) -> ControllerTool:
    """Install one architecture-matched, manifest-verified OCI binary bundle."""
    _binaries(os_name)
    image = "%s:%s" % (oci_repository, version)
    with tempfile.TemporaryDirectory(prefix="sparkrun-coldsnap-oci-") as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        if os_name == "darwin":
            from sparkrun.plugins.coldsnap.oci_bundle import OCIBundleError, fetch_binary_bundle

            logger.log(PROGRESS, "ColdSnap: downloading OCI controller bundle %s for %s/%s", image, os_name, arch)
            try:
                with progress_heartbeat(logger, "ColdSnap: downloading OCI controller bundle"):
                    resolved = fetch_binary_bundle(oci_repository, version, os_name, arch, bundle)
            except (OCIBundleError, OSError, ValueError, tarfile.TarError) as error:
                raise ColdSnapToolError("ColdSnap OCI bundle download failed: %s" % error) from error
        else:
            resolved = _extract_oci_with_docker(oci_repository, version, arch, bundle)
        payloads = _read_oci_bundle(bundle, version, commit, os_name, arch)
        root = _cache_path(cache_dir, version, os_name, arch)
        binary_digests: dict[str, str] = {}
        for name, payload in payloads.items():
            binary_digests[name] = hashlib.sha256(payload).hexdigest()
            _atomic_write(
                root / name,
                payload,
                stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH,
            )
        cache_manifest = {
            "format": 1,
            "version": version,
            "commit": commit,
            "repository": repository,
            "platform": "%s-%s" % (os_name, arch),
            "sha256": binary_digests,
            "archives": {},
            "oci": {"reference": image, "resolved": resolved},
        }
        _atomic_write(
            root / "manifest.json",
            (json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n").encode(),
            stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH,
        )
    installed = _verify_cached(
        root, version, commit, target_arch=arch if target_tools else "", controller_platform=None if target_tools else (os_name, arch)
    )
    if installed is None:
        raise ColdSnapToolError("OCI-installed ColdSnap controller v%s failed identity verification" % version)
    return ControllerTool(
        installed.path,
        installed.adapter_path,
        installed.version,
        "oci",
        installed.sglang_adapter_path,
        installed.criu_rpc_path,
    )


def ensure_controller_tool(config: Any) -> ControllerTool:
    return _ensure_tool(config)


def ensure_target_tool(config: Any, arch: str) -> ControllerTool:
    if arch not in {"amd64", "arm64"}:
        raise ColdSnapToolError("Unsupported target architecture: %s" % arch)
    return _ensure_tool(config, target_arch=arch)


def _ensure_tool(config: Any, *, target_arch: str = "") -> ControllerTool:
    version, commit, repository, oci_repository, configured_path, allow_download = _settings(config)
    if configured_path:
        if target_arch:
            target_path = config.plugin_settings("coldsnap").get("controller", {}).get("target_path")
            if target_path:
                verified = _verify_cached(Path(target_path).expanduser(), version, commit, target_arch=target_arch)
                if verified is None:
                    raise ColdSnapToolError("Configured controller.target_path is not a verified target binary bundle")
                return verified
            if ("linux", target_arch) != _platform():
                raise ColdSnapToolError(
                    "Cross-platform development controllers require controller.target_path with a release-matched Linux target bundle"
                )
        return _resolve_explicit(configured_path, version)
    os_name, arch = ("linux", target_arch) if target_arch else _platform()
    root = _cache_path(config.cache_dir, version, os_name, arch)
    cached = _verify_cached(
        root, version, commit, target_arch=arch if target_arch else "", controller_platform=None if target_arch else (os_name, arch)
    )
    if cached is not None:
        return cached
    if not allow_download:
        raise ColdSnapToolError("ColdSnap controller v%s is not cached and plugins.coldsnap.controller.download is false" % version)
    logger.log(PROGRESS, "ColdSnap: downloading %s v%s for %s/%s", "target tools" if target_arch else "controller", version, os_name, arch)
    install_options: _InstallOptions = {"target_tools": True} if target_arch else {}
    try:
        return install_controller_tool(config.cache_dir, version, repository, os_name, arch, commit=commit, **install_options)
    except ColdSnapToolError as release_error:
        logger.log(
            PROGRESS,
            "ColdSnap: GitHub release acquisition failed; trying OCI binary bundle",
        )
        try:
            return install_controller_tool_from_oci(
                config.cache_dir,
                version,
                repository,
                oci_repository,
                os_name,
                arch,
                commit=commit,
                **install_options,
            )
        except ColdSnapToolError as oci_error:
            logger.log(
                PROGRESS,
                "ColdSnap: OCI binary-bundle acquisition failed; using temporary pinned Git source-build fallback",
            )
            try:
                return install_controller_tool_from_ssh(
                    config.cache_dir, version, repository, os_name, arch, commit=commit, **install_options
                )
            except ColdSnapToolError as build_error:
                raise ColdSnapToolError(
                    "GitHub release acquisition failed: %s OCI binary-bundle acquisition failed: %s "
                    "Temporary Git source-build fallback also failed: %s" % (release_error, oci_error, build_error)
                ) from build_error


__all__ = [
    "DEFAULT_CONTROLLER_COMMIT",
    "DEFAULT_CONTROLLER_VERSION",
    "DEFAULT_BINARY_OCI_REPOSITORY",
    "DEFAULT_RELEASE_REPOSITORY",
    "ColdSnapReleaseAccessError",
    "ColdSnapToolError",
    "ControllerTool",
    "ensure_controller_tool",
    "explicit_controller_environment",
    "install_controller_tool",
    "install_controller_tool_from_oci",
    "install_controller_tool_from_ssh",
]
