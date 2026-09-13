# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""On-demand acquisition of the SparkRoute release binary.

The LiteLLM gateway is acquired at start time by shelling ``uvx --from
'litellm[proxy]==<pinned>' litellm`` (see :mod:`sparkrun.proxy.engine`), so
"fetch a pinned third-party runtime when the gateway starts" is already how
this layer behaves.  SparkRoute is a Go binary rather than a Python package,
so this module is its ``uvx``: resolve the current platform, fetch the matching
release asset from GitHub, and verify it before it is ever executed.

Two rules make this safe enough to run a downloaded binary with the operator's
privileges, and neither should be relaxed:

* **Pinned version, pinned digest.** A sparkrun release names exactly one
  SparkRoute version and carries the SHA-256 of every asset it will accept.
  A platform with no pinned digest is a hard failure, not a download — "we
  don't know what this should hash to" and "run it anyway" must never be the
  same code path.
* **Verify before execute.** The download lands in a private temp file and is
  hashed *before* anything is unpacked from it; the binary only becomes
  executable once its archive matched.  A partial or tampered download is
  never left somewhere that looks like a usable binary.

**Assets are archives, not bare binaries.** SparkRoute publishes
``sparkroute_<version>_<os>_<arch>.tar.gz`` (``.zip`` on Windows) holding the
executable alongside its licence and notice files, and its ``checksums.txt``
digests those archives.  So the pinned digest is the *archive's* — it is the
value the release actually publishes, and pinning the unpacked binary instead
would mean pinning a number nobody else can independently confirm.

That is also why the verified archive is kept in the cache next to the binary
it produced: the pinned digest is then re-checkable on every call against a
file we still hold, which is the property the bare-binary layout used to have.
Extraction never trusts a path from the archive — the single member is located
by name and streamed to a destination this module chooses — so a crafted
archive cannot write outside the cache directory.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import stat
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from sparkrun.plugins.sparkroute._gateway_version import __version__ as SPARKROUTE_VERSION

logger = logging.getLogger(__name__)

#: GitHub repository publishing the verified SparkRoute release archives.
SPARKROUTE_REPO = "sparksq/sparkroute"

#: The single SparkRoute version this sparkrun release runs.


#: Executable name inside the release archive, and in the cache.
SPARKROUTE_BINARY = "sparkroute"

#: ``(version, os, arch) -> sha256`` of the release **archive** for every asset
#: we will unpack and execute.
#:
#: Populated from the release's published ``checksums.txt`` when a version is
#: pinned.  An absent entry is deliberately fatal — see the module docstring.
#: Source: https://github.com/sparksq/sparkroute/releases/tag/v0.0.2
RELEASE_CHECKSUMS: dict[tuple[str, str, str], str] = {
    ("0.0.2", "darwin", "amd64"): "a286cf46e18be49d1bc46ccca177cec1093909b0410e725afe1e81d8e4d4931e",
    ("0.0.2", "darwin", "arm64"): "2e3e8c61b94b2c780f235b7e123ef5e805c86f280290a4a53994527949007ab8",
    ("0.0.2", "linux", "amd64"): "81ba51e6d32f825818fa65f1689be4967327804a6e294fbbcd5f3aae35e11cbe",
    ("0.0.2", "linux", "arm64"): "1b59689f246fe98cf1f598f0c7b0da7a20a679582cbd0ef1bf48b9f60a238807",
    ("0.0.2", "windows", "amd64"): "81023da51b0abe53aea2135138495eb5c78e32d4edbf3952f7f9b7ed67086e66",
    ("0.0.2", "windows", "arm64"): "eb0fe99465785e3bd7f26f0d08f0e3f26d18e950c10ff3df96dc70a5b1e76663",
}

#: Point at a locally-built binary instead of a release asset.  Development
#: aid for working on SparkRoute itself; skips version and digest checks, so
#: it warns every time.
BINARY_OVERRIDE_ENV = "SPARKRUN_SPARKROUTE_BINARY"

#: Names this variable had before, newest first, still honoured so an existing
#: development setup continues to use its explicitly selected binary.
LEGACY_BINARY_OVERRIDE_ENVS = (
    "SPARKRUN_FOXSCI_ROUTE_BINARY",
    "SPARKRUN_LLM_GATEWAY_BINARY",
)

#: Refuse absurd downloads rather than filling the cache dir.  Also bounds
#: extraction, so a decompression bomb cannot outgrow the same limit.
MAX_DOWNLOAD_BYTES = 256 << 20

_DOWNLOAD_TIMEOUT_SECONDS = 120

_OS_ALIASES = {"linux": "linux", "darwin": "darwin", "windows": "windows"}
_ARCH_ALIASES = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "aarch64": "arm64",
    "arm64": "arm64",
}


class GatewayReleaseError(RuntimeError):
    """Acquiring or verifying the gateway binary failed."""


def platform_target() -> tuple[str, str]:
    """Return the ``(os, arch)`` release-asset target for this machine.

    Raises:
        GatewayReleaseError: The platform has no SparkRoute build.
    """
    raw_os = platform.system().lower()
    raw_arch = platform.machine().lower()
    target_os = _OS_ALIASES.get(raw_os)
    target_arch = _ARCH_ALIASES.get(raw_arch)
    if target_os is None or target_arch is None:
        raise GatewayReleaseError("No SparkRoute build for this platform (%s/%s)" % (raw_os, raw_arch))
    return target_os, target_arch


def archive_suffix(target_os: str) -> str:
    """Archive format published for *target_os*."""
    return ".zip" if target_os == "windows" else ".tar.gz"


def binary_name(target_os: str) -> str:
    """Executable name inside the archive for *target_os*."""
    return SPARKROUTE_BINARY + (".exe" if target_os == "windows" else "")


def asset_name(version: str, target_os: str, target_arch: str) -> str:
    """Release asset filename for *version* on the given target."""
    return "%s_%s_%s_%s%s" % (SPARKROUTE_BINARY, version, target_os, target_arch, archive_suffix(target_os))


def release_url(version: str, asset: str) -> str:
    """Download URL for *asset* of release *version*."""
    return "https://github.com/%s/releases/download/v%s/%s" % (SPARKROUTE_REPO, version, asset)


def expected_digest(version: str, target_os: str, target_arch: str) -> str:
    """Return the pinned SHA-256 of the release archive for a target.

    Raises:
        GatewayReleaseError: No digest is pinned, so the asset must not run.
    """
    digest = RELEASE_CHECKSUMS.get((version, target_os, target_arch))
    if not digest:
        raise GatewayReleaseError(
            "No pinned checksum for SparkRoute %s on %s/%s; refusing to run an unverified binary" % (version, target_os, target_arch)
        )
    return digest


def _version_dir(version: str, cache_dir: Path | None = None) -> Path:
    if cache_dir is None:
        from sparkrun.core.config import DEFAULT_CACHE_DIR

        cache_dir = DEFAULT_CACHE_DIR
    return Path(cache_dir) / "gateways" / SPARKROUTE_BINARY / version


def binary_path(version: str, *, cache_dir: Path | None = None) -> Path:
    """Cache location for the verified binary of *version*."""
    target_os, _ = platform_target()
    return _version_dir(version, cache_dir) / binary_name(target_os)


def archive_path(version: str, *, cache_dir: Path | None = None) -> Path:
    """Cache location for the verified release archive of *version*."""
    target_os, target_arch = platform_target()
    return _version_dir(version, cache_dir) / asset_name(version, target_os, target_arch)


def _binary_override() -> tuple[str, str] | None:
    """Return ``(env var, binary path)``, preferring the current name.

    The variable name is returned rather than assumed, so every message about
    the override names the one the caller actually set — telling someone their
    binary came from a variable they never exported is how a stale legacy
    export survives a debugging session.
    """
    override = os.environ.get(BINARY_OVERRIDE_ENV)
    if override:
        return BINARY_OVERRIDE_ENV, override
    for name in LEGACY_BINARY_OVERRIDE_ENVS:
        legacy = os.environ.get(name)
        if legacy:
            logger.warning("%s is deprecated; use %s", name, BINARY_OVERRIDE_ENV)
            return name, legacy
    return None


def ensure_binary(
    version: str | None = None,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Return a path to a verified SparkRoute binary, downloading if needed.

    Args:
        version: Release to acquire.  Defaults to :data:`SPARKROUTE_VERSION`.
        cache_dir: Sparkrun cache root.  Defaults to the configured one.
        force: Re-download even when a cached archive already verifies.

    Raises:
        GatewayReleaseError: The platform is unsupported, no digest is pinned,
            the download failed, or the asset did not match its digest.
    """
    override = _binary_override()
    if override:
        source_env, override_path = override
        path = Path(override_path)
        if not path.is_file():
            raise GatewayReleaseError("%s points at %s, which is not a file" % (source_env, path))
        logger.warning(
            "Using SparkRoute binary from %s (%s) — version and checksum verification are skipped",
            source_env,
            path,
        )
        return path

    version = version or SPARKROUTE_VERSION
    target_os, target_arch = platform_target()
    digest = expected_digest(version, target_os, target_arch)
    destination = binary_path(version, cache_dir=cache_dir)
    archive = archive_path(version, cache_dir=cache_dir)

    if not force and archive.is_file():
        actual = _file_digest(archive)
        if actual == digest:
            # The archive pin says nothing about a modified extracted binary.
            # Restore from the freshly verified archive, including offline.
            _extract_binary(archive, target_os, destination)
            return destination
        # A cached file that no longer matches is corruption or tampering, not
        # a version change (the version is in the path).  Re-fetch rather than
        # unpack it, and say so.
        logger.warning("Cached SparkRoute archive at %s failed verification; re-downloading", archive)

    asset = asset_name(version, target_os, target_arch)
    url = release_url(version, asset)
    archive.parent.mkdir(parents=True, exist_ok=True)
    _restrict_dir(archive.parent)

    logger.info("Downloading SparkRoute %s (%s/%s)", version, target_os, target_arch)
    tmp_path = _download(url, archive.parent)
    try:
        actual = _file_digest(tmp_path)
        if actual != digest:
            raise GatewayReleaseError("SparkRoute %s failed checksum verification (expected %s, got %s)" % (asset, digest, actual))
        # Verified: only now may it be unpacked, and only now does the archive
        # take the name a later call will trust.
        os.replace(tmp_path, archive)
    finally:
        tmp_path.unlink(missing_ok=True)

    _extract_binary(archive, target_os, destination)
    return destination


def _extract_binary(archive: Path, target_os: str, destination: Path) -> None:
    """Unpack the single executable member of *archive* to *destination*.

    Member *paths* from the archive are never used to build a filesystem path:
    the wanted entry is located by name and its content streamed to a temp file
    that becomes *destination*.  A crafted archive therefore cannot traverse
    out of the cache directory, and a partially written binary never appears
    under the final name.
    """
    wanted = binary_name(target_os)
    try:
        if archive.suffix == ".zip":
            source = _open_zip_member(archive, wanted)
        else:
            source = _open_tar_member(archive, wanted)
    except (tarfile.TarError, zipfile.BadZipFile, OSError) as exc:
        raise GatewayReleaseError("Could not read the SparkRoute archive %s: %s" % (archive.name, exc)) from exc

    handle, tmp_name = tempfile.mkstemp(dir=str(destination.parent), prefix=".sparkroute-", suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        with source, os.fdopen(handle, "wb") as out:
            written = 0
            while True:
                chunk = source.read(1 << 20)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_DOWNLOAD_BYTES:
                    raise GatewayReleaseError("SparkRoute binary exceeds the %d-byte limit" % MAX_DOWNLOAD_BYTES)
                out.write(chunk)
        if not written:
            raise GatewayReleaseError("SparkRoute archive %s contains an empty %s" % (archive.name, wanted))
        tmp_path.chmod(stat.S_IRWXU)
        # Avoid replacing an unchanged executable in use. A replaced symlink,
        # changed mode, or corrupted binary must be repaired from the archive.
        if (
            destination.is_file()
            and not destination.is_symlink()
            and (os.name == "nt" or stat.S_IMODE(destination.stat().st_mode) == stat.S_IRWXU)
            and _file_digest(destination) == _file_digest(tmp_path)
        ):
            tmp_path.unlink()
        else:
            os.replace(tmp_path, destination)
    except GatewayReleaseError:
        tmp_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise GatewayReleaseError("Could not unpack the SparkRoute binary: %s" % exc) from exc


def _open_tar_member(archive: Path, wanted: str):
    """Return a stream over *wanted* inside a tar archive."""
    handle = tarfile.open(archive, mode="r:gz")
    try:
        member = _tar_member(handle, wanted)
        stream = handle.extractfile(member)
        if stream is None:
            raise GatewayReleaseError("%s in %s is not a regular file" % (wanted, archive.name))
    except BaseException:
        handle.close()
        raise
    return _ClosingStream(stream, handle)


def _tar_member(handle: tarfile.TarFile, wanted: str) -> tarfile.TarInfo:
    """Locate the executable member, which the release stages as ``./<name>``."""
    for member in handle.getmembers():
        if _member_matches(member.name, wanted):
            if not member.isfile():
                raise GatewayReleaseError("%s is not a regular file in the archive" % wanted)
            return member
    raise GatewayReleaseError("The SparkRoute archive does not contain %s" % wanted)


def _open_zip_member(archive: Path, wanted: str):
    """Return a stream over *wanted* inside a zip archive."""
    handle = zipfile.ZipFile(archive)
    try:
        for info in handle.infolist():
            if not info.is_dir() and _member_matches(info.filename, wanted):
                return _ClosingStream(handle.open(info), handle)
        raise GatewayReleaseError("The SparkRoute archive does not contain %s" % wanted)
    except BaseException:
        handle.close()
        raise


def _member_matches(name: str, wanted: str) -> bool:
    """True when an archive entry names the wanted top-level executable.

    The release stages the binary at the archive root, but ``tar -C dir .``
    records it as ``./sparkroute``.  Only the root is accepted — matching any
    depth would let a crafted archive nominate a different file.
    """
    normalized = name.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized == wanted


class _ClosingStream:
    """A member stream that closes its owning archive when it closes."""

    def __init__(self, stream, archive):
        self._stream = stream
        self._archive = archive

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def close(self) -> None:
        try:
            self._stream.close()
        finally:
            self._archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _download(url: str, directory: Path) -> Path:
    """Fetch *url* into a private temp file inside *directory*."""
    if not url.startswith("https://"):
        raise GatewayReleaseError("Refusing to download the gateway binary over a non-HTTPS URL")

    handle, tmp_name = tempfile.mkstemp(dir=str(directory), prefix=".sparkroute-", suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as out:
            with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310 - scheme checked above
                # urllib follows redirects internally (GitHub redirects release
                # assets to its object store); re-check the scheme we actually
                # ended up on so a redirect can't downgrade the transport.
                final_url = response.geturl()
                if not final_url.startswith("https://"):
                    raise GatewayReleaseError("Gateway binary download was redirected to a non-HTTPS URL")
                written = 0
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_DOWNLOAD_BYTES:
                        raise GatewayReleaseError("Gateway binary exceeds the %d-byte download limit" % MAX_DOWNLOAD_BYTES)
                    out.write(chunk)
    except GatewayReleaseError:
        tmp_path.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, OSError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise GatewayReleaseError("Could not download SparkRoute from %s: %s" % (url, exc)) from exc
    return tmp_path


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restrict_dir(path: Path) -> None:
    try:
        path.chmod(0o700)
    except OSError:
        logger.debug("Could not chmod 0700 %s", path, exc_info=True)


def resolve_sparkrun_executable() -> str:
    """Return the executable the gateway should use to call back into sparkrun.

    Passed as the gateway's ``-sparkrun-command``.  It must be a **single
    executable path**, not an argv list: the gateway runs it as
    ``exec.Command(command, "gateway-bridge")``, appending exactly one
    argument, so a ``python -m sparkrun`` style invocation cannot be expressed.

    ``sys.executable`` is not a substitute either — under ``uvx``, pipx, or a
    venv it names the interpreter, and the bridge needs sparkrun's console
    script with its environment.

    Raises:
        GatewayReleaseError: No ``sparkrun`` console script is on PATH, so
            there is nothing the gateway could call back into.  This happens
            in a source checkout that was never installed; ``uv sync`` (or any
            pip/pipx/uvx install) provides the script.
    """
    script = shutil.which("sparkrun")
    if not script:
        raise GatewayReleaseError(
            "No 'sparkrun' executable on PATH for the gateway to call back into. "
            "The gateway appends a single argument to -sparkrun-command, so a "
            "'python -m sparkrun' invocation cannot be used; install sparkrun so its "
            "console script exists (e.g. 'uv sync')."
        )
    return script


__all__ = [
    "BINARY_OVERRIDE_ENV",
    "SPARKROUTE_BINARY",
    "SPARKROUTE_REPO",
    "SPARKROUTE_VERSION",
    "GatewayReleaseError",
    "LEGACY_BINARY_OVERRIDE_ENVS",
    "MAX_DOWNLOAD_BYTES",
    "RELEASE_CHECKSUMS",
    "archive_path",
    "archive_suffix",
    "asset_name",
    "binary_name",
    "binary_path",
    "ensure_binary",
    "expected_digest",
    "platform_target",
    "release_url",
    "resolve_sparkrun_executable",
]
