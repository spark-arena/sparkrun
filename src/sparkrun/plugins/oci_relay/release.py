# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see LICENSE_EXCEPTION.

"""Pinned, verified target-binary acquisition; no target needs internet access."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import tarfile
import tempfile
import urllib.request

from . import __version__

logger = logging.getLogger(__name__)
MAX_ARCHIVE = 128 << 20


class BinaryUnavailable(RuntimeError):
    pass


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_elf(path: Path, arch: str) -> None:
    with path.open("rb") as stream:
        header = stream.read(20)
    expected = {"amd64": 62, "arm64": 183}.get(arch)
    if (
        expected is None or len(header) != 20 or header[:6] != b"\x7fELF\x02\x01"
        or int.from_bytes(header[18:20], "little") != expected
    ):
        raise BinaryUnavailable(f"{path} is not a Linux {arch} executable")


def acquire(arch: str, settings: dict, *, offline: bool) -> tuple[Path, str]:
    version = settings.get("version", __version__)
    if version != __version__:
        raise BinaryUnavailable(f"adapter {__version__} requires the matching engine release")
    paths = settings.get("binary_paths", {})
    hashes = settings.get("binary_sha256", {})
    if not isinstance(paths, dict) or not isinstance(hashes, dict):
        raise BinaryUnavailable("binary_paths and binary_sha256 must map architecture to values")
    if arch in paths:
        path = Path(paths[arch]).expanduser()
        expected = hashes.get(arch)
        if not isinstance(expected, str) or len(expected) != 64:
            raise BinaryUnavailable(f"a trusted binary_sha256.{arch} is required for an explicit binary")
        if not path.is_file() or file_digest(path) != expected:
            raise BinaryUnavailable(f"binary checksum mismatch for {arch}")
        verify_elf(path, arch)
        return path, expected
    development = settings.get("development_binary")
    if development:
        path = Path(development).expanduser()
        if not path.is_file():
            raise BinaryUnavailable(f"development binary does not exist: {path}")
        verify_elf(path, arch)
        logger.warning("OCI Relay uses explicitly configured development binary %s", path)
        return path, file_digest(path)

    releases = json.loads(Path(__file__).with_name("releases.json").read_text())
    pinned = releases.get(version, {}).get(f"linux/{arch}")
    if not isinstance(pinned, dict) or len(str(pinned.get("sha256", ""))) != 64:
        raise BinaryUnavailable(
            f"no trusted release checksum is pinned for OCI Relay {version} linux/{arch}; "
            "configure binary_paths and binary_sha256, or an explicit development_binary"
        )
    url = pinned["url"]
    if not isinstance(url, str) or not url.startswith("https://"):
        raise BinaryUnavailable("release URL must use HTTPS")
    cache = Path(settings.get("cache_dir", "~/.cache/oci-relay")).expanduser() / version / f"linux-{arch}"
    archive = cache / "release.tar.gz"
    binary = cache / "oci-relay"
    if not archive.is_file() or file_digest(archive) != pinned["sha256"]:
        if offline:
            raise BinaryUnavailable("offline mode requires a verified cached release or explicit binary")
        cache.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".download-", dir=cache)
        try:
            with os.fdopen(fd, "wb") as output, urllib.request.urlopen(url, timeout=30) as response:
                if not response.url.startswith("https://"):
                    raise BinaryUnavailable("release download redirected away from HTTPS")
                count = 0
                while chunk := response.read(64 << 10):
                    count += len(chunk)
                    if count > MAX_ARCHIVE:
                        raise BinaryUnavailable("release archive exceeds size limit")
                    output.write(chunk)
            if file_digest(Path(temporary)) != pinned["sha256"]:
                raise BinaryUnavailable("release archive checksum mismatch")
            os.replace(temporary, archive)
        finally:
            Path(temporary).unlink(missing_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".binary-", dir=cache)
    try:
        with os.fdopen(fd, "wb") as output, tarfile.open(archive, "r:gz") as tar:
            matches = [entry for entry in tar.getmembers() if entry.name in {"oci-relay", "./oci-relay"}]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size > MAX_ARCHIVE:
                raise BinaryUnavailable("release must contain one regular oci-relay executable")
            stream = tar.extractfile(matches[0])
            if stream is None:
                raise BinaryUnavailable("release executable missing")
            with stream:
                while chunk := stream.read(64 << 10):
                    output.write(chunk)
        verify_elf(Path(temporary), arch)
        os.chmod(temporary, 0o555)
        os.replace(temporary, binary)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return binary, file_digest(binary)


def acquire_decoder(arch: str, settings: dict, binary: Path) -> tuple[Path, str] | None:
    """Acquire only a decoder bound to a trusted release or explicit dev settings."""
    paths, hashes = settings.get("decoder_paths", {}), settings.get("decoder_sha256", {})
    if not isinstance(paths, dict) or not isinstance(hashes, dict):
        raise BinaryUnavailable("decoder_paths and decoder_sha256 must map architecture to values")
    if arch in paths:
        helper = Path(paths[arch]).expanduser()
        expected = hashes.get(arch)
        if not isinstance(expected, str) or len(expected) != 64 or file_digest(helper) != expected:
            raise BinaryUnavailable("decoder checksum mismatch or missing trusted hash")
        verify_elf(helper, arch)
        return helper, expected
    if settings.get("development_binary"):
        if not settings.get("development_unpigz"):
            return None
        helper = Path(settings["development_unpigz"]).expanduser()
        verify_elf(helper, arch)
        return helper, file_digest(helper)
    if arch in settings.get("binary_paths", {}):
        return None
    # Revalidate the archive, not an editable sidecar manifest. Legacy releases
    # have no helper and remain usable through the ordinary compressed import.
    pins = json.loads(Path(__file__).with_name("releases.json").read_text())
    pin = pins.get(__version__, {}).get("linux/" + arch, {})
    archive = binary.parent / "release.tar.gz"
    if not archive.is_file() or file_digest(archive) != pin.get("sha256"):
        raise BinaryUnavailable("decoder release archive checksum mismatch")
    with tarfile.open(archive, "r:gz") as tar:
        def entry(name, maximum, optional=False):
            matches = [e for e in tar.getmembers() if e.name in {name, "./" + name}]
            if not matches and optional:
                return None
            if len(matches) != 1 or not matches[0].isfile() or not 0 < matches[0].size <= maximum:
                raise BinaryUnavailable("release must contain one regular " + name)
            with tar.extractfile(matches[0]) as stream:
                return stream.read(maximum + 1)
        manifest_raw = entry("bundle.json", 64 << 10, optional=True)
        if manifest_raw is None:
            return None
        manifest = json.loads(manifest_raw)
        if (manifest.get("format") != 1 or manifest.get("version") != __version__
                or manifest.get("platform") != "linux/" + arch
                or "receiver-unpigz-v1" not in manifest.get("capabilities", [])):
            raise BinaryUnavailable("unsupported decoder bundle metadata")
        for name, payload in (("oci-relay", binary.read_bytes()), ("unpigz", entry("unpigz", 16 << 20))):
            info = manifest.get("files", {}).get(name, {})
            if info.get("sha256") != hashlib.sha256(payload).hexdigest() or info.get("size") != len(payload):
                raise BinaryUnavailable("bundle checksum mismatch: " + name)
        # Retain redistributed licenses alongside the executable on controller.
        licenses = entry("UNPIGZ_LICENSES.txt", 1 << 20)
    helper = binary.parent / "unpigz"
    fd, temporary = tempfile.mkstemp(prefix=".decoder-", dir=binary.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
        verify_elf(Path(temporary), arch)
        os.chmod(temporary, 0o555)
        os.replace(temporary, helper)
        (binary.parent / "UNPIGZ_LICENSES.txt").write_bytes(licenses)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return helper, file_digest(helper)
