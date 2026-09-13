# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Tests for SparkRoute binary acquisition.

The load-bearing property is that nothing unverified ever becomes executable:
a missing pin, a digest mismatch, a plain-HTTP URL and an oversized response
must each fail *before* the file lands where the engine would exec it.

Release assets are archives, so unpacking is part of that boundary — the
digest is checked against the *archive*, and the member is located by name
rather than by trusting a path the archive supplies.
"""

from __future__ import annotations

import hashlib
import io
import stat
import tarfile
import zipfile
from unittest import mock

import pytest

from sparkrun.plugins.sparkroute import release

BINARY_BODY = b"#!/fake/sparkroute\n" + b"x" * 1024


def _tar_archive(members: dict[str, bytes]) -> bytes:
    """Build a .tar.gz the way the release workflow's ``tar -C stage .`` does."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as handle:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            handle.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


def _zip_archive(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as handle:
        for name, body in members.items():
            handle.writestr(name, body)
    return buffer.getvalue()


@pytest.fixture
def payload() -> bytes:
    """A release archive staging the binary at its root, plus licence files."""
    return _tar_archive(
        {
            "./sparkroute": BINARY_BODY,
            "./LICENSE": b"AGPL-3.0-only",
            "./NOTICE": b"notice",
        }
    )


@pytest.fixture
def pinned(monkeypatch, payload):
    """Pin a digest for a fixed platform so tests don't depend on the host."""
    monkeypatch.setattr(release, "platform_target", lambda: ("linux", "arm64"))
    monkeypatch.setattr(
        release,
        "RELEASE_CHECKSUMS",
        {("9.9.9", "linux", "arm64"): hashlib.sha256(payload).hexdigest()},
    )
    monkeypatch.delenv(release.BINARY_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv(release.LEGACY_BINARY_OVERRIDE_ENVS[0], raising=False)
    return "9.9.9"


class _Stream:
    """Minimal urlopen stand-in that yields *body* in chunks."""

    def __init__(self, body: bytes, url: str = "https://objects.example/asset"):
        self._buffer = io.BytesIO(body)
        self._url = url

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def geturl(self) -> str:
        return self._url

    def read(self, size: int) -> bytes:
        return self._buffer.read(size)


# ---------------------------------------------------------------------------
# Platform / naming
# ---------------------------------------------------------------------------


def test_platform_target_normalizes_arch(monkeypatch):
    monkeypatch.setattr(release.platform, "system", lambda: "Linux")
    monkeypatch.setattr(release.platform, "machine", lambda: "aarch64")
    assert release.platform_target() == ("linux", "arm64")


def test_platform_target_rejects_unknown_platform(monkeypatch):
    monkeypatch.setattr(release.platform, "system", lambda: "Haiku")
    monkeypatch.setattr(release.platform, "machine", lambda: "m68k")
    with pytest.raises(release.GatewayReleaseError, match="No SparkRoute build"):
        release.platform_target()


def test_asset_and_url_shape():
    """Mirrors the release workflow's ``base=sparkroute_${VERSION}_${GOOS}_${GOARCH}``."""
    assert release.asset_name("1.2.3", "linux", "arm64") == "sparkroute_1.2.3_linux_arm64.tar.gz"
    assert release.asset_name("1.2.3", "windows", "amd64") == "sparkroute_1.2.3_windows_amd64.zip"
    assert release.binary_name("linux") == "sparkroute"
    assert release.binary_name("windows") == "sparkroute.exe"
    assert release.release_url("1.2.3", "asset") == "https://github.com/sparksq/sparkroute/releases/download/v1.2.3/asset"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_unpinned_platform_refuses_to_download(monkeypatch, tmp_path):
    monkeypatch.setattr(release, "platform_target", lambda: ("linux", "arm64"))
    monkeypatch.setattr(release, "RELEASE_CHECKSUMS", {})
    monkeypatch.delenv(release.BINARY_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv(release.LEGACY_BINARY_OVERRIDE_ENVS[0], raising=False)
    with mock.patch.object(release.urllib.request, "urlopen") as urlopen:
        with pytest.raises(release.GatewayReleaseError, match="No pinned checksum"):
            release.ensure_binary("9.9.9", cache_dir=tmp_path)
    urlopen.assert_not_called()


def test_default_release_has_valid_pins_for_every_supported_platform():
    expected = {(release.SPARKROUTE_VERSION, system, arch) for system in ("darwin", "linux", "windows") for arch in ("amd64", "arm64")}
    assert expected <= release.RELEASE_CHECKSUMS.keys()
    for key in expected:
        digest = release.RELEASE_CHECKSUMS[key]
        assert len(digest) == 64
        assert len(bytes.fromhex(digest)) == 32


def test_download_verifies_then_unpacks_an_executable(pinned, payload, tmp_path):
    with mock.patch.object(release.urllib.request, "urlopen", return_value=_Stream(payload)):
        path = release.ensure_binary(pinned, cache_dir=tmp_path)
    assert path.name == "sparkroute"
    assert path.read_bytes() == BINARY_BODY
    assert path.stat().st_mode & stat.S_IXUSR
    # The verified archive is kept so the pin stays re-checkable, and no
    # partial download is left behind looking like a usable binary.
    assert sorted(p.name for p in path.parent.iterdir()) == [
        "sparkroute",
        "sparkroute_9.9.9_linux_arm64.tar.gz",
    ]


def test_only_the_binary_is_unpacked(pinned, payload, tmp_path):
    """License material stays in the retained, verified distribution archive."""
    with mock.patch.object(release.urllib.request, "urlopen", return_value=_Stream(payload)):
        path = release.ensure_binary(pinned, cache_dir=tmp_path)
    assert not (path.parent / "LICENSE").exists()
    with tarfile.open(release.archive_path(pinned, cache_dir=tmp_path)) as archive:
        assert archive.extractfile("./LICENSE").read() == b"AGPL-3.0-only"
        assert archive.extractfile("./NOTICE").read() == b"notice"


def test_corrupted_extracted_binary_is_repaired_offline(pinned, payload, tmp_path):
    with mock.patch.object(release.urllib.request, "urlopen", return_value=_Stream(payload)):
        path = release.ensure_binary(pinned, cache_dir=tmp_path)
    path.write_bytes(b"corrupted executable")
    with mock.patch.object(release.urllib.request, "urlopen") as urlopen:
        repaired = release.ensure_binary(pinned, cache_dir=tmp_path)
    urlopen.assert_not_called()
    assert repaired.read_bytes() == BINARY_BODY


def test_cached_symlink_is_replaced_without_overwriting_target(pinned, payload, tmp_path):
    with mock.patch.object(release.urllib.request, "urlopen", return_value=_Stream(payload)):
        path = release.ensure_binary(pinned, cache_dir=tmp_path)
    target = tmp_path / "external"
    target.write_bytes(b"external file")
    path.unlink()
    path.symlink_to(target)
    with mock.patch.object(release.urllib.request, "urlopen") as urlopen:
        repaired = release.ensure_binary(pinned, cache_dir=tmp_path)
    urlopen.assert_not_called()
    assert not repaired.is_symlink()
    assert repaired.read_bytes() == BINARY_BODY
    assert target.read_bytes() == b"external file"


def test_digest_mismatch_leaves_nothing_executable(pinned, tmp_path):
    with mock.patch.object(release.urllib.request, "urlopen", return_value=_Stream(b"tampered")):
        with pytest.raises(release.GatewayReleaseError, match="checksum verification"):
            release.ensure_binary(pinned, cache_dir=tmp_path)
    destination = release.binary_path(pinned, cache_dir=tmp_path)
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []


def test_cached_binary_is_reused_without_downloading(pinned, payload, tmp_path):
    with mock.patch.object(release.urllib.request, "urlopen", return_value=_Stream(payload)):
        first = release.ensure_binary(pinned, cache_dir=tmp_path)
    with mock.patch.object(release.urllib.request, "urlopen") as urlopen:
        second = release.ensure_binary(pinned, cache_dir=tmp_path)
    assert first == second
    urlopen.assert_not_called()


def test_deleted_binary_is_reextracted_without_downloading(pinned, payload, tmp_path):
    """The cached archive still matches its pin, so it is trusted to re-unpack."""
    with mock.patch.object(release.urllib.request, "urlopen", return_value=_Stream(payload)):
        path = release.ensure_binary(pinned, cache_dir=tmp_path)
    path.unlink()
    with mock.patch.object(release.urllib.request, "urlopen") as urlopen:
        again = release.ensure_binary(pinned, cache_dir=tmp_path)
    urlopen.assert_not_called()
    assert again.read_bytes() == BINARY_BODY


def test_corrupted_cache_is_refetched_not_executed(pinned, payload, tmp_path):
    archive = release.archive_path(pinned, cache_dir=tmp_path)
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"corrupted")
    with mock.patch.object(release.urllib.request, "urlopen", return_value=_Stream(payload)):
        path = release.ensure_binary(pinned, cache_dir=tmp_path)
    assert path.read_bytes() == BINARY_BODY


def test_oversized_download_is_abandoned(pinned, monkeypatch, tmp_path):
    monkeypatch.setattr(release, "MAX_DOWNLOAD_BYTES", 16)
    with mock.patch.object(release.urllib.request, "urlopen", return_value=_Stream(b"y" * 4096)):
        with pytest.raises(release.GatewayReleaseError, match="download limit"):
            release.ensure_binary(pinned, cache_dir=tmp_path)
    assert list(release.binary_path(pinned, cache_dir=tmp_path).parent.iterdir()) == []


def test_non_https_redirect_is_rejected(pinned, payload, tmp_path):
    stream = _Stream(payload, url="http://objects.example/asset")
    with mock.patch.object(release.urllib.request, "urlopen", return_value=stream):
        with pytest.raises(release.GatewayReleaseError, match="non-HTTPS"):
            release.ensure_binary(pinned, cache_dir=tmp_path)


def test_plain_http_release_url_is_rejected(pinned, monkeypatch, tmp_path):
    monkeypatch.setattr(release, "release_url", lambda *_a: "http://github.com/x/y")
    with mock.patch.object(release.urllib.request, "urlopen") as urlopen:
        with pytest.raises(release.GatewayReleaseError, match="non-HTTPS"):
            release.ensure_binary(pinned, cache_dir=tmp_path)
    urlopen.assert_not_called()


# ---------------------------------------------------------------------------
# Unpacking
# ---------------------------------------------------------------------------


def _pin(monkeypatch, archive: bytes, target=("linux", "arm64")):
    monkeypatch.setattr(release, "platform_target", lambda: target)
    monkeypatch.setattr(release, "RELEASE_CHECKSUMS", {("9.9.9", *target): hashlib.sha256(archive).hexdigest()})
    monkeypatch.delenv(release.BINARY_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv(release.LEGACY_BINARY_OVERRIDE_ENVS[0], raising=False)


def test_windows_asset_is_unpacked_from_a_zip(monkeypatch, tmp_path):
    archive = _zip_archive({"sparkroute.exe": BINARY_BODY, "LICENSE": b"x"})
    _pin(monkeypatch, archive, target=("windows", "amd64"))
    with mock.patch.object(release.urllib.request, "urlopen", return_value=_Stream(archive)):
        path = release.ensure_binary("9.9.9", cache_dir=tmp_path)
    assert path.name == "sparkroute.exe"
    assert path.read_bytes() == BINARY_BODY


def test_archive_without_the_binary_is_rejected(monkeypatch, tmp_path):
    archive = _tar_archive({"./README.md": b"nope"})
    _pin(monkeypatch, archive)
    with mock.patch.object(release.urllib.request, "urlopen", return_value=_Stream(archive)):
        with pytest.raises(release.GatewayReleaseError, match="does not contain"):
            release.ensure_binary("9.9.9", cache_dir=tmp_path)
    assert not release.binary_path("9.9.9", cache_dir=tmp_path).exists()


def test_traversing_member_name_cannot_escape_the_cache(monkeypatch, tmp_path):
    """A member named ``../../sparkroute`` must not satisfy the lookup, and
    must not be written anywhere: the destination is this module's choice, but
    only a root-level entry is accepted as the binary at all."""
    archive = _tar_archive({"../../sparkroute": BINARY_BODY})
    _pin(monkeypatch, archive)
    with mock.patch.object(release.urllib.request, "urlopen", return_value=_Stream(archive)):
        with pytest.raises(release.GatewayReleaseError, match="does not contain"):
            release.ensure_binary("9.9.9", cache_dir=tmp_path)
    assert not (tmp_path.parent / "sparkroute").exists()


def test_nested_member_is_not_mistaken_for_the_binary(monkeypatch, tmp_path):
    archive = _tar_archive({"./tools/sparkroute": BINARY_BODY})
    _pin(monkeypatch, archive)
    with mock.patch.object(release.urllib.request, "urlopen", return_value=_Stream(archive)):
        with pytest.raises(release.GatewayReleaseError, match="does not contain"):
            release.ensure_binary("9.9.9", cache_dir=tmp_path)


def test_unreadable_archive_is_reported_not_executed(monkeypatch, tmp_path):
    archive = b"not an archive at all"
    _pin(monkeypatch, archive)
    with mock.patch.object(release.urllib.request, "urlopen", return_value=_Stream(archive)):
        with pytest.raises(release.GatewayReleaseError, match="Could not read"):
            release.ensure_binary("9.9.9", cache_dir=tmp_path)
    assert not release.binary_path("9.9.9", cache_dir=tmp_path).exists()


def test_decompression_bomb_is_bounded(monkeypatch, tmp_path):
    archive = _tar_archive({"./sparkroute": b"z" * 8192})
    _pin(monkeypatch, archive)
    monkeypatch.setattr(release, "MAX_DOWNLOAD_BYTES", 64)
    with mock.patch.object(release.urllib.request, "urlopen", return_value=_Stream(archive)):
        with pytest.raises(release.GatewayReleaseError, match="exceeds"):
            release.ensure_binary("9.9.9", cache_dir=tmp_path)
    assert not release.binary_path("9.9.9", cache_dir=tmp_path).exists()


# ---------------------------------------------------------------------------
# Development override
# ---------------------------------------------------------------------------


def test_binary_override_bypasses_acquisition(monkeypatch, tmp_path):
    local = tmp_path / "sparkroute"
    local.write_bytes(b"locally built")
    monkeypatch.setenv(release.BINARY_OVERRIDE_ENV, str(local))
    with mock.patch.object(release.urllib.request, "urlopen") as urlopen:
        assert release.ensure_binary(cache_dir=tmp_path) == local
    urlopen.assert_not_called()


def test_legacy_override_env_is_still_honoured(monkeypatch, tmp_path):
    """An existing development setup must not silently start downloading."""
    local = tmp_path / "sparkroute"
    local.write_bytes(b"locally built")
    monkeypatch.delenv(release.BINARY_OVERRIDE_ENV, raising=False)
    monkeypatch.setenv(release.LEGACY_BINARY_OVERRIDE_ENVS[0], str(local))
    with mock.patch.object(release.urllib.request, "urlopen") as urlopen:
        assert release.ensure_binary(cache_dir=tmp_path) == local
    urlopen.assert_not_called()


def test_messages_name_the_env_var_that_was_actually_set(monkeypatch, tmp_path, caplog):
    """Crediting the current name for a path that came from a legacy export is
    how a stale export survives a debugging session."""
    local = tmp_path / "sparkroute"
    local.write_bytes(b"locally built")
    monkeypatch.delenv(release.BINARY_OVERRIDE_ENV, raising=False)
    monkeypatch.setenv(release.LEGACY_BINARY_OVERRIDE_ENVS[0], str(local))
    with caplog.at_level("WARNING"):
        release.ensure_binary(cache_dir=tmp_path)
    assert release.LEGACY_BINARY_OVERRIDE_ENVS[0] in caplog.text
    assert "Using SparkRoute binary from %s" % release.BINARY_OVERRIDE_ENV not in caplog.text


def test_a_legacy_override_pointing_nowhere_names_the_legacy_var(monkeypatch, tmp_path):
    monkeypatch.delenv(release.BINARY_OVERRIDE_ENV, raising=False)
    monkeypatch.setenv(release.LEGACY_BINARY_OVERRIDE_ENVS[0], str(tmp_path / "missing"))
    with pytest.raises(release.GatewayReleaseError, match=release.LEGACY_BINARY_OVERRIDE_ENVS[0]):
        release.ensure_binary(cache_dir=tmp_path)


def test_current_override_env_wins_over_the_legacy_one(monkeypatch, tmp_path):
    current = tmp_path / "current"
    current.write_bytes(b"current")
    legacy = tmp_path / "legacy"
    legacy.write_bytes(b"legacy")
    monkeypatch.setenv(release.BINARY_OVERRIDE_ENV, str(current))
    monkeypatch.setenv(release.LEGACY_BINARY_OVERRIDE_ENVS[0], str(legacy))
    assert release.ensure_binary(cache_dir=tmp_path) == current


def test_binary_override_must_point_at_a_file(monkeypatch, tmp_path):
    monkeypatch.setenv(release.BINARY_OVERRIDE_ENV, str(tmp_path / "missing"))
    with pytest.raises(release.GatewayReleaseError, match="not a file"):
        release.ensure_binary(cache_dir=tmp_path)


# ---------------------------------------------------------------------------
# Callback command
# ---------------------------------------------------------------------------


def test_sparkrun_executable_is_the_console_script():
    with mock.patch.object(release.shutil, "which", return_value="/opt/venv/bin/sparkrun"):
        assert release.resolve_sparkrun_executable() == "/opt/venv/bin/sparkrun"


def test_missing_console_script_is_an_error_not_a_module_invocation():
    """The gateway runs -sparkrun-command as ``exec.Command(cmd,
    "gateway-bridge")``, appending exactly one argument — so a ``python -m
    sparkrun`` form cannot be expressed and must not be silently attempted."""
    with mock.patch.object(release.shutil, "which", return_value=None):
        with pytest.raises(release.GatewayReleaseError, match="single argument"):
            release.resolve_sparkrun_executable()
