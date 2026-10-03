"""Checksum verification for cached HF models.

Presence checks (:func:`sparkrun.models.download.is_model_cached`, the
``model_sync.sh`` scan) cannot see a truncated or raced transfer: the snapshot
symlink and the blob both exist, but the bytes are wrong.  HF content-addresses
weight blobs (``blobs/<sha256>`` where the filename is the sha256 of the
contents), so verification is a hash pass with no network access and no
manifest.  Corruption from a raced rsync therefore shows up as a mismatch even
though every file is present.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from sparkrun.core.config import resolve_hf_cache_home
from sparkrun.models.download import _snapshot_dirs_for_revision, is_gguf_model, model_cache_path

logger = logging.getLogger(__name__)

# Generous: verification hashes every weight blob (hundreds of GB on large
# models).  NVMe sustains well over 1 GB/s of sha256; slow disks need room.
DEFAULT_VERIFY_TIMEOUT = 1800

# Written next to a model's cache after a clean hash pass; a marker younger
# than every weight blob short-circuits the next verification (see
# ``verify_model_local`` and the matching fast path in ``model_verify.sh``).
VERIFY_MARKER_NAME = ".sparkrun-verified"

# Weight patterns mirror model_sync.sh / is_model_cached.  Only weight blobs
# are hashed: they are the LFS files whose names are sha256 digests, and they
# are the payload a raced transfer actually damages.
_WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.gguf")


def model_verify_disabled() -> bool:
    """True when the pre-flight verification is switched off."""
    return bool(os.environ.get("SPARKRUN_NO_MODEL_VERIFY"))


def _weight_files(snapshots: Path, revision: str | None):
    """Yield weight-file paths under the revision's snapshot directories.

    Recursive (``rglob``), mirroring the remote scan: some repos shard their
    weights into a subdirectory, and a top-level-only glob would report those
    caches as unverified and trigger downloads they don't need.
    """
    model_cache = snapshots.parent
    if revision:
        dirs = _snapshot_dirs_for_revision(model_cache, snapshots, revision)
    else:
        dirs = _snapshot_dirs_for_revision(model_cache, snapshots, "main")
        if not dirs:
            dirs = [d for d in snapshots.iterdir() if d.is_dir()] if snapshots.is_dir() else []
    for d in dirs:
        for pattern in _WEIGHT_PATTERNS:
            for f in sorted(d.rglob(pattern)):
                if f.is_file():  # follows the symlink; dangling entries are skipped by callers' checks
                    yield f


def verify_model_local(model_id: str, cache_dir: str | None = None, revision: str | None = None) -> list[Path] | None:
    """Hash the local cache's weight blobs and compare with their names.

    Returns ``None`` when there is nothing to verify (no cache or no matching
    revision), otherwise the list of corrupt/missing weight blobs — empty when
    the cache verifies.  GGUF caches have a different layout and are not
    verified here (callers exclude them upstream of this module).

    Steady-state fast path: a verification marker younger than every weight
    blob short-circuits the hash pass, so repeat launches don't re-hash
    hundreds of gigabytes.  Any blob written after the last clean pass (fresh
    mtime — downloads and re-downloads alike) invalidates the marker.
    """
    if is_gguf_model(model_id):
        return []
    cache = Path(resolve_hf_cache_home(cache_dir))
    safe_name = model_id.replace("/", "--")
    model_cache = cache / "hub" / f"models--{safe_name}"
    snapshots = model_cache / "snapshots"
    if not snapshots.is_dir():
        return None

    weights = list(_weight_files(snapshots, revision))
    if not weights:
        return None
    marker = model_cache / VERIFY_MARKER_NAME
    try:
        if marker.is_file() and all(p.stat().st_mtime <= marker.stat().st_mtime for p in weights):
            return []
    except OSError:
        pass

    bad: list[Path] = []
    for f in weights:
        blob = f.resolve()
        if not blob.is_file():
            logger.warning("missing blob for snapshot entry: %s", f)
            bad.append(blob)
            continue
        h = hashlib.sha256()
        with open(blob, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != blob.name:
            logger.warning("checksum mismatch: %s", blob)
            bad.append(blob)
    if not bad:
        try:
            marker.touch()
        except OSError:
            pass
    return bad


def render_verify_script(model_id: str, cache: str, revision: str | None) -> str:
    """Build the remote verification script for *model_id*.

    *cache* must already be resolved (``resolve_hf_cache_home``).  The script
    shares ``_hf_snapshots.sh`` with the sync scripts so revision resolution
    cannot drift from the download path.
    """
    from sparkrun.scripts import read_script
    from sparkrun.utils.shell import quote, validate_interpolated_path

    revision_arg = quote(revision or "")
    cache = validate_interpolated_path(cache, field_name="cache_dir")
    cache_path = validate_interpolated_path(model_cache_path(model_id, cache), field_name="model cache path")
    return read_script("model_verify.sh").format(cache_path=cache_path, revision=revision_arg)


def verify_model_on_hosts(
    model_id: str,
    hosts: list[str],
    cache_dir: str | None = None,
    revision: str | None = None,
    timeout: int = DEFAULT_VERIFY_TIMEOUT,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
) -> list[str]:
    """Return the hosts whose cached copy fails checksum verification.

    Runs ``model_verify.sh`` on every host in parallel.  Verification hashes
    the pinned snapshot's weight blobs (see the module docstring), so a host
    that merely *looks* cached but holds a raced/truncated blob is reported
    bad and the caller re-distributes to it.
    """
    if not hosts:
        return []
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    script = render_verify_script(model_id, resolve_hf_cache_home(cache_dir), revision)
    results = run_remote_scripts_parallel(
        list(hosts),
        script,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        ssh_options=ssh_options,
        timeout=timeout,
        quiet=True,
        allow_local=True,
        session_guard=True,
    )
    failed = [r for r in results if not r.success]
    for r in failed:
        # The script echoes the offending blob names; surface them instead of
        # leaving the operator a bare "failed on N hosts".
        detail = ((r.stdout or "") + (r.stderr or "")).strip()
        if detail:
            logger.warning("Model verification on %s reported:\n%s", r.host, detail[-1500:])
    return [r.host for r in failed]


def repair_model_on_host(
    model_id: str,
    host: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    hf_token: str | None = None,
    timeout: int = DEFAULT_VERIFY_TIMEOUT,
    ssh_user: str | None = None,
    ssh_key: str | None = None,
    ssh_options: list[str] | None = None,
) -> bool:
    """Repair a host's cache in place: hash, remove corrupt blobs, re-download.

    Used when the *source* of a distribution failed verification: the corrupt
    blobs are deleted (so nothing serves half-transferred bytes), then the
    ensure script runs with ``SPARKRUN_FORCE_DOWNLOAD=1`` so the presence
    shortcut cannot hide the now-missing shard.  Returns True when the host
    ends up with a usable cache.
    """
    from sparkrun.models.distribute import _build_model_ensure_script
    from sparkrun.orchestration.ssh import run_script_on_host

    kw = dict(ssh_user=ssh_user, ssh_key=ssh_key, ssh_options=ssh_options, session_guard=True)
    cache = resolve_hf_cache_home(cache_dir)

    verify = "export SPARKRUN_VERIFY_REPAIR=1\n" + render_verify_script(model_id, cache, revision)
    verify_result = run_script_on_host(host, verify, ssh_kwargs=kw, timeout=timeout, quiet=True)
    if not verify_result.success:
        # rc=1 with no repair output means the cache is absent, not corrupt;
        # the forced ensure below downloads it either way.
        logger.debug("verify(+repair) on %s exited %d; continuing to forced download", host, verify_result.returncode)

    ensure = _build_model_ensure_script(model_id, cache, revision=revision, hf_token=hf_token, force_download=True)
    ensure_result = run_script_on_host(host, ensure, ssh_kwargs=kw, timeout=7200)
    if not ensure_result.success:
        logger.error("forced re-download failed on %s (rc=%d)", host, ensure_result.returncode)
    return ensure_result.success
