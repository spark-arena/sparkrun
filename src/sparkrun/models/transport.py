"""Builtin manifest-scoped copies with atomic file replacement and bounded retry."""

from __future__ import annotations

import base64
import logging
from pathlib import PurePosixPath
import time

from sparkrun.core.model_distribution import ModelCopyRequest, ModelTransferResult
from sparkrun.models.artifacts import ModelArtifactError
from sparkrun.models.host_io import ModelHostIO
from sparkrun.orchestration.ssh import RemoteResult, build_ssh_opts_string, rsync_retry_disabled, should_run_locally
from sparkrun.orchestration.transfer import (
    classify_rsync_failure,
    rsync_attribute_errors_only,
    rsync_had_vanished_files,
    rsync_has_attribute_permission_error,
)
from sparkrun.utils.shell import quote

logger = logging.getLogger(__name__)


def copy_script(request: ModelCopyRequest, target, ssh_kwargs: dict, *, relaxed: bool = False) -> str:
    """Copy only selected logical files; never traverse unrelated live blobs.

    Dereferencing the selected snapshot writes a self-contained snapshot with
    regular files. This avoids copying both blob and snapshot payloads or
    exporting links into a different host's cache-wide blob store.
    """
    source = str(PurePosixPath(request.source_cache_root) / request.manifest.relative_snapshot) + "/"
    destination = str(PurePosixPath(target.cache_root) / request.manifest.relative_snapshot) + "/"
    local_destination = (
        request.source_host is None and should_run_locally(target.host, ssh_kwargs.get("ssh_user"))
    ) or request.source_host == target.host
    args = ["rsync", "-r", "--copy-links", "--mkpath", "--protect-args", "--ignore-times", "--from0", "--partial-dir=.sparkrun-partial"]
    if not relaxed:
        args.append("--times")
    if not local_destination:
        opts = build_ssh_opts_string(**{key: ssh_kwargs[key] for key in ("ssh_user", "ssh_key", "ssh_options") if key in ssh_kwargs})
        args += ["-e", "ssh " + opts]
        user = ssh_kwargs.get("ssh_user")
        destination = (user + "@" if user else "") + target.transfer_host + ":" + destination
    data = base64.b64encode(b"\0".join(path.encode() for path in target.required_files) + b"\0").decode()
    command = " ".join(str(quote(arg)) for arg in args)
    return (
        "set -euo pipefail\nexport LC_ALL=C\n"
        "list=$(mktemp)\ntrap 'rm -f -- \"$list\"' EXIT\n"
        f'printf %s {quote(data)} | base64 -d > "$list"\n'
        + command
        + ' --files-from="$list" -- '
        + str(quote(source))
        + " "
        + str(quote(destination))
        + "\n"
    )


def builtin_copy(request: ModelCopyRequest, io: ModelHostIO, ssh_kwargs: dict) -> ModelTransferResult:
    outcomes, errors = {}, {}
    deadline = time.monotonic() + request.timeout
    for target in request.targets:
        if request.cancelled():
            raise ModelArtifactError("model copy cancelled")
        if not target.required_files:
            outcomes[target.host] = "already_present"
            continue
        relaxed = False
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                outcomes[target.host], errors[target.host] = "failed", "model transfer deadline exceeded"
                break
            result = io.execute(request.source_host, copy_script(request, target, ssh_kwargs, relaxed=relaxed), timeout=remaining)
            if request.cancelled():
                raise ModelArtifactError("model copy cancelled")
            transfer = RemoteResult(
                target.host, result.returncode, result.stdout.decode(errors="replace"), result.stderr.decode(errors="replace")
            )
            if transfer.success or rsync_attribute_errors_only(transfer):
                outcomes[target.host] = "complete"
                if request.progress is not None:
                    request.progress(target.host, sum(f.size for f in request.manifest.files if f.path in target.required_files))
                break
            # Keep the first diagnostic even if the retry succeeds. Its paths
            # explain source loss; never assert which process caused it.
            logger.warning(
                "Model copy %s -> %s attempt %d failed: %s",
                request.source_host or "control",
                target.host,
                attempt + 1,
                transfer.stderr[-1500:],
            )
            if attempt == 0 and not rsync_retry_disabled():
                if rsync_had_vanished_files(transfer):
                    report = io.observe(request.source_host, request.manifest, request.source_cache_root)
                    if report.complete:
                        continue
                elif rsync_has_attribute_permission_error(transfer):
                    relaxed = True
                    continue
            outcomes[target.host] = "failed"
            errors[target.host] = classify_rsync_failure(transfer) + ": " + transfer.stderr[-1500:]
            break
    return ModelTransferResult(request.manifest.identity, outcomes, errors)
