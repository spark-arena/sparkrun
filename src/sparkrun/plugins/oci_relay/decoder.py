# SPDX-FileCopyrightText: 2026 Spark Arena
# SPDX-License-Identifier: Apache-2.0
"""Receiver decoder resource policy; payload processing remains entirely in Go."""

import logging

from .host import OperationError
from .progress import PROGRESS, size
from .source_policy import docker_facts

logger = logging.getLogger(__name__)


def arguments(runner, host, helper, settings, facts, peer_limits):
    mode = settings.get("receiver_decoder", "auto")
    if mode == "none" or settings.get("receiver_import", "pull") != "pull":
        return []

    def unavailable(reason):
        if mode == "unpigz":
            raise OperationError(reason)
        logger.log(PROGRESS, "OCI Relay: %s: compressed import (%s)", host, reason)
        return []

    if helper is None:
        return unavailable("bundled decoder unavailable for this engine release")
    info = docker_facts(runner, host)
    if info.get("driver") != "overlay2":
        return unavailable("bundled decoding is selected only for overlay2")
    # A separate operation-owned writable directory is mounted into read-only
    # native-cache helpers. It never grants write access to Docker's stores.
    directory = runner.directory(host)
    reserve = settings.get("decode_reserve_bytes", 16 << 30)
    try:
        available = []
        for path in (directory, info["root"]):
            raw = runner.execute(host, ["env", "LC_ALL=C", "df", "-Pk", path])
            available.append(int(raw.decode().splitlines()[-1].split()[3]) * 1024)
        # Count scratch, Docker's downloaded copy and unpacked layer data on
        # the most constrained filesystem. Leave 1 GiB for concurrent activity.
        cap = min(settings.get("max_decode_bytes", 64 << 30), max(0, (min(available) - reserve - (1 << 30)) // 3))
        cap = (cap // (1 << 20)) * (1 << 20)
    except Exception:
        return unavailable("decoder disk headroom unavailable")
    if cap < 4 << 20:
        return unavailable("insufficient decoder disk headroom")
    workers = settings.get("decode_workers", min(8, peer_limits["source_streams"], max(1, int(facts.get("cpus", 4)))))
    logger.log(PROGRESS, "OCI Relay: %s: bundled unpigz, %d workers, up to %s scratch", host, workers, size(cap))
    return ["--decoder", mode, "--unpigz", helper, "--decode-spool-dir", directory,
            "--decode-workers", str(workers), "--max-decode-bytes", str(cap), "--decode-reserve-bytes", str(reserve)]
