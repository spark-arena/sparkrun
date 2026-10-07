# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see LICENSE_EXCEPTION.

"""Operation-scoped registry retention, separate from the memory budget."""

import logging

from .progress import PROGRESS, size

logger = logging.getLogger(__name__)
DEFAULT_CACHE_BYTES = 16 << 30
FREE_RESERVE_BYTES = 16 << 30


def registry_cache_budget(runner, host, directory, settings):
    requested = settings.get("registry_cache_bytes", DEFAULT_CACHE_BYTES)
    if not requested:
        return 0
    try:
        # POSIX output, 1024-byte blocks, and space available to the management
        # user (excluding filesystem blocks reserved for root). The directory
        # already exists; no shell interpolation or payload allocation here.
        raw = runner.execute(host, ["env", "LC_ALL=C", "df", "-Pk", directory])
        fields = raw.decode().splitlines()[-1].split()
        available = int(fields[3]) * 1024
        if available < 0 or len(fields) < 6:
            raise ValueError("invalid disk availability")
    except Exception:
        logger.log(PROGRESS, "OCI Relay: disk headroom unavailable; registry cache disabled")
        return 0
    budget = min(requested, max(0, available - FREE_RESERVE_BYTES))
    logger.log(PROGRESS, "OCI Relay: registry disk cache up to %s; %s available, keeping %s free",
               size(budget), size(available), size(FREE_RESERVE_BYTES))
    return budget
