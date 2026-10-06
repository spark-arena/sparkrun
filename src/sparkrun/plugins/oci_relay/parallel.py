# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see LICENSE_EXCEPTION.
"""Bounded host setup with a completion barrier before operation cleanup."""

from concurrent.futures import ThreadPoolExecutor, as_completed


def parallel(items, operation):
    """Return keyed results; settle started work before propagating any failure.

    Workers may register owned processes/directories with Runner. Cleanup must
    never race a still-running worker that could create another resource.
    """
    items = list(items)
    if not items:
        return {}
    with ThreadPoolExecutor(max_workers=min(8, len(items)), thread_name_prefix="oci-relay-setup") as pool:
        pending = {pool.submit(operation, item): item for item in items}
        try:
            results = {pending[future]: future.result() for future in as_completed(pending)}
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return {item: results[item] for item in items}
