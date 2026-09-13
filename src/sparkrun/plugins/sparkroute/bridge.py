# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""One-request JSON stdio runner for the hidden gateway bridge command."""

from __future__ import annotations

import json
import logging
import sys
from typing import BinaryIO, TextIO

from .operations import execute
from .protocol import PROTOCOL_VERSION, ProtocolError, error_response, read_request, success_response

logger = logging.getLogger(__name__)


def run_stdio(stdin: BinaryIO | None = None, stdout: TextIO | None = None) -> int:
    input_stream = stdin if stdin is not None else sys.stdin.buffer
    output_stream = stdout if stdout is not None else sys.stdout
    request_id = ""
    # Replies are emitted in the version that was *asked for*, so a caller's
    # correlation check passes and it reads the actual error — including
    # ``unsupported_version`` for a mismatched installation.
    schema_version = PROTOCOL_VERSION
    exit_code = 0
    try:
        request = read_request(input_stream)
        request_id = request.request_id
        schema_version = request.schema_version
        response = success_response(request_id, execute(request), schema_version)
    except ProtocolError as exc:
        response = error_response(exc.request_id or request_id, exc, exc.schema_version or schema_version)
        exit_code = 2 if exc.code in {"empty_request", "invalid_json", "invalid_request", "unsupported_version"} else 0
    except Exception:
        # Raw exceptions may include resolved recipe or transport material.
        # Keep stderr diagnostic output content-free just like the response.
        logger.error("unexpected gateway bridge failure")
        response = error_response(
            request_id,
            ProtocolError("internal_error", "unexpected Sparkrun bridge failure", retryable=True),
            schema_version,
        )
        exit_code = 1
    json.dump(response, output_stream, sort_keys=True, separators=(",", ":"))
    output_stream.write("\n")
    output_stream.flush()
    return exit_code


__all__ = ["run_stdio"]
