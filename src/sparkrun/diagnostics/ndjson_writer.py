"""Append-only, immediate-flush NDJSON writer.

Each call to :meth:`NDJSONWriter.emit` writes one JSON line with envelope
fields (``_type``, ``_seq``, ``_ts``) and flushes immediately so records
survive crashes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class NDJSONWriter:
    """Append-only NDJSON writer with immediate flush.

    Every emitted record is wrapped in an envelope with:
    - ``_type``: record type string
    - ``_seq``: monotonically increasing sequence number
    - ``_ts``: ISO-8601 UTC timestamp

    Usage::

        with NDJSONWriter("output.ndjson") as w:
            w.emit("host_hardware", {"host": "10.0.0.1", "cpu_cores": 16})
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._seq = 0
        self._fh = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def seq(self) -> int:
        return self._seq

    def open(self) -> NDJSONWriter:
        self._fh = self._path.open("a", encoding="utf-8")
        return self

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def emit(self, record_type: str, data: dict | None = None) -> dict:
        """Write one NDJSON record and flush immediately.

        Args:
            record_type: Value for the ``_type`` envelope field.
            data: Payload fields merged into the record.

        Returns:
            The full record dict that was written.
        """
        self._seq += 1
        record = {
            "_type": record_type,
            "_seq": self._seq,
            "_ts": datetime.now(timezone.utc).isoformat(),
        }
        if data:
            record.update(data)

        line = json.dumps(record, default=str)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()
        else:
            logger.warning("NDJSONWriter not open; record %d dropped", self._seq)

        return record

    def __enter__(self) -> NDJSONWriter:
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
