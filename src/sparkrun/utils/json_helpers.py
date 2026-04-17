"""JSON serialization helpers for sparkrun."""

from __future__ import annotations

import dataclasses
import json
from typing import Any


class SparkrunJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles sparkrun objects.

    Automatically serializes objects with to_dict() and dataclasses.
    """

    def default(self, o: Any) -> Any:
        if hasattr(o, "to_dict") and callable(o.to_dict):
            return o.to_dict()
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        return super().default(o)


def dumps_json(data: Any, pretty: bool = False) -> str:
    """Serialize data to JSON string using SparkrunJSONEncoder."""
    kwargs: dict[str, Any] = {"cls": SparkrunJSONEncoder}
    if pretty:
        kwargs["indent"] = 2
    return json.dumps(data, **kwargs)
