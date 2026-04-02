"""JSON serialization helpers for sparkrun."""
from __future__ import annotations

import dataclasses
import json
from typing import Any


class SparkrunJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles sparkrun objects.

    Automatically serializes objects with to_dict() and dataclasses.
    """

    def default(self, obj: Any) -> Any:
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)


def dumps_json(data: Any, pretty: bool = False) -> str:
    """Serialize data to JSON string using SparkrunJSONEncoder."""
    kwargs: dict[str, Any] = {"cls": SparkrunJSONEncoder}
    if pretty:
        kwargs["indent"] = 2
    return json.dumps(data, **kwargs)
