"""Tests for sparkrun.utils.json_helpers."""

from __future__ import annotations

import dataclasses
import json

from sparkrun.utils.json_helpers import SparkrunJSONEncoder, dumps_json


class _HasToDict:
    """Test helper with to_dict() method."""

    def __init__(self, value):
        self.value = value

    def to_dict(self):
        return {"value": self.value}


@dataclasses.dataclass
class _SampleDC:
    name: str
    count: int


class TestSparkrunJSONEncoder:
    def test_plain_dict(self):
        result = json.dumps({"a": 1}, cls=SparkrunJSONEncoder)
        assert json.loads(result) == {"a": 1}

    def test_to_dict_object(self):
        obj = _HasToDict(42)
        result = json.dumps(obj, cls=SparkrunJSONEncoder)
        assert json.loads(result) == {"value": 42}

    def test_dataclass(self):
        dc = _SampleDC(name="test", count=5)
        result = json.dumps(dc, cls=SparkrunJSONEncoder)
        assert json.loads(result) == {"name": "test", "count": 5}

    def test_nested_to_dict(self):
        data = {"items": [_HasToDict("a"), _HasToDict("b")]}
        result = json.dumps(data, cls=SparkrunJSONEncoder)
        parsed = json.loads(result)
        assert parsed == {"items": [{"value": "a"}, {"value": "b"}]}

    def test_nested_dataclass(self):
        data = [_SampleDC("x", 1), _SampleDC("y", 2)]
        result = json.dumps(data, cls=SparkrunJSONEncoder)
        parsed = json.loads(result)
        assert parsed == [{"name": "x", "count": 1}, {"name": "y", "count": 2}]


class TestDumpsJson:
    def test_basic(self):
        result = dumps_json({"key": "val"})
        assert json.loads(result) == {"key": "val"}

    def test_pretty(self):
        result = dumps_json({"key": "val"}, pretty=True)
        assert "\n" in result
        assert json.loads(result) == {"key": "val"}

    def test_not_pretty(self):
        result = dumps_json({"key": "val"}, pretty=False)
        assert "\n" not in result

    def test_with_to_dict(self):
        result = dumps_json(_HasToDict(99))
        assert json.loads(result) == {"value": 99}

    def test_with_dataclass(self):
        result = dumps_json(_SampleDC("dc", 7))
        assert json.loads(result) == {"name": "dc", "count": 7}
