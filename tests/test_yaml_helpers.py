"""LiteralBlockDumper: output meant for people (recipe export)."""

from __future__ import annotations

import yaml

from sparkrun.utils.yaml_helpers import LiteralBlockDumper


def _dump(data) -> str:
    return yaml.dump(data, Dumper=LiteralBlockDumper, indent=2, sort_keys=False, default_flow_style=False)


def test_long_numeric_lists_are_written_inline():
    sizes = [1, 2, 4, 8, 12, 16, 20, 24]
    text = _dump({"compilation_config": {"cudagraph_capture_sizes": sizes}})
    assert "cudagraph_capture_sizes: [1, 2, 4, 8, 12, 16, 20, 24]" in text
    assert yaml.safe_load(text)["compilation_config"]["cudagraph_capture_sizes"] == sizes


def test_floats_count_as_numbers():
    assert "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]" in _dump({"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})


def test_numeric_lists_of_any_length_are_inline():
    assert _dump({"x": [7]}) == "x: [7]\n"
    assert _dump({"x": [0, 1.5]}) == "x: [0, 1.5]\n"
    assert _dump({"x": []}) == "x: []\n"


def test_lists_of_other_things_stay_block():
    for items in (["a", "b", "c", "d", "e", "f"], [True, False, True, False, True, False], [1, 2, 3, 4, 5, "six"], [{"a": 1}] * 6):
        text = _dump({"x": items})
        assert "[" not in text.split("\n", 1)[0]
        assert yaml.safe_load(text)["x"] == items


def test_very_long_numeric_lists_wrap_but_round_trip():
    sizes = list(range(1, 200, 3))
    assert yaml.safe_load(_dump({"x": sizes}))["x"] == sizes


def test_multiline_strings_still_use_literal_blocks():
    assert "command: |\n  vllm serve m\n  --port 8000\n" in _dump({"command": "vllm serve m\n--port 8000\n"})


def test_a_single_string_is_inline():
    assert _dump({"custom_ops": ["all"]}) == "custom_ops: [all]\n"
    assert yaml.safe_load(_dump({"x": ["a: b"]}))["x"] == ["a: b"]  # quoted when it needs to be
    assert yaml.safe_load(_dump({"x": ["yes"]}))["x"] == ["yes"]  # not read back as a boolean


def test_a_single_multiline_string_keeps_its_literal_block():
    text = _dump({"pre_exec": ["echo one\necho two\n"]})
    assert "[" not in text and "|" in text
    assert yaml.safe_load(text)["pre_exec"] == ["echo one\necho two\n"]


def test_several_strings_stay_block():
    assert _dump({"mods": ["a", "b"]}) == "mods:\n- a\n- b\n"


def test_a_single_long_string_stays_block():
    path = "/home/someone/.cache/huggingface/hub/models--org--model/snapshots/abc"
    assert _dump({"x": [path]}) == "x:\n- %s\n" % path
