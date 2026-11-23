"""The one parser for literal flag values in ``command:`` templates."""

from __future__ import annotations

import pytest

from sparkrun.utils.command_flags import command_flag_value

COMMAND = """vllm serve {model} \\
  --port-range 9000 \\
  --gpu-memory-utilization 0.3 \\
  --served-model-name="my-model" \\
  --enforce-eager \\
  --tensor-parallel-size {tensor_parallel}"""


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (("--gpu-memory-utilization",), "0.3"),
        (("--served-model-name",), "my-model"),  # = spelling, quotes stripped
        (("--port",), None),  # whole token only: never --port-range
        (("--tensor-parallel-size",), None),  # placeholder belongs to the config chain
        (("--missing",), None),
        (("--GPU-Memory-Utilization",), "0.3"),
    ],
)
def test_reads_literal_values(flags, expected):
    assert command_flag_value(COMMAND, *flags) == expected


def test_last_occurrence_of_any_alias_wins():
    """argparse semantics: the engine uses the last value given for an option."""
    command = "vllm serve --kv-cache-memory 1G --kv-cache-memory-bytes 4G --kv-cache-memory 8G"
    assert command_flag_value(command, "--kv-cache-memory-bytes", "--kv-cache-memory") == "8G"
    assert command_flag_value("serve --kv-cache-dtype fp8 --kv-cache-dtype auto", "--kv-cache-dtype") == "auto"


def test_bare_flags_and_negative_numbers():
    assert command_flag_value(COMMAND, "--enforce-eager") is None
    assert command_flag_value(COMMAND, "--enforce-eager", bare="") == ""
    assert command_flag_value("llama-server --n-gpu-layers -1 --port 8000", "--n-gpu-layers") == "-1"


def test_placeholders_kept_when_asked():
    assert command_flag_value(COMMAND, "--tensor-parallel-size", literal_only=False) == "{tensor_parallel}"


def test_continuation_glued_to_value():
    assert command_flag_value("serve --gpu-memory-utilization 0.5\\\n --port 1", "--gpu-memory-utilization") == "0.5"


@pytest.mark.parametrize("command", [None, "", 42])
def test_non_string_input(command):
    assert command_flag_value(command, "--port") is None
