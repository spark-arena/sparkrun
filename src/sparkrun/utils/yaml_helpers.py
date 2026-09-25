"""YAML read/write helpers for sparkrun."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path) -> dict:
    """Load a YAML file, returning an empty dict on parse failure."""
    with Path(path).open() as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


class LiteralBlockDumper(yaml.SafeDumper):
    """YAML dumper for human-read output (recipe export, recipe state).

    * Multiline strings use literal block style (``|``).
    * All-numeric lists and single-string lists use flow style
      (``[1, 2, 4, 8]``, ``[all]``): see :func:`_prefers_flow`.
    """

    pass


#: A single-string list goes inline up to this length.
FLOW_SINGLE_STRING_MAX_CHARS = 40


def _prefers_flow(items: list) -> bool:
    """Whether a list reads better inline than one item per line.

    * **Any list of plain numbers.** Numbers are data, not structure: a
      ``cudagraph_capture_sizes`` ladder is dozens of integers, and block
      style turns it into a column that dwarfs the recipe around it. Booleans
      are excluded (``bool`` is an ``int`` subclass, but a list of flags is not
      numeric data).
    * **A single short string**, e.g. ``custom_ops: [all]``: one element in
      block style spends two lines on what is visibly a value. A long one (a
      path, a command) stays in block style, where it has room, and a
      multiline one keeps its ``|`` literal block.
    """
    if not items:
        return False
    if len(items) == 1 and isinstance(items[0], str):
        return "\n" not in items[0] and len(items[0]) <= FLOW_SINGLE_STRING_MAX_CHARS
    return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in items)


def _list_representer(dumper: LiteralBlockDumper, data: list) -> yaml.SequenceNode:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True if _prefers_flow(data) else None)


def _literal_str_representer(dumper: LiteralBlockDumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


# noinspection PyTypeChecker
LiteralBlockDumper.add_representer(str, _literal_str_representer)
# noinspection PyTypeChecker
LiteralBlockDumper.add_representer(list, _list_representer)
