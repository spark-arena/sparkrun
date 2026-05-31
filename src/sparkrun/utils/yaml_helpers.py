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
    """YAML dumper that uses literal block style (|) for multiline strings."""

    pass


def _literal_str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


# noinspection PyTypeChecker
LiteralBlockDumper.add_representer(str, _literal_str_representer)
