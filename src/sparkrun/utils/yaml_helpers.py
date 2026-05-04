"""YAML formatting helpers for sparkrun."""

from __future__ import annotations

import yaml


class LiteralBlockDumper(yaml.SafeDumper):
    """YAML dumper that uses literal block style (|) for multiline strings."""

    pass


def _literal_str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


# noinspection PyTypeChecker
LiteralBlockDumper.add_representer(str, _literal_str_representer)
