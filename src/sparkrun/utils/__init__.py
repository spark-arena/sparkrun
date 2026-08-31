"""Shared utility functions for sparkrun.

Small, self-contained helpers used across multiple modules.  The
implementations live in focused submodules (``net``, ``text``, ``env``,
``yaml_helpers``); they are re-exported here so existing
``from sparkrun.utils import <name>`` imports keep working.
"""

from __future__ import annotations

from sparkrun.utils.env import merge_env, resolve_ssh_user, suppress_noisy_loggers
from sparkrun.utils.images import (
    ImageRef,
    image_has_explicit_version,
    is_pullable_image_ref,
    parse_image_ref,
)
from sparkrun.utils.net import get_local_ips, is_local_host, is_valid_ip
from sparkrun.utils.text import coerce_value, format_duration, parse_kv_output, parse_scoped_name
from sparkrun.utils.yaml_helpers import load_yaml

__all__ = [
    "ImageRef",
    "coerce_value",
    "format_duration",
    "get_local_ips",
    "image_has_explicit_version",
    "is_local_host",
    "is_pullable_image_ref",
    "is_valid_ip",
    "load_yaml",
    "parse_image_ref",
    "merge_env",
    "parse_kv_output",
    "parse_scoped_name",
    "resolve_ssh_user",
    "suppress_noisy_loggers",
]
