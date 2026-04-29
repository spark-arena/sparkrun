"""Prometheus text exposition format parser."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Matches: metric_name{label="val",label2="val2"} value
# or:      metric_name value
_METRIC_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)"  # metric name
    r"(\{[^}]*\})?"  # optional labels
    r"\s+"  # whitespace
    r"([0-9eE.+\-]+|NaN|Inf|\+Inf|-Inf)"  # value
)


def parse_prometheus_text(text: str) -> dict[str, float]:
    """Parse Prometheus text exposition format into a flat dict.

    Each metric line becomes a dict entry where the key is the full
    metric name including labels (e.g. ``"nv_gpu_utilization_percent{gpu=\\"0\\"}"``).
    Lines without labels use just the metric name as key.

    Comment lines (starting with ``#``) and empty lines are skipped.

    Args:
        text: Raw Prometheus exposition text.

    Returns:
        Dict mapping metric keys to float values.
    """
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        m = _METRIC_RE.match(line)
        if m is None:
            continue

        name = m.group(1)
        labels = m.group(2) or ""
        raw_value = m.group(3)

        key = name + labels

        try:
            value = float(raw_value) if raw_value in ("NaN", "+Inf", "-Inf", "Inf") else float(raw_value)
        except ValueError:
            logger.debug("Skipping unparseable metric value: %s", line)
            continue

        metrics[key] = value

    return metrics


def extract_label(key: str, label_name: str) -> str | None:
    """Extract a label value from a Prometheus metric key.

    Args:
        key: Full metric key like ``'nv_gpu_info{gpu="0",name="GH200"}'``.
        label_name: Label to extract (e.g. ``"name"``).

    Returns:
        Label value string, or None if not found.
    """
    # Match label_name="value" within braces
    pattern = r'%s="([^"]*)"' % re.escape(label_name)
    m = re.search(pattern, key)
    return m.group(1) if m else None
