"""Tests for ``SparkrunConfig.default_benchmark_framework`` field."""

from __future__ import annotations

from pathlib import Path

import yaml

from sparkrun.core.config import SparkrunConfig


def test_default_benchmark_framework_returns_llama_benchy_when_unset(tmp_path: Path):
    """When config.yaml is absent or has no defaults block, the value
    falls back to ``llama-benchy``.
    """
    config_path = tmp_path / "config.yaml"
    # File does not exist
    config = SparkrunConfig(config_path=config_path)
    assert config.default_benchmark_framework == "llama-benchy"


def test_default_benchmark_framework_returns_llama_benchy_when_defaults_missing_key(tmp_path: Path):
    """A populated defaults block without ``benchmark_framework`` still
    falls back to the built-in default.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"defaults": {"transformers": "t5"}}))

    config = SparkrunConfig(config_path=config_path)
    assert config.default_benchmark_framework == "llama-benchy"


def test_default_benchmark_framework_loads_explicit_value(tmp_path: Path):
    """An explicit ``defaults.benchmark_framework`` overrides the fallback."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"defaults": {"benchmark_framework": "tool-eval-bench"}}))

    config = SparkrunConfig(config_path=config_path)
    assert config.default_benchmark_framework == "tool-eval-bench"


def test_default_benchmark_framework_coerces_to_string(tmp_path: Path):
    """Non-string YAML scalars are coerced to ``str``."""
    config_path = tmp_path / "config.yaml"
    # Write raw YAML so the value parses as plain string (most common case).
    config_path.write_text("defaults:\n  benchmark_framework: custom-fw\n")

    config = SparkrunConfig(config_path=config_path)
    assert isinstance(config.default_benchmark_framework, str)
    assert config.default_benchmark_framework == "custom-fw"


def test_default_benchmark_framework_handles_malformed_defaults(tmp_path: Path):
    """A malformed defaults block (non-dict) must not crash; falls back."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("defaults: not-a-dict\n")

    config = SparkrunConfig(config_path=config_path)
    assert config.default_benchmark_framework == "llama-benchy"
