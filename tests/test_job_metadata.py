"""Tests for ``sparkrun.orchestration.job_metadata`` — backends persistence (A1)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from sparkrun.core.backend_select import BackendBundle
from sparkrun.orchestration.collectives import NcclBackend, RcclBackend
from sparkrun.orchestration.job_metadata import (
    load_job_metadata,
    save_job_metadata,
)


@pytest.fixture
def mock_recipe():
    """Recipe stub with the attributes save_job_metadata reads."""
    r = mock.MagicMock()
    r.runtime = "vllm"
    r.model = "Qwen/Qwen3-1.7B"
    r.defaults = {"port": 8000}
    r.qualified_name = "test-recipe"
    r.executor = ""
    r.executor_config = None
    r.__getstate__ = mock.MagicMock(return_value={})
    return r


def test_save_job_metadata_persists_backends(tmp_path: Path, mock_recipe):
    """``backends`` kwarg is serialized to ``meta['backends']`` with
    ``{host: {vendor, backend}}`` shape."""
    cluster_id = "sparkrun_abc123def456"
    hosts = ["nv-host", "amd-host"]
    backends = {
        "nv-host": BackendBundle(accelerator_vendor="nvidia", collective=NcclBackend()),
        "amd-host": BackendBundle(accelerator_vendor="amd", collective=RcclBackend()),
    }

    save_job_metadata(
        cluster_id,
        mock_recipe,
        hosts,
        cache_dir=str(tmp_path),
        backends=backends,
    )

    meta = load_job_metadata(cluster_id, cache_dir=str(tmp_path))
    assert meta is not None
    assert "backends" in meta
    assert meta["backends"] == {
        "nv-host": {"vendor": "nvidia", "backend": "nccl"},
        "amd-host": {"vendor": "amd", "backend": "rccl"},
    }


def test_save_job_metadata_omits_backends_when_empty(tmp_path: Path, mock_recipe):
    """Empty backends dict is omitted from persisted metadata."""
    cluster_id = "sparkrun_aaaaaaaaaaaa"
    save_job_metadata(
        cluster_id,
        mock_recipe,
        ["h1"],
        cache_dir=str(tmp_path),
        backends={},
    )
    meta = load_job_metadata(cluster_id, cache_dir=str(tmp_path))
    assert meta is not None
    assert "backends" not in meta


def test_save_job_metadata_backends_none_omitted(tmp_path: Path, mock_recipe):
    """backends=None (default) is omitted from persisted metadata."""
    cluster_id = "sparkrun_bbbbbbbbbbbb"
    save_job_metadata(
        cluster_id,
        mock_recipe,
        ["h1"],
        cache_dir=str(tmp_path),
    )
    meta = load_job_metadata(cluster_id, cache_dir=str(tmp_path))
    assert meta is not None
    assert "backends" not in meta


def test_save_job_metadata_backends_roundtrip(tmp_path: Path, mock_recipe):
    """Single-host NVIDIA backend roundtrips through YAML serialization."""
    cluster_id = "sparkrun_cccccccccccc"
    backends = {
        "10.0.0.1": BackendBundle(accelerator_vendor="nvidia", collective=NcclBackend()),
    }
    save_job_metadata(
        cluster_id,
        mock_recipe,
        ["10.0.0.1"],
        cache_dir=str(tmp_path),
        backends=backends,
    )
    meta = load_job_metadata(cluster_id, cache_dir=str(tmp_path))
    assert meta is not None
    persisted = meta["backends"]["10.0.0.1"]
    # Schema: {vendor, backend} — names that survive readback unchanged.
    assert persisted["vendor"] == "nvidia"
    assert persisted["backend"] == "nccl"
