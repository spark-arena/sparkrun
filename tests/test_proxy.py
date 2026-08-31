"""Tests for sparkrun proxy package — discovery, config, engine, CLI."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from click.testing import CliRunner


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def jobs_dir(tmp_path: Path) -> Path:
    """Create a temporary jobs directory with sample metadata."""
    d = tmp_path / "jobs"
    d.mkdir()
    return d


@pytest.fixture
def sample_job_meta() -> dict[str, Any]:
    """A sample job metadata dict."""
    return {
        "cluster_id": "sparkrun_bbbbbbbbbbbbbbb1_bbbbbbbbbbb1",
        "recipe": "qwen3-1.7b-vllm",
        "model": "Qwen/Qwen3-1.7B",
        "runtime": "vllm",
        "hosts": ["192.168.11.13"],
        "port": 8000,
        "tensor_parallel": 1,
    }


@pytest.fixture
def sample_job_meta_with_served_name() -> dict[str, Any]:
    """A sample job metadata with served_model_name."""
    return {
        "cluster_id": "sparkrun_bbbbbbbbbbbbbbb2_bbbbbbbbbbb2",
        "recipe": "qwen3-custom",
        "model": "Qwen/Qwen3-1.7B",
        "runtime": "vllm",
        "hosts": ["192.168.11.14"],
        "port": 9000,
        "served_model_name": "my-qwen",
        "tensor_parallel": 2,
    }


@pytest.fixture
def populated_jobs_dir(jobs_dir: Path, sample_job_meta, sample_job_meta_with_served_name) -> Path:
    """Jobs directory with two metadata files."""
    with open(jobs_dir / "abc123.yaml", "w") as f:
        yaml.safe_dump(sample_job_meta, f)
    with open(jobs_dir / "def456.yaml", "w") as f:
        yaml.safe_dump(sample_job_meta_with_served_name, f)
    return jobs_dir.parent


@pytest.fixture
def proxy_config_path(tmp_path: Path) -> Path:
    """Path for a proxy config file."""
    return tmp_path / "proxy.yaml"


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    """Temporary state directory for engine."""
    d = tmp_path / "proxy_state"
    d.mkdir()
    return d


def _make_recipe(name="test", model="Qwen/Qwen3-1.7B", runtime="vllm", defaults=None):
    """Create a real Recipe object for testing."""
    from sparkrun.core.recipe import Recipe

    return Recipe.from_dict(
        {
            "name": name,
            "model": model,
            "runtime": runtime,
            "container": "test-image:latest",
            "defaults": defaults or {},
        }
    )


# =====================================================================
# Tests: derive_cluster_id with port/served_model_name
# =====================================================================


class TestGenerateClusterId:
    """Test derive_cluster_id() with port and served_model_name."""

    def test_backward_compat_no_overrides(self):
        """Omitting overrides produces same hash as original behavior."""
        from sparkrun.orchestration.job_metadata import derive_cluster_id

        recipe = _make_recipe()
        hosts = ["10.0.0.1", "10.0.0.2"]

        # No overrides, no defaults with port/served_name
        id1 = derive_cluster_id(recipe, hosts)
        id2 = derive_cluster_id(recipe, hosts, overrides=None)
        id3 = derive_cluster_id(recipe, hosts, overrides={})
        assert id1 == id2 == id3

    def test_different_ports_different_ids(self):
        """Same model on different ports produces different IDs."""
        from sparkrun.orchestration.job_metadata import derive_cluster_id

        recipe = _make_recipe()
        hosts = ["10.0.0.1"]

        id_8000 = derive_cluster_id(recipe, hosts, overrides={"port": 8000})
        id_9000 = derive_cluster_id(recipe, hosts, overrides={"port": 9000})
        assert id_8000 != id_9000

    def test_different_served_names_different_ids(self):
        """Same model with different served names produces different IDs."""
        from sparkrun.orchestration.job_metadata import derive_cluster_id

        recipe = _make_recipe()
        hosts = ["10.0.0.1"]

        id_a = derive_cluster_id(recipe, hosts, overrides={"served_model_name": "model-a"})
        id_b = derive_cluster_id(recipe, hosts, overrides={"served_model_name": "model-b"})
        assert id_a != id_b

    def test_port_from_recipe_defaults(self):
        """Port from recipe defaults is included in hash."""
        from sparkrun.orchestration.job_metadata import derive_cluster_id

        recipe_with_port = _make_recipe(defaults={"port": 8080})
        recipe_no_port = _make_recipe()
        hosts = ["10.0.0.1"]

        id_with = derive_cluster_id(recipe_with_port, hosts)
        id_without = derive_cluster_id(recipe_no_port, hosts)
        assert id_with != id_without

    def test_override_takes_precedence_over_default(self):
        """Override port takes precedence over recipe default."""
        from sparkrun.orchestration.job_metadata import derive_cluster_id

        recipe = _make_recipe(defaults={"port": 8000})
        hosts = ["10.0.0.1"]

        id_default = derive_cluster_id(recipe, hosts)
        id_override = derive_cluster_id(recipe, hosts, overrides={"port": 9000})
        assert id_default != id_override

    def test_same_override_matches_default(self):
        """When override equals default, ID matches no-override case."""
        from sparkrun.orchestration.job_metadata import derive_cluster_id

        recipe = _make_recipe(defaults={"port": 8000})
        hosts = ["10.0.0.1"]

        id_default = derive_cluster_id(recipe, hosts)
        id_same = derive_cluster_id(recipe, hosts, overrides={"port": 8000})
        assert id_default == id_same

    def test_tp1_does_not_change_id(self):
        """tp=1 (default) should not change the cluster ID."""
        from sparkrun.orchestration.job_metadata import derive_cluster_id

        recipe = _make_recipe()
        hosts = ["10.0.0.1"]

        id_no_tp = derive_cluster_id(recipe, hosts)
        id_tp1 = derive_cluster_id(recipe, hosts, overrides={"tensor_parallel": 1})
        assert id_no_tp == id_tp1

    def test_different_tp_different_ids(self):
        """Different non-default TP values produce different IDs."""
        from sparkrun.orchestration.job_metadata import derive_cluster_id

        recipe = _make_recipe()
        hosts = ["10.0.0.1"]

        id_tp2 = derive_cluster_id(recipe, hosts, overrides={"tensor_parallel": 2})
        id_tp4 = derive_cluster_id(recipe, hosts, overrides={"tensor_parallel": 4})
        assert id_tp2 != id_tp4

    def test_tp_changes_id_vs_default(self):
        """Non-default TP should produce different ID than no TP."""
        from sparkrun.orchestration.job_metadata import derive_cluster_id

        recipe = _make_recipe()
        hosts = ["10.0.0.1"]

        id_default = derive_cluster_id(recipe, hosts)
        id_tp2 = derive_cluster_id(recipe, hosts, overrides={"tensor_parallel": 2})
        assert id_default != id_tp2

    def test_pp_changes_id(self):
        """Non-default PP should produce different ID than no PP."""
        from sparkrun.orchestration.job_metadata import derive_cluster_id

        recipe = _make_recipe()
        hosts = ["10.0.0.1"]

        id_default = derive_cluster_id(recipe, hosts)
        id_pp2 = derive_cluster_id(recipe, hosts, overrides={"pipeline_parallel": 2})
        assert id_default != id_pp2

    def test_pp1_does_not_change_id(self):
        """pp=1 (default) should not change the cluster ID."""
        from sparkrun.orchestration.job_metadata import derive_cluster_id

        recipe = _make_recipe()
        hosts = ["10.0.0.1"]

        id_no_pp = derive_cluster_id(recipe, hosts)
        id_pp1 = derive_cluster_id(recipe, hosts, overrides={"pipeline_parallel": 1})
        assert id_no_pp == id_pp1

    def test_tp_and_pp_from_defaults(self):
        """TP and PP from recipe defaults should affect cluster ID."""
        from sparkrun.orchestration.job_metadata import derive_cluster_id

        recipe_plain = _make_recipe()
        recipe_parallel = _make_recipe(defaults={"tensor_parallel": 2, "pipeline_parallel": 2})
        hosts = ["10.0.0.1"]

        id_plain = derive_cluster_id(recipe_plain, hosts)
        id_parallel = derive_cluster_id(recipe_parallel, hosts)
        assert id_plain != id_parallel


# =====================================================================
# Tests: save_job_metadata with port/served_model_name
# =====================================================================


class TestSaveJobMetadata:
    """Test port and served_model_name persistence in job metadata."""

    def test_port_persisted(self, tmp_path: Path):
        """Port from overrides is saved in metadata."""
        from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

        recipe = _make_recipe()
        save_job_metadata(
            "sparkrun_aaaaaaaaaaaaaaa1_aaaaaaaaaaa1",
            recipe,
            ["10.0.0.1"],
            overrides={"port": 9000},
            cache_dir=str(tmp_path),
        )

        meta = load_job_metadata("sparkrun_aaaaaaaaaaaaaaa1_aaaaaaaaaaa1", cache_dir=str(tmp_path))
        assert meta is not None
        assert meta["port"] == 9000

    def test_served_model_name_persisted(self, tmp_path: Path):
        """served_model_name from overrides is saved in metadata."""
        from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

        recipe = _make_recipe()
        save_job_metadata(
            "sparkrun_aaaaaaaaaaaaaaa2_aaaaaaaaaaa2",
            recipe,
            ["10.0.0.1"],
            overrides={"served_model_name": "my-model"},
            cache_dir=str(tmp_path),
        )

        meta = load_job_metadata("sparkrun_aaaaaaaaaaaaaaa2_aaaaaaaaaaa2", cache_dir=str(tmp_path))
        assert meta is not None
        assert meta["served_model_name"] == "my-model"

    def test_port_from_recipe_defaults(self, tmp_path: Path):
        """Port from recipe defaults is saved when no override."""
        from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

        recipe = _make_recipe(defaults={"port": 8080})
        save_job_metadata(
            "sparkrun_aaaaaaaaaaaaaaa3_aaaaaaaaaaa3",
            recipe,
            ["10.0.0.1"],
            cache_dir=str(tmp_path),
        )

        meta = load_job_metadata("sparkrun_aaaaaaaaaaaaaaa3_aaaaaaaaaaa3", cache_dir=str(tmp_path))
        assert meta is not None
        assert meta["port"] == 8080

    def test_no_port_no_served_name(self, tmp_path: Path):
        """Missing port/served_name fields when not set anywhere."""
        from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

        recipe = _make_recipe()
        save_job_metadata(
            "sparkrun_aaaaaaaaaaaaaaa4_aaaaaaaaaaa4",
            recipe,
            ["10.0.0.1"],
            cache_dir=str(tmp_path),
        )

        meta = load_job_metadata("sparkrun_aaaaaaaaaaaaaaa4_aaaaaaaaaaa4", cache_dir=str(tmp_path))
        assert meta is not None
        assert "port" not in meta
        assert "served_model_name" not in meta

    def test_api_key_persisted_via_runtime(self, tmp_path: Path):
        """Runtime's resolve_api_key result is persisted to metadata."""
        from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

        recipe = _make_recipe(defaults={"api_key": "sk-abc"})

        class _Rt:
            def resolve_api_key(self, recipe, overrides=None):
                return recipe.defaults.get("api_key")

        save_job_metadata(
            "sparkrun_aaaaaaaaaaaaaaa5_aaaaaaaaaaa5",
            recipe,
            ["10.0.0.1"],
            cache_dir=str(tmp_path),
            runtime=_Rt(),
        )
        meta = load_job_metadata("sparkrun_aaaaaaaaaaaaaaa5_aaaaaaaaaaa5", cache_dir=str(tmp_path))
        assert meta is not None
        assert meta["api_key"] == "sk-abc"

    def test_api_key_omitted_when_runtime_returns_none(self, tmp_path: Path):
        """No api_key field when resolve_api_key returns None."""
        from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

        recipe = _make_recipe()

        class _Rt:
            def resolve_api_key(self, recipe, overrides=None):
                return None

        save_job_metadata(
            "sparkrun_aaaaaaaaaaaaaaa6_aaaaaaaaaaa6",
            recipe,
            ["10.0.0.1"],
            cache_dir=str(tmp_path),
            runtime=_Rt(),
        )
        meta = load_job_metadata("sparkrun_aaaaaaaaaaaaaaa6_aaaaaaaaaaa6", cache_dir=str(tmp_path))
        assert meta is not None
        assert "api_key" not in meta

    def test_api_key_skipped_when_no_runtime(self, tmp_path: Path):
        """Backward compat: no runtime arg means no api_key resolution."""
        from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

        recipe = _make_recipe(defaults={"api_key": "sk-ignored"})
        save_job_metadata(
            "sparkrun_aaaaaaaaaaaaaaa7_aaaaaaaaaaa7",
            recipe,
            ["10.0.0.1"],
            cache_dir=str(tmp_path),
        )
        meta = load_job_metadata("sparkrun_aaaaaaaaaaaaaaa7_aaaaaaaaaaa7", cache_dir=str(tmp_path))
        assert meta is not None
        assert "api_key" not in meta


# =====================================================================
# Tests: Discovery
# =====================================================================


class TestDiscovery:
    """Test endpoint discovery from job metadata."""

    def test_discover_basic(self, populated_jobs_dir: Path):
        """Discover endpoints from job metadata files."""
        from sparkrun.proxy.discovery import discover_endpoints

        endpoints = discover_endpoints(
            cache_dir=str(populated_jobs_dir),
            check_health=False,
        )
        assert len(endpoints) == 2

    def test_discover_host_filter(self, populated_jobs_dir: Path):
        """Host filter limits discovered endpoints."""
        from sparkrun.proxy.discovery import discover_endpoints

        endpoints = discover_endpoints(
            host_filter=["192.168.11.13"],
            cache_dir=str(populated_jobs_dir),
            check_health=False,
        )
        assert len(endpoints) == 1
        assert endpoints[0].host == "192.168.11.13"

    def test_discover_port_fallback(self, jobs_dir: Path):
        """Missing port in metadata defaults to 8000."""
        from sparkrun.proxy.discovery import discover_endpoints

        meta = {
            "cluster_id": "sparkrun_aaaaaaaaaaaaaaa4_aaaaaaaaaaa4",
            "recipe": "test",
            "model": "test/model",
            "runtime": "vllm",
            "hosts": ["10.0.0.1"],
            # No port field
        }
        with open(jobs_dir / "noport.yaml", "w") as f:
            yaml.safe_dump(meta, f)

        endpoints = discover_endpoints(
            cache_dir=str(jobs_dir.parent),
            check_health=False,
        )
        assert len(endpoints) == 1
        assert endpoints[0].port == 8000

    def test_discover_served_model_name(self, populated_jobs_dir: Path):
        """served_model_name is extracted from metadata."""
        from sparkrun.proxy.discovery import discover_endpoints

        endpoints = discover_endpoints(
            cache_dir=str(populated_jobs_dir),
            check_health=False,
        )
        named = [ep for ep in endpoints if ep.served_model_name]
        assert len(named) == 1
        assert named[0].served_model_name == "my-qwen"

    def test_discover_empty_dir(self, tmp_path: Path):
        """Empty jobs dir returns empty list."""
        from sparkrun.proxy.discovery import discover_endpoints

        endpoints = discover_endpoints(
            cache_dir=str(tmp_path),
            check_health=False,
        )
        assert endpoints == []

    def test_discover_no_dir(self, tmp_path: Path):
        """Missing jobs dir returns empty list."""
        from sparkrun.proxy.discovery import discover_endpoints

        endpoints = discover_endpoints(
            cache_dir=str(tmp_path / "nonexistent"),
            check_health=False,
        )
        assert endpoints == []

    def test_health_check_success(self, populated_jobs_dir: Path):
        """Successful health check sets healthy=True and populates actual_models."""
        from sparkrun.proxy.discovery import discover_endpoints

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(
            {
                "data": [{"id": "Qwen/Qwen3-1.7B"}],
            }
        ).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("sparkrun.proxy.discovery.urllib.request.urlopen", return_value=mock_response):
            endpoints = discover_endpoints(
                cache_dir=str(populated_jobs_dir),
                check_health=True,
            )

        healthy = [ep for ep in endpoints if ep.healthy]
        assert len(healthy) == 2
        assert "Qwen/Qwen3-1.7B" in healthy[0].actual_models

    def test_api_key_threaded_into_endpoint(self, jobs_dir: Path):
        """api_key from job metadata flows into DiscoveredEndpoint."""
        from sparkrun.proxy.discovery import discover_endpoints

        meta = {
            "cluster_id": "sparkrun_apikey",
            "recipe": "vllm-auth",
            "model": "Org/Authed",
            "runtime": "vllm",
            "hosts": ["10.0.0.5"],
            "port": 8000,
            "api_key": "sk-upstream",
        }
        with open(jobs_dir / "apikey.yaml", "w") as f:
            yaml.safe_dump(meta, f)

        endpoints = discover_endpoints(
            cache_dir=str(jobs_dir.parent),
            check_health=False,
        )
        assert len(endpoints) == 1
        assert endpoints[0].api_key == "sk-upstream"

    def test_health_check_sends_bearer_header(self, jobs_dir: Path):
        """Health check uses Bearer auth when api_key is set on the endpoint."""
        from sparkrun.proxy.discovery import discover_endpoints

        meta = {
            "cluster_id": "sparkrun_authed",
            "recipe": "vllm-auth",
            "model": "Org/Authed",
            "runtime": "vllm",
            "hosts": ["10.0.0.7"],
            "port": 8000,
            "api_key": "sk-bearer",
        }
        with open(jobs_dir / "authed.yaml", "w") as f:
            yaml.safe_dump(meta, f)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"data": [{"id": "Org/Authed"}]}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        captured: list = []

        def _spy(req, timeout=None):
            captured.append(req)
            return mock_response

        with patch("sparkrun.proxy.discovery.urllib.request.urlopen", side_effect=_spy):
            endpoints = discover_endpoints(
                cache_dir=str(jobs_dir.parent),
                check_health=True,
            )

        healthy = [ep for ep in endpoints if ep.healthy]
        assert len(healthy) == 1
        assert captured, "urlopen was not called"
        # urllib normalises header keys: Authorization → "Authorization"
        auth_value = captured[0].get_header("Authorization")
        assert auth_value == "Bearer sk-bearer"

    def test_health_check_no_auth_when_no_api_key(self, populated_jobs_dir: Path):
        """No Authorization header is sent when endpoint has no api_key."""
        from sparkrun.proxy.discovery import discover_endpoints

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"data": []}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        captured: list = []

        def _spy(req, timeout=None):
            captured.append(req)
            return mock_response

        with patch("sparkrun.proxy.discovery.urllib.request.urlopen", side_effect=_spy):
            discover_endpoints(
                cache_dir=str(populated_jobs_dir),
                check_health=True,
            )

        for req in captured:
            assert req.get_header("Authorization") is None

    def test_health_check_failure(self, populated_jobs_dir: Path):
        """Failed health check sets healthy=False."""
        import urllib.error

        from sparkrun.proxy.discovery import discover_endpoints

        with patch(
            "sparkrun.proxy.discovery.urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            endpoints = discover_endpoints(
                cache_dir=str(populated_jobs_dir),
                check_health=True,
            )

        healthy = [ep for ep in endpoints if ep.healthy]
        assert len(healthy) == 0

    def test_dedup_by_identity(self, jobs_dir: Path):
        """Endpoints on different IPs serving same models are deduplicated."""
        from sparkrun.proxy.discovery import discover_endpoints

        # Two metadata files for the same server on different network interfaces.
        # Neither carries ib_ip_map/mgmt_ip_map, so they get different host:port
        # keys and rely on identity dedup after health checks.
        meta_old = {
            "cluster_id": "sparkrun_old",
            "recipe": "qwen3-sglang",
            "model": "Qwen/Qwen3.5-35B",
            "runtime": "sglang",
            "hosts": ["192.168.11.14"],
            "port": 8000,
            "tensor_parallel": 1,
        }
        meta_new = {
            "cluster_id": "sparkrun_new",
            "recipe": "qwen3-sglang",
            "model": "Qwen/Qwen3.5-35B",
            "runtime": "sglang",
            "hosts": ["10.24.11.14"],
            "port": 8000,
            "tensor_parallel": 1,
        }

        import time

        # Older metadata file
        with open(jobs_dir / "old.yaml", "w") as f:
            yaml.safe_dump(meta_old, f)
        time.sleep(0.05)
        # Newer metadata file
        with open(jobs_dir / "new.yaml", "w") as f:
            yaml.safe_dump(meta_new, f)

        cache_dir = str(jobs_dir.parent)

        # Mock health checks — both return same models
        models_response = json.dumps(
            {
                "data": [{"id": "qwen3.5-35b"}],
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = models_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch(
            "sparkrun.proxy.discovery.urllib.request.urlopen",
            return_value=mock_resp,
        ):
            endpoints = discover_endpoints(cache_dir=cache_dir, check_health=True)

        # Should be deduplicated to 1 endpoint (same models on same port)
        assert len(endpoints) == 1
        # Newest metadata file wins
        assert endpoints[0].host == "10.24.11.14"

    def test_dedup_ib_to_mgmt_normalization(self, jobs_dir: Path):
        """IB IPs are normalised to management IPs via ib_ip_map."""
        from sparkrun.proxy.discovery import discover_endpoints

        # Stale metadata with IB IP as host (no maps — old format)
        meta_stale = {
            "cluster_id": "sparkrun_stale",
            "recipe": "old-recipe",
            "model": "Qwen/Qwen3-1.7B",
            "runtime": "vllm",
            "hosts": ["192.168.11.13"],
            "port": 8000,
            "tensor_parallel": 1,
        }
        # Current metadata with mgmt IP and ib_ip_map
        meta_current = {
            "cluster_id": "sparkrun_current",
            "recipe": "new-recipe",
            "model": "Qwen/Qwen3.5-0.8B",
            "runtime": "sglang",
            "hosts": ["10.24.11.13"],
            "port": 8000,
            "tensor_parallel": 2,
            "ib_ip_map": {"10.24.11.13": "192.168.11.13"},
            "mgmt_ip_map": {"10.24.11.13": "10.24.11.13"},
        }

        import time

        with open(jobs_dir / "stale.yaml", "w") as f:
            yaml.safe_dump(meta_stale, f)
        time.sleep(0.05)
        with open(jobs_dir / "current.yaml", "w") as f:
            yaml.safe_dump(meta_current, f)

        cache_dir = str(jobs_dir.parent)

        endpoints = discover_endpoints(cache_dir=cache_dir, check_health=False)

        # IB IP 192.168.11.13 normalised to 10.24.11.13 via ib_to_mgmt map,
        # so both entries share the same host:port key. Newest wins.
        assert len(endpoints) == 1
        assert endpoints[0].host == "10.24.11.13"
        assert endpoints[0].recipe_name == "new-recipe"
        assert endpoints[0].runtime == "sglang"

    def test_discover_live_uses_running_containers(self, tmp_path: Path):
        """Live discovery cross-references api.list_jobs with api.status."""
        from sparkrun.api import JobInfo
        from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload
        from sparkrun.proxy.discovery import discover_endpoints

        running_meta = {
            "cluster_id": "sparkrun_bbbbbbbbbbbbbbb1_bbbbbbbbbbb1",
            "recipe": "qwen3.5-0.8b-bf16-sglang",
            "model": "Qwen/Qwen3.5-0.8B",
            "runtime": "sglang",
            "hosts": ["10.24.11.13", "10.24.11.14"],
            "port": 8000,
            "tensor_parallel": 2,
            "served_model_name": "qwen3.5-0.8b",
            "mgmt_ip_map": {"10.24.11.13": "10.24.11.13", "10.24.11.14": "10.24.11.14"},
        }
        stale_meta = {
            "cluster_id": "sparkrun_old999",
            "recipe": "nemotron3-super-120b-nvfp4-trtllm",
            "model": "nvidia/NVIDIA-Nemotron-3-Super-120B",
            "runtime": "trtllm",
            "hosts": ["10.24.11.13"],
            "port": 8000,
            "tensor_parallel": 1,
        }

        jobs = [
            JobInfo(
                cluster_id=running_meta["cluster_id"],
                recipe=running_meta["recipe"],
                runtime=running_meta["runtime"],
                hosts=tuple(running_meta["hosts"]),
                metadata=running_meta,
            ),
            JobInfo(
                cluster_id=stale_meta["cluster_id"],
                recipe=stale_meta["recipe"],
                runtime=stale_meta["runtime"],
                hosts=tuple(stale_meta["hosts"]),
                metadata=stale_meta,
            ),
        ]

        snapshot = ClusterStatus(
            hosts=(
                HostOccupancy(
                    host="10.24.11.13",
                    workloads=(RunningWorkload(cluster_id=running_meta["cluster_id"]),),
                ),
                HostOccupancy(
                    host="10.24.11.14",
                    workloads=(RunningWorkload(cluster_id=running_meta["cluster_id"]),),
                ),
            ),
        )

        with (
            patch("sparkrun.proxy.discovery.api.list_jobs", return_value=jobs),
            patch("sparkrun.proxy.discovery.api.status", return_value=snapshot),
        ):
            endpoints = discover_endpoints(
                check_health=False,
                host_list=["10.24.11.13", "10.24.11.14"],
                ssh_kwargs={"ssh_user": "drew"},
            )

        # Only the actually-running cluster should appear
        assert len(endpoints) == 1
        ep = endpoints[0]
        assert ep.cluster_id == "sparkrun_bbbbbbbbbbbbbbb1_bbbbbbbbbbb1"
        assert ep.recipe_name == "qwen3.5-0.8b-bf16-sglang"
        assert ep.runtime == "sglang"
        assert ep.host == "10.24.11.13"
        assert ep.tensor_parallel == 2
        assert ep.served_model_name == "qwen3.5-0.8b"

    def test_discover_live_fallback_on_failure(self, populated_jobs_dir: Path):
        """Falls back to metadata-only discovery when api.status fails."""
        from sparkrun.proxy.discovery import discover_endpoints

        with patch(
            "sparkrun.proxy.discovery.api.status",
            side_effect=RuntimeError("SSH failed"),
        ):
            endpoints = discover_endpoints(
                cache_dir=str(populated_jobs_dir),
                check_health=False,
                host_list=["10.24.11.13"],
                ssh_kwargs={"ssh_user": "drew"},
            )

        # Should still find endpoints via metadata-only fallback
        assert len(endpoints) > 0


# =====================================================================
# Tests: ProxyConfig
# =====================================================================


class TestProxyConfig:
    """Test ProxyConfig load/save and alias management."""

    def test_defaults_when_missing(self, proxy_config_path: Path):
        """Default values when config file doesn't exist."""
        from sparkrun.proxy.config import ProxyConfig

        cfg = ProxyConfig(proxy_config_path)
        assert cfg.port == 4000
        assert cfg.host == "0.0.0.0"
        assert cfg.master_key is None
        assert cfg.auto_discover is True
        assert cfg.discover_interval == 30
        assert cfg.aliases == {}

    def test_save_and_load(self, proxy_config_path: Path):
        """Config round-trips through save/load."""
        from sparkrun.proxy.config import ProxyConfig

        cfg = ProxyConfig(proxy_config_path)
        cfg.set_proxy(port=5000, host="127.0.0.1")
        cfg.add_alias("gpt-4", "Qwen/Qwen3-1.7B")
        cfg.save()

        cfg2 = ProxyConfig(proxy_config_path)
        assert cfg2.port == 5000
        assert cfg2.host == "127.0.0.1"
        assert cfg2.aliases == {"gpt-4": "Qwen/Qwen3-1.7B"}

    def test_alias_crud(self, proxy_config_path: Path):
        """Add, list, and remove aliases."""
        from sparkrun.proxy.config import ProxyConfig

        cfg = ProxyConfig(proxy_config_path)

        # Add
        cfg.add_alias("alias1", "model1")
        cfg.add_alias("alias2", "model2")
        assert len(cfg.list_aliases()) == 2

        # Update
        cfg.add_alias("alias1", "model1-updated")
        assert cfg.aliases["alias1"] == "model1-updated"

        # Remove
        assert cfg.remove_alias("alias1") is True
        assert cfg.remove_alias("nonexistent") is False
        assert len(cfg.list_aliases()) == 1

    def test_default_recipes_empty(self, proxy_config_path: Path):
        """default_recipes returns empty dict when not configured."""
        from sparkrun.proxy.config import ProxyConfig

        cfg = ProxyConfig(proxy_config_path)
        assert cfg.default_recipes == {}


# =====================================================================
# Tests: Engine — config generation
# =====================================================================


class TestEngineConfig:
    """Test litellm config generation."""

    def test_build_config_basic(self):
        """Build litellm config from endpoints."""
        from sparkrun.proxy.discovery import DiscoveredEndpoint
        from sparkrun.proxy.engine import build_litellm_config

        endpoints = [
            DiscoveredEndpoint(
                cluster_id="sparkrun_abc",
                model="Qwen/Qwen3-1.7B",
                served_model_name=None,
                runtime="vllm",
                host="192.168.11.13",
                port=8000,
                healthy=True,
                actual_models=["Qwen/Qwen3-1.7B"],
                recipe_name="qwen3-1.7b-vllm",
            ),
        ]

        config = build_litellm_config(endpoints, master_key="test-key")

        assert len(config["model_list"]) == 1
        entry = config["model_list"][0]
        assert entry["model_name"] == "Qwen/Qwen3-1.7B"
        assert entry["litellm_params"]["model"] == "openai/Qwen/Qwen3-1.7B"
        assert entry["litellm_params"]["api_base"] == "http://192.168.11.13:8000/v1"
        assert config["general_settings"]["master_key"] == "test-key"
        assert config["litellm_settings"]["drop_params"] is True

    def test_build_config_no_aliases_in_config(self):
        """Aliases are no longer baked into litellm config (applied via API)."""
        from sparkrun.proxy.discovery import DiscoveredEndpoint
        from sparkrun.proxy.engine import build_litellm_config

        endpoints = [
            DiscoveredEndpoint(
                cluster_id="sparkrun_abc",
                model="Qwen/Qwen3-1.7B",
                served_model_name=None,
                runtime="vllm",
                host="10.0.0.1",
                port=8000,
                healthy=True,
                actual_models=["Qwen/Qwen3-1.7B"],
            ),
        ]

        config = build_litellm_config(endpoints)

        assert "router_settings" not in config

    def test_build_config_skips_unhealthy(self):
        """Unhealthy endpoints are excluded from config."""
        from sparkrun.proxy.discovery import DiscoveredEndpoint
        from sparkrun.proxy.engine import build_litellm_config

        endpoints = [
            DiscoveredEndpoint(
                cluster_id="sparkrun_abc",
                model="model-a",
                served_model_name=None,
                runtime="vllm",
                host="10.0.0.1",
                port=8000,
                healthy=True,
                actual_models=["model-a"],
            ),
            DiscoveredEndpoint(
                cluster_id="sparkrun_def",
                model="model-b",
                served_model_name=None,
                runtime="vllm",
                host="10.0.0.2",
                port=8000,
                healthy=False,
                actual_models=[],
            ),
        ]

        config = build_litellm_config(endpoints)
        assert len(config["model_list"]) == 1
        assert config["model_list"][0]["model_name"] == "model-a"

    def test_build_config_uses_endpoint_api_key(self):
        """Endpoint api_key flows through to litellm_params; absent → 'not-needed'."""
        from sparkrun.proxy.discovery import DiscoveredEndpoint
        from sparkrun.proxy.engine import build_litellm_config

        endpoints = [
            DiscoveredEndpoint(
                cluster_id="sparkrun_auth",
                model="Org/Authed",
                served_model_name=None,
                runtime="vllm",
                host="10.0.0.5",
                port=8000,
                healthy=True,
                actual_models=["Org/Authed"],
                api_key="sk-upstream",
            ),
            DiscoveredEndpoint(
                cluster_id="sparkrun_open",
                model="Org/Open",
                served_model_name=None,
                runtime="vllm",
                host="10.0.0.6",
                port=8000,
                healthy=True,
                actual_models=["Org/Open"],
            ),
        ]

        config = build_litellm_config(endpoints)
        keys = {m["model_name"]: m["litellm_params"]["api_key"] for m in config["model_list"]}
        assert keys == {"Org/Authed": "sk-upstream", "Org/Open": "not-needed"}

    def test_build_config_deduplicates(self):
        """Same model on same host:port is deduplicated."""
        from sparkrun.proxy.discovery import DiscoveredEndpoint
        from sparkrun.proxy.engine import build_litellm_config

        ep = DiscoveredEndpoint(
            cluster_id="sparkrun_abc",
            model="model-a",
            served_model_name=None,
            runtime="vllm",
            host="10.0.0.1",
            port=8000,
            healthy=True,
            actual_models=["model-a"],
        )

        config = build_litellm_config([ep, ep])
        assert len(config["model_list"]) == 1

    def test_write_config(self, tmp_path: Path):
        """Config dict is written to YAML file."""
        from sparkrun.proxy.engine import write_config

        config_dict = {"model_list": [], "general_settings": {"master_key": "test"}}
        path = write_config(config_dict, config_path=tmp_path / "test_config.yaml")

        assert path.exists()
        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["general_settings"]["master_key"] == "test"


# =====================================================================
# Tests: Engine — subprocess lifecycle
# =====================================================================


class TestEngineLifecycle:
    """Test ProxyEngine start/stop/is_running."""

    def test_start_dry_run(self, state_dir: Path):
        """Dry-run start returns 0 without launching."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)
        with patch("shutil.which", return_value="/usr/bin/uvx"):
            rc = engine.start(
                config_path=state_dir / "fake.yaml",
                dry_run=True,
            )
        assert rc == 0

    def test_start_no_uvx(self, state_dir: Path):
        """Start fails gracefully when uvx is not found."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)
        with patch("shutil.which", return_value=None):
            rc = engine.start(config_path=state_dir / "fake.yaml")
        assert rc == 1

    def test_start_daemonized(self, state_dir: Path):
        """Daemonized start saves PID to state file."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None  # Process still running

        with patch("shutil.which", return_value="/usr/bin/uvx"), patch("subprocess.Popen", return_value=mock_proc), patch("time.sleep"):
            rc = engine.start(config_path=state_dir / "fake.yaml")

        assert rc == 0
        assert engine._read_pid() == 12345

    def test_stop_sends_sigterm(self, state_dir: Path):
        """Stop sends SIGTERM and clears state."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)
        engine._save_state(99999)

        with patch("os.kill") as mock_kill:
            result = engine.stop()

        assert result is True
        mock_kill.assert_called_once_with(99999, signal.SIGTERM)
        assert not engine.state_file.exists()

    def test_stop_stale_pid(self, state_dir: Path):
        """Stop handles stale PID gracefully."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)
        engine._save_state(99999)

        with patch("os.kill", side_effect=ProcessLookupError):
            result = engine.stop()

        assert result is False
        assert not engine.state_file.exists()

    def test_stop_no_state(self, state_dir: Path):
        """Stop with no state file returns False."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)
        assert engine.stop() is False

    def test_is_running_true(self, state_dir: Path):
        """is_running returns True when PID is alive."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)
        engine._save_state(os.getpid())  # Current process is alive

        assert engine.is_running() is True

    def test_is_running_false_no_state(self, state_dir: Path):
        """is_running returns False with no state file."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)
        assert engine.is_running() is False

    def test_get_state(self, state_dir: Path):
        """get_state returns saved state dict."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)
        engine._save_state(12345)

        state = engine.get_state()
        assert state is not None
        assert state["pid"] == 12345
        assert state["port"] == 4000

    def test_get_state_missing(self, state_dir: Path):
        """get_state returns None when no state file."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)
        assert engine.get_state() is None


# =====================================================================
# Tests: Engine — management API
# =====================================================================


class TestWaitForExit:
    """Test process-exit detection, which gates every proxy restart."""

    def test_zombie_child_counts_as_exited(self):
        """A dead-but-unreaped child must not read as still running.

        ``os.kill(pid, 0)`` succeeds on a zombie, so naive polling waits out
        the full timeout and then escalates to SIGKILL against a process
        that already exited — leaving the restart with no replacement.  The
        auto-discover daemon spawns each replacement proxy itself, so from
        its second restart onward the old proxy is exactly this case.
        """
        import subprocess
        import sys
        import time

        from sparkrun.proxy.engine import _wait_for_exit

        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        time.sleep(0.5)  # exited, but deliberately not reaped

        # Precondition: the zombie still answers a liveness signal.
        os.kill(proc.pid, 0)

        started = time.monotonic()
        assert _wait_for_exit(proc.pid, 10.0) is True
        assert time.monotonic() - started < 5.0, "should return promptly, not time out"

        proc.returncode = 0  # already reaped by _wait_for_exit

    def test_live_process_reported_as_running(self):
        """A genuinely running process must not be reported as exited."""
        import subprocess
        import sys

        from sparkrun.proxy.engine import _wait_for_exit

        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            assert _wait_for_exit(proc.pid, 1.0) is False
        finally:
            proc.kill()
            proc.wait()


class TestEngineModelQueryAPI:
    """Test the read-only management API client methods.

    Only queries remain: LiteLLM's mutation endpoints (``/model/new``,
    ``/model/delete``) require a DB-backed model store and answer
    ``500 No DB Connected`` against a sparkrun-launched proxy, so model
    changes go through the config file instead.
    """

    def test_list_models_via_api(self, state_dir: Path):
        """list_models_via_api parses response."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)

        response_data = {
            "data": [
                {"model_name": "model-a"},
                {"model_name": "model-b"},
            ]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            models = engine.list_models_via_api()

        assert len(models) == 2
        assert models[0]["model_name"] == "model-a"

    def test_list_models_api_failure(self, state_dir: Path):
        """list_models_via_api returns empty list on failure."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)

        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            models = engine.list_models_via_api()

        assert models == []

    def test_mutation_endpoints_are_gone(self, state_dir: Path):
        """The DB-dependent mutators must not come back as silent no-ops.

        They returned False/0 on failure, which callers read as "nothing to
        do" — the exact shape of the bug this replaced.
        """
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)
        for dead in ("add_model_via_api", "remove_model_via_api", "add_alias_via_api", "remove_alias_via_api"):
            assert not hasattr(engine, dead), "%s should have been removed" % dead


def _ep(model: str, host: str = "10.0.0.1", port: int = 8000, api_key: str | None = None):
    """Build a healthy DiscoveredEndpoint for the sync tests."""
    from sparkrun.proxy.discovery import DiscoveredEndpoint

    return DiscoveredEndpoint(
        cluster_id="sparkrun_%s" % model.replace("/", "_"),
        model=model,
        served_model_name=None,
        runtime="vllm",
        host=host,
        port=port,
        healthy=True,
        actual_models=[model],
        api_key=api_key,
    )


class TestApplyDesiredState:
    """Test config-regeneration + restart, the only way models change."""

    def _engine(self, state_dir: Path, running: bool = True):
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir, master_key="sk-test")
        engine._test_running = running
        return engine

    def _seed(self, engine, endpoints, aliases=None):
        """Write a config representing the proxy's current state."""
        from sparkrun.proxy.engine import build_litellm_config, write_config

        write_config(
            build_litellm_config(endpoints, master_key=engine.master_key, aliases=aliases),
            engine.config_path,
        )

    def test_noop_when_already_in_sync(self, state_dir: Path):
        """An unchanged endpoint set must not rewrite or restart anything."""
        engine = self._engine(state_dir)
        ep = _ep("test/model")
        self._seed(engine, [ep])
        mtime_before = engine.config_path.stat().st_mtime_ns

        with (
            patch.object(type(engine), "is_running", return_value=True),
            patch.object(engine, "_restart_proxy") as mock_restart,
        ):
            added, removed = engine.apply_desired_state([ep])

        assert (added, removed) == (0, 0)
        mock_restart.assert_not_called()
        assert engine.config_path.stat().st_mtime_ns == mtime_before

    def test_new_endpoint_rewrites_config_and_restarts(self, state_dir: Path):
        """A newly discovered endpoint lands in the config and restarts the proxy."""
        import yaml

        engine = self._engine(state_dir)
        old, new = _ep("old/model"), _ep("new/model", host="10.0.0.2")
        self._seed(engine, [old])

        with (
            patch.object(type(engine), "is_running", return_value=True),
            patch.object(engine, "_restart_proxy", return_value=4321) as mock_restart,
        ):
            added, removed = engine.apply_desired_state([old, new])

        assert (added, removed) == (1, 0)
        mock_restart.assert_called_once()
        written = yaml.safe_load(engine.config_path.read_text())
        names = {m["model_name"] for m in written["model_list"]}
        assert names == {"old/model", "new/model"}

    def test_vanished_endpoint_is_removed(self, state_dir: Path):
        """An endpoint that disappeared is dropped from the config."""
        import yaml

        engine = self._engine(state_dir)
        self._seed(engine, [_ep("old/model")])

        with (
            patch.object(type(engine), "is_running", return_value=True),
            patch.object(engine, "_restart_proxy", return_value=4321),
        ):
            added, removed = engine.apply_desired_state([])

        assert (added, removed) == (0, 1)
        written = yaml.safe_load(engine.config_path.read_text())
        assert written["model_list"] == []

    def test_not_running_writes_config_without_restart(self, state_dir: Path):
        """With no proxy running the config is still updated for next start."""
        engine = self._engine(state_dir, running=False)
        self._seed(engine, [])

        with (
            patch.object(type(engine), "is_running", return_value=False),
            patch.object(engine, "_restart_proxy") as mock_restart,
        ):
            added, removed = engine.apply_desired_state([_ep("test/model")])

        assert (added, removed) == (1, 0)
        mock_restart.assert_not_called()

    def test_restart_false_suppresses_restart(self, state_dir: Path):
        """restart=False updates the config but leaves the process alone."""
        engine = self._engine(state_dir)
        self._seed(engine, [])

        with (
            patch.object(type(engine), "is_running", return_value=True),
            patch.object(engine, "_restart_proxy") as mock_restart,
        ):
            added, _removed = engine.apply_desired_state([_ep("test/model")], restart=False)

        assert added == 1
        mock_restart.assert_not_called()

    def test_restart_failure_raises(self, state_dir: Path):
        """A failed restart must surface, not be swallowed as a no-op."""
        from sparkrun.proxy.engine import ProxyRestartError

        engine = self._engine(state_dir)
        self._seed(engine, [])

        with (
            patch.object(type(engine), "is_running", return_value=True),
            patch.object(engine, "_restart_proxy", return_value=None),
            pytest.raises(ProxyRestartError),
        ):
            engine.apply_desired_state([_ep("test/model")])

    def test_restart_never_rotates_master_key(self, state_dir: Path):
        """general_settings is carried over, so a bare engine can't change auth.

        The CLI constructs ``ProxyEngine()`` bare in several places; if the
        regenerated config took that default master key, a sync would
        silently re-key the running proxy and lock out every client.
        """
        import yaml
        from sparkrun.proxy.engine import ProxyEngine, build_litellm_config, write_config

        write_config(build_litellm_config([], master_key="sk-REAL-RUNNING-KEY"), state_dir / "litellm_config.yaml")

        bare = ProxyEngine(state_dir=state_dir, master_key="sk-wrong-default")
        with (
            patch.object(type(bare), "is_running", return_value=True),
            patch.object(bare, "_restart_proxy", return_value=1),
        ):
            bare.apply_desired_state([_ep("test/model")])

        written = yaml.safe_load(bare.config_path.read_text())
        assert written["general_settings"]["master_key"] == "sk-REAL-RUNNING-KEY"

    def test_sync_models_preserves_configured_aliases(self, state_dir: Path):
        """sync_models without explicit aliases must not drop them."""
        import yaml

        engine = self._engine(state_dir)
        ep = _ep("Qwen/Qwen3-1.7B")
        self._seed(engine, [ep], aliases={"my-model": "Qwen/Qwen3-1.7B"})

        with (
            patch.object(type(engine), "is_running", return_value=True),
            patch.object(engine, "_restart_proxy", return_value=1),
            patch.object(engine, "_configured_aliases", return_value={"my-model": "Qwen/Qwen3-1.7B"}),
        ):
            engine.sync_models([ep, _ep("other/model", host="10.0.0.9")])

        written = yaml.safe_load(engine.config_path.read_text())
        assert "my-model" in {m["model_name"] for m in written["model_list"]}


class TestEngineAliases:
    """Test alias handling now that aliases live in the config file."""

    def test_alias_emitted_for_live_target(self):
        """An alias becomes an extra model_list entry on the target's backend."""
        from sparkrun.proxy.engine import build_litellm_config

        config = build_litellm_config([_ep("Qwen/Qwen3-1.7B")], aliases={"my-model": "Qwen/Qwen3-1.7B"})

        alias_entries = [m for m in config["model_list"] if m["model_name"] == "my-model"]
        assert len(alias_entries) == 1
        assert alias_entries[0]["litellm_params"]["model"] == "openai/Qwen/Qwen3-1.7B"
        assert alias_entries[0]["litellm_params"]["api_base"] == "http://10.0.0.1:8000/v1"

    def test_alias_skipped_when_target_absent(self):
        """An alias whose target has no backend is omitted, not emitted broken."""
        from sparkrun.proxy.engine import build_litellm_config

        config = build_litellm_config([_ep("other/model")], aliases={"my-model": "Qwen/Qwen3-1.7B"})

        assert "my-model" not in {m["model_name"] for m in config["model_list"]}

    def test_alias_spans_every_backend_of_target(self):
        """A tp-replicated model gets one alias entry per backend."""
        from sparkrun.proxy.engine import build_litellm_config

        eps = [_ep("Qwen/Qwen3-1.7B", host="10.0.0.1"), _ep("Qwen/Qwen3-1.7B", host="10.0.0.2")]
        config = build_litellm_config(eps, aliases={"my-model": "Qwen/Qwen3-1.7B"})

        alias_bases = {m["litellm_params"]["api_base"] for m in config["model_list"] if m["model_name"] == "my-model"}
        assert alias_bases == {"http://10.0.0.1:8000/v1", "http://10.0.0.2:8000/v1"}

    def test_endpoints_from_config_ignores_aliases(self, state_dir: Path):
        """Recovering endpoints from config must not resurrect aliases as models."""
        from sparkrun.proxy.engine import ProxyEngine, build_litellm_config, write_config

        engine = ProxyEngine(state_dir=state_dir)
        write_config(
            build_litellm_config([_ep("Qwen/Qwen3-1.7B")], aliases={"my-model": "Qwen/Qwen3-1.7B"}),
            engine.config_path,
        )

        recovered = engine._endpoints_from_config()

        assert [e.model for e in recovered] == ["Qwen/Qwen3-1.7B"]
        assert recovered[0].host == "10.0.0.1"
        assert recovered[0].port == 8000

    def test_sync_aliases_keeps_existing_models(self, state_dir: Path):
        """Adding an alias must not drop the models already being served."""
        import yaml
        from sparkrun.proxy.engine import ProxyEngine, build_litellm_config, write_config

        engine = ProxyEngine(state_dir=state_dir, master_key="sk-test")
        write_config(build_litellm_config([_ep("Qwen/Qwen3-1.7B")], master_key="sk-test"), engine.config_path)

        with (
            patch.object(type(engine), "is_running", return_value=True),
            patch.object(engine, "_restart_proxy", return_value=1),
        ):
            added, _removed = engine.sync_aliases({"my-model": "Qwen/Qwen3-1.7B"})

        assert added == 1
        written = yaml.safe_load(engine.config_path.read_text())
        names = {m["model_name"] for m in written["model_list"]}
        assert names == {"Qwen/Qwen3-1.7B", "my-model"}

    def test_sync_aliases_removes_dropped_alias(self, state_dir: Path):
        """Removing an alias from config removes its entry from the proxy."""
        import yaml
        from sparkrun.proxy.engine import ProxyEngine, build_litellm_config, write_config

        engine = ProxyEngine(state_dir=state_dir, master_key="sk-test")
        write_config(
            build_litellm_config([_ep("Qwen/Qwen3-1.7B")], master_key="sk-test", aliases={"my-model": "Qwen/Qwen3-1.7B"}),
            engine.config_path,
        )

        with (
            patch.object(type(engine), "is_running", return_value=True),
            patch.object(engine, "_restart_proxy", return_value=1),
        ):
            _added, removed = engine.sync_aliases({})

        assert removed == 1
        written = yaml.safe_load(engine.config_path.read_text())
        assert {m["model_name"] for m in written["model_list"]} == {"Qwen/Qwen3-1.7B"}


# =====================================================================
# Tests: launch_inference auto_port (port conflict avoidance)
# =====================================================================


class TestLaunchInferenceAutoPort:
    """Test auto_port behavior in launch_inference (used by proxy load and benchmark)."""

    def _make_mocks(self):
        """Create mock recipe, runtime, and config for launch_inference tests."""
        mock_recipe = MagicMock()
        mock_recipe.build_config_chain.return_value = {"port": 8000}
        mock_recipe.model = "test/model"
        mock_recipe.model_revision = None
        mock_recipe.name = "test"
        mock_recipe.env = {}
        mock_recipe.builder = None
        mock_recipe.mode = "solo"
        mock_recipe.max_nodes = None

        mock_runtime = MagicMock()
        mock_runtime.resolve_container.return_value = "test:latest"
        mock_runtime.is_delegating_runtime.return_value = True
        mock_runtime.generate_command.return_value = "serve cmd"
        mock_runtime.run.return_value = 0

        mock_config = MagicMock()
        mock_config.hf_cache_dir = "/tmp/cache"
        mock_config.cache_dir = "/tmp/cache"

        return mock_recipe, mock_runtime, mock_config

    def test_auto_port_calls_find_available_port(self):
        """auto_port=True uses find_available_port to resolve the port."""
        from sparkrun.core.launcher import launch_inference

        mock_recipe, mock_runtime, mock_config = self._make_mocks()

        with (
            patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={}),
            patch("sparkrun.orchestration.primitives.find_available_port", return_value=8000) as mock_fap,
            patch("sparkrun.orchestration.job_metadata.derive_cluster_id", return_value="test_id"),
            patch("sparkrun.orchestration.job_metadata.save_job_metadata"),
        ):
            result = launch_inference(
                recipe=mock_recipe,
                runtime=mock_runtime,
                host_list=["10.0.0.1"],
                overrides={},
                config=mock_config,
                is_solo=True,
                auto_port=True,
                dry_run=True,
            )

        assert result.serve_port == 8000
        mock_fap.assert_called_once_with("10.0.0.1", 8000, ssh_kwargs={}, dry_run=True)

    def test_auto_port_increments_when_occupied(self):
        """Returns incremented port when desired port is in use."""
        from sparkrun.core.launcher import launch_inference

        mock_recipe, mock_runtime, mock_config = self._make_mocks()

        with (
            patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={}),
            patch("sparkrun.orchestration.primitives.find_available_port", return_value=8002),
            patch("sparkrun.orchestration.job_metadata.derive_cluster_id", return_value="test_id"),
            patch("sparkrun.orchestration.job_metadata.save_job_metadata"),
        ):
            result = launch_inference(
                recipe=mock_recipe,
                runtime=mock_runtime,
                host_list=["10.0.0.1"],
                overrides={},
                config=mock_config,
                is_solo=True,
                auto_port=True,
                dry_run=True,
            )

        assert result.serve_port == 8002

    def test_auto_port_uses_recipe_default_port(self):
        """Reads desired port from recipe config chain."""
        from sparkrun.core.launcher import launch_inference

        mock_recipe, mock_runtime, mock_config = self._make_mocks()
        mock_recipe.build_config_chain.return_value = {"port": 9000}

        with (
            patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={}),
            patch("sparkrun.orchestration.primitives.find_available_port", return_value=9000) as mock_fap,
            patch("sparkrun.orchestration.job_metadata.derive_cluster_id", return_value="test_id"),
            patch("sparkrun.orchestration.job_metadata.save_job_metadata"),
        ):
            result = launch_inference(
                recipe=mock_recipe,
                runtime=mock_runtime,
                host_list=["10.0.0.1"],
                overrides={},
                config=mock_config,
                is_solo=True,
                auto_port=True,
                dry_run=True,
            )

        assert result.serve_port == 9000
        mock_fap.assert_called_once_with("10.0.0.1", 9000, ssh_kwargs={}, dry_run=True)

    def test_no_auto_port_uses_config_chain(self):
        """auto_port=False reads port from config chain without probing."""
        from sparkrun.core.launcher import launch_inference

        mock_recipe, mock_runtime, mock_config = self._make_mocks()
        mock_recipe.build_config_chain.return_value = {"port": 9000}

        with (
            patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={}),
            patch("sparkrun.orchestration.job_metadata.derive_cluster_id", return_value="test_id"),
            patch("sparkrun.orchestration.job_metadata.save_job_metadata"),
        ):
            result = launch_inference(
                recipe=mock_recipe,
                runtime=mock_runtime,
                host_list=["10.0.0.1"],
                overrides={},
                config=mock_config,
                is_solo=True,
                auto_port=False,
                dry_run=True,
            )

        assert result.serve_port == 9000

    def test_dry_run_passes_through(self):
        """dry_run flag is forwarded to find_available_port."""
        from sparkrun.core.launcher import launch_inference

        mock_recipe, mock_runtime, mock_config = self._make_mocks()

        with (
            patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={}),
            patch("sparkrun.orchestration.primitives.find_available_port", return_value=8000) as mock_fap,
            patch("sparkrun.orchestration.job_metadata.derive_cluster_id", return_value="test_id"),
            patch("sparkrun.orchestration.job_metadata.save_job_metadata"),
        ):
            launch_inference(
                recipe=mock_recipe,
                runtime=mock_runtime,
                host_list=["10.0.0.1"],
                overrides={},
                config=mock_config,
                is_solo=True,
                auto_port=True,
                dry_run=True,
            )

        mock_fap.assert_called_once_with("10.0.0.1", 8000, ssh_kwargs={}, dry_run=True)


class TestAutoPortDoesNotMoveIdentity:
    """``auto_port`` must not change the workload's *identity*.

    ``generate_intent_id`` hashes the port, and the ``auto_port`` probe
    rewrites ``overrides["port"]`` in place.  If the cluster_id were derived
    after that, a workload's identity would depend on which port happened to
    be free — and every lookup path (``stop`` / ``logs`` / ``--ensure`` /
    proxy discovery), which derives from the recipe's *requested* port, would
    fail to find the running job.  The proxy is the caller that sets
    ``auto_port=True``.
    """

    HOSTS = ["10.0.0.1"]

    def _recipe(self):
        from sparkrun.core.recipe import Recipe

        return Recipe(
            {
                "sparkrun_version": "2",
                "runtime": "vllm",
                "model": "test/m",
                "mode": "solo",
                "defaults": {"port": 8000},
            }
        )

    def _launch(self, recipe, overrides, *, available_port):
        from sparkrun.core.launcher import launch_inference

        mock_runtime = MagicMock()
        mock_runtime.resolve_container.return_value = "test:latest"
        mock_runtime.is_delegating_runtime.return_value = True
        mock_runtime.generate_command.return_value = "serve cmd"
        mock_runtime.run.return_value = 0

        mock_config = MagicMock()
        mock_config.hf_cache_dir = "/tmp/cache"
        mock_config.cache_dir = "/tmp/cache"

        with (
            patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={}),
            patch("sparkrun.orchestration.primitives.find_available_port", return_value=available_port),
            patch("sparkrun.orchestration.job_metadata.save_job_metadata"),
        ):
            return launch_inference(
                recipe=recipe,
                runtime=mock_runtime,
                host_list=list(self.HOSTS),
                overrides=overrides,
                config=mock_config,
                is_solo=True,
                auto_port=True,
                dry_run=True,
            )

    def test_cluster_id_reflects_requested_port_not_the_probed_one(self):
        """The port was taken and the probe moved to 8002 — the cluster_id must
        still be the one the lookup paths compute from the requested 8000."""
        from sparkrun.orchestration.job_metadata import derive_cluster_id

        recipe = self._recipe()
        overrides: dict = {}

        result = self._launch(recipe, overrides, available_port=8002)

        assert result.serve_port == 8002  # actually bound where the probe landed
        assert result.cluster_id == derive_cluster_id(recipe, self.HOSTS, overrides={})
        # ...and specifically NOT the identity the shifted port would produce.
        assert result.cluster_id != derive_cluster_id(recipe, self.HOSTS, overrides={"port": 8002})

    def test_identity_is_stable_across_differing_probe_results(self):
        """Two loads of the same recipe landing on different free ports share
        one identity, so the second replaces the first instead of leaking it."""
        first = self._launch(self._recipe(), {}, available_port=8000)
        second = self._launch(self._recipe(), {}, available_port=8003)

        assert first.cluster_id == second.cluster_id

    def test_actual_port_still_reaches_metadata_for_routing(self):
        """Identity is declarative (requested port); the bound port is factual
        and must still flow into overrides → job metadata → proxy routing."""
        overrides: dict = {}

        result = self._launch(self._recipe(), overrides, available_port=8002)

        assert overrides["port"] == 8002
        assert result.serve_port == 8002

    def test_explicitly_requested_port_still_distinguishes_workloads(self):
        """A deliberate ``--port`` is part of the identity — two intentional
        deployments on different ports stay distinct."""
        a = self._launch(self._recipe(), {"port": 8000}, available_port=8000)
        b = self._launch(self._recipe(), {"port": 9000}, available_port=9000)

        assert a.cluster_id != b.cluster_id


# =====================================================================
# Tests: CLI commands
# =====================================================================


class TestCLI:
    """Test Click CLI commands via CliRunner."""

    def test_proxy_help(self):
        """proxy --help shows subcommands."""
        from sparkrun.cli._proxy import proxy

        runner = CliRunner()
        result = runner.invoke(proxy, ["--help"])
        assert result.exit_code == 0
        assert "start" in result.output
        assert "stop" in result.output
        assert "start" in result.output

    def test_alias_list_empty(self, tmp_path: Path):
        """alias list shows message when empty."""
        from sparkrun.cli._proxy import proxy

        runner = CliRunner()
        with (
            patch("sparkrun.proxy.config.ProxyConfig.__init__", return_value=None),
            patch("sparkrun.proxy.config.ProxyConfig.aliases", new_callable=lambda: property(lambda s: {})),
        ):
            result = runner.invoke(proxy, ["alias", "list"])

        assert result.exit_code == 0
        assert "No aliases configured" in result.output

    def test_stop_not_running(self, state_dir: Path):
        """stop shows message when proxy isn't running."""
        from sparkrun.cli._proxy import proxy

        runner = CliRunner()
        with patch("sparkrun.proxy.engine.ProxyEngine.is_running", return_value=False):
            result = runner.invoke(proxy, ["stop"])

        assert result.exit_code == 0
        assert "No proxy is currently running" in result.output

    def test_status_no_state(self, state_dir: Path):
        """status shows message when no state exists."""
        from sparkrun.cli._proxy import proxy

        runner = CliRunner()
        with patch("sparkrun.proxy.engine.ProxyEngine.get_state", return_value=None):
            result = runner.invoke(proxy, ["status"])

        assert result.exit_code == 0
        assert "No proxy state found" in result.output

    def test_start_dry_run(self, tmp_path: Path):
        """start --dry-run shows what would be done."""
        from sparkrun.proxy.discovery import DiscoveredEndpoint
        from sparkrun.cli._proxy import proxy

        endpoints = [
            DiscoveredEndpoint(
                cluster_id="sparkrun_abc",
                model="test/model",
                served_model_name=None,
                runtime="vllm",
                host="10.0.0.1",
                port=8000,
                healthy=True,
                actual_models=["test/model"],
                recipe_name="test-recipe",
            ),
        ]

        runner = CliRunner()
        with (
            patch("sparkrun.proxy.discovery.discover_endpoints", return_value=endpoints),
            patch("sparkrun.proxy.config.ProxyConfig.__init__", return_value=None),
            patch("sparkrun.proxy.config.ProxyConfig.port", new_callable=lambda: property(lambda s: 4000)),
            patch("sparkrun.proxy.config.ProxyConfig.host", new_callable=lambda: property(lambda s: "0.0.0.0")),
            patch("sparkrun.proxy.config.ProxyConfig.host_configured", new_callable=lambda: property(lambda s: False)),
            patch("sparkrun.proxy.config.ProxyConfig.master_key", new_callable=lambda: property(lambda s: "sk-test")),
            patch("sparkrun.proxy.config.ProxyConfig.aliases", new_callable=lambda: property(lambda s: {})),
            patch("sparkrun.proxy.config.ProxyConfig.enable_ui", new_callable=lambda: property(lambda s: False)),
            patch("sparkrun.proxy.config.ProxyConfig.gateway", new_callable=lambda: property(lambda s: None)),
            patch("sparkrun.proxy.config.ProxyConfig.auto_discover", new_callable=lambda: property(lambda s: True)),
            patch(
                "sparkrun.proxy.config.ProxyConfig.discover_interval",
                new_callable=lambda: property(lambda s: 30),
            ),
        ):
            result = runner.invoke(proxy, ["start", "--dry-run"])

        assert result.exit_code == 0
        assert "dry-run" in result.output

    def test_models_not_running(self):
        """models shows message when proxy isn't running."""
        from sparkrun.cli._proxy import proxy

        runner = CliRunner()
        with patch("sparkrun.proxy.engine.ProxyEngine.is_running", return_value=False):
            result = runner.invoke(proxy, ["models"])

        assert result.exit_code == 0
        assert "not running" in result.output


# =====================================================================
# Tests: Auto-discover
# =====================================================================


class TestAutodiscover:
    """Test auto-discovery background process."""

    def test_start_autodiscover_writes_config(self, tmp_path: Path):
        """start_autodiscover writes config YAML and spawns a subprocess."""
        from sparkrun.proxy.engine import ProxyEngine

        state_dir = tmp_path / "proxy"
        engine = ProxyEngine(state_dir=state_dir)

        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_popen.return_value = mock_proc

            pid = engine.start_autodiscover(
                proxy_pid=9999,
                interval=60,
                host_list=["10.24.11.13", "10.24.11.14"],
                ssh_kwargs={"ssh_user": "drew"},
            )

        assert pid == 12345

        # Verify config file was written
        cfg_path = state_dir / "autodiscover.yaml"
        assert cfg_path.exists()
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["proxy_pid"] == 9999
        assert cfg["interval"] == 60
        assert cfg["host_list"] == ["10.24.11.13", "10.24.11.14"]
        assert cfg["ssh_kwargs"] == {"ssh_user": "drew"}

    def test_stop_autodiscover_sends_sigterm(self, tmp_path: Path):
        """stop_autodiscover sends SIGTERM to the auto-discover PID."""
        from sparkrun.proxy.engine import ProxyEngine

        state_dir = tmp_path / "proxy"
        state_dir.mkdir(parents=True)
        engine = ProxyEngine(state_dir=state_dir)

        # Save state with autodiscover PID
        engine._save_state(pid=100, autodiscover_pid=200)

        with patch("os.kill") as mock_kill:
            engine.stop_autodiscover()
            mock_kill.assert_called_once_with(200, signal.SIGTERM)

    def test_stop_kills_both_proxy_and_autodiscover(self, tmp_path: Path):
        """stop() kills both proxy and auto-discover PIDs."""
        from sparkrun.proxy.engine import ProxyEngine

        state_dir = tmp_path / "proxy"
        state_dir.mkdir(parents=True)
        engine = ProxyEngine(state_dir=state_dir)

        engine._save_state(pid=100, autodiscover_pid=200)

        with patch("os.kill") as mock_kill:
            result = engine.stop()

        assert result is True
        # Should have killed both: autodiscover (SIGTERM) and proxy (SIGTERM)
        assert mock_kill.call_count == 2
        mock_kill.assert_any_call(200, signal.SIGTERM)
        mock_kill.assert_any_call(100, signal.SIGTERM)

    def test_update_autodiscover_pid(self, tmp_path: Path):
        """update_autodiscover_pid records PID in state file."""
        from sparkrun.proxy.engine import ProxyEngine

        state_dir = tmp_path / "proxy"
        state_dir.mkdir(parents=True)
        engine = ProxyEngine(state_dir=state_dir)

        engine._save_state(pid=100)
        assert engine._read_autodiscover_pid() is None

        engine.update_autodiscover_pid(300)
        assert engine._read_autodiscover_pid() == 300

    def test_autodiscover_loop_exits_on_dead_proxy(self, tmp_path: Path):
        """run_autodiscover exits when proxy PID is gone."""
        from sparkrun.proxy.autodiscover import run_autodiscover

        cfg_path = tmp_path / "autodiscover.yaml"
        cfg = {
            "proxy_pid": 999999,  # non-existent PID
            "interval": 1,
            "proxy_port": 4000,
        }
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        # Should exit quickly since PID 999999 doesn't exist
        run_autodiscover(str(cfg_path))
