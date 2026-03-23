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
        "cluster_id": "sparkrun_abc123",
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
        "cluster_id": "sparkrun_def456",
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
    return Recipe.from_dict({
        "name": name,
        "model": model,
        "runtime": runtime,
        "container": "test-image:latest",
        "defaults": defaults or {},
    })


# =====================================================================
# Tests: generate_cluster_id with port/served_model_name
# =====================================================================

class TestGenerateClusterId:
    """Test generate_cluster_id() with port and served_model_name."""

    def test_backward_compat_no_overrides(self):
        """Omitting overrides produces same hash as original behavior."""
        from sparkrun.orchestration.job_metadata import generate_cluster_id

        recipe = _make_recipe()
        hosts = ["10.0.0.1", "10.0.0.2"]

        # No overrides, no defaults with port/served_name
        id1 = generate_cluster_id(recipe, hosts)
        id2 = generate_cluster_id(recipe, hosts, overrides=None)
        id3 = generate_cluster_id(recipe, hosts, overrides={})
        assert id1 == id2 == id3

    def test_different_ports_different_ids(self):
        """Same model on different ports produces different IDs."""
        from sparkrun.orchestration.job_metadata import generate_cluster_id

        recipe = _make_recipe()
        hosts = ["10.0.0.1"]

        id_8000 = generate_cluster_id(recipe, hosts, overrides={"port": 8000})
        id_9000 = generate_cluster_id(recipe, hosts, overrides={"port": 9000})
        assert id_8000 != id_9000

    def test_different_served_names_different_ids(self):
        """Same model with different served names produces different IDs."""
        from sparkrun.orchestration.job_metadata import generate_cluster_id

        recipe = _make_recipe()
        hosts = ["10.0.0.1"]

        id_a = generate_cluster_id(recipe, hosts, overrides={"served_model_name": "model-a"})
        id_b = generate_cluster_id(recipe, hosts, overrides={"served_model_name": "model-b"})
        assert id_a != id_b

    def test_port_from_recipe_defaults(self):
        """Port from recipe defaults is included in hash."""
        from sparkrun.orchestration.job_metadata import generate_cluster_id

        recipe_with_port = _make_recipe(defaults={"port": 8080})
        recipe_no_port = _make_recipe()
        hosts = ["10.0.0.1"]

        id_with = generate_cluster_id(recipe_with_port, hosts)
        id_without = generate_cluster_id(recipe_no_port, hosts)
        assert id_with != id_without

    def test_override_takes_precedence_over_default(self):
        """Override port takes precedence over recipe default."""
        from sparkrun.orchestration.job_metadata import generate_cluster_id

        recipe = _make_recipe(defaults={"port": 8000})
        hosts = ["10.0.0.1"]

        id_default = generate_cluster_id(recipe, hosts)
        id_override = generate_cluster_id(recipe, hosts, overrides={"port": 9000})
        assert id_default != id_override

    def test_same_override_matches_default(self):
        """When override equals default, ID matches no-override case."""
        from sparkrun.orchestration.job_metadata import generate_cluster_id

        recipe = _make_recipe(defaults={"port": 8000})
        hosts = ["10.0.0.1"]

        id_default = generate_cluster_id(recipe, hosts)
        id_same = generate_cluster_id(recipe, hosts, overrides={"port": 8000})
        assert id_default == id_same


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
            "sparkrun_test123", recipe, ["10.0.0.1"],
            overrides={"port": 9000},
            cache_dir=str(tmp_path),
        )

        meta = load_job_metadata("sparkrun_test123", cache_dir=str(tmp_path))
        assert meta is not None
        assert meta["port"] == 9000

    def test_served_model_name_persisted(self, tmp_path: Path):
        """served_model_name from overrides is saved in metadata."""
        from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

        recipe = _make_recipe()
        save_job_metadata(
            "sparkrun_test456", recipe, ["10.0.0.1"],
            overrides={"served_model_name": "my-model"},
            cache_dir=str(tmp_path),
        )

        meta = load_job_metadata("sparkrun_test456", cache_dir=str(tmp_path))
        assert meta is not None
        assert meta["served_model_name"] == "my-model"

    def test_port_from_recipe_defaults(self, tmp_path: Path):
        """Port from recipe defaults is saved when no override."""
        from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

        recipe = _make_recipe(defaults={"port": 8080})
        save_job_metadata(
            "sparkrun_test789", recipe, ["10.0.0.1"],
            cache_dir=str(tmp_path),
        )

        meta = load_job_metadata("sparkrun_test789", cache_dir=str(tmp_path))
        assert meta is not None
        assert meta["port"] == 8080

    def test_no_port_no_served_name(self, tmp_path: Path):
        """Missing port/served_name fields when not set anywhere."""
        from sparkrun.orchestration.job_metadata import save_job_metadata, load_job_metadata

        recipe = _make_recipe()
        save_job_metadata(
            "sparkrun_noport", recipe, ["10.0.0.1"],
            cache_dir=str(tmp_path),
        )

        meta = load_job_metadata("sparkrun_noport", cache_dir=str(tmp_path))
        assert meta is not None
        assert "port" not in meta
        assert "served_model_name" not in meta


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
            "cluster_id": "sparkrun_noport",
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
        mock_response.read.return_value = json.dumps({
            "data": [{"id": "Qwen/Qwen3-1.7B"}],
        }).encode()
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
        models_response = json.dumps({
            "data": [{"id": "qwen3.5-35b"}],
        }).encode()

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
        """Live discovery builds endpoints from query_cluster_status results."""
        from sparkrun.proxy.discovery import discover_endpoints
        from sparkrun.core.cluster_manager import ClusterGroup, ClusterStatusResult

        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        cache_dir = str(tmp_path)

        # Save metadata for the running cluster
        meta = {
            "cluster_id": "sparkrun_abc123",
            "recipe": "qwen3.5-0.8b-bf16-sglang",
            "model": "Qwen/Qwen3.5-0.8B",
            "runtime": "sglang",
            "hosts": ["10.24.11.13", "10.24.11.14"],
            "port": 8000,
            "tensor_parallel": 2,
            "served_model_name": "qwen3.5-0.8b",
            "mgmt_ip_map": {"10.24.11.13": "10.24.11.13", "10.24.11.14": "10.24.11.14"},
        }
        with open(jobs_dir / "abc123.yaml", "w") as f:
            yaml.safe_dump(meta, f)

        # Also save stale metadata on the same host:port (should be ignored)
        stale = {
            "cluster_id": "sparkrun_old999",
            "recipe": "nemotron3-super-120b-nvfp4-trtllm",
            "model": "nvidia/NVIDIA-Nemotron-3-Super-120B",
            "runtime": "trtllm",
            "hosts": ["10.24.11.13"],
            "port": 8000,
            "tensor_parallel": 1,
        }
        with open(jobs_dir / "old999.yaml", "w") as f:
            yaml.safe_dump(stale, f)

        # Mock query_cluster_status to return only the running cluster
        mock_result = ClusterStatusResult(
            groups={
                "sparkrun_abc123": ClusterGroup(
                    cluster_id="sparkrun_abc123",
                    members=[
                        ("10.24.11.13", "node_0", "Up 5 minutes", "sglang:latest"),
                        ("10.24.11.14", "node_1", "Up 5 minutes", "sglang:latest"),
                    ],
                    meta=meta,
                ),
            },
            solo_entries=[],
            errors={},
            idle_hosts=[],
            pending_ops=[],
            total_containers=2,
            host_count=2,
        )

        with patch(
            "sparkrun.core.cluster_manager.query_cluster_status",
            return_value=mock_result,
        ):
            endpoints = discover_endpoints(
                cache_dir=cache_dir,
                check_health=False,
                host_list=["10.24.11.13", "10.24.11.14"],
                ssh_kwargs={"ssh_user": "drew"},
            )

        # Only the actually-running cluster should appear
        assert len(endpoints) == 1
        ep = endpoints[0]
        assert ep.cluster_id == "sparkrun_abc123"
        assert ep.recipe_name == "qwen3.5-0.8b-bf16-sglang"
        assert ep.runtime == "sglang"
        assert ep.host == "10.24.11.13"
        assert ep.tensor_parallel == 2
        assert ep.served_model_name == "qwen3.5-0.8b"

    def test_discover_live_fallback_on_failure(self, populated_jobs_dir: Path):
        """Falls back to metadata discovery when live query fails."""
        from sparkrun.proxy.discovery import discover_endpoints

        with patch(
            "sparkrun.core.cluster_manager.query_cluster_status",
            side_effect=RuntimeError("SSH failed"),
        ):
            endpoints = discover_endpoints(
                cache_dir=str(populated_jobs_dir),
                check_health=False,
                host_list=["10.24.11.13"],
                ssh_kwargs={"ssh_user": "drew"},
            )

        # Should still find endpoints via metadata fallback
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

        with patch("shutil.which", return_value="/usr/bin/uvx"), \
             patch("subprocess.Popen", return_value=mock_proc), \
             patch("time.sleep"):
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

class TestEngineManagementAPI:
    """Test management API client methods."""

    def test_add_model_via_api(self, state_dir: Path):
        """add_model_via_api constructs correct request."""
        from sparkrun.proxy.discovery import DiscoveredEndpoint
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)

        ep = DiscoveredEndpoint(
            cluster_id="sparkrun_abc",
            model="test/model",
            served_model_name=None,
            runtime="vllm",
            host="10.0.0.1",
            port=8000,
            healthy=True,
            actual_models=["test/model"],
        )

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "ok"}'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            result = engine.add_model_via_api(ep)

        assert result is True
        # Verify the request was made
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.method == "POST"
        assert "/model/new" in req.full_url

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

    def test_remove_model_via_api(self, state_dir: Path):
        """remove_model_via_api sends POST /model/delete."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "ok"}'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            result = engine.remove_model_via_api("model-id-123")

        assert result is True
        req = mock_urlopen.call_args[0][0]
        assert req.method == "POST"
        assert "/model/delete" in req.full_url
        payload = json.loads(req.data)
        assert payload["id"] == "model-id-123"

    def test_remove_model_api_failure(self, state_dir: Path):
        """remove_model_via_api returns False on failure."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)

        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            result = engine.remove_model_via_api("model-id-123")

        assert result is False

    def test_sync_models_adds_new(self, state_dir: Path):
        """sync_models adds models not yet registered."""
        from sparkrun.proxy.discovery import DiscoveredEndpoint
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)

        ep = DiscoveredEndpoint(
            cluster_id="sparkrun_abc",
            model="test/model",
            served_model_name=None,
            runtime="vllm",
            host="10.0.0.1",
            port=8000,
            healthy=True,
            actual_models=["test/model"],
        )

        # No models registered yet
        with patch.object(engine, "list_models_via_api", return_value=[]), \
             patch.object(engine, "add_model_via_api", return_value=True) as mock_add:
            added, removed = engine.sync_models([ep])

        assert added == 1
        assert removed == 0
        mock_add.assert_called_once_with(ep)

    def test_sync_models_removes_stale(self, state_dir: Path):
        """sync_models removes models whose backends are gone."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)

        registered = [
            {
                "model_name": "old/model",
                "model_info": {"id": "old-id-123"},
                "litellm_params": {"api_base": "http://10.0.0.99:8000/v1"},
            },
        ]

        # No healthy endpoints — the old model should be removed
        with patch.object(engine, "list_models_via_api", return_value=registered), \
             patch.object(engine, "remove_model_via_api", return_value=True) as mock_rm:
            added, removed = engine.sync_models([])

        assert added == 0
        assert removed == 1
        mock_rm.assert_called_once_with("old-id-123")

    def test_sync_models_skips_healthy(self, state_dir: Path):
        """sync_models does not remove models with healthy backends."""
        from sparkrun.proxy.discovery import DiscoveredEndpoint
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)

        ep = DiscoveredEndpoint(
            cluster_id="sparkrun_abc",
            model="test/model",
            served_model_name=None,
            runtime="vllm",
            host="10.0.0.1",
            port=8000,
            healthy=True,
            actual_models=["test/model"],
        )

        registered = [
            {
                "model_name": "test/model",
                "model_info": {"id": "good-id"},
                "litellm_params": {"api_base": "http://10.0.0.1:8000/v1"},
            },
        ]

        with patch.object(engine, "list_models_via_api", return_value=registered), \
             patch.object(engine, "remove_model_via_api") as mock_rm, \
             patch.object(engine, "add_model_via_api") as mock_add:
            added, removed = engine.sync_models([ep])

        assert added == 0
        assert removed == 0
        mock_rm.assert_not_called()
        mock_add.assert_not_called()


# =====================================================================
# Tests: Engine — alias API
# =====================================================================

class TestEngineAliasAPI:
    """Test API-based alias management methods."""

    def test_add_alias_via_api(self, state_dir: Path):
        """add_alias_via_api finds target backends and registers alias."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)

        registered = [
            {
                "model_name": "Qwen/Qwen3-1.7B",
                "litellm_params": {
                    "model": "openai/Qwen/Qwen3-1.7B",
                    "api_base": "http://10.0.0.1:8000/v1",
                    "api_key": "not-needed",
                },
            },
        ]

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "ok"}'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch.object(engine, "list_models_via_api", return_value=registered), \
             patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            result = engine.add_alias_via_api("my-model", "Qwen/Qwen3-1.7B")

        assert result is True
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        import json
        body = json.loads(req.data)
        assert body["model_name"] == "my-model"
        assert body["litellm_params"]["api_base"] == "http://10.0.0.1:8000/v1"

    def test_add_alias_target_not_found(self, state_dir: Path):
        """add_alias_via_api returns False when target model is not registered."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)

        with patch.object(engine, "list_models_via_api", return_value=[]):
            result = engine.add_alias_via_api("my-model", "nonexistent/model")

        assert result is False

    def test_remove_alias_via_api(self, state_dir: Path):
        """remove_alias_via_api removes all entries with the alias name."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)

        registered = [
            {
                "model_name": "my-model",
                "model_info": {"id": "alias-id-1"},
                "litellm_params": {"api_base": "http://10.0.0.1:8000/v1"},
            },
            {
                "model_name": "real-model",
                "model_info": {"id": "real-id"},
                "litellm_params": {"api_base": "http://10.0.0.1:8000/v1"},
            },
        ]

        with patch.object(engine, "list_models_via_api", return_value=registered), \
             patch.object(engine, "remove_model_via_api", return_value=True) as mock_rm:
            removed = engine.remove_alias_via_api("my-model")

        assert removed == 1
        mock_rm.assert_called_once_with("alias-id-1")

    def test_sync_aliases_adds_missing(self, state_dir: Path):
        """sync_aliases adds aliases not yet registered."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)

        registered = [
            {
                "model_name": "Qwen/Qwen3-1.7B",
                "litellm_params": {
                    "model": "openai/Qwen/Qwen3-1.7B",
                    "api_base": "http://10.0.0.1:8000/v1",
                },
            },
        ]

        with patch.object(engine, "list_models_via_api", return_value=registered), \
             patch.object(engine, "add_alias_via_api", return_value=True) as mock_add:
            added, removed = engine.sync_aliases({"my-model": "Qwen/Qwen3-1.7B"})

        assert added == 1
        assert removed == 0
        mock_add.assert_called_once_with("my-model", "Qwen/Qwen3-1.7B")

    def test_sync_aliases_skips_existing(self, state_dir: Path):
        """sync_aliases does not re-add aliases already registered."""
        from sparkrun.proxy.engine import ProxyEngine

        engine = ProxyEngine(state_dir=state_dir)

        registered = [
            {
                "model_name": "Qwen/Qwen3-1.7B",
                "litellm_params": {
                    "model": "openai/Qwen/Qwen3-1.7B",
                    "api_base": "http://10.0.0.1:8000/v1",
                },
            },
            {
                "model_name": "my-model",
                "litellm_params": {
                    "model": "openai/Qwen/Qwen3-1.7B",
                    "api_base": "http://10.0.0.1:8000/v1",
                },
            },
        ]

        with patch.object(engine, "list_models_via_api", return_value=registered), \
             patch.object(engine, "add_alias_via_api") as mock_add:
            added, removed = engine.sync_aliases({"my-model": "Qwen/Qwen3-1.7B"})

        assert added == 0
        assert removed == 0
        mock_add.assert_not_called()


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

        with patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={}), \
             patch("sparkrun.orchestration.primitives.find_available_port", return_value=8000) as mock_fap, \
             patch("sparkrun.orchestration.job_metadata.generate_cluster_id", return_value="test_id"), \
             patch("sparkrun.orchestration.job_metadata.save_job_metadata"):
            result = launch_inference(
                recipe=mock_recipe, runtime=mock_runtime,
                host_list=["10.0.0.1"], overrides={}, config=mock_config,
                is_solo=True, auto_port=True, dry_run=True,
            )

        assert result.serve_port == 8000
        mock_fap.assert_called_once_with("10.0.0.1", 8000, ssh_kwargs={}, dry_run=True)

    def test_auto_port_increments_when_occupied(self):
        """Returns incremented port when desired port is in use."""
        from sparkrun.core.launcher import launch_inference

        mock_recipe, mock_runtime, mock_config = self._make_mocks()

        with patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={}), \
             patch("sparkrun.orchestration.primitives.find_available_port", return_value=8002), \
             patch("sparkrun.orchestration.job_metadata.generate_cluster_id", return_value="test_id"), \
             patch("sparkrun.orchestration.job_metadata.save_job_metadata"):
            result = launch_inference(
                recipe=mock_recipe, runtime=mock_runtime,
                host_list=["10.0.0.1"], overrides={}, config=mock_config,
                is_solo=True, auto_port=True, dry_run=True,
            )

        assert result.serve_port == 8002

    def test_auto_port_uses_recipe_default_port(self):
        """Reads desired port from recipe config chain."""
        from sparkrun.core.launcher import launch_inference

        mock_recipe, mock_runtime, mock_config = self._make_mocks()
        mock_recipe.build_config_chain.return_value = {"port": 9000}

        with patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={}), \
             patch("sparkrun.orchestration.primitives.find_available_port", return_value=9000) as mock_fap, \
             patch("sparkrun.orchestration.job_metadata.generate_cluster_id", return_value="test_id"), \
             patch("sparkrun.orchestration.job_metadata.save_job_metadata"):
            result = launch_inference(
                recipe=mock_recipe, runtime=mock_runtime,
                host_list=["10.0.0.1"], overrides={}, config=mock_config,
                is_solo=True, auto_port=True, dry_run=True,
            )

        assert result.serve_port == 9000
        mock_fap.assert_called_once_with("10.0.0.1", 9000, ssh_kwargs={}, dry_run=True)

    def test_no_auto_port_uses_config_chain(self):
        """auto_port=False reads port from config chain without probing."""
        from sparkrun.core.launcher import launch_inference

        mock_recipe, mock_runtime, mock_config = self._make_mocks()
        mock_recipe.build_config_chain.return_value = {"port": 9000}

        with patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={}), \
             patch("sparkrun.orchestration.job_metadata.generate_cluster_id", return_value="test_id"), \
             patch("sparkrun.orchestration.job_metadata.save_job_metadata"):
            result = launch_inference(
                recipe=mock_recipe, runtime=mock_runtime,
                host_list=["10.0.0.1"], overrides={}, config=mock_config,
                is_solo=True, auto_port=False, dry_run=True,
            )

        assert result.serve_port == 9000

    def test_dry_run_passes_through(self):
        """dry_run flag is forwarded to find_available_port."""
        from sparkrun.core.launcher import launch_inference

        mock_recipe, mock_runtime, mock_config = self._make_mocks()

        with patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={}), \
             patch("sparkrun.orchestration.primitives.find_available_port", return_value=8000) as mock_fap, \
             patch("sparkrun.orchestration.job_metadata.generate_cluster_id", return_value="test_id"), \
             patch("sparkrun.orchestration.job_metadata.save_job_metadata"):
            launch_inference(
                recipe=mock_recipe, runtime=mock_runtime,
                host_list=["10.0.0.1"], overrides={}, config=mock_config,
                is_solo=True, auto_port=True, dry_run=True,
            )

        mock_fap.assert_called_once_with("10.0.0.1", 8000, ssh_kwargs={}, dry_run=True)


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
        with patch("sparkrun.proxy.config.ProxyConfig.__init__", return_value=None), \
             patch("sparkrun.proxy.config.ProxyConfig.list_aliases", return_value=[]):
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
        with patch("sparkrun.proxy.discovery.discover_endpoints", return_value=endpoints), \
             patch("sparkrun.proxy.config.ProxyConfig.__init__", return_value=None), \
             patch("sparkrun.proxy.config.ProxyConfig.port", new_callable=lambda: property(lambda s: 4000)), \
             patch("sparkrun.proxy.config.ProxyConfig.host", new_callable=lambda: property(lambda s: "0.0.0.0")), \
             patch("sparkrun.proxy.config.ProxyConfig.master_key", new_callable=lambda: property(lambda s: "sk-test")), \
             patch("sparkrun.proxy.config.ProxyConfig.aliases", new_callable=lambda: property(lambda s: {})):
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
