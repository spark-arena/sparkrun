"""Tests for container image SHA pinning (P1) and long-term archival ref (P2).

Covers:
- resolve_image_sha helper unit tests
- BenchmarkRunState extras round-trip for SHA and longterm_ref
- BenchmarkResult.generate_metadata prefers persisted longterm_image_ref over
  live builder resolution
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from sparkrun.benchmarking.base import BenchmarkResult
from sparkrun.benchmarking.run_state import BenchmarkRunState
from sparkrun.orchestration.primitives import resolve_image_sha
from sparkrun.orchestration.ssh import RemoteResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_remote_result(returncode: int, stdout: str, host: str = "host1") -> RemoteResult:
    return RemoteResult(host=host, returncode=returncode, stdout=stdout, stderr="")


def _make_state(bid: str = "bench_aabbccddeeff") -> BenchmarkRunState:
    return BenchmarkRunState(
        benchmark_id=bid,
        cluster_id="cluster-abc",
        recipe_qualified_name="@registry/my-recipe",
        framework="llama-benchy",
        profile="default",
        base_args={"pp": [2048]},
        schedule=[{"depth": 0, "concurrency": 1}],
    )


# ---------------------------------------------------------------------------
# resolve_image_sha unit tests
# ---------------------------------------------------------------------------


def test_resolve_image_sha_returns_sha_on_success(monkeypatch):
    """Returns sha256:... string when docker inspect succeeds on the first host."""
    expected = "sha256:abcdef1234567890"
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.run_command_on_host",
        lambda host, cmd, **kw: _make_remote_result(0, expected + "\n"),
    )
    result = resolve_image_sha("myimage:latest", ["host1"])
    assert result == expected


def test_resolve_image_sha_returns_none_on_failure(monkeypatch):
    """Returns None when docker inspect exits non-zero on all hosts."""
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.run_command_on_host",
        lambda host, cmd, **kw: _make_remote_result(1, ""),
    )
    result = resolve_image_sha("myimage:latest", ["host1", "host2"])
    assert result is None


def test_resolve_image_sha_returns_none_dry_run(monkeypatch):
    """Returns None immediately on dry_run without calling run_command_on_host."""
    called = []
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.run_command_on_host",
        lambda *a, **kw: called.append(True) or _make_remote_result(0, "sha256:abc"),
    )
    result = resolve_image_sha("myimage:latest", ["host1"], dry_run=True)
    assert result is None
    assert not called, "run_command_on_host should not be called in dry_run mode"


def test_resolve_image_sha_returns_none_empty_hosts(monkeypatch):
    """Returns None when hosts list is empty."""
    called = []
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.run_command_on_host",
        lambda *a, **kw: called.append(True) or _make_remote_result(0, "sha256:abc"),
    )
    result = resolve_image_sha("myimage:latest", [])
    assert result is None
    assert not called


def test_resolve_image_sha_iterates_hosts(monkeypatch):
    """First host fails (rc!=0), second succeeds — second host's SHA is returned."""
    expected = "sha256:deadbeef"
    responses = {
        "host1": _make_remote_result(1, "", host="host1"),
        "host2": _make_remote_result(0, expected + "\n", host="host2"),
    }
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.run_command_on_host",
        lambda host, cmd, **kw: responses[host],
    )
    result = resolve_image_sha("myimage:latest", ["host1", "host2"])
    assert result == expected


def test_resolve_image_sha_rejects_non_sha256_output(monkeypatch):
    """Returns None when docker inspect output does not start with 'sha256:'."""
    monkeypatch.setattr(
        "sparkrun.orchestration.primitives.run_command_on_host",
        lambda host, cmd, **kw: _make_remote_result(0, "not-a-sha\n"),
    )
    result = resolve_image_sha("myimage:latest", ["host1"])
    assert result is None


def test_resolve_image_sha_skips_exception_host(monkeypatch):
    """Host raising an exception is skipped; next host is tried."""
    expected = "sha256:cafebabe"

    def _side_effect(host, cmd, **kw):
        if host == "host1":
            raise OSError("connection refused")
        return _make_remote_result(0, expected + "\n", host=host)

    monkeypatch.setattr("sparkrun.orchestration.primitives.run_command_on_host", _side_effect)
    result = resolve_image_sha("myimage:latest", ["host1", "host2"])
    assert result == expected


# ---------------------------------------------------------------------------
# BenchmarkRunState extras round-trip
# ---------------------------------------------------------------------------


def test_sha_pin_persists_in_state_extras(tmp_path: Path):
    """container_image_sha survives a save/load cycle in state.extras."""
    state = _make_state()
    state.extras["container_image_sha"] = "sha256:abc123"
    state.save(str(tmp_path))

    loaded = BenchmarkRunState.load(state.benchmark_id, str(tmp_path))
    assert loaded is not None
    assert loaded.extras.get("container_image_sha") == "sha256:abc123"


def test_longterm_ref_persists_in_state_extras(tmp_path: Path):
    """container_image_longterm_ref survives a save/load cycle in state.extras."""
    state = _make_state()
    state.extras["container_image_longterm_ref"] = "ghcr.io/x/y@sha256:def456"
    state.extras["container_image_longterm_pinned"] = True
    state.save(str(tmp_path))

    loaded = BenchmarkRunState.load(state.benchmark_id, str(tmp_path))
    assert loaded is not None
    assert loaded.extras.get("container_image_longterm_ref") == "ghcr.io/x/y@sha256:def456"
    assert loaded.extras.get("container_image_longterm_pinned") is True


# ---------------------------------------------------------------------------
# BenchmarkResult.generate_metadata — longterm_image_ref preference
# ---------------------------------------------------------------------------


def _make_minimal_recipe(container: str = "myimage:latest") -> MagicMock:
    """Minimal recipe mock for generate_metadata."""
    recipe = MagicMock()
    recipe.name = "my-recipe"
    recipe.qualified_name = "@registry/my-recipe"
    recipe.container = container
    recipe.model = "org/model"
    recipe.runtime = "vllm-distributed"
    recipe.metadata = {}
    recipe.model_revision = None
    recipe.source_registry = "registry"
    recipe.source_registry_url = "https://github.com/example/registry"
    recipe.export.return_value = "recipe-yaml-content"
    recipe.build_config_chain.return_value = {}
    return recipe


def _make_bench_result_with_launch(recipe: MagicMock, builder_return: tuple) -> BenchmarkResult:
    """BenchmarkResult with a launch_result whose builder returns builder_return."""
    builder = MagicMock()
    builder.resolve_long_term_image.return_value = builder_return

    launch_result = MagicMock()
    launch_result.recipe = recipe
    launch_result.overrides = {}
    launch_result.cluster_id = "cluster-abc"
    launch_result.host_list = ["host1"]
    launch_result.container_image = recipe.container
    launch_result.runtime_info = {}
    launch_result.builder = builder

    br = BenchmarkResult()
    br.launch_result = launch_result
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    br.start_time = now
    br.end_time = now
    br.framework = MagicMock()
    br.framework.framework_name = "llama-benchy"
    br.profile = "default"
    br.benchmark_args = {}
    return br


def test_generate_metadata_prefers_persisted_longterm_ref():
    """When longterm_image_ref is pre-set, generate_metadata uses it and does
    not call builder.resolve_long_term_image."""
    recipe = _make_minimal_recipe("myimage:latest")
    persisted_ref = "ghcr.io/x/y@sha256:persisted"
    live_ref = "ghcr.io/x/y@sha256:live-different"

    br = _make_bench_result_with_launch(recipe, (live_ref, True))
    br.longterm_image_ref = persisted_ref
    br.longterm_image_pinned = True

    meta = br.generate_metadata()

    assert meta["recipe"]["container"] == persisted_ref
    assert meta["recipe"]["container_pinned"] is True
    # Builder should NOT have been called since we had a persisted ref
    br.launch_result.builder.resolve_long_term_image.assert_not_called()


def test_generate_metadata_falls_back_to_builder_when_no_persisted_ref():
    """When longterm_image_ref is None, generate_metadata falls back to
    builder.resolve_long_term_image."""
    recipe = _make_minimal_recipe("myimage:latest")
    live_ref = "ghcr.io/x/y@sha256:ddd"

    br = _make_bench_result_with_launch(recipe, (live_ref, True))
    # longterm_image_ref is None by default

    meta = br.generate_metadata()

    assert meta["recipe"]["container"] == live_ref
    assert meta["recipe"]["container_pinned"] is True
    br.launch_result.builder.resolve_long_term_image.assert_called_once()


def test_generate_metadata_unpinned_builder_leaves_raw_container():
    """When builder returns pinned=False, recipe_container stays as-is."""
    recipe = _make_minimal_recipe("myimage:latest")

    br = _make_bench_result_with_launch(recipe, ("ghcr.io/x/y@sha256:ddd", False))
    meta = br.generate_metadata()

    # pinned=False from builder means we don't override recipe_container
    assert meta["recipe"]["container_pinned"] is False
    assert meta["recipe"]["container"] == recipe.container
