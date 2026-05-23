"""Tests for the sparkrun.api package skeleton (Task 5).

Covers:

- Public re-exports are reachable through ``sparkrun.api``.
- Data models have the expected fields and defaults.
- Error hierarchy is rooted at :class:`SparkrunError`.
- ``HostsUnreachable`` carries the hosts attribute.
- Resolve helpers translate underlying errors to typed API errors.
- ``sparkrun.api`` can be imported without pulling in ``click``.
"""

from __future__ import annotations

import sys

import pytest

import sparkrun.api as api


# --------------------------------------------------------------------------
# Public surface — re-exports reachable
# --------------------------------------------------------------------------


def test_public_re_exports():
    """Each name listed in api.__all__ must be reachable as an attribute."""
    for name in api.__all__:
        assert hasattr(api, name), "sparkrun.api missing %r" % name


def test_run_options_defaults():
    opts = api.RunOptions(recipe="my-recipe")
    assert opts.recipe == "my-recipe"
    assert opts.hosts is None
    assert opts.cluster is None
    assert opts.overrides == {}
    assert opts.solo is False
    assert opts.dry_run is False
    assert opts.follow is True
    assert opts.detached is True
    assert opts.trust is None
    assert opts.scheduler is None


def test_run_options_immutable():
    opts = api.RunOptions(recipe="x")
    with pytest.raises(Exception):  # FrozenInstanceError
        opts.recipe = "y"  # type: ignore[misc]


def test_run_options_overrides_isolated_per_instance():
    """The default_factory=dict for overrides is fresh per instance."""
    a = api.RunOptions(recipe="x")
    b = api.RunOptions(recipe="y")
    assert a.overrides is not b.overrides


def test_run_result_fields():
    result = api.RunResult(
        cluster_id="sparkrun_abc",
        host_list=("h1", "h2"),
        placement=None,
        scheduler="greedy",
        runtime="vllm",
        executor="docker",
        started_at=1234.5,
        dry_run=False,
        is_solo=False,
    )
    assert result.cluster_id == "sparkrun_abc"
    assert result.host_list == ("h1", "h2")
    assert result.metadata == {}


def test_stop_result_defaults():
    r = api.StopResult(cluster_id="x", hosts_targeted=("h1",), containers_removed=2)
    assert r.errors == ()


def test_log_line_defaults():
    line = api.LogLine(host="h", container="c", text="hello")
    assert line.stream == "stdout"
    assert line.timestamp is None


def test_job_info_defaults():
    j = api.JobInfo(cluster_id="x")
    assert j.recipe is None
    assert j.runtime is None
    assert j.hosts == ()
    assert j.metadata == {}


# --------------------------------------------------------------------------
# Error hierarchy
# --------------------------------------------------------------------------


def test_error_hierarchy_rooted_at_sparkrun_error():
    for cls in (
        api.InsufficientCapacity,
        api.LayoutRequired,
        api.RecipeNotFound,
        api.HostsUnreachable,
        api.JobNotFound,
        api.TrustRejected,
    ):
        assert issubclass(cls, api.SparkrunError)


def test_sparkrun_error_inherits_from_exception():
    assert issubclass(api.SparkrunError, Exception)


def test_hosts_unreachable_carries_hosts_tuple():
    e = api.HostsUnreachable("dead", hosts=["a", "b"])
    assert e.hosts == ("a", "b")
    assert "dead" in str(e)


def test_hosts_unreachable_empty_default():
    e = api.HostsUnreachable("no hosts")
    assert e.hosts == ()


def test_errors_are_catchable_as_sparkrun_error():
    """Callers should be able to use one generic except clause."""
    try:
        raise api.InsufficientCapacity("too few")
    except api.SparkrunError as e:
        assert "too few" in str(e)


# --------------------------------------------------------------------------
# Resolve helpers
# --------------------------------------------------------------------------


def test_resolve_cluster_synthesizes_anonymous_for_hosts_only():
    """`hosts_input` alone synthesizes an anonymous ClusterDefinition (name='')."""
    from sparkrun.api._resolve import resolve_cluster

    cluster = resolve_cluster(None, ["a", "b"])
    assert cluster.name == ""
    assert cluster.hosts == ["a", "b"]
    assert cluster.hosts_hardware == {}


def test_resolve_cluster_returns_loaded_cluster_unchanged():
    """Pre-loaded ClusterDefinition is returned as-is when no hosts override."""
    from sparkrun.api._resolve import resolve_cluster
    from sparkrun.core.cluster_manager import ClusterDefinition

    cluster = ClusterDefinition(name="c", hosts=["c1", "c2"])
    assert resolve_cluster(cluster) is cluster


def test_resolve_cluster_hosts_override_definition_hosts():
    """Explicit hosts_input overrides the cluster's host list (other fields preserved)."""
    from sparkrun.api._resolve import resolve_cluster
    from sparkrun.core.cluster_manager import ClusterDefinition

    cluster = ClusterDefinition(name="c", hosts=["c1"], user="alice")
    resolved = resolve_cluster(cluster, ["override-host"])
    assert resolved.hosts == ["override-host"]
    assert resolved.name == "c"
    assert resolved.user == "alice"


def test_resolve_cluster_raises_when_no_source():
    """No cluster, no hosts, no defaults → HostsUnreachable."""
    from sparkrun.api._resolve import resolve_cluster

    with pytest.raises(api.HostsUnreachable):
        resolve_cluster(None, None)


def test_resolve_cluster_by_name(tmp_path):
    """A string cluster name uses the provided ClusterManager."""
    from sparkrun.api._resolve import resolve_cluster
    from sparkrun.core.cluster_manager import ClusterManager

    mgr = ClusterManager(tmp_path)
    mgr.create(name="prod", hosts=["h1"])
    resolved = resolve_cluster("prod", cluster_mgr=mgr)
    assert resolved.name == "prod"
    assert resolved.hosts == ["h1"]


def test_resolve_recipe_passthrough_for_recipe_object():
    """Pre-loaded Recipe is returned as-is."""
    from sparkrun.api._resolve import resolve_recipe
    from sparkrun.core.recipe import Recipe

    # Recipe takes a raw dict at construction.
    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "test/model"})
    assert resolve_recipe(recipe) is recipe


def test_resolve_recipe_unknown_raises_recipe_not_found(tmp_path):
    """A string name that doesn't resolve raises RecipeNotFound."""
    from sparkrun.api._resolve import resolve_recipe
    from sparkrun.core.config import SparkrunConfig

    cfg = SparkrunConfig()
    with pytest.raises(api.RecipeNotFound):
        resolve_recipe("this-recipe-does-not-exist-anywhere", config=cfg)


# --------------------------------------------------------------------------
# Click independence — the API must not require click to import
# --------------------------------------------------------------------------


def test_api_imports_without_click_in_sys_modules():
    """Importing ``sparkrun.api`` must not pull in click as a side effect.

    A fresh subprocess gets a clean ``sys.modules`` so we can detect
    accidental click imports leaking through the API layer.
    """
    import subprocess

    code = (
        "import sys, importlib;"
        "importlib.import_module('sparkrun.api');"
        "assert 'click' not in sys.modules, 'click should not be pulled in by sparkrun.api'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "sparkrun.api should be importable without click.\nstdout: %s\nstderr: %s" % (
        result.stdout,
        result.stderr,
    )
