"""Tests for ``sparkrun.api._hosts`` — the single placement authority.

``resolve_effective_hosts`` is the shared placement function used by
``api.run``, the benchmark flow, and the CLI ``run`` command.  These
tests exercise its branch logic directly (the indirect coverage via
``api.run`` only hits the dry-run happy path), focusing on the failure
and edge branches flagged for 0.3.0:

* solo short-circuit (``--solo`` and ``recipe.mode == 'solo'``),
* single-host bypass (no scheduler call),
* ``max_nodes`` hard-error vs. cap-note,
* runtime ``world_size`` baking into the scheduling request,
* typed-exception raising (``InsufficientCapacity``).

The scheduler / status calls are monkeypatched on ``sparkrun.api`` so
the tests never touch real hosts (mirrors ``test_threading_pass.py``).
"""

from __future__ import annotations

import pytest

import sparkrun.api as api
from sparkrun.api._errors import InsufficientCapacity, SparkrunError
from sparkrun.api._hosts import resolve_effective_hosts
from sparkrun.core.cluster_manager import ClusterDefinition


# ---------------------------------------------------------------------------
# Stub recipe — mirrors the _Recipe stubs in test_threading_pass.py
# ---------------------------------------------------------------------------


class _Recipe:
    """Minimal recipe stub for resolve_effective_hosts."""

    def __init__(self, *, max_nodes=None, mode="auto", parallelism=None, layout=None):
        self.max_nodes = max_nodes
        self.mode = mode
        self.layout = layout
        self.qualified_name = "stub/recipe"
        self._parallelism = parallelism or {}

    def build_config_chain(self, overrides):
        # Merge configured parallelism with caller overrides (overrides win).
        chain = dict(self._parallelism)
        chain.update(overrides or {})
        return chain


def _stub_schedule(monkeypatch, hosts_used, *, capture=None):
    """Patch ``api.schedule`` to return a synthetic assignment over *hosts_used*."""
    from sparkrun.core.scheduler import RankAssignment, RankSlot, SchedulingResult

    def _schedule(request, **kwargs):
        if capture is not None:
            capture["request"] = request
            capture["scheduler_kw"] = kwargs.get("scheduler")
        assignment = RankAssignment(
            by_rank=tuple(RankSlot(h, 0) for h in hosts_used),
            hosts_used=tuple(hosts_used),
        )
        return SchedulingResult(assignment=assignment, scheduler_name="greedy")

    monkeypatch.setattr(api, "schedule", _schedule)


# ---------------------------------------------------------------------------
# Single-host bypass and solo short-circuit (scheduler is NOT consulted)
# ---------------------------------------------------------------------------


def test_single_host_bypasses_scheduler(monkeypatch):
    """One input host short-circuits: no scheduler call, solo, no placement."""

    def _boom(*a, **k):
        raise AssertionError("scheduler must not be called for a single host")

    monkeypatch.setattr(api, "schedule", _boom)

    recipe = _Recipe(parallelism={"tensor_parallel": 2})
    host_list, is_solo, notes, placement = resolve_effective_hosts(
        ["only-host"],
        recipe,
        {},
    )
    assert host_list == ["only-host"]
    assert is_solo is True
    assert placement is None
    assert notes == []


def test_solo_flag_truncates_and_clears_placement(monkeypatch):
    """``solo=True`` skips scheduling entirely and trims to the head host."""

    def _boom(*a, **k):
        raise AssertionError("scheduler must not be called in solo mode")

    monkeypatch.setattr(api, "schedule", _boom)

    recipe = _Recipe(parallelism={"tensor_parallel": 2})
    host_list, is_solo, notes, placement = resolve_effective_hosts(
        ["h1", "h2", "h3"],
        recipe,
        {},
        solo=True,
    )
    assert host_list == ["h1"]
    assert is_solo is True
    assert placement is None
    assert any("solo mode enabled" in n for n in notes)


def test_recipe_mode_solo_forces_single_host(monkeypatch):
    """``recipe.mode == 'solo'`` trims to one host at the final solo gate.

    Unlike the ``solo`` flag, ``recipe.mode`` does not gate the scheduler
    block (that block keys off the ``solo`` argument only), so the
    scheduler still runs; the final ``is_solo`` gate then trims to the
    head host and clears the placement.
    """
    _stub_schedule(monkeypatch, ["h1", "h2"])
    monkeypatch.setattr(api, "status", lambda *a, **k: None)

    recipe = _Recipe(mode="solo", parallelism={"tensor_parallel": 2})
    host_list, is_solo, notes, placement = resolve_effective_hosts(
        ["h1", "h2"],
        recipe,
        {},
    )
    assert host_list == ["h1"]
    assert is_solo is True
    assert placement is None
    assert any("solo mode enabled" in n for n in notes)


def test_no_parallelism_configured_skips_scheduler(monkeypatch):
    """Multiple hosts but no parallelism settings -> scheduler is not consulted.

    With no tp/pp/dp set, ``parallelism_configured`` is False so the
    scheduling block is skipped; all hosts pass through and is_solo is False.
    """

    def _boom(*a, **k):
        raise AssertionError("scheduler must not be called without parallelism")

    monkeypatch.setattr(api, "schedule", _boom)

    recipe = _Recipe()
    host_list, is_solo, notes, placement = resolve_effective_hosts(
        ["h1", "h2"],
        recipe,
        {},
    )
    assert host_list == ["h1", "h2"]
    assert is_solo is False
    assert placement is None
    assert notes == []


# ---------------------------------------------------------------------------
# Scheduler success path + notes + placement
# ---------------------------------------------------------------------------


def test_scheduler_trims_hosts_and_emits_note(monkeypatch):
    """Scheduler returns fewer hosts than provided -> trim + 'N of M' note."""
    _stub_schedule(monkeypatch, ["h1", "h2"])
    # Avoid status side effects.
    monkeypatch.setattr(api, "status", lambda *a, **k: None)

    recipe = _Recipe(parallelism={"tensor_parallel": 2})
    cluster = ClusterDefinition(name="c", hosts=["h1", "h2", "h3", "h4"])
    host_list, is_solo, notes, placement = resolve_effective_hosts(
        ["h1", "h2", "h3", "h4"],
        recipe,
        {},
        cluster_def=cluster,
    )
    assert host_list == ["h1", "h2"]
    assert is_solo is False
    assert placement is not None
    assert placement.hosts_used == ("h1", "h2")
    assert any("2 nodes required, using 2 of 4 hosts" in n for n in notes)


def test_runtime_world_size_baked_into_request(monkeypatch):
    """When a runtime is supplied, its ``world_size`` is baked into total_ranks."""
    capture: dict = {}
    _stub_schedule(monkeypatch, ["h1", "h2", "h3", "h4"], capture=capture)
    monkeypatch.setattr(api, "status", lambda *a, **k: None)

    class _Runtime:
        def world_size(self, parallelism, recipe=None, cluster=None):
            # Force a rank count that differs from tp alone.
            return 4

    recipe = _Recipe(parallelism={"tensor_parallel": 2})
    cluster = ClusterDefinition(name="c", hosts=["h1", "h2", "h3", "h4"])
    host_list, is_solo, notes, placement = resolve_effective_hosts(
        ["h1", "h2", "h3", "h4"],
        recipe,
        {},
        cluster_def=cluster,
        runtime=_Runtime(),
    )
    # The request's parallelism carries the runtime-derived total_ranks.
    request = capture["request"]
    assert request.parallelism.total_ranks == 4
    assert request.parallelism.world_size() == 4
    assert host_list == ["h1", "h2", "h3", "h4"]


def test_scheduler_choice_threaded_through(monkeypatch):
    """The ``scheduler`` argument is forwarded to ``api.schedule``."""
    capture: dict = {}
    _stub_schedule(monkeypatch, ["h1", "h2"], capture=capture)
    monkeypatch.setattr(api, "status", lambda *a, **k: None)

    recipe = _Recipe(parallelism={"tensor_parallel": 2})
    cluster = ClusterDefinition(name="c", hosts=["h1", "h2"])
    resolve_effective_hosts(
        ["h1", "h2"],
        recipe,
        {},
        cluster_def=cluster,
        scheduler="occupancy-dense",
    )
    assert capture["scheduler_kw"] == "occupancy-dense"


def test_status_failure_is_best_effort(monkeypatch):
    """A failing ``api.status`` query does not abort scheduling (status=None)."""
    capture: dict = {}
    _stub_schedule(monkeypatch, ["h1", "h2"], capture=capture)

    def _status_boom(*a, **k):
        raise RuntimeError("partial reachability")

    monkeypatch.setattr(api, "status", _status_boom)

    recipe = _Recipe(parallelism={"tensor_parallel": 2})
    cluster = ClusterDefinition(name="c", hosts=["h1", "h2"])
    host_list, _, _, _ = resolve_effective_hosts(
        ["h1", "h2"],
        recipe,
        {},
        cluster_def=cluster,
    )
    assert capture["request"].status is None
    assert host_list == ["h1", "h2"]


# ---------------------------------------------------------------------------
# InsufficientCapacity translation
# ---------------------------------------------------------------------------


def test_insufficient_capacity_too_few_hosts_message(monkeypatch):
    """When fewer hosts than required, the typed error explains the shortfall."""

    def _schedule(request, **kwargs):
        raise InsufficientCapacity("scheduler said no")

    monkeypatch.setattr(api, "schedule", _schedule)
    monkeypatch.setattr(api, "status", lambda *a, **k: None)

    # tp=4 needs 4 nodes, but only 2 hosts provided.
    recipe = _Recipe(parallelism={"tensor_parallel": 4})
    cluster = ClusterDefinition(name="c", hosts=["h1", "h2"])
    with pytest.raises(InsufficientCapacity) as exc:
        resolve_effective_hosts(
            ["h1", "h2"],
            recipe,
            {},
            cluster_def=cluster,
        )
    err = exc.value
    assert "requires 4 nodes" in str(err)
    assert "only 2 hosts provided" in str(err)
    assert err.required == 4
    assert err.host_list == ("h1", "h2")


def test_insufficient_capacity_occupied_cluster_message(monkeypatch):
    """Enough hosts but no free slots -> 'insufficient free capacity' message."""

    def _schedule(request, **kwargs):
        raise InsufficientCapacity("no free accelerator slots")

    monkeypatch.setattr(api, "schedule", _schedule)
    monkeypatch.setattr(api, "status", lambda *a, **k: None)

    # tp=2 needs 2 nodes and 2 hosts are provided (occupancy, not shortfall).
    recipe = _Recipe(parallelism={"tensor_parallel": 2})
    cluster = ClusterDefinition(name="c", hosts=["h1", "h2"])
    with pytest.raises(InsufficientCapacity) as exc:
        resolve_effective_hosts(
            ["h1", "h2"],
            recipe,
            {},
            cluster_def=cluster,
        )
    assert "insufficient free capacity for 2 node(s)" in str(exc.value)


# ---------------------------------------------------------------------------
# max_nodes — hard error vs. cap note
# ---------------------------------------------------------------------------


def test_max_nodes_hard_error_when_scheduler_exceeds(monkeypatch):
    """Scheduler returns more hosts than recipe.max_nodes -> SparkrunError."""
    _stub_schedule(monkeypatch, ["h1", "h2", "h3"])
    monkeypatch.setattr(api, "status", lambda *a, **k: None)

    recipe = _Recipe(max_nodes=2, parallelism={"tensor_parallel": 3})
    cluster = ClusterDefinition(name="c", hosts=["h1", "h2", "h3"])
    with pytest.raises(SparkrunError) as exc:
        resolve_effective_hosts(
            ["h1", "h2", "h3"],
            recipe,
            {},
            cluster_def=cluster,
        )
    assert "max_nodes=2" in str(exc.value)


def test_max_nodes_cap_note_when_scheduler_not_run(monkeypatch):
    """No parallelism (scheduler skipped) but more hosts than max_nodes -> cap note.

    With no parallelism configured the scheduler block is skipped, so the
    orthogonal ``max_nodes`` cap applies: host list is trimmed and a note
    emitted (NOT a hard error, since required <= max_nodes is trivially
    satisfied when parallelism is unset).
    """

    def _boom(*a, **k):
        raise AssertionError("scheduler must not run without parallelism")

    monkeypatch.setattr(api, "schedule", _boom)

    recipe = _Recipe(max_nodes=2)
    host_list, is_solo, notes, placement = resolve_effective_hosts(
        ["h1", "h2", "h3", "h4"],
        recipe,
        {},
    )
    assert host_list == ["h1", "h2"]
    assert placement is None
    assert any("max_nodes=2, using 2 of 4 hosts" in n for n in notes)


def test_max_nodes_orthogonal_hard_error_with_parallelism(monkeypatch):
    """Orthogonal max_nodes block raises when required > max_nodes and the
    scheduler block was skipped.

    The scheduler block requires ``not solo``; with ``solo=True`` it is
    skipped even though parallelism is configured.  The orthogonal cap
    block at the bottom then sees ``parallelism_configured=True`` and
    ``required (4) > max_nodes (2)`` and raises a hard error *before* the
    solo truncation runs.
    """

    def _boom(*a, **k):
        raise AssertionError("scheduler must not run when solo=True")

    monkeypatch.setattr(api, "schedule", _boom)

    recipe = _Recipe(max_nodes=2, parallelism={"tensor_parallel": 4})
    with pytest.raises(SparkrunError) as exc:
        resolve_effective_hosts(
            ["h1", "h2", "h3", "h4"],
            recipe,
            {},
            solo=True,
        )
    assert "requires 4 nodes" in str(exc.value)
    assert "max_nodes=2" in str(exc.value)
