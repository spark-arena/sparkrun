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


def test_solo_flag_picks_single_host_via_scheduler(monkeypatch):
    """``solo=True`` runs a 1-rank schedule to pick one host *with room*.

    Solo no longer blindly trims to ``host_list[0]``; it issues a
    ``world_size == 1`` scheduling request so the configured scheduler picks
    an occupancy-appropriate host.  Here the stub returns ``h2`` (not the
    head), proving the chosen host comes from the scheduler, not a slice.
    """
    capture: dict = {}
    _stub_schedule(monkeypatch, ["h2"], capture=capture)
    monkeypatch.setattr(api, "status", lambda *a, **k: None)

    recipe = _Recipe(parallelism={"tensor_parallel": 2})
    host_list, is_solo, notes, placement = resolve_effective_hosts(
        ["h1", "h2", "h3"],
        recipe,
        {},
        solo=True,
    )
    assert host_list == ["h2"]
    assert is_solo is True
    assert placement is None
    assert any("solo mode enabled" in n for n in notes)
    # Solo forces a single-rank request regardless of the recipe's tp=2.
    assert capture["request"].parallelism.world_size() == 1


def test_solo_raises_when_no_host_has_room(monkeypatch):
    """Solo fails closed when the scheduler can fit no host (every host full)."""
    from sparkrun.api._errors import InsufficientCapacity as _IC

    def _schedule(request, **kwargs):
        raise _IC("no free accelerator slots")

    monkeypatch.setattr(api, "schedule", _schedule)
    monkeypatch.setattr(api, "status", lambda *a, **k: None)

    recipe = _Recipe(parallelism={"tensor_parallel": 2})
    with pytest.raises(_IC):
        resolve_effective_hosts(["h1", "h2", "h3"], recipe, {}, solo=True)


def test_recipe_mode_solo_forces_single_host(monkeypatch):
    """``recipe.mode == 'solo'`` routes to the single-host occupancy pick.

    Like the ``solo`` flag, ``recipe.mode == 'solo'`` now skips the multi-node
    scheduling block and issues a 1-rank request through the single-host pick;
    the final ``is_solo`` gate reports solo and clears the placement.
    """
    _stub_schedule(monkeypatch, ["h1"])
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


def test_expert_parallel_only_recipe_engages_scheduler(monkeypatch):
    """A recipe configuring only expert/context parallelism must still schedule.

    Regression: ``parallelism_configured`` previously checked only tp/pp/dp, so
    an ep-only recipe (whose runtime derives world_size from ``tp*ep``) skipped
    the multi-node scheduling block and was treated as single-host.  With
    ep included in the gate, the scheduler is consulted and the runtime's
    world_size override is baked into the request.
    """
    capture: dict = {}
    _stub_schedule(monkeypatch, ["h1", "h2"], capture=capture)
    monkeypatch.setattr(api, "status", lambda *a, **k: None)

    class _Runtime:
        def world_size(self, parallelism, recipe=None, cluster=None):
            # e.g. Atlas: tp(1) * ep(2) -> 2 ranks.
            return parallelism.expert_parallel * parallelism.tensor_parallel

    recipe = _Recipe(parallelism={"expert_parallel": 2})
    cluster = ClusterDefinition(name="c", hosts=["h1", "h2"])
    host_list, is_solo, notes, placement = resolve_effective_hosts(
        ["h1", "h2"],
        recipe,
        {},
        cluster_def=cluster,
        runtime=_Runtime(),
    )
    assert host_list == ["h1", "h2"]
    assert is_solo is False
    assert placement is not None
    assert capture["request"].parallelism.world_size() == 2


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
# Usable-memory cap: per-rank VRAM claim + resolved per-GPU caps
# ---------------------------------------------------------------------------


class _RecipeWithEstimate(_Recipe):
    """Recipe stub whose estimate_vram is controllable for cap-wiring tests."""

    def __init__(self, *, per_gpu_gb=None, raises=False, **kw):
        super().__init__(**kw)
        self._per_gpu_gb = per_gpu_gb
        self._raises = raises

    def estimate_vram(self, cli_overrides=None, auto_detect=True, cache_dir=None):
        import types

        if self._raises:
            raise RuntimeError("model metadata unavailable")
        return types.SimpleNamespace(total_per_gpu_gb=self._per_gpu_gb)


def test_vram_claim_and_resolved_caps_threaded_into_request(monkeypatch):
    """estimate_vram populates resources.memory_gb and folds the GB10 cap into host_hardware."""
    capture: dict = {}
    _stub_schedule(monkeypatch, ["h1", "h2"], capture=capture)
    monkeypatch.setattr(api, "status", lambda *a, **k: None)

    recipe = _RecipeWithEstimate(per_gpu_gb=50.0, parallelism={"tensor_parallel": 2})
    cluster = ClusterDefinition(name="c", hosts=["h1", "h2"])
    resolve_effective_hosts(["h1", "h2"], recipe, {}, cluster_def=cluster)

    request = capture["request"]
    assert request.resources is not None
    assert request.resources.memory_gb == 50.0
    assert request.resources.util_fraction == 1.0
    # Default DGX hosts → platform cap 0.85 folded into the spec field.
    assert request.host_hardware["h1"].accelerators[0].max_gpu_memory_utilization == 0.85


def test_vram_estimate_failure_yields_no_claim(monkeypatch):
    """A failing estimate degrades to resources=None (memory-blind, no regression)."""
    capture: dict = {}
    _stub_schedule(monkeypatch, ["h1", "h2"], capture=capture)
    monkeypatch.setattr(api, "status", lambda *a, **k: None)

    recipe = _RecipeWithEstimate(raises=True, parallelism={"tensor_parallel": 2})
    cluster = ClusterDefinition(name="c", hosts=["h1", "h2"])
    resolve_effective_hosts(["h1", "h2"], recipe, {}, cluster_def=cluster)

    assert capture["request"].resources is None


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


# ---------------------------------------------------------------------------
# C1: self-intent occupancy exclusion
# ---------------------------------------------------------------------------


def test_exclude_intent_subtracts_self_from_occupancy(monkeypatch):
    """A relaunch must not count its own still-running containers as load.

    ``h1`` is reported fully busy by a workload whose intent matches the launch;
    with ``exclude_intent_id`` that workload is subtracted before scheduling, so
    the scheduler sees ``h1`` free again."""
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload

    status = ClusterStatus(
        hosts=(
            HostOccupancy(
                host="h1",
                used_slots=1,
                free_slots=0,
                workloads=(RunningWorkload(cluster_id="abc123-tok", intent_id="abc123"),),
            ),
            HostOccupancy(host="h2", used_slots=0, free_slots=1),
        ),
        executor="docker",
    )
    monkeypatch.setattr(api, "status", lambda *a, **k: status)
    capture = {}
    _stub_schedule(monkeypatch, ["h1"], capture=capture)

    recipe = _Recipe(parallelism={"tensor_parallel": 1})
    resolve_effective_hosts(["h1", "h2"], recipe, {}, exclude_intent_id="abc123")

    sched_status = capture["request"].status
    h1 = sched_status.for_host("h1")
    assert h1.used_slots == 0
    assert h1.free_slots == 1
    assert h1.workloads == ()


def test_exclude_intent_leaves_foreign_workloads_intact(monkeypatch):
    """Only the matching intent is subtracted; other workloads stay as load."""
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload

    status = ClusterStatus(
        hosts=(
            HostOccupancy(
                host="h1",
                used_slots=1,
                free_slots=0,
                workloads=(RunningWorkload(cluster_id="other-tok", intent_id="other"),),
            ),
        ),
        executor="docker",
    )
    monkeypatch.setattr(api, "status", lambda *a, **k: status)
    capture = {}
    _stub_schedule(monkeypatch, ["h1"], capture=capture)

    recipe = _Recipe(parallelism={"tensor_parallel": 1})
    resolve_effective_hosts(["h1", "h2"], recipe, {}, exclude_intent_id="abc123")

    h1 = capture["request"].status.for_host("h1")
    assert h1.used_slots == 1
    assert len(h1.workloads) == 1


# ---------------------------------------------------------------------------
# A2: user-facing node counts honour the runtime world_size override
# ---------------------------------------------------------------------------


def test_insufficient_capacity_message_uses_world_size(monkeypatch):
    """The 'requires N nodes' message/field reflects ``runtime.world_size``
    (the baked rank count), not the raw ``tp*pp*dp`` ``total_nodes``."""

    class _RT:
        def world_size(self, parallelism, recipe=None, cluster=None):
            return 4  # overrides the tp=2 formula (total_nodes would say 2)

    monkeypatch.setattr(api, "status", lambda *a, **k: None)

    def _raise(*a, **k):
        raise InsufficientCapacity("no room")

    monkeypatch.setattr(api, "schedule", _raise)

    recipe = _Recipe(parallelism={"tensor_parallel": 2})
    with pytest.raises(InsufficientCapacity) as exc:
        resolve_effective_hosts(["h1", "h2"], recipe, {}, runtime=_RT())
    assert exc.value.required == 4
    assert "requires 4 nodes" in str(exc.value)


# ---------------------------------------------------------------------------
# A1 / C5: occupancy-aware max_nodes trim threads placement
# ---------------------------------------------------------------------------


def test_max_nodes_trim_with_cluster_is_occupancy_aware_and_threads_placement(monkeypatch):
    """No parallelism + max_nodes + a known cluster → the trim runs through the
    scheduler (least-loaded subset) and threads the resulting placement instead
    of a blind prefix slice with placement=None."""
    monkeypatch.setattr(api, "status", lambda *a, **k: None)
    capture = {}
    _stub_schedule(monkeypatch, ["h3", "h4"], capture=capture)

    cluster = ClusterDefinition(name="c", hosts=["h1", "h2", "h3", "h4"])
    recipe = _Recipe(max_nodes=2)  # no parallelism configured
    host_list, is_solo, notes, placement = resolve_effective_hosts(
        ["h1", "h2", "h3", "h4"],
        recipe,
        {},
        cluster_def=cluster,
    )
    assert host_list == ["h3", "h4"]
    assert placement is not None
    assert placement.hosts_used == ("h3", "h4")
    assert capture["request"].parallelism.world_size() == 2
    assert any("max_nodes=2, using 2 of 4 hosts" in n for n in notes)
