"""Tests for per-machine container images (``containers:``).

Covers the four things that fail *silently* if broken: workload identity
(a wrong intent_id evicts someone else's job), the resolver's fail-loud
validation (a typo'd hostname would otherwise run an untuned image), the
runtime guardrail, and the ``pull`` transfer mode's per-node dispatch.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sparkrun.core.images import (
    ImagePlan,
    ImagePlanError,
    derive_container_entries,
    parse_container_entries,
    resolve_image_plan,
)
from sparkrun.core.recipe import Recipe
from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint, generate_intent_id

HOSTS = ["spark-01", "spark-02", "spark-03"]


def _recipe(containers=None, container="img:latest", **extra):
    d = {
        "recipe_version": "2",
        "name": "pmi",
        "model": "org/model",
        "runtime": "sglang",
        "container": container,
        **extra,
    }
    if containers is not None:
        d["containers"] = containers
    return Recipe.from_dict(d)


def _tuned(n=3):
    return [{"host": h, "image": "myorg/tuned:%s" % h} for h in HOSTS[:n]]


# ---------------------------------------------------------------------------
# Parsing / round-trip
# ---------------------------------------------------------------------------


def test_parse_drops_incomplete_entries():
    assert parse_container_entries([{"host": "a", "image": "i"}, {"host": "b"}, {"image": "j"}, "nope"]) == [{"host": "a", "image": "i"}]


def test_parse_of_non_list_is_empty():
    assert parse_container_entries(None) == []
    assert parse_container_entries({"host": "a"}) == []


def test_recipe_round_trips_containers():
    r = _recipe(_tuned())
    restored = Recipe._deserialize(r.__getstate__())
    assert restored.containers == r.containers
    assert "containers" in r._build_export_dict()


def test_recipe_without_block_exports_no_containers_key():
    assert "containers" not in _recipe()._build_export_dict()


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_no_block_is_uniform_and_not_heterogeneous():
    plan = resolve_image_plan(_recipe(), "img:latest", HOSTS)
    assert plan.images_by_node == ("img:latest",) * 3
    assert plan.declared == ()
    assert plan.heterogeneous is False


def test_resolves_per_host():
    plan = resolve_image_plan(_recipe(_tuned()), "img:latest", HOSTS)
    assert plan.images_by_node == tuple("myorg/tuned:%s" % h for h in HOSTS)
    assert plan.heterogeneous is True
    assert plan.head_image() == "myorg/tuned:spark-01"


def test_unlisted_host_falls_back_to_container():
    plan = resolve_image_plan(_recipe(_tuned(2)), "img:latest", HOSTS)
    assert plan.images_by_node == ("myorg/tuned:spark-01", "myorg/tuned:spark-02", "img:latest")


def test_unlisted_host_without_fallback_raises():
    """No image at all for a selected machine is a hard error, not a guess."""
    with pytest.raises(ImagePlanError, match="spark-03"):
        resolve_image_plan(_recipe(_tuned(2), container=""), "", HOSTS)


def test_host_not_in_cluster_raises():
    """A typo'd hostname must not silently fall through to the generic image."""
    entries = _tuned(2) + [{"host": "spark-99", "image": "myorg/tuned:typo"}]
    with pytest.raises(ImagePlanError, match="spark-99"):
        resolve_image_plan(_recipe(entries), "img:latest", HOSTS, cluster_hosts=HOSTS)


def test_declaring_more_machines_than_used_is_legal():
    """The block usually covers the whole cluster; a tp-2 launch uses two."""
    plan = resolve_image_plan(_recipe(_tuned()), "img:latest", HOSTS[:2], cluster_hosts=HOSTS)
    assert plan.images_by_node == ("myorg/tuned:spark-01", "myorg/tuned:spark-02")


def test_duplicate_host_raises():
    entries = [{"host": "spark-01", "image": "a:1"}, {"host": "spark-01", "image": "b:1"}]
    with pytest.raises(ImagePlanError, match="more than once"):
        resolve_image_plan(_recipe(entries), "img:latest", HOSTS)


def test_derive_container_entries_groups_by_image():
    plan = ImagePlan(default_image="d:1", images_by_node=("a:1", "b:1", "a:1"))
    entries = derive_container_entries(plan, HOSTS)
    assert {e.name: e.target for e in entries} == {"a:1": [0, 2], "b:1": [1]}


# ---------------------------------------------------------------------------
# Workload identity
# ---------------------------------------------------------------------------


def test_intent_id_unchanged_without_block():
    """The orphaning guard: an existing recipe must hash exactly as before.

    If this moves, every running workload becomes invisible to ``stop`` /
    ``logs`` / ``--ensure``, which all recompute the intent from the recipe.

    The expected digest is rebuilt here from the *pre-feature* key construction
    rather than captured from the current implementation — a snapshot of the
    code under test would move happily along with a regression.
    """
    import hashlib

    from sparkrun.orchestration.job_metadata import INTENT_ID_LEN

    expected_key = "\0".join(["sglang", "org/model", "image=img:latest"])
    expected = hashlib.sha256(expected_key.encode()).hexdigest()[:INTENT_ID_LEN]
    assert generate_intent_id(_recipe()) == expected


def test_intent_id_differs_when_one_machine_image_differs():
    """Otherwise launching the second silently evicts the first."""
    a = _recipe(_tuned())
    changed = _tuned()
    changed[1]["image"] = "myorg/tuned-nightly:spark-02"
    assert generate_intent_id(a) != generate_intent_id(_recipe(changed))


def test_intent_id_ignores_declaration_order():
    a = _recipe(_tuned())
    b = _recipe(list(reversed(_tuned())))
    assert generate_intent_id(a) == generate_intent_id(b)


def test_intent_id_is_placement_independent():
    """Hashing the *declared* map, not the resolved one.

    The scheduler picks hosts per launch; if the intent moved with that choice,
    a job placed on a different subset could never be found again.
    """
    r = _recipe(_tuned())
    intent = generate_intent_id(r)
    resolved = []
    for subset in (HOSTS, HOSTS[:2], HOSTS[1:]):
        plan = resolve_image_plan(r, "img:latest", subset, cluster_hosts=HOSTS)
        resolved.append(plan.images_by_node)
        assert generate_intent_id(r) == intent
    # The premise: placement genuinely produced different per-node maps, so the
    # invariance above is meaningful rather than trivially true.
    assert len(set(resolved)) == len(resolved)


def test_block_changes_the_fingerprint():
    assert derive_recipe_fingerprint(_recipe(_tuned())) != derive_recipe_fingerprint(_recipe())


def test_image_override_clears_the_block():
    from sparkrun.core.resolve import apply_recipe_overrides

    r = _recipe(_tuned())
    apply_recipe_overrides((), image="override:1", recipe=r)
    assert r.containers == []
    assert r.container == "override:1"
    assert resolve_image_plan(r, "override:1", HOSTS).heterogeneous is False


# ---------------------------------------------------------------------------
# Runtime guardrail
# ---------------------------------------------------------------------------


def test_ray_and_trtllm_fail_closed_by_default():
    from sparkrun.runtimes.base import RuntimePlugin
    from sparkrun.runtimes.trtllm import TrtllmRuntime
    from sparkrun.runtimes.vllm_ray import VllmRayRuntime

    assert RuntimePlugin.supports_heterogeneous_images is False
    assert VllmRayRuntime.supports_heterogeneous_images is False
    assert TrtllmRuntime.supports_heterogeneous_images is False


def _launch_harness(monkeypatch, tmp_path, runtime_supports, builder="", builder_transforms=True):
    """Mock just enough of launch_inference's preamble to reach the guards.

    The guards must fire *before* any side effect, so nothing here mocks SSH,
    distribution or the container engine — if a guard were misplaced, the test
    would fail on a missing mock rather than passing quietly.
    """
    from sparkrun.core import launcher

    monkeypatch.setattr(
        "sparkrun.orchestration.distribution.resolve_auto_transfer_mode",
        lambda *a, **kw: type("R", (), {"mode": "local"})(),
    )
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda *a, **kw: {})
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.derive_cluster_id", lambda *a, **kw: "sparkrun_test00000000")
    monkeypatch.setattr(launcher, "resolve_effective_cache_dir", lambda *a, **kw: str(tmp_path))
    monkeypatch.setattr(
        launcher,
        "builder_transforms_image",
        lambda recipe, v=None: bool(getattr(recipe, "builder", "")) and builder_transforms,
    )

    class _Cfg:
        hf_cache_dir = tmp_path / "hf"
        cache_dir = tmp_path / "cache"

        def get_registry_manager(self):
            return None

    class _Runtime:
        runtime_name = "stub"
        supports_heterogeneous_images = runtime_supports

        def resolve_container(self, recipe, overrides):
            return "stub:latest"

        def get_family(self):
            return "stub"

        def prepare(self, *a, **kw):
            # Sentinel: reached only once both guards and the image-plan
            # resolution have passed, and well before any SSH.
            raise AssertionError("launch proceeded past the per-machine-image guard")

        def run(self, *a, **kw):  # pragma: no cover - the sentinel fires first
            return type("R", (), {"containers": {}, "head_host": "h1"})()

    class _Recipe:
        runtime = "stub"
        model = "stub-model"
        env = {}
        builder_config = {}
        mods = []
        source_registry = None
        source_registry_url = None
        defaults = {"port": 8000}
        pre_exec = []
        post_exec = []
        post_commands = []
        layout = None
        stop_after_post = False
        executor = ""
        executor_config = None
        is_url_sourced = False
        cluster_config = None
        qualified_name = "stub-recipe"
        name = "stub-recipe"
        container = "stub:latest"
        containers = [{"host": "h1", "image": "tuned:h1"}]
        model_revision = None

        def __init__(self):
            self.builder = builder

        def build_config_chain(self, overrides=None):
            merged = dict(self.defaults)
            merged.update(overrides or {})
            return type("_CC", (), {"get": lambda _s, k, d=None: merged.get(k, d)})()

        def __getstate__(self):
            return {}

    def _launch():
        from sparkrun.core.launcher import launch_inference

        return launch_inference(
            recipe=_Recipe(),
            runtime=_Runtime(),
            host_list=["h1"],
            overrides={},
            config=_Cfg(),
            cluster=None,
            is_solo=True,
            dry_run=True,
            sync_tuning=False,
        )

    return _launch


def test_launcher_rejects_containers_on_unsupported_runtime(monkeypatch, tmp_path):
    """Fails before any side effect — no pull, no sync, no container."""
    from sparkrun.core.recipe import RecipeError

    launch = _launch_harness(monkeypatch, tmp_path, runtime_supports=False)
    with pytest.raises(RecipeError, match="does not support"):
        launch()


def test_launcher_rejects_containers_with_an_image_building_builder(monkeypatch, tmp_path):
    from sparkrun.core.recipe import RecipeError

    launch = _launch_harness(monkeypatch, tmp_path, runtime_supports=True, builder="eugr")
    with pytest.raises(RecipeError, match="builder"):
        launch()


def test_launcher_allows_containers_with_a_non_transforming_builder(monkeypatch, tmp_path):
    """docker-pull returns the ref untouched, so it composes — no guard fires.

    Reaching the sentinel in ``_Runtime.run`` is the assertion: the launch got
    all the way past both guards and the builder phase.
    """
    launch = _launch_harness(monkeypatch, tmp_path, runtime_supports=True, builder="docker-pull", builder_transforms=False)
    with pytest.raises(AssertionError, match="past the per-machine-image guard"):
        launch()


def test_native_distributed_runtimes_opt_in():
    from sparkrun.runtimes.llama_cpp import LlamaCppRuntime
    from sparkrun.runtimes.sglang import SglangRuntime
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    assert SglangRuntime.supports_heterogeneous_images is True
    assert VllmDistributedRuntime.supports_heterogeneous_images is True
    assert LlamaCppRuntime.supports_heterogeneous_images is True


def test_environment_builders_do_not_transform_the_image():
    from sparkrun.builders.base import BuilderPlugin
    from sparkrun.builders.docker_pull import DockerPullBuilder
    from sparkrun.builders.uv_venv import UvVenvBuilder

    assert BuilderPlugin.transforms_image is True
    assert UvVenvBuilder.transforms_image is False
    assert DockerPullBuilder.transforms_image is False


# ---------------------------------------------------------------------------
# Launch wiring
# ---------------------------------------------------------------------------


def test_cluster_context_resolves_image_by_host():
    from sparkrun.runtimes._cluster_ops import ClusterContext

    ctx = ClusterContext(
        hosts=list(HOSTS),
        head_host=HOSTS[0],
        worker_hosts=HOSTS[1:],
        num_nodes=3,
        ssh_kwargs={},
        volumes={},
        all_env={},
        cluster_id="sparkrun_aaaaaaaaaaaa_bbbbbbbb",
        image="fallback:1",
        dry_run=True,
        config=None,
        images_by_node=("a:1", "b:1", "c:1"),
    )
    assert [ctx.image_for_host(h) for h in HOSTS] == ["a:1", "b:1", "c:1"]
    assert ctx.heterogeneous_images() is True
    # An unknown host degrades to the cluster image rather than raising.
    assert ctx.image_for_host("who") == "fallback:1"


def test_cluster_context_without_map_uses_single_image():
    from sparkrun.runtimes._cluster_ops import ClusterContext

    ctx = ClusterContext(
        hosts=list(HOSTS),
        head_host=HOSTS[0],
        worker_hosts=HOSTS[1:],
        num_nodes=3,
        ssh_kwargs={},
        volumes={},
        all_env={},
        cluster_id="sparkrun_aaaaaaaaaaaa_bbbbbbbb",
        image="only:1",
        dry_run=True,
        config=None,
    )
    assert [ctx.image_for_host(h) for h in HOSTS] == ["only:1"] * 3
    assert ctx.heterogeneous_images() is False


# ---------------------------------------------------------------------------
# `pull` transfer mode
# ---------------------------------------------------------------------------


def _single_image(mode, **kw):
    from sparkrun.orchestration.distribution import _distribute_single_image

    return _distribute_single_image("img:1", list(HOSTS), list(HOSTS), mode, None, None, {}, False, False, **kw)


def test_pull_mode_pulls_on_every_node_and_never_locally():
    from sparkrun.orchestration import distribution as dist

    with (
        patch("sparkrun.containers.sync.sync_image_to_hosts", return_value=[]) as m_pull,
        patch.object(dist, "_distribute_from_head") as m_head,
    ):
        assert _single_image("pull") == []
    assert m_pull.call_args[0][1] == HOSTS
    m_head.assert_not_called()


def test_heterogeneous_delegated_becomes_per_node_pull():
    """Delegated would fan a machine-tuned image onto the wrong machine."""
    with (
        patch("sparkrun.containers.sync.sync_image_to_hosts", return_value=[]) as m_pull,
        patch("sparkrun.containers.distribute.distribute_image_from_head") as m_head,
    ):
        assert _single_image("delegated", heterogeneous=True) == []
    m_pull.assert_called_once()
    m_head.assert_not_called()


def test_homogeneous_delegated_is_unchanged():
    with (
        patch("sparkrun.containers.sync.sync_image_to_hosts") as m_pull,
        patch("sparkrun.containers.distribute.distribute_image_from_head", return_value=[]) as m_head,
    ):
        assert _single_image("delegated") == []
    m_head.assert_called_once()
    m_pull.assert_not_called()


def test_rebuild_reaches_every_node_pull():
    """The documented workaround for a re-pushed tag has to actually apply."""
    with patch("sparkrun.containers.sync.sync_image_to_hosts", return_value=[]) as m_pull:
        _single_image("pull", force_pull=True)
    assert m_pull.call_args.kwargs["force_pull"] is True


def test_auto_pull_falls_back_to_push_for_failed_nodes_only():
    with (
        patch("sparkrun.containers.sync.sync_image_to_hosts", return_value=["spark-02"]),
        patch("sparkrun.containers.distribute.distribute_image_from_local", return_value=[]) as m_push,
    ):
        from sparkrun.orchestration.distribution import _distribute_single_image

        failed = _distribute_single_image(
            "img:1",
            list(HOSTS),
            list(HOSTS),
            "pull",
            None,
            None,
            {},
            False,
            True,  # auto_delegated
        )
    assert failed == []
    assert m_push.call_args[0][1] == ["spark-02"]


def test_explicit_pull_does_not_fall_back():
    """An explicitly-named mode is honored literally — the `delegated` rule."""
    with (
        patch("sparkrun.containers.sync.sync_image_to_hosts", return_value=["spark-02"]),
        patch("sparkrun.containers.distribute.distribute_image_from_local") as m_push,
    ):
        assert _single_image("pull") == ["spark-02"]
    m_push.assert_not_called()


def test_model_pull_downloads_per_node():
    from sparkrun.orchestration.distribution import _distribute_single_model

    with patch("sparkrun.models.distribute.distribute_model_per_node", return_value=[]) as m:
        _distribute_single_model("org/m", list(HOSTS), list(HOSTS), "/cache", "/cache", "pull", None, None, {}, None, None, False, False)
    assert m.call_args[0][1] == HOSTS


def test_model_pull_with_shared_cache_downloads_on_head_only():
    """N nodes writing one NFS path concurrently is waste at best."""
    from sparkrun.core.cluster_manager import ModelDistributionPrefs
    from sparkrun.orchestration.distribution import _distribute_single_model

    with (
        patch("sparkrun.models.distribute.distribute_model_per_node") as m_node,
        patch("sparkrun.models.distribute.distribute_model_from_head", return_value=[]) as m_head,
    ):
        _distribute_single_model(
            "org/m",
            list(HOSTS),
            list(HOSTS),
            "/cache",
            "/cache",
            "pull",
            None,
            None,
            {},
            None,
            None,
            False,
            False,
            prefs=ModelDistributionPrefs(skip_fan_out=True),
        )
    m_node.assert_not_called()
    assert m_head.call_args.kwargs["skip_fan_out"] is True
