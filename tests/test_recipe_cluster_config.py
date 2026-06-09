"""Tests for the internal ``cluster_config`` recipe escape hatch.

Covers recipe parsing / round-trip, the resolved-model volume helper, the
``skip_model`` distribution gate, and the ``launch_inference`` choke point that
applies the overrides (cache dirs, resolved_model_path → serve-arg + skip).
"""

from __future__ import annotations

from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.recipe import ClusterConfig, Recipe
from sparkrun.orchestration.primitives import resolved_model_volume


def _recipe_dict(**cluster_config):
    d = {
        "recipe_version": "2",
        "name": "esc",
        "model": "Qwen/Qwen3-1.7B",
        "runtime": "vllm-distributed",
        "container": "img:latest",
    }
    if cluster_config:
        d["cluster_config"] = cluster_config
    return d


# ---------------------------------------------------------------------------
# Recipe parsing / round-trip
# ---------------------------------------------------------------------------


def test_cluster_config_parsed_into_recipe():
    r = Recipe.from_dict(_recipe_dict(remote_cache_dir="/nfs/hf", local_cache_dir="/tmp/hf", resolved_model_path="/nfs/models/qwen3"))
    assert r.cluster_config == ClusterConfig(remote_cache_dir="/nfs/hf", local_cache_dir="/tmp/hf", resolved_model_path="/nfs/models/qwen3")


def test_cluster_config_absent_is_none_and_not_swept():
    r = Recipe.from_dict(_recipe_dict())
    assert r.cluster_config is None
    # Unknown-key sweep must not capture it (it's a known key now).
    assert "cluster_config" not in r.runtime_config


def test_cluster_config_empty_block_is_none():
    assert Recipe.from_dict(_recipe_dict(**{})).cluster_config is None  # no block
    # An explicitly empty dict also collapses to None.
    d = _recipe_dict()
    d["cluster_config"] = {}
    assert Recipe.from_dict(d).cluster_config is None


def test_cluster_config_partial_block():
    r = Recipe.from_dict(_recipe_dict(resolved_model_path="/nfs/m"))
    assert r.cluster_config == ClusterConfig(resolved_model_path="/nfs/m")
    assert r.cluster_config.remote_cache_dir is None


def test_cluster_config_getstate_setstate_round_trip():
    r = Recipe.from_dict(_recipe_dict(remote_cache_dir="/r", local_cache_dir="/l", resolved_model_path="/m"))
    state = r.__getstate__()
    assert state["cluster_config"] == {"remote_cache_dir": "/r", "local_cache_dir": "/l", "resolved_model_path": "/m"}
    restored = Recipe._deserialize(state)
    assert restored.cluster_config == r.cluster_config


def test_cluster_config_survives_export():
    r = Recipe.from_dict(_recipe_dict(resolved_model_path="/nfs/m"))
    exported = r.export()
    assert "cluster_config" in exported
    assert "resolved_model_path" in exported


# ---------------------------------------------------------------------------
# resolved_model_volume helper
# ---------------------------------------------------------------------------


def test_resolved_model_volume_identity_mount():
    r = Recipe.from_dict(_recipe_dict(resolved_model_path="/nfs/models/qwen3"))
    assert resolved_model_volume(r) == {"/nfs/models/qwen3": "/nfs/models/qwen3"}


def test_resolved_model_volume_empty_when_unset():
    assert resolved_model_volume(Recipe.from_dict(_recipe_dict())) == {}
    assert resolved_model_volume(None) == {}
    assert resolved_model_volume(object()) == {}


def test_executor_volumes_and_resolved_model_path_coexist():
    """executor_config.volumes and cluster_config.resolved_model_path are
    independent -v channels — they coexist on the same docker run, never clobber."""
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.docker import DockerExecutor
    from sparkrun.orchestration.primitives import build_volumes

    recipe = Recipe.from_dict(_recipe_dict(resolved_model_path="/mnt/quant/M3"))
    # resolved_model_path + HF cache flow through the volumes DICT (run_cmd volumes=).
    vols = build_volumes("/hf", extra=resolved_model_volume(recipe))
    # executor_config.volumes flow through ExecutorConfig → separate -v flags.
    cmd = DockerExecutor(ExecutorConfig(volumes=["/mnt/quant/nvfp4_calib"])).run_cmd("img:1", volumes=vols)

    assert "-v /mnt/quant/M3:/mnt/quant/M3" in cmd  # resolved model (cluster_config)
    assert "-v /hf:/cache/huggingface" in cmd  # HF cache
    assert "-v /mnt/quant/nvfp4_calib:/mnt/quant/nvfp4_calib" in cmd  # executor_config.volumes


# ---------------------------------------------------------------------------
# {resolved_model_path} command-template placeholder
# ---------------------------------------------------------------------------


def _render(recipe):
    return recipe.render_command(recipe.build_config_chain())


def test_resolved_model_path_placeholder_uses_configured_path():
    d = _recipe_dict(resolved_model_path="/mnt/quant/M3")
    d["command"] = "sglang serve --model-path {model} --chat-template {resolved_model_path}/chat_template.jinja"
    r = Recipe.from_dict(d)
    rendered = _render(r)
    assert "--chat-template /mnt/quant/M3/chat_template.jinja" in rendered
    # {model} still resolves to the repo id pre-launch (the launcher repoints it
    # at the path at serve time); {resolved_model_path} already gives the path.
    assert "--model-path Qwen/Qwen3-1.7B" in rendered


def test_resolved_model_path_placeholder_falls_back_to_model():
    d = _recipe_dict()  # no cluster_config
    d["command"] = "sglang serve --chat-template {resolved_model_path}/chat_template.jinja"
    r = Recipe.from_dict(d)
    # Falls back to model so the same template stays valid for normal use.
    assert "--chat-template Qwen/Qwen3-1.7B/chat_template.jinja" in _render(r)


def test_resolved_model_path_in_config_chain():
    r = Recipe.from_dict(_recipe_dict(resolved_model_path="/mnt/quant/M3"))
    assert r.build_config_chain().get("resolved_model_path") == "/mnt/quant/M3"
    r2 = Recipe.from_dict(_recipe_dict())
    assert r2.build_config_chain().get("resolved_model_path") == "Qwen/Qwen3-1.7B"


# ---------------------------------------------------------------------------
# distribute_from_config skip_model gate (localhost fast path)
# ---------------------------------------------------------------------------


def _patch_distribution(monkeypatch, download_calls):
    monkeypatch.setattr("sparkrun.orchestration.distribution.is_local_host", lambda h: True)
    monkeypatch.setattr("sparkrun.orchestration.distribution._is_cross_user", lambda kw: False)
    monkeypatch.setattr("sparkrun.orchestration.distribution._get_hf_token", lambda: "")
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda *a, **kw: {})
    monkeypatch.setattr("sparkrun.containers.registry.ensure_image", lambda *a, **kw: 0)

    def _dl(model, **kw):
        download_calls.append(model)
        return 0

    monkeypatch.setattr("sparkrun.models.download.download_model", _dl)


class _Cfg:
    class _P:
        def __str__(self):
            return "/tmp/cache"

    cache_dir = _P()


def test_distribute_skip_model_skips_download(monkeypatch):
    calls: list = []
    _patch_distribution(monkeypatch, calls)
    from sparkrun.orchestration.distribution import distribute_from_config

    recipe = Recipe.from_dict(_recipe_dict())
    distribute_from_config(recipe, "img:latest", ["localhost"], "/tmp/cache", _Cfg(), dry_run=False, skip_model=True)
    assert calls == []  # model download skipped entirely


def test_distribute_without_skip_downloads_model(monkeypatch):
    calls: list = []
    _patch_distribution(monkeypatch, calls)
    from sparkrun.orchestration.distribution import distribute_from_config

    recipe = Recipe.from_dict(_recipe_dict())
    distribute_from_config(recipe, "img:latest", ["localhost"], "/tmp/cache", _Cfg(), dry_run=False, skip_model=False)
    assert calls == ["Qwen/Qwen3-1.7B"]  # model ensured locally


# ---------------------------------------------------------------------------
# launch_inference choke point
# ---------------------------------------------------------------------------


class _StubRuntime:
    runtime_name = "stub"
    requires_capability: frozenset = frozenset()
    last_kwargs: dict = {}

    def is_delegating_runtime(self):
        return False

    def resolve_container(self, recipe, overrides=None):
        return "stub:latest"

    def prepare(self, *a, **k):
        return None

    def get_head_container_name(self, cluster_id, is_solo=False):
        return "%s_solo" % cluster_id

    def generate_command(self, **kwargs):
        return "echo serve"

    def resolve_api_key(self, recipe, overrides=None):
        return None

    def _collect_runtime_info(self, *a, **k):
        return {}

    def run(self, **kwargs):
        type(self).last_kwargs = dict(kwargs)
        return 0


class _Recipe:
    runtime = "stub"
    model = "Qwen/Qwen3-1.7B"
    env: dict = {}
    builder = None
    mods: list = []
    source_registry = None
    source_registry_url = None
    defaults = {"port": 8000}
    pre_exec: list = []
    post_exec: list = []
    post_commands: list = []
    layout = None
    stop_after_post = False
    executor = ""
    executor_config = None
    is_url_sourced = False
    qualified_name = "stub-recipe"
    name = "stub-recipe"
    container = "stub:latest"
    model_revision = None

    def __init__(self, cluster_config=None):
        self.cluster_config = cluster_config

    def build_config_chain(self, overrides=None):
        outer = self

        class _CC:
            def get(self, k, default=None):
                return (overrides or {}).get(k, outer.defaults.get(k, default))

        return _CC()

    def __getstate__(self):
        return {}


def _patch_launch(monkeypatch, tmp_path, captured):
    from sparkrun.core import launcher

    monkeypatch.setattr(
        "sparkrun.orchestration.distribution.resolve_auto_transfer_mode",
        lambda *a, **kw: type("R", (), {"mode": "local"})(),
    )

    def _dist(*a, **kw):
        captured["dist_args"] = a
        captured["dist_kwargs"] = kw
        return (None, {}, {})

    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", _dist)
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.save_job_metadata", lambda *a, **kw: None)
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.derive_cluster_id", lambda *a, **kw: "sparkrun_testabc12345")
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda *a, **kw: {})

    def _resolve_cache(cache_dir, *a, **kw):
        captured["remote_cache_in"] = cache_dir
        return cache_dir or str(tmp_path)

    monkeypatch.setattr(launcher, "resolve_effective_cache_dir", _resolve_cache)
    monkeypatch.setattr("sparkrun.orchestration.primitives.try_clear_page_cache", lambda *a, **kw: None)
    monkeypatch.setattr("sparkrun.orchestration.executor.resolve_executor", lambda **kw: type("Ex", (), {})())


def test_launch_inference_applies_cluster_config_overrides(monkeypatch, tmp_path):
    from sparkrun.core.launcher import launch_inference

    captured: dict = {}
    _patch_launch(monkeypatch, tmp_path, captured)

    class _Cfg2:
        hf_cache_dir = tmp_path / "hf"
        cache_dir = tmp_path / "cache"

        def get_registry_manager(self):
            return None

    recipe = _Recipe(
        cluster_config=ClusterConfig(
            remote_cache_dir="/nfs/remote",
            local_cache_dir="/nfs/local",
            resolved_model_path="/nfs/models/qwen3",
        )
    )
    overrides: dict = {}

    launch_inference(
        recipe=recipe,
        runtime=_StubRuntime(),
        host_list=["nv-host"],
        overrides=overrides,
        config=_Cfg2(),
        cluster=ClusterDefinition(name="t", hosts=["nv-host"]),
        is_solo=True,
        dry_run=True,
        sync_tuning=False,
    )

    # Remote cache override flowed into cache resolution.
    assert captured["remote_cache_in"] == "/nfs/remote"
    # distribute_from_config received the resolved remote cache (positional 3),
    # the local override, and skip_model=True.
    assert captured["dist_args"][3] == "/nfs/remote"
    assert captured["dist_kwargs"]["local_cache_dir"] == "/nfs/local"
    assert captured["dist_kwargs"]["skip_model"] is True
    # Non-mutating contract: the caller's recipe/overrides are left untouched
    # (launch_inference is the shared run/benchmark/proxy pipeline and the same
    # recipe object may be reused across launches).
    assert recipe.model == "Qwen/Qwen3-1.7B"
    assert "served_model_name" not in overrides
    # The serve-arg repoint + preserved served name happen on the copy handed
    # to the runtime, not on the caller's objects.
    run_kwargs = _StubRuntime.last_kwargs
    assert run_kwargs["recipe"].model == "/nfs/models/qwen3"
    assert run_kwargs["overrides"]["served_model_name"] == "Qwen/Qwen3-1.7B"
    assert run_kwargs["recipe"] is not recipe


def test_launch_inference_no_cluster_config_is_noop(monkeypatch, tmp_path):
    from sparkrun.core.launcher import launch_inference

    captured: dict = {}
    _patch_launch(monkeypatch, tmp_path, captured)

    class _Cfg2:
        hf_cache_dir = tmp_path / "hf"
        cache_dir = tmp_path / "cache"

        def get_registry_manager(self):
            return None

    recipe = _Recipe(cluster_config=None)
    overrides: dict = {}

    launch_inference(
        recipe=recipe,
        runtime=_StubRuntime(),
        host_list=["nv-host"],
        overrides=overrides,
        config=_Cfg2(),
        cluster=ClusterDefinition(name="t", hosts=["nv-host"]),
        is_solo=True,
        dry_run=True,
        sync_tuning=False,
    )

    assert captured["dist_kwargs"]["skip_model"] is False
    assert recipe.model == "Qwen/Qwen3-1.7B"
    assert "served_model_name" not in overrides


def test_launch_inference_cluster_disables_model_distribution(monkeypatch, tmp_path):
    """``distribution.model.enabled: false`` skips model distribution (no resolved path)."""
    from sparkrun.core.cluster_manager import ClusterDistributionConfig, ModelDistributionPrefs
    from sparkrun.core.launcher import launch_inference

    captured: dict = {}
    _patch_launch(monkeypatch, tmp_path, captured)

    class _Cfg2:
        hf_cache_dir = tmp_path / "hf"
        cache_dir = tmp_path / "cache"

        def get_registry_manager(self):
            return None

    recipe = _Recipe(cluster_config=None)
    cluster = ClusterDefinition(
        name="t",
        hosts=["nv-host"],
        distribution=ClusterDistributionConfig(model=ModelDistributionPrefs(enabled=False)),
    )

    launch_inference(
        recipe=recipe,
        runtime=_StubRuntime(),
        host_list=["nv-host"],
        overrides={},
        config=_Cfg2(),
        cluster=cluster,
        is_solo=True,
        dry_run=True,
        sync_tuning=False,
    )

    # Distribution disabled at the cluster level → skip_model, but model arg is
    # left untouched (no resolved_model_path repointing).
    assert captured["dist_kwargs"]["skip_model"] is True
    assert recipe.model == "Qwen/Qwen3-1.7B"
