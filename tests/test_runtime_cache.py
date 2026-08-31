"""Persistent compilation/autotune cache (issue #256).

Covers the settings chain, the host path keying, the generated
create/stamp/prune script, and the two env-tier invariants that would fail
*silently* if broken: HF paths surviving the ``XDG_CACHE_HOME`` catch-all, and
``recipe.env`` outranking the injected cache env.

See ``.slop/runtime-cache-design.md``.
"""

from __future__ import annotations

import pytest

from sparkrun.core.recipe import Recipe
from sparkrun.core.runtime_cache import (
    LAST_USED_MARKER,
    RUNTIME_CACHE_CONTAINER_PATH,
    RuntimeCacheSettings,
    build_runtime_cache_mounts,
    image_key,
    model_key,
    resolve_runtime_cache_root,
    resolve_runtime_cache_settings,
    sanitize_key_component,
)
from sparkrun.orchestration.runtime_cache import generate_runtime_cache_script
from sparkrun.runtimes.sglang import SglangRuntime
from sparkrun.runtimes.trtllm import TrtllmRuntime
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime


def _recipe(**extra) -> Recipe:
    d = {
        "recipe_version": "2",
        "name": "rc",
        "model": "Qwen/Qwen3-30B-A3B",
        "runtime": "vllm-distributed",
        "container": "vllm/vllm-openai:v0.11.0",
    }
    d.update(extra)
    return Recipe.from_dict(d)


class _Cfg:
    """Minimal stand-in for SparkrunConfig's runtime_cache surface."""

    def __init__(self, runtime_cache=None):
        self.runtime_cache = runtime_cache or {}


class _Cluster:
    def __init__(self, runtime_cache=None):
        self.runtime_cache = runtime_cache


# ---------------------------------------------------------------------------
# Settings chain
# ---------------------------------------------------------------------------


def test_baseline_defaults_are_on_model_keyed_not_image_keyed():
    s = resolve_runtime_cache_settings()
    assert s.enabled is True
    assert s.key_by_model is True
    assert s.key_by_image is False
    assert s.prune_enabled is True
    assert s.prune_max_age_days == 30


def test_config_layer_beats_baseline():
    s = resolve_runtime_cache_settings(config=_Cfg({"key_by_image": True, "prune": {"max_age_days": 7}}))
    assert s.key_by_image is True
    assert s.prune_max_age_days == 7


def test_cluster_beats_config():
    s = resolve_runtime_cache_settings(config=_Cfg({"enabled": True}), cluster=_Cluster({"enabled": False}))
    assert s.enabled is False


def test_cli_beats_cluster():
    s = resolve_runtime_cache_settings(cluster=_Cluster({"enabled": False}), cli_override={"enabled": True})
    assert s.enabled is True


def test_recipe_beats_cli():
    s = resolve_runtime_cache_settings(cli_override={"enabled": True}, recipe=_recipe(runtime_cache={"enabled": False}))
    assert s.enabled is False


def test_recipe_bool_shorthand():
    assert resolve_runtime_cache_settings(recipe=_recipe(runtime_cache=False)).enabled is False


def test_env_kill_switch_beats_every_layer():
    s = resolve_runtime_cache_settings(
        recipe=_recipe(runtime_cache={"enabled": True}),
        cli_override={"enabled": True},
        env_disabled=True,
    )
    assert s.enabled is False


def test_trtllm_runtime_default_opts_into_image_keying():
    """The one cache that cannot validate itself protects itself, per-runtime.

    The global default is off; a stable user launching a TRT-LLM recipe must
    still get image-keyed directories without editing anything.
    """
    assert resolve_runtime_cache_settings(runtime=TrtllmRuntime()).key_by_image is True
    assert resolve_runtime_cache_settings(runtime=VllmDistributedRuntime()).key_by_image is False


def test_config_beats_runtime_default():
    s = resolve_runtime_cache_settings(runtime=TrtllmRuntime(), config=_Cfg({"key_by_image": False}))
    assert s.key_by_image is False


def test_non_boolean_value_is_ignored_not_coerced():
    s = resolve_runtime_cache_settings(config=_Cfg({"enabled": "sometimes"}))
    assert s.enabled is True


def test_string_booleans_are_accepted():
    assert resolve_runtime_cache_settings(config=_Cfg({"enabled": "no"})).enabled is False


# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------


def test_sanitize_strips_unsafe_characters():
    out = sanitize_key_component("Qwen/Qwen3 30B;rm -rf /")
    assert "/" not in out and ";" not in out and " " not in out
    assert all(c.isalnum() or c in "._-" for c in out)


def test_sanitize_never_returns_empty():
    assert sanitize_key_component("///") == "unknown"
    assert sanitize_key_component("") == "unknown"


def test_model_key_is_readable_and_revision_sensitive():
    base = model_key(_recipe())
    assert base.startswith("Qwen__Qwen3-30B-A3B-")
    assert model_key(_recipe(model_revision="abc123")) != base


def test_model_key_distinguishes_gguf_quants():
    assert model_key(_recipe(model="org/m:Q4_K_M")) != model_key(_recipe(model="org/m:Q8_0"))


def test_image_key_prefers_identity_over_mutable_tag():
    """Two different images behind one ``:latest`` tag must not share a tree."""
    a = image_key("repo/img:latest", identity="sha256:aaa")
    b = image_key("repo/img:latest", identity="sha256:bbb")
    assert a != b
    # ...and without an identity the ref still separates distinct tags.
    assert image_key("repo/img:v1") != image_key("repo/img:v2")


# ---------------------------------------------------------------------------
# Mount construction
# ---------------------------------------------------------------------------


def _mounts(runtime=None, settings=None, **kw):
    return build_runtime_cache_mounts(
        runtime=runtime or VllmDistributedRuntime(),
        recipe=kw.pop("recipe", _recipe()),
        settings=settings or RuntimeCacheSettings(),
        root=kw.pop("root", "/home/u/.cache/sparkrun/runtime-cache"),
        image=kw.pop("image", "vllm/vllm-openai:v0.11.0"),
        **kw,
    )


def test_disabled_yields_no_plan_at_all():
    assert _mounts(settings=RuntimeCacheSettings(enabled=False)) is None


def test_default_layout_is_family_then_model():
    m = _mounts()
    assert m.family_root == "/home/u/.cache/sparkrun/runtime-cache/vllm"
    assert m.leaf == "%s/%s" % (m.family_root, model_key(_recipe()))


def test_image_key_inserts_a_level_above_the_model_key():
    """Enabling image-keying must not reshuffle the existing levels."""
    m = _mounts(settings=RuntimeCacheSettings(key_by_image=True), image_identity="sha256:aa")
    assert m.leaf == "%s/%s/%s" % (m.family_root, image_key("vllm/vllm-openai:v0.11.0", "sha256:aa"), model_key(_recipe()))


def test_unkeyed_leaf_is_the_family_root():
    m = _mounts(settings=RuntimeCacheSettings(key_by_model=False, key_by_image=False))
    assert m.leaf == m.family_root


def test_container_path_is_constant_regardless_of_keying():
    """All keying is host-side — a recipe never sees a key."""
    for settings in (
        RuntimeCacheSettings(),
        RuntimeCacheSettings(key_by_image=True),
        RuntimeCacheSettings(key_by_model=False, key_by_image=False),
    ):
        m = _mounts(settings=settings, image_identity="sha256:aa")
        assert list(m.volumes.values()) == [RUNTIME_CACHE_CONTAINER_PATH]
        assert m.env["XDG_CACHE_HOME"] == RUNTIME_CACHE_CONTAINER_PATH


def test_vllm_declares_its_explicit_cache_vars():
    env = _mounts().env
    assert env["VLLM_CACHE_ROOT"] == "/cache/runtime/vllm"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/cache/runtime/inductor"
    assert env["TRITON_CACHE_DIR"] == "/cache/runtime/triton"


def test_flashinfer_workspace_base_is_declared_because_cache_dir_is_inert():
    """``FLASHINFER_CACHE_DIR`` is not an env var flashinfer reads.

    In ``flashinfer/jit/env.py`` it is a module *attribute* computed as
    ``FLASHINFER_WORKSPACE_BASE / ".cache" / "flashinfer"``, the base defaulting
    to ``Path.home()`` — verified against flashinfer 0.6.11 and 0.6.18.  So
    declaring only ``FLASHINFER_CACHE_DIR`` left every JIT-compiled kernel in
    the ``--rm`` container's throwaway ``HOME`` and recompiled it next launch.

    Both are asserted: the base is the lever that works, the cache dir is kept
    against a later release honoring it.  Pointing them at one subtree is
    deliberate — the base grows ``.cache/flashinfer/<ver>/<arch>/`` beneath it.
    """
    for runtime in (VllmDistributedRuntime(), SglangRuntime()):
        m = _mounts(runtime=runtime)
        assert m.env["FLASHINFER_WORKSPACE_BASE"] == "/cache/runtime/flashinfer"
        assert m.env["FLASHINFER_CACHE_DIR"] == "/cache/runtime/flashinfer"
        assert "%s/flashinfer" % m.leaf in m.dirs


def test_cute_dsl_cache_is_declared_because_it_defaults_into_tmpdir():
    """NVIDIA's CuTeDSL generated-IR cache reaches neither XDG nor ``HOME``.

    ``nvidia_cutlass_dsl``'s ``get_default_generated_ir_path()`` falls back to
    ``$TMPDIR/<user>/cutlass_python_cache``, so the XDG catch-all cannot cover
    it.  vLLM's ``vllm_flash_attn.cute``, FlashInfer's sparse kernels and the
    b12x kernel stack all compile through this DSL.
    """
    for runtime in (VllmDistributedRuntime(), SglangRuntime()):
        m = _mounts(runtime=runtime)
        assert m.env["CUTE_DSL_CACHE_DIR"] == "/cache/runtime/cute_dsl"
        assert m.env["FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"] == "/cache/runtime/flash_attn_cute_dsl"


def test_b12x_compile_cache_rides_the_xdg_catch_all():
    """b12x resolves ``B12X_COMPILE_CACHE_DIR`` → ``$XDG_CACHE_HOME/b12x/compile``.

    No explicit entry is needed, but only as long as the catch-all is set — the
    b12x images are what the ``eugr`` / ``@official-recipes`` DeepSeek-V4 Flash
    recipes run on, and their compile cache is minutes per launch.
    """
    m = _mounts()
    assert m.env["XDG_CACHE_HOME"] == RUNTIME_CACHE_CONTAINER_PATH
    assert "B12X_COMPILE_CACHE_DIR" not in m.env


def test_sglang_declares_inductor_and_triton():
    env = _mounts(runtime=SglangRuntime()).env
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/cache/runtime/inductor"
    assert env["TRITON_CACHE_DIR"] == "/cache/runtime/triton"


def test_sglang_declares_its_own_cache_root_not_just_flashinfers():
    """``SGLANG_CACHE_DIR`` is where SGLang writes its FlashInfer autotune results.

    Distinct from ``FLASHINFER_CACHE_DIR`` (FlashInfer's own JIT cubins), and not
    reachable via ``XDG_CACHE_HOME`` — the upstream default is a literal
    ``expanduser("~/.cache/sglang")``, so leaving it unset re-ran the autotune
    sweep into the container's throwaway ``HOME`` on every launch.
    """
    m = _mounts(runtime=SglangRuntime())
    assert m.env["SGLANG_CACHE_DIR"] == "/cache/runtime/sglang"
    assert m.env["FLASHINFER_CACHE_DIR"] == "/cache/runtime/flashinfer"
    assert "%s/sglang" % m.leaf in m.dirs


def test_sglang_jit_cache_is_declared_because_it_does_not_track_the_root():
    """Unset, SGLang's JIT build cache falls back to a hardcoded ``~/.cache/sglang/jit``.

    Setting ``SGLANG_CACHE_DIR`` alone would leave it in the container.
    """
    m = _mounts(runtime=SglangRuntime())
    assert m.env["SGLANG_JIT_CACHE_DIR"] == "/cache/runtime/sglang/jit"
    assert "%s/sglang/jit" % m.leaf in m.dirs


def test_sglang_declares_tilelang_cache():
    """TileLang defaults to ``~/.tilelang/cache`` — outside XDG *and* the SGLang root."""
    assert _mounts(runtime=SglangRuntime()).env["TILELANG_CACHE_DIR"] == "/cache/runtime/tilelang"


def test_declared_directories_are_created():
    m = _mounts()
    assert m.leaf in m.dirs
    assert "%s/inductor" % m.leaf in m.dirs
    assert "%s/triton" % m.leaf in m.dirs


def test_runtime_declaring_nothing_still_gets_the_xdg_catch_all():
    class _Bare(VllmDistributedRuntime):
        def runtime_cache_paths(self, *, fingerprint=""):
            return {}

    m = _mounts(runtime=_Bare())
    assert m.env == {"XDG_CACHE_HOME": RUNTIME_CACHE_CONTAINER_PATH}
    assert m.dirs == [m.leaf]


# --- TRT-LLM autotuner: a file, keyed by fingerprint ---


def test_trtllm_autotuner_is_a_file_keyed_by_fingerprint():
    m = _mounts(runtime=TrtllmRuntime(), fingerprint="deadbeef")
    assert m.env["TLLM_AUTOTUNER_CACHE_PATH"] == "/cache/runtime/autotune/deadbeef.cache"
    # Only the *parent* is created — the runtime writes the file itself.
    assert "%s/autotune" % m.leaf in m.dirs
    assert "%s/autotune/deadbeef.cache" % m.leaf not in m.dirs


def test_trtllm_autotuner_filename_varies_with_fingerprint_at_a_fixed_dir_key():
    """The design invariant: per-artifact correctness never rides on the dir key.

    Two configurations of the same model land in the same directory (model-keyed)
    but must not share the autotuner file, because it validates nothing.
    """
    a = _mounts(runtime=TrtllmRuntime(), fingerprint="aaa")
    b = _mounts(runtime=TrtllmRuntime(), fingerprint="bbb")
    assert a.leaf == b.leaf
    assert a.env["TLLM_AUTOTUNER_CACHE_PATH"] != b.env["TLLM_AUTOTUNER_CACHE_PATH"]


def test_trtllm_without_a_fingerprint_omits_the_autotuner_pointer():
    """Better no cache than one file shared by every configuration."""
    assert "TLLM_AUTOTUNER_CACHE_PATH" not in _mounts(runtime=TrtllmRuntime(), fingerprint="").env


def test_a_broken_runtime_hook_degrades_to_xdg_only():
    class _Broken(VllmDistributedRuntime):
        def runtime_cache_paths(self, *, fingerprint=""):
            raise RuntimeError("boom")

    m = _mounts(runtime=_Broken())
    assert m.env == {"XDG_CACHE_HOME": RUNTIME_CACHE_CONTAINER_PATH}


def test_explicit_dir_setting_wins_over_the_probed_root():
    assert resolve_runtime_cache_root(RuntimeCacheSettings(dir="/mnt/fast/rc"), "/home/u/.cache/sparkrun") == "/mnt/fast/rc"


def test_root_defaults_under_the_sparkrun_cache_dir():
    assert resolve_runtime_cache_root(RuntimeCacheSettings(), "/home/u/.cache/sparkrun") == "/home/u/.cache/sparkrun/runtime-cache"


# ---------------------------------------------------------------------------
# Generated script: create + stamp + prune
# ---------------------------------------------------------------------------


def test_script_creates_dirs_and_stamps_the_marker():
    m = _mounts()
    script = generate_runtime_cache_script(m)
    assert "mkdir -p" in script
    assert m.leaf in script
    assert LAST_USED_MARKER in script


def test_script_prunes_siblings_but_never_the_active_leaf():
    script = generate_runtime_cache_script(_mounts())
    assert "rm -rf" in script
    assert "-mtime +30" in script
    # The just-touched tree is skipped explicitly, not merely by its fresh mtime.
    assert "continue" in script


def test_prune_report_line_carries_the_path():
    """``ensure_runtime_cache_on_hosts`` re-logs these lines, so a literal
    ``%s`` here would report every prune without saying what was deleted."""
    script = generate_runtime_cache_script(_mounts())
    assert "printf 'runtime-cache: pruned %s\\n' \"$_tree\"" in script
    assert "%%s" not in script


def test_prune_ages_by_the_marker_not_the_directory():
    """Reading a cache never touches the directory, so mtime-aging the *tree*
    would delete exactly the warm caches it should keep."""
    script = generate_runtime_cache_script(_mounts())
    assert '"$_marker" -maxdepth 0 -mtime' in script
    assert '"$_tree" -maxdepth 0 -mtime' not in script


def test_no_prune_stanza_when_pruning_is_disabled():
    m = _mounts(settings=RuntimeCacheSettings(prune_enabled=False))
    assert "rm -rf" not in generate_runtime_cache_script(m)


def test_no_prune_stanza_when_unkeyed():
    """With no key components the leaf *is* the family root — it has no siblings,
    and a sweep there would delete other runtimes' trees."""
    m = _mounts(settings=RuntimeCacheSettings(key_by_model=False, key_by_image=False))
    assert "rm -rf" not in generate_runtime_cache_script(m)


def test_prune_refuses_a_shallow_root():
    """Defense in depth around the one ``rm -rf`` in the codebase."""
    m = build_runtime_cache_mounts(
        runtime=VllmDistributedRuntime(),
        recipe=_recipe(),
        settings=RuntimeCacheSettings(),
        root="",  # → family_root == "/vllm", far too shallow to sweep
    )
    assert "rm -rf" not in generate_runtime_cache_script(m)


def test_script_is_shell_safe_for_hostile_model_names():
    m = _mounts(recipe=_recipe(model="org/m$(touch /tmp/pwned);x"))
    script = generate_runtime_cache_script(m)
    assert "touch /tmp/pwned" not in script
    assert "$(" not in script.replace("$(find", "")


# ---------------------------------------------------------------------------
# Env tiers — the two silent-failure invariants
# ---------------------------------------------------------------------------


def _solo_env(runtime, recipe, runtime_cache, recipe_env=None):
    """Reproduce ``_run_solo``'s merge order without launching anything."""
    from sparkrun.utils import merge_env

    return merge_env(
        runtime_cache.env if runtime_cache else {},
        runtime.get_common_env(),
        runtime.get_solo_env(),
        recipe_env or recipe.env,
        runtime.get_extra_env(),
    )


def test_hf_paths_survive_the_xdg_catch_all():
    """``huggingface_hub`` honors XDG_CACHE_HOME.  If HF_HOME/HF_HUB_CACHE ever
    stop outranking the cache env, the model cache silently relocates off its
    own mount and every launch re-downloads the weights."""
    runtime = VllmDistributedRuntime()
    env = _solo_env(runtime, _recipe(), _mounts())
    assert env["HF_HOME"] == "/cache/huggingface"
    assert env["HF_HUB_CACHE"] == "/cache/huggingface/hub"
    assert env["XDG_CACHE_HOME"] == RUNTIME_CACHE_CONTAINER_PATH


def test_recipe_env_beats_the_injected_cache_env():
    """The cache env is the *lowest* tier.  Routing it through ``get_extra_env``
    — which wins — would clobber a recipe that points VLLM_CACHE_ROOT itself."""
    runtime = VllmDistributedRuntime()
    env = _solo_env(runtime, _recipe(), _mounts(), recipe_env={"VLLM_CACHE_ROOT": "/mine"})
    assert env["VLLM_CACHE_ROOT"] == "/mine"


def test_disabled_cache_leaves_the_env_untouched():
    runtime = VllmDistributedRuntime()
    with_cache = _solo_env(runtime, _recipe(), _mounts())
    without = _solo_env(runtime, _recipe(), None)
    assert "XDG_CACHE_HOME" not in without
    assert set(with_cache) - set(without) == {
        "XDG_CACHE_HOME",
        "VLLM_CACHE_ROOT",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "FLASHINFER_CACHE_DIR",
        "FLASHINFER_WORKSPACE_BASE",
        "CUTE_DSL_CACHE_DIR",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
    }


# ---------------------------------------------------------------------------
# The two volume chokepoints (solo + cluster) must agree
# ---------------------------------------------------------------------------


def _solo_volumes(recipe, runtime_cache):
    """Reproduce ``_run_solo``'s volume assembly."""
    from sparkrun.orchestration.primitives import build_volumes, resolved_model_volume

    runtime = VllmDistributedRuntime()
    return build_volumes(
        "/hf",
        extra={
            **(runtime_cache.volumes if runtime_cache else {}),
            **runtime.get_extra_volumes(),
            **resolved_model_volume(recipe),
        },
    )


def test_solo_path_mounts_the_cache_alongside_the_hf_cache():
    m = _mounts()
    volumes = _solo_volumes(_recipe(), m)
    assert volumes["/hf"] == "/cache/huggingface"
    assert volumes[m.leaf] == RUNTIME_CACHE_CONTAINER_PATH


def test_solo_path_without_a_cache_is_unchanged():
    assert _solo_volumes(_recipe(), None) == {"/hf": "/cache/huggingface"}


def test_cluster_context_mounts_and_env_match_the_solo_path():
    """The two chokepoints are separate code; a divergence would mean a cluster
    launch silently loses the cache a solo launch of the same recipe keeps."""
    from sparkrun.runtimes._cluster_ops import ClusterContext

    recipe, m = _recipe(), _mounts()
    ctx = ClusterContext.build(
        VllmDistributedRuntime(),
        ["h1", "h2"],
        "img:v1",
        "cid",
        recipe.env,
        "/hf",
        None,
        True,
        recipe=recipe,
        runtime_cache=m,
    )
    assert ctx.volumes[m.leaf] == RUNTIME_CACHE_CONTAINER_PATH
    assert ctx.volumes["/hf"] == "/cache/huggingface"
    assert ctx.all_env["VLLM_CACHE_ROOT"] == "/cache/runtime/vllm"
    # Same HF invariant as the solo path.
    assert ctx.all_env["HF_HOME"] == "/cache/huggingface"
    assert ctx.all_env["HF_HUB_CACHE"] == "/cache/huggingface/hub"


def test_cluster_context_without_a_cache_is_unchanged():
    from sparkrun.runtimes._cluster_ops import ClusterContext

    recipe = _recipe()
    ctx = ClusterContext.build(
        VllmDistributedRuntime(), ["h1"], "img:v1", "cid", recipe.env, "/hf", None, True, recipe=recipe, runtime_cache=None
    )
    assert ctx.volumes == {"/hf": "/cache/huggingface"}
    assert "XDG_CACHE_HOME" not in ctx.all_env


# ---------------------------------------------------------------------------
# Executor seam
# ---------------------------------------------------------------------------


def test_base_executor_ensure_runtime_cache_is_a_safe_no_op():
    """A provider executor that never overrides must not break a launch."""
    from sparkrun.orchestration.executors.k8s import K8sExecutor

    assert K8sExecutor().ensure_runtime_cache(_mounts(), ["h1"]) is None


def test_host_executors_prepare_the_cache_over_ssh(monkeypatch):
    from sparkrun.orchestration.executors.docker import DockerExecutor
    from sparkrun.orchestration.executors.local import LocalExecutor

    for executor_cls in (DockerExecutor, LocalExecutor):
        seen = {}
        monkeypatch.setattr(
            "sparkrun.orchestration.runtime_cache.run_remote_scripts_parallel",
            lambda hosts, script, **kw: seen.update(hosts=hosts, script=script, kw=kw) or [],
        )
        m = _mounts()
        executor_cls().ensure_runtime_cache(m, ["h1", "h2"])
        assert seen["hosts"] == ["h1", "h2"]
        assert m.leaf in seen["script"]
        # SSH-to-self must work without a self-SSH mesh configured.
        assert seen["kw"]["allow_local"] is True


def test_preparation_failure_never_raises(monkeypatch):
    """Best-effort by contract: a cache we couldn't prepare costs a recompile."""
    from sparkrun.orchestration.executors.docker import DockerExecutor

    def _boom(*a, **kw):
        raise OSError("ssh exploded")

    monkeypatch.setattr("sparkrun.orchestration.runtime_cache.run_remote_scripts_parallel", _boom)
    assert DockerExecutor().ensure_runtime_cache(_mounts(), ["h1"]) is None


# ---------------------------------------------------------------------------
# Local executor: container paths must reverse-map to host paths
# ---------------------------------------------------------------------------


def test_local_executor_rewrites_cache_env_to_host_paths():
    from sparkrun.orchestration.executors.local import _hostify_env

    m = _mounts()
    hostified = _hostify_env(dict(m.env), dict(m.volumes))
    assert hostified["XDG_CACHE_HOME"] == m.leaf
    assert hostified["VLLM_CACHE_ROOT"] == "%s/vllm" % m.leaf


# ---------------------------------------------------------------------------
# Recipe round-trip
# ---------------------------------------------------------------------------


def test_recipe_runtime_cache_round_trips_and_is_not_swept_into_runtime_config():
    recipe = _recipe(runtime_cache={"key_by_image": True})
    assert recipe.runtime_cache == {"key_by_image": True}
    assert "runtime_cache" not in recipe.runtime_config

    restored = Recipe._deserialize(recipe.__getstate__())
    assert restored.runtime_cache == {"key_by_image": True}
    assert recipe.to_dict()["runtime_cache"] == {"key_by_image": True}


@pytest.mark.parametrize("value", [{}, None])
def test_recipe_without_runtime_cache_omits_it_from_export(value):
    recipe = _recipe() if value is None else _recipe(runtime_cache=value)
    assert "runtime_cache" not in recipe.to_dict()


# ---------------------------------------------------------------------------
# Launcher wiring — the plan must actually reach runtime.run()
# ---------------------------------------------------------------------------


class _StubRuntime:
    """Records ``run()`` kwargs; declares just enough for the cache hooks."""

    runtime_name = "vllm-distributed"
    requires_capability: frozenset = frozenset()
    last_kwargs: dict = {}

    def get_family(self):
        return "vllm"

    def runtime_cache_paths(self, *, fingerprint=""):
        from sparkrun.core.runtime_cache import CachePath

        return {"TRITON_CACHE_DIR": CachePath("triton")}

    def runtime_cache_defaults(self):
        return {}

    def is_delegating_runtime(self):
        return False

    def resolve_container(self, recipe, overrides=None):
        return "stub:latest"

    def prepare(self, *args, **kwargs):
        return None

    def get_head_container_name(self, cluster_id, is_solo=False):
        return "%s_solo" % cluster_id

    def generate_command(self, **kwargs):
        return "echo serve"

    def resolve_api_key(self, recipe, overrides=None):
        return None

    def _collect_runtime_info(self, *args, **kwargs):
        return {}

    def run(self, **kwargs):
        type(self).last_kwargs = dict(kwargs)
        return 0


def _launch(monkeypatch, tmp_path, **launch_kw):
    from sparkrun.core import launcher
    from sparkrun.core.launcher import launch_inference

    monkeypatch.setattr(
        "sparkrun.orchestration.distribution.resolve_auto_transfer_mode",
        lambda *a, **kw: type("R", (), {"mode": "local"})(),
    )
    monkeypatch.setattr("sparkrun.orchestration.distribution.distribute_from_config", lambda *a, **kw: (None, {}, {}, {}))
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.save_job_metadata", lambda *a, **kw: None)
    monkeypatch.setattr("sparkrun.orchestration.job_metadata.derive_cluster_id", lambda *a, **kw: "sparkrun_testabc12345")
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda *a, **kw: {})
    monkeypatch.setattr(launcher, "resolve_effective_cache_dir", lambda *a, **kw: str(tmp_path / "hf"))
    monkeypatch.setattr(launcher, "resolve_effective_runtime_cache_dir", lambda *a, **kw: str(tmp_path / "sparkrun"))
    monkeypatch.setattr("sparkrun.orchestration.primitives.try_clear_page_cache", lambda *a, **kw: None)
    monkeypatch.setattr("sparkrun.orchestration.executor.resolve_executor", lambda **kw: type("Ex", (), {})())

    class _Cfg:
        hf_cache_dir = tmp_path / "hf"
        cache_dir = tmp_path / "sparkrun"
        runtime_cache: dict = {}

        def get_registry_manager(self):
            return None

    runtime = _StubRuntime()
    runtime.last_kwargs = {}
    launch_inference(
        recipe=launch_kw.pop("recipe", _recipe()),
        runtime=runtime,
        host_list=["h1"],
        overrides={},
        config=_Cfg(),
        is_solo=True,
        dry_run=True,
        sync_tuning=False,
        **launch_kw,
    )
    return _StubRuntime.last_kwargs


def test_launcher_threads_the_cache_plan_to_runtime_run(monkeypatch, tmp_path):
    mounts = _launch(monkeypatch, tmp_path)["runtime_cache"]
    assert mounts is not None
    assert mounts.leaf.startswith(str(tmp_path / "sparkrun" / "runtime-cache" / "vllm"))
    assert mounts.env["TRITON_CACHE_DIR"] == "/cache/runtime/triton"
    assert mounts.volumes == {mounts.leaf: RUNTIME_CACHE_CONTAINER_PATH}


def test_launcher_honors_the_cli_off_switch(monkeypatch, tmp_path):
    assert _launch(monkeypatch, tmp_path, runtime_cache_override={"enabled": False})["runtime_cache"] is None


def test_launcher_honors_the_recipe_off_switch(monkeypatch, tmp_path):
    kwargs = _launch(monkeypatch, tmp_path, recipe=_recipe(runtime_cache={"enabled": False}))
    assert kwargs["runtime_cache"] is None


def test_launcher_honors_the_env_kill_switch(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKRUN_NO_RUNTIME_CACHE", "1")
    assert _launch(monkeypatch, tmp_path)["runtime_cache"] is None


def test_dry_run_never_touches_the_hosts(monkeypatch, tmp_path):
    """A dry run must stay read-only — no mkdir, no marker, no prune."""
    called = []
    monkeypatch.setattr(
        "sparkrun.orchestration.runtime_cache.run_remote_scripts_parallel",
        lambda *a, **kw: called.append(a) or [],
    )
    assert _launch(monkeypatch, tmp_path)["runtime_cache"] is not None
    assert called == []


def test_image_identity_is_probed_only_when_image_keying_is_on(monkeypatch, tmp_path):
    """The probe is an SSH round-trip; the shared-tree default has no use for it."""
    probes = []
    monkeypatch.setattr(
        "sparkrun.core.runtime_cache.probe_image_identity",
        lambda image, hosts, ssh_kwargs, dry_run=False: probes.append(image) or "sha256:abc",
    )
    _launch(monkeypatch, tmp_path)
    assert probes == []

    _launch(monkeypatch, tmp_path, recipe=_recipe(runtime_cache={"key_by_image": True}))
    assert probes == ["stub:latest"]


def test_image_identity_separates_a_repulled_mutable_tag(monkeypatch, tmp_path):
    """A ``:latest`` re-pull must not inherit the previous image's cache — the
    exact staleness issue #256 reports, and why trtllm turns image-keying on."""
    identities = iter(["sha256:old", "sha256:new"])
    monkeypatch.setattr(
        "sparkrun.core.runtime_cache.probe_image_identity",
        lambda *a, **kw: next(identities),
    )
    recipe = _recipe(runtime_cache={"key_by_image": True})
    first = _launch(monkeypatch, tmp_path, recipe=recipe)["runtime_cache"].leaf
    second = _launch(monkeypatch, tmp_path, recipe=recipe)["runtime_cache"].leaf
    assert first != second


def test_unresolvable_image_identity_falls_back_to_the_ref(monkeypatch, tmp_path):
    monkeypatch.setattr("sparkrun.core.runtime_cache.probe_image_identity", lambda *a, **kw: None)
    m = _launch(monkeypatch, tmp_path, recipe=_recipe(runtime_cache={"key_by_image": True}))["runtime_cache"]
    assert m is not None and image_key("stub:latest") in m.leaf


def test_image_identity_probe_is_skipped_on_dry_run_and_never_raises(monkeypatch):
    from sparkrun.core.runtime_cache import probe_image_identity

    assert probe_image_identity("img:v1", ["h1"], {}, dry_run=True) is None
    monkeypatch.setattr(
        "sparkrun.containers.distribute._check_remote_image_identities",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("ssh down")),
    )
    assert probe_image_identity("img:v1", ["h1"], {}) is None
