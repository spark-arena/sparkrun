"""Tests for ``derive_recipe_fingerprint`` — the serve-configuration digest.

The fingerprint is the provenance peer of ``generate_intent_id``: the intent_id
stays narrow (runtime / model / port / served-model-name / parallelism) so
lookup paths keep matching a live workload, while the fingerprint separates two
recipes that share an intent but serve *different* configurations.

Two invariants are load-bearing and each has a failure mode that shipped:

* **Distinguishing** — recipes differing only in a serve argument must not
  collide, or a benchmark resumes into another recipe's results (issue #232).
* **Stability** — the digest must not move for reasons unrelated to declared
  configuration (auto-detected metadata, runtime-injected hooks), or a resume
  silently misses its own prior state.
"""

from __future__ import annotations

from sparkrun.benchmarking.run_state import derive_benchmark_id
from sparkrun.core.recipe import Recipe
from sparkrun.orchestration.job_metadata import (
    RECIPE_FINGERPRINT_LEN,
    derive_recipe_fingerprint,
    generate_cluster_id,
    generate_intent_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _recipe(**changes) -> Recipe:
    """A minimal v2 recipe; ``changes`` are merged over the base document."""
    data = {
        "recipe_version": "2",
        "model": "Qwen/Qwen3-1.7B",
        "runtime": "vllm",
        "container": "vllm/vllm-openai:latest",
        "defaults": {"port": 8000, "tensor_parallel": 1, "max_num_batched_tokens": 16384},
        "metadata": {"description": "demo"},
    }
    defaults = dict(data["defaults"])
    defaults.update(changes.pop("defaults", {}))
    data.update(changes)
    data["defaults"] = defaults
    return Recipe(data)


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------


def test_fingerprint_format():
    fp = derive_recipe_fingerprint(_recipe())
    assert len(fp) == RECIPE_FINGERPRINT_LEN
    assert all(c in "0123456789abcdef" for c in fp)


def test_fingerprint_is_deterministic():
    assert derive_recipe_fingerprint(_recipe()) == derive_recipe_fingerprint(_recipe())


# ---------------------------------------------------------------------------
# Distinguishing: same intent, different serve configuration (issue #232)
# ---------------------------------------------------------------------------


def test_serve_arg_difference_changes_fingerprint():
    """The #232 repro: two recipes differing ONLY in max_num_batched_tokens.

    They share an intent — same runtime, model, port, parallelism — so the
    intent_id alone cannot tell them apart.
    """
    a = _recipe(defaults={"max_num_batched_tokens": 16384})
    b = _recipe(defaults={"max_num_batched_tokens": 32768})

    assert generate_intent_id(a) == generate_intent_id(b), "precondition: the intent_id does NOT separate these"
    assert derive_recipe_fingerprint(a) != derive_recipe_fingerprint(b)


def test_serve_arg_difference_changes_benchmark_id():
    """End-to-end: the #232 recipes must derive distinct benchmark IDs.

    Same cluster_id (they place identically), same framework/profile/schedule —
    only the fingerprint separates them.  This is the assertion that would have
    caught the original defect.
    """
    a = _recipe(defaults={"max_num_batched_tokens": 16384})
    b = _recipe(defaults={"max_num_batched_tokens": 32768})
    cluster_id = generate_cluster_id(generate_intent_id(a), "abcdef012345")

    id_a = derive_benchmark_id(cluster_id, "llama-benchy", "default", {}, None, recipe_fingerprint=derive_recipe_fingerprint(a))
    id_b = derive_benchmark_id(cluster_id, "llama-benchy", "default", {}, None, recipe_fingerprint=derive_recipe_fingerprint(b))
    assert id_a != id_b


def test_cli_override_changes_fingerprint():
    """A serve argument supplied as a CLI override counts as configuration."""
    r = _recipe()
    assert derive_recipe_fingerprint(r) != derive_recipe_fingerprint(r, {"max_num_batched_tokens": 32768})


def test_hardcoded_command_flag_changes_fingerprint():
    """A serve flag hardcoded into ``command`` — not declared under ``defaults``
    — never reaches the config chain, so the command template itself must be
    hashed or the two recipes collide.
    """
    a = _recipe(command="vllm serve {model} --max-num-batched-tokens 16384")
    b = _recipe(command="vllm serve {model} --max-num-batched-tokens 32768")

    assert generate_intent_id(a) == generate_intent_id(b), "precondition: the intent_id does NOT separate these"
    assert derive_recipe_fingerprint(a) != derive_recipe_fingerprint(b)


def test_container_env_and_revision_change_fingerprint():
    base = derive_recipe_fingerprint(_recipe())
    assert derive_recipe_fingerprint(_recipe(container="vllm/vllm-openai:v0.11.0")) != base
    assert derive_recipe_fingerprint(_recipe(env={"VLLM_USE_V1": "1"})) != base
    assert derive_recipe_fingerprint(_recipe(model_revision="abc123")) != base


def test_v1_mods_and_build_args_change_fingerprint():
    """v1 ``mods`` are injected into ``pre_exec`` during resolution and
    ``build_args`` lands in ``runtime_config``; both change the image that
    gets built, so both are hashed from their declared form.
    """
    base = derive_recipe_fingerprint(_recipe())
    assert derive_recipe_fingerprint(_recipe(mods=["some-mod"])) != base
    assert derive_recipe_fingerprint(_recipe(build_args={"VLLM_COMMIT": "abc123"})) != base


def test_declared_hooks_change_fingerprint():
    """Declared hooks alter what is measured, so they belong in the digest."""
    assert derive_recipe_fingerprint(_recipe(post_commands=["echo warmup"])) != derive_recipe_fingerprint(_recipe())


# ---------------------------------------------------------------------------
# Stability: things that must NOT move the digest
# ---------------------------------------------------------------------------


def test_auto_detected_metadata_does_not_change_fingerprint():
    """``Recipe.estimate_vram(auto_detect=True)`` writes HuggingFace-probed
    facts back into ``recipe.metadata`` mid-run (core/recipe.py), and the
    benchmark flow triggers that before deriving the ID.  Hashing metadata
    would make the digest depend on network reachability — the same recipe
    would resume against itself only when the probe happened to succeed.
    """
    r = _recipe()
    before = derive_recipe_fingerprint(r)
    r.metadata["model_params"] = 1_720_000_000
    r.metadata["model_dtype"] = "bfloat16"
    r.metadata["num_layers"] = 28
    assert derive_recipe_fingerprint(r) == before


def test_runtime_injected_hooks_do_not_change_fingerprint():
    """v1 mods / builders extend ``recipe.pre_exec`` in place during resolution
    (core/mods.py).  Those are resolved artifacts, not declared configuration.
    """
    r = _recipe()
    before = derive_recipe_fingerprint(r)
    r.pre_exec.append("docker build -t generated .")
    assert derive_recipe_fingerprint(r) == before


def test_nested_dict_ordering_does_not_change_fingerprint():
    """A dict-valued serve argument must hash by content, not insertion order."""
    a = _recipe(defaults={"speculative_config": {"method": "ngram", "num_speculative_tokens": 5}})
    b = _recipe(defaults={"speculative_config": {"num_speculative_tokens": 5, "method": "ngram"}})
    assert derive_recipe_fingerprint(a) == derive_recipe_fingerprint(b)


def test_metadata_only_recipes_share_a_fingerprint():
    """Two recipes differing only in documentation metadata are the same workload."""
    a = _recipe(metadata={"description": "first"})
    b = _recipe(metadata={"description": "second", "maintainer": "someone"})
    assert derive_recipe_fingerprint(a) == derive_recipe_fingerprint(b)


def test_registry_provenance_does_not_change_fingerprint():
    """A registry reference and its local YAML identify the same workload."""
    recipe = _recipe()
    local = derive_recipe_fingerprint(recipe)

    recipe.source_registry = "example"
    recipe.source_registry_url = "https://github.com/example/recipes.git"
    recipe._qualified_name_override = "@example/qwen"

    assert derive_recipe_fingerprint(recipe) == local


def test_implicit_model_revision_template_variable_preserves_fingerprint():
    """Template-variable injection must not invalidate published identities.

    ``model_revision`` was historically hashed as a top-level attribute.  It
    later became an implicit config-chain entry so command templates could use
    ``{model_revision}``; that second representation must not move the digest.
    """
    recipe = _recipe(model_revision="abc123")
    assert recipe.build_config_chain().get("model_revision") == "abc123"

    class LegacyRecipe(Recipe):
        def build_config_chain(self, overrides=None, user_config=None):
            chain = super().build_config_chain(overrides, user_config)
            return {key: chain.get(key) for key in chain.keys() if key != "model_revision"}

    legacy = LegacyRecipe(_recipe(model_revision="abc123")._raw)

    assert derive_recipe_fingerprint(recipe) == derive_recipe_fingerprint(legacy)


def test_explicit_model_revision_override_still_changes_fingerprint():
    recipe = _recipe(model_revision="abc123")

    assert derive_recipe_fingerprint(recipe) != derive_recipe_fingerprint(
        recipe,
        {"model_revision": "def456"},
    )


# ---------------------------------------------------------------------------
# Host-dependence: why the digest must be derived before the launch
# ---------------------------------------------------------------------------


def test_platform_flag_defaults_move_the_fingerprint():
    """The reason ``api.plan`` derives it rather than ``save_job_metadata``.

    ``launch_inference`` calls ``apply_platform_runtime_flag_defaults``, which
    ``setdefault``s platform flags into ``recipe.defaults`` keyed off the *head
    host's* hardware — before it persists job metadata.  So a digest taken
    inside the launcher depends on where the job landed, and no caller can
    reproduce the value its own job was stored under without probing the same
    hardware.  Matching a job by fingerprint after the fact then silently never
    matches.
    """
    import pytest

    from sparkrun.core.hardware import default_dgx_spark_hardware
    from sparkrun.core.launcher import apply_platform_runtime_flag_defaults

    # llama-cpp on GB10 is the case that exists today (``mmap: False``); the
    # rule is about the mechanism, not this particular flag.
    r = _recipe(runtime="llama-cpp")
    declared = derive_recipe_fingerprint(r)

    applied = apply_platform_runtime_flag_defaults(r, "llama-cpp", default_dgx_spark_hardware())
    if not applied:
        pytest.skip("this platform contributes no runtime-flag defaults for llama-cpp")

    assert derive_recipe_fingerprint(r) != declared, (
        "platform defaults changed recipe.defaults without moving the digest; "
        "if this ever holds, the pre-launch derivation is no longer needed"
    )
