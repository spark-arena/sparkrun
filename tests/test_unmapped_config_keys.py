"""Tests for the unmapped ``defaults:`` / ``-o`` key report (issue #276).

A structured runtime renders its serve command by iterating a flag map, so a
key the map does not list is *dropped* — no error, no warning, nothing in the
rendered command.  ``report_unmapped_config_keys`` is what makes that audible.
"""

import pytest

from sparkrun.core.launcher import report_unmapped_config_keys
from sparkrun.core.recipe import Recipe
from sparkrun.runtimes.atlas import AtlasRuntime
from sparkrun.runtimes.base import BASE_CONSUMED_CONFIG_KEYS, RuntimePlugin


def _recipe(**overrides) -> Recipe:
    base = {
        "name": "test-recipe",
        "model": "Sehyo/Qwen3.5-35B-A3B-NVFP4",
        "runtime": "atlas",
    }
    base.update(overrides)
    return Recipe.from_dict(base)


def _joined(recipe, runtime, overrides=None) -> str:
    return "\n".join(report_unmapped_config_keys(recipe, runtime, overrides))


# --- The reporting itself ---


def test_unmapped_recipe_default_is_reported():
    """The #276 shape: a misspelled correctness pin that reaches nothing."""
    report = _joined(_recipe(defaults={"lm_head_dytpe": "bf16"}), AtlasRuntime())

    assert "lm_head_dytpe" in report
    assert "atlas" in report


def test_mapped_recipe_default_is_not_reported():
    """A key the runtime does emit must stay silent, or the report is noise."""
    assert _joined(_recipe(defaults={"lm_head_dtype": "bf16"}), AtlasRuntime()) == ""


def test_unmapped_override_is_reported_separately():
    """An override was typed at this invocation, so it gets its own wording.

    A recipe default that does nothing is an inherited defect; a ``-o`` that
    does nothing is a failed instruction the user just gave.
    """
    messages = report_unmapped_config_keys(_recipe(), AtlasRuntime(), {"lm_head_dtpe": "bf16"})

    assert len(messages) == 1
    assert "-o lm_head_dtpe" in messages[0]
    assert "no effect" in messages[0]


def test_override_and_default_are_reported_once_each():
    """A key given both ways is reported as the override, not twice."""
    messages = report_unmapped_config_keys(
        _recipe(defaults={"bogus_key": "a"}),
        AtlasRuntime(),
        {"bogus_key": "b"},
    )

    assert len(messages) == 1
    assert "-o bogus_key" in messages[0]


# --- What must never be reported ---


@pytest.mark.parametrize("key", sorted(BASE_CONSUMED_CONFIG_KEYS))
def test_base_consumed_keys_are_never_reported(key):
    """Keys the shared machinery reads for every runtime are not "dropped"."""
    assert _joined(_recipe(defaults={key: 1}), AtlasRuntime()) == ""


def test_command_template_placeholder_is_not_reported():
    """A key referenced from ``command:`` is the documented escape hatch.

    RECIPES.md says runtime-specific keys are passed through to template
    substitution, so a flat flag-map diff would flag every one of them.
    """
    recipe = _recipe(
        defaults={"nonstandard_knob": 7},
        command="spark serve {model} --nonstandard-knob {nonstandard_knob}",
    )

    assert _joined(recipe, AtlasRuntime()) == ""


def test_placeholder_referenced_from_another_default_is_not_reported():
    """``render_template`` iterates, so one default may exist only to feed another."""
    recipe = _recipe(defaults={"metrics_port": 9090, "warmup_prompt": "ping-{metrics_port}"})

    assert _joined(recipe, AtlasRuntime()) == ""


@pytest.mark.parametrize("key", ["_gguf_model_path", "env.VLLM_USE_V1", "runtime_cache.enabled"])
def test_internal_and_namespaced_keys_are_not_reported(key):
    """Underscore-prefixed values are sparkrun-injected; dotted ones route by prefix."""
    assert _joined(_recipe(), AtlasRuntime(), {key: "x"}) == ""


# --- The opt-in contract ---


def test_runtime_that_does_not_declare_is_silent():
    """``None`` is the base default: an undeclared runtime reports nothing.

    A wrong key set is worse than no key set — it either cries wolf on a
    working recipe or restores exactly the silence being fixed.
    """

    class _Undeclared(RuntimePlugin):
        runtime_name = "undeclared"

        def generate_command(self, *args, **kwargs):  # pragma: no cover - unused
            return ""

    assert _Undeclared().known_config_keys() is None
    assert report_unmapped_config_keys(_recipe(defaults={"bogus_key": 1}), _Undeclared()) == []


def test_runtime_without_the_hook_is_tolerated():
    """An out-of-tree runtime built against an older base class must not break the launch."""

    class _Legacy:
        runtime_name = "legacy"

    assert report_unmapped_config_keys(_recipe(defaults={"bogus_key": 1}), _Legacy()) == []


def test_runtime_whose_hook_raises_is_tolerated():
    """The report is a diagnostic; a misbehaving runtime costs only the report."""

    class _Broken:
        runtime_name = "broken"

        def known_config_keys(self):
            raise RuntimeError("boom")

    assert report_unmapped_config_keys(_recipe(defaults={"bogus_key": 1}), _Broken()) == []
