"""Declarative extension points an inference gateway consumes.

Two seams, both deliberately *outside* the workload's identity: a recipe's
capability lists and a runtime's native API dialects describe what a deployment
can do, not how it is configured to serve.
"""

from __future__ import annotations

from sparkrun.core.recipe import Recipe
from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint


def _recipe(**changes) -> Recipe:
    data = {
        "recipe_version": "2",
        "model": "Qwen/Qwen3-1.7B",
        "runtime": "vllm",
        "container": "vllm/vllm-openai:latest",
        "defaults": {"port": 8000, "tensor_parallel": 1},
    }
    data.update(changes)
    return Recipe(data)


# --------------------------------------------------------------------------
# Recipe capabilities
# --------------------------------------------------------------------------


def test_capabilities_are_parsed_as_real_attributes():
    r = _recipe(capabilities=["single_vector_embedding"], unsupported_capabilities=["tool_calling"])
    assert r.capabilities == ["single_vector_embedding"]
    assert r.unsupported_capabilities == ["tool_calling"]


def test_capabilities_default_to_empty():
    r = _recipe()
    assert r.capabilities == []
    assert r.unsupported_capabilities == []


def test_capabilities_do_not_leak_into_runtime_config():
    """They are declared in ``_KNOWN_KEYS`` so the unknown-key sweep skips them.

    If they landed in ``runtime_config`` they would reach the serve command as
    flags *and* enter the fingerprint.
    """
    r = _recipe(capabilities=["single_vector_embedding"])
    assert "capabilities" not in r.runtime_config
    assert "unsupported_capabilities" not in r.runtime_config


def test_capabilities_are_outside_the_recipe_fingerprint():
    """Editing a capability list must not change the workload's identity.

    Otherwise declaring a capability on a running deployment would give it a
    new fingerprint and force it to be re-admitted.
    """
    plain = derive_recipe_fingerprint(_recipe())
    declared = derive_recipe_fingerprint(_recipe(capabilities=["single_vector_embedding"], unsupported_capabilities=["tool_calling"]))
    assert plain == declared


def test_capabilities_survive_the_state_round_trip():
    """Every launch serializes the recipe into job metadata and back."""
    original = _recipe(capabilities=["single_vector_embedding"], unsupported_capabilities=["tool_calling"])
    restored = _recipe()
    restored.__setstate__(original.__getstate__())

    assert restored.capabilities == ["single_vector_embedding"]
    assert restored.unsupported_capabilities == ["tool_calling"]


def test_capability_entries_are_coerced_to_strings():
    r = _recipe(capabilities=[1, "vision"])
    assert r.capabilities == ["1", "vision"]


# --------------------------------------------------------------------------
# Runtime native protocols
# --------------------------------------------------------------------------


def test_native_protocols_are_fail_closed_by_default():
    """Over-claiming sends wrong-shaped bytes to a server that cannot parse
    them; under-claiming only costs a translation."""
    from sparkrun.runtimes.base import RuntimePlugin

    assert RuntimePlugin.native_protocols(object(), _recipe()) == ["openai"]


def test_every_shipped_runtime_claims_only_dialects_it_serves():
    """No in-tree runtime may claim a dialect without overriding the hook."""
    from sparkrun.core.bootstrap import get_runtime, init_sparkrun, list_runtimes

    v = init_sparkrun()
    recipe = _recipe()
    names = list_runtimes(v)
    assert names, "no runtimes discovered — the assertion below would be vacuous"

    for name in names:
        protocols = get_runtime(name, v).native_protocols(recipe)
        assert protocols, "%s claimed no protocol at all" % name
        assert all(p == p.lower() for p in protocols), (name, protocols)
        assert protocols[0] == "openai", "%s reordered away from its serving dialect" % name
