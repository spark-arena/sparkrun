"""Tests for ``distribution_config`` parsing, esp. per-subkey defaulting.

An omitted ``models`` / ``containers`` subkey falls back to the auto-default
single entry (``{model}`` / ``{container}``); a subkey that is present is
honored literally (including an explicit empty ``entries`` or ``enabled: false``).
"""

from __future__ import annotations

from sparkrun.core.recipe import Recipe


def _recipe(dist_cfg=None):
    d = {
        "recipe_version": "2",
        "name": "dc",
        "model": "org/primary",
        "runtime": "vllm-distributed",
        "container": "img:latest",
    }
    if dist_cfg is not None:
        d["distribution_config"] = dist_cfg
    return Recipe.from_dict(d)


def _names(rc):
    return [e.name for e in rc.entries]


def test_no_block_uses_default_both():
    dc = _recipe().distribution_config
    assert _names(dc.models) == ["{model}"]
    assert _names(dc.containers) == ["{container}"]


def test_models_present_containers_defaulted():
    """The headline ergonomics fix: add a model, don't re-list the container."""
    dc = _recipe({"models": {"entries": [{"name": "{model}"}, {"name": "aux/backbone"}]}}).distribution_config
    assert _names(dc.models) == ["{model}", "aux/backbone"]
    # containers omitted → auto-default entry, not empty
    assert _names(dc.containers) == ["{container}"]


def test_containers_present_models_defaulted():
    dc = _recipe({"containers": {"entries": [{"name": "custom:img"}]}}).distribution_config
    assert _names(dc.models) == ["{model}"]
    assert _names(dc.containers) == ["custom:img"]


def test_present_empty_entries_is_honored_not_defaulted():
    """An explicit empty list means 'distribute nothing', not 'use default'."""
    dc = _recipe({"models": {"entries": []}}).distribution_config
    assert _names(dc.models) == []
    # the omitted containers subkey still defaults
    assert _names(dc.containers) == ["{container}"]


def test_present_disabled_subkey_is_honored():
    dc = _recipe({"models": {"enabled": False, "entries": [{"name": "{model}"}]}}).distribution_config
    assert dc.models.enabled is False
    assert _names(dc.models) == ["{model}"]


def test_string_shorthand_entries():
    dc = _recipe({"models": {"entries": ["{model}", "aux/backbone"]}}).distribution_config
    assert _names(dc.models) == ["{model}", "aux/backbone"]


def test_target_defaults_to_all_nodes():
    dc = _recipe({"models": {"entries": [{"name": "aux/backbone"}]}}).distribution_config
    assert dc.models.entries[0].target == [-1]


def test_provided_block_marked_externally_provided():
    dc = _recipe({"models": {"entries": [{"name": "{model}"}]}}).distribution_config
    assert dc.externally_provided is True


def test_resolve_substitutes_placeholders_across_defaulted_subkey():
    r = _recipe({"models": {"entries": [{"name": "{model}"}, {"name": "aux/backbone"}]}})
    dc = r.distribution_config.resolve(r, resolved_container=r.container, overrides={})
    assert _names(dc.models) == ["org/primary", "aux/backbone"]
    # the defaulted container subkey still resolves {container}
    assert _names(dc.containers) == ["img:latest"]
