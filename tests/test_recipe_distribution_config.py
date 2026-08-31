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


# --- per-entry revision ---------------------------------------------------
#
# A commit SHA is only meaningful in the repo it came from.  The recipe's
# top-level `model_revision` is stamped onto the auto-generated entry, and
# nothing else inherits it — a draft model added by runtime.prepare() is a
# different repo, and pinning it to the served model's SHA fails the download
# with "Revision Not Found" after the served model has already synced.


def _rev_recipe(model_revision=None, dist_cfg=None):
    d = {
        "recipe_version": "2",
        "name": "dc",
        "model": "org/primary",
        "runtime": "vllm-distributed",
        "container": "img:latest",
    }
    if model_revision is not None:
        d["model_revision"] = model_revision
    if dist_cfg is not None:
        d["distribution_config"] = dist_cfg
    return Recipe.from_dict(d)


def _revs(rc):
    return [e.revision for e in rc.entries]


def test_model_revision_stamped_on_auto_default_entry():
    dc = _rev_recipe(model_revision="deadbeef").distribution_config
    assert _revs(dc.models) == ["deadbeef"]


def test_model_revision_stamped_on_inherited_default_when_models_omitted():
    """`models` omitted → the inherited default entry still carries the pin."""
    dc = _rev_recipe("deadbeef", {"containers": {"entries": [{"name": "custom:img"}]}}).distribution_config
    assert _revs(dc.models) == ["deadbeef"]


def test_added_model_does_not_inherit_recipe_revision():
    """The bug: a draft model pinned to the served model's SHA 404s."""
    r = _rev_recipe(model_revision="deadbeef")
    r.distribution_config.add_model("org/draft")
    entries = {e.name: e.revision for e in r.distribution_config.models.entries}
    assert entries == {"{model}": "deadbeef", "org/draft": None}


def test_add_model_pins_its_own_revision():
    r = _rev_recipe(model_revision="deadbeef")
    r.distribution_config.add_model("org/draft", revision="cafe1234")
    entries = {e.name: e.revision for e in r.distribution_config.models.entries}
    assert entries == {"{model}": "deadbeef", "org/draft": "cafe1234"}


def test_add_model_pin_upgrades_an_earlier_unpinned_entry():
    """A later caller with a revision outranks an earlier unpinned add.

    Dedup used to return on the first name match, so whichever call happened to
    run first won — and an unpinned add silently discarded a pin that arrived
    afterwards, fetching the draft model unpinned.
    """
    r = _rev_recipe(model_revision="deadbeef")
    r.distribution_config.add_model("org/draft")
    r.distribution_config.add_model("org/draft", revision="cafe1234")
    entries = {e.name: e.revision for e in r.distribution_config.models.entries}
    assert entries == {"{model}": "deadbeef", "org/draft": "cafe1234"}


def test_add_model_without_a_pin_never_clears_one():
    """The reverse order must not un-pin what a previous caller pinned."""
    r = _rev_recipe(model_revision="deadbeef")
    r.distribution_config.add_model("org/draft", revision="cafe1234")
    r.distribution_config.add_model("org/draft")
    entries = {e.name: e.revision for e in r.distribution_config.models.entries}
    assert entries == {"{model}": "deadbeef", "org/draft": "cafe1234"}


def test_hand_written_entries_are_authoritative_about_revision():
    """A recipe that lists its own entries states every revision it wants."""
    dc = _rev_recipe(
        "deadbeef",
        {"models": {"entries": [{"name": "{model}"}, {"name": "aux/backbone", "revision": "abc123"}]}},
    ).distribution_config
    assert _revs(dc.models) == [None, "abc123"]


def test_revision_survives_state_round_trip():
    """Job metadata re-reads the recipe; an unpinned draft must stay unpinned."""
    r = _rev_recipe(model_revision="deadbeef")
    r.distribution_config.add_model("org/draft")
    restored = Recipe._deserialize(r.__getstate__())
    entries = {e.name: e.revision for e in restored.distribution_config.models.entries}
    assert entries == {"{model}": "deadbeef", "org/draft": None}
