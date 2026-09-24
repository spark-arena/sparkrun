"""Private saved measurement inputs and resume validation; no launch or publication."""

from __future__ import annotations

from sparkrun.benchmarking.metadata import benchmark_recipe_fingerprint, public_benchmark_data
from sparkrun.benchmarking.run_state import BenchmarkStateError, derive_benchmark_id
from sparkrun.core.recipe import Recipe


def _image_digest(reference):
    """Recognize a digest reference without resolving mutable tags or doing I/O."""
    value = (reference or "").rsplit("@", 1)[-1]
    return value if value.startswith("sha256:") else None


def validate_image_references(references, image):
    """Reject known image changes; return whether equivalence was established."""
    references = {value for value in references if value}
    if not references or not image:
        return False  # missing evidence stays unknown
    if image in references:
        return True
    digest = _image_digest(image)
    if digest and digest in {_image_digest(ref) for ref in references}:
        return True
    raise BenchmarkStateError("Running job image differs from the saved benchmark; explicitly start fresh")


def measurement_specification(recipe: Recipe, overrides: dict) -> dict:
    """Use the existing full Recipe serialization to preserve in-memory edits."""
    return {
        "version": 1,
        "recipe": public_benchmark_data(recipe.__getstate__()),
        "overrides": public_benchmark_data(overrides),
        "fingerprint": benchmark_recipe_fingerprint(recipe, overrides),
    }


def _saved_recipe(spec: dict) -> tuple[Recipe, dict]:
    if type(spec.get("version")) is not int or spec["version"] != 1:
        raise BenchmarkStateError("Unsupported benchmark measurement specification")
    data, overrides = spec.get("recipe"), spec.get("overrides")
    if not isinstance(data, dict) or not isinstance(overrides, dict) or data.get("_serialization_version") != Recipe._SERIALIZATION_VERSION:
        raise BenchmarkStateError("Invalid saved benchmark recipe specification")
    recipe = Recipe._deserialize(data)
    if benchmark_recipe_fingerprint(recipe, overrides) != spec.get("fingerprint"):
        raise BenchmarkStateError("Saved benchmark recipe no longer matches its fingerprint; explicitly start fresh")
    return recipe, overrides


def _matches_identity(state, recipe, overrides) -> bool:
    from sparkrun.orchestration.job_metadata import generate_cluster_id, PLACEMENT_TOKEN_LEN

    cluster_id = generate_cluster_id(state.intent_id, "0" * PLACEMENT_TOKEN_LEN) if state.intent_id else state.cluster_id
    return state.benchmark_id == derive_benchmark_id(
        cluster_id,
        state.framework,
        state.profile,
        state.base_args,
        state.schedule,
        recipe_fingerprint=benchmark_recipe_fingerprint(recipe, overrides),
        hosts=state.host_list,
    )


def _configuration_fingerprint(recipe, overrides):
    """Compare serving inputs after image equivalence has been verified separately."""
    from copy import copy

    recipe = copy(recipe)
    recipe.container = "<verified-image>"
    recipe.defaults = {key: value for key, value in recipe.defaults.items() if key != "image"}
    # Identity reads the declared snapshot when `overrides:` were applied; it
    # needs the same normalization, and must not be shared with the caller's.
    declared = getattr(recipe, "_declared", None)
    if declared is not None:
        recipe._declared = {
            **declared,
            "container": "<verified-image>",
            "defaults": {key: value for key, value in declared["defaults"].items() if key != "image"},
        }
    overrides = {key: value for key, value in (overrides or {}).items() if key != "image"}
    return benchmark_recipe_fingerprint(recipe, overrides)


def record_job_specification(state, meta: dict | None) -> None:
    """Establish the initial baseline once; later observations must match it."""
    if (state.measurement_spec or {}).get("job_fingerprint"):
        validate_job_specification(state, meta)
        return
    if state.measurement_spec is not None and meta and meta.get("recipe_state"):
        actual = Recipe._deserialize(meta["recipe_state"])
        overrides = meta.get("overrides") or {}
        state.measurement_spec["job_fingerprint"] = benchmark_recipe_fingerprint(actual, overrides)
        state.measurement_spec["job_configuration_fingerprint"] = _configuration_fingerprint(actual, overrides)


def validate_job_specification(state, meta: dict | None, *, recipe=None) -> None:
    """One acceptance rule before appending measurements; never rewrite evidence."""
    spec = state.measurement_spec or {}
    expected = spec.get("job_fingerprint")
    if meta is None:
        if expected:
            raise BenchmarkStateError("Running job recipe provenance is missing")
        return
    if not state.matches_hosts(meta.get("hosts")):
        raise BenchmarkStateError("Running job hosts differ from the saved benchmark")
    if recipe is None and state.measurement_spec is not None:
        recipe, _ = _saved_recipe(state.measurement_spec)
    if recipe is not None:
        for field in ("model", "runtime"):
            if meta.get(field) and meta[field] != getattr(recipe, field):
                raise BenchmarkStateError("Running job %s differs from the saved benchmark" % field)
    equivalent_image = validate_image_references(
        (state.extras.get(key) for key in ("container_image", "container_image_sha", "container_image_longterm_ref")),
        meta.get("effective_container_image"),
    )
    if not expected:
        return
    if not meta.get("recipe_state"):
        raise BenchmarkStateError("Running job recipe provenance is missing")
    actual = Recipe._deserialize(meta["recipe_state"])
    overrides = meta.get("overrides") or {}
    if benchmark_recipe_fingerprint(actual, overrides) == expected:
        return
    normalized = spec.get("job_configuration_fingerprint")
    if not normalized and state.extras.get("measurement_recipe_state"):
        # Older baselines may establish the same evidence from their recorded
        # effective context, but only if it verifies against the saved hash.
        saved = Recipe._deserialize(state.extras["measurement_recipe_state"])
        saved_overrides = state.extras.get("measurement_overrides") or {}
        if benchmark_recipe_fingerprint(saved, saved_overrides) == expected:
            normalized = _configuration_fingerprint(saved, saved_overrides)
    if equivalent_image and normalized and _configuration_fingerprint(actual, overrides) == normalized:
        return
    raise BenchmarkStateError("Running job configuration differs from the saved benchmark")


def restore_measurement_specification(state, meta: dict | None, *, config) -> tuple[Recipe, dict]:
    """Restore saved inputs, or verify legacy inputs against their benchmark ID.

    Legacy state cannot silently trust a mutable recipe name. If its identity
    cannot be reproduced, the caller must start fresh. No checkpoint is written
    here, including on successful legacy reconstruction.
    """
    if state.measurement_spec is not None:
        recipe, overrides = _saved_recipe(state.measurement_spec)
    else:
        from sparkrun.core.resolve import load_recipe

        recipe, _, _ = load_recipe(config, state.recipe_qualified_name, resolve=False)
        overrides = public_benchmark_data((meta or {}).get("overrides") or {})
    if not _matches_identity(state, recipe, overrides):
        raise BenchmarkStateError("Cannot verify the original benchmark recipe; explicitly start fresh")
    if meta is not None:
        validate_job_specification(state, meta, recipe=recipe)
    return recipe, overrides
