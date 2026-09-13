# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Project sparkrun's recipe catalog into the gateway's ``sparkrun`` managed set.

See ``docs/SPARKROUTE_BRIDGE.md`` and the public SparkRoute integration guide:
https://github.com/sparksq/sparkroute/blob/v0.0.2/docs/SPARKRUN_ON_DEMAND.md.

Three properties are load-bearing and each has a failure mode behind it:

* **Catalog, not discovery.** A binding is projected whether its workload is
  ready, stopped, or was never started. Generating deployments from live
  endpoints would deadlock — an offline workload would delete the very
  ``activatable`` deployment the gateway needs in order to start it again.
* **Deterministic.** Array order participates in the gateway's content
  revision, so an unstable ordering would produce a new revision (and a new
  runtime generation) on every reconcile even when nothing changed. Every list
  here is sorted, except ``cluster_candidates``, whose order encodes placement
  preference.
* **Self-contained.** Generated entities reference only generated entities.
  Referencing an operator entity would let their edits block our deletions.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Single provider for every generated deployment.
#:
#: The name is an *identity*, not a protocol label — which is why it is plain
#: ``sparkrun`` rather than the earlier ``sparkrun:openai-compatible``. Which
#: inference dialects a deployment speaks is
#: :data:`Deployment.native_protocols`, and ``Provider.type`` stays
#: ``sparkrun`` as the lifecycle-managed provider class.
#:
#: Not per-cluster and not per-protocol: a deployment references exactly one
#: provider, but an ``activatable`` binding may list several
#: ``cluster_candidates`` and one workload may serve several dialects. Keying
#: providers on either would force two deployments over a single workload —
#: two binding revisions and two fenced activations against one job.
SPARKRUN_PROVIDER = "sparkrun"

#: Dialect every sparkrun runtime serves; the floor for ``native_protocols``.
DEFAULT_PROTOCOL = "openai"

#: Prefix for generated deployment names.  Virtual models are deliberately
#: *unprefixed*: clients call those names.
DEPLOYMENT_PREFIX = "sparkrun:"

#: Namespace for deployments imported from a live discovery snapshot. The
#: model itself is hashed because upstream names may contain characters the
#: gateway does not allow in entity IDs.
DISCOVERED_DEPLOYMENT_PREFIX = "sparkrun:discovered:"

#: Version tag inside the binding-revision digest, so the derivation can change
#: without silently reusing an old identity.
BINDING_REVISION_VERSION = 1

#: Length of the binding revision, matching the recipe fingerprint's.
BINDING_REVISION_LEN = 12

#: Capabilities the gateway will not attempt under a permissive policy: they
#: select a distinct API or a gateway-owned resource contract, so a wrong guess
#: is not a recoverable upstream error. Sparkrun emits these only when a recipe
#: declares them. Any ``x-`` extension is likewise always positive-declaration.
HARD_CAPABILITIES = frozenset(
    {
        "files",
        "stored_completions",
        "responses",
        "responses_compact",
        "background_responses",
        "conversations",
        "single_vector_embedding",
        "token_counting",
    }
)

#: Declaring this makes the deployment an embedding target; the virtual model
#: must require it too, so an invalid generated route fails validation rather
#: than at request time.
EMBEDDING_CAPABILITY = "single_vector_embedding"


class ProjectionError(ValueError):
    """A binding could not be projected into a valid managed set."""


def derive_binding_revision(
    *,
    deployment: str,
    recipe: str,
    recipe_revision: str,
    cluster_candidates: list[str],
    overrides: dict[str, str],
) -> str:
    """Digest identifying one deployment binding.

    Derivation fixed by the gateway contract: SHA-256 over compact canonical
    JSON with sorted keys, truncated to :data:`BINDING_REVISION_LEN`.

    Virtual-model names and aliases are **excluded** on purpose. Several
    virtual models may route to one deployment, so a client-facing rename must
    not change workload identity — a changed binding revision makes endpoints
    registered under the old one ineligible, forcing revalidation.
    ``cluster_candidates`` order is preserved because it expresses placement
    preference; ``overrides`` keys sort canonically.
    """
    payload = {
        "version": BINDING_REVISION_VERSION,
        "controller": "sparkrun",
        "deployment": deployment,
        "recipe": recipe,
        "recipe_revision": recipe_revision,
        "cluster_candidates": list(cluster_candidates),
        "overrides": dict(overrides),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()[:BINDING_REVISION_LEN]


def deployment_name(recipe_revision: str) -> str:
    """Generated deployment identity for a recipe fingerprint."""
    return "%s%s" % (DEPLOYMENT_PREFIX, recipe_revision)


def discovered_deployment_name(model: str) -> str:
    """Stable generated identity for a discovery-only upstream model."""
    digest = hashlib.sha256(model.encode()).hexdigest()[:BINDING_REVISION_LEN]
    return "%s%s" % (DISCOVERED_DEPLOYMENT_PREFIX, digest)


def build_sparkrun_set(
    entries: list["ProjectedBinding"],
    aliases: dict[str, str] | None = None,
    *,
    permissive: bool = True,
    discovered_models: list[str] | None = None,
    discovered_clusters: dict[str, list[str]] | None = None,
    discovered_apis: dict[str, dict[str, Any]] | None = None,
    binding_clusters: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Render projected bindings into the complete ``sparkrun`` managed set.

    Args:
        entries: Resolved bindings, already deduplicated by
            :func:`dedupe_bindings`.
        aliases: ``alias -> virtual model name``. An alias whose target is not
            in this set is skipped, so it starts working by itself once the
            target is bound.
        permissive: Emit ``capability_policy.unknown = "try"``, letting the
            gateway attempt undeclared optional capabilities instead of
            refusing them. See ``ProxyConfig.capability_policy``.
        discovered_models: Upstream model names observed by the last explicit
            discovery sync. These render as ``discovered`` sources and never
            replace or remove activatable bindings.

    Returns:
        ``{"providers": [...], "deployments": [...], "virtual_models": [...]}``
        with every list deterministically ordered.
    """
    deployments: list[dict[str, Any]] = []
    profiles: dict[str, dict[str, Any]] = {}
    models: dict[str, dict[str, Any]] = {}

    for entry in sorted(entries, key=lambda e: e.recipe_revision):
        name = deployment_name(entry.recipe_revision)
        deployment: dict[str, Any] = {
            "name": name,
            "title": deployment_title(
                entry.model, (binding_clusters or {}).get(entry.recipe_revision) or entry.cluster_candidates, fallback="unassigned"
            ),
            "provider": SPARKRUN_PROVIDER,
            "model": entry.model,
            # Order is observable only when several translated fallbacks are
            # possible and none is native, so it is emitted deterministically
            # rather than sorted — preference, not a set.
            "native_protocols": list(entry.native_protocols),
        }

        from .metadata import reduce_model_metadata

        # Match saved launch metadata to this recipe, not merely its public
        # model name: another recipe may serve that name with different limits.
        observed = (discovered_apis or {}).get(entry.model, {}).get("recipe_metadata", {}).get(entry.recipe_revision, {})
        fields = reduce_model_metadata(entry.model_metadata, observed)
        if fields:
            deployment["model_metadata"] = fields
        declared = sorted(set(entry.capabilities))
        unsupported = sorted(set(entry.unsupported_capabilities))
        overlap = set(declared) & set(unsupported)
        if overlap:
            # The gateway rejects this outright; catching it here names the
            # recipe, which its validation error cannot.
            raise ProjectionError("recipe %s declares %s as both supported and unsupported" % (entry.recipe, ", ".join(sorted(overlap))))
        if declared:
            deployment["capabilities"] = declared

        policy: dict[str, Any] = {}
        if permissive:
            policy["unknown"] = "try"
        if unsupported:
            policy["unsupported"] = unsupported
        if policy:
            deployment["capability_policy"] = policy

        deployment["endpoint_source"] = {
            "type": "activatable",
            "controller": "sparkrun",
            "revision": derive_binding_revision(
                deployment=name,
                recipe=entry.recipe,
                recipe_revision=entry.recipe_revision,
                cluster_candidates=entry.cluster_candidates,
                overrides=entry.overrides,
            ),
            "recipe": entry.recipe,
            "recipe_revision": entry.recipe_revision,
            "cluster_candidates": list(entry.cluster_candidates),
            "cold_start": entry.cold_start,
        }
        if entry.overrides:
            deployment["endpoint_source"]["overrides"] = dict(sorted(entry.overrides.items()))
        deployments.append(deployment)

        model = models.setdefault(entry.virtual_model, {"name": entry.virtual_model, "targets": []})
        model["targets"].append(name)
        _collect_profiles(profiles, entry.virtual_model, entry.request_profiles, name)
        # An embedding route must require the capability, so a binding that
        # loses the declaration fails validation instead of failing requests.
        if EMBEDDING_CAPABILITY in declared:
            model.setdefault("required_capabilities", set()).add(EMBEDDING_CAPABILITY)

    # A live-discovery import is a warm route, not a cold-start declaration.
    # Skip names already represented by an activatable binding: the
    # controller adopts the same discovered endpoint for that deployment, and
    # a second target would count one upstream twice.
    bound_models = {entry.virtual_model for entry in entries}
    for upstream_model in sorted(set(discovered_models or ()) - bound_models):
        name = discovered_deployment_name(upstream_model)
        deployment: dict[str, Any] = {
            "name": name,
            "title": deployment_title(upstream_model, (discovered_clusters or {}).get(upstream_model, []), fallback="discovered"),
            "provider": SPARKRUN_PROVIDER,
            "model": upstream_model,
            "native_protocols": (discovered_apis or {}).get(upstream_model, {}).get("native_protocols") or [DEFAULT_PROTOCOL],
            "capabilities": (discovered_apis or {}).get(upstream_model, {}).get("capabilities", []),
            "endpoint_source": {"type": "discovered", "controller": "sparkrun"},
        }
        discovered_metadata = (discovered_apis or {}).get(upstream_model, {}).get("model_metadata")
        if discovered_metadata:
            deployment["model_metadata"] = deepcopy(discovered_metadata)
        if permissive:
            deployment["capability_policy"] = {"unknown": "try"}
        deployments.append(deployment)
        models[upstream_model] = {"name": upstream_model, "targets": [name]}
        _collect_profiles(profiles, upstream_model, (discovered_apis or {}).get(upstream_model, {}).get("request_profiles", {}), name)

    deployments.sort(key=lambda deployment: deployment["name"])

    virtual_models: list[dict[str, Any]] = []
    for name in sorted(models):
        model = models[name]
        rendered: dict[str, Any] = {"name": name}
        required = model.get("required_capabilities")
        if required:
            rendered["required_capabilities"] = sorted(required)
        rendered["pools"] = [
            {
                "priority": 0,
                "targets": [{"deployment": target, "weight": 1} for target in sorted(model["targets"])],
            }
        ]
        virtual_models.append(rendered)

    if aliases:
        by_name = {model["name"]: model for model in virtual_models}
        for alias_name, target_model in sorted(aliases.items()):
            target = by_name.get(target_model)
            if target is None:
                logger.debug("Alias %r skipped: no bound virtual model named %r", alias_name, target_model)
                continue
            target.setdefault("aliases", []).append(alias_name)
    for model in virtual_models:
        if "aliases" in model:
            model["aliases"] = sorted(model["aliases"])

    _append_profiles(virtual_models, profiles)
    return {
        "providers": [{"name": SPARKRUN_PROVIDER, "type": "sparkrun"}],
        "deployments": deployments,
        "virtual_models": virtual_models,
    }


def deployment_title(model: str, clusters: list[str], *, fallback: str) -> str:
    """Human label only: never participates in deployment or binding identity."""
    cluster = ",".join(dict.fromkeys(clusters)) or fallback
    return f"sparkrun:{cluster}:{model}"


class ProjectedBinding:
    """One resolved binding, ready to render."""

    __slots__ = (
        "recipe",
        "recipe_revision",
        "model",
        "virtual_model",
        "cluster_candidates",
        "overrides",
        "capabilities",
        "unsupported_capabilities",
        "native_protocols",
        "cold_start",
        "request_profiles",
        "model_metadata",
    )

    def __init__(
        self,
        *,
        recipe: str,
        recipe_revision: str,
        model: str,
        virtual_model: str,
        cluster_candidates: list[str] | None = None,
        overrides: dict[str, str] | None = None,
        request_profiles: dict[str, Any] | None = None,
        model_metadata: dict[str, Any] | None = None,
        capabilities: list[str] | None = None,
        unsupported_capabilities: list[str] | None = None,
        native_protocols: list[str] | None = None,
        cold_start: str = "wait",
    ) -> None:
        self.recipe = recipe
        self.recipe_revision = recipe_revision
        self.model = model
        self.virtual_model = virtual_model
        self.cluster_candidates = list(cluster_candidates or ())
        self.overrides = {str(k): str(v) for k, v in (overrides or {}).items()}
        self.request_profiles = deepcopy(request_profiles or {})
        self.model_metadata = deepcopy(model_metadata or {})
        self.capabilities = list(capabilities or ())
        self.unsupported_capabilities = list(unsupported_capabilities or ())
        self.native_protocols = list(native_protocols or (DEFAULT_PROTOCOL,))
        self.cold_start = cold_start


def dedupe_bindings(entries: list[ProjectedBinding]) -> list[ProjectedBinding]:
    """Collapse bindings that share a recipe fingerprint.

    Deployment identity is ``sparkrun:<recipe_revision>``, but
    ``derive_recipe_fingerprint`` deliberately excludes a recipe's name and
    registry — so the *same* recipe present in two registries (a local copy of
    a community one, which users routinely have) yields one deployment name
    from two catalog entries, and emitting both would be rejected as a
    duplicate deployment.

    Collapsing them is correct rather than a workaround: two recipes with
    identical declared serve configuration already share an ``intent_id``, so
    ``sparkrun run`` on either adopts the same container. They are one
    workload. The lexically-first qualified name becomes the representative, so
    the choice is stable across reconciles rather than depending on catalog
    iteration order.
    """
    by_revision: dict[str, ProjectedBinding] = {}
    for entry in entries:
        existing = by_revision.get(entry.recipe_revision)
        if existing is None:
            by_revision[entry.recipe_revision] = entry
            continue
        if entry.cluster_candidates != existing.cluster_candidates:
            raise ProjectionError(
                "Recipe fingerprint %s is bound to conflicting clusters; define separate deployments in the configuration UI"
                % entry.recipe_revision
            )
        merged_profiles = deepcopy(existing.request_profiles)
        for selector, parameters in entry.request_profiles.items():
            if selector in merged_profiles and merged_profiles[selector] != parameters:
                raise ProjectionError(f"Recipes {existing.recipe!r} and {entry.recipe!r} define conflicting request profile {selector!r}")
            merged_profiles[selector] = deepcopy(parameters)
        existing.request_profiles = entry.request_profiles = merged_profiles
        if entry.recipe < existing.recipe:
            # Keep the lexically-first recipe name, but preserve the richer
            # declaration: dropping a capability because a duplicate lacked it
            # would silently disable a route.
            entry.capabilities = sorted(set(entry.capabilities) | set(existing.capabilities))
            entry.unsupported_capabilities = sorted(set(entry.unsupported_capabilities) | set(existing.unsupported_capabilities))
            by_revision[entry.recipe_revision] = entry
        else:
            existing.capabilities = sorted(set(existing.capabilities) | set(entry.capabilities))
            existing.unsupported_capabilities = sorted(set(existing.unsupported_capabilities) | set(entry.unsupported_capabilities))
        logger.debug(
            "Bindings %r and %r share fingerprint %s; projecting one deployment",
            entry.recipe,
            existing.recipe,
            entry.recipe_revision,
        )
    return list(by_revision.values())


def _runtime_protocols(recipe: Any, *, sctx: Any = None) -> list[str]:
    """Ask the recipe's runtime which dialects it serves natively.

    Degrades to ``["openai"]`` rather than failing the reconcile when the
    runtime plugin cannot be resolved. That is the fail-closed direction:
    ``openai`` is what every sparkrun runtime serves, and omitting a dialect
    only costs a translation, whereas failing the whole reconcile over an
    unloadable plugin would take down configuration for every other binding.
    """
    from sparkrun.api._resolve import resolve_runtime

    try:
        runtime = resolve_runtime(recipe, sctx=sctx)
        protocols = list(runtime.native_protocols(recipe) or ())
    except Exception:
        logger.debug("Could not resolve native protocols for %s", getattr(recipe, "qualified_name", recipe), exc_info=True)
        return [DEFAULT_PROTOCOL]
    return protocols or [DEFAULT_PROTOCOL]


def _runtime_capabilities(recipe: Any, *, sctx: Any = None) -> list[str]:
    from sparkrun.api._resolve import resolve_runtime

    try:
        return list(resolve_runtime(recipe, sctx=sctx).native_capabilities(recipe))
    except Exception:
        logger.debug("Could not resolve native APIs for %s", getattr(recipe, "qualified_name", recipe), exc_info=True)
        return []


def resolve_bindings(bindings: list[dict[str, Any]], *, sctx: Any = None) -> list[ProjectedBinding]:
    """Resolve ``proxy.yaml`` binding entries into projectable bindings.

    Each entry names a recipe; sparkrun resolves it to get the fingerprint that
    becomes ``recipe_revision``, the served model name, and any declared
    capabilities.

    Recipe resolution is deliberately *not* tolerant of a missing recipe: a
    binding that silently vanished from the desired set would remove a route
    the user believes exists, which is worse than refusing to reconcile.

    Args:
        bindings: Entries from ``ProxyConfig.bindings``.
        sctx: Optional shared session context (avoids re-scanning registries).

    Raises:
        ProjectionError: A binding is malformed or its recipe cannot resolve.
    """
    # Deferred: sparkrun.proxy.discovery imports sparkrun.api, so a
    # module-level import here would be circular.
    import sparkrun.api as api
    from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint

    resolved: list[ProjectedBinding] = []
    for index, entry in enumerate(bindings):
        name = str(entry.get("recipe") or "").strip()
        if not name:
            raise ProjectionError("bindings[%d] has no 'recipe'" % index)

        overrides = {str(k): str(v) for k, v in (entry.get("overrides") or {}).items()}
        candidates = entry.get("cluster_candidates")
        if candidates is None:
            single = entry.get("cluster")
            candidates = [single] if single else []
        candidates = [str(c) for c in candidates]

        try:
            recipe, normalized = api.resolve_catalog_recipe(name, overrides, sctx=sctx)
        except api.SparkrunError as exc:
            raise ProjectionError("bindings[%d]: recipe %r could not be resolved (%s)" % (index, name, exc)) from exc

        cold_start = str(entry.get("cold_start") or "wait")
        if cold_start not in ("wait", "reject"):
            raise ProjectionError("bindings[%d]: cold_start must be 'wait' or 'reject', got %r" % (index, cold_start))

        from .recipe_config import recipe_sparkroute, SparkrouteRecipeError

        try:
            settings = recipe_sparkroute(recipe)
        except SparkrouteRecipeError as error:
            raise ProjectionError(str(error)) from error
        from .metadata import build_model_metadata

        served = str(entry.get("model") or getattr(recipe, "effective_served_model_name", "") or recipe.model)
        protocols = entry.get("native_protocols") or _runtime_protocols(recipe, sctx=sctx)
        resolved.append(
            ProjectedBinding(
                recipe=name if name.startswith("catalog:") else getattr(recipe, "qualified_name", None) or name,
                recipe_revision=derive_recipe_fingerprint(recipe, normalized),
                model=served,
                virtual_model=served,
                cluster_candidates=candidates,
                overrides=overrides,
                # A binding may narrow or extend what the recipe declares;
                # neither side can be inferred, so both are explicit.
                model_metadata=build_model_metadata(recipe, [served], overrides=normalized).get(served, {}),
                request_profiles=settings.get("request_profiles", {}),
                capabilities=sorted(
                    set(str(c) for c in (entry.get("capabilities") or getattr(recipe, "capabilities", []) or []))
                    | set(settings.get("capabilities", []))
                    | set(_runtime_capabilities(recipe, sctx=sctx))
                ),
                unsupported_capabilities=[
                    str(c) for c in (entry.get("unsupported_capabilities") or getattr(recipe, "unsupported_capabilities", []) or [])
                ],
                native_protocols=[str(p).strip().lower() for p in protocols if str(p).strip()],
                cold_start=cold_start,
            )
        )
    return resolved


__all__ = [
    "BINDING_REVISION_LEN",
    "DEFAULT_PROTOCOL",
    "BINDING_REVISION_VERSION",
    "DEPLOYMENT_PREFIX",
    "DISCOVERED_DEPLOYMENT_PREFIX",
    "EMBEDDING_CAPABILITY",
    "HARD_CAPABILITIES",
    "SPARKRUN_PROVIDER",
    "ProjectedBinding",
    "ProjectionError",
    "build_sparkrun_set",
    "dedupe_bindings",
    "derive_binding_revision",
    "discovered_deployment_name",
    "deployment_name",
    "resolve_bindings",
]


def _collect_profiles(profiles, parent, declarations, deployment):
    for selector, overrides in declarations.items():
        name = f"{parent}:{selector}"
        profile = profiles.setdefault(name, {"parent": parent, "overrides": deepcopy(overrides), "targets": set()})
        if profile["overrides"] != overrides:
            raise ProjectionError(f"Recipes serving {parent!r} define conflicting request profile {selector!r}")
        profile["targets"].add(deployment)


def _append_profiles(models, profiles):
    by_name = {model["name"]: model for model in models}
    occupied = set(by_name) | {alias for model in models for alias in model.get("aliases", [])}
    for name, profile in sorted(profiles.items()):
        parent = by_name[profile["parent"]]
        suffix = name[len(parent["name"]) :]
        aliases = [alias + suffix for alias in parent.get("aliases", [])]
        for public_name in [name, *aliases]:
            if public_name in occupied:
                raise ProjectionError(f"Recipe request profile conflicts with existing model or alias {public_name!r}")
            occupied.add(public_name)
        model = deepcopy(parent)
        model.update(name=name, request_overrides=deepcopy(profile["overrides"]))
        if aliases:
            model["aliases"] = aliases
        model["pools"] = [{"priority": 0, "targets": [{"deployment": target, "weight": 1} for target in sorted(profile["targets"])]}]
        models.append(model)
    models.sort(key=lambda model: model["name"])
