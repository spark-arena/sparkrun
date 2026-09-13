"""Controller-local recipe catalog for graphical and automated clients.

Search is cache-only. References persist source identity rather than a copy of
registry contents, so a pinned fingerprint detects subsequent recipe changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import RecipeNotFound, SparkrunError

MAX_RECIPE_BYTES = 256 * 1024
_REFERENCE = re.compile(r"^catalog:([0-9a-f]{32})$")


def _root(sctx) -> Path:
    return Path(sctx.config.config_path).parent / "recipe-catalog"


def _atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".catalog-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _reference(path: Path, registry: str | None, sctx, *, imported: bool = False) -> str:
    value = {"path": str(path.resolve()), "registry": registry, "imported": imported}
    if registry:
        value["registry_url"] = sctx.registry_manager.get_registry(registry).url
    identity = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:32]
    record = _root(sctx) / "references" / (identity + ".json")
    if not record.exists():
        _atomic(record, value)
    return "catalog:" + identity


def _selection(reference: str, sctx) -> tuple[Path, str | None, bool]:
    from sparkrun.core.registry import RegistryError

    match = _REFERENCE.fullmatch(reference)
    if match:
        try:
            record = json.loads((_root(sctx) / "references" / (match[1] + ".json")).read_text())
            path = Path(record["path"])
            registry = record.get("registry")
            if registry:
                entry = sctx.registry_manager.get_registry(registry)
                if not entry.enabled or entry.url != record.get("registry_url"):
                    raise RecipeNotFound("Selected recipe registry changed or is disabled; choose it again")
            if not path.is_file():
                raise RecipeNotFound("Selected recipe no longer exists; choose it again")
            return path, registry, bool(record.get("imported"))
        except (OSError, ValueError, KeyError, RegistryError) as exc:
            raise RecipeNotFound("Recipe selection is unavailable; choose it again") from exc
    if reference.startswith("catalog:") or "://" in reference:
        raise RecipeNotFound("Select a cached registry recipe or a controller-local file")
    path = Path(reference).expanduser()
    if not path.is_absolute():
        from sparkrun.utils import parse_scoped_name

        scope, name = parse_scoped_name(reference)
        matches = sctx.registry_manager.find_recipe_in_registries(name, include_hidden=True)
        matches = [(registry, selected) for registry, selected in matches if not scope or registry == scope]
        if len(matches) != 1:
            raise RecipeNotFound("Choose an exact cached recipe or an absolute path on the control node")
        path = Path(matches[0][1])
    if not path.is_file():
        raise RecipeNotFound("Recipe was not found on this controller")
    path = path.resolve()
    return path, sctx.registry_manager.registry_for_path(path), path.parent == _root(sctx) / "imports"


def list_registries(*, sctx=None) -> list[dict[str, Any]]:
    """List configured registries without initializing or updating them."""
    sctx = resolve_sctx(sctx)
    manager = sctx.registry_manager
    return [
        {
            "name": entry.name,
            "enabled": entry.enabled,
            "visible": getattr(entry, "visible", True),
            "trusted": getattr(entry, "trusted", False),
            "cached": manager._cache_dir(entry.name).is_dir(),
        }
        for entry in manager.list_registries()
    ]


def list_clusters(*, sctx=None) -> list[dict[str, Any]]:
    """List named cluster definitions; no SSH or live capacity probe."""
    sctx = resolve_sctx(sctx)
    manager = sctx.cluster_manager
    default = manager.get_default()
    return [
        {
            "name": cluster.name,
            "description": cluster.description or "",
            "host_count": len(cluster.hosts),
            "default": cluster.name == default,
        }
        for cluster in manager.list_clusters()
    ]


def catalog_recipes(
    query: str = "",
    *,
    registry: str = "",
    runtime: str = "",
    local_only: bool = False,
    offset: int = 0,
    limit: int = 50,
    filters: dict[str, str] | None = None,
    sctx=None,
) -> dict[str, Any]:
    """Search cached recipes with exact file identity and bounded pagination."""
    from sparkrun.api._recipes import search_recipes
    from sparkrun.core.recipe import recipe_summary

    if not 1 <= limit <= 100 or offset < 0 or len(query) > 256:
        raise SparkrunError("Invalid catalog page or query")
    filters = filters or {}
    if (
        not isinstance(filters, dict)
        or set(filters) - set(CATALOG_FACETS)
        or any(not isinstance(v, str) or len(v) > 128 for v in filters.values())
    ):
        raise SparkrunError("Invalid recipe filters")
    sctx = resolve_sctx(sctx)
    cleanup_catalog_imports(sctx=sctx)
    found = (
        []
        if local_only
        else search_recipes(
            query or None,
            registry=registry or None,
            runtime=runtime or None,
            include_hidden=True,
            include_local=False,
            ensure_initialized=False,
            sctx=sctx,
        )
    )
    entries = [] if local_only else [entry.to_dict() for entry in found]
    if not registry:
        # Explicit controller-owned roots, independent of the bridge's cwd.
        roots = [Path(sctx.config.config_path).parent / "recipes", _root(sctx) / "imports"]
        for root in roots:
            if root.is_dir():
                for path in sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml")):
                    try:
                        row = recipe_summary(path)
                        if (
                            row
                            and (not runtime or row.get("runtime") == runtime)
                            and query.lower() in " ".join(str(row.get(k, "")) for k in ("name", "model", "description")).lower()
                        ):
                            entries.append(row)
                    except (OSError, ValueError):
                        continue
    rows = []
    seen = set()
    for entry in entries:
        path = Path(entry["path"]).resolve()
        if path in seen:
            continue
        seen.add(path)
        row = {key: entry.get(key) for key in ("name", "model", "runtime", "description", "min_nodes", "tp", "registry")}
        for key in ("name", "model", "runtime", "description", "registry"):
            if row.get(key) is not None:
                row[key] = str(row[key])[:1024]
        row["source_path"] = str(path)[:4096]
        row["reference"] = _reference(path, entry.get("registry"), sctx, imported=path.parent == _root(sctx) / "imports")
        row.update(_declared_facets(path))
        rows.append(row)
    rows.sort(key=lambda row: (bool(row["registry"]), str(row["name"]), row["source_path"]))
    facets = {key: sorted({str(row.get(key)) if row.get(key) is not None else "unknown" for row in rows}) for key in CATALOG_FACETS}
    rows = [
        row
        for row in rows
        if all((str(row.get(key)) if row.get(key) is not None else "unknown") == value for key, value in filters.items() if value)
    ]
    next_offset = offset + limit
    registries = list_registries(sctx=sctx)
    return {
        "recipes": rows[offset:next_offset],
        "facets": facets,
        "total": len(rows),
        "next_offset": next_offset if next_offset < len(rows) else None,
        "unavailable_registries": [r["name"] for r in registries if r["enabled"] and not r["cached"]],
    }


def resolve_catalog_recipe(reference: str, overrides: dict | None = None, *, sctx=None):
    """Resolve one exact selection and normalize overrides like the CLI run path.

    Returns (Recipe, launch overrides). Image and env overrides are applied to
    the recipe before runtime selection and fingerprint derivation.
    """
    from sparkrun.core.recipe import Recipe, RecipeError
    from sparkrun.core.resolve import apply_recipe_overrides
    from sparkrun.utils import coerce_value

    sctx = resolve_sctx(sctx)
    path, registry, imported = _selection(reference, sctx)
    if path.stat().st_size > MAX_RECIPE_BYTES:
        raise SparkrunError("Recipe exceeds the size limit")
    try:
        recipe = Recipe.load(path, resolve=False)
        recipe.source_registry = registry
        # Imported files have not inherited the trust of a local author.
        recipe.is_url_sourced = imported
        values = {str(key): coerce_value(value) if isinstance(value, str) else value for key, value in (overrides or {}).items()}
        image = values.pop("image", None)
        env = ["%s=%s" % (key, values.pop(key)) for key in list(values) if key.startswith("env.")]
        recipe, values = apply_recipe_overrides(env, image=image, recipe=recipe, **values)
        return recipe, values
    except (RecipeError, ValueError, TypeError) as exc:
        raise SparkrunError("Recipe is invalid: %s" % type(exc).__name__) from exc


def get_recipe_details(reference: str, overrides: dict | None = None, *, sctx=None) -> dict[str, Any]:
    """Resolve a selection into a safe model-configuration preview, without launching."""
    from sparkrun.api._resolve import resolve_runtime
    from sparkrun.core.launcher import resolve_recipe_trust
    from sparkrun.core.recipe_items import registered_recipe_items
    from sparkrun.core.validation import validate_recipe
    from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint

    sctx = resolve_sctx(sctx)
    path, registry, imported = _selection(reference, sctx)
    recipe, normalized = resolve_catalog_recipe(reference, overrides, sctx=sctx)
    runtime = resolve_runtime(recipe, sctx=sctx)
    trusted = resolve_recipe_trust(recipe, False)
    issues = [
        issue.to_dict()
        for issue in validate_recipe(
            recipe, runtime=runtime, overrides=normalized, config=sctx.config, v=sctx.variables, include_unmapped_keys=False
        )
    ][:30]
    for issue in issues:
        if issue["code"] in {"unknown-top-level-key", "misplaced-config-key"}:
            issue["severity"] = "error"
    from sparkrun.core.launcher import _enforce_recipe_mount_trust
    from sparkrun.core.recipe import RecipeError

    try:
        _enforce_recipe_mount_trust(recipe, trusted)
    except RecipeError:
        issues.append(
            {
                "severity": "error",
                "code": "recipe_trust_required",
                "message": "This recipe requires explicit trust for host access; review it on the control node.",
            }
        )
    hooks = bool(recipe.pre_exec or recipe.post_exec or recipe.post_commands or recipe.mods)
    if hooks and not trusted:
        issues.append(
            {
                "severity": "error",
                "code": "recipe_trust_required",
                "message": "Recipe hooks require registry trust before unattended launch.",
            }
        )
    if recipe.stop_after_post:
        issues.append(
            {"severity": "error", "code": "stops_after_start", "message": "This recipe stops itself after its post-launch hooks."}
        )
    if imported and (recipe.mods or recipe.builder_config):
        issues.append(
            {
                "severity": "error",
                "code": "import_auxiliary_files",
                "message": "This upload references mods or builder configuration. Use a registry or a local recipe with its auxiliary files instead.",
            }
        )
    if imported:
        path.touch()  # abandoned imports expire only after a week without a preview
    selected_keys = set(getattr(recipe, "_raw", {}))
    plugins = sorted({item.owner for item in registered_recipe_items() if item.key in selected_keys})
    defaults = {
        key: recipe.defaults[key]
        for key in (
            "tensor_parallel",
            "pipeline_parallel",
            "data_parallel",
            "max_model_len",
            "gpu_memory_utilization",
            "port",
            "served_model_name",
        )
        if key in recipe.defaults
    }
    return {
        "reference": _reference(path, registry, sctx, imported=imported),
        "name": recipe.qualified_name,
        "source_path": str(path),
        "registry": registry,
        "model": recipe.effective_served_model_name or recipe.model,
        "hf_model": recipe.model,
        "runtime": recipe.runtime,
        "description": recipe.description or "",
        "min_nodes": recipe.min_nodes,
        "defaults": defaults,
        "metadata": _declared_facets(path),
        "recipe_revision": derive_recipe_fingerprint(recipe, normalized),
        "plugin_items": recipe.export_plugin_items(),
        "native_api_options": runtime.native_api_options(),
        "native_protocols": list(runtime.native_protocols(recipe) or ("openai",)),
        "capabilities": sorted(set(getattr(recipe, "capabilities", []) or []) | set(runtime.native_capabilities(recipe))),
        "required_plugins": plugins,
        "available_plugins": sorted({item.owner for item in registered_recipe_items()}),
        "trusted": trusted,
        "issues": issues,
    }


def import_recipe(content: str, *, sctx=None) -> dict[str, Any]:
    """Import a single YAML document. Never executes hooks or resolves build assets."""
    import yaml
    from sparkrun.core.recipe import Recipe

    if len(content.encode()) > MAX_RECIPE_BYTES:
        raise SparkrunError("Recipe exceeds the size limit")
    try:
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise ValueError("expected object")
        Recipe.from_dict(data)
    except Exception as exc:
        raise SparkrunError("Import must contain a valid recipe YAML document") from exc
    sctx = resolve_sctx(sctx)
    identity = hashlib.sha256(content.encode()).hexdigest()
    path = _root(sctx) / "imports" / (identity + ".yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Immutable content-addressed import, also safe across concurrent imports.
    if not path.exists():
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".import-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(content)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return get_recipe_details(_reference(path, None, sctx, imported=True), sctx=sctx)


def refresh_registries(*, progress=None, sctx=None) -> dict[str, Any]:
    """Explicitly initialize/update registries, preserving per-registry outcomes."""
    sctx = resolve_sctx(sctx)
    manager = sctx.registry_manager
    updated = manager.update(progress=progress) or {}
    return {
        "updated": {str(key): bool(value) for key, value in updated.items()},
        "failed": sorted(str(key) for key, value in updated.items() if not value),
    }


def retain_catalog_recipe(reference: str, *, sctx=None) -> None:
    """Retain a managed import when a client commits a persistent binding."""
    sctx = resolve_sctx(sctx)
    path, registry, imported = _selection(reference, sctx)
    if not imported:
        return
    canonical = _reference(path, registry, sctx, imported=True)
    record = _root(sctx) / "references" / (canonical.split(":")[1] + ".json")
    value = json.loads(record.read_text())
    value["retained"] = True
    _atomic(record, value)


def cleanup_catalog_imports(*, sctx=None, max_age_seconds: float = 7 * 86400) -> int:
    """Remove abandoned staged uploads. Persistently bound imports never expire."""
    sctx = resolve_sctx(sctx)
    removed = 0
    for path in (_root(sctx) / "imports").glob("*.yaml"):
        if path.stat().st_mtime >= time.time() - max_age_seconds:
            continue
        ref = _reference(path, None, sctx, imported=True)
        record = _root(sctx) / "references" / (ref.split(":")[1] + ".json")
        value = json.loads(record.read_text())
        if not value.get("retained"):
            path.unlink(missing_ok=True)
            record.unlink(missing_ok=True)
            removed += 1
    return removed


CATALOG_FACETS = ("min_nodes", "tp", "pp", "quantization", "context_length", "parameters_b")


def _declared_facets(path: Path) -> dict[str, Any]:
    """Only declared YAML metadata; no name heuristics, network, or HF resolver."""
    import math
    import yaml

    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_RECIPE_BYTES + 1)
        if len(raw) > MAX_RECIPE_BYTES:
            return {}
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            return {}
        defaults = data.get("defaults") or {}
        metadata = data.get("metadata") or {}
        if not isinstance(defaults, dict) or not isinstance(metadata, dict):
            return {}

        def number(value):
            return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 else None

        params = number(metadata.get("model_params"))
        quant = metadata.get("quantization", defaults.get("quantization"))
        return {
            "min_nodes": number(data.get("min_nodes", 1)),
            "tp": number(defaults.get("tensor_parallel")),
            "pp": number(defaults.get("pipeline_parallel")),
            "quantization": quant[:128] if isinstance(quant, str) and quant else None,
            "context_length": number(defaults.get("max_model_len")),
            "parameters_b": params / 1e9 if params else None,
            "benchmarks": _benchmark_context(metadata.get("benchmarks")),
        }
    except (OSError, ValueError, yaml.YAMLError):
        return {}


def configure_registry(
    action: str, name: str, *, url: str = "", subpath: str = "", acknowledge_trust: bool = False, sctx=None
) -> dict[str, Any]:
    """Explicit configuration changes; adding never clones or grants trust."""
    from sparkrun.core.registry import RegistryEntry, RegistryError

    sctx = resolve_sctx(sctx)
    manager = sctx.registry_manager
    try:
        if action == "add":
            manager.add_registry(RegistryEntry(name=name, url=url, subpath=subpath, trusted=False))
        elif action == "trust":
            if acknowledge_trust is not True:
                raise SparkrunError("Review the registry and explicitly acknowledge that its recipes may execute hooks")
            manager.trust_registry(name)
        elif action in {"remove", "enable", "disable", "untrust"}:
            getattr(manager, action + "_registry")(name)
        else:
            raise SparkrunError("Unsupported registry action")
    except (RegistryError, ValueError) as exc:
        raise SparkrunError("Registry configuration failed: %s" % str(exc)) from exc
    return {"registries": list_registries(sctx=sctx)}


def catalog_cluster_capacity(cluster: str, *, sctx=None) -> dict[str, Any]:
    """Explicit live advisory occupancy probe; never reserves or launches."""
    from sparkrun.api._status import status

    sctx = resolve_sctx(sctx)
    definition = sctx.cluster_manager.get(cluster)
    observed = status(list(definition.hosts), cluster=definition, sctx=sctx)
    rows = []
    for host in definition.hosts[:256]:
        occupancy = observed.for_host(host)
        rows.append(
            {
                "host": host,
                "reachable": occupancy is not None,
                "free_slots": occupancy.free_slots if occupancy else None,
                "used_slots": occupancy.used_slots if occupancy else None,
                "workloads": len(occupancy.workloads) if occupancy else None,
            }
        )
    return {"cluster": cluster, "observed_at": time.time(), "hosts": rows, "advisory": True}


def _benchmark_context(value) -> list[dict[str, Any]]:
    """Bounded, explicitly declared context; these are not measured by browsing."""
    import math

    if not isinstance(value, list):
        return []
    result = []
    for entry in value[:10]:
        if not isinstance(entry, dict):
            continue
        row = {}
        for key in ("output_tokens_per_second", "time_to_first_token_ms", "input_tokens", "output_tokens", "concurrency"):
            item = entry.get(key)
            if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) and item >= 0:
                row[key] = item
        for key in ("hardware", "runtime", "date", "description"):
            item = entry.get(key)
            if isinstance(item, str):
                row[key] = item[:256]
        if row:
            result.append(row)
    return result
