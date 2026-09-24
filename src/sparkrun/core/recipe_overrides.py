"""Recipe ``overrides:`` — conditional tuning for one model on many layouts.

A recipe carries its base settings in ``defaults`` / ``env`` / ``container``
and a list of conditional layers on top::

    overrides:
      - when: {nodes: 1}
        defaults: {max_num_seqs: 1, max_model_len: 65536}
      - when: {arch: sm_120}
        defaults: {gpu_memory_utilization: 0.95}
      - when: {config: {speculator: none}}
        defaults: {max_num_seqs: 32}

**What belongs here.** Settings unique to this model on a given layout or
hardware *combination*. Facts about the hardware itself (``CUTE_DSL_ARCH``,
allocator env, memory ceilings) belong in the platform tier
(:mod:`sparkrun.platforms`). If a recipe needs an override to restate one, the
platform is missing it.

Semantics:

* Keys in one ``when`` AND together. A value is a scalar (equality), a list
  (any-of), or a mapping of operators (:data:`OPERATORS`).
* Matching entries apply in list order and later wins. Precedence overall:
  runtime < platform < recipe ``defaults`` < matched overrides < CLI ``-o``.
* ``config:`` predicates read the configuration **before** any override
  applies, so overrides cannot feed each other and order never changes
  whether one matches.
* A selector or operator this sparkrun does not know makes the entry **not
  match**, with a warning. Registries version independently of sparkrun, and
  "matched because we could not tell" would apply tuning to the wrong launch.
* Hardware selectors must evaluate the same on every host the launch may use.
  Launch-wide layers take one value per launch, so a split answer is an
  error rather than silently using the head host's.

**Identity.** Applying overrides mutates the recipe's *effective*
``defaults`` / ``env`` / ``container`` so every consumer sees them unchanged,
but the recipe snapshots the *declared* values first
(:meth:`Recipe.declared_defaults` and friends). ``generate_intent_id`` and
``derive_recipe_fingerprint`` read the snapshot, because ``stop`` / ``logs`` /
``--ensure`` recompute them from a freshly loaded recipe that never had
overrides applied. Layers therefore may not set anything the intent hashes or
anything a ``when`` can read (:data:`FORBIDDEN_DEFAULT_KEYS`).
"""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from sparkrun.core.hardware import normalize_arch
from sparkrun.core.parallelism import PARALLELISM_KEYS

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.hardware import HostHardware
    from sparkrun.core.recipe import Recipe
    from sparkrun.runtimes.base import RuntimePlugin

logger = logging.getLogger(__name__)

#: Layers an override may set.
OVERRIDE_LAYERS = ("defaults", "env", "container")

#: Recipe surfaces reserved for a later phase. Named so a recipe that uses one
#: is told "not yet", not "unknown key".
RESERVED_LAYERS = frozenset({"mods", "builder_config", "executor_config", "command"})

_SHAPE_SELECTORS = {short: long for long, short in PARALLELISM_KEYS}
#: Launch-shape selectors: parallelism short names plus derived node shape.
SHAPE_SELECTORS = frozenset(_SHAPE_SELECTORS) | {"nodes", "gpus_per_node"}
RUNTIME_SELECTORS = frozenset({"runtime", "runtime_family"})
HARDWARE_SELECTORS = frozenset({"platform", "vendor", "accelerator", "arch", "capability", "memory_gb"})
CONFIG_SELECTOR = "config"
SELECTORS = SHAPE_SELECTORS | RUNTIME_SELECTORS | HARDWARE_SELECTORS | {CONFIG_SELECTOR}

OPERATORS = frozenset({"eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte"})
_NUMERIC_OPERATORS = frozenset({"gt", "gte", "lt", "lte"})

#: Selectors whose values are names or tag sets: ordering them is meaningless,
#: so ``gt`` / ``lt`` there is reported rather than silently false.
_UNORDERED_SELECTORS = frozenset({"runtime", "runtime_family", "platform", "vendor", "accelerator", "arch", "capability"})

#: ``defaults`` keys an override may not set: what the intent hashes (a
#: conditional value there would make ``stop`` / ``logs`` / ``--ensure`` stop
#: matching the workload), and parallelism, which ``when`` reads.
FORBIDDEN_DEFAULT_KEYS = frozenset({"port", "served_model_name"}) | {long for long, _ in PARALLELISM_KEYS}


class OverrideConflictError(ValueError):
    """A hardware predicate evaluates differently across the launch's hosts."""


@dataclass(frozen=True)
class RecipeOverride:
    """One ``overrides:`` entry, structurally validated."""

    index: int
    when: dict[str, Any]
    defaults: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    container: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"when": deepcopy(self.when)}
        if self.defaults:
            d["defaults"] = deepcopy(self.defaults)
        if self.env:
            d["env"] = dict(self.env)
        if self.container is not None:
            d["container"] = self.container
        return d

    def config_keys(self) -> set[str]:
        """Config keys this entry's ``when`` reads (they count as consumed)."""
        cfg = self.when.get(CONFIG_SELECTOR)
        return {str(k) for k in cfg} if isinstance(cfg, dict) else set()


def parse_overrides(raw: Any) -> list[RecipeOverride]:
    """Parse and structurally validate a recipe's ``overrides:`` block.

    Raises :class:`~sparkrun.core.recipe.RecipeError` for shapes that cannot
    be honored (not a list, empty ``when``, unknown or forbidden layer keys).
    Unknown *selectors* are not structural: they are kept, and make the entry
    not match at evaluation time.
    """
    from sparkrun.core.recipe import RecipeError

    if raw is None:
        return []
    if not isinstance(raw, list):
        raise RecipeError("overrides: must be a list of {when: …, defaults/env/container: …} entries")
    parsed: list[RecipeOverride] = []
    for index, entry in enumerate(raw):
        where = "overrides[%d]" % index
        if not isinstance(entry, dict):
            raise RecipeError("%s must be a mapping" % where)
        when = entry.get("when")
        if not isinstance(when, dict) or not when:
            raise RecipeError("%s needs a non-empty `when:` mapping (use `defaults:` for unconditional values)" % where)
        if CONFIG_SELECTOR in when and (not isinstance(when[CONFIG_SELECTOR], dict) or not when[CONFIG_SELECTOR]):
            raise RecipeError("%s: `when.config` must be a non-empty mapping of config key → predicate" % where)
        layer_keys = set(entry) - {"when"}
        reserved = sorted(layer_keys & RESERVED_LAYERS)
        if reserved:
            raise RecipeError("%s: %s cannot be overridden yet (supported: %s)" % (where, ", ".join(reserved), ", ".join(OVERRIDE_LAYERS)))
        unknown = sorted(layer_keys - set(OVERRIDE_LAYERS))
        if unknown:
            raise RecipeError("%s: unknown key(s) %s (supported: when, %s)" % (where, ", ".join(unknown), ", ".join(OVERRIDE_LAYERS)))
        if not layer_keys:
            raise RecipeError("%s sets nothing (add defaults, env or container)" % where)

        defaults = entry.get("defaults") or {}
        if not isinstance(defaults, dict):
            raise RecipeError("%s: defaults must be a mapping" % where)
        forbidden = sorted(set(defaults) & FORBIDDEN_DEFAULT_KEYS)
        if forbidden:
            raise RecipeError(
                "%s: %s cannot be overridden — they identify the workload or are read by `when:`" % (where, ", ".join(forbidden))
            )
        env = entry.get("env") or {}
        if not isinstance(env, dict):
            raise RecipeError("%s: env must be a mapping" % where)
        container = entry.get("container")
        if container is not None and (not isinstance(container, str) or not container.strip()):
            raise RecipeError("%s: container must be a non-empty image reference" % where)
        parsed.append(
            RecipeOverride(
                index=index,
                when=deepcopy(when),
                defaults=deepcopy(defaults),
                env={str(k): str(v) for k, v in env.items()},
                container=container.strip() if container else None,
            )
        )
    return parsed


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverrideContext:
    """Everything a ``when`` may read, gathered once per launch."""

    shape: dict[str, int] = field(default_factory=dict)
    runtime: str | None = None
    runtime_family: str | None = None
    config: Mapping[str, Any] | Any = field(default_factory=dict)
    hosts: dict[str, HostHardware] = field(default_factory=dict)


@dataclass(frozen=True)
class OverrideDecision:
    index: int
    matched: bool
    reason: str
    unknown: tuple[str, ...] = ()


@dataclass(frozen=True)
class OverrideResolution:
    """What applying a recipe's overrides did, for display and diagnostics."""

    decisions: tuple[OverrideDecision, ...] = ()
    defaults: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    container: str | None = None
    skipped_env: tuple[str, ...] = ()
    skipped_container: bool = False

    @property
    def matched(self) -> tuple[int, ...]:
        return tuple(d.index for d in self.decisions if d.matched)

    def describe(self) -> list[str]:
        """One line per entry, for ``--dry-run`` / plan display."""
        lines = []
        for d in self.decisions:
            lines.append("overrides[%d]: %s — %s" % (d.index, "matched" if d.matched else "skipped", d.reason))
        if self.skipped_env:
            lines.append("overrides: env %s left to the CLI value" % ", ".join(self.skipped_env))
        if self.skipped_container:
            lines.append("overrides: container left to --image")
        return lines


def _compile(value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        return [(str(op), operand) for op, operand in value.items()]
    if isinstance(value, list):
        return [("in", value)]
    return [("eq", value)]


def _norm(selector: str, value: Any) -> Any:
    if value is None:
        return None
    if selector == "arch":
        return normalize_arch(value) or str(value).strip().lower()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        return text.lower()


def _test(selector: str, op: str, operand: Any, actual: Any, *, absent_is_value: bool = False) -> bool:
    """One predicate against one value. ``actual`` may be a set (capability).

    An unknown value never satisfies ``eq`` / ``in`` / ordering. For config keys
    (*absent_is_value*) an unset key is a real answer, so ``ne`` / ``not_in``
    hold; for hardware, "unknown" is not "different", so nothing holds.
    """
    if actual is None:
        return absent_is_value and op in ("ne", "not_in")
    if isinstance(actual, (set, frozenset)):
        values = {_norm(selector, v) for v in actual}
        wanted = {_norm(selector, v) for v in (operand if isinstance(operand, list) else [operand])}
        if op in ("eq", "in"):
            return bool(values & wanted)
        if op in ("ne", "not_in"):
            return not values & wanted
        return False
    a = _norm(selector, actual)
    if op in ("eq", "ne", "in", "not_in"):
        # ``True`` must not equal ``1``: a boolean compares only with a boolean.
        candidates = operand if op in ("in", "not_in") and isinstance(operand, list) else [operand]
        normalized = [_norm(selector, c) for c in candidates]
        hit = any(isinstance(a, bool) == isinstance(c, bool) and a == c for c in normalized)
        return hit if op in ("eq", "in") else not hit
    if op in _NUMERIC_OPERATORS:
        b = _norm(selector, operand)
        if not isinstance(a, float) or not isinstance(b, float) or isinstance(a, bool) or isinstance(b, bool):
            return False
        return {"gt": a > b, "gte": a >= b, "lt": a < b, "lte": a <= b}[op]
    return False


def _check(selector: str, predicates: list[tuple[str, Any]], actual: Any, *, absent_is_value: bool = False) -> tuple[bool, list[str]]:
    unknown = [op for op, _ in predicates if op not in OPERATORS]
    if unknown:
        return False, ["%s: unknown operator %s" % (selector, ", ".join(sorted(unknown)))]
    unordered = [op for op, _ in predicates if op in _NUMERIC_OPERATORS and selector in _UNORDERED_SELECTORS]
    if unordered:
        return False, ["%s: %s needs a numeric selector" % (selector, ", ".join(sorted(unordered)))]
    return all(_test(selector, op, operand, actual, absent_is_value=absent_is_value) for op, operand in predicates), []


def _accelerator_facts(host_hardware: HostHardware) -> list[dict[str, Any]]:
    # Deferred: platforms import orchestration, which the recipe layer must not
    # pull in at import time.
    from sparkrun.core.limits import resolve_accelerator_memory_gb
    from sparkrun.platforms import resolve_accelerator_arch, resolve_accelerator_platform

    facts = []
    for accel in host_hardware.accelerators:
        platform = resolve_accelerator_platform(accel, host_hardware)
        facts.append(
            {
                "platform": platform.platform_name if platform is not None else None,
                "vendor": accel.vendor,
                "accelerator": accel.model,
                "arch": resolve_accelerator_arch(accel, host_hardware),
                "capability": frozenset(accel.capabilities),
                "memory_gb": resolve_accelerator_memory_gb(accel, host_hardware),
            }
        )
    return facts


def _hardware_predicates(override: RecipeOverride) -> dict[str, list[tuple[str, Any]]] | None:
    """The entry's hardware predicates, or ``None`` when it has none it can evaluate.

    An entry with an unknown selector or operator never matches, so it cannot
    split hosts either.
    """
    if any(selector not in SELECTORS for selector in override.when):
        return None
    hardware = {sel: _compile(val) for sel, val in override.when.items() if sel in HARDWARE_SELECTORS}
    if not hardware or any(_check(sel, preds, None)[1] for sel, preds in hardware.items()):
        return None
    return hardware


def partition_hosts_by_hardware(overrides: list[RecipeOverride], hosts: Mapping[str, HostHardware]) -> list[tuple[str, ...]]:
    """Split *hosts* into groups that answer every hardware ``when:`` the same way.

    Groups keep the hosts' order and are ordered by where each group's first
    host appears, so the result is deterministic. A recipe with no hardware
    predicates, or a homogeneous cluster, yields a single group. A launch
    placed within one group can never hit :class:`OverrideConflictError`.
    """
    predicates = [p for p in (_hardware_predicates(o) for o in overrides) if p is not None]
    groups: dict[tuple[bool, ...], list[str]] = {}
    for host, hardware in hosts.items():
        signature = tuple(_host_matches(p, hardware) for p in predicates)
        groups.setdefault(signature, []).append(host)
    return [tuple(members) for members in groups.values()]


def _host_matches(hardware_when: dict[str, list[tuple[str, Any]]], host_hardware: HostHardware) -> bool:
    """A host matches when one accelerator satisfies every hardware predicate."""
    for facts in _accelerator_facts(host_hardware):
        if all(_check(sel, preds, facts.get(sel))[0] for sel, preds in hardware_when.items()):
            return True
    return False


def evaluate_override(override: RecipeOverride, ctx: OverrideContext) -> OverrideDecision:
    """Decide one entry. Raises :class:`OverrideConflictError` on a host split."""
    unknown: list[str] = []
    failed: list[str] = []
    hardware_when: dict[str, list[tuple[str, Any]]] = {}

    for selector, value in override.when.items():
        if selector not in SELECTORS:
            unknown.append("unknown selector %r" % selector)
            continue
        if selector == CONFIG_SELECTOR:
            for key, predicate in value.items():
                ok, bad = _check("config.%s" % key, _compile(predicate), _config_get(ctx.config, str(key)), absent_is_value=True)
                unknown.extend(bad)
                if not ok and not bad:
                    failed.append("config.%s" % key)
            continue
        predicates = _compile(value)
        if selector in HARDWARE_SELECTORS:
            _ok, bad = _check(selector, predicates, None)
            if bad:
                unknown.extend(bad)
            else:
                hardware_when[selector] = predicates
            continue
        actual = ctx.shape.get(selector) if selector in SHAPE_SELECTORS else getattr(ctx, selector)
        ok, bad = _check(selector, predicates, actual)
        unknown.extend(bad)
        if not ok and not bad:
            failed.append("%s=%s" % (selector, actual if actual is not None else "unknown"))

    if unknown:
        return OverrideDecision(override.index, False, "; ".join(unknown) + " (this sparkrun cannot evaluate it)", tuple(unknown))
    if failed:
        return OverrideDecision(override.index, False, "%s did not match" % ", ".join(failed))

    if hardware_when:
        if not ctx.hosts:
            return OverrideDecision(override.index, False, "no host hardware known to evaluate %s" % ", ".join(sorted(hardware_when)))
        verdicts = {host: _host_matches(hardware_when, hw) for host, hw in ctx.hosts.items()}
        if len(set(verdicts.values())) > 1:
            yes = sorted(h for h, v in verdicts.items() if v)
            no = sorted(h for h, v in verdicts.items() if not v)
            raise OverrideConflictError(
                "overrides[%d] (%s) matches %s but not %s; a launch-wide override needs every host to agree — "
                "narrow the hosts (--hosts) or split the recipe"
                % (override.index, ", ".join(sorted(hardware_when)), ", ".join(yes), ", ".join(no))
            )
        if not next(iter(verdicts.values())):
            return OverrideDecision(override.index, False, "%s did not match the hosts" % ", ".join(sorted(hardware_when)))
    return OverrideDecision(override.index, True, "all conditions matched")


def _config_get(config: Any, key: str) -> Any:
    try:
        return config.get(key)
    except Exception:
        logger.debug("override config predicate: could not read %r", key, exc_info=True)
        return None


def build_override_context(
    recipe: Recipe,
    cli_overrides: dict[str, Any] | None = None,
    *,
    runtime: RuntimePlugin | None = None,
    cluster: ClusterDefinition | None = None,
    hosts: list[str] | tuple[str, ...] = (),
    host_hardware: dict[str, HostHardware] | None = None,
    solo: bool = False,
    declared: bool = True,
) -> OverrideContext:
    """Gather what ``when`` can read for one launch, from declared config.

    *host_hardware* (observed this launch) wins over *cluster* inventory, which
    falls back to the platform assumption, the same order placement uses.
    """
    from sparkrun.core.parallelism import extract_parallelism

    chain = recipe.build_config_chain(cli_overrides, declared=declared)
    parallelism = extract_parallelism(chain)

    from sparkrun.core.hardware import resolve_hardware

    hardware: dict[str, HostHardware] = {}
    for host in hosts:
        observed = (host_hardware or {}).get(host)
        if observed is not None and observed.source == "detected":
            hardware[host] = observed
        else:
            # Inventory, then the application-policy assumption: the order
            # placement uses. A host must never drop out of every group.
            hardware[host] = cluster.hardware_for(host) if cluster is not None else resolve_hardware(None)

    gpus = [hw.total_gpus for hw in hardware.values() if hw.total_gpus > 0]
    gpus_per_node = min(gpus) if gpus else 1

    world = parallelism.world_size()
    if runtime is not None and cluster is not None:
        try:
            world = runtime.world_size(parallelism, recipe=recipe, cluster=cluster)
        except Exception:
            logger.debug("override context: runtime world_size failed; using tp*pp*dp", exc_info=True)

    layout = getattr(recipe, "layout", None)
    placements = getattr(layout, "placements", None) or []
    if solo or recipe.mode == "solo":
        nodes = 1
    elif placements:
        nodes = len({p.host for p in placements})
    else:
        nodes = max(1, math.ceil(world / gpus_per_node))

    shape = {short: int(getattr(parallelism, long)) for long, short in PARALLELISM_KEYS}
    shape.update(nodes=nodes, gpus_per_node=gpus_per_node)
    return OverrideContext(
        shape=shape,
        runtime=recipe.runtime or (runtime.runtime_name if runtime is not None else None),
        runtime_family=runtime.get_family() if runtime is not None else None,
        config=chain,
        hosts=hardware,
    )


def apply_recipe_override_layers(recipe: Recipe, ctx: OverrideContext) -> OverrideResolution:
    """Evaluate *recipe*'s overrides against *ctx* and apply the matches.

    Idempotent: effective values are rebuilt from the declared snapshot on
    every call, so re-applying after a changed context never stacks layers.
    CLI-set env keys and ``--image`` are never overwritten.
    """
    overrides = getattr(recipe, "overrides", None) or []
    recipe.restore_declared_values()
    if not overrides:
        resolution = OverrideResolution()
        recipe.override_resolution = resolution
        return resolution

    decisions = tuple(evaluate_override(o, ctx) for o in overrides)
    for decision in decisions:
        if decision.unknown:
            logger.warning("Recipe %s: overrides[%d] skipped — %s", recipe.name, decision.index, decision.reason)

    defaults: dict[str, Any] = {}
    env: dict[str, str] = {}
    container: str | None = None
    for override, decision in zip(overrides, decisions, strict=True):
        if not decision.matched:
            continue
        defaults.update(deepcopy(override.defaults))
        env.update(override.env)
        if override.container is not None:
            container = override.container

    cli_env = set(getattr(recipe, "_cli_env_keys", ()) or ())
    skipped_env = tuple(sorted(k for k in env if k in cli_env))
    applied_env = {k: v for k, v in env.items() if k not in cli_env}
    skipped_container = container is not None and bool(getattr(recipe, "_cli_image", False))

    recipe.snapshot_declared_values()
    recipe.defaults.update(defaults)
    recipe.env.update(applied_env)
    if container is not None and not skipped_container:
        recipe.container = container

    resolution = OverrideResolution(
        decisions=decisions,
        defaults=defaults,
        env=applied_env,
        container=None if skipped_container else container,
        skipped_env=skipped_env,
        skipped_container=skipped_container,
    )
    recipe.override_resolution = resolution
    return resolution


__all__ = [
    "CONFIG_SELECTOR",
    "FORBIDDEN_DEFAULT_KEYS",
    "HARDWARE_SELECTORS",
    "OPERATORS",
    "OVERRIDE_LAYERS",
    "OverrideConflictError",
    "OverrideContext",
    "OverrideDecision",
    "OverrideResolution",
    "RecipeOverride",
    "SELECTORS",
    "apply_recipe_override_layers",
    "build_override_context",
    "evaluate_override",
    "parse_overrides",
    "partition_hosts_by_hardware",
]
