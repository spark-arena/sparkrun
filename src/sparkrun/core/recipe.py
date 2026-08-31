"""Recipe loading, validation, and v1->v2 migration."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, asdict as dataclass_asdict
from json import dumps as json_dumps
from pathlib import Path
from typing import Any, TYPE_CHECKING, Optional

import yaml

from vpd.next.util import read_yaml
from scitrera_app_framework.api import Variables, EnvPlacement

from sparkrun.core.images import parse_container_entries
from sparkrun.core.layout import RecipeLayout
from sparkrun.core.recipe_items import get_recipe_item, registered_recipe_items
from sparkrun.utils.text import mask_non_placeholder_braces, render_template, unmask_braces, uses_brace_escapes

if TYPE_CHECKING:
    from sparkrun.core.registry import RegistryManager
    from sparkrun.models.vram import VRAMEstimate

logger = logging.getLogger(__name__)

# Matches a backslash followed by trailing whitespace before a newline.
# In bash, ``\<newline>`` is a line continuation but ``\ <newline>`` is
# an escaped space — a common YAML editing mistake that silently breaks
# multi-line commands.
_TRAILING_SPACE_CONTINUATION_RE = re.compile(r"\\ +\n")

_RAY_BACKEND_RE = re.compile(r"--distributed-executor-backend\s+ray\b")
# --kv-cache-dtype (vllm/sglang/atlas: --kv-cache-dtype) or tokenary (--kvcache-dtype),
# space- or =-separated. Captured so a recipe that sets the flag only inside the
# free-form command: template is still picked up by the VRAM estimator (issue #248).
_KV_CACHE_DTYPE_FLAG_RE = re.compile(r"--kv-?cache-dtype[=\s]+(\S+)", re.IGNORECASE)
# The name a workload is *served under*, spelled differently per runtime:
# vllm / sglang / modular-max use --served-model-name, atlas --model-name,
# llama.cpp --alias.  (llama.cpp's short "-a" is deliberately excluded: a bare
# short flag is too easy to collide with another tool's option.)  Captured so a
# recipe that sets the name only inside the free-form command: template is still
# visible to the benchmark, the proxy and the container labels — see
# :func:`extract_served_model_name_from_command`.
_SERVED_MODEL_NAME_FLAG_RE = re.compile(r"--(?:served-model-name|model-name|alias)[=\s]+(\S+)", re.IGNORECASE)
_CMD_VLLM_RE = re.compile(r"^vllm\s+serve\b")
_CMD_SGLANG_RE = re.compile(r"^(?:sglang\s+serve|python3?\s+-m\s+sglang\.launch_server)\b")
_CMD_LLAMA_CPP_RE = re.compile(r"^llama-server\b")
_CMD_TRTLLM_RE = re.compile(r"^(?:trtllm-serve|mpirun\b.*trtllm)")

_KNOWN_KEYS = {
    "sparkrun_version",
    "recipe_version",
    "name",
    "description",
    "model",
    "model_revision",
    "runtime",
    "runtime_version",
    "mode",
    "min_nodes",
    "max_nodes",
    "container",
    "containers",
    "defaults",
    "env",
    "command",
    "runtime_config",
    "cluster_only",
    "solo_only",
    "benchmark",
    "metadata",
    "pre_exec",
    "post_exec",
    "post_commands",
    "mods",
    "stop_after_post",
    "builder",
    "builder_config",
    "executor",
    "executor_config",
    "scheduler",
    "distribution_config",
    "layout",
    "cluster_config",
    "runtime_cache",
    "capabilities",
    "unsupported_capabilities",
}


@dataclass
class DistributionModelEntry:
    """A single model to distribute during the distribution phase.

    ``revision`` is **per-entry and authoritative** — there is no fallback to
    the recipe's ``model_revision``.  A launch distributes several unrelated
    repos (the served model plus any speculative draft model a runtime adds in
    ``prepare()``), and a commit SHA is only meaningful in the repo it came
    from: pinning the draft model to the served model's SHA asks the Hub for a
    revision that repo has never had, and the download dies with
    ``Revision Not Found`` after the served model has already synced.

    The recipe-level pin reaches the served model by being stamped onto the
    auto-generated entry at construction (see
    :func:`_default_distribution_config`), so a recipe that hand-writes its
    ``distribution_config`` entries states every revision it wants — an entry
    with no ``revision`` is authoritatively unpinned.
    """

    name: str
    target: list[int] = field(default_factory=list)
    """Node indices to distribute to. ``[-1]`` means all nodes."""
    revision: str | None = None


@dataclass
class DistributionContainerEntry:
    """A single container image to distribute during the distribution phase."""

    name: str
    target: list[int] = field(default_factory=list)
    """Node indices to distribute to. ``[-1]`` means all nodes."""


@dataclass
class DistributionResourceConfig:
    """Distribution settings for a resource type (models or containers)."""

    enabled: bool = True
    entries: list[DistributionModelEntry | DistributionContainerEntry] = field(default_factory=list)

    explicit: bool = False
    """True when the recipe wrote this resource's block itself.

    Finer-grained than :attr:`DistributionConfig.externally_provided`, which is
    whole-config: a recipe that customizes only ``models`` still gets the
    *auto-generated* container entry, and the launcher must be able to tell that
    apart from a hand-written one before it derives container entries from the
    per-machine image plan (deriving over a hand-written block would discard the
    user's choice; *not* deriving over the auto one would ship only the fallback
    image to every machine).
    """


@dataclass
class DistributionConfig:
    """Controls model and container distribution behavior.

    Auto-generated by default; can be set in recipe YAML or mutated
    by runtimes during ``prepare()``.
    """

    models: DistributionResourceConfig = field(default_factory=DistributionResourceConfig)
    containers: DistributionResourceConfig = field(default_factory=DistributionResourceConfig)
    externally_provided: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DistributionConfig":
        """Create a DistributionConfig from a raw dict."""

        def model_entry_from_dict(entry: dict[str, Any]) -> DistributionModelEntry:
            return DistributionModelEntry(
                name=entry["name"],
                target=list(entry.get("target", [])),
                revision=entry.get("revision"),
            )

        def container_entry_from_dict(entry: dict[str, Any]) -> DistributionContainerEntry:
            return DistributionContainerEntry(
                name=entry["name"],
                target=list(entry.get("target", [])),
            )

        def resource_config_from_dict(
            raw: dict[str, Any] | None,
            entry_factory,
        ) -> DistributionResourceConfig:
            if raw is None:
                return DistributionResourceConfig()

            return DistributionResourceConfig(
                enabled=raw.get("enabled", True),
                entries=[
                    entry if isinstance(entry, (DistributionModelEntry, DistributionContainerEntry)) else entry_factory(entry)
                    for entry in raw.get("entries", [])
                ],
                explicit=bool(raw.get("explicit", False)),
            )

        return cls(
            models=(
                data["models"]
                if isinstance(data.get("models"), DistributionResourceConfig)
                else resource_config_from_dict(
                    data.get("models"),
                    model_entry_from_dict,
                )
            ),
            containers=(
                data["containers"]
                if isinstance(data.get("containers"), DistributionResourceConfig)
                else resource_config_from_dict(
                    data.get("containers"),
                    container_entry_from_dict,
                )
            ),
            externally_provided=data.get("externally_provided", True),
        )

    def add_model(self, model: str, revision: str | None = None):
        """Add a model distribution config to the distribution config.

        ``revision`` pins *this* model only.  Runtimes call this from
        ``prepare()`` to add a speculative draft model, which is a different
        repo from the served model and must never inherit its pin — see
        :class:`DistributionModelEntry`.
        """
        # scan through existing models and make sure that we don't add a duplicate
        for existing_model in self.models.entries:
            if existing_model.name == model:
                # A later caller with a pin outranks an earlier unpinned add:
                # dropping it would fetch the model unpinned and silently
                # discard the revision the recipe asked for.
                if revision:
                    existing_model.revision = revision
                return
        self.models.entries.append(DistributionModelEntry(name=model, revision=revision))

    def add_container(self, model_container_config: DistributionContainerEntry):
        self.containers.entries.append(model_container_config)

    def resolve(self, recipe: "Recipe", resolved_container: str = None, overrides: dict[str, Any] | None = None) -> "DistributionConfig":
        """Resolve templated names in distribution config using the config chain.

        Replaces ``{model}``, ``{container}`` placeholders in entry names with
        concrete values from the resolved config.  Mutates the entries in-place
        so runtimes can read them during ``prepare()``.

        Args:
            overrides: CLI overrides dict (optional, uses applied overrides if not given).
        """
        # noinspection PyProtectedMember
        effective_overrides = overrides or recipe._applied_overrides
        config_chain = recipe.build_config_chain(effective_overrides)

        # Inject model and container into chain for substitution
        if resolved_container:  # resolved_container override always takes precedence
            config_chain["container"] = resolved_container
        config_chain["model"] = recipe.model

        for entry in self.models.entries:
            if isinstance(entry, DistributionModelEntry):
                entry.name = render_template(entry.name, config_chain)

        for entry in self.containers.entries:
            if isinstance(entry, DistributionContainerEntry):
                entry.name = render_template(entry.name, config_chain)

        return self


def _default_distribution_config(
    model: str = "{model}",
    container: str = "{container}",
    model_revision: str | None = None,
) -> DistributionConfig:
    """Create the default distribution config for a recipe.

    ``model_revision`` is the recipe's top-level pin.  Stamping it onto the
    auto-generated entry here — rather than applying it at distribution time to
    whatever entries happen to be present — is what keeps it attached to the
    model it actually describes once a runtime has added a draft model
    alongside it.
    """
    return DistributionConfig(
        models=DistributionResourceConfig(
            enabled=True,
            entries=[DistributionModelEntry(name=model, revision=model_revision)],
        ),
        containers=DistributionResourceConfig(
            enabled=True,
            entries=[DistributionContainerEntry(name=container)],
        ),
        externally_provided=False,
    )


def _parse_distribution_config(data: dict[str, Any]) -> DistributionConfig:
    """Parse ``distribution_config`` from raw recipe YAML data.

    An *omitted* ``models`` or ``containers`` subkey falls back to the
    auto-default single entry (``{model}`` / ``{container}``) so a recipe can,
    e.g., add a second model to distribute without having to re-list the
    container it never customized.  A subkey that IS present is honored
    literally — an explicit ``entries: []`` or ``enabled: false`` still means
    "distribute nothing", not "use the default".

    The recipe's top-level ``model_revision`` is stamped onto the auto-generated
    model entry (and onto the one inherited when ``models`` is omitted), so a
    recipe that lists its own entries owns their revisions outright.
    """
    raw = data.get("distribution_config")
    model_revision = data.get("model_revision")
    # fallback if not provided (expected to be the default case)
    if not raw or not isinstance(raw, dict):
        return _default_distribution_config(model_revision=model_revision)

    default = _default_distribution_config(model_revision=model_revision)

    def _parse_models(models_raw: Any) -> DistributionResourceConfig:
        if not isinstance(models_raw, dict):
            models_raw = {}
        entries_raw = models_raw.get("entries", [])
        if not isinstance(entries_raw, list):
            entries_raw = []
        entries: list[DistributionModelEntry | DistributionContainerEntry] = []
        for e in entries_raw:
            if isinstance(e, dict):
                entries.append(
                    DistributionModelEntry(
                        name=e.get("name", ""),
                        target=e.get("target", [-1]),
                        revision=e.get("revision"),
                    )
                )
            elif isinstance(e, str):
                entries.append(DistributionModelEntry(name=e))
        return DistributionResourceConfig(enabled=models_raw.get("enabled", True), entries=entries, explicit=True)

    def _parse_containers(containers_raw: Any) -> DistributionResourceConfig:
        if not isinstance(containers_raw, dict):
            containers_raw = {}
        entries_raw = containers_raw.get("entries", [])
        if not isinstance(entries_raw, list):
            entries_raw = []
        entries: list[DistributionModelEntry | DistributionContainerEntry] = []
        for e in entries_raw:
            if isinstance(e, dict):
                entries.append(
                    DistributionContainerEntry(
                        name=e.get("name", ""),
                        target=e.get("target", [-1]),
                    )
                )
            elif isinstance(e, str):
                entries.append(DistributionContainerEntry(name=e))
        return DistributionResourceConfig(enabled=containers_raw.get("enabled", True), entries=entries, explicit=True)

    return DistributionConfig(
        models=_parse_models(raw["models"]) if "models" in raw else default.models,
        containers=_parse_containers(raw["containers"]) if "containers" in raw else default.containers,
    )


def _sort_dict_by_patterns(data: dict[str, Any], patterns: list[str]) -> dict[str, Any]:
    """Return a new dict with keys ordered according to *patterns*.

    Each entry in *patterns* is either an exact key name or an
    ``fnmatch``-style glob (e.g. ``"model*"``).  Keys are emitted in
    the order of the first pattern they match; keys that match no
    pattern are appended alphabetically at the end.
    """
    from fnmatch import fnmatch

    ordered: dict[str, Any] = {}
    remaining = set(data.keys())

    for pattern in patterns:
        # Collect matching keys in their original insertion order
        matched = [k for k in data if k in remaining and fnmatch(k, pattern)]
        matched.sort()
        for k in matched:
            ordered[k] = data[k]
            remaining.discard(k)

    # Append unmatched keys alphabetically
    for k in sorted(remaining):
        ordered[k] = data[k]

    return ordered


def extract_kv_cache_dtype_from_command(command: str | None) -> str | None:
    """Extract a ``--kv-cache-dtype`` value from a free-form command template.

    Recipes sometimes set the KV cache dtype only inside ``command:`` rather
    than in ``defaults.kv_cache_dtype`` or ``metadata.kv_dtype``.  Without
    parsing it, the VRAM estimator silently falls back to ``bfloat16`` and
    (for MLA models using ``fp8_ds_mla`` / ``nvfp4_ds_mla``) sizes the KV cache
    with the wrong formula — a ~10x over-estimate (issue #248).

    Returns the first match's value (``"auto"`` treated as absent), or ``None``.
    """
    if not command:
        return None
    m = _KV_CACHE_DTYPE_FLAG_RE.search(command)
    if not m:
        return None
    value = m.group(1)
    if value.lower() in ("auto", ""):
        return None
    return value


def extract_served_model_name_from_command(command: str | None) -> str | None:
    """Extract the served-model name from a free-form command template.

    The supported spelling is ``defaults.served_model_name`` — every runtime
    reconciles that into the rendered command via
    ``RuntimePlugin._augment_served_model_name``.  A recipe that instead writes
    ``--served-model-name <name>`` straight into ``command:`` bypasses that
    machinery, and the name becomes invisible to the config chain — so the
    benchmark asks the endpoint for the *model id*, which the server does not
    answer to, and every task fails with ``404 ... does not exist`` (issue #257).
    The proxy and the container labels have the same blind spot.

    Two guards on the captured value:

    * A ``{placeholder}`` is rejected — an unrendered template means the value
      really does live in the config chain, which resolves it properly; the
      literal ``{served_model_name}`` would be worse than no answer.
    * vLLM accepts several names after the flag (``--served-model-name a b c``)
      and reports the first as the canonical id, so only the first is taken.

    Returns the name, or ``None`` when the flag is absent or unusable.

    This is deliberately a *last resort*, never a replacement for the config
    chain: callers consult their own resolved value first.

    Non-string input yields ``None`` rather than raising: callers reach this
    via ``getattr(recipe, "command", None)`` on objects that only duck-type as
    recipes, and this sits on the launch path's best-effort metadata write,
    where a ``TypeError`` would fail a launch that was otherwise fine.
    """
    if not command or not isinstance(command, str):
        return None
    m = _SERVED_MODEL_NAME_FLAG_RE.search(command)
    if not m:
        return None
    value = m.group(1).strip("\"'")
    if not value or "{" in value or "}" in value:
        return None
    return value


def resolve_served_model_name(recipe: "Recipe", declared: Any = None) -> str:
    """The name a workload is actually served under: *declared* → command → model.

    The single resolution order shared by every consumer that needs the served
    name for *display or routing* (benchmark target, proxy discovery, container
    labels).  ``declared`` is the caller's own already-resolved value — a config
    chain lookup, a CLI override, a ``defaults`` read — and always wins.

    Note the deliberate non-consumer: :func:`~sparkrun.orchestration.job_metadata.generate_intent_id`
    still hashes only the *declared* name.  Widening it would change the intent
    id of every recipe that hardcodes the flag, orphaning workloads already
    running under the old id from ``stop`` / ``logs`` / ``--ensure``.
    """
    if declared is not None and str(declared):
        return str(declared)
    return extract_served_model_name_from_command(recipe.command) or recipe.model


def _resolve_runtime_from_command_hint(recipe: Recipe) -> None:
    """Infer runtime from command prefix when no explicit runtime is set.

    Only fires when runtime is the default ``""`` (empty) and the
    recipe has a ``command`` field.  Recognises:

    - ``vllm serve ...`` → ``"vllm"`` (vllm flavor left for downstream resolvers)
    - ``sglang serve ...`` or ``python -m sglang.launch_server ...`` → ``"sglang"``
    - ``llama-server ...`` → ``"llama-cpp"``
    """
    if recipe.runtime:  # if runtime defined, then we do nothing
        return
    cmd = (recipe.command or "").strip()
    if not cmd:
        return
    # vllm serve → keep as "vllm" for _resolve_vllm_variant to pick the variant
    if _CMD_VLLM_RE.match(cmd):
        recipe.runtime = "vllm"
    elif _CMD_SGLANG_RE.match(cmd):
        recipe.runtime = "sglang"
    elif _CMD_LLAMA_CPP_RE.match(cmd):
        recipe.runtime = "llama-cpp"
    elif _CMD_TRTLLM_RE.match(cmd):
        recipe.runtime = "trtllm"


def _collapse_brace_escapes(value: str) -> str:
    """Collapse vpd-style brace escapes (``{{`` -> ``{``, ``}}`` -> ``}``).

    Recipes double their braces so a literal ``{`` survives vpd
    ``{placeholder}`` substitution — e.g. a JSON-valued flag written as
    ``--diffusion-config '{{"canvas_length": 256}}'``.  Once substitution has
    run, the doubled braces are collapsed back to single braces, matching
    eugr's own ``run-recipe.sh``.

    A value not written in that convention is returned untouched.  Collapsing
    unconditionally would rewrite the ``}}`` that merely closes nested plain
    JSON (``{"a":{"b":1}}``), dropping a brace from a value that was already
    correct.

    Used for *values* (recipe defaults), which are never placeholder templates
    themselves.  Command templates go through :func:`_mask_brace_escapes` /
    :func:`_unmask_brace_escapes` instead, which survive substitution.
    """
    if not uses_brace_escapes(value):
        return value
    return value.replace("{{", "{").replace("}}", "}")


def _mask_brace_escapes(value: str) -> str:
    """Hide ``{{``/``}}`` escapes (and other literal braces) from substitution.

    Thin wrapper over :func:`sparkrun.utils.text.mask_non_placeholder_braces`,
    which is shared with lifecycle-hook rendering so both paths treat braces
    identically.  See that function for the scan rules.
    """
    return mask_non_placeholder_braces(value, escapes=True)


def _unmask_brace_escapes(value: str) -> str:
    """Restore :func:`_mask_brace_escapes` sentinels as literal braces."""
    return unmask_braces(value)


def _resolve_v1_migration(recipe: Recipe) -> None:
    """v1 format recipes -> eugr builder (runtime left for vllm variant resolution)."""
    if recipe.recipe_version != "1":
        return
    if recipe.runtime in ("vllm", ""):
        if not recipe.builder:
            recipe.builder = "eugr"


def _resolve_brace_escapes(recipe: Recipe) -> None:
    """Collapse ``{{``/``}}`` escapes in defaults values, for every recipe version.

    A recipe may escape literal braces in a *value* as well as in the command
    template (e.g. a JSON-valued flag supplied as a default).  Collapse them
    for string values only; non-string defaults (numeric port, max_num_seqs,
    gpu_memory_utilization, ...) are passed through untouched — the regression
    behind the ``'int' object has no attribute 'replace'`` crash in issue #213.

    Gated on the *value* rather than on ``recipe_version``, matching
    :meth:`Recipe.render_command`.  This previously lived inside
    ``_resolve_v1_migration``, so it reached only v1 recipes whose runtime was
    vllm — a v2 (or v1 sglang) recipe leaked ``{{...}}`` into the rendered
    command, one layer below the same bug the command template had.
    """
    escaped = sorted(k for k, v in recipe.defaults.items() if isinstance(v, str) and uses_brace_escapes(v))
    if escaped and recipe.recipe_version != "1":
        logger.warning(
            "Recipe '%s' declares recipe_version '%s' but these defaults use the v1 doubled-brace escape "
            "('{{' / '}}'): %s. Write literal braces plainly instead. The escape is honored for now but will not "
            "be supported by v3 recipes.",
            recipe.name,
            recipe.recipe_version,
            ", ".join(escaped),
        )
    recipe.defaults = {k: (_collapse_brace_escapes(v) if isinstance(v, str) else v) for k, v in recipe.defaults.items()}


def _resolve_eugr_signals(recipe: Recipe) -> None:
    """build_args or mods present -> eugr builder (runtime left for vllm variant resolution)."""
    if recipe.runtime not in ("vllm", ""):
        return
    rc = recipe.runtime_config
    if rc.get("build_args") or recipe.container.strip().startswith("ghcr.io/spark-arena/dgx-vllm-eugr-nightly"):
        if not recipe.builder:
            recipe.builder = "eugr"


def _resolve_vllm_variant(recipe: Recipe) -> None:
    """Bare 'vllm' (or empty) -> 'vllm-distributed' (default) or 'vllm-ray' (Ray hints)."""
    if recipe.runtime not in ("vllm", ""):
        return
    # An explicit CLI override wins over everything, including a literal
    # `--distributed-executor-backend ray` baked into the command template —
    # so `-o distributed_executor_backend=mp` can flip a legacy recipe off Ray.
    # noinspection PyProtectedMember
    override = recipe._applied_overrides.get("distributed_executor_backend")
    if override is not None:
        recipe.runtime = "vllm-ray" if str(override).lower() == "ray" else "vllm-distributed"
        return
    if str(recipe.defaults.get("distributed_executor_backend", "")).lower() == "ray":
        recipe.runtime = "vllm-ray"
        return
    if recipe.command and _RAY_BACKEND_RE.search(recipe.command):
        recipe.runtime = "vllm-ray"
        return
    recipe.runtime = "vllm-distributed"


_RECIPE_RESOLVERS = [
    _resolve_runtime_from_command_hint,
    _resolve_v1_migration,
    _resolve_brace_escapes,
    _resolve_eugr_signals,
    _resolve_vllm_variant,
]


def resolve_runtime(data: dict[str, Any], overrides: dict[str, Any] | None = None) -> str:
    """Lightweight runtime resolution from raw data (for listing/display).

    Mirrors the runtime-affecting resolvers in :data:`_RECIPE_RESOLVERS`
    without constructing a full Recipe.

    Args:
        data: Raw recipe dict.
        overrides: Optional CLI overrides (checked before defaults for
            the vllm-variant decision).
    """
    runtime = data.get("runtime") or ""

    # Command-hint resolver (mirrors _resolve_runtime_from_command_hint)
    # Only fires when runtime is not explicitly set
    cmd = (data.get("command") or "").strip()
    if not runtime and cmd:
        if _CMD_SGLANG_RE.match(cmd):
            return "sglang"
        if _CMD_LLAMA_CPP_RE.match(cmd):
            return "llama-cpp"
        if _CMD_TRTLLM_RE.match(cmd):
            return "trtllm"
        # vllm serve or unrecognised → fall through to vllm variant resolution

    # v1 migration and eugr detection now only affect builder, not runtime.
    # Runtime falls through to vllm variant resolution below.

    runtime_config = data.get("runtime_config") or {}
    if runtime_config is not None and not isinstance(runtime_config, dict):
        raise RecipeError("Recipe 'runtime_config' field must be a mapping, got %s" % type(runtime_config).__name__)
    if runtime in ("vllm", ""):
        effective = dict(overrides or {})
        defaults = data.get("defaults")
        if defaults is not None and not isinstance(defaults, dict):
            raise RecipeError("Recipe 'defaults' field must be a mapping, got %s" % type(defaults).__name__)
        defaults = defaults or {}
        # An explicit override wins over the literal-command hint (mirrors
        # _resolve_vllm_variant): -o distributed_executor_backend=mp flips a
        # legacy ray-in-command recipe to vllm-distributed.
        override = effective.get("distributed_executor_backend")
        if override is not None:
            return "vllm-ray" if str(override).lower() == "ray" else "vllm-distributed"
        if str(defaults.get("distributed_executor_backend", "")).lower() == "ray":
            return "vllm-ray"
        if _RAY_BACKEND_RE.search(cmd):
            return "vllm-ray"
        return "vllm-distributed"
    return runtime


def resolve_builder(data: dict[str, Any]) -> str:
    """Lightweight builder resolution from raw data (for listing/display).

    Detects eugr signals (v1 version, build_args, mods) and returns
    ``"eugr"`` or ``""`` without constructing a full Recipe.
    """
    builder = data.get("builder", "")
    if builder:
        return builder
    version = str(data.get("sparkrun_version", data.get("recipe_version", "2")))
    if version == "1":
        runtime = data.get("runtime", "")
        if runtime in ("vllm", ""):
            return "eugr"
    runtime_config = data.get("runtime_config") or {}
    runtime = data.get("runtime", "")
    if runtime in ("vllm", "") and (
        data.get("build_args")
        or data.get("mods")
        or (isinstance(runtime_config, dict) and (runtime_config.get("build_args") or runtime_config.get("mods")))
    ):
        return "eugr"
    return ""


def is_recipe_file(path: Path) -> bool:
    """Check if a YAML file is a valid sparkrun recipe.

    Requires: parseable YAML dict, resolvable runtime, model, and container fields.
    """
    try:
        data = read_yaml(str(path))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if not data.get("model") or not data.get("container"):
        return False
    try:
        rt = resolve_runtime(data)
    except Exception:
        return False
    return rt != "unknown"


def discover_cwd_recipes(directory: Path | None = None) -> list[Path]:
    """Scan a directory (default CWD) for flat .yaml/.yml files that are valid recipes."""
    if directory is None:
        directory = Path.cwd()
    if not directory.is_dir():
        return []
    candidates: list[Path] = []
    for pattern in ("*.yaml", "*.yml"):
        candidates.extend(directory.glob(pattern))
    return sorted(p for p in candidates if is_recipe_file(p))


SPARK_ARENA_PREFIX = "@spark-arena/"
SPARK_ARENA_API_URL = "https://spark-arena.com/api/recipes/%s/raw"


def expand_recipe_shortcut(name: str) -> str:
    """Expand known recipe shortcuts to full URLs.

    Currently supports:
        @spark-arena/UUID  ->  https://spark-arena.com/api/recipes/UUID/raw
    """
    if name.startswith(SPARK_ARENA_PREFIX):
        recipe_id = name[len(SPARK_ARENA_PREFIX) :]
        return SPARK_ARENA_API_URL % recipe_id
    return name


def simplify_recipe_ref(url: str) -> str:
    """Simplify a recipe URL to a shortcut if possible (inverse of expand).

    Currently supports:
        https://spark-arena.com/api/recipes/UUID/raw  ->  @spark-arena/UUID

    Returns the original string unchanged if no simplification applies.
    """
    m = re.match(r"https?://spark-arena\.com/api/recipes/([^/]+)/raw$", url)
    if m:
        return "%s%s" % (SPARK_ARENA_PREFIX, m.group(1))
    return url


def is_recipe_url(name: str) -> bool:
    """Check if recipe_name looks like an HTTP(S) URL."""
    return name.startswith(("http://", "https://"))


# Hosts allowed to serve recipes without an explicit confirmation. Any
# other https host requires the caller to pass ``allow_untrusted_host=True``
# (the CLI prompts / honours --trust). http:// is never allowed (MITM).
RECIPE_URL_ALLOWED_HOSTS: tuple[str, ...] = ("spark-arena.com",)

# Cap fetched recipe size to avoid a hostile/buggy endpoint streaming
# unbounded data onto the control machine. Recipes are small YAML files.
_RECIPE_FETCH_MAX_BYTES = 5 * 1024 * 1024
_RECIPE_FETCH_MAX_REDIRECTS = 3


def _recipe_url_host(url: str) -> str:
    from urllib.parse import urlparse

    return (urlparse(url).hostname or "").lower()


def _validate_recipe_url(url: str, *, allow_untrusted_host: bool) -> None:
    """Raise RecipeError if *url* is not a safe recipe source.

    Enforces https-only (no MITM) and an allowlist of known hosts unless
    the caller explicitly opts in to an untrusted host.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise RecipeError(
            "Refusing to fetch recipe over %r: only https:// recipe URLs are allowed "
            "(plaintext http is vulnerable to tampering)." % (parsed.scheme or url)
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise RecipeError("Refusing to fetch recipe: URL has no host: %s" % url)
    if host not in RECIPE_URL_ALLOWED_HOSTS and not allow_untrusted_host:
        raise RecipeUntrustedHostError(url, host)


def _url_cache_path(url: str) -> Path:
    """Return the local cache path for a remote recipe URL."""
    import hashlib

    from sparkrun.core.config import DEFAULT_CACHE_DIR

    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return DEFAULT_CACHE_DIR / "remote-recipes" / ("%s.yaml" % url_hash)


def fetch_and_cache_recipe(url: str, *, allow_untrusted_host: bool = False) -> Path:
    """Fetch a recipe from URL and cache it locally.

    Only ``https://`` URLs are accepted, and only from
    :data:`RECIPE_URL_ALLOWED_HOSTS` unless *allow_untrusted_host* is set
    (the CLI prompts / honours ``--trust`` before passing it). Redirects
    are re-validated against the same rules and the chain is capped; the
    response body is size-capped.

    On success, writes/updates the cache file and returns its path.
    On network failure, falls back to cached copy if available.
    Raises RecipeError if fetch fails and no cache exists.
    """
    from urllib.error import HTTPError, URLError
    from urllib.request import HTTPRedirectHandler, Request, build_opener

    _validate_recipe_url(url, allow_untrusted_host=allow_untrusted_host)

    cache_path = _url_cache_path(url)

    class _ValidatingRedirectHandler(HTTPRedirectHandler):
        max_redirections = _RECIPE_FETCH_MAX_REDIRECTS

        def redirect_request(self, req, fp, code, msg, headers, newurl):
            # Re-apply the same scheme/host policy to every redirect hop so
            # an allowed host cannot bounce us to http:// or an arbitrary host.
            _validate_recipe_url(newurl, allow_untrusted_host=allow_untrusted_host)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    try:
        opener = build_opener(_ValidatingRedirectHandler())
        req = Request(url, headers={"User-Agent": "sparkrun"})
        with opener.open(req, timeout=30) as resp:
            # Read one byte past the cap so we can detect oversize bodies.
            content = resp.read(_RECIPE_FETCH_MAX_BYTES + 1)
        if len(content) > _RECIPE_FETCH_MAX_BYTES:
            raise RecipeError("Refusing to fetch recipe from %s: response exceeds %d bytes." % (url, _RECIPE_FETCH_MAX_BYTES))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)
        return cache_path
    except (HTTPError, URLError, OSError) as e:
        if cache_path.exists():
            reason = e.code if isinstance(e, HTTPError) else e.reason
            logger.warning(
                "Failed to fetch recipe (using cached copy): %s",
                reason,
            )
            return cache_path
        if isinstance(e, HTTPError):
            raise RecipeError("Failed to fetch recipe from %s: HTTP %d" % (url, e.code))
        raise RecipeError("Failed to fetch recipe from %s: %s" % (url, e.reason if isinstance(e, URLError) else e))


# # Backward-compat aliases (old underscore names)
# _expand_recipe_shortcut = expand_recipe_shortcut
# _simplify_recipe_ref = simplify_recipe_ref
# _is_recipe_url = is_recipe_url
# _fetch_and_cache_recipe = fetch_and_cache_recipe


@dataclass
class LaunchOverrides:
    """Internal, undocumented launch/infra overrides from a recipe's
    ``cluster_config:`` block.

    Despite the YAML key (kept for back-compat), these are **per-recipe launch
    overrides**, not a cluster definition — distinct from
    :class:`sparkrun.core.cluster_manager.ClusterDistributionConfig` (a named
    cluster's distribution settings) and ``ResolvedClusterConfig`` (resolved
    transfer config).  They are intentionally *not* part of the documented
    recipe schema — an escape hatch for temporary / environment-specific work
    that doesn't belong in the cluster config (e.g. pointing at pre-placed
    model weights on a shared NFS path).  Applied at the single
    ``launch_inference`` choke point so both the CLI and ``api.run`` honour
    them, and only for *trusted* recipes (they can expose host paths to the
    container — see ``launcher._enforce_recipe_mount_trust``).

    Fields:
        remote_cache_dir: Overrides the remote HuggingFace cache dir on target
            hosts (where models are mounted into the container, and where
            distribution would land). Takes precedence over the cluster config.
        local_cache_dir: Overrides the control-machine download cache dir.
        resolved_model_path: Absolute path to a directory of model weights that
            is already present on every node (e.g. a shared NFS mount). When
            set, sparkrun skips model download + distribution entirely,
            identity-mounts the path into the container, and points the serving
            runtime's model argument at it.
    """

    remote_cache_dir: str | None = None
    local_cache_dir: str | None = None
    resolved_model_path: str | None = None

    def is_empty(self) -> bool:
        return not (self.remote_cache_dir or self.local_cache_dir or self.resolved_model_path)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.remote_cache_dir:
            d["remote_cache_dir"] = self.remote_cache_dir
        if self.local_cache_dir:
            d["local_cache_dir"] = self.local_cache_dir
        if self.resolved_model_path:
            d["resolved_model_path"] = self.resolved_model_path
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LaunchOverrides | None":
        if not isinstance(data, dict):
            return None
        cc = cls(
            remote_cache_dir=(str(data["remote_cache_dir"]) if data.get("remote_cache_dir") else None),
            local_cache_dir=(str(data["local_cache_dir"]) if data.get("local_cache_dir") else None),
            resolved_model_path=(str(data["resolved_model_path"]) if data.get("resolved_model_path") else None),
        )
        return None if cc.is_empty() else cc


# Back-compat alias: this type was previously named ``ClusterConfig``, which
# collided with the cluster-side config types.  Keep the old name importable.
ClusterConfig = LaunchOverrides


def is_local_model_path(model: str | None) -> bool:
    """True when a recipe's ``model:`` value is an absolute host path.

    An absolute path in ``model:`` is user-facing sugar for
    :attr:`LaunchOverrides.resolved_model_path`: the weights are already present
    on every node (e.g. a shared mount), so sparkrun skips model download +
    distribution, identity-mounts the directory into the container, and serves
    the runtime directly from it.  See :func:`sparkrun.core.launcher.launch_inference`.

    Only *absolute* paths are treated this way — no ``file://`` scheme, no ``~``
    expansion, no relative paths — so an ordinary HuggingFace repo id
    (``org/name``) or GGUF spec (``org/name-GGUF:Q4_K_M``) is never mistaken for
    a local path.  Because this identity-mounts a host directory, it is gated to
    trusted recipes at the launch choke point (see
    ``launcher._enforce_recipe_mount_trust``).
    """
    return isinstance(model, str) and os.path.isabs(model)


class RecipeError(Exception):
    """Raised when a recipe is invalid or cannot be loaded."""


class RecipeUntrustedHostError(RecipeError):
    """Raised when a recipe URL points at a host not on the allowlist.

    The caller (CLI) may catch this, confirm with the user, and retry
    ``fetch_and_cache_recipe(url, allow_untrusted_host=True)``.
    """

    def __init__(self, url: str, host: str):
        self.url = url
        self.host = host
        super().__init__(
            "Recipe URL host %r is not in the trusted allowlist (%s). "
            "Re-run with --trust to fetch from this host." % (host, ", ".join(RECIPE_URL_ALLOWED_HOSTS))
        )


class RecipeAmbiguousError(RecipeError):
    """Raised when a recipe name matches more than one registry recipe.

    Ambiguity is not only *across* registries: a registry's recipe dir is
    scanned recursively, so ``a/foo.yaml`` and ``b/foo.yaml`` in the same
    registry are two different recipes matching the stem ``foo``.

    ``labels`` holds one user-typeable ``@registry/...`` name per entry in
    ``matches`` (see :meth:`RegistryManager.qualified_recipe_name`), so
    callers can present options that are actually distinguishable.  When
    omitted it degrades to ``@registry/<stem>``.
    """

    def __init__(self, name: str, matches: list[tuple[str, Path]], labels: list[str] | None = None):
        self.name = name
        self.matches = matches
        self.labels = labels if labels is not None else ["@%s/%s" % (reg, Path(p).stem) for reg, p in matches]
        from sparkrun.core.registry import format_ambiguity

        super().__init__(format_ambiguity("Recipe", name, matches, self.labels))


class Recipe:
    """A loaded and validated sparkrun recipe."""

    def __init__(self, data: dict[str, Any], source_path: str | None = None):
        self._raw = data
        self.source_path = source_path
        self.source_registry: str | None = None  # set by _load_recipe after resolution
        self.source_registry_url: str | None = None  # set by _load_recipe after resolution
        # True when the recipe was fetched from a remote URL (e.g. a
        # spark-arena link). URL-sourced recipes are never auto-trusted —
        # their hooks require --trust / interactive confirmation. Set by
        # _load_recipe; see core.launcher.resolve_recipe_trust.
        self.is_url_sourced: bool = False

        self._qualified_name_override: str | None = None  # optional override for qualified_name

        # Detect version
        self.recipe_version = str(data.get("recipe_version", "2"))

        # Core fields — name defaults to source filename stem if not provided
        default_name = Path(source_path).stem if source_path else "unnamed"
        self.name: str = default_name  # data.get("name", default_name)
        self.description: str = data.get("description", "")
        self.model: str = data.get("model", "")
        self.model_revision: str | None = data.get("model_revision")
        self.runtime: str = data.get("runtime", "")  # init to empty string if not provided
        self.runtime_version: str = data.get("runtime_version", "")

        # Topology
        self.mode: str = data.get("mode", "auto")  # "solo", "cluster", "auto"
        self.min_nodes: int = int(data.get("min_nodes", 1))
        self.max_nodes: int | None = data.get("max_nodes")
        if self.mode == "solo":
            self.max_nodes = self.min_nodes = 1
        elif self.mode == "auto" and self.min_nodes > 1:
            self.mode = "cluster"
        elif self.mode == "auto" and self.max_nodes == 1:
            self.mode = "solo"

        # Topology - Handle solo_only/cluster_only as first-class fields (works for both v1 and v2)
        if data.get("cluster_only"):
            self.min_nodes = max(self.min_nodes, 2)
            self.mode = "cluster"
        if data.get("solo_only"):
            self.max_nodes = 1
            self.mode = "solo"

        # Container
        self.container: str = data.get("container", "")

        # Optional per-machine images.  ``container`` above stays the fallback
        # for any host without an entry; see :mod:`sparkrun.core.images` for why
        # these bind to hostnames rather than ranks.  Parsed permissively — the
        # cluster's host list isn't known here, so validation happens in
        # ``resolve_image_plan`` at launch time.
        self.containers: list[dict[str, str]] = parse_container_entries(data.get("containers"))

        # Configuration
        self.defaults: dict[str, Any] = dict(data.get("defaults") or {})
        # Use recipe-provided env values literally.  Do NOT expand control-machine
        # variables (e.g. ``$AWS_SECRET_ACCESS_KEY``): a third-party recipe could
        # otherwise exfiltrate host secrets by injecting them into the container.
        self.env: dict[str, str] = {str(k): str(v) for k, v in (data.get("env") or {}).items()}
        self.command: str | None = data.get("command")

        # Metadata section (v2 extension for VRAM estimation, model info)
        raw_metadata = data.get("metadata", {})
        self.metadata: dict[str, Any] = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}

        # Plugin-owned top-level recipe items.  The core owns only lifecycle;
        # each registration owns parsing, validation, and canonical export.
        # Parse after core identity fields so a handler may bind to model,
        # runtime, or revision without reaching into raw YAML itself.
        self.plugin_items: dict[str, Any] = {}
        self._plugin_item_raw: dict[str, Any] = {}
        for registration in registered_recipe_items():
            if registration.key not in data:
                continue
            raw_item = data[registration.key]
            self._plugin_item_raw[registration.key] = raw_item
            try:
                self.plugin_items[registration.key] = registration.handler.parse(raw_item, self)
            except Exception as error:
                raise RecipeError(
                    "Plugin %s could not parse top-level recipe item '%s': %s" % (registration.owner, registration.key, error)
                ) from error

        # Metadata values supplement missing top-level fields
        # if not self.name or self.name == default_name:
        #     meta_name = self.metadata.get("name")
        #     if meta_name:
        #         self.name = str(meta_name)
        if not self.description:
            meta_desc = self.metadata.get("description")
            if meta_desc:
                self.description = str(meta_desc)

        # Maintainer (metadata-only field)
        self.maintainer: str = str(self.metadata.get("maintainer", ""))

        # Runtime-specific config: explicit runtime_config key takes priority,
        # then unknown top-level keys are auto-swept in.
        self.runtime_config: dict[str, Any] = dict(data.get("runtime_config", {}))
        plugin_keys = {registration.key for registration in registered_recipe_items()}
        for k, v in data.items():
            if k not in _KNOWN_KEYS and k not in plugin_keys and k not in self.runtime_config:
                self.runtime_config[k] = v

        # Gateway capability declarations.  Optional, and *not* serve
        # configuration: they describe what the served model can do, which an
        # inference gateway needs in order to admit or reject a request before
        # it reaches the backend.  Declared as real attributes (rather than
        # swept into ``runtime_config``) precisely so they stay out of
        # ``derive_recipe_fingerprint`` — editing a capability list must not
        # change the workload's identity and force a running deployment to be
        # re-admitted.
        #
        # Nothing in a recipe reveals these, so sparkrun never infers them; an
        # undeclared capability is left to the gateway's own policy.
        self.capabilities: list[str] = [str(c) for c in (data.get("capabilities") or [])]
        self.unsupported_capabilities: list[str] = [str(c) for c in (data.get("unsupported_capabilities") or [])]

        # Lifecycle hooks
        self.pre_exec: list[str | dict[str, str]] = list(data.get("pre_exec", []))
        self.post_exec: list[str] = list(data.get("post_exec", []))
        self.post_commands: list[str] = list(data.get("post_commands", []))
        self.stop_after_post: bool = bool(data.get("stop_after_post", False))

        # Mods (generic, builder-agnostic): list of references resolved to
        # pre_exec entries by core/mods.py before container launch.
        # v1 recipes carry mods under runtime_config; migrate to top-level
        # so exports round-trip cleanly and the resolver sees a single source.
        self.mods: list[str] = list(data.get("mods", []) or [])
        if not self.mods and isinstance(self.runtime_config.get("mods"), list):
            self.mods = list(self.runtime_config.pop("mods"))

        # Builder plugin
        self.builder: str = data.get("builder", "")
        self.builder_config: dict[str, Any] = dict(data.get("builder_config", {}))

        # Executor config (container engine settings: auto_remove, restart_policy, etc.)
        raw_exec = data.get("executor_config", {})
        self.executor_config: dict[str, Any] = dict(raw_exec) if isinstance(raw_exec, dict) else {}

        # Experimental executor selector.  ``""`` (default) → DockerExecutor.
        # ``"local"`` → native-subprocess LocalExecutor (no container).
        # No CLI surface; recipe-only.
        self.executor: str = str(data.get("executor", "") or "")

        # Optional scheduler selector.  ``""`` (default) → GreedyScheduler.
        # ``"occupancy-sparse"`` / ``"occupancy-dense"`` opt in to
        # occupancy-sparse / occupancy-dense placement + fractional GPU sharing.
        # Overridden by ``--scheduler`` on the CLI / ``RunOptions.scheduler``.
        self.scheduler: str = str(data.get("scheduler", "") or "")

        # Distribution config (auto-generated default, mutable by runtimes)
        self.distribution_config: DistributionConfig = _parse_distribution_config(data)

        # Explicit placement layout for heterogeneous clusters (optional).
        # Parsed permissively here; the placement engine in
        # :mod:`sparkrun.schedulers.greedy` validates at apply time.
        raw_layout = data.get("layout")
        self.layout: RecipeLayout | None = RecipeLayout.from_dict(raw_layout) if isinstance(raw_layout, dict) else None

        # Internal, undocumented launch overrides (see :class:`LaunchOverrides`).
        self.cluster_config: LaunchOverrides | None = LaunchOverrides.from_dict(data.get("cluster_config"))

        # Compilation/autotune cache knobs — highest layer of the chain in
        # :func:`sparkrun.core.runtime_cache.resolve_runtime_cache_settings`.
        raw_rt_cache = data.get("runtime_cache")
        if isinstance(raw_rt_cache, bool):
            raw_rt_cache = {"enabled": raw_rt_cache}
        self.runtime_cache: dict[str, Any] = dict(raw_rt_cache) if isinstance(raw_rt_cache, dict) else {}

        # Applied overrides (populated by resolve())
        self._applied_overrides: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Runtime resolution (separated from __init__ for override support)
    # ------------------------------------------------------------------

    def _effective_default(self, key: str, fallback: Any = None) -> Any:
        """Get effective value: applied overrides -> recipe defaults -> fallback.

        Used by resolvers so they naturally see CLI overrides without
        per-resolver maintenance.
        """
        v = self._applied_overrides.get(key)
        if v is not None:
            return v
        return self.defaults.get(key, fallback)

    def resolve(self, overrides: dict[str, Any] | None = None) -> Recipe:
        """Run the resolver chain, optionally with CLI overrides.

        Overrides are visible to resolvers via ``_effective_default()`` so
        they can influence runtime resolution (e.g.
        ``distributed_executor_backend=ray`` switches vllm-distributed to
        vllm-ray).

        Can be called multiple times safely — resets runtime to its raw
        YAML value before re-running the chain.
        """
        self._applied_overrides = dict(overrides) if overrides else {}
        self.runtime = self._raw.get("runtime", "")
        self.builder = self._raw.get("builder", "")
        for resolver in _RECIPE_RESOLVERS:
            resolver(self)
        return self

    @property
    def qualified_name(self) -> str:
        """Fully qualified name for unambiguous CLI display.

        Returns @registry/name for registry recipes, source_path for
        path/URL recipes, or bare name for bundled/CWD recipes.
        """
        if self._qualified_name_override:
            return self._qualified_name_override
        if self.source_registry:
            return "@%s/%s" % (self.source_registry, self.name)
        if self.source_path:
            p = self.source_path
            if p.startswith(("http://", "https://")):
                return p
            sp = Path(p)
            if sp.is_absolute() or "/" in p:
                return p
        return self.name

    @property
    def spark_arena_benchmarks(self) -> list[dict[str, Any]]:
        """List of ``{tp, uuid}`` dicts linking to Spark Arena benchmark results."""
        return self.metadata.get("spark_arena_benchmarks", [])

    @property
    def slug(self) -> str:
        """URL/filesystem-safe slug derived from name."""
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")

    def get_default(self, key: str, fallback: Any = None) -> Any:
        """Get a value from defaults with optional fallback."""
        return self.defaults.get(key, fallback)

    @property
    def effective_served_model_name(self) -> str:
        """Resolved served-model name: CLI override → recipe default → ``command:`` → model id.

        Mirrors the runtime serve-argument resolution (``--served-model-name``
        falls back to the model id when unset) so observers can read the name a
        workload is actually served under off the container labels.

        The ``command:`` step is the last resort described in
        :func:`resolve_served_model_name`: without it, a recipe that hardcodes
        the flag in its command template labels its containers with the model id
        while the server answers only to the hardcoded name.
        """
        return resolve_served_model_name(self, self._effective_default("served_model_name"))

    def build_config_chain(self, cli_overrides: dict[str, Any] | None = None, user_config: dict[str, Any] | None = None) -> Variables:
        """Build cascading config: CLI overrides -> user config -> recipe defaults.

        Also injects ``model`` and ``resolved_model_path`` into the chain for
        ``{...}`` template substitution.  ``resolved_model_path`` resolves to the
        recipe's ``cluster_config.resolved_model_path`` when set (pre-placed
        on-disk weights), otherwise falls back to ``model`` so the same template
        — e.g. ``--chat-template {resolved_model_path}/chat_template.jinja`` —
        works for both the override and normal cases.
        """
        base = dict(self.defaults)
        base.setdefault("model", self.model)
        _rmp = getattr(getattr(self, "cluster_config", None), "resolved_model_path", None)
        base.setdefault("resolved_model_path", _rmp or self.model)
        return Variables(sources=(cli_overrides or {}, user_config or {}, base), env_placement=EnvPlacement.IGNORED)

    def render_command(self, config_chain: Variables) -> str | None:
        """Render the command template with values from the config chain.

        Returns None if no command template is defined.
        """
        if not self.command:
            return None

        rendered = self.command.strip()

        # Literal braces are masked to sentinels *first* so they are invisible
        # to vpd's placeholder regex — otherwise a JSON-valued flag swallows any
        # placeholder nested inside it — then restored once substitution is
        # done, so the runtime receives valid JSON.  Substitution iterates to
        # resolve nested references.
        #
        # Escape mode is read off the *template*, not the recipe version.  It
        # used to be `recipe_version == "1"`, so a v1 command pasted into a v2
        # recipe emitted literal '{{...}}' to the runtime — a launch that dies
        # on the serve command after the model has downloaded, with an error
        # naming neither sparkrun nor the recipe.  Forcing escapes on for every
        # version is *not* the fix: '}}' closes nested plain JSON
        # ('{"a":{"b":1}}') as often as it escapes one brace, so it would
        # silently eat a closing brace from the idiomatic v2 spelling.  '{{' is
        # the disambiguator (see utils.text.uses_brace_escapes).
        escapes = uses_brace_escapes(rendered)
        if escapes and self.recipe_version != "1":
            logger.warning(
                "Recipe '%s' declares recipe_version '%s' but its command template uses the doubled-brace escape "
                "('{{' / '}}'), which is the v1 convention — usually a command pasted from a v1 recipe. Write literal "
                "braces plainly instead ('--flag '%s''); a placeholder nested inside JSON still resolves. The escape "
                "is honored for now but will not be supported by v3 recipes.",
                self.name,
                self.recipe_version,
                '{"key": "value"}',
            )

        # Hook templates deliberately stay unescaped (see
        # sparkrun.orchestration.hooks.render_hook_command).
        rendered = render_template(rendered, config_chain, escapes=escapes)

        # Fix trailing spaces after backslash line-continuations.
        # ``\<space><newline>`` → ``\<newline>``
        rendered = _TRAILING_SPACE_CONTINUATION_RE.sub("\\\n", rendered)

        return rendered

    def validate(self) -> list[str]:
        """Validate the recipe and return a list of warnings/errors.

        The flat string form kept for back-compat.  Callers that need to tell
        a launch-blocking problem from a cosmetic one use
        :func:`sparkrun.core.validation.validate_recipe`, which calls the two
        halves below directly.
        """
        return self.validate_structure() + self.validate_metadata()

    def validate_structure(self) -> list[str]:
        """Problems that make the recipe unlaunchable.

        A missing model or an out-of-range node range is not something any
        later stage can work around, which is why these are the half
        :func:`sparkrun.core.validation.validate_recipe` reports as errors.
        """
        issues = []
        if not self.name:
            issues.append("Recipe missing 'name' field")
        if not self.model:
            issues.append("Recipe missing 'model' field")
        if not self.runtime:
            issues.append("Recipe missing 'runtime' field")
        if self.mode not in ("solo", "cluster", "auto"):
            issues.append("Invalid mode '%s'; expected 'solo', 'cluster', or 'auto'" % self.mode)
        if self.min_nodes < 1:
            issues.append("min_nodes must be >= 1, got %d" % self.min_nodes)
        if self.max_nodes is not None and self.max_nodes < self.min_nodes:
            issues.append("max_nodes (%s) < min_nodes (%s)" % (self.max_nodes, self.min_nodes))
        return issues

    def validate_metadata(self) -> list[str]:
        """Problems in ``metadata:`` that degrade a *estimate*, not the launch.

        Every field here feeds VRAM estimation or display.  A bad value costs
        an estimate — the estimator skips the claim and placement falls back to
        capacity-only — so these are reported as warnings, never as a reason to
        refuse to launch.  Real published recipes carry prose in
        ``metadata.quantization`` ("NVFP4 (compressed-tensors, mixed
        precision)"); aborting on that would strand a working recipe over a
        label.
        """
        issues: list[str] = []

        if self.metadata:
            from sparkrun.models.kv import arch_fields, is_valid_kv_dtype
            from sparkrun.models.vram import parse_param_count, bytes_per_element

            mp = self.metadata.get("model_params")
            if mp is not None and parse_param_count(mp) is None:
                issues.append("metadata.model_params %r is not a valid parameter count" % mp)
            md = self.metadata.get("model_dtype")
            if md is not None and bytes_per_element(str(md)) is None:
                issues.append("metadata.model_dtype %r is not a recognized dtype" % md)
            kd = self.metadata.get("kv_dtype")
            # Accepts either a per-element dtype or a packed slot layout a KV
            # strategy claims (nvfp4_ds_mla, ...), which is why this asks the
            # registry rather than enumerating architectures here.
            if kd is not None and not is_valid_kv_dtype(str(kd)):
                issues.append("metadata.kv_dtype %r is not a recognized dtype" % kd)
            # Architecture fields declared by the KV strategies. These are
            # documented as user-overridable and are coerced in estimate_vram, so
            # an unchecked bad value surfaces as a traceback from `recipe show
            # --json` — or, on the launch path, as a debug-level log and a
            # silently dropped memory claim, which skips the fit check.
            for _field in arch_fields():
                _issue = _field.validate(self.metadata.get(_field.name))
                if _issue:
                    issues.append(_issue)
            mt = self.metadata.get("model_type")
            if mt is not None and not isinstance(mt, str):
                issues.append("metadata.model_type %r must be a string" % mt)
            mq = self.metadata.get("quantization")
            if mq is not None:
                _KNOWN_QUANT_METHODS = {
                    "awq",
                    "gptq",
                    "marlin",
                    "fp8",
                    "nvfp4",
                    "mxfp4",
                    "bitsandbytes",
                    "compressed-tensors",
                    "auto-round",
                    "autoround",
                    "auto_round",
                    "gguf",
                    "int4",
                    "int8",
                    "none",
                }
                if str(mq).lower().strip() not in _KNOWN_QUANT_METHODS:
                    issues.append("metadata.quantization %r is not a recognized method" % mq)

        for key, value in self.plugin_items.items():
            registration = get_recipe_item(key)
            if registration is None:
                issues.append("No plugin is registered to validate top-level recipe item '%s'" % key)
                continue
            try:
                plugin_issues = registration.handler.validate(value, self)
            except Exception as error:
                issues.append("Plugin %s failed to validate top-level recipe item '%s': %s" % (registration.owner, key, error))
                continue
            issues.extend("%s.%s" % (key, issue) for issue in plugin_issues)

        return issues

    def plugin_item(self, key: str, default: Any = None) -> Any:
        """Return a parsed plugin-owned top-level item."""

        return self.plugin_items.get(key, default)

    @classmethod
    def load(cls, path: str | Path, resolve: bool = True) -> Recipe:
        """Load a recipe from a YAML file path.

        Args:
            path: Path to the recipe YAML file.
            resolve: Run the resolver chain immediately (default True).
                Pass ``False`` when CLI overrides need to influence
                resolution — call ``recipe.resolve(overrides)`` later.
        """
        path = Path(path)
        if not path.exists():
            raise RecipeError("Recipe file not found: %s" % path)
        data = read_yaml(str(path))
        if not isinstance(data, dict):
            raise RecipeError("Recipe file must contain a YAML mapping: %s" % path)
        recipe = cls(data, source_path=str(path))
        if resolve:
            recipe.resolve()
        return recipe

    @classmethod
    def from_dict(cls, data: dict[str, Any], overrides: dict[str, Any] | None = None) -> Recipe:
        """Create a recipe from a dict (useful for testing).

        Always resolves immediately for backward compatibility.
        """
        recipe = cls(data)
        recipe.resolve(overrides)
        return recipe

    def estimate_vram(
        self,
        cli_overrides: dict[str, Any] | None = None,
        auto_detect: bool = True,
        cache_dir: str | None = None,
        total_gpu_memory_gb: float | None = None,
    ) -> VRAMEstimate:
        """Estimate VRAM usage for this recipe.

        Merges metadata fields with auto-detected HF config (if available).
        CLI overrides for max_model_len, tensor_parallel are respected.

        Args:
            cli_overrides: CLI override values (e.g. tensor_parallel, max_model_len).
            auto_detect: Whether to query HuggingFace Hub for model config.
            cache_dir: Optional HuggingFace cache directory for model lookups.

        Returns:
            VRAMEstimate dataclass with estimation results.
        """
        from sparkrun.models.kv import arch_fields, is_kv_layout
        from sparkrun.models.vram import (
            bytes_per_element,
            estimate_vram as _estimate_vram,
            extract_model_info,
            fetch_model_config,
            fetch_safetensors_params,
            fetch_safetensors_size,
            parse_param_count,
        )
        from sparkrun.models.quantization import (
            QuantizationInfo,
            fetch_hf_quant_config,
            resolve_quantization,
        )

        config = self.build_config_chain(cli_overrides)

        # Start with metadata values
        from sparkrun.models.vram import normalize_dtype

        _raw_dtype = self.metadata.get("model_dtype")
        model_dtype = normalize_dtype(str(_raw_dtype)) if _raw_dtype else None
        model_params_raw = self.metadata.get("model_params")
        _raw_kv = self.metadata.get("kv_dtype")
        # An explicit CLI kv_cache_dtype override always wins over a metadata
        # value, which may be an auto-write from a previous estimate on this
        # object.  Without this, a value frozen into metadata on call one
        # shadows a different override passed on call two — a frozen MLA slot
        # layout vs generic KV dtype is a ~10x switch.
        _cli_kv = (cli_overrides or {}).get("kv_cache_dtype")
        if _cli_kv and str(_cli_kv) not in ("auto", ""):
            kv_dtype = str(_cli_kv)
        else:
            kv_dtype = normalize_dtype(str(_raw_kv)) if _raw_kv else None
        num_layers = self.metadata.get("num_layers")
        num_kv_heads = self.metadata.get("num_kv_heads")
        head_dim = self.metadata.get("head_dim")
        model_vram = self.metadata.get("model_vram")
        kv_vram_per_token = self.metadata.get("kv_vram_per_token")
        model_type = self.metadata.get("model_type")
        # Architecture fields declared by the KV strategies — auto-detected
        # below, overridable in metadata.  Read as a sweep over the declaration
        # rather than a hand-written list, so a new architecture's fields reach
        # the estimator (and the write-back at the bottom) without an edit here.
        arch_extra: dict[str, Any] = {f.name: self.metadata.get(f.name) for f in arch_fields()}
        quant_info: QuantizationInfo | None = None
        _storage_dtype: str | None = None  # raw torch_dtype before quant override
        effective_recipe_quant: str | None = None  # recipe-level quantization override

        # Auto-detect from HF if fields are missing and model is specified.
        needs_detection = (model_vram is None and (not model_dtype or model_params_raw is None)) or (
            kv_vram_per_token is None and (not num_layers or not num_kv_heads or not head_dim)
        )
        # Even with kv_vram_per_token pinned we must still learn which KV
        # architecture the model uses.  The override replaces the KV *sizing*
        # (the user supplies the exact bytes-per-token), but the *sharding rule*
        # only depends on the architecture: an MLA latent is replicated across TP
        # ranks (divided by PP only), an ordinary KV cache shards by TP*PP.
        # Defaulting to the ordinary rule silently TP-divides the override of a
        # DeepSeek model, which under-claims memory and lets the scheduler
        # over-commit a placement.  This cannot be short-circuited by pinned
        # architecture: pinning num_layers/num_kv_heads/head_dim is exactly how
        # a user suppresses detection, and they may not have pinned the
        # architecture markers while delegating KV sizing.  Detection is cheap
        # and write-back stops it recurring, so always run it here.
        #
        # The three signals that identify an architecture without HF are asked
        # for generically — a pinned marker any strategy declared, a packed KV
        # layout any strategy claims, or a model_type (which this method writes
        # back unconditionally, so its presence means detection already ran).
        # Note this must not ask *which* strategy applies: dense is the answer
        # for most models and is never a positive identification, so gating on
        # it would refetch on every call forever.
        needs_detection = needs_detection or (
            kv_vram_per_token is not None and not model_type and not any(arch_extra.values()) and not is_kv_layout(str(kv_dtype or ""))
        )
        if auto_detect and self.model and needs_detection:
            hf_config = fetch_model_config(self.model, revision=self.model_revision, cache_dir=cache_dir)
            hf_quant_config = fetch_hf_quant_config(self.model, revision=self.model_revision, cache_dir=cache_dir)

            # Resolve quantization from all sources (works even without hf_config for GGUF)
            recipe_quant_meta = self.metadata.get("quantization")
            recipe_quant_default = config.get("quantization")
            effective_recipe_quant = recipe_quant_meta or (str(recipe_quant_default) if recipe_quant_default else None)
            quant_info = resolve_quantization(
                hf_config=hf_config,
                hf_quant_config=hf_quant_config,
                recipe_quant=effective_recipe_quant,
                model_id=self.model,
            )

            if hf_config:
                hf_info = extract_model_info(hf_config)

                # use quant_config to capture on disk storage as quantized if it exists
                # but otherwise fall back to the model dtype
                _storage_dtype = hf_info.get("quant_dtype") or hf_info.get("model_dtype")

                # Fill in missing fields (metadata takes precedence)
                if not model_dtype:
                    if quant_info:
                        model_dtype = quant_info.weight_dtype
                    else:
                        model_dtype = _storage_dtype
                if not num_layers:
                    num_layers = hf_info.get("num_layers")
                if not num_kv_heads:
                    num_kv_heads = hf_info.get("num_kv_heads")
                if not head_dim:
                    head_dim = hf_info.get("head_dim")

                # Architecture markers (MLA's compressed-latent dims, and
                # whatever a future strategy declares).  extract_model_info
                # returns them under the same names the strategies declare, so
                # the fill-in is a sweep, not a per-field list.
                for _name, _value in arch_extra.items():
                    if not _value:
                        arch_extra[_name] = hf_info.get(_name)
                if not model_type:
                    model_type = hf_info.get("model_type")

                # Use kv_cache_quant from hf_quant_config to inform kv_dtype
                if not kv_dtype and quant_info and quant_info.kv_cache_quant:
                    kv_dtype = quant_info.kv_cache_quant
            else:
                # No HF config (e.g. GGUF models) — still use quant_info if available
                if not model_dtype and quant_info:
                    model_dtype = quant_info.weight_dtype

        # Parse model_params
        model_params = parse_param_count(model_params_raw) if model_params_raw is not None else None

        # Fallback: derive model weight info from safetensors when metadata
        # doesn't provide it.
        #
        # fetch_safetensors_size() returns total bytes computed from
        # per-dtype tensor metadata (via API or index).  How we use it
        # depends on whether quantization is pre-baked or applied at runtime:
        #
        # - Pre-quantized (quant from HF config): the returned bytes
        #   already reflect the quantized weights.  Use directly as
        #   model_vram since the per-dtype byte calculation IS the VRAM.
        #
        # - Runtime-quantized (quant from recipe): the returned bytes
        #   reflect the on-disk format (e.g. bf16).  Derive model_params
        #   from total_size / storage_bpe so the VRAM estimator can apply
        #   the target dtype (e.g. fp8).
        _is_runtime_quant = bool(
            effective_recipe_quant
            and effective_recipe_quant not in ("none", "auto", "")
            and _storage_dtype
            and _storage_dtype != model_dtype
        )

        if model_params is None and model_vram is None and auto_detect and self.model:
            total_size = fetch_safetensors_size(self.model, revision=self.model_revision, cache_dir=cache_dir)
            if total_size is not None:
                if _is_runtime_quant:
                    # Runtime quantization: derive params from storage dtype
                    _derive_bpe = bytes_per_element(str(_storage_dtype))
                    if _derive_bpe is not None and _derive_bpe > 0:
                        model_params = int(total_size / _derive_bpe)
                    else:
                        model_vram = total_size / (1024**3)
                else:
                    # Pre-quantized or unquantized: bytes = actual VRAM
                    model_vram = total_size / (1024**3)
            else:
                # Last resort: param count from HF API
                api_params = fetch_safetensors_params(self.model, revision=self.model_revision)
                if api_params is not None:
                    model_params = api_params

        # Get effective max_model_len and tensor_parallel from config chain
        max_model_len = config.get("max_model_len")
        if max_model_len is not None:
            if str(max_model_len).lower() == "auto":
                max_model_len = None
            else:
                max_model_len = int(max_model_len)

        tp_val = config.get("tensor_parallel")
        tensor_parallel = int(tp_val) if tp_val is not None else 1

        pp_val = config.get("pipeline_parallel")
        pipeline_parallel = int(pp_val) if pp_val is not None else 1

        # Check for kv_cache_dtype in defaults (runtime-specific).
        if not kv_dtype:
            kv_cache_default = config.get("kv_cache_dtype")
            if kv_cache_default and str(kv_cache_default) != "auto":
                kv_dtype = str(kv_cache_default)

        # Last resort: parse --kv-cache-dtype out of the free-form command
        # template.  Recipes that set the flag only in command: (rather than
        # in defaults) would otherwise silently fall back to bfloat16 — for
        # MLA models using fp8_ds_mla / nvfp4_ds_mla that's a ~10x KV-cache
        # over-estimate (issue #248).  Warn so the user knows the estimate
        # depends on a command-template parse and is encouraged to pin the
        # value in defaults/metadata instead.
        _kv_from_command: str | None = None
        if not kv_dtype:
            _kv_from_command = extract_kv_cache_dtype_from_command(self.command)
            if _kv_from_command:
                kv_dtype = _kv_from_command

        # GPU memory utilization (runtime budget fraction)
        gpu_mem_val = config.get("gpu_memory_utilization")
        gpu_memory_utilization = float(gpu_mem_val) if gpu_mem_val is not None else None

        result = _estimate_vram(
            model_params=model_params,
            model_dtype=str(model_dtype) if model_dtype else None,
            kv_dtype=str(kv_dtype) if kv_dtype else None,
            num_layers=int(num_layers) if num_layers is not None else None,
            num_kv_heads=int(num_kv_heads) if num_kv_heads is not None else None,
            head_dim=int(head_dim) if head_dim is not None else None,
            max_model_len=max_model_len,
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            model_vram=float(model_vram) if model_vram is not None else None,
            kv_vram_per_token=float(kv_vram_per_token) if kv_vram_per_token is not None else None,
            gpu_memory_utilization=gpu_memory_utilization,
            total_gpu_memory_gb=total_gpu_memory_gb,
            model_type=str(model_type) if model_type else None,
            arch={f.name: f.coerce(arch_extra[f.name]) for f in arch_fields() if arch_extra.get(f.name)},
        )
        if _kv_from_command:
            result.warnings.append(
                "kv_cache_dtype %r inferred from command: template; pin it in "
                "defaults.kv_cache_dtype or metadata.kv_dtype for a stable estimate" % _kv_from_command
            )

        # Write back auto-detected values so downstream consumers
        # (e.g. benchmark result export) can use them without re-fetching.
        #
        # This must stay complete: the written-back architecture fields satisfy
        # ``needs_detection`` above, so anything omitted here is lost on the
        # second call.  A single ``sparkrun run`` estimates three times on one
        # Recipe (host resolution, the displayed banner, then the scheduling
        # pass inside ``api.run``), and the last one feeds the placement's
        # ``ResourceRequest`` — so a partial write-back silently reverts the
        # estimate on the path that decides where ranks land.
        if model_dtype:
            self.metadata["model_dtype"] = normalize_dtype(str(model_dtype))
        if num_layers is not None and "num_layers" not in self.metadata:
            self.metadata["num_layers"] = int(num_layers)
        if num_kv_heads is not None and "num_kv_heads" not in self.metadata:
            self.metadata["num_kv_heads"] = int(num_kv_heads)
        if head_dim is not None and "head_dim" not in self.metadata:
            self.metadata["head_dim"] = int(head_dim)
        if model_params is not None and "model_params" not in self.metadata:
            self.metadata["model_params"] = model_params
        if quant_info and "quantization" not in self.metadata:
            self.metadata["quantization"] = quant_info.method
        if quant_info and quant_info.bits and "quant_bits" not in self.metadata:
            self.metadata["quant_bits"] = quant_info.bits
        # Persist the resolved dtype so benchmark export and telemetry (which
        # read only metadata) record the KV cache configuration that actually
        # ran.  The read side now prefers a fresh cli override over this
        # metadata value, so persisting it cannot shadow a later override the
        # way it did before.
        if kv_dtype and "kv_dtype" not in self.metadata:
            self.metadata["kv_dtype"] = normalize_dtype(str(kv_dtype))
        if model_type and "model_type" not in self.metadata:
            self.metadata["model_type"] = str(model_type)
        # Architecture markers.  A model whose architecture declares none of
        # them leaves these unset on every call, which re-derives the same
        # (correct) verdict.  Sweeping the declaration is what makes the
        # completeness requirement above structural rather than a convention:
        # a field cannot be read at the top and forgotten here.
        for _field in arch_fields():
            _value = arch_extra.get(_field.name)
            if _value and _field.name not in self.metadata:
                self.metadata[_field.name] = _field.coerce(_value)

        return result

    # ------------------------------------------------------------------
    # Internal serialization (full round-trip state, not canonical export)
    # ------------------------------------------------------------------

    _SERIALIZATION_VERSION = 1

    def __getstate__(self) -> dict[str, Any]:
        """Serialize the full effective Recipe state into a plain dict.

        Unlike ``export()`` (which produces a clean canonical recipe),
        this captures *all* resolved fields so the object can be
        faithfully restored without re-running the resolver chain.
        """
        return {
            "_serialization_version": self._SERIALIZATION_VERSION,
            "name": self.name,
            "source_path": self.source_path,
            "source_registry": self.source_registry,
            "source_registry_url": self.source_registry_url,
            "is_url_sourced": self.is_url_sourced,
            "_qualified_name_override": self._qualified_name_override,
            "recipe_version": self.recipe_version,
            "description": self.description,
            "model": self.model,
            "model_revision": self.model_revision,
            "runtime": self.runtime,
            "runtime_version": self.runtime_version,
            "mode": self.mode,
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "container": self.container,
            "containers": [dict(e) for e in self.containers],
            "defaults": dict(self.defaults),
            "env": dict(self.env),
            "command": self.command,
            "metadata": dict(self.metadata),
            "plugin_items": {
                key: (
                    get_recipe_item(key).handler.export(self.plugin_items[key], self)
                    if get_recipe_item(key) is not None and key in self.plugin_items
                    else self._plugin_item_raw[key]
                )
                for key in sorted(set(self._plugin_item_raw) | set(self.plugin_items))
            },
            "maintainer": self.maintainer,
            "runtime_config": dict(self.runtime_config),
            "capabilities": list(self.capabilities),
            "unsupported_capabilities": list(self.unsupported_capabilities),
            "pre_exec": list(self.pre_exec),
            "post_exec": list(self.post_exec),
            "post_commands": list(self.post_commands),
            "stop_after_post": self.stop_after_post,
            "mods": list(self.mods),
            "builder": self.builder,
            "builder_config": dict(self.builder_config),
            "executor": self.executor,
            "executor_config": dict(self.executor_config),
            "scheduler": self.scheduler,
            "distribution_config": dataclass_asdict(self.distribution_config),
            "layout": self.layout.to_dict() if self.layout else None,
            "cluster_config": self.cluster_config.to_dict() if self.cluster_config else None,
            "runtime_cache": dict(self.runtime_cache),
            "_applied_overrides": dict(self._applied_overrides),
            "_raw": dict(self._raw),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore Recipe fields from a state dict produced by ``__getstate__``."""
        self._raw = state.get("_raw", {})
        self.source_path = state.get("source_path")
        self.source_registry = state.get("source_registry")
        self.source_registry_url = state.get("source_registry_url")
        self.is_url_sourced = state.get("is_url_sourced", False)
        self._qualified_name_override = state.get("_qualified_name_override")
        self.recipe_version = state.get("recipe_version", "2")
        self.name = state.get("name", "unnamed")
        self.description = state.get("description", "")
        self.model = state.get("model", "")
        self.model_revision = state.get("model_revision")
        self.runtime = state.get("runtime", "")
        self.runtime_version = state.get("runtime_version", "")
        self.mode = state.get("mode", "auto")
        self.min_nodes = state.get("min_nodes", 1)
        self.max_nodes = state.get("max_nodes")
        self.container = state.get("container", "")
        self.containers = parse_container_entries(state.get("containers"))
        self.defaults = dict(state.get("defaults") or {})
        self.env = dict(state.get("env") or {})
        self.command = state.get("command")
        self.metadata = dict(state.get("metadata") or {})
        self.plugin_items = {}
        self._plugin_item_raw = dict(state.get("plugin_items") or {})
        self.maintainer = state.get("maintainer", "")
        self.runtime_config = dict(state.get("runtime_config") or {})
        self.capabilities = list(state.get("capabilities") or [])
        self.unsupported_capabilities = list(state.get("unsupported_capabilities") or [])
        self.pre_exec = list(state.get("pre_exec") or [])
        self.post_exec = list(state.get("post_exec") or [])
        self.post_commands = list(state.get("post_commands") or [])
        self.stop_after_post = bool(state.get("stop_after_post", False))
        self.mods = list(state.get("mods") or [])
        self.builder = state.get("builder", "")
        self.builder_config = dict(state.get("builder_config") or {})
        self.executor = str(state.get("executor", "") or "")
        self.executor_config = dict(state.get("executor_config") or {})
        self.scheduler = str(state.get("scheduler", "") or "")
        self._applied_overrides = dict(state.get("_applied_overrides") or {})
        dist_cfg: dict | None = state.get("distribution_config", None)
        self.distribution_config = _parse_distribution_config(self._raw) if dist_cfg is None else DistributionConfig.from_dict(dist_cfg)
        layout_state = state.get("layout")
        self.layout = RecipeLayout.from_dict(layout_state) if isinstance(layout_state, dict) else None
        self.cluster_config = LaunchOverrides.from_dict(state.get("cluster_config"))
        self.runtime_cache = dict(state.get("runtime_cache") or {})
        for key, raw_item in self._plugin_item_raw.items():
            registration = get_recipe_item(key)
            if registration is not None:
                self.plugin_items[key] = registration.handler.parse(raw_item, self)

    @classmethod
    def _deserialize(cls, data: dict[str, Any]) -> Recipe:
        """Construct a Recipe from a serialized state dict (no resolution)."""
        instance = cls.__new__(cls)
        instance.__setstate__(data)
        return instance

    def _serialize_yaml(self) -> str:
        """Serialize full Recipe state to a YAML string."""
        from sparkrun.utils.yaml_helpers import LiteralBlockDumper

        return yaml.dump(
            self.__getstate__(),
            Dumper=LiteralBlockDumper,
            indent=2,
            sort_keys=False,
            default_flow_style=False,
        )

    @classmethod
    def _deserialize_yaml(cls, text: str) -> Recipe:
        """Restore a Recipe from a YAML string produced by ``_serialize_yaml``."""
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise RecipeError("Expected a YAML mapping for Recipe deserialization")
        return cls._deserialize(data)

    def __repr__(self) -> str:
        return "Recipe(name=%r, runtime=%r, model=%r)" % (self.name, self.runtime, self.model)

    # Preferred key ordering for export.  Entries are either exact key names
    # or fnmatch-style patterns (e.g. "model*" matches "model", "model_revision").
    # Keys not listed here are appended alphabetically after the last group.
    EXPORT_KEY_ORDER: list[str] = [
        "recipe_version",
        "model*",
        "runtime*",
        "builder*",
        "min_nodes",
        "max_nodes",
        "container",
        "containers",
        "solo_only",
        "cluster_only",
        "layout",
        "cluster_config",
        "metadata",
        "build_args",
        "mods",
        "defaults",
        "env",
        "pre_exec",
        "command",
        "post_exec",
        "post_commands",
        "stop_after_post",
    ]

    # Top-level keys that are folded into metadata on export.
    _METADATA_PROMOTED_KEYS = {"description", "maintainer"}

    def _build_export_dict(self) -> dict[str, Any]:
        """Build a canonical recipe dict from resolved instance attributes.

        Applies normalizations performed by the constructor and resolvers:
        - Uses resolved ``runtime`` (e.g. ``"vllm-distributed"`` not ``"vllm"``).
        - Folds top-level ``description`` into ``metadata.description``.
        - Omits empty/default-valued fields to keep output minimal.
        - Drops v1-only and internal keys (``recipe_version``, ``sparkrun_version``,
          ``name``, ``mode``, ``runtime_config``, unknown sweep keys).
        """
        d: dict[str, Any] = {"recipe_version": self.recipe_version, "model": self.model}

        # -- Core fields (always present) --
        if self.model_revision:
            d["model_revision"] = self.model_revision
        d["runtime"] = self._raw.get("runtime", self.runtime)  # use bare original if given
        if self.runtime_version:
            d["runtime_version"] = self.runtime_version

        # -- Topology -- (accounts for v1 solo_only/cluster_only flags as well)
        if self.min_nodes != 1:
            d["min_nodes"] = self.min_nodes
        if self.max_nodes is not None:
            d["max_nodes"] = self.max_nodes

        # -- Container --
        if self.container:
            d["container"] = self.container
        if self.containers:
            d["containers"] = [dict(e) for e in self.containers]

        # # -- Preserve Raw Topology flags from v1 --
        # if self._raw.get("solo_only"):
        #     d["solo_only"] = True
        # if self._raw.get("cluster_only"):
        #     d["cluster_only"] = True

        # -- Explicit placement layout (optional) --
        if self.layout is not None:
            layout_dict = self.layout.to_dict()
            if layout_dict:
                d["layout"] = layout_dict

        # -- Cluster config (internal launch overrides; undocumented) --
        if self.cluster_config is not None:
            cc_dict = self.cluster_config.to_dict()
            if cc_dict:
                d["cluster_config"] = cc_dict

        # -- Runtime (compilation/autotune) cache knobs --
        if self.runtime_cache:
            d["runtime_cache"] = dict(self.runtime_cache)

        # -- Metadata (absorb promoted keys) --
        d["metadata"] = meta = dict(self.metadata)
        if self.description:
            meta["description"] = self.description
        if self.maintainer:
            meta["maintainer"] = self.maintainer

        # transfer SELECTED model parameters to recipe
        if meta and meta.get("model_dtype", None) is not None:
            meta["model_dtype"] = str(meta["model_dtype"])
        if meta and meta.get("kv_dtype", None) is not None:
            meta["kv_dtype"] = str(meta["kv_dtype"])
        if meta and meta.get("model_params", None) is not None:
            meta["model_params"] = str(meta["model_params"])
        if meta and meta.get("quantization", None) is not None:
            meta["quantization"] = str(meta["quantization"])
        if meta and meta.get("quant_bits", None) is not None:
            meta["quant_bits"] = int(meta["quant_bits"])

        # -- Builder --
        if self.builder:
            d["builder"] = self.builder
        if self.builder_config:
            d["builder_config"] = dict(self.builder_config)

        # -- Scheduler --
        if self.scheduler:
            d["scheduler"] = self.scheduler

        # -- Configuration --
        if self.defaults:
            d["defaults"] = dict(self.defaults)
        if self.env:
            d["env"] = dict(self.env)

        # -- Mods (resolved to pre_exec at run time; preserve source list on export) --
        if self.mods:
            d["mods"] = list(self.mods)

        # -- Lifecycle hooks --
        if self.pre_exec:
            d["pre_exec"] = list(self.pre_exec)
        if self.command:
            d["command"] = self.command
        if self.post_exec:
            d["post_exec"] = list(self.post_exec)
        if self.post_commands:
            d["post_commands"] = list(self.post_commands)
        if self.stop_after_post:
            d["stop_after_post"] = True

        # TODO: consider if we include embedded benchmarks in export or not!
        #       (currently we do not)

        # check for content in runtime_config and then sweep it to top-level for greater compat w/ v1 style
        if self.runtime_config:
            d.update(self.runtime_config)

        # Plugin items stay at the top level they claimed. Unknown state from
        # a serialized recipe is preserved verbatim if its plugin is disabled.
        for key in sorted(set(self._plugin_item_raw) | set(self.plugin_items)):
            registration = get_recipe_item(key)
            if registration is not None and key in self.plugin_items:
                d[key] = registration.handler.export(self.plugin_items[key], self)
            else:
                d[key] = self._plugin_item_raw[key]

        # add distribution_config iff it was provided in the input recipe
        dist_cfg = self.distribution_config
        if dist_cfg and dist_cfg.externally_provided:
            d["distribution_config"] = dataclass_asdict(dist_cfg)

        return d

    def to_dict(
        self,
        overrides: Optional[dict] = None,
        container_image: Optional[str] = None,
    ) -> dict[str, Any]:
        """Convert the recipe to a canonical dictionary.

        Builds a clean dict from resolved attributes (not raw input),
        applies overrides, filters ephemeral fields, and sorts keys.
        """
        export_dict = self._build_export_dict()

        # Bake overrides into defaults so the export is self-contained
        if overrides:
            defaults = dict(export_dict.get("defaults") or {})
            defaults.update(overrides)
            export_dict["defaults"] = defaults

        # Override container with effective image (post-builder)
        if container_image:
            export_dict["container"] = container_image

        # filter out pre-/post- commands that were added by
        # runtime, builder, etc. because those should be reproducible
        # implicitly by relying on the runtime & builder in the future as well
        for key in ("pre_exec", "post_exec", "post_commands"):
            val = self._raw.get(key, [])
            if val:
                export_dict[key] = val
            else:
                export_dict.pop(key, None)

        # ensure that `stop_after_post` is excluded if False
        if not export_dict.get("stop_after_post", False):
            export_dict.pop("stop_after_post", None)

        return _sort_dict_by_patterns(export_dict, self.EXPORT_KEY_ORDER)

    def export(
        self,
        path: Optional[str | Path] = None,
        json: bool = False,
        overrides: Optional[dict] = None,
        container_image: Optional[str] = None,
    ) -> Optional[str | Path]:
        """Export the recipe as canonical YAML.

        Builds a clean dict from resolved attributes (not raw input),
        applies preferred key ordering, and writes YAML.

        Args:
            path: Write to file instead of returning text.
            json: Output JSON instead of YAML.
            overrides: When provided, merge into the exported ``defaults``
                dict so the export captures the effective configuration.
            container_image: When provided, override the ``container`` field
                (accounts for builder mutations).
        """
        from sparkrun.utils.yaml_helpers import LiteralBlockDumper

        ordered = self.to_dict(overrides=overrides, container_image=container_image)

        text = (
            json_dumps(ordered, sort_keys=False)
            if json
            else yaml.dump(ordered, Dumper=LiteralBlockDumper, indent=2, sort_keys=False, default_flow_style=False)
        )

        if path is None:
            return text

        dest = Path(path)
        dest.write_text(text, encoding="utf-8")
        return dest


def _ambiguous(name: str, matches: list[tuple[str, Path]], registry_manager: RegistryManager) -> RecipeAmbiguousError:
    """Build a :class:`RecipeAmbiguousError` with path-qualified labels.

    Labels come from :meth:`RegistryManager.qualified_recipe_name`, so two
    nested matches in one registry render as distinct, re-typeable names
    rather than the same ``@registry/stem`` twice.
    """
    labels = [registry_manager.qualified_recipe_name(reg, path) for reg, path in matches]
    return RecipeAmbiguousError(name, matches, labels=labels)


def find_recipe(
    name: str,
    search_paths: list[Path] | None = None,
    registry_manager: RegistryManager | None = None,
    local_files: list[Path] | None = None,
) -> Path:
    """Find a recipe by name across search paths.

    Supports @registry/recipe-name syntax for scoped lookups.

    Search order:
    1. @registry/name scoped lookup (if @ prefix present)
    2. Exact/relative file path (if exists)
    3. Given search paths
    4. Registry paths (if registry_manager provided)
    5. Registry file-stem matching (if registry_manager provided)

    Raises:
        RecipeAmbiguousError: If name matches multiple recipes — either across
            registries (no @scope) or at multiple paths within the scoped one.
        RecipeError: If recipe not found.
    """
    # Parse @registry/name prefix
    from sparkrun.utils import parse_scoped_name

    scoped_registry, lookup_name = parse_scoped_name(name)

    # Scoped lookup: search only the specified registry
    if scoped_registry and registry_manager:
        matches = registry_manager.find_recipe_in_registries(
            lookup_name,
            include_hidden=True,
        )
        scoped_matches = [(reg, path) for reg, path in matches if reg == scoped_registry]
        if len(scoped_matches) == 1:
            return scoped_matches[0][1]
        # A registry's recipe dir is scanned recursively, so one stem can match
        # several files in the *same* registry. Silently taking the first sorted
        # hit would pick a recipe the user never named (e.g. the 3x-cluster
        # variant when they meant the 4x one) — surface it instead.
        if scoped_matches:
            raise _ambiguous(lookup_name, scoped_matches, registry_manager)
        raise RecipeError("Recipe '%s' not found in registry '%s'" % (lookup_name, scoped_registry))

    # 1. Check if it's a direct path
    direct = Path(lookup_name)
    if direct.exists():
        return direct
    # Also try with .yaml extension
    if not lookup_name.endswith((".yaml", ".yml")):
        for ext in (".yaml", ".yml"):
            candidate = Path(lookup_name + ext)
            if candidate.exists():
                return candidate

    # 2. Check local_files (CWD-discovered recipes) by stem match
    if local_files:
        for lf in local_files:
            if lf.stem == lookup_name:
                return lf
        # Also try with extension stripped if user passed name.yaml
        if lookup_name.endswith((".yaml", ".yml")):
            bare = Path(lookup_name).stem
            for lf in local_files:
                if lf.stem == bare:
                    return lf

    # 3. Search user-provided paths (flat first, then recursive by stem)
    for search_dir in search_paths or []:
        for ext in ("", ".yaml", ".yml"):
            candidate = search_dir / (lookup_name + ext)
            if candidate.exists():
                return candidate
    for search_dir in search_paths or []:
        for ext in (".yaml", ".yml"):
            for m in search_dir.rglob(f"**/{lookup_name}{ext}"):
                return m

    # 4. Search registry paths with ambiguity detection.
    # Use find_recipe_in_registries() which tracks per-registry matches
    # so that identical recipe names across registries raise an error.
    if registry_manager:
        matches = registry_manager.find_recipe_in_registries(lookup_name)
        if len(matches) == 1:
            _registry_name, recipe_path = matches[0]
            return recipe_path
        elif len(matches) > 1:
            raise _ambiguous(lookup_name, matches, registry_manager)

    search_desc = [str(p) for p in (search_paths or [])]
    if registry_manager:
        search_desc.append("registry paths")
    raise RecipeError("Recipe '%s' not found. Searched: %s" % (lookup_name, search_desc))


def find_recipe_in_registry(name: str, registry_name: str, registry_manager: RegistryManager) -> Path:
    """Find a recipe in a specific registry by name.

    Args:
        name: Recipe file stem.
        registry_name: Registry to search.
        registry_manager: Registry manager instance.

    Returns:
        Path to the recipe file.

    Raises:
        RecipeAmbiguousError: If the name matches several paths in that registry.
        RecipeError: If recipe not found in that registry.
    """
    matches = registry_manager.find_recipe_in_registries(name, include_hidden=True)
    scoped = [(reg, path) for reg, path in matches if reg == registry_name]
    if len(scoped) == 1:
        return scoped[0][1]
    if scoped:
        # Recursive scan means one stem can match several files in a single
        # registry; don't guess which one the caller meant.
        raise _ambiguous(name, scoped, registry_manager)
    raise RecipeError("Recipe '%s' not found in registry '%s'" % (name, registry_name))


def recipe_summary(path: Path, registry_name: str | None = None) -> dict[str, Any] | None:
    """Build a lightweight recipe summary dict from a YAML file.

    Returns a metadata dict suitable for recipe listing and search, or
    ``None`` if the file cannot be read or does not contain a dict.

    This is intentionally cheaper than constructing a full :class:`Recipe`
    — it skips version migration, resolver chains, and env expansion.
    """
    try:
        data = read_yaml(str(path))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    stem = path.stem
    defaults = data.get("defaults", {})
    qualified = ("@%s/%s" % (registry_name, stem)) if registry_name else stem
    builder = resolve_builder(data)
    entry: dict[str, Any] = {
        "name": qualified,
        "file": stem,
        "path": str(path),
        "model": data.get("model", ""),
        "description": data.get("description", ""),
        "runtime": resolve_runtime(data),
        "min_nodes": data.get("min_nodes", 1),
        "tp": defaults.get("tensor_parallel", "") if isinstance(defaults, dict) else "",
        "gpu_mem": defaults.get("gpu_memory_utilization", "") if isinstance(defaults, dict) else "",
    }
    if builder:
        entry["builder"] = builder
    if registry_name:
        entry["registry"] = registry_name
    return entry


def list_recipes(
    search_paths: list[Path] | None = None,
    registry_manager: RegistryManager | None = None,
    include_hidden: bool = False,
    local_files: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """List all available recipes with name and path."""
    from sparkrun.core.registry import RECIPE_ASSET, iter_asset_files

    recipes: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    # Process CWD-discovered local files first (no registry label)
    for f in local_files or []:
        if f.stem in seen_names:
            continue
        seen_names.add(f.stem)
        entry = recipe_summary(f)
        if entry is not None:
            recipes.append(entry)

    all_paths = list(search_paths or [])

    # Add registry paths if available
    if registry_manager:
        all_paths.extend(registry_manager.get_recipe_paths(include_hidden=include_hidden))

    for search_dir in all_paths:
        if not search_dir.is_dir():
            continue

        # Determine if this is a registry path
        registry_name = None
        if registry_manager:
            for reg in registry_manager.list_registries():
                if reg.enabled:
                    reg_path = registry_manager.cache_root / reg.name / reg.subpath
                    if search_dir == reg_path or search_dir.is_relative_to(reg_path):
                        registry_name = reg.name
                        break

        for f in iter_asset_files(search_dir, RECIPE_ASSET):
            if f.stem not in seen_names:
                seen_names.add(f.stem)
                entry = recipe_summary(f, registry_name=registry_name)
                if entry is not None:
                    recipes.append(entry)

    return recipes


#: Recipe summary fields a free-text query is matched against.
RECIPE_QUERY_FIELDS: tuple[str, ...] = ("name", "file", "model", "description")


def recipe_matches_query(entry: dict[str, Any], query: str | None) -> bool:
    """Return True when a recipe summary matches a free-text *query*.

    Case-insensitive substring match over :data:`RECIPE_QUERY_FIELDS`. An
    empty/None query matches everything.

    This is the single matching predicate shared by
    ``RegistryManager.search_recipes`` and the local (CWD) recipe search, so
    a recipe sitting in the working directory is found on the same terms as
    one from a registry.
    """
    if not query:
        return True
    needle = query.lower()
    return any(needle in str(entry.get(f, "")).lower() for f in RECIPE_QUERY_FIELDS)


def filter_recipes(
    recipes: list[dict[str, Any]],
    *,
    runtime: str | None = None,
    registry: str | None = None,
) -> list[dict[str, Any]]:
    """Filter a recipe list by runtime and/or registry.

    Args:
        recipes: Recipe metadata dicts (from :func:`list_recipes` or
            ``RegistryManager.search_recipes``).
        runtime: Keep only recipes with this runtime (case-insensitive).
        registry: Keep only recipes from this registry name.

    Returns:
        Filtered list (may be empty).
    """
    result = recipes
    if registry:
        result = [r for r in result if r.get("registry") == registry]
    if runtime:
        rt_lower = runtime.lower()
        result = [r for r in result if r.get("runtime", "").lower() == rt_lower]
    return result
