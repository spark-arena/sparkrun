"""Recipe loading, validation, and v1->v2 migration."""

from __future__ import annotations

import logging
import re
from os import path as osp
from pathlib import Path
from typing import Any, TYPE_CHECKING, Optional

import yaml

from vpd.next.util import read_yaml
from vpd.legacy.yaml_dict import vpd_chain, VirtualPathDictChain
from vpd.legacy.arguments import arg_substitute

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
_CMD_VLLM_RE = re.compile(r"^vllm\s+serve\b")
_CMD_SGLANG_RE = re.compile(r"^(?:sglang\s+serve|python3?\s+-m\s+sglang\.launch_server)\b")
_CMD_LLAMA_CPP_RE = re.compile(r"^llama-server\b")

_KNOWN_KEYS = {
    "sparkrun_version", "recipe_version", "name", "description", "model",
    "model_revision",
    "runtime", "runtime_version", "mode", "min_nodes", "max_nodes",
    "container", "defaults", "env", "command", "runtime_config",
    "cluster_only", "solo_only",
    "benchmark", "metadata",
}


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

    return


def _resolve_v1_migration(recipe: Recipe) -> None:
    """v1 format recipes -> eugr-vllm runtime."""
    if recipe.recipe_version != "1":
        return
    if recipe.runtime in ("vllm", ""):
        recipe.runtime = "eugr-vllm"


def _resolve_eugr_signals(recipe: Recipe) -> None:
    """build_args or mods present -> eugr-vllm."""
    if recipe.runtime not in ("vllm", ""):
        return
    rc = recipe.runtime_config
    if rc.get("build_args") or rc.get("mods"):
        recipe.runtime = "eugr-vllm"


def _resolve_vllm_variant(recipe: Recipe) -> None:
    """Bare 'vllm' (or empty) -> 'vllm-distributed' (default) or 'vllm-ray' (Ray hints)."""
    if recipe.runtime not in ("vllm", ""):
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
    _resolve_eugr_signals,
    _resolve_vllm_variant,
]


def resolve_runtime(data: dict[str, Any]) -> str:
    """Lightweight runtime resolution from raw data (for listing/display).

    Mirrors the runtime-affecting resolvers in :data:`_RECIPE_RESOLVERS`
    without constructing a full Recipe.
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
        # vllm serve or unrecognised → fall through to vllm variant resolution

    # v1 migration and eugr detection (mirror _resolve_v1_migration and _resolve_eugr_signals)
    version = str(data.get("sparkrun_version", data.get("recipe_version", "2")))
    if runtime in ("vllm", "") and version == "1":
        return "eugr-vllm"

    # In Recipe.__init__, unknown keys are swept into runtime_config, and
    # _resolve_eugr_signals inspects recipe.runtime_config for build_args/mods.
    runtime_config = data.get("runtime_config") or {}
    if runtime_config is not None and not isinstance(runtime_config, dict):
        raise RecipeError("Recipe 'runtime_config' field must be a mapping, got %s" % type(runtime_config).__name__)
    if runtime in ("vllm", "") and (
            data.get("build_args")
            or data.get("mods")
            or runtime_config.get("build_args")
            or runtime_config.get("mods")
    ):
        return "eugr-vllm"
    if runtime in ("vllm", ""):
        defaults = data.get("defaults")
        if defaults is not None and not isinstance(defaults, dict):
            raise RecipeError("Recipe 'defaults' field must be a mapping, got %s" % type(defaults).__name__)
        defaults = defaults or {}
        if str(defaults.get("distributed_executor_backend", "")).lower() == "ray":
            return "vllm-ray"
        if _RAY_BACKEND_RE.search(cmd):
            return "vllm-ray"
        return "vllm-distributed"
    return runtime


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


class RecipeError(Exception):
    """Raised when a recipe is invalid or cannot be loaded."""


class RecipeAmbiguousError(RecipeError):
    """Raised when a recipe name matches multiple registries."""

    def __init__(self, name: str, matches: list[tuple[str, Path]]):
        self.name = name
        self.matches = matches
        registries = ", ".join(reg for reg, _ in matches)
        super().__init__(
            "Recipe '%s' found in multiple registries: %s. "
            "Use @registry/%s to specify." % (name, registries, name)
        )


class Recipe:
    """A loaded and validated sparkrun recipe."""

    def __init__(self, data: dict[str, Any], source_path: str | None = None):
        self._raw = data
        self.source_path = source_path
        self.source_registry: str | None = None  # set by _load_recipe after resolution
        self.source_registry_url: str | None = None  # set by _load_recipe after resolution

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
        if self.mode == 'solo':
            self.max_nodes = self.min_nodes = 1
        elif self.mode == 'auto' and self.min_nodes > 1:
            self.mode = 'cluster'
        elif self.mode == 'auto' and self.max_nodes == 1:
            self.mode = 'solo'

        # Topology - Handle solo_only/cluster_only as first-class fields (works for both v1 and v2)
        if data.get("cluster_only"):
            self.min_nodes = max(self.min_nodes, 2)
            self.mode = "cluster"
        if data.get("solo_only"):
            self.max_nodes = 1
            self.mode = "solo"

        # Container
        self.container: str = data.get("container", "")

        # Configuration
        self.defaults: dict[str, Any] = dict(data.get("defaults", {}))
        self.env: dict[str, str] = {
            str(k): osp.expandvars(str(v)) for k, v in data.get("env", {}).items()
        }
        self.command: str | None = data.get("command")

        # Metadata section (v2 extension for VRAM estimation, model info)
        raw_metadata = data.get("metadata", {})
        self.metadata: dict[str, Any] = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}

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
        for k, v in data.items():
            if k not in _KNOWN_KEYS and k not in self.runtime_config:
                self.runtime_config[k] = v

        # Post-init resolver chain — all runtime/migration logic lives here
        for resolver in _RECIPE_RESOLVERS:
            resolver(self)

    @property
    def slug(self) -> str:
        """URL/filesystem-safe slug derived from name."""
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")

    def get_default(self, key: str, fallback: Any = None) -> Any:
        """Get a value from defaults with optional fallback."""
        return self.defaults.get(key, fallback)

    def build_config_chain(self, cli_overrides: dict[str, Any] | None = None,
                           user_config: dict[str, Any] | None = None) -> VirtualPathDictChain:
        """Build cascading config: CLI overrides -> user config -> recipe defaults.

        Also injects 'model' into the chain for template substitution.
        """
        base = dict(self.defaults)
        base.setdefault("model", self.model)
        return vpd_chain(cli_overrides or {}, user_config or {}, base)

    def render_command(self, config_chain: VirtualPathDictChain) -> str | None:
        """Render the command template with values from the config chain.

        Returns None if no command template is defined.
        """
        if not self.command:
            return None

        rendered = self.command.strip()

        # Use vpd arg_substitute for {placeholder} replacement
        # Iterate to handle nested substitutions
        last = None
        while last != rendered:
            last = rendered
            rendered = arg_substitute(rendered, config_chain)

        # Fix trailing spaces after backslash line-continuations.
        # ``\<space><newline>`` → ``\<newline>``
        rendered = _TRAILING_SPACE_CONTINUATION_RE.sub("\\\n", rendered)

        return rendered

    def validate(self) -> list[str]:
        """Validate the recipe and return a list of warnings/errors."""
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

        # Validate metadata if present
        if self.metadata:
            from sparkrun.models.vram import parse_param_count, bytes_per_element

            mp = self.metadata.get("model_params")
            if mp is not None and parse_param_count(mp) is None:
                issues.append("metadata.model_params %r is not a valid parameter count" % mp)
            md = self.metadata.get("model_dtype")
            if md is not None and bytes_per_element(str(md)) is None:
                issues.append("metadata.model_dtype %r is not a recognized dtype" % md)
            kd = self.metadata.get("kv_dtype")
            if kd is not None and bytes_per_element(str(kd)) is None:
                issues.append("metadata.kv_dtype %r is not a recognized dtype" % kd)

        return issues

    @classmethod
    def load(cls, path: str | Path) -> Recipe:
        """Load a recipe from a YAML file path."""
        path = Path(path)
        if not path.exists():
            raise RecipeError("Recipe file not found: %s" % path)
        data = read_yaml(str(path))
        if not isinstance(data, dict):
            raise RecipeError("Recipe file must contain a YAML mapping: %s" % path)
        return cls(data, source_path=str(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Recipe:
        """Create a recipe from a dict (useful for testing)."""
        return cls(data)

    def estimate_vram(
            self,
            cli_overrides: dict[str, Any] | None = None,
            auto_detect: bool = True,
            cache_dir: str | None = None,
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
        from sparkrun.models.vram import (
            bytes_per_element,
            estimate_vram as _estimate_vram,
            extract_model_info,
            fetch_model_config,
            fetch_safetensors_size,
            parse_param_count,
        )

        config = self.build_config_chain(cli_overrides)

        # Start with metadata values
        model_dtype = self.metadata.get("model_dtype")
        model_params_raw = self.metadata.get("model_params")
        kv_dtype = self.metadata.get("kv_dtype")
        num_layers = self.metadata.get("num_layers")
        num_kv_heads = self.metadata.get("num_kv_heads")
        head_dim = self.metadata.get("head_dim")
        model_vram = self.metadata.get("model_vram")
        kv_vram_per_token = self.metadata.get("kv_vram_per_token")

        # Auto-detect from HF if fields are missing and model is specified
        if auto_detect and self.model:
            needs_detection = (
                                      model_vram is None
                                      and (not model_dtype or model_params_raw is None)
                              ) or (
                                      kv_vram_per_token is None
                                      and (not num_layers or not num_kv_heads or not head_dim)
                              )
            if needs_detection:
                hf_config = fetch_model_config(self.model, revision=self.model_revision, cache_dir=cache_dir)
                if hf_config:
                    hf_info = extract_model_info(hf_config)
                    # Fill in missing fields (metadata takes precedence)
                    if not model_dtype:
                        model_dtype = hf_info.get("model_dtype")
                    if not num_layers:
                        num_layers = hf_info.get("num_layers")
                    if not num_kv_heads:
                        num_kv_heads = hf_info.get("num_kv_heads")
                    if not head_dim:
                        head_dim = hf_info.get("head_dim")

        # Parse model_params
        model_params = parse_param_count(model_params_raw) if model_params_raw is not None else None

        # Fallback: derive model_params from safetensors index when metadata
        # doesn't provide it.  The index file records total_size in bytes;
        # dividing by bytes-per-element gives an approximate parameter count.
        if model_params is None and model_vram is None and auto_detect and self.model and model_dtype:
            bpe = bytes_per_element(str(model_dtype))
            if bpe is not None and bpe > 0:
                total_size = fetch_safetensors_size(self.model, revision=self.model_revision, cache_dir=cache_dir)
                if total_size is not None:
                    model_params = int(total_size / bpe)

        # Get effective max_model_len and tensor_parallel from config chain
        max_model_len = config.get("max_model_len")
        if max_model_len is not None:
            max_model_len = int(max_model_len)

        tp_val = config.get("tensor_parallel")
        tensor_parallel = int(tp_val) if tp_val is not None else 1

        # Check for kv_cache_dtype in defaults (runtime-specific)
        if not kv_dtype:
            kv_cache_default = config.get("kv_cache_dtype")
            if kv_cache_default and str(kv_cache_default) != "auto":
                kv_dtype = str(kv_cache_default)

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
            model_vram=float(model_vram) if model_vram is not None else None,
            kv_vram_per_token=float(kv_vram_per_token) if kv_vram_per_token is not None else None,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        # Write back auto-detected values so downstream consumers
        # (e.g. benchmark result export) can use them without re-fetching.
        if model_dtype and "model_dtype" not in self.metadata:
            self.metadata["model_dtype"] = str(model_dtype)
        if num_layers is not None and "num_layers" not in self.metadata:
            self.metadata["num_layers"] = int(num_layers)
        if num_kv_heads is not None and "num_kv_heads" not in self.metadata:
            self.metadata["num_kv_heads"] = int(num_kv_heads)
        if head_dim is not None and "head_dim" not in self.metadata:
            self.metadata["head_dim"] = int(head_dim)
        if model_params is not None and "model_params" not in self.metadata:
            self.metadata["model_params"] = model_params

        return result

    def __repr__(self) -> str:
        return "Recipe(name=%r, runtime=%r, model=%r)" % (self.name, self.runtime, self.model)

    # Preferred key ordering for export.  Entries are either exact key names
    # or fnmatch-style patterns (e.g. "model*" matches "model", "model_revision").
    # Keys not listed here are appended alphabetically after the last group.
    EXPORT_KEY_ORDER: list[str] = [
        'recipe_version',
        "model*",
        "runtime*",
        "min_nodes", "max_nodes",
        "container",
        "solo_only", "cluster_only",
        "metadata",
        "defaults",
        "env",
        "command",
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
        d: dict[str, Any] = {
            'recipe_version': self.recipe_version,
            "model": self.model
        }

        # -- Core fields (always present) --
        if self.model_revision:
            d["model_revision"] = self.model_revision
        d["runtime"] = self._raw.get('runtime', self.runtime)  # use bare original if given
        if self.runtime_version:
            d["runtime_version"] = self.runtime_version

        # -- Topology --
        if self.min_nodes != 1:
            d["min_nodes"] = self.min_nodes
        if self.max_nodes is not None:
            d["max_nodes"] = self.max_nodes

        # -- Container --
        if self.container:
            d["container"] = self.container

        # -- Preserve Raw Topology flags from v1 --
        if self._raw.get("solo_only"):
            d["solo_only"] = True
        if self._raw.get("cluster_only"):
            d["cluster_only"] = True

        # -- Metadata (absorb promoted keys) --
        meta = dict(self.metadata)
        if self.description:
            meta["description"] = self.description
        if self.maintainer:
            meta["maintainer"] = self.maintainer

        # transfer SELECTED model parameters to recipe
        if meta and meta.get('model_dtype', None) is not None:
            meta['model_dtype'] = str(meta['model_dtype'])
        # TODO: kv_dtype should be included and reflect command overrides on kv dtype not just hf auto-detect
        # if meta and meta.get('kv_dtype', None) is not None:
        #     meta['kv_dtype'] = str(meta['kv_dtype'])
        if meta and meta.get('model_params', None) is not None:
            meta['model_params'] = str(meta['model_params'])

        # -- Configuration --
        if self.defaults:
            d["defaults"] = dict(self.defaults)
        if self.env:
            d["env"] = dict(self.env)
        if self.command:
            d["command"] = self.command

        # TODO: consider if we include embedded benchmarks in export or not!
        #       (currently we do not)

        return d

    def export(self, path: Optional[str | Path]) -> Optional[str]:
        """Export the recipe as canonical YAML.

        Builds a clean dict from resolved attributes (not raw input),
        applies preferred key ordering, and writes YAML.
        """
        export_dict = self._build_export_dict()
        ordered = _sort_dict_by_patterns(export_dict, self.EXPORT_KEY_ORDER)
        recipe_text = yaml.safe_dump(ordered, indent=2, sort_keys=False)
        if path is None:
            return recipe_text
        Path(path).write_text(recipe_text, encoding="utf-8")
        return None


def find_recipe(name: str, search_paths: list[Path] | None = None,
                registry_manager: RegistryManager | None = None,
                local_files: list[Path] | None = None) -> Path:
    """Find a recipe by name across search paths.

    Supports @registry/recipe-name syntax for scoped lookups.

    Search order:
    1. @registry/name scoped lookup (if @ prefix present)
    2. Exact/relative file path (if exists)
    3. Given search paths
    4. Registry paths (if registry_manager provided)
    5. Registry file-stem matching (if registry_manager provided)

    Raises:
        RecipeAmbiguousError: If name matches multiple registries without @scope.
        RecipeError: If recipe not found.
    """
    # Parse @registry/name prefix
    from sparkrun.utils import parse_scoped_name
    scoped_registry, lookup_name = parse_scoped_name(name)

    # Scoped lookup: search only the specified registry
    if scoped_registry and registry_manager:
        matches = registry_manager.find_recipe_in_registries(
            lookup_name, include_hidden=True,
        )
        scoped_matches = [(reg, path) for reg, path in matches if reg == scoped_registry]
        if scoped_matches:
            return scoped_matches[0][1]
        raise RecipeError(
            "Recipe '%s' not found in registry '%s'" % (lookup_name, scoped_registry)
        )

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
    for search_dir in (search_paths or []):
        for ext in ("", ".yaml", ".yml"):
            candidate = search_dir / (lookup_name + ext)
            if candidate.exists():
                return candidate
    for search_dir in (search_paths or []):
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
            raise RecipeAmbiguousError(lookup_name, matches)

    search_desc = [str(p) for p in (search_paths or [])]
    if registry_manager:
        search_desc.append("registry paths")
    raise RecipeError(
        "Recipe '%s' not found. Searched: %s"
        % (lookup_name, search_desc)
    )


def find_recipe_in_registry(name: str, registry_name: str,
                            registry_manager: RegistryManager) -> Path:
    """Find a recipe in a specific registry by name.

    Args:
        name: Recipe file stem.
        registry_name: Registry to search.
        registry_manager: Registry manager instance.

    Returns:
        Path to the recipe file.

    Raises:
        RecipeError: If recipe not found in that registry.
    """
    matches = registry_manager.find_recipe_in_registries(name, include_hidden=True)
    for reg, path in matches:
        if reg == registry_name:
            return path
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
    entry: dict[str, Any] = {
        "name": stem,
        "file": stem,
        "path": str(path),
        "model": data.get("model", ""),
        "description": data.get("description", ""),
        "runtime": resolve_runtime(data),
        "min_nodes": data.get("min_nodes", 1),
        "tp": defaults.get("tensor_parallel", "") if isinstance(defaults, dict) else "",
        "gpu_mem": defaults.get("gpu_memory_utilization", "") if isinstance(defaults, dict) else "",
    }
    if registry_name:
        entry["registry"] = registry_name
    return entry


def list_recipes(search_paths: list[Path] | None = None,
                 registry_manager: RegistryManager | None = None,
                 include_hidden: bool = False,
                 local_files: list[Path] | None = None) -> list[dict[str, Any]]:
    """List all available recipes with name and path."""
    recipes: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    # Process CWD-discovered local files first (no registry label)
    for f in (local_files or []):
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

        for f in sorted(search_dir.rglob("*.yaml")):
            if f.stem not in seen_names:
                seen_names.add(f.stem)
                entry = recipe_summary(f, registry_name=registry_name)
                if entry is not None:
                    recipes.append(entry)

    return recipes


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
