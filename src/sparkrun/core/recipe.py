"""Recipe loading, validation, and v1->v2 migration."""

from __future__ import annotations

import logging
import re
from json import dumps as json_dumps
from os import path as osp
from pathlib import Path
from typing import Any, TYPE_CHECKING, Optional

import yaml

from vpd.next.util import read_yaml
from vpd.legacy.arguments import arg_substitute
from scitrera_app_framework.api import Variables, EnvPlacement

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
    "stop_after_post",
    "builder",
    "builder_config",
    "executor_config",
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
    elif _CMD_TRTLLM_RE.match(cmd):
        recipe.runtime = "trtllm"


def _resolve_v1_migration(recipe: Recipe) -> None:
    """v1 format recipes -> eugr builder (runtime left for vllm variant resolution)."""
    if recipe.recipe_version != "1":
        return
    if recipe.runtime in ("vllm", "") and not recipe.builder:
        recipe.builder = "eugr"


def _resolve_eugr_signals(recipe: Recipe) -> None:
    """build_args or mods present -> eugr builder (runtime left for vllm variant resolution)."""
    if recipe.runtime not in ("vllm", ""):
        return
    rc = recipe.runtime_config
    if (
        rc.get("build_args") or rc.get("mods") or recipe.container.strip().startswith("ghcr.io/spark-arena/dgx-vllm-eugr-nightly")
    ) and not recipe.builder:
        recipe.builder = "eugr"


def _resolve_vllm_variant(recipe: Recipe) -> None:
    """Bare 'vllm' (or empty) -> 'vllm-distributed' (default) or 'vllm-ray' (Ray hints)."""
    if recipe.runtime not in ("vllm", ""):
        return
    # noinspection PyProtectedMember
    if str(recipe._effective_default("distributed_executor_backend", "")).lower() == "ray":
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
        # Overrides take precedence over defaults
        deb = effective.get("distributed_executor_backend") or defaults.get("distributed_executor_backend", "")
        if str(deb).lower() == "ray":
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


def _url_cache_path(url: str) -> Path:
    """Return the local cache path for a remote recipe URL."""
    import hashlib

    from sparkrun.core.config import DEFAULT_CACHE_DIR

    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return DEFAULT_CACHE_DIR / "remote-recipes" / ("%s.yaml" % url_hash)


def fetch_and_cache_recipe(url: str) -> Path:
    """Fetch a recipe from URL and cache it locally.

    On success, writes/updates the cache file and returns its path.
    On network failure, falls back to cached copy if available.
    Raises RecipeError if fetch fails and no cache exists.
    """
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    cache_path = _url_cache_path(url)

    try:
        req = Request(url, headers={"User-Agent": "sparkrun"})
        with urlopen(req, timeout=30) as resp:
            content = resp.read()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)
        return cache_path
    except (HTTPError, URLError, OSError) as e:
        if cache_path.exists():
            reason = e.code if isinstance(e, HTTPError) else (getattr(e, "reason", None) or str(e))
            logger.warning(
                "Failed to fetch recipe (using cached copy): %s",
                reason,
            )
            return cache_path
        if isinstance(e, HTTPError):
            raise RecipeError("Failed to fetch recipe from %s: HTTP %d" % (url, e.code)) from e
        raise RecipeError("Failed to fetch recipe from %s: %s" % (url, e.reason if isinstance(e, URLError) else e)) from e


# Backward-compat aliases (old underscore names)
_expand_recipe_shortcut = expand_recipe_shortcut
_simplify_recipe_ref = simplify_recipe_ref
_is_recipe_url = is_recipe_url
_fetch_and_cache_recipe = fetch_and_cache_recipe


class RecipeError(Exception):
    """Raised when a recipe is invalid or cannot be loaded."""


class RecipeAmbiguousError(RecipeError):
    """Raised when a recipe name matches multiple registries."""

    def __init__(self, name: str, matches: list[tuple[str, Path]]):
        self.name = name
        self.matches = matches
        registries = ", ".join(reg for reg, _ in matches)
        super().__init__("Recipe '%s' found in multiple registries: %s. Use @registry/%s to specify." % (name, registries, name))


class Recipe:
    """A loaded and validated sparkrun recipe."""

    def __init__(self, data: dict[str, Any], source_path: str | None = None):
        self._raw = data
        self.source_path = source_path
        self.source_registry: str | None = None  # set by _load_recipe after resolution
        self.source_registry_url: str | None = None  # set by _load_recipe after resolution

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

        # Configuration
        self.defaults: dict[str, Any] = dict(data.get("defaults") or {})
        self.env: dict[str, str] = {str(k): osp.expandvars(str(v)) for k, v in (data.get("env") or {}).items()}
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

        # Lifecycle hooks
        self.pre_exec: list[str | dict[str, str]] = list(data.get("pre_exec", []))
        self.post_exec: list[str] = list(data.get("post_exec", []))
        self.post_commands: list[str] = list(data.get("post_commands", []))
        self.stop_after_post: bool = bool(data.get("stop_after_post", False))

        # Builder plugin
        self.builder: str = data.get("builder", "")
        self.builder_config: dict[str, Any] = dict(data.get("builder_config", {}))

        # Executor config (container engine settings: auto_remove, restart_policy, etc.)
        raw_exec = data.get("executor_config", {})
        self.executor_config: dict[str, Any] = dict(raw_exec) if isinstance(raw_exec, dict) else {}

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

    def build_config_chain(self, cli_overrides: dict[str, Any] | None = None, user_config: dict[str, Any] | None = None) -> Variables:
        """Build cascading config: CLI overrides -> user config -> recipe defaults.

        Also injects 'model' into the chain for template substitution.
        """
        base = dict(self.defaults)
        base.setdefault("model", self.model)
        return Variables(sources=(cli_overrides or {}, user_config or {}, base), env_placement=EnvPlacement.IGNORED)

    def render_command(self, config_chain: Variables) -> str | None:
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

        assert isinstance(rendered, str)

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

        return issues

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
        kv_dtype = normalize_dtype(str(_raw_kv)) if _raw_kv else None
        num_layers = self.metadata.get("num_layers")
        num_kv_heads = self.metadata.get("num_kv_heads")
        head_dim = self.metadata.get("head_dim")
        model_vram = self.metadata.get("model_vram")
        kv_vram_per_token = self.metadata.get("kv_vram_per_token")
        quant_info: QuantizationInfo | None = None
        _storage_dtype: str | None = None  # raw torch_dtype before quant override
        effective_recipe_quant: str | None = None  # recipe-level quantization override

        # Auto-detect from HF if fields are missing and model is specified
        if auto_detect and self.model:
            needs_detection = (model_vram is None and (not model_dtype or model_params_raw is None)) or (
                kv_vram_per_token is None and (not num_layers or not num_kv_heads or not head_dim)
            )
            if needs_detection:
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

                    # Capture the raw storage dtype (torch_dtype) before
                    # quantization override — needed later when deriving
                    # model_params from on-disk total_size.
                    _storage_dtype = hf_info.get("model_dtype")

                    # Fill in missing fields (metadata takes precedence)
                    if not model_dtype:
                        model_dtype = quant_info.weight_dtype if quant_info else _storage_dtype
                    if not num_layers:
                        num_layers = hf_info.get("num_layers")
                    if not num_kv_heads:
                        num_kv_heads = hf_info.get("num_kv_heads")
                    if not head_dim:
                        head_dim = hf_info.get("head_dim")

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
            max_model_len = int(str(max_model_len))

        tp_val = config.get("tensor_parallel")
        tensor_parallel = int(str(tp_val)) if tp_val is not None else 1

        pp_val = config.get("pipeline_parallel")
        pipeline_parallel = int(str(pp_val)) if pp_val is not None else 1

        # Check for kv_cache_dtype in defaults (runtime-specific)
        if not kv_dtype:
            kv_cache_default = config.get("kv_cache_dtype")
            if kv_cache_default and str(kv_cache_default) != "auto":
                kv_dtype = str(kv_cache_default)

        # GPU memory utilization (runtime budget fraction)
        gpu_mem_val = config.get("gpu_memory_utilization")
        gpu_memory_utilization = float(str(gpu_mem_val)) if gpu_mem_val is not None else None

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
        )

        # Write back auto-detected values so downstream consumers
        # (e.g. benchmark result export) can use them without re-fetching.
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
        if kv_dtype:
            self.metadata["kv_dtype"] = normalize_dtype(str(kv_dtype))

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
            "defaults": dict(self.defaults),
            "env": dict(self.env),
            "command": self.command,
            "metadata": dict(self.metadata),
            "maintainer": self.maintainer,
            "runtime_config": dict(self.runtime_config),
            "pre_exec": list(self.pre_exec),
            "post_exec": list(self.post_exec),
            "post_commands": list(self.post_commands),
            "stop_after_post": self.stop_after_post,
            "builder": self.builder,
            "builder_config": dict(self.builder_config),
            "executor_config": dict(self.executor_config),
            "_applied_overrides": dict(self._applied_overrides),
            "_raw": dict(self._raw),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore Recipe fields from a state dict produced by ``__getstate__``."""
        self._raw = state.get("_raw", {})
        self.source_path = state.get("source_path")
        self.source_registry = state.get("source_registry")
        self.source_registry_url = state.get("source_registry_url")
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
        self.defaults = dict(state.get("defaults") or {})
        self.env = dict(state.get("env") or {})
        self.command = state.get("command")
        self.metadata = dict(state.get("metadata") or {})
        self.maintainer = state.get("maintainer", "")
        self.runtime_config = dict(state.get("runtime_config") or {})
        self.pre_exec = list(state.get("pre_exec") or [])
        self.post_exec = list(state.get("post_exec") or [])
        self.post_commands = list(state.get("post_commands") or [])
        self.stop_after_post = bool(state.get("stop_after_post", False))
        self.builder = state.get("builder", "")
        self.builder_config = dict(state.get("builder_config") or {})
        self.executor_config = dict(state.get("executor_config") or {})
        self._applied_overrides = dict(state.get("_applied_overrides") or {})

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
        "solo_only",
        "cluster_only",
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
        d["metadata"] = meta = dict(self.metadata)
        if self.description:
            meta["description"] = self.description
        if self.maintainer:
            meta["maintainer"] = self.maintainer

        # transfer SELECTED model parameters to recipe
        if meta and meta.get("model_dtype") is not None:
            meta["model_dtype"] = str(meta["model_dtype"])
        if meta and meta.get("kv_dtype") is not None:
            meta["kv_dtype"] = str(meta["kv_dtype"])
        if meta and meta.get("model_params") is not None:
            meta["model_params"] = str(meta["model_params"])
        if meta and meta.get("quantization") is not None:
            meta["quantization"] = str(meta["quantization"])
        if meta and meta.get("quant_bits") is not None:
            meta["quant_bits"] = int(meta["quant_bits"])

        # -- Builder --
        if self.builder:
            d["builder"] = self.builder
        if self.builder_config:
            d["builder_config"] = dict(self.builder_config)

        # -- Configuration --
        if self.defaults:
            d["defaults"] = dict(self.defaults)
        if self.env:
            d["env"] = dict(self.env)

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
        RecipeAmbiguousError: If name matches multiple registries without @scope.
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
        if scoped_matches:
            return scoped_matches[0][1]
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
            raise RecipeAmbiguousError(lookup_name, matches)

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
