"""User configuration management for sparkrun."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING, Optional

import yaml
from vpd.next.util import read_yaml

if TYPE_CHECKING:
    from scitrera_app_framework import Variables
    from sparkrun.core.registry import RegistryManager
    from sparkrun.proxy.config import ProxyConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "sparkrun"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sparkrun"

# Defaults for the vllm-tune backing engine (https://github.com/SeraphimSerapis/vllm-tune).
# Overridable via `tuning.vllm_tune_repo` / `tuning.vllm_tune_ref` in config.yaml.
DEFAULT_VLLM_TUNE_REPO = "https://github.com/SeraphimSerapis/vllm-tune.git"
DEFAULT_VLLM_TUNE_REF = "main"

# Defer to huggingface_hub's own resolution of the cache root, which
# respects HF_HOME, HF_HUB_CACHE, and HUGGINGFACE_HUB_CACHE env vars.
try:
    from huggingface_hub.constants import HF_HOME as _HF_HOME

    DEFAULT_HF_CACHE_DIR = Path(_HF_HOME)
except ImportError:  # pragma: no cover — huggingface_hub is a required dep
    DEFAULT_HF_CACHE_DIR = Path.home() / ".cache" / "huggingface"


def resolve_sparkrun_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve sparkrun's own cache directory (~/.cache/sparkrun/).

    For HuggingFace model cache, use ``resolve_cache_dir()`` instead.
    """
    if cache_dir is not None:
        return Path(cache_dir)
    return DEFAULT_CACHE_DIR


def resolve_hf_cache_home(cache_dir: str | None) -> str:
    """Resolve an optional cache directory override to a concrete path.

    Returns *cache_dir* if provided, otherwise the HuggingFace cache
    directory as resolved by ``huggingface_hub`` (respecting ``HF_HOME``
    and related env vars).
    """
    return cache_dir or str(DEFAULT_HF_CACHE_DIR)


def resolve_hf_token() -> Optional[str]:
    try:
        # noinspection PyUnusedImports
        from huggingface_hub import get_token

        return get_token()
    except ImportError:
        pass

    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def get_config_root(v: Variables | None = None) -> Path:
    """Config root from SAF stateful root, falling back to DEFAULT_CONFIG_DIR."""
    if v is not None:
        from scitrera_app_framework.core import is_stateful_ready

        stateful_root = is_stateful_ready(v)
        if stateful_root:
            return Path(stateful_root)
    return DEFAULT_CONFIG_DIR


class SparkrunConfig:
    """Manages sparkrun user configuration."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or (DEFAULT_CONFIG_DIR / "config.yaml")
        self._data: dict[str, Any] = {}
        self._proxy_config: "ProxyConfig | None" = None
        self._load()

    def _load(self):
        if self.config_path.exists():
            self._data = read_yaml(str(self.config_path)) or {}
        else:
            self._data = {}

    @property
    def cache_dir(self) -> Path:
        return Path(self._data.get("cache_dir", str(DEFAULT_CACHE_DIR)))

    @property
    def hf_cache_dir(self) -> Path:
        return Path(self._data.get("hf_cache_dir", str(DEFAULT_HF_CACHE_DIR)))

    @property
    def runtime_cache(self) -> dict[str, Any]:
        """User-level ``runtime_cache:`` block (compilation/autotune cache).

        Top-level rather than nested under a ``cache:`` section because
        ``cache_dir`` / ``hf_cache_dir`` are already top-level scalars — a
        ``cache:`` block would straddle two spellings of the same idea.  See
        :func:`sparkrun.core.runtime_cache.resolve_runtime_cache_settings` for
        the layered chain this participates in.
        """
        raw = self._data.get("runtime_cache")
        return dict(raw) if isinstance(raw, dict) else {}

    @property
    def default_benchmark_output_dir(self) -> Path:
        defaults = self._data.get("defaults", {})
        dir_val = defaults.get("benchmark_output_dir")
        return Path(os.path.expanduser(str(dir_val))) if dir_val else Path.cwd()

    @property
    def default_hosts(self) -> list[str]:
        cluster = self._data.get("cluster", {})
        return cluster.get("hosts", [])

    @property
    def default_image_prefix(self) -> str:
        defaults = self._data.get("defaults", {})
        return defaults.get("image_prefix", "")

    @property
    def default_transformers_tag(self) -> str:
        defaults = self._data.get("defaults", {})
        return defaults.get("transformers", "t4")

    @property
    def default_benchmark_framework(self) -> str:
        """Site-wide default benchmarking framework name.

        Resolved from ``defaults.benchmark_framework`` in ``config.yaml``,
        falling back to ``"llama-benchy"`` when unset.  CLI invocations
        without an explicit ``--framework`` flag use this value.
        """
        defaults = self._data.get("defaults", {})
        val = defaults.get("benchmark_framework") if isinstance(defaults, dict) else None
        return str(val) if val else "llama-benchy"

    @property
    def default_executor(self) -> str | None:
        """System-wide executor pin (``"docker"`` / ``"local"`` / ``"k8s"``).

        Falls below recipe-level ``executor:`` and the runtime's
        ``default_executor()`` in the resolution chain — so a user can
        set a sane site-wide default without overriding per-recipe
        choices.  ``None`` (default) means "no opinion".
        """
        defaults = self._data.get("defaults", {})
        val = defaults.get("executor") or self._data.get("default_executor")
        return str(val).strip().lower() if val else None

    @property
    def executor_config(self) -> dict[str, Any]:
        """System-wide ``executor_config`` overrides (e.g. ``k8s_namespace``).

        Merged into the executor resolution chain below recipe overrides
        and runtime adjustments.  Empty dict when unset.
        """
        cfg = self._data.get("executor_config")
        return dict(cfg) if isinstance(cfg, dict) else {}

    @property
    def k8s_defaults(self) -> dict[str, Any]:
        """CLI / setup-time Kubernetes defaults (the ``k8s:`` block).

        Distinct from :attr:`executor_config` (which feeds executor-time
        ``kubeconfig`` / ``k8s_*`` overrides): this block holds the target
        a plain ``sparkrun setup k8s ...`` invocation defaults to, plus
        the ``kubectl`` binary settings (``path`` / ``version`` / per-
        context ``pinned`` versions).  Empty dict when unset.
        """
        cfg = self._data.get("k8s")
        return dict(cfg) if isinstance(cfg, dict) else {}

    @property
    def k8s_launcher_image(self) -> str | None:
        """Container image for the in-cluster launcher Job (``k8s.launcher_image``).

        The job-driven launch path runs sparkrun's orchestration inside
        this image (typically a published sparkrun container).  ``None``
        when unset — callers must then supply an explicit image.
        """
        val = self.k8s_defaults.get("launcher_image")
        return str(val) if val else None

    def _kubectl_settings(self) -> dict[str, Any]:
        kubectl = self.k8s_defaults.get("kubectl")
        return kubectl if isinstance(kubectl, dict) else {}

    def _k8s_subsection(self, key: str) -> dict[str, Any]:
        sub = self.k8s_defaults.get(key)
        return sub if isinstance(sub, dict) else {}

    @property
    def kueue_version(self) -> str | None:
        """Pinned Kueue release to install (``k8s.kueue.version``)."""
        val = self._k8s_subsection("kueue").get("version")
        return str(val) if val else None

    @property
    def jobset_version(self) -> str | None:
        """Pinned JobSet release to install (``k8s.jobset.version``)."""
        val = self._k8s_subsection("jobset").get("version")
        return str(val) if val else None

    @property
    def kubectl_path(self) -> str | None:
        """Explicit ``kubectl`` binary path override (``k8s.kubectl.path``)."""
        val = self._kubectl_settings().get("path")
        return str(val) if val else None

    @property
    def kubectl_version(self) -> str | None:
        """Pinned ``kubectl`` version (``k8s.kubectl.version``)."""
        val = self._kubectl_settings().get("version")
        return str(val) if val else None

    def kubectl_pinned_version(self, context: str | None) -> str | None:
        """Server-matched ``kubectl`` version pinned for *context*, if any."""
        if not context:
            return None
        pinned = self._kubectl_settings().get("pinned")
        if isinstance(pinned, dict):
            val = pinned.get(context)
            return str(val) if val else None
        return None

    def pin_kubectl_version(self, context: str, version: str) -> None:
        """Persist a per-context ``kubectl`` version pin under ``k8s.kubectl.pinned``."""
        k8s = self.get("k8s")
        if not isinstance(k8s, dict):
            k8s = {}
        kubectl = k8s.get("kubectl")
        if not isinstance(kubectl, dict):
            kubectl = {}
        pinned = kubectl.get("pinned")
        if not isinstance(pinned, dict):
            pinned = {}
        pinned[context] = version
        kubectl["pinned"] = pinned
        k8s["kubectl"] = kubectl
        self.set("k8s", k8s)

    @property
    def ssh_user(self) -> str | None:
        if hasattr(self, "_ssh_user_override"):
            return self._ssh_user_override
        ssh = self._data.get("ssh", {})
        return ssh.get("user")

    @ssh_user.setter
    def ssh_user(self, value: str | None) -> None:
        self._ssh_user_override = value

    @property
    def ssh_key(self) -> str | None:
        ssh = self._data.get("ssh", {})
        key = ssh.get("key")
        return os.path.expanduser(key) if key else None

    @property
    def ssh_options(self) -> list[str]:
        ssh = self._data.get("ssh", {})
        return ssh.get("options", [])

    @property
    def max_parallel_ssh(self) -> int:
        """Cap on concurrent SSH/rsync fan-out workers.

        Bounds the thread-pool size used by the parallel orchestration
        helpers (``run_remote_scripts_parallel``, ``run_rsync_parallel``,
        ``run_pipeline_to_remotes_parallel``, cleanup, status queries).
        Defaults to :data:`DEFAULT_MAX_PARALLEL_SSH` (20) — chosen to stay
        under sshd's ``MaxStartups`` default (10 unauthenticated, but
        authenticated sessions are not throttled the same way) while still
        fanning out widely.  Set via ``ssh.max_parallel_ssh`` in
        ``config.yaml``.  Values ``<= 0`` fall back to the default.
        """
        from sparkrun.orchestration.ssh import DEFAULT_MAX_PARALLEL_SSH

        ssh = self._data.get("ssh", {})
        raw = ssh.get("max_parallel_ssh") if isinstance(ssh, dict) else None
        try:
            val = int(raw)
        except (TypeError, ValueError):
            return DEFAULT_MAX_PARALLEL_SSH
        return val if val > 0 else DEFAULT_MAX_PARALLEL_SSH

    @property
    def jobs_autoprune(self) -> bool:
        """Whether ``sparkrun run`` prunes stale job metadata as it launches.

        The job metadata cache is append-only — only an explicit ``stop``
        removes an entry — so crashed jobs accumulate indefinitely and make
        the cache useless as a completion source.  ``run`` already holds a
        live status snapshot (``api.plan`` takes exactly one), so it can prune
        safely and for free: nothing currently running is ever touched.

        Set ``jobs.autoprune: false`` in ``config.yaml`` to keep every job
        metadata file forever and prune only via
        ``sparkrun setup prune-job-metadata-cache``.
        """
        from scitrera_app_framework import ext_parse_bool

        jobs = self._data.get("jobs", {})
        raw = jobs.get("autoprune") if isinstance(jobs, dict) else None
        if raw is None:
            return True
        parsed = ext_parse_bool(raw)
        # `ext_parse_bool` returns None for anything it doesn't recognise;
        # an unparseable value must not silently disable pruning.
        return True if parsed is None else parsed

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dot-separated key path."""
        parts = key.split(".")
        current = self._data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set(self, key: str, value: Any) -> None:
        """Set a config value by dot-separated key path."""
        parts = key.split(".")
        current = self._data
        for part in parts[:-1]:
            next_value = current.get(part)
            if not isinstance(next_value, dict):
                next_value = {}
                current[part] = next_value
            current = next_value
        current[parts[-1]] = value

    def save(self) -> None:
        """Persist the current config data to disk."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.safe_dump(self._data, f, default_flow_style=False, sort_keys=False)

    @property
    def feature_channel(self) -> str:
        """Active release channel for feature-flag defaults (normalized).

        Reads ``features.channel`` when set, otherwise falls back to the
        persisted self-update channel. Lets a ``stable`` install preview an
        entire channel's feature set (``features.channel: alpha``) without
        changing which code it actually runs.
        """
        from sparkrun.core.channels import normalize_channel

        features = self._data.get("features", {})
        raw = features.get("channel") if isinstance(features, dict) else None
        if raw:
            return normalize_channel(raw)
        return self.self_update_channel

    def feature_override(self, name: str) -> bool | None:
        """Return the explicit ``features.<name>`` override, or ``None`` when unset.

        The ``features.channel`` key is reserved for :attr:`feature_channel`
        and is never treated as a flag override.
        """
        if name == "channel":
            return None
        features = self._data.get("features", {})
        if not isinstance(features, dict):
            return None
        if name not in features:
            return None
        from scitrera_app_framework import ext_parse_bool

        return bool(ext_parse_bool(features[name]))

    def is_feature_enabled(self, name: str) -> bool:
        """Resolve whether feature *name* is enabled for this config.

        Thin wrapper over :func:`sparkrun.core.features.is_feature_enabled`
        that binds this config (and thus its channel + overrides).
        """
        from sparkrun.core.features import is_feature_enabled

        return is_feature_enabled(name, config=self)

    @property
    def self_update_channel(self) -> str:
        """Return the persisted update channel, normalized (default ``stable``)."""
        from sparkrun.core.channels import normalize_channel

        return normalize_channel(self.get("self_update.channel"))

    def set_self_update_channel(self, channel: str) -> None:
        """Persist the update channel (normalized) plus its source and requirement."""
        from sparkrun.core.channels import channel_requirement, is_git_channel, normalize_channel

        canonical = normalize_channel(channel)
        self.set("self_update.channel", canonical)
        self.set("self_update.source", "git" if is_git_channel(canonical) else "pypi")
        self.set("self_update.requirement", channel_requirement(canonical))
        self.save()

    def _get_defaults_section(self, section: str, name: str) -> dict[str, Any]:
        """Return ``defaults.<section>.<name>`` as a dict, or ``{}`` when missing or malformed."""
        defaults = self._data.get("defaults", {})
        if not isinstance(defaults, dict):
            return {}
        bucket = defaults.get(section, {})
        if not isinstance(bucket, dict):
            return {}
        entry = bucket.get(name, {})
        return entry if isinstance(entry, dict) else {}

    def get_defaults_builder(self, name: str) -> dict[str, Any]:
        """Return per-builder defaults from ``defaults.builders.<name>``.

        Example user config::

            defaults:
              builders:
                eugr:
                  use_sentinel_image: false

        Builders should treat the returned dict as a soft default — recipe
        fields and explicit overrides still win.
        """
        return self._get_defaults_section("builders", name)

    def get_defaults_runtime(self, name: str) -> dict[str, Any]:
        """Return per-runtime defaults from ``defaults.runtimes.<name>``.

        Example user config::

            defaults:
              runtimes:
                vllm-distributed:
                  some_option: value

        Runtimes should treat the returned dict as a soft default — recipe
        fields and explicit overrides still win.
        """
        return self._get_defaults_section("runtimes", name)

    @property
    def monitor_backend(self) -> str | None:
        """Monitoring backend preference: ``"bash"`` or ``"nv-monitor"``."""
        return self._data.get("monitor_backend")

    @property
    def vllm_tune_repo(self) -> str:
        """Git URL for the vllm-tune backing engine used by ``sparkrun tune vllm``."""
        tuning = self._data.get("tuning", {})
        if isinstance(tuning, dict):
            url = tuning.get("vllm_tune_repo")
            if url:
                return str(url)
        return DEFAULT_VLLM_TUNE_REPO

    @property
    def vllm_tune_ref(self) -> str:
        """Git ref (tag/branch/SHA) pinning the vllm-tune backing engine."""
        tuning = self._data.get("tuning", {})
        if isinstance(tuning, dict):
            ref = tuning.get("vllm_tune_ref")
            if ref:
                return str(ref)
        return DEFAULT_VLLM_TUNE_REF

    @property
    def external_plugin_paths(self) -> list[Path]:
        """Directories to load out-of-tree plugins from (``plugins.paths``).

        Each entry is a directory prepended to ``sys.path`` at startup; every
        importable top-level module/package inside it is imported, scanned for
        sparkrun plugin base classes (runtimes, executors, transports, …), and
        given the chance to self-register via a module-level ``register(v)``
        hook.  Empty list (the default) disables external plugin loading, so a
        stock install pays zero cost and exposes zero extra surface.  Because
        the config file and these directories are user-owned, loading them is
        trusted by definition — the same model as a pip-installed package.
        """
        plugins = self._data.get("plugins", {})
        if not isinstance(plugins, dict):
            return []
        raw = plugins.get("paths", [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            return []
        return [Path(os.path.expanduser(str(entry))) for entry in raw if entry]

    def get_recipe_search_paths(self) -> list[Path]:
        """Return ordered list of paths to search for recipes."""
        paths = []
        # 1. Current directory recipes/
        cwd_recipes = Path.cwd() / "recipes"
        if cwd_recipes.is_dir():
            paths.append(cwd_recipes)
        # 2. User config recipes/
        user_recipes = DEFAULT_CONFIG_DIR / "recipes"
        if user_recipes.is_dir():
            paths.append(user_recipes)
        # 3. Extra search paths from config
        for extra in self._data.get("recipe_paths", []):
            p = Path(os.path.expanduser(extra))
            if p.is_dir():
                paths.append(p)
        return paths

    def get_registry_manager(self) -> "RegistryManager":
        """Create a RegistryManager using the config root and cache dir."""
        from sparkrun.core.registry import RegistryManager

        return RegistryManager(
            config_root=self.config_path.parent if self.config_path else DEFAULT_CONFIG_DIR,
            cache_root=self.cache_dir / "registries",
        )

    def get_proxy_config(self) -> "ProxyConfig":
        """Return a cached :class:`ProxyConfig` for this config object.

        The canonical access path is :attr:`SparkrunContext.proxy_config`;
        callers that don't have a ``sctx`` (scripts, tests, internal
        helpers) can reach the same instance via this factory.  The
        instance is constructed lazily on first call and reused on
        subsequent calls — mirroring :meth:`get_registry_manager`.
        """
        if self._proxy_config is None:
            from sparkrun.proxy.config import ProxyConfig

            self._proxy_config = ProxyConfig()
        return self._proxy_config
