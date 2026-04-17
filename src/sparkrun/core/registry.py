"""Git-based recipe registry system for sparkrun.

This module provides a registry manager that tracks and syncs recipe collections
from remote git repositories using sparse checkouts for efficiency.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from vpd.next.util import read_yaml

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Exception raised for registry-specific errors."""

    pass


@dataclass
class RegistryEntry:
    """Represents a recipe registry source.

    Attributes:
        name: Unique identifier for the registry
        url: Git repository URL
        subpath: Path within the repository containing recipes
        description: Human-readable description
        enabled: Whether this registry is active
        visible: If False, recipes hidden from default listings
        tuning_subpath: Path within repo for tuning configs
        benchmark_subpath: Path within repo for benchmark profiles
    """

    name: str
    url: str
    subpath: str
    description: str = ""
    enabled: bool = True
    visible: bool = True
    tuning_subpath: str = ""
    benchmark_subpath: str = ""


FALLBACK_DEFAULT_REGISTRIES = [
    RegistryEntry(
        name="sparkrun-testing",
        url="https://github.com/dbotwinick/sparkrun-recipe-registry.git",
        subpath="testing/recipes",
        description="Sparkrun testing registry for recipes, tuning configs, and benchmark profiles",
        tuning_subpath="testing/tuning",
        benchmark_subpath="testing/benchmarking",
        visible=False,
    ),
    RegistryEntry(
        name="official",
        url="https://github.com/spark-arena/recipe-registry.git",
        subpath="official-recipes",
        description="Official Spark Arena registry for recipes, tuning configs, and benchmark profiles",
        tuning_subpath="tuning",
        benchmark_subpath="benchmarking",
        visible=True,
    ),
    RegistryEntry(
        name="eugr",
        url="https://github.com/eugr/spark-vllm-docker",
        subpath="recipes",
        description="Official eugr/spark-vllm-docker repo recipes",
        visible=True,
    ),
    RegistryEntry(
        name="sparkrun-transitional",
        url="https://github.com/dbotwinick/sparkrun-recipe-registry.git",
        subpath="transitional/recipes",
        description="Transitional registry for recipes",
        tuning_subpath="testing/tuning",
        visible=True,
    ),
    RegistryEntry(
        name="experimental",
        url="https://github.com/spark-arena/recipe-registry.git",
        subpath="experimental-recipes",
        description="Spark Arena registry for experimental recipes",
        visible=False,
    ),
    RegistryEntry(
        name="community",
        url="https://github.com/spark-arena/community-recipe-registry.git",
        subpath="recipes",
        description="Community recipe registry",
        visible=False,
    ),
]

# Git URLs whose .sparkrun/registry.yaml manifests are used for first-run
# registry discovery (see RegistryManager._init_defaults_from_manifests).
DEFAULT_REGISTRIES_GIT = [
    "https://github.com/dbotwinick/sparkrun-recipe-registry.git",
    "https://github.com/spark-arena/recipe-registry.git",
    "https://github.com/spark-arena/community-recipe-registry.git",
]

# List of git URLs for registries that have been superseded and should be cleaned up.
# Comparison strips trailing .git from entry URLs before matching.
DEPRECATED_REGISTRIES: list[str] = [
    "https://github.com/scitrera/oss-spark-run",
    # 'https://github.com/eugr/spark-vllm-docker',
]

# Reserved name prefixes — only URLs from allowed GitHub orgs may use these.
# This prevents third-party registries from impersonating official sources.
RESERVED_NAME_PREFIXES = (
    "arena",
    "spark-arena",
    "sparkarena",
    "sparkrun",
    "official",
    "experimental",
    "transitional",
    "community",
    "eugr",
    "dbotwinick",
    "raphaelamorim",
    "scitrera",
)

RESERVED_PREFIX_ALLOWED_ORGS = (
    "spark-arena",
    "scitrera",
    "eugr",
    "dbotwinick",
    "raphaelamorim",
)


def validate_registry_name(name: str, url: str) -> None:
    """Raise RegistryError if name uses a reserved prefix from a non-allowed URL.

    Reserved prefixes protect official registry namespaces. Only repositories
    hosted under allowed GitHub organizations may use these prefixes.

    Args:
        name: Registry name to validate.
        url: Git repository URL associated with the registry.

    Raises:
        RegistryError: If the name uses a reserved prefix and the URL is not
            from an allowed GitHub organization.
    """
    name_lower = name.lower()
    matched_prefix = None
    for prefix in RESERVED_NAME_PREFIXES:
        if name_lower.startswith(prefix):
            matched_prefix = prefix
            break

    if matched_prefix is None:
        return

    # Extract GitHub org from URL
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.hostname and parsed.hostname.lower() in ("github.com", "www.github.com"):
            # Path is like /org/repo or /org/repo.git
            parts = parsed.path.strip("/").split("/")
            if parts:
                org = parts[0].lower()
                if org in RESERVED_PREFIX_ALLOWED_ORGS:
                    return
    except Exception:
        pass

    allowed = ", ".join(RESERVED_PREFIX_ALLOWED_ORGS)
    raise RegistryError(
        "Registry name %r uses reserved prefix %r. Only GitHub organizations [%s] may use this prefix." % (name, matched_prefix, allowed)
    )


class RegistryManager:
    """Manages recipe registries with git-based syncing.

    The manager tracks registry configurations, handles shallow git clones
    with sparse checkouts, and provides recipe discovery across all registries.
    """

    def __init__(self, config_root: Path, cache_root: Path | None = None) -> None:
        """Initialize the registry manager.

        Args:
            config_root: Directory containing registries.yaml
            cache_root: Optional cache directory, defaults to ~/.cache/sparkrun/registries
        """
        self.config_root = Path(config_root)
        self.cache_root = Path(cache_root) if cache_root else Path.home() / ".cache/sparkrun/registries"
        self.config_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._manifest_discovery_attempted = False

    @property
    def _registries_path(self) -> Path:
        """Path to the registries configuration file."""
        return self.config_root / "registries.yaml"

    def _cache_dir(self, name: str) -> Path:
        """Get the cache directory for a specific registry.

        Args:
            name: Registry name

        Returns:
            Path to the cache directory
        """
        return self.cache_root / name

    def _recipe_dir(self, entry: RegistryEntry) -> Path | None:
        """Get the recipe directory within a cached registry.

        Args:
            entry: Registry entry

        Returns:
            Path to the recipe directory, or None if not cached
        """
        cache_dir = self._cache_dir(entry.name)
        recipe_path = cache_dir / entry.subpath
        return recipe_path if recipe_path.exists() else None

    def _default_registries(self) -> list[RegistryEntry]:
        """Return the default registry list.

        On first run (no ``registries.yaml``), attempts manifest-based
        discovery from ``DEFAULT_REGISTRIES_GIT``.  Discovered manifest
        entries take priority; ``FALLBACK_DEFAULT_REGISTRIES`` entries are
        then layered on for any names not already present.  This lets git
        manifests override/refresh entries while hardcoded fallbacks fill
        gaps (e.g. when a manifest URL is unreachable).

        When manifest entries are discovered, the combined list is persisted
        to ``registries.yaml`` so subsequent loads read from file.

        Manifest discovery is attempted at most once per ``RegistryManager``
        instance to avoid repeated slow network calls.
        """
        discovered: list[RegistryEntry] = []
        if not self._manifest_discovery_attempted:
            self._manifest_discovery_attempted = True
            discovered = self._init_defaults_from_manifests()

        # Layer fallback entries whose names don't collide with manifest entries
        seen_names = {e.name for e in discovered}
        combined = list(discovered)
        for fallback in FALLBACK_DEFAULT_REGISTRIES:
            if fallback.name not in seen_names:
                combined.append(fallback)
                seen_names.add(fallback.name)

        # Persist so subsequent _load_registries() reads from file
        if discovered:
            self._save_registries(combined)

        return combined

    def _init_defaults_from_manifests(self) -> list[RegistryEntry]:
        """Try to discover default registries from git manifest files.

        For each URL in ``DEFAULT_REGISTRIES_GIT``, clones the repo and reads
        its ``.sparkrun/registry.yaml`` manifest.  Entries are collected,
        deduplicated by name, and validated.

        URLs that fail to clone are skipped individually — successful URLs
        still contribute their entries (partial success).  Only if ALL URLs
        fail does this return ``[]``.

        This method does **not** save to ``registries.yaml``; the caller
        (:meth:`_default_registries`) handles persistence after layering
        fallback entries.

        This method bypasses :meth:`add_registry` to avoid a re-entrancy bug
        where ``add_registry`` → ``_load_registries`` → ``_default_registries``
        would see the ``_manifest_discovery_attempted`` flag already set and
        fall back to ``FALLBACK_DEFAULT_REGISTRIES``.
        """
        all_entries: list[RegistryEntry] = []
        seen_names: set[str] = set()

        for url in DEFAULT_REGISTRIES_GIT:
            try:
                entries = self._discover_manifest_entries(url)
                for entry in entries:
                    if entry.name in seen_names:
                        logger.debug("Skipping duplicate manifest entry %r", entry.name)
                        continue
                    validate_registry_name(entry.name, entry.url)
                    seen_names.add(entry.name)
                    all_entries.append(entry)
            except Exception as e:
                logger.warning("Manifest discovery failed for %s: %s", url, e)
                # Continue to next URL instead of aborting entirely

        return all_entries

    def _load_registries_from_file(self) -> list[RegistryEntry]:
        """Load registries from the YAML config file without any fallback logic.

        Returns:
            List of registry entries parsed from registries.yaml.

        Raises:
            Exception: If the file cannot be read or parsed.
        """
        data = read_yaml(self._registries_path)
        if not isinstance(data, dict):
            return []
        registries = data.get("registries", [])
        return [
            RegistryEntry(
                name=r["name"],
                url=r["url"],
                subpath=r["subpath"],
                description=r.get("description", ""),
                enabled=r.get("enabled", True),
                visible=r.get("visible", True),
                tuning_subpath=r.get("tuning_subpath", ""),
                benchmark_subpath=r.get("benchmark_subpath", ""),
            )
            for r in registries
        ]

    def _load_registries(self) -> list[RegistryEntry]:
        """Load registries from YAML configuration.

        Returns:
            List of registry entries, or default registries if config doesn't exist
        """
        if not self._registries_path.exists():
            logger.debug("No registries.yaml found, using defaults")
            return self._default_registries()

        try:
            entries = self._load_registries_from_file()
            # Filter out any entries whose URL matches a deprecated registry
            filtered = []
            for entry in entries:
                if self._is_deprecated_url(entry.url):
                    logger.warning(
                        "Filtering deprecated registry %r (url: %s) from config",
                        entry.name,
                        entry.url,
                    )
                else:
                    filtered.append(entry)
            return filtered
        except Exception as e:
            logger.warning("Failed to load registries.yaml: %s", e)
            return self._default_registries()

    def _save_registries(self, entries: list[RegistryEntry]) -> None:
        """Save registries to YAML configuration.

        Args:
            entries: List of registry entries to save
        """
        data_list = []
        for e in entries:
            d: dict[str, Any] = {"name": e.name, "url": e.url, "subpath": e.subpath}
            if e.description:
                d["description"] = e.description
            if not e.enabled:
                d["enabled"] = False
            if not e.visible:
                d["visible"] = False
            if e.tuning_subpath:
                d["tuning_subpath"] = e.tuning_subpath
            if e.benchmark_subpath:
                d["benchmark_subpath"] = e.benchmark_subpath
            data_list.append(d)

        data = {"registries": data_list}
        with open(self._registries_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.debug("Saved registries to %s", self._registries_path)

    @staticmethod
    def _git_env() -> dict[str, str]:
        """Return environment variables for non-interactive git operations."""
        import os

        env = os.environ.copy()
        # Prevent git from prompting for credentials — fail immediately instead
        env["GIT_TERMINAL_PROMPT"] = "0"
        return env

    def _clone_dir_for_url(self, url: str) -> Path:
        """Return a deterministic cache directory for a given git URL.

        Uses a hash of the URL to create a shared clone location.
        """
        import hashlib

        url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
        return self.cache_root / ("_url_%s" % url_hash)

    @staticmethod
    def _build_sparse_paths(entry: RegistryEntry) -> list[str]:
        """Build the sparse-checkout path list for a single registry entry.

        Always includes the recipe subpath and ``.sparkrun`` (for manifests).
        Tuning and benchmark subpaths are added when configured.
        """
        paths = [entry.subpath]
        if entry.tuning_subpath:
            paths.append(entry.tuning_subpath)
        if entry.benchmark_subpath:
            paths.append(entry.benchmark_subpath)
        paths.append(".sparkrun")
        return paths

    def _sparse_checkout_paths_for_url(self, url: str) -> list[str]:
        """Collect all subpaths that need to be checked out for a given URL.

        Returns the union of subpath, tuning_subpath, and benchmark_subpath
        for all enabled registries pointing to the given URL.
        """
        paths: set[str] = set()
        for entry in self._load_registries():
            if entry.url == url and entry.enabled:
                paths.update(self._build_sparse_paths(entry))
        return sorted(paths)

    def _sync_url(self, url: str, progress: Callable[[str, bool], None] | None = None) -> bool:
        """Clone or pull a shared checkout for a URL, then update sparse paths.

        Returns True on success, False on failure.
        """
        clone_dir = self._clone_dir_for_url(url)
        sparse_paths = self._sparse_checkout_paths_for_url(url)
        git_env = self._git_env()

        try:
            if (clone_dir / ".git").exists():
                # Fetch + hard reset to ensure deleted files are removed
                # and rebased histories are handled correctly
                result = subprocess.run(
                    ["git", "-C", str(clone_dir), "fetch", "origin"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    env=git_env,
                )
                if result.returncode != 0:
                    logger.warning("git fetch failed for %s: %s", url, result.stderr.strip())
                    return False
                result = subprocess.run(
                    ["git", "-C", str(clone_dir), "reset", "--hard", "FETCH_HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    env=git_env,
                )
                if result.returncode != 0:
                    logger.warning("git reset failed for %s: %s", url, result.stderr.strip())
                    return False
            else:
                # Fresh sparse clone
                clone_dir.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(
                    ["git", "clone", "--filter=blob:none", "--sparse", str(url), str(clone_dir)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    env=git_env,
                )
                if result.returncode != 0:
                    logger.warning("git clone failed for %s: %s", url, result.stderr.strip())
                    return False

            # Update sparse-checkout paths
            if sparse_paths:
                result = subprocess.run(
                    ["git", "-C", str(clone_dir), "sparse-checkout", "set"] + sparse_paths,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    env=git_env,
                )
                if result.returncode != 0:
                    logger.warning("sparse-checkout set failed for %s: %s", url, result.stderr.strip())

            return True
        except subprocess.TimeoutExpired:
            logger.warning("Git operation timed out for %s", url)
            return False

    def _link_registry_to_shared(self, entry: RegistryEntry) -> None:
        """Create/update symlink from per-registry cache dir to shared clone subpath."""
        shared_dir = self._clone_dir_for_url(entry.url)
        per_registry_dir = self._cache_dir(entry.name)

        # Remove old per-registry dir if it's a real directory (not a symlink)
        if per_registry_dir.exists() and not per_registry_dir.is_symlink():
            import shutil

            shutil.rmtree(per_registry_dir)

        # Create symlink: per_registry_dir -> shared_dir
        per_registry_dir.parent.mkdir(parents=True, exist_ok=True)
        if per_registry_dir.is_symlink():
            per_registry_dir.unlink()
        per_registry_dir.symlink_to(shared_dir)

    def _clone_or_pull_single(self, entry: RegistryEntry) -> bool:
        """Clone or update a registry repository (single-URL implementation).

        Uses shallow clone with sparse checkout for efficiency. Git command
        failures are logged but not raised (best-effort sync).

        Args:
            entry: Registry entry to sync

        Returns:
            True if the operation succeeded, False otherwise.
        """
        cache_dir = self._cache_dir(entry.name)
        git_env = self._git_env()

        try:
            if (cache_dir / ".git").exists():
                # Update existing repository
                logger.debug("Updating registry %s", entry.name)

                # Ensure sparse checkout covers all configured subpaths
                # (picks up tuning_subpath / benchmark_subpath added after
                # the initial clone)
                sparse_paths = self._build_sparse_paths(entry)
                subprocess.run(
                    ["git", "-C", str(cache_dir), "sparse-checkout", "set"] + sparse_paths,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    env=git_env,
                )

                # Fetch + hard reset to ensure deleted files are removed
                # and rebased histories are handled correctly
                result = subprocess.run(
                    ["git", "-C", str(cache_dir), "fetch", "--depth=1", "origin"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    env=git_env,
                )
                if result.returncode != 0:
                    logger.debug("Git fetch failed for %s: %s", entry.name, result.stderr)
                    return False
                result = subprocess.run(
                    ["git", "-C", str(cache_dir), "reset", "--hard", "FETCH_HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    env=git_env,
                )
                if result.returncode != 0:
                    logger.debug("Git reset failed for %s: %s", entry.name, result.stderr)
                    return False
            else:
                # Fresh clone with sparse checkout
                logger.debug("Cloning registry %s", entry.name)
                cache_dir.mkdir(parents=True, exist_ok=True)

                # Shallow clone with blob filtering
                result = subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "--filter=blob:none",
                        "--sparse",
                        entry.url,
                        str(cache_dir),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    env=git_env,
                )
                if result.returncode != 0:
                    logger.debug("Git clone failed for %s: %s", entry.name, result.stderr)
                    return False

                # Configure sparse checkout for all subpaths
                sparse_paths = self._build_sparse_paths(entry)
                result = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(cache_dir),
                        "sparse-checkout",
                        "set",
                    ]
                    + sparse_paths,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    env=git_env,
                )
                if result.returncode != 0:
                    logger.debug(
                        "Sparse checkout setup failed for %s: %s",
                        entry.name,
                        result.stderr,
                    )
                    return False
        except subprocess.TimeoutExpired:
            logger.debug("Git operation timed out for %s", entry.name)
            return False
        except Exception as e:
            logger.debug("Failed to sync registry %s: %s", entry.name, e)
            return False

        return True

    def _clone_or_pull(self, entry: RegistryEntry) -> bool:
        """Clone or update a registry, using shared clones for same-URL registries."""
        # Check if any other registries share this URL
        all_entries = self._load_registries()
        same_url_entries = [e for e in all_entries if e.url == entry.url and e.enabled]

        if len(same_url_entries) > 1:
            # Use shared clone
            success = self._sync_url(entry.url)
            if success:
                self._link_registry_to_shared(entry)
            return success
        else:
            # Single registry for this URL — use original clone behavior
            return self._clone_or_pull_single(entry)

    def add_registry(self, entry: RegistryEntry) -> None:
        """Add a new registry.

        Args:
            entry: Registry entry to add

        Raises:
            RegistryError: If a registry with the same name already exists,
                or uses a reserved name prefix from a non-allowed URL.
        """
        validate_registry_name(entry.name, entry.url)
        registries = self._load_registries()
        if any(r.name == entry.name for r in registries):
            raise RegistryError(f"Registry {entry.name!r} already exists")
        registries.append(entry)
        self._save_registries(registries)
        logger.info("Added registry %s", entry.name)

    def _discover_manifest_entries(self, url: str) -> list[RegistryEntry]:
        """Clone a repo, read its .sparkrun/registry.yaml manifest, return entries.

        Does NOT save or add entries — purely a discovery/parsing operation.

        Args:
            url: Git repository URL.

        Returns:
            List of RegistryEntry objects declared in the manifest.

        Raises:
            RegistryError: If clone fails, no manifest found, or manifest is empty.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "repo"
            git_env = self._git_env()
            result = subprocess.run(
                ["git", "clone", "--depth=1", "--single-branch", str(url), str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                stdin=subprocess.DEVNULL,
                env=git_env,
            )
            if result.returncode != 0:
                raise RegistryError("Failed to clone %s: %s" % (url, result.stderr.strip()))

            manifest_path = tmp_path / ".sparkrun" / "registry.yaml"
            if not manifest_path.exists():
                raise RegistryError("No .sparkrun/registry.yaml manifest found in %s" % url)

            manifest = yaml.safe_load(manifest_path.read_text()) or {}
            registries_data = manifest.get("registries", [])
            if not registries_data:
                raise RegistryError("Manifest in %s declares no registries" % url)

            # Support both canonical keys (subpath, tuning_subpath,
            # benchmark_subpath) used in registries.yaml and the shorter
            # keys (recipes, tuning, benchmarks) used in repo manifests.
            return [
                RegistryEntry(
                    name=reg_data["name"],
                    url=url,
                    subpath=reg_data.get("subpath", reg_data.get("recipes", "recipes")),
                    description=reg_data.get("description", ""),
                    enabled=reg_data.get("enabled", True),
                    visible=reg_data.get("visible", True),
                    tuning_subpath=reg_data.get("tuning_subpath", reg_data.get("tuning", "")),
                    benchmark_subpath=reg_data.get("benchmark_subpath", reg_data.get("benchmarks", "")),
                )
                for reg_data in registries_data
            ]

    def add_registry_from_url(self, url: str) -> list[RegistryEntry]:
        """Add registries by discovering them from a repo's .sparkrun/registry.yaml manifest.

        Clones the repo temporarily, reads the manifest, and adds all declared registries.

        Args:
            url: Git repository URL.

        Returns:
            List of RegistryEntry objects added.

        Raises:
            RegistryError: If clone fails or no manifest found.
        """
        entries = self._discover_manifest_entries(url)
        added = []
        for entry in entries:
            validate_registry_name(entry.name, entry.url)
            try:
                self.add_registry(entry)
                added.append(entry)
                logger.info("Added registry '%s' from manifest", entry.name)
            except RegistryError:
                logger.warning("Registry '%s' already exists, skipping", entry.name)
        return added

    def remove_registry(self, name: str) -> None:
        """Remove a registry by name.

        Args:
            name: Registry name to remove

        Raises:
            RegistryError: If the registry is not found
        """
        registries = self._load_registries()
        filtered = [r for r in registries if r.name != name]
        if len(filtered) == len(registries):
            raise RegistryError(f"Registry {name!r} not found")
        self._save_registries(filtered)
        logger.info("Removed registry %s", name)

    @staticmethod
    def _is_deprecated_url(url: str) -> bool:
        """Check whether a registry URL matches a deprecated entry.

        Strips trailing ``.git`` from the URL before comparison so that
        ``https://github.com/org/repo.git`` matches ``https://github.com/org/repo``.
        """
        normalized = url.rstrip("/")
        if normalized.endswith(".git"):
            normalized = normalized[:-4]
        for dep_url in DEPRECATED_REGISTRIES:
            dep_normalized = dep_url.rstrip("/")
            if dep_normalized.endswith(".git"):
                dep_normalized = dep_normalized[:-4]
            if normalized == dep_normalized:
                return True
        return False

    def restore_missing_defaults(self) -> list[str]:
        """Add default registry entries that are missing from the config.

        Checks ``FALLBACK_DEFAULT_REGISTRIES`` for entries whose name is not
        present in the current ``registries.yaml``.  Missing entries are
        appended and persisted.

        Returns:
            List of registry names that were added.
        """
        try:
            entries = self._load_registries_from_file()
        except Exception:
            entries = self._load_registries()

        existing_names = {e.name for e in entries}
        added: list[str] = []

        for default in FALLBACK_DEFAULT_REGISTRIES:
            if default.name not in existing_names:
                entries.append(default)
                added.append(default.name)
                logger.info("Restored missing default registry: %s", default.name)

        if added:
            self._save_registries(entries)

        return added

    def cleanup_deprecated(self) -> list[str]:
        """Remove deprecated registries and their caches.

        Matches on the registry URL (not the name) against
        ``DEPRECATED_REGISTRIES``.

        Returns list of registry names that were cleaned up.
        """
        if not DEPRECATED_REGISTRIES:
            return []

        try:
            entries = self._load_registries_from_file()
        except Exception:
            entries = self._load_registries()
        cleaned = []
        remaining = []

        deprecated_urls: set[str] = set()
        for entry in entries:
            if self._is_deprecated_url(entry.url):
                deprecated_urls.add(entry.url)
                # Remove per-registry cache (symlink or directory)
                cache_dir = self._cache_dir(entry.name)
                if cache_dir.exists():
                    import shutil

                    if cache_dir.is_symlink():
                        cache_dir.unlink()
                    else:
                        shutil.rmtree(cache_dir)
                cleaned.append(entry.name)
                logger.info("Removed deprecated registry: %s", entry.name)
            else:
                remaining.append(entry)

        if cleaned:
            self._save_registries(remaining)

            # Clean up orphaned shared clones: if no remaining registry
            # references a deprecated URL, remove the shared _url_* dir.
            import shutil

            remaining_urls = {e.url for e in remaining}
            for dep_url in deprecated_urls:
                if dep_url not in remaining_urls:
                    shared_dir = self._clone_dir_for_url(dep_url)
                    if shared_dir.exists():
                        shutil.rmtree(shared_dir)
                        logger.info("Removed orphaned shared clone: %s", shared_dir.name)

        return cleaned

    def clear_cache(self) -> int:
        """Remove all cached registry clones for a clean slate.

        Removes per-registry symlinks and shared ``_url_*`` clone
        directories from :attr:`cache_root`.

        Returns:
            Number of cache entries removed.
        """
        import shutil

        count = 0
        if self.cache_root.exists():
            for child in self.cache_root.iterdir():
                if child.is_symlink():
                    child.unlink()
                    count += 1
                elif child.is_dir():
                    shutil.rmtree(child)
                    count += 1
        logger.debug("Cleared %d cache entries from %s", count, self.cache_root)
        return count

    def reset_to_defaults(self) -> list[RegistryEntry]:
        """Delete the registries config, clear cache, and re-initialize from defaults.

        Removes ``registries.yaml`` (if it exists), clears all cached git
        clones, resets the manifest discovery flag, and re-runs the default
        initialization path (manifest discovery first, then hardcoded
        fallback).  The resulting registries are saved to
        ``registries.yaml`` and returned.

        Returns:
            The new list of registry entries.
        """
        if self._registries_path.exists():
            self._registries_path.unlink()
            logger.info("Removed existing registries.yaml")

        # Clear all cached clones so the subsequent update does fresh clones
        cleared = self.clear_cache()
        if cleared:
            logger.info("Cleared %d cached registry clones", cleared)

        # Allow manifest discovery to run again
        self._manifest_discovery_attempted = False

        entries = self._default_registries()
        self._save_registries(entries)
        logger.info("Reset registries to defaults (%d entries)", len(entries))
        return entries

    def _set_registry_enabled(self, name: str, enabled: bool) -> None:
        """Set the enabled state of a registry by name.

        Args:
            name: Registry name to modify.
            enabled: Target enabled state.

        Raises:
            RegistryError: If the registry is not found.
        """
        entries = self._load_registries()
        for e in entries:
            if e.name == name:
                e.enabled = enabled
                self._save_registries(entries)
                logger.info("%s registry %s", "Enabled" if enabled else "Disabled", name)
                return
        raise RegistryError("Registry %r not found" % name)

    def enable_registry(self, name: str) -> None:
        """Enable a registry by name.

        Raises:
            RegistryError: If the registry is not found
        """
        self._set_registry_enabled(name, True)

    def disable_registry(self, name: str) -> None:
        """Disable a registry by name.

        Raises:
            RegistryError: If the registry is not found
        """
        self._set_registry_enabled(name, False)

    def list_registries(self) -> list[RegistryEntry]:
        """List all configured registries.

        Returns:
            List of all registry entries
        """
        return self._load_registries()

    def get_registry(self, name: str) -> RegistryEntry:
        """Get a single registry by name.

        Args:
            name: Registry name

        Returns:
            Registry entry

        Raises:
            RegistryError: If the registry is not found
        """
        registries = self._load_registries()
        for entry in registries:
            if entry.name == name:
                return entry
        raise RegistryError(f"Registry {name!r} not found")

    def update(
        self,
        name: str | None = None,
        progress: Callable[[str, bool], None] | None = None,
    ) -> dict[str, bool]:
        """Update one or all registries.

        Performs shallow clone or pull for specified registry or all enabled
        registries if name is None.

        Args:
            name: Optional registry name to update, or None for all.
            progress: Optional callback invoked after each registry with
                ``(registry_name, success)``.

        Returns:
            Mapping of registry name to success status for each registry
            that was attempted.
        """
        registries = self._load_registries()
        results: dict[str, bool] = {}

        if name is not None:
            # Update single registry
            entry = self.get_registry(name)
            if entry.enabled:
                ok = self._clone_or_pull(entry)
                results[entry.name] = ok
                if progress:
                    progress(entry.name, ok)
            else:
                logger.warning("Registry %s is disabled, skipping update", name)
                results[entry.name] = False
                if progress:
                    progress(entry.name, False)
        else:
            # Update all enabled registries
            for entry in registries:
                if entry.enabled:
                    ok = self._clone_or_pull(entry)
                    results[entry.name] = ok
                    if progress:
                        progress(entry.name, ok)

        return results

    def ensure_initialized(self) -> None:
        """Ensure registries are initialized.

        If no cache exists, runs update() to perform initial sync.
        """
        registries = self._load_registries()
        needs_init = False

        for entry in registries:
            if entry.enabled:
                cache_dir = self._cache_dir(entry.name)
                if not (cache_dir / ".git").exists():
                    needs_init = True
                    break

        if needs_init:
            logger.debug("Initializing registries")
            self.update()

    def get_recipe_paths(self, include_hidden: bool = False) -> list[Path]:
        """Get all recipe directories from cached registries.

        Args:
            include_hidden: If True, include recipes from invisible registries

        Returns:
            List of paths to recipe directories (only from enabled registries)
        """
        paths = []
        registries = self._load_registries()

        for entry in registries:
            if not entry.enabled:
                continue
            if not include_hidden and not entry.visible:
                continue

            recipe_dir = self._recipe_dir(entry)
            if recipe_dir:
                paths.append(recipe_dir)
            else:
                logger.debug("Registry %s not cached or recipe path not found", entry.name)

        return paths

    def _list_dir_recipes(self, recipe_dir: Path, registry_name: str) -> list[dict[str, Any]]:
        """List all recipes in a directory with metadata.

        Args:
            recipe_dir: Directory to scan for .yaml recipe files.
            registry_name: Name of the registry this directory belongs to.

        Returns:
            List of recipe metadata dicts.
        """
        if not recipe_dir.is_dir():
            return []

        from sparkrun.core.recipe import recipe_summary

        recipes = []
        for f in sorted(recipe_dir.rglob("*.yaml")):
            entry = recipe_summary(f, registry_name=registry_name)
            if entry is not None:
                recipes.append(entry)
        return recipes

    def search_recipes(self, query: str, include_hidden: bool = False) -> list[dict[str, Any]]:
        """Search for recipes across all registries.

        Performs case-insensitive substring matching on recipe name, file stem,
        model, and description fields.

        Args:
            query: Search query string
            include_hidden: If True, include recipes from invisible registries

        Returns:
            List of recipe metadata dicts with 'registry' field added
        """
        results = []
        query_lower = query.lower()
        registries = self._load_registries()

        for entry in registries:
            if not entry.enabled:
                continue
            if not include_hidden and not entry.visible:
                continue
            recipe_dir = self._recipe_dir(entry)
            if recipe_dir is None:
                continue
            for recipe in self._list_dir_recipes(recipe_dir, entry.name):
                searchable = [
                    recipe.get("name", "").lower(),
                    recipe.get("file", "").lower(),
                    recipe.get("model", "").lower(),
                    recipe.get("description", "").lower(),
                ]
                if any(query_lower in s for s in searchable):
                    results.append(recipe)

        return results

    def registry_for_path(self, path: Path) -> str | None:
        """Return the registry name that owns the given path, or None."""
        registries = self._load_registries()
        for entry in registries:
            if not entry.enabled:
                continue
            recipe_dir = self._recipe_dir(entry)
            if recipe_dir and path.is_relative_to(recipe_dir):
                return entry.name
        return None

    def find_recipe_in_registries(self, name: str, include_hidden: bool = False) -> list[tuple[str, Path]]:
        """Find a recipe by file stem across all registries.

        Searches for recipes whose file stem matches the given name.

        Args:
            name: Recipe file stem to find (e.g. 'glm-4.7-flash-awq')
            include_hidden: If True, include recipes from invisible registries

        Returns:
            List of (registry_name, recipe_path) tuples for disambiguation
        """
        matches = []
        registries = self._load_registries()

        for entry in registries:
            if not entry.enabled:
                continue
            if not include_hidden and not entry.visible:
                continue
            recipe_dir = self._recipe_dir(entry)
            if recipe_dir is None:
                continue
            # Flat lookup first (existing behavior)
            for ext in (".yaml", ".yml"):
                candidate = recipe_dir / (name + ext)
                if candidate.exists():
                    matches.append((entry.name, candidate))

        # If flat lookup found nothing, search subdirectories by stem
        if not matches:
            for entry in registries:
                if not entry.enabled:
                    continue
                if not include_hidden and not entry.visible:
                    continue
                recipe_dir = self._recipe_dir(entry)
                if recipe_dir is None:
                    continue
                for ext in (".yaml", ".yml"):
                    for candidate in sorted(recipe_dir.rglob(f"{name}{ext}")):
                        matches.append((entry.name, candidate))

        return matches

    def _tuning_dir(self, entry: RegistryEntry) -> Path | None:
        """Get the tuning directory within a cached registry.

        Args:
            entry: Registry entry

        Returns:
            Path to the tuning directory, or None if not available
        """
        if not entry.tuning_subpath:
            return None
        cache_dir = self._cache_dir(entry.name)
        tuning_path = cache_dir / entry.tuning_subpath
        return tuning_path if tuning_path.exists() else None

    def find_tuning_configs(self, runtime: str, registry_name: str | None = None) -> list[tuple[str, Path]]:
        """Find tuning config files for a given runtime.

        Searches flat layout: ``tuning/<runtime>/.../*.json``.  Configs
        are shape-based (not model-specific), so no model filtering is
        needed — files from different models coexist by filename.

        Args:
            runtime: Runtime name (e.g. "sglang", "vllm")
            registry_name: If provided, only search this registry.
                Otherwise search all enabled registries with tuning.

        Returns:
            List of (registry_name, config_path) tuples
        """
        matches = []
        for entry in self._load_registries():
            if not entry.enabled or not entry.tuning_subpath:
                continue
            if registry_name and entry.name != registry_name:
                continue
            tuning_dir = self._tuning_dir(entry)
            if tuning_dir is None:
                continue

            runtime_dir = tuning_dir / runtime
            if not runtime_dir.is_dir():
                continue

            for f in sorted(runtime_dir.rglob("*.json")):
                matches.append((entry.name, f))

        return matches

    def list_tuning_configs(self) -> list[dict[str, Any]]:
        """List all available tuning configs across registries.

        Returns:
            List of dicts with registry, runtime, file, and path fields.
        """
        configs = []
        for entry in self._load_registries():
            if not entry.enabled or not entry.tuning_subpath:
                continue
            tuning_dir = self._tuning_dir(entry)
            if tuning_dir is None:
                continue

            for runtime_dir in sorted(tuning_dir.iterdir()):
                if not runtime_dir.is_dir():
                    continue
                runtime = runtime_dir.name
                for f in sorted(runtime_dir.rglob("*.json")):
                    configs.append(
                        {
                            "registry": entry.name,
                            "runtime": runtime,
                            "file": f.name,
                            "path": str(f),
                        }
                    )
        return configs

    def _benchmark_dir(self, entry: RegistryEntry) -> Path | None:
        """Get benchmark directory within a cached registry.

        Args:
            entry: Registry entry to look up.

        Returns:
            Path to the benchmark directory, or None if not available
        """
        if not entry.benchmark_subpath:
            return None
        cache_dir = self._cache_dir(entry.name)
        benchmark_path = cache_dir / entry.benchmark_subpath
        return benchmark_path if benchmark_path.exists() else None

    def find_benchmark_profile_in_registries(
        self,
        name: str,
        include_hidden: bool = False,
    ) -> list[tuple[str, Path]]:
        """Find benchmark profile by file stem across registries.

        Args:
            name: Profile file stem (e.g. 'spark-arena-v1')
            include_hidden: If True, include profiles from invisible registries

        Returns:
            List of (registry_name, profile_path) tuples for disambiguation
        """
        matches = []
        for entry in self._load_registries():
            if not entry.enabled:
                continue
            if not include_hidden and not entry.visible:
                continue
            benchmark_dir = self._benchmark_dir(entry)
            if benchmark_dir is None:
                continue
            for ext in (".yaml", ".yml"):
                candidate = benchmark_dir / (name + ext)
                if candidate.exists():
                    matches.append((entry.name, candidate))
        return matches

    def list_benchmark_profiles(
        self,
        registry_name: str | None = None,
        include_hidden: bool = False,
    ) -> list[dict[str, Any]]:
        """List all benchmark profiles across registries.

        Args:
            registry_name: If provided, only list from this registry
            include_hidden: If True, include profiles from invisible registries

        Returns:
            List of dicts with keys: registry, file, name, description, path
        """
        import yaml

        profiles = []
        for entry in self._load_registries():
            if not entry.enabled:
                continue
            if registry_name and entry.name != registry_name:
                continue
            # When a specific registry is requested by name, skip the
            # visibility filter — the user is explicitly targeting it.
            if not registry_name and not include_hidden and not entry.visible:
                continue
            benchmark_dir = self._benchmark_dir(entry)
            if benchmark_dir is None:
                continue
            for f in sorted(benchmark_dir.glob("*.yaml")) + sorted(benchmark_dir.glob("*.yml")):
                # Read metadata from the profile
                profile_name = f.stem
                description = ""
                try:
                    with open(f) as fh:
                        data = yaml.safe_load(fh) or {}
                    profile_name = data.get("name", f.stem)
                    description = data.get("description", "")
                    # Also check metadata.description
                    metadata = data.get("metadata", {})
                    if isinstance(metadata, dict):
                        if not description and metadata.get("description"):
                            description = metadata["description"]
                        if metadata.get("name") and not data.get("name"):
                            profile_name = metadata["name"]
                except Exception:
                    pass
                profiles.append(
                    {
                        "registry": entry.name,
                        "file": f.stem,
                        "name": profile_name,
                        "description": description,
                        "path": str(f),
                    }
                )
        return profiles
