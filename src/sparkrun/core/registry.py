"""Git-based recipe registry system for sparkrun.

This module provides a registry manager that tracks and syncs recipe collections
from remote git repositories using sparse checkouts for efficiency.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, replace as _dataclass_replace
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml

from vpd.next.util import read_yaml

from sparkrun.utils.shell import validate_git_url

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Exception raised for registry-specific errors."""

    pass


class RegistryFilterError(RegistryError):
    """A recipe listing was scoped to a registry that cannot be used.

    Raised by :func:`resolve_registry_filter` for all three ways a registry
    filter can be wrong, discriminated by :attr:`reason`:

    * ``"unknown"`` — no registry by that name is configured.
    * ``"disabled"`` — the registry exists but is disabled.
    * ``"conflict"`` — an ``@registry`` query scope contradicts an
      explicitly-supplied registry filter.

    :attr:`available` lists the configured registry names so callers can
    render a useful message without a second lookup.
    """

    def __init__(self, message: str, *, registry: str, reason: str, available: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.registry = registry
        self.reason = reason
        self.available = available


def resolve_registry_filter(
    query: str | None,
    registry: str | None,
    registry_manager: "RegistryManager",
) -> tuple[str | None, str | None]:
    """Resolve the registry filter for a recipe listing or search.

    Handles the ``@registry`` scope shorthand in *query* — ``@community``
    and ``@community/`` mean "the community registry", and anything after
    the ``/`` is kept as the remaining free-text query, so
    ``@community/qwen`` is that registry filtered by ``qwen``. A bare ``@``
    names no registry and is left alone as a plain query.

    Whichever way the registry arrives — the shorthand or an explicit
    *registry* argument — it is validated the same way, so a typo raises
    instead of silently yielding an empty result set.

    Args:
        query: Free-text query, optionally carrying an ``@registry`` scope.
        registry: Explicit registry filter (may be None).
        registry_manager: Manager used to validate the resulting name.

    Returns:
        ``(registry, query)`` with any scope stripped off the query. A scope
        with no remaining query yields ``None`` for the query.

    Raises:
        RegistryFilterError: Unknown or disabled registry, or a scope that
            conflicts with *registry*.
    """
    if query and query.startswith("@"):
        scope, _, rest = query[1:].partition("/")
        if scope:  # a bare "@" names no registry — leave it as a plain query
            if registry is not None and registry != scope:
                raise RegistryFilterError(
                    "Conflicting registry: '@%s' in the query vs --registry %s." % (scope, registry),
                    registry=scope,
                    reason="conflict",
                )
            registry, query = scope, (rest.strip() or None)

    if registry is None:
        return registry, query

    entries = registry_manager.list_registries()
    available = tuple(sorted(e.name for e in entries))
    match = next((e for e in entries if e.name == registry), None)
    if match is None:
        raise RegistryFilterError(
            "Unknown registry '%s'. Available registries: %s" % (registry, ", ".join(available) or "(none configured)"),
            registry=registry,
            reason="unknown",
            available=available,
        )
    if not match.enabled:
        raise RegistryFilterError(
            "Registry '%s' is disabled. Enable it with `sparkrun registry enable %s`." % (registry, registry),
            registry=registry,
            reason="disabled",
            available=available,
        )

    return registry, query


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
        mods_subpath: Path within repo for shared mods (run.sh + supporting files)
        trusted: Whether recipes from this registry auto-run lifecycle hooks
            (pre_exec / post_exec / post_commands) without an interactive
            confirmation prompt.  Defaults to False so user-added third-party
            registries are untrusted until explicitly opted-in via
            ``sparkrun registry trust <name>`` or ``registry add --trust``.
        declared_by: Name of the plugin that declared this registry, or ``""``
            for an ordinary user-owned entry read from ``registries.yaml``.
            **Runtime-only and never serialized** — see
            :mod:`sparkrun.core.registry_defaults`.  A declared entry is an
            overlay: it exists while its plugin is loaded and is not written to
            the config file, which is what makes uninstalling a plugin remove
            its registries cleanly.  Mutating one (``registry disable`` /
            ``trust`` / ``untrust``) *materializes* it by clearing this field,
            after which it is an ordinary file entry the user owns.
    """

    name: str
    url: str
    subpath: str
    description: str = ""
    enabled: bool = True
    visible: bool = True
    tuning_subpath: str = ""
    benchmark_subpath: str = ""
    mods_subpath: str = ""
    trusted: bool = False
    declared_by: str = ""


#: Asset subpaths a registry may omit.  Shared with
#: :meth:`RegistryManager._backfill_default_subpaths`, which repairs exactly
#: these — an omitted one makes that asset kind unresolvable rather than
#: defaulted.
OPTIONAL_SUBPATH_FIELDS = ("tuning_subpath", "benchmark_subpath", "mods_subpath")

#: Every :class:`RegistryEntry` field that names a directory inside the
#: registry's checkout.  A new one must be added here or it escapes
#: :func:`assert_safe_registry_entry` and becomes an unvalidated path from a
#: remote manifest.
SUBPATH_FIELDS = ("subpath",) + OPTIONAL_SUBPATH_FIELDS


@dataclass(frozen=True)
class RegistryAsset:
    """A kind of per-registry file that sparkrun resolves by file stem.

    Recipes and benchmark profiles differ only in *where* they live and *how
    deep* the scan goes.  Everything around that — which registries are
    eligible, extension precedence, flat-beats-nested, the ``@registry/...``
    label used to disambiguate — is identical, so it lives once in
    :meth:`RegistryManager.find_asset_in_registries` and is parameterized by
    an instance of this class.

    Attributes:
        kind: Human-readable noun for messages ("recipe", "benchmark profile").
        subpath_field: :class:`RegistryEntry` attribute naming the subdirectory.
        recursive: Scan subdirectories when the flat lookup misses.  Recipe
            registries nest by model family; benchmark registries are flat.
        extensions: Accepted file extensions, in precedence order — the first
            one that exists wins, so a same-stem ``.yaml`` and ``.yml`` in one
            directory are one asset spelled two ways, not an ambiguity.
    """

    kind: str
    subpath_field: str
    recursive: bool = True
    extensions: tuple[str, ...] = (".yaml", ".yml")


#: Recipes: nested by model family, ``.yaml`` or ``.yml``.
RECIPE_ASSET = RegistryAsset("recipe", "subpath")
#: Benchmark profiles: nested like recipes, so a registry can group profiles
#: by suite or hardware without their names becoming unreachable.
BENCHMARK_ASSET = RegistryAsset("benchmark profile", "benchmark_subpath")
#: Tuning configs: shape-based JSON under ``<tuning>/<runtime>/``.  Only the
#: directory resolution is shared — lookup is by runtime, not by stem.
TUNING_ASSET = RegistryAsset("tuning config", "tuning_subpath", extensions=(".json",))
#: Shared mods (run.sh + supporting files).  Directory resolution only.
MODS_ASSET = RegistryAsset("mods", "mods_subpath")


def format_ambiguity(kind: str, name: str, matches: list[tuple[str, Path]], labels: list[str]) -> str:
    """Build the message shared by recipe and benchmark-profile ambiguity errors.

    Kept as a formatter rather than a base class: the two error types live in
    hierarchies callers already catch (``RecipeError`` / ``ProfileError``), and
    the only thing genuinely common is the wording.

    Args:
        kind: Capitalized noun for the asset ("Recipe", "Benchmark profile").
        name: The name that was ambiguous.
        matches: The ``(registry, path)`` matches.
        labels: Typeable ``@registry/...`` names, parallel to *matches*.

    Returns:
        A message naming where the collision is and how to resolve it.
    """
    registries = {reg for reg, _ in matches}
    where = (
        "in registry '%s'" % next(iter(registries))
        if len(registries) == 1
        else "in multiple registries: %s" % ", ".join(sorted(registries))
    )
    return "%s '%s' is ambiguous — %d matches %s (%s). Use the full name to specify." % (
        kind,
        name,
        len(matches),
        where,
        ", ".join(labels),
    )


def iter_asset_files(directory: Path, asset: RegistryAsset) -> list[Path]:
    """Return a directory's asset files, sorted, one per stem per directory.

    This is the *catalog* peer of :func:`_scan_asset_dir` (which resolves one
    name), and it applies the same rules so listing and lookup can never
    disagree — an asset that is runnable is listed, and vice versa.

    Covers every extension the asset declares. A same-stem ``.yaml`` and
    ``.yml`` in one directory are one asset spelled two ways, so ``.yaml``
    wins; the same stem in *different* directories stays two distinct assets,
    so this never dedupes the catalog by name.

    Args:
        directory: Directory to scan.
        asset: Which kind of asset to list.

    Returns:
        Sorted list of asset file paths.
    """
    globber = directory.rglob if asset.recursive else directory.glob
    chosen: dict[tuple[Path, str], Path] = {}
    for ext in asset.extensions:
        for f in globber("*" + ext):
            chosen.setdefault((f.parent, f.stem), f)
    return sorted(chosen.values())


def _scan_asset_dir(
    base: Path,
    name: str,
    asset: RegistryAsset,
    accept: Callable[[Path], bool] | None = None,
) -> list[Path]:
    """Return one registry's matches for *name*, newest rules applied.

    Flat wins: if ``<base>/<name>.<ext>`` exists it is the answer and no
    recursive scan happens.  Otherwise — and only when the asset is recursive —
    subdirectories are scanned, keeping at most one match per containing
    directory so the same stem in two subdirs stays two distinct assets.

    *accept* (used for the benchmark category filter) rejects a candidate
    without masking the next one: a wrong-category ``.yaml`` still lets a
    right-category ``.yml`` in the same directory through.
    """
    for ext in asset.extensions:
        candidate = base / (name + ext)
        if candidate.exists() and (accept is None or accept(candidate)):
            return [candidate]

    if not asset.recursive:
        return []

    seen_dirs: set[Path] = set()
    found: list[Path] = []
    for ext in asset.extensions:
        for candidate in sorted(base.rglob(name + ext)):
            if candidate.parent in seen_dirs:
                continue
            if accept is not None and not accept(candidate):
                continue
            seen_dirs.add(candidate.parent)
            found.append(candidate)
    return found


# Git URLs whose .sparkrun/registry.yaml manifests are used for first-run
# registry discovery (see RegistryManager._init_defaults_from_manifests).
#
# This list is **only** consulted during bootstrap-time manifest discovery.
# It does NOT control trust: trust is now a per-registry ``trusted`` field
# stored locally in ``registries.yaml`` (see ``RegistryEntry.trusted``).
BOOTSTRAP_REGISTRY_URLS = [
    "https://github.com/dbotwinick/sparkrun-recipe-registry.git",
    "https://github.com/spark-arena/recipe-registry.git",
    "https://github.com/spark-arena/community-recipe-registry.git",
]

# Shared by the shipped default and the URL migration, so a migrated config and
# a fresh install describe the ``eugr`` registry identically.
EUGR_REGISTRY_DESCRIPTION = "Official mirror of eugr/spark-vllm-docker recipes and mods"

# Prior description shipped alongside the pre-mirror upstream URL. The migration
# refreshes a description only when it still matches this exactly, so a user who
# edited theirs keeps it.
_LEGACY_EUGR_DESCRIPTION = "Official eugr/spark-vllm-docker repo recipes"


FALLBACK_DEFAULT_REGISTRIES = [
    RegistryEntry(
        name="sparkrun-testing",
        url="https://github.com/dbotwinick/sparkrun-recipe-registry.git",
        subpath="testing/recipes",
        description="Sparkrun testing registry for recipes, tuning configs, and benchmark profiles",
        tuning_subpath="testing/tuning",
        benchmark_subpath="testing/benchmarking",
        visible=False,
        trusted=True,
    ),
    RegistryEntry(
        name="official",
        url="https://github.com/spark-arena/recipe-registry.git",
        subpath="official-recipes",
        description="Official Spark Arena registry for recipes, tuning configs, and benchmark profiles",
        tuning_subpath="tuning",
        benchmark_subpath="benchmarking",
        visible=True,
        trusted=True,
    ),
    RegistryEntry(
        name="eugr",
        # Our mirror of eugr/spark-vllm-docker's recipes+mods, not the upstream
        # repo itself: the upstream repo is primarily a container build whose
        # recipes are a small corner of it, and mirroring lets the registry
        # carry its own .sparkrun/registry.yaml.  Existing configs pointing at
        # the upstream URL are rewritten by MIGRATED_REGISTRY_URLS.
        url="https://github.com/spark-arena/eugr-recipes",
        subpath="recipes",
        description=EUGR_REGISTRY_DESCRIPTION,
        mods_subpath="mods",
        visible=True,
        trusted=True,
    ),
    RegistryEntry(
        name="sparkrun-transitional",
        url="https://github.com/dbotwinick/sparkrun-recipe-registry.git",
        subpath="transitional/recipes",
        description="Transitional registry for recipes",
        tuning_subpath="testing/tuning",
        visible=True,
        trusted=True,
    ),
    RegistryEntry(
        name="experimental",
        url="https://github.com/spark-arena/recipe-registry.git",
        subpath="experimental-recipes",
        description="Spark Arena registry for experimental recipes",
        visible=False,
        trusted=True,
    ),
    RegistryEntry(
        name="community",
        url="https://github.com/spark-arena/community-recipe-registry.git",
        subpath="recipes",
        description="Community recipe registry",
        # Must mirror the repo's own .sparkrun/registry.yaml manifest
        # (recipes / tuning / benchmarks). The manifest is only consulted on
        # first-run discovery, so a registries.yaml written from this fallback
        # keeps whatever is spelled here forever — and an omitted subpath is
        # not merely a default, it makes that asset kind *unresolvable*
        # (``asset_dir`` returns nothing) and drops the path from the sparse
        # checkout, so `registry update` never fetches it either.
        tuning_subpath="tuning",
        benchmark_subpath="benchmarking",
        visible=False,
        trusted=True,
    ),
    RegistryEntry(
        name="atlas",
        # Atlas moved its recipes from Avarok-Cybersecurity/atlas-recipes to the
        # Atlas-Inf org.  The layout (a ``recipes`` subpath) is identical on both
        # sides, so existing configs are rewritten in place by
        # MIGRATED_REGISTRY_URLS rather than dropped and re-added.
        url="https://github.com/Atlas-Inf/sparkrun-recipes.git",
        subpath="recipes",
        description="Atlas recipes",
        visible=False,
        trusted=True,
    ),
]


#: scp-style SSH remote — ``[user@]host:org/repo``.  The ``[^/]`` on the path
#: is what keeps this from also matching ``https://host/...``, where the
#: character after the colon is a slash.
_SCP_LIKE_SSH_RE = re.compile(r"^(?:[^@/\s]+@)?(?P<host>[A-Za-z0-9._-]+):(?P<path>[^/].*)$")


def _normalize_registry_url(url: str) -> str:
    """Canonicalize a git URL so two spellings of one repo compare equal.

    Drops the scheme, any ``user@`` credentials, and a trailing ``/`` or
    ``.git``; rewrites the scp-style SSH form (``git@github.com:org/repo``) to
    the same ``host/org/repo`` shape as an https URL; and lowercases the
    result.

    Stripping only ``/`` and ``.git`` — as this did — meant `official` spelled
    ``git@github.com:spark-arena/recipe-registry.git``, ``http://…`` or with
    any capitalisation did **not** match the shipped default, so the trust
    backfill marked it untrusted and its recipes prompted for hook
    confirmation forever (issue #257).  A non-TTY has no way to answer that
    prompt.

    The bound on how far to canonicalize: **never merge two URLs git would
    resolve to different repos.** Scheme, credentials and case are all things
    git ignores when picking the repo, so folding them is safe. Query strings
    and fragments are deliberately *not* stripped — they are not part of a git
    URL, so anything carrying one simply fails to match, which fails closed.

    Lowercasing the path is safe for the comparison sets this feeds — the
    trusted / deprecated / migrated URL lists are all GitHub, which is
    case-insensitive — so a case-only difference cannot smuggle in a repo
    other than the one we ship.
    """
    raw = (url or "").strip()
    if not raw:
        return ""

    if "://" in raw:
        rest = raw.split("://", 1)[1]
    elif (m := _SCP_LIKE_SSH_RE.match(raw)) is not None:
        rest = "%s/%s" % (m.group("host"), m.group("path").lstrip("/"))
    else:
        rest = raw

    # Drop ``user[:pass]@`` from the authority only — a later ``@`` belongs to
    # the path and is part of the repo's identity.
    authority, sep, tail = rest.partition("/")
    if "@" in authority:
        rest = authority.rpartition("@")[2] + sep + tail

    rest = rest.strip().lower().rstrip("/")
    if rest.endswith(".git"):
        rest = rest[:-4]
    return rest.rstrip("/")


#: Schema/migration revision stamped into ``registries.yaml`` as
#: ``config_version``.  Bump when adding a **one-shot** migration below.
#:
#: A plain integer rather than the sparkrun version: it tracks the config
#: format, not the release cadence, and a downgrade must not read as "needs
#: migrating".
CONFIG_VERSION = 1

#: Top-level ``registries.yaml`` key holding names the user removed that a
#: plugin still declares (see :meth:`RegistryManager._load_suppressed`).
#:
#: Additive: an older sparkrun ignores an unknown top-level key, and the overlay
#: is not a file rewrite, so this needs no :data:`CONFIG_VERSION` bump.  Per the
#: :data:`_MIGRATIONS` docstring, only migrations that cannot detect their own
#: applicability from file content belong there.
SUPPRESSED_REGISTRIES_KEY = "suppressed_plugin_registries"

#: Version implied by a file that carries no ``config_version`` key but does
#: carry an explicit per-entry ``trusted`` field — i.e. one written after the
#: trust model landed but before the marker did.
_IMPLIED_VERSION_TRUST_PRESENT = 1


def _default_trusted_urls() -> set[str]:
    """Normalized URLs of every registry that ships ``trusted=True``.

    The single source of truth for "which registries are trusted out of the
    box", consumed by the legacy-config trust migration so a newly-trusted
    default reaches upgrading users and not just fresh installs.
    """
    return {_normalize_registry_url(e.url) for e in FALLBACK_DEFAULT_REGISTRIES if e.trusted}


def _migration_v1_backfill_trust(entries: list["RegistryEntry"]) -> None:
    """v1 — populate ``trusted`` on a file that predates the trust model.

    An entry is marked trusted when its URL matches a registry that
    :data:`FALLBACK_DEFAULT_REGISTRIES` ships as trusted; everything else stays
    untrusted.

    Deriving this from the default registry list (rather than from
    :data:`BOOTSTRAP_REGISTRY_URLS`, which exists for manifest discovery) keeps
    a single source of truth for "which registries ship trusted" — otherwise
    adding a trusted default silently fails to reach users upgrading from a
    pre-trust ``registries.yaml``.
    """
    trusted_urls = _default_trusted_urls()
    for entry in entries:
        entry.trusted = _normalize_registry_url(entry.url) in trusted_urls


#: One-shot migrations, ascending: ``(version, name, fn(entries))``.
#:
#: **Only** migrations that cannot detect their own applicability from file
#: content belong here.  Rewrites that *can* — following a moved URL, dropping
#: a deprecated registry, refreshing a stale shipped description — stay
#: unconditional in :meth:`RegistryManager._load_registries`, because running
#: them on every load is what repairs a file that arrived from a backup,
#: another machine, a hand-edit or a fork.  Version-gating those would make
#: them strictly weaker in exchange for saving a handful of string compares.
#:
#: To add one: append it here and bump :data:`CONFIG_VERSION` to match.
_MIGRATIONS: tuple[tuple[int, str, "Callable[[list[RegistryEntry]], None]"], ...] = ((1, "backfill_trust", _migration_v1_backfill_trust),)


# List of git URLs for registries that have been superseded and should be cleaned up.
# Comparison strips trailing .git from entry URLs before matching.
#
# A URL listed here is **removed** from the user's config, taking its recipes
# with it.  When a registry *moves* rather than dies, list it in
# :data:`MIGRATED_REGISTRY_URLS` instead — that rewrites the entry in place and
# keeps the user's per-entry state (enabled/visible/trust).
DEPRECATED_REGISTRIES: list[str] = [
    "https://github.com/scitrera/oss-spark-run",
    # NOT the old eugr URL: it moved, it wasn't retired. See MIGRATED_REGISTRY_URLS.
]

# Git URLs that have **moved**, mapped old -> new.  Applied as a one-time
# rewrite of the user's ``registries.yaml`` on load, so an existing install
# follows the move instead of pinning to the old repo forever.
#
# Keys and values are compared/stored via :func:`_normalize_registry_url`, so a
# ``.git`` suffix or trailing slash on either side still matches.
#
# The ``eugr`` recipes moved from eugr's container-build repo to our mirror of
# its ``recipes/`` + ``mods/`` trees.  The layout (``recipes``/``mods``
# subpaths) is identical on both sides, so only the URL needs rewriting.
#
# The ``atlas`` recipes moved with the project itself, from the
# ``Avarok-Cybersecurity`` org to ``Atlas-Inf``.  Both repos expose the recipes
# under a ``recipes`` subpath, so again only the URL needs rewriting.
MIGRATED_REGISTRY_URLS: dict[str, str] = {
    "https://github.com/eugr/spark-vllm-docker": "https://github.com/spark-arena/eugr-recipes",
    "https://github.com/Avarok-Cybersecurity/atlas-recipes": "https://github.com/Atlas-Inf/sparkrun-recipes.git",
}


def _migrated_url_for(url: str) -> str | None:
    """New URL for *url* if it names a moved registry, else ``None``.

    Normalizes **both sides** rather than indexing the dict directly. The keys
    above are written in their natural https form, so a direct lookup only
    worked while the raw key and the canonicalized entry URL happened to
    coincide — they stopped coinciding the moment the normalizer learned to
    drop the scheme, which silently disabled every URL migration.

    Evaluated per call against the live dict rather than cached into a
    canonical-key map at import: a module-level snapshot ignores any later
    edit to :data:`MIGRATED_REGISTRY_URLS`, which is both a testing footgun and
    a trap for anyone who assumes the constant is the source of truth. The dict
    holds a handful of entries, so the scan costs nothing.
    """
    canonical = _normalize_registry_url(url)
    for old, new in MIGRATED_REGISTRY_URLS.items():
        if _normalize_registry_url(old) == canonical:
            return new
    return None


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

# Specific registry names reserved for specific GitHub orgs (exact-match,
# not prefix).  Org names must be lowercase to match :func:`_get_git_org`,
# which lowercases the URL path component before returning.
EXTERNAL_RESERVED_NAMES = {
    "atlas": ("atlas-inf",),
    # ColdSnap's recipe registries, declared by the in-tree ColdSnap plugin (see
    # sparkrun.core.registry_defaults).  Exact-match rather than a
    # RESERVED_NAME_PREFIXES entry on purpose: RESERVED_PREFIX_ALLOWED_ORGS is a
    # *global* allowlist, so covering "coldsnap-*" that way would also grant
    # sparksq every other reserved prefix (official-*, sparkrun-*, arena-*).
    # That may be a reasonable thing to decide later; it is not a side effect
    # this reservation should smuggle in.
    "coldsnap": ("sparksq",),
    "coldsnap-vanilla": ("sparksq",),
}


def _get_git_org(url: str) -> str | None:
    """Extract the GitHub organization from a git URL."""
    # Extract GitHub org from URL
    # noinspection PyBroadException
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.hostname and parsed.hostname.lower() in ("github.com", "www.github.com"):
            # Path is like /org/repo or /org/repo.git
            parts = parsed.path.strip("/").split("/")
            if parts:
                org = parts[0].lower()
                return org
    except Exception:
        pass

    return None


#: Charset for a registry name.  A name is not merely a label — it is used
#: verbatim as a **directory name** under the cache root — so this is a
#: filesystem-safety guard rather than a style preference.  Requiring the first
#: character to be alphanumeric rules out three cases for free: ``.`` / ``..`` /
#: dotfiles, a leading ``-`` (which git would read as an option), and the
#: ``_url_`` prefix :meth:`RegistryManager._clone_dir_for_url` reserves for
#: shared clones.
_SAFE_REGISTRY_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: Upper bound on a registry name, so a manifest cannot produce a path that
#: blows past ``NAME_MAX`` and fails at an arbitrary later point instead of at
#: validation.
_MAX_REGISTRY_NAME_LEN = 100

#: Charset for one path segment of an asset subpath.  Same rule as a name, and
#: for the same reason — each segment becomes a real directory component.
#: Excluding ``:`` is load-bearing on Windows, where ``C:/x`` is *absolute*.
_SAFE_SUBPATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def assert_safe_registry_name(name: str) -> None:
    """Raise :class:`RegistryError` unless *name* is safe as a directory name.

    :meth:`RegistryManager._cache_dir` resolves a registry name as
    ``cache_root / name``, and registry names arrive from
    ``.sparkrun/registry.yaml`` manifests in **remote repositories**.  A name
    containing a path separator or ``..`` therefore escapes the cache root —
    and :meth:`RegistryManager._link_registry_to_shared` goes on to ``rmtree``
    whatever real directory it landed on, so this is a delete primitive, not
    just an untidy path.

    Two subtler cases fall out of requiring the first character to be
    alphanumeric (see :data:`_SAFE_REGISTRY_NAME_RE`): a name that *is* a
    shared-clone directory (``_url_<hash>``) would have the checkout its
    siblings share deleted out from under it, and a leading ``-`` would reach
    git as an option.

    Args:
        name: Candidate registry name.

    Raises:
        RegistryError: The name is empty, over-long, or outside the charset.
    """
    if not name:
        raise RegistryError("Registry name must not be empty")
    if len(name) > _MAX_REGISTRY_NAME_LEN:
        raise RegistryError("Registry name %r is too long (max %d characters)" % (name, _MAX_REGISTRY_NAME_LEN))
    if not _SAFE_REGISTRY_NAME_RE.match(name):
        raise RegistryError(
            "Registry name %r is not a valid directory name: it must start with a letter or digit "
            "and contain only letters, digits, '.', '_' or '-' (a registry name is used as a cache "
            "directory name)" % name
        )


def assert_safe_registry_subpath(subpath: str, field: str = "subpath") -> None:
    """Raise :class:`RegistryError` unless *subpath* stays inside its registry.

    Asset subpaths are resolved against the registry's cache directory
    (:meth:`RegistryManager.asset_dir` is ``_cache_dir(name) / subpath``) and
    handed to ``git sparse-checkout set``.  They come from remote manifests, so
    ``../../..`` is an escape with teeth: :func:`iter_asset_files` would
    ``rglob`` a directory outside the clone and
    :func:`~sparkrun.core.recipe.find_recipe` would offer whatever YAML it
    found there as a runnable recipe.

    An empty subpath is accepted — that is how a registry declares it serves no
    assets of that kind, and :meth:`RegistryManager.asset_dir` already returns
    ``None`` for it.

    Backslashes are rejected rather than normalized: a backslash separates
    paths on Windows and is a legal filename character on POSIX, so treating it
    as either is wrong on the other platform.

    Args:
        subpath: Candidate subpath, ``/``-separated and relative.
        field: Field name, used only to make the error message locate itself.

    Raises:
        RegistryError: The subpath is absolute, contains a traversal segment,
            or holds a character outside the segment charset.
    """
    if not subpath:
        return
    if "\\" in subpath:
        raise RegistryError("Registry %s %r must not contain a backslash (use '/' to separate path segments)" % (field, subpath))
    if subpath.startswith("/"):
        raise RegistryError("Registry %s %r must be relative to the repository root, not absolute" % (field, subpath))

    segments = [seg for seg in subpath.split("/") if seg]
    if not segments:
        raise RegistryError("Registry %s %r contains no path segments" % (field, subpath))
    for seg in segments:
        if not _SAFE_SUBPATH_SEGMENT_RE.match(seg):
            raise RegistryError(
                "Registry %s %r contains an unsafe path segment %r: each segment must start with a "
                "letter or digit and contain only letters, digits, '.', '_' or '-'" % (field, subpath, seg)
            )


def assert_safe_registry_entry(entry: RegistryEntry) -> None:
    """Validate every path-forming field on *entry*.

    The single chokepoint for "this entry can be turned into filesystem paths
    safely", so a caller cannot cover the name and forget the subpaths.  Says
    nothing about *namespace* legitimacy — that is
    :func:`validate_registry_name`, which is a separate question with a
    separate answer (a name can be perfectly safe and still be an
    impersonation).

    Raises:
        RegistryError: Any name or subpath is unsafe.
    """
    assert_safe_registry_name(entry.name)
    for field in SUBPATH_FIELDS:
        assert_safe_registry_subpath(getattr(entry, field), field=field)


def validate_registry_name(name: str, url: str) -> None:
    """Raise RegistryError if the name is unsafe or impersonates a reserved namespace.

    Two checks, in order: the name must be usable as a cache directory name
    (:func:`assert_safe_registry_name`), and a reserved prefix may only be
    claimed by a repository hosted under an allowed GitHub organization.
    Reserved prefixes protect official registry namespaces.

    Args:
        name: Registry name to validate.
        url: Git repository URL associated with the registry.

    Raises:
        RegistryError: If the name is not a safe directory name, or it uses a
            reserved prefix and the URL is not from an allowed GitHub
            organization.
    """
    assert_safe_registry_name(name)
    name_lower = name.lower()

    # check specific EXTERNAL_RESERVED_NAMES entries
    if name_lower in EXTERNAL_RESERVED_NAMES:
        if _get_git_org(url) in EXTERNAL_RESERVED_NAMES[name_lower]:
            return
        else:
            raise RegistryError(
                "Registry name %s is reserved. Only GitHub organizations [%s] may use this prefix."
                % (name, "|".join(EXTERNAL_RESERVED_NAMES[name_lower]))
            )

    # check for reserved prefixes
    matched_prefix = None
    for prefix in RESERVED_NAME_PREFIXES:
        if name_lower.startswith(prefix):
            matched_prefix = prefix
            break
    if matched_prefix is None:
        return

    # Extract GitHub org from URL
    org = _get_git_org(url)
    if org in RESERVED_PREFIX_ALLOWED_ORGS:
        return

    allowed = ", ".join(RESERVED_PREFIX_ALLOWED_ORGS)
    raise RegistryError(
        "Registry name %r uses reserved prefix %r. Only GitHub organizations [%s] may use this prefix." % (name, matched_prefix, allowed)
    )


_PROFILE_CATEGORY_CACHE: dict[tuple[str, float, int], str | None] = {}


def _profile_category_from_data(data: Any) -> str | None:
    """Extract a benchmark category from a parsed profile YAML mapping.

    Resolution order: explicit ``category:`` (top-level or inside a
    ``benchmark:`` block) → derived from the framework's ``primary_category``
    → ``None`` (let callers default).
    """
    if not isinstance(data, dict):
        return None

    explicit = data.get("category")
    block = data.get("benchmark") if isinstance(data.get("benchmark"), dict) else None
    if explicit is None and block is not None:
        explicit = block.get("category")
    if isinstance(explicit, str) and explicit:
        return explicit

    framework_name = data.get("framework")
    if framework_name is None and block is not None:
        framework_name = block.get("framework")
    if not isinstance(framework_name, str) or not framework_name:
        return None

    try:
        from sparkrun.core.bootstrap import get_benchmarking_framework
    except Exception:
        return None
    try:
        fw = get_benchmarking_framework(framework_name)
    except Exception:
        return None
    return getattr(fw, "primary_category", None)


def _profile_category(path: Path) -> str | None:
    """Return the category for the profile at *path*, cached by (path, mtime, size)."""
    try:
        st = path.stat()
        key = (str(path), st.st_mtime, st.st_size)
    except OSError:
        return None
    cached = _PROFILE_CATEGORY_CACHE.get(key)
    if cached is not None or key in _PROFILE_CATEGORY_CACHE:
        return cached
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        _PROFILE_CATEGORY_CACHE[key] = None
        return None
    cat = _profile_category_from_data(data)
    _PROFILE_CATEGORY_CACHE[key] = cat
    return cat


def is_dir_link(path: Path) -> bool:
    """True when *path* is a symlink **or** a Windows directory junction.

    ``Path.is_symlink()`` reports False for junctions, so code that only checks
    for symlinks will treat one as an ordinary directory — and then delete
    *through* it, taking the shared clone's contents with it.
    """
    return path.is_symlink() or os.path.isjunction(path)


def remove_dir_link(path: Path) -> None:
    """Remove a link without following it. Junctions need ``rmdir``, not ``unlink``."""
    if path.is_symlink():
        path.unlink()
    else:
        os.rmdir(path)


def link_directory(link: Path, target: Path) -> None:
    """Point *link* at directory *target*, portably.

    Windows grants the symlink privilege only to elevated processes (or with
    Developer Mode enabled), so a plain symlink fails there with
    ``WinError 1314: A required privilege is not held by the client``.  A
    directory *junction* is the unprivileged equivalent for local paths, so fall
    back to that rather than requiring every Windows user to elevate.

    Raises:
        OSError: Neither a symlink nor a junction could be created.
    """
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError as e:
        if os.name != "nt":
            raise
        logger.debug("symlink %s -> %s failed (%s); falling back to a junction", link, target, e)

    try:
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
        )
        failed = result.returncode != 0 or not link.exists()
        detail = (result.stderr or result.stdout or "").strip()
    except OSError as e:
        # Couldn't even launch mklink; report it as the link failure it is
        # rather than letting a bare "No such file or directory: 'cmd'" escape.
        failed, detail = True, str(e)

    if failed:
        raise OSError(
            "Could not link %s -> %s: creating a symlink needs elevation or Developer Mode, "
            "and the junction fallback failed: %s" % (link, target, detail)
        )
    logger.debug("Created junction %s -> %s", link, target)


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

    def _iter_registries(
        self,
        *,
        include_hidden: bool = False,
        only: str | None = None,
    ) -> Iterator[RegistryEntry]:
        """Yield eligible registries, applying the standard filters once.

        Args:
            include_hidden: Yield invisible registries too.  Callers that do
                not filter on visibility at all pass True.
            only: Restrict to this registry name.

        Yields:
            Enabled registry entries passing the filters, in config order.
        """
        for entry in self._load_registries():
            if not entry.enabled:
                continue
            if only is not None and entry.name != only:
                continue
            if not include_hidden and not entry.visible:
                continue
            yield entry

    def asset_dir(self, entry: RegistryEntry, asset: RegistryAsset) -> Path | None:
        """Get an asset directory within a cached registry.

        Args:
            entry: Registry entry.
            asset: Which kind of asset directory to resolve.

        Returns:
            Path to the directory, or None when the registry does not declare
            one or it is not cached.
        """
        subpath = getattr(entry, asset.subpath_field, "")
        if not subpath:
            return None
        path = self._cache_dir(entry.name) / subpath
        return path if path.exists() else None

    def find_asset_in_registries(
        self,
        name: str,
        asset: RegistryAsset,
        *,
        include_hidden: bool = False,
        accept: Callable[[Path], bool] | None = None,
    ) -> list[tuple[str, Path]]:
        """Find an asset by file stem across registries.

        Each registry is searched **independently** (see
        :func:`_scan_asset_dir`): a flat hit in one registry never suppresses
        another registry's recursive scan.

        Args:
            name: File stem to find (may include a subpath, e.g. ``fam/foo``).
            asset: Which kind of asset to look for.
            include_hidden: Include assets from invisible registries.
            accept: Optional per-candidate predicate (e.g. a category filter).

        Returns:
            List of ``(registry_name, path)`` tuples for disambiguation.
        """
        matches: list[tuple[str, Path]] = []
        for entry in self._iter_registries(include_hidden=include_hidden):
            base = self.asset_dir(entry, asset)
            if base is None:
                continue
            matches.extend((entry.name, path) for path in _scan_asset_dir(base, name, asset, accept))
        return matches

    def qualified_asset_name(self, registry_name: str, path: Path, asset: RegistryAsset) -> str:
        """Render an asset path as a user-typeable ``@registry/...`` name.

        A recursive scan means a bare stem is not always unique within one
        registry (``a/foo.yaml`` and ``b/foo.yaml`` are different assets), so
        this returns the *disambiguating* name — the extension-less path
        relative to the asset directory.  ``parse_scoped_name`` splits only on
        the first ``/``, so the result is accepted verbatim by the resolvers:
        ``@official/qwen3.6/vllm/qwen3.6-27b-fp8-mtp-vllm``.

        Falls back to the bare stem when the path is not under that registry's
        asset dir (or the registry is unknown / not cached).

        Args:
            registry_name: Registry the path was matched in.
            path: Path to the asset file.
            asset: Which kind of asset the path refers to.

        Returns:
            A name of the form ``@<registry>/<relative-path-without-extension>``.
        """
        for entry in self._load_registries():
            if entry.name != registry_name:
                continue
            base = self.asset_dir(entry, asset)
            if base and path.is_relative_to(base):
                return "@%s/%s" % (registry_name, path.relative_to(base).with_suffix("").as_posix())
            break
        return "@%s/%s" % (registry_name, path.stem)

    def _recipe_dir(self, entry: RegistryEntry) -> Path | None:
        """Get the recipe directory within a cached registry."""
        return self.asset_dir(entry, RECIPE_ASSET)

    def _load_suppressed(self) -> list[str]:
        """Names the user removed that a plugin still declares (tombstones).

        ``registry remove`` on a declared registry cannot simply drop it — the
        declaration would put it straight back on the next launch, which is
        indistinguishable from a bug.  So the removal is recorded here instead.
        Kept as a plain list under :data:`SUPPRESSED_REGISTRIES_KEY` in
        ``registries.yaml``; an older sparkrun ignores the key.
        """
        if not self._registries_path.exists():
            return []
        try:
            data = read_yaml(self._registries_path)
        except Exception:
            return []
        if not isinstance(data, dict):
            return []
        raw = data.get(SUPPRESSED_REGISTRIES_KEY) or []
        if not isinstance(raw, list):
            return []
        return [str(name) for name in raw if isinstance(name, str) and name]

    def _apply_plugin_overlay(self, entries: list[RegistryEntry]) -> list[RegistryEntry]:
        """Merge plugin-declared registries onto the user's own entries.

        Applied *after* every convergent rewrite and after any save, so declared
        entries never enter the migration or persistence path — the overlay is
        computed fresh on each load and only the user's file is durable.

        Resolution, in order:

        1. **A collision with a shipped default is refused and warned.**  The
           declaration loses.  Refusing rather than ranking is deliberate: the
           legitimate use of this seam is contributing a *new* name, and letting
           a declaration silently shadow a curated one would be a namespace
           redirect that ``validate_registry_name`` cannot catch (it only guards
           *reserved* names).  Pathological, so make it loud.

           Checked **first**, before the file-entry rule below, precisely so it
           is loud in the common case too: a shipped default is normally present
           in the user's file, so an ordering that let rule 2 match first would
           silently ignore a plugin trying to redefine ``official`` — the one
           case most worth reporting.
        2. **A file entry of the same name wins outright, silently.**  That is
           the supported override (materialize-on-mutation puts entries there,
           and it is how a user repoints a declared registry at an internal
           mirror), so it is a normal outcome, not a conflict.
        3. **A tombstone suppresses**, likewise silently — the user said no.
        """
        from sparkrun.core.registry_defaults import iter_declared_registries

        declarations = iter_declared_registries()
        if not declarations:
            return entries

        existing = {e.name for e in entries}
        suppressed = set(self._load_suppressed())
        shipped = {e.name for e in FALLBACK_DEFAULT_REGISTRIES}

        merged = list(entries)
        for declaration in declarations:
            name = declaration.entry.name
            if name in shipped:
                logger.warning(
                    "Plugin %r declares registry %r, which is a shipped default; ignoring the declaration",
                    declaration.owner,
                    name,
                )
                continue
            if name in existing:
                logger.debug(
                    "Registry %r is configured locally; ignoring the declaration from plugin %r",
                    name,
                    declaration.owner,
                )
                continue
            if name in suppressed:
                logger.debug(
                    "Registry %r declared by plugin %r was removed by the user; not re-adding",
                    name,
                    declaration.owner,
                )
                continue
            merged.append(declaration.effective_entry())
            existing.add(name)
        return merged

    def _default_registries(self) -> list[RegistryEntry]:
        """Return the default registry list.

        On first run (no ``registries.yaml``), attempts manifest-based
        discovery from ``BOOTSTRAP_REGISTRY_URLS``.  Discovered manifest
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
                # Copy — never hand out the module-level entry itself.  Callers
                # mutate what they get back (``untrust_registry`` flips
                # ``trusted``, ``disable_registry`` flips ``enabled``), and on a
                # fresh install this is the only path that produces entries, so
                # aliasing rewrote the shipped defaults process-wide.  Harmless
                # enough in a one-shot CLI; in the long-lived desktop sidecar it
                # silently moved the trust baseline that
                # :func:`_default_trusted_urls` — and so every later
                # migration — is derived from.
                combined.append(_dataclass_replace(fallback))
                seen_names.add(fallback.name)

        # Persist so subsequent _load_registries() reads from file.  The overlay
        # is applied *after* this, so a declared registry is never written into
        # a fresh registries.yaml either.
        if discovered:
            self._save_registries(combined)

        return self._apply_plugin_overlay(combined)

    def _init_defaults_from_manifests(self) -> list[RegistryEntry]:
        """Try to discover default registries from git manifest files.

        For each URL in ``BOOTSTRAP_REGISTRY_URLS``, clones the repo and reads
        its ``.sparkrun/registry.yaml`` manifest.  Entries are collected,
        deduplicated by name, and validated.

        URLs that fail to clone are skipped individually — successful URLs
        still contribute their entries (partial success).  Only if ALL URLs
        fail does this return ``[]``.

        Entries discovered via this bootstrap path are marked as
        ``trusted=True``.  Trust is granted by **sparkrun** because the
        URL came from the curated ``BOOTSTRAP_REGISTRY_URLS`` list — the
        manifest YAML itself does not (and cannot) dictate trust, even if
        it sets ``trusted: true`` on its own entries.  The standalone
        :meth:`_discover_manifest_entries` helper (used by
        :meth:`add_registry_from_url`) keeps the manifest-derived default
        of ``trusted=False`` and the bootstrap override happens here.

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

        for url in BOOTSTRAP_REGISTRY_URLS:
            try:
                entries = self._discover_manifest_entries(url)
                for entry in entries:
                    if entry.name in seen_names:
                        logger.debug("Skipping duplicate manifest entry %r", entry.name)
                        continue
                    validate_registry_name(entry.name, entry.url)
                    # Bootstrap-discovered entries are trusted because they
                    # came in via the curated BOOTSTRAP_REGISTRY_URLS list,
                    # not because the manifest declared itself trustworthy.
                    entry.trusted = True
                    seen_names.add(entry.name)
                    all_entries.append(entry)
            except Exception as e:
                logger.warning("Manifest discovery failed for %s: %s", url, e)
                # Continue to next URL instead of aborting entirely

        return all_entries

    def _load_registries_from_file(self) -> list[RegistryEntry]:
        """Load registries from the YAML config file without any fallback logic.

        Entries whose name or subpaths are not safe to turn into filesystem
        paths (:func:`assert_safe_registry_entry`) are **skipped with a
        warning** rather than raising.  Skipping is deliberately narrower than
        the enclosing ``except`` in :meth:`_load_registries`, which discards the
        whole file and reverts to the shipped defaults: one bad entry — a
        hand-edit, a merge, a manifest read by an older build that had no
        charset check — must not take the user's other registries with it.  The
        namespace check (:func:`validate_registry_name`) is deliberately *not*
        applied here; it gates *adding* a registry, and running it on load would
        break an existing config retroactively.

        Returns:
            List of usable registry entries parsed from registries.yaml.

        Raises:
            Exception: If the file cannot be read or parsed.
        """
        data = read_yaml(self._registries_path)
        registries = data.get("registries", [])
        entries: list[RegistryEntry] = []
        for r in registries:
            entry = RegistryEntry(
                name=r["name"],
                url=r["url"],
                subpath=r["subpath"],
                description=r.get("description", ""),
                enabled=r.get("enabled", True),
                visible=r.get("visible", True),
                tuning_subpath=r.get("tuning_subpath", ""),
                benchmark_subpath=r.get("benchmark_subpath", ""),
                mods_subpath=r.get("mods_subpath", ""),
                trusted=r.get("trusted", False),
            )
            try:
                assert_safe_registry_entry(entry)
            except RegistryError as e:
                logger.warning("Skipping unusable registry entry in %s: %s", self._registries_path, e)
                continue
            entries.append(entry)
        return entries

    def _read_config_version(self) -> int:
        """Migration revision of the on-disk registries.yaml.

        Reads the ``config_version`` marker.  When it is absent — a file
        written before the marker existed — the revision is *inferred* from
        whether the entries carry an explicit ``trusted`` field, which is the
        only pre-marker evidence available.

        That inference is sound only because :meth:`_save_registries` now
        writes ``trusted`` on **every** entry, in both directions.  It used to
        omit ``trusted: false``, so "no trusted key anywhere" meant either
        *pre-trust file* or *everything is untrusted* — indistinguishable.
        That ambiguity is what made the one-shot migration re-fire on every
        load, and what silently reverted a user who had untrusted every
        registry (issue #257 follow-up).  The two mechanisms deliberately back
        each other up: the marker is authoritative, and the explicit field
        keeps this fallback correct if the marker is ever lost to a hand-edit,
        a merge, or a tool that rewrites the file.

        Returns 0 for a file that predates both.
        """
        try:
            data = read_yaml(self._registries_path)
        except Exception:
            return CONFIG_VERSION  # unreadable: never "migrate" what we can't see
        if not isinstance(data, dict):
            return CONFIG_VERSION

        raw_version = data.get("config_version")
        if isinstance(raw_version, int) and not isinstance(raw_version, bool):
            return raw_version

        registries = data.get("registries") or []
        if not isinstance(registries, list) or not registries:
            # No entries to migrate; treat as current so nothing re-fires.
            return CONFIG_VERSION
        if any(isinstance(raw, dict) and "trusted" in raw for raw in registries):
            return _IMPLIED_VERSION_TRUST_PRESENT
        return 0

    def _load_entries_for_mutation(self) -> list[RegistryEntry]:
        """Raw entries, with any pending one-shot migration already applied.

        For the mutating commands (``restore_missing_defaults``,
        ``cleanup_deprecated``).  They end in a save, and a save stamps
        ``config_version`` — so reading a file that still has migrations
        pending would mark it done and skip those migrations forever.

        Deliberately returns the **raw** list rather than
        :meth:`_load_registries`'s: that one filters deprecated entries out,
        and ``cleanup_deprecated`` has to see them to report them and drop
        their caches.
        """
        if self._registries_path.exists() and self._read_config_version() < CONFIG_VERSION:
            self._load_registries()  # side effect: apply and persist migrations
        try:
            return self._load_registries_from_file()
        except Exception:
            return self._load_registries()

    def _run_one_shot_migrations(self, entries: list[RegistryEntry]) -> bool:
        """Apply every one-shot migration newer than the file's revision.

        Always returns True when a migration pass ran, even if no entry
        changed: the point is to stamp the marker so it never runs again.
        Persisting only on a content change is what left a file with nothing
        to backfill looking un-migrated forever.

        A file stamped *ahead* of this build is left alone — we never migrate
        backwards, and an unknown future revision is not an error.
        """
        from_version = self._read_config_version()
        if from_version >= CONFIG_VERSION:
            if from_version > CONFIG_VERSION:
                logger.debug(
                    "registries.yaml is at config_version %d, newer than this sparkrun (%d); leaving it alone",
                    from_version,
                    CONFIG_VERSION,
                )
            return False

        for version, name, fn in _MIGRATIONS:
            if version <= from_version:
                continue
            fn(entries)
            logger.info("Applied registries.yaml migration v%d (%s)", version, name)
        return True

    @staticmethod
    def _backfill_default_subpaths(entries: list[RegistryEntry]) -> bool:
        """Fill in asset subpaths a shipped default gained after this file was written.

        An omitted subpath is not a harmless default — it makes that asset kind
        **unresolvable**.  ``asset_dir`` returns nothing when the field is
        blank, so ``--profile <name>`` reports "not found" no matter how it is
        spelled, and ``_build_sparse_paths`` drops the directory from the sparse
        checkout so ``registry update`` never fetches it either.  The symptom is
        therefore a registry that appears healthy and silently cannot serve
        benchmark profiles, tuning configs or mods.

        Nothing re-reads a registry's ``.sparkrun/registry.yaml`` manifest once
        ``registries.yaml`` exists — manifests are consulted only on first-run
        discovery — so a file written from :data:`FALLBACK_DEFAULT_REGISTRIES`
        (which happens whenever discovery was offline) keeps whatever that list
        spelled at the time, forever.

        Only ever *adds*: a user who deliberately blanked a subpath gets it
        back, which is the accepted trade for repairing the far more common
        case, but a subpath the user has customised is never overwritten.
        Matching is by URL, so a renamed registry is still repaired.

        Returns:
            True when any entry was modified (caller re-saves the file).
        """
        by_url = {_normalize_registry_url(e.url): e for e in FALLBACK_DEFAULT_REGISTRIES}
        changed = False
        for entry in entries:
            shipped = by_url.get(_normalize_registry_url(entry.url))
            if shipped is None:
                continue
            for field in OPTIONAL_SUBPATH_FIELDS:
                if not getattr(entry, field) and getattr(shipped, field):
                    setattr(entry, field, getattr(shipped, field))
                    logger.info(
                        "Backfilled %s=%r on registry %r from shipped default",
                        field,
                        getattr(shipped, field),
                        entry.name,
                    )
                    changed = True
        return changed

    @staticmethod
    def _migrate_registry_urls(entries: list[RegistryEntry]) -> bool:
        """Rewrite entries whose URL appears in :data:`MIGRATED_REGISTRY_URLS`.

        A moved registry keeps the user's own choices — name, subpaths,
        ``enabled``/``visible`` — and follows the repo to its new home.
        Without this an existing install would keep pulling the old repo
        indefinitely, since nothing re-reads the shipped default URL once
        ``registries.yaml`` exists.

        **Trust is re-asserted from the shipped default** when the new URL is
        one we ship as ``trusted=True``.  The one-shot backfill in
        :meth:`_migrate_trust_field` only ever runs against a pre-trust file,
        so a registry that became a trusted default *after* that file was
        written is stuck untrusted forever — which is exactly the ``eugr``
        case, and would leave its mods prompting on every launch.  Moving the
        registry to a repo we publish is the natural point to apply the
        default.

        The tradeoff: ``_save_registries`` omits ``trusted: false``, so an
        explicit ``registry untrust`` is indistinguishable on disk from "never
        granted", and a user who had untrusted a now-moved default gets it
        re-granted once.  Re-run ``sparkrun registry untrust <name>`` to
        restore it; the move only happens once.

        The ``eugr`` description is refreshed too, but only when it still
        matches the string that shipped alongside the old URL, so a
        user-customized description survives.

        Returns:
            True when at least one entry changed (the caller persists).
        """
        changed = False
        trusted_urls = _default_trusted_urls()
        for entry in entries:
            new_url = _migrated_url_for(entry.url)
            if not new_url or _normalize_registry_url(entry.url) == _normalize_registry_url(new_url):
                continue
            logger.info("Registry %r moved: %s -> %s", entry.name, entry.url, new_url)
            entry.url = new_url
            if entry.description == _LEGACY_EUGR_DESCRIPTION:
                entry.description = EUGR_REGISTRY_DESCRIPTION
            if not entry.trusted and _normalize_registry_url(new_url) in trusted_urls:
                logger.info("Registry %r ships trusted; applying that default after the move", entry.name)
                entry.trusted = True
            changed = True
        return changed

    def _load_registries(self) -> list[RegistryEntry]:
        """Load registries from YAML configuration.

        Returns:
            List of registry entries, or default registries if config doesn't exist
        """
        if not self._registries_path.exists():
            logger.debug("No registries.yaml found, using defaults")
            return self._default_registries()

        # Read the file's revision BEFORE anything writes, since a write stamps
        # the marker and would make the file look already-migrated.
        pending_migrations = self._read_config_version() < CONFIG_VERSION

        try:
            entries = self._load_registries_from_file()

            # --- Convergent rewrites: content-detected, run on every load. ---
            # Follow moved registries BEFORE anything else inspects the URL.
            # Ordering is load-bearing for the trust backfill: that marks an
            # entry trusted by matching its URL against `_default_trusted_urls()`,
            # which holds the *new* URLs — a pre-trust config still carrying an
            # old URL would otherwise be backfilled as untrusted.
            urls_migrated = self._migrate_registry_urls(entries)

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

            # Backfill asset subpaths a shipped default has gained since this
            # file was written.  Content-detected, so it runs on every load.
            subpaths_backfilled = self._backfill_default_subpaths(filtered)

            # --- One-shot migrations: version-gated, run at most once ever. ---
            migrated = self._run_one_shot_migrations(filtered) if pending_migrations else False

            if migrated or urls_migrated or subpaths_backfilled:
                self._save_registries(filtered)
            # Overlay last: declared entries must not reach the rewrite or save
            # path above, or a plugin's registry would be persisted into the
            # user's file and outlive the plugin.
            return self._apply_plugin_overlay(filtered)
        except Exception as e:
            logger.warning("Failed to load registries.yaml: %s", e)
            return self._default_registries()

    def _save_registries(self, entries: list[RegistryEntry], *, suppressed: list[str] | None = None) -> None:
        """Save registries to YAML configuration.

        Plugin-declared entries (``declared_by`` set) are **skipped**.  Every
        mutating command reads through :meth:`_load_registries`, which now
        includes the overlay, and each of them rewrites the whole document — so
        without this filter the first ``registry disable`` of anything would
        quietly persist every declared registry, and they would then outlive
        their plugin.  Materializing one is done by *clearing* ``declared_by``
        (see :meth:`_materialize_declared`), which is what lets it through here.

        Args:
            entries: Registry entries to save; declared entries are ignored.
            suppressed: Tombstone list to write.  ``None`` preserves whatever is
                already on disk — this method rebuilds the document from
                scratch, so anything not re-emitted is dropped.
        """
        if suppressed is None:
            suppressed = self._load_suppressed()

        data_list = []
        for e in entries:
            if e.declared_by:
                continue
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
            if e.mods_subpath:
                d["mods_subpath"] = e.mods_subpath
            # ``trusted`` is written in BOTH directions, unlike the other flags.
            # The convention here is "omit the field default" — ``enabled`` /
            # ``visible`` default True so only False is written, and ``trusted``
            # defaults False so only True used to be.  For those two, absence is
            # unambiguous; for trust it was not, because a file with nothing
            # trusted was byte-identical to one written before the trust model
            # existed.  Writing it always is what makes
            # :meth:`_read_config_version`'s fallback sound and keeps an explicit
            # ``registry untrust`` from being read as "never migrated".
            d["trusted"] = bool(e.trusted)
            data_list.append(d)

        # Stamped on every write, not just by the migration runner: this method
        # rebuilds the document from scratch, so anything it didn't re-emit
        # would be silently dropped.
        data: dict[str, Any] = {"config_version": CONFIG_VERSION, "registries": data_list}
        if suppressed:
            data[SUPPRESSED_REGISTRIES_KEY] = sorted(set(suppressed))
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
        Tuning, benchmark, and mods subpaths are added when configured.
        """
        paths = [entry.subpath]
        if entry.tuning_subpath:
            paths.append(entry.tuning_subpath)
        if entry.benchmark_subpath:
            paths.append(entry.benchmark_subpath)
        if entry.mods_subpath:
            paths.append(entry.mods_subpath)
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
                validate_git_url(url)
                result = subprocess.run(
                    ["git", "clone", "--filter=blob:none", "--sparse", "--", url, str(clone_dir)],
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

        # Remove old per-registry dir if it's a real directory (not a link)
        if per_registry_dir.exists() and not is_dir_link(per_registry_dir):
            import shutil

            shutil.rmtree(per_registry_dir)

        # Link per_registry_dir -> shared_dir (junction on Windows; see
        # link_directory for why a plain symlink isn't enough there).
        per_registry_dir.parent.mkdir(parents=True, exist_ok=True)
        if is_dir_link(per_registry_dir):
            remove_dir_link(per_registry_dir)
        link_directory(per_registry_dir, shared_dir)

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

        # A cached clone is keyed by registry *name*, but its `origin` is the
        # URL it was cloned from. If the entry's URL has since changed (a
        # MIGRATED_REGISTRY_URLS rewrite, or a hand-edited registries.yaml),
        # the `git fetch origin` below would silently keep pulling the old
        # repo. Drop the stale cache so the fresh-clone path runs.
        self._drop_cache_if_url_changed(entry)

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
                validate_git_url(entry.url)
                result = subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "--filter=blob:none",
                        "--sparse",
                        "--",
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

    def _drop_cache_if_url_changed(self, entry: RegistryEntry) -> bool:
        """Remove ``entry``'s cached clone when it came from a different URL.

        The per-registry cache dir is keyed by name, so a registry that changes
        URL keeps a checkout of the *old* repo whose ``origin`` still points
        there.  ``_clone_or_pull_single`` fetches from ``origin``, so without
        this the new URL would never actually be pulled.

        Best-effort: an unreadable or remote-less cache is left alone rather
        than deleted, since a failed sync is recoverable and a wrong delete is
        not.  The cache dir may be a link into a shared clone (several
        registries on one URL), so unlink rather than delete *through* it.

        Returns:
            True when a stale cache was dropped.
        """
        cache_dir = self._cache_dir(entry.name)
        if not (cache_dir / ".git").exists():
            return False

        try:
            result = subprocess.run(
                ["git", "-C", str(cache_dir), "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                stdin=subprocess.DEVNULL,
                env=self._git_env(),
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("Could not read origin for registry %s: %s", entry.name, e)
            return False

        if result.returncode != 0 or not isinstance(result.stdout, str):
            # No origin remote, or output we can't read. "Can't tell" is not
            # "changed" — a failed sync is recoverable, a wrong delete isn't.
            logger.debug("Registry %s cache has no readable origin; leaving it alone", entry.name)
            return False

        cached_url = _normalize_registry_url(result.stdout.strip())
        if cached_url == _normalize_registry_url(entry.url):
            return False

        logger.info(
            "Registry %r cache was cloned from %s but is now %s; re-cloning",
            entry.name,
            cached_url,
            entry.url,
        )
        try:
            if is_dir_link(cache_dir):
                remove_dir_link(cache_dir)
            else:
                import shutil

                shutil.rmtree(cache_dir)
        except OSError as e:
            logger.warning("Could not remove stale cache for registry %s: %s", entry.name, e)
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
                uses a reserved name prefix from a non-allowed URL, has an
                invalid/unsafe git URL, or carries a name/subpath that would
                escape the registry cache.
        """
        try:
            validate_git_url(entry.url)
        except ValueError as exc:
            raise RegistryError("Invalid registry URL: %s" % exc) from exc
        # Covers the subpaths too: validate_registry_name only sees the name,
        # and this is the public entry point for programmatic adds.
        assert_safe_registry_entry(entry)
        validate_registry_name(entry.name, entry.url)
        registries = self._load_registries()
        if any(r.name == entry.name for r in registries):
            raise RegistryError(f"Registry {entry.name!r} already exists")
        registries.append(entry)
        # Adding a name back is an explicit reversal of a prior removal, so it
        # clears any tombstone.  Otherwise a user who removed a plugin-declared
        # registry and later re-added it under their own URL would keep a
        # suppression entry that silences the declaration forever, invisibly.
        suppressed = [name for name in self._load_suppressed() if name != entry.name]
        self._save_registries(registries, suppressed=suppressed)
        logger.info("Added registry %s", entry.name)

    def _discover_manifest_entries(self, url: str) -> list[RegistryEntry]:
        """Clone a repo, read its .sparkrun/registry.yaml manifest, return entries.

        Does NOT save or add entries — purely a discovery/parsing operation.

        Every declared entry is validated with
        :func:`assert_safe_registry_entry` before it is returned.  This is a
        **remote, untrusted document** whose names and subpaths go on to form
        real filesystem paths, so an unsafe entry is dropped with a warning; the
        rest of the manifest still applies, matching the per-URL partial-success
        behavior of :meth:`_init_defaults_from_manifests`.  A manifest with no
        usable entry left raises, so the caller never silently adds nothing.

        Args:
            url: Git repository URL.

        Returns:
            List of RegistryEntry objects declared in the manifest.

        Raises:
            RegistryError: If clone fails, no manifest found, the manifest is
                empty, or no declared entry survived validation.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "repo"
            git_env = self._git_env()
            validate_git_url(url)
            result = subprocess.run(
                # Blob-filtered + sparse: only ``.sparkrun`` is ever read here,
                # so pulling the recipe trees would be wasted transfer — and
                # this runs once per bootstrap URL on first run.
                ["git", "clone", "--depth=1", "--single-branch", "--filter=blob:none", "--sparse", "--", url, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                stdin=subprocess.DEVNULL,
                env=git_env,
            )
            if result.returncode != 0:
                raise RegistryError("Failed to clone %s: %s" % (url, result.stderr.strip()))

            # ``--sparse`` checks out root-level files only, so the manifest
            # directory has to be asked for explicitly — without this the clone
            # succeeds and the manifest is simply absent.
            result = subprocess.run(
                ["git", "-C", str(tmp_path), "sparse-checkout", "set", ".sparkrun"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                stdin=subprocess.DEVNULL,
                env=git_env,
            )
            if result.returncode != 0:
                raise RegistryError("Failed to check out .sparkrun/ from %s: %s" % (url, result.stderr.strip()))

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
            entries: list[RegistryEntry] = []
            for reg_data in registries_data:
                entry = RegistryEntry(
                    name=reg_data["name"],
                    url=url,
                    subpath=reg_data.get("subpath", reg_data.get("recipes", "recipes")),
                    description=reg_data.get("description", ""),
                    enabled=reg_data.get("enabled", True),
                    visible=reg_data.get("visible", True),
                    tuning_subpath=reg_data.get("tuning_subpath", reg_data.get("tuning", "")),
                    benchmark_subpath=reg_data.get("benchmark_subpath", reg_data.get("benchmarks", "")),
                    mods_subpath=reg_data.get("mods_subpath", reg_data.get("mods", "")),
                )
                try:
                    assert_safe_registry_entry(entry)
                except RegistryError as e:
                    logger.warning("Ignoring unsafe entry in the manifest of %s: %s", url, e)
                    continue
                entries.append(entry)

            if not entries:
                raise RegistryError("Manifest in %s declares no usable registries (all entries were rejected)" % url)
            return entries

    def add_registry_from_url(self, url: str, trust: bool = False) -> list[RegistryEntry]:
        """Add registries by discovering them from a repo's .sparkrun/registry.yaml manifest.

        Clones the repo temporarily, reads the manifest, and adds all declared registries.

        Args:
            url: Git repository URL.
            trust: If True, mark every newly-added entry as ``trusted=True``
                after the add step.  Default False keeps user-added
                registries untrusted until an explicit opt-in.

        Returns:
            List of RegistryEntry objects added.

        Raises:
            RegistryError: If clone fails, no manifest found, or URL is invalid.
        """
        try:
            validate_git_url(url)
        except ValueError as exc:
            raise RegistryError("Invalid registry URL: %s" % exc) from exc
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
        if trust and added:
            # Flip the trust bit on every newly-added entry and persist.
            for entry in added:
                self.trust_registry(entry.name)
                entry.trusted = True
        return added

    @staticmethod
    def _materialize_declared(entry: RegistryEntry) -> None:
        """Turn a plugin-declared entry into a user-owned one, in place.

        Clearing ``declared_by`` is what lets :meth:`_save_registries` write it.
        After this the file entry wins over the declaration
        (:meth:`_apply_plugin_overlay` rule 1), so the user's change is durable
        and survives the plugin being upgraded, disabled or removed — which is
        the point: they took ownership of it.
        """
        if entry.declared_by:
            logger.info(
                "Registry %r was declared by plugin %r; saving it to your configuration so this change persists",
                entry.name,
                entry.declared_by,
            )
            entry.declared_by = ""

    def remove_registry(self, name: str) -> None:
        """Remove a registry by name.

        A plugin-declared registry is **tombstoned** rather than deleted: it is
        not in ``registries.yaml`` to remove, and dropping it from the in-memory
        list would let the declaration put it straight back on the next launch —
        a removal that silently fails to remove. The tombstone is cleared by
        adding the registry back (:meth:`add_registry`).

        Args:
            name: Registry name to remove

        Raises:
            RegistryError: If the registry is not found
        """
        registries = self._load_registries()
        target = next((r for r in registries if r.name == name), None)
        if target is None:
            raise RegistryError(f"Registry {name!r} not found")

        filtered = [r for r in registries if r.name != name]
        if target.declared_by:
            suppressed = self._load_suppressed()
            if name not in suppressed:
                suppressed.append(name)
            self._save_registries(filtered, suppressed=suppressed)
            logger.info(
                "Removed registry %s (declared by plugin %r; suppressed so it is not re-added)",
                name,
                target.declared_by,
            )
            return

        self._save_registries(filtered)
        logger.info("Removed registry %s", name)

    @staticmethod
    def _is_deprecated_url(url: str) -> bool:
        """Check whether a registry URL matches a deprecated entry.

        Shares :func:`_normalize_registry_url` with the trust and migrated-URL
        comparisons.  This used to inline its own weaker copy of that logic, so
        a deprecated registry spelled as an SSH remote (or with a different
        scheme or capitalisation) was never cleaned up.
        """
        normalized = _normalize_registry_url(url)
        return any(normalized == _normalize_registry_url(dep_url) for dep_url in DEPRECATED_REGISTRIES)

    def restore_missing_defaults(self) -> list[str]:
        """Add default registry entries that are missing from the config.

        Checks ``FALLBACK_DEFAULT_REGISTRIES`` for entries whose name is not
        present in the current ``registries.yaml``.  Missing entries are
        appended and persisted.

        Returns:
            List of registry names that were added.
        """
        entries = self._load_entries_for_mutation()

        existing_names = {e.name for e in entries}
        added: list[str] = []

        for default in FALLBACK_DEFAULT_REGISTRIES:
            if default.name not in existing_names:
                entries.append(_dataclass_replace(default))  # copy — see _default_registries
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

        entries = self._load_entries_for_mutation()
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

                    if is_dir_link(cache_dir):
                        remove_dir_link(cache_dir)
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
                if is_dir_link(child):
                    remove_dir_link(child)
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
                self._materialize_declared(e)
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

    def _set_registry_trusted(self, name: str, trusted: bool) -> None:
        """Set the trusted state of a registry by name.

        Args:
            name: Registry name to modify.
            trusted: Target trust state.

        Raises:
            RegistryError: If the registry is not found.
        """
        entries = self._load_registries()
        for e in entries:
            if e.name == name:
                e.trusted = trusted
                self._materialize_declared(e)
                self._save_registries(entries)
                logger.info("%s registry %s", "Trusted" if trusted else "Untrusted", name)
                return
        raise RegistryError("Registry %r not found" % name)

    def trust_registry(self, name: str) -> None:
        """Mark a registry as trusted (recipes from it get auto-trust for hooks).

        Raises:
            RegistryError: If the registry is not found.
        """
        self._set_registry_trusted(name, True)

    def untrust_registry(self, name: str) -> None:
        """Mark a registry as untrusted (recipes from it require --trust or prompt).

        Raises:
            RegistryError: If the registry is not found.
        """
        self._set_registry_trusted(name, False)

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
        for entry in self._iter_registries(include_hidden=include_hidden):
            recipe_dir = self._recipe_dir(entry)
            if recipe_dir:
                paths.append(recipe_dir)
            else:
                logger.debug("Registry %s not cached or recipe path not found", entry.name)

        return paths

    def _list_dir_recipes(self, recipe_dir: Path, registry_name: str) -> list[dict[str, Any]]:
        """List all recipes in a directory with metadata.

        Args:
            recipe_dir: Directory to scan for ``.yaml`` / ``.yml`` recipe files.
            registry_name: Name of the registry this directory belongs to.

        Returns:
            List of recipe metadata dicts.
        """
        if not recipe_dir.is_dir():
            return []

        from sparkrun.core.recipe import recipe_summary

        recipes = []
        for f in iter_asset_files(recipe_dir, RECIPE_ASSET):
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
        from sparkrun.core.recipe import recipe_matches_query

        results = []
        for entry in self._iter_registries(include_hidden=include_hidden):
            recipe_dir = self._recipe_dir(entry)
            if recipe_dir is None:
                continue
            for recipe in self._list_dir_recipes(recipe_dir, entry.name):
                if recipe_matches_query(recipe, query):
                    results.append(recipe)

        return results

    def registry_for_path(self, path: Path) -> str | None:
        """Return the registry name that owns the given path, or None."""
        # Ownership is not a visibility question — a hidden registry still owns
        # its files, so this deliberately does not filter on `visible`.
        for entry in self._iter_registries(include_hidden=True):
            recipe_dir = self._recipe_dir(entry)
            if recipe_dir and path.is_relative_to(recipe_dir):
                return entry.name
        return None

    def qualified_recipe_name(self, registry_name: str, path: Path) -> str:
        """Render a registry recipe path as a user-typeable ``@registry/...`` name.

        A registry's recipe dir is scanned recursively, so a bare stem is not
        always unique within one registry (``a/foo.yaml`` and ``b/foo.yaml``
        are different recipes).  This returns the *disambiguating* name — the
        extension-less path relative to the registry's recipe dir — which
        :func:`~sparkrun.core.recipe.find_recipe` accepts verbatim because
        ``parse_scoped_name`` splits only on the first ``/``:
        ``@official/qwen3.6/vllm/qwen3.6-27b-fp8-mtp-vllm``.

        Falls back to the bare stem when the path is not under the registry's
        recipe dir (or the registry is unknown / not cached).

        Args:
            registry_name: Registry the path was matched in.
            path: Path to the recipe file.

        Returns:
            A name of the form ``@<registry>/<relative-path-without-extension>``.
        """
        return self.qualified_asset_name(registry_name, path, RECIPE_ASSET)

    def find_recipe_in_registries(self, name: str, include_hidden: bool = False) -> list[tuple[str, Path]]:
        """Find a recipe by file stem across all registries.

        Searches for recipes whose file stem matches the given name.

        Args:
            name: Recipe file stem to find (e.g. 'glm-4.7-flash-awq')
            include_hidden: If True, include recipes from invisible registries

        Returns:
            List of (registry_name, recipe_path) tuples for disambiguation
        """
        return self.find_asset_in_registries(name, RECIPE_ASSET, include_hidden=include_hidden)

    def _tuning_dir(self, entry: RegistryEntry) -> Path | None:
        """Get the tuning directory within a cached registry."""
        return self.asset_dir(entry, TUNING_ASSET)

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
        # Tuning lookup is not visibility-filtered; `_tuning_dir` already
        # returns None for a registry that declares no tuning subpath.
        for entry in self._iter_registries(include_hidden=True, only=registry_name or None):
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
        for entry in self._iter_registries(include_hidden=True):
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

    def _mods_dir(self, entry: RegistryEntry) -> Path | None:
        """Get the mods directory within a cached registry."""
        return self.asset_dir(entry, MODS_ASSET)

    def ensure_registry_on_host(
        self,
        name: str,
        host: str,
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
        extra_sparse_paths: list[str] | None = None,
    ) -> str:
        """Clone or update a registry's git repo on a remote head node.

        Used by delegated-mode flows that need registry-backed resources
        (e.g. mods) to live on the head node rather than the control
        machine. Mirrors the local clone layout under
        ``~/.cache/sparkrun/registries/_url_<hash>/`` on the remote host
        with sparse-checkout configured for the union of all subpaths
        declared by registries sharing this URL.

        Args:
            name: Registry name (used to look up the URL and subpaths).
            host: Remote head node hostname.
            ssh_kwargs: SSH connection kwargs.
            dry_run: When True, return the would-be path without acting.
            extra_sparse_paths: Additional sparse-checkout paths to union
                with the URL's normal set. Used by the eugr fallback to
                guarantee ``mods/`` is checked out even when the local
                registry entry predates the ``mods_subpath`` field.

        Returns:
            Absolute path on the remote host where the registry is checked out.
        """
        import hashlib

        from sparkrun.orchestration.primitives import run_script_on_host
        from sparkrun.utils.shell import quote

        entry = self.get_registry(name)
        validate_git_url(entry.url)
        sparse_paths = list(self._sparse_checkout_paths_for_url(entry.url))
        if extra_sparse_paths:
            for p in extra_sparse_paths:
                if p and p not in sparse_paths:
                    sparse_paths.append(p)
        url_hash = hashlib.sha256(entry.url.encode()).hexdigest()[:12]
        remote_clone_dir = "~/.cache/sparkrun/registries/_url_%s" % url_hash
        sparse_args = " ".join(quote(p) for p in sparse_paths) if sparse_paths else ""

        # Redirect git's chatty output to stderr (1>&2) so stdout stays clean,
        # then emit the resolved path via a printf sentinel.  Parsing the
        # sentinel is robust even if some SSH config merges streams or a
        # future change adds more shell noise.
        script = (
            "set -e\n"
            "REPO_DIR=%(path)s\n"
            'if [ -d "$REPO_DIR/.git" ]; then\n'
            '  git -C "$REPO_DIR" fetch origin 1>&2\n'
            '  git -C "$REPO_DIR" reset --hard FETCH_HEAD 1>&2\n'
            "else\n"
            '  mkdir -p "$(dirname "$REPO_DIR")"\n'
            '  git clone --filter=blob:none --sparse %(url)s "$REPO_DIR" 1>&2\n'
            "fi\n"
        ) % {"path": remote_clone_dir, "url": quote(entry.url)}
        if sparse_args:
            script += 'git -C "$REPO_DIR" sparse-checkout set %s 1>&2\n' % sparse_args
        script += 'printf "__SPARKRUN_REPO_DIR__%s\\n" "$REPO_DIR"\n'

        logger.info("Ensuring registry %r on head node %s...", name, host)
        if dry_run:
            return remote_clone_dir

        result = run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=180)
        if not result.success:
            raise RegistryError(
                "Failed to ensure registry %r on %s: %s" % (name, host, result.stderr.strip() if result.stderr else "(no output)")
            )
        # Scan stdout in reverse for the sentinel line so any unexpected
        # preceding output (e.g. SSH config merging stderr into stdout) is
        # ignored.  Fall back to the computed path if the sentinel is absent.
        marker = "__SPARKRUN_REPO_DIR__"
        if result.stdout:
            for line in reversed(result.stdout.splitlines()):
                if line.startswith(marker):
                    return line[len(marker) :].strip()
        return remote_clone_dir

    def _benchmark_dir(self, entry: RegistryEntry) -> Path | None:
        """Get benchmark directory within a cached registry."""
        return self.asset_dir(entry, BENCHMARK_ASSET)

    def find_benchmark_profile_in_registries(
        self,
        name: str,
        include_hidden: bool = False,
        category: str | None = None,
    ) -> list[tuple[str, Path]]:
        """Find benchmark profile by file stem across registries.

        Args:
            name: Profile file stem (e.g. 'spark-arena-v1')
            include_hidden: If True, include profiles from invisible registries
            category: Optional category filter. When set, only profiles whose
                declared (or framework-derived) category matches are returned.

        Returns:
            List of (registry_name, profile_path) tuples for disambiguation
        """
        accept = None if category is None else (lambda p: _profile_category(p) == category)
        return self.find_asset_in_registries(name, BENCHMARK_ASSET, include_hidden=include_hidden, accept=accept)

    def list_benchmark_profiles(
        self,
        registry_name: str | None = None,
        include_hidden: bool = False,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all benchmark profiles across registries.

        Args:
            registry_name: If provided, only list from this registry
            include_hidden: If True, include profiles from invisible registries
            category: Optional category filter. When set, only profiles whose
                declared (or framework-derived) category matches are returned.

        Returns:
            List of dicts with keys: registry, file, name, description, path,
            category
        """
        import yaml

        profiles = []
        # Naming a registry outranks its visibility default — the user is
        # explicitly targeting it, so the visibility filter is dropped.
        for entry in self._iter_registries(
            include_hidden=include_hidden or bool(registry_name),
            only=registry_name or None,
        ):
            benchmark_dir = self._benchmark_dir(entry)
            if benchmark_dir is None:
                continue
            # Shares the lookup scanner, so the catalog and
            # find_benchmark_profile_in_registries can never disagree about
            # which files exist.
            for f in iter_asset_files(benchmark_dir, BENCHMARK_ASSET):
                # Read metadata from the profile
                profile_name = f.stem
                description = ""
                profile_category: str | None = None
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
                    profile_category = _profile_category_from_data(data)
                except Exception:
                    pass
                if category is not None and profile_category != category:
                    continue
                profiles.append(
                    {
                        "registry": entry.name,
                        "file": f.stem,
                        "name": profile_name,
                        "description": description,
                        "path": str(f),
                        "category": profile_category,
                    }
                )
        return profiles
