"""Job and cluster metadata storage.

Persists cluster_id → recipe mapping in ``~/.cache/sparkrun/jobs/`` so
``cluster status`` and other commands can display recipe info for
running clusters.

Identifier model
----------------

A *cluster_id* identifies a single workload execution and is the stable
reference returned by ``sparkrun.api.run``. It is split into two pieces
so load-aware schedulers (whose placement decisions are not reproducible
from CLI inputs alone) can still recover workloads at stop / logs time:

* ``intent_id`` — deterministic :data:`INTENT_ID_LEN`-char hex derived
  from ``recipe.runtime`` + ``recipe.model`` + port + served-model-name +
  every non-default parallelism dimension.  **Hosts are not hashed**, so
  the same recipe + parallelism + port always produces the same
  ``intent_id`` regardless of which hosts the scheduler picked.
* ``placement_token`` — :data:`PLACEMENT_TOKEN_LEN`-char hex
  (``secrets.token_hex(PLACEMENT_TOKEN_BYTES)``) generated at launch
  time to disambiguate multiple parallel deployments of the same
  intent.

The composite ``cluster_id = "sparkrun_" + intent_id + "_" + placement_token``
has two ``_`` separators (after ``sparkrun`` and after intent_id).

Separately, :func:`derive_recipe_fingerprint` digests a recipe's full *serve
configuration*.  It is **not** part of the cluster_id: the intent_id stays
narrow on purpose so lookup paths keep matching a live workload, while
consumers that must tell differently-configured workloads apart (benchmark
identity) hash the fingerprint alongside it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING, Optional

import yaml

from sparkrun.utils.fs import open_private_write

if TYPE_CHECKING:
    from sparkrun.core.backend_select import BackendBundle
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.recipe import Recipe
    from sparkrun.runtimes.base import RuntimePlugin

logger = logging.getLogger(__name__)


@dataclass
class JobStatus:
    """Result of checking whether a sparkrun job is running."""

    running: bool
    cluster_id: str
    healthy: bool | None = None  # None = not checked
    metadata: dict | None = None
    container_statuses: dict[str, bool] = field(default_factory=dict)
    hosts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the job status to a JSON-serializable dictionary."""
        from dataclasses import asdict

        result = asdict(self)
        if self.metadata:
            result["recipe"] = self.metadata.get("recipe")
        return result


def check_job_running(
    *,
    cluster_id: str | None = None,
    recipe: "Recipe | None" = None,
    hosts: list[str] | None = None,
    overrides: dict | None = None,
    ssh_kwargs: dict | None = None,
    cache_dir: str | None = None,
    check_http_models: bool = False,
    port: int | None = None,
) -> JobStatus:
    """Check whether a sparkrun job is currently running.

    Resolves the cluster_id (from params or recipe+hosts), checks head-node
    containers for liveness, and optionally performs an HTTP health check.

    Args:
        cluster_id: Explicit cluster ID.  If not given, generated from
            *recipe*, *hosts*, and *overrides*.
        recipe: Recipe object (used to generate cluster_id if not given).
        hosts: Host list.  Falls back to job metadata if not provided.
        overrides: Recipe overrides (port, served_model_name, etc.).
        ssh_kwargs: SSH connection parameters.
        cache_dir: Cache directory for job metadata lookup.
        check_http_models: When True and container is running, probe the
            ``/v1/models`` endpoint.
        port: Explicit port for health checks.  Falls back to metadata
            then default 8000.

    Returns:
        :class:`JobStatus` with liveness and optional health info.
    """
    from sparkrun.orchestration.executor import resolve_executor

    # Resolve cluster_id
    if cluster_id is None:
        if recipe is None or hosts is None:
            raise ValueError("Either cluster_id or both recipe and hosts must be provided")
        cluster_id = derive_cluster_id(recipe, hosts, overrides=overrides)

    # Load metadata
    meta = load_job_metadata(cluster_id, cache_dir=cache_dir)

    # Resolve hosts
    if hosts is None:
        if meta and meta.get("hosts"):
            hosts = meta["hosts"]
        else:
            return JobStatus(running=False, cluster_id=cluster_id, metadata=meta, hosts=[])

    head_host = hosts[0]
    is_solo = len(hosts) == 1

    # Determine candidate container names on the head host (preserved for
    # backward-compatible ``container_statuses`` shape).
    candidates: list[str] = []
    if is_solo:
        candidates.append("%s_solo" % cluster_id)
    else:
        # Native distributed: node_0; Ray: head
        candidates.append("%s_node_0" % cluster_id)
        candidates.append("%s_head" % cluster_id)

    # Source liveness from the executor's canonical introspection path
    # (``executor.query_status``) rather than per-container ``docker
    # inspect`` probes.  Use metadata-derived overrides so we query via
    # the same executor that launched the workload — mirrors what
    # ``api.stop`` / ``api.logs`` do.
    cli_overrides: dict | None = None
    if meta:
        meta_exec = meta.get("executor")
        meta_exec_cfg = meta.get("executor_config")
        cli_overrides = {}
        if meta_exec:
            cli_overrides["executor"] = meta_exec
        if isinstance(meta_exec_cfg, dict):
            cli_overrides.update(meta_exec_cfg)
        if not cli_overrides:
            cli_overrides = None

    executor = resolve_executor(
        cli_overrides=cli_overrides,
        rootless=False,
        auto_user=False,
    )
    status_snapshot = executor.query_status(hosts, ssh_kwargs=ssh_kwargs)

    running = cluster_id in status_snapshot.running_cluster_ids()

    # Reconstruct the legacy ``container_statuses`` dict shape.  We can't
    # recover exact container *names* from a ``RunningWorkload`` (which
    # carries docker IDs, not names), so we mark every candidate name
    # uniformly based on whether the cluster has any workload on the
    # head host.
    head_occupancy = status_snapshot.for_host(head_host)
    cluster_on_head = False
    if head_occupancy is not None:
        cluster_on_head = any(w.cluster_id == cluster_id for w in head_occupancy.workloads)

    container_statuses: dict[str, bool] = {name: cluster_on_head for name in candidates}

    # Optional health check
    healthy: bool | None = None
    if check_http_models and running:
        from sparkrun.orchestration.primitives import wait_for_healthy

        effective_port = port or (meta.get("port") if meta else None) or 8000
        url = "http://%s:%d/v1/models" % (head_host, effective_port)
        healthy = wait_for_healthy(url, max_retries=1, retry_interval=0, max_consecutive_refused=2)

    return JobStatus(
        running=running,
        cluster_id=cluster_id,
        healthy=healthy,
        metadata=meta,
        container_statuses=container_statuses,
        hosts=hosts,
    )


def _resolve_override(key: str, overrides: dict | None, defaults: dict | None):
    """Resolve a value from overrides -> recipe defaults."""
    val = overrides.get(key) if overrides else None
    if val is None and defaults:
        val = defaults.get(key)
    return val


# Cluster-id format constants.  The composite cluster_id is
# "sparkrun_<intent_id>_<placement_token>" where intent_id is
# :data:`INTENT_ID_LEN` hex chars (sha256 prefix) and placement_token is
# :data:`PLACEMENT_TOKEN_LEN` hex chars
# (``secrets.token_hex(PLACEMENT_TOKEN_BYTES)``).
INTENT_ID_LEN = 16
PLACEMENT_TOKEN_BYTES = 6  # secrets.token_hex(PLACEMENT_TOKEN_BYTES) → PLACEMENT_TOKEN_LEN hex chars
PLACEMENT_TOKEN_LEN = PLACEMENT_TOKEN_BYTES * 2
# Length of the serve-configuration digest from :func:`derive_recipe_fingerprint`.
# Not part of the cluster_id — it identifies *configuration*, not a workload.
RECIPE_FINGERPRINT_LEN = 12

_INTENT_ID_RE = re.compile(r"^[0-9a-f]{%d}$" % INTENT_ID_LEN)
_PLACEMENT_TOKEN_RE = re.compile(r"^[0-9a-f]{%d}$" % PLACEMENT_TOKEN_LEN)
_NEW_CLUSTER_ID_RE = re.compile(r"^sparkrun_([0-9a-f]{%d})_([0-9a-f]{%d})$" % (INTENT_ID_LEN, PLACEMENT_TOKEN_LEN))
# Canonical container name: ``sparkrun_<intent>_<placement>[_<role>]``.
# Used by every consumer that splits container names into (cluster_id, role):
# the Docker/local executors' ``query_status`` (the status source),
# ``cluster_manager.classify_cluster_status``, the cluster monitor TUI.
_CONTAINER_NAME_RE = re.compile(
    r"^sparkrun_(?P<intent>[0-9a-f]{%d})_(?P<placement>[0-9a-f]{%d})(?:_(?P<role>.+))?$" % (INTENT_ID_LEN, PLACEMENT_TOKEN_LEN)
)


def generate_intent_id(recipe: "Recipe", overrides: dict | None = None) -> str:
    """Deterministic :data:`INTENT_ID_LEN`-char hex *intent* identifier (no ``sparkrun_`` prefix).

    Hashes ``recipe.runtime`` + ``recipe.model`` + ``recipe.container`` + port
    + served-model-name + every non-default parallelism dimension in
    :data:`sparkrun.core.parallelism.PARALLELISM_KEYS` (tp, pp, dp, ep,
    cp).  Hosts are **not** hashed — same recipe + parallelism + port
    always yields the same intent_id regardless of scheduler placement.

    The container image is included because the intent_id is not only a
    discovery key: ``api.run`` treats a matching intent as *this launch's own
    workload*, subtracting it from the occupancy snapshot and then evicting
    it.  Two recipes serving the same model on the same port through
    different images — say a stable build and a nightly — are different
    workloads a user will reasonably want side by side, and without the image
    in the hash, launching the second silently destroyed the first.  (``--image``
    writes through to ``recipe.container``, so an image override is covered.)

    It remains deliberately narrow otherwise: serve arguments are *not*
    hashed, so ``stop`` / ``status`` / ``logs`` keep finding a live workload
    after a relaunch that only tweaked a flag.  Callers that must distinguish
    those hash :func:`derive_recipe_fingerprint` alongside this.

    Use :func:`generate_cluster_id` to compose this with a fresh
    placement token at launch time.
    """
    from sparkrun.core.parallelism import PARALLELISM_KEYS

    port = _resolve_override("port", overrides, recipe.defaults)
    served_name = _resolve_override("served_model_name", overrides, recipe.defaults)

    parts: list[str] = [recipe.runtime, recipe.model]
    # Empty/unset container (recipe relies on the runtime's default image)
    # contributes nothing, so recipes that predate an explicit container keep
    # hashing as before rather than all colliding on a placeholder.
    if getattr(recipe, "container", None):
        parts.append("image=%s" % recipe.container)
    if port is not None:
        parts.append("port=%s" % port)
    if served_name is not None:
        parts.append("name=%s" % served_name)

    # Include every non-default parallelism dimension in the hash so
    # configs that differ only in dp/ep/cp also get distinct intent IDs.
    # Iterating PARALLELISM_KEYS keeps this in lockstep with
    # save_job_metadata (single source of truth for parallelism dims).
    for long_key, short_key in PARALLELISM_KEYS:
        val = _resolve_override(long_key, overrides, recipe.defaults)
        if val is not None and int(val) != 1:
            parts.append("%s=%s" % (short_key, int(val)))

    key = "\0".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:INTENT_ID_LEN]


def derive_recipe_fingerprint(recipe: "Recipe", overrides: dict | None = None) -> str:
    """Deterministic :data:`RECIPE_FINGERPRINT_LEN`-char hex digest of a recipe's *serve configuration*.

    The provenance peer of :func:`generate_intent_id`.  The intent_id answers
    "which served endpoint is this?" and is deliberately narrow so ``stop`` /
    ``status`` / ``logs`` keep finding a live workload across relaunches; this
    answers "what exactly is being served?", so two recipes that share an
    intent — same runtime, model, port, parallelism — but differ in a serve
    argument (e.g. ``--max-num-batched-tokens``, ``--speculative-config``) are
    distinguishable.  Consumers that must not conflate differently-configured
    workloads hash this *alongside* the intent_id rather than widening the
    intent_id itself (see
    :func:`sparkrun.benchmarking.run_state.derive_benchmark_id`).

    Hashes **declared** configuration only:

    * the intent_id (runtime, model, port, served-model-name, parallelism)
    * the resolved config chain — recipe ``defaults`` layered with *overrides*,
      i.e. the serve-argument surface
    * ``container``, ``command``, ``env``, ``model_revision``,
      ``runtime_version``, ``layout``, ``min_nodes`` / ``max_nodes``.
      ``command`` matters on its own: a serve flag hardcoded into the command
      template (rather than declared under ``defaults``) never reaches the
      config chain, so hashing the template is what catches it.
    * ``mods`` and ``runtime_config`` — the latter absorbs unknown top-level
      keys such as v1 ``build_args``, which change the image that gets built
    * the recipe's *declared* ``pre_exec`` / ``post_exec`` / ``post_commands``,
      read from the raw recipe so runtime- and builder-injected hooks don't
      move the digest (v1 ``mods`` are injected into ``pre_exec`` during
      resolution, which is why they are hashed from the declared list above)

    Deliberately excluded, so the digest stays stable across relaunches of the
    same logical workload: hosts and placement, resolved container digests /
    pinned image SHAs, and ``metadata`` — the latter carries provenance plus
    *auto-detected* model facts that a HuggingFace probe writes back into
    ``recipe.metadata`` mid-run, so hashing it would make the digest depend on
    network reachability.
    """

    def _val(value: Any) -> str:
        # Canonical per-value encoding: sorts nested mapping keys (so a dict
        # value's insertion order can't move the digest) while preserving list
        # order (hook and arg sequence are semantically significant).
        return json.dumps(value, sort_keys=True, default=str)

    parts: list[str] = [generate_intent_id(recipe, overrides=overrides)]

    config_chain = recipe.build_config_chain(overrides)
    for key in sorted(config_chain.keys()):
        parts.append("%s=%s" % (key, _val(config_chain.get(key))))

    for attr in (
        "container",
        "command",
        "model_revision",
        "runtime_version",
        "min_nodes",
        "max_nodes",
        "env",
        "mods",
        "runtime_config",
    ):
        parts.append("%s=%s" % (attr, _val(getattr(recipe, attr, None))))

    layout = recipe.layout.to_dict() if getattr(recipe, "layout", None) is not None else None
    parts.append("layout=%s" % _val(layout))

    # Declared hooks only — ``recipe.pre_exec`` and friends are extended in
    # place by v1 mods / builders during resolution (see core/mods.py), and
    # those additions are resolved artifacts, not declared configuration.
    raw = getattr(recipe, "_raw", None) or {}
    for hook in ("pre_exec", "post_exec", "post_commands"):
        parts.append("%s=%s" % (hook, _val(raw.get(hook) or [])))

    key = "\0".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:RECIPE_FINGERPRINT_LEN]


def generate_placement_token() -> str:
    """Generate a fresh placement token (:data:`PLACEMENT_TOKEN_LEN`-char hex string).

    Each launch gets its own token so multiple parallel deployments of
    the same intent on different host sets are distinguishable.  Format
    is ``secrets.token_hex(PLACEMENT_TOKEN_BYTES)``.
    """
    return secrets.token_hex(PLACEMENT_TOKEN_BYTES)


def derive_placement_token_from_hosts(hosts: "list[str] | tuple[str, ...]") -> str:
    """Deterministic placement_token derived from a host set.

    Used by lookup-style call sites (status / stop / logs / ensure)
    that need a stable cluster_id from a ``(recipe, hosts)`` pair
    without consulting a launcher.  Hosts are sorted before hashing so
    ordering does not affect the result.
    """
    host_key = "\0".join(sorted(str(h) for h in hosts))
    return hashlib.sha256(host_key.encode()).hexdigest()[:PLACEMENT_TOKEN_LEN]


def derive_cluster_id(recipe: "Recipe", hosts: "list[str] | tuple[str, ...]", overrides: dict | None = None) -> str:
    """Deterministic cluster_id from ``(recipe, hosts)``.

    Convenience for lookup paths: composes :func:`generate_intent_id`
    with :func:`derive_placement_token_from_hosts` so callers that need
    the "same recipe + hosts → same cluster_id" lookup semantics don't
    have to repeat the derivation themselves.  New launches via
    :func:`sparkrun.api.run` use :func:`generate_placement_token`
    instead, so derived and live cluster_ids occupy disjoint token
    spaces and never collide.
    """
    intent_id = generate_intent_id(recipe, overrides=overrides)
    placement_token = derive_placement_token_from_hosts(hosts)
    return generate_cluster_id(intent_id, placement_token)


def generate_cluster_id(intent_id: str, placement_token: str) -> str:
    """Compose a sparkrun cluster identifier from *intent_id* and *placement_token*.

    Returns ``"sparkrun_<intent_id>_<placement_token>"``.  Both inputs
    are validated against :data:`INTENT_ID_LEN` / :data:`PLACEMENT_TOKEN_LEN`;
    malformed values raise :class:`ValueError`.
    """
    if not isinstance(intent_id, str) or not _INTENT_ID_RE.fullmatch(intent_id):
        raise ValueError("intent_id must be %d hex chars, got %r" % (INTENT_ID_LEN, intent_id))
    if not isinstance(placement_token, str) or not _PLACEMENT_TOKEN_RE.fullmatch(placement_token):
        raise ValueError("placement_token must be %d hex chars, got %r" % (PLACEMENT_TOKEN_LEN, placement_token))
    return "sparkrun_%s_%s" % (intent_id, placement_token)


def parse_cluster_id(cluster_id: str) -> tuple[str, str]:
    """Decompose *cluster_id* into ``(intent_id, placement_token)``.

    Accepts only the canonical
    ``sparkrun_<intent_id>_<placement_token>`` form (with hex segments
    of length :data:`INTENT_ID_LEN` / :data:`PLACEMENT_TOKEN_LEN`).
    Raises :class:`ValueError` for anything else.
    """
    m = _NEW_CLUSTER_ID_RE.match(cluster_id)
    if m:
        return m.group(1), m.group(2)
    raise ValueError("Not a sparkrun cluster_id: %r" % cluster_id)


def is_cluster_id(cluster_id: str) -> bool:
    """``True`` when *cluster_id* parses as the canonical sparkrun format."""
    return _NEW_CLUSTER_ID_RE.fullmatch(cluster_id) is not None


def parse_container_name(name: str) -> tuple[str, str] | None:
    """Decompose a container name into ``(cluster_id, role)``.

    Accepts the canonical
    ``sparkrun_<intent_id>_<placement_token>[_<role>]`` form, plus the
    ``..._solo`` shorthand for single-container launches.  Returns
    ``None`` for names that don't parse so callers can keep an
    "unknown" branch without try/except.

    The ``cluster_id`` returned is the full
    ``sparkrun_<intent>_<placement>`` — distinct workloads of the same
    recipe replay (same intent, different placement token) parse to
    distinct cluster_ids.
    """
    if name.endswith("_solo"):
        return (name.removesuffix("_solo"), "solo")
    m = _CONTAINER_NAME_RE.match(name)
    if m is None:
        return None
    cluster_id = "sparkrun_%s_%s" % (m.group("intent"), m.group("placement"))
    role = m.group("role") or "?"
    return (cluster_id, role)


def save_job_metadata(
    cluster_id: str,
    recipe: "Recipe",
    hosts: list[str],
    overrides: dict | None = None,
    cache_dir: str | None = None,
    ib_ip_map: dict[str, str] | None = None,
    mgmt_ip_map: dict[str, str] | None = None,
    recipe_ref: str | None = None,
    runtime_info: dict[str, str] | None = None,
    container_image: Optional[str] = None,
    runtime: "RuntimePlugin | None" = None,
    backends: "dict[str, BackendBundle] | None" = None,
    *,
    recipe_fingerprint: str | None = None,
    owner: str | None = None,
    cluster_name: str | None = None,
    ssh_user: str | None = None,
    sctx: "SparkrunContext | None" = None,
) -> None:
    """Persist job metadata so ``cluster status`` can display recipe info.

    Writes a small YAML file to ``{cache_dir}/jobs/{digest}.yaml`` where
    *digest* is the portion of *cluster_id* after the ``sparkrun_``
    prefix.

    Args:
        backends: Per-host backend bundles resolved by the launcher.
            Persisted as ``{host: {vendor, backend}}`` so ``stop``/``logs``
            can recover the collective backend without re-probing.
        recipe_fingerprint: Pre-computed :func:`derive_recipe_fingerprint`
            digest to persist verbatim.  **Callers that need the digest to
            match a value they computed themselves must pass it**: by the time
            the launcher saves metadata it has already folded platform
            runtime-flag defaults into ``recipe.defaults``
            (:func:`sparkrun.core.launcher.apply_platform_runtime_flag_defaults`),
            so deriving here would digest a *host-dependent* recipe that no
            caller can reproduce without probing the same hardware.
        owner: Opaque tag naming the component that created this job (e.g. an
            automated supervisor).  Lets it distinguish workloads it launched
            from identically-configured ones a human started, and refuse to
            tear the latter down.  ``None`` omits the key, which every
            pre-existing job also reads as.
        cluster_name: Name of the cluster this workload was launched on
            (empty / ``None`` for an anonymous ``--hosts`` launch).  It is
            the job's durable *connection* identity: ``stop`` / ``logs``
            addressed by cluster_id resolve hosts from ``hosts`` above and
            so name no cluster, which left them on an anonymous definition
            carrying no SSH user, no executor pin and no transport — the
            teardown then ran as the control node's login and reported
            success while the workload kept serving (issue #277).
        ssh_user: The SSH user this launch actually connected as.  The
            fallback for when the recorded cluster can no longer be
            resolved (renamed, deleted, or a different control node), and
            the only identity an anonymous launch has.  Omitted when the
            launch fell through to the ssh client's own default, so
            "recorded" always means "we know", never "we guessed".
        sctx: Optional shared :class:`SparkrunContext`.  When provided
            (and *cache_dir* is unset) ``sctx.config.cache_dir`` is the
            cache root.
    """
    cache_dir = _resolve_cache_dir(cache_dir, sctx)

    digest = _filename_digest(cluster_id)
    jobs_dir = Path(cache_dir) / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    # Metadata can carry the resolved upstream API key (see below), so keep the
    # directory owner-only.  Best-effort: a pre-existing dir from an older
    # sparkrun is also tightened here.
    try:
        os.chmod(jobs_dir, 0o700)
    except OSError:
        logger.debug("Could not chmod 0700 %s", jobs_dir, exc_info=True)

    from sparkrun.core.parallelism import PARALLELISM_KEYS

    # Stamp the metadata with the producing sparkrun version so future
    # readers can detect schema/cluster-id format drift and migrate or
    # warn appropriately.  See ``load_job_metadata`` for the read side.
    try:
        from sparkrun import __version__ as _sparkrun_version
    except Exception:
        _sparkrun_version = "unknown"

    # Decompose cluster_id so callers can index by intent_id /
    # placement_token without re-parsing.  Any non-canonical cluster_id
    # is a caller bug; let the :class:`ValueError` from
    # :func:`parse_cluster_id` propagate.
    intent_id_meta, placement_token_meta = parse_cluster_id(cluster_id)

    meta: dict = {
        "sparkrun_version": _sparkrun_version,
        "cluster_id": cluster_id,
        "recipe": recipe.qualified_name,
        # Deriving here is the fallback for callers that do not pass one; see
        # the argument's docstring for why the launcher must.
        "recipe_fingerprint": recipe_fingerprint or derive_recipe_fingerprint(recipe, overrides),
        "model": recipe.model,
        "runtime": recipe.runtime,
        "hosts": hosts,
        "intent_id": intent_id_meta,
        # Launch time, so consumers can order jobs by recency.  The read side
        # (``api.list_jobs``) has always looked for this key; nothing wrote it,
        # so every job's ``started_at`` was ``None`` and the documented
        # "most recent first" ordering silently degraded to alphabetical by
        # cluster_id.  Recorded here rather than derived from file mtime
        # because mtime is only a proxy — a rewrite (relaunch of the same
        # cluster_id, a backup restore, an rsync of the cache) moves it.
        "started_at": time.time(),
        "placement_token": placement_token_meta,
    }
    if recipe_ref:
        meta["recipe_ref"] = recipe_ref
    # Omitted rather than written empty: the read side must be able to tell
    # "nobody claimed this job" from "written before sparkrun recorded owners".
    if owner:
        meta["owner"] = owner

    # How to reach this job again.  ``hosts`` records *where* it runs; these
    # record *as what* — see the argument docs above.  Both are omitted when
    # unknown rather than written empty, so the read side can tell "anonymous
    # launch" from "launched before sparkrun recorded this".
    if cluster_name:
        meta["cluster"] = str(cluster_name)
    if ssh_user:
        meta["ssh_user"] = str(ssh_user)

    # Store all parallelism values (not just tensor_parallel)
    for long_key, _ in PARALLELISM_KEYS:
        val = _resolve_override(long_key, overrides, recipe.defaults)
        if val is not None:
            meta[long_key] = int(val)
    # Persist port for proxy discovery
    port = None
    if overrides:
        port = overrides.get("port")
    if port is None and recipe.defaults:
        port = recipe.defaults.get("port")
    if port is not None:
        meta["port"] = int(port)

    # Persist served_model_name for proxy discovery.  The command-template
    # fallback matters here as much as in the benchmark: the proxy *routes* on
    # this name, so a recipe that hardcodes --served-model-name in command:
    # would be advertised (and proxied) under the model id the server rejects.
    served_name = None
    if overrides:
        served_name = overrides.get("served_model_name")
    if served_name is None and recipe.defaults:
        served_name = recipe.defaults.get("served_model_name")
    if served_name is None:
        from sparkrun.core.recipe import extract_served_model_name_from_command

        served_name = extract_served_model_name_from_command(getattr(recipe, "command", None))
    if served_name is not None:
        meta["served_model_name"] = str(served_name)

    # Resolve upstream API key via the runtime plugin so proxy discovery
    # can authenticate to the inference endpoint.  Runtimes that don't
    # support api-keys return None from resolve_api_key().
    if runtime is not None:
        try:
            api_key = runtime.resolve_api_key(recipe, overrides)
        except Exception:
            logger.debug("resolve_api_key failed for %s", cluster_id, exc_info=True)
            api_key = None
        if api_key:
            meta["api_key"] = str(api_key)

    if ib_ip_map:
        meta["ib_ip_map"] = ib_ip_map
    if mgmt_ip_map:
        meta["mgmt_ip_map"] = mgmt_ip_map
    if runtime_info:
        meta["runtime_info"] = runtime_info
    if container_image:
        meta["effective_container_image"] = container_image

    # Persist per-host backend bundle so stop/logs can recover collective
    # backend selection without re-probing hardware.  Schema:
    #   backends: { host: { vendor, backend } }
    if backends:
        meta["backends"] = {
            host: {"vendor": bundle.accelerator_vendor, "backend": bundle.collective.name} for host, bundle in backends.items()
        }

    # Persist executor selection so stop/logs can reproduce the same
    # executor (Docker vs experimental local) without re-running the
    # launcher's resolution logic.
    executor_selector = recipe.executor or ""
    if executor_selector:
        meta["executor"] = executor_selector
    recipe_exec_cfg = recipe.executor_config
    if isinstance(recipe_exec_cfg, dict) and recipe_exec_cfg:
        meta["executor_config"] = dict(recipe_exec_cfg)

    # Full overrides dict for export reconstruction
    if overrides:
        meta["overrides"] = dict(overrides)

    # Serialize full recipe state for faithful export reconstruction.
    try:
        meta["recipe_state"] = recipe.__getstate__()
    except Exception:
        logger.debug("Failed to serialize recipe state for %s", cluster_id, exc_info=True)

    meta_path = jobs_dir / f"{digest}.yaml"
    # ``meta`` may hold the resolved upstream ``api_key`` (and a full recipe
    # state that can include env secrets), so create the file owner-only from
    # the start — never a umask-default 0644 window where another local user
    # could read the key.  O_TRUNC mirrors the previous "w" overwrite semantics.
    # O_NOFOLLOW refuses to write through a symlink: if another local user
    # pre-planted ``<digest>.yaml`` as a link to a file they can read, the open
    # fails (ELOOP) rather than leaking the key through the link's target.
    # (open_private_write applies it only where it exists — naming it directly
    # is an AttributeError on a Windows control node, which meant no job
    # metadata was written there at all.)
    fd = open_private_write(meta_path)
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(meta, f, default_flow_style=False)
    # If the file pre-existed as a regular file with looser perms, O_CREAT won't
    # re-chmod it; tighten explicitly (best-effort).  O_NOFOLLOW above already
    # guaranteed the fd is not a symlink, so this chmod can't be redirected.
    try:
        os.chmod(meta_path, 0o600)
    except OSError:
        logger.debug("Could not chmod 0600 %s", meta_path, exc_info=True)
    logger.debug("Saved job metadata to %s", meta_path)


def remove_job_metadata(
    cluster_id: str,
    cache_dir: str | None = None,
    *,
    sctx: "SparkrunContext | None" = None,
) -> None:
    """Delete the cached job metadata file for a cluster_id.

    No-op if the file does not exist.  When *cache_dir* is unset, the
    cache root is resolved from ``sctx.config.cache_dir`` (when *sctx*
    is provided) and falls back to :data:`DEFAULT_CACHE_DIR`.
    """
    cache_dir = _resolve_cache_dir(cache_dir, sctx)
    digest = _filename_digest(cluster_id)
    meta_path = Path(cache_dir) / "jobs" / f"{digest}.yaml"
    meta_path.unlink(missing_ok=True)
    logger.debug("Removed job metadata %s", meta_path)


#: Filename (under the cache root) of the last observed occupancy snapshot.
RUNNING_SNAPSHOT_FILE = "running.json"

#: How long a recorded snapshot is trusted.  Short, because it is used to
#: *hide* things: a workload that died five minutes ago should stop being
#: offered, and one launched from another terminal should start being offered.
RUNNING_SNAPSHOT_MAX_AGE_S = 600


def save_running_snapshot(
    cluster_ids: "set[str] | frozenset[str] | tuple[str, ...] | list[str]",
    hosts: "list[str] | tuple[str, ...]",
    *,
    cache_dir: str | None = None,
    sctx: "SparkrunContext | None" = None,
) -> None:
    """Record which workloads were observed running, and where we looked.

    Shell completion cannot afford an SSH sweep — it runs on every TAB, and a
    host that no longer resolves would hang the terminal with no way to signal
    what it is waiting on.  But several commands (``run``, ``status``,
    ``stop``) already pay for a sweep, so they can leave the answer behind for
    completion to read for free.

    *hosts* is recorded alongside because a sweep is frequently **partial** —
    placement queries a candidate subset, not the whole cluster.  Without it a
    reader cannot distinguish "not running" from "not looked at", and would
    silently hide a live workload on an unswept host.

    Best-effort and silent on failure: this is a convenience cache, and no
    command should fail because it could not be written.
    """
    import json

    cache_dir = _resolve_cache_dir(cache_dir, sctx)
    payload = {
        "at": time.time(),
        "cluster_ids": sorted(str(c) for c in cluster_ids if c),
        "hosts": sorted(str(h) for h in hosts if h),
    }
    try:
        path = Path(cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        fd = open_private_write(path / RUNNING_SNAPSHOT_FILE)
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
    except Exception:
        logger.debug("Could not write running snapshot", exc_info=True)


def load_running_snapshot(
    *,
    cache_dir: str | None = None,
    max_age_s: float | None = None,
    sctx: "SparkrunContext | None" = None,
) -> "tuple[frozenset[str], frozenset[str]] | None":
    """Read the last observed occupancy snapshot.

    Returns ``(cluster_ids, hosts_covered)``, or ``None`` when there is no
    snapshot or it is older than *max_age_s* — in which case callers must fall
    back to showing everything rather than hiding what they cannot vouch for.

    *max_age_s* resolves to :data:`RUNNING_SNAPSHOT_MAX_AGE_S` at call time
    rather than binding it as a default, so the module constant is a real knob
    instead of a value frozen when this function was defined.
    """
    import json

    if max_age_s is None:
        max_age_s = RUNNING_SNAPSHOT_MAX_AGE_S
    cache_dir = _resolve_cache_dir(cache_dir, sctx)
    try:
        with open(Path(cache_dir) / RUNNING_SNAPSHOT_FILE) as f:
            data = json.load(f)
        if time.time() - float(data["at"]) > max_age_s:
            return None
        return frozenset(data.get("cluster_ids") or ()), frozenset(data.get("hosts") or ())
    except Exception:
        return None


#: Jobs older than this are candidates for pruning.
PRUNE_MAX_AGE_DAYS = 30

#: …unless they are among this many most recent for their intent.
PRUNE_KEEP_PER_INTENT = 3


def prune_job_metadata(
    *,
    cache_dir: str | None = None,
    max_age_days: int = PRUNE_MAX_AGE_DAYS,
    keep_per_intent: int = PRUNE_KEEP_PER_INTENT,
    protected_cluster_ids: "set[str] | frozenset[str] | tuple[str, ...] | None" = None,
    dry_run: bool = False,
    sctx: "SparkrunContext | None" = None,
) -> list[str]:
    """Delete stale job metadata, returning the cluster_ids removed.

    The cache is append-only: every launch writes a file and only an explicit
    ``stop`` (or the ``logs`` staleness path) removes one, so a crashed job —
    the common case under ``auto_remove``, where the container is gone before
    anything asks about it — accumulates forever.  In practice that reaches
    hundreds of dead entries against a couple of dozen live intents, which
    makes the cache useless as a completion source and slow to read.

    A job is **kept** when it is both:

    - among the *keep_per_intent* most recent for its ``intent_id`` — so every
      workload you actually run keeps a short history rather than being
      erased wholesale, and
    - younger than *max_age_days*.

    Anything else is deleted.  ``keep_per_intent`` is a per-intent recency
    window rather than a global count on purpose: a global "keep newest N"
    would silently drop every trace of an intent you run rarely.

    ``protected_cluster_ids`` is never deleted regardless of age. Callers pass
    the cluster_ids they have just observed **running**, which is what makes
    this safe to run automatically: a live workload's metadata is load-bearing
    for ``stop`` / ``logs`` / proxy discovery, and deleting it would strand the
    deployment. Age alone is not a sufficient guard — a long-lived server can
    easily outlive the cutoff.

    Best-effort: an unreadable or undeletable file is skipped, never raised.

    Args:
        max_age_days: Age cutoff in days. ``0`` disables the age test, making
            *keep_per_intent* the only rule.
        dry_run: Report what would be deleted without deleting it.

    Returns:
        cluster_ids removed (or that would be, under *dry_run*), most recent
        first.
    """
    from sparkrun.api._jobs import list_jobs

    cache_dir = _resolve_cache_dir(cache_dir, sctx)
    protected = set(protected_cluster_ids or ())

    try:
        jobs = list_jobs(cache_dir=cache_dir)
    except Exception:
        logger.debug("prune_job_metadata: could not list jobs", exc_info=True)
        return []

    cutoff = (time.time() - max_age_days * 86400) if max_age_days else None
    seen_per_intent: dict[str, int] = {}
    removed: list[str] = []

    # `list_jobs` is already recency-descending, so the per-intent counter
    # walks newest-first and the first `keep_per_intent` entries it sees for
    # an intent are exactly the ones to keep.
    for job in jobs:
        intent = job.intent_id or job.cluster_id
        rank = seen_per_intent.get(intent, 0)
        seen_per_intent[intent] = rank + 1

        if job.cluster_id in protected:
            continue
        # Both conditions must hold to keep — an entry that is recent for its
        # intent but ancient in absolute terms is still ancient.  (Making this
        # an "or" instead keeps the newest K of every intent forever, which
        # leaves a cache that never shrinks below one entry per intent ever
        # launched.)
        is_recent_for_intent = rank < keep_per_intent
        is_young = cutoff is None or (job.started_at or 0.0) >= cutoff
        if is_recent_for_intent and is_young:
            continue

        if not dry_run:
            try:
                remove_job_metadata(job.cluster_id, cache_dir=cache_dir)
            except Exception:
                logger.debug("prune_job_metadata: failed to remove %s", job.cluster_id, exc_info=True)
                continue
        removed.append(job.cluster_id)

    if removed:
        logger.debug("prune_job_metadata: %s %d job(s)", "would remove" if dry_run else "removed", len(removed))
    return removed


def load_job_metadata(
    cluster_id: str,
    cache_dir: str | None = None,
    *,
    sctx: "SparkrunContext | None" = None,
) -> dict | None:
    """Load job metadata for a cluster_id.  Returns ``None`` if not found.

    When *cache_dir* is unset, the cache root is resolved from
    ``sctx.config.cache_dir`` (when *sctx* is provided) and falls back
    to :data:`DEFAULT_CACHE_DIR`.

    Metadata schema may evolve across sparkrun versions; readers can
    inspect ``data["sparkrun_version"]`` to detect potential drift and
    handle migration.  Today this function returns the data verbatim;
    a version-mismatch policy can land here later.
    """
    cache_dir = _resolve_cache_dir(cache_dir, sctx)
    digest = _filename_digest(cluster_id)
    meta_path = Path(cache_dir) / "jobs" / f"{digest}.yaml"
    if not meta_path.exists():
        return None
    try:
        from sparkrun.utils import load_yaml

        data = load_yaml(meta_path)
        return data or None
    except Exception:
        logger.debug("Failed to load job metadata for %s", cluster_id, exc_info=True)
        return None


def _filename_digest(cluster_id: str) -> str:
    """Return the metadata filename stem for *cluster_id*.

    Strips the ``sparkrun_`` prefix when present; otherwise returns
    *cluster_id* verbatim so caller-supplied bare digests still
    round-trip.
    """
    return cluster_id.removeprefix("sparkrun_")


def _resolve_cache_dir(cache_dir: str | None, sctx: "SparkrunContext | None") -> str:
    """Resolve the effective cache root for job-metadata I/O.

    Priority: explicit *cache_dir* > ``sctx.config.cache_dir`` > module
    default :data:`DEFAULT_CACHE_DIR`.  Used by every public function in
    this module so the resolution chain stays consistent.
    """
    if cache_dir is not None:
        return cache_dir
    if sctx is not None:
        try:
            return str(sctx.config.cache_dir)
        except Exception:
            logger.debug("sctx.config.cache_dir unavailable; using default", exc_info=True)
    from sparkrun.core.config import DEFAULT_CACHE_DIR

    return str(DEFAULT_CACHE_DIR)
