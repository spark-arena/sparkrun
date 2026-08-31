"""Common inference launch pipeline.

Shared by ``sparkrun run``, ``sparkrun benchmark``, and
``sparkrun proxy load``.  Callers are responsible for recipe loading,
host resolution, override building, and node trimming *before*
calling :func:`launch_inference`.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from sparkrun.core.backend_select import BackendBundle
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.progress import LaunchProgress
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.registry import RegistryManager
    from sparkrun.orchestration.comm_env import ClusterCommEnv
    from sparkrun.runtimes.base import RuntimePlugin
    from sparkrun.builders.base import BuilderPlugin

logger = logging.getLogger(__name__)


@dataclass
class LaunchResult:
    """Result of :func:`launch_inference`."""

    rc: int
    cluster_id: str
    host_list: list[str]
    is_solo: bool
    runtime: RuntimePlugin
    recipe: Recipe
    overrides: dict[str, Any]
    container_image: str
    effective_cache_dir: str
    serve_port: int
    config: SparkrunConfig
    recipe_ref: str | None = None
    comm_env: "ClusterCommEnv | None" = None
    ib_ip_map: dict[str, str] = field(default_factory=dict)
    ib_iface_map: dict[str, str] = field(default_factory=dict)
    serve_command: str = ""
    runtime_info: dict[str, str] = field(default_factory=dict)
    builder: BuilderPlugin | None = None
    backends: dict[str, "BackendBundle"] = field(default_factory=dict)
    """Per-host backend bundles resolved from fingerprint/hardware metadata.

    Populated when at least one host's hardware resolved cleanly through
    :func:`sparkrun.core.backend_select.select_backends`.  Empty dict
    when no resolution was performed (e.g. caller bypassed cluster
    threading) — runtimes then fall back to the legacy NCCL generator
    in :func:`sparkrun.runtimes._cluster_ops.resolve_comm_env`.
    """


def resolve_recipe_trust(recipe: Recipe, trust_cli: bool) -> bool:
    """Decide whether recipe hooks (pre_exec/post_exec/post_commands) are trusted.

    A recipe is trusted when any of these hold:

    * the user passed ``--trust`` on the CLI (``trust_cli=True``);
    * the recipe was loaded from a local filesystem path (no
      ``source_registry`` *and* not fetched from a URL);
    * the recipe came from a registry that the local ``registries.yaml``
      marks as ``trusted: true`` (per-registry opt-in stored in
      :class:`sparkrun.core.registry.RegistryEntry`).

    Recipes fetched from a remote URL (``recipe.is_url_sourced``) are
    **never** auto-trusted, even though they carry no ``source_registry``:
    a "run this link" recipe must not silently execute its hooks. They
    require ``--trust`` or interactive confirmation.

    Trust is a **local** decision: it lives in the user's
    ``~/.config/sparkrun/registries.yaml``.  A manifest in the source
    repository cannot grant itself trust; the user must opt in either via
    the bootstrap defaults (registries shipped as ``trusted=True`` in
    :data:`sparkrun.core.registry.FALLBACK_DEFAULT_REGISTRIES`) or
    explicitly via ``sparkrun registry trust <name>`` /
    ``sparkrun registry add --trust <url>``.

    Args:
        recipe: The loaded recipe (used for ``source_registry``
            introspection).
        trust_cli: CLI ``--trust`` flag value.

    Returns:
        True when the hook commands may run without per-launch
        confirmation, False when they should be gated by an interactive
        prompt.
    """
    if trust_cli:
        return True
    # URL-sourced recipes carry no source_registry but must not be
    # auto-trusted — they are the least-trustworthy source.  Direct
    # attribute access (not getattr-with-default) so a future rename
    # fails loudly in tests rather than silently auto-trusting.
    if recipe.is_url_sourced:
        return False
    if recipe.source_registry is None:
        return True  # local filesystem recipe
    # Any failure to consult the local registries.yaml → untrusted.
    try:
        from sparkrun.core.config import SparkrunConfig
        from sparkrun.core.registry import RegistryError

        mgr = SparkrunConfig().get_registry_manager()
        entry = mgr.get_registry(recipe.source_registry)
        return bool(entry.trusted)
    except RegistryError:
        return False
    except Exception:
        logger.debug("resolve_recipe_trust: failed to consult registries.yaml", exc_info=True)
        return False


#: ``executor_config`` keys an untrusted recipe must not set.  Each maps to a
#: ``docker run`` flag that can break container isolation or expose host state:
#: ``privileged``/``cap_add``/``security_opt`` defeat the rootless hardening;
#: ``devices`` grants raw device access (``/dev/mem``, ``/dev/sda``); ``user``
#: can run the container as root; ``volumes`` bind-mounts host paths.  Honouring
#: any of these for a "run this link" recipe is a host-takeover primitive, so we
#: fail closed unless the recipe is trusted.  Innocuous resource knobs
#: (``shm_size``, ``ipc``, ``network``, ``memory_limit``, ``ulimit``,
#: ``restart_policy``, ``auto_remove``, ``labels`` …) are intentionally absent.
_TRUST_GATED_EXECUTOR_KEYS = frozenset({"privileged", "cap_add", "security_opt", "devices", "user", "volumes"})

#: Executors an untrusted recipe is allowed to select.  Only the Docker executor
#: runs the workload inside a rootless, namespaced container — the sandbox that
#: justifies running a registry/URL recipe's serve ``command`` without a prompt.
#: ``local`` runs the command natively via ``setsid bash -c`` (no container at
#: all) and ``k8s`` wedges it into ``kubectl run``; selecting either from an
#: untrusted recipe is arbitrary host code execution, so they require trust.
_TRUSTED_DEFAULT_EXECUTORS = frozenset({"", "docker"})


def _enforce_recipe_mount_trust(recipe: Recipe, trusted: bool) -> None:
    """Reject untrusted recipes that try to escape container isolation.

    Several recipe-controlled surfaces can expose host state to the workload
    container or defeat the rootless-by-default hardening:

    * ``cluster_config`` — the (undocumented) ``resolved_model_path`` /
      ``remote_cache_dir`` / ``local_cache_dir`` launch overrides, which
      identity-mount a host directory and repoint the serve argument at it;
    * ``executor_config`` privilege keys (:data:`_TRUST_GATED_EXECUTOR_KEYS`) —
      ``privileged``/``cap_add``/``security_opt``/``devices``/``user``/``volumes``.
      These sit *above* the executor's rootless ``apply_runtime_adjustments``
      layer in :func:`resolve_executor`'s chain, so a recipe that sets them wins
      over the hardening (e.g. ``privileged: true`` → ``docker run --privileged``).

    All are infra-level escape hatches.  Honouring them for an untrusted
    (registry- or URL-sourced) recipe would let "run this link" mount ``/``,
    grant ``--privileged``, or pass through ``/dev/mem`` and take over the host.
    They are allowed only for trusted recipes (local, default-registry, or
    ``--trust``); an untrusted recipe that sets them fails closed here, at the
    single launch choke point shared by ``run``/``benchmark``/``proxy``.

    Raises:
        RecipeError: when an untrusted recipe declares any gated surface.
    """
    if trusted:
        return
    from sparkrun.core.recipe import RecipeError, is_local_model_path

    # Direct attribute access (not getattr-with-default): these are always set
    # by ``Recipe.__init__`` / ``__setstate__``, so a rename should raise loudly
    # here rather than silently disabling the security gate.
    cc = recipe.cluster_config
    if cc is not None and not cc.is_empty():
        raise RecipeError(
            "This recipe sets cluster_config host/path overrides (pre-placed model "
            "path or cache-dir redirection), which can expose host paths to the "
            "container. They are only honoured for trusted recipes; re-run with "
            "--trust after auditing this recipe."
        )

    # An absolute path in ``model:`` is sugar for ``resolved_model_path``: it
    # identity-mounts that host directory into the container (see
    # ``resolved_model_volume``).  Same host-exposure risk as the cluster_config
    # hatch above, so gate it the same way — an untrusted "run this link" recipe
    # must not be able to mount an arbitrary host path via ``model:``.
    if is_local_model_path(recipe.model):
        raise RecipeError(
            "This recipe's model is an absolute host path (%r), which identity-mounts "
            "that host directory into the container. It is only honoured for trusted "
            "recipes; re-run with --trust after auditing this recipe." % recipe.model
        )
    exec_cfg = recipe.executor_config
    if isinstance(exec_cfg, dict):
        # Gate on *presence* of the key, not truthiness: ``security_opt: []`` is a
        # deliberate attempt to clear the no-new-privileges hardening, so an empty
        # list must be caught too.
        gated = sorted(k for k in _TRUST_GATED_EXECUTOR_KEYS if k in exec_cfg)
        if gated:
            raise RecipeError(
                "This recipe sets privileged executor_config keys %s, which can "
                "break container isolation (extra host bind mounts, --privileged, "
                "raw --device access, or running as root). They are only honoured "
                "for trusted recipes; re-run with --trust after auditing this "
                "recipe." % gated
            )

    # The executor *selector* is itself an isolation control: only ``docker``
    # runs the serve command inside a rootless container.  ``local`` runs it
    # natively (``setsid bash -c``) and ``k8s`` via ``kubectl run`` — both with
    # no container boundary — so an untrusted recipe selecting either is direct
    # host code execution.  Check both the dedicated ``executor`` field and the
    # selector smuggled through ``executor_config`` (``executor`` /
    # ``executor_type``, the same keys ``ExecutorConfig.from_chain`` reads).
    # ``Recipe.__init__`` always stores ``executor`` as a str; the isinstance
    # guard only keeps non-str sentinels (e.g. test mocks) from being coerced
    # into a spurious selector — a real untrusted recipe cannot reach here with
    # a non-str executor.
    raw_selected = recipe.executor if isinstance(recipe.executor, str) else ""
    selected = raw_selected.strip().lower()
    if isinstance(exec_cfg, dict):
        smuggled = exec_cfg.get("executor") or exec_cfg.get("executor_type")
        if smuggled and not selected and isinstance(smuggled, str):
            selected = smuggled.strip().lower()
    if selected not in _TRUSTED_DEFAULT_EXECUTORS:
        raise RecipeError(
            "This recipe selects the %r executor, which runs the workload outside "
            "a Docker container (no rootless / namespace isolation). It is only "
            "honoured for trusted recipes; re-run with --trust after auditing this "
            "recipe." % selected
        )


def _verify_pre_placed_model(recipe, hosts, ssh_kwargs, *, runtime, cluster, config, overrides) -> None:
    """Fail fast when pre-placed model weights are missing on the target substrate.

    An absolute-path ``model:`` or ``cluster_config.resolved_model_path`` tells
    sparkrun the weights already exist at that path on every node, so download +
    distribution are skipped and the path is identity-mounted.  This verifies
    that promise *before* the skip via the launching executor's
    :meth:`~sparkrun.orchestration.executors._base.Executor.verify_mount_sources`
    (host substrate → SSH ``test -e``; provider executors probe their own
    volumes).  Raises :class:`RecipeError` listing the host→path gaps.

    Best-effort and non-fatal by design: an unresolvable executor (e.g. a
    gated-off provider) or an unreachable/unverifiable host is *skipped* rather
    than blocking the launch — only a *confirmed*-missing path raises.  The
    caller guards ``dry_run`` (no SSH).
    """
    from sparkrun.core.recipe import RecipeError
    from sparkrun.orchestration.executor import resolve_executor
    from sparkrun.orchestration.primitives import resolved_model_volume

    paths = list(resolved_model_volume(recipe))  # identity-mount source path(s)
    if not paths:
        return

    try:
        executor = resolve_executor(recipe=recipe, cluster=cluster, runtime=runtime, config=config, cli_overrides=overrides)
    except Exception:
        # If the executor can't be resolved here, the runtime will surface the
        # real error at launch — don't pre-empt it with a preflight failure.
        logger.debug("pre-placed model preflight: executor unresolvable; skipping probe", exc_info=True)
        return

    try:
        missing = executor.verify_mount_sources(paths, hosts, ssh_kwargs=ssh_kwargs) or {}
    except Exception:
        logger.debug("pre-placed model preflight: probe failed; skipping", exc_info=True)
        return

    if missing:
        detail = "; ".join("%s: %s" % (host, ", ".join(miss)) for host, miss in sorted(missing.items()))
        raise RecipeError(
            "Pre-placed model weights were not found on the target host(s). The model "
            "path must already exist on every node (download + distribution are skipped "
            "for on-disk weights). Missing — %s" % detail
        )


def _verify_image_command_passthrough(
    recipe,
    image,
    hosts,
    ssh_kwargs,
    *,
    runtime,
    cluster,
    config,
    executor_config,
    rootless,
    auto_user,
    host_hardware,
    v,
) -> None:
    """Fail fast when the image's ENTRYPOINT would swallow sparkrun's command.

    sparkrun appends its launcher as CMD *arguments*, so an image whose
    ENTRYPOINT consumes them (``ENTRYPOINT ["vllm","serve"]``) runs a different
    program than intended — while the passthrough wrappers most NGC images ship
    (``/opt/nvidia/nvidia_entrypoint.sh``, ending in ``exec "$@"``) are fine and
    must be left alone.  Only a probe can tell them apart; see
    :meth:`~sparkrun.orchestration.executors._base.Executor.verify_command_passthrough`.

    Fails closed rather than auto-clearing the ENTRYPOINT.  The probe does
    establish that clearing it *works*, but not that clearing it is *harmless* —
    a consuming entrypoint may also perform setup the workload needs — so this
    names both supported fixes and leaves the choice to the operator.

    The executor is resolved with the same arguments the launch itself uses
    below, so the probe container starts under the launch's own accelerator
    flags (``host_hardware`` is what pins DGX Spark to ``--gpus`` over CDI).

    Best-effort, matching :func:`_verify_pre_placed_model`: an unresolvable
    executor, an unreachable host, or any probe error is skipped rather than
    blocking.  Only a *confirmed* consuming entrypoint raises.
    """
    from sparkrun.core.recipe import RecipeError
    from sparkrun.orchestration.executor import resolve_executor

    if not image or not hosts:
        return

    try:
        executor = resolve_executor(
            recipe=recipe,
            cluster=cluster,
            runtime=runtime,
            config=config,
            cli_overrides=executor_config if isinstance(executor_config, dict) else None,
            rootless=rootless,
            auto_user=auto_user,
            host_hardware=host_hardware,
            v=v,
        )
    except Exception:
        logger.debug("image entrypoint preflight: executor unresolvable; skipping probe", exc_info=True)
        return

    try:
        probe = executor.verify_command_passthrough(image, hosts, ssh_kwargs=ssh_kwargs)
    except Exception:
        logger.debug("image entrypoint preflight: probe failed; skipping", exc_info=True)
        return

    if probe is None or not probe.consumes_command:
        return

    raise RecipeError(
        "Container image %s declares ENTRYPOINT %s, which consumes the command sparkrun "
        "appends rather than running it, so the workload would never start — the image's "
        "own program parses sparkrun's launcher as its flags. Verified on %s: the same "
        "command runs correctly once the entrypoint is cleared.\n"
        "\n"
        "Fix it in the recipe:\n"
        "\n"
        "    executor_config:\n"
        '      entrypoint: ""\n'
        "\n"
        "or for a one-off run, pass:  -o entrypoint=''" % (image, probe.entrypoint or "(unknown)", probe.host)
    )


def resolve_effective_runtime_cache_dir(
    host_list: list[str],
    ssh_kwargs: dict,
    config: SparkrunConfig,
    dry_run: bool = False,
) -> str:
    """Resolve the target hosts' sparkrun cache dir for the runtime cache.

    Same shape and the same reasoning as :func:`resolve_effective_cache_dir`:
    the compilation cache lives on the *targets*, so its root must resolve
    against their ``$HOME``, not the control machine's.  A probe failure
    degrades to the control machine's path rather than raising — the runtime
    cache is an optimization, and a wrong guess costs a recompile (the
    directory simply won't pre-exist) while an exception would cost the launch.
    """
    from sparkrun.utils import is_local_host
    from sparkrun.orchestration.primitives import probe_remote_sparkrun_cache

    head = host_list[0] if host_list else None
    ssh_user = ssh_kwargs.get("ssh_user")
    cross_user = ssh_user is not None and ssh_user != os.environ.get("USER", "root")

    if head and not dry_run and (not is_local_host(head) or cross_user):
        try:
            return probe_remote_sparkrun_cache(head, **ssh_kwargs)
        except Exception:
            logger.debug("runtime_cache: cache-dir probe failed on %s; using local default", head, exc_info=True)

    return str(config.cache_dir)


def resolve_effective_cache_dir(
    cache_dir: str | None,
    host_list: list[str],
    ssh_kwargs: dict,
    config: SparkrunConfig,
    dry_run: bool = False,
) -> str:
    """Resolve the remote HF cache path to a concrete absolute string.

    - If *cache_dir* is given (cluster ``cache_dir`` or CLI override), use it
      as-is.
    - Otherwise, when targeting a remote host (or running cross-user), probe
      the head node via SSH so the resolved path reflects the SSH login user's
      ``$HOME`` / ``HF_HOME`` rather than the control machine's.
    - For the single-localhost same-user fast path, fall back to the control
      machine's HF cache.

    Returning a concrete path here avoids embedding shell-expansion expressions
    downstream, where ``shlex.quote``-aware code paths (volume mounts, ssh
    quoted commands) would prevent the expansion from running.
    """
    if cache_dir:
        return cache_dir

    from sparkrun.utils import is_local_host
    from sparkrun.orchestration.primitives import probe_remote_hf_cache

    head = host_list[0] if host_list else None
    ssh_user = ssh_kwargs.get("ssh_user")
    cross_user = ssh_user is not None and ssh_user != os.environ.get("USER", "root")

    if head and not is_local_host(head):
        return probe_remote_hf_cache(head, dry_run=dry_run, **ssh_kwargs)
    if head and cross_user:
        return probe_remote_hf_cache(head, dry_run=dry_run, **ssh_kwargs)

    return str(config.hf_cache_dir)


def apply_platform_runtime_flag_defaults(recipe: Recipe, runtime_name: str, host_hardware) -> dict[str, object]:
    """Fold platform/runtime/accelerator flag defaults into ``recipe.defaults``.

    Resolves the hardware platform for *host_hardware* and applies its
    :meth:`~sparkrun.platforms.base.HardwarePlatformPlugin.default_runtime_flags`
    for *runtime_name* at the **recipe-default tier** (``setdefault``).  This
    means a platform default (e.g. ``mmap: False`` for llama.cpp on GB10) only
    takes effect when the recipe — and therefore any CLI override layered above
    it — is silent on that key; an explicit recipe ``mmap: true`` is preserved.

    Returns the subset of defaults that were actually applied (for logging /
    testing); an empty dict means nothing was added.
    """
    from sparkrun.platforms import resolve_platform

    if host_hardware is None or not getattr(host_hardware, "accelerators", None):
        return {}

    platform = resolve_platform(host_hardware)
    if platform is None:
        return {}

    accel = host_hardware.accelerators[0]
    try:
        flag_defaults = platform.default_runtime_flags(runtime_name, accel) or {}
    except Exception:
        logger.debug("Platform %r default_runtime_flags raised", getattr(platform, "platform_name", "?"), exc_info=True)
        return {}

    applied: dict[str, object] = {}
    for key, value in flag_defaults.items():
        if key not in recipe.defaults:
            recipe.defaults[key] = value
            applied[key] = value
    return applied


def resolve_platform_env_defaults(runtime: RuntimePlugin, host_hardware) -> dict[str, str]:
    """Return the platform's container-env defaults for *runtime*.

    The env peer of :func:`apply_platform_runtime_flag_defaults`: resolves the
    hardware platform for *host_hardware* and asks it for
    :meth:`~sparkrun.platforms.base.HardwarePlatformPlugin.default_env`, passing
    both the runtime's name and its family (``get_family()``) so a platform can
    target one variant or a whole family (``"vllm"``).

    Unlike the flag defaults this returns rather than mutates: env is merged as
    a *tier* by the caller (below the cluster's env and the recipe's ``env``),
    so a user can override a platform default with any value, including an
    empty one — a ``setdefault`` into ``recipe.env`` could not express that.

    Never raises: a platform whose hook misbehaves contributes nothing.
    """
    from sparkrun.platforms import resolve_platform

    if host_hardware is None or not getattr(host_hardware, "accelerators", None):
        return {}

    platform = resolve_platform(host_hardware)
    if platform is None:
        return {}

    accel = host_hardware.accelerators[0]
    try:
        env = platform.default_env(runtime.runtime_name, accel, runtime_family=runtime.get_family()) or {}
    except Exception:
        logger.debug("Platform %r default_env raised", getattr(platform, "platform_name", "?"), exc_info=True)
        return {}
    return {str(k): str(val) for k, val in env.items()}


def resolve_per_host_backends(
    host_list: list[str],
    cluster=None,
) -> dict[str, "BackendBundle"]:
    """Resolve a :class:`BackendBundle` per host via :func:`select_backends`.

    For each host in *host_list*, calls
    :meth:`ClusterDefinition.hardware_for` (or defaults to DGX Spark
    when *cluster* is ``None``) and routes the result through
    :func:`sparkrun.core.backend_select.select_backends`.

    Hosts whose hardware fails to resolve a backend (unknown vendor,
    multi-vendor host, etc.) are silently skipped: runtimes fall back
    to the legacy NCCL generator in
    :func:`sparkrun.runtimes._cluster_ops.resolve_comm_env` for those
    hosts.  This keeps the cluster-launch surface live for
    partial-vendor coverage rather than failing-fast on a single bad
    fingerprint.

    Args:
        host_list: Resolved cluster hosts.
        cluster: Optional :class:`ClusterDefinition` carrying per-host
            hardware metadata.

    Returns:
        Mapping host -> :class:`BackendBundle`.  Empty dict when no
        host resolved successfully (e.g. all-Apple or all-CPU cluster).
    """
    from sparkrun.core.backend_select import NoMatchingBackendError, select_backends
    from sparkrun.core.hardware import default_dgx_spark_hardware

    backends: dict[str, BackendBundle] = {}
    for host in host_list:
        if cluster is not None:
            hw = cluster.hardware_for(host)
        else:
            hw = default_dgx_spark_hardware()
        try:
            backends[host] = select_backends(hw)
        except NoMatchingBackendError as e:
            logger.debug("No backend resolved for host %s: %s", host, e)
    return backends


def launch_inference(
    *,
    recipe: Recipe,
    runtime: RuntimePlugin,
    host_list: list[str],
    overrides: dict[str, Any],
    config: SparkrunConfig | None = None,
    v=None,
    sctx: SparkrunContext | None = None,
    is_solo: bool = False,
    cache_dir: str | None = None,
    local_cache_dir: str | None = None,
    transfer_mode: str | None = None,
    transfer_interface: str | None = None,
    # Model-distribution preferences.  ``None`` → derive from the resolved
    # ``cluster`` def; an explicit bool overrides (used by the benchmark path
    # which launches with explicit hosts and loses the named cluster).
    preserve_model_perms: bool | None = None,
    skip_model_fan_out: bool | None = None,
    recipe_ref: str | None = None,
    registry_mgr: RegistryManager | None = None,
    auto_port: bool = False,
    sync_tuning: bool = True,
    dry_run: bool = False,
    detached: bool = True,
    follow: bool = True,
    # Runtime-specific kwargs forwarded to runtime.run()
    ray_port: int | None = None,
    dashboard_port: int | None = None,
    dashboard: bool | None = None,
    init_port: int | None = None,
    topology: str | None = None,
    cluster_id_override: str | None = None,
    # Executor config (dict for config chain layering)
    executor_config: dict | None = None,
    extra_docker_opts: list[str] | None = None,
    # note: transition to rootless by default
    rootless: bool = True,
    auto_user: bool = True,
    progress: LaunchProgress | None = None,
    # Phase X threading: named cluster definition (carries per-host hardware
    # metadata).  When None, the runtime falls back to the legacy
    # host-list-only path (1 GPU / host, no per-host hardware lookups).
    cluster=None,
    # Precomputed placement from ``sparkrun.api.run`` (single source of
    # truth for "what runs where").  When provided, the runtime layer
    # uses it verbatim; when ``None``, the runtime recomputes locally
    # for back-compat with callers that haven't been threaded yet.
    placement=None,
    # When True, suppress the interactive confirmation prompt for
    # recipe-defined pre_exec hooks (and post_exec/post_commands run in
    # post_launch_lifecycle).  CLI flag --trust + local/official-registry
    # recipes set this to True via resolve_recipe_trust().
    trust: bool = False,
    # Called once, immediately before the runtime starts containers — after
    # every step that can fail cheaply (distribution, model download, tuning)
    # has succeeded.  ``sparkrun.api.run`` uses it to evict the deployments
    # this launch supersedes, so an interrupted or failed launch cannot tear
    # down a running workload it never got close to replacing.  Not called on
    # ``dry_run``.
    before_start: "Callable[[], None] | None" = None,
    # Highest-precedence layer of the runtime-cache settings chain (the CLI's
    # --runtime-cache / --no-runtime-cache lands here as ``{"enabled": bool}``).
    # ``None`` means "nothing was asked for" and defers to recipe / cluster /
    # config / runtime defaults.
    runtime_cache_override: dict | None = None,
) -> LaunchResult:
    """Launch an inference workload.

    This is the shared pipeline used by ``run``, ``benchmark``, and
    ``proxy load``.  It handles:

    1. Job metadata persistence
    2. Builder phase (if recipe defines a builder)
    3. Runtime preparation
    4. Resource distribution (container image + model)
    5. Tuning config sync and distribution
    6. GGUF model resolution
    7. Serve command generation
    8. Page cache clear
    9. ``runtime.run()``

    Args:
        recipe: Loaded and validated recipe.
        runtime: Resolved runtime plugin.
        host_list: Resolved and trimmed host list.
        overrides: Merged overrides dict (from recipe_override_options + extras).
        config: SparkrunConfig instance.
        v: SAF Variables instance (optional, uses singleton if None).
        is_solo: Whether to launch in solo mode.
        cache_dir: Remote/cluster cache dir (None = resolve from config).
        local_cache_dir: Control-machine cache dir for downloads (None = same as cache_dir).
        transfer_mode: Resource transfer mode override (None = "auto").
        transfer_interface: Network interface for transfers (cx7 or mgmt; None = cx7 default).
        recipe_ref: Simplified recipe reference for display (e.g. @spark-arena/UUID).
        registry_mgr: Registry manager for tuning config sync.
        auto_port: If True, auto-increment port when the desired port is in use.
        sync_tuning: Whether to sync tuning configs from registries.
        dry_run: Show what would be done without executing.
        detached: Run containers in detached mode.
        follow: whether to follow logs
        ray_port: Ray GCS port (forwarded to runtime.run).
        dashboard_port: Ray dashboard port (forwarded to runtime.run).
        dashboard: Tri-state Ray dashboard toggle, forwarded verbatim to
            runtime.run. ``True``/``False`` force it; ``None`` lets the Ray
            runtime resolve it against ``recipe.runtime_config.dashboard``
            (defaulting on).
        init_port: Distributed init port (forwarded to runtime.run).
        executor_config: Executor config
        rootless: Run containers in rootless mode (applies defaults to executor_config)
        auto_user: Automatically set user and group IDs to match host. (applies defaults to executor_config)


    Returns:
        LaunchResult with the outcome and all resolved context.
    """
    from sparkrun.orchestration.job_metadata import derive_cluster_id, save_job_metadata
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    # Resolve config, v, and progress from sctx when provided
    if sctx is not None:
        if config is None:
            config = sctx.config
        if v is None:
            v = sctx.variables
        if progress is None:
            progress = sctx.progress
    p = progress  # short alias

    from sparkrun.orchestration.distribution import resolve_auto_transfer_mode

    # -- Phase 1: Prepare --
    if p:
        p.phase(1)

    # Resolve the recipe-wide trust flag once so pre_exec (here) and
    # post_exec/post_commands (post_launch_lifecycle) make the same
    # decision for the same recipe.
    recipe_trusted = resolve_recipe_trust(recipe, trust)

    # Security: host-path overrides let a recipe grant the container access to
    # arbitrary host paths.  Untrusted (registry/URL-sourced) recipes must not
    # be able to do this — fail closed before any of them is applied.
    _enforce_recipe_mount_trust(recipe, recipe_trusted)

    # Internal recipe escape hatch: ``cluster_config`` launch overrides
    # (undocumented).  Applied at this single launch choke point so both the
    # CLI and ``api.run`` honour them.  Recipe-level values take precedence
    # over cluster/global config.  See :class:`sparkrun.core.recipe.ClusterConfig`.
    from sparkrun.core.recipe import is_local_model_path

    _cluster_config = recipe.cluster_config
    _resolved_model_path: str | None = None
    if _cluster_config is not None:
        if _cluster_config.remote_cache_dir:
            cache_dir = _cluster_config.remote_cache_dir
        if _cluster_config.local_cache_dir:
            local_cache_dir = _cluster_config.local_cache_dir
        if _cluster_config.resolved_model_path:
            _resolved_model_path = _cluster_config.resolved_model_path
            # ``launch_inference`` is the shared run/benchmark/proxy pipeline and
            # the caller may reuse the same recipe/overrides across launches, so
            # repoint the serve argument on shallow copies rather than mutating
            # the caller's objects in place.
            recipe = copy.copy(recipe)
            overrides = dict(overrides)
            # Preserve a clean served-model name (default to the original repo
            # id) before repointing the serve argument at the on-disk weights.
            if recipe.model and "served_model_name" not in overrides and "served_model_name" not in (recipe.defaults or {}):
                overrides["served_model_name"] = recipe.model
            # Every runtime reads ``recipe.model`` for the serve argument, so
            # repoint it at the pre-placed weights.  Download + distribution are
            # skipped below; the path is identity-mounted into the container.
            recipe.model = _resolved_model_path
            logger.info(
                "cluster_config.resolved_model_path set; serving on-disk weights at %s (skipping model download/distribution)",
                _resolved_model_path,
            )

    # User-facing sugar for the same behaviour: an absolute path in ``model:``
    # is pre-placed on-disk weights.  ``recipe.model`` already *is* the path
    # (no repoint needed) and ``resolved_model_volume`` picks up the identity
    # mount from it; here we only need to skip download/distribution and give
    # the served model a clean name (the directory basename, not the full path).
    if not _resolved_model_path and is_local_model_path(recipe.model):
        _resolved_model_path = recipe.model
        overrides = dict(overrides)
        if "served_model_name" not in overrides and "served_model_name" not in (recipe.defaults or {}):
            overrides["served_model_name"] = os.path.basename(recipe.model.rstrip("/")) or recipe.model
        logger.info(
            "model is an absolute path; serving on-disk weights at %s (skipping model download/distribution)",
            recipe.model,
        )
    _skip_model_distribution = bool(_resolved_model_path)

    ssh_kwargs = build_ssh_kwargs(config)

    # Preflight: pre-placed weights (resolved_model_path / absolute-path model)
    # skip download + distribution, so verify the path actually exists on the
    # substrate where the workload will run before committing to that skip.
    # Substrate-aware via the launching executor (host → SSH ``test -e``;
    # provider executors probe their own volumes). Best-effort: skipped on
    # dry-run, and an unresolvable executor / unreachable host never blocks.
    if _resolved_model_path and not dry_run:
        _verify_pre_placed_model(
            recipe,
            host_list,
            ssh_kwargs,
            runtime=runtime,
            cluster=cluster,
            config=config,
            overrides=overrides,
        )

    effective_local_cache = local_cache_dir or str(config.hf_cache_dir)
    effective_cache_dir = resolve_effective_cache_dir(
        cache_dir,
        host_list,
        ssh_kwargs,
        config,
        dry_run=dry_run,
    )
    transfer_result = resolve_auto_transfer_mode(
        transfer_mode or "auto",
        host_list,
        ssh_kwargs=ssh_kwargs,
        dry_run=dry_run,
        topology=topology,
    )
    effective_transfer_mode = transfer_result.mode

    # Derive the deterministic cluster_id from recipe + (trimmed) hosts.
    #
    # This MUST precede port resolution below.  ``generate_intent_id`` hashes
    # the port, and the ``auto_port`` probe rewrites ``overrides["port"]`` in
    # place — so deriving afterwards would make the workload's *identity*
    # depend on whichever port happened to be free at launch.  Every lookup
    # path (``stop`` / ``logs`` / ``--ensure`` / proxy discovery) derives from
    # the recipe's *requested* port, so the identity must too, or none of them
    # can find the running job.  The identity is declarative (what was asked
    # for); the *actual* bound port is factual and travels in job metadata.
    cluster_id = cluster_id_override or derive_cluster_id(recipe, host_list, overrides=overrides)

    # -- Port resolution --
    if auto_port:
        from sparkrun.orchestration.primitives import find_available_port

        config_chain = recipe.build_config_chain(overrides)
        desired_port = int(config_chain.get("port") or 8000)
        head_host = host_list[0]
        serve_port = find_available_port(
            head_host,
            desired_port,
            ssh_kwargs=ssh_kwargs,
            dry_run=dry_run,
        )
        overrides["port"] = serve_port
    else:
        config_chain = recipe.build_config_chain(overrides)
        serve_port = int(config_chain.get("port") or 8000)

    # Resolve container image
    container_image = runtime.resolve_container(recipe, overrides)

    # Resolve recipe.mods to pre_exec entries (builder-agnostic).
    # Part of preparation — surfaces resolution failures before any
    # builder/distribution work, and keeps the builder ignorant of mods.
    if recipe.mods:
        if registry_mgr is None and config is not None:
            registry_mgr = config.get_registry_manager()
        if registry_mgr is not None:
            from sparkrun.core.mods import resolve_and_inject_mods

            resolve_and_inject_mods(
                recipe,
                registry_mgr,
                config=config,
                transfer_mode=effective_transfer_mode,
                head=host_list[0] if host_list else None,
                ssh_kwargs=ssh_kwargs,
                dry_run=dry_run,
            )
        else:
            logger.warning("Cannot resolve recipe.mods: no RegistryManager available")

    if p:
        p.phase_end()

    # -- Phase 2: Builder --
    builder = None
    if recipe.builder:
        if p:
            p.phase(2)
        from sparkrun.builders.base import BuilderUnavailableError
        from sparkrun.core.bootstrap import get_builder

        # Only *lookup* is tolerated failing here. A ValueError out of
        # prepare() is a real build failure, and reporting it as "builder not
        # found, skipping" would launch the workload without the environment
        # it asked for. A gated builder is likewise never skipped: the user
        # named one that exists (see BuilderUnavailableError).
        try:
            builder = get_builder(recipe.builder, v)
        except BuilderUnavailableError:
            raise
        except ValueError:
            builder = None
            logger.warning("Builder '%s' not found, skipping", recipe.builder)

        if builder is not None:
            container_image = builder.prepare(
                container_image,
                recipe,
                host_list,
                config=config,
                dry_run=dry_run,
                transfer_mode=effective_transfer_mode,
                ssh_kwargs=ssh_kwargs,
            )
        if p:
            p.phase_end()
    else:
        if p:
            p.phase_skip(2, "no builder")

    # Resolve per-host backends from cluster hardware (or DGX Spark default).
    # Used by NCCL/RCCL/HCCL env emission inside the cluster orchestrator;
    # empty dict means runtimes fall back to the legacy NCCL generator in
    # _cluster_ops.resolve_comm_env.
    backends = resolve_per_host_backends(host_list, cluster=cluster)

    # Pre-placement compatibility gate: verify the runtime can target every
    # placed host before any side effects (container pull, model sync, etc.).
    # Skipped when no cluster hardware is available (e.g. --hosts / --hosts-file
    # bypass, or a host without fingerprint data); a missing hardware entry in
    # ClusterDefinition.hardware_for() falls back to DGX Spark defaults, so
    # only runtimes with requires_capability constraints are affected.
    if cluster is not None and runtime.requires_capability:
        from sparkrun.runtimes.compatibility import (
            IncompatibleHardwareError,
            check_runtime_host_compatibility,
        )

        compat_errors: list[str] = []
        for host in host_list:
            hw = cluster.hardware_for(host)
            compat_errors.extend(check_runtime_host_compatibility(runtime, host, hw))
        if compat_errors:
            raise IncompatibleHardwareError(runtime.runtime_name, compat_errors)

    # Per-host platform validation: emit warnings for vendor-specific concerns
    # (missing RoCEv2 on DGX Spark, non-NVIDIA on generic platform, etc.).
    # This runs regardless of whether a cluster was threaded — hosts without
    # explicit metadata fall back to DGX Spark defaults so the check always
    # has something sensible to validate against.
    from sparkrun.platforms import resolve_platform

    _head_hw = None
    for host in host_list:
        if cluster is not None:
            _hw = cluster.hardware_for(host)
        else:
            from sparkrun.core.hardware import default_dgx_spark_hardware

            _hw = default_dgx_spark_hardware()
        if _head_hw is None:
            _head_hw = _hw
        _platform = resolve_platform(_hw)
        if _platform is not None:
            for _warn in _platform.validate_host(_hw):
                logger.warning("Host %s: %s", host, _warn)

    # Platform/runtime/accelerator flag defaults (e.g. mmap off for GB10 +
    # llama.cpp).  Keyed off the head host's accelerator; applied at the
    # recipe-default tier so explicit recipe/CLI values still win.  The serve
    # command is built once, so a single representative host is the right scope.
    _applied_flags = apply_platform_runtime_flag_defaults(recipe, runtime.runtime_name, _head_hw)
    if _applied_flags:
        logger.debug("Applied platform runtime-flag defaults: %s", _applied_flags)

    # Save job metadata
    if not dry_run:
        try:
            save_job_metadata(
                cluster_id,
                recipe,
                host_list,
                overrides=overrides,
                cache_dir=str(config.cache_dir),
                recipe_ref=recipe_ref,
                container_image=container_image,
                runtime=runtime,
                backends=backends,
            )
        except Exception:
            # Not fatal to the launch, but it is not cosmetic either: without
            # metadata, `stop` and `logs` can't recover this job's hosts from
            # the cluster id alone.  Warn rather than whisper at debug — a
            # silent debug line hid a total write failure on Windows.
            logger.warning(
                "Could not save job metadata for %s; `sparkrun logs`/`stop` may not find this job by cluster id (pass --hosts if so)",
                cluster_id,
                exc_info=True,
            )

    # Pre-launch preparation (post-container builds)
    runtime.prepare(
        recipe,
        host_list,
        config=config,
        dry_run=dry_run,
        transfer_mode=effective_transfer_mode,
        overrides=overrides,
    )

    # -- Phase 3: Distribution --
    comm_env = None
    ib_ip_map: dict[str, str] = {}
    ib_iface_map: dict[str, str] = {}
    if not runtime.is_delegating_runtime():
        if p:
            p.phase(3)
        from sparkrun.orchestration.distribution import distribute_from_config

        # Cluster-level model-distribution preferences (shared/NFS caches).
        # Explicit kwargs win (used by the benchmark path, which launches with
        # explicit hosts and so loses the named cluster identity); otherwise
        # fall back to the resolved cluster's prefs.  Anonymous/explicit-hosts
        # clusters carry the safe defaults (preserve_perms=True,
        # skip_fan_out=False).
        from sparkrun.core.cluster_manager import ModelDistributionPrefs

        _dist_model = getattr(getattr(cluster, "distribution", None), "model", None)
        _model_prefs = ModelDistributionPrefs(
            preserve_perms=(preserve_model_perms if preserve_model_perms is not None else getattr(_dist_model, "preserve_perms", True)),
            skip_fan_out=(skip_model_fan_out if skip_model_fan_out is not None else getattr(_dist_model, "skip_fan_out", False)),
        )
        # Skip model download + distribution when the cluster disables it
        # (``distribution.model.enabled: false``) or a recipe points at
        # pre-placed weights (``cluster_config.resolved_model_path``).
        _model_dist_enabled = getattr(_dist_model, "enabled", True)
        _skip_model = _skip_model_distribution or not _model_dist_enabled

        # Skip container-image distribution for container-less executors (the
        # `local` executor has no image to distribute — and the image may not
        # even exist). Resolve the executor name cheaply here (the full
        # resolve_executor(...) below is byte-identical, just later) and map it
        # to its class's needs_image. k8s takes its own launch path
        # (api/_run.py run_k8s) so only `local` reaches here as container-less.
        # Any resolution error defaults to distributing the image (never break
        # launch on the skip decision).
        _skip_container = False
        try:
            from sparkrun.orchestration.executor import get_executor, resolve_executor_name

            _exec_name = resolve_executor_name(
                cli_overrides=executor_config if isinstance(executor_config, dict) else None,
                recipe=recipe,
                cluster=cluster,
                runtime=runtime,
                config=config,
                v=v,
            )
            _skip_container = not getattr(get_executor(_exec_name, v), "needs_image", True)
        except Exception:
            logger.debug("Could not resolve executor for image-skip decision; distributing image", exc_info=True)
            _skip_container = False

        # Preflight: does this image actually run the command sparkrun appends?
        # Runs from the distribution hook — i.e. once the image is resident on
        # every target but *before* the model sync — because that is the only
        # point where the image can be probed on the substrate and the launch
        # has not yet paid for the long, routinely-interrupted transfer.
        # Skipped on dry-run (no SSH) and for container-less executors.
        def _probe_image_entrypoint() -> None:
            if dry_run or _skip_container:
                return
            _verify_image_command_passthrough(
                recipe,
                container_image,
                host_list,
                ssh_kwargs,
                runtime=runtime,
                cluster=cluster,
                config=config,
                executor_config=executor_config,
                rootless=rootless,
                auto_user=auto_user,
                host_hardware=_head_hw,
                v=v,
            )

        comm_env, ib_ip_map, mgmt_ip_map, ib_iface_map = distribute_from_config(
            recipe,
            container_image,
            host_list,
            effective_cache_dir,
            config,
            dry_run,
            model_revision=recipe.model_revision,
            recipe_name=recipe.name,
            transfer_mode=effective_transfer_mode,
            transfer_interface=transfer_interface,
            local_cache_dir=effective_local_cache,
            pre_ib=transfer_result,
            topology=topology,
            prefs=_model_prefs,
            skip_model=_skip_model,
            skip_container=_skip_container,
            after_container_sync=_probe_image_entrypoint,
        )
        # Re-save job metadata with IP maps from IB detection
        if not dry_run and (ib_ip_map or mgmt_ip_map):
            try:
                save_job_metadata(
                    cluster_id,
                    recipe,
                    host_list,
                    overrides=overrides,
                    cache_dir=str(config.cache_dir),
                    ib_ip_map=ib_ip_map,
                    mgmt_ip_map=mgmt_ip_map,
                    recipe_ref=recipe_ref,
                    runtime=runtime,
                    backends=backends,
                )
            except Exception:
                logger.debug("Failed to update job metadata: %s", cluster_id, exc_info=True)
        if p:
            p.phase_end()
    else:
        if p:
            p.phase_skip(3, "delegating runtime")

    # -- Phase 4: Tuning --
    _needs_tuning = (sync_tuning and not dry_run) or not runtime.is_delegating_runtime()
    if _needs_tuning:
        if p:
            p.phase(4)
    else:
        if p:
            p.phase_skip(4, "disabled")

    if sync_tuning and not dry_run:
        from sparkrun.tuning.sync import sync_registry_tuning

        try:
            synced = sync_registry_tuning(
                registry_mgr,
                recipe.runtime,
                dry_run=dry_run,
                registry_name=recipe.source_registry,
            )
            if synced:
                logger.info("Synced %d tuning config(s) from registries.", synced)
        except Exception:
            logger.debug("Failed to sync tuning configs", exc_info=True)

    # Distribute tuning configs to remote hosts
    if not runtime.is_delegating_runtime():
        from sparkrun.tuning.distribute import distribute_tuning_to_hosts

        try:
            tuning_failed = distribute_tuning_to_hosts(
                recipe.runtime,
                host_list,
                dry_run=dry_run,
                transfer_mode=effective_transfer_mode,
                **ssh_kwargs,
            )
            if tuning_failed:
                logger.warning(
                    "Tuning config distribution failed on: %s",
                    ", ".join(tuning_failed),
                )
        except Exception:
            logger.debug("Failed to distribute tuning configs", exc_info=True)

    if _needs_tuning and p:
        p.phase_end()

    # GGUF model resolution
    from sparkrun.models.download import is_gguf_model, resolve_gguf_container_path, resolve_mmproj_container_path

    if is_gguf_model(recipe.model) and not dry_run and not _resolved_model_path:
        gguf_container_path = resolve_gguf_container_path(
            recipe.model,
            effective_cache_dir,
        )
        if gguf_container_path:
            overrides["_gguf_model_path"] = gguf_container_path
            overrides["model"] = gguf_container_path
            logger.info("GGUF model pre-synced, container path: %s", gguf_container_path)

        # Multimodal projector (mmproj) resolution for vision GGUF models.
        # The selector lives in runtime_config (``mmproj:`` top-level key is
        # auto-swept there); the resolved container path is injected into the
        # override layer so ``{mmproj}`` substitutes and llama.cpp can
        # auto-inject ``--mmproj``.  llama.cpp-specific.
        if recipe.runtime == "llama-cpp":
            mmproj_selector = recipe.runtime_config.get("mmproj")
            _disabled = str(mmproj_selector).lower() in ("false", "none", "off", "no", "0", "disable", "disabled")
            if mmproj_selector is None or not _disabled:
                mmproj_container_path = resolve_mmproj_container_path(
                    recipe.model,
                    effective_cache_dir,
                    selector=None if mmproj_selector is None else str(mmproj_selector),
                )
                if mmproj_container_path:
                    overrides["_mmproj_path"] = mmproj_container_path
                    overrides["mmproj"] = mmproj_container_path
                    logger.info("mmproj projector resolved, container path: %s", mmproj_container_path)

    # Generate serve command
    serve_command = runtime.generate_command(
        recipe=recipe,
        overrides=overrides,
        is_cluster=not is_solo,
        num_nodes=len(host_list),
        head_ip=None,  # determined during launch
    )

    # Best-effort page cache clear
    if not runtime.is_delegating_runtime():
        from sparkrun.orchestration.primitives import try_clear_page_cache

        try_clear_page_cache(host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run)

    # -- Phase 5: Launch runtime --
    if p:
        from sparkrun.utils.cli_formatters import RUNTIME_DISPLAY

        _rt_display = RUNTIME_DISPLAY.get(runtime.runtime_name, runtime.runtime_name)
        p.phase(5, "Launching %s runtime" % _rt_display)

    # Last point before containers start.  Everything that can fail slowly and
    # cheaply — image distribution, model download, tuning sync — is behind us,
    # so a caller can safely tear down the deployment this launch replaces.
    # Doing it any earlier means an interrupted `sparkrun run` leaves the
    # cluster with neither the old workload nor the new one.
    if before_start is not None and not dry_run:
        before_start()

    # Build runtime.run() kwargs — include runtime-specific options only
    # when they were explicitly provided.
    run_kwargs: dict[str, Any] = {"follow": follow}
    if ray_port is not None:
        run_kwargs["ray_port"] = ray_port
    if dashboard_port is not None:
        run_kwargs["dashboard_port"] = dashboard_port
    # Forward the tri-state dashboard toggle verbatim (True / False / None). The
    # Ray runtime resolves None against the recipe and emits --include-dashboard
    # accordingly; non-Ray runtimes ignore it via run()'s **kwargs.
    run_kwargs["dashboard"] = dashboard
    if init_port is not None:
        run_kwargs["init_port"] = init_port
    if topology is not None:
        run_kwargs["topology"] = topology

    # Build executor via the unified resolution chain (single source of
    # truth shared with cli._stop_logs).  Order: CLI → recipe → runtime
    # → per-executor adjustments (Docker reads rootless/auto_user here)
    # → SparkrunConfig → platform → per-executor defaults → dataclass field
    # defaults.  The head host's hardware supplies the platform tier (e.g. DGX
    # Spark pinning docker's GPU request to --gpus rather than CDI); one
    # executor is built per launch, so a representative host is the right scope.
    from sparkrun.orchestration.executor import resolve_executor

    executor = resolve_executor(
        recipe=recipe,
        cluster=cluster,
        runtime=runtime,
        config=config,
        cli_overrides=executor_config if isinstance(executor_config, dict) else None,
        rootless=rootless,
        auto_user=auto_user,
        host_hardware=_head_hw,
        v=v,
    )

    # -- Runtime (compilation / autotune) cache --
    #
    # Mount a persistent host dir at /cache/runtime so torch.compile, Inductor,
    # Triton, FlashInfer and the TRT-LLM autotuner survive the `--rm` container
    # (issue #256).  Resolution order and the keying rules live in
    # :mod:`sparkrun.core.runtime_cache`; the short version is that the
    # directory key is *hygiene* (disk footprint, hit rate) and never
    # correctness, so a disabled or mis-keyed cache costs a recompile at worst.
    #
    # Wholly best-effort: any failure here degrades to "no cache" rather than
    # failing a launch that would otherwise work.
    runtime_cache_mounts = None
    try:
        from sparkrun.core.runtime_cache import (
            build_runtime_cache_mounts,
            probe_image_identity,
            resolve_runtime_cache_root,
            resolve_runtime_cache_settings,
            runtime_cache_disabled_by_env,
        )
        from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint

        _rc_settings = resolve_runtime_cache_settings(
            runtime=runtime,
            config=config,
            cluster=cluster,
            recipe=recipe,
            cli_override=runtime_cache_override,
            env_disabled=runtime_cache_disabled_by_env(),
        )
        if _rc_settings.enabled:
            _rc_root = resolve_runtime_cache_root(
                _rc_settings,
                resolve_effective_runtime_cache_dir(host_list, ssh_kwargs, config, dry_run=dry_run),
            )
            # Derived *after* apply_platform_runtime_flag_defaults, deliberately.
            # The fingerprint then reflects the platform flags this hardware
            # actually gets — which is what a per-configuration autotuner cache
            # wants, since those tactics are hardware-specific anyway.  It is
            # stable across relaunches on the same cluster, so hits still land.
            runtime_cache_mounts = build_runtime_cache_mounts(
                runtime=runtime,
                recipe=recipe,
                settings=_rc_settings,
                root=_rc_root,
                image=container_image,
                # Only probed when the key needs it — see probe_image_identity.
                image_identity=(
                    probe_image_identity(container_image, host_list, ssh_kwargs, dry_run=dry_run) if _rc_settings.key_by_image else None
                ),
                fingerprint=derive_recipe_fingerprint(recipe, overrides),
            )
    except Exception:
        logger.debug("runtime_cache: resolution failed; launching without it", exc_info=True)
        runtime_cache_mounts = None

    if runtime_cache_mounts is not None:
        logger.debug("runtime cache: %s -> /cache/runtime", runtime_cache_mounts.leaf)
        if not dry_run:
            # Create + stamp + sweep on the substrate.  Docker would otherwise
            # materialize a missing -v source as root-owned, and `local` has no
            # daemon to create it at all.
            try:
                executor.ensure_runtime_cache(runtime_cache_mounts, host_list, ssh_kwargs=ssh_kwargs)
            except Exception:
                logger.debug("runtime_cache: host preparation failed; continuing", exc_info=True)

    # Container env tiers, lowest first:
    #   platform (hardware tuning, e.g. PYTORCH_CUDA_ALLOC_CONF on GB10)
    #   < cluster env_file (e.g. CONTAINER_* imported from a legacy
    #     spark-vllm-docker .env)
    #   < recipe env (which already carries any -e CLI override).
    # Single chokepoint — covers solo and cluster mode, all runtimes and
    # executors.  Keyed off the head host, like the platform flag defaults.
    platform_env = resolve_platform_env_defaults(runtime, _head_hw)
    if platform_env:
        logger.debug("Applied platform env defaults: %s", sorted(platform_env))
    cluster_env = cluster.resolve_env() if (cluster is not None and getattr(cluster, "env", None)) else {}
    effective_env = recipe.env
    if platform_env or cluster_env:
        effective_env = {**platform_env, **cluster_env, **(recipe.env or {})}

    # Launch
    rc = runtime.run(
        hosts=host_list,
        image=container_image,
        serve_command=serve_command,
        recipe=recipe,
        overrides=overrides,
        cluster_id=cluster_id,
        env=effective_env,
        cache_dir=effective_cache_dir,
        config=config,
        dry_run=dry_run,
        detached=detached,
        comm_env=comm_env,
        ib_ip_map=ib_ip_map,
        ib_iface_map=ib_iface_map,
        executor=executor,
        progress=progress,
        extra_docker_opts=extra_docker_opts,
        cluster=cluster,
        placement=placement,
        backends=backends or None,
        trust=recipe_trusted,
        runtime_cache=runtime_cache_mounts,
        **run_kwargs,
    )

    if p:
        p.phase_end()

    # Collect runtime version info from the head container (non-blocking)
    runtime_info: dict[str, str] = {}
    if rc == 0 and not dry_run:
        try:
            head_host = host_list[0] if host_list else "localhost"
            head_container = runtime.get_head_container_name(cluster_id, is_solo=is_solo)
            # Resolve builder for version info collection
            ver_builder = None
            if recipe.builder:
                from sparkrun.core.bootstrap import get_builder

                try:
                    ver_builder = get_builder(recipe.builder, v)
                except ValueError:
                    pass
            # noinspection PyProtectedMember
            runtime_info = runtime._collect_runtime_info(
                head_host,
                head_container,
                ssh_kwargs,
                dry_run=False,
                builder=ver_builder,
            )
            # Collect container image labels (separate docker inspect call)
            if ver_builder:
                try:
                    label_info = ver_builder.collect_container_labels(
                        head_container,
                        head_host,
                        ssh_kwargs,
                    )
                    # Merge without overwriting existing keys
                    for k, lv in label_info.items():
                        if k not in runtime_info:
                            runtime_info[k] = lv
                except Exception:
                    logger.debug("Container label collection failed", exc_info=True)
            if runtime_info:
                try:
                    save_job_metadata(
                        cluster_id,
                        recipe,
                        host_list,
                        overrides=overrides,
                        cache_dir=str(config.cache_dir),
                        recipe_ref=recipe_ref,
                        runtime_info=runtime_info,
                        container_image=container_image,
                        runtime=runtime,
                        backends=backends,
                    )
                except Exception:
                    logger.debug("Failed to save runtime_info to job metadata", exc_info=True)
        except Exception:
            logger.debug("Runtime info collection failed", exc_info=True)

    return LaunchResult(
        rc=rc,
        cluster_id=cluster_id,
        host_list=host_list,
        is_solo=is_solo,
        runtime=runtime,
        recipe=recipe,
        overrides=overrides,
        container_image=container_image,
        effective_cache_dir=effective_cache_dir,
        serve_port=serve_port,
        config=config,
        recipe_ref=recipe_ref,
        comm_env=comm_env,
        ib_ip_map=ib_ip_map,
        ib_iface_map=ib_iface_map,
        serve_command=serve_command,
        runtime_info=runtime_info,
        builder=builder,
        backends=backends,
    )


@dataclass(frozen=True)
class ServeReadiness:
    """Outcome of waiting for a launched workload's head endpoint.

    ``reason`` is empty when ready, else ``"port"`` (never started
    listening / container exited) or ``"health"`` (listening but never
    returned HTTP 200).
    """

    ready: bool
    head_host: str
    head_ip: str
    port: int
    container: str
    reason: str = ""

    @property
    def health_url(self) -> str:
        return "http://%s:%d/v1/models" % (self.head_ip, self.port)


def wait_for_serve_ready(
    result: LaunchResult,
    *,
    ssh_kwargs: dict | None = None,
    dry_run: bool = False,
    port_max_retries: int = 120,
    port_retry_interval: int = 2,
    health_max_retries: int = 120,
    health_retry_interval: int = 5,
) -> ServeReadiness:
    """Wait for a detached launch's head endpoint to answer ``/v1/models``.

    ``launch_inference(detached=True)`` returns once the *containers* are
    up, which for a large model is minutes before the server accepts a
    request.  Callers that need to act on a serving endpoint must wait
    for it explicitly.

    Two stages, and the order matters: an inference server refuses
    connections outright until its engine has finished initializing, so
    the entire startup is indistinguishable from a crash when probing
    the URL alone.  :func:`wait_for_port` polls the head *host* for a
    listening socket and aborts early if the head container has exited;
    only then is :func:`wait_for_healthy`'s connection-refused-means-dead
    heuristic sound.

    Args:
        result: The :class:`LaunchResult` to wait on.
        ssh_kwargs: SSH parameters for probing the head host.
        dry_run: Report ready without waiting.
        port_max_retries: Port-poll attempts (``port_retry_interval`` apart).
        port_retry_interval: Seconds between port polls.
        health_max_retries: Health-poll attempts (``health_retry_interval`` apart).
        health_retry_interval: Seconds between health polls.

    Returns:
        A :class:`ServeReadiness` describing the head endpoint and
        whether it became ready.
    """
    from sparkrun.orchestration.docker import generate_container_name, generate_node_container_name
    from sparkrun.orchestration.health import wait_for_healthy, wait_for_port
    from sparkrun.orchestration.primitives import detect_host_ip
    from sparkrun.utils import is_local_host

    head_host = result.host_list[0] if result.host_list else "localhost"

    if result.is_solo:
        container = generate_container_name(result.cluster_id, "solo")
    else:
        container = generate_node_container_name(result.cluster_id, 0)

    if is_local_host(head_host):
        head_ip = "127.0.0.1"
    else:
        try:
            head_ip = detect_host_ip(head_host, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
        except RuntimeError:
            head_ip = head_host

    port = result.serve_port
    base = ServeReadiness(True, head_host, head_ip, port, container)
    if dry_run:
        return base

    if not wait_for_port(
        head_host,
        port,
        max_retries=port_max_retries,
        retry_interval=port_retry_interval,
        ssh_kwargs=ssh_kwargs,
        dry_run=dry_run,
        container_name=container,
    ):
        return ServeReadiness(False, head_host, head_ip, port, container, reason="port")

    if not wait_for_healthy(
        base.health_url,
        max_retries=health_max_retries,
        retry_interval=health_retry_interval,
        dry_run=dry_run,
    ):
        return ServeReadiness(False, head_host, head_ip, port, container, reason="health")

    return base


def post_launch_lifecycle(
    result: LaunchResult,
    remote_cache_dir: str,
    trust: bool = False,
    dry_run: bool = False,
    progress: LaunchProgress | None = None,
) -> None:
    """Run post-serve lifecycle: port polling, health checks, hooks, conditional stop.

    Called after a successful detached launch when recipe defines post_exec or post_commands.
    Handles:
    1. Determining head container name
    2. Detecting head IP
    3. Waiting for port and health check
    4. Building hook context
    5. Running post_exec and post_commands
    6. Handling stop_after_post

    Args:
        result: LaunchResult from launch_inference.
        remote_cache_dir: Remote cache directory for hook context.
        trust: Trust post_commands from non-default registries without prompting.
        dry_run: Show what would be done without executing.
    """
    import sys

    import click

    from sparkrun.orchestration.hooks import (
        build_hook_context,
        run_post_commands,
        run_post_exec,
    )
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    p = progress  # short alias
    if p:
        p.phase(6)

    recipe = result.recipe
    runtime = result.runtime
    host_list = result.host_list
    overrides = result.overrides
    config = result.config

    _ssh_kw = build_ssh_kwargs(config)

    click.echo("Waiting for server to become ready...")
    readiness = wait_for_serve_ready(result, ssh_kwargs=_ssh_kw, dry_run=dry_run)
    head_host = readiness.head_host
    head_ip = readiness.head_ip
    effective_port = readiness.port
    head_container = readiness.container

    if not readiness.ready:
        if readiness.reason == "port":
            click.echo("Error: Server port %d never became ready" % effective_port, err=True)
        else:
            click.echo("Error: Server health check never passed at %s" % readiness.health_url, err=True)
        sys.exit(1)

    config_chain = recipe.build_config_chain(overrides)

    # Build hook context with extended variables
    hook_context = build_hook_context(
        config_chain,
        head_host=head_host,
        head_ip=head_ip,
        port=effective_port,
        cluster_id=result.cluster_id,
        container_name=head_container,
        cache_dir=remote_cache_dir,
    )

    # Resolve trust once for both post_exec (inside head container) and
    # post_commands (on control machine).  Same gate as the pre_exec
    # decision computed in launch_inference().
    _is_trusted = resolve_recipe_trust(recipe, trust)

    try:
        # Run post_exec inside head container
        if recipe.post_exec:
            click.echo("Running post_exec commands...")
            run_post_exec(
                head_host,
                head_container,
                recipe.post_exec,
                hook_context,
                ssh_kwargs=_ssh_kw,
                dry_run=dry_run,
                trust=_is_trusted,
                cache_dir=remote_cache_dir,
            )

        # Run post_commands on control machine
        if recipe.post_commands:
            click.echo("Running post_commands on control machine...")
            run_post_commands(recipe.post_commands, hook_context, dry_run=dry_run, trust=_is_trusted)
    except RuntimeError as e:
        click.echo("Error in post hooks: %s" % e, err=True)
        sys.exit(1)

    click.echo("Post hooks completed successfully.")
    if p:
        p.phase_end()

    # If stop_after_post, stop the workload and exit
    if recipe.stop_after_post:
        click.echo("Stopping workload (stop_after_post=true)...")
        runtime.stop(
            hosts=host_list,
            cluster_id=result.cluster_id,
            config=config,
            dry_run=dry_run,
        )
        sys.exit(0)
