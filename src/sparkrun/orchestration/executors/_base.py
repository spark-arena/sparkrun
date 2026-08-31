"""Executor ABC, ExecutorConfig dataclass, and SAF extension point.

This module is the canonical home for the :class:`Executor` plugin
interface.  Concrete implementations live alongside in
``executors/docker.py``, ``executors/local.py``, ``executors/k8s.py``.

The public facade :mod:`sparkrun.orchestration.executor` re-exports
the names defined here, plus the resolution helpers
(:func:`resolve_executor` / :func:`get_executor`).
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Mapping, TYPE_CHECKING

from scitrera_app_framework import Plugin, Variables, ext_parse_bool, get_extensions

from sparkrun.orchestration.teardown import TEARDOWN_REMOVED_MARKER
from sparkrun.scripts import read_script
from sparkrun.utils import merge_env
from sparkrun.utils.shell import b64_encode_cmd, quote

if TYPE_CHECKING:
    from sparkrun.containers.entrypoint import EntrypointProbe
    from sparkrun.core.cluster_status import ClusterStatus, TerminationInfo
    from sparkrun.core.hardware import HostHardware
    from sparkrun.core.log_source import LogSource
    from sparkrun.core.runtime_cache import RuntimeCacheMounts

logger = logging.getLogger(__name__)

EXT_EXECUTOR = "sparkrun.executor"

# Canonical container/pod label schema for sparkrun-launched workloads.
# Concrete executors emit these in their native annotation system
# (Docker --label, k8s metadata.labels, etc.) and query_status reads
# them back to enrich the ClusterStatus snapshot.
LABEL_CLUSTER_ID = "sparkrun.cluster_id"
LABEL_INTENT_ID = "sparkrun.intent_id"
LABEL_RECIPE = "sparkrun.recipe"
LABEL_RANK = "sparkrun.rank"
LABEL_RUNTIME = "sparkrun.runtime"
LABEL_MODEL = "sparkrun.model"
LABEL_SERVED_MODEL_NAME = "sparkrun.served_model_name"


def _registered_executor_names() -> set[str]:
    """Return executor selectors registered with SAF under :data:`EXT_EXECUTOR`.

    Falls back to the built-in static set when SAF isn't initialized
    (e.g. test paths that build :class:`ExecutorConfig` directly without
    going through ``init_sparkrun``).
    """
    try:
        from sparkrun.core.bootstrap import get_variables

        v = get_variables()
    except Exception:  # pragma: no cover - degraded path
        v = None

    if v is not None:
        try:
            plugins = get_extensions(EXT_EXECUTOR, v=v)
            names = {p.executor_name for p in plugins.values() if getattr(p, "executor_name", "")}
            if names:
                return names
        except Exception:
            logger.debug("Falling back to static executor name set", exc_info=True)

    # Static fallback — keeps the validation logic functioning for
    # in-tree tests that import :class:`ExecutorConfig` directly without
    # bootstrapping the SAF plugin registry.
    return {"docker", "local", "k8s"}


@dataclass
class ExecutorConfig:
    """Typed view of resolved executor settings.

    Constructed from a config chain (or plain dict) after CLI → recipe
    → runtime → per-executor adjustments → SparkrunConfig →
    per-executor defaults layering, driven by
    :func:`sparkrun.orchestration.executor.resolve_executor`.
    """

    auto_remove: bool = True
    restart_policy: str | None = None
    privileged: bool = True
    gpus: str = "all"
    ipc: str = "host"
    shm_size: str = "25gb"
    network: str = "host"
    user: str | None = None
    security_opt: list[str] | None = None
    cap_add: list[str] | None = None
    ulimit: list[str] | None = None
    devices: list[str] | None = None
    volumes: list[str] | None = None
    """Extra bind mounts as Docker ``-v`` specs (e.g. ``"/mnt/quant:/mnt/quant"``).

    A bare path (no ``:``) is expanded to an identity mount (``/p`` →
    ``/p:/p``); ``src:dst`` and ``src:dst:ro`` forms are passed through.  These
    are emitted in addition to the standard HuggingFace cache mount, so a recipe
    or cluster can grant the container access to extra host paths (calibration
    output dirs, datasets, …) without touching the model cache."""
    entrypoint: str | None = None
    """Container entrypoint override.

    ``None`` leaves the image entrypoint untouched.  ``""`` is meaningful:
    Docker treats it as "clear the image ENTRYPOINT"; K8s uses it to switch
    from args mode into an explicit ``bash -c`` command override.
    """
    memory_limit: str | None = None
    labels: list[str] | None = None
    accelerator_vendor: str | None = None
    """Per-host accelerator vendor that drives device-flag emission.

    ``None`` (default) preserves legacy NVIDIA behavior: the executor
    emits ``--gpus <gpus>``.  Set to ``"nvidia"`` explicitly when a
    per-host config is computed from :class:`HostHardware`.  Other
    supported values:

    - ``"amd"``: ROCm — emits ``--device /dev/kfd --device /dev/dri --group-add video``.
    - ``"intel"``: Intel Gaudi — emits ``--device /dev/accel``.
    - ``"apple"`` / ``"cpu"``: no device flag (Apple MLX requires a
      non-Docker executor; this leaves CPU-only containers untouched).

    For all non-NVIDIA values, ``gpus`` is ignored.
    """

    gpu_access_mode: str = "cdi"
    """How an NVIDIA GPU request is spelled on the container command line.

    - ``"cdi"`` (default): Container Device Interface —
      ``--device nvidia.com/gpu=<id>``.  Portable (Docker >= 25) and
      *required* on daemons that reject ``--gpus`` (e.g. Thunder Compute),
      but it depends on a valid, non-stale ``/etc/cdi/nvidia.yaml``.
    - ``"gpus"``: the classic nvidia-container-runtime flag,
      ``--gpus <gpus>``.

    Resolved through the normal executor chain, so it can be pinned per
    recipe / cluster / config; the platform tier
    (:meth:`~sparkrun.platforms.base.HardwarePlatformPlugin.default_executor_config`)
    supplies the per-hardware default — DGX Spark pins ``"gpus"``.  Only the
    Docker executor consumes it (``gpus`` itself is what Local/K8s read).
    """

    # Executor selector.  ``"docker"`` (default) uses
    # :class:`DockerExecutor`; ``"local"`` opts into the experimental
    # :class:`LocalExecutor` that runs the serve command as a native
    # subprocess; ``"k8s"`` opts into the experimental
    # :class:`K8sExecutor` draft.
    executor_type: str = "docker"

    # LocalExecutor-only fields (ignored by Docker/K8s).
    working_dir: str | None = None
    log_dir: str | None = None
    log_file: str | None = None
    pid_dir: str | None = None
    pid_file: str | None = None
    env_file: str | None = None
    command_prefix: str | None = None

    # K8sExecutor-only fields (ignored by Docker/Local). All
    # experimental — the K8s executor is a draft pending real-world
    # validation.  Empty values fall back to whatever ``kubectl`` picks
    # up from its current context.
    k8s_namespace: str | None = None
    k8s_context: str | None = None
    k8s_node_selector: str | None = None
    k8s_image_pull_policy: str | None = None
    kubeconfig: str | None = None
    # Absolute path to the kubectl binary the executor should invoke.
    # Usually resolved from sparkrun's managed cache by
    # :meth:`K8sExecutor.finalize_config`; falls back to bare ``kubectl``
    # (PATH lookup at script-run time) when unset.
    kubectl_path: str | None = None

    @classmethod
    def from_chain(cls, chain) -> ExecutorConfig:
        """Build from a config chain or plain dict.

        Only keys present (non-None) in *chain* override the
        dataclass-level field defaults.  Per-executor defaults are
        applied **upstream** (the bottom layer of
        :func:`resolve_executor`'s chain), not inside this method —
        this keeps ``ExecutorConfig`` executor-agnostic.
        """
        kwargs: dict = {}

        # Bool fields — only forward when explicitly present.
        for key in ("auto_remove", "privileged"):
            v = chain.get(key)
            if v is not None:
                kwargs[key] = ext_parse_bool(v)

        # Plain (non-nullable) string fields — only forward when present.
        for key in ("gpus", "gpu_access_mode", "ipc", "shm_size", "network"):
            v = chain.get(key)
            if v is not None:
                kwargs[key] = str(v)

        # Nullable string fields — falsy values map to dataclass default.
        for key in (
            "restart_policy",
            "user",
            "memory_limit",
            "accelerator_vendor",
            "working_dir",
            "log_dir",
            "log_file",
            "pid_dir",
            "pid_file",
            "env_file",
            "command_prefix",
            "k8s_namespace",
            "k8s_context",
            "k8s_node_selector",
            "k8s_image_pull_policy",
            "kubeconfig",
            "kubectl_path",
        ):
            v = chain.get(key)
            if v:
                kwargs[key] = v

        # Empty string is meaningful for entrypoint: it clears Docker's image
        # ENTRYPOINT and tells K8s to use an explicit bash command override.
        entrypoint = chain.get("entrypoint")
        if entrypoint is not None:
            kwargs["entrypoint"] = str(entrypoint)

        # List-or-string fields — promote bare strings to single-item lists.
        for key in ("security_opt", "cap_add", "ulimit", "devices", "volumes", "labels"):
            raw = chain.get(key)
            if raw is None:
                continue
            if isinstance(raw, str):
                raw = [raw]
            kwargs[key] = raw or None

        # Executor selector: prefer ``executor``, fall back to
        # ``executor_type`` for forward compat.  Validity is checked
        # against the SAF plugin registry (see
        # :func:`_registered_executor_names`); unknown values warn and
        # degrade to ``"docker"``.
        exec_type_raw = chain.get("executor")
        if exec_type_raw is None:
            exec_type_raw = chain.get("executor_type")
        if exec_type_raw:
            executor_type = str(exec_type_raw).strip().lower()
            known = _registered_executor_names()
            if executor_type not in known:
                logger.warning(
                    "Unknown executor type %r; falling back to 'docker'. Known: %s",
                    executor_type,
                    sorted(known),
                )
                executor_type = "docker"
            kwargs["executor_type"] = executor_type

        return cls(**kwargs)

    def __post_init__(self):
        # Docker does not allow --rm with --restart
        if self.restart_policy:
            self.auto_remove = False


class Executor(Plugin):
    """Abstract base for executors.

    Each concrete executor (Docker, local-native-subprocess, K8s, …) is
    a SAF :class:`Plugin` registered at the ``sparkrun.executor``
    extension point and discovered via :func:`find_types_in_modules`.
    The plugin instance returned from the extension registry acts as a
    *class registry* — callers look up the type and instantiate a
    fresh per-launch ``Executor`` via
    :func:`sparkrun.orchestration.executor.resolve_executor`.
    Per-launch config lives on ``self.config``; the registered SAF
    singleton uses a default ``ExecutorConfig`` only to keep its
    method signatures uniform.

    Subclasses must define:

    - ``executor_name``: identifier used to select this executor
      (matched against ``recipe.executor`` / ``SparkrunConfig.default_executor``).
    - The abstract low-level command generators below.

    Subclasses *should* override:

    - :meth:`default_config` — per-executor defaults dict.  Lowest
      layer of the resolution chain (after the dataclass defaults).
    - :meth:`apply_runtime_adjustments` — runtime-driven adjustments
      (e.g. Docker reads ``rootless`` / ``auto_user`` flags here).
    """

    eager = False  # don't initialize until requested

    # --- Subclass must define ---
    executor_name: ClassVar[str] = ""

    # Whether this executor distributes/uses a container image. Container-less
    # executors set False so the launch skips image distribution.
    needs_image: ClassVar[bool] = True

    # Status-discovery substrate.  Executors sharing a scope inspect *disjoint*
    # state on the *same* substrate (e.g. docker containers vs local pidfiles on
    # the SSH hosts) and are merged into one snapshot by
    # ``query_status_for_cluster``; a provider executor with its own control
    # plane declares a distinct scope (e.g. ``"k8s"`` / ``"modal"``) and is
    # queried alone.  A cluster's scope is the ``status_scope`` of the executor
    # it would launch with, so status sweeps exactly the executors that could
    # have placed workloads on that cluster.
    status_scope: ClassVar[str] = "host"

    # --- Optional channel-aware gating ---
    # When set to a registered feature-flag name (e.g. ``"executor.k8s"``),
    # this executor hides itself from the SAF extension registry (via
    # :meth:`is_multi_extension`) whenever that flag resolves off for the
    # active channel/config. ``None`` means always available.
    required_feature_flag: ClassVar[str | None] = None

    # --- SAF Plugin interface ---

    def name(self) -> str:
        return "sparkrun.executor.%s" % self.executor_name

    def extension_point_name(self, v: Variables) -> str:
        return EXT_EXECUTOR

    def is_enabled(self, v: Variables) -> bool:
        # Must return False for multi-extension plugins to prevent SAF's
        # single-extension cache from short-circuiting subsequent plugin
        # initializations under the same extension point.
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        # SAF only exposes a multi-extension plugin (via get_extensions) when
        # this returns True at registration. An executor gated behind a
        # feature flag hides itself here — it stays in the plugin registry but
        # is absent from get_extensions / list_executors / resolution, so a
        # gated selector fails closed (ExecutorUnavailableError). See
        # core.features and docs (Feature Flags).
        if self.required_feature_flag:
            from sparkrun.core.features import feature_gate_enabled

            return feature_gate_enabled(self.required_feature_flag, v)
        return True

    def initialize(self, v: Variables, logger=None) -> "Executor":
        return self

    def __init__(self, config: ExecutorConfig | None = None):
        super().__init__()
        self.config = config or ExecutorConfig()

    def finalize_config(self, *, config=None, v: Variables | None = None) -> None:
        """Post-construction hook to enrich ``self.config``.

        Called once by :func:`sparkrun.orchestration.executor.resolve_executor`
        after the executor is built, with the resolving
        :class:`~sparkrun.core.config.SparkrunConfig` (or ``None``).
        Executors that need session-level state the executor-agnostic
        chain can't provide (e.g. the kubectl binary path from sparkrun's
        managed cache) override this.  Default is a no-op — directly
        constructed executors (tests, ad-hoc callers) keep working
        unchanged.
        """
        return None

    # --- Per-executor layering hooks ---

    @classmethod
    def default_config(cls) -> dict:
        """Per-executor default settings — lowest-priority chain layer.

        Override to ship sensible defaults (e.g. DockerExecutor's
        ``shm_size``/``ipc=host``).  Bare base class contributes
        nothing — the dataclass field defaults already provide the
        bottom layer.
        """
        return {}

    @classmethod
    def apply_runtime_adjustments(cls, *, rootless: bool = True, auto_user: bool = True, **kwargs) -> dict:
        """Runtime-driven adjustments — sits above SparkrunConfig.

        Only executors that care about a given knob need to consume
        it.  Docker reads ``rootless``/``auto_user`` to flip
        ``privileged`` and inject ``--user $SHELL_USER``; Local and
        K8s ignore both.  ``**kwargs`` is accepted for forward
        compatibility (future signals like ``cluster=…``).
        """
        return {}

    # --- Low-level command generators (abstract) ---

    @abstractmethod
    def run_cmd(
        self,
        image: str,
        command: str = "",
        container_name: str | None = None,
        detach: bool = True,
        env: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
        extra_opts: list[str] | None = None,
        *,
        sparkrun_labels: dict[str, str] | None = None,
    ) -> str:
        """Generate a container run command string.

        Args:
            sparkrun_labels: Canonical sparkrun workload identity labels
                (``sparkrun.cluster_id`` / ``sparkrun.intent_id`` / etc.)
                from :meth:`workload_labels_for_cluster`.  Emitted as
                ``--label key=value`` flags (Docker) or annotations
                (K8s); ignored by executors that have no container
                concept (LocalExecutor).  Distinct from
                ``ExecutorConfig.labels`` (user-supplied), which still
                gets emitted alongside.
        """
        ...

    @abstractmethod
    def exec_cmd(
        self,
        container_name: str,
        command: str,
        detach: bool = False,
        env: dict[str, str] | None = None,
    ) -> str:
        """Generate a container exec command string."""
        ...

    @abstractmethod
    def stop_cmd(self, container_name: str, force: bool = True) -> str:
        """Generate a container stop/remove command string."""
        ...

    @abstractmethod
    def logs_cmd(
        self,
        container_name: str,
        follow: bool = False,
        tail: int | None = None,
    ) -> str:
        """Generate a container logs command string."""
        ...

    def read_logs_cmd(
        self,
        source: "LogSource",
        *,
        follow: bool = False,
        tail: int | None = None,
    ) -> str:
        """Generate the command that reads *source* on this substrate.

        The read-path peer of :meth:`logs_cmd`: the runtime says *what* to
        read (:class:`~sparkrun.core.log_source.LogSource`), the executor
        says *how* to read it here.

        The default ignores ``source.mode`` / ``source.path`` and reads the
        substrate's own log stream — correct for every executor whose
        workload writes to something the substrate already captures
        (``kubectl logs`` for k8s; the host logfile ``run_cmd`` redirects to
        for ``local``).  Only executors with an in-container filesystem
        indirection — docker, whose sleep-infinity containers hide the serve
        log from ``docker logs`` — need to override.
        """
        return self.logs_cmd(source.container, follow=follow, tail=tail)

    @abstractmethod
    def status_cmd(self, container_name: str) -> str:
        """Generate a command that exits 0 iff *container_name* is alive."""
        ...

    def exists_cmd(self, container_name: str) -> str:
        """Generate a command that exits 0 iff *container_name* is **present**.

        The teardown peer of :meth:`status_cmd`, which asks the narrower
        question "is it *running*?".  Present means anything teardown must
        still remove — for Docker that includes an exited-but-not-removed
        container, which ``status_cmd``'s ``docker ps`` cannot see and
        ``docker rm -f`` must still delete.

        The default is :meth:`status_cmd`, correct for every substrate where
        a workload's only state *is* its liveness (the ``local`` executor: a
        pidfile whose process is gone leaves nothing to remove).  Executors
        with a separate dead-but-present state override.
        """
        return self.status_cmd(container_name)

    def teardown_script(self, container_names: list[str] | tuple[str, ...]) -> str:
        """Generate a script that removes *container_names* here, and **verifies it**.

        The one seam through which every teardown path runs — ``sparkrun
        stop``, ``stop --all``, and post-launch-failure cleanup — so a
        workload is always torn down by the substrate that started it.  Before
        this existed, teardown emitted ``docker rm -f`` unconditionally, which
        meant a ``local`` executor's native process was asked about via Docker,
        truthfully reported as absent, and left running while the caller
        printed success.

        The generated script must:

        - remove every name in *container_names* (idempotently: a name that
          isn't there is not a failure, and any residue such as a stale
          pidfile is still cleaned up),
        - exit non-zero, naming survivors on stderr, if anything is still
          present afterwards,
        - print :data:`~sparkrun.orchestration.teardown.TEARDOWN_REMOVED_MARKER`
          with the number of workloads that were **actually present** before
          the removal — not the number of names attempted.

        The default composes :meth:`exists_cmd` and :meth:`stop_cmd`, so an
        executor gets a correct teardown from the primitives it already
        defines (this is the whole of the ``local`` and ``k8s`` implementations).

        Executors whose substrate can be *unavailable* rather than merely empty
        — a daemon or CLI that may be down, where "not present" and "cannot
        tell" are different answers — must override to check substrate health
        first, or an unreachable backend reads as a successful teardown.  That
        is exactly why :class:`~sparkrun.orchestration.executors.docker.DockerExecutor`
        overrides it.
        """
        from sparkrun.orchestration.teardown import format_teardown_removed

        if not container_names:
            return "echo %s\n" % quote(format_teardown_removed(0))

        lines = ["_sr_removed=0"]
        for name in container_names:
            # Count what was there *before* removing it, so a candidate name
            # that never existed isn't reported as a container we stopped.
            lines.append("if %s; then _sr_removed=$((_sr_removed + 1)); fi" % self.exists_cmd(name))
            # Runs unconditionally: teardown is idempotent, and for substrates
            # that leave residue behind a dead workload (a `local` pidfile) the
            # stop is what prunes it.
            lines.append(self.stop_cmd(name))

        lines.append('_sr_left=""')
        for name in container_names:
            lines.append('if %s; then _sr_left="$_sr_left "%s; fi' % (self.exists_cmd(name), quote(name)))
        lines.extend(
            [
                'if [ -n "$_sr_left" ]; then',
                '  echo "workloads still present:$_sr_left" >&2',
                "  exit 1",
                "fi",
                'echo "%s$_sr_removed"' % TEARDOWN_REMOVED_MARKER,
            ]
        )
        return "\n".join(lines) + "\n"

    @abstractmethod
    def inspect_exists_cmd(self, image: str) -> str:
        """Generate a command to check if an image exists locally."""
        ...

    @abstractmethod
    def pull_cmd(self, image: str) -> str:
        """Generate an image pull command string."""
        ...

    # --- Status introspection ---

    def query_status(
        self,
        hosts: list[str],
        *,
        ssh_kwargs: dict | None = None,
        host_hardware: "Mapping[str, HostHardware] | None" = None,
    ) -> "ClusterStatus":
        """Inspect sparkrun-launched workloads running on *hosts*.

        Default implementation returns a zero-occupancy snapshot — every
        host is treated as fully free with no running workloads.  This
        is a safe degradation for executors that don't yet implement
        introspection (e.g. the K8sExecutor draft); they satisfy the
        contract without lying about state they can't see.

        Concrete executors override to query their backend
        (``docker ps``, ``kubectl get pods``, local process state).
        """
        from sparkrun.core.cluster_status import empty_status

        return empty_status(hosts, executor=self.executor_name)

    def describe_terminated(
        self,
        sources: "list[LogSource]",
        *,
        ssh_kwargs: dict | None = None,
    ) -> "dict[tuple[str, str], TerminationInfo]":
        """Report on workloads :meth:`query_status` no longer reports as running.

        The post-mortem peer of :meth:`query_status`: that answers "what is
        running"; this answers, for something that is *not*, whether its remains
        are still on this substrate and how the operator inspects them.

        Keyed by ``(host, container)`` rather than by name alone, because a name
        is only unique per host — Ray worker containers share one name across
        every node.  A source this executor cannot speak for (unreachable host,
        no container engine, an inconclusive probe) is simply absent from the
        mapping, which callers must read as "cannot tell".

        Every part of the answer is substrate-specific — a stopped Docker
        container, a leftover ``local`` pidfile, a Failed Pod — which is why it
        belongs here rather than in the caller.  Notably the
        :attr:`~sparkrun.core.cluster_status.TerminationInfo.investigate_hints`:
        ``docker logs`` is wrong advice on a k8s cluster and meaningless for a
        native process.

        Best-effort, like :meth:`query_status` and the ``verify_*`` preflights:
        it must never raise, and it reports
        :attr:`~sparkrun.core.cluster_status.TerminationInfo.exists` as ``None``
        rather than guessing.  The base implementation returns ``{}`` — "cannot
        tell" — so an executor that does not implement it degrades to preserving
        cached job metadata rather than deleting it.
        """
        return {}

    # --- Preflight ---

    def verify_mount_sources(
        self,
        paths: list[str],
        hosts: list[str],
        *,
        ssh_kwargs: dict | None = None,
    ) -> dict[str, list[str]]:
        """Report identity-mount *paths* absent on this executor's substrate.

        Preflight peer of :meth:`query_status`: it answers "do these host paths
        already exist where the workload will run and the bind-mount will
        resolve?" for *this* executor's substrate.  Used to validate pre-placed
        model weights (an absolute-path ``model:`` or
        ``cluster_config.resolved_model_path``) *before* the launch skips model
        download + distribution — so a missing/typo'd path fails fast with a
        clear error instead of a cryptic container crash on the target.

        Returns ``{host: [missing_path, ...]}`` for hosts with confirmed-missing
        paths; an empty dict means "all present, or could not verify".  The
        default is the safe no-op (``{}``) — a provider executor that can't
        cheaply probe its substrate never blocks a launch.  Host-substrate
        executors (docker / local) override to SSH-probe the hosts; provider
        executors (k8s / modal) would probe their own volume/PVC/control plane.

        Args:
            paths: Absolute host paths that must pre-exist on every host.
            hosts: Target hosts for the launch.
            ssh_kwargs: Connection settings for host-substrate probes.
        """
        del paths, hosts, ssh_kwargs  # base is a no-op; substrate-aware executors override
        return {}

    def ensure_runtime_cache(
        self,
        mounts: "RuntimeCacheMounts",
        hosts: list[str],
        *,
        ssh_kwargs: dict | None = None,
    ) -> None:
        """Prepare the persistent compilation/autotune cache on this substrate.

        Write-path peer of :meth:`verify_mount_sources`: that one asks "do the
        paths I will mount already exist?", this one *makes* the paths it is
        about to mount.  Three things in one pass — create the directories,
        stamp the last-used marker, sweep aged sibling trees (see
        :mod:`sparkrun.orchestration.runtime_cache`).

        Creating them explicitly is not optional on a bind-mount substrate:
        Docker materializes a missing ``-v`` source as a **root-owned**
        directory, which breaks rootless docker and breaks the ``local``
        executor outright.

        The default is the safe no-op — a provider executor whose cache lives
        in a managed volume overrides (or legitimately does nothing).
        Best-effort by contract, like :meth:`verify_mount_sources` and
        :meth:`query_status`: this must never raise, because a cache we could
        not prepare costs a recompile while a raised exception costs the
        launch.

        Args:
            mounts: The resolved plan from
                :func:`sparkrun.core.runtime_cache.build_runtime_cache_mounts`.
            hosts: Target hosts for the launch.
            ssh_kwargs: Connection settings for host-substrate executors.
        """
        del mounts, hosts, ssh_kwargs  # base is a no-op; substrate-aware executors override

    def verify_command_passthrough(
        self,
        image: str,
        hosts: list[str],
        *,
        ssh_kwargs: dict | None = None,
    ) -> "EntrypointProbe | None":
        """Report whether *image* would swallow the command sparkrun appends.

        Second preflight peer of :meth:`query_status`, alongside
        :meth:`verify_mount_sources`: that one asks "do the paths I will mount
        exist on this substrate?", this one asks "will the command I append
        actually run on this image?".

        sparkrun always emits its launcher as CMD *arguments*, so an image whose
        ENTRYPOINT consumes them (``ENTRYPOINT ["vllm","serve"]``) runs
        something else entirely — while a passthrough wrapper ending in
        ``exec "$@"`` (``/opt/nvidia/nvidia_entrypoint.sh``, inherited by most
        NGC images) is harmless and must *not* be cleared.  The two are
        indistinguishable by inspection, so substrate-aware executors settle it
        empirically; see :mod:`sparkrun.containers.entrypoint`.

        Returns the probe result, or ``None`` when this executor has no opinion.
        The default is the safe no-op — container-less (``local``) and provider
        executors that don't compose a Docker-style argv never block a launch.
        Best-effort like :meth:`verify_mount_sources`: only a *confirmed*
        consuming entrypoint is reported; "could not tell" reads as fine.

        Args:
            image: Resolved container image reference.
            hosts: Target hosts for the launch.
            ssh_kwargs: Connection settings for host-substrate probes.
        """
        del image, hosts, ssh_kwargs  # base is a no-op; substrate-aware executors override
        return None

    @classmethod
    def workload_labels_for_cluster(
        cls,
        cluster_id: str,
        recipe=None,
        runtime=None,
        rank: int | None = None,
    ) -> dict[str, str]:
        """Build the canonical sparkrun label dict from a cluster + recipe + runtime.

        Convenience wrapper over :meth:`workload_labels` that pulls
        ``recipe.qualified_name`` and ``runtime.runtime_name`` off the
        objects callers already carry at launch time.  Returns an empty
        dict only when *cluster_id* itself is falsy; otherwise the
        cluster_id (and derived intent_id when canonical) is always
        emitted.

        Args:
            cluster_id: Full sparkrun cluster identifier.
            recipe: Optional :class:`Recipe` — :attr:`Recipe.qualified_name`
                becomes ``sparkrun.recipe`` when truthy, and
                :attr:`Recipe.model` becomes ``sparkrun.model``.
            runtime: Optional :class:`RuntimePlugin` —
                ``runtime.runtime_name`` becomes ``sparkrun.runtime``.
            rank: Optional rank index.
        """
        if not cluster_id:
            return {}
        recipe_name = getattr(recipe, "qualified_name", None) if recipe is not None else None
        model = getattr(recipe, "model", None) if recipe is not None else None
        served_model_name = getattr(recipe, "effective_served_model_name", None) if recipe is not None else None
        runtime_name = getattr(runtime, "runtime_name", None) if runtime is not None else None
        return cls.workload_labels(
            cluster_id=cluster_id,
            recipe_name=recipe_name,
            runtime_name=runtime_name,
            rank=rank,
            model=model,
            served_model_name=served_model_name,
        )

    @classmethod
    def workload_labels(
        cls,
        cluster_id: str,
        recipe_name: str | None = None,
        runtime_name: str | None = None,
        rank: int | None = None,
        intent_id: str | None = None,
        model: str | None = None,
        served_model_name: str | None = None,
    ) -> dict[str, str]:
        """Return the canonical sparkrun label set for a workload.

        Used by run-script generators to tag containers/pods so
        :meth:`query_status` can recover workload identity from the
        backend's native introspection (``docker ps``, ``kubectl get``).

        Returns a plain ``dict[str, str]``; concrete executors choose
        how to emit them (``--label key=value`` for Docker, k8s
        ``metadata.labels``, …).

        Args:
            cluster_id: Full sparkrun cluster identifier.
            recipe_name: Recipe qualified name (omitted when falsy).
            runtime_name: Runtime family (omitted when falsy).
            rank: Global rank index (``0`` is emitted; ``None`` is not).
            intent_id: Hex intent identifier.  When omitted, the
                method attempts to derive it from *cluster_id* (using
                the canonical ``sparkrun_<intent>_<token>`` shape) so
                callers that already carry a cluster_id get the label
                for free.
            model: HuggingFace model ID (omitted when falsy).
            served_model_name: Name the model is served under via the
                inference API (omitted when falsy).
        """
        labels: dict[str, str] = {LABEL_CLUSTER_ID: cluster_id}
        if intent_id is None:
            intent_id = _intent_id_from_cluster_id(cluster_id)
        if intent_id:
            labels[LABEL_INTENT_ID] = intent_id
        if recipe_name:
            labels[LABEL_RECIPE] = recipe_name
        if runtime_name:
            labels[LABEL_RUNTIME] = runtime_name
        if rank is not None:
            labels[LABEL_RANK] = str(rank)
        if model:
            labels[LABEL_MODEL] = model
        if served_model_name:
            labels[LABEL_SERVED_MODEL_NAME] = served_model_name
        return labels

    # --- Naming helpers (concrete, shared across executors) ---

    @staticmethod
    def container_name(cluster_id: str, role: str = "head") -> str:
        """Generate a deterministic container name: ``{cluster_id}_{role}``."""
        return "%s_%s" % (cluster_id, role)

    @staticmethod
    def node_container_name(cluster_id: str, rank: int) -> str:
        """Generate a ranked node container name: ``{cluster_id}_node_{rank}``."""
        return "%s_node_%d" % (cluster_id, rank)

    @staticmethod
    def enumerate_containers(cluster_id: str, num_hosts: int) -> list[str]:
        """Return all possible container names for a cluster.

        Covers solo, Ray (head/worker), and native (node_N) patterns.
        """
        names = [
            "%s_solo" % cluster_id,
            "%s_head" % cluster_id,
            "%s_worker" % cluster_id,
        ]
        for rank in range(num_hosts):
            names.append("%s_node_%d" % (cluster_id, rank))
        return names

    # --- High-level script generators (concrete) ---

    def generate_launch_script(
        self,
        image: str,
        container_name: str,
        command: str,
        env: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
        nccl_env: dict[str, str] | None = None,
        detach: bool = True,
        extra_docker_opts: list[str] | None = None,
        *,
        sparkrun_labels: dict[str, str] | None = None,
    ) -> str:
        """Generate a script that cleans up then launches a container."""

        all_env = merge_env(nccl_env, env)
        cleanup = self.stop_cmd(container_name)
        run = self.run_cmd(
            image=image,
            command=command,
            container_name=container_name,
            detach=detach,
            env=all_env,
            volumes=volumes,
            extra_opts=extra_docker_opts,
            sparkrun_labels=sparkrun_labels,
        )

        template = read_script("container_launch.sh")
        return template.format(
            container_name=quote(container_name),
            image=quote(image),
            cleanup_cmd=cleanup,
            run_cmd=run,
        )

    def generate_exec_serve_script(
        self,
        container_name: str,
        serve_command: str,
        env: dict[str, str] | None = None,
        detached: bool = True,
        volumes: dict[str, str] | None = None,
        *,
        sparkrun_labels: dict[str, str] | None = None,
    ) -> str:
        """Generate a script that executes the serve command inside a running container.

        ``sparkrun_labels`` is accepted for API symmetry with the other
        generators; Docker exec does not support attaching labels (those
        live on the parent container), so it is ignored here.  Callers
        attach labels when the container itself is created via
        ``generate_launch_script`` / ``generate_node_script`` /
        ``generate_ray_*_script``.

        ``volumes`` is accepted for API symmetry; the container mounts were
        already established at container-create time
        (``generate_launch_script``), so the Docker exec path ignores it.
        Only container-less executors (LocalExecutor) consult it — to
        reverse-map container-path env values back to their host source.
        """
        del sparkrun_labels  # accepted but unused — exec inherits labels from the container
        del volumes  # accepted but unused — mounts were set at container-create time

        env_exports = ""
        if env:
            for key, value in sorted(env.items()):
                env_exports += "export %s=%s; " % (key, quote(str(value)))

        full_cmd = "%s%s" % (env_exports, serve_command)

        # Base64 encode the command to avoid all bash string-escaping/quoting bugs
        # when passing it into `docker exec ... bash -c "..."`
        b64_cmd = b64_encode_cmd(full_cmd)

        script_name = "exec_serve_detached.sh" if detached else "exec_serve_foreground.sh"
        template = read_script(script_name)
        return template.format(
            container_name=quote(container_name),
            b64_cmd=b64_cmd,
        )

    def generate_ray_head_script(
        self,
        image: str,
        container_name: str,
        ray_port: int = 46379,
        dashboard_port: int = 8265,
        dashboard: bool = False,
        env: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
        nccl_env: dict[str, str] | None = None,
        extra_docker_opts: list[str] | None = None,
        *,
        sparkrun_labels: dict[str, str] | None = None,
    ) -> str:
        """Generate a script that starts a Ray head node in a container."""

        all_env = merge_env({"RAY_memory_monitor_refresh_ms": "0"}, nccl_env, env)

        # Ray starts the dashboard by default when --include-dashboard is absent,
        # so disabling it must be explicit. When on, bind 0.0.0.0 so the dashboard
        # is reachable at the head IP (the container runs with host networking).
        if dashboard:
            dashboard_flags = "--include-dashboard=True --dashboard-host 0.0.0.0 --dashboard-port %d " % dashboard_port
        else:
            dashboard_flags = "--include-dashboard=False "

        cleanup = self.stop_cmd(container_name)
        run = self.run_cmd(
            image=image,
            command=("ray start --block --head --port %d --node-ip-address $NODE_IP %s--disable-usage-stats" % (ray_port, dashboard_flags)),
            container_name=container_name,
            detach=True,
            env=all_env,
            volumes=volumes,
            extra_opts=extra_docker_opts,
            sparkrun_labels=sparkrun_labels,
        )

        template = read_script("ray_head.sh")
        return template.format(
            cleanup_cmd=cleanup,
            run_cmd=run,
        )

    def generate_ray_worker_script(
        self,
        image: str,
        container_name: str,
        head_ip: str,
        ray_port: int = 46379,
        env: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
        nccl_env: dict[str, str] | None = None,
        extra_docker_opts: list[str] | None = None,
        *,
        sparkrun_labels: dict[str, str] | None = None,
    ) -> str:
        """Generate a script that starts a Ray worker node."""

        all_env = merge_env({"RAY_memory_monitor_refresh_ms": "0"}, nccl_env, env)

        cleanup = self.stop_cmd(container_name)
        run = self.run_cmd(
            image=image,
            command=("ray start --block --address=%s:%d --node-ip-address $NODE_IP" % (head_ip, ray_port)),
            container_name=container_name,
            detach=True,
            env=all_env,
            volumes=volumes,
            extra_opts=extra_docker_opts,
            sparkrun_labels=sparkrun_labels,
        )

        template = read_script("ray_worker.sh")
        return template.format(
            cleanup_cmd=cleanup,
            run_cmd=run,
            head_ip=head_ip,
            ray_port=ray_port,
        )

    def generate_node_script(
        self,
        image: str,
        container_name: str,
        serve_command: str,
        label: str = "node",
        env: dict[str, str] | None = None,
        volumes: dict[str, str] | None = None,
        nccl_env: dict[str, str] | None = None,
        extra_docker_opts: list[str] | None = None,
        *,
        sparkrun_labels: dict[str, str] | None = None,
    ) -> str:
        """Generate a script that launches a container with a direct entrypoint command."""

        all_env = merge_env(nccl_env, env)
        cleanup = self.stop_cmd(container_name)
        run = self.run_cmd(
            image=image,
            command=serve_command,
            container_name=container_name,
            detach=True,
            env=all_env,
            volumes=volumes,
            extra_opts=extra_docker_opts,
            sparkrun_labels=sparkrun_labels,
        )

        return (
            "#!/bin/bash\n"
            "set -uo pipefail\n"
            "\n"
            "printf 'Cleaning up existing container: %%s\\n' %(name)s\n"
            "%(cleanup)s\n"
            "\n"
            "printf 'Launching %%s: %%s\\n' %(label)s %(name)s\n"
            "%(run_cmd)s\n"
            "\n"
            "# Verify container started\n"
            "sleep 1\n"
            "if docker ps --format '{{.Names}}' | grep -q '^%(name)s$'; then\n"
            "    printf 'Container %%s launched successfully\\n' %(name)s\n"
            "else\n"
            "    printf 'ERROR: Container %%s failed to start\\n' %(name)s >&2\n"
            "    docker logs %(name)s 2>&1 | tail -20 || true\n"
            "    exit 1\n"
            "fi\n"
        ) % {
            "name": quote(container_name),
            "cleanup": cleanup,
            "run_cmd": run,
            "label": quote(label),
        }


def _intent_id_from_cluster_id(cluster_id: str) -> str | None:
    """Derive the intent_id from a sparkrun cluster_id.

    Returns ``None`` when *cluster_id* doesn't parse as a canonical
    ``sparkrun_<intent>_<placement_token>`` — callers may treat absence
    as "this workload's identifier is not canonical, fall back to
    cluster_id matching".
    """
    from sparkrun.orchestration.job_metadata import parse_cluster_id

    if not cluster_id:
        return None
    try:
        intent_id, _ = parse_cluster_id(cluster_id)
        return intent_id
    except ValueError:
        return None


def accelerator_vendor_for(host_hardware) -> str | None:
    """Return the single shared accelerator vendor on *host_hardware*, or ``None``.

    Used by per-host executor config resolution to decide whether to
    emit NVIDIA ``--gpus``, ROCm device flags, etc.  Returns ``None``
    when the host has zero accelerators or advertises more than one
    vendor (e.g. an Apple M5 host with a discrete NVIDIA GPU); the
    caller is expected to fall back to recipe-level overrides or refuse
    to auto-pack.
    """
    if host_hardware is None:
        return None
    vendors = getattr(host_hardware, "vendors", None)
    if not vendors or len(vendors) != 1:
        return None
    return next(iter(vendors))
