"""Experimental Kubernetes executor and native JobSet lifecycle adapter.

Native API launches use the manifest path in ``run.py``. This executor handles
portable-ID JobSet discovery and confirmed controller teardown for those launches.
The remaining command generators below implement the older direct-Pod path:

:class:`K8sExecutor` launches workloads as Kubernetes Pods via the
``kubectl`` CLI instead of running Docker containers directly on each
host.  The orchestration / SSH dispatch layer is unchanged: the
executor still emits bash scripts that get piped via
``ssh <host> bash -s`` (or run locally).  Each "host" becomes a place
where ``kubectl`` is invoked — typically the control machine itself
when ``host_list`` is a single ``localhost``.

This is an **experimental draft**.  It is intentionally minimal:

* Pods are created with ``kubectl run`` rather than full manifests, so
  many advanced spec features (init containers, sidecars, custom
  scheduler hints) are unreachable.  Use ``command_prefix`` /
  ``extra_opts``-flavored fields if you need to wedge in extra
  ``kubectl run`` arguments.
* ``--privileged``, ``--shm-size``, and other Docker-specific options
  are silently dropped.  Set ``k8s_node_selector`` /
  ``k8s_image_pull_policy`` instead.
* GPU allocation goes through ``--limits=nvidia.com/gpu=N`` when
  ``gpus`` looks like ``"device=0,1"`` (2 GPUs) or ``"all"`` (1 GPU
  as a conservative default).  Anything fancier is dropped with a
  warning — recipes that need fine-grained scheduling should use a
  proper manifest pathway (out of scope for the draft).
* Ray cluster strategy is unsupported (raises).
* Multi-host: one Pod per host_list entry.  No StatefulSet or Job
  orchestration — the calling code already iterates hosts.
* ``inspect_exists_cmd`` / ``pull_cmd`` are no-ops (the cluster pulls
  on demand).

The Pod name reuses the runtime's ``container_name`` so the lifecycle
identity (``cluster_id_solo`` / ``cluster_id_node_0`` / …) is
consistent across executors and the existing
``enumerate_cluster_containers`` helper keeps working.
"""

from __future__ import annotations

from .config import K8sSettings

from dataclasses import dataclass
import logging
import json
import re
from typing import TYPE_CHECKING

from sparkrun.orchestration.executors._base import Executor, ExecutorConfig, ExecutorTarget, interactive_shell_command
from sparkrun.core.application_profile import get_application_profile, resource_name
from sparkrun.core.ownership import OWNER_LABEL, assert_resource_namespace
from sparkrun.plugins.k8s.orchestration.client import KubectlClient
from sparkrun.utils.shell import b64_wrap_bash, quote

if TYPE_CHECKING:
    from sparkrun.core.cluster_status import TerminationInfo
    from sparkrun.core.log_source import LogSource

logger = logging.getLogger(__name__)

_GPUS_DEVICE_RE = re.compile(r"device=([0-9,]+)")


@dataclass
class K8sExecutorConfig(ExecutorConfig):
    """Kubernetes-only executor settings, resolved by the shared config chain."""

    executor_type: str = "k8s"
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
    def from_chain(cls, chain):
        config = super().from_chain(chain)
        for key in ("k8s_namespace", "k8s_context", "k8s_node_selector", "k8s_image_pull_policy", "kubeconfig", "kubectl_path"):
            value = chain.get(key)
            if value:
                setattr(config, key, str(value))
        return config


class K8sExecutor(Executor):
    """``kubectl``-driven executor (experimental draft).

    Provides JobSet status/teardown and direct-Pod command generators.
    The user is expected to
    have ``kubectl`` on PATH and a current context that points at a
    cluster reachable from the script's execution host.
    """

    config: K8sExecutorConfig

    def __init__(self, config: ExecutorConfig | None = None):
        resolved = K8sExecutorConfig() if config is None else config
        if not isinstance(resolved, K8sExecutorConfig):
            raise TypeError("K8sExecutor requires K8sExecutorConfig")
        super().__init__(resolved)
        self.config = resolved

    executor_name = "k8s"
    config_class = K8sExecutorConfig
    required_feature_flag = "executor.k8s"
    # Own control plane (kubectl), not the SSH-host substrate — queried alone
    # for a k8s cluster, never merged with the docker/local host executors.
    status_scope = "k8s"
    supports_host_endpoint = False

    def is_multi_extension(self, v):
        from sparkrun.core.features import feature_gate_enabled
        from . import FEATURE_FLAG

        return feature_gate_enabled(FEATURE_FLAG, v) and super().is_multi_extension(v)

    # No Docker-flavoured defaults; ``--privileged`` / ``--shm-size``
    # etc. don't apply.  No rootless/auto_user handling either.

    # ------------------------------------------------------------------
    # Binary resolution + common kubectl prefix
    # ------------------------------------------------------------------

    def finalize_config(self, *, config=None, v=None) -> None:
        """Resolve the target and an available kubectl binary into executor config.

        Only an *already-available* binary is used (an explicit config
        path, a cached download, or ``kubectl`` on PATH) — launch-time
        never triggers an implicit network download.  Acquiring / updating
        the binary is the job of ``sparkrun setup k8s kubectl``.  When
        nothing resolves, ``kubectl_path`` stays ``None`` and the emitted
        commands fall back to a bare ``kubectl`` (PATH lookup on the
        execution host).
        """
        from sparkrun.core.config import SparkrunConfig

        if isinstance(config, SparkrunConfig):
            from .orchestration.context import resolve_kube_target

            target = resolve_kube_target(
                config,
                kubeconfig=self.config.kubeconfig,
                context=self.config.k8s_context,
                namespace=self.config.k8s_namespace,
            )
            self.config.kubeconfig = target.kubeconfig
            self.config.k8s_context = target.context
            self.config.k8s_namespace = target.namespace or resource_name()
        if self.config.kubectl_path or config is None:
            return
        # Binary resolution needs the real config surface (cache dir, pins).
        # Partial stand-ins (tests exercising chain ordering) are skipped.
        if not isinstance(config, SparkrunConfig):
            return
        from sparkrun.plugins.k8s.orchestration import ensure_kubectl
        from sparkrun.plugins.k8s.orchestration.errors import K8sError

        version = K8sSettings(config).kubectl_version or K8sSettings(config).kubectl_pinned_version(self.config.k8s_context)
        try:
            binary = ensure_kubectl(
                config.cache_dir,
                version=version,
                explicit_path=K8sSettings(config).kubectl_path,
                allow_download=False,
            )
        except K8sError:
            logger.debug("No managed kubectl available; K8sExecutor will use bare 'kubectl'.")
            return
        self.config.kubectl_path = str(binary.path)

    def resolve_target(self, *, dry_run=False) -> ExecutorTarget:
        from .orchestration.context import _pin_client_target

        client = self._client()
        # Previews do not require a configured local kubectl. Real plans pin
        # current-context before it can influence identity or status queries.
        if not dry_run:
            _pin_client_target(client)
        return ExecutorTarget(
            self.executor_name,
            {
                "kubeconfig": client.kubeconfig,
                "k8s_context": client.context,
                "k8s_namespace": client.namespace,
                "kubectl_path": self.config.kubectl_path,
            },
            destination_key=json.dumps([client.kubeconfig, client.context, client.namespace], separators=(",", ":")),
        )

    def _client(self) -> KubectlClient:
        """Build a :class:`KubectlClient` from this executor's config."""
        cfg = self.config
        return KubectlClient(
            cfg.kubectl_path or "kubectl",
            kubeconfig=cfg.kubeconfig,
            context=cfg.k8s_context,
            namespace=cfg.k8s_namespace or resource_name(),
        )

    def stop_workload(self, cluster_id: str, *, metadata: dict | None = None) -> int | None:
        from .lifecycle import stop_native_workload

        return stop_native_workload(self._client(), cluster_id, (metadata or {}).get("native_resource"))

    def query_status(self, hosts, *, ssh_kwargs=None, host_hardware=None):
        from .lifecycle import query_native_status

        return query_native_status(self._client(), hosts)

    def native_log_sources(self, cluster_id, *, metadata, scope, cancel=None):
        from .orchestration.logs import resolve_sources

        reference = metadata.get("native_resource")
        if not isinstance(reference, dict) or reference.get("kind") not in {"JobSet", "Job"}:
            raise ValueError("Unsupported native Kubernetes workload reference")
        return resolve_sources(
            self._client(),
            name=reference.get("name"),
            kind=reference["kind"].lower(),
            cluster_id=cluster_id,
            uid=reference.get("uid"),
            scope=scope,
            cancel=cancel,
        )

    def read_logs_cmd(self, source, *, follow=False, tail=None):
        if source.resource is None:
            return super().read_logs_cmd(source, follow=follow, tail=tail)
        from .orchestration.logs import read_command

        return read_command(self._client(), source, follow=follow, tail=tail)

    def validate_log_source(self, source, *, cancel=None, timeout=None):
        from .orchestration.logs import validate_source

        validate_source(self._client(), source, cancel=cancel, timeout=timeout)

    def _kubectl_prefix(self) -> str:
        """Build ``kubectl [--kubeconfig K] [--context C] [-n NS]`` prefix."""
        return self._client().prefix_string()

    def _gpu_limit(self) -> str | None:
        """Translate ``gpus`` into a ``nvidia.com/gpu`` resource limit.

        ``"all"`` → 1 (conservative — k8s schedules per node).
        ``"device=0,2"`` → 2.
        Anything else → ``None`` (skip the limit, log a warning).
        """
        gpus = (self.config.gpus or "").strip()
        if not gpus or gpus.lower() in ("none", "0"):
            return None
        if gpus.lower() == "all":
            return "1"
        m = _GPUS_DEVICE_RE.match(gpus)
        if m:
            count = len([x for x in m.group(1).split(",") if x.strip()])
            return str(max(count, 1))
        logger.warning(
            "K8sExecutor: gpus=%r is not translatable to a numeric GPU limit; scheduling without a GPU resource request.",
            gpus,
        )
        return None

    def _command_entrypoint(self) -> str:
        """Return the executable used for explicit K8s command overrides."""
        if self.config.entrypoint:
            return self.config.entrypoint
        return "bash"

    # ------------------------------------------------------------------
    # Low-level command generators (Executor ABC)
    # ------------------------------------------------------------------

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
        """Emit a ``kubectl run`` command that creates a single Pod.

        ``sparkrun_labels`` are emitted as one ``--labels=key=value`` flag
        per pair — kubectl run accepts comma-separated labels but
        emitting separately sidesteps any quoting ambiguity in shell
        contexts.  User-supplied ``cfg.labels`` is still emitted below.
        """
        if not container_name:
            raise ValueError("K8sExecutor.run_cmd requires container_name (used as Pod name)")
        if not image:
            raise ValueError("K8sExecutor.run_cmd requires image")

        cfg = self.config
        prefix = self._kubectl_prefix()
        parts: list[str] = [
            prefix,
            "run",
            quote(container_name),
            "--image=%s" % quote(image),
            "--restart=Never",
        ]
        if cfg.k8s_image_pull_policy:
            parts.append("--image-pull-policy=%s" % quote(cfg.k8s_image_pull_policy))
        if cfg.k8s_node_selector:
            # --overrides is the kubectl run lever for arbitrary fields
            # but node-selector has its own flag — easier surface.
            parts.append("--overrides=%s" % quote(_node_selector_overrides(cfg.k8s_node_selector)))
        gpu_limit = self._gpu_limit()
        if gpu_limit:
            parts.append("--limits=%s" % quote("nvidia.com/gpu=%s" % gpu_limit))
        if cfg.memory_limit:
            parts.append("--limits=%s" % quote("memory=%s" % cfg.memory_limit))
        if env:
            for key, value in sorted(env.items()):
                parts.append("--env=%s" % quote("%s=%s" % (key, value)))
        if cfg.labels:
            for lbl in cfg.labels:
                parts.append("--labels=%s" % quote(lbl))
        sparkrun_labels = {**(sparkrun_labels or {}), OWNER_LABEL: get_application_profile().id}
        if sparkrun_labels:
            for key, value in sorted(sparkrun_labels.items()):
                parts.append("--labels=%s" % quote("%s=%s" % (key, value)))
        if extra_opts:
            # extra_opts are docker --run flags; pass through verbatim
            # only when they look like kubectl-compatible ``--key=val``.
            for opt in extra_opts:
                if opt.startswith("--") and "=" in opt:
                    parts.append(opt)

        # Command runs inside the pod; wrap via base64 to dodge quoting
        # bugs in deeply nested shells.
        if command:
            if cfg.entrypoint is not None:
                parts.append("--command")
            parts.append("--")
            parts.extend([quote(self._command_entrypoint()), "-c", b64_wrap_bash(command)])

        return " ".join(parts)

    def exec_cmd(
        self,
        container_name: str,
        command: str,
        detach: bool = False,
        env: dict[str, str] | None = None,
    ) -> str:
        """Run *command* inside an already-running Pod via ``kubectl exec``."""
        prefix = self._kubectl_prefix()
        env_prelude = ""
        if env:
            env_prelude = "; ".join("export %s=%s" % (k, quote(str(v))) for k, v in sorted(env.items())) + "; "
        full = env_prelude + command
        flags = "-i" if not detach else "-d"
        # kubectl exec doesn't have a -d; for "detach" we just fire-and-forget.
        if detach:
            # Pseudo-detach: pipe to nohup inside the pod.
            full = "nohup bash -c " + b64_wrap_bash(full) + " >/dev/null 2>&1 &"
            flags = "-i"
            return "%s exec %s %s -- bash -c %s" % (prefix, flags, quote(container_name), b64_wrap_bash(full))
        return "%s exec %s %s -- bash -c %s" % (prefix, flags, quote(container_name), b64_wrap_bash(full))

    def interactive_exec_argv(self, host, container_name, command=None, *, tty=False, ssh_kwargs=None, resource=None):
        """``kubectl exec -i [-t] <pod> [-c <container>] -- sh -c ...``, run on the control machine.

        The control plane is reached from here, so *host* and *ssh_kwargs* are
        unused. A native resource names its pod and container separately (and
        its namespace, which overrides the configured one: pflag keeps the last
        ``-n``); otherwise *container_name* is the pod, as in :meth:`exec_cmd`.
        """
        argv = self._client().base_args()
        if resource is not None:
            argv += ["-n", resource.namespace, "exec", "-i"] + (["-t"] if tty else []) + [resource.name, "-c", container_name]
        else:
            argv += ["exec", "-i"] + (["-t"] if tty else []) + [container_name]
        return argv + ["--", "sh", "-c", interactive_shell_command(command)]

    def stop_cmd(self, container_name: str, force: bool = True) -> str:
        """Delete the Pod; ``--ignore-not-found`` so it's idempotent."""
        prefix = self._kubectl_prefix()
        flags = "--ignore-not-found"
        if force:
            flags += " --grace-period=0 --force"
        assert_resource_namespace(container_name)
        selector = quote('{.metadata.labels.sparkrun\\.distribution}{"|"}{.metadata.labels.app\\.kubernetes\\.io/managed-by}')
        legacy = ' || [ "$_identity" = "|sparkrun" ]' if get_application_profile().id == "sparkrun" else ""
        return (
            "_identity=$(%s get pod %s --ignore-not-found -o jsonpath=%s) || exit 1; "
            'if [ -z "$_identity" ]; then :; '
            'elif [ "${_identity%%%%|*}" = %s ]%s; then '
            "%s delete pod %s %s; else exit 1; fi"
        ) % (prefix, quote(container_name), selector, quote(get_application_profile().id), legacy, prefix, quote(container_name), flags)

    def logs_cmd(
        self,
        container_name: str,
        follow: bool = False,
        tail: int | None = None,
    ) -> str:
        """Stream Pod logs via ``kubectl logs``."""
        prefix = self._kubectl_prefix()
        parts = [prefix, "logs"]
        if follow:
            parts.append("-f")
        if tail is not None:
            parts.append("--tail=%d" % int(tail))
        parts.append(quote(container_name))
        return " ".join(parts)

    def status_cmd(self, container_name: str) -> str:
        """Exit 0 iff the Pod is in a Running phase."""
        prefix = self._kubectl_prefix()
        # jsonpath returns empty string when Pod is missing → fails the test.
        return "[ \"$(%s get pod %s -o jsonpath='{.status.phase}' 2>/dev/null)\" = 'Running' ]" % (prefix, quote(container_name))

    def describe_terminated(
        self,
        sources: "list[LogSource]",
        *,
        ssh_kwargs: dict | None = None,
    ) -> "dict[tuple[str, str], TerminationInfo]":
        """Read the terminal phase of Pods the status sweep no longer reports.

        A ``Succeeded`` / ``Failed`` Pod is retained by the API server until
        something deletes it, so unlike a ``--rm`` container its remains are
        normally still there — and ``kubectl logs`` still works on it, which is
        exactly what the operator wants next.  An empty phase means the Pod is
        gone from the API server entirely.
        """
        from sparkrun.core.cluster_status import TerminationInfo
        from sparkrun.orchestration.ssh import run_remote_scripts_parallel

        if not sources:
            return {}

        ssh_kwargs = ssh_kwargs or {}
        by_host: dict[str, list[str]] = {}
        for source in sources:
            by_host.setdefault(source.host, []).append(source.container)

        hosts = list(by_host)
        prefix = self._kubectl_prefix()
        names = sorted({s.container for s in sources})
        script = (
            "\n".join(
                "printf '%%s\\t%%s\\n' %s \"$(%s get pod %s -o jsonpath='{.status.phase}' 2>/dev/null)\"" % (quote(n), prefix, quote(n))
                for n in names
            )
            + "\n"
        )
        try:
            results = run_remote_scripts_parallel(
                hosts,
                script,
                ssh_user=ssh_kwargs.get("ssh_user"),
                ssh_key=ssh_kwargs.get("ssh_key"),
                ssh_options=ssh_kwargs.get("ssh_options"),
                timeout=ssh_kwargs.get("timeout", 15),
                quiet=True,
                allow_local=True,
            )
        except Exception:  # noqa: BLE001 — best-effort, like query_status
            logger.debug("describe_terminated: probe failed", exc_info=True)
            return {}

        found: dict[tuple[str, str], TerminationInfo] = {}
        result_by_host = {r.host: r for r in results}
        for host in hosts:
            r = result_by_host.get(host)
            if r is None or r.returncode != 0:
                logger.debug("describe_terminated: inconclusive for %r (rc=%s)", host, getattr(r, "returncode", "n/a"))
                continue
            phases = {}
            for line in (r.stdout or "").splitlines():
                name, sep, phase = line.partition("\t")
                if sep and name.strip():
                    phases[name.strip()] = phase.strip()
            for container in by_host[host]:
                phase = phases.get(container) or ""
                if phase:
                    found[(host, container)] = TerminationInfo(
                        exists=True,
                        detail="pod phase %s" % phase,
                        investigate_hints=(
                            "kubectl logs %s" % container,
                            "kubectl describe pod %s" % container,
                        ),
                    )
                else:
                    found[(host, container)] = TerminationInfo(exists=False, detail="no pod by that name exists")
        return found

    def inspect_exists_cmd(self, image: str) -> str:
        """No-op: Kubernetes pulls images on Pod creation."""
        return "true"

    def pull_cmd(self, image: str) -> str:
        """No-op: Kubernetes pulls images on Pod creation."""
        return "true"

    # ------------------------------------------------------------------
    # High-level script generators (override Executor defaults)
    # ------------------------------------------------------------------

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
        """Preflight: delete any stale Pod with the same name.

        The actual workload Pod is created by
        :meth:`generate_exec_serve_script` so the serve command (which
        sparkrun knows after solo-mode runtimes resolve it) lands in the
        Pod's primary container.  ``sparkrun_labels`` is preserved by
        forwarding through to :meth:`generate_exec_serve_script` —
        callers thread the same dict into both calls so the Pod that
        actually gets created carries the labels.
        """
        del sparkrun_labels  # preflight only — labels attach at Pod-create time
        cleanup = self.stop_cmd(container_name)
        return (
            "#!/bin/bash\n"
            "set -uo pipefail\n"
            "# K8sExecutor preflight: ensure no stale Pod with this name.\n"
            "%(cleanup)s\n"
            'printf "K8sExecutor: preflight complete for %%s\\n" %(name)s\n'
        ) % {
            "cleanup": cleanup,
            "name": quote(container_name),
        }

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
        """Create the workload Pod with *serve_command* as its entrypoint.

        For K8s there is no separate ``docker run`` + ``docker exec``
        split — the Pod *is* the workload, so we create it directly
        with the serve command.  We need an image, which we don't have
        here; pull it from the executor config via ``self.config`` or
        require the runtime to have placed it in env.  As a draft, we
        rely on a sentinel ``SPARKRUN_K8S_IMAGE`` environment variable
        passed in *env* — runtime authors who target K8s explicitly
        should set this in their generated env.

        ``sparkrun_labels`` is forwarded to :meth:`run_cmd` so the
        Pod manifest carries the canonical sparkrun label set.
        ``volumes`` is accepted for API symmetry but unused here (Pod
        mounts are out of scope for the draft executor).
        """
        del volumes  # accepted but unused — Pod mounts are out of scope
        image = (env or {}).get("SPARKRUN_K8S_IMAGE", "")
        if not image:
            # Last-ditch: fall back to a marker that fails loudly so
            # the operator sees the missing wiring rather than a
            # mysterious "image '' not found" message from the API.
            image = "sparkrun-k8s-image-not-configured"
        env_for_pod = dict(env or {})
        env_for_pod.pop("SPARKRUN_K8S_IMAGE", None)
        return "#!/bin/bash\nset -uo pipefail\n%s\n" % self.run_cmd(
            image=image,
            command=serve_command,
            container_name=container_name,
            detach=detached,
            env=env_for_pod,
            sparkrun_labels=sparkrun_labels,
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
        """Per-rank Pod launcher for native cluster runtimes."""
        from sparkrun.utils import merge_env

        all_env = merge_env(nccl_env, env)
        cleanup = self.stop_cmd(container_name)
        run = self.run_cmd(
            image=image,
            command=serve_command,
            container_name=container_name,
            detach=True,
            env=all_env,
            extra_opts=extra_docker_opts,
            sparkrun_labels=sparkrun_labels,
        )
        return (
            "#!/bin/bash\n"
            "set -uo pipefail\n"
            'printf "Cleaning up existing Pod: %%s\\n" %(name)s\n'
            "%(cleanup)s\n"
            "\n"
            'printf "Launching %%s: %%s\\n" %(label)s %(name)s\n'
            "%(run)s\n"
        ) % {
            "name": quote(container_name),
            "label": quote(label),
            "cleanup": cleanup,
            "run": run,
        }

    def generate_ray_head_script(self, *args, **kwargs) -> str:
        """K8sExecutor does not support Ray clustering in the draft."""
        raise NotImplementedError(
            "K8sExecutor draft does not support Ray cluster strategy. Use a native runtime (e.g. vllm-distributed, sglang) or DockerExecutor."
        )

    def generate_ray_worker_script(self, *args, **kwargs) -> str:
        """K8sExecutor does not support Ray clustering in the draft."""
        raise NotImplementedError(
            "K8sExecutor draft does not support Ray cluster strategy. Use a native runtime (e.g. vllm-distributed, sglang) or DockerExecutor."
        )


def _node_selector_overrides(selector: str) -> str:
    """Translate ``key=value[,key=value]`` into a kubectl ``--overrides`` JSON.

    kubectl run lost ``--node-selector`` in newer versions; use the
    ``--overrides`` JSON path so we stay compatible.
    """
    import json

    pairs: dict[str, str] = {}
    for token in selector.split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        k, _, v = token.partition("=")
        pairs[k.strip()] = v.strip()
    overrides = {"apiVersion": "v1", "spec": {"nodeSelector": pairs}}
    return json.dumps(overrides, separators=(",", ":"))
