"""Experimental Kubernetes executor (draft).

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

import logging
import re

from sparkrun.orchestration.executors._base import Executor
from sparkrun.utils.shell import b64_wrap_bash, quote

logger = logging.getLogger(__name__)

_GPUS_DEVICE_RE = re.compile(r"device=([0-9,]+)")


class K8sExecutor(Executor):
    """``kubectl``-driven executor (experimental draft).

    Generates Pod-level lifecycle commands.  The user is expected to
    have ``kubectl`` on PATH and a current context that points at a
    cluster reachable from the script's execution host.
    """

    executor_name = "k8s"

    # No Docker-flavoured defaults; ``--privileged`` / ``--shm-size``
    # etc. don't apply.  No rootless/auto_user handling either.

    # ------------------------------------------------------------------
    # Common kubectl prefix
    # ------------------------------------------------------------------

    def _kubectl_prefix(self) -> str:
        """Build ``kubectl [--kubeconfig K] [--context C] [-n NS]`` prefix."""
        cfg = self.config
        parts: list[str] = ["kubectl"]
        if cfg.kubeconfig:
            parts.extend(["--kubeconfig", quote(cfg.kubeconfig)])
        if cfg.k8s_context:
            parts.extend(["--context", quote(cfg.k8s_context)])
        if cfg.k8s_namespace:
            parts.extend(["-n", quote(cfg.k8s_namespace)])
        return " ".join(parts)

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
            parts.append("--")
            parts.extend(["bash", "-c", b64_wrap_bash(command)])

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

    def stop_cmd(self, container_name: str, force: bool = True) -> str:
        """Delete the Pod; ``--ignore-not-found`` so it's idempotent."""
        prefix = self._kubectl_prefix()
        flags = "--ignore-not-found"
        if force:
            flags += " --grace-period=0 --force"
        return "%s delete pod %s %s 2>/dev/null || true" % (prefix, quote(container_name), flags)

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
        """
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
