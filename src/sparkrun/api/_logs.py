"""``sparkrun.api.logs`` — stream logs from a running sparkrun workload.

Returns a lazy :class:`Iterator` of :class:`LogLine` records.  Callers
consume the iterator (CLI renders to TTY; tests/automation can pipe to
processing).  The function never blocks beyond the executor's own
log streaming behavior.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator

from sparkrun.api._errors import JobNotFound
from sparkrun.api._models import LogLine

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition

logger = logging.getLogger(__name__)


def logs(
    cluster_id: str,
    *,
    hosts: list[str] | tuple[str, ...] | None = None,
    cluster: "str | ClusterDefinition | None" = None,
    follow: bool = False,
    tail: int | None = None,
    cache_dir: str | None = None,
) -> Iterator[LogLine]:
    """Yield :class:`LogLine` records from the head container of *cluster_id*.

    Args:
        cluster_id: The cluster ID returned by :func:`sparkrun.api.run`.
        hosts: Explicit host list; defaults to the hosts recorded in
            ``~/.cache/sparkrun/jobs/`` for *cluster_id*.
        cluster: Optional cluster definition; used to resolve the
            executor that should read logs.
        follow: When ``True``, stream new lines as they arrive
            (executor's native follow mode).
        tail: When set, start *tail* lines from the end of the log.
        cache_dir: Override for the sparkrun cache root.

    Raises:
        JobNotFound: When no hosts can be determined for *cluster_id*.
    """
    from sparkrun.api._resolve import resolve_cluster_def
    from sparkrun.orchestration.executor import resolve_executor
    from sparkrun.orchestration.job_metadata import load_job_metadata

    cluster_def = resolve_cluster_def(cluster)
    meta = load_job_metadata(cluster_id, cache_dir=cache_dir)

    if hosts:
        target_hosts = list(hosts)
    elif meta and meta.get("hosts"):
        target_hosts = list(meta["hosts"])
    else:
        raise JobNotFound("No hosts known for cluster_id %r" % cluster_id)

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
        cluster=cluster_def,
        cli_overrides=cli_overrides,
        rootless=False,
        auto_user=False,
    )

    head_host = target_hosts[0]
    head_name = ("%s_solo" % cluster_id) if len(target_hosts) <= 1 else ("%s_node_0" % cluster_id)

    return _stream_logs(executor, head_host, head_name, follow=follow, tail=tail)


def _stream_logs(executor, head_host: str, container: str, *, follow: bool, tail: int | None) -> Iterator[LogLine]:
    """Iterate :class:`LogLine` records by reading the executor's log command output.

    Spawned subprocess inherits the executor's ``logs_cmd`` semantics:
    ``docker logs -f``, ``kubectl logs -f``, or ``tail -F``.  Each
    stdout line is wrapped as a :class:`LogLine`.
    """
    import subprocess

    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.primitives import build_ssh_kwargs
    from sparkrun.orchestration.ssh import build_ssh_cmd

    config = SparkrunConfig()
    ssh_kwargs = build_ssh_kwargs(config)
    tail_cmd = executor.logs_cmd(container, follow=follow, tail=tail)
    ssh_cmd = build_ssh_cmd(head_host, **ssh_kwargs) + ["bash", "-c", tail_cmd]

    proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            yield LogLine(host=head_host, container=container, text=line.rstrip("\n"))
    finally:
        try:
            proc.terminate()
        except Exception:
            logger.debug("Log subprocess terminate failed", exc_info=True)


__all__ = ["logs"]
