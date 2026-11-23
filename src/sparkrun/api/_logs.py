"""``sparkrun.api.logs`` — stream logs from a running sparkrun workload.

The composition layer of the log path.  The runtime says *what* to read
(:meth:`~sparkrun.runtimes.base.RuntimePlugin.log_sources`), the executor
says *how* to read it on its substrate
(:meth:`~sparkrun.orchestration.executors._base.Executor.read_logs_cmd`),
:mod:`sparkrun.orchestration.logs` runs the commands and merges the output,
and this module resolves the workload and wires the three together.

Returns a lazy :class:`Iterator` of :class:`LogLine` records: the CLI
renders them, the desktop sidecar streams them, tests consume them.
Resolution and validation happen eagerly at call time (so a bad target
raises here rather than on the first ``next()``); only the reading is lazy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import math
import os
import threading
from typing import TYPE_CHECKING, Callable, Generator

from sparkrun.api._errors import JobNotFound, SparkrunError
from sparkrun.core.log_source import SCOPE_ALL, SCOPE_HEAD, LogLine, LogSource, source_sort_key

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.executors._base import Executor

logger = logging.getLogger(__name__)


def _nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a non-negative integer" % name)


def _publish_without_overwrite(partial_path: str, resolved_path: str, path: str) -> None:
    """Atomically publish *partial_path* at *resolved_path*, refusing an existing target.

    Never an existence check followed by a replace: another writer can create the
    target in between, and the replace would destroy it.
    """
    exists = SparkrunError(f"Target file {path!r} already exists (use overwrite=True to overwrite)")
    try:
        os.link(partial_path, resolved_path)
    except FileExistsError as e:
        raise exists from e
    except (AttributeError, OSError) as link_err:
        # No hard links here (FAT, some SMB mounts). Windows rename refuses an
        # existing target atomically; POSIX rename replaces it, so there is no
        # safe no-overwrite fallback there.
        if os.name != "nt":
            raise SparkrunError(
                f"Cannot publish log dump to {path!r} without risking an overwrite ({link_err}); retry with overwrite=True"
            ) from link_err
        try:
            os.rename(partial_path, resolved_path)
        except FileExistsError as e:
            raise exists from e
        except OSError as e:
            raise SparkrunError(f"Cannot finalize log dump to {path!r}: {e}") from e
        return
    # Published. A leftover partial would only block the next dump to this path.
    try:
        os.unlink(partial_path)
    except OSError as e:
        logger.warning("Log dump published to %s but %s could not be removed: %s", path, partial_path, e)


@dataclass(frozen=True)
class LogTarget:
    """Resolved workload log sources; obtain one with :func:`open_log_target`.

    Resolution and the availability precheck happen once. Each read opens fresh
    reader processes; the target itself owns no running processes and needs no
    close. Close partially consumed iterators (or set their cancellation event)
    to release readers. Reopen the target after workload placement changes.
    """

    workload_id: str
    runtime: str
    sources: tuple[LogSource, ...]
    _executor: Executor = field(repr=False, compare=False)
    _ssh_kwargs: dict = field(repr=False, compare=False)
    _cancel: threading.Event | None = field(default=None, repr=False, compare=False)
    _host: str | None = field(default=None, repr=False, compare=False)
    _container: str | None = field(default=None, repr=False, compare=False)
    _scope: str = field(default=SCOPE_HEAD, repr=False, compare=False)

    def read(
        self,
        *,
        follow: bool = False,
        tail: int | None = None,
        cancel: threading.Event | None = None,
        check: bool = True,
    ) -> Generator[LogLine, None, None]:
        """Read lazily, rank-grouped or interleaved in arrival order when following.

        ``tail=None`` reads all available lines; zero starts at the end. Reader
        failures raise by default; ``check=False`` allows best-effort following.
        Cancellation ends iteration normally and defaults to the event captured
        when opening this target. It never stops the workload.
        """
        from sparkrun.orchestration.logs import read_log_sources

        if tail is not None:
            _nonnegative_int("tail", tail)
        return read_log_sources(
            self._executor,
            self.sources,
            follow=follow,
            tail=tail,
            ssh_kwargs=self._ssh_kwargs,
            cancel=self._cancel if cancel is None else cancel,
            check=check,
        )

    def follow(
        self,
        *,
        tail: int | None = 200,
        cancel: threading.Event | None = None,
        check: bool = True,
    ) -> Generator[LogLine, None, None]:
        """Alias for ``read(follow=True, ...)``, initially tailing 200 lines."""
        return self.read(follow=True, tail=tail, cancel=cancel, check=check)

    def page(
        self,
        source: LogSource,
        *,
        skip: int,
        count: int = 500,
        cancel: threading.Event | None = None,
        timeout: float = 30.0,
    ) -> list[LogLine]:
        """Read up to ``count`` lines preceding the last ``skip`` lines of a source.

        Uses the existing tail primitive and retains at most ``count`` records;
        transfer grows with ``skip + count``. Offsets are relative to the end at
        read time, not a snapshot cursor: appends/rotation can shift pages.
        Reader failures and timeouts raise. Cancellation returns an empty list;
        check the event before interpreting that as exhausted history.
        """
        from sparkrun.orchestration.logs import read_log_sources

        if source not in self.sources:
            raise ValueError("Log source does not belong to this target")
        _nonnegative_int("skip", skip)
        _nonnegative_int("count", count)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Log page timeout must be positive and finite")
        cancel = self._cancel if cancel is None else cancel
        if count == 0 or (cancel is not None and cancel.is_set()):
            return []
        reader = read_log_sources(
            self._executor, (source,), tail=skip + count, ssh_kwargs=self._ssh_kwargs, cancel=cancel, check=True, timeout=timeout
        )
        page: list[LogLine] = []
        total = 0
        try:
            for line in reader:
                total += 1
                if len(page) < count:
                    page.append(line)
        finally:
            reader.close()
        if cancel is not None and cancel.is_set():
            return []
        return page[: max(0, total - skip)]

    def dump(
        self,
        path: str,
        *,
        overwrite: bool = False,
        cancel: threading.Event | None = None,
        prefix: bool | str | Callable[[LogLine], str] | None = None,
    ) -> int:
        """Stream complete logs for this target to *path*.

        Writes to ``<path>.partial`` and atomically renames to *path* upon completion.
        If *cancel* (or the target's own cancellation event) is set during dumping,
        the partial file is removed and the partial line count is returned (callers
        should check ``cancel.is_set()`` to distinguish cancellation from completion).
        Refuses an existing path unless *overwrite* is True.
        """

        if not path:
            raise SparkrunError("Dump path cannot be empty")

        resolved_path = os.path.abspath(os.path.expanduser(str(os.fspath(path))))
        if os.path.exists(resolved_path) and not overwrite:
            raise SparkrunError(f"Target file {path!r} already exists (use overwrite=True to overwrite)")

        parent = os.path.dirname(resolved_path)
        if not os.path.isdir(parent):
            raise SparkrunError(f"Directory {parent!r} does not exist")

        cancel = self._cancel if cancel is None else cancel
        partial_path = f"{resolved_path}.partial"

        multi_host = len({s.host for s in self.sources}) > 1
        if prefix is None:
            if len(self.sources) > 1:
                is_cluster_scope = self._host is None
                include_host = multi_host or is_cluster_scope

                def _prefix_fn(line: LogLine) -> str:
                    parts = []
                    if include_host:
                        parts.append(line.host)
                    parts.append(line.role)
                    if line.rank is not None:
                        parts.append(str(line.rank))
                    return f"[{'/'.join(parts)}]"

                effective_prefix = _prefix_fn
            else:
                effective_prefix = False
        else:
            effective_prefix = prefix

        sorted_sources = sorted(self.sources, key=source_sort_key)
        if self._container and self._host:
            scope_desc = f"container ({self._host}/{self._container})"
        elif self._container:
            scope_desc = f"container ({self._container})"
        elif self._host:
            scope_desc = f"node ({self._host})"
        else:
            scope_desc = f"cluster ({self._scope})"
        src_labels = ", ".join(s.label for s in sorted_sources)
        header = f"# workload: {self.workload_id} | scope: {scope_desc} | sources: {src_labels}\n"

        try:
            fd = os.open(partial_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as e:
            raise SparkrunError(f"A log dump to {path!r} is already in progress ({partial_path!r} exists)") from e
        except OSError as e:
            raise SparkrunError(f"Cannot write to {path!r}: {e}") from e

        partial_created = True
        completed = False
        try:
            try:
                fh = open(fd, "w", encoding="utf-8")
            except BaseException:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise

            try:
                fh.write(header)
                count = 0
                reader = self.read(follow=False, tail=None, cancel=cancel, check=True)
                try:
                    for line in reader:
                        if callable(effective_prefix):
                            p = effective_prefix(line)
                            if p:
                                fh.write(f"{p} " if not p.endswith(" ") else p)
                            fh.write(f"{line.text}\n")
                        elif effective_prefix is True:
                            tag = f"[{line.host}/{line.role}]" if multi_host else f"[{line.role}]"
                            fh.write(f"{tag} {line.text}\n")
                        elif isinstance(effective_prefix, str) and effective_prefix:
                            fh.write(f"{effective_prefix} " if not effective_prefix.endswith(" ") else effective_prefix)
                            fh.write(f"{line.text}\n")
                        else:
                            fh.write(f"{line.text}\n")
                        count += 1
                finally:
                    reader.close()
            finally:
                fh.close()

            if cancel is not None and cancel.is_set():
                try:
                    os.unlink(partial_path)
                except OSError:
                    pass
                completed = True
                return count

            if overwrite:
                try:
                    os.replace(partial_path, resolved_path)
                except OSError as e:
                    raise SparkrunError(f"Cannot finalize log dump to {path!r}: {e}") from e
            else:
                _publish_without_overwrite(partial_path, resolved_path, path)

            completed = True
            return count
        finally:
            if not completed and partial_created:
                try:
                    os.unlink(partial_path)
                except OSError:
                    pass


def logs(
    workload_id: str | None = None,
    *,
    recipe: str | Recipe | None = None,
    hosts: list[str] | tuple[str, ...] | None = None,
    overrides: dict | None = None,
    cluster: str | ClusterDefinition | None = None,
    scope: str = SCOPE_HEAD,
    host: str | None = None,
    container: str | None = None,
    source_id: str | None = None,
    follow: bool = False,
    tail: int | None = None,
    cache_dir: str | None = None,
    sctx: SparkrunContext | None = None,
    cancel: threading.Event | None = None,
    check: bool = True,
) -> Generator[LogLine, None, None]:
    """Resolve a workload eagerly and return a lazy iterator of :class:`LogLine`.

    Target arguments and filters follow :func:`open_log_target`. Reading follows
    :meth:`LogTarget.read`: finite reads are rank-grouped, following interleaves
    arrival order, ``tail=None`` reads all lines, and failures raise unless
    ``check=False``. Explicit cancellation overrides the operation's event.
    """
    if tail is not None:
        _nonnegative_int("tail", tail)
    target = open_log_target(
        workload_id,
        recipe=recipe,
        hosts=hosts,
        overrides=overrides,
        cluster=cluster,
        scope=scope,
        host=host,
        container=container,
        source_id=source_id,
        cache_dir=cache_dir,
        sctx=sctx,
        cancel=cancel,
    )
    if follow:
        return target.follow(tail=tail, cancel=cancel, check=check)
    return target.read(follow=False, tail=tail, cancel=cancel, check=check)


def open_log_target(
    workload_id: str | None = None,
    *,
    recipe: "str | Recipe | None" = None,
    hosts: list[str] | tuple[str, ...] | None = None,
    overrides: dict | None = None,
    cluster: "str | ClusterDefinition | None" = None,
    scope: str = SCOPE_HEAD,
    host: str | None = None,
    container: str | None = None,
    source_id: str | None = None,
    cache_dir: str | None = None,
    sctx: "SparkrunContext | None" = None,
    cancel: threading.Event | None = None,
) -> LogTarget:
    """Resolve log sources and check availability without opening log readers.

    Either ``workload_id`` *or* (``recipe`` + a host source) is required.
    When both are given, ``workload_id`` wins — the same contract as
    :func:`sparkrun.api.stop`.  The recipe form resolves through live intent
    discovery (:func:`~sparkrun.api._resolve.discover_cluster_id_by_intent`)
    rather than deriving a cluster_id, so it finds the workload regardless of
    which placement token the scheduler assigned it.

    Args:
        workload_id: Workload identity returned in ``RunResult.workload_id``.
        recipe: Recipe name or object, when addressing the workload by
            recipe instead of by id.
        hosts: Explicit host list.  Required for the recipe form; for the
            workload_id form it defaults to the hosts recorded in
            ``~/.cache/sparkrun/jobs/``.
        overrides: Recipe overrides used at launch.  They participate in the
            intent, so port / parallelism overrides must match for the
            recipe form to resolve.
        cluster: Optional cluster name or definition.
        scope: :data:`SCOPE_HEAD` (default) reads only the primary log;
            :data:`SCOPE_ALL` reads every worker/rank too.
        host: Select sources on this host after resolving the full topology.
            Exact matches win; otherwise ignore hostname case and a ``user@``
            prefix. An ambiguous normalized host is rejected. This selector
            never changes the resolved connection user or placement.
        container: Exact container/source name filter. If any selector is
            supplied, filter all sources regardless of scope; all selectors
            intersect. Use these instead of reducing ``hosts`` to one worker.
        source_id: Exact opaque source identity, including the native resource
            and container instance when available. Intersects other filters.
        cache_dir: Override for the sparkrun cache root.  Defaults to
            the supplied or current application configuration, without plugin bootstrap.
        sctx: Optional shared :class:`SparkrunContext`.
        cancel: Default cancellation event for subsequent reads; otherwise
            capture the scoped operation's event. Individual reads may override.

    Raises:
        JobNotFound: No hosts can be determined, or no running workload
            matches the recipe or source filters.
        AmbiguousWorkload: The recipe matches several running workloads.
        SparkrunError: Neither ``workload_id`` nor ``recipe`` was given, or
            *scope* is not a valid value.
    """
    from sparkrun.api._workload import open_workload

    if scope not in (SCOPE_HEAD, SCOPE_ALL):
        raise SparkrunError("Invalid log scope %r: expected %r or %r" % (scope, SCOPE_HEAD, SCOPE_ALL))
    for name, value in (("host", host), ("container", container), ("source_id", source_id)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("%s must be a non-empty string" % name)

    access = open_workload(
        workload_id,
        recipe=recipe,
        hosts=hosts,
        overrides=overrides,
        cluster=cluster,
        cache_dir=cache_dir,
        operation="Log resolution",
        sctx=sctx,
    )
    # Persistent metadata and runtime naming still use the internal cluster_id.
    cluster_id, meta, target_hosts, cluster_def = access.cluster_id, access.meta, access.hosts, access.cluster
    runtime, executor, sctx, ssh_kwargs, cache_dir = access.runtime, access.executor, access.sctx, access.ssh_kwargs, access.cache_dir

    cancel = cancel if cancel is not None else getattr(getattr(sctx, "operation", None), "cancel", None)
    filtered = any(value is not None for value in (host, container, source_id))
    effective_scope = SCOPE_ALL if filtered else scope
    if meta and meta.get("native_resource") is not None:
        try:
            sources = executor.native_log_sources(cluster_id, metadata=meta, scope=effective_scope, cancel=cancel)
        except (SparkrunError, ValueError):
            raise
        except Exception as exc:
            raise SparkrunError("Native log resolution failed: %s" % exc) from exc
        if cancel is not None and cancel.is_set():
            sources = []
        else:
            sources = _filter_sources(sources, cluster_id, host=host, container=container, source_id=source_id)

        return LogTarget(
            workload_id=cluster_id,
            runtime=runtime.runtime_name,
            sources=tuple(sources),
            _executor=executor,
            _ssh_kwargs={},
            _cancel=cancel,
            _host=host,
            _container=container,
            _scope=effective_scope,
        )

    sources = runtime.log_sources(
        cluster_id,
        target_hosts,
        is_solo=len(target_hosts) <= 1,
        scope=effective_scope,
    )
    sources = _filter_sources(sources, cluster_id, host=host, container=container, source_id=source_id)

    # Liveness precheck: check ALL nodes (not just the head log source) —
    # in a multi-node job the head may have crashed while workers are still
    # running, and the user should still be able to read logs from surviving
    # nodes.  Uses ``executor.query_status`` (one parallel SSH sweep) — the
    # same source of truth as ``api.status`` and ``check_job_running``.
    # See ``_verify_log_source_alive`` for the full decision tree.
    #
    # Substrate knowledge stays behind the executor throughout: this module
    # asks *what is running* and *what became of it*, never *how to look*.
    all_sources = (
        sources
        if effective_scope == SCOPE_ALL
        else runtime.log_sources(
            cluster_id,
            target_hosts,
            is_solo=len(target_hosts) <= 1,
            scope=SCOPE_ALL,
        )
    )
    # Explicit selectors check only their sources. Reads never remove metadata
    # or cache claims, even when all selected containers are confirmed absent.
    _verify_log_source_alive(
        executor,
        all_sources,
        ssh_kwargs,
        cluster_id,
        cache_dir,
        scope=effective_scope,
        host_hardware=cluster_def.hosts_hardware or None,
    )

    return LogTarget(
        workload_id=cluster_id,
        runtime=runtime.runtime_name,
        sources=tuple(sources),
        _executor=executor,
        _ssh_kwargs=dict(ssh_kwargs),
        _cancel=cancel,
        _host=host,
        _container=container,
        _scope=effective_scope,
    )


def _filter_sources(sources, cluster_id, *, host=None, container=None, source_id=None):
    """Apply the same selectors to host-native and controller-native sources."""
    if host is not None:
        exact = [source for source in sources if source.host == host]
        if exact:
            sources = exact
        else:
            normalized = host.strip().rsplit("@", 1)[-1].lower()
            sources = [source for source in sources if source.host.strip().rsplit("@", 1)[-1].lower() == normalized]
            if len({source.host for source in sources}) > 1:
                raise SparkrunError("Ambiguous log host %r; use the exact host including its user prefix" % host)
    if container is not None:
        sources = [source for source in sources if source.container == container]
    if source_id is not None:
        sources = [source for source in sources if source.source_id == source_id]
    if not sources and any(value is not None for value in (host, container, source_id)):
        raise JobNotFound(
            "No log sources match host=%r container=%r source_id=%r for workload %s" % (host, container, source_id, cluster_id)
        )
    return sources


def _verify_log_source_alive(
    executor,
    sources,
    ssh_kwargs: dict,
    cluster_id: str,
    cache_dir: str | None,
    *,
    scope: str = SCOPE_HEAD,
    host_hardware: dict | None = None,
) -> None:
    """Raise :class:`JobNotFound` if the workload isn't running.

    Uses :meth:`Executor.query_status` — the same source of truth as
    ``api.status``, ``check_job_running``, and the monitor TUI — to do
    one parallel SSH sweep across all hosts, rather than probing each
    source sequentially.

    Three outcomes:

    1. **Head alive** (container in the status snapshot) → proceed.
    2. **Head dead, some workers alive** → raise with a pointer to
       ``--all-sources`` so the user can read from surviving nodes.
       With ``scope=SCOPE_ALL`` the reader can handle this, so proceed.
    3. **All sources dead** (none in the snapshot) →
       :meth:`Executor.describe_terminated` distinguishes
       **stopped-but-inspectable** (preserve metadata, render the executor's
       investigation hints) from **fully gone** (report absence without cleanup).
       ``query_status`` reports only what is *running*, so it structurally
       cannot make that distinction itself.

    Every substrate-specific question — is anything left behind, what state is
    it in, what should the operator run next — is answered by the executor.
    This function contributes only sparkrun-level guidance (``--all-sources``,
    ``sparkrun stop``), so it reads the same on docker, local and k8s.

    Best-effort: if the status query fails or a host is unreachable
    (in ``ClusterStatus.errors``), the precheck is skipped so a network
    blip never causes a false "not running" verdict.
    """
    if not sources:
        return

    head = sources[0]

    # One parallel SSH sweep via the status API — same source of truth
    # as `api.status`, `check_job_running`, and the monitor TUI.
    all_hosts = list(dict.fromkeys(s.host for s in sources))
    try:
        # The cluster's saved inventory, as ``api.status`` passes it: without
        # it the executor falls back to ``resolve_hardware()``, which raises
        # under a ``require-metadata`` profile and skips this check silently.
        snapshot = executor.query_status(all_hosts, ssh_kwargs=ssh_kwargs, host_hardware=host_hardware)
    except Exception:  # noqa: BLE001 — best-effort; let log reader surface its own error
        return

    def _container_status(host: str, container_name: str) -> bool | None:
        """True if running, False if confirmed absent, None if host unreachable."""
        occ = snapshot.for_host(host)
        if occ is None:
            return None  # host in errors / unreachable → inconclusive
        for w in occ.workloads:
            if w.cluster_id == cluster_id:
                for c in w.containers:
                    if c.name == container_name:
                        return True
                # The cluster_id is here but this container isn't named.
                # ``RunningWorkload.containers`` is optional — an executor that
                # doesn't populate it can't answer per-container questions at
                # all, so this is *inconclusive*, not "alive".  The difference
                # only shows on workers: counting one alive by mistake reports
                # "partially running, try --all-sources" for a workload that is
                # entirely dead.
                if not w.containers:
                    return None
        return False  # host reachable, container not in snapshot

    head_status = _container_status(head.host, head.container)
    if head_status is None:
        return  # inconclusive → skip precheck
    if head_status:
        return  # head is alive → proceed normally

    # Head is confirmed dead.  Check workers — if any are still alive,
    # point the user at ``--all-sources`` rather than letting the reader
    # fail on the dead head container with a raw docker error.
    alive_worker_hosts = []
    for source in sources[1:]:
        status = _container_status(source.host, source.container)
        if status is None:
            return  # inconclusive → skip precheck
        if status:
            alive_worker_hosts.append(source.host)

    if alive_worker_hosts:
        # With ``--all-sources`` the reader can read from the surviving
        # workers, so proceed.  With the default (head-only) scope the
        # reader would fail on the dead head container — raise a helpful
        # error pointing at ``--all-sources`` instead.
        if scope == SCOPE_ALL:
            return
        raise JobNotFound(
            "Workload %s is partially running — the head container on %s has stopped, "
            "but worker containers are still alive on %s.\n"
            "Try `sparkrun logs %s --all-sources` to read from surviving nodes, "
            "or `sparkrun stop %s` to clean up." % (cluster_id, head.host, ", ".join(alive_worker_hosts), cluster_id, cluster_id)
        )

    # Every source is confirmed dead.  Ask the executor what became of the head
    # — whether anything is left to inspect, and how to inspect it.  A missing
    # entry means "cannot tell" (unreachable host, an executor with no
    # post-mortem support), which must not be read as "gone".
    try:
        terminated = executor.describe_terminated([head], ssh_kwargs=ssh_kwargs)
    except Exception:  # noqa: BLE001 — best-effort, like the status sweep above
        logger.debug("describe_terminated failed for %s", cluster_id, exc_info=True)
        terminated = {}
    info = terminated.get((head.host, head.container))

    if info is None or info.exists is not False:
        raise JobNotFound(
            "Workload %s is not currently running on %s%s.\n"
            "The job metadata has been preserved so you can investigate.%s\n"
            "Run `sparkrun stop %s` to clean up when ready." % (cluster_id, head.host, _detail_suffix(info), _hint_block(info), cluster_id)
        )

    # Absence is only an observation: a launch persists metadata and claims
    # before creating containers, and a relaunch may replace this ID while we
    # probe. Reads must never finalize workloads or release their cache claims.
    raise JobNotFound(
        "Workload %s is not running on %s and nothing remains to read%s.\n"
        "The job metadata has been preserved. Run `sparkrun stop %s` to clean up when ready.\n"
        "Run `sparkrun status` to see running workloads.%s"
        % (
            cluster_id,
            head.host,
            _detail_suffix(info),
            cluster_id,
            _hint_block(info),
        )
    )


def _detail_suffix(info) -> str:
    """Render an executor's substrate-native state as a parenthetical, if any."""
    detail = getattr(info, "detail", None)
    return " (%s)" % detail if detail else ""


def _hint_block(info) -> str:
    """Render an executor's investigation hints as an indented block.

    The commands are the executor's — ``docker logs`` on a Docker host,
    ``kubectl logs`` on k8s, a plain ``cat`` for a ``local`` job — so this only
    lays them out.  An executor with nothing useful to suggest contributes
    nothing rather than a heading with an empty list under it.
    """
    hints = getattr(info, "investigate_hints", ()) or ()
    if not hints:
        return ""
    return "\n" + "\n".join("  %s" % h for h in hints)


__all__ = ["LogTarget", "logs", "open_log_target"]
