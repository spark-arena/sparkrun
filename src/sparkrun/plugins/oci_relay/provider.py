# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see LICENSE_EXCEPTION.

"""Source-adjacent relay orchestration for Sparkrun's image-copy API."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import replace
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from urllib.parse import urlsplit

from . import __version__
from .host import Lines, OperationError, Runner, pump
from .parallel import parallel
from .paths import arguments as path_arguments, qualify as qualify_paths
from .progress import Progress
from .release import BinaryUnavailable, acquire
from .source_policy import SourceUnavailable, detect, validate
from .tuning import limits, probe

logger = logging.getLogger(__name__)


def _probe(runner, host, binary, endpoint, credential, *, paths=None):
    output = runner.execute(
        host, [binary, "peer", "--check", *(path_arguments(paths, urlsplit(endpoint).port) if paths else ["--endpoint", endpoint]),
               "--session", credential, "--timeout-seconds", "10"],
        timeout=15,
    )
    result = json.loads(output)
    if result.get("state") != "READY" or result.get("version") != 1:
        raise OperationError("relay preflight returned an invalid result")


def _free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _bridge(runner, process, source_host, source_binary, source_dir):
    attachment = runner.start(source_host, [source_binary, "attach", "--socket", source_dir + "/source.sock"])
    Lines(attachment.stderr)
    threading.Thread(target=pump, args=(process.stdout, attachment.stdin), daemon=True).start()
    threading.Thread(target=pump, args=(attachment.stdout, process.stdin), daemon=True).start()
    return attachment


def _stdio_probe(runner, host, binary, endpoint, credential, source_host, source_binary, source_dir):
    process = runner.start(host, [binary, "peer", "--check", "--stdio", "--endpoint", endpoint,
                                  "--session", credential, "--timeout-seconds", "10"])
    events = Lines(process.stderr, events=True)
    attachment = _bridge(runner, process, source_host, source_binary, source_dir)
    try:
        result = events.event(time.monotonic() + 15)
        if process.wait(timeout=5) != 0 or result.get("state") != "READY" or result.get("version") != 1:
            raise OperationError("stdio preflight returned an invalid result")
    finally:
        process.close()
        attachment.close()


def _forward(runner, source_host, source_port, receiver, routes):
    from sparkrun.orchestration.ssh import should_run_locally

    if not callable(getattr(runner.session, "open_forward", None)):
        raise OperationError("host transport does not support managed forwarding")
    local_port = routes.get("local_port")
    if local_port is None:
        if source_host is None or should_run_locally(source_host, runner.session.ssh_user):
            local_port = source_port
        else:
            failure = None
            for _ in range(3):
                local_port = _free_port()
                handle = runner.session.open_forward(
                    source_host, listen_port=local_port, target_host="127.0.0.1", target_port=source_port,
                )
                runner.processes.append(handle)
                Lines(handle.stdout)
                diagnostics = Lines(handle.stderr)
                deadline = time.monotonic() + 10
                while handle.poll() is None and time.monotonic() < deadline:
                    try:
                        with socket.create_connection(("127.0.0.1", local_port), timeout=0.2):
                            break
                    except OSError:
                        time.sleep(0.05)
                else:
                    failure = OperationError("source SSH forward failed: " + "\n".join(diagnostics.tail)[-2000:])
                    handle.close()
                    continue
                failure = None
                break
            if failure:
                raise failure
        routes["local_port"] = local_port
    if should_run_locally(receiver, runner.session.ssh_user):
        return f"https://127.0.0.1:{local_port}"
    handle = runner.session.open_forward(
        receiver, listen_port=0, target_host="127.0.0.1", target_port=local_port, reverse=True,
    )
    runner.processes.append(handle)
    Lines(handle.stdout)
    diagnostics = Lines(handle.stderr)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and handle.poll() is None:
        match = re.search(r"Allocated port (\d+)", "\n".join(diagnostics.tail))
        if match:
            return f"https://127.0.0.1:{int(match[1])}"
        time.sleep(0.05)
    raise OperationError("receiver SSH forward failed: " + "\n".join(diagnostics.tail)[-2000:])


def _source_address(request, settings, facts):
    if settings.get("advertise"):
        return settings["advertise"]
    if request.source_host is not None:
        return facts.get("source_address", request.source_host)
    # Resolve the controller's route to the selected data-network target.
    address = socket.getaddrinfo(request.transfer_hosts[0], 9, type=socket.SOCK_DGRAM)[0]
    with socket.socket(address[0], socket.SOCK_DGRAM) as route:
        route.connect(address[4])
        return route.getsockname()[0]


class RelayProvider:
    def __init__(self):
        # Serialize image groups for now: a new group cannot multiply the host budgets.
        self._operations = threading.Lock()

    @staticmethod
    def _settings(request):
        if request.config is None:
            from sparkrun.core.config import SparkrunConfig

            return SparkrunConfig().plugin_settings("oci-relay")
        return request.config.plugin_settings("oci-relay")

    def pull(self, request):
        """Decline before transfer when core's existing local-image policy applies."""
        from sparkrun.plugins import ImageDistributionUnsupported
        from sparkrun.utils.images import parse_image_ref
        from .source_policy import LOCAL_DOCKER

        settings = self._settings(request)
        if request.offline or (settings.get("registry_source", True) is False and settings.get("source_mode") != "registry"):
            return None
        if settings.get("source_mode", "auto") not in {"auto", "registry"} or settings.get("manifest"):
            return None
        ref = parse_image_ref(request.image)
        # Docker cannot attach an upstream RepoDigest through its tag API. Keep
        # digest-named runtime references on core's existing pull path for now.
        if ref.digest:
            if settings.get("source_mode") == "registry":
                raise ImageDistributionUnsupported("registry plugin currently requires a tag; standalone registry sources accept digests")
            return None
        started = time.monotonic()
        timeout = request.timeout or settings.get("timeout_seconds", 3600)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 86400:
            raise ValueError("relay timeout must be between 1 and 86400 seconds")
        if not request.dry_run and not request.force_pull and settings.get("source_mode") != "registry":
            runner = Runner(request.session, settings)
            try:
                inspected = runner.connection(request.source_host).execute(
                    request.source_host or "localhost", LOCAL_DOCKER + ["image", "inspect", "--format={{.Id}}", request.image], timeout=min(30, timeout),
                )
                if inspected.returncode == 0:
                    # Includes best-effort :latest refresh: core retains its
                    # established fallback-to-local semantics for existing images.
                    return None
            finally:
                runner.close()
        if not ref.tag:
            request = replace(request, image=request.image + ":latest")
        remaining = timeout - (time.monotonic() - started)
        if remaining < 1:
            raise OperationError("registry source selection exceeded operation deadline")
        return self.copy(replace(request, timeout=remaining), registry=True)

    def copy(self, request, *, registry=False):
        from sparkrun.plugins import ImageCopyResult, ImageDistributionUnsupported

        settings = self._settings(request)
        if registry:
            settings = dict(settings, source_mode="registry")
        if settings.get("source_mode") == "registry" and request.offline:
            raise ImageDistributionUnsupported("registry source is unavailable offline")
        # Resolve defaults per copy without changing the user's settings.
        try:
            settings = validate(settings)
        except SourceUnavailable as error:
            raise ImageDistributionUnsupported(str(error)) from error
        if request.dry_run:
            logger.info(
                "[dry-run] OCI Relay image=%s source=%s targets=%s transport=%s source_mode=%s (auto resolved at execution)",
                request.image, request.source_host or "controller", list(request.targets),
                settings.get("transport", "auto"),
                settings.get("source_mode", "auto"),
            )
            return ImageCopyResult(dict.fromkeys(request.targets, "complete"))
        timeout = request.timeout or settings.get("timeout_seconds", 3600)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 86400:
            raise ValueError("relay timeout must be between 1 and 86400 seconds")
        started = time.monotonic()
        if not self._operations.acquire(timeout=timeout):
            raise OperationError("image-copy queue exceeded operation deadline")
        try:
            remaining = timeout - (time.monotonic() - started)
            if remaining < 1:
                raise OperationError("image-copy deadline expired before admission")
            return self._copy(request, settings, int(remaining))
        except (BinaryUnavailable, SourceUnavailable) as error:
            raise ImageDistributionUnsupported(str(error)) from error
        finally:
            self._operations.release()

    def _copy(self, request, settings, timeout):
        from sparkrun.plugins import ImageCopyResult

        transport = settings.get("transport", "auto")
        if transport not in {"auto", "http2-direct", "http2-ssh", "ssh-stdio"}:
            raise ValueError("unknown relay transport")
        spool = settings.get("max_spool_bytes", 8 << 30)
        if type(spool) is not int or not 4 << 20 <= spool <= 1 << 50:
            raise ValueError("invalid source spool budget")
        runner = Runner(request.session, settings)
        operation_start = phase_start = time.monotonic()
        deadline = operation_start + timeout
        timings = {}
        completed = False
        progress = Progress(request.image)

        def mark(name):
            nonlocal phase_start
            now = time.monotonic()
            timings[name + "_seconds"] = now - phase_start
            phase_start = now

        heartbeat_stop = threading.Event()
        source_process = None
        try:
            execution_hosts = list(dict.fromkeys([request.source_host, *request.targets]))
            architectures = parallel(execution_hosts, runner.architecture)
            # Acquire/verify each architecture once. Concurrent extraction into
            # the same release cache would needlessly duplicate work.
            releases = {arch: acquire(arch, settings, offline=request.offline)
                        for arch in dict.fromkeys(architectures.values())}
            binaries = parallel(execution_hosts, lambda host: runner.stage_binary(
                host, *releases[architectures[host]], __version__,
            ))
            mark("binaries")
            progress.phase("selecting source and network routes")
            source_host = request.source_host
            if settings.get("source_mode") == "registry":
                target_arches = {architectures[host] for host in request.targets}
                if len(target_arches) != 1:
                    raise SourceUnavailable("registry distribution requires one target platform per operation")
                target_platform = "linux/" + next(iter(target_arches))
                if settings.get("platform") and settings["platform"].split("/")[:2] != target_platform.split("/"):
                    raise SourceUnavailable("configured registry platform does not match the receivers")
                settings = dict(settings, platform=settings.get("platform") or target_platform)
            source_binary = binaries[source_host]
            probe_target = next((address for host, address in zip(request.targets, request.transfer_hosts, strict=True)
                                 if host != source_host), "127.0.0.1")
            facts = {source_host: probe(runner, source_host, probe_target)}
            advertise = _source_address(request, settings, facts[source_host]) if transport in {"auto", "http2-direct"} else "127.0.0.1"
            facts.update(parallel(
                [host for host in request.targets if host not in facts],
                lambda host: probe(runner, host, advertise if transport in {"auto", "http2-direct"} else source_host or "127.0.0.1"),
            ))
            def colocated(host):
                source_id = facts[source_host].get("host_id")
                return host == source_host or bool(source_id and source_id == facts[host].get("host_id"))

            data_paths = {}
            if "data_paths" in settings:
                data_paths = parallel([host for host in request.targets if not colocated(host)], lambda host: qualify_paths(
                    runner, source_host, host, settings["data_paths"], facts,
                ))
                if transport == "http2-direct" and not all(data_paths.values()):
                    raise OperationError("no qualified direct data path for one or more receivers")
                for host, paths in data_paths.items():
                    if paths:
                        facts[host] = dict(facts[host], network_gbps=sum(p["network_gbps"] for p in paths))
                # One source budget for the operation; never multiply by receivers.
                facts[source_host] = dict(facts[source_host], network_gbps=max(
                    (sum(p["network_gbps"] for p in paths) for paths in data_paths.values()), default=0,
                ))
            roles = Counter(facts[host].get("host_id") or str(host) for host in [source_host, *request.targets])
            source_limits = limits(settings, facts[source_host], transport,
                                   local_roles=roles[facts[source_host].get("host_id") or str(source_host)])
            logger.info("OCI Relay source limits: %s; route facts: %s", source_limits, facts[source_host])
            selection = detect(runner, source_host, request.image, settings, facts[source_host])
            logger.info("OCI Relay source selection: requested=%s selected=%s reason=%s",
                        settings.get("source_mode", "auto"), selection.mode, selection.reason)
            settings = dict(settings, source_mode=selection.mode)
            mark("discovery")
            source_dir = runner.directory(source_host)
            if settings.get("source_mode") == "docker-save":
                logger.info("OCI Relay preparing one uncompressed Docker export; disk cap=%d bytes; export manifest may differ from registry", spool)
            ids = {f"receiver-{index}": host for index, host in enumerate(request.targets)}
            progress.bind(ids)
            runner.execute(
                source_host,
                [source_binary, "session", "--out", source_dir, "--peers", ",".join(ids),
                 "--duration", f"{min(86400, timeout + 60)}s"],
            )
            # This is protected management data, never a progress/log payload.
            session_data = runner.execute(source_host, ["cat", source_dir + "/session.json"])
            if len(session_data) > 4 << 20:
                raise OperationError("session metadata exceeds limit")
            session = json.loads(session_data)
            if set(session.get("peers", {})) != {*ids, "manager"}:
                raise OperationError("session returned unexpected peers")
            def credential_file(identity):
                host = ids[identity]
                directory = runner.directory(host)
                credential = directory + "/peer.json"
                runner.json(host, credential, session["peers"][identity])
                return credential

            target_files = parallel(ids, credential_file)
            manifest = ""
            if settings.get("manifest"):
                path = Path(settings["manifest"]).expanduser()
                with path.open("rb") as stream:
                    raw = stream.read((4 << 20) + 1)
                if len(raw) > 4 << 20:
                    raise OperationError("manifest exceeds size limit")
                manifest = source_dir + "/manifest.json"
                runner.write(source_host, manifest, raw)
            plan = {
                "version": 1, "source": selection.mode, "image": request.image,
                "platform": settings.get("platform", ""), "manifest": manifest,
                "session_dir": source_dir, "socket": source_dir + "/source.sock",
                "listen": settings.get("listen", "0.0.0.0:0" if transport in {"auto", "http2-direct"} else "127.0.0.1:0"),
                "advertise": advertise, "allow_preparation_read": settings.get("allow_preparation_read") is True,
                "max_buffer_bytes": source_limits["max_buffer_bytes"], "max_spool_bytes": spool,
                "max_upload_bytes": settings.get("max_upload_bytes", spool),
                "spool_dir": settings.get("spool_dir", source_dir),
                "source_streams": source_limits["source_streams"],
                "source_join_milliseconds": settings.get("source_join_milliseconds", 0),
                "timeout_seconds": max(1, int(deadline - time.monotonic())),
                "lease_seconds": 45, "managed_stdin": True,
            }
            if selection.mode == "registry":
                plan.update(registry_config=settings.get("registry_config", ""),
                            registry_plain_http=settings.get("registry_plain_http", False),
                            registry_cache_bytes=settings.get("registry_cache_bytes", 0))
            mark("session")
            progress.phase(f"preparing {selection.mode} source")
            source_process = runner.start_source(source_host, source_binary, source_dir, plan)
            source_events = Lines(source_process.stdout, events=True)
            source_errors = Lines(source_process.stderr)

            def heartbeat():
                while not heartbeat_stop.is_set():
                    try:
                        source_process.stdin.write(b'{"type":"heartbeat"}\n')
                    except (OSError, ValueError):
                        return
                    heartbeat_stop.wait(5)

            threading.Thread(target=heartbeat, daemon=True).start()
            try:
                ready = source_events.event(deadline)
            except OperationError as error:
                raise OperationError(str(error) + ": " + "\n".join(source_errors.tail)[-4000:]) from error
            while ready.get("type") == "progress":
                ready = source_events.event(deadline)
            if ready.get("type") != "ready" or ready.get("version") != 1:
                raise OperationError("source failed readiness: " + "\n".join(source_errors.tail)[-4000:])
            endpoint = ready["endpoint"]
            parts = urlsplit(endpoint)
            if parts.scheme != "https" or not parts.port:
                raise OperationError("invalid source endpoint")
            mark("preparation")
            progress.phase("checking receiver connections")
            routes = {}
            target_routes = {}
            ready_paths = {}

            def direct_probe(identity):
                try:
                    if ids[identity] in data_paths:
                        available = []
                        for path in data_paths[ids[identity]]:
                            try:
                                _probe(runner, ids[identity], binaries[ids[identity]], endpoint,
                                       target_files[identity], paths=[path])
                                available.append(path)
                            except Exception as error:
                                logger.warning("OCI Relay data path preflight failed for %s: %s", ids[identity], error)
                        if not available:
                            raise OperationError("no configured data path passed authenticated preflight")
                        ready_paths[identity] = available
                    else:
                        _probe(runner, ids[identity], binaries[ids[identity]], endpoint, target_files[identity])
                except Exception as error:
                    if transport == "http2-direct":
                        raise
                    return str(error)
                return None

            direct_errors = parallel(ids, direct_probe) if transport in {"auto", "http2-direct"} else {}
            # Negotiate every route before starting any destination pull.
            # Forward allocation shares the source tunnel, so keep that part
            # serialized; successful direct probes need no further host calls.
            for identity, host in ids.items():
                chosen = None
                errors = []
                if transport in {"auto", "http2-direct"}:
                    if direct_errors[identity] is None:
                        chosen = ("http2-direct", endpoint)
                    else:
                        errors.append(direct_errors[identity])
                if chosen is None and transport in {"auto", "http2-ssh"}:
                    try:
                        forwarded = _forward(runner, source_host, parts.port, host, routes)
                        _probe(runner, host, binaries[host], forwarded, target_files[identity])
                        chosen = ("http2-ssh", forwarded)
                    except Exception as error:
                        errors.append(str(error))
                        if transport == "http2-ssh":
                            raise
                if chosen is None:
                    if transport not in {"auto", "ssh-stdio"}:
                        raise OperationError("no permitted data route")
                    _stdio_probe(runner, host, binaries[host], endpoint, target_files[identity],
                                 source_host, source_binary, source_dir)
                    chosen = ("ssh-stdio", endpoint)
                    if errors:
                        logger.info("OCI Relay uses stdio for %s after route preflight: %s", host, "; ".join(errors))
                target_routes[identity] = chosen
                logger.info("OCI Relay route %s -> %s: %s", source_host or "controller", host, chosen[0])
            mark("preflight")
            progress.phase("checking receiver caches and starting transfers")

            peers = {}
            peer_diagnostics = {}
            for identity, host in ids.items():
                route, peer_endpoint = target_routes[identity]
                peer_limits = limits(settings, facts[host], route, local_roles=roles[facts[host].get("host_id") or str(host)])
                logger.info("OCI Relay receiver %s limits: %s; route facts: %s", host, peer_limits, facts[host])
                arguments = [
                    binaries[host], "peer", *(path_arguments(ready_paths[identity], parts.port)
                        if identity in ready_paths else ["--endpoint", peer_endpoint]), "--session", target_files[identity],
                    "--tag", request.image, "--replace-tag", "--max-buffer-bytes", str(peer_limits["max_buffer_bytes"]),
                    "--max-source-streams", str(peer_limits["source_streams"]),
                    "--import", settings.get("receiver_import", "pull"),
                    "--max-import-bytes", str(settings.get("max_import_bytes", 0)),
                    "--timeout-seconds", str(max(1, int(deadline - time.monotonic()))),
                ]
                if selection.mode == "registry":
                    arguments.append("--skip-present")
                if route == "ssh-stdio":
                    arguments.append("--stdio")
                if identity in ready_paths:
                    arguments.extend(["--connections-per-path", str(settings.get("connections_per_path", 1))])
                process = runner.start_receiver(host, binaries[host], str(Path(target_files[identity]).parent), arguments, ready)
                peers[identity] = process
                peer_diagnostics[identity] = Lines(process.stderr)
                if route == "ssh-stdio":
                    _bridge(runner, process, source_host, source_binary, source_dir)
                else:
                    Lines(process.stdout)

            def check_peers():
                for identity, process in peers.items():
                    if process.poll() not in {None, 0}:
                        raise OperationError(f"receiver {ids[identity]} exited before completion: "
                                             + "\n".join(peer_diagnostics[identity].tail)[-2000:])

            final = source_events.event(deadline, check_peers)
            while final.get("type") == "progress":
                if final.get("version") != 1 or final.get("transfer") != ready["transfer"]:
                    raise OperationError("source returned invalid progress identity")
                progress.event(final)
                logger.info("OCI Relay progress: %s", final.get("metrics", {}))
                final = source_events.event(deadline, check_peers)
            if final.get("type") != "result" or final.get("version") != 1 or final.get("transfer") != ready["transfer"]:
                raise OperationError("source returned an invalid final result")
            outcomes = final.get("receivers", {})
            if set(outcomes) != set(ids):
                raise OperationError("source did not report every required receiver")
            result = {}
            errors = {}
            for identity, host in ids.items():
                observation = outcomes[identity]
                if observation.get("cleanup_error"):
                    logger.warning("OCI Relay receiver %s cleanup incomplete: %s", host, observation["cleanup_error"])
                code = peers[identity].wait(timeout=max(0.1, min(15, deadline - time.monotonic())))
                valid = (
                    code == 0 and observation.get("state") == "COMPLETE"
                    and observation.get("manifest_digest") == ready["manifest_digest"]
                    and observation.get("tag") == request.image and observation.get("image_id")
                    and observation.get("image_id") == ready.get("config_digest", observation.get("image_id"))
                )
                result[host] = ("already_present" if observation.get("already_present") else "complete") if valid else "failed"
                progress.outcome(identity, observation, valid)
                if not valid:
                    errors[host] = observation.get("error") or "\n".join(peer_diagnostics[identity].tail)[-2000:] or "receiver failed"
            source_code = source_process.wait(timeout=15)
            if source_code and all(value in {"complete", "already_present"} for value in result.values()):
                raise OperationError("source process failed after reporting complete receivers")
            logger.info("OCI Relay receiver timings: %s", {
                ids[identity]: {name: observation.get(name, 0) for name in (
                    "pull_seconds", "last_download_seconds", "last_extract_seconds",
                    "import_method", "reused_layers", "import_archive_bytes", "load_seconds",
                    "paths", "store", "local_bytes", "config_digest", "docker_image_id", "installed_manifest_digest",
                    "cached_blob_layers", "discovery_seconds", "discovery_images", "discovery_inspected",
                    "discovery_complete", "discovery_stop_reason", "discovery_stale", "cache_probe_seconds", "cache_probe_limited",
                )} for identity, observation in outcomes.items()
            })
            logger.info("OCI Relay completed: %s", final.get("metrics", {}))
            if "registry_metrics" in final:
                progress.event({"registry_metrics": final["registry_metrics"]})
                logger.info("OCI Relay registry: %s", final["registry_metrics"])
            mark("transfer_import")
            completed = all(value in {"complete", "already_present"} for value in result.values())
            return ImageCopyResult(result, errors)
        finally:
            cleanup_start = time.monotonic()
            progress.phase("cleaning up transfer resources")
            cleanup = None
            try:
                heartbeat_stop.set()
                if source_process is not None and source_process.poll() is None:
                    try:
                        source_process.stdin.write(b'{"type":"cancel"}\n')
                        source_process.wait(timeout=15)
                    except (OSError, ValueError, subprocess.TimeoutExpired):
                        source_process.cancel()
                cleanup = runner.close()
                for error in cleanup:
                    logger.warning("OCI Relay cleanup incomplete: %s", error)
                timings.update(cleanup_seconds=time.monotonic() - cleanup_start,
                               total_seconds=time.monotonic() - operation_start,
                               state="COMPLETE" if completed else "FAILED")
                logger.info("OCI Relay timings: %s", timings)
            finally:
                progress.close(completed and cleanup is not None, cleanup is None or bool(cleanup))


PROVIDER = RelayProvider()
