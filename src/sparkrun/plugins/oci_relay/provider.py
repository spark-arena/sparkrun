# SPDX-FileCopyrightText: 2026 Spark Arena
# SPDX-License-Identifier: Apache-2.0

"""Source-adjacent relay orchestration for Sparkrun's image-copy API."""

from __future__ import annotations

import json
import logging
import platform
from collections import Counter
from dataclasses import replace
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from urllib.parse import urlsplit

from . import __version__, pins, decoder
from .disk import registry_cache_budget
from .host import Lines, OperationError, Runner, pump
from .parallel import parallel
from .paths import arguments as path_arguments, qualify as qualify_paths
from .progress import PROGRESS, Progress
from .release import BinaryUnavailable, acquire, acquire_decoder
from .source_policy import SourceUnavailable, detect, validate
from .tuning import limits, probe

logger = logging.getLogger(__name__)


class RegistrySourceUnavailable(OperationError):
    """Registry metadata preparation failed before starting any receiver."""


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
    supports_offline_pull = True
    # Sparkrun honors this default only for pre-transfer unsupported requests in
    # provider=auto mode. Explicit host configuration takes precedence.
    fallback_on_unsupported = True

    def __init__(self):
        # Serialize image groups for now: a new group cannot multiply the host budgets.
        self._operations = threading.Lock()

    @staticmethod
    def _settings(request):
        if request.config is None:
            from sparkrun.core.config import SparkrunConfig

            return SparkrunConfig().plugin_settings("oci-relay")
        return request.config.plugin_settings("oci-relay")

    @staticmethod
    def _require_pin_api():
        import sparkrun.plugins as api

        if getattr(api, "IMAGE_RUNTIME_API_VERSION", None) != 1:
            raise api.ImageDistributionUnsupported(
                "digest-pinned relay imports require Sparkrun develop-next with image-runtime API 1"
            )

    def local_image(self, request):
        if pins.digest(request.image) is None:
            return None
        runner = Runner(request.session, self._settings(request))
        try:
            return pins.resolve(runner, request.source_host, request.image)
        finally:
            runner.close()

    def _pull_pinned(self, request, settings):
        from sparkrun.plugins import ImageCopyResult, ImageDistributionUnsupported
        from sparkrun.core.progress import progress_heartbeat

        self._require_pin_api()
        pins.digest(request.image)
        settings = validate(settings)
        started = time.monotonic()
        timeout = request.timeout or settings.get("timeout_seconds", 3600)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 86400:
            raise ValueError("relay timeout must be between 1 and 86400 seconds")
        registry = (not request.offline and settings.get("source_mode", "auto") in {"auto", "registry"}
                    and (settings.get("registry_source", True) or settings.get("source_mode") == "registry")
                    and not settings.get("manifest"))
        if request.offline and request.force_pull:
            raise OperationError("offline mode cannot fetch a fresh pinned image")
        if request.dry_run:
            return self.copy(request, registry=registry)
        logger.log(PROGRESS, "OCI Relay: checking verified registry pin on receivers: %s", request.image)
        runner = Runner(request.session, settings)
        with progress_heartbeat(logger, "OCI Relay: checking verified registry pin"):
            try:
                present = {} if request.force_pull else parallel(
                    request.targets, lambda host: pins.resolve(runner, host, request.image),
                )
                if time.monotonic() - started >= timeout:
                    raise OperationError("pinned cache checks exceeded operation deadline")
                if present and all(present.values()):
                    logger.log(PROGRESS, "OCI Relay: pinned image already verified on all %d receiver(s); no transfer", len(present))
                    return ImageCopyResult(dict.fromkeys(request.targets, "already_present"), runtime_images=present)
                if not registry:
                    # Offline/local mode can reuse a verified pin on a receiver,
                    # including imports whose Docker store lacks the RepoDigest.
                    source_host = request.source_host
                    source_id = pins.resolve(runner, source_host, request.image)
                    if source_id is None:
                        source_host = next((host for host, value in present.items() if value), None)
                        source_id = present.get(source_host)
                    if source_id is None:
                        raise ImageDistributionUnsupported("pinned image is not resident on an available relay source")
                    request = replace(request, source_host=source_host)
            finally:
                runner.close()
        if registry:
            logger.log(PROGRESS, "OCI Relay: fetching exact registry pin and distributing directly to receivers: %s", request.image)
        remaining = timeout - (time.monotonic() - started)
        if remaining < 1:
            raise OperationError("pinned source selection exceeded operation deadline")
        return self.copy(replace(request, timeout=remaining), registry=registry)

    def pull(self, request):
        """Own registry pulls and controller latest refreshes before Docker imports."""
        from sparkrun.utils.images import is_pullable_image_ref, parse_image_ref
        from .source_policy import LOCAL_DOCKER

        settings = self._settings(request)
        if pins.digest(request.image):
            return self._pull_pinned(request, settings)
        if request.offline or (settings.get("registry_source", True) is False and settings.get("source_mode") != "registry"):
            return None
        if settings.get("source_mode", "auto") not in {"auto", "registry"} or settings.get("manifest"):
            return None
        ref = parse_image_ref(request.image)
        started = time.monotonic()
        timeout = request.timeout or settings.get("timeout_seconds", 3600)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 86400:
            raise ValueError("relay timeout must be between 1 and 86400 seconds")
        cached_image = None
        if not request.dry_run and not request.force_pull and settings.get("source_mode") != "registry" and (request.source_host is not None or platform.system() == "Linux"):
            logger.log(PROGRESS, "OCI Relay: checking source image on %s", request.source_host or "controller")
            runner = Runner(request.session, settings)
            try:
                inspected = runner.connection(request.source_host).execute(
                    request.source_host or "localhost", LOCAL_DOCKER + ["image", "inspect", "--format={{.Id}}", request.image], timeout=min(30, timeout),
                )
                if inspected.returncode == 0:
                    # Core refreshes controller-local latest references, while
                    # delegated sources retain a present image unless forced.
                    refresh = (request.source_host is None and is_pullable_image_ref(request.image)
                               and ref.tag in {None, "latest"})
                    if not refresh:
                        logger.log(PROGRESS, "OCI Relay: using existing source image %s", request.image)
                        return None
                    cached_image = inspected.stdout.decode().strip()
                    if not re.fullmatch(r"sha256:[0-9a-f]{64}", cached_image):
                        raise OperationError("Docker returned an invalid cached image ID")
            finally:
                runner.close()
        if not ref.tag:
            request = replace(request, image=request.image + ":latest")
        action = "refreshing latest image from registry" if cached_image else "pulling from registry"
        logger.log(PROGRESS, "OCI Relay: %s and distributing directly to receivers: %s", action, request.image)
        remaining = timeout - (time.monotonic() - started)
        if remaining < 1:
            raise OperationError("registry source selection exceeded operation deadline")
        try:
            return self.copy(replace(request, timeout=remaining), registry=True)
        except RegistrySourceUnavailable as error:
            if cached_image is None:
                raise
            # Best-effort refresh can reuse the original local image only
            # before receiver startup. Never switch identity after a partial
            # transfer, corrupt payload, receiver failure or cancellation.
            remaining = timeout - (time.monotonic() - started)
            if remaining < 1:
                raise OperationError("registry refresh exhausted the operation deadline")
            logger.warning("OCI Relay: registry refresh unavailable before transfer (%s); using cached image %s (%s)",
                           error, request.image, cached_image)
            return self.copy(replace(request, timeout=remaining), source_image=cached_image)

    def copy(self, request, *, registry=False, source_image=None):
        from sparkrun.plugins import ImageCopyResult, ImageDistributionUnsupported

        settings = self._settings(request)
        pin = pins.digest(request.image)
        if pin:
            self._require_pin_api()
            if request.offline and settings.get("source_mode") == "registry":
                settings = dict(settings, source_mode="auto")
        if registry:
            settings = dict(settings, source_mode="registry")
        if settings.get("source_mode") == "registry" and request.source_host is None and platform.system() != "Linux":
            # A non-Linux control node coordinates; a Linux receiver fetches
            # registry data using its own registry credentials and network.
            if not request.targets:
                raise ImageDistributionUnsupported("registry relay requires a Linux execution host")
            request = replace(request, source_host=request.targets[0])
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
            if pin and settings.get("source_mode") != "registry":
                source_image = self.local_image(request)
                if source_image is None:
                    raise ImageDistributionUnsupported("no verified local source for registry pin")
                remaining = timeout - (time.monotonic() - started)
                if remaining < 1:
                    raise OperationError("pinned source checks exceeded operation deadline")
            return self._copy(request, settings, int(remaining), source_image=source_image)
        except (BinaryUnavailable, SourceUnavailable) as error:
            raise ImageDistributionUnsupported(str(error)) from error
        finally:
            self._operations.release()

    def _copy(self, request, settings, timeout, *, source_image=None):
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
        source_image = source_image or request.image
        pin = pins.digest(request.image)
        destination = pins.retention_tag(request.image) if pin else request.image

        def mark(name):
            nonlocal phase_start
            now = time.monotonic()
            timings[name + "_seconds"] = now - phase_start
            phase_start = now

        heartbeat_stop = threading.Event()
        source_process = None
        try:
            if pin:
                progress.phase("checking registry pin metadata storage")
                parallel(request.targets, lambda host: pins.preflight(runner, host, request.image))
            execution_hosts = list(dict.fromkeys([request.source_host, *request.targets]))
            architectures = parallel(execution_hosts, runner.architecture)
            # Acquire/verify each architecture once. Concurrent extraction into
            # the same release cache would needlessly duplicate work.
            releases = {arch: acquire(arch, settings, offline=request.offline)
                        for arch in dict.fromkeys(architectures.values())}
            binaries = parallel(execution_hosts, lambda host: runner.stage_binary(
                host, *releases[architectures[host]], __version__,
            ))
            helpers = {}
            if settings.get("receiver_decoder", "auto") != "none" and settings.get("receiver_import", "pull") == "pull":
                supported = [host for host in request.targets if "receiver-unpigz-v1" in getattr(runner, "binary_capabilities", {}).get(host, [])]
                decoders = {arch: acquire_decoder(arch, settings, releases[arch][0])
                            for arch in dict.fromkeys(architectures[host] for host in supported)}
                helpers = parallel([host for host in supported if decoders[architectures[host]] is not None],
                                   lambda host: runner.stage_decoder(host, *decoders[architectures[host]]))
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
            selection = detect(runner, source_host, source_image, settings, facts[source_host])
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
                "version": 1, "source": selection.mode, "image": source_image,
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
                from .registry_ranges import plan as range_plan

                plan.update(registry_config=settings.get("registry_config", ""),
                            registry_plain_http=settings.get("registry_plain_http", False),
                            registry_cache_bytes=registry_cache_budget(runner, source_host, plan["spool_dir"], settings))
                plan.update(range_plan(settings, getattr(runner, "binary_capabilities", {}).get(source_host, [])))
            mark("session")
            progress.phase("fetching registry manifest and configuration" if selection.mode == "registry"
                           else f"preparing {selection.mode} source")
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
                while ready.get("type") == "progress":
                    progress.event(ready)
                    ready = source_events.event(deadline)
            except OperationError as error:
                failure = RegistrySourceUnavailable if selection.mode == "registry" else OperationError
                raise failure(str(error) + ": " + "\n".join(source_errors.tail)[-4000:]) from error
            if ready.get("type") != "ready" or ready.get("version") != 1:
                # Malformed protocol events are not registry availability failures.
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
                    "--tag", destination, "--replace-tag", "--max-buffer-bytes", str(peer_limits["max_buffer_bytes"]),
                    "--max-source-streams", str(peer_limits["source_streams"]),
                    "--import", settings.get("receiver_import", "pull"),
                    "--max-import-bytes", str(settings.get("max_import_bytes", 0)),
                    "--timeout-seconds", str(max(1, int(deadline - time.monotonic()))),
                ]
                if selection.mode == "registry" or pin:
                    arguments.append("--skip-present")
                if route == "ssh-stdio":
                    arguments.append("--stdio")
                if identity in ready_paths:
                    arguments.extend(["--connections-per-path", str(settings.get("connections_per_path", 1))])
                # New binaries negotiate striping automatically. Pass overrides
                # only when requested, keeping default use of older releases valid.
                for key in ("stripe_threshold_bytes", "stripe_streams", "stripe_piece_bytes", "http2_stream_window_bytes"):
                    if key in settings:
                        arguments.extend(["--" + key.replace("_", "-"), str(settings[key])])
                if "receiver-unpigz-v1" in getattr(runner, "binary_capabilities", {}).get(host, []):
                    arguments.extend(decoder.arguments(runner, host, helpers.get(host), settings, facts[host], peer_limits))
                elif settings.get("receiver_decoder") == "unpigz":
                    raise OperationError("receiver binary lacks bundled decoder capability")
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
            if pin and selection.mode == "registry" and final.get("registry_root_digest") != pin:
                raise OperationError("registry source did not verify the requested root digest")
            outcomes = final.get("receivers", {})
            if set(outcomes) != set(ids):
                raise OperationError("source did not report every required receiver")
            result = {}
            errors = {}
            runtime_images = {}
            for identity, host in ids.items():
                observation = outcomes[identity]
                if observation.get("cleanup_error"):
                    logger.warning("OCI Relay receiver %s cleanup incomplete: %s", host, observation["cleanup_error"])
                code = peers[identity].wait(timeout=max(0.1, min(15, deadline - time.monotonic())))
                valid = (
                    code == 0 and observation.get("state") == "COMPLETE"
                    and observation.get("manifest_digest") == ready["manifest_digest"]
                    and observation.get("tag") == destination and observation.get("image_id")
                    and observation.get("image_id") == ready.get("config_digest", observation.get("image_id"))
                    and (not pin or (pins.valid_id(observation.get("docker_image_id"))
                                     and pins.valid_id(ready.get("config_digest"))))
                )
                result[host] = ("already_present" if observation.get("already_present") else "complete") if valid else "failed"
                if valid and pin:
                    runtime_images[host] = observation["docker_image_id"]
                progress.outcome(identity, observation, valid)
                if not valid:
                    errors[host] = observation.get("error") or "\n".join(peer_diagnostics[identity].tail)[-2000:] or "receiver failed"
            source_code = source_process.wait(timeout=15)
            if source_code and all(value in {"complete", "already_present"} for value in result.values()):
                raise OperationError("source process failed after reporting complete receivers")
            logger.info("OCI Relay receiver timings: %s", {
                ids[identity]: {name: observation.get(name, 0) for name in (
                    "pull_seconds", "last_download_seconds", "last_extract_seconds", "decoder",
                    "import_method", "reused_layers", "import_archive_bytes", "load_seconds",
                    "paths", "store", "local_bytes", "config_digest", "docker_image_id", "installed_manifest_digest",
                    "cached_blob_layers", "discovery_seconds", "discovery_images", "discovery_inspected",
                    "discovery_complete", "discovery_stop_reason", "discovery_stale", "cache_probe_seconds", "cache_probe_limited",
                )} for identity, observation in outcomes.items()
            })
            logger.info("OCI Relay transfer metrics: %s", final.get("metrics", {}))
            if "registry_metrics" in final:
                progress.event({"registry_metrics": final["registry_metrics"]})
                logger.info("OCI Relay registry: %s", final["registry_metrics"])
            mark("transfer_import")
            receivers_complete = all(value in {"complete", "already_present"} for value in result.values())
            if pin and receivers_complete:
                progress.phase("saving verified registry pin records")
                try:
                    parallel(request.targets, lambda host: pins.record(
                        runner, host, request.image, runtime_images[host], ready["config_digest"],
                    ))
                except OperationError as error:
                    raise OperationError(
                        "images imported and verified, but saving registry pin records failed; "
                        "restore cache storage and retry (installed images can be reused): " + str(error)
                    ) from error
            # Metadata publication is required for durable pinned-image reuse.
            # Do not let cleanup print success when publication raised above.
            completed = receivers_complete
            if pin:
                return ImageCopyResult(result, errors, runtime_images=runtime_images)
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
