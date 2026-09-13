"""Same-host startup measurements and strategy-provided inference readiness."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from importlib.resources import files
from sparkrun.core.readiness import DOCKER_HOST_OBSERVER, OPENAI_CHAT_STREAM, INFERENCE_STYLES, ObservationUnavailable

logger = logging.getLogger(__name__)


def validate_observation(value, *, require_inference=False, expected_style=None, normalize=False):
    if not isinstance(value, dict) or value.get("format") != 1 or value.get("observer") != "rank0":
        raise ValueError("invalid startup observation")
    if value.get("measurement") not in {"rank0-acceptance-v1", "sparkrun-rank0-v1"}:
        raise ValueError("unsupported startup measurement")
    if not isinstance(value.get("container_id"), str) or not value["container_id"]:
        raise ValueError("startup observation has no container identity")
    start = value.get("container_started_unix_ns")
    if type(start) is not int or start <= 0:
        raise ValueError("invalid startup timestamps")
    for key in ("port_open_unix_ns", "http_ready_unix_ns"):
        if key in value and (type(value[key]) is not int or value[key] < start):
            raise ValueError("invalid startup readiness timestamp")
    if value.get("inference_requested") is False:
        if (
            require_inference
            or value["measurement"] != "sparkrun-rank0-v1"
            or value.get("endpoint_ready") is not True
            or value.get("inference_ready") is not False
            or value.get("response_validated") is not False
            or "first_token_unix_ns" in value
            or "first_token_field" in value
            or "port_open_unix_ns" not in value
            or "http_ready_unix_ns" not in value
        ):
            raise ValueError("invalid endpoint-only startup observation")
    else:
        if value.get("inference_ready") is not True:
            raise ValueError("startup observation has no successful inference")
        first = value.get("first_token_unix_ns")
        if type(first) is not int or first < start:
            raise ValueError("invalid startup timestamps")
        if value.get("first_token_field") not in ("content", "reasoning", "reasoning_content"):
            raise ValueError("startup observation has no text token")
    # ColdSnap rank0-acceptance-v1 receipts have a fixed Docker/Chat contract.
    # Native host observations record the actual protocol; older receipts that
    # predate multiple handlers remain Chat. Never relabel a strategy receipt.
    native_style = value.get("inference_style", OPENAI_CHAT_STREAM)
    if (
        value["measurement"] == "sparkrun-rank0-v1"
        and value.get("inference_requested") is not False
        and (not isinstance(native_style, str) or native_style not in INFERENCE_STYLES)
    ):
        raise ValueError("startup observation has incompatible inference_style")
    normalized = dict(value)
    for key, expected in (
        ("executor", "docker"),
        ("observer_location", "rank0-container" if value["measurement"] == "rank0-acceptance-v1" else DOCKER_HOST_OBSERVER.location),
        ("start_boundary", DOCKER_HOST_OBSERVER.start_boundary),
        (
            "inference_style",
            None
            if value.get("inference_requested") is False
            else native_style
            if value["measurement"] == "sparkrun-rank0-v1"
            else OPENAI_CHAT_STREAM,
        ),
    ):
        if key in normalized and normalized[key] != expected:
            raise ValueError("startup observation has incompatible " + key)
        normalized[key] = expected
    if expected_style is not None and normalized["inference_style"] != expected_style:
        raise ValueError("startup observation does not satisfy the requested inference style")
    return normalized if normalize else value


def run_probe(host, config, *, ssh_kwargs=None, cancel=None):
    """Run one head-local probe; cancellation terminates only this subprocess."""
    from sparkrun.orchestration.ssh import build_ssh_cmd, should_run_locally

    if cancel is not None and cancel.is_set():
        raise InterruptedError("cancelled")
    source = files("sparkrun.scripts").joinpath("startup_probe.py").read_text()
    source += (
        "\nfrom math import inf\ntry:\n    print(json.dumps(observe("
        + repr(config)
        + ")))\nexcept ObservationUnavailable as error:\n    print(json.dumps({'unsupported_observer': str(error)}))\n"
    )
    script = "exec python3 - <<'SPARKRUN_STARTUP_PY'\n" + source + "\nSPARKRUN_STARTUP_PY\n"
    kwargs = {key: value for key, value in (ssh_kwargs or {}).items() if key in {"ssh_user", "ssh_key", "ssh_options"}}
    command = ["bash", "-s"] if should_run_locally(host, kwargs.get("ssh_user")) else [*build_ssh_cmd(host, **kwargs), "bash", "-s"]
    deadline = time.monotonic() + sum(config[key] for key in ("port_timeout_s", "health_timeout_s", "inference_timeout_s")) + 30
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    try:
        pending = script
        while True:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("startup observation exceeded its deadline")
            try:
                stdout, stderr = process.communicate(pending, timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                pending = None
        if process.returncode:
            detail = stderr[-1000:]
            if config.get("api_key"):
                detail = detail.replace(config["api_key"], "[redacted]")
            raise RuntimeError("rank-0 readiness probe failed: " + detail)
        if len(stdout) > 65536:
            raise RuntimeError("startup observation exceeds its size limit")
        value = json.loads(stdout)
        if isinstance(value, dict) and isinstance(value.get("unsupported_observer"), str):
            raise ObservationUnavailable(value["unsupported_observer"])
        return validate_observation(
            value,
            normalize=True,
            require_inference=config.get("inference", True),
            expected_style=config.get("inference_style") if config.get("inference", True) else None,
        )
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()


def observe_launch(result, *, settings, style=None, observer=None, ssh_kwargs=None, cancel=None, timeline=None, parent=None):
    """Use a strategy receipt once, otherwise measure on the Docker head host."""
    from sparkrun.core.launcher import ServeReadiness
    from sparkrun.orchestration.primitives import detect_host_ip
    from sparkrun.utils import is_local_host

    host = result.host_list[0] if result.host_list else "localhost"
    address = "127.0.0.1" if is_local_host(host) else host
    container = result.runtime.get_head_container_name(result.cluster_id, is_solo=result.is_solo)
    if cancel is not None and cancel.is_set():
        return ServeReadiness(False, host, address, result.serve_port, container, reason="cancelled")
    observation = getattr(result, "startup_observation", None)
    try:
        if observation:
            observation = validate_observation(dict(observation), normalize=True)
            if settings.inference and (not observation.get("inference_ready") or observation["inference_style"] != style):
                # A valid receipt for one API cannot satisfy another API's
                # requested readiness check. Measure again without relabeling it.
                observation = None
        if not observation:
            if observer != DOCKER_HOST_OBSERVER:
                raise ObservationUnavailable("executor observer adapter is not implemented")
            if not is_local_host(host):
                address = detect_host_ip(host, ssh_kwargs=ssh_kwargs) or host
            if cancel is not None and cancel.is_set():
                raise InterruptedError("cancelled")
            resolve_key = getattr(result.runtime, "resolve_api_key", None)
            api_key = resolve_key(result.recipe, result.overrides) if resolve_key else None

            observation = run_probe(
                host,
                {
                    "container": container,
                    "address": address,
                    "port": result.serve_port,
                    "port_timeout_s": settings.port_timeout_s,
                    "health_timeout_s": settings.health_timeout_s,
                    "inference": settings.inference,
                    "inference_style": style,
                    "health_path": result.runtime.readiness_health_path,
                    "inference_timeout_s": settings.inference_timeout_s,
                    "prompt": settings.inference_prompt,
                    "model": (result.overrides or {}).get("served_model_name")
                    or getattr(result.recipe, "effective_served_model_name", None),
                    "api_key": api_key,
                },
                ssh_kwargs=ssh_kwargs,
                cancel=cancel,
            )
        if cancel is not None and cancel.is_set():
            raise InterruptedError("cancelled")
    except InterruptedError:
        return ServeReadiness(False, host, address, result.serve_port, container, reason="cancelled")
    except ObservationUnavailable:
        raise
    except (OSError, ValueError, RuntimeError) as error:
        logger.warning("Inference readiness observation failed: %s", error)
        return ServeReadiness(False, host, address, result.serve_port, container, reason="inference")
    start = observation["container_started_unix_ns"]
    for name, key in (("port_open", "port_open_unix_ns"), ("http_ready", "http_ready_unix_ns"), ("ttft", "first_token_unix_ns")):
        if key not in observation:
            continue
        span = "serve.startup_" + name
        # Idempotent against the span already being on *this timeline*, not
        # against the result carrying an observation.  The latter is true both
        # for a repeated wait and for a strategy-supplied receipt, so it kept
        # ColdSnap's ``rank0-acceptance-v1`` measurement out of every timing
        # artifact the launch produced — the one case where the numbers came
        # from somewhere sparkrun could not re-measure.
        if timeline is None or timeline.find(span) is not None:
            continue
        timeline.add_span(
            span,
            clock="host:" + host,
            duration_s=(observation[key] - start) / 1e9,
            wall_start=start / 1e9,
            parent=parent,
            measurement=observation["measurement"],
            observer="rank0",
            # All three are elapsed from the same origin, so they overlap.
            # Tree siblings are otherwise additive; without the marker a
            # consumer summing them reports a startup ~1.5x its real length.
            composition="non_additive",
            timing_semantics="from_container_start",
        )
    result.startup_observation = observation
    return ServeReadiness(
        True,
        host,
        address,
        result.serve_port,
        container,
        startup_observation=observation,
        port_wait_s=observation.get("port_wait_s", 0),
        health_wait_s=observation.get("health_wait_s", 0),
    )
