"""Standalone same-host Docker-start readiness probe, fed over SSH stdin."""

import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Kept local because this standalone script is sent over SSH to the serving host.
# tests/test_readiness_capabilities.py checks these against core.readiness.
OPENAI_CHAT_STREAM = "openai-chat-stream-v1"
OPENAI_RESPONSES_STREAM = "openai-responses-stream-v1"
ANTHROPIC_MESSAGES_STREAM = "anthropic-messages-stream-v1"


class ObservationUnavailable(RuntimeError):
    """Unsupported target environment; preserve the legacy endpoint wait."""


def verify_host_observer():
    """Never combine a remote/VM daemon clock with this host's /proc or clock."""
    if sys.platform != "linux":
        raise ObservationUnavailable("Docker host observation requires Linux")
    endpoint = os.environ.get("DOCKER_HOST") if not os.environ.get("DOCKER_CONTEXT") else None
    if not endpoint:
        endpoint = json.loads(
            subprocess.check_output(["docker", "context", "inspect", "--format", "{{json .Endpoints.docker.Host}}"], timeout=10)
        )
    if not isinstance(endpoint, str) or not endpoint.startswith("unix://"):
        raise ObservationUnavailable("Docker host observation requires a local Unix-socket daemon")
    info = json.loads(subprocess.check_output(["docker", "info", "--format", "{{json .}}"], timeout=10))
    if (
        info.get("OSType") != "linux"
        or "docker desktop" in info.get("OperatingSystem", "").lower()
        or any("rootless" in option for option in info.get("SecurityOptions", []))
    ):
        raise ObservationUnavailable("Docker host observation requires a native, non-rootless Linux daemon")


def unix_ns(timestamp):
    whole, dot, fraction = timestamp.rstrip("Z").partition(".")
    seconds = int(datetime.datetime.fromisoformat(whole).replace(tzinfo=datetime.timezone.utc).timestamp())
    return seconds * 1_000_000_000 + (int((fraction + "000000000")[:9]) if dot else 0)


def inspect_container(name):
    data = subprocess.check_output(["docker", "inspect", "--format", "{{json .}}", name], timeout=10)
    info = json.loads(data)
    if info.get("HostConfig", {}).get("NetworkMode") != "host":
        raise ObservationUnavailable("Docker host observation requires the serving container's host network")
    if not info["State"]["Running"]:
        raise RuntimeError("head container is not running")
    return info["Id"], unix_ns(info["State"]["StartedAt"])


def port_listening(port):
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table) as stream:
                for row in stream:
                    fields = row.split()
                    if len(fields) > 3 and fields[3] == "0A" and fields[1].split(":")[-1] == "%04X" % port:
                        return True
        except FileNotFoundError:
            continue
    return False


def observe(config):
    style = config.get("inference_style", OPENAI_CHAT_STREAM)
    if config.get("inference", True) and style not in INFERENCE_PROBES:
        raise ValueError("unsupported inference readiness style")
    verify_host_observer()
    port_wait_start = time.monotonic()
    container_id, started = inspect_container(config["container"])
    result = {
        "format": 1,
        "measurement": "sparkrun-rank0-v1",
        "observer": "rank0",
        "executor": "docker",
        "observer_location": "rank0-host",
        "start_boundary": "docker.State.StartedAt",
        "inference_style": style if config.get("inference", True) else None,
        "container_id": container_id,
        "container_started_unix_ns": started,
        "observer_started_unix_ns": time.time_ns(),
        "inference_ready": False,
        "inference_requested": config.get("inference", True),
        "endpoint_ready": False,
        "response_validated": False,
        "http_ready_path": config.get("health_path", "/health"),
    }
    deadline = time.monotonic() + config["port_timeout_s"]
    while not port_listening(config["port"]):
        if time.monotonic() >= deadline:
            raise TimeoutError("port")
        inspect_container(config["container"])
        time.sleep(0.1)
    result["port_open_unix_ns"] = time.time_ns()
    result["port_wait_s"] = time.monotonic() - port_wait_start
    health_wait_start = time.monotonic()
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if config.get("api_key"):
        http.addheaders = [("Authorization", "Bearer " + config["api_key"])]
    address = config["address"]
    base = "http://%s:%d" % (("[" + address + "]") if ":" in address else address, config["port"])
    deadline = time.monotonic() + config["health_timeout_s"]
    while True:
        try:
            with http.open(base + result["http_ready_path"], timeout=1) as response:
                if response.status == 200:
                    result["http_ready_unix_ns"] = time.time_ns()
                    result["health_wait_s"] = time.monotonic() - health_wait_start
                    result["endpoint_ready"] = True
                    break
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("health")
        inspect_container(config["container"])
        time.sleep(0.1)
    if not result["inference_requested"]:
        if inspect_container(config["container"]) != (container_id, started):
            raise RuntimeError("head container changed during readiness observation")
        return result
    return INFERENCE_PROBES[style](config, http, base, result)


def observe_openai_chat_stream(config, http, base, result):
    """OpenAI chat/SSE v1: first non-empty content or reasoning delta."""
    payload = {
        "model": probe_model(config, http, base),
        "messages": [{"role": "user", "content": config["prompt"]}],
        "stream": True,
        "temperature": 0,
        "max_tokens": 64,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    request_start = time.monotonic()
    result["request_started_unix_ns"] = time.time_ns()
    content = []
    finished = False

    def accepted():
        if inspect_container(config["container"]) != (result["container_id"], result["container_started_unix_ns"]):
            raise RuntimeError("head container changed during readiness observation")
        wall_request = (result["first_token_unix_ns"] - result["request_started_unix_ns"]) / 1e9
        if abs(wall_request - result["request_ttft_seconds"]) > 0.25:
            raise RuntimeError("host clock changed during inference observation")
        result["response_seconds"] = time.monotonic() - request_start
        return result

    with http.open(request, timeout=config["inference_timeout_s"]) as response:
        total = 0
        event = bytearray()
        while True:
            line = response.readline(65537)
            total += len(line)
            if len(line) > 65536 or total > 1024 * 1024:
                raise RuntimeError("inference response exceeds its size limit")
            if time.monotonic() - request_start > config["inference_timeout_s"]:
                raise TimeoutError("inference")
            if not line:
                raise RuntimeError("inference stream ended before acceptance")
            if line.strip():
                if line.startswith(b"data:"):
                    event.extend(line[5:].strip() + b"\n")
                if len(event) > 65536:
                    raise RuntimeError("inference event exceeds its size limit")
                continue
            if not event:
                continue
            value = bytes(event).strip()
            event.clear()
            if value == b"[DONE]":
                if not result["inference_ready"]:
                    raise RuntimeError("inference stream contained no text")
                if not finished or "".join(content).strip() != config["expected"]:
                    raise RuntimeError("inference response failed exact validation")
                result.update(response_validated=True, content="".join(content))
                return accepted()
            chunk = json.loads(value)
            if not isinstance(chunk, dict) or "error" in chunk:
                raise RuntimeError("server reported an inference error")
            choices = chunk.get("choices", [])
            if not isinstance(choices, list) or len(choices) > 1:
                raise RuntimeError("inference stream returned invalid choices")
            for choice in choices:
                if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                    raise RuntimeError("inference stream returned an invalid choice")
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    raise RuntimeError("inference stream returned an invalid delta")
                if choice.get("finish_reason"):
                    finished = True
                for field in ("content", "reasoning", "reasoning_content"):
                    text = delta.get(field)
                    if field == "content" and isinstance(text, str):
                        content.append(text)
                    if isinstance(text, str) and text:
                        if not result["inference_ready"]:
                            result.update(
                                first_token_unix_ns=time.time_ns(),
                                first_token_field=field,
                                request_ttft_seconds=time.monotonic() - request_start,
                                inference_ready=True,
                                prompt_sha256=hashlib.sha256(config["prompt"].encode()).hexdigest(),
                                max_tokens=64,
                                temperature=0,
                                model=payload["model"],
                            )
                        if "expected" not in config:
                            return accepted()


def probe_model(config, http, base):
    """Prefer the launched model; standalone callers may use model discovery."""
    if isinstance(config.get("model"), str) and config["model"]:
        return config["model"]
    with http.open(base + "/v1/models", timeout=10) as response:
        models = json.load(response)["data"]
    if not models or not isinstance(models[0].get("id"), str) or not models[0]["id"]:
        raise RuntimeError("server did not advertise a model")
    return models[0]["id"]


def stream_events(response, request_start, timeout):
    """Read bounded SSE data frames while enforcing the inference deadline."""
    total = 0
    event = bytearray()
    while True:
        line = response.readline(65537)
        total += len(line)
        if len(line) > 65536 or total > 1024 * 1024:
            raise RuntimeError("inference response exceeds its size limit")
        if time.monotonic() - request_start > timeout:
            raise TimeoutError("inference")
        if not line:
            raise RuntimeError("inference stream ended before acceptance")
        if line.strip():
            if line.startswith(b"data:"):
                event.extend(line[5:].strip() + b"\n")
            if len(event) > 65536:
                raise RuntimeError("inference event exceeds its size limit")
            continue
        if event:
            value = bytes(event).strip()
            event.clear()
            chunk = json.loads(value)
            if not isinstance(chunk, dict) or "error" in chunk or chunk.get("type") == "error":
                raise RuntimeError("server reported an inference error")
            yield chunk


def observe_native_event_stream(config, http, base, result, *, style):
    """Responses/Anthropic SSE: first text or thinking, optional exact acceptance."""
    model = probe_model(config, http, base)
    payload = {"model": model, "stream": True, "temperature": 0}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if style == OPENAI_RESPONSES_STREAM:
        path = "/v1/responses"
        payload.update(input=config["prompt"], max_output_tokens=64, store=False)
    else:
        path = "/v1/messages"
        payload.update(messages=[{"role": "user", "content": config["prompt"]}], max_tokens=64)
        headers["anthropic-version"] = "2023-06-01"
        if config.get("api_key"):
            headers["x-api-key"] = config["api_key"]
    request = urllib.request.Request(base + path, data=json.dumps(payload).encode(), headers=headers)
    request_start = time.monotonic()
    result["request_started_unix_ns"] = time.time_ns()
    content = []
    finished = False

    def accepted():
        if inspect_container(config["container"]) != (result["container_id"], result["container_started_unix_ns"]):
            raise RuntimeError("head container changed during readiness observation")
        wall_request = (result["first_token_unix_ns"] - result["request_started_unix_ns"]) / 1e9
        if abs(wall_request - result["request_ttft_seconds"]) > 0.25:
            raise RuntimeError("host clock changed during inference observation")
        result["response_seconds"] = time.monotonic() - request_start
        return result

    with http.open(request, timeout=config["inference_timeout_s"]) as response:
        for chunk in stream_events(response, request_start, config["inference_timeout_s"]):
            kind = chunk.get("type")
            field, text, terminal = None, None, False
            if style == OPENAI_RESPONSES_STREAM:
                if kind in ("response.failed", "response.incomplete"):
                    raise RuntimeError("server reported an unsuccessful response")
                if kind == "response.output_text.delta":
                    field, text = "content", chunk.get("delta")
                elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                    field, text = "reasoning", chunk.get("delta")
                elif kind == "response.completed":
                    completed = chunk.get("response")
                    if not isinstance(completed, dict):
                        raise RuntimeError("inference stream returned an invalid response")
                    finished = completed.get("status") == "completed"
                    terminal = True
            else:
                if kind in ("content_block_delta", "content_block_start"):
                    delta = chunk.get("delta" if kind == "content_block_delta" else "content_block", {})
                    if not isinstance(delta, dict):
                        raise RuntimeError("inference stream returned an invalid content block")
                    if delta.get("type") in ("text_delta", "text"):
                        field, text = "content", delta.get("text")
                    elif delta.get("type") in ("thinking_delta", "thinking"):
                        field, text = "reasoning", delta.get("thinking")
                elif kind == "message_delta":
                    delta = chunk.get("delta", {})
                    finished = isinstance(delta, dict) and delta.get("stop_reason") in ("end_turn", "stop_sequence", "max_tokens")
                elif kind == "message_stop":
                    terminal = True
            if terminal:
                if not result["inference_ready"]:
                    raise RuntimeError("inference stream contained no text")
                if not finished or "".join(content).strip() != config.get("expected"):
                    raise RuntimeError("inference response failed exact validation")
                result.update(response_validated=True, content="".join(content))
                return accepted()
            if field and text is not None and not isinstance(text, str):
                raise RuntimeError("inference stream returned invalid text")
            if not text:
                continue
            if field == "content":
                content.append(text)
            if not result["inference_ready"]:
                result.update(
                    first_token_unix_ns=time.time_ns(),
                    first_token_field=field,
                    request_ttft_seconds=time.monotonic() - request_start,
                    inference_ready=True,
                    prompt_sha256=hashlib.sha256(config["prompt"].encode()).hexdigest(),
                    max_tokens=64,
                    temperature=0,
                    model=model,
                )
            if "expected" not in config:
                return accepted()


def observe_openai_responses_stream(config, http, base, result):
    return observe_native_event_stream(config, http, base, result, style=OPENAI_RESPONSES_STREAM)


def observe_anthropic_messages_stream(config, http, base, result):
    return observe_native_event_stream(config, http, base, result, style=ANTHROPIC_MESSAGES_STREAM)


INFERENCE_PROBES = {
    OPENAI_CHAT_STREAM: observe_openai_chat_stream,
    OPENAI_RESPONSES_STREAM: observe_openai_responses_stream,
    ANTHROPIC_MESSAGES_STREAM: observe_anthropic_messages_stream,
}
