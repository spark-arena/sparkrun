# Startup readiness and timing

By default, watched supported Docker launches for vLLM and SGLang use one
streaming chat request as the final readiness signal. A listening port or a
successful health response alone does not establish that a model can generate.
Runtimes and executors without compatible declarations retain endpoint checks.
Selection uses runtime and executor capability declarations, not runtime-family
names. Docker support currently requires Linux, a local Unix-socket daemon,
and the serving container's host network; rootless daemons and Docker Desktop
are excluded from this host-local observer.

The readiness observer runs on the head host (rank 0), using host Python 3,
Docker inspection, and the local inference endpoint. It records:

| Metric | Boundary measured from the serving container's Docker `State.StartedAt` |
| --- | --- |
| TTR port-open | First observed TCP LISTEN state on the serving port |
| TTR HTTP-ready | First observed HTTP 200 from `/health` |
| Startup TTFT | First non-empty `content`, `reasoning`, or `reasoning_content` streaming delta |

The first two are distinct from inference readiness. Empty role, usage, and
keepalive events do not count as generated text. The observer uses the launched
recipe's served model name (falling back to `/v1/models` when unavailable),
sends a single temperature-zero, max-64-token streaming inference
request, and closes it after receiving first text. Ordinary readiness does not
require an exact final reply. A timed-out, empty, or failed stream does not
signal ready. Recipe/override API keys use the runtime's existing key resolver;
keys travel over stdin, not command-line arguments, and are excluded from
the observation and logs.

All timestamps used in the startup duration come from the head host, avoiding
control-node clock skew. The probe verifies container identity and start time
again before accepting its observation. Port detection is passive via
`/proc/net/tcp{,6}`, not a TCP connection. The probe disables proxy environment
handling for its local HTTP calls. Detectable clock steps during inference
invalidate the observation.

The normal launch log prints the three Docker-start metrics when inference is
ready. That line is written while the container log stream is still scrolling,
so `sparkrun run` repeats the same three numbers as a `Startup readiness` block
beside the timing tree once the stream stops, from the same projection. Both
suppress with `--no-timings` only for the block; the live line always prints.

They also appear as overlapping `serve.startup_port_open`,
`serve.startup_http_ready`, and `serve.startup_ttft` spans in the launch timeline,
each marked `composition: non_additive` with `timing_semantics:
from_container_start`. Do not sum these spans or confuse them with total CLI wall
time. The rendered timing tree omits them when the block above it reported the
same figures — its rows sum to its total and these are not terms in that sum —
but they remain in the exported timeline that the diagnostics record and
benchmark metadata read. A strategy-supplied receipt records the same spans as a
sparkrun-measured observation; a repeated wait on one timeline does not record
them twice. Images/model distribution and any preparation before container start
remain separate phases.
`ServeReadiness.port_wait_s` and `health_wait_s` retain their wait-duration
meaning; Docker-start timestamps are in `startup_observation`.

## Configuration

Readiness settings resolve field by field, from lowest to highest priority:

1. Built-in defaults.
2. The top-level `readiness` block in `~/.config/sparkrun/config.yaml`.
3. The top-level `readiness` block in the launched recipe.

Only explicitly supplied recipe fields override the global settings. An
omitted field inherits; an explicit `false` overrides `true` (and vice versa).
The effective policy is separate from model/runtime flags and does not mutate
the global configuration or get baked into exported recipes.

These are the built-in defaults, also usable as a global configuration block:

```yaml
readiness:
  port_timeout_s: 1800
  health_timeout_s: 900
  inference: true
  inference_style: auto
  inference_timeout_s: 120
  inference_prompt: "Reply with exactly: sparkrun-ready"
```

For an embedding-only recipe, override just inference in that recipe's YAML:

```yaml
readiness:
  inference: false
```

This disables the inference request only for this recipe. Docker-start port/HTTP TTR
measurements remain enabled, the log reports endpoint readiness, and TTFT is
`not applicable (inference disabled)`, not zero. No model-list or inference request
is needed by the endpoint-only probe. Other recipes retain the global defaults.

A chat recipe can override the prompt and budget independently:

```yaml
readiness:
  inference: true
  inference_prompt: "Reply with exactly: ready"
  inference_timeout_s: 180
```

The inference timeout is a separate finite, positive budget after port and HTTP
health readiness. For port/health timeouts, zero or negative means wait until
ready, the container fails, or the caller cancels. Unknown recipe fields,
non-boolean inference settings, empty prompts, and invalid timeouts are rejected
when loading the recipe. Omit a field to inherit it; `null` is not an override.
Unknown inference styles are also rejected in global configuration rather than
silently falling back to `auto`.

### Runtime styles and executor observation support

`inference_style: auto` chooses the runtime's preferred style from the recipe's
declared native APIs. An explicit selection can be made in global config or a
recipe. For example, to check Responses on a vLLM recipe:

```yaml
metadata:
  native_apis: [chat_completions, responses, messages]
readiness:
  inference_style: openai-responses-stream-v1
```

| Inference style | Native API declaration | Streaming endpoint |
| --- | --- | --- |
| `openai-chat-stream-v1` | `chat_completions` | `/v1/chat/completions` |
| `openai-responses-stream-v1` | `responses` | `/v1/responses` |
| `anthropic-messages-stream-v1` | `messages` | `/v1/messages` |

Each handler accepts the first non-empty text or reasoning delta. Readiness
checks one selected API; it does not establish that every advertised API works.
Style identifies the protocol/check;
`measurement` identifies the timing/acceptance profile, not the protocol.

Runtime authors declare `readiness_styles` as a preference-ordered tuple and
`readiness_health_path` as the HTTP readiness endpoint. The base declarations
are empty/`None`. vLLM supports all three styles in the order above; SGLang
currently supports Chat only. Both use `/health`. A runtime subclass can explicitly
opt out with an empty tuple. With inference enabled, explicitly selecting a
style the runtime or recipe does not declare fails before launch. `auto` on an opted-out
runtime retains legacy endpoint checks. `inference: false` overrides the probe
request, while supported endpoint-only TTR collection remains available.

The runtime's `native_apis(recipe)` declaration feeds the recipe catalog,
SparkRoute deployment projection, and readiness selection. vLLM defaults to
Chat Completions, Responses, and Anthropic Messages, including custom/nightly
images. Recognized version tags older than 0.12.0 retain the Chat default.
Recipe `metadata.native_apis` explicitly narrows or overrides that family default;
use `[chat_completions]` for a custom build that only serves Chat. Values must be
non-empty and supported by the runtime family. `responses` also requires
`chat_completions` because they share the OpenAI wire-family declaration.
With `[messages]`, `auto` selects the Anthropic probe instead of Chat. Invalid
API declarations are rejected before launch even when inference is disabled.

Executor authors implement `readiness_observer()` for their resolved
configuration, returning a `ReadinessObserver` descriptor or `None`. The
descriptor separates observation adapter, location, and optional start-time
boundary from runtime protocol support. The initial implemented adapter is
`docker-host-v1`, with location `rank0-host` and boundary
`docker.State.StartedAt`. Docker declares it only for host networking, with
additional target-side checks before observation. If that environment is
unsupported, Sparkrun keeps the legacy endpoint wait without inventing startup
metrics. A failed supported probe still fails readiness.

Local and Kubernetes currently return no observer; no Docker commands are sent
to them by readiness. Their future adapters can provide process/container start
boundaries, or inference-only observation without startup timing. Those adapters
and an inference-only timing schema are not implemented by this change. Never
substitute request latency, Pod creation, or control-node launch time for the
recorded serving-instance start boundary.

The policy chain remains defaults → global config → recipe. Capability
declarations constrain that policy; they are not another override layer.

Watched launches, post-launch hooks, proxy registration, and fresh benchmark
launches use the same effective policy. Benchmarks wait for startup readiness
before running their framework's requests. The probe needs Python 3 and Docker
access on the head host; it does not install software there.

Default log-following launches and post-launch hooks use this readiness path.
`--no-follow` retains its fast return after the existing boot-liveness check;
it does not block for inference or promise a TTFT. Ctrl+C detaches from the
workload and cancels the local probe/SSH process. Probe operations also have
bounded timeouts; cancelling observation does not stop the serving container.

## Execution strategies

An execution strategy can return `ActivationResult.startup_observation` with
its successful inference observation. `LaunchResult` passes it to readiness,
which reuses it without sending a second inference when it covers the selected
protocol. A valid receipt for a different protocol triggers a fresh probe when
the executor supports it. ColdSnap's `rank0-acceptance-v1` receipts always attest
to Chat; they cannot be relabeled as Responses or Anthropic. The optional mapping is
empty for existing strategies. Its format-1 contract includes measurement
profile, rank-0 observer, container identity/start, first-token timestamp/field,
and `inference_ready: true`; port/HTTP timestamps and full-response validation
are additional provenance. Endpoint-only observations instead carry
`inference_requested: false`, `endpoint_ready: true`, and both TTR timestamps,
without a first-token timestamp or a claim of successful inference. They cannot
satisfy a wait that requires inference. The current supported profiles are
`sparkrun-rank0-v1` and ColdSnap's `rank0-acceptance-v1`.

ColdSnap plugins that support this handoff supply already-validated full
acceptance response timing. Older compatible ColdSnap controllers can mark
`runtime_info.inference_readiness` as `accepted` without supplying timestamps;
this suppresses a duplicate inference but does not invent TTFT. Updating this
upstream code does not itself update the vendored ColdSnap plugin or controller.

Recipe readiness settings control Sparkrun's probes, not a strategy's own
acceptance contract. Disabling the Sparkrun inference probe does not disable
ColdSnap's mandatory acceptance or discard an already-measured TTFT; the host
continues to reuse that successful observation without another inference.

## Benchmark metadata

Fresh benchmark launches export the rank-local observation under
`sparkrun_benchmark.timing.startup` in benchmark YAML, and `timing.startup` in
Arena submission metadata and the public API result's `metadata`. The same
projection supplies all three. This is an optional, additive block in the
version-1 YAML format; framework result rows and their request-latency/TTFT
fields are unchanged.

For example, this **illustrative schema sample is not a qualification result**:

```yaml
timing:
  startup:
    format: 1
    measurement: sparkrun-rank0-v1
    observer: rank0
    executor: docker
    observer_location: rank0-host
    inference_style: openai-chat-stream-v1
    start_boundary: docker.State.StartedAt
    ttr_port_open_s: 12.25
    ttr_http_ready_s: 12.5
    ttft_s: 13.0
    ttft_status: measured
    inference_requested: true
    inference_ready: true
    response_validated: false
    first_token_field: content
    http_ready_path: /health
    observer_start_delay_s: 0.125
    prompt_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    max_tokens: 64
    temperature: 0
    request_ttft_s: 0.5
```

All `_s` fields are seconds. The three startup metrics are measured from Docker
start; optional `request_ttft_s` measures only the readiness request, not startup
or a framework request. Optional `observer_start_delay_s` exposes how late the
observer began after container start. Prompt hash and generation settings are
included when supplied by the observer; raw prompt/response text, credentials,
container identity, and host addresses are not included in this block.

The startup check runs once before framework work. An already-observed launch
(including ColdSnap acceptance or a post-launch hook's readiness) reuses its
observation without another startup inference. Framework warmup, coherence
checks, and measured requests remain unchanged and separate. ColdSnap retains
its `rank0-acceptance-v1` profile and full-response validation provenance.
Older format-1 receipts are normalized using their fixed Docker/OpenAI profile
contract, without mutating them or requiring a controller/plugin upgrade.
ColdSnap records location `rank0-container`; Sparkrun's host probe records
`rank0-host`. Both use the rank-0 host clock, but run in different contexts.
Conflicting style, executor, location, or boundary declarations are rejected.

With recipe `readiness.inference: false`, normal Docker launches export both
TTRs and `ttft_status: not_applicable`, without a `ttft_s` field or a chat
request. Missing timestamps are omitted, not filled with zero. Unsupported or
legacy paths without a startup observation omit the block entirely.
`--skip-run` also omits it because the benchmark did not observe that launch.
Resumed results (including partially reused runs) omit startup metrics rather
than attributing a current or stale launch to reused measurements. Resumed
benchmarks retain endpoint-only waits and do not add a startup inference probe.

Existing Arena `timing.serve_ready` fields remain endpoint **wait durations**,
not Docker-start TTR. Existing benchmark JSON/CSV framework outputs are unchanged;
use YAML or API metadata for startup measurements. The benchmark log labels
the Docker-start TTR/TTFT separately from endpoint waits.

## Comparison and qualification

These are first-observed event timestamps, not kernel event tracing. The
observer begins when the post-launch readiness wait starts, and can therefore
observe a fast/already-ready endpoint late. The observation retains
`observer_started_unix_ns` to expose that delay. Port/HTTP readiness may precede
first inference by much more than a second.

For controlled qualification, `run_probe` can take an `expected` final response.
It still issues one inference request and timestamps first text, but then
requires the complete stream, finish reason, and exact final content before
returning `response_validated: true`. ColdSnap's maintained harness uses this
mode for normal-launch controls and reads ColdSnap acceptance for restore
cases. Match prompts, token limits, topology, engine, and cache policy. Keep
profile labels; do not mix rank-local samples with historical control-node
observer measurements. This feature is measurement infrastructure, not new
GPU qualification or published performance data.
