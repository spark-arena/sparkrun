---
name: workloads
description: Launch, inspect, log, stop, benchmark, and manage inference workloads with Sparkrun 0.4.0, including gateways and optional daemon clients.
---

# Workloads and benchmarks

This reference targets Sparkrun 0.4.0. Check `sparkrun --version` and installed
command `--help` when capabilities differ. Use configuration for installation, hosts,
plugins and features; use registry for recipe authoring. Replace placeholders
with actual values before executing examples.

## Launch and readiness

```bash
sparkrun recipe search <query>
sparkrun show <recipe> --tp 2
sparkrun run <recipe> --cluster <name> --dry-run
sparkrun run <recipe> --cluster <name> --no-follow
sparkrun run <recipe> --hosts <host1>,<host2> --tp 2 --no-follow
sparkrun run <recipe> --tp 1 --port 9000 --gpu-mem 0.8 --max-model-len 32768 --no-follow
sparkrun run <recipe> -o max_model_len=8192 --served-model-name my-model --no-follow
sparkrun run <recipe> --ensure --no-follow
```

Normal `run` follows startup logs and returns when readiness completes.
`--no-follow` suppresses model logs **but still waits for readiness**; use it
when the tool budget allows model startup. `--ensure` reuses a matching workload
and still waits. Downloads and launch can take much longer than two minutes.

```bash
sparkrun run <recipe> --cluster <name> --no-ready-wait
sparkrun cluster check-job <workload_id> --check-http-models --json
```

`--no-ready-wait` skips the final readiness wait, not preparation or launch, and
cannot be combined with readiness-dependent post-launch hooks. Report it as
launched, not ready. Container presence or `/v1/models` success alone is not the
full inference readiness probe. `--follow` stays attached after readiness;
reserve it and `--foreground` for interactive or managed background sessions.
Ctrl+C while watching logs detaches without stopping the workload.

Placement considers the cluster, occupancy, and parallelism. On one-GPU DGX
Sparks, TP × PP generally determines the required GPUs/hosts. Do not assume every
configured host is used or the first is always selected as head. Use reported
placement. VRAM estimates can be unverified/lower bounds, not a guarantee of fit.
`--solo` remains supported; prefer `--tp 1` for a single GPU.

## Status, bounded logs, and stop

```bash
sparkrun status --cluster <name>
sparkrun cluster check-job <workload_id> --json
sparkrun logs <workload_id> --tail 200
sparkrun logs <workload_id> --all-sources --tail 200
sparkrun stop <workload_id> --dry-run
sparkrun stop <workload_id>
sparkrun stop --all --cluster <name>
```

`status` aliases `cluster status`; run one. Prefer the exact commands and workload
ID printed by status (`workload_id` in JSON output); it identifies a workload,
not a saved host cluster. Recipe targets also work:
preserve launch overrides (`--tp`, `--port`, `--served-model-name`) or use the
recorded ID. Do not invent IDs/container names from a presumed hash format.

Bare `logs` reads existing logs and exits; `--tail 200` bounds output. The default
source is the head; `--all-sources` includes workers/native executor sources.
`--follow` streams indefinitely. Python/daemon host/container/source filters are
not additional CLI flags. `check-job` exits 0 for running, 1 for not running;
inspect errors rather than treating every failure as absent. Only use `stop --all`
when the user authorized stopping that scope.

## Monitoring

```bash
sparkrun cluster monitor --cluster <name> --simple --interval 5
sparkrun cluster monitor --cluster <name> --json --interval 5
```

Both modes stream until interrupted; they are not one-shot. Use a cancellable
background session or a deliberate short tool timeout and stop the local monitor.
The default TUI needs an interactive terminal. Use status for a finite presence check.

## Benchmark categories and Arena

```bash
sparkrun benchmark <recipe> --cluster <name>
sparkrun benchmark perf <recipe> --profile @official/spark-arena-v2
sparkrun benchmark tools <recipe> --skip-run
sparkrun benchmark perf <recipe> -b depth=0,2048,4096 -b tg=32,128
sparkrun benchmark tools <recipe> -b parallel=2 -b max_turns=8
sparkrun benchmark perf <recipe> --no-stop --timeout 14400
sparkrun benchmark perf <recipe> --resume
sparkrun benchmark resume <benchmark_id>
sparkrun benchmark <recipe> --arena
sparkrun benchmark <recipe> --arena --local-test
```

Bare `benchmark` defaults to performance (`perf`/`performance`, usually
llama-benchy). `tools` uses tool-eval-bench. Discover profiles with
`registry list-benchmark-profiles`. Use framework-specific `-b KEY=VALUE`, and
`-o KEY=VALUE` for recipe overrides. Use bundled framework defaults unless the user
requests a pin; do not force a stale tool-eval-bench ref.

Benchmark cleans up inference it launches unless `--no-stop` is set. `--skip-run`
measures an existing server without taking ownership. `--resume` reuses matching
saved progress; `--fresh` replaces it. Saved benchmark IDs support resume and
publication retry. Allow sufficient time in the calling tool as well as the
benchmark's `--timeout`. Report actual result paths and failed/partial cases.

Arena upload is opt-in via **`benchmark --arena`**, after `sparkrun arena login`.
An Arena profile alone does not upload. `--local-test` requires `--arena` and skips
login/upload. `arena benchmark` is an advanced hidden alias; prefer the main
benchmark command. Login requires an interactive user session.

## Diagnostics

```bash
sparkrun run <recipe> --no-follow --collect-diagnostics run.ndjson
sparkrun setup diagnose --cluster <name> --output diagnostics.ndjson
```

Run capture includes DEBUG logs, structured plugin feedback, and failure log tails
without changing console verbosity. No `-vvv` is needed. Use a new output path;
bundles do not overwrite. Review before sharing: redaction is best-effort, and
hostnames/operational details remain. This capture CLI flag belongs to `run`,
not `benchmark`; Python/daemon clients can capture benchmarks via the shared API.

## Preparation, offline use, and tuning

```bash
sparkrun run <recipe> --cluster <name> --offline --no-follow
sparkrun tune sglang <recipe> --hosts <host1>
sparkrun tune vllm <recipe> --cluster <name> --tp 1
```

`--offline` requires usable cached images/models; it does not mean skip validation.
Use the setup reference for cache inspection/repair. Advanced `build` prepares
images/environments and assets without launching inference; inspect its help for
the selected executor. Kernel tuning requires the `cli.tune` feature and consumes
host resources; only run it when requested, using the matching runtime. Saved
tuning configs normally sync during launch; `run --no-sync-tuning` opts out.

## Gateways

LiteLLM is bundled and defaults on **all channels**. SparkRoute is an external
plugin. `gateway` controls shared support; `gateway.<provider>` controls each
implementation. Explicit feature enabling selects it for subsequent starts;
plugin installation/enabling alone does not switch the selection.

```bash
sparkrun plugins list --all
sparkrun setup features list --all
sparkrun setup features enable gateway.litellm
sparkrun proxy start --host 127.0.0.1 --cluster <name>
sparkrun proxy status --json
sparkrun proxy models
sparkrun proxy models --refresh
sparkrun proxy load <recipe> --cluster <name>
sparkrun proxy unload <recipe> --cluster <name>
sparkrun proxy alias add my-model org/model-name
sparkrun proxy alias list
sparkrun proxy alias remove my-model
sparkrun proxy sync
sparkrun proxy stop
```

Use `proxy start --restart` to apply a switch to a running gateway. A missing or
disabled selected provider is an error, not a silent fallback. Model queries use
gateway-provided credentials (including LiteLLM's `LITELLM_MASTER_KEY`); do not
invent `proxy models --master-key` or expose secrets in transcripts. Admin UI/token
support is provider-specific. Linux service management uses `proxy systemd`;
sudo prompts require an interactive terminal.

## Optional daemon clients

Ordinary CLI commands run directly even when a daemon is active. Use daemon
clients for explicitly requested API workflows/debugging, not as a prerequisite
to ordinary CLI use. MCP is disabled.

```bash
sparkrun daemon status
sparkrun daemon jobs list
sparkrun daemon jobs show <job_id>
sparkrun daemon --url https://controller:8443 --token-file <token_file> status
sparkrun daemon --socket <socket_path> jobs list
```

Connection flags precede the daemon subcommand. Inspect `/v1/ops` for actual
operation names/schemas; plugins use `plugins.<plugin_id>.*`. Jobs may need prompt
responses. Events follow streams; `jobs show` is a finite poll. Never print tokens.

<Tool_Usage>
Run these commands with the `bash` tool. Raise `timeout` (up to 3600 seconds) for
downloads, readiness waits, and benchmarks; the default 300-second deadline is too
short for most launches. Start indefinite streams (`logs --follow`, `cluster
monitor --simple`, `cluster monitor --json`) as background jobs (`async: true` or
`timeout: 0`) and stop them when enough samples are collected. A background job or
a timed-out bash call kills only the local CLI process, never the remote workload.
Interactive prompts (`setup wizard`, SSH/sudo, trust confirmations, Arena login)
need the user's own terminal. Never claim a timed-out launch was rolled back.
</Tool_Usage>

References: [readiness](https://github.com/spark-arena/sparkrun/blob/main/docs/STARTUP_READINESS.md),
[gateways](https://github.com/spark-arena/sparkrun/blob/main/docs/PROXY.md),
[daemon API](https://github.com/spark-arena/sparkrun/blob/main/docs/DAEMON_API.md).
