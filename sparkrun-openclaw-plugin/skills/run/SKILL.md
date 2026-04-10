---
name: run
description: "ALWAYS invoke this skill before running any sparkrun CLI commands. Never run sparkrun directly without loading this skill first. Covers launching, monitoring, stopping, and checking status of inference workloads on NVIDIA DGX Spark."
---

<Purpose>
Provides complete reference for launching, monitoring, and stopping LLM inference workloads using sparkrun on NVIDIA DGX Spark systems. Covers the full lifecycle: browse recipes, check VRAM fit, launch jobs, view logs, check status, stop workloads, run benchmarks, tune kernels, and manage the inference proxy.
</Purpose>

<Use_When>
- User wants to run an LLM inference model on DGX Spark
- User wants to check status of running workloads
- User wants to stop a running inference job
- User wants to view logs from a running workload
- User wants to preview VRAM requirements before launching
- User wants to benchmark an inference workload
- User wants to tune MoE kernels for better performance
- User wants to manage the inference proxy
- User wants to monitor cluster metrics
- User asks "how do I run", "start", "launch", "deploy" a model
</Use_When>

<Do_Not_Use_When>
- User wants to install sparkrun or set up a cluster -- use the setup skill instead
- User wants to manage recipe registries or create custom recipes -- use the registry skill instead
- User is asking about sparkrun internals or development
</Do_Not_Use_When>

<Steps>

## Run a Recipe

```bash
# Single host
sparkrun run <recipe> --tp 1 --no-follow
sparkrun run <recipe> --hosts <ip> --no-follow

# Multi-node cluster
sparkrun run <recipe> --cluster <name> --no-follow
sparkrun run <recipe> --hosts <ip1>,<ip2>,... --no-follow
sparkrun run <recipe> --tp <N> --no-follow

# With overrides
sparkrun run <recipe> --port 9000 --gpu-mem 0.8 --no-follow
sparkrun run <recipe> -o max_model_len=8192 -o attention_backend=triton --no-follow
sparkrun run <recipe> --served-model-name my-model --no-follow
sparkrun run <recipe> --pp 2 --tp 2 --no-follow   # pipeline + tensor parallelism
sparkrun run <recipe> --max-model-len 32768 --no-follow

# Idempotent launch (exit 0 if already running)
sparkrun run <recipe> --ensure --no-follow

# With Docker options
sparkrun run <recipe> --restart unless-stopped --no-follow
sparkrun run <recipe> --no-rm --no-follow   # keep containers after stop
sparkrun run <recipe> --transfer-mode push --no-follow  # force push-based distribution

# Dry-run (show what would happen)
sparkrun run <recipe> --dry-run
```

**CRITICAL: Always use `--no-follow`** when running from an agent/skill context to avoid blocking on log streaming. Then use `sparkrun cluster status` or `sparkrun logs` separately to check on the job.

## Check Status

```bash
# Show all sparkrun containers across cluster hosts
# NOTE: `sparkrun status` is an alias for `sparkrun cluster status` — only run one.
sparkrun cluster status
sparkrun cluster status --cluster <name>
sparkrun cluster status --hosts <ip1>,<ip2>,...

# Output shows:
#   - Grouped containers by job (with recipe name if cached)
#   - Container role, host, status, and image
#   - Ready-to-use logs and stop commands for each job
#   - Pending operations (downloads/distributions in progress)
#   - Idle hosts

# Check if a specific job is running (scripting-friendly)
sparkrun cluster check-job <recipe> --cluster <name>
sparkrun cluster check-job <cluster_id> --hosts <ip1>,<ip2>
sparkrun cluster check-job <recipe> --check-http-models   # also verify /v1/models endpoint
sparkrun cluster check-job <recipe> --json                 # JSON output
# Exit code: 0 = running, 1 = not running
```

## Monitor Cluster Metrics

```bash
# Plain-text output (agent-friendly)
sparkrun cluster monitor --cluster <name> --simple

# JSON streaming (automation-friendly)
sparkrun cluster monitor --cluster <name> --json

# Custom interval
sparkrun cluster monitor --cluster <name> --interval 5
```

## View Logs

```bash
# By recipe name
sparkrun logs <recipe> --cluster <name>
sparkrun logs <recipe> --hosts <ip1>,<ip2>,...
sparkrun logs <recipe> --tp <N>

# By cluster ID (from sparkrun status output)
sparkrun logs <cluster_id>
sparkrun logs <cluster_id> --cluster <name>

# Control log tail length
sparkrun logs <recipe> --tail 200
```

## Stop a Workload

```bash
# By recipe name
sparkrun stop <recipe> --cluster <name>
sparkrun stop <recipe> --hosts <ip1>,<ip2>,...
sparkrun stop <recipe> --tp <N>

# By cluster ID (from sparkrun status output)
sparkrun stop <cluster_id>
sparkrun stop <cluster_id> --cluster <name>

# Stop all sparkrun containers (no recipe needed)
sparkrun stop --all --cluster <name>
sparkrun stop --all --hosts <ip1>,<ip2>,...

# Dry-run
sparkrun stop <recipe> --dry-run
```

## Browse and Inspect Recipes

```bash
# List all available recipes (no filter)
sparkrun list

# List with filters
sparkrun list --all                         # include hidden registry recipes
sparkrun list --registry <name>             # filter by registry
sparkrun list --runtime vllm                # filter by runtime
sparkrun list <query>                       # filter by name

# Search for recipes by name, model, runtime, or description (contains-match)
sparkrun recipe search <query>
sparkrun recipe search <query> --registry <name> --runtime sglang

# Inspect a specific known recipe (by exact name or file path)
sparkrun recipe show <recipe>
sparkrun recipe show <recipe> --tp <N>

# Validate or estimate VRAM
sparkrun recipe validate <recipe>
sparkrun recipe vram <recipe> --tp <N> --max-model-len 32768

# Export a recipe
sparkrun recipe export <recipe>
sparkrun recipe export <recipe> --json
sparkrun recipe export <recipe> --save out.yaml
```

Use `sparkrun recipe search` as the first attempt when looking for a particular recipe. Use `sparkrun recipe show` when given a specific recipe name or file -- it may not appear in search results.

## Benchmark

```bash
# Full flow: launch inference -> benchmark -> stop
sparkrun benchmark <recipe> --tp 1
sparkrun benchmark <recipe> --cluster <name>
sparkrun benchmark <recipe> --tp 2 --profile <profile_name>

# Benchmark an already-running instance (skip launch)
sparkrun benchmark <recipe> --skip-run --tp 1

# Keep inference running after benchmark completes
sparkrun benchmark <recipe> --no-stop --tp 1

# Override benchmark args (use -b for benchmark args, -o for recipe overrides)
sparkrun benchmark <recipe> -b depth=0,2048,4096 -b tg=32,128

# Specify framework and timeout
sparkrun benchmark <recipe> --framework llama-benchy --timeout 3600

# Dry-run
sparkrun benchmark <recipe> --dry-run
```

## Kernel Tuning

```bash
# Tune SGLang fused MoE kernels
sparkrun tune sglang <recipe> --hosts <ip>
sparkrun tune sglang <recipe> --cluster <name> --tp 1 --tp 2 --tp 4
sparkrun tune sglang <recipe> -H <ip> --parallel 2

# Tune vLLM fused MoE kernels
sparkrun tune vllm <recipe> --hosts <ip>
sparkrun tune vllm <recipe> --cluster <name> --tp 4
```

## Inference Proxy

```bash
# Start the unified OpenAI-compatible proxy
sparkrun proxy start --cluster <name>
sparkrun proxy start --port 4000

# Check proxy status and registered models
sparkrun proxy status
sparkrun proxy models
sparkrun proxy models --refresh

# Load/unload models through the proxy
sparkrun proxy load <recipe> --cluster <name>
sparkrun proxy unload <recipe> --cluster <name>

# Manage model aliases
sparkrun proxy alias add my-model "Qwen/Qwen3-1.7B"
sparkrun proxy alias remove my-model
sparkrun proxy alias list

# Stop the proxy
sparkrun proxy stop
```

</Steps>

<Tool_Usage>
Use the `sparkrun_exec` tool for all sparkrun commands. This tool accepts the full sparkrun CLI command as a string.

When running workloads:
1. Always use `--no-follow` flag with `sparkrun run`
2. After launching, run `sparkrun cluster status` to confirm containers are running
3. Use the logs/stop commands from status output to manage jobs
4. For monitoring, use `--simple` or `--json` mode (TUI requires interactive terminal)
</Tool_Usage>

<Key_Options>

**`sparkrun run` options:**

| Option | Description |
|--------|-------------|
| `--hosts, -H` | Comma-separated host list |
| `--hosts-file` | File with hosts (one per line) |
| `--cluster` | Use a saved cluster |
| `--tp, --tensor-parallel` | Override tensor parallelism (= node count) |
| `--pp, --pipeline-parallel` | Override pipeline parallelism |
| `--port` | Override serve port |
| `--gpu-mem` | GPU memory utilization (0.0-1.0) |
| `--max-model-len` | Override maximum model context length |
| `--served-model-name` | Override the served model name |
| `--image` | Override container image |
| `-o KEY=VALUE` | Override any recipe default |
| `--ensure` | Only launch if not already running; exit 0 if already up |
| `--no-rm` | Don't auto-remove containers on exit |
| `--restart POLICY` | Docker restart policy (no, always, unless-stopped, on-failure[:N]) |
| `--transfer-mode` | Resource transfer mode (auto, local, push, delegated) |
| `--dry-run, -n` | Show what would be done |
| `--no-follow` | Don't attach to logs after launch |
| `--foreground` | Run in foreground (blocking) |

</Key_Options>

<Important_Notes>
- **Always use `--no-follow`** when running from an automated/agent context to avoid blocking
- Use `sparkrun cluster status` after launching to confirm containers are running
- `--tp N` must match the number of hosts (DGX Spark = 1 GPU per host)
- `sparkrun stop` and `sparkrun logs` accept both recipe names and cluster IDs as targets
- If `--tp`, `--port`, or `--served-model-name` were used during `run`, pass the same values to `stop` and `logs`
- Use `sparkrun show <recipe> --tp N` to preview VRAM estimates before running
- Container names follow the pattern `sparkrun_{hash}_{role}` where the hash is derived from runtime + model + sorted hosts + overrides
- Ctrl+C while following logs detaches safely -- it never kills the inference job
- Use `sparkrun stop --all` to stop all sparkrun containers without specifying a recipe
- `--solo` is deprecated; use `--tp 1` instead
- Recipe names support `@registry/name` syntax for explicit registry selection
- `sparkrun update` upgrades sparkrun itself (if installed via uv) and updates all registries
</Important_Notes>

Task: {{ARGUMENTS}}
