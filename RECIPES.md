# sparkrun Recipe Reference

A recipe is a YAML file defining an inference workload: model, container, runtime, configuration, and lifecycle hooks.

```bash
sparkrun run my-recipe --solo          # use defaults
sparkrun run my-recipe -H host1,host2  # override hosts
sparkrun run my-recipe -o port=9000    # override any default
```

## Minimal Recipe

```yaml
model: Qwen/Qwen3-1.7B
runtime: vllm
container: scitrera/dgx-spark-vllm:latest
defaults:
  port: 8000
  tensor_parallel: 1
```

Everything else is optional. When `command` is omitted, the runtime generates it from `defaults`.

---

## Field Reference

### Core

| Field             | Type   | Required    | Default         | Description                                                       |
|-------------------|--------|-------------|-----------------|-------------------------------------------------------------------|
| `model`           | string | **yes**     | —               | HuggingFace model ID, GGUF spec (`Qwen/Qwen3-1.7B-GGUF:Q4_K_M`), or an absolute path to pre-placed on-disk weights |
| `model_revision`  | string | no          | `null`          | Pin to a specific HF revision (branch, tag, or commit hash)       |
| `runtime`         | string | no          | auto-detected   | Runtime identifier. See [Runtime Resolution](#runtime-resolution) |
| `runtime_version` | string | no          | `""`            | Informational version tag                                         |
| `container`       | string | recommended | runtime default | Container image reference                                         |

GGUF models use colon syntax (`repo:quant`) to download only the matching quantization files. When pre-synced, sparkrun
rewrites `-hf` to `-m` with the resolved container cache path.

**Local model weights:** an *absolute path* in `model:` (e.g. `model: /nfs/models/qwen3`) is treated as pre-placed
on-disk weights that already exist on every node (typically a shared NFS mount). sparkrun then skips model download +
distribution entirely, identity-mounts that directory into the container at the same path, and serves the runtime
directly from it; the served-model name defaults to the directory basename. Only absolute paths trigger this — an
ordinary repo id (`org/name`) or GGUF spec is never mistaken for a path, and `~`/relative paths are not expanded.
Because this bind-mounts a host directory into the container, it is honored only for **trusted** recipes (local,
default-registry, or `--trust`); an untrusted registry/URL recipe with an absolute-path model fails closed. This is the
ergonomic front-end for the lower-level `cluster_config.resolved_model_path` escape hatch.

Since distribution is skipped, the path must genuinely already exist on **every target node** — the control machine
need not be a cluster member and need not see the path itself. Before launch, sparkrun verifies the path on the targets
via the executor (host substrate → SSH `test -e`; provider executors probe their own volumes) and fails fast listing any
node where it's missing, rather than letting the container crash. This preflight is best-effort: an unreachable or
unverifiable node never blocks the launch.

`model_revision` affects download, cache checking, VRAM auto-detection, and model sync. Pin to a commit hash for
reproducible deployments.

### Topology

| Field          | Type                                | Default  | Description                                               |
|----------------|-------------------------------------|----------|-----------------------------------------------------------|
| `mode`         | `"auto"` \| `"solo"` \| `"cluster"` | `"auto"` | Deployment mode. `auto` = decided by host count           |
| `min_nodes`    | int                                 | `1`      | Minimum hosts. `> 1` forces `mode: cluster`               |
| `max_nodes`    | int                                 | `null`   | Maximum hosts. `null` = no limit. `1` forces `mode: solo` |
| `solo_only`    | bool                                | `false`  | Shorthand: `max_nodes: 1, mode: solo`                     |
| `cluster_only` | bool                                | `false`  | Shorthand: `min_nodes: 2, mode: cluster`                  |

Note: `mode`, `solo_only`, and `cluster_only` are deprecated. Developers are encouraged to use  `min_nodes` and
`max_nodes` instead.

On DGX Spark (1 GPU per node), `tensor_parallel: N` = N hosts.

### Configuration

| Field            | Type   | Default | Description                                                                                    |
|------------------|--------|---------|------------------------------------------------------------------------------------------------|
| `defaults`       | map    | `{}`    | Default values for serve flags. CLI overrides take priority                                    |
| `env`            | map    | `{}`    | Container environment variables. Values are passed through **literally** (see note below) |
| `command`        | string | `null`  | Command template. `{key}` placeholders resolved from config chain                              |
| `runtime_config` | map    | `{}`    | Runtime-specific config. Unknown top-level keys are auto-swept here                            |

> **`env` values are not expanded.** Control-machine environment variables are
> no longer substituted into recipe `env`. A `${HF_TOKEN}` in a recipe reaches
> the container as the literal string `${HF_TOKEN}`, not the token. This closed
> an exfiltration path: a third-party recipe could otherwise write
> `env: {LEAK: "${AWS_SECRET_ACCESS_KEY}"}` and have the control machine's
> secrets substituted into a container it controls.
>
> To forward a secret, use a **cluster-level** `env` block with an `env_file`
> (`core/cluster_manager.py:ClusterDefinition.resolve_env`) — `${VAR}`
> references there resolve at launch time from that file only, never from
> `os.environ`, so secrets stay out of both the recipe and the cluster YAML.

**Container env tiers** (highest priority first):

1. Recipe `env` — including both CLI forms, which are written into it:
   - `-e KEY=VALUE` / `--env KEY=VALUE` (repeatable) — value used **verbatim**.
     Hidden from `--help` unless `SPARKRUN_ADVANCED=1`.
   - `-o env.KEY=VALUE` — the generic override path, so the value goes through
     `coerce_value`: `-o env.X=true` arrives as `True`, not `true`. Prefer `-e`
     for anything a runtime parses itself.

   Both split on the **first** `=` only (`-e ALLOC=expandable_segments:True` is
   safe), and `-e KEY=` sets an empty value — the way to blank out a platform or
   cluster default. `-e` wins if a key is given both ways.
2. Cluster `env` / `env_file`.
3. **Platform** — `HardwarePlatformPlugin.default_env(runtime_name, accelerator,
   runtime_family=…)`, targeted at platform × runtime × accelerator. Today
   DGX Spark (GB10) sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for
   the `vllm` and `sglang` families, which reduces allocator fragmentation on
   GB10's unified memory. Runtimes that manage their own memory (llama.cpp,
   TRT-LLM) are untargeted. Because this is a *tier* rather than a default
   written into the recipe, you can override it with any value — including an
   empty one (`-o env.PYTORCH_CUDA_ALLOC_CONF=`).

Below all of these sit the runtime's own `get_common_env()` / cluster env.

### Metadata

Informational fields for VRAM estimation, display, and search. Not passed to runtime.

```yaml
metadata:
  description: "Qwen3 8B — general purpose"
  maintainer: "you <you@example.com>"
  category: agent
  model_params: 8B           # or 8000000000
  model_dtype: bfloat16      # float32, float16, bfloat16, int8, fp8, int4, awq4, gptq, nvfp4, q4_k_m, q8_0, ...
  kv_dtype: fp8_e5m2         # KV cache dtype (default: bfloat16); also accepts nvfp4_ds_mla / fp8_ds_mla
  num_layers: 32
  num_kv_heads: 8
  head_dim: 128
  model_vram: 16.5           # GB override — skips param-based calculation
  kv_vram_per_token: 0.0001  # GB/token override — skips architecture-based calculation
  quantization: awq          # Quantization method: awq, gptq, fp8, nvfp4, mxfp4, auto-round, bitsandbytes, compressed-tensors, none
  quant_bits: 4              # Quantization bit width (4, 8)
```

sparkrun auto-detects `model_params`, `model_dtype`, `num_layers`, `num_kv_heads`, `head_dim`, `quantization`, and
`quant_bits` from HuggingFace Hub config when not provided. Both `config.json` (`quantization_config` block) and
`hf_quant_config.json` (modelopt supplement, e.g. NVIDIA NVFP4 models) are checked. When `hf_quant_config.json`
contains `kv_cache_quant_algo`, it is used to set `kv_dtype` if not already specified. Metadata values always take
precedence over auto-detected values.

The `kv_cache_dtype` is resolved in priority order: CLI override → `metadata.kv_dtype` → HuggingFace quant config →
`defaults.kv_cache_dtype` → `--kv-cache-dtype` parsed from the `command:` template (last resort, with a warning).
When the dtype is resolved from the command template, it is written back to `metadata.kv_dtype` so subsequent
estimates are stable. When no dtype is resolved at all, the estimator defaults to `bfloat16` for computation but
leaves `VRAMEstimate.kv_dtype` as `None` so the CLI can display `bfloat16 (default)` rather than silently reporting
a guessed value.

#### Multi-head Latent Attention (DeepSeek)

MLA models cache one *compressed latent* per token per layer instead of a K and V entry per attention head, so the
generic `2 * num_layers * num_kv_heads * head_dim * bytes` sizing overestimates their KV cache by one to two orders of
magnitude. sparkrun detects MLA from the HuggingFace config (`qk_rope_head_dim`, plus `kv_lora_rank` on V2/V3 or
`head_dim` on V4) and sizes the latent cache instead.

MLA is one *KV cache strategy* among several; each one declares the `metadata` fields it reads, so the table below is
generated by the strategy rather than by the estimator. Auto-detected fields, all overridable in `metadata`:

| Field               | Meaning                                                                                  |
|---------------------|------------------------------------------------------------------------------------------|
| `kv_lora_rank`      | Compressed-latent width, **excluding** the RoPE tail. Its presence switches KV sizing to the MLA path. |
| `qk_rope_head_dim`  | RoPE tail cached alongside the latent.                                                    |
| `compress_ratios`   | DeepSeek V4 per-layer cache compression; layers at ratio ≤ 1 are sliding-window layers.   |
| `model_type`        | Selects the packed slot layout (e.g. `deepseek_v4`).                                      |
| `index_head_dim`    | Sparse-attention indexer width. Not sized; its presence marks the estimate as a floor.     |

The two generations spell the latent differently, and sparkrun normalizes them: V2/V3 name `kv_lora_rank` and cache
the RoPE tail *in addition* to it (512 + 64 = 576 elements), while V4 has no `kv_lora_rank` and folds both into
`head_dim` (512 total, of which 64 is the tail). Auto-detection reports the **NoPE width** for both — 512 for V3, 448
for V4 — so the tail is added exactly once. Pin `kv_lora_rank` by hand only if you are giving the NoPE width.

Runtimes that pack the latent, its block scales and the RoPE tail into a fixed-width uint8 slot are named through
`kv_dtype` / `defaults.kv_cache_dtype`: `fp8_ds_mla` and `nvfp4_ds_mla` (656 bytes per token per layer; 584 on
DeepSeek V4). Naming one of these is on its own enough to select MLA sizing.

Two consequences worth knowing when reading an estimate:

- **The latent cache is replicated on every tensor-parallel rank** — it has no head dimension to shard — so raising
  `--tp` does not shrink it. Pipeline parallelism still splits it by layer.
- For sparse-attention models the estimate covers the **latent cache only**, so treat it as a floor. DeepSeek V4
  also keeps a sliding-window cache and V3.2 a sparse-indexer cache (roughly 132 bytes per token per layer); neither
  is sized. Whichever apply are named in a warning on the estimate.

Use `kv_vram_per_token` to override the whole calculation if a runtime's real footprint differs. Note it follows
the same sharding rule: on an MLA model the override is divided by pipeline parallelism only, not by `--tp`, so
the figure you supply is one rank's replicated latent cache rather than a cluster-wide total.

> **Pinning architecture metadata on an MLA model.** Pinning `num_layers`, `num_kv_heads` and `head_dim` *together
> with* `model_dtype` (or `model_vram`) suppresses the HuggingFace config fetch — all of them are needed; pinning the
> three architecture fields alone still leaves `model_dtype` missing, which keeps detection running. Once the fetch is
> suppressed, MLA can no longer be auto-detected and the estimate falls back to the generic formula: for DeepSeek-V3,
> 122 GB instead of 2.1 GB at 32k context. Pin `qk_rope_head_dim` (and `kv_lora_rank`, unless `head_dim` already holds
> the whole cached width) alongside them, or name an `*_ds_mla` layout in `kv_dtype`; any one of those signals is
> enough. Pinning `qk_rope_head_dim` on its own is accepted but warns, since it cannot distinguish a compressed-latent
> cache from an ordinary one.

### Benchmark

```yaml
benchmark:
  category: performance       # optional; derived from the framework's primary
                              # category when omitted (perf for llama-benchy,
                              # tools for tool-eval-bench, ...).
  framework: llama-benchy     # default framework for the category
  timeout: 3600
  args: # or put args at top level (auto-swept)
    pp: [ 2048 ]
    tg: [ 32, 128 ]
    concurrency: [ 1, 2, 5 ]
```

Used by `sparkrun benchmark <recipe>`. CLI `-o` overrides apply on top.

**Category subcommands.** `sparkrun benchmark` accepts a category positional
that pins the kind of benchmark (and the default framework for it):

```
sparkrun benchmark performance <recipe>     # alias: sparkrun benchmark perf
sparkrun benchmark tools       <recipe>
sparkrun benchmark <recipe>                  # back-compat: == performance
sparkrun benchmark run <recipe>              # legacy entry; no category
```

Pinning a category and an incompatible `--framework` raises
`FrameworkCategoryMismatch`. New categories appear automatically once a
plugin registers them (`BenchmarkingPlugin.categories`).

**Resume / fresh.** Resumable runs (frameworks that implement
`build_task_list`) write progress state to
`~/.cache/sparkrun/benchmarks/<benchmark_id>/`. CLI flags:

```
--resume   # non-interactive: resume if compatible state exists, else fresh
--fresh    # delete prior state and start over (mutually exclusive with --resume)
```

When neither flag is set and stdin is a TTY, the CLI prompts. Non-TTY
defaults to resume. The library API (`sparkrun.api.benchmark`) exposes the
full `ResumeMode` enum (`AUTO`, `IF_EXISTS`, `FRESH`, `REQUIRED`); pass
`on_prompt_required=...` to inject a callback in lieu of the prompt.

**Container image pinning.** On the first successful launch of a resumable
run, sparkrun captures two distinct references and persists them in
`state.extras`:

- `container_image_sha` — content-addressable image ID resolved via
  `docker image inspect` on a target host. On resume the orchestration
  overrides `overrides["image"]` with this SHA so a re-pushed tag or rebuilt
  local image cannot silently change the bits between sessions.
- `container_image_longterm_ref` — output of the builder's
  `resolve_long_term_image()`. Used only for archival provenance in the
  result YAML; it is not used at launch time. Persisted so resumed sessions
  emit identical archive references.

**Spark Arena.** `--arena` on any category subcommand runs the opinionated
Spark Arena flow (auth check, hardcoded profile `@official/spark-arena-v2`,
post-run upload). `sparkrun arena benchmark` continues to work as a sibling
entry point that calls into the same shared helpers (`preflight_arena` and
`finalize_arena`).

### Version

| Field            | Type           | Default | Description                                                     |
|------------------|----------------|---------|-----------------------------------------------------------------|
| `recipe_version` | `"2"` \| `"1"` | `"2"`   | v1 = legacy eugr format, auto-detected from `build_args`/`mods` |

---

## Config Chain

Resolution order (highest priority first):

```
CLI overrides  →  recipe defaults  →  runtime defaults
```

```bash
sparkrun run my-recipe -o max_model_len=8192 --port 9000
```

1. CLI: `{port: 9000, max_model_len: 8192}`
2. Recipe defaults: `{port: 8000, tensor_parallel: 2, gpu_memory_utilization: 0.9}`
3. Result: `{port: 9000, max_model_len: 8192, tensor_parallel: 2, gpu_memory_utilization: 0.9}`

`{model}` is always injected from the top-level `model` field. Substitution is iterative (handles nested references like
`base_url: "http://localhost:{port}"`).

---

## Command Templates

When `command` is set, sparkrun renders `{key}` placeholders from the config chain. When omitted, the runtime builds the
command from structured `defaults`.

```yaml
command: |
  vllm serve \
      {model} \
      --port {port} \
      -tp {tensor_parallel} \
      --gpu-memory-utilization {gpu_memory_utilization}
```

- Unresolved placeholders are left as-is
- Trailing spaces after backslash continuations (`\ \n`) are auto-fixed
- The runtime may auto-append flags (e.g. `--served-model-name`) if the template omits them

### Literal braces (JSON-valued flags)

**Write literal braces plainly. No escaping.** A `{...}` span containing another brace can't be a placeholder, so JSON
passes through untouched — and a placeholder nested inside it still resolves:

```yaml
command: |
  vllm serve {model} \
      --speculative-config '{"method":"mtp","num_speculative_tokens":{num_speculative_tokens}}' \
      --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
```

This applies to `defaults` values as well as to the command template.

> **Deprecated: doubled braces.** v1 (eugr) recipes escape a literal brace by doubling it (`'{{"a":1}}'`). sparkrun
> still honors that — a doubled brace collapses to one, so a v1 command pasted into a v2 recipe works rather than
> shipping `{{...}}` to the runtime — but it logs a deprecation warning outside v1 and **will not be supported in v3**.
> Do not write new recipes this way. The convention is detected from `{{` and then applies to the whole template, so a
> template mixing both spellings collapses the `}}` that merely closes a nested plain-JSON object.

Lifecycle hooks (`pre_exec` / `post_exec` / `post_commands`) render `{key}` and pass plain JSON through identically,
but never collapse `{{` — a hook body is shell, where a doubled brace may be meant literally.

**When to use templates:** full control over flags, ordering, runtime-specific features not in the flag map. **When to
omit:** standard configs where the runtime's `generate_command()` is sufficient.

---

## Runtime Resolution

When `runtime` is empty or `"vllm"`:

| Condition                                                             | Resolved Runtime   |
|-----------------------------------------------------------------------|--------------------|
| `recipe_version: "1"` or `build_args`/`mods` present (deprecated)      | `eugr-vllm`        |
| `distributed_executor_backend: ray` in defaults or command            | `vllm-ray`         |
| Bare `vllm` or empty                                                  | `vllm-distributed` |
| Command starts with `sglang serve` / `python -m sglang.launch_server` | `sglang`           |
| Command starts with `llama-server`                                    | `llama-cpp`        |
| Command starts with `trtllm-serve` / `mpirun...trtllm`                | `trtllm`           |

Explicit `runtime` always wins. Command-hint detection only fires when `runtime` is omitted.

### Available Runtimes

| Runtime            | Clustering                                          | Strategy                                       |
|--------------------|-----------------------------------------------------|------------------------------------------------|
| `vllm-distributed` | Native (`--nnodes`, `--node-rank`, `--master-addr`) | Each node runs `vllm serve`                    |
| `vllm-ray`         | Ray head/worker                                     | Ray cluster, exec serve on head                |
| `sglang`           | Native (`--dist-init-addr`, `--node-rank`)          | Each node runs `sglang.launch_server`          |
| `llama-cpp`        | RPC (experimental)                                  | Workers run `rpc-server`, head uses `--rpc`    |
| `trtllm`           | MPI (`mpirun` + rsh wrapper)                        | `sleep infinity` containers + `mpirun` on head |
| `eugr-vllm`        | Ray (inherits vllm-ray)                             | eugr container builds + Ray cluster            |
| `atlas`            | Native (`--rank`, `--world-size`, `--master-addr`)  | Atlas Spark (azeezish/atlas-gb10) — pure-Rust LLM inference; each rank runs `atlas serve`, rank 0 only HTTP |
| `modular-max`      | None — single-node only                             | Modular MAX (`max serve`); tensor parallelism uses local GPUs via `--devices` (never multi-node) |

### Common Defaults Keys

| Key                      | vLLM                       | SGLang                  | llama.cpp            | TRT-LLM               | Atlas                 | Description                       |
|--------------------------|----------------------------|-------------------------|----------------------|-----------------------|-----------------------|-----------------------------------|
| `port`                   | `--port`                   | `--port`                | `--port`             | `--port`              | `--port`              | Serve port                        |
| `host`                   | `--host`                   | `--host`                | `--host`             | `--host`              | `--host`              | Bind address                      |
| `tensor_parallel`        | `-tp`                      | `--tp-size`             | `--split-mode row`   | `--tp_size`           | `--tp-size`           | TP degree (= node count on Spark) |
| `pipeline_parallel`      | `-pp`                      | `--pp-size`             | `--split-mode layer` | `--pp_size`           | —                     | PP degree                         |
| `ep_size`                | —                          | `--ep-size`             | —                    | —                     | `--ep-size`           | Expert-parallel degree            |
| `gpu_memory_utilization` | `--gpu-memory-utilization` | `--mem-fraction-static` | —                    | —                     | `--gpu-memory-utilization` | GPU memory fraction          |
| `max_model_len`          | `--max-model-len`          | `--context-length`      | `--ctx-size`         | `--max_seq_len`       | `--max-seq-len`       | Max sequence length               |
| `served_model_name`      | `--served-model-name`      | `--served-model-name`   | `--alias`            | —                     | `--model-name`        | Model name in API                 |
| `dtype`                  | `--dtype`                  | `--dtype`               | —                    | —                     | (auto)                | Model dtype                       |
| `quantization`           | `--quantization`           | `--quantization`        | —                    | —                     | (auto)                | Quantization method               |
| `trust_remote_code`      | `--trust-remote-code`      | `--trust-remote-code`   | —                    | `--trust_remote_code` | (no-op)               | Allow remote code                 |
| `kv_cache_dtype`         | `--kv-cache-dtype`         | `--kv-cache-dtype`      | —                    | via extra config      | `--kv-cache-dtype`    | KV cache dtype                    |

Any key can appear in `defaults` — there is no fixed schema. Runtime-specific keys (e.g. `tool_call_parser`, `ctx_size`,
`n_gpu_layers`, `reasoning_parser`) are passed through to command template substitution.

#### Atlas high-speed swap

Atlas's block-level KV streaming (`--high-speed-swap`) is enabled with the boolean `high_speed_swap` key, but it
also needs tuning keys to be usable — Atlas requires `--high-speed-swap-dir` whenever the toggle is set. These keys
map to the corresponding `--high-speed-swap-*` flags:

```yaml
defaults:
  high_speed_swap: true
  high_speed_swap_dir: /mnt/nvme/atlas-hss      # required; must differ from the --swap-space-gb mount
  high_speed_swap_gb: 64                          # total NVMe budget (GiB)
  high_speed_swap_resident_blocks: 512           # HBM scratch slots
  high_speed_swap_rank: 32                        # predictor low-rank dim
  high_speed_swap_qd: 8                           # io_uring submission queue depth
  high_speed_swap_cache_blocks_per_seq: 64       # per-sequence HBM cache cap
```

---

## Runtime Cache

Persists compilation and autotune output — torch.compile / Inductor graphs, Triton cubins,
FlashInfer JIT modules, the TRT-LLM autotuner — on the target hosts so a relaunch skips minutes of
recompilation. Containers are `--rm`, so without this it is all discarded on exit.

**On by default.** A host directory is mounted at `/cache/runtime` (sibling of
`/cache/huggingface`) and the relevant env vars are pointed into it, at the lowest env tier — a
recipe's own `env:` always wins.

```yaml
runtime_cache: false          # shorthand: turn it off for this recipe

runtime_cache:                # ...or set individual knobs
  enabled: true
  dir: /mnt/fast/rc           # host-side root (default: ~/.cache/sparkrun/runtime-cache)
  key_by_model: true          # separate tree per model (default: on)
  key_by_image: false         # separate tree per container image (default: off)
  prune:
    enabled: true
    max_age_days: 30
```

Layered: **recipe `runtime_cache:` → CLI `--runtime-cache` / `--no-runtime-cache` → cluster
`runtime_cache:` → `SparkrunConfig` `runtime_cache:` → runtime defaults → baseline.**
`SPARKRUN_NO_RUNTIME_CACHE=1` is a kill switch that beats every layer.

**About `key_by_image`.** It defaults off because the caches that dominate are content-addressed
internally and are correct — and much more useful — in a tree shared across image versions. Turn it
on for a runtime whose cache does not validate what wrote it. `trtllm` already does this for itself
(its autotuner records the TRT-LLM version and GPU but checks neither on load), so TRT-LLM recipes
need no `runtime_cache:` block to be safe.

Trees are aged by a last-used marker that every launch re-stamps, so a cache you use daily never
ages out. Sweep manually with `sparkrun setup prune-runtime-cache --cluster <name> --dry-run`.

---

## Executor Config

Controls how the workload is launched. Layered:
**CLI flags → recipe `executor` / `executor_config` → cluster `executor` / `executor_config` → executor adjustments → runtime executor-config defaults → `SparkrunConfig` → per-executor defaults → dataclass field defaults.**

### Selecting an executor

The `executor:` field picks which executor backend handles the launch. See
[`docs/EXECUTORS.md`](docs/EXECUTORS.md) for the full reference.

```yaml
executor: docker      # default. Docker-driven (production path).
# executor: local     # experimental. Native subprocess. No container.
# executor: k8s       # experimental draft. kubectl run-driven.
```

Recipes that omit `executor:` fall back to whatever `SparkrunConfig.default_executor` is, then to `docker`.

### Docker executor fields (default)

```yaml
executor_config:
  auto_remove: false
  restart_policy: unless-stopped
  shm_size: 20gb
```

| Key              | Type   | Default     | Docker flag    | Description                                                                                                         |
|------------------|--------|-------------|----------------|---------------------------------------------------------------------------------------------------------------------|
| `auto_remove`    | bool   | `true`      | `--rm`         | Remove container on exit                                                                                            |
| `restart_policy` | string | `null`      | `--restart`    | `always`, `unless-stopped`, `on-failure:N`. **Mutually exclusive with `auto_remove`** — forces `auto_remove: false` |
| `privileged`     | bool   | `true`      | `--privileged` | Privileged mode                                                                                                     |
| `gpus`           | string | `"all"`     | (see below)    | GPU device spec (`all`, `device=0,1`)                                                                               |
| `gpu_access_mode`| string | platform    | `--device` / `--gpus` | How the GPU request is spelled: `cdi` → `--device nvidia.com/gpu=<id>`, `gpus` → `--gpus <gpus>`. Defaults per hardware platform (DGX Spark/GB10 → `gpus`, everything else → `cdi`) |
| `ipc`            | string | `"host"`    | `--ipc`        | IPC namespace                                                                                                       |
| `shm_size`       | string | `"10.24gb"` | `--shm-size`   | Shared memory size                                                                                                  |
| `network`        | string | `"host"`    | `--network`    | Network mode                                                                                                        |
| `entrypoint`     | string | `null`      | `--entrypoint` | Override the image `ENTRYPOINT`. `null` leaves it untouched; `""` clears it so sparkrun's serve command runs directly |
| `user`           | string | `null`      | `--user`       | UID:GID or `$SHELL_USER` (auto: `$(id -u):$(id -g)` + mount passwd/group)                                           |
| `security_opt`   | list   | `null`      | `--security-opt` | Repeated. Defaults to `[no-new-privileges]` in rootless mode.                                                     |
| `cap_add`        | list   | `null`      | `--cap-add`    | Repeated.                                                                                                           |
| `ulimit`         | list   | `null`      | `--ulimit`     | Repeated. Rootless adds `memlock=-1:-1`, `stack=67108864`.                                                           |
| `devices`        | list   | `null`      | `--device`     | Repeated. Rootless adds `/dev/infiniband`.                                                                          |
| `memory_limit`   | string | `null`      | `--memory`     | Container memory cap.                                                                                               |
| `labels`         | list   | `null`      | `--label`      | Repeated.                                                                                                           |

Use `entrypoint: ""` for images that ship an ENTRYPOINT which consumes sparkrun's generated `bash -c <serve command>` tail:

```yaml
executor_config:
  entrypoint: ""
```

Or, for a one-off run without editing the recipe:

```bash
sparkrun run <recipe> -o entrypoint=''
```

**Which images need this.** sparkrun always appends its launcher as CMD *arguments*, so what an ENTRYPOINT does with
them decides whether the workload starts at all — and the two idioms look identical to `docker image inspect`:

| ENTRYPOINT | Behaviour | Needs `entrypoint: ""` |
|---|---|---|
| absent | the appended CMD *is* the argv | no |
| `["/opt/nvidia/nvidia_entrypoint.sh"]` (and most NGC-derived images) | passthrough wrapper — does setup, then `exec "$@"` | **no** — clearing it *skips that setup* |
| `["vllm","serve"]` | consuming — parses sparkrun's `bash -c …` as its own flags | **yes** |

A consuming ENTRYPOINT fails in a way that points nowhere near the cause: the container's program rejects some flag it
mis-parsed out of sparkrun's launcher, and the serve log dies with the container. sparkrun therefore probes the image
during distribution (before the model sync) and refuses the launch with the fix above; see
[docs/EXECUTORS.md](docs/EXECUTORS.md#entrypoint-preflight). Set `SPARKRUN_NO_IMAGE_PROBE=1` to skip the probe.

### LocalExecutor fields (experimental, `executor: local`)

`LocalExecutor` runs the runtime's serve command as a native subprocess on the
target host — there is no Docker container in the loop. **Limitations**: no
image, no volumes, no Ray strategy, GPU visibility only honors `gpus: all` or
`gpus: device=0,2`.

```yaml
executor: local
executor_config:
  working_dir: /opt/myproject
  log_dir: /var/log/sparkrun
  pid_dir: /var/run/sparkrun
  env_file: /etc/sparkrun.env
  command_prefix: nice -n 10
  gpus: "device=0,2"            # → CUDA_VISIBLE_DEVICES=0,2
```

| Key              | Default                                        | Description                                                                       |
|------------------|------------------------------------------------|-----------------------------------------------------------------------------------|
| `working_dir`    | `null`                                         | `cd <working_dir>` before launch.                                                 |
| `log_dir`        | `$HOME/.cache/sparkrun/local/logs`             | Per-container `<log_dir>/<container_name>.log`.                                   |
| `log_file`       | `null`                                         | Overrides `<log_dir>/...` entirely.                                               |
| `pid_dir`        | `$HOME/.cache/sparkrun/local/pids`             | Per-container `<pid_dir>/<container_name>.pid`.                                   |
| `pid_file`       | `null`                                         | Overrides `<pid_dir>/...` entirely.                                               |
| `env_file`       | `null`                                         | Sourced via `set -a; . <env_file>; set +a` before launch.                         |
| `command_prefix` | `null`                                         | Prepended verbatim (e.g. `nice -n 10 ionice -c2`).                                |

### K8sExecutor fields (experimental draft, `executor: k8s`)

`K8sExecutor` launches workloads as Kubernetes Pods via `kubectl run`. **Limitations**:
`kubectl run` (not full manifests) so init containers / sidecars / volume claims
are unreachable; Docker-specific options (`privileged`, `shm_size`, `ipc`,
`network`) are silently dropped; no Ray cluster strategy.

```yaml
executor: k8s
executor_config:
  k8s_namespace: ml-prod
  k8s_context: prod-east
  k8s_node_selector: nodepool=dgx-spark
  k8s_image_pull_policy: IfNotPresent
  kubeconfig: /etc/k8s/admin.conf
  memory_limit: 128Gi
```

| Key                      | Default | Description                                                                                          |
|--------------------------|---------|------------------------------------------------------------------------------------------------------|
| `k8s_namespace`          | `null`  | `kubectl -n <ns>`.                                                                                   |
| `k8s_context`            | `null`  | `kubectl --context <ctx>`.                                                                           |
| `k8s_node_selector`      | `null`  | `key=value[,key=value]`. Emitted as `--overrides` JSON.                                              |
| `k8s_image_pull_policy`  | `null`  | `--image-pull-policy`.                                                                               |
| `kubeconfig`             | `null`  | `--kubeconfig`.                                                                                      |
| `entrypoint`             | `null`  | When set, emits `kubectl run --command`. `""` uses `bash -c <serve command>` to bypass image ENTRYPOINT wrappers. |

**CLI override:** `sparkrun run --no-rm --restart always my-recipe`

All runtimes automatically inherit executor settings — no per-runtime changes
needed.

---

## Lifecycle Hooks

### Execution Order

```
1. builder.prepare_image()          — build/pull container image
2. Distribution                     — sync container image + model to hosts
3. Container launch                 — docker run
4. pre_exec                         — inside ALL containers, before serve
5. Serve command                    — exec inside container (solo) or direct entrypoint (native cluster)
6. [wait for healthy]               — port check + HTTP /v1/models (only when post hooks defined)
7. post_exec                        — inside HEAD container only
8. post_commands                    — on CONTROL MACHINE
9. [stop_after_post]                — optional auto-stop
```

### Trust model

`pre_exec`, `post_exec`, and `post_commands` execute shell commands derived
from the recipe. A recipe is **trusted** (hooks run without prompting) when
any of these hold:

- the user passed `--trust` on the CLI;
- the recipe was loaded from a **local path** (no `source_registry`) and was
  not fetched from a URL;
- the recipe came from a registry whose `trusted` flag is true in the user's
  local `registries.yaml`. Every registry shipped as a built-in default is
  first-party and ships trusted; a registry added with
  `sparkrun registry add <url>` is untrusted until the user opts in with
  `--trust` or `sparkrun registry trust <name>`.

Recipes fetched from a **URL** — including `@spark-arena/<uuid>` shortcuts —
are never auto-trusted, regardless of the above.

Otherwise the user is prompted before each hook surface runs. Trust also gates
`executor` selection and the `executor_config` privilege keys; see
[`docs/SECURITY.md`](docs/SECURITY.md) for the full model.

### pre_exec

Runs inside **every container** before the serve command. Sequential, fail-fast.

```yaml
pre_exec:
  # Shell command: docker exec <container> bash -c '<cmd>'
  - "pip install flash-attn --no-build-isolation"
  - "sed -i 's/old/new/' /workspace/config.json"

  # File injection: docker cp from control machine into container
  - copy: /local/path/to/mods
    dest: /workspace/mods          # default: /workspace/mods/<basename>

  # File injection from a remote host (delegated)
  - copy: /path/on/remote/host
    dest: /workspace/patches
    source_host: 10.0.0.5          # rsync from source_host → target, then docker cp
```

`{key}` placeholders in string commands are resolved from the config chain.

### post_exec

Runs inside the **head container only**, after the server passes health checks (`/v1/models` returns 200). Sequential,
fail-fast.

```yaml
post_exec:
  - "curl -s http://localhost:{port}/v1/models | python3 -m json.tool"
  - "echo 'Server ready: {model}'"
```

### post_commands

Runs on the **control machine** (where sparkrun runs) via `subprocess.run(shell=True)`. Sequential, fail-fast.

```yaml
post_commands:
  - "curl -s http://{head_ip}:{port}/v1/models"
  - "python3 scripts/warmup.py --url {base_url}"
```

### stop_after_post

Stop the workload after post hooks complete. Useful for batch/one-shot workflows:

```yaml
post_commands:
  - "python3 benchmark.py --url {base_url} --output results.json"
stop_after_post: true
```

### Hook Template Variables

| Variable            | Available In                             | Source                                |
|---------------------|------------------------------------------|---------------------------------------|
| All `defaults` keys | `pre_exec`, `post_exec`, `post_commands` | Config chain                          |
| `{model}`           | All                                      | Recipe field                          |
| `{head_host}`       | `post_exec`, `post_commands`             | Detected at runtime                   |
| `{head_ip}`         | `post_exec`, `post_commands`             | Detected at runtime                   |
| `{port}`            | All                                      | Config chain                          |
| `{cluster_id}`      | `post_exec`, `post_commands`             | Generated                             |
| `{container_name}`  | `post_exec`, `post_commands`             | Generated                             |
| `{cache_dir}`       | `post_exec`, `post_commands`             | Resolved                              |
| `{base_url}`        | `post_exec`, `post_commands`             | Derived: `http://{head_ip}:{port}/v1` |

---

## Builders

Builders prepare the container image **before** distribution. Most recipes don't need one — sparkrun pulls images
automatically.

| Builder       | Description                                                       |
|---------------|-------------------------------------------------------------------|
| `docker-pull` | Default fallback. Relies on distribution phase for pulling        |
| `eugr`        | Builds eugr-style containers with `build_args` and `mods` patches |

```yaml
builder: eugr
builder_config:
  repo_url: https://github.com/eugr/spark-vllm-docker.git
  branch: main
  rebuild: true   # force a fresh image (see below)
```

`builder_config` is passed directly to the builder plugin's `prepare_image()`. Most contents are
builder-specific, with one standard cross-builder key:

- **`rebuild`** (bool) — request the freshest possible image. It is a no-op for `docker-pull` (the
  distribution phase already handles pulling). For `eugr` it forces a from-scratch rebuild of
  locally-built images (bypassing the build cache) and a fresh `docker pull` of pullable registry
  images. It behaves identically across `local`, `push`, and `delegated` transfer modes.

The `rebuild` key can also be set per-invocation with `sparkrun run --rebuild <recipe>`
(or `--no-rebuild` to force it off); the CLI flag overrides the recipe's `builder_config.rebuild`.

#### eugr is pull-first

Mirroring upstream `build-and-copy.sh`, the eugr builder **pulls a prebuilt image by default and only
builds from wheels when `build_args` request it**. A build is requested when `build_args` contains
`--use-wheels` **or any custom build flag** (`--rebuild-vllm`, `--rebuild-flashinfer`, `--vllm-ref`,
`--exp-mxfp4`, `--force-download`, `--apply-vllm-pr`, …) — exactly the flags that set the script's
`CUSTOM_BUILD_REQUESTED`. `build_args` that are empty or contain only the deprecated `--tf5` no-op do
**not** trigger a build.

**Pull path (default):**

- The eugr nightly `:latest` sentinels — `ghcr.io/spark-arena/dgx-vllm-eugr-nightly(-tf5):latest` and the
  Docker Hub `eugr/spark-vllm:latest` (short or `docker.io/`-qualified) — resolve to sparkrun's
  authoritative `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` and are pulled.
- Any other non-pullable eugr image (e.g. a v1 `container: vllm-node`) is reused if already present,
  otherwise substituted with our nightly and pulled — eugr no longer builds it implicitly.
- `--rebuild` force-pulls a fresh copy.
- `--exp-b12x` (alias `--experimental-b12x`) selects the b12x preset. Like upstream — where it swaps
  the vLLM fork/ref and Torch family and points `PREBUILT_RUNNER_IMAGE` at `eugr/spark-vllm-b12x`
  *without* setting `CUSTOM_BUILD_REQUESTED` — it does not trigger a build on its own: the sentinel
  resolves to `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x:latest` and is pulled. Naming a b12x
  prebuilt image (ours, or `eugr/spark-vllm-b12x:latest`) selects the variant the same way.

**Build path (`--use-wheels` or a custom build flag):**

- The nightly sentinels build locally as `sparkrun-eugr-vllm` (or `sparkrun-eugr-vllm-b12x` when
  `--exp-b12x` is present, mirroring upstream's `vllm-node-b12x` default tag, so a b12x source build
  can't overwrite the standard nightly); other images build under their own name.
- `build_args` are forwarded verbatim to `build-and-copy.sh` (so `--use-wheels`, `--vllm-ref`,
  `--exp-b12x`, etc. take effect). The build cache is honored for the standard `--use-wheels` build;
  `--rebuild` forces a from-scratch rebuild.
- Flag incompatibilities are the script's to enforce — e.g. `--exp-b12x --use-wheels` is rejected
  upstream (b12x vLLM wheels aren't published) and surfaces as a build failure rather than being
  second-guessed here.

Set `defaults.builders.eugr.use_sentinel_image: false` in `~/.config/sparkrun/config.yaml` to opt out of
pull-first entirely: images are used verbatim and a missing one is built via `build-and-copy.sh`.

### eugr Builder (v1 Compatibility, Legacy)

v1 recipes with `build_args`/`mods` auto-route to `eugr-vllm` runtime and the eugr builder:

```yaml
recipe_version: "1"
model: Qwen/Qwen3.5-0.8B
runtime: vllm
container: vllm-node-tf5

build_args:
  - "--tf5"

mods:
  - mods/fix-qwen3.5-chat-template

defaults:
  port: 52001
  tensor_parallel: 1
  gpu_memory_utilization: 0.175
```

The eugr builder clones the repo, runs docker build with build_args, appends mods to pre-exec lifecycle hooks, and
returns the built image name.

---

## GGUF Recipes (llama.cpp)

```yaml
model: Qwen/Qwen3-1.7B-GGUF:Q8_0
runtime: llama-cpp
max_nodes: 1
container: scitrera/dgx-spark-llama-cpp:b8076-cu131

defaults:
  port: 8000
  host: 0.0.0.0
  n_gpu_layers: 99
  ctx_size: 8192

command: |
  llama-server \
      -hf {model} \
      --host {host} --port {port} \
      --n-gpu-layers {n_gpu_layers} \
      --ctx-size {ctx_size} \
      --flash-attn on --jinja --no-webui
```

- Colon syntax (`repo:quant`) downloads only matching quant files
- `max_model_len` is auto-mapped to `ctx_size` for cross-runtime CLI compatibility
- `tensor_parallel` → `--split-mode row`, `pipeline_parallel` → `--split-mode layer` (mutually exclusive)
- Pre-synced GGUF: `-hf` auto-rewritten to `-m` with container cache path

### Vision GGUF models (multimodal projector)

Vision GGUF models ship a companion **multimodal projector** (`mmproj-*.gguf`) alongside the quantized weights. sparkrun
downloads it automatically (no extra config) and resolves its container path. There is no need to hardcode a snapshot
path.

```yaml
model: unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M
runtime: llama-cpp
max_nodes: 1
container: ghcr.io/spark-arena/dgx-llama-cpp:latest

# Optional projector selector (swept into runtime_config). Defaults to `auto`,
# which prefers F16 → BF16 → F32 → F8. Pin a precision with e.g. `F32`, give a
# filename, or set `false` to disable auto-injection.
mmproj: auto

defaults:
  port: 8001
  host: 0.0.0.0
  alias: qwen3-vl-8b
  n_gpu_layers: 99
  ctx_size: 8192
  flash_attn: on            # valued in modern llama.cpp (on/off/auto)
  jinja: true
  top_p: 0.8
  top_k: 20
  temperature: 0.7          # → --temp
  min_p: 0.0
  presence_penalty: 1.5
  webui: false              # inverted → --no-webui
  mmap: false               # inverted → --no-mmap
```

- The example above is **command-less**: with the projector and sampling options in `defaults`, sparkrun builds the full
  `llama-server` command and auto-injects `--mmproj <resolved path>`.
- To place the projector explicitly in a `command:` template, use `{mmproj}` (e.g. `--mmproj {mmproj}`); auto-injection
  is skipped whenever the rendered command already contains `--mmproj`.
- `flash_attn` accepts `on`/`off`/`auto` (booleans map to `on`/`off`); `webui`/`mmap` are inverted toggles that emit
  `--no-webui`/`--no-mmap` only when set false.
- **GB10 default:** on DGX Spark (GB10), llama.cpp defaults to `mmap: false` (i.e. `--no-mmap`) because memory-mapped
  GGUF loading performs poorly on its unified memory. This platform default applies only when the recipe is silent on
  `mmap`; set `mmap: true` in `defaults` to opt back in.

---

## Recipe Discovery

Search order:

1. **`@registry/recipe-name`** — scoped lookup in specific registry
2. **URL** — HTTP/HTTPS fetched and cached
3. **File path** — exact or with `.yaml`/`.yml` extension
4. **CWD scan** — `.yaml`/`.yml` files that are valid recipes (must have `model`, `container`, resolvable `runtime`)
5. **Registry search** — flat + recursive lookup, applied **independently per registry**

Each enabled registry is searched on its own terms: a flat `<recipe-dir>/<name>.yaml` wins within that
registry, and only if that registry has no flat match is its recipe dir scanned recursively. A flat hit in
one registry never suppresses the recursive scan of another — so a recipe that lives only in a
subdirectory stays reachable even when a different registry has a flat file with the same stem.

Because the scan is recursive, a bare stem is not always unique *within* one registry —
`a/foo.yaml` and `b/foo.yaml` are two different recipes. Any name matching more than one recipe raises an
error rather than guessing, listing path-qualified names you can use verbatim:

```
Recipe 'foo' is ambiguous — 2 matches in registry 'official' (@official/a/foo, @official/b/foo).
Use the full name to specify.
```

`@registry/name` disambiguates across registries, and `@registry/subdir/name` disambiguates within one
(the scope is split on the first `/`, so everything after it is a path under the registry's recipe dir).
When stdin is a TTY the CLI offers these as a numbered prompt instead of erroring.

---

## Complete Example

```yaml
recipe_version: "2"
model: Qwen/Qwen3-8B
runtime: vllm
container: scitrera/dgx-spark-vllm:latest
min_nodes: 1
max_nodes: 4

metadata:
  description: "Qwen3 8B — general purpose"
  maintainer: you
  model_dtype: bfloat16

defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 2
  gpu_memory_utilization: 0.9
  max_model_len: 32768
  served_model_name: qwen3-8b
  trust_remote_code: true
  enable_prefix_caching: true

env:
  NCCL_CUMEM_ENABLE: "0"

executor_config:
  restart_policy: unless-stopped

pre_exec:
  - "pip install flash-attn==2.7.3 --no-build-isolation"

command: |
  vllm serve \
      {model} \
      --served-model-name {served_model_name} \
      --port {port} --host {host} \
      -tp {tensor_parallel} \
      --gpu-memory-utilization {gpu_memory_utilization} \
      --max-model-len {max_model_len} \
      --trust-remote-code \
      --enable-prefix-caching

post_commands:
  - "curl -s http://{head_ip}:{port}/v1/models | python3 -m json.tool"
```
