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
| `model`           | string | **yes**     | —               | HuggingFace model ID or GGUF spec (`Qwen/Qwen3-1.7B-GGUF:Q4_K_M`) |
| `model_revision`  | string | no          | `null`          | Pin to a specific HF revision (branch, tag, or commit hash)       |
| `runtime`         | string | no          | auto-detected   | Runtime identifier. See [Runtime Resolution](#runtime-resolution) |
| `runtime_version` | string | no          | `""`            | Informational version tag                                         |
| `container`       | string | recommended | runtime default | Container image reference                                         |

GGUF models use colon syntax (`repo:quant`) to download only the matching quantization files. When pre-synced, sparkrun
rewrites `-hf` to `-m` with the resolved container cache path.

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
| `env`            | map    | `{}`    | Container environment variables. Supports `$VAR` / `${VAR}` expansion from control machine env |
| `command`        | string | `null`  | Command template. `{key}` placeholders resolved from config chain                              |
| `runtime_config` | map    | `{}`    | Runtime-specific config. Unknown top-level keys are auto-swept here                            |

### Metadata

Informational fields for VRAM estimation, display, and search. Not passed to runtime.

```yaml
metadata:
  description: "Qwen3 8B — general purpose"
  maintainer: "you <you@example.com>"
  category: agent
  model_params: 8B           # or 8000000000
  model_dtype: bfloat16      # float32, float16, bfloat16, int8, fp8, int4, awq4, gptq, nvfp4, q4_k_m, q8_0, ...
  kv_dtype: fp8_e5m2         # KV cache dtype (default: bfloat16)
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

### Benchmark

```yaml
benchmark:
  framework: llama-benchy     # default framework
  timeout: 3600
  args: # or put args at top level (auto-swept)
    pp: [ 2048 ]
    tg: [ 32, 128 ]
    concurrency: [ 1, 2, 5 ]
```

Used by `sparkrun benchmark <recipe>`. CLI `-o` overrides apply on top.

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

### Common Defaults Keys

| Key                      | vLLM                       | SGLang                  | llama.cpp            | TRT-LLM               | Description                       |
|--------------------------|----------------------------|-------------------------|----------------------|-----------------------|-----------------------------------|
| `port`                   | `--port`                   | `--port`                | `--port`             | `--port`              | Serve port                        |
| `host`                   | `--host`                   | `--host`                | `--host`             | `--host`              | Bind address                      |
| `tensor_parallel`        | `-tp`                      | `--tp-size`             | `--split-mode row`   | `--tp_size`           | TP degree (= node count on Spark) |
| `pipeline_parallel`      | `-pp`                      | `--pp-size`             | `--split-mode layer` | `--pp_size`           | PP degree                         |
| `gpu_memory_utilization` | `--gpu-memory-utilization` | `--mem-fraction-static` | —                    | —                     | GPU memory fraction               |
| `max_model_len`          | `--max-model-len`          | `--context-length`      | `--ctx-size`         | `--max_seq_len`       | Max sequence length               |
| `served_model_name`      | `--served-model-name`      | `--served-model-name`   | `--alias`            | —                     | Model name in API                 |
| `dtype`                  | `--dtype`                  | `--dtype`               | —                    | —                     | Model dtype                       |
| `quantization`           | `--quantization`           | `--quantization`        | —                    | —                     | Quantization method               |
| `trust_remote_code`      | `--trust-remote-code`      | `--trust-remote-code`   | —                    | `--trust_remote_code` | Allow remote code                 |
| `kv_cache_dtype`         | `--kv-cache-dtype`         | `--kv-cache-dtype`      | —                    | via extra config      | KV cache dtype                    |

Any key can appear in `defaults` — there is no fixed schema. Runtime-specific keys (e.g. `tool_call_parser`, `ctx_size`,
`n_gpu_layers`, `reasoning_parser`) are passed through to command template substitution.

---

## Executor Config

Controls container engine behavior. Layered: **CLI flags → recipe `executor_config` → built-in defaults**.

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
| `gpus`           | string | `"all"`     | `--gpus`       | GPU device spec                                                                                                     |
| `ipc`            | string | `"host"`    | `--ipc`        | IPC namespace                                                                                                       |
| `shm_size`       | string | `"10.24gb"` | `--shm-size`   | Shared memory size                                                                                                  |
| `network`        | string | `"host"`    | `--network`    | Network mode                                                                                                        |

**CLI override:** `sparkrun run --no-rm --restart always my-recipe`

All runtimes automatically inherit executor settings — no per-runtime changes needed.

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
```

`builder_config` is passed directly to the builder plugin's `prepare_image()`. Contents are builder-specific.

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

---

## Recipe Discovery

Search order:

1. **`@registry/recipe-name`** — scoped lookup in specific registry
2. **URL** — HTTP/HTTPS fetched and cached
3. **File path** — exact or with `.yaml`/`.yml` extension
4. **CWD scan** — `.yaml`/`.yml` files that are valid recipes (must have `model`, `container`, resolvable `runtime`)
5. **Registry search** — flat + recursive lookup across all enabled registries

Ambiguous names (same recipe in multiple registries) raise an error — use `@registry/name` to disambiguate.

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
