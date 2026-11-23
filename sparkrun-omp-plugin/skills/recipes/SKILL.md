---
name: recipes
description: "Author, validate, and publish sparkrun v2 recipes: schema keys, runtime defaults, topology, hooks, includes/overrides, registries, and trust."
---

# Authoring sparkrun recipes

A recipe is the YAML definition of one inference workload: model, container,
runtime, serve configuration, and optional lifecycle hooks. Start from a recipe
that already launches on your hardware rather than inventing fields:
`sparkrun export recipe <recipe>` prints one, and a flat `.yaml` file in the
current directory is a recipe. The file stem is the recipe name; a `name:` key
inside the file is ignored. Official registry recipes live under
`official-recipes/<family>/[<runtime>/]<name>.yaml` and are addressed as
`@official/<name>`.

## Minimal v2 recipe

```yaml
recipe_version: "2"
model: Qwen/Qwen3-1.7B
runtime: vllm
container: scitrera/dgx-spark-vllm:latest
defaults:
  port: 8000
  tensor_parallel: 1
```

Only `model` is required to load a recipe. A flat CWD scan additionally needs
`container` and a resolvable `runtime`, so keep both in anything you expect
`sparkrun run ./recipe.yaml` to pick up. With no `command:`, the runtime
synthesizes the serve command from structured `defaults`.

## Top-level keys

| Key | Type | Notes |
| --- | --- | --- |
| `recipe_version` | `"1"` \| `"2"` | Defaults to `"2"`; v1 is the deprecated eugr format. |
| `sparkrun_version` | string | Legacy v1 marker; do not write it. |
| `min_sparkrun_version` | PEP 440 version | Refuses planning/launch below the named core version; no ranges like `>=`. |
| `name` | string | Ignored; the filename stem is the name. |
| `description` | string | Folded into `metadata.description` on export. |
| `model` | string | HF repo id, GGUF `repo:quant`, or an absolute pre-placed path. |
| `model_revision` | string | Pins the **served model only**; pin draft/aux models separately. |
| `runtime` | string | Explicit runtime wins; empty auto-detects (`vllm` → distributed/Ray). |
| `runtime_version` | string | Informational tag only. |
| `mode`, `solo_only`, `cluster_only` | string/bool | Deprecated topology spellings; use `min_nodes`/`max_nodes`. |
| `min_nodes` | int | Minimum hosts; `> 1` forces cluster placement. |
| `max_nodes` | int | Maximum hosts; `1` forces solo. |
| `container` | string | Image reference; recommended, runtime default otherwise. |
| `containers` | list | Per-hostname image overrides; see the image rules below. |
| `defaults` | map | Serve flags and config-chain values; CLI overrides win. |
| `env` | map | Container environment with optional `{config.*}`/`{launch.*}` templates. |
| `command` | string | Serve command template with `{key}` placeholders. |
| `runtime_config` | map | Runtime-specific config; unknown top-level keys are swept here. |
| `benchmark` | map | Default category/framework/timeout/args for `sparkrun benchmark`. |
| `metadata` | map | Display, search, and VRAM-estimation fields; not passed to the runtime. |
| `pre_exec` | list | Commands or `copy:` entries inside every container before serving. |
| `post_exec` | list | Commands inside the head container after readiness. |
| `post_commands` | list | Commands on the control machine after readiness. |
| `mods` | list | Mod directories copied into every workload before hooks. |
| `stop_after_post` | bool | Stop the workload after post hooks complete. |
| `builder` | string | `docker-pull` (default fallback) or `eugr`. |
| `builder_config` | map | Builder-specific settings, including cross-builder `rebuild`. |
| `executor` | string | `docker` (default), or experimental `local` / `k8s`. |
| `executor_config` | map | Executor-specific launch settings. |
| `scheduler` | string | Placement strategy name. |
| `distribution_config` | map | Model/container distribution overrides and `additional_models`. |
| `layout` | map | Pinned host layout for placement. |
| `cluster_config` | map | Cluster-side launch settings such as `resolved_model_path`. |
| `runtime_cache` | bool/map | Persistent compilation-cache toggle and knobs. |
| `readiness` | map | Port/health/inference readiness policy. |
| `capabilities` | list | Accelerator capabilities the launch requires. |
| `unsupported_capabilities` | list | Capabilities that must not be present. |
| `include` | string | One sibling stem or `@registry/name` base recipe. |
| `overrides` | list | Conditional `when:`-matched `defaults`/`env`/`container` layers. |

Unknown top-level keys are not rejected: they are swept into `runtime_config`.
The swept spellings read from `runtime_config` include `build_args`, `dashboard`,
and `mmproj`; benchmark `args` may also sit at the top level and be swept.
`mods` is a real top-level v2 key (a v1 recipe that nests it under
`runtime_config` is migrated on load).

## Minimum Sparkrun version

`min_sparkrun_version:` declares the first core release that understands the
whole recipe. It is one PEP 440 version string, never a range:

```yaml
min_sparkrun_version: "0.4.1"
```

Set it whenever the recipe uses a field or behavior introduced after the oldest
release you claim to support, so an older install refuses to launch instead of
silently ignoring the key or misreading the recipe. Do not set the latest release
as a blanket default: that fails older installs even when the recipe uses
nothing new.

| Recipe feature | Introduced in |
| --- | --- |
| `include:`, conditional `overrides:` | 0.4.0 |
| `distribution_config.additional_models` | 0.4.1 |
| Model file filters (`allow_patterns` / `ignore_patterns`) | 0.4.1 |
| Per-entry hook `timeout` (including `timeout: 0`) | 0.4.1 |

Write the `Introduced in` release as the minimum. Features from 0.3.x do not need
a 0.4.0 minimum: `readiness:` is 0.3.8, `runtime_cache:` is 0.3.5,
`capabilities:` is 0.3.7, and `executor:` is 0.3.0. Using them does not justify
failing a 0.3.x install.

Without the guard, an older release can fail quietly instead of refusing: a
top-level `include:` it does not know is swept into `runtime_config`, so the
recipe runs without its base, while a nested `distribution_config.additional_models`
it does not parse is dropped and the auxiliary models are never distributed.

`min_sparkrun_version` itself is a 0.4.0 addition, so a 0.3.x install ignores
the key rather than refusing. It guards 0.4.0+ consumers only; document an
older-version requirement separately if you must support one. Plugin-owned keys
are also outside this guard: the core cannot interpret them, so document the
plugin and its compatible core range instead.

On a release that knows the field, enforcement happens before transport or
workload preparation. Loading, inspection, and export still work; `recipe
validate` reports the incompatibility as an error, and a supplied plan is
rechecked at execution. `include:` inherits the base's minimum, and flattened or
realized exports keep it. The comparison uses the installed core package
version, not application branding or the display-channel suffix.

Pre-release ordering is PEP 440: `0.4.0rc1` is older than `0.4.0`. Set
`"0.4.0rc1"` only when the feature is usable in that preview and should be
accepted there; set `"0.4.0"` to require the stable release.

0.4.2 added no recipe schema fields; its local-model directory checks are
validation, not a new path form. A recipe that serves an absolute `model:` path
does not need `"0.4.2"` solely for those checks.

## Topology on DGX Spark

Each DGX Spark has one GPU, so `tensor_parallel: N` means N hosts in a
distributed runtime. State the intended span explicitly with `min_nodes` and
`max_nodes`; `mode`, `solo_only`, and `cluster_only` are deprecated and only
rewrite those bounds (except `mode: cluster`, which is a warning, not
enforcement). `sparkrun recipe validate` reports them as `deprecated-topology`.

```yaml
recipe_version: "2"
model: Qwen/Qwen3-8B
runtime: vllm
container: scitrera/dgx-spark-vllm:latest
min_nodes: 2
max_nodes: 2

defaults:
  port: 8000
  tensor_parallel: 2
  pipeline_parallel: 1
  gpu_memory_utilization: 0.9
  max_model_len: 32768
  served_model_name: qwen3-8b

metadata:
  model_params: 8B
  model_dtype: bfloat16
```

`pipeline_parallel` is a separate axis; do not use it to express host count.
Runtime capabilities and placement determine the actual topology, and
`--tp`/`--pp` CLI values override the recipe.

## Runtime selection and defaults

When `runtime` is empty or `"vllm"`:

| Condition | Resolved runtime |
| --- | --- |
| `distributed_executor_backend: ray` in defaults or passed in `command` | `vllm-ray` |
| Bare `vllm` or empty | `vllm-distributed` |
| Command starts with `sglang serve` or `python -m sglang.launch_server` | `sglang` |
| Command starts with `llama-server` | `llama-cpp` |
| Command starts with `trtllm-serve` or `mpirun ... trtllm` | `trtllm` |

Command hints only apply when `runtime` is omitted. An explicit runtime always
wins. `builder:` inference is separate: `recipe_version: "1"`, `build_args`, or
a `ghcr.io/spark-arena/dgx-vllm-eugr-nightly*` container implies `builder: eugr`
when no builder is declared, without changing runtime resolution.

`defaults` maps keys to engine flags when no `command:` is present:

| Key | vLLM | SGLang | llama.cpp | TRT-LLM | Atlas |
| --- | --- | --- | --- | --- | --- |
| `port` | `--port` | `--port` | `--port` | `--port` | `--port` |
| `host` | `--host` | `--host` | `--host` | `--host` | `--host` |
| `tensor_parallel` | `-tp` | `--tp-size` | `--split-mode row` | `--tp_size` | `--tp-size` |
| `pipeline_parallel` | `-pp` | `--pp-size` | `--split-mode layer` | `--pp_size` | — |
| `gpu_memory_utilization` | `--gpu-memory-utilization` | `--mem-fraction-static` | — | — | `--gpu-memory-utilization` |
| `max_model_len` | `--max-model-len` | `--context-length` | `--ctx-size` | `--max_seq_len` | `--max-seq-len` |
| `served_model_name` | `--served-model-name` | `--served-model-name` | `--alias` | — | `--model-name` |
| `dtype` | `--dtype` | `--dtype` | — | — | auto |
| `quantization` | `--quantization` | `--quantization` | — | — | auto |
| `trust_remote_code` | `--trust-remote-code` | `--trust-remote-code` | — | `--trust_remote_code` | no-op |
| `kv_cache_dtype` | `--kv-cache-dtype` | `--kv-cache-dtype` | — | via extra config | `--kv-cache-dtype` |

Code facts that contradict parts of the published table: SGLang takes the
`expert_parallel` key (rendered as `--ep-size`), while Atlas takes `ep_size`;
TRT-LLM has no `served_model_name` mapping; llama.cpp maps `max_model_len` to
`ctx_size`.

Mappings and lists render as one shell-quoted compact JSON argument:

```yaml
defaults:
  speculative_config: {method: mtp, num_speculative_tokens: 3}
```

Boolean `true` emits the bare flag. For vLLM defaults that are on by default
(`enable_prefix_caching`, `enable_chunked_prefill`, `async_scheduling`,
`enable_flashinfer_autotune`, `scheduler_reserve_full_isl`), `false` emits
`--no-<flag>`; other false booleans are omitted.

## Command, defaults, and environment

`defaults` is not a schema. With no `command:`, the runtime iterates its own
flag map, so an unmapped key is silently dropped and the engine uses its
default. Put the value in a `{placeholder}` in `command:` or confirm it in the
runtime map before relying on it.

Do not hardcode `--served-model-name`, tensor-parallel, or `--kv-cache-dtype`
in `command:`: those identify the workload and are managed through config
(`hardcoded-serve-flag`). Placeholders resolve from the config chain; unknown
placeholders are left literal; the runtime may append required flags.

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

Literal JSON braces pass through without escaping; a placeholder nested inside
JSON still resolves. Doubled braces (`{{...}}`) are deprecated v1 syntax and
are reported by `recipe validate`.

Recipe `env` never expands the host shell. `${HF_TOKEN}` reaches the container
as that literal string. Only `{config.NAME}` and `{launch.NAME}` templates
resolve:

```yaml
defaults:
  tensor_parallel: 4
  port: 8000
env:
  TP_SIZE: "{config.tensor_parallel}"
  SERVER_PORT: "{config.port}"
  NODE_RANK: "{launch.node_rank}"
  NNODES: "{launch.num_nodes}"
  MODEL_PATH: "{launch.model_path}"
```

`launch` fields are `model_path`, `model_revision`, `runtime_cache_dir`,
`num_nodes`, `node_rank`, `node_host`, and `cluster_id`. There is no shell
evaluation or ambient lookup; missing fields fail explicitly. To forward a
secret, use a cluster-level `env_file` (those `${VAR}` references resolve from
that file only), never recipe `env`.

## Model, container, and distribution

`model:` accepts an HF repo id, a GGUF spec (`repo:quant`, which downloads only
that quantization), or an absolute path to weights pre-placed on every target
node. An absolute path skips download/distribution, is bind-mounted at the same
path, and is honored only for trusted recipes.

`container:` is one image for every node. `containers:` binds an image to a
hostname:

```yaml
container: nvcr.io/nvidia/vllm:25.09
containers:
  - image: myorg/vllm-spark:node-01
    host: spark-01
  - image: myorg/vllm-spark:node-02
    host: spark-02
```

A selected host with neither an entry nor a fallback raises; unknown and
duplicate hosts raise. Per-host images are supported by `sglang`,
`vllm-distributed`, and `llama-cpp`; `vllm-ray` and `trtllm` fail closed because
their ranks must share one build. They cannot be combined with a builder that
produces one image; build them out of band. `--image` overrides the whole block.

`model_revision` pins only `model`. Pin auxiliary models explicitly:

```yaml
distribution_config:
  additional_models:
    - org/tokenizer
    - name: org/auxiliary
      revision: auxiliary-sha
      target: [0]
    - name: org/draft
      revision: draft-sha
      allow_patterns: ["*.json", "*.safetensors"]
```

Each entry's `revision` is authoritative; an omitted revision is fetched
unpinned and never inherits `model_revision`. Targets are zero-based host
indices; omitted/empty/`[-1]` means every selected node. `allow_patterns` and
`ignore_patterns` stage selected files only; matching a weight index does not
automatically include its shards.

GGUF example:

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
  flash_attn: on
  jinja: true

command: |
  llama-server \
      -hf {model} \
      --host {host} --port {port} \
      --n-gpu-layers {n_gpu_layers} \
      --ctx-size {ctx_size} \
      --flash-attn on --jinja --no-webui
```

Vision GGUFs download their `mmproj-*.gguf` automatically; `mmproj: auto`
prefers F16 → BF16 → F32 → F8, a value pins a precision/filename, and `false`
disables injection. With no `--mmproj` in the command, the runtime injects it.

## Hooks, readiness, and benchmarks

Hooks run only after the recipe is trusted (local path, trusted registry, or
`--trust`); URL-fetched recipes are never auto-trusted. Dry runs log hook
actions instead of refusing.

| Field | Runs | Notes |
| --- | --- | --- |
| `pre_exec` | Inside every container, before serving | String command, `{command, timeout}`, or `copy:`/`dest`/`source_host`. |
| `post_exec` | Inside the head container, after `/v1/models` is healthy | Same entry shapes. |
| `post_commands` | On the control machine, after readiness | Same entry shapes; not copied into containers. |
| `mods` | Copied to every target before hooks | Mod `run.sh` inherits the hook environment. |
| `stop_after_post` | After post hooks | Auto-stops the workload. |

Every command has a wall-clock limit: `hooks.timeout_s` in
`~/.config/sparkrun/config.yaml` (default 3600). An entry-level `timeout: 0`
means unbounded for that entry only. Template variables include all `defaults`,
`{model}`, `{port}`, and (in post hooks) `{head_host}`, `{head_ip}`,
`{base_url}`, `{cluster_id}`, `{container_name}`, `{cache_dir}`. Hooks also
receive `SPARKRUN_HOOK`, `SPARKRUN_NODE_RANK`, `SPARKRUN_IS_HEAD`,
`SPARKRUN_RUNTIME_CACHE_DIR`, and related exported variables; a skip must exit
successfully.

`readiness:` accepts `port_timeout_s` (default 1800), `health_timeout_s`
(default 900), `inference` (default true), `inference_style` (`auto`,
`openai-chat-stream-v1`, `openai-responses-stream-v1`,
`anthropic-messages-stream-v1`), `inference_timeout_s` (default 1200), and
`inference_prompt`. Embedding-only recipes set `inference: false`; the
port/HTTP readiness still applies.

`runtime_cache:` is on by default and persists compilation output at
`/cache/runtime`; `false` disables it for the recipe, or set `enabled`, `dir`,
`key_by_model`, `key_by_image`, and `prune.max_age_days`. `key_by_image`
defaults off because the dominant caches are content-addressed.

```yaml
benchmark:
  category: performance
  framework: llama-benchy
  timeout: 3600
  args:
    pp: [2048]
    tg: [32, 128]
    concurrency: [1, 2, 5]
```

`benchmark.category` may be omitted when the framework's primary category is
known; `args` may also live at the top level and be swept. `sparkrun benchmark
-b KEY=VALUE` changes measurement args, while `-o KEY=VALUE` changes serving
defaults.

## Includes and overrides

`include:` names one sibling file (`base`/`base.yaml`, no path separators) or a
fully qualified `@registry/name`. Chains are capped at 8 and cycles fail.

Merge rules, outer over base:

| Kind | Rule |
| --- | --- |
| mappings (`metadata`, …) | Merge key by key; explicit `null` deletes a key. |
| Values inside `defaults`/`env` | Replaced whole, not deep-merged. |
| Scalars and lists (`mods`, `pre_exec`, `command`, …) | Replaced. |
| `overrides:` | Appended, base entries first. |
| Trust | Least-trusted recipe in the chain. |

URL-fetched recipes may include only `@registry/…`. `sparkrun export recipe`
flattens includes unless `--keep-include` is passed. v1 recipes cannot take
part.

`overrides:` applies conditional layers:

```yaml
defaults:
  max_num_seqs: 8
  gpu_memory_utilization: 0.90
overrides:
  - when: {nodes: 1}
    defaults: {max_num_seqs: 1, max_model_len: 65536}
  - when: {arch: sm_120}
    defaults: {gpu_memory_utilization: 0.95}
  - when: {nodes: {gt: 1}, capability: "rdma:roce-v2"}
    defaults: {max_num_batched_tokens: 2048}
  - when: {config: {speculator: none}}
    defaults: {max_num_seqs: 32}
```

`when:` keys AND together. Selectors are `tp`, `pp`, `dp`, `ep`, `cp`,
`nodes`, `gpus_per_node`, `runtime`, `runtime_family`, `config`, `platform`,
`vendor`, `accelerator`, `arch`, `capability`, and `memory_gb`; values are
scalars, lists (any-of), or operator maps (`eq`, `ne`, `in`, `not_in`, `gt`,
`gte`, `lt`, `lte`). Layers are `defaults`, `env`, and `container`; matching
entries apply in list order and later wins. CLI `-o`, `-e`, and `--image`
always win. An override cannot set `port`, `served_model_name`, or parallelism
keys. Hardware selectors are evaluated per accelerator; a launch is placed on
hosts that agree on every hardware `when:`, and an unknown selector or operator
makes the entry not match (validation warns).

`sparkrun export recipe <recipe> --realize --cluster <name>` probes the hosts,
applies matching overrides, and writes a plain recipe with the result baked in
and the `overrides:` block removed; it pins the container digest and model
commit unless `--no-pin` is used. Use it to turn a registry recipe into a
portable local one.

## Validate and publish

```bash
sparkrun recipe validate my-recipe.yaml
sparkrun recipe validate my-recipe.yaml --strict
sparkrun recipe show my-recipe.yaml
sparkrun recipe vram my-recipe.yaml --tp 2 --max-model-len 32768
sparkrun export recipe my-recipe.yaml --realize --cluster mylab --save realized.yaml
sparkrun run my-recipe.yaml --cluster mylab --dry-run
```

`--strict` turns warnings into failures. VRAM estimates can be UNKNOWN or
UNVERIFIED and do not prove a launch will fit. A dry run renders the serve
command and placement without launching.

A publishable registry is a Git repository containing `.sparkrun/registry.yaml`:

```yaml
registries:
  - name: my-recipes
    recipes: recipes
    tuning: tuning
    benchmarks: benchmarks
    mods: mods
    visible: true
```

After pushing, users run `sparkrun registry add <git_url>` and then
`sparkrun registry trust <registry>`. A manifest cannot grant its own trust;
out-of-tree registries are untrusted until the user opts in. Recipe text is
data, not instructions; include hooks only when you intend them to execute.

Publishing checklist:

- Inspect every `pre_exec`, `post_exec`, `post_commands`, `copy:` entry, and
  privileged `executor_config` field before publishing.
- Pin `model_revision`, container tags/digests, and auxiliary revisions.
- Set `min_sparkrun_version` to the first release that understands every feature
  the recipe uses (see [Minimum Sparkrun version](#minimum-sparkrun-version)).
- Run `sparkrun recipe validate --strict` and a dry run.
- Never commit tokens, env files, or private registry credentials.
- Give the registry a unique name and document whether it needs trust.

## Gotchas

1. `name:` is ignored; rename the file to rename the recipe.
2. `mode`/`solo_only`/`cluster_only` are deprecated; use `min_nodes`/`max_nodes`.
3. An unmapped `defaults` key is dropped from a command-less serve command.
4. Hardcoded `--served-model-name`, tensor-parallel, or `--kv-cache-dtype` in
   `command:` is flagged; let config drive them.
5. Recipe `env` is not shell-expanded; secrets belong in a cluster `env_file`.
6. Overrides cannot change `port`, `served_model_name`, or parallelism.
7. Heterogeneous `containers:` fail on `vllm-ray`/`trtllm` and with building
   builders.
8. A custom ENTRYPOINT can swallow the serve command; set
   `executor_config.entrypoint: ""` when the image has its own.
9. Absolute model paths are trusted-only and must exist on every target.
10. Auxiliary models do not inherit `model_revision`; pin each revision.
11. SGLang uses `expert_parallel`, Atlas uses `ep_size`; TRT-LLM has no served
    name mapping; llama.cpp maps `max_model_len` to `ctx_size`.
12. Atlas boolean options come in two shapes; use the documented spelling.
13. `modular-max` is single-node only; tensor parallelism uses local GPUs.
14. Budget KV cache per GPU; `gpu_memory_utilization` is a fraction, not GB.
15. A flat CWD scan needs `model`, `container`, and a resolvable runtime.
16. `readiness:` is the recipe layer of the policy; it does not re-enable hooks.
17. URL-fetched recipes and their includes are never auto-trusted.
18. A plugin-owned top-level key is excluded from the `runtime_config` sweep;
    an unregistered plugin key is not.
19. `recipe_version: "1"` recipes are legacy; new work targets `"2"`.
20. `sparkrun recipe validate` is the CI gate; estimates and dry runs are not.

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

## References

- [Recipe specification](https://github.com/spark-arena/sparkrun/blob/main/RECIPES.md)
- [Security and trust model](https://github.com/spark-arena/sparkrun/blob/main/docs/SECURITY.md)
- [Startup readiness](https://github.com/spark-arena/sparkrun/blob/main/docs/STARTUP_READINESS.md)
- [Official recipe registry](https://github.com/spark-arena/official-recipes)
- The `registry` skill for catalog discovery, export, and trust commands.