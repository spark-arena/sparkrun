# TensorFold

TensorFold ([ashhart/TensorFold](https://github.com/ashhart/TensorFold)) is a
Zig/CUDA native LLM server with an OpenAI-compatible API.  One sparkrun runtime
covers the engine's deployment shapes; the recipe selects the shape with
`defaults.tf_family`:

| `tf_family` | shape | engine |
|---|---|---|
| absent | solo on one GB10 | upstream TensorFold release |
| `glm` | native TP2 | upstream TensorFold release (GLM-5.3-Flash) |
| `nemotron` | native TP2 | upstream TensorFold release (Nemotron 3.5 Super VL) |
| `dsv41` | native TP2/TP4 | community DeepSeek-V4.1 fork, prepared bundle |

The runtime is gated to `gb10` hosts (`requires_capability`).  TensorFold's
released CUDA fatbins target sm_80/86/120/121, but only the GB10 (sm_121) path
is qualified by sparkrun today.

## Container images are built outside sparkrun

Like the other first-party runtime images, the upstream TensorFold image is not
defined in this repository.  It only needs the pinned release archive plus a
CUDA/NCCL-capable base:

- glibc ≥ 2.28 and a driver providing `libcuda.so.1`;
- **NCCL ≥ 2.30** when native ranks are used (TensorFold refuses older NCCL);
- libibverbs/verbs providers for RoCE, and cuBLASLt for `--vision`/projections.

The DGX Spark platform default names the upstream image tag; recipes may
override `container:` to pin an exact engine version or a locally built image.
If an image exists only on the targets (pre-built locally, not published),
set `distribution_config.containers.enabled: false`; the launcher then skips
the control-machine pull/save and leaves the local tag to each host.  The
runtime still probes the image on every target before the model starts.

The DeepSeek-V4.1 fork is **not** a first-party image: its recipes must name an
explicit `container:` built outside sparkrun (the fork's serving and follower
binaries, its three/four patch set and the tokenizers helper for bundle
preparation).  Validation refuses a `dsv41` launch without an explicit image.

## Solo: one Spark

The runtime emits, inside the container:

```text
tensorfold-native serve <model> --host 0.0.0.0 --port <port> --name <served>
  --no-update-check [--context N] [--parallel N] [--vision] [--no-thinking] ...
```

`--host`/`--port` are mandatory and are always emitted: the engine defaults to
`127.0.0.1:8080`, while sparkrun's readiness watcher and proxy expect the
recipe/chain port on the head host.  The engine exits on SIGTERM after a
graceful drain (`TF_DRAIN_S`, 20 s by default), which is what `sparkrun stop`
uses.

Served APIs: Chat/Completions, Responses and Anthropic Messages.  The runtime
declares all three to the proxy/readiness layers and uses `/health`.

### Common `defaults:` keys

| recipe key | TensorFold flag | notes |
|---|---|---|
| `port`, `host` | `--port`, `--host` | `host` defaults to `0.0.0.0` |
| `served_model_name` | `--name` | advertised model id |
| `max_model_len` | `--context` | prompt + reply window |
| `max_num_seqs` | `--parallel` | concurrent streams, **not** tensor parallelism |
| `max_tokens` | `--max-tokens` | default reply cap |
| `temperature`, `top_p`, `top_k`, `min_p` | same | sampling defaults |
| `thinking` | `--thinking` / `--no-thinking` | false emits `--no-thinking` |
| `vision`, `no_drafts`, `loop_guard`, `metrics_open`, `dashboard` | presence flags | `metrics_open`/`dashboard` are rank-0 only |
| `drafter`, `drafter_bits`, `alias`, `chat_template` | same | |
| `api_key` | `--api-key` | also surfaced to proxy discovery |
| `command_binary` | replaces `tensorfold-native` | for images with an entrypoint |

Anything else is reported by the unmapped-key check; use a recipe `command:`
template for exotic flags (solo only).

## NFS / pre-placed weights

TensorFold serves a local checkpoint directory as happily as an HF repo id, so
sparkrun's existing pre-placed model path is the NFS mechanism — no
runtime-specific code:

```yaml
model: /mnt/nfs/models/qwen3.8-27b-mlx4     # absolute path, present on every host
```

or, for a recipe that keeps a portable model id:

```yaml
model: TensorFold/Qwen3.8-27B-MLX-4bit
cluster_config:
  resolved_model_path: /mnt/nfs/models/qwen3.8-27b-mlx4
```

In both forms sparkrun skips download and distribution, verifies the path on
every target (`test -e`), identity-mounts the directory into the container at
the same path, and points the serve command at it.  `{model}` in a command
template also resolves to that path.  Two consequences:

- **Trust matters.** An absolute `model:` (or `cluster_config`) is a
  trusted-recipe feature: a local recipe or `--trust` is required.  A registry
  recipe cannot carry an NFS path that sparkrun will silently honour.
- The path must be the complete checkpoint directory (HF snapshot layout or the
  engine's expected model directory), not a partial copy.

A shared Hugging Face cache on NFS is the other supported shape: set the
cluster's `cache_dir` to the shared mount and disable distribution
(`distribution.model.enabled: false` or `skip_fan_out: true`), and the engine
reads the mounted cache (`HF_HOME`/`HF_HUB_CACHE` are set by the runtime).
Prefer the absolute-path form for TensorFold: it survives SHA-pinned downloads
that leave no `refs/` entry in the cache.

## Native TP: family-specific rank placement

TensorFold's native CUDA build has **no generic `--tp` flag** (upstream
`docs/multi-rank-receipts.md`).  Each family places its own ranks through
environment variables, only rank 0 serves HTTP, and the followers must already
be running when rank 0 starts:

| `tf_family` | rank/world variables | extra | qualified |
|---|---|---|---|
| `glm` | `TF_GLM_RANK` / `TF_GLM_WORLD` | `NCCL_COMM_ID` | 2 ranks (EXL3/TR3); NVFP4 contributed |
| `nemotron` | `TF_NEMOTRON_RANK` / `TF_NEMOTRON_WORLD` | `NCCL_COMM_ID`, NCCL ≥ 2.30 | 2 ranks for Super VL 120B |
| `dsv41` | `TF_TP_RANK` / `TF_TP_WORLD` + follower binary | `TF_TP_MASTER/PORT`, bundle env | 2 or 4 ranks in the reference deployment |

Set the family in `defaults:` and let `--tp` place the hosts (`tensor_parallel`
is the node count; one GPU per Spark):

```yaml
defaults:
  tensor_parallel: 2
  tf_family: glm          # or nemotron / dsv41
min_nodes: 2
max_nodes: 2
```

The orchestrator starts the follower ranks first, then rank 0, and skips the
head-port gate because the engine owns the join; the launcher's endpoint watcher
still owns readiness.  The runtime reserves a contiguous rendezvous port pair
(`port` and `port + 1` — some families use the second for their follower control
channel).

Recipe defaults are passed to both ranks; only rank 0 gets
`--host/--port/--name/--api-key`.  A recipe `command:` template is refused for TP
because it cannot emit different argv for the serving and follower ranks.

### Limits

- Upstream qualifies two ranks; GLM's engine accepts up to 16 and Nemotron's
  world must divide the expert shares, but sparkrun treats >2 as experimental.
- TP is not a speed multiplier yet on two GB10s (upstream issue #981); the
  reason to run it is capacity.
- Both ranks verify the checkpoint before serving; a shared NFS mount is the
  simplest way to guarantee they see the same one.
- The container needs the fabric: the Docker executor already uses host
  networking, privileged mode and the optional `/dev/infiniband` device; the
  runtime adds `IPC_LOCK` and unlimited memlock.  Set `NCCL_IB_GID_INDEX` etc.
  through cluster/recipe `env` for your fabric.

## `tf_family: dsv41` — DeepSeek-V4.1 TP2/TP4 (experimental)

The released TensorFold binary does not serve DeepSeek-V4.1 (upstream PR #827),
so this family targets a community fork.  Rank 0 runs `tensorfold-native serve`
with `TF_TP_*`; the followers run the different `tf-dsv41-lanes` binary, and the
engine reads a **prepared bundle**, not the raw checkpoint.

### Bundle layout

`model:` is the absolute bundle root, identical on every node (NFS or pre-placed
local mounts):

```text
<bundle>/model/                 config/tokenizer/index/encoding/, vision shards, chat_template.jinja
<bundle>/cache/rank-cache/      rank<N>of<W>-<hash>.bin
<bundle>/cache/dsv41_token_map.json
<bundle>/kit/                   kernel kit (aot/, MANIFEST, ...)
<bundle>/engram/                shards 47 and 48 of the base checkpoint
```

The fork's `tf-dsv41-rank-cache`/`node_inputs.sh` tool writes the per-rank
caches from the EXL3 pack (CPU only, about six minutes per rank); the kit comes
from the reference recipe, and the Engram tables are the base checkpoint's
shards 47/48.  A TP2 pair needs ~2×106 GB of rank caches; TP4 uses ~4×69 GB;
Engram is ~203 GB and is latency-sensitive, so the reference deployment keeps a
local per-node copy and overrides `engram_dir`.  These assets are prepared once,
outside sparkrun; the runtime only mounts and serves them.

`model:` is an absolute path, so the recipe must be trusted (local or
`--trust`), and the family requires an explicit `container:` because the fork
image is built outside sparkrun.

The bundle can be split by latency: Engram is read during generation and is
the usual candidate for node-local storage.  Copy shards 47/48 to a local
directory on every node, point `engram_dir` at it, and bind-mount that directory
into the container with the executor:

```yaml
defaults:
  engram_dir: /var/tmp/tensorfold-dsv41/engram
executor_config:
  volumes:
    - /var/tmp/tensorfold-dsv41/engram   # identity mount, same path inside
```

`model/`, `cache/` and `kit/` can stay on the shared NFS bundle; only
`TF_DS_ENGRAM`/`--engram` changes.  This is how the reference TP2 deployment
keeps Engram on each Spark's NVMe while the large shared artifacts stay on the
network.

### Runtime knobs

| key | meaning |
|---|---|
| `tensor_parallel` | 2 or 4 (required; single-rank DeepSeek is not qualified) |
| `max_model_len` | serve `--context` (1M in the reference lane) |
| `max_num_seqs` | serve `--parallel` and the follower `--streams` default |
| `streams`, `pool` | follower `--streams` / `--pool` |
| `round_rows`, `window`, `tf_2d` | TP4 2D-split settings (`TF_DS_*`); defaults 48 / 2097152 / `parity` |
| `served_k` | TP4 served-lane shape (`TF_DS_SERVED_K`, default `solo`) |
| `cache_dir`, `kit_dir`, `engram_dir`, `token_map` | bundle subdirectory overrides (absolute paths inside the mounted bundle, or a separately bind-mounted path) |
| `rdma_hca`, `net_if`, `rdma_gid_index`, `nccl_ib_gid_index` | fabric env |
| `cuda_kernels` | `TENSORFOLD_CUDA_KERNELS` (default `<bundle>/kit/aot`) |
| `tf_env` | extra `TF_*`/`NCCL_*` assignments, a mapping |
| `served_model_name`, sampling keys | same as upstream |

## Validation checklist for the hardware run

1. Build/publish the upstream image outside this repository; build or obtain the
   DeepSeek fork image and check both `tensorfold-native --version` and the
   follower binary.
2. `sparkrun run <recipe> --dry-run` — confirm the rendered command has
   `--host 0.0.0.0 --port <port>`, and for TP the rank env/`NCCL_COMM_ID`.
3. Solo: one Spark, then `/v1/models`, `sparkrun proxy`, `sparkrun cluster
   monitor` (`[tensorfold]` lines are colorized), `sparkrun logs`, `stop`.
4. NFS: an absolute `model:` present on the target; confirm no download and the
   identity mount.
5. TP2: GLM/Nemotron recipes; rank 1 must start before rank 0; both logs under
   the normal `sparkrun logs`; stop removes both containers.
6. DeepSeek: prepare the bundle first, run TP2 before TP4, verify `/v1/models`
   and a chat completion, then stop/restart and confirm the caches are reused.
7. Report the exact engine tag, CUDA/NCCL versions, model pack and revision.

## Jayleaton Zig engine family

The runtime also drives the Jayleaton Zig port of DeepSeek-V4.1
(`jayleaton/deepseek-v41-tensorfold-spark`, the `tensorfold-dsv41-zig` image).  It
is a **different engine variant**, not another build of the bertholomus fork:

- rank 0: `tensorfold-dsv41 serve <pack> ...`
- rank 1: `tf-dsv41-m1 follow <pack> <assets>`
- rank 1 starts first, then rank 0 (the native cluster path's `workers_first`
  order);
- the serving profile is the `TF_DSV41_*` set, not `TF_DS_*`;
- per-rank Engram, prepared weights, AOT/cache assets and session state live on
  local NVMe, while the pack itself may stay on NFS.

Select it with `tf_family: jayleaton`; it is TP=2 only and the recipe must name
an explicit image (there is no platform default for this fork).  A pre-built
image is required; the runtime does not build it for you.  The same is true of
the rank assets: pack the rank's Engram shards, generate `rope.bin`,
`engram-host.bin` and the AOT cubins with the Jayleaton tools, and put them in
the mount paths below.  `scripts/run-zig.sh <rank> --check-aot` from the
upstream recipe is the asset gate.

```yaml
model: marker/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw-Abliterated
container: tensorfold-dsv41-zig:local
cluster_config:
  resolved_model_path: /home/marker/.cache/huggingface/jayleaton-flash17-ablit
defaults:
  tf_family: jayleaton
  tensor_parallel: 2
  port: 8000
  served_model_name: deepseek-v4.1-flash
  tf_port: 29571
  assets_dir: /assets
  engram_dir: /engram
  prepared_dir: /prepared
  state_dir: /state
  cache_dir: /cache
executor_config:
  volumes:
    - /var/tmp/jayleaton/engram:/engram:ro
    - /var/tmp/jayleaton/assets:/assets:ro
    - /var/tmp/jayleaton/prepared:/prepared
    - /var/tmp/jayleaton/state:/state
    - /var/tmp/jayleaton/state/sessions:/sessions
    - /var/tmp/jayleaton/cache:/cache
env:
  TF_DSV41_CONTEXT: "1048576"
  TF_DSV41_BIAS_VL: /home/marker/.cache/huggingface/jayleaton-flash17-ablit
  # ... the rest of config/prod-zig-flash17.env.example
```

The model id is only the recipe identity; `cluster_config.resolved_model_path`
is the pre-placed pack path and skips download/distribution.  Registering that
pack as a synthetic HF-cache entry (`model/marker/...` in `hf cache ls`) makes
the same files available in the container's HF-cache mount, but the engine reads
the pack via the explicit resolved path.  Because the Engram and prepared
directories are rank-specific, pass the head host first explicitly when the
cluster's default order is not the head's:

```bash
sparkrun run <recipe> --hosts <head>,<worker>
```

The engine ignores `dashboard`/`metrics_open`; use its own `/health`,
`/v1/models` and request log (`TF_DSV41_REQUEST_LOG`).  A canary is
`17 * 23` -> `391`; thinking is on by default via `TF_DSV41_THINKING=1`.

## Known limitations / non-goals

- No generic TP for arbitrary models.  Only `glm`, `nemotron`, the bertholomus DeepSeek
  fork (`dsv41`) and the Jayleaton Zig fork (`jayleaton`) are wired; other
  families run solo and a multi-node request is refused with an actionable
  message.
- >2 ranks upstream is unqualified; the DeepSeek fork's TP4 is image-pinned and
  experimental.
- No macOS/`--speed-up` (upstream-only) and no CPU execution.
- TensorFold moves fast (several releases a week; only the latest gets security
  fixes).  Mirror the runtime's `container:` override to a pinned image tag and
  bump it deliberately.
- The engine binds unauthenticated on `0.0.0.0`.  Set `api_key` (forwarded as
  `--api-key`) when more than the control machine can reach the host, exactly as
  with any other sparkrun runtime.
