# TensorFold

TensorFold ([ashhart/TensorFold](https://github.com/ashhart/TensorFold)) is a
Zig/CUDA native LLM server with an OpenAI-compatible API.  Sparkrun ships two
runtimes for it:

| runtime | engine | shapes | status |
|---|---|---|---|
| `tensorfold` | upstream release, pinned by the image | solo (one Spark); native TP2 for GLM-5.3-Flash and Nemotron 3.5 Super VL | primary path |
| `tensorfold-dsv41` | community fork `bertholomus/TensorFold` + the tonyd2wild TP2/TP4 patches | TP2 / TP4 across 2 or 4 Sparks | **experimental** |

Both are gated to `gb10` hosts (`requires_capability`).  TensorFold's released
CUDA fatbins target sm_80/86/120/121, but only the GB10 (sm_121) path is
qualified by sparkrun today; another accelerator needs its own image and
qualification.

## Why an image

Upstream publishes release archives, not containers.  The sparkrun image wraps
the pinned archive and adds the libraries the engine may `dlopen`:

- **NCCL ≥ 2.30** — native tensor-parallel ranks.  The engine refuses older
  NCCL; the CUDA runtime base holds 2.28, so the image raises it explicitly.
- **libibverbs / verbs providers** — RoCE/RDMA gathers between Sparks.
- cuBLASLt from the CUDA runtime base — `--vision` and some projection paths.

The DeepSeek image additionally compiles the fork from source (Zig 0.17.0) and
installs a small Python venv only for the rank-cache preparation helper.  The
serving process itself uses no Python.

Build the image on a Spark (arm64) and tag it with the platform default so no
registry is needed.  The control machine can be a Spark or a separate arm64
machine; sparkrun uses a locally present image without contacting a registry
and then distributes it to the cluster hosts (`docker save | ssh docker load`):

```sh
docker/tensorfold/build.sh --load
# = ghcr.io/spark-arena/tensorfold:1.1.3

docker/tensorfold-dsv41/build.sh --load
# = ghcr.io/spark-arena/tensorfold-dsv41:3ff3b2a
```

`image.env` in each directory is the one pinned source of truth; the drift
guards in `tests/test_tensorfold_image.py` keep the platform defaults in step.

## Solo: one Spark

```sh
sparkrun run recipes/tensorfold-qwen3.8-27b.yaml --hosts spark-a
sparkrun run recipes/tensorfold-nemotron-3.5-lightning.yaml --hosts spark-a
```

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
declares all three to the proxy/readiness layers and uses the engine's
`/health` endpoint.

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
| `vision`, `no_drafts`, `loop_guard`, `metrics_open` | presence flags | |
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
- The path must be the **complete checkpoint directory** (HF snapshot layout or
  the engine's expected model directory), not a partial copy.  For TensorFold's
  per-rank DeepSeek lane the bundle layout is below.

A shared Hugging Face cache on NFS is the other supported shape: set the
cluster's `cache_dir` to the shared mount and disable distribution
(`distribution.model.enabled: false` or `skip_fan_out: true`), and the engine
reads the mounted cache (`HF_HOME`/`HF_HUB_CACHE` are set by the runtime).
Prefer the absolute-path form for TensorFold: it survives SHA-pinned downloads
that leave no `refs/` entry in the cache.

## TP2: GLM-5.3-Flash and Nemotron 3.5 Super VL

TensorFold's native CUDA build has **no generic `--tp` flag**.  Each engine
family places its own ranks through environment variables, only rank 0 serves
HTTP, and the follower ranks must already be running when rank 0 starts:

| family | rank/world variables | extra | qualified |
|---|---|---|---|
| `glm` | `TF_GLM_RANK` / `TF_GLM_WORLD` | `NCCL_COMM_ID=<head_ip>:<port>` | 2 ranks (EXL3/TR3); NVFP4 contributed |
| `nemotron` | `TF_NEMOTRON_RANK` / `TF_NEMOTRON_WORLD` | `NCCL_COMM_ID`, NCCL ≥ 2.30 | 2 ranks for Super VL 120B |

Set the family in `defaults:` and let `--tp` place the hosts:

```yaml
runtime: tensorfold
defaults:
  tensor_parallel: 2
  tf_family: glm          # or nemotron
min_nodes: 2
max_nodes: 2
```

Then `sparkrun run recipes/tensorfold-glm-5.3-flash-tp2.yaml --tp 2` (or just
the recipe, which declares it) starts rank 1 first, then rank 0; the head-port
gate is skipped because the engine owns the join, and the launcher watches the
serve endpoint afterwards.  The runtime reserves a contiguous
`NCCL_COMM_ID` port pair (`port` and `port + 1` — some families use the second
for their follower control channel).

Recipe defaults are passed to both ranks; only rank 0 gets
`--host/--port/--name/--api-key`.  A recipe `command:` template is refused for
TP because it cannot emit different argv for the serving and follower ranks.

### Limits to know before planning a deployment

- **Two ranks only are release-qualified.**  GLM's engine accepts a world up to
  16, and a three-node NVFP4 run is reported, but sparkrun treats >2 as
  experimental and surfaces a warning.  Nemotron's world must divide the expert
  shares.
- **TP is not a speed multiplier yet.**  On two GB10s, upstream issue #981
  reports TP=2 serial decode roughly equal to TP=1; the reason to run TP is
  capacity (a larger model, more headroom), not throughput.  DeepSeek EXL3
  packs are the exception where the fork shows a real gain.
- Both ranks verify the checkpoint (layout and content hashes) before serving.
  They must see the same checkpoint; a shared NFS mount is the simplest way to
  guarantee that.
- The container needs the fabric: sparkrun's Docker executor already uses host
  networking, privileged mode and the optional `/dev/infiniband` device; the
  runtime adds `IPC_LOCK` and unlimited memlock.  Set `NCCL_IB_GID_INDEX` etc.
  through cluster/recipe `env` for your fabric; `rdma` environment from the
  normal IB detection also flows through.

## `tensorfold-dsv41`: DeepSeek-V4.1 TP2/TP4 (experimental)

The released TensorFold binary does **not** serve DeepSeek-V4.1.  This runtime
wraps the community fork that does, and is deliberately a separate runtime:

- rank 0 runs `tensorfold-native serve` with `TF_TP_*`;
- followers run a different binary, `tf-dsv41-lanes`, with positional argv;
- the engine reads a **prepared bundle**, not the raw checkpoint: per-rank
  weight caches, a kernel kit and the Engram tables.

### Bundle layout

`model:` is the absolute bundle root, identical on every node:

```text
<bundle>/model/                 config/tokenizer/index/encoding/, vision shards, chat_template.jinja
<bundle>/cache/rank-cache/      rank<N>of<W>-<hash>.bin
<bundle>/cache/dsv41_token_map.json
<bundle>/kit/                   kernel kit (aot/, MANIFEST, ...)
<bundle>/engram/                shards 47 and 48 of the base checkpoint
```

### Prepare once

```sh
scripts/tensorfold-dsv41-prepare.sh \
    --pack   /mnt/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw \
    --bundle /mnt/nfs/sparkrun/tensorfold-dsv41-flash-tp2 \
    --world  2 \
    --kit    /opt/tensorfold-recipes/kit-v0.6.1 \
    --engram-source /mnt/models/DeepSeek-V4.1-Flash-1017
```

The script assembles the model directory, copies the kit and Engram shards, and
runs the CPU-only rank-cache writer in the image (about six minutes per rank).
Plan for ~2×106 GB of rank caches on TP2 / ~4×69 GB on TP4, plus ~203 GB of
Engram, and a 196–240 GB pack as input.  On a shared NFS bundle every node sees
the caches; Engram is latency-sensitive, so copy `<bundle>/engram` to local NVMe
per worker and set `defaults.engram_dir` to that local path.

The bundle path is an absolute `model:` and therefore needs a trusted recipe
(local or `--trust`).  Edit the model path and `rdma_hca`/`net_if` in
`recipes/tensorfold-dsv41-flash-tp2.yaml` or `...-tp4.yaml` before running.

### Runtime knobs

| key | meaning |
|---|---|
| `tensor_parallel` | 2 or 4 (required; single-rank DeepSeek is not qualified) |
| `max_model_len` | serve `--context` (1M in the reference lane) |
| `max_num_seqs` | serve `--parallel` and the follower `--streams` default |
| `streams`, `pool` | follower `--streams` / `--pool` (and shared KV positions) |
| `round_rows`, `window`, `tf_2d` | TP4 2D-split settings (`TF_DS_*`); defaults: 48 / 2097152 / `parity` |
| `rdma_hca`, `net_if`, `rdma_gid_index`, `nccl_ib_gid_index` | fabric env |
| `cuda_kernels` | `TENSORFOLD_CUDA_KERNELS` (default `<bundle>/kit/aot`) |
| `tf_env` | extra `TF_*`/`NCCL_*` assignments, a mapping |
| `served_model_name`, sampling keys | same as upstream |

The fork's patch set pins request sampling (`TF_SEED_RANDOM`, `TF_TOP_K_MAX`)
through the image; recipes expose the normal sampling defaults.

## Validation checklist for the hardware run

1. `docker/tensorfold/build.sh --load` (and the dsv41 build if used); check
   `tensorfold-native --version`.
2. `sparkrun run <recipe> --dry-run` — confirm the rendered command has
   `--host 0.0.0.0 --port <port>`, and for TP the rank env/`NCCL_COMM_ID`.
3. Solo: run one Spark, then `curl http://<host>:<port>/v1/models`, start
   `sparkrun proxy`, and check `sparkrun cluster monitor` colorizes
   `[tensorfold]` lines (the colorizer already shipped before this runtime).
4. NFS: run with an absolute `model:` on a path present on the target; confirm
   no download/distribution phase and that the path is identity-mounted.
5. TP2: run `tensorfold-glm-5.3-flash-tp2` and
   `tensorfold-nemotron-3.5-super-vl-tp2`; rank 1 should be started before rank
   0; both ranks' logs are under the normal `sparkrun logs`.  Stop the launch
   and confirm both containers are removed.
6. DeepSeek: build the bundle with the prep script, run TP2 first (smaller
   world), verify `/v1/models` and a chat completion, then stop/restart and
   check the caches are reused (no repeated rank-cache generation).
7. Report the exact engine tag, NGC/CUDA base, NCCL version, model pack and
   revision — the engine's cadence is fast and only the latest release gets
   security fixes.

## Known limitations / non-goals

- No generic TP for arbitrary models.  Only `glm` and `nemotron` (upstream) and
  the DeepSeek fork are wired; other families run solo and a multi-node request
  is refused with an actionable message.
- No >2-rank qualification upstream; the DeepSeek fork supports TP4 in its
  reference deployment, but sparkrun's support is experimental and image-pinned.
- No `--speed-up` (macOS-only) and no CPU execution.
- TensorFold is a moving target: several releases a week, only the latest gets
  security fixes.  The image version is pinned; bump it deliberately.
- The engine binds unauthenticated on `0.0.0.0`.  Set `defaults.api_key`
  (forwarded as `--api-key`) when more than the control machine can reach the
  host, exactly as with any other sparkrun runtime.
