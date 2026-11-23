# TensorFold

TensorFold ([ashhart/TensorFold](https://github.com/ashhart/TensorFold)) is a
Zig/CUDA native LLM server with an OpenAI-compatible API.  sparkrun supports the
upstream engine only: one runtime, one launch shape, no per-fork or per-model
families.

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

## Native TP: upstream rank placement

Upstream TensorFold exposes the rank flags generically on the CUDA server:

```text
tensorfold-native serve <model> --tp <world> --rank <r> --master <head>
  --master-port <port> ...
```

sparkrun sets `tensor_parallel` to the number of hosts (one GPU per Spark), puts
one rank on each host, and emits those flags for every rank.  Only rank 0 gets
the HTTP flags (`--host/--port/--name/--api-key`); the follower ranks run the
same engine without a listener.  The orchestrator starts the followers first,
then rank 0, and skips the head-port gate because the engine owns the join; the
launcher's endpoint watcher still owns readiness.

```yaml
defaults:
  tensor_parallel: 2
min_nodes: 2
max_nodes: 2
```

A recipe `command:` template is refused for TP because it cannot emit different
argv for the serving and follower ranks.  Rank options are not universal family
support: upstream decides which models accept `--tp` above one and refuses the
rest by name.  sparkrun does not second-guess that decision.

### Limits

- Upstream qualifies the family-specific worlds; sparkrun does not add a
  separate support matrix.  A model that refuses `--tp` fails at engine
  startup with the engine's own message.
- TP is not a speed multiplier yet on two GB10s (upstream issue #981); the
  reason to run it is capacity.
- Both ranks verify the checkpoint before serving; a shared NFS mount is the
  simplest way to guarantee they see the same one.
- The container needs the fabric: the Docker executor already uses host
  networking, privileged mode and the optional `/dev/infiniband` device; the
  runtime adds `IPC_LOCK` and unlimited memlock.  Set `NCCL_IB_GID_INDEX` etc.
  through cluster/recipe `env` for your fabric.

## Validation checklist for the hardware run

1. Build/publish the upstream image outside this repository; check
   `tensorfold-native --version`.
2. `sparkrun run <recipe> --dry-run` — confirm the rendered command has
   `--host 0.0.0.0 --port <port>`, and for TP the `--tp/--rank/--master` flags.
3. Solo: one Spark, then `/v1/models`, `sparkrun proxy`, `sparkrun cluster
   monitor` (`[tensorfold]` lines are colorized), `sparkrun logs`, `stop`.
4. NFS: an absolute `model:` present on the target; confirm no download and the
   identity mount.
5. TP: a model the upstream engine supports at that world; rank 1 must start
   before rank 0; both logs under the normal `sparkrun logs`; stop removes both
   containers.
6. Report the exact engine tag, CUDA/NCCL versions, model pack and revision.

## Known limitations / non-goals

- Only upstream TensorFold is supported.  Out-of-tree forks and their prepared
  bundles are not modelled by this runtime; when upstream serves a checkpoint,
  it runs through this same path.
- No macOS/`--speed-up` (upstream-only) and no CPU execution.
- TensorFold moves fast (several releases a week; only the latest gets security
  fixes).  Mirror the runtime's `container:` override to a pinned image tag and
  bump it deliberately.
- The engine binds unauthenticated on `0.0.0.0`.  Set `api_key` (forwarded as
  `--api-key`) when more than the control machine can reach the host, exactly as
  with any other sparkrun runtime.
