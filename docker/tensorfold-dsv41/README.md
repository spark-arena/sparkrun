# TensorFold DeepSeek-V4.1 community image

This directory builds the **experimental** DeepSeek-V4.1-Flash TP2/TP4 engine
image for the `tensorfold-dsv41` sparkrun runtime.  It is not the upstream
TensorFold release:

| component | source |
|---|---|
| engine | [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold), branch `deepseek-v41-zig`, pinned in `image.env` |
| patches | [tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks), `runs/2026-10-18-1017-tensorfold-tp2/patches/` |
| build recipe | same deployment's `launch/build.sh` (Zig 0.17.0, CUDA 13, sm_121) |

Both upstream projects and the patches are Apache-2.0; the fork is a
derivative work of TensorFold and keeps its license/notices.  The image is
compiled from source at build time and contains no model weights.

## Build

On a DGX Spark (arm64), from the repository root:

```sh
docker/tensorfold-dsv41/build.sh --load
```

The resulting local tag matches the platform default
(`ghcr.io/spark-arena/tensorfold-dsv41:3ff3b2a`), so a recipe can reference it
without a registry.  Cross-building from an x86 workstation is possible for
syntax/compile checks but produces x86 binaries; do not run those on a Spark.

## What is inside

- `/opt/tensorfold-dsv41/bin/tensorfold-native` — serving binary for rank 0
- `/opt/tensorfold-dsv41/bin/tf-dsv41-lanes` — follower rank binary
- `/opt/tensorfold-dsv41/bin/tf-dsv41-rank-cache` — CPU-only rank-cache writer
- `/opt/tensorfold-dsv41/bin/node_inputs.sh` — per-rank cache/token-map helper
- NCCL ≥ 2.30 and the verbs providers for RoCE

Model data is **not** in the image.  See `docs/TENSORFOLD.md` for the prepared
bundle layout and `scripts/tensorfold-dsv41-prepare.sh` for the one-time
bundle preparation.
