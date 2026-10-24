# TensorFold native server image

Wraps a pinned [TensorFold](https://github.com/ashhart/TensorFold) release
archive (Apache-2.0) in a CUDA runtime image for sparkrun's `tensorfold`
runtime.  The release archive is fetched from GitHub Releases and verified
against its published SHA-256 sidecar at build time.

| value | pin |
|---|---|
| engine | `TF_VERSION` in `image.env` |
| CUDA runtime base | `CUDA_TAG` |
| NCCL | `NCCL_APT_VERSION` (≥ 2.30 is required for native TP ranks) |

The image contains no model weights, Python or PyTorch; it is ~2 GB and only
needs the NVIDIA driver at run time.  `libnccl2` is raised above the CUDA base
image's held 2.28 package because TensorFold refuses older NCCL for multi-node
tensor parallelism.

## Build

```sh
docker/tensorfold/build.sh --load              # native arm64 on a Spark
PLATFORM=linux/amd64 docker/tensorfold/build.sh --load   # smoke build
```

`build.sh` sources `image.env`, builds `ghcr.io/spark-arena/tensorfold:<tag>`
and verifies `tensorfold-native --version`.  Publishing is explicit
(`--push`); CI publishes the multi-arch manifest on manual dispatch.

The resulting tag matches the DGX Spark platform default, so sparkrun uses the
locally built image without a registry and distributes it to cluster hosts.
