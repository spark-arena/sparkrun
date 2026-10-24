# TensorFold DeepSeek-V4.1 community build (TP2/TP4) for sparkrun.
#
# This is NOT the upstream release.  The released TensorFold native CUDA
# binary does not serve DeepSeek-V4.1 yet (upstream PR #827 is open), so this
# image pins the community fork from bertholomus/TensorFold at the commit the
# tonyd2wild DGX Spark TP2/TP4 deployment used, plus the three resolution
# patches from that deployment:
#
#   TensorFold fork : bertholomus/TensorFold @ 3ff3b2ac1d71a789cb930068c46d9abed025eefa
#                     (branch deepseek-v41-zig, Apache-2.0)
#   patches         : tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks
#                     runs/2026-10-18-1017-tensorfold-tp2/patches (Apache-2.0)
#
# The engine is compiled from source inside the builder stage.  Build this on
# an arm64 host (a DGX Spark) for the GB10 target:
#
#   docker/tensorfold-dsv41/build.sh --load
#
# Nothing model-specific is baked into the image: the weights, per-rank caches,
# kernel kit and Engram shards live in the prepared bundle documented in
# docs/TENSORFOLD.md.

ARG CUDA_DEVEL_TAG=13.0.1-devel-ubuntu24.04
ARG CUDA_TAG=13.0.1-runtime-ubuntu24.04

# Zig 0.17.0 from its official PyPI wheel: the ziglang.org download site is
# frequently slow/throttled, while the wheel is version-pinned and available
# for both build architectures.  Copying the whole package keeps the compiler's
# lib/ directory beside the executable, which Zig resolves relative to itself.
FROM python:3.12-slim AS zigstage
ARG ZIG_VERSION=0.17.0
RUN pip install --no-cache-dir "ziglang==${ZIG_VERSION}"

FROM nvidia/cuda:${CUDA_DEVEL_TAG} AS builder

ARG TF_SHA=3ff3b2ac1d71a789cb930068c46d9abed025eefa
ARG TARGETARCH
ARG TARGET_SM=121

ENV DEBIAN_FRONTEND=noninteractive \
    ZIG_GLOBAL_CACHE_DIR=/tmp/zig-cache \
    ZIG_LOCAL_CACHE_DIR=/tmp/zig-local-cache \
    HOME=/tmp

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        patch \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=zigstage /usr/local/lib/python3.12/site-packages/ziglang /opt/zig
RUN /opt/zig/zig version

# Source at the pinned fork commit plus the three deployment patches.
COPY patches/ /tmp/patches/
RUN set -eux; \
    curl -fsSL -o /tmp/src.tar.gz "https://codeload.github.com/bertholomus/TensorFold/tar.gz/${TF_SHA}"; \
    mkdir -p /src; \
    tar -xzf /tmp/src.tar.gz -C /src --strip-components=1; \
    cd /src; \
    for p in /tmp/patches/*.patch; do \
        echo "== applying $(basename "$p")"; \
        patch -p1 --batch --forward --fuzz=0 < "$p"; \
    done; \
    rm -rf /tmp/patches /tmp/src.tar.gz

# `install` writes the family tools to zig-out/bin and tensorfold-native to
# zig-out/native/bin.  The DeepSeek TP2/TP4 lane needs all of them.
RUN set -eux; \
    cd /src; \
    case "${TARGETARCH}" in \
        arm64) TARGET_TRIPLE=aarch64-linux-gnu.2.28 ;; \
        amd64) TARGET_TRIPLE=x86_64-linux-gnu.2.28 ;; \
        *) exit 1 ;; \
    esac; \
    /opt/zig/zig build install native \
        -Dtarget="${TARGET_TRIPLE}" \
        -Doptimize=ReleaseFast \
        -Dnvcc=/usr/local/cuda/bin/nvcc \
        -Dsm="${TARGET_SM}"; \
    test -x zig-out/bin/tf-dsv41-lanes; \
    test -x zig-out/native/bin/tensorfold-native; \
    test -x zig-out/bin/tf-dsv41-rank-cache

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_TAG}

ARG TF_SHA=3ff3b2ac1d71a789cb930068c46d9abed025eefa
ARG NCCL_APT_VERSION=2.32.3-1+cuda13.4
ARG TOKENIZERS_VERSION=0.21.1

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TENSORFOLD_NO_UPDATE_CHECK=1 \
    PATH=/opt/tensorfold-dsv41/bin:/opt/tensorfold-dsv41/py/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends --allow-change-held-packages \
        ca-certificates \
        libnccl2="${NCCL_APT_VERSION}" \
        libibverbs1 \
        ibverbs-providers \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# The one-time preparation script (scripts/tensorfold-dsv41-prepare.sh) runs
# node_inputs.sh inside this image; its token-map step needs tokenizers.  The
# serving process never imports Python, but keeping the prep dependencies in
# the same image is what makes preparation a single command on every node.
RUN python3 -m venv /opt/tensorfold-dsv41/py \
    && /opt/tensorfold-dsv41/py/bin/pip install --no-cache-dir "tokenizers==${TOKENIZERS_VERSION}" \
    && /opt/tensorfold-dsv41/py/bin/python3 -c 'import tokenizers; print(tokenizers.__version__)'

COPY --from=builder /src/zig-out/bin/ /opt/tensorfold-dsv41/bin/
COPY --from=builder /src/zig-out/native/bin/tensorfold-native /opt/tensorfold-dsv41/bin/tensorfold-native
COPY --from=builder /src/tools/dsv41-zig/inputs/node_inputs.sh /opt/tensorfold-dsv41/bin/node_inputs.sh
COPY --from=builder /src/tools/dsv41-zig/inputs/make_token_map.py /opt/tensorfold-dsv41/bin/make_token_map.py

RUN set -eux; \
    chmod +x /opt/tensorfold-dsv41/bin/node_inputs.sh /opt/tensorfold-dsv41/bin/make_token_map.py; \
    ln -sf /opt/tensorfold-dsv41/bin/tensorfold-native /usr/local/bin/tensorfold-native; \
    ln -sf /opt/tensorfold-dsv41/bin/tensorfold-native /usr/local/bin/tensorfold; \
    mkdir -p /var/cache/tensorfold /root/.cache/tensorfold; \
    /opt/tensorfold-dsv41/bin/tensorfold-native --version

LABEL org.opencontainers.image.title="sparkrun-tensorfold-dsv41" \
      org.opencontainers.image.description="TensorFold DeepSeek-V4.1 TP2/TP4 community build for sparkrun" \
      org.opencontainers.image.source="https://github.com/spark-arena/sparkrun" \
      dev.sparkrun.tensorfold.fork="bertholomus/TensorFold@${TF_SHA}" \
      dev.sparkrun.nccl.version="${NCCL_APT_VERSION}"

CMD ["sleep", "infinity"]
