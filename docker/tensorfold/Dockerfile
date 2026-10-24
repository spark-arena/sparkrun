# TensorFold native server image for sparkrun.
#
# Upstream TensorFold ships a Zig release archive, not a container.  This image
# wraps the pinned archive for one architecture and adds the two runtime
# libraries the engine may dlopen:
#
#   NCCL          multi-node tensor parallelism (the release only *needs*
#                 libcuda.so.1 from the driver for solo serving; NCCL is loaded
#                 when native ranks are launched).  TensorFold 1.1.x refuses
#                 NCCL < 2.30, so the base image's held 2.28 package is upgraded
#                 explicitly here rather than inherited.
#   libibverbs    RoCE/RDMA gathers between GB10 nodes.
#
# cuBLASLt comes from the CUDA runtime base and is used by `--vision` and some
# projection paths.  No Python is installed: the engine is a Zig binary and the
# archive carries its grammar library.
#
# Build on the target architecture (Dockerfile has no emulation story):
#   docker build -t ghcr.io/spark-arena/tensorfold:1.1.3 docker/tensorfold
# The pinned inputs live in image.env and are also consumed by build.sh, the
# GitHub workflow, and tests/test_tensorfold_image.py.

ARG CUDA_TAG=13.0.1-runtime-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}

ARG TF_VERSION=1.1.3
# libnccl2 is held in the CUDA base at 2.28.x; --allow-change-held-packages
# is required to raise it, and the exact pinned version keeps the build
# reproducible across NVIDIA repo updates.  cuda13.4 is the newest packaging
# of NCCL 2.32.3 available from the ubuntu2404 CUDA repository; it is a
# userspace library and works with the 13.0 runtime base (verified by loading
# it without a GPU).
ARG NCCL_APT_VERSION=2.32.3-1+cuda13.4
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TENSORFOLD_NO_UPDATE_CHECK=1 \
    PATH=/opt/tensorfold/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends --allow-change-held-packages \
        ca-certificates \
        curl \
        libnccl2="${NCCL_APT_VERSION}" \
        libibverbs1 \
        ibverbs-providers \
    && rm -rf /var/lib/apt/lists/*

# The release archive is a flat tarball with bin/, lib/ and licenses; keep its
# directories together (the grammar library is looked up beside the binary).
RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) PLATFORM=linux-x86_64 ;; \
        arm64) PLATFORM=linux-aarch64 ;; \
        *) echo "TensorFold has no release archive for TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    BASE="https://github.com/ashhart/TensorFold/releases/download/v${TF_VERSION}/tensorfold-${TF_VERSION}-${PLATFORM}"; \
    curl -fsSL -o /tmp/tensorfold.tar.gz "${BASE}.tar.gz"; \
    curl -fsSL -o /tmp/tensorfold.sha256 "${BASE}.tar.gz.sha256"; \
    cd /tmp; \
    # The sidecar names the tarball, so verify in a directory with that name.
    cp tensorfold.tar.gz "tensorfold-${TF_VERSION}-${PLATFORM}.tar.gz"; \
    sed "s#tensorfold-${TF_VERSION}-${PLATFORM}.tar.gz#tensorfold.tar.gz#" \
        tensorfold.sha256 > tensorfold-local.sha256; \
    sha256sum -c tensorfold-local.sha256; \
    mkdir -p /opt; \
    tar -xzf tensorfold.tar.gz -C /opt; \
    mv "/opt/tensorfold-${TF_VERSION}-${PLATFORM}" /opt/tensorfold; \
    ln -s /opt/tensorfold/bin/tensorfold-native /usr/local/bin/tensorfold-native; \
    ln -s /opt/tensorfold/bin/tensorfold-native /usr/local/bin/tensorfold; \
    rm -f /tmp/tensorfold*; \
    /opt/tensorfold/bin/tensorfold-native --version

# The engine writes ~/.cache/tensorfold/{window-costs,packs,...} and has no XDG
# support.  Root's HOME is writable in a normal container; the explicit cache
# directory gives an operator somewhere to mount a persistent volume.
RUN mkdir -p /var/cache/tensorfold /root/.cache/tensorfold

LABEL org.opencontainers.image.title="sparkrun-tensorfold" \
      org.opencontainers.image.description="TensorFold native server for sparkrun" \
      org.opencontainers.image.source="https://github.com/spark-arena/sparkrun" \
      dev.sparkrun.tensorfold.version="${TF_VERSION}" \
      dev.sparkrun.nccl.version="${NCCL_APT_VERSION}"

# No ENTRYPOINT: sparkrun starts this image with `sleep infinity` and drives it
# with `docker exec`.  An entrypoint that consumed argv would break the exec
# contract (the same reason Executor.verify_command_passthrough exists).
CMD ["sleep", "infinity"]
