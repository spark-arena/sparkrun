#!/usr/bin/env bash
# Build the TensorFold DeepSeek-V4.1 (TP2/TP4) image.
#
# Build on a DGX Spark (linux/arm64) — this compiles the pinned fork from source
# and cross-builds the GB10 target.  The build needs network access (the fork
# tarball and Zig wheel) and takes several minutes; the resulting image is
# self-contained.
#
#   docker/tensorfold-dsv41/build.sh --load
#
# Publishing is explicit; CI does it from a native arm64 runner (see
# .github/workflows/tensorfold-dsv41-image.yml).
set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")"
# shellcheck disable=SC1091
set -a; source ./image.env; set +a

MODE=build
PLATFORM="${PLATFORM:-linux/arm64}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --push) MODE=push ;;
        --load) MODE=load ;;
        --platform) shift; PLATFORM="$1" ;;
        --help|-h)
            awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$SELF"
            exit 0
            ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

ARGS=(buildx build --platform "$PLATFORM"
    --build-arg "TF_SHA=${TF_FORK_SHA}"
    --build-arg "CUDA_DEVEL_TAG=${CUDA_DEVEL_TAG}"
    --build-arg "CUDA_TAG=${CUDA_TAG}"
    --build-arg "NCCL_APT_VERSION=${NCCL_APT_VERSION}"
    --build-arg "ZIG_VERSION=${ZIG_VERSION}"
    --build-arg "TOKENIZERS_VERSION=${TOKENIZERS_VERSION}"
    --build-arg "TARGET_SM=${TARGET_SM}"
    --build-arg "TARGETARCH=${PLATFORM##*/}"
    -t "${IMAGE_REPO}:${IMAGE_TAG}"
    .)
if [ "$MODE" = push ]; then
    ARGS+=(--push)
else
    ARGS+=(--load)
fi

echo "==> ${MODE} ${IMAGE_REPO}:${IMAGE_TAG} for ${PLATFORM} (fork ${TF_FORK_SHA:0:7})"
docker "${ARGS[@]}"

if [ "$MODE" != push ]; then
    echo "==> verify"
    docker run --rm "${IMAGE_REPO}:${IMAGE_TAG}" bash -lc \
        '/opt/tensorfold-dsv41/bin/tensorfold-native --version && test -x /opt/tensorfold-dsv41/bin/tf-dsv41-lanes && echo followers-present'
fi
