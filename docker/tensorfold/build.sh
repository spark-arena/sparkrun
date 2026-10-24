#!/usr/bin/env bash
# Build the sparkrun TensorFold image.
#
# Build on the architecture you intend to run: the release archive is
# per-architecture and this is the primary DGX Spark path (linux/arm64 /
# sm_121).  On a Spark, run it from the repository root:
#
#   docker/tensorfold/build.sh --load
#
# On an x86_64 workstation for a smoke build:
#
#   PLATFORM=linux/amd64 docker/tensorfold/build.sh --load
#
# Publishing is always explicit; CI does the multi-arch manifest from native
# runners (see .github/workflows/tensorfold-image.yml).
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
    --build-arg "TF_VERSION=${TF_VERSION}"
    --build-arg "CUDA_TAG=${CUDA_TAG}"
    --build-arg "NCCL_APT_VERSION=${NCCL_APT_VERSION}"
    --build-arg "TARGETARCH=${PLATFORM##*/}"
    -t "${IMAGE_REPO}:${IMAGE_TAG}"
    .)
if [ "$MODE" = push ]; then
    ARGS+=(--push)
else
    ARGS+=(--load)
fi

echo "==> ${MODE} ${IMAGE_REPO}:${IMAGE_TAG} for ${PLATFORM} (TensorFold ${TF_VERSION})"
docker "${ARGS[@]}"

if [ "$MODE" != push ]; then
    echo "==> verify"
    docker run --rm "${IMAGE_REPO}:${IMAGE_TAG}" tensorfold-native --version
fi
