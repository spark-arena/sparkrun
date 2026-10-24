#!/usr/bin/env bash
# Prepare the TensorFold DeepSeek-V4.1 (TP2/TP4) bundle for sparkrun.
#
# This is the one-time, model-specific preparation that the upstream TensorFold
# runtime does not need: the community DeepSeek lane serves from per-rank caches
# and a kernel kit, not from the raw EXL3 checkpoint.  The output is a bundle
# directory that the `tensorfold-dsv41` runtime mounts read-only on every node:
#
#   <bundle>/model/                 config/tokenizer/index/encoding/, vision
#                                   shards and the chat template
#   <bundle>/cache/rank-cache/      rank<R>of<W>-<hash>.bin (one per rank)
#   <bundle>/cache/dsv41_token_map.json
#   <bundle>/kit/                   kernel kit from the reference recipe
#   <bundle>/engram/                shards 47 and 48 of the base checkpoint
#
# Rank-cache generation is CPU-only and reads the checkpoint tensor by tensor;
# expect about six minutes per rank and a bundle sized roughly:
#   TP2, EXL3 2.9 bpw:  ~2 x 106 GB rank cache + ~203 GB Engram
#   TP4, EXL3 3.5 bpw:  ~4 x 69 GB rank cache + ~203 GB Engram
# The pack itself is only an input and is not copied into the bundle.
#
# Usage:
#   scripts/tensorfold-dsv41-prepare.sh \
#       --pack /mnt/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw \
#       --bundle /mnt/nfs/sparkrun/tensorfold-dsv41-flash \
#       --world 2 \
#       --kit /opt/tensorfold-recipes/kit-v0.6.1 \
#       --engram-source /mnt/models/DeepSeek-V4.1-Flash-1017 \
#       --image ghcr.io/spark-arena/tensorfold-dsv41:3ff3b2a
#
# Run it once on any host that can read the pack and write the bundle.  On a
# shared NFS bundle every node then sees the same path; Engram is latency
# sensitive, so the reference deployment copies `<bundle>/engram` to local NVMe
# on each worker and overrides `engram_dir` in the recipe.
set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
REPO_ROOT="$(cd "$(dirname "$SELF")/.." && pwd)"
DEFAULT_CHAT_TEMPLATE="$REPO_ROOT/docker/tensorfold-dsv41/chat_template.jinja"

PACK=
BUNDLE=
WORLD=
KIT=
ENGRAM_SOURCE=
IMAGE="${TF_DSV41_IMAGE:-ghcr.io/spark-arena/tensorfold-dsv41:3ff3b2a}"
CHAT_TEMPLATE="$DEFAULT_CHAT_TEMPLATE"
SKIP_ENGRAM=0
SKIP_CACHES=0

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$SELF"
    exit "${1:-2}"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --pack) shift; PACK=${1:?--pack needs a directory} ;;
        --bundle) shift; BUNDLE=${1:?--bundle needs a directory} ;;
        --world) shift; WORLD=${1:?--world needs 2 or 4} ;;
        --kit) shift; KIT=${1:?--kit needs a directory} ;;
        --engram-source) shift; ENGRAM_SOURCE=${1:?--engram-source needs a directory} ;;
        --chat-template) shift; CHAT_TEMPLATE=${1:?--chat-template needs a file} ;;
        --image) shift; IMAGE=${1:?--image needs a tag} ;;
        --skip-engram) SKIP_ENGRAM=1 ;;
        --skip-caches) SKIP_CACHES=1 ;;
        --help|-h) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
    shift
done

[ -n "$PACK" ] || { echo "--pack is required" >&2; exit 2; }
[ -n "$BUNDLE" ] || { echo "--bundle is required" >&2; exit 2; }
[ "$WORLD" = 2 ] || [ "$WORLD" = 4 ] || { echo "--world must be 2 or 4" >&2; exit 2; }
[ -n "$KIT" ] || { echo "--kit is required (the reference recipe's kit/ directory)" >&2; exit 2; }

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo "==> $*"; }

[ -d "$PACK" ] || die "pack directory not found: $PACK"
[ -f "$PACK/config.json" ] || die "$PACK/config.json missing (is this the EXL3 checkpoint?)"
[ -f "$PACK/tokenizer.json" ] || die "$PACK/tokenizer.json missing"
[ -d "$KIT" ] || die "kit directory not found: $KIT"
[ -f "$CHAT_TEMPLATE" ] || die "chat template not found: $CHAT_TEMPLATE"
command -v docker >/dev/null 2>&1 || die "docker is required to run the rank-cache writer"

docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "image '$IMAGE' is not present locally; build it with docker/tensorfold-dsv41/build.sh --load"

note "bundle: $BUNDLE (world $WORLD)"
mkdir -p "$BUNDLE/model" "$BUNDLE/cache/rank-cache" "$BUNDLE/kit" "$BUNDLE/engram"

# --- model runtime directory ------------------------------------------------
# The engine needs the checkpoint's metadata, tokenizer, template and vision
# shard(s); the weights themselves are read from the per-rank caches.  The
# reference deployment copies every top-level JSON plus the 43/44 vision shards
# (the 1017 pack ships its vision tower in shard 43 on two ranks, 43+44 on four).
note "assembling model metadata + vision shards"
for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json \
         model.safetensors.index.json quantization_config.json exl3_k_map.json generation_config.json; do
    [ -f "$PACK/$f" ] && cp -f "$PACK/$f" "$BUNDLE/model/"
done
[ -d "$PACK/encoding" ] && cp -R "$PACK/encoding" "$BUNDLE/model/"
for pattern in 'model-00043-of-*.safetensors' 'model-00044-of-*.safetensors'; do
    for shard in "$PACK"/$pattern; do
        [ -f "$shard" ] && cp -f "$shard" "$BUNDLE/model/"
    done
done
cp -f "$CHAT_TEMPLATE" "$BUNDLE/model/chat_template.jinja"
[ -f "$BUNDLE/model/chat_template.jinja" ] || die "chat template was not copied"

# --- kit --------------------------------------------------------------------
note "copying kernel kit"
cp -R "$KIT/." "$BUNDLE/kit/"
[ -d "$BUNDLE/kit/aot" ] || echo "WARNING: $BUNDLE/kit/aot is absent; set cuda_kernels in the recipe if your kit stores the AOT set elsewhere" >&2

# --- Engram tables ----------------------------------------------------------
if [ "$SKIP_ENGRAM" = 0 ]; then
    [ -n "$ENGRAM_SOURCE" ] || die "--engram-source is required unless --skip-engram is given"
    note "copying Engram shards 47/48 from $ENGRAM_SOURCE"
    copied=0
    for shard in "$ENGRAM_SOURCE"/model-00047-of-*.safetensors "$ENGRAM_SOURCE"/model-00048-of-*.safetensors; do
        [ -f "$shard" ] || continue
        cp -f "$shard" "$BUNDLE/engram/"
        copied=$((copied + 1))
    done
    [ "$copied" -ge 2 ] || die "expected shards 47 and 48 in $ENGRAM_SOURCE; use --skip-engram if the bundle already has them"
else
    note "skipping Engram copy"
fi
[ -d "$BUNDLE/engram" ] && [ -n "$(ls -A "$BUNDLE/engram" 2>/dev/null)" ] || [ "$SKIP_ENGRAM" = 1 ] \
    || echo "WARNING: $BUNDLE/engram is empty; the lane will fail without the Engram tables" >&2

# --- per-rank caches --------------------------------------------------------
if [ "$SKIP_CACHES" = 0 ]; then
    note "building per-rank caches (CPU only, ~6 min/rank)"
    for rank in $(seq 0 $((WORLD - 1))); do
        note "rank $rank of $WORLD"
        docker run --rm --network none \
            --user "$(id -u):$(id -g)" -e HOME=/tmp \
            -v "$PACK:/pack:ro" \
            -v "$BUNDLE/cache:/cache" \
            --entrypoint /opt/tensorfold-dsv41/bin/node_inputs.sh \
            "$IMAGE" /pack /cache "$rank" "$WORLD"
    done
    expected=$(find "$BUNDLE/cache/rank-cache" -maxdepth 1 -name 'rank*of*' | wc -l)
    [ "$expected" -ge "$WORLD" ] || die "expected $WORLD rank caches, found $expected"
else
    note "skipping rank-cache generation"
fi

[ -f "$BUNDLE/cache/dsv41_token_map.json" ] || echo "WARNING: $BUNDLE/cache/dsv41_token_map.json is missing; run without --skip-caches" >&2

note "bundle summary"
du -sh "$BUNDLE"/* 2>/dev/null || true
ls -la "$BUNDLE/cache/rank-cache" | tail -n +2
cat <<EOF

Bundle ready: $BUNDLE

On every node the bundle must live at the same absolute path (a shared NFS
mount is the supported layout).  For lowest-latency Engram access, copy
$BUNDLE/engram to node-local disk and point the recipe's engram_dir at it.
Then edit a recipe under recipes/ (model: must be this absolute path) and run:

  sparkrun run recipes/tensorfold-dsv41-flash-tp2.yaml --hosts <spark-a>,<spark-b>

The absolute model: path is a trusted-recipe feature: a local recipe or
--trust is required.
EOF
