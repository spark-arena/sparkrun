#!/bin/bash
set -uo pipefail
echo "Checking model cache for {model_id}..."
# Rendered control-side by models.download.model_cache_path — the one place
# that knows the HF models--org--name mangling.  Deriving it here again in bash
# is how the two drifted (issue #291): the old tr-based slash replacement
# emitted a single hyphen, so every org/model id missed its own cache and fell
# through to the download path.
CACHE_PATH="{cache_path}"
# Pre-quoted control-side; empty string when the entry is unpinned.
MODEL_REVISION={revision}

# sparkrun:include _hf_snapshots.sh

# Which snapshots count as "this model".  With a pinned revision that is only
# the matching one — scanning every snapshot is what let a host serve some
# other revision and report a cache hit.
SNAPSHOT_DIRS=$(sparkrun_hf_snapshot_dirs "$CACHE_PATH" "$MODEL_REVISION")

# Check for actual weight files (not just config.json from VRAM auto-detect).
# The scan stays recursive: some repos shard weights into a subdirectory, and
# the control-side peer's top-level-only glob would report those as uncached
# and re-download them on every launch.
FOUND_WEIGHTS=false
while IFS= read -r SNAPSHOT_DIR; do
    [ -n "$SNAPSHOT_DIR" ] || continue
    for pattern in "*.safetensors" "*.bin" "*.pt" "*.gguf"; do
        if find "$SNAPSHOT_DIR" -name "$pattern" -print -quit 2>/dev/null | grep -q .; then
            FOUND_WEIGHTS=true
            break
        fi
    done
    if [ "$FOUND_WEIGHTS" = true ]; then
        break
    fi
done <<<"$SNAPSHOT_DIRS"

if [ "$FOUND_WEIGHTS" = true ]; then
    echo "Model already cached: {model_id}"
    exit 0
fi

# Positional params carry the optional --revision so the value is never
# interpolated into the command line as text.
if [ -n "$MODEL_REVISION" ]; then
    set -- --revision "$MODEL_REVISION"
else
    set --
fi

echo "Downloading model: {model_id}..."
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "{model_id}" "$@" --cache-dir "{cache}/hub"
elif command -v uvx &>/dev/null; then
    uvx hf download "{model_id}" "$@" --cache-dir "{cache}/hub"
else
    echo "Installing uv for model download access..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uvx &>/dev/null; then
        uvx hf download "{model_id}" "$@" --cache-dir "{cache}/hub"
    else
        echo "ERROR: failed to install uv; cannot download model on this host" >&2
        exit 1
    fi
fi
