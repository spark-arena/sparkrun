#!/bin/bash
set -uo pipefail
# Recipe-sourced, so pre-quoted control-side rather than interpolated as text:
# a model id is registry content and reaches a script run on every host.
MODEL_ID={model_id}
echo "Checking model cache for $MODEL_ID..."
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
    echo "Model already cached: $MODEL_ID"
    exit 0
fi

# Positional params carry the optional --revision so the value is never
# interpolated into the command line as text.
if [ -n "$MODEL_REVISION" ]; then
    set -- --revision "$MODEL_REVISION"
else
    set --
fi

echo "Downloading model: $MODEL_ID..."
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$MODEL_ID" "$@" --cache-dir "{cache}/hub"
elif command -v uvx &>/dev/null; then
    uvx hf download "$MODEL_ID" "$@" --cache-dir "{cache}/hub"
else
    # No HuggingFace client. The installer URL is version-pinned in
    # sparkrun.core.tooling — never the unversioned one, which is whatever
    # Astral published this morning.
    echo "No HuggingFace client on this host; installing pinned uv {uv_version}..."
    if curl -LsSf "{uv_install_url}" | sh; then
        export PATH="{uv_bin_dir}:$PATH"
    fi
    if command -v uvx &>/dev/null; then
        uvx hf download "$MODEL_ID" "$@" --cache-dir "{cache}/hub"
    else
        echo "ERROR: no HuggingFace client on $(hostname), and uv {uv_version} could not be installed." >&2
        echo "       This host may have no outbound network access." >&2
        echo "  Fix by any of:" >&2
        echo "    - install 'huggingface-cli' or 'uv' on this host" >&2
        echo "    - download on the control machine instead: transfer_mode 'local' or 'push'" >&2
        echo "    - pre-place the weights in {cache}/hub (they are detected and reused)" >&2
        exit 1
    fi
fi
