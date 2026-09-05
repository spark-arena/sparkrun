#!/bin/bash
set -uo pipefail
# Recipe-sourced, so pre-quoted control-side rather than interpolated as text.
# GGUF_QUANT is a *fragment* of a glob, so it becomes a variable rather than a
# quoted literal: "*'Q4_K_M'*.gguf" would match nothing.
REPO_ID={repo_id}
GGUF_QUANT={quant}
echo "Checking GGUF model cache for $REPO_ID (quant: $GGUF_QUANT)..."
# Rendered control-side by models.download.model_cache_path — see the note in
# model_sync.sh; the bash-side derivation was wrong for every ``org/model`` id.
CACHE_PATH="{cache_path}"
# Pre-quoted control-side; empty string when the entry is unpinned.
MODEL_REVISION={revision}

# sparkrun:include _hf_snapshots.sh

# With a pinned revision, only that snapshot counts — see model_sync.sh.
SNAPSHOT_DIRS=$(sparkrun_hf_snapshot_dirs "$CACHE_PATH" "$MODEL_REVISION")

# The directory is keyed by repo, so the quant still has to match a file.
GGUF_MATCH=""
while IFS= read -r SNAPSHOT_DIR; do
    [ -n "$SNAPSHOT_DIR" ] || continue
    GGUF_MATCH=$(find "$SNAPSHOT_DIR" -name "*$GGUF_QUANT*.gguf" -print -quit 2>/dev/null)
    if [ -n "$GGUF_MATCH" ]; then
        break
    fi
done <<<"$SNAPSHOT_DIRS"

if [ -n "$GGUF_MATCH" ]; then
    echo "GGUF model already cached: $GGUF_MATCH"
    exit 0
fi

# Positional params carry the optional --revision so the value is never
# interpolated into the command line as text.
if [ -n "$MODEL_REVISION" ]; then
    set -- --revision "$MODEL_REVISION"
else
    set --
fi

echo "Downloading GGUF model: $REPO_ID (quant: $GGUF_QUANT)..."
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$REPO_ID" --include "*$GGUF_QUANT*" "*mmproj*" "$@" --cache-dir "{cache}/hub"
elif command -v uvx &>/dev/null; then
    uvx hf download "$REPO_ID" --include "*$GGUF_QUANT*" "*mmproj*" "$@" --cache-dir "{cache}/hub"
else
    # No HuggingFace client. The installer URL is version-pinned in
    # sparkrun.core.tooling — never the unversioned one, which is whatever
    # Astral published this morning.
    echo "No HuggingFace client on this host; installing pinned uv {uv_version}..."
    if curl -LsSf "{uv_install_url}" | sh; then
        export PATH="{uv_bin_dir}:$PATH"
    fi
    if command -v uvx &>/dev/null; then
        uvx hf download "$REPO_ID" --include "*$GGUF_QUANT*" "*mmproj*" "$@" --cache-dir "{cache}/hub"
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
