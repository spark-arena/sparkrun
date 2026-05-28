#!/bin/bash
# Install or update the vllm-tune backing engine on the remote host.
#
# Inputs (env vars, set by the caller before piping this script over SSH):
#   VLLM_TUNE_REPO   Git URL to clone (e.g. https://github.com/SeraphimSerapis/vllm-tune.git)
#   VLLM_TUNE_REF    Git ref to check out (tag, branch, or SHA)
#   VLLM_TUNE_DEST   Absolute install directory (e.g. $HOME/.cache/sparkrun/vllm-tune/<ref>)
#
# On success, prints the absolute path to the installed vllm-tune.sh on stdout
# and exits 0.  Idempotent: re-running with the same ref is a fetch + checkout,
# not a re-clone.
set -euo pipefail

: "${VLLM_TUNE_REPO:?VLLM_TUNE_REPO is required}"
: "${VLLM_TUNE_REF:?VLLM_TUNE_REF is required}"
: "${VLLM_TUNE_DEST:?VLLM_TUNE_DEST is required}"

command -v git >/dev/null 2>&1 || { echo "git not found on PATH" >&2; exit 127; }

mkdir -p "$(dirname "$VLLM_TUNE_DEST")"

if [[ -d "$VLLM_TUNE_DEST/.git" ]]; then
    # Existing clone — fetch the requested ref and reset to it.
    cd "$VLLM_TUNE_DEST"
    # Fetch by ref name (works for both branches and tags); fall back to a full
    # fetch if the ref isn't a remote-tracking name (e.g. an arbitrary SHA).
    if ! git fetch --depth 1 origin "$VLLM_TUNE_REF" 2>/dev/null; then
        git fetch --tags origin
    fi
    if git rev-parse --verify --quiet "FETCH_HEAD" >/dev/null; then
        git checkout -q --detach FETCH_HEAD
    else
        git checkout -q --detach "$VLLM_TUNE_REF"
    fi
else
    # Fresh install.
    if [[ -e "$VLLM_TUNE_DEST" ]]; then
        echo "Refusing to clone into non-empty non-git path: $VLLM_TUNE_DEST" >&2
        exit 1
    fi
    # --branch accepts tags and branch names but not arbitrary SHAs; if it
    # fails, fall back to clone-then-checkout.
    if ! git clone --quiet --depth 1 --branch "$VLLM_TUNE_REF" "$VLLM_TUNE_REPO" "$VLLM_TUNE_DEST" 2>/dev/null; then
        git clone --quiet "$VLLM_TUNE_REPO" "$VLLM_TUNE_DEST"
        cd "$VLLM_TUNE_DEST"
        git checkout -q --detach "$VLLM_TUNE_REF"
    fi
fi

chmod +x "$VLLM_TUNE_DEST/vllm-tune.sh" 2>/dev/null || true
chmod +x "$VLLM_TUNE_DEST"/*.sh 2>/dev/null || true

echo "$VLLM_TUNE_DEST/vllm-tune.sh"
