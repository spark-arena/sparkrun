#!/bin/bash
set -uo pipefail

# Distribute an HF model cache directory from this host to target hosts via rsync.
# Placeholders filled by Python: {model_path}, {targets}, {ssh_opts}, {ssh_user}

MODEL_PATH="{model_path}"
TARGETS="{targets}"
SSH_OPTS="{ssh_opts}"
SSH_USER="{ssh_user}"

echo "Distributing model $MODEL_PATH to targets: $TARGETS"

FAILED=0
for TARGET in $TARGETS; do
    if [ -n "$SSH_USER" ]; then
        DEST="$SSH_USER@$TARGET:$MODEL_PATH/"
    else
        DEST="$TARGET:$MODEL_PATH/"
    fi
    echo "  Syncing $MODEL_PATH -> $TARGET ..."
    if rsync -az --mkpath --partial --links -e "ssh $SSH_OPTS" "$MODEL_PATH/" "$DEST"; then
        echo "  OK: $TARGET"
    else
        echo "  FAILED: $TARGET" >&2
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: $FAILED target(s) failed" >&2
    exit 1
fi
echo "Model distributed successfully to all targets"
