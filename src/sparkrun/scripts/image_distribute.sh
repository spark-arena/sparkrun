#!/bin/bash
set -uo pipefail

# Distribute a Docker image from this host to target hosts via docker save/load.
# Placeholders filled by Python: {image}, {targets}, {ssh_opts}, {ssh_user}, {max_parallel}
#
# NOTE: this file is consumed via Python str.format(); it must contain NO
# literal curly-brace characters (shell functions are therefore avoided in
# favor of backgrounded subshells, which use parentheses).

IMAGE="{image}"
TARGETS="{targets}"
SSH_OPTS="{ssh_opts}"
SSH_USER="{ssh_user}"
MAX_PARALLEL="{max_parallel}"

echo "Distributing image $IMAGE to targets: $TARGETS"

# Each transfer writes its exit status to a per-target status file so the
# parent can detect failures after `wait` (background jobs don't propagate
# non-zero exit codes to the script otherwise).
STATUS_DIR="$(mktemp -d)"
trap 'rm -rf "$STATUS_DIR"' EXIT

# Bounded concurrency: keep at most MAX_PARALLEL transfers in flight.  These
# are large IB transfers from a single head, so a small cap avoids saturating
# the head's NIC/disk while still overlapping per-target SSH/connect latency.
RUNNING=0
for TARGET in $TARGETS; do
    if [ -n "$SSH_USER" ]; then
        DEST="$SSH_USER@$TARGET"
    else
        DEST="$TARGET"
    fi
    echo "  Sending $IMAGE -> $TARGET ..."
    (
        if docker save "$IMAGE" | ssh $SSH_OPTS "$DEST" 'docker load'; then
            echo "  OK: $TARGET"
            echo 0 > "$STATUS_DIR/$TARGET.status"
        else
            echo "  FAILED: $TARGET" >&2
            echo 1 > "$STATUS_DIR/$TARGET.status"
        fi
    ) &
    RUNNING=$((RUNNING + 1))
    if [ "$RUNNING" -ge "$MAX_PARALLEL" ]; then
        if wait -n 2>/dev/null; then
            RUNNING=$((RUNNING - 1))
        else
            # `wait -n` unsupported (older bash) — drain all and reset.
            wait
            RUNNING=0
        fi
    fi
done
wait

FAILED=0
for TARGET in $TARGETS; do
    STATUS_FILE="$STATUS_DIR/$TARGET.status"
    if [ ! -f "$STATUS_FILE" ] || [ "$(cat "$STATUS_FILE")" != "0" ]; then
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: $FAILED target(s) failed" >&2
    exit 1
fi
echo "Image distributed successfully to all targets"
