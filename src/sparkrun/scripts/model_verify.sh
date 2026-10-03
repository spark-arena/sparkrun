#!/bin/bash
set -uo pipefail
# Verify a cached HF model against the cache's own content addressing: weight
# blobs are LFS files whose blob filename IS the sha256 of their contents, so
# hashing them needs no network access and no manifest.  Presence-only checks
# (any weight file found) cannot see a truncated or raced transfer -- the
# snapshot symlink and the blob both exist while the bytes are wrong.
#
# The scan is scoped to the pinned revision's snapshot directories, exactly
# like the Python verifier (sparkrun.models.verify): an absent revision or a
# snapshot of dangling links resolves to zero entries and FAILS -- a marker
# left over from a previous revision can never wave a new revision through.
#
# Exit codes: 0 = verified; 1 = missing/corrupt (with SPARKRUN_VERIFY_REPAIR=1,
# corrupt blobs are removed first so a re-download refetches them).
#
# Placeholders filled by Python: {cache_path}, {revision}
# NOTE: this file is consumed via Python str.format(); it must contain NO
# literal curly-brace characters.

CACHE_PATH="{cache_path}"
MODEL_REVISION={revision}
MARKER="$CACHE_PATH/.sparkrun-verified"

# sparkrun:include _hf_snapshots.sh

SNAPSHOT_DIRS="$(sparkrun_hf_snapshot_dirs "$CACHE_PATH" "$MODEL_REVISION")"

# Collect the revision's weight entries.  find -L follows the snapshot
# symlinks and silently skips dangling ones, so an entry listed here always
# resolves to a real blob right now -- a revision whose blobs never arrived
# yields zero entries and fails via FOUND=0, whatever its marker says.
ENTRY_LIST="$(mktemp)"
trap 'rm -f "$ENTRY_LIST"' EXIT
while IFS= read -r SNAPSHOT_DIR; do
    [ -n "$SNAPSHOT_DIR" ] || continue
    find -L "$SNAPSHOT_DIR" \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" -o -name "*.gguf" \) -type f -print0 2>/dev/null >>"$ENTRY_LIST"
done <<<"$SNAPSHOT_DIRS"
FOUND="$(tr -dc '\000' <"$ENTRY_LIST" | wc -c)"
FOUND=$((FOUND + 0))

if [ "$FOUND" -eq 0 ]; then
    echo "no weight files found under $CACHE_PATH for the requested revision"
    exit 1
fi

# Steady-state fast path: a verification marker younger than every resolved
# weight blob means nothing changed since the last clean hash pass.  The
# comparison is scoped to this revision's entries -- a rev-A-era marker
# cannot clear a rev-B snapshot, whose entries are missing (zero entries,
# failed above) or carry fresh mtimes.  SPARKRUN_VERIFY_REPAIR implies a
# full pass, since the caller is about to remove blobs.
MARKER_FRESH=1
if [ ! -f "$MARKER" ] || [ -n "$(printenv SPARKRUN_VERIFY_REPAIR)" ]; then
    MARKER_FRESH=0
fi
if [ "$MARKER_FRESH" = 1 ]; then
    while IFS= read -r -d '' f; do
        blob="$(readlink -f "$f" 2>/dev/null)"
        if [ ! -f "$blob" ] || [ "$blob" -nt "$MARKER" ]; then
            MARKER_FRESH=0
            break
        fi
    done <"$ENTRY_LIST"
    if [ "$MARKER_FRESH" = 1 ]; then
        echo "verified marker is fresh; skipping hash pass"
        exit 0
    fi
    echo "cache changed since last verification; re-hashing"
fi

FAIL=0
while IFS= read -r -d '' f; do
    blob="$(readlink -f "$f" 2>/dev/null)"
    if [ ! -f "$blob" ]; then
        echo "missing blob for snapshot entry: $f"
        FAIL=$((FAIL + 1))
        continue
    fi
    sum="$(sha256sum "$blob" 2>/dev/null | cut -d" " -f1)"
    if [ "$sum" != "$(basename "$blob")" ]; then
        echo "checksum mismatch: $blob"
        if [ "$(printenv SPARKRUN_VERIFY_REPAIR)" = "1" ]; then
            rm -f "$blob"
            echo "removed corrupt blob: $blob"
        fi
        FAIL=$((FAIL + 1))
    fi
done <"$ENTRY_LIST"

if [ "$FAIL" -gt 0 ]; then
    echo "verification failed: $FAIL problem(s) across $FOUND weight file(s)"
    exit 1
fi
touch "$MARKER"
echo "verified: $FOUND weight file(s) match their checksums"
exit 0
