#!/bin/bash
# Refresh ~/.ssh/known_hosts entries for a list of SSH endpoints.
#
# Injected by sparkrun.scripts.inject_shell_vars:
#   KEYSCAN_TARGETS  space-separated IPs / hostnames (validated, lowercased
#                    control-side: ssh-keyscan prints hostnames lowercased)
#   KEYSCAN_TIMEOUT  ssh-keyscan -T value in seconds
#
# Per target, compares what the host presents now with what known_hosts
# already records:
#   unchanged  every presented key is already recorded -> nothing written
#   added      new key types only -> the missing keys are appended
#   replaced   a recorded key of a type the host presents differs -> the
#              stale entries are removed (ssh-keygen -R) and the current
#              keys written.  Reported, because a changed host key means a
#              re-imaged host -- or something worth a look.
#   unanswered nothing came back (unreachable from this machine)
#   failed     a changed key whose stale entry could not be removed
#
# Appending without removing is what this replaced: ssh accepts a host when
# ANY recorded key matches, so it silently hid a changed key, kept the stale
# one trusted, and -- since -H salts every hash -- grew the file on each run.
set -uo pipefail
set -f
export LC_ALL=C

KEYSCAN_TARGETS="${KEYSCAN_TARGETS:-}"
KEYSCAN_TIMEOUT="${KEYSCAN_TIMEOUT:-5}"

if ! command -v ssh-keyscan >/dev/null 2>&1 || ! command -v ssh-keygen >/dev/null 2>&1; then
    echo "KEYSCAN_ERROR=ssh-keyscan/ssh-keygen not found"
    exit 1
fi

KH="$HOME/.ssh/known_hosts"
mkdir -p "$HOME/.ssh"
touch "$KH"
TMP=$(mktemp) || { echo "KEYSCAN_ERROR=mktemp failed"; exit 1; }
trap 'rm -f "$TMP" "$TMP.old"' EXIT

# One invocation probes every target concurrently, so an unreachable
# address costs one timeout in total rather than one per address.  Unhashed
# output, so results can be grouped by target.
# shellcheck disable=SC2086
SCAN=$(ssh-keyscan -T "$KEYSCAN_TIMEOUT" $KEYSCAN_TARGETS 2>/dev/null)

# Serialize writers that share a home directory (e.g. NFS-mounted homes
# across nodes).  Best effort: without flock the writes still happen.
exec 9>"$HOME/.ssh/.known_hosts.sparkrun.lock"
if command -v flock >/dev/null 2>&1; then
    flock -w 30 9 || true
fi

append_hashed() {
    # $1: lines "<host> <type> <key>".  Written hashed, like ssh-keyscan -H.
    printf '%s\n' "$1" > "$TMP"
    ssh-keygen -q -H -f "$TMP" >/dev/null 2>&1 || true
    grep -v '^#' "$TMP" >> "$KH"
}

ADDED=0
UNCHANGED=0
REPLACED=""
UNANSWERED=""
FAILED=""
for t in $KEYSCAN_TARGETS; do
    scanned=$(printf '%s\n' "$SCAN" | awk -v h="$t" '$1 == h && NF >= 3 { print $2 " " $3 }' | sort -u)
    if [ -z "$scanned" ]; then
        UNANSWERED="$UNANSWERED $t"
        continue
    fi
    recorded=$(ssh-keygen -F "$t" -f "$KH" 2>/dev/null | awk '!/^#/ && !/^@/ && NF >= 3 { print $2 " " $3 }' | sort -u)
    missing=$(comm -23 <(printf '%s\n' "$scanned") <(printf '%s\n' "$recorded") | sed '/^$/d')

    # Changed = a recorded key whose type the host now presents with another
    # key.  Checked even when nothing is missing: the append-only refresh this
    # replaced left hosts holding both the stale and the current key.
    scanned_types=$(printf '%s\n' "$scanned" | awk '{ print $1 }' | sort -u)
    stale_types=$(comm -13 <(printf '%s\n' "$scanned") <(printf '%s\n' "$recorded") | sed '/^$/d' | awk '{ print $1 }' | sort -u)
    changed=$(comm -12 <(printf '%s\n' "$scanned_types") <(printf '%s\n' "$stale_types") | sed '/^$/d')

    if [ -n "$changed" ]; then
        if ! ssh-keygen -R "$t" -f "$KH" >/dev/null 2>&1; then
            FAILED="$FAILED $t"
            continue
        fi
        append_hashed "$(printf '%s\n' "$scanned" | sed "s/^/$t /")"
        REPLACED="$REPLACED $t"
    elif [ -n "$missing" ]; then
        append_hashed "$(printf '%s\n' "$missing" | sed "s/^/$t /")"
        ADDED=$((ADDED + 1))
    else
        UNCHANGED=$((UNCHANGED + 1))
    fi
done

echo "KEYSCAN_ADDED=$ADDED"
echo "KEYSCAN_UNCHANGED=$UNCHANGED"
echo "KEYSCAN_REPLACED=${REPLACED# }"
echo "KEYSCAN_UNANSWERED=${UNANSWERED# }"
echo "KEYSCAN_FAILED=${FAILED# }"
