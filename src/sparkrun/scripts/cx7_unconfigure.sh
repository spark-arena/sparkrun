#!/bin/bash
# Remove CX7 netplan configuration created by sparkrun setup.
#
# Removing our own file only releases the interfaces if our file is what holds
# them.  A host configured by another netplan file, a NetworkManager profile,
# a .network unit or /etc/network/interfaces keeps its CX7 addresses after we
# delete ours -- so this reports foreign owners instead of claiming a clean
# uninstall it did not perform.  It never touches config sparkrun did not write.
#
# No `set -e`: a probe that fails must not abort a teardown, and the two steps
# that really can fail check their own exit codes rather than trusting the
# shell to notice.
set -uo pipefail

# sparkrun:include _net_persist.sh

NETPLAN_FILE="/etc/netplan/40-cx7.yaml"

_sparkrun_owns_iface() {
    # Does *our* netplan file declare this interface?  The file is mode 600,
    # so this only resolves when running as root; when it cannot be read we
    # fall back to "our file exists", never calling a device foreign on a
    # permissions error.
    [ -f "$NETPLAN_FILE" ] || return 1
    if grep -q . "$NETPLAN_FILE" 2>/dev/null; then
        grep -Eq "^[[:space:]]*$1:" "$NETPLAN_FILE" 2>/dev/null
    else
        return 0
    fi
}

# --- Attribute every RDMA-backed interface -----------------------------------
# device/infiniband is what makes a NIC a fabric adapter, so this finds the
# CX7 ports without repeating the detection script's HCA walk.
FOREIGN_COUNT=0
for netdir in /sys/class/net/*; do
    [ -e "$netdir/device/infiniband" ] || continue
    ifc=$(basename "$netdir")
    ip4=$(ip -4 addr show "$ifc" 2>/dev/null | grep -oP 'inet \K[0-9.]+' | head -1)
    [ -n "$ip4" ] || continue

    result=$(sparkrun_net_persistence "$ifc" "$ip4")
    state=$(printf '%s' "$result" | cut -d'|' -f1)
    source=$(printf '%s' "$result" | cut -d'|' -f2)
    detail=$(printf '%s' "$result" | cut -d'|' -f3-)

    [ "$state" = "persistent" ] || continue
    if [ "$source" = "netplan" ] && _sparkrun_owns_iface "$ifc"; then
        continue
    fi
    echo "FOREIGN: $ifc ($ip4) is persisted by $source${detail:+ ($detail)} — left in place"
    FOREIGN_COUNT=$((FOREIGN_COUNT + 1))
done

# --- Remove our own file, if we wrote one ------------------------------------
if [ -f "$NETPLAN_FILE" ]; then
    if ! sudo -n rm -f "$NETPLAN_FILE"; then
        echo "FAILED: could not remove $NETPLAN_FILE" >&2
        exit 1
    fi
    echo "REMOVED: $NETPLAN_FILE"

    echo "Applying netplan configuration..." >&2
    if ! sudo -n netplan apply; then
        echo "FAILED: netplan apply" >&2
        exit 1
    fi
    echo "APPLIED: netplan (CX7 interfaces released)"
elif [ "$FOREIGN_COUNT" -gt 0 ]; then
    echo "SKIPPED: $NETPLAN_FILE not found; CX7 networking is configured elsewhere (see FOREIGN above)"
else
    echo "SKIPPED: $NETPLAN_FILE not found"
fi

echo "CX7_FOREIGN=$FOREIGN_COUNT"
echo "OK: CX7 unconfigure complete"
