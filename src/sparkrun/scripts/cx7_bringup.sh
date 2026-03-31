#!/bin/bash
# Bring CX7 interfaces up with link-local addresses for topology detection.
# Arguments are passed as CX7_IFACES env var (space-separated interface names).
# Outputs CX7_BRINGUP_COUNT=N on stdout.
# Diagnostic messages go to stderr.
#
# Only uses sudo when an interface is administratively down. Interfaces that
# are already UP (even without carrier) are left untouched (no sudo required).

set -uo pipefail

if [ -z "${CX7_IFACES:-}" ]; then
    echo "CX7_BRINGUP_COUNT=0"
    exit 0
fi

echo "Bringing up CX7 interfaces: $CX7_IFACES" >&2

COUNT=0
NEEDED_SUDO=0
for iface in $CX7_IFACES; do
    # Check if interface is administratively UP by reading flags
    # Flags field: <FLAG1,FLAG2,...>  — UP means admin-enabled
    flags=$(ip link show dev "$iface" 2>/dev/null | head -1 | grep -oP '<\K[^>]+' || echo "")
    if echo ",$flags," | grep -q ",UP,"; then
        echo "  $iface: already up" >&2
        COUNT=$((COUNT + 1))
    else
        # Interface is administratively down — need sudo to bring it up
        if sudo ip link set "$iface" up 2>/dev/null; then
            echo "  $iface: brought up" >&2
            COUNT=$((COUNT + 1))
            NEEDED_SUDO=1
        else
            echo "  $iface: failed to bring up (sudo required)" >&2
        fi
    fi
done

# Wait briefly for link state to settle (only if we changed something)
if [ $NEEDED_SUDO -gt 0 ]; then
    sleep 2
fi

echo "CX7_BRINGUP_COUNT=$COUNT"
