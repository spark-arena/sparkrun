#!/bin/bash
# Remove CX7 netplan configuration created by sparkrun setup.
set -euo pipefail

NETPLAN_FILE="/etc/netplan/40-cx7.yaml"

if [ -f "$NETPLAN_FILE" ]; then
    sudo -n rm -f "$NETPLAN_FILE"
    echo "REMOVED: $NETPLAN_FILE"

    echo "Applying netplan configuration..." >&2
    sudo -n netplan apply
    echo "APPLIED: netplan (CX7 interfaces released)"
else
    echo "SKIPPED: $NETPLAN_FILE not found"
fi

echo "OK: CX7 unconfigure complete"
