#!/bin/bash
# Neighbor MAC discovery for CX7 topology detection.
# Arguments are passed as CX7_IFACES env var (space-separated interface names).
# Outputs CX7_NEIGHBOR_N_LOCAL_IFACE and CX7_NEIGHBOR_N_REMOTE_MAC on stdout.
# Diagnostic messages go to stderr.
#
# Three-tier discovery (each tier only runs if previous found nothing):
#   1. IPv6 all-nodes multicast ping (no sudo) — works on any UP interface
#   2. IPv4 broadcast ping (no sudo) — works when interface has an IP
#   3. arping with sudo (fallback) — DAD mode, works without any IP

set -uo pipefail

if [ -z "${CX7_IFACES:-}" ]; then
    echo "CX7_NEIGHBOR_COUNT=0"
    exit 0
fi

echo "Discovering neighbors on CX7 interfaces: $CX7_IFACES" >&2

# Helper: read neighbor table and collect MACs
# Usage: _collect_neighbors [-6]
# Sets COUNT and outputs CX7_NEIGHBOR_* vars
_collect_neighbors() {
    local ip_ver="${1:-}"
    for iface in $CX7_IFACES; do
        local found=0
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            local remote_mac
            remote_mac=$(echo "$line" | grep -oP 'lladdr \K[0-9a-f:]+')
            local state
            state=$(echo "$line" | awk '{print $NF}')

            if [ -n "$remote_mac" ] && [ "$state" != "FAILED" ]; then
                echo "  $iface -> $remote_mac ($state)" >&2
                echo "CX7_NEIGHBOR_${COUNT}_LOCAL_IFACE=$iface"
                echo "CX7_NEIGHBOR_${COUNT}_REMOTE_MAC=$remote_mac"
                COUNT=$((COUNT + 1))
                found=1
                # Take only the first valid neighbor per interface
                break
            fi
        done < <(ip $ip_ver neigh show dev "$iface" 2>/dev/null)
    done
}

COUNT=0

# Phase 1: IPv6 all-nodes multicast (no sudo needed)
# Works on any UP interface — fe80:: link-local is assigned automatically.
echo "  Phase 1: IPv6 multicast discovery..." >&2
for iface in $CX7_IFACES; do
    ping6 -I "$iface" -c 3 -w 3 ff02::1 >/dev/null 2>&1 || true
done
sleep 1
_collect_neighbors -6

# Phase 2: IPv4 broadcast ping (no sudo needed)
if [ "$COUNT" -eq 0 ]; then
    echo "  Phase 2: IPv4 broadcast discovery..." >&2
    for iface in $CX7_IFACES; do
        # Link-local broadcast
        ping -I "$iface" -c 3 -w 3 -b 169.254.255.255 >/dev/null 2>&1 || true
        # Subnet broadcast if interface has an IP
        iface_bcast=$(ip -4 addr show dev "$iface" 2>/dev/null | grep -oP 'brd \K[0-9.]+' | head -1)
        if [ -n "$iface_bcast" ]; then
            ping -I "$iface" -c 2 -w 2 -b "$iface_bcast" >/dev/null 2>&1 || true
        fi
    done
    sleep 1
    _collect_neighbors
fi

# Phase 3: arping with sudo (last resort)
if [ "$COUNT" -eq 0 ]; then
    echo "  Phase 3: arping discovery (sudo)..." >&2
    for iface in $CX7_IFACES; do
        sudo arping -I "$iface" -c 3 -b -w 3 0.0.0.0 >/dev/null 2>&1 || true
    done
    sleep 1
    _collect_neighbors
fi

echo "CX7_NEIGHBOR_COUNT=$COUNT"
