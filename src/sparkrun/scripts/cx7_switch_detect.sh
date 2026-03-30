#!/bin/bash
# Detect whether CX7 interfaces are connected through a switch.
#
# Method: Cross-subnet L2 reachability test.
# In a direct connection, each cable is its own L2 segment — an interface
# on subnet1 cannot reach an interface on subnet2 at L2.
# Through a switch, all interfaces share the same L2 fabric and CAN
# reach across subnets via ARP.
#
# Environment variables:
#   CX7_LOCAL_IFACE  — local interface name to test from
#   CX7_REMOTE_IP    — remote IP on a DIFFERENT subnet to arping
#
# Fallback: If cross-subnet test is not possible (no IPs configured),
# listens for STP BPDUs as a secondary check.
#
# Outputs CX7_SWITCH_DETECTED=0|1|-1 on stdout.
# Diagnostic messages go to stderr.

set -uo pipefail

LOCAL_IFACE="${CX7_LOCAL_IFACE:-}"
REMOTE_IP="${CX7_REMOTE_IP:-}"

# --- Method 1: Cross-subnet arping (reliable, needs configured IPs) ---
if [ -n "$LOCAL_IFACE" ] && [ -n "$REMOTE_IP" ]; then
    echo "Cross-subnet L2 test: arping $REMOTE_IP from $LOCAL_IFACE" >&2

    if command -v arping >/dev/null 2>&1; then
        # Send 2 ARP requests with 3-second timeout
        if sudo arping -I "$LOCAL_IFACE" -c 2 -w 3 "$REMOTE_IP" >/dev/null 2>&1; then
            echo "  Reply received — switch detected" >&2
            echo "CX7_SWITCH_DETECTED=1"
            exit 0
        else
            echo "  No reply — no switch (direct connection)" >&2
            echo "CX7_SWITCH_DETECTED=0"
            exit 0
        fi
    else
        echo "  arping not available" >&2
    fi
fi

# --- Method 2: STP BPDU listening (fallback, less reliable) ---
echo "Fallback: listening for STP BPDUs..." >&2

if ! command -v tcpdump >/dev/null 2>&1; then
    echo "  tcpdump not available, cannot detect switch" >&2
    echo "CX7_SWITCH_DETECTED=-1"
    exit 0
fi

# Determine which interface to listen on
LISTEN_IFACE="${LOCAL_IFACE:-}"
if [ -z "$LISTEN_IFACE" ] && [ -n "${CX7_IFACES:-}" ]; then
    # shellcheck disable=SC2086
    set -- $CX7_IFACES
    LISTEN_IFACE="$1"
fi

if [ -z "$LISTEN_IFACE" ]; then
    echo "  No interface to listen on" >&2
    echo "CX7_SWITCH_DETECTED=-1"
    exit 0
fi

# STP BPDUs: dst 01:80:c2:00:00:00 (sent every ~2s by managed switches)
# LLDP: dst 01:80:c2:00:00:0e (sent every ~30s)
echo "  Listening on $LISTEN_IFACE for 5s..." >&2
if sudo timeout 5 tcpdump -i "$LISTEN_IFACE" -c 1 \
    '(ether dst 01:80:c2:00:00:00 or ether dst 01:80:c2:00:00:0e)' \
    -q 2>/dev/null | grep -q .; then
    echo "  Switch control frame detected" >&2
    echo "CX7_SWITCH_DETECTED=1"
else
    echo "  No switch frames detected" >&2
    echo "CX7_SWITCH_DETECTED=0"
fi
