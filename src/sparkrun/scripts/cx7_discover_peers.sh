#!/bin/bash
# CX7 peer discovery via broadcast ping + ARP table scan.
# Arguments: one or more /24 subnet CIDRs (e.g. 192.168.11.0/24).
# Outputs PEER_N_IP=x.x.x.x and PEER_COUNT=N on stdout.
# Diagnostic messages go to stderr.

set -uo pipefail

if [ $# -eq 0 ]; then
    echo "PEER_COUNT=0"
    exit 0
fi

# Ping broadcast on each subnet to populate ARP table
for SUBNET in "$@"; do
    BCAST=$(python3 -c "import ipaddress,sys; print(ipaddress.IPv4Network(sys.argv[1], strict=False).broadcast_address)" "$SUBNET" 2>/dev/null)
    [ -z "$BCAST" ] && continue
    echo "Pinging broadcast $BCAST for $SUBNET ..." >&2
    ping -b -c 2 -w 3 "$BCAST" >/dev/null 2>&1 || true
done

# Allow ARP table to settle
sleep 1

# Filter ARP neighbors through target subnets, excluding local IPs.
# Using Python avoids shell injection risks from IP parsing.
python3 - "$@" <<'PYEOF'
import ipaddress, re, subprocess, sys

subnets = []
for arg in sys.argv[1:]:
    try:
        subnets.append(ipaddress.IPv4Network(arg, strict=False))
    except ValueError:
        continue

if not subnets:
    print("PEER_COUNT=0")
    sys.exit(0)

# Collect local IPs to exclude
local_out = subprocess.run(
    ["ip", "-4", "addr", "show"],
    capture_output=True, text=True,
).stdout
local_ips = set(re.findall(r"inet (\d+\.\d+\.\d+\.\d+)", local_out))

# Read ARP / neighbor table
neigh_out = subprocess.run(
    ["ip", "neigh", "show"],
    capture_output=True, text=True,
).stdout

seen = set()
count = 0
for line in neigh_out.splitlines():
    parts = line.split()
    if len(parts) < 4:
        continue
    ip_str = parts[0]
    # Only include entries with a valid ARP state
    state = parts[-1]
    if state not in ("REACHABLE", "STALE", "DELAY", "PROBE"):
        continue
    if ip_str in local_ips or ip_str in seen:
        continue
    try:
        addr = ipaddress.IPv4Address(ip_str)
    except ValueError:
        continue
    for net in subnets:
        if addr in net:
            print("PEER_%d_IP=%s" % (count, ip_str))
            seen.add(ip_str)
            count += 1
            break

print("PEER_COUNT=%d" % count)
PYEOF
