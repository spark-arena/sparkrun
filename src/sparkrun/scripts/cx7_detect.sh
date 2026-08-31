#!/bin/bash
# CX7 network interface detection for DGX Spark
# Outputs key=value pairs on stdout for CX7 configuration.
# Diagnostic messages go to stderr so they don't pollute parsing.

set -uo pipefail

echo "Running CX7 interface detection..." >&2

if ! [ -d /sys/class/infiniband ]; then
    echo "CX7_DETECTED=0"
    exit 0
fi

# --- Detect management interface and IP ---
MGMT_IFACE=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'dev \K\S+' || echo "eth0")
MGMT_IP=$(ip -4 addr show "$MGMT_IFACE" 2>/dev/null | grep -oP 'inet \K[0-9.]+' | head -1)
echo "Management interface: $MGMT_IFACE ($MGMT_IP)" >&2

# --- Find active CX7/RoCE interfaces ---
IFACE_COUNT=0
declare -a IFACE_NAMES=()
declare -a IFACE_IPS=()
declare -a IFACE_PREFIXES=()
declare -a IFACE_SUBNETS=()
declare -a IFACE_MTUS=()
declare -a IFACE_STATES=()
declare -a IFACE_HCAS=()
declare -a IFACE_MACS=()

for ib_path in /sys/class/infiniband/*; do
    [ -e "$ib_path" ] || continue
    hca_name=$(basename "$ib_path")

    # Check Port 1 state
    state_file="$ib_path/ports/1/state"
    if [ ! -f "$state_file" ]; then continue; fi
    state_val=$(cat "$state_file" 2>/dev/null)
    if [[ "$state_val" != *"ACTIVE"* ]]; then
        echo "Device $hca_name: not active ($state_val)" >&2
        continue
    fi

    # Find net interface backing this device
    net_dir="$ib_path/device/net"
    if [ ! -d "$net_dir" ]; then continue; fi
    net_if=$(ls "$net_dir" | head -n 1)
    if [ -z "$net_if" ]; then continue; fi

    # Skip if this is the management interface
    if [ "$net_if" = "$MGMT_IFACE" ]; then
        echo "Device $hca_name: skipping (management interface $net_if)" >&2
        continue
    fi

    # Get interface state
    operstate=$(cat "/sys/class/net/$net_if/operstate" 2>/dev/null || echo "unknown")

    # Get MTU
    mtu=$(cat "/sys/class/net/$net_if/mtu" 2>/dev/null || echo "0")

    # Get IPv4 address and prefix
    addr_info=$(ip -4 addr show "$net_if" 2>/dev/null | grep -oP 'inet \K[0-9./]+' | head -1)
    if [ -n "$addr_info" ]; then
        ip_addr=$(echo "$addr_info" | cut -d/ -f1)
        prefix=$(echo "$addr_info" | cut -d/ -f2)
        # Calculate subnet using Python (always available on DGX OS)
        subnet=$(python3 -c "import ipaddress; print(ipaddress.IPv4Network('${addr_info}', strict=False))" 2>/dev/null || echo "")
    else
        ip_addr=""
        prefix=""
        subnet=""
    fi

    echo "Device $hca_name: Active, interface=$net_if, ip=$ip_addr/$prefix, mtu=$mtu, state=$operstate" >&2

    IFACE_NAMES+=("$net_if")
    IFACE_IPS+=("$ip_addr")
    IFACE_PREFIXES+=("$prefix")
    IFACE_SUBNETS+=("$subnet")
    IFACE_MTUS+=("$mtu")
    IFACE_STATES+=("$operstate")
    IFACE_HCAS+=("$hca_name")

    # MAC address
    mac_addr=$(cat "/sys/class/net/$net_if/address" 2>/dev/null || echo "")
    IFACE_MACS+=("$mac_addr")

    IFACE_COUNT=$((IFACE_COUNT + 1))
done

if [ "$IFACE_COUNT" -eq 0 ]; then
    echo "No active CX7/RoCE interfaces found." >&2
    echo "CX7_DETECTED=0"
    exit 0
fi

# --- Check for existing netplan config ---
NETPLAN_EXISTS=0
if [ -f /etc/netplan/40-cx7.yaml ]; then
    NETPLAN_EXISTS=1
fi

# --- Check passwordless sudo ---
SUDO_OK=0
if sudo -n true 2>/dev/null; then
    SUDO_OK=1
fi

# --- Collect non-CX7 used subnets from route table ---
# Build a set of CX7 interface names for exclusion
declare -A CX7_IF_SET
for name in "${IFACE_NAMES[@]}"; do
    CX7_IF_SET["$name"]=1
done

USED_SUBNETS=""
while IFS= read -r route_line; do
    dest=$(echo "$route_line" | awk '{print $1}')
    # Skip default routes and non-CIDR entries
    if [ "$dest" = "default" ]; then continue; fi
    if [[ "$dest" != *"/"* ]]; then continue; fi

    # Check if route is on a CX7 interface
    dev=""
    if echo "$route_line" | grep -q "dev "; then
        dev=$(echo "$route_line" | grep -oP 'dev \K\S+')
    fi
    if [ -n "$dev" ] && [ -n "${CX7_IF_SET[$dev]+x}" ]; then
        continue  # Skip CX7 interface routes
    fi

    if [ -n "$USED_SUBNETS" ]; then
        USED_SUBNETS="${USED_SUBNETS},$dest"
    else
        USED_SUBNETS="$dest"
    fi
done < <(ip -4 route 2>/dev/null)

# --- Output key=value pairs ---
echo "---------------------------------------------------" >&2
echo "Detection complete: $IFACE_COUNT CX7 interface(s)" >&2
echo "---------------------------------------------------" >&2

echo "CX7_DETECTED=1"
echo "CX7_MGMT_IP=$MGMT_IP"
echo "CX7_MGMT_IFACE=$MGMT_IFACE"
echo "CX7_NETPLAN_EXISTS=$NETPLAN_EXISTS"
echo "CX7_SUDO_OK=$SUDO_OK"
echo "CX7_IFACE_COUNT=$IFACE_COUNT"

for i in $(seq 0 $((IFACE_COUNT - 1))); do
    echo "CX7_IFACE_${i}_NAME=${IFACE_NAMES[$i]}"
    echo "CX7_IFACE_${i}_IP=${IFACE_IPS[$i]}"
    echo "CX7_IFACE_${i}_PREFIX=${IFACE_PREFIXES[$i]}"
    echo "CX7_IFACE_${i}_SUBNET=${IFACE_SUBNETS[$i]}"
    echo "CX7_IFACE_${i}_MTU=${IFACE_MTUS[$i]}"
    echo "CX7_IFACE_${i}_STATE=${IFACE_STATES[$i]}"
    echo "CX7_IFACE_${i}_HCA=${IFACE_HCAS[$i]}"
    echo "CX7_IFACE_${i}_MAC=${IFACE_MACS[$i]}"
done

echo "CX7_USED_SUBNETS=$USED_SUBNETS"
