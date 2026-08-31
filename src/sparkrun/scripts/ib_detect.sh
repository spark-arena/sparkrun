#!/bin/bash
# InfiniBand/RDMA detection for DGX Spark
# Outputs key=value pairs on stdout for NCCL configuration.
# Diagnostic messages go to stderr so they don't pollute parsing.
#
# Based on the proven detection in scripts/infiniband-detect.sh.

set -uo pipefail

# --- Helper: Find RoCEv2 GID Index via show_gids ---
find_rocev2_ipv4_index() {
    local hca=$1
    if ! command -v show_gids &>/dev/null; then
        return 1
    fi
    show_gids \
    | awk -v dev="$hca" \
      '$1 == dev && $6 == "v2" && $5 ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/ {print $3; exit}'
}

echo "Running InfiniBand detection..." >&2

if ! [ -d /sys/class/infiniband ]; then
    echo "IB_DETECTED=0"
    exit 0
fi

ACTIVE_HCAS=()
ACTIVE_NETIFS=()
UCX_DEVS=()
GID_INDEX=""

# Iterate over InfiniBand devices in sysfs
for ib_path in /sys/class/infiniband/*; do
    [ -e "$ib_path" ] || continue
    hca_name=$(basename "$ib_path")

    # Check Port 1 state
    state_file="$ib_path/ports/1/state"
    if [ ! -f "$state_file" ]; then continue; fi

    state_val=$(cat "$state_file" 2>/dev/null)

    if [[ "$state_val" == *"ACTIVE"* ]]; then
        ACTIVE_HCAS+=("$hca_name")
        UCX_DEVS+=("${hca_name}:1")

        # Determine GID Index (run once on first active card)
        if [ -z "$GID_INDEX" ]; then
            idx=$(find_rocev2_ipv4_index "$hca_name")
            if [ -n "$idx" ]; then
                GID_INDEX=$idx
                echo "Device $hca_name: Active (RoCEv2 GID Index: $idx)" >&2
            else
                # Fallback: scan sysfs gid_attrs for RoCE v2
                port_gid_dir="$ib_path/ports/1/gid_attrs/types"
                if [ -d "$port_gid_dir" ]; then
                    for type_file in "$port_gid_dir"/*; do
                        gid_type=$(cat "$type_file" 2>/dev/null || true)
                        if [[ "$gid_type" == *"RoCE v2"* ]] || [[ "$gid_type" == *"RoCEv2"* ]]; then
                            GID_INDEX=$(basename "$type_file")
                            echo "Device $hca_name: Active (RoCEv2 GID Index: $GID_INDEX via sysfs)" >&2
                            break
                        fi
                    done
                fi
                if [ -z "$GID_INDEX" ]; then
                    echo "Device $hca_name: Active (GID detect failed, defaulting to 3)" >&2
                    GID_INDEX=3
                fi
            fi
        else
            echo "Device $hca_name: Active" >&2
        fi

        # Find Ethernet interface backing this device
        net_dir="$ib_path/device/net"
        if [ -d "$net_dir" ]; then
            net_if=$(ls "$net_dir" | head -n 1)
            [ -n "$net_if" ] && ACTIVE_NETIFS+=("$net_if")
        fi
    fi
done

if [ ${#ACTIVE_HCAS[@]} -eq 0 ]; then
    echo "No active RDMA devices found." >&2
    echo "IB_DETECTED=0"
    exit 0
fi

# Detect management network interface (default route) and its IP
DEFAULT_IF=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'dev \K\S+' || echo "eth0")
MGMT_IP=$(ip -4 addr show "$DEFAULT_IF" 2>/dev/null | grep -oP 'inet \K[0-9.]+' | head -1)

# Build comma-separated lists
HCA_LIST=$(IFS=,; echo "${ACTIVE_HCAS[*]}")
NET_LIST=$(IFS=,; echo "${ACTIVE_NETIFS[*]}")
UCX_LIST=$(IFS=,; echo "${UCX_DEVS[*]}")

# Resolve IPv4 addresses of active IB network interfaces
IB_IPS=()
for net_if in "${ACTIVE_NETIFS[@]}"; do
    ip_addr=$(ip -4 addr show "$net_if" 2>/dev/null | grep -oP 'inet \K[0-9.]+' | head -1)
    if [ -n "$ip_addr" ]; then
        IB_IPS+=("$ip_addr")
    fi
done
IB_IP_LIST=$(IFS=,; echo "${IB_IPS[*]}")

echo "---------------------------------------------------" >&2
echo "Detection complete:" >&2
echo "  HCA_LIST   = $HCA_LIST" >&2
echo "  NET_LIST   = $NET_LIST" >&2
echo "  IB_IPS     = $IB_IP_LIST" >&2
echo "  GID_INDEX  = $GID_INDEX" >&2
echo "  DEFAULT_IF = $DEFAULT_IF" >&2
echo "  MGMT_IP    = $MGMT_IP" >&2
echo "---------------------------------------------------" >&2

# Output key=value pairs on stdout (parsed by sparkrun)
echo "IB_DETECTED=1"
echo "DETECTED_GID_INDEX=$GID_INDEX"
echo "DETECTED_HCA_LIST=$HCA_LIST"
echo "DETECTED_SOCKET_IFNAME=$DEFAULT_IF"
echo "DETECTED_NET_LIST=$NET_LIST"
echo "DETECTED_UCX_LIST=$UCX_LIST"
echo "DETECTED_IB_IPS=$IB_IP_LIST"
echo "DETECTED_MGMT_IP=$MGMT_IP"
