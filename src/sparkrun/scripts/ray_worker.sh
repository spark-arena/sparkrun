#!/bin/bash
set -uo pipefail

# sparkrun:include _mgmt_iface.sh

# Detect management IP
DEFAULT_IF=$(sparkrun_mgmt_iface "")
if [ -z "$DEFAULT_IF" ]; then
    echo "ERROR: Could not identify a management interface" >&2
    exit 1
fi
NODE_IP=$(ip -4 addr show "$DEFAULT_IF" | grep -oP '(?<=inet\s)\d+(\.\d+){{3}}' | head -n1)
if [ -z "$NODE_IP" ]; then
    echo "ERROR: Could not detect management IP" >&2
    exit 1
fi
echo "Detected Node IP: $NODE_IP"

# Cleanup
{cleanup_cmd}

# Launch Ray worker
echo "Launching Ray worker, connecting to {head_ip}:{ray_port}..."
{run_cmd}

echo "Ray worker started on $NODE_IP, connected to {head_ip}:{ray_port}"
