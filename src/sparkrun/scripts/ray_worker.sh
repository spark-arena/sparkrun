#!/bin/bash
set -uo pipefail

# Detect management IP
DEFAULT_IF=$(ip route get 8.8.8.8 | grep -oP 'dev \K\S+')
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
