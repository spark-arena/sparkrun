#!/bin/bash
set -uo pipefail

printf "Cleaning up existing container: %s\n" {container_name}
{cleanup_cmd}

printf "Launching container: %s\n" {container_name}
printf "Image: %s\n" {image}
{run_cmd}

# Add host management IP as alias on eth0 for bridge network compatibility
# This allows containers to be reached via the host's management IP
if [ -n "{HOST_IP}" ]; then
    docker exec {container_name} ip addr add "{HOST_IP}/24" dev eth0 2>/dev/null || true
fi

printf "Container %s launched successfully\n" {container_name}
