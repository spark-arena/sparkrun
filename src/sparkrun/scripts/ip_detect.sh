#!/bin/bash
set -uo pipefail

# sparkrun:include _mgmt_iface.sh

DEFAULT_IF=$(sparkrun_mgmt_iface "")
if [ -z "$DEFAULT_IF" ]; then
    echo "ERROR: Could not identify a management interface" >&2
    exit 1
fi
NODE_IP=$(sparkrun_mgmt_ip "$DEFAULT_IF")
if [ -z "$NODE_IP" ]; then
    echo "ERROR: Could not detect management IP" >&2
    exit 1
fi
echo "$NODE_IP"
