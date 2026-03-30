#!/bin/bash
# CX7 netplan configuration for DGX Spark
# Variables at the top are filled by Python .format() before piping via SSH.
# The heredoc body uses shell $VAR expansion (no curly brace escaping needed).
set -euo pipefail

ADAPTER1="{adapter1}"
ADAPTER2="{adapter2}"
IP1="{ip1}"
IP2="{ip2}"
MTU="{mtu}"
PREFIX="{prefix_len}"

echo "Configuring CX7 interfaces:" >&2
echo "  $ADAPTER1 -> $IP1/$PREFIX (MTU $MTU)" >&2
echo "  $ADAPTER2 -> $IP2/$PREFIX (MTU $MTU)" >&2

sudo tee /etc/netplan/40-cx7.yaml > /dev/null <<EOF
network:
  version: 2
  ethernets:
    $ADAPTER1:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: $MTU
      addresses: [$IP1/$PREFIX]
    $ADAPTER2:
      dhcp4: no
      dhcp6: no
      link-local: []
      mtu: $MTU
      addresses: [$IP2/$PREFIX]
EOF

sudo chmod 600 /etc/netplan/40-cx7.yaml
echo "Applying netplan configuration..." >&2
sudo netplan apply

# Verify
echo "Verifying configuration..." >&2
ip -4 addr show "$ADAPTER1" 2>/dev/null | head -3 >&2
ip -4 addr show "$ADAPTER2" 2>/dev/null | head -3 >&2
echo "CX7_CONFIGURED=1"
