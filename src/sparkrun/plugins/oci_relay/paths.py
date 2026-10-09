# SPDX-FileCopyrightText: 2026 Spark Arena
# SPDX-License-Identifier: Apache-2.0
"""Explicit data-network maps; route discovery never changes host networking."""
from __future__ import annotations

import ipaddress
import json
import logging
import re

logger = logging.getLogger(__name__)

_PATH_FACTS = r'''
set -eu
ip -j route get "$2" from "$1"
printf '\000'
ip -j address show to "$1/$3"
printf '\000'
device=$(ip -o route get "$2" from "$1" | awk '{for (i=1;i<NF;i++) if ($i=="dev") {print $(i+1); exit}}')
case "$device" in ''|*[!a-zA-Z0-9_.:-]*) exit 1;; esac
cat "/sys/class/net/$device/speed" 2>/dev/null || true
printf '\000'
'''


def validate_paths(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError("data_paths must contain between 1 and 8 host-to-IP maps")
    for path in value:
        if not isinstance(path, dict) or not path:
            raise ValueError("each data path must map execution hosts (or controller) to local IPs")
        for host, address in path.items():
            if not isinstance(host, str) or not host or not isinstance(address, str):
                raise ValueError("data path hosts and addresses must be strings")
            ip = ipaddress.ip_address(address)
            if ip.is_unspecified or ip.is_multicast or ip.is_loopback or ip.is_link_local:
                raise ValueError("data paths require concrete, routable interface addresses")
    return value


def _route(runner, host, local, remote):
    prefix = "32" if ipaddress.ip_address(local).version == 4 else "128"
    raw = runner.execute(host, ["sh", "-c", _PATH_FACTS, "relay-path", local, remote, prefix])
    fields = raw.split(b"\0")
    if len(raw) > 64 << 10 or len(fields) != 4 or fields[-1]:
        raise ValueError("invalid or oversized data-path facts")
    route_json, address_json, speed_raw, _ = fields
    routes = json.loads(route_json)
    if not isinstance(routes, list) or not routes or not isinstance(routes[0], dict):
        raise ValueError("route response is invalid")
    route = routes[0]
    device = route.get("dev", "")
    if not isinstance(device, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", device):
        raise ValueError("route has no usable interface")
    if route.get("type", "unicast") != "unicast" or route.get("gateway"):
        raise ValueError("multi-link paths must be directly connected")
    # Require the requested local address to actually belong to this interface.
    addresses = json.loads(address_json)
    if not any(ipaddress.ip_address(item["local"]) == ipaddress.ip_address(local)
               for interface in addresses if interface.get("ifname") == device for item in interface.get("addr_info", [])):
        raise ValueError("local data address is not assigned to the routed interface")
    try:
        speed = int(speed_raw) / 1000
    except ValueError:
        speed = 0
    return device, speed if 0 < speed <= 1600 else 0


def qualify(runner, source_host, target, maps, facts):
    """Return directly connected routes using distinct devices at both ends.

    Unavailable routes degrade to the surviving link. Authentication preflight
    still runs later; these probes alone never authorize data transfer.
    """
    source_key = source_host if source_host is not None else "controller"
    result, source_devices, target_devices = [], set(), set()
    for path in validate_paths(maps):
        if source_key not in path or target not in path:
            raise ValueError("each data path must include the source and every target execution host")
        source, local = path[source_key], path[target]
        try:
            src_dev, src_speed = _route(runner, source_host, source, local)
            dst_dev, dst_speed = _route(runner, target, local, source)
            if src_dev in source_devices or dst_dev in target_devices:
                raise ValueError("data paths share an interface")
            for host, speed in ((source_host, src_speed), (target, dst_speed)):
                product = facts.get(host, {}).get("product_name", "").replace("_", " ").lower()
                if product == "nvidia dgx spark":
                    if host == source_host:
                        src_speed = min(speed, 100)
                    else:
                        dst_speed = min(speed, 100)
            result.append({"source": source, "local": local, "source_device": src_dev,
                           "target_device": dst_dev, "network_gbps": min(src_speed, dst_speed)})
            source_devices.add(src_dev)
            target_devices.add(dst_dev)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            logger.warning("OCI Relay skipped a data path for %s: %s", target, error)
    return result


def arguments(paths, port):
    result = []
    for path in paths:
        address = path["source"]
        authority = f"[{address}]" if ":" in address else address
        result.extend(["--path", f"https://{authority}:{port},{path['local']}"])
    return result
