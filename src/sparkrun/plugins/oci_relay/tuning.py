# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: Apache-2.0
"""Sparkrun policy only: translate route/host information into explicit limits."""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)
MIB = 1 << 20

# Fixed NUL-delimited fields keep absent files distinct without requiring
# Python/jq on execution hosts. Destination and paths are positional arguments.
_FACTS = r'''
section() { "$@" 2>/dev/null || true; printf '\000'; }
section cat /proc/sys/kernel/random/boot_id
section cat /proc/meminfo
section cat /proc/self/cgroup
section getconf _NPROCESSORS_ONLN
section cat /sys/class/dmi/id/product_name
section ip -j route get "$1"
'''
_LIMITS = r'''
cat "$1" 2>/dev/null || true
printf '\000'
shift
for directory do
    cat "$directory/memory.max" "$directory/memory.current" 2>/dev/null || true
    printf '\000'
done
'''


def _sections(runner, host, script, arguments, count):
    raw = runner.execute(host, ["sh", "-c", script, "relay-probe", *arguments])
    fields = raw.split(b"\0")
    if len(raw) > 64 << 10 or len(fields) != count + 1 or fields[-1] != b"":
        raise ValueError("invalid host probe fields")
    return fields[:-1]


def probe(runner, host, destination):
    """Best-effort route facts in two host commands, including ancestor caps."""
    facts = {}
    try:
        host_id, memory, cgroup, cpus, product, route = _sections(runner, host, _FACTS, [destination], 6)
    except (OSError, RuntimeError, ValueError):
        return facts
    facts["host_id"] = host_id.decode(errors="replace").strip()
    facts["product_name"] = product.decode(errors="replace").strip()
    match = re.search(rb"^MemAvailable:\s+(\d+)\s+kB", memory, re.MULTILINE)
    if match:
        facts["available_memory"] = int(match[1]) * 1024
    try:
        if 0 < int(cpus) <= 65536:
            facts["cpus"] = int(cpus)
    except ValueError:
        pass
    directories = []
    membership = next((line[3:] for line in cgroup.decode(errors="replace").splitlines() if line.startswith("0::")), "")
    if membership.startswith("/") and ".." not in membership.split("/"):
        directory = "/sys/fs/cgroup" + membership.rstrip("/")
        # A child may say max beneath a capped parent. Include the root too.
        while directory.startswith("/sys/fs/cgroup/"):
            directories.append(directory)
            directory = directory.rsplit("/", 1)[0]
        directories.append("/sys/fs/cgroup")
    interface, speed_path = "", "/dev/null"
    try:
        routes = json.loads(route)
        if not isinstance(routes, list) or not routes or not isinstance(routes[0], dict):
            raise ValueError("invalid route")
        address = routes[0].get("prefsrc") or routes[0].get("src")
        if address:
            import ipaddress

            facts["source_address"] = str(ipaddress.ip_address(address))
        interface = routes[0].get("dev", "")
        if isinstance(interface, str) and re.fullmatch(r"[A-Za-z0-9_.:-]+", interface):
            speed_path = "/sys/class/net/" + interface + "/speed"
    except (ValueError, TypeError):
        pass
    try:
        speed, *caps = _sections(runner, host, _LIMITS, [speed_path, *directories], 1 + len(directories))
    except (OSError, RuntimeError, ValueError):
        # Unknown cgroup headroom must not inherit an unrestricted host budget.
        facts.pop("available_memory", None)
        return facts
    for raw in caps:
        values = raw.split()
        if len(values) == 2 and values[0] != b"max":
            try:
                maximum, current = map(int, values)
                if maximum >= 0 and current >= 0:
                    remaining = max(0, maximum - current)
                    facts["available_memory"] = min(facts.get("available_memory", remaining), remaining)
            except ValueError:
                continue
    try:
        # Virtual interfaces often report no speed. Unknown stays conservative.
        speed = int(speed)
        if 0 < speed <= 1600000:
            link_gbps = speed / 1000
            effective = link_gbps
            if facts.get("product_name", "").replace("_", " ").lower() == "nvidia dgx spark":
                # Spark's advertised 200 Gbps port has an effective 100 Gbps
                # host path. Do not add the second port: this is one route.
                effective = min(link_gbps, 100.0)
            facts.update(network_gbps=effective, link_gbps=link_gbps, interface=interface)
    except ValueError:
        pass
    return facts


def limits(settings, facts, transport, *, local_roles=1):
    """Bound startup hints, not a line-rate claim or a runtime feedback controller."""
    speed = settings.get("network_gbps", facts.get("network_gbps", 0))
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 0 <= speed <= 1600:
        raise ValueError("network_gbps must be between 0 and 1600")
    memory, streams = 128 * MIB, 4
    if transport in {"auto", "http2-direct"}:
        if speed >= 200:
            memory, streams = 1024 * MIB, 32
        elif speed >= 100:
            memory, streams = 1024 * MIB, 16
        elif speed >= 25:
            memory, streams = 256 * MIB, 8
    elif transport == "http2-ssh":
        memory, streams = 128 * MIB, 4
    available = facts.get("available_memory")
    if available is not None:
        # Leave headroom for Docker, TLS, Python, workloads and file cache.
        memory = min(memory, available // 16 // local_roles)
    else:
        memory = min(memory, 128 * MIB)
    memory = memory // MIB * MIB
    streams = min(streams, max(1, facts.get("cpus", 4)), max(1, memory // (8 * MIB)))
    memory = settings.get("max_buffer_bytes", memory)
    streams = settings.get("source_streams", streams)
    if type(memory) is not int or not 8 * MIB <= memory <= 64 << 30:
        raise ValueError("max_buffer_bytes must be between 8 MiB and 64 GiB; host may lack memory headroom")
    if type(streams) is not int or not 1 <= streams <= 32:
        raise ValueError("source_streams must be between 1 and 32")
    return {"max_buffer_bytes": memory, "source_streams": streams}
