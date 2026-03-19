"""Docker command string generation.

These functions are pure generators -- they produce command strings
that will be embedded into scripts and executed remotely via SSH.
They do not execute Docker commands directly.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Standard docker run options for DGX Spark GPU workloads
_DEFAULT_DOCKER_OPTS = [
    "--privileged",
    "--gpus all",
    "--ipc=host",
    "--shm-size=10.24gb",
    "--network host",
]


def docker_run_cmd(
    image: str,
    command: str = "",
    container_name: str | None = None,
    detach: bool = True,
    env: dict[str, str] | None = None,
    volumes: dict[str, str] | None = None,
    extra_opts: list[str] | None = None,
    auto_remove: bool = True,
    restart_policy: str | None = None,
) -> str:
    """Generate a ``docker run`` command string.

    Args:
        image: Container image reference (e.g. ``nvcr.io/nvidia/vllm:latest``).
        command: Command to run inside the container.
        container_name: Optional ``--name`` for the container.
        detach: Run in detached mode (``-d``).
        env: Environment variables to set (``-e KEY=VALUE``).
        volumes: Volume mounts (``-v host:container``).
        extra_opts: Additional docker run options.
        auto_remove: Add ``--rm`` flag (default True). Forced to False
            when *restart_policy* is set (Docker does not allow both).
        restart_policy: Docker restart policy (e.g. ``always``,
            ``unless-stopped``, ``on-failure:3``).

    Returns:
        Complete ``docker run`` command string.
    """
    # Docker does not allow --rm with --restart
    if restart_policy:
        auto_remove = False

    parts = ["docker", "run"]

    if detach:
        parts.append("-d")

    parts.extend(_DEFAULT_DOCKER_OPTS)

    if auto_remove:
        parts.append("--rm")

    if restart_policy:
        parts.extend(["--restart", restart_policy])

    if container_name:
        parts.extend(["--name", container_name])

    if env:
        for key, value in sorted(env.items()):
            parts.extend(["-e", f"{key}={value}"])

    if volumes:
        for host_path, container_path in sorted(volumes.items()):
            parts.extend(["-v", f"{host_path}:{container_path}"])

    if extra_opts:
        parts.extend(extra_opts)

    parts.append(image)

    if command:
        parts.append(command)

    result = " ".join(parts)

    if env:
        logger.debug("docker run %s env (%d vars):", container_name or image, len(env))
        for key, value in sorted(env.items()):
            logger.debug("  %s=%s", key, value)
    logger.debug("docker run command: %s", result)

    return result


def docker_exec_cmd(
    container_name: str,
    command: str,
    detach: bool = False,
    env: dict[str, str] | None = None,
) -> str:
    """Generate a ``docker exec`` command string.

    Args:
        container_name: Name of the running container.
        command: Command to execute inside the container.
        detach: Run in detached mode.
        env: Environment variables to set.

    Returns:
        Complete ``docker exec`` command string.
    """
    parts = ["docker", "exec"]
    if detach:
        parts.append("-d")
    if env:
        for key, value in sorted(env.items()):
            parts.extend(["-e", f"{key}={value}"])
    parts.extend([container_name, "bash", "-c", f"'{command}'"])
    return " ".join(parts)


def docker_stop_cmd(container_name: str, force: bool = True) -> str:
    """Generate a docker stop/rm command string.

    Args:
        container_name: Name of the container to stop.
        force: If True, use ``docker rm -f``; otherwise ``docker stop``.

    Returns:
        Command string that stops (and optionally removes) the container.
    """
    if force:
        return f"docker rm -f {container_name} 2>/dev/null || true"
    return f"docker stop {container_name} 2>/dev/null || true"


def docker_inspect_exists_cmd(image: str) -> str:
    """Generate a command to check if a docker image exists locally.

    Args:
        image: Image reference to check.

    Returns:
        Command string that exits 0 if the image exists locally.
    """
    return f"docker image inspect {image} >/dev/null 2>&1"


def docker_pull_cmd(image: str) -> str:
    """Generate a ``docker pull`` command.

    Args:
        image: Image reference to pull.

    Returns:
        Command string.
    """
    return f"docker pull {image}"


def docker_logs_cmd(
    container_name: str,
    follow: bool = False,
    tail: int | None = None,
) -> str:
    """Generate a ``docker logs`` command.

    Args:
        container_name: Name of the container.
        follow: If True, follow log output (``-f``).
        tail: Number of lines to show from the end.

    Returns:
        Command string.
    """
    parts = ["docker", "logs"]
    if follow:
        parts.append("-f")
    if tail is not None:
        parts.extend(["--tail", str(tail)])
    parts.append(container_name)
    return " ".join(parts)




def generate_container_name(cluster_id: str, role: str = "head") -> str:
    """Generate a deterministic container name.

    Args:
        cluster_id: Cluster identifier (e.g. ``sparkrun0``).
        role: Container role -- ``"head"``, ``"worker"``, or ``"solo"``.

    Returns:
        Container name in the form ``{cluster_id}_{role}``.
    """
    return f"{cluster_id}_{role}"


def generate_node_container_name(cluster_id: str, rank: int) -> str:
    """Generate a container name for a ranked node: ``{cluster_id}_node_{rank}``.

    Used by native-cluster runtimes (SGLang, vllm-distributed) where
    each node gets a rank-indexed container name.

    Args:
        cluster_id: Cluster identifier (e.g. ``sparkrun0``).
        rank: Node rank (0 = head, 1+ = workers).

    Returns:
        Container name string.
    """
    return "%s_node_%d" % (cluster_id, rank)


def enumerate_cluster_containers(cluster_id: str, num_hosts: int) -> list[str]:
    """Return all possible container names for a cluster.

    Covers solo, Ray (head/worker), and native (node_N) patterns so
    callers can clean up containers regardless of the runtime that
    created them.

    Args:
        cluster_id: Cluster identifier (e.g. ``sparkrun0``).
        num_hosts: Number of hosts in the cluster (used to generate
            native ``node_N`` names).

    Returns:
        List of container name strings.
    """
    names = [
        generate_container_name(cluster_id, "solo"),
        generate_container_name(cluster_id, "head"),
        generate_container_name(cluster_id, "worker"),
    ]
    for rank in range(num_hosts):
        names.append(generate_node_container_name(cluster_id, rank))
    return names
