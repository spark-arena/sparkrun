"""Script generators for remote execution.

Each function generates a complete bash script as a string.
These scripts are fed to remote hosts via ``ssh host bash -s``.
"""

from __future__ import annotations

import logging

from sparkrun.orchestration.docker import (
    docker_run_cmd,
    docker_stop_cmd,
)
from sparkrun.orchestration.primitives import merge_env
from sparkrun.scripts import read_script

logger = logging.getLogger(__name__)


def generate_ip_detect_script() -> str:
    """Generate a script that detects the host's management IP address.

    The detected IP is printed as the last line of stdout.

    Returns:
        Bash script content as a string.
    """
    return read_script("ip_detect.sh")


def generate_container_launch_script(
    image: str,
    container_name: str,
    command: str,
    env: dict[str, str] | None = None,
    volumes: dict[str, str] | None = None,
    nccl_env: dict[str, str] | None = None,
    detach: bool = True,
    extra_docker_opts: list[str] | None = None,
    auto_remove: bool = True,
    restart_policy: str | None = None,
) -> str:
    """Generate a script that launches a Docker container.

    Combines cleanup of any existing container with the same name
    followed by ``docker run``.

    Args:
        image: Container image reference.
        container_name: Name for the container.
        command: Command to run inside the container.
        env: Additional environment variables.
        volumes: Volume mounts (host_path -> container_path).
        nccl_env: NCCL-specific environment variables.
        detach: Run in detached mode.
        extra_docker_opts: Additional ``docker run`` options.

    Returns:
        Complete bash script as a string.
    """
    all_env = merge_env(nccl_env, env)

    cleanup = docker_stop_cmd(container_name)
    run_cmd = docker_run_cmd(
        image=image,
        command=command,
        container_name=container_name,
        detach=detach,
        env=all_env,
        volumes=volumes,
        extra_opts=extra_docker_opts,
        auto_remove=auto_remove,
        restart_policy=restart_policy,
    )

    template = read_script("container_launch.sh")
    return template.format(
        container_name=container_name,
        image=image,
        cleanup_cmd=cleanup,
        run_cmd=run_cmd,
    )


def generate_ray_head_script(
    image: str,
    container_name: str,
    ray_port: int = 46379,
    dashboard_port: int = 8265,
    dashboard: bool = False,
    env: dict[str, str] | None = None,
    volumes: dict[str, str] | None = None,
    nccl_env: dict[str, str] | None = None,
    auto_remove: bool = True,
    restart_policy: str | None = None,
) -> str:
    """Generate a script that starts a Ray head node in a container.

    The script:

    1. Detects the node IP.
    2. Cleans up any existing container with the same name.
    3. Launches the Ray head container.
    4. Outputs the node IP as the last line of stdout.

    Args:
        image: Container image reference.
        container_name: Name for the head container.
        ray_port: Ray GCS port.
        dashboard_port: Ray dashboard port.
        dashboard: Enable the Ray dashboard.
        env: Additional environment variables.
        volumes: Volume mounts.
        nccl_env: NCCL-specific environment variables.

    Returns:
        Complete bash script as a string.
    """
    all_env = merge_env({"RAY_memory_monitor_refresh_ms": "0"}, nccl_env, env)

    dashboard_flags = ""
    if dashboard:
        dashboard_flags = (
            f"--include-dashboard=True "
            f"--dashboard-host 0.0.0.0 "
            f"--dashboard-port {dashboard_port} "
        )

    cleanup = docker_stop_cmd(container_name)
    run_cmd = docker_run_cmd(
        image=image,
        command=(
            f"ray start --block --head "
            f"--port {ray_port} "
            f"--node-ip-address $NODE_IP "
            f"{dashboard_flags}"
            f"--disable-usage-stats"
        ),
        container_name=container_name,
        detach=True,
        env=all_env,
        volumes=volumes,
        auto_remove=auto_remove,
        restart_policy=restart_policy,
    )

    template = read_script("ray_head.sh")
    return template.format(
        cleanup_cmd=cleanup,
        run_cmd=run_cmd,
    )


def generate_ray_worker_script(
    image: str,
    container_name: str,
    head_ip: str,
    ray_port: int = 46379,
    env: dict[str, str] | None = None,
    volumes: dict[str, str] | None = None,
    nccl_env: dict[str, str] | None = None,
    auto_remove: bool = True,
    restart_policy: str | None = None,
) -> str:
    """Generate a script that starts a Ray worker node.

    The script detects the local node IP, cleans up any existing
    worker container, and launches a Ray worker that connects to
    the head node at *head_ip*.

    Args:
        image: Container image reference.
        container_name: Name for the worker container.
        head_ip: IP address of the Ray head node.
        ray_port: Ray GCS port on the head node.
        env: Additional environment variables.
        volumes: Volume mounts.
        nccl_env: NCCL-specific environment variables.

    Returns:
        Complete bash script as a string.
    """
    all_env = merge_env({"RAY_memory_monitor_refresh_ms": "0"}, nccl_env, env)

    cleanup = docker_stop_cmd(container_name)
    run_cmd = docker_run_cmd(
        image=image,
        command=(
            f"ray start --block "
            f"--address={head_ip}:{ray_port} "
            f"--node-ip-address $NODE_IP"
        ),
        container_name=container_name,
        detach=True,
        env=all_env,
        volumes=volumes,
        auto_remove=auto_remove,
        restart_policy=restart_policy,
    )

    template = read_script("ray_worker.sh")
    return template.format(
        cleanup_cmd=cleanup,
        run_cmd=run_cmd,
        head_ip=head_ip,
        ray_port=ray_port,
    )


def generate_exec_serve_script(
    container_name: str,
    serve_command: str,
    env: dict[str, str] | None = None,
    detached: bool = True,
) -> str:
    """Generate a script that executes the serve command inside a running container.

    If *detached* is True, uses ``nohup`` to survive SSH disconnects and
    tails the log file for initial output.

    Args:
        container_name: Name of the running container.
        serve_command: The inference serve command to run.
        env: Additional environment variables to export inside the container.
        detached: If True, run in background (survives SSH disconnect).

    Returns:
        Complete bash script as a string.
    """
    env_exports = ""
    if env:
        for key, value in sorted(env.items()):
            env_exports += f"export {key}='{value}'; "

    escaped_cmd = serve_command.replace("'", "'\\''")
    full_cmd = f"{env_exports}{escaped_cmd}"

    script_name = "exec_serve_detached.sh" if detached else "exec_serve_foreground.sh"
    template = read_script(script_name)
    return template.format(
        container_name=container_name,
        full_cmd=full_cmd,
    )
