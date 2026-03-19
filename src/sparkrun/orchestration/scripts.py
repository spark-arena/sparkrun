"""Script generators for remote execution.

Each function generates a complete bash script as a string.
These scripts are fed to remote hosts via ``ssh host bash -s``.
"""

from __future__ import annotations

import logging

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
) -> str:
    """Generate a script that launches a Docker container.

    Combines cleanup of any existing container with the same name
    followed by ``docker run``.

    .. note:: This is a backward-compatibility shim.  New code should
       use :meth:`DockerExecutor.generate_launch_script` instead.

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
    from sparkrun.orchestration.executor_docker import DockerExecutor

    return DockerExecutor().generate_launch_script(
        image=image,
        container_name=container_name,
        command=command,
        env=env,
        volumes=volumes,
        nccl_env=nccl_env,
        detach=detach,
        extra_docker_opts=extra_docker_opts,
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
) -> str:
    """Generate a script that starts a Ray head node in a container.

    .. note:: Backward-compatibility shim.  New code should use
       :meth:`DockerExecutor.generate_ray_head_script`.
    """
    from sparkrun.orchestration.executor_docker import DockerExecutor

    return DockerExecutor().generate_ray_head_script(
        image=image,
        container_name=container_name,
        ray_port=ray_port,
        dashboard_port=dashboard_port,
        dashboard=dashboard,
        env=env,
        volumes=volumes,
        nccl_env=nccl_env,
    )


def generate_ray_worker_script(
    image: str,
    container_name: str,
    head_ip: str,
    ray_port: int = 46379,
    env: dict[str, str] | None = None,
    volumes: dict[str, str] | None = None,
    nccl_env: dict[str, str] | None = None,
) -> str:
    """Generate a script that starts a Ray worker node.

    .. note:: Backward-compatibility shim.  New code should use
       :meth:`DockerExecutor.generate_ray_worker_script`.
    """
    from sparkrun.orchestration.executor_docker import DockerExecutor

    return DockerExecutor().generate_ray_worker_script(
        image=image,
        container_name=container_name,
        head_ip=head_ip,
        ray_port=ray_port,
        env=env,
        volumes=volumes,
        nccl_env=nccl_env,
    )


def generate_exec_serve_script(
    container_name: str,
    serve_command: str,
    env: dict[str, str] | None = None,
    detached: bool = True,
) -> str:
    """Generate a script that executes the serve command inside a running container.

    .. note:: Backward-compatibility shim.  New code should use
       :meth:`DockerExecutor.generate_exec_serve_script`.
    """
    from sparkrun.orchestration.executor_docker import DockerExecutor

    return DockerExecutor().generate_exec_serve_script(
        container_name=container_name,
        serve_command=serve_command,
        env=env,
        detached=detached,
    )
