"""Unit tests for sparkrun.orchestration.docker module."""

from sparkrun.orchestration.docker import (
    docker_exec_cmd,
    docker_inspect_exists_cmd,
    docker_logs_cmd,
    docker_pull_cmd,
    docker_stop_cmd,
    generate_container_name,
)


def test_docker_exec_basic():
    """Generate basic docker exec."""
    cmd = docker_exec_cmd("my-container", "echo hello")

    assert cmd.startswith("docker exec")
    assert "my-container" in cmd
    assert "bash -c" in cmd
    from sparkrun.utils.shell import b64_encode_cmd

    expected = b64_encode_cmd("echo hello")
    assert expected in cmd


def test_docker_exec_detach():
    """With detach flag."""
    cmd = docker_exec_cmd("my-container", "echo hello", detach=True)
    assert "-d" in cmd


def test_docker_exec_with_env():
    """With environment variables."""
    env = {"PATH": "/usr/local/bin", "HOME": "/root"}
    cmd = docker_exec_cmd("my-container", "echo hello", env=env)

    # Should be sorted
    assert "-e HOME=/root" in cmd
    assert "-e PATH=/usr/local/bin" in cmd


def test_docker_exec_with_env_spaces():
    """With environment variables containing spaces."""
    env = {"MY_VAR": "hello world", "PATH": "/usr/local/bin"}
    cmd = docker_exec_cmd("my-container", "echo hello", env=env)

    # Should be sorted and properly quoted
    assert "-e 'MY_VAR=hello world'" in cmd
    assert "-e PATH=/usr/local/bin" in cmd


def test_docker_stop_force():
    """Verify docker rm -f."""
    cmd = docker_stop_cmd("my-container", force=True)
    assert "docker rm -f my-container" in cmd
    assert "2>/dev/null || true" in cmd


def test_docker_stop_graceful():
    """Verify docker stop."""
    cmd = docker_stop_cmd("my-container", force=False)
    assert "docker stop my-container" in cmd
    assert "2>/dev/null || true" in cmd


def test_docker_inspect_exists():
    """Verify inspect command format."""
    cmd = docker_inspect_exists_cmd("test-image:latest")
    assert cmd == "docker image inspect test-image:latest >/dev/null 2>&1"


def test_docker_pull():
    """Verify pull command format."""
    cmd = docker_pull_cmd("nvcr.io/nvidia/vllm:v0.5.3")
    assert cmd == "docker pull nvcr.io/nvidia/vllm:v0.5.3"


def test_docker_logs_basic():
    """Basic logs command."""
    cmd = docker_logs_cmd("my-container")
    assert cmd == "docker logs my-container"


def test_docker_logs_follow_tail():
    """With -f and --tail options."""
    cmd = docker_logs_cmd("my-container", follow=True, tail=100)
    assert cmd == "docker logs -f --tail 100 my-container"


def test_generate_container_name():
    """Test name generation for head, worker, solo roles."""
    assert generate_container_name("sparkrun0", "head") == "sparkrun0_head"
    assert generate_container_name("sparkrun0", "worker") == "sparkrun0_worker"
    assert generate_container_name("cluster-abc", "solo") == "cluster-abc_solo"
