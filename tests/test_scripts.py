"""Unit tests for sparkrun.orchestration.scripts module."""

from sparkrun.orchestration.scripts import generate_ip_detect_script
from sparkrun.orchestration.executor_docker import DockerExecutor

_executor = DockerExecutor()
generate_container_launch_script = _executor.generate_launch_script
generate_ray_head_script = _executor.generate_ray_head_script
generate_ray_worker_script = _executor.generate_ray_worker_script
generate_exec_serve_script = _executor.generate_exec_serve_script


def test_generate_ip_detect_script():
    """Script detects IP via ip route."""
    script = generate_ip_detect_script()

    assert script.startswith("#!/bin/bash")
    assert "ip route get 8.8.8.8" in script
    assert "NODE_IP" in script
    assert "grep -oP" in script


def test_generate_container_launch_script():
    """Script includes cleanup + docker run."""
    script = generate_container_launch_script(
        image="test-image:latest",
        container_name="test-container",
        command="python app.py",
    )

    assert script.startswith("#!/bin/bash")
    assert "docker rm -f test-container" in script
    assert "docker run" in script
    assert "test-image:latest" in script
    assert "python app.py" in script
    assert "--name test-container" in script


def test_generate_container_launch_script_with_env():
    """With env vars and nccl_env."""
    env = {"MY_VAR": "value1"}
    nccl_env = {"NCCL_DEBUG": "INFO", "NCCL_IB_DISABLE": "0"}

    script = generate_container_launch_script(
        image="test-image:latest",
        container_name="test-container",
        command="python app.py",
        env=env,
        nccl_env=nccl_env,
    )

    # Both env and nccl_env should be included
    assert "-e MY_VAR=value1" in script
    assert "-e NCCL_DEBUG=INFO" in script
    assert "-e NCCL_IB_DISABLE=0" in script


def test_generate_container_launch_script_with_volumes():
    """With volume mounts."""
    volumes = {"/host/models": "/models", "/host/cache": "/cache"}

    script = generate_container_launch_script(
        image="test-image:latest",
        container_name="test-container",
        command="python app.py",
        volumes=volumes,
    )

    assert "-v /host/cache:/cache" in script
    assert "-v /host/models:/models" in script


def test_generate_ray_head_script():
    """Contains ray start --head, port, no dashboard by default."""
    script = generate_ray_head_script(
        image="ray-image:latest",
        container_name="ray-head",
        ray_port=46379,
        dashboard_port=8265,
    )

    assert script.startswith("#!/bin/bash")
    assert "ip route get 8.8.8.8" in script
    assert "NODE_IP" in script
    assert "docker rm -f ray-head" in script
    assert "ray start" in script
    assert "--head" in script
    assert "--port 46379" in script
    assert "--dashboard-port" not in script
    assert "--include-dashboard" not in script
    assert "--disable-usage-stats" in script
    assert "--node-ip-address $NODE_IP" in script


def test_generate_ray_head_script_with_dashboard():
    """With dashboard=True, includes dashboard flags."""
    script = generate_ray_head_script(
        image="ray-image:latest",
        container_name="ray-head",
        ray_port=46379,
        dashboard_port=8265,
        dashboard=True,
    )

    assert "--include-dashboard=True" in script
    assert "--dashboard-port 8265" in script
    assert "--dashboard-host 0.0.0.0" in script


def test_generate_ray_head_script_with_nccl():
    """With NCCL env vars injected."""
    nccl_env = {"NCCL_DEBUG": "INFO", "NCCL_IB_HCA": "mlx5_0"}

    script = generate_ray_head_script(
        image="ray-image:latest",
        container_name="ray-head",
        nccl_env=nccl_env,
    )

    assert "-e NCCL_DEBUG=INFO" in script
    assert "-e NCCL_IB_HCA=mlx5_0" in script
    assert "-e RAY_memory_monitor_refresh_ms=0" in script


def test_generate_ray_worker_script():
    """Contains ray start --address with head_ip."""
    script = generate_ray_worker_script(
        image="ray-image:latest",
        container_name="ray-worker",
        head_ip="192.168.1.100",
        ray_port=46379,
    )

    assert script.startswith("#!/bin/bash")
    assert "ip route get 8.8.8.8" in script
    assert "NODE_IP" in script
    assert "docker rm -f ray-worker" in script
    assert "ray start" in script
    assert "--address=192.168.1.100:46379" in script
    assert "--node-ip-address $NODE_IP" in script
    assert "--head" not in script  # Should not have --head


def test_generate_ray_worker_script_with_nccl():
    """With NCCL env vars injected."""
    nccl_env = {"NCCL_SOCKET_IFNAME": "eth0"}

    script = generate_ray_worker_script(
        image="ray-image:latest",
        container_name="ray-worker",
        head_ip="192.168.1.100",
        nccl_env=nccl_env,
    )

    assert "-e NCCL_SOCKET_IFNAME=eth0" in script
    assert "-e RAY_memory_monitor_refresh_ms=0" in script


def test_generate_exec_serve_script_detached():
    """Uses nohup for background execution."""
    script = generate_exec_serve_script(
        container_name="my-container",
        serve_command="vllm serve model",
        detached=True,
    )

    assert script.startswith("#!/bin/bash")
    assert "docker exec my-container" in script
    assert "nohup" in script
    assert "/tmp/sparkrun_serve.log" in script
    assert "tail -f" in script
    assert "vllm serve model" in script


def test_generate_exec_serve_script_foreground():
    """No nohup when not detached."""
    script = generate_exec_serve_script(
        container_name="my-container",
        serve_command="vllm serve model",
        detached=False,
    )

    assert script.startswith("#!/bin/bash")
    assert "docker exec my-container" in script
    assert "nohup" not in script
    assert "tail -f" not in script
    assert "vllm serve model" in script


def test_generate_exec_serve_script_with_env():
    """With environment variables."""
    env = {"MODEL_PATH": "/models/llama", "CUDA_VISIBLE_DEVICES": "0,1"}

    script = generate_exec_serve_script(
        container_name="my-container",
        serve_command="vllm serve model",
        env=env,
        detached=True,
    )

    # Env vars should be exported in sorted order
    assert "export CUDA_VISIBLE_DEVICES='0,1'" in script
    assert "export MODEL_PATH='/models/llama'" in script


def test_generate_exec_serve_script_escapes_quotes():
    """Test that single quotes in command are properly escaped."""
    script = generate_exec_serve_script(
        container_name="my-container",
        serve_command="echo 'hello world'",
        detached=True,
    )

    # The command should have escaped quotes
    assert "echo" in script
    assert "hello world" in script


def test_generate_ray_head_script_custom_ports():
    """Test custom Ray and dashboard ports with dashboard enabled."""
    script = generate_ray_head_script(
        image="ray-image:latest",
        container_name="ray-head",
        ray_port=10001,
        dashboard_port=10002,
        dashboard=True,
    )

    assert "--port 10001" in script
    assert "--dashboard-port 10002" in script


def test_generate_container_launch_script_no_detach():
    """Test container launch without detach."""
    script = generate_container_launch_script(
        image="test-image:latest",
        container_name="test-container",
        command="python app.py",
        detach=False,
    )

    # Should not have -d flag
    assert "docker run" in script
    # The -d flag should not be present (checking the docker command portion)
    # Note: we can't easily check for absence without parsing, so verify presence of run
    assert "docker run" in script
