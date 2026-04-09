"""Unit tests for the Executor abstraction.

Verifies that ``DockerExecutor`` produces identical output to the
existing ``docker.py`` functions, and that ``ExecutorConfig`` layering
works correctly.
"""

from __future__ import annotations

from sparkrun.orchestration.docker import (
    enumerate_cluster_containers,
    generate_container_name,
    generate_node_container_name,
)
from sparkrun.orchestration.executor import (
    EXECUTOR_DEFAULTS,
    Executor,
    ExecutorConfig,
)
from sparkrun.orchestration.executor_docker import DockerExecutor
from sparkrun.utils.shell import b64_encode_cmd

# ---------------------------------------------------------------------------
# ExecutorConfig tests
# ---------------------------------------------------------------------------


class TestExecutorConfig:
    """Tests for ExecutorConfig dataclass and from_chain."""

    def test_defaults(self):
        cfg = ExecutorConfig()
        assert cfg.auto_remove is True
        assert cfg.restart_policy is None
        assert cfg.privileged is True
        assert cfg.gpus == "all"
        assert cfg.ipc == "host"
        assert cfg.shm_size == "10.24gb"
        assert cfg.network == "host"

    def test_restart_forces_no_auto_remove(self):
        cfg = ExecutorConfig(restart_policy="always")
        assert cfg.auto_remove is False
        assert cfg.restart_policy == "always"

    def test_restart_overrides_explicit_auto_remove(self):
        cfg = ExecutorConfig(auto_remove=True, restart_policy="unless-stopped")
        assert cfg.auto_remove is False

    def test_from_chain_plain_dict(self):
        chain = {"auto_remove": False, "gpus": "0", "shm_size": "1gb"}
        cfg = ExecutorConfig.from_chain(chain)
        assert cfg.auto_remove is False
        assert cfg.gpus == "0"
        assert cfg.shm_size == "1gb"
        assert cfg.privileged is True  # default

    def test_from_chain_with_restart(self):
        chain = {"restart_policy": "on-failure:3", "auto_remove": True}
        cfg = ExecutorConfig.from_chain(chain)
        assert cfg.restart_policy == "on-failure:3"
        assert cfg.auto_remove is False  # forced by __post_init__

    def test_from_chain_string_bools(self):
        chain = {"auto_remove": "false", "privileged": "true"}
        cfg = ExecutorConfig.from_chain(chain)
        assert cfg.auto_remove is False
        assert cfg.privileged is True

    def test_from_chain_empty_restart_is_none(self):
        chain = {"restart_policy": ""}
        cfg = ExecutorConfig.from_chain(chain)
        assert cfg.restart_policy is None

    def test_user_field(self):
        cfg = ExecutorConfig(user="1000:1000")
        assert cfg.user == "1000:1000"

    def test_user_shell_user(self):
        cfg = ExecutorConfig(user="$SHELL_USER")
        assert cfg.user == "$SHELL_USER"

    def test_security_opt_field(self):
        cfg = ExecutorConfig(security_opt=["no-new-privileges"])
        assert cfg.security_opt == ["no-new-privileges"]

    def test_user_default_none(self):
        cfg = ExecutorConfig()
        assert cfg.user is None
        assert cfg.security_opt is None

    def test_from_chain_user_and_security_opt(self):
        chain = {"user": "$SHELL_USER", "security_opt": ["no-new-privileges"]}
        cfg = ExecutorConfig.from_chain(chain)
        assert cfg.user == "$SHELL_USER"
        assert cfg.security_opt == ["no-new-privileges"]

    def test_from_chain_security_opt_string(self):
        chain = {"security_opt": "no-new-privileges"}
        cfg = ExecutorConfig.from_chain(chain)
        assert cfg.security_opt == ["no-new-privileges"]

    def test_from_chain_empty_user_is_none(self):
        chain = {"user": ""}
        cfg = ExecutorConfig.from_chain(chain)
        assert cfg.user is None

    def test_memory_limit_field(self):
        cfg = ExecutorConfig(memory_limit="32G")
        assert cfg.memory_limit == "32G"

    def test_from_chain_memory_limit(self):
        chain = {"memory_limit": "16G"}
        cfg = ExecutorConfig.from_chain(chain)
        assert cfg.memory_limit == "16G"

    def test_config_chain_layering(self):
        """Verify config chain resolution: CLI > recipe > defaults."""
        from scitrera_app_framework.api import EnvPlacement, Variables

        cli_opts = {"auto_remove": False}
        recipe_opts = {"restart_policy": "always", "shm_size": "20gb"}
        chain = Variables(sources=(cli_opts, recipe_opts, EXECUTOR_DEFAULTS), env_placement=EnvPlacement.IGNORED)
        cfg = ExecutorConfig.from_chain(chain)

        assert cfg.auto_remove is False  # CLI wins
        assert cfg.restart_policy == "always"  # recipe
        assert cfg.shm_size == "20gb"  # recipe
        assert cfg.gpus == "all"  # default

    def test_config_chain_privileged_false(self):
        """Verify privileged=False survives config chain (falsy value preserved)."""
        from scitrera_app_framework.api import EnvPlacement, Variables

        cli_opts = {
            "privileged": False,
            "user": "$SHELL_USER",
            "security_opt": ["no-new-privileges"],
            "cap_add": ["IPC_LOCK", "SYS_PTRACE"],
            "ulimit": ["memlock=-1:-1"],
        }
        chain = Variables(sources=(cli_opts, {}, EXECUTOR_DEFAULTS), env_placement=EnvPlacement.IGNORED)
        cfg = ExecutorConfig.from_chain(chain)

        assert cfg.privileged is False
        assert cfg.user == "$SHELL_USER"
        assert cfg.security_opt == ["no-new-privileges"]
        assert cfg.cap_add == ["IPC_LOCK", "SYS_PTRACE"]
        assert cfg.ulimit == ["memlock=-1:-1"]


# ---------------------------------------------------------------------------
# Naming helper tests
# ---------------------------------------------------------------------------


class TestNamingHelpers:
    """Verify Executor naming helpers match docker.py functions."""

    def test_container_name(self):
        assert Executor.container_name("sparkrun0", "head") == generate_container_name("sparkrun0", "head")
        assert Executor.container_name("sparkrun0", "worker") == generate_container_name("sparkrun0", "worker")
        assert Executor.container_name("sparkrun0", "solo") == generate_container_name("sparkrun0", "solo")

    def test_node_container_name(self):
        for rank in range(5):
            assert Executor.node_container_name("sparkrun0", rank) == generate_node_container_name("sparkrun0", rank)

    def test_enumerate_containers(self):
        expected = enumerate_cluster_containers("sparkrun0", 3)
        actual = Executor.enumerate_containers("sparkrun0", 3)
        assert actual == expected


# ---------------------------------------------------------------------------
# ExecutorConfig with restart/auto_remove interaction
# ---------------------------------------------------------------------------


class TestDockerExecutorConfig:
    """Verify that ExecutorConfig settings propagate to docker commands."""

    def test_no_rm_flag(self):
        cfg = ExecutorConfig(auto_remove=False)
        executor = DockerExecutor(cfg)
        cmd = executor.run_cmd("img:latest")
        assert "--rm" not in cmd

    def test_rm_flag_default(self):
        executor = DockerExecutor()
        cmd = executor.run_cmd("img:latest")
        assert "--rm" in cmd

    def test_restart_policy(self):
        cfg = ExecutorConfig(restart_policy="always")
        executor = DockerExecutor(cfg)
        cmd = executor.run_cmd("img:latest")
        assert "--restart always" in cmd
        assert "--rm" not in cmd

    def test_restart_on_failure(self):
        cfg = ExecutorConfig(restart_policy="on-failure:3")
        executor = DockerExecutor(cfg)
        cmd = executor.run_cmd("img:latest")
        assert "--restart on-failure:3" in cmd
        assert "--rm" not in cmd

    def test_custom_shm_size(self):
        cfg = ExecutorConfig(shm_size="20gb")
        executor = DockerExecutor(cfg)
        cmd = executor.run_cmd("img:latest")
        assert "--shm-size=20gb" in cmd

    def test_custom_network(self):
        cfg = ExecutorConfig(network="bridge")
        executor = DockerExecutor(cfg)
        cmd = executor.run_cmd("img:latest")
        assert "--network=bridge" in cmd

    def test_memory_limit(self):
        cfg = ExecutorConfig(memory_limit="64G")
        executor = DockerExecutor(cfg)
        cmd = executor.run_cmd("img:latest")
        assert "--memory=64G" in cmd

    def test_no_privileged(self):
        cfg = ExecutorConfig(privileged=False)
        executor = DockerExecutor(cfg)
        cmd = executor.run_cmd("img:latest")
        assert "--privileged" not in cmd

    def test_user_explicit(self):
        cfg = ExecutorConfig(user="1000:1000")
        executor = DockerExecutor(cfg)
        cmd = executor.run_cmd("img:latest")
        assert "--user 1000:1000" in cmd
        assert "/etc/passwd" not in cmd
        assert "/etc/group" not in cmd
        assert "HOME=/tmp" not in cmd

    def test_user_shell_user_resolves(self):
        cfg = ExecutorConfig(user="$SHELL_USER")
        executor = DockerExecutor(cfg)
        cmd = executor.run_cmd("img:latest")
        assert "--user $(id -u):$(id -g)" in cmd
        assert "-v /etc/passwd:/etc/passwd:ro" in cmd
        assert "-v /etc/group:/etc/group:ro" in cmd
        assert "-e HOME=/tmp" in cmd

    def test_security_opt(self):
        cfg = ExecutorConfig(security_opt=["no-new-privileges"])
        executor = DockerExecutor(cfg)
        cmd = executor.run_cmd("img:latest")
        assert "--security-opt no-new-privileges" in cmd

    def test_security_opt_multiple(self):
        cfg = ExecutorConfig(security_opt=["no-new-privileges", "seccomp=unconfined"])
        executor = DockerExecutor(cfg)
        cmd = executor.run_cmd("img:latest")
        assert "--security-opt no-new-privileges" in cmd
        assert "--security-opt seccomp=unconfined" in cmd

    def test_rootless_config(self):
        """Verify the combination of settings that --rootless would produce."""
        cfg = ExecutorConfig(
            privileged=False,
            user="$SHELL_USER",
            security_opt=["no-new-privileges"],
            cap_add=["IPC_LOCK", "SYS_PTRACE", "SYS_NICE", "NET_ADMIN"],
            ulimit=["memlock=-1:-1"],
        )
        executor = DockerExecutor(cfg)
        cmd = executor.run_cmd("img:latest")
        assert "--privileged" not in cmd
        assert "--user $(id -u):$(id -g)" in cmd
        assert "-v /etc/passwd:/etc/passwd:ro" in cmd
        assert "-v /etc/group:/etc/group:ro" in cmd
        assert "-e HOME=/tmp" in cmd
        assert "--security-opt no-new-privileges" in cmd
        assert "--cap-add IPC_LOCK" in cmd
        assert "--cap-add SYS_PTRACE" in cmd
        assert "--cap-add SYS_NICE" in cmd
        assert "--cap-add NET_ADMIN" in cmd
        assert "--ulimit memlock=-1:-1" in cmd

    def test_cap_add_single(self):
        cfg = ExecutorConfig(cap_add=["SYS_PTRACE"])
        executor = DockerExecutor(cfg)
        cmd = executor.run_cmd("img:latest")
        assert "--cap-add SYS_PTRACE" in cmd

    def test_ulimit_single(self):
        cfg = ExecutorConfig(ulimit=["memlock=-1:-1"])
        executor = DockerExecutor(cfg)
        cmd = executor.run_cmd("img:latest")
        assert "--ulimit memlock=-1:-1" in cmd

    def test_no_user_by_default(self):
        executor = DockerExecutor()
        cmd = executor.run_cmd("img:latest")
        assert "--user" not in cmd
        assert "/etc/passwd" not in cmd
        assert "/etc/group" not in cmd
        assert "HOME=/tmp" not in cmd
        assert "--security-opt" not in cmd
        assert "--cap-add" not in cmd
        assert "--ulimit" not in cmd

    def test_shell_user_home_overridable(self):
        """Recipe env vars can override the default HOME=/tmp."""
        cfg = ExecutorConfig(user="$SHELL_USER")
        executor = DockerExecutor(cfg)
        cmd = executor.run_cmd("img:latest", env={"HOME": "/workspace"})
        # Default HOME=/tmp from _build_default_opts appears first
        assert "-e HOME=/tmp" in cmd
        # Recipe override appears later — Docker uses the last -e value
        assert "-e HOME=/workspace" in cmd
        assert cmd.index("-e HOME=/tmp") < cmd.index("-e HOME=/workspace")

    def test_run_cmd_env_and_volumes_spaces(self):
        executor = DockerExecutor()
        env = {"MY_VAR": "hello world", "PATH": "/usr/bin:/bin"}
        volumes = {"/host dir": "/container dir", "/tmp": "/tmp"}
        cmd = executor.run_cmd("img:latest", container_name="my_container", env=env, volumes=volumes)
        assert "-e 'MY_VAR=hello world'" in cmd
        assert "-e PATH=/usr/bin:/bin" in cmd
        assert "-v '/host dir:/container dir'" in cmd
        assert "-v /tmp:/tmp" in cmd

    def test_exec_cmd_env_spaces(self):
        executor = DockerExecutor()
        env = {"MY_VAR": "hello world", "PATH": "/usr/bin:/bin"}
        cmd = executor.exec_cmd("my_container", "echo hello", env=env)
        assert "-e 'MY_VAR=hello world'" in cmd
        assert "-e PATH=/usr/bin:/bin" in cmd

    def test_run_cmd_extra_opts(self):
        executor = DockerExecutor()
        cmd = executor.run_cmd(
            "img:latest",
            container_name="my_container",
            extra_opts=["-p", "8001:8001", "--gpus", "all", '--env="FOO=BAR"'],
        )
        assert "-p 8001:8001" in cmd
        assert "--gpus all" in cmd
        assert "--env=FOO=BAR" in cmd  # Quotes correctly consumed by shlex.split
        assert "img:latest" in cmd


# ---------------------------------------------------------------------------
# High-level script generator tests
# ---------------------------------------------------------------------------


class TestScriptGenerators:
    """Verify high-level script generators produce valid scripts."""

    def setup_method(self):
        self.executor = DockerExecutor()

    def test_generate_launch_script(self):
        script = self.executor.generate_launch_script(
            image="img:latest",
            container_name="sparkrun0_solo",
            command="sleep infinity",
        )
        assert "docker rm -f sparkrun0_solo" in script
        assert "docker run" in script
        assert b64_encode_cmd("sleep infinity") in script
        assert "img:latest" in script

    def test_generate_launch_script_with_env(self):
        script = self.executor.generate_launch_script(
            image="img:latest",
            container_name="test",
            command="sleep infinity",
            env={"KEY": "val"},
        )
        assert "KEY=val" in script

    def test_generate_exec_serve_script(self):
        script = self.executor.generate_exec_serve_script(
            container_name="sparkrun0_solo",
            serve_command="vllm serve model",
        )
        assert "sparkrun0_solo" in script
        expected_b64 = b64_encode_cmd("vllm serve model")
        assert expected_b64 in script

    def test_generate_exec_serve_script_with_spaces(self):
        script = self.executor.generate_exec_serve_script(
            container_name="my container",
            serve_command="echo hello",
        )
        assert "'my container'" in script

    def test_generate_exec_serve_script_escapes_double_quotes(self):
        script = self.executor.generate_exec_serve_script(
            container_name="sparkrun0_solo",
            serve_command='vllm serve --hf-overrides \'{"rope": "yarn"}\'',
        )
        expected_b64 = b64_encode_cmd('vllm serve --hf-overrides \'{"rope": "yarn"}\'')
        assert expected_b64 in script

    def test_generate_ray_head_script(self):
        script = self.executor.generate_ray_head_script(
            image="img:latest",
            container_name="sparkrun0_head",
            ray_port=46379,
        )
        assert "docker rm -f sparkrun0_head" in script
        # The command contains the port, stats, etc.
        # We check that a command starting with 'ray start' is encoded.
        assert b64_encode_cmd("ray start --block --head").split("=")[0] in script

    def test_generate_ray_worker_script(self):
        script = self.executor.generate_ray_worker_script(
            image="img:latest",
            container_name="sparkrun0_worker",
            head_ip="10.0.0.1",
            ray_port=46379,
        )
        assert "docker rm -f sparkrun0_worker" in script
        assert b64_encode_cmd("ray start --block").split("=")[0] in script

    def test_generate_node_script(self):
        script = self.executor.generate_node_script(
            image="img:latest",
            container_name="sparkrun0_node_0",
            serve_command="vllm serve model",
            label="vllm node",
        )
        assert "docker rm -f sparkrun0_node_0" in script
        assert "docker run" in script
        assert b64_encode_cmd("vllm serve model") in script
        assert "'vllm node'" in script

    def test_generate_node_script_with_restart(self):
        cfg = ExecutorConfig(restart_policy="always")
        executor = DockerExecutor(cfg)
        script = executor.generate_node_script(
            image="img:latest",
            container_name="sparkrun0_node_0",
            serve_command="vllm serve model",
        )
        assert "--restart always" in script
        assert "--rm" not in script
