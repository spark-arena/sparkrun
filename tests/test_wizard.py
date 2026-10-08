"""Tests for the sparkrun setup wizard."""

from __future__ import annotations

from unittest import mock

import pytest
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.core.cluster_manager import ClusterManager
from sparkrun.orchestration.ssh import RemoteResult


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cluster_mgr(tmp_path):
    """Return a ClusterManager rooted in an isolated temp directory."""
    config_root = tmp_path / "wizard_config"
    config_root.mkdir(parents=True, exist_ok=True)
    return ClusterManager(config_root)


@pytest.fixture
def patched_cluster_mgr(cluster_mgr):
    """Patch _get_cluster_manager everywhere to use the isolated ClusterManager.

    Without this, the CLI calls get_config_root(v=None) which falls back to
    the real ~/.config/sparkrun/ instead of the test's isolated directory.
    """
    with (
        mock.patch("sparkrun.cli._common._get_cluster_manager", return_value=cluster_mgr),
        mock.patch("sparkrun.cli._setup._get_cluster_manager", return_value=cluster_mgr),
        mock.patch("sparkrun.cli._setup._sudo._get_cluster_manager", return_value=cluster_mgr),
    ):
        yield cluster_mgr


# ---------------------------------------------------------------------------
# Basic help / invocation tests
# ---------------------------------------------------------------------------


def test_wizard_help(runner):
    """sparkrun setup wizard --help shows expected options."""
    result = runner.invoke(main, ["setup", "wizard", "--help"])
    assert result.exit_code == 0
    assert "--hosts" in result.output
    assert "--cluster" in result.output
    assert "--user" in result.output
    assert "--dry-run" in result.output
    assert "--yes" in result.output


# ---------------------------------------------------------------------------
# Smart setup routing tests
# ---------------------------------------------------------------------------


def test_setup_bare_no_default(runner, v, patched_cluster_mgr):
    """Bare 'sparkrun setup' with no default cluster invokes wizard."""
    assert patched_cluster_mgr.get_default() is None
    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout="CX7_DETECTED=0\n",
            stderr="",
        )
        result = runner.invoke(main, ["setup"], input="10.0.0.1\ntest\n\nn\nn\nn\n")
    assert "Welcome to sparkrun" in result.output


def test_setup_bare_with_default(runner, v, patched_cluster_mgr):
    """Bare 'sparkrun setup' with a default cluster shows help."""
    patched_cluster_mgr.create("mylab", ["10.0.0.1", "10.0.0.2"])
    patched_cluster_mgr.set_default("mylab")
    result = runner.invoke(main, ["setup"])
    assert "Setup and configuration commands" in result.output
    assert "Welcome to sparkrun" not in result.output


# ---------------------------------------------------------------------------
# Dry-run tests
# ---------------------------------------------------------------------------


def test_wizard_dry_run(runner, v, patched_cluster_mgr):
    """Wizard --dry-run with --hosts and --cluster previews without side effects."""
    with mock.patch("subprocess.run") as mock_sub:
        mock_sub.return_value = mock.Mock(
            returncode=0,
            stdout="CX7_DETECTED=0\n",
            stderr="",
        )
        result = runner.invoke(
            main,
            [
                "setup",
                "wizard",
                "--hosts",
                "10.0.0.1,10.0.0.2",
                "--cluster",
                "drytest",
                "--dry-run",
                "--yes",
            ],
        )

    assert result.exit_code == 0
    assert "Setup preview complete" in result.output
    assert "drytest" in result.output


# ---------------------------------------------------------------------------
# Cluster creation tests
# ---------------------------------------------------------------------------


def test_wizard_creates_cluster(runner, v, patched_cluster_mgr):
    """Wizard creates a cluster and sets it as default."""
    with (
        mock.patch("subprocess.run") as mock_sub,
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts") as mock_cx7,
        mock.patch("sparkrun.cli._setup._ssh._run_ssh_mesh", return_value=True),
    ):
        mock_sub.return_value = mock.Mock(
            returncode=0,
            stdout="CX7_DETECTED=0\n",
            stderr="",
        )
        mock_cx7.return_value = {
            "10.0.0.1": mock.Mock(detected=False),
            "10.0.0.2": mock.Mock(detected=False),
        }

        result = runner.invoke(
            main,
            [
                "setup",
                "wizard",
                "--hosts",
                "10.0.0.1,10.0.0.2",
                "--cluster",
                "newlab",
                "--yes",
            ],
        )

    assert result.exit_code == 0
    assert "Created cluster 'newlab'" in result.output
    assert patched_cluster_mgr.get_default() == "newlab"


def test_wizard_existing_cluster(runner, v, patched_cluster_mgr):
    """Wizard offers to use an existing cluster."""
    patched_cluster_mgr.create("existing", ["10.0.0.1", "10.0.0.2"])

    # Answer: use existing (y), select 1, skip SSH (n), skip sudoers (n), skip earlyoom (n)
    with mock.patch("subprocess.run") as mock_sub:
        mock_sub.return_value = mock.Mock(
            returncode=0,
            stdout="CX7_DETECTED=0\n",
            stderr="",
        )
        result = runner.invoke(
            main,
            [
                "setup",
                "wizard",
            ],
            input="y\n1\nn\nn\nn\n",
        )

    assert result.exit_code == 0
    assert "existing" in result.output


def test_wizard_name_collision(runner, v, patched_cluster_mgr):
    """Wizard handles cluster name collision by offering update."""
    patched_cluster_mgr.create("mylab", ["10.0.0.1", "10.0.0.2"])

    with (
        mock.patch("subprocess.run") as mock_sub,
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts") as mock_cx7,
        mock.patch("sparkrun.cli._setup._ssh._run_ssh_mesh", return_value=True),
    ):
        mock_sub.return_value = mock.Mock(
            returncode=0,
            stdout="CX7_DETECTED=0\n",
            stderr="",
        )
        mock_cx7.return_value = {"10.0.0.5": mock.Mock(detected=False)}
        result = runner.invoke(
            main,
            [
                "setup",
                "wizard",
                "--hosts",
                "10.0.0.5",
                "--cluster",
                "mylab",
                "--yes",
            ],
        )

    assert result.exit_code == 0
    assert "Updated cluster 'mylab'" in result.output


# ---------------------------------------------------------------------------
# Yes mode
# ---------------------------------------------------------------------------


def test_wizard_yes_mode(runner, v, patched_cluster_mgr):
    """--yes --hosts --cluster runs without interactive prompts."""
    with (
        mock.patch("subprocess.run") as mock_sub,
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts") as mock_cx7,
        mock.patch("sparkrun.cli._setup._ssh._run_ssh_mesh", return_value=True),
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel") as mock_rsp,
        mock.patch("sparkrun.orchestration.sudo.run_with_sudo_fallback") as mock_sudo,
        mock.patch("sparkrun.orchestration.sudo.run_sudo_script_on_host") as mock_sudo_host,
    ):
        mock_sub.return_value = mock.Mock(
            returncode=0,
            stdout="CX7_DETECTED=0\n",
            stderr="",
        )
        mock_cx7.return_value = {"10.0.0.1": mock.Mock(detected=False)}
        mock_rsp.return_value = [RemoteResult("10.0.0.1", 0, "", "")]
        mock_sudo.return_value = (
            {"10.0.0.1": RemoteResult("10.0.0.1", 0, "OK", "")},
            [],
        )
        mock_sudo_host.return_value = RemoteResult("10.0.0.1", 0, "OK", "")

        result = runner.invoke(
            main,
            [
                "setup",
                "wizard",
                "--hosts",
                "10.0.0.1",
                "--cluster",
                "auto",
                "--yes",
            ],
        )

    assert result.exit_code == 0
    assert "Setup finished" in result.output


# ---------------------------------------------------------------------------
# Single host
# ---------------------------------------------------------------------------


def test_wizard_single_host(runner, v, patched_cluster_mgr):
    """A single host only needs controller access, not a peer mesh."""
    with (
        mock.patch("subprocess.run") as mock_sub,
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts") as mock_cx7,
        mock.patch("sparkrun.cli._setup._ssh._run_ssh_mesh", return_value=True) as mock_mesh,
    ):
        mock_sub.return_value = mock.Mock(
            returncode=0,
            stdout="CX7_DETECTED=0\n",
            stderr="",
        )
        mock_cx7.return_value = {"10.0.0.1": mock.Mock(detected=False)}

        result = runner.invoke(
            main,
            [
                "setup",
                "wizard",
                "--hosts",
                "10.0.0.1",
                "--cluster",
                "solo",
                "--yes",
            ],
        )

    assert result.exit_code == 0
    mock_mesh.assert_not_called()


# ---------------------------------------------------------------------------
# CX7 detection paths
# ---------------------------------------------------------------------------


def test_wizard_no_cx7(runner, v, patched_cluster_mgr):
    """No CX7 detected shows host prompt."""
    with mock.patch("subprocess.run") as mock_sub:
        mock_sub.return_value = mock.Mock(
            returncode=0,
            stdout="CX7_DETECTED=0\n",
            stderr="",
        )
        result = runner.invoke(
            main,
            [
                "setup",
                "wizard",
                "--dry-run",
            ],
            input="10.0.0.1,10.0.0.2\ntestcluster\n\nn\nn\nn\n",
        )

    assert result.exit_code == 0
    assert "Enter host IPs/hostnames" in result.output
    mock_sub.assert_not_called()


def test_wizard_cx7_peer_discovery(runner, v, patched_cluster_mgr):
    """Local CX7 detection + peer scan populates host list."""
    cx7_output = (
        "CX7_DETECTED=1\n"
        "CX7_MGMT_IP=10.0.0.1\n"
        "CX7_MGMT_IFACE=eth0\n"
        "CX7_NETPLAN_EXISTS=0\n"
        "CX7_SUDO_OK=1\n"
        "CX7_IFACE_COUNT=2\n"
        "CX7_IFACE_0_NAME=enp1\n"
        "CX7_IFACE_0_IP=192.168.11.1\n"
        "CX7_IFACE_0_PREFIX=24\n"
        "CX7_IFACE_0_SUBNET=192.168.11.0/24\n"
        "CX7_IFACE_0_MTU=9000\n"
        "CX7_IFACE_0_STATE=up\n"
        "CX7_IFACE_0_HCA=mlx5_0\n"
        "CX7_IFACE_1_NAME=enp2\n"
        "CX7_IFACE_1_IP=192.168.12.1\n"
        "CX7_IFACE_1_PREFIX=24\n"
        "CX7_IFACE_1_SUBNET=192.168.12.0/24\n"
        "CX7_IFACE_1_MTU=9000\n"
        "CX7_IFACE_1_STATE=up\n"
        "CX7_IFACE_1_HCA=mlx5_1\n"
        "CX7_USED_SUBNETS=10.0.0.0/24\n"
    )

    with (
        mock.patch("sparkrun.core.setup_probe.local_setup_step_supported", return_value=True),
        mock.patch("subprocess.run") as mock_sub,
        mock.patch(
            "sparkrun.orchestration.networking.discover_cx7_peers",
            return_value=["192.168.11.2"],
        ),
    ):
        mock_sub.return_value = mock.Mock(
            returncode=0,
            stdout=cx7_output,
            stderr="",
        )
        result = runner.invoke(
            main,
            [
                "setup",
                "wizard",
            ],
            input="10.0.0.1,10.0.0.2\npeertest\n\nn\nn\nn\n",
        )

    assert result.exit_code == 0
    assert "CX7 interfaces detected" in result.output
    assert "peer" in result.output.lower()


# ---------------------------------------------------------------------------
# Sudo password tests
# ---------------------------------------------------------------------------


def test_wizard_nopasswd(runner, v, patched_cluster_mgr):
    """No password prompt when all hosts have NOPASSWD."""
    with (
        mock.patch("subprocess.run") as mock_sub,
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts") as mock_cx7,
        mock.patch("sparkrun.cli._setup._ssh._run_ssh_mesh", return_value=True),
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel") as mock_rsp,
        mock.patch("sparkrun.orchestration.sudo.run_with_sudo_fallback") as mock_sudo,
        mock.patch("sparkrun.orchestration.sudo.run_sudo_script_on_host") as mock_sudo_host,
    ):
        mock_sub.return_value = mock.Mock(
            returncode=0,
            stdout="CX7_DETECTED=0\n",
            stderr="",
        )
        mock_cx7.return_value = {"10.0.0.1": mock.Mock(detected=False)}
        mock_rsp.return_value = [RemoteResult("10.0.0.1", 0, "", "")]
        mock_sudo.return_value = (
            {"10.0.0.1": RemoteResult("10.0.0.1", 0, "OK", "")},
            [],
        )
        mock_sudo_host.return_value = RemoteResult("10.0.0.1", 0, "OK", "")

        result = runner.invoke(
            main,
            [
                "setup",
                "wizard",
                "--hosts",
                "10.0.0.1",
                "--cluster",
                "nopasswd",
                "--yes",
            ],
        )

    assert result.exit_code == 0
    assert "[sudo] password" not in result.output


def _action_probe(hosts, **kwargs):
    from sparkrun.core.hardware import default_dgx_spark_hardware
    from sparkrun.core.setup_models import HostState
    from sparkrun.core.setup_probe import resolve_setup_context

    states = {
        h: HostState(
            h,
            facts={
                "CHECK_OS": "Linux",
                "CHECK_APT": "1",
                "CHECK_SYSTEMD": "1",
                "CHECK_USER": "tester",
                "CHECK_DOCKER_INSTALLED": "1",
                "CHECK_DOCKER_GROUP": "0",
                "CHECK_DOCKER_USABLE": "0",
                "CHECK_NVIDIA_CTK": "1",
                "CHECK_CDI_SPEC": "0",
                "CHECK_SUDOERS_CHOWN": "1",
                "CHECK_SUDOERS_DROPCACHES": "1",
                "CHECK_EARLYOOM_ACTIVE": "0",
            },
            hardware=default_dgx_spark_hardware(),
        )
        for h in hosts
    }
    return states, resolve_setup_context(
        states, config=kwargs["config"], cluster=kwargs.get("cluster"), cluster_name=kwargs.get("cluster_name")
    )


def test_wizard_generates_cdi(runner, v, patched_cluster_mgr):
    """Only a target whose resolved executor needs CDI gets a spec generated."""
    from sparkrun.core.setup_manifest import ManifestManager

    patched_cluster_mgr.create("cditest", ["10.0.0.1"], executor_config={"gpu_access_mode": "cdi"})
    with (
        mock.patch("subprocess.run", return_value=mock.Mock(returncode=0, stdout="CX7_DETECTED=0\n", stderr="")),
        mock.patch("sparkrun.core.setup_probe.probe_setup_hosts", side_effect=_action_probe),
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=[RemoteResult("10.0.0.1", 0, "OK", "")]),
        mock.patch(
            "sparkrun.orchestration.sudo.dispatch_sudo_script",
            return_value=RemoteResult("10.0.0.1", 0, "GENERATED: /etc/cdi/nvidia.yaml", ""),
        ) as dispatch,
    ):
        result = runner.invoke(main, ["setup", "wizard", "--cluster", "cditest", "--yes"])
    assert result.exit_code == 0, result.output
    assert any("nvidia-ctk cdi generate" in call.args[1] for call in dispatch.call_args_list)
    manifest = ManifestManager(patched_cluster_mgr.clusters_dir).load("cditest")
    assert manifest is not None
    assert manifest.phases["nvidia_cdi"].hosts == ["10.0.0.1"]


def test_wizard_cdi_dry_run(runner, v, patched_cluster_mgr):
    """Wizard --dry-run previews the CDI phase without running sudo."""
    with mock.patch("subprocess.run") as mock_sub:
        mock_sub.return_value = mock.Mock(returncode=0, stdout="CX7_DETECTED=0\n", stderr="")
        result = runner.invoke(
            main,
            ["setup", "wizard", "--hosts", "10.0.0.1", "--cluster", "cdidry", "--dry-run", "--yes"],
        )

    assert result.exit_code == 0
    assert "NVIDIA CDI spec" in result.output
    mock_sub.assert_not_called()


def test_wizard_sudo_password_reuse(runner, v, patched_cluster_mgr):
    """Password collected once is reused across phases."""
    with (
        mock.patch("sparkrun.core.setup_probe.probe_setup_hosts", side_effect=_action_probe),
        mock.patch("subprocess.run") as mock_sub,
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts") as mock_cx7,
        mock.patch("sparkrun.cli._setup._ssh._run_ssh_mesh", return_value=True),
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel") as mock_rsp,
        mock.patch("sparkrun.orchestration.sudo.run_with_sudo_fallback") as mock_sudo,
        mock.patch("sparkrun.orchestration.sudo.run_sudo_script_on_host") as mock_sudo_host,
    ):
        mock_sub.return_value = mock.Mock(
            returncode=0,
            stdout="CX7_DETECTED=0\n",
            stderr="",
        )
        mock_cx7.return_value = {"10.0.0.1": mock.Mock(detected=False)}
        mock_rsp.side_effect = lambda hosts, script, **kwargs: [
            RemoteResult(h, 1, "", "sudo: password required") if script == "sudo -n true" else RemoteResult(h, 0, "OK", "") for h in hosts
        ]
        mock_sudo.return_value = (
            {"10.0.0.1": RemoteResult("10.0.0.1", 0, "OK", "")},
            [],
        )
        mock_sudo_host.return_value = RemoteResult("10.0.0.1", 0, "OK", "")

        result = runner.invoke(
            main,
            [
                "setup",
                "wizard",
                "--hosts",
                "10.0.0.1",
                "--cluster",
                "sudotest",
                "--yes",
            ],
            input="testpassword\n",
        )

    assert result.exit_code == 0
    password_prompts = result.output.count("[sudo] password")
    assert password_prompts == 1, "Expected 1 password prompt, got %d\n%s" % (password_prompts, result.output)


# ---------------------------------------------------------------------------
# Already-configured topology: the wizard agrees with `setup check`
# (scenarios adapted from PR #309)
# ---------------------------------------------------------------------------


def _good_cx7(host, mtu=9000):
    from sparkrun.orchestration.networking import CX7HostDetection, CX7Interface, CX7Persistence

    ifaces = [
        CX7Interface(
            name="enp1s0f%dnp%d" % (idx, idx),
            ip="192.168.1%d.%s" % (idx, host.rsplit(".", 1)[-1]),
            prefix=24,
            subnet="192.168.1%d.0/24" % idx,
            mtu=mtu,
            state="up",
            hca="mlx5_%d" % idx,
            persistence=CX7Persistence.PERSISTENT,
            persistence_source="netplan",
        )
        for idx in range(2)
    ]
    return CX7HostDetection(host=host, interfaces=ifaces, netplan_exists=True, detected=True)


def _topology_probe(mesh_ok_by_host, calls, mtu=9000):
    """A probe_setup_hosts stand-in: configured CX7, mesh results per host."""

    def probe(hosts, **kwargs):
        from sparkrun.core.hardware import default_dgx_spark_hardware
        from sparkrun.core.setup_models import HostState
        from sparkrun.core.setup_probe import resolve_setup_context

        calls.append(kwargs)
        peers = len(hosts) - 1 + len(kwargs.get("extra_mesh_peers") or ())
        states = {
            h: HostState(
                h,
                facts={
                    "CHECK_OS": "Linux",
                    "CHECK_USER": "tester",
                    "CHECK_NETPLAN": "1",
                    "CHECK_MESH_TOTAL": str(peers),
                    "CHECK_MESH_OK": str(peers if mesh_ok_by_host.get(h, True) else peers - 1),
                },
                hardware=default_dgx_spark_hardware(),
                cx7=_good_cx7(h, mtu),
            )
            for h in hosts
        }
        context = resolve_setup_context(
            states, config=kwargs["config"], cluster=kwargs.get("cluster"), cluster_name=kwargs.get("cluster_name")
        )
        context.extra_mesh_peers = tuple(kwargs.get("extra_mesh_peers") or ())
        return states, context

    return probe


HOSTS = ["10.0.0.1", "10.0.0.2"]


def _invoke_topology_wizard(runner, probe, *extra_args, input=None, mtu=9000, control_sshd=True):
    with (
        mock.patch("subprocess.run", return_value=mock.Mock(returncode=0, stdout="CX7_DETECTED=0\n", stderr="")),
        mock.patch("sparkrun.core.setup_probe.probe_setup_hosts", side_effect=probe),
        mock.patch("sparkrun.utils.net.local_ip_for", return_value="10.0.0.9"),
        mock.patch("sparkrun.utils.net.accepts_tcp", return_value=control_sshd),
        mock.patch("sparkrun.cli._setup._ssh._run_ssh_mesh", return_value=True) as mesh,
        mock.patch("sparkrun.cli._setup._ssh._detect_and_update_mgmt_ips") as mgmt,
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value={h: _good_cx7(h, mtu) for h in HOSTS}),
        mock.patch("sparkrun.orchestration.networking.apply_cx7_plan", return_value=[]) as cx7_apply,
        mock.patch(
            "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
            side_effect=lambda hs, *a, **k: [RemoteResult(h, 0, "", "") for h in hs],
        ),
        mock.patch("sparkrun.orchestration.sudo.dispatch_sudo_script", return_value=RemoteResult("x", 0, "OK", "")),
    ):
        result = runner.invoke(main, ["setup", "wizard", "--hosts", ",".join(HOSTS), "--cluster", "topo", *extra_args], input=input)
    return result, mesh, mgmt, cx7_apply


def test_wizard_skips_configured_mesh_and_cx7(runner, v, patched_cluster_mgr):
    """Every mesh leg and the CX7 plan already hold → nothing asked, nothing applied."""
    calls = []
    # No --yes: unrelated prompts take their defaults; these phases must not ask.
    result, mesh, mgmt, cx7_apply = _invoke_topology_wizard(runner, _topology_probe({}, calls), input="\n" * 20)

    assert result.exit_code == 0, result.output
    assert "SSH mesh already working across 2 host(s) + this machine" in result.output
    assert "All hosts already configured." in result.output
    assert "Set up SSH mesh" not in result.output
    assert "Configure CX7 networking?" not in result.output
    mesh.assert_not_called()
    cx7_apply.assert_not_called()
    # The skipped mesh still counts as working for management-IP normalization.
    mgmt.assert_called_once()
    # The control machine is probed as a mesh peer, not just host <-> host.
    assert calls and all(list(c["extra_mesh_peers"]) == ["10.0.0.9"] for c in calls)


def test_wizard_saves_topology_for_an_already_configured_cx7(runner, v, patched_cluster_mgr):
    """The planner still runs, so the cluster records its topology (NCCL reads it)."""
    result, *_ = _invoke_topology_wizard(runner, _topology_probe({}, []), "--yes")
    assert result.exit_code == 0, result.output
    assert patched_cluster_mgr.get("topo").topology == "switch"


def test_wizard_cx7_checks_passing_but_mtu_off_plan_asks_first(runner, v, patched_cluster_mgr):
    """Per-host checks pass at MTU 1500; the planner disagrees, so the user decides."""
    result, _mesh, _mgmt, cx7_apply = _invoke_topology_wizard(
        runner, _topology_probe({}, [], mtu=1500), input="\n" + "n\n" + "\n" * 20, mtu=1500
    )
    assert result.exit_code == 0, result.output
    assert "differ from the cluster plan" in result.output
    assert "All hosts already configured." not in result.output
    cx7_apply.assert_not_called()


def test_wizard_meshes_when_control_leg_missing(runner, v, patched_cluster_mgr):
    """One host cannot reach a peer (here: the control machine) → the mesh runs."""
    result, mesh, _mgmt, _cx7 = _invoke_topology_wizard(runner, _topology_probe({"10.0.0.2": False}, []), "--yes")

    assert result.exit_code == 0, result.output
    assert "SSH mesh already working" not in result.output
    mesh.assert_called_once()
    assert "10.0.0.9" in mesh.call_args.args[0]


def test_wizard_without_control_sshd_does_not_require_the_control_leg(runner, v, patched_cluster_mgr):
    """No SSH server here: the host→control leg cannot exist, so it is not demanded."""
    calls = []
    result, mesh, _mgmt, _cx7 = _invoke_topology_wizard(runner, _topology_probe({}, calls), "--yes", control_sshd=False)
    assert result.exit_code == 0, result.output
    assert "SSH mesh already working across 2 host(s) — skipping." in result.output
    assert all(list(c["extra_mesh_peers"]) == [] for c in calls)
    mesh.assert_not_called()


def test_wizard_cross_user_never_probes_the_control_leg(runner, v, patched_cluster_mgr):
    """A different SSH user does not join the control machine to the mesh."""
    calls = []
    with mock.patch("sparkrun.cli._setup._ssh._default_ssh_user", return_value="localperson"):
        result, *_ = _invoke_topology_wizard(runner, _topology_probe({}, calls), "--yes", "--user", "clusteruser")
    assert result.exit_code == 0, result.output
    assert calls and all(list(c["extra_mesh_peers"]) == [] for c in calls)


def test_wizard_dry_run_never_claims_configured(runner, v, patched_cluster_mgr):
    """Without probe facts nothing is "already configured" — dry-run previews as before."""
    with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0, stdout="CX7_DETECTED=0\n", stderr="")):
        result = runner.invoke(main, ["setup", "wizard", "--hosts", "10.0.0.1,10.0.0.2", "--cluster", "topo", "--dry-run", "--yes"])
    assert result.exit_code == 0, result.output
    assert "already working" not in result.output
    assert "already configured" not in result.output
