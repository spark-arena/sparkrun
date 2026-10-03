"""Tests for the sparkrun setup wizard."""

from __future__ import annotations

from unittest import mock

import pytest
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.api.setup import SshProbe
from sparkrun.cli._setup._check import HostState, SetupProbe
from sparkrun.core.cluster_manager import ClusterManager
from sparkrun.orchestration.docker_info import DockerDriverInfo
from sparkrun.orchestration.networking import CX7HostDetection, CX7Interface, CX7Persistence
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

    Also stubs the wizard's readiness probe to "no data" so every phase
    prompts and applies exactly as it did before that probe existed — tests
    that exercise the probe itself patch it again in their own body.
    """
    from sparkrun.cli._setup._check import SetupProbe

    with (
        mock.patch("sparkrun.cli._common._get_cluster_manager", return_value=cluster_mgr),
        mock.patch("sparkrun.cli._setup._get_cluster_manager", return_value=cluster_mgr),
        mock.patch("sparkrun.cli._setup._sudo._get_cluster_manager", return_value=cluster_mgr),
        mock.patch("sparkrun.cli._setup._check.probe_host_states", return_value=SetupProbe()),
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
    assert "Setup Complete!" in result.output
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
    assert "Setup Complete!" in result.output


# ---------------------------------------------------------------------------
# Single host
# ---------------------------------------------------------------------------


def test_wizard_single_host(runner, v, patched_cluster_mgr):
    """Single remote host still runs SSH mesh with control machine."""
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
    assert mock_mesh.called


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
    assert "No CX7 interfaces detected" in result.output


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
                "--dry-run",
            ],
            input="10.0.0.1,10.0.0.2\npeertest\n\nn\nn\nn\n",
        )

    assert result.exit_code == 0
    assert "CX7 detected" in result.output
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


def test_wizard_generates_cdi(runner, v, patched_cluster_mgr):
    """Wizard runs the NVIDIA CDI phase and reports the generated spec."""
    with (
        mock.patch("subprocess.run") as mock_sub,
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts") as mock_cx7,
        mock.patch("sparkrun.cli._setup._ssh._run_ssh_mesh", return_value=True),
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel") as mock_rsp,
        mock.patch("sparkrun.orchestration.sudo.run_with_sudo_fallback") as mock_sudo,
        mock.patch("sparkrun.orchestration.sudo.run_sudo_script_on_host") as mock_sudo_host,
    ):
        mock_sub.return_value = mock.Mock(returncode=0, stdout="CX7_DETECTED=0\n", stderr="")
        mock_cx7.return_value = {"10.0.0.1": mock.Mock(detected=False)}
        mock_rsp.return_value = [RemoteResult("10.0.0.1", 0, "", "")]
        mock_sudo.return_value = (
            {"10.0.0.1": RemoteResult("10.0.0.1", 0, "GENERATED: /etc/cdi/nvidia.yaml (1 device(s))", "")},
            [],
        )
        mock_sudo_host.return_value = RemoteResult("10.0.0.1", 0, "OK", "")

        result = runner.invoke(
            main,
            ["setup", "wizard", "--hosts", "10.0.0.1", "--cluster", "cditest", "--yes"],
        )

    assert result.exit_code == 0
    assert "Phase 4b: NVIDIA CDI" in result.output
    assert "GENERATED: /etc/cdi/nvidia.yaml" in result.output
    assert "CDI:" in result.output

    # The phase is recorded in the setup manifest.
    from sparkrun.core.setup_manifest import ManifestManager

    manifest = ManifestManager(patched_cluster_mgr.clusters_dir).load("cditest")
    assert manifest is not None
    assert "nvidia_cdi" in manifest.phases


def test_wizard_cdi_dry_run(runner, v, patched_cluster_mgr):
    """Wizard --dry-run previews the CDI phase without running sudo."""
    with mock.patch("subprocess.run") as mock_sub:
        mock_sub.return_value = mock.Mock(returncode=0, stdout="CX7_DETECTED=0\n", stderr="")
        result = runner.invoke(
            main,
            ["setup", "wizard", "--hosts", "10.0.0.1", "--cluster", "cdidry", "--dry-run", "--yes"],
        )

    assert result.exit_code == 0
    assert "Would generate CDI spec" in result.output


def test_wizard_sudo_password_reuse(runner, v, patched_cluster_mgr):
    """Password collected once is reused across phases."""
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
        mock_rsp.return_value = [RemoteResult("10.0.0.1", 1, "", "sudo: password required")]
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
    assert password_prompts == 1, "Expected 1 password prompt, got %d" % password_prompts


# ---------------------------------------------------------------------------
# Readiness probe: skip phases whose checks already pass (the `setup check`
# agreement — the wizard must detect an already-configured cluster).
# ---------------------------------------------------------------------------

_ALL_GOOD_FACTS = {
    "CHECK_USER": "drew",
    "CHECK_DOCKER_INSTALLED": "1",
    "CHECK_DOCKER_USABLE": "1",
    "CHECK_DOCKER_GROUP": "1",
    "CHECK_GPU_PRESENT": "1",
    "CHECK_NVIDIA_CTK": "1",
    "CHECK_CDI_SPEC": "1",
    "CHECK_EARLYOOM_INSTALLED": "1",
    "CHECK_EARLYOOM_ACTIVE": "1",
    "CHECK_SUDOERS_CHOWN": "1",
    "CHECK_SUDOERS_DROPCACHES": "1",
    "CHECK_MESH_TOTAL": "1",
    "CHECK_MESH_OK": "1",
    "CHECK_COMPLETE": "1",
}


def _good_cx7(host: str) -> CX7HostDetection:
    """A configured CX7 pair: both ports up with persistent addresses."""
    ifaces = [
        CX7Interface(
            name="enp%d" % idx,
            ip="192.168.1%d.1" % idx,
            prefix=24,
            subnet="192.168.1%d.0/24" % idx,
            mtu=9000,
            state="up",
            hca="mlx5_%d" % idx,
            persistence=CX7Persistence.PERSISTENT,
            persistence_source="netplan",
        )
        for idx in range(2)
    ]
    return CX7HostDetection(host=host, interfaces=ifaces, netplan_exists=True, detected=True)


def _probed(facts_by_host: dict[str, dict[str, str]]) -> SetupProbe:
    """Build a SetupProbe as probe_host_states would, from per-host facts."""
    return SetupProbe(states={host: HostState(host=host, facts=facts, cx7=_good_cx7(host)) for host, facts in facts_by_host.items()})


def _gate_ok(hosts):
    return mock.patch("sparkrun.api.setup.probe_ssh_access", return_value=[SshProbe(host=h, ok=True) for h in hosts])


def test_wizard_skips_already_configured_phases(runner, v, patched_cluster_mgr):
    """Every check green → the wizard reports and skips, prompting for nothing.

    The user reported exactly this: CX7 and the rest already set up, but the
    wizard walked them through every step again while `sparkrun setup check`
    correctly reported all-clear.
    """
    hosts = ["10.0.0.1", "10.0.0.2"]
    probe = _probed({h: dict(_ALL_GOOD_FACTS) for h in hosts})

    with (
        mock.patch("shutil.which", return_value=False),  # skip the uv-tool install branch
        mock.patch("subprocess.run") as mock_sub,
        _gate_ok(hosts),
        mock.patch("sparkrun.cli._setup._check.probe_host_states", return_value=probe),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value={h: mock.Mock(detected=False) for h in hosts}),
        mock.patch("sparkrun.models.distribute.detect_shared_cache", return_value=False),
        mock.patch("sparkrun.cli._setup._ssh._run_ssh_mesh") as mock_mesh,
        mock.patch("sparkrun.cli._setup._ssh._detect_and_update_mgmt_ips") as mock_mgmt,
        mock.patch("sparkrun.orchestration.sudo.run_with_sudo_fallback") as mock_sudo,
        mock.patch("sparkrun.telemetry.emit.send_event"),
    ):
        mock_sub.return_value = mock.Mock(returncode=0, stdout="CX7_DETECTED=0\n", stderr="")
        result = runner.invoke(
            main,
            ["setup", "wizard", "--hosts", ",".join(hosts), "--cluster", "donelab", "--yes"],
        )

    assert result.exit_code == 0, result.output
    assert "Checking existing configuration on 2 host(s)" in result.output
    assert "SSH mesh already working" in result.output
    assert "CX7 already configured and persisted" in result.output
    assert "already in the docker group" in result.output
    assert "CDI spec already present" in result.output
    assert "Sudoers entries already installed" in result.output
    assert "earlyoom already installed and active" in result.output
    assert "Setup Complete!" in result.output
    assert "[sudo] password" not in result.output

    # Skipped means skipped: no mesh fan-out, no sudo payload anywhere.
    mock_mesh.assert_not_called()
    mock_sudo.assert_not_called()
    # But the mesh skip still counts as working for downstream phases
    # (management-IP normalization runs exactly as after a fresh mesh).
    mock_mgmt.assert_called_once()


def test_wizard_probe_failure_prompts_and_applies_as_before(runner, v, patched_cluster_mgr):
    """No probe data (probe failed) → every phase behaves as it did before."""
    with (
        mock.patch("shutil.which", return_value=False),  # skip the uv-tool install branch
        mock.patch("subprocess.run") as mock_sub,
        _gate_ok(["10.0.0.1"]),
        # patched_cluster_mgr already stubs the probe to an empty SetupProbe.
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value={"10.0.0.1": mock.Mock(detected=False)}),
        mock.patch("sparkrun.models.distribute.detect_shared_cache", return_value=False),
        mock.patch("sparkrun.cli._setup._ssh._run_ssh_mesh", return_value=True) as mock_mesh,
        mock.patch("sparkrun.cli._setup._ssh._detect_and_update_mgmt_ips"),
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel") as mock_rsp,
        mock.patch("sparkrun.orchestration.sudo.run_with_sudo_fallback") as mock_sudo,
        mock.patch("sparkrun.telemetry.emit.send_event"),
    ):
        mock_sub.return_value = mock.Mock(returncode=0, stdout="CX7_DETECTED=0\n", stderr="")
        mock_rsp.return_value = [RemoteResult("10.0.0.1", 0, "", "")]
        mock_sudo.return_value = ({"10.0.0.1": RemoteResult("10.0.0.1", 0, "OK", "")}, [])

        result = runner.invoke(
            main,
            ["setup", "wizard", "--hosts", "10.0.0.1", "--cluster", "freshlab", "--yes"],
        )

    assert result.exit_code == 0, result.output
    assert "SSH mesh already working" not in result.output
    assert "already in the docker group" not in result.output
    assert "Setup Complete!" in result.output
    mock_mesh.assert_called_once()  # mesh ran
    mock_sudo.assert_called()  # sudo payloads ran


def test_wizard_partial_gap_still_applies(runner, v, patched_cluster_mgr):
    """One host missing the docker group → that phase applies; the rest skip."""
    hosts = ["10.0.0.1", "10.0.0.2"]
    facts = {h: dict(_ALL_GOOD_FACTS) for h in hosts}
    facts["10.0.0.2"]["CHECK_DOCKER_GROUP"] = "0"
    facts["10.0.0.2"]["CHECK_DOCKER_USABLE"] = "0"
    probe = _probed(facts)

    with (
        mock.patch("subprocess.run") as mock_sub,
        _gate_ok(hosts),
        mock.patch("sparkrun.cli._setup._check.probe_host_states", return_value=probe),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value={h: mock.Mock(detected=False) for h in hosts}),
        mock.patch("sparkrun.models.distribute.detect_shared_cache", return_value=False),
        mock.patch("sparkrun.cli._setup._ssh._run_ssh_mesh") as mock_mesh,
        mock.patch("sparkrun.cli._setup._ssh._detect_and_update_mgmt_ips"),
        mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel") as mock_rsp,
        mock.patch("sparkrun.orchestration.sudo.run_with_sudo_fallback") as mock_sudo,
        mock.patch(
            "sparkrun.orchestration.docker_info.detect_docker_drivers",
            return_value={h: DockerDriverInfo(driver="overlay2", snapshotter=False) for h in hosts},
        ),
        mock.patch("sparkrun.telemetry.emit.send_event"),
    ):
        mock_sub.return_value = mock.Mock(returncode=0, stdout="CX7_DETECTED=0\n", stderr="")
        mock_rsp.return_value = [RemoteResult(h, 0, "", "") for h in hosts]
        mock_sudo.return_value = ({h: RemoteResult(h, 0, "OK", "") for h in hosts}, [])

        result = runner.invoke(
            main,
            ["setup", "wizard", "--hosts", ",".join(hosts), "--cluster", "gaplab", "--yes"],
        )

    assert result.exit_code == 0, result.output
    # The satisfied phases skip…
    assert "SSH mesh already working" in result.output
    assert "Sudoers entries already installed" in result.output
    # …and the gapped one applies without prompting in --yes mode.
    assert "already in the docker group" not in result.output
    assert "Docker:     OK (2/2)" in result.output
    mock_mesh.assert_not_called()
    mock_sudo.assert_called_once()
