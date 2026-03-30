"""sparkrun setup uninstall — reverse setup wizard changes."""

from __future__ import annotations

import logging
import os
import sys

import click

logger = logging.getLogger(__name__)

# Phase teardown order (reverse of install order)
TEARDOWN_PHASES = [
    "earlyoom",
    "sudoers",
    "docker_group",
    "cx7",
    "ssh_mesh",
]

# Phases where default confirmation is N (dangerous)
DANGEROUS_PHASES = {"cx7", "ssh_mesh", "docker_group"}


def _confirm_phase(phase: str, description: str, yes: bool) -> bool:
    """Confirm a teardown phase with the user."""
    default = phase not in DANGEROUS_PHASES
    if yes:
        click.echo("  %s: %s" % (phase, description))
        return True
    return click.confirm("  %s: %s — proceed?" % (phase, description), default=default)


def _check_running_containers(host_list, ssh_kwargs):
    """Check for running sparkrun containers on hosts. Returns list of (host, container_name)."""
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    script = "docker ps --filter 'name=sparkrun_' --format '{{.Names}}' 2>/dev/null || true"
    try:
        results = run_remote_scripts_parallel(
            host_list, script, timeout=15, quiet=True, **ssh_kwargs
        )
    except Exception:
        return []

    found = []
    for r in results:
        if r.success and r.stdout.strip():
            for name in r.stdout.strip().splitlines():
                if name.strip():
                    found.append((r.host, name.strip()))
    return found


def _teardown_earlyoom(host_list, manifest_phase, ssh_kwargs, sudo_fn, dry_run):
    """Teardown earlyoom on hosts."""
    from sparkrun.scripts import read_script

    installed_pkg = manifest_phase.extra.get("installed_package", False) if manifest_phase else False
    remove_package = "true" if installed_pkg else "false"

    script = read_script("earlyoom_uninstall.sh").format(remove_package=remove_package)

    results = {}
    for h in host_list:
        if dry_run:
            click.echo("    [dry-run] %s: would remove earlyoom config%s" % (
                h, " and package" if installed_pkg else ""))
            continue
        r = sudo_fn(h, script)
        results[h] = r
        if r.success:
            click.echo("    [OK]   %s" % h)
        else:
            click.echo("    [FAIL] %s: %s" % (h, r.stderr.strip()[:100]))
    return results


def _teardown_sudoers(host_list, manifest_phase, user, ssh_kwargs, sudo_fn, dry_run):
    """Remove sparkrun sudoers entries."""
    from sparkrun.utils.shell import validate_unix_username

    validate_unix_username(user)

    files = []
    if manifest_phase and manifest_phase.extra.get("files"):
        files = manifest_phase.extra["files"]
    else:
        files = [
            "/etc/sudoers.d/sparkrun-chown-%s" % user,
            "/etc/sudoers.d/sparkrun-dropcaches-%s" % user,
        ]

    script = "#!/bin/bash\nset -euo pipefail\n"
    for f in files:
        script += 'if [ -f "%s" ]; then sudo -n rm -f "%s"; echo "REMOVED: %s"; else echo "SKIPPED: %s"; fi\n' % (f, f, f, f)

    results = {}
    for h in host_list:
        if dry_run:
            click.echo("    [dry-run] %s: would remove %s" % (h, ", ".join(files)))
            continue
        r = sudo_fn(h, script)
        results[h] = r
        if r.success:
            click.echo("    [OK]   %s" % h)
        else:
            click.echo("    [FAIL] %s: %s" % (h, r.stderr.strip()[:100]))
    return results


def _teardown_docker_group(host_list, user, ssh_kwargs, sudo_fn, dry_run):
    """Remove user from docker group."""
    from sparkrun.utils.shell import validate_unix_username

    validate_unix_username(user)

    script = '#!/bin/bash\nset -euo pipefail\ngpasswd -d "%s" docker 2>/dev/null && echo "REMOVED: %s from docker group" || echo "SKIPPED: %s not in docker group"\n' % (user, user, user)

    results = {}
    for h in host_list:
        if dry_run:
            click.echo("    [dry-run] %s: would remove '%s' from docker group" % (h, user))
            continue
        r = sudo_fn(h, script)
        results[h] = r
        if r.success:
            click.echo("    [OK]   %s" % h)
        else:
            click.echo("    [FAIL] %s: %s" % (h, r.stderr.strip()[:100]))
    return results


def _teardown_cx7(host_list, manifest_phase, ssh_kwargs, sudo_fn, dry_run):
    """Remove CX7 netplan configuration."""
    from sparkrun.scripts import read_script

    script = read_script("cx7_unconfigure.sh")

    results = {}
    for h in host_list:
        if dry_run:
            click.echo("    [dry-run] %s: would remove /etc/netplan/40-cx7.yaml" % h)
            continue
        r = sudo_fn(h, script)
        results[h] = r
        if r.success:
            click.echo("    [OK]   %s" % h)
        else:
            click.echo("    [FAIL] %s: %s" % (h, r.stderr.strip()[:100]))

    # Remove CX7 IPs from known_hosts on all hosts
    if manifest_phase and manifest_phase.extra.get("cx7_ips") and not dry_run:
        cx7_ips = manifest_phase.extra["cx7_ips"]
        from sparkrun.orchestration.ssh import run_remote_scripts_parallel

        keygen_cmds = "\n".join('ssh-keygen -R "%s" 2>/dev/null || true' % ip for ip in cx7_ips)
        try:
            run_remote_scripts_parallel(host_list, keygen_cmds, timeout=30, quiet=True, **ssh_kwargs)
        except Exception:
            logger.debug("Failed to clean CX7 IPs from known_hosts", exc_info=True)

    return results


def _teardown_ssh_mesh(host_list, manifest_phase, user, ssh_kwargs, dry_run):
    """Remove SSH authorized_keys entries (does NOT delete keypairs)."""
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    if dry_run:
        for h in host_list:
            click.echo("    [dry-run] %s: would remove mesh authorized_keys entries" % h)
        return {}

    # Step 1: Collect public keys from all mesh hosts
    click.echo("    Collecting public keys from mesh hosts...")
    collect_script = "cat ~/.ssh/id_ed25519.pub 2>/dev/null || cat ~/.ssh/id_rsa.pub 2>/dev/null || echo ''"
    try:
        key_results = run_remote_scripts_parallel(
            host_list, collect_script, timeout=15, quiet=True, **ssh_kwargs
        )
    except Exception as e:
        click.echo("    [FAIL] Could not collect keys: %s" % e)
        return {}

    pubkeys = []
    for r in key_results:
        if r.success and r.stdout.strip():
            pubkeys.append(r.stdout.strip().splitlines()[0].strip())

    if not pubkeys:
        click.echo("    No public keys found to remove.")
        return {}

    # Step 2: Remove matching lines from authorized_keys on each host
    remove_cmds = []
    for key in pubkeys:
        parts = key.split()
        if len(parts) >= 2:
            key_data = parts[1]
            remove_cmds.append("sed -i '\\|%s|d' ~/.ssh/authorized_keys 2>/dev/null || true" % key_data)

    if not remove_cmds:
        click.echo("    No valid keys to remove.")
        return {}

    removal_script = "#!/bin/bash\n" + "\n".join(remove_cmds) + "\necho 'MESH_KEYS_REMOVED=1'"

    click.echo("    Removing %d key(s) from authorized_keys on %d host(s)..." % (len(pubkeys), len(host_list)))
    try:
        results = run_remote_scripts_parallel(
            host_list, removal_script, timeout=30, quiet=True, **ssh_kwargs
        )
        result_map = {}
        for r in results:
            result_map[r.host] = r
            if r.success:
                click.echo("    [OK]   %s" % r.host)
            else:
                click.echo("    [FAIL] %s: %s" % (r.host, r.stderr.strip()[:100]))
        return result_map
    except Exception as e:
        click.echo("    [FAIL] Key removal failed: %s" % e)
        return {}


@click.command("uninstall", hidden=True)
@click.argument("cluster_name", required=False, default=None)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmations (full teardown)")
@click.option("--dry-run", "-n", is_flag=True, help="Preview without executing")
@click.option("--keep-cluster", is_flag=True, help="Keep local cluster definition")
@click.option("--phase", "phases", multiple=True, help="Only uninstall specific phase(s)")
@click.option("--force", is_flag=True, help="Proceed even if running containers detected")
@click.pass_context
def setup_uninstall(ctx, cluster_name, yes, dry_run, keep_cluster, phases, force):
    """Reverse setup wizard changes for a cluster.

    Undoes setup phases in reverse order: earlyoom, sudoers, docker group,
    CX7 networking, SSH mesh. Each phase can be confirmed individually.

    Dangerous phases (CX7, SSH mesh, docker group) default to N unless --yes.

    \b
    Examples:
      sparkrun setup uninstall
      sparkrun setup uninstall mylab
      sparkrun setup uninstall --dry-run
      sparkrun setup uninstall --phase earlyoom --phase sudoers
      sparkrun setup uninstall --yes --keep-cluster
    """
    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.config import SparkrunConfig, get_config_root
    from sparkrun.core.setup_manifest import ManifestManager
    from sparkrun.orchestration.primitives import build_ssh_kwargs
    from sparkrun.orchestration.sudo import run_sudo_script_on_host

    config = SparkrunConfig()
    config_root = get_config_root()
    cluster_mgr = ClusterManager(config_root)
    manifest_mgr = ManifestManager(cluster_mgr.clusters_dir)

    # ── Resolve cluster ──────────────────────────────────────────
    if not cluster_name:
        cluster_name = cluster_mgr.get_default()
    if not cluster_name:
        click.echo("Error: No cluster specified and no default cluster set.", err=True)
        click.echo("Usage: sparkrun setup uninstall [CLUSTER_NAME]", err=True)
        sys.exit(1)

    try:
        cluster_def = cluster_mgr.get(cluster_name)
    except Exception:
        click.echo("Error: Cluster '%s' not found." % cluster_name, err=True)
        sys.exit(1)

    host_list = list(cluster_def.hosts)
    user = cluster_def.user or os.environ.get("USER", "root")

    click.echo()
    click.echo("Uninstall setup for cluster '%s'" % cluster_name)
    click.echo("=" * 48)
    click.echo("  Hosts: %s" % ", ".join(host_list))
    click.echo("  User:  %s" % user)
    click.echo()

    # ── Load manifest ────────────────────────────────────────────
    manifest = manifest_mgr.load(cluster_name)
    if manifest is None:
        click.echo("Warning: No setup manifest found for '%s'." % cluster_name)
        click.echo("Will infer from cluster definition. Dangerous phases default to N.")
        click.echo()

    # ── Build SSH kwargs ─────────────────────────────────────────
    ssh_kwargs = build_ssh_kwargs(config)
    if user:
        ssh_kwargs["ssh_user"] = user

    # ── Container check ──────────────────────────────────────────
    if not dry_run:
        running = _check_running_containers(host_list, ssh_kwargs)
        if running:
            click.echo("Warning: Running sparkrun containers detected:")
            for host, name in running:
                click.echo("  %s: %s" % (host, name))
            click.echo()
            if not force:
                if not yes and not click.confirm(
                    "Continue with uninstall? (use --force to suppress)", default=False
                ):
                    click.echo("Aborted.")
                    return
                elif yes:
                    click.echo("Proceeding despite running containers (--yes).")
            else:
                click.echo("Proceeding (--force).")
            click.echo()

    # ── Sudo helper ──────────────────────────────────────────────
    sudo_password = None

    def _ensure_sudo_password():
        nonlocal sudo_password
        if sudo_password is not None:
            return sudo_password
        if dry_run:
            return None

        from sparkrun.orchestration.ssh import run_remote_scripts_parallel

        try:
            test_results = run_remote_scripts_parallel(
                host_list, "sudo -n true", quiet=True, timeout=10, **ssh_kwargs
            )
            if all(r.success for r in test_results):
                return None
        except Exception:
            pass

        sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)
        return sudo_password

    def _sudo_on_host(host, script, timeout=300):
        pw = _ensure_sudo_password()
        return run_sudo_script_on_host(
            host, script, pw or "", ssh_kwargs=ssh_kwargs, timeout=timeout, dry_run=dry_run
        )

    # ── Filter phases ────────────────────────────────────────────
    phase_filter = set(phases) if phases else None
    active_phases = []
    for phase in TEARDOWN_PHASES:
        if phase_filter and phase not in phase_filter:
            continue
        # Only include phases that were actually applied (or all if no manifest)
        if manifest and phase in manifest.phases and manifest.phases[phase].applied:
            active_phases.append(phase)
        elif manifest and phase not in manifest.phases:
            continue  # Phase wasn't recorded, skip
        elif manifest is None:
            active_phases.append(phase)  # No manifest, include all

    if not active_phases and not phase_filter:
        click.echo("No setup phases to uninstall.")
        if not keep_cluster:
            click.echo()
            if yes or click.confirm("Delete cluster definition '%s'?" % cluster_name, default=True):
                cluster_mgr.delete(cluster_name)
                manifest_mgr.delete(cluster_name)
                click.echo("Cluster '%s' deleted." % cluster_name)
        return

    # ── Phase-by-phase teardown ──────────────────────────────────
    summary = {}

    for phase in active_phases:
        manifest_phase = manifest.phases.get(phase) if manifest else None
        phase_hosts = manifest_phase.hosts if manifest_phase else host_list

        click.echo("Phase: %s" % phase)
        click.echo("-" * 30)

        if phase == "earlyoom":
            desc = "Stop earlyoom and remove sparkrun config on %d host(s)" % len(phase_hosts)
            if _confirm_phase(phase, desc, yes):
                _teardown_earlyoom(phase_hosts, manifest_phase, ssh_kwargs, _sudo_on_host, dry_run)
                summary[phase] = "removed"
            else:
                summary[phase] = "skipped"

        elif phase == "sudoers":
            desc = "Remove sparkrun sudoers entries on %d host(s)" % len(phase_hosts)
            if _confirm_phase(phase, desc, yes):
                _teardown_sudoers(phase_hosts, manifest_phase, user, ssh_kwargs, _sudo_on_host, dry_run)
                summary[phase] = "removed"
            else:
                summary[phase] = "skipped"

        elif phase == "docker_group":
            desc = "Remove '%s' from docker group on %d host(s) (prevents Docker without sudo)" % (user, len(phase_hosts))
            if _confirm_phase(phase, desc, yes):
                _teardown_docker_group(phase_hosts, user, ssh_kwargs, _sudo_on_host, dry_run)
                summary[phase] = "removed"
            else:
                summary[phase] = "skipped"

        elif phase == "cx7":
            desc = "Remove CX7 netplan config on %d host(s) (disables high-speed networking)" % len(phase_hosts)
            if _confirm_phase(phase, desc, yes):
                _teardown_cx7(phase_hosts, manifest_phase, ssh_kwargs, _sudo_on_host, dry_run)
                summary[phase] = "removed"
            else:
                summary[phase] = "skipped"

        elif phase == "ssh_mesh":
            desc = "Remove SSH mesh authorized_keys on %d host(s) (may break inter-node SSH)" % len(phase_hosts)
            if _confirm_phase(phase, desc, yes):
                _teardown_ssh_mesh(phase_hosts, manifest_phase, user, ssh_kwargs, dry_run)
                summary[phase] = "removed"
            else:
                summary[phase] = "skipped"

        click.echo()

    # ── Cluster cleanup ──────────────────────────────────────────
    if not keep_cluster:
        desc = "Delete cluster '%s' and manifest" % cluster_name
        if yes or click.confirm("  %s — proceed?" % desc, default=True):
            if not dry_run:
                cluster_mgr.delete(cluster_name)
                manifest_mgr.delete(cluster_name)
            summary["cluster"] = "deleted"
            click.echo("  Cluster '%s' %s." % (cluster_name, "would be deleted" if dry_run else "deleted"))
        else:
            summary["cluster"] = "kept"
    else:
        summary["cluster"] = "kept (--keep-cluster)"

    # ── Summary ──────────────────────────────────────────────────
    click.echo()
    click.echo("Uninstall Summary")
    click.echo("=" * 48)
    for key, val in summary.items():
        click.echo("  %-14s %s" % (key + ":", val))
    click.echo()
