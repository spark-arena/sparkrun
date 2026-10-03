"""CLI-level sudo helpers for sparkrun setup commands."""

from __future__ import annotations

from typing import Any

import click

from .._common import _get_cluster_manager


def _record_setup_phase(cluster_name, user, host_list, phase, **extra):
    """Record a setup phase in the manifest (best-effort, never raises)."""
    try:
        resolved = cluster_name
        if not resolved:
            mgr = _get_cluster_manager()
            resolved = mgr.get_default()
        if not resolved:
            return
        from sparkrun.core.setup_manifest import ManifestManager

        mgr = _get_cluster_manager()
        manifest_mgr = ManifestManager(mgr.clusters_dir)
        manifest_mgr.record_phase(resolved, user, host_list, phase, **extra)
    except Exception:
        import logging

        logging.getLogger(__name__).debug("Failed to record manifest phase '%s'", phase, exc_info=True)


def ensure_sudo_password(
    host_list,
    user,
    ssh_kwargs,
    sudo_ssh_kwargs=None,
    dry_run=False,
    allow_indirect=False,
    default_user=None,
):
    """Test NOPASSWD, prompt for password if needed, optionally fall back to indirect sudo.

    Args:
        host_list: Hosts to test sudo on.
        user: Primary SSH/sudo user.
        ssh_kwargs: SSH kwargs for the cluster user (used for indirect sudo).
        sudo_ssh_kwargs: SSH kwargs for sudo operations (may differ from ssh_kwargs
            if sudo user differs).  Defaults to *ssh_kwargs* if not provided.
        dry_run: If True, skip testing and return (None, None).
        allow_indirect: If True and primary user's sudo fails, prompt for
            an alternate user with sudo access (indirect sudo path).
        default_user: Default alternate user for indirect sudo prompt.

    Returns:
        Tuple of ``(sudo_password, indirect_sudo_user)`` where
        *indirect_sudo_user* is ``None`` unless the primary user's sudo
        failed and *allow_indirect* was True.
    """
    if dry_run:
        return None, None

    if sudo_ssh_kwargs is None:
        sudo_ssh_kwargs = ssh_kwargs

    # Try NOPASSWD on all hosts using the current sudo user
    from sparkrun.orchestration.primitives import run_local_script, should_run_locally
    from sparkrun.orchestration.ssh import RemoteResult, run_remote_scripts_parallel

    sudo_user = sudo_ssh_kwargs.get("ssh_user", user)
    local_hosts = [h for h in host_list if should_run_locally(h, sudo_user)]
    remote_hosts = [h for h in host_list if not should_run_locally(h, sudo_user)]
    try:
        test_results = []
        for h in local_hosts:
            lr = run_local_script("sudo -n true", dry_run=False)
            test_results.append(RemoteResult(host=h, returncode=lr.returncode, stdout=lr.stdout, stderr=lr.stderr))
        if remote_hosts:
            test_results.extend(
                run_remote_scripts_parallel(
                    remote_hosts,
                    "sudo -n true",
                    quiet=True,
                    timeout=10,
                    **sudo_ssh_kwargs,
                )
            )
        if all(r.success for r in test_results):
            return None, None
    except Exception:
        pass

    # Prompt for sudo password
    sudo_password = click.prompt("[sudo] password for %s" % sudo_user, hide_input=True)

    # Verify sudo works with this password on at least one host
    from sparkrun.orchestration.sudo import run_sudo_script_on_host

    test_host = host_list[0]
    if should_run_locally(test_host, sudo_user):
        import subprocess

        proc = subprocess.run(
            ["sudo", "-S", "true"],
            input=sudo_password + "\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
        test_r = RemoteResult(host=test_host, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    else:
        test_r = run_sudo_script_on_host(
            test_host,
            "true",
            sudo_password,
            ssh_kwargs=sudo_ssh_kwargs,
            timeout=10,
        )
    if not test_r.success and allow_indirect:
        # Sudo failed for cluster user — offer alternate user
        alt_default = default_user if sudo_user != default_user else ""
        click.echo("  Sudo failed for '%s'. Specify a user with sudo access." % sudo_user)
        alt_user = click.prompt("Sudo user", default=alt_default)
        try:
            from sparkrun.utils.shell import validate_unix_username

            validate_unix_username(alt_user)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="'Sudo user'") from exc
        sudo_password = click.prompt("[sudo] password for %s" % alt_user, hide_input=True)
        return sudo_password, alt_user

    return sudo_password, None


def install_sudoers_entry(
    host_list,
    sudoers_script,
    sudo_password,
    user,
    ssh_kwargs=None,
):
    """Install a sudoers entry on every host, retrying failures per-host.

    A cluster's hosts do not necessarily share a sudo password, so the
    password collected at the cluster-wide prompt can fail on a subset
    (``sudo -S`` "Sorry, try again"). Those hosts are retried one at a time
    with a per-host prompt — the same pattern the main action's password
    fallback uses — so one outlier password no longer leaves a host
    permanently unconfigured across re-runs.

    Args:
        host_list: Hosts to install on.
        sudoers_script: Rendered sudoers-install script.
        sudo_password: Password from the cluster-wide prompt, or ``None``
            where every host already has NOPASSWD sudo.
        user: Sudo user named in the per-host prompt.
        ssh_kwargs: SSH connection parameters.

    Returns:
        Tuple of ``(ok_count, fail_count, failed_hosts)`` where
        *failed_hosts* are the hosts still unconfigured after the retry.
    """
    from sparkrun.orchestration.sudo import run_sudo_script_on_host

    results: dict[str, Any] = {}
    for h in host_list:
        results[h] = run_sudo_script_on_host(
            h,
            sudoers_script,
            sudo_password,
            ssh_kwargs=ssh_kwargs,
            timeout=300,
            dry_run=False,
        )

    failed_hosts = [h for h in host_list if not results[h].success]
    if failed_hosts:
        click.echo()
        click.echo("Sudo authentication failed on %d host(s). Retrying individually..." % len(failed_hosts))
        for h in failed_hosts:
            per_host_pw = click.prompt("[sudo] password for %s @ %s" % (user, h), hide_input=True)
            results[h] = run_sudo_script_on_host(
                h,
                sudoers_script,
                per_host_pw,
                ssh_kwargs=ssh_kwargs,
                timeout=300,
                dry_run=False,
            )

    ok = 0
    fail = 0
    still_failed: list[str] = []
    for h in host_list:
        r = results[h]
        if r.success:
            ok += 1
            click.echo("  [OK]   %s: %s" % (h, r.stdout.strip()))
        else:
            fail += 1
            still_failed.append(h)
            click.echo("  [FAIL] %s: %s" % (h, r.stderr.strip()[:200]), err=True)
    return ok, fail, still_failed
