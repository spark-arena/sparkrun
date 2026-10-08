"""CLI-level sudo helpers for sparkrun setup commands."""

from __future__ import annotations

import logging

import click

from .._common import _get_cluster_manager

logger = logging.getLogger(__name__)


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
        logger.debug("Failed to record manifest phase '%s'", phase, exc_info=True)


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
        # Verify only where sudo asked for a password: an unreachable host says
        # nothing about the password, and a NOPASSWD host must never be sent one.
        # ssh's own failures exit 255 and a timed-out probe -1: neither host
        # said anything about a password.
        needs_password = [r.host for r in test_results if not r.success and r.returncode not in (255, -1)]
    except Exception:
        logger.debug("Passwordless sudo probe failed; collecting the password unverified", exc_info=True)
        return click.prompt("[sudo] password for %s" % sudo_user, hide_input=True), None
    if not needs_password:
        return None, None  # the rest are unreachable: no password helps there

    # Prompt, then verify on every host that needs a password. Hosts need
    # not share one: if any accepts it, keep it — the rest are asked for their
    # own when an action reaches them (SudoPasswords). Only a password every
    # host rejects is re-asked here, like sudo's own retries.
    from concurrent.futures import ThreadPoolExecutor

    from sparkrun.orchestration.sudo import is_sudo_auth_failure, run_sudo_script_on_host

    def verify(host):
        return run_sudo_script_on_host(host, "true", sudo_password, ssh_kwargs=sudo_ssh_kwargs, timeout=10)

    for attempt in range(3):
        if attempt:
            click.echo("Sorry, try again.")
        sudo_password = click.prompt("[sudo] password for %s" % sudo_user, hide_input=True)
        with ThreadPoolExecutor(max_workers=min(len(needs_password), 16)) as pool:
            verdicts = list(pool.map(verify, needs_password))
        if any(r.success for r in verdicts):
            return sudo_password, None
        # Keep asking where the password itself was rejected; hosts that failed
        # otherwise (dropped connection, not a sudoer) do not decide for them.
        needs_password = [r.host for r in verdicts if is_sudo_auth_failure(r)]
        if not needs_password:
            break  # not a wrong password: this user cannot sudo there

    if allow_indirect:
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


def interactive_sudo_passwords(user):
    """Terminal prompts for :class:`~sparkrun.orchestration.sudo.SudoPasswords`.

    The shared password is asked once, the first time a host needs one; a
    host that rejects it gets its own prompt, and that password is reused for
    the rest of the command.
    """
    from sparkrun.orchestration.sudo import SudoPasswords

    def shared():
        return click.prompt("[sudo] password for %s" % user, hide_input=True)

    return SudoPasswords(prompt_shared=shared, prompt_host=host_password_prompt(user))


def host_password_prompt(user):
    """``host -> password`` prompt for a host whose sudo rejected the shared password."""

    def prompt(host):
        click.echo("Sudo rejected the password on %s." % host, err=True)
        return click.prompt("[sudo] password for %s @ %s" % (user, host), hide_input=True)

    return prompt


def run_sudo_action(host_list, script, fallback_script, ssh_kwargs, passwords, *, dry_run=False, progress=None):
    """Run a root action: ``sudo -n`` everywhere, then a password per failing host.

    *script* runs non-interactively on every host in parallel; each host where
    that fails runs *fallback_script* (as root) with the password *passwords*
    holds for it, re-asking a host whose sudo rejects it.  *progress(host,
    result)* reports each ``sudo -n`` success, then each password run before
    (``result=None``) and after it.

    Returns:
        ``{host: RemoteResult}`` for every host that ran.
    """
    from sparkrun.orchestration.sudo import run_sudo_script_on_host, run_with_sudo_fallback

    result_map, failed = run_with_sudo_fallback(host_list, script, fallback_script, ssh_kwargs, dry_run=dry_run)
    if dry_run:
        return result_map
    if failed:
        # A host whose sudo needs no password failed for another reason; its
        # retry runs as root via `sudo -n`, never with a password on stdin.
        _probe, needs_password = run_with_sudo_fallback(failed, "sudo -n true", "true", ssh_kwargs)
        passwords.nopasswd.update(h for h in failed if h not in needs_password)
    if progress:
        for host in host_list:
            if host not in failed and host in result_map:
                progress(host, result_map[host])
    for host in failed:
        if progress:
            progress(host, None)
        result_map[host] = passwords.run(
            host,
            lambda password, host=host: run_sudo_script_on_host(host, fallback_script, password, ssh_kwargs=ssh_kwargs, timeout=300),
        )
        if progress:
            progress(host, result_map[host])
    return result_map


def save_sudoers_entry(label, host_list, user, ssh_kwargs, passwords, *, config, cluster_name, explicit_hosts, dry_run, cache_dir=""):
    """``--save-sudo``: install one scoped sudoers entry where it is not already in effect.

    Uses the ``sudoers`` setup step's installer and readiness probe, so the
    entry, the skip decision and ``setup check`` agree.  Only *label*'s entry
    is installed — the wizard's ``sudoers`` step is what installs them all.
    It is also held to that step's boundary: a target whose hardware plan
    omits ``sudoers`` (or where the step is disabled) is refused before any
    prompt, as ``setup earlyoom`` / ``setup cx7`` refuse theirs.
    """
    from sparkrun.core.setup_actions import SUDOERS_ENTRIES, SetupActionContext, install_sudoers_entries, sudoers_entry_path
    from sparkrun.core.setup_models import OK
    from sparkrun.core.setup_probe import probe_sudoers_entries

    from sparkrun.utils.shell import validate_sudoers_path, validate_unix_username

    # Both are interpolated into a sudoers rule; refuse before any prompt.
    try:
        validate_unix_username(user)
        if cache_dir:
            validate_sudoers_path(cache_dir)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    from ._step_runner import require_setup_step_targets

    require_setup_step_targets(
        "sudoers", host_list, ssh_kwargs, config, cluster_name=cluster_name, explicit_hosts=explicit_hosts, dry_run=dry_run
    )
    path = sudoers_entry_path(label, user)
    if dry_run:
        click.echo("  [dry-run] Would install sudoers entry on %d host(s):" % len(host_list))
        for h in host_list:
            click.echo("    %s: %s" % (h, path))
        click.echo()
        return
    # Setup commands connect as *user*, so the probe answers for that user.
    fact = SUDOERS_ENTRIES[label][1]
    present = probe_sudoers_entries(host_list, ssh_kwargs=ssh_kwargs, cache_dir=cache_dir or None)
    passwords.nopasswd.update(h for h in host_list if present.get(h, {}).get("CHECK_SUDO_NOPASSWD") == "1")
    targets = [h for h in host_list if present.get(h, {}).get(fact) != "1"]
    if not targets:
        click.echo("Sudoers entry %s already in effect on every host." % path)
        click.echo()
        return
    click.echo("Installing sudoers entry %s on %d host(s)..." % (path, len(targets)))
    action = SetupActionContext(user, ssh_kwargs, passwords=passwords)
    installed = []
    for h in targets:
        result = install_sudoers_entries(h, action, (label,), cache_dir=cache_dir)
        if result.status == OK:
            installed.append(h)
            click.echo("  [OK]   %s: %s" % (h, result.detail))
        else:
            click.echo("  [FAIL] %s: %s" % (h, result.detail[:200]), err=True)
    click.echo("Sudoers install: %d OK, %d failed." % (len(installed), len(targets) - len(installed)))
    if installed:
        _record_setup_phase(cluster_name, user, installed, "sudoers", files=[path])
    click.echo()
