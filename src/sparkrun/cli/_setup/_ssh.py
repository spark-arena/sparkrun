"""SSH mesh, access bootstrap, and management IP detection helpers for sparkrun setup."""

from __future__ import annotations

from sparkrun.core.application_profile import render_identity_text

from dataclasses import dataclass, field
from pathlib import Path

import click


def _default_ssh_user() -> str:
    """Return the local username, cross-platform.

    ``os.environ["USER"]`` is POSIX-only: Windows sets ``USERNAME`` instead, so
    the previous ``os.environ.get("USER", "root")`` meant every Windows control
    machine opened its first SSH connection as ``root`` — a user that exists on
    a DGX Spark but is never the one setup should be using.  ``getpass.getuser``
    consults ``LOGNAME``/``USER``/``LNAME``/``USERNAME`` and then the password
    database, so it works on both.

    Returns an empty string when the username genuinely can't be determined;
    callers should then prompt without a default rather than invent one.
    """
    import getpass

    try:
        return getpass.getuser() or ""
    except Exception:
        return ""


@dataclass
class SshAccessOutcome:
    """Result of the control→host SSH access gate."""

    #: Hosts we can reach non-interactively with key auth.
    ok_hosts: list[str] = field(default_factory=list)
    #: The (possibly corrected) SSH username to use downstream.
    user: str = ""
    #: True when a key was installed on at least one host during this gate.
    bootstrapped: bool = False
    #: Hosts still unreachable after any bootstrap attempt.
    blocked: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return bool(self.ok_hosts) and not self.blocked


def _ensure_ssh_access(host_list, user, config, dry_run=False, yes=False):
    """Verify — and when possible bootstrap — key-based SSH to *host_list*.

    Every later setup phase assumes passwordless SSH works.  Running this first
    turns "four screens of Permission denied from four separate probes" into a
    single diagnosis and one offer to fix it.

    On authentication failure the user is offered a key-based bootstrap: locate
    or generate a local identity, then install its public key on each failing
    host using password auth (one password prompt per host).  Success is
    confirmed by re-probing, never by trusting the install's exit code.

    Args:
        host_list: Cluster hosts to check.
        user: SSH username to try.
        config: :class:`~sparkrun.core.config.SparkrunConfig` for key/options.
        dry_run: Skip all network access and report success.
        yes: Non-interactive; diagnose and advise, but never prompt.

    Returns:
        An :class:`SshAccessOutcome`.  ``user`` may differ from the argument if
        the operator corrected it at the prompt.
    """
    from sparkrun.api.setup import (
        SshAccessError,
        ensure_local_key,
        install_public_key_interactive,
        probe_ssh_access,
    )

    if not host_list:
        return SshAccessOutcome(ok_hosts=[], user=user)

    if dry_run:
        click.echo("[dry-run] Would verify SSH access to %d host(s) as '%s'." % (len(host_list), user))
        return SshAccessOutcome(ok_hosts=list(host_list), user=user)

    ssh_options = config.ssh_options if config else []
    ssh_key = config.ssh_key if config else None

    def _probe(as_user):
        return probe_ssh_access(list(host_list), as_user, key=ssh_key, options=ssh_options)

    click.echo("Checking SSH access to %d host(s) as '%s'..." % (len(host_list), user))
    probes = _probe(user)
    ok = [p.host for p in probes if p.ok]
    if len(ok) == len(host_list):
        click.echo("  All %d host(s) reachable with key-based SSH." % len(host_list))
        return SshAccessOutcome(ok_hosts=ok, user=user)

    auth_failed = [p for p in probes if p.auth_failed]
    host_key_failed = [p for p in probes if p.host_key_failed]
    unreachable = [p for p in probes if not p.reachable]

    if ok:
        click.echo("  Reachable: %s" % ", ".join(ok))
    if auth_failed:
        click.echo("  Authentication failed: %s" % ", ".join(p.host for p in auth_failed))
    if host_key_failed:
        click.echo("  Host key mismatch: %s" % ", ".join(p.host for p in host_key_failed), err=True)
        click.echo("    The stored host key changed. Resolve manually before continuing:", err=True)
        for p in host_key_failed:
            click.echo("      ssh-keygen -R %s" % p.host, err=True)
    if unreachable:
        click.echo("  Not reachable: %s" % ", ".join(p.host for p in unreachable), err=True)
        for p in unreachable:
            if p.error:
                click.echo("    %s: %s" % (p.host, p.error.splitlines()[0]), err=True)
        click.echo("    Check the address, the network route, and that sshd is running.", err=True)

    if not auth_failed:
        return SshAccessOutcome(ok_hosts=ok, user=user, blocked=[p.host for p in probes if not p.ok])

    click.echo()
    if yes:
        click.echo(
            "SSH key authentication is not set up for '%s' on %d host(s)." % (user, len(auth_failed)),
            err=True,
        )
        click.echo("Re-run without --yes to set it up interactively.", err=True)
        return SshAccessOutcome(ok_hosts=ok, user=user, blocked=[p.host for p in auth_failed])

    # A wrong username is the single most common cause of a clean
    # "Permission denied (publickey,password)" on an untouched cluster, so
    # offer to correct it before spending password prompts on the wrong user.
    if not ok and click.confirm("Try a different SSH username first?", default=False):
        new_user = click.prompt("SSH username", default=user or None)
        if new_user and new_user != user:
            user = new_user
            click.echo("Re-checking SSH access as '%s'..." % user)
            probes = _probe(user)
            ok = [p.host for p in probes if p.ok]
            if len(ok) == len(host_list):
                click.echo("  All %d host(s) reachable with key-based SSH." % len(host_list))
                return SshAccessOutcome(ok_hosts=ok, user=user)
            auth_failed = [p for p in probes if p.auth_failed]
            if not auth_failed:
                return SshAccessOutcome(ok_hosts=ok, user=user, blocked=[p.host for p in probes if not p.ok])

    if not click.confirm(
        "Set up key-based SSH access to %d host(s) now?" % len(auth_failed),
        default=True,
    ):
        return SshAccessOutcome(ok_hosts=ok, user=user, blocked=[p.host for p in auth_failed])

    # Keep sparkrun's own key beside its config (mountable into a containerized
    # daemon, and never mixed in with the user's personal identities).
    key_dir = Path(config.config_path).parent / "ssh" if config is not None else None
    try:
        local_key = ensure_local_key(preferred=ssh_key, key_dir=key_dir)
    except SshAccessError as e:
        click.echo("Could not prepare a local SSH key: %s" % e, err=True)
        return SshAccessOutcome(ok_hosts=ok, user=user, blocked=[p.host for p in auth_failed])

    if local_key.generated:
        click.echo("  Generated a new SSH key: %s" % local_key.path)
        if config is not None:
            try:
                config.set("ssh.key", str(local_key.path))
                config.save()
                click.echo("  Recorded as ssh.key in %s" % config.config_path)
            except Exception as e:
                click.echo("  Warning: could not persist ssh.key: %s" % e, err=True)
    else:
        click.echo("  Using SSH key: %s" % local_key.path)

    click.echo()
    click.echo("Installing the public key on %d host(s)." % len(auth_failed))
    click.echo("You will be prompted for %s's password once per host." % user)
    click.echo()

    attempted = []
    for probe in auth_failed:
        click.echo("  %s:" % probe.host)
        try:
            install_public_key_interactive(
                probe.host,
                user,
                local_key.public_key,
                options=ssh_options,
            )
        except SshAccessError as e:
            click.echo("    %s" % e, err=True)
            continue
        except ValueError as e:
            click.echo("    Refusing to install key: %s" % e, err=True)
            continue
        attempted.append(probe.host)

    # The install's exit code is a hint; a passwordless probe is the proof.
    click.echo()
    click.echo("Verifying key-based access...")
    probes = _probe(user)
    ok = [p.host for p in probes if p.ok]
    blocked = [p.host for p in probes if not p.ok]
    if ok:
        click.echo("  Key-based SSH working on %d/%d host(s)." % (len(ok), len(host_list)))
    if blocked:
        click.echo("  Still failing: %s" % ", ".join(blocked), err=True)
        click.echo("  Install the key manually on those hosts:", err=True)
        click.echo("    %s" % local_key.public_key, err=True)

    return SshAccessOutcome(ok_hosts=ok, user=user, bootstrapped=bool(attempted), blocked=blocked)


def _detect_and_update_mgmt_ips(host_list, cluster_name, cluster_mgr, ssh_kwargs, dry_run=False):
    """Detect management IPs on cluster hosts and update the cluster definition if needed.

    After SSH mesh, the hosts in the cluster definition may be CX7 or other
    non-management IPs.  This function SSHes into each host to discover its
    management IP (default-route interface) and, when any differ from the
    stored addresses, updates the cluster definition to prefer management IPs.

    If a host's management IP matches the local machine, 127.0.0.1 is used.

    Args:
        host_list: Current cluster host list (may be mutated in place).
        cluster_name: Name of the cluster to update (may be None).
        cluster_mgr: ClusterManager instance (may be None).
        ssh_kwargs: SSH connection keyword arguments.
        dry_run: Preview mode.

    Returns:
        The (possibly updated) host list.
    """
    from sparkrun.orchestration.scripts import generate_ip_detect_script
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel
    from sparkrun.utils.net import is_valid_ip, local_ip_for

    if dry_run or not host_list:
        return host_list

    click.echo("Detecting management IPs on cluster hosts...")
    script = generate_ip_detect_script()
    results = run_remote_scripts_parallel(
        host_list,
        script,
        timeout=15,
        quiet=True,
        **ssh_kwargs,
    )

    # Build mapping: original host -> detected mgmt IP
    mgmt_map: dict[str, str] = {}
    for r in results:
        if r.success:
            ip = r.last_line.strip()
            if is_valid_ip(ip):
                mgmt_map[r.host] = ip

    if not mgmt_map:
        click.echo("  Could not detect management IPs (non-fatal).")
        return host_list

    # Determine local machine's IP for 127.0.0.1 substitution
    local_ip = local_ip_for(host_list[0]) if host_list else None

    # Build corrected host list (preserving order, deduplicating)
    new_hosts: list[str] = []
    seen: set[str] = set()
    changes: list[str] = []
    for h in host_list:
        mgmt = mgmt_map.get(h)
        if mgmt and mgmt != h:
            # Host was given as a non-management IP — prefer mgmt
            if local_ip and mgmt == local_ip:
                resolved = "127.0.0.1"
                label = "  %s -> 127.0.0.1 (local, mgmt=%s)" % (h, mgmt)
            else:
                resolved = mgmt
                label = "  %s -> %s" % (h, mgmt)
            if resolved in seen:
                changes.append("  %s -> %s (duplicate, dropped)" % (h, resolved))
                continue
            new_hosts.append(resolved)
            seen.add(resolved)
            changes.append(label)
        elif h == local_ip and mgmt == local_ip:
            # Already the local machine's routable IP — use 127.0.0.1
            resolved = "127.0.0.1"
            if resolved in seen:
                changes.append("  %s -> 127.0.0.1 (duplicate, dropped)" % h)
                continue
            new_hosts.append(resolved)
            seen.add(resolved)
            changes.append("  %s -> 127.0.0.1 (local)" % h)
        else:
            if h in seen:
                changes.append("  %s (duplicate, dropped)" % h)
                continue
            new_hosts.append(h)
            seen.add(h)

    deduped = len(new_hosts) < len(host_list)

    if not changes:
        click.echo("  All hosts are already using management IPs.")
        return host_list

    if deduped and all("duplicate" in c for c in changes):
        click.echo("  Deduplicating cluster hosts:")
    else:
        click.echo("  Updating cluster hosts to management IPs:")
    for c in changes:
        click.echo(c)

    # Update in place so callers see the new list
    host_list[:] = new_hosts

    # Persist to cluster definition
    if cluster_name and cluster_mgr:
        try:
            cluster_mgr.update(name=cluster_name, hosts=new_hosts)
            click.echo("  Cluster '%s' updated." % cluster_name)
        except Exception as e:
            click.echo("  Warning: could not update cluster: %s" % e, err=True)

    return host_list


def _run_ssh_mesh_native(mesh_hosts, user, ssh_key=None, dry_run=False):
    """Mesh SSH keys without a local POSIX shell (Windows control machines).

    Thin console wrapper over :func:`sparkrun.api.setup.mesh_ssh_keys_native`.
    Unlike the bash script it cannot prompt for passwords, so it needs
    control→host key auth to already work — which the wizard's SSH access gate
    (:func:`_ensure_ssh_access`) establishes beforehand.

    Returns:
        True when every host was meshed.
    """
    from sparkrun.api.setup import mesh_ssh_keys_native

    click.echo("Meshing SSH keys (no local shell; using remote execution)...")
    result = mesh_ssh_keys_native(list(mesh_hosts), user, key=ssh_key, dry_run=dry_run)

    if dry_run:
        click.echo("[dry-run] Would mesh SSH keys across %d host(s)." % len(mesh_hosts))
        return True

    for host, err in sorted(result.collect_failures.items()):
        click.echo("  Could not read a key from %s: %s" % (host, err.splitlines()[0] if err else "unknown"), err=True)
    for host, err in sorted(result.install_failures.items()):
        click.echo("  Could not authorize keys on %s: %s" % (host, err.splitlines()[0] if err else "unknown"), err=True)

    if result.public_keys:
        click.echo("  Meshed %d host key(s) across %d host(s)." % (len(result.public_keys), len(mesh_hosts)))
    if not result.ok:
        click.echo(render_identity_text("  Mesh incomplete. Run '{app_command} setup ssh --diagnose' for details."), err=True)
    return result.ok


def _run_ssh_mesh(mesh_hosts, user, cluster_hosts=None, ssh_key=None, discover_ips=True, dry_run=False, control_is_member=False):
    """Run SSH mesh (mesh_ssh_keys.sh) and optionally discover/distribute host keys.

    Shared core logic used by ``setup_ssh`` and the setup wizard.

    Args:
        mesh_hosts: All hosts for the mesh (including extras and self).
        user: SSH username.
        cluster_hosts: Hosts for Phase 2 IP discovery (subset of mesh_hosts).
            Defaults to *mesh_hosts* if ``None``.
        ssh_key: SSH key path (optional).
        discover_ips: Run Phase 2 (discover IPs, distribute host keys).
        dry_run: Preview mode.
        control_is_member: The control machine is a cluster member (e.g.
            ``127.0.0.1`` was resolved to a routable IP).  When True and the
            SSH user differs from the OS user, loopback host keys are included
            in the distribution so that ``ssh <user>@127.0.0.1`` works.

    Returns:
        ``True`` if mesh completed successfully, ``False`` otherwise.
    """
    import os
    import shutil
    import subprocess
    from sparkrun.scripts import get_script_path

    if len(mesh_hosts) < 1:
        click.echo("SSH mesh requires at least 1 host (got 0).", err=True)
        return False

    cluster_hosts = cluster_hosts or list(mesh_hosts)

    # Phase 1: Mesh SSH keys.  ``mesh_ssh_keys.sh`` needs a POSIX shell on the
    # *control* machine; fall back to the native implementation, which does all
    # its shell work on the (always-Linux) cluster hosts.
    #
    # On Windows this applies even when a `bash` *is* on PATH.  Git Bash and WSL
    # both take a POSIX view of the filesystem, so the Windows path to the
    # bundled script is either escape-mangled or simply not addressable:
    #
    #   /bin/bash: C:UsersdrewDesktop...mesh_ssh_keys.sh: No such file or directory
    if os.name == "nt" or shutil.which("bash") is None:
        ok = _run_ssh_mesh_native(mesh_hosts, user, ssh_key=ssh_key, dry_run=dry_run)
        if dry_run:
            if discover_ips:
                click.echo("  Phase 2 (discover IPs + distribute host keys) would run after mesh.")
            return True
        if not ok:
            return False
    else:
        with get_script_path("mesh_ssh_keys.sh") as script_path:
            cmd = ["bash", str(script_path), user] + mesh_hosts

            if dry_run:
                click.echo("[dry-run] Would run SSH mesh:")
                click.echo("  " + " ".join(cmd))
                if discover_ips:
                    click.echo("  Phase 2 (discover IPs + distribute host keys) would run after mesh.")
                return True

            # Run interactively — the script prompts for passwords
            result = subprocess.run(cmd)
            if result.returncode != 0:
                return False

    # Phase 2: Distribute host keys for management IPs and discovered IPs
    if not discover_ips:
        return True

    click.echo()

    ssh_kwargs = {"ssh_user": user}
    if ssh_key:
        ssh_kwargs["ssh_key"] = ssh_key

    from sparkrun.orchestration.networking import distribute_host_keys
    from sparkrun.orchestration.primitives import check_tcp_reachability
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    # Quick SSH connectivity check before Phase 2 — catch cross-user auth
    # failures early with actionable output instead of opaque errors.
    click.echo("Verifying SSH connectivity to cluster hosts...")
    _verify_results = run_remote_scripts_parallel(
        cluster_hosts,
        "true",
        timeout=10,
        quiet=True,
        **ssh_kwargs,
    )
    _failed = [r for r in _verify_results if not r.success]
    if _failed:
        click.echo()
        click.echo("WARNING: SSH authentication failed for %d host(s):" % len(_failed), err=True)
        for r in _failed:
            click.echo("  %s: %s" % (r.host, r.stderr.strip() if r.stderr else "unknown error"), err=True)
        click.echo(err=True)
        click.echo("Common causes when SSH user differs from local user:", err=True)
        click.echo("  1. Home directory permissions: chmod go-w ~%s" % user, err=True)
        click.echo("  2. sshd AuthorizedKeysFile points to a non-default location", err=True)
        click.echo("  3. AllowUsers/AllowGroups in sshd_config restricts the user", err=True)
        click.echo(err=True)
        click.echo(render_identity_text("Run '{app_command} setup ssh --diagnose' for detailed diagnostics."), err=True)
        click.echo("Continuing with Phase 2 (some operations may fail)...", err=True)
        click.echo()
    else:
        click.echo("  All %d host(s) reachable." % len(cluster_hosts))

    # Start with cluster host management IPs — these must be in every
    # node's known_hosts so inter-node SSH/rsync works without prompts.
    all_discovered_ips: list[str] = []
    seen_ips: set[str] = set()
    for h in cluster_hosts:
        if h not in seen_ips:
            all_discovered_ips.append(h)
            seen_ips.add(h)

    # Also include all mesh hosts (extras, control machine) that aren't
    # already in the cluster list.
    for h in mesh_hosts:
        if h not in seen_ips:
            all_discovered_ips.append(h)
            seen_ips.add(h)

    # Discover additional network IPs (CX7, InfiniBand, etc.) — only for
    # multi-host clusters where inter-node discovery is meaningful.
    if len(cluster_hosts) >= 2:
        from sparkrun.orchestration.networking import discover_host_network_ips

        click.echo("Discovering additional network IPs on cluster hosts...")
        discovered = discover_host_network_ips(cluster_hosts, ssh_kwargs=ssh_kwargs)

        if discovered:
            for host, ips in sorted(discovered.items()):
                click.echo("  %s: %s" % (host, ", ".join(ips)))
                for ip in ips:
                    if ip not in seen_ips:
                        all_discovered_ips.append(ip)
                        seen_ips.add(ip)
        else:
            click.echo("  No additional network IPs found.")

    # When the SSH user differs from the OS user and the control machine is
    # a cluster member, cross-user SSH to 127.0.0.1 needs a known_hosts
    # entry.  Include loopback addresses in the keyscan list so that
    # ``ssh <user>@127.0.0.1`` works without host-key prompts.
    import os

    _os_user = os.environ.get("USER")
    if control_is_member and user != _os_user:
        for loopback in ("127.0.0.1", "localhost"):
            if loopback not in seen_ips:
                all_discovered_ips.append(loopback)
                seen_ips.add(loopback)
        click.echo("  Including loopback host keys (cross-user SSH to local node)")

    # Informational reachability check from control machine
    click.echo()
    click.echo("Checking reachability from control machine...")
    reachability = check_tcp_reachability(all_discovered_ips)
    reachable = [ip for ip, ok in reachability.items() if ok]
    unreachable = [ip for ip, ok in reachability.items() if not ok]
    if reachable:
        click.echo("  Reachable: %s" % ", ".join(reachable))
    if unreachable:
        click.echo("  Not reachable from control (normal for IB): %s" % ", ".join(unreachable))

    # Distribute host keys for all IPs (management + discovered)
    click.echo()
    click.echo("Distributing host keys for %d IP(s)..." % len(all_discovered_ips))
    dist = distribute_host_keys(
        all_discovered_ips,
        cluster_hosts,
        ssh_kwargs=ssh_kwargs,
        # Fabric IPs are unreachable from a non-member control machine;
        # scanning them there only costs a timeout.
        local_ips=reachable,
    )
    _report_host_key_distribution(dist, len(all_discovered_ips))

    return True


def _report_host_key_distribution(dist, ip_count: int) -> None:
    """Render a :class:`~sparkrun.orchestration.networking.HostKeyDistribution`."""
    ks_ok = sum(1 for o in dist.hosts if o.success)
    for o in dist.hosts:
        if not o.success:
            click.echo("  Warning: known_hosts refresh failed on %s: %s" % (o.target, o.error), err=True)
    if dist.local is not None and not dist.local.success:
        click.echo("  Warning: known_hosts refresh failed on this machine: %s" % dist.local.error, err=True)

    for machine, targets in sorted(dist.replaced.items()):
        where = "this machine" if machine == "local" else machine
        click.echo(
            "  CHANGED host key(s) replaced in %s's known_hosts: %s" % (where, ", ".join(targets)),
            err=True,
        )
    if dist.replaced:
        click.echo(
            "  A changed host key is expected after re-imaging a host; if none was re-imaged, investigate.",
            err=True,
        )

    # Claim this machine only if something was actually registered here —
    # a scan in which every target went unanswered is not a distribution.
    local_ok = dist.local is not None and dist.local.registered
    click.echo(
        "  Host keys for %d IP(s) distributed to %d/%d host(s)%s."
        % (ip_count, ks_ok, len(dist.hosts), " + this machine" if local_ok else "")
    )
