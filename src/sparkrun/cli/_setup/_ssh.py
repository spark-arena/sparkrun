"""SSH mesh and management IP detection helpers for sparkrun setup."""

from __future__ import annotations

import click


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
    from sparkrun.orchestration.primitives import local_ip_for
    from sparkrun.orchestration.scripts import generate_ip_detect_script
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel
    from sparkrun.utils import is_valid_ip

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
    import subprocess
    from sparkrun.scripts import get_script_path

    if len(mesh_hosts) < 2:
        click.echo("SSH mesh requires at least 2 hosts (got %d)." % len(mesh_hosts), err=True)
        return False

    cluster_hosts = cluster_hosts or list(mesh_hosts)

    # Phase 1: Mesh SSH keys
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
    if not discover_ips or len(cluster_hosts) < 2:
        return True

    click.echo()

    ssh_kwargs = {"ssh_user": user}
    if ssh_key:
        ssh_kwargs["ssh_key"] = ssh_key

    from sparkrun.orchestration.networking import discover_host_network_ips, distribute_host_keys
    from sparkrun.orchestration.primitives import check_tcp_reachability
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    # Quick SSH connectivity check before Phase 2 — catch cross-user auth
    # failures early with actionable output instead of opaque errors.
    click.echo("Verifying SSH connectivity to cluster hosts...")
    _verify_results = run_remote_scripts_parallel(
        cluster_hosts, "true", timeout=10, quiet=True, **ssh_kwargs,
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
        click.echo("Run 'sparkrun setup ssh --diagnose' for detailed diagnostics.", err=True)
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

    # Discover additional network IPs (CX7, InfiniBand, etc.)
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
    ks_results = distribute_host_keys(
        all_discovered_ips,
        cluster_hosts,
        ssh_kwargs=ssh_kwargs,
    )
    ks_ok = sum(1 for r in ks_results if r.success)
    ks_fail = sum(1 for r in ks_results if not r.success)
    if ks_fail:
        click.echo("  Warning: keyscan failed on %d host(s)." % ks_fail, err=True)
    click.echo("  Host keys for %d IP(s) distributed to %d host(s) + local." % (len(all_discovered_ips), ks_ok))

    return True
