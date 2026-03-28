"""sparkrun setup group and subcommands."""

from __future__ import annotations

import sys

import click

from ._common import (
    _detect_shell,
    _get_cluster_manager,
    _require_uv,
    resolve_cluster_config,
    _resolve_setup_context,
    _shell_rc_file,
    dry_run_option,
    host_options,
)


@click.group(invoke_without_command=True)
@click.pass_context
def setup(ctx):
    """Setup and configuration commands."""
    if ctx.invoked_subcommand is not None:
        return
    # Smart routing: auto-launch wizard when no default cluster is set
    mgr = _get_cluster_manager()
    if mgr.get_default() is None:
        from ._wizard import setup_wizard

        ctx.invoke(setup_wizard)
    else:
        click.echo(ctx.get_help())


@setup.command("completion", hidden=True)
@click.option("--shell", type=click.Choice(["bash", "zsh", "fish"]), default=None, help="Shell type (auto-detected if not specified)")
@click.pass_context
def setup_completion(ctx, shell):
    """Install shell tab-completion for sparkrun.

    Detects your current shell and appends the completion setup to
    your shell config file (~/.bashrc, ~/.zshrc, or ~/.config/fish/...).

    Examples:

      sparkrun setup completion

      sparkrun setup completion --shell bash
    """
    if not shell:
        shell, rc_file = _detect_shell()
    else:
        rc_file = _shell_rc_file(shell)

    completion_var = "_SPARKRUN_COMPLETE"

    if shell == "bash":
        snippet = 'eval "$(%s=bash_source sparkrun)"' % completion_var
    elif shell == "zsh":
        snippet = 'eval "$(%s=zsh_source sparkrun)"' % completion_var
    elif shell == "fish":
        snippet = "%s=fish_source sparkrun | source" % completion_var

    # Check if already installed
    if rc_file.exists():
        contents = rc_file.read_text()
        if completion_var in contents:
            click.echo("Completion already configured in %s" % rc_file)
            return

    # Ensure parent directory exists (for fish)
    rc_file.parent.mkdir(parents=True, exist_ok=True)

    with open(rc_file, "a") as f:
        f.write("\n# sparkrun tab-completion\n")
        f.write(snippet + "\n")

    click.echo("Completion installed for %s in %s" % (shell, rc_file))
    click.echo("Restart your shell or run: source %s" % rc_file)


@setup.command("install")
@click.option("--shell", type=click.Choice(["bash", "zsh", "fish"]), default=None, help="Shell type (auto-detected if not specified)")
@click.option("--no-update-registries", is_flag=True, help="Skip updating recipe registries after installation")
@click.pass_context
def setup_install(ctx, shell, no_update_registries):
    """Install sparkrun and tab-completion.

    Requires uv (https://docs.astral.sh/uv/).  Typical usage:

    \b
      uvx sparkrun setup install

    This installs sparkrun as a uv tool (real binary on PATH), cleans up
    any old aliases/functions from previous installs, configures
    tab-completion, and updates recipe registries.
    """
    import subprocess

    if not shell:
        shell, rc_file = _detect_shell()
    else:
        rc_file = _shell_rc_file(shell)

    # Step 1: Install sparkrun via uv tool
    uv = _require_uv()

    click.echo("Installing sparkrun via uv tool install...")
    result = subprocess.run(
        [uv, "tool", "install", "sparkrun", "--force"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo("Error installing sparkrun: %s" % result.stderr.strip(), err=True)
        sys.exit(1)
    click.echo("sparkrun installed on PATH")

    # Step 2: Clean up old aliases/functions from previous installs
    if rc_file.exists():
        old_markers = [
            "alias sparkrun=",
            "alias sparkrun ",
            "function sparkrun",
            "sparkrun()",
        ]
        contents = rc_file.read_text()
        lines = contents.splitlines(keepends=True)
        cleaned = [ln for ln in lines if not any(m in ln for m in old_markers)]
        if len(cleaned) != len(lines):
            rc_file.write_text("".join(cleaned))
            click.echo("Removed old sparkrun alias/function from %s" % rc_file)

    # Step 3: Set up tab-completion
    ctx.invoke(setup_completion, shell=shell)

    # Step 4: Update recipe registries from the newly installed binary
    if not no_update_registries:
        click.echo()
        click.echo("Updating recipe registries...")
        reg_result = subprocess.run(
            ["sparkrun", "registry", "update"],
            capture_output=False,
        )
        if reg_result.returncode != 0:
            click.echo("Warning: registry update failed (non-fatal).", err=True)

    click.echo()
    click.echo("Restart your shell or run: source %s" % rc_file)


@setup.command("update")
@click.option("--no-update-registries", is_flag=True, help="Skip updating recipe registries after upgrading sparkrun")
@click.pass_context
def setup_update(ctx, no_update_registries):
    """Update sparkrun and recipe registries.

    Runs ``uv tool upgrade sparkrun`` to fetch the latest release, then
    updates all enabled recipe registries from git.  Use
    ``--no-update-registries`` to skip the registry sync step.

    Only works when sparkrun was installed via ``uv tool install``.
    """
    import subprocess

    from sparkrun import __version__ as old_version

    uv = _require_uv()

    # Guard: only upgrade if sparkrun was installed via uv tool
    check = subprocess.run(
        [uv, "tool", "list"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0 or "sparkrun" not in check.stdout:
        click.echo(
            "Error: sparkrun was not installed via 'uv tool install'.\n"
            "Cannot safely upgrade — manage updates through your package manager instead.",
            err=True,
        )
        sys.exit(1)

    click.echo("Checking for updates (current: %s)..." % old_version)
    result = subprocess.run(
        [uv, "tool", "upgrade", "sparkrun"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo("Error updating sparkrun: %s" % result.stderr.strip(), err=True)
        sys.exit(1)

    # The running process still has the old module cached, and reload
    # won't help because uv tool installs into a separate virtualenv.
    # Ask the newly installed binary instead.
    ver_result = subprocess.run(
        ["sparkrun", "--version"],
        capture_output=True,
        text=True,
    )
    if ver_result.returncode == 0:
        new_version = ver_result.stdout.strip().rsplit(None, 1)[-1]
        if new_version == old_version:
            click.echo("sparkrun %s is already the latest version." % old_version)
        else:
            click.echo("sparkrun updated: %s -> %s" % (old_version, new_version))
    else:
        click.echo("sparkrun updated (could not determine new version)")

    # Update recipe registries from the newly installed binary — the
    # running process still has old code, so we must shell out.
    if not no_update_registries:
        click.echo()
        click.echo("Updating recipe registries...")
        reg_result = subprocess.run(
            ["sparkrun", "registry", "update"],
            capture_output=False,
        )
        if reg_result.returncode != 0:
            click.echo("Warning: registry update failed (non-fatal).", err=True)


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

    # Phase 2: Discover additional IPs and distribute host keys
    if not discover_ips or len(cluster_hosts) < 2:
        return True

    click.echo()
    click.echo("Discovering additional network IPs on cluster hosts...")

    ssh_kwargs = {"ssh_user": user}
    if ssh_key:
        ssh_kwargs["ssh_key"] = ssh_key

    from sparkrun.orchestration.networking import discover_host_network_ips, distribute_host_keys
    from sparkrun.orchestration.primitives import check_tcp_reachability

    discovered = discover_host_network_ips(cluster_hosts, ssh_kwargs=ssh_kwargs)

    if not discovered:
        click.echo("  No additional network IPs found.")
        return True

    # Collect all unique discovered IPs
    all_discovered_ips: list[str] = []
    seen_ips: set[str] = set()
    for host, ips in sorted(discovered.items()):
        click.echo("  %s: %s" % (host, ", ".join(ips)))
        for ip in ips:
            if ip not in seen_ips:
                all_discovered_ips.append(ip)
                seen_ips.add(ip)

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

    # Distribute host keys for ALL discovered IPs
    click.echo()
    click.echo("Distributing host keys for %d discovered IP(s)..." % len(all_discovered_ips))
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


@setup.command("ssh")
@host_options
@click.option("--extra-hosts", default=None, help="Additional comma-separated hosts to include (e.g. control machine)")
@click.option("--include-self/--no-include-self", default=True, show_default=True, help="Include this machine's hostname in the mesh")
@click.option("--user", "-u", default=None, help="SSH username (default: current user)")
@click.option(
    "--discover-ips/--no-discover-ips", default=True, show_default=True, help="After meshing, discover IB/CX7 IPs and distribute host keys"
)
@dry_run_option
@click.pass_context
def setup_ssh(ctx, hosts, hosts_file, cluster_name, extra_hosts, include_self, user, discover_ips, dry_run):
    """Set up passwordless SSH mesh across cluster hosts.

    Ensures every host can SSH to every other host without password prompts.
    Creates ed25519 keys if missing and distributes public keys.

    After the mesh is established, sparkrun automatically discovers
    additional network IPs (InfiniBand, CX7) on cluster hosts and
    distributes their host keys so inter-node SSH works over all
    networks. Use --no-discover-ips to skip this phase.

    By default, the machine running sparkrun is included in the mesh
    (--include-self). Use --no-include-self to exclude it.

    You will be prompted for passwords on first connection to each host.

    Examples:

      sparkrun setup ssh --hosts 192.168.11.13,192.168.11.14

      sparkrun setup ssh --cluster mylab --user ubuntu

      sparkrun setup ssh --cluster mylab --extra-hosts 10.0.0.1

      sparkrun setup ssh --hosts 10.0.0.1,10.0.0.2 --no-discover-ips
    """
    import os

    from sparkrun.core.hosts import resolve_hosts
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.primitives import local_ip_for

    config = SparkrunConfig()

    # Resolve hosts and look up cluster user if applicable
    cluster_mgr = _get_cluster_manager()
    host_list = resolve_hosts(
        hosts=hosts,
        hosts_file=hosts_file,
        cluster_name=cluster_name,
        cluster_manager=cluster_mgr,
        config_default_hosts=config.default_hosts,
    )

    # Determine the cluster's configured user (if hosts came from a cluster)
    cluster_user = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr).user

    # Resolve 127.0.0.1 to a routable IP so other nodes can SSH back
    _resolved_loopback = False
    for i, h in enumerate(host_list):
        if h == "127.0.0.1":
            # Find first non-loopback host to determine routing
            other = next((x for x in host_list if x != "127.0.0.1"), None)
            if other:
                resolved = local_ip_for(other)
                if resolved and resolved != "127.0.0.1":
                    click.echo("Resolved 127.0.0.1 -> %s (routable IP)" % resolved)
                    host_list[i] = resolved
                    _resolved_loopback = True
                else:
                    click.echo("Warning: Could not resolve 127.0.0.1 to a routable IP, dropping it.", err=True)
                    host_list[i] = ""  # mark for removal
            else:
                click.echo("Warning: All hosts are 127.0.0.1, cannot resolve.", err=True)
                host_list[i] = ""
    host_list = [h for h in host_list if h]

    # Resolve effective user early (needed for self-inclusion decision)
    local_user = os.environ.get("USER", "root")
    if user is None:
        user = cluster_user or config.ssh_user or local_user

    # Track original cluster hosts before extras/self are appended
    cluster_hosts = list(host_list)
    seen = set(host_list)
    added: list[str] = []
    if extra_hosts:
        for h in extra_hosts.split(","):
            h = h.strip()
            if h and h not in seen:
                host_list.append(h)
                seen.add(h)
                added.append(h)

    # Include the control machine unless opted out.
    # Use the local IP that can route to the first cluster host, since
    # remote hosts may not be able to resolve this machine's hostname.
    self_host: str | None = None
    cross_user = user != local_user
    if include_self and host_list:
        self_host = local_ip_for(host_list[0])
        if self_host and self_host in seen and cross_user:
            # Control machine was explicitly listed — keep it. The user
            # included it intentionally so the cluster user should exist there.
            click.echo(
                "Note: SSH user '%s' differs from local user '%s'. "
                "The mesh script will handle cross-user key exchange for %s automatically." % (user, local_user, self_host)
            )
        elif self_host and self_host not in seen and not cross_user:
            host_list.append(self_host)
            seen.add(self_host)
            added.append("%s (this machine)" % self_host)
        elif self_host and self_host not in seen and cross_user:
            # Don't auto-add the control machine — the cluster user
            # likely doesn't exist here.  The mesh script's cross-user
            # block will still install the local user's key on the
            # remote hosts for passwordless control→cluster SSH.
            click.echo(
                "Note: Skipping control machine (%s) in mesh — user '%s' differs from "
                "local user '%s'. Control→cluster SSH is handled automatically by the mesh script." % (self_host, user, local_user)
            )

    if not host_list:
        click.echo("Error: No hosts specified. Use --hosts, --hosts-file, or --cluster.", err=True)
        sys.exit(1)

    if len(host_list) < 2:
        click.echo(
            "Error: SSH mesh requires at least 2 hosts (got %d)." % len(host_list),
            err=True,
        )
        sys.exit(1)

    if not dry_run:
        click.echo("Setting up SSH mesh for user '%s' across %d hosts..." % (user, len(host_list)))
        click.echo("Cluster Hosts: %s" % ", ".join(sorted(cluster_hosts)))
        if added:
            click.echo("Added: %s" % ", ".join(added))
        click.echo()

    ok = _run_ssh_mesh(
        host_list,
        user,
        cluster_hosts=cluster_hosts,
        ssh_key=config.ssh_key,
        discover_ips=discover_ips,
        dry_run=dry_run,
        control_is_member=_resolved_loopback or (self_host is not None and self_host in set(cluster_hosts)),
    )
    sys.exit(0 if ok else 1)


@setup.command("cx7")
@host_options
@click.option("--user", "-u", default=None, help="SSH username (default: from config or current user)")
@dry_run_option
@click.option("--force", is_flag=True, help="Reconfigure even if existing config is valid")
@click.option("--mtu", default=9000, show_default=True, type=int, help="MTU for CX7 interfaces")
@click.option("--subnet1", default=None, help="Override subnet for CX7 partition 1 (e.g. 192.168.11.0/24)")
@click.option("--subnet2", default=None, help="Override subnet for CX7 partition 2 (e.g. 192.168.12.0/24)")
@click.pass_context
def setup_cx7(ctx, hosts, hosts_file, cluster_name, user, dry_run, force, mtu, subnet1, subnet2):
    """Configure CX7 network interfaces on cluster hosts.

    Detects ConnectX-7 interfaces, assigns static IPs on two /24 subnets
    with jumbo frames (MTU 9000), and applies netplan configuration.

    Existing valid configurations are preserved unless --force is used.
    IP addresses are derived from each host's management IP last octet.

    Requires passwordless sudo on target hosts.

    Examples:

      sparkrun setup cx7 --hosts 10.24.11.13,10.24.11.14

      sparkrun setup cx7 --cluster mylab --dry-run

      sparkrun setup cx7 --cluster mylab --subnet1 192.168.11.0/24 --subnet2 192.168.12.0/24

      sparkrun setup cx7 --cluster mylab --force
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.networking import (
        CX7HostDetection,
        configure_cx7_host,
        detect_cx7_for_hosts,
        distribute_cx7_host_keys,
        select_subnets,
        plan_cluster_cx7,
        apply_cx7_plan,
    )

    # Validate subnet pair
    if (subnet1 is None) != (subnet2 is None):
        click.echo("Error: --subnet1 and --subnet2 must be specified together.", err=True)
        sys.exit(1)

    config = SparkrunConfig()
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)

    # Step 1: Detect
    detections = detect_cx7_for_hosts(host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run)

    # Check all hosts have CX7
    no_cx7 = [h for h, d in detections.items() if not d.detected]
    if no_cx7:
        click.echo("Warning: No CX7 interfaces on: %s" % ", ".join(no_cx7), err=True)

    hosts_with_cx7 = {h: d for h, d in detections.items() if d.detected}
    if not hosts_with_cx7:
        click.echo("Error: No CX7 interfaces detected on any host.", err=True)
        sys.exit(1)

    # Step 2: Select subnets
    try:
        s1, s2 = select_subnets(detections, override1=subnet1, override2=subnet2)
    except RuntimeError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    click.echo()
    click.echo("Subnets: %s, %s" % (s1, s2))
    click.echo("MTU: %d" % mtu)
    click.echo()

    # Step 3: Plan
    plan = plan_cluster_cx7(detections, s1, s2, mtu=mtu, force=force)

    # Display plan
    for hp in plan.host_plans:
        det = detections.get(hp.host)
        mgmt_label = " (%s)" % det.mgmt_ip if det and det.mgmt_ip else ""
        click.echo("  %s%s" % (hp.host, mgmt_label))
        for a in hp.assignments:
            status = "OK" if not hp.needs_change else "configure"
            click.echo("    %-20s -> %s/%d  MTU %d  [%s]" % (a.iface_name, a.ip, plan.prefix_len, plan.mtu, status))
        if not hp.assignments and hp.needs_change:
            click.echo("    %s" % hp.reason)
        click.echo()

    # Show warnings
    for w in plan.warnings:
        click.echo("Warning: %s" % w, err=True)

    # Step 4: Check if all valid
    if plan.all_valid and not force:
        click.echo("All hosts already configured. Use --force to reconfigure.")
        return

    # Count
    needs_config = sum(1 for hp in plan.host_plans if hp.needs_change and len(hp.assignments) == 2)
    already_ok = sum(1 for hp in plan.host_plans if not hp.needs_change)
    has_errors = sum(1 for hp in plan.host_plans if hp.needs_change and len(hp.assignments) != 2)

    if needs_config == 0:
        if has_errors:
            click.echo("Error: %d host(s) have issues that prevent configuration." % has_errors, err=True)
            for e in plan.errors:
                click.echo("  %s" % e, err=True)
            sys.exit(1)
        click.echo("No hosts need configuration changes.")
        return

    if dry_run:
        click.echo("[dry-run] Would configure %d host(s), %d already valid." % (needs_config, already_ok))
        return

    # Step 5: Apply — prompt for sudo password if needed
    sudo_hosts_needing_pw = {
        hp.host
        for hp in plan.host_plans
        if hp.needs_change and len(hp.assignments) == 2 and not detections.get(hp.host, CX7HostDetection(host="")).sudo_ok
    }
    sudo_password = None
    if sudo_hosts_needing_pw:
        click.echo("Sudo password required for %d host(s)." % len(sudo_hosts_needing_pw))
        sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)

    click.echo("Applying configuration to %d host(s)..." % needs_config)
    results = apply_cx7_plan(
        plan,
        ssh_kwargs=ssh_kwargs,
        dry_run=dry_run,
        sudo_password=sudo_password,
        sudo_hosts=sudo_hosts_needing_pw,
    )

    # Build a map of host -> result for easy lookup
    result_map = {r.host: r for r in results}

    # Check for sudo failures and retry with per-host passwords
    if sudo_hosts_needing_pw and not dry_run:
        failed_sudo_hosts = [r.host for r in results if not r.success and r.host in sudo_hosts_needing_pw]
        if failed_sudo_hosts:
            click.echo()
            click.echo("Sudo authentication failed on %d host(s). Retrying individually..." % len(failed_sudo_hosts))
            # Build a lookup of host -> host_plan for retry
            host_plan_map = {hp.host: hp for hp in plan.host_plans}
            for fhost in failed_sudo_hosts:
                hp = host_plan_map.get(fhost)
                if not hp:
                    continue
                per_host_pw = click.prompt("[sudo] password for %s @ %s" % (user, fhost), hide_input=True)
                retry_result = configure_cx7_host(
                    hp,
                    mtu=plan.mtu,
                    prefix_len=plan.prefix_len,
                    ssh_kwargs=ssh_kwargs,
                    dry_run=dry_run,
                    sudo_password=per_host_pw,
                )
                result_map[fhost] = retry_result

    # Collect final results in plan order
    final_results = [result_map[hp.host] for hp in plan.host_plans if hp.host in result_map]
    configured = sum(1 for r in final_results if r.success)
    failed = sum(1 for r in final_results if not r.success)

    for r in final_results:
        if not r.success:
            click.echo("  [FAIL] %s: %s" % (r.host, r.stderr.strip()[:100]), err=True)

    click.echo()
    parts = []
    if configured:
        parts.append("%d configured" % configured)
    if already_ok:
        parts.append("%d already valid" % already_ok)
    if failed:
        parts.append("%d failed" % failed)
    if has_errors:
        parts.append("%d skipped (errors)" % has_errors)
    click.echo("Results: %s." % ", ".join(parts))

    # Step 6: Distribute CX7 host keys to known_hosts
    # Collect ALL CX7 IPs (both existing valid and newly configured) so that
    # every host (and the control machine) can SSH to every CX7 IP.
    all_cx7_ips = []
    for hp in plan.host_plans:
        for a in hp.assignments:
            if a.ip:
                all_cx7_ips.append(a.ip)

    if all_cx7_ips and not dry_run:
        click.echo()
        click.echo("Distributing CX7 host keys to known_hosts...")
        ks_results = distribute_cx7_host_keys(
            all_cx7_ips,
            host_list,
            ssh_kwargs=ssh_kwargs,
            dry_run=dry_run,
        )
        ks_ok = sum(1 for r in ks_results if r.success)
        ks_fail = sum(1 for r in ks_results if not r.success)
        if ks_fail:
            click.echo("  Warning: keyscan failed on %d host(s)." % ks_fail, err=True)
        click.echo("  Host keys for %d CX7 IPs distributed to %d host(s) + local." % (len(all_cx7_ips), ks_ok))

    if failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Docker group membership script (inline — too short for a separate .sh file)
# ---------------------------------------------------------------------------

# TODO: inline script
_DOCKER_GROUP_SCRIPT = """\
#!/bin/bash
set -uo pipefail
TARGET_USER="{user}"
if id -nG "$TARGET_USER" 2>/dev/null | grep -qw docker; then
    echo "DOCKER_GROUP=already_member"
else
    sudo -n usermod -aG docker "$TARGET_USER" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "DOCKER_GROUP=added"
    else
        echo "DOCKER_GROUP=needs_sudo"
        exit 1
    fi
fi
"""

# TODO: inline script
_DOCKER_GROUP_FALLBACK_SCRIPT = """\
#!/bin/bash
set -uo pipefail
TARGET_USER="{user}"
usermod -aG docker "$TARGET_USER"
echo "DOCKER_GROUP=added"
"""


def _docker_group_summary(stdout: str, user: str | None = None) -> str:
    """Extract status from docker-group script output."""
    label = "'%s' " % user if user else ""
    for line in stdout.strip().splitlines():
        if line.startswith("DOCKER_GROUP="):
            val = line.split("=", 1)[1]
            if val == "already_member":
                return "%salready a member" % label
            elif val == "added":
                return "added %sto docker group" % label
    return stdout.strip()[:80]


@setup.command("docker-group")
@host_options
@click.option("--user", "-u", default=None, help="Target user (default: SSH user)")
@dry_run_option
@click.pass_context
def setup_docker_group(ctx, hosts, hosts_file, cluster_name, user, dry_run):
    """Ensure user is a member of the docker group on cluster hosts.

    Runs ``usermod -aG docker <user>`` on each host so that Docker
    commands work without sudo.  Requires sudo on target hosts.

    A re-login (or ``newgrp docker``) is needed on hosts where the
    user was newly added.

    Examples:

      sparkrun setup docker-group --hosts 10.24.11.13,10.24.11.14

      sparkrun setup docker-group --cluster mylab

      sparkrun setup docker-group --cluster mylab --user ubuntu
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.sudo import run_with_sudo_fallback, run_sudo_script_on_host

    config = SparkrunConfig()
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)

    click.echo("Ensuring user '%s' is in the docker group on %d host(s)..." % (user, len(host_list)))
    click.echo()

    script = _DOCKER_GROUP_SCRIPT.format(user=user)
    fallback = _DOCKER_GROUP_FALLBACK_SCRIPT.format(user=user)

    result_map, still_failed = run_with_sudo_fallback(
        host_list,
        script,
        fallback,
        ssh_kwargs,
        dry_run=dry_run,
    )

    # Report immediate successes
    for h in host_list:
        r = result_map.get(h)
        if r and r.success:
            click.echo("  [OK]   %s: %s" % (h, _docker_group_summary(r.stdout, user=user)))

    # Prompt and retry if needed
    if still_failed and not dry_run:
        sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)
        for h in still_failed:
            r = run_sudo_script_on_host(
                h,
                fallback,
                sudo_password,
                ssh_kwargs=ssh_kwargs,
                timeout=30,
                dry_run=dry_run,
            )
            result_map[h] = r

    # Final summary
    ok_count = sum(1 for h in host_list if result_map.get(h) and result_map[h].success)
    fail_count = sum(1 for h in host_list if result_map.get(h) and not result_map[h].success)

    for h in host_list:
        r = result_map.get(h)
        if r and not r.success:
            click.echo("  [FAIL] %s: %s" % (h, r.stderr.strip()[:200]), err=True)
        elif r and r.success and h in (still_failed or []):
            click.echo("  [OK]   %s: %s" % (h, _docker_group_summary(r.stdout, user=user)))

    click.echo()
    parts = []
    if ok_count:
        parts.append("%d OK" % ok_count)
    if fail_count:
        parts.append("%d failed" % fail_count)
    click.echo("Results: %s." % ", ".join(parts) if parts else "No hosts processed.")

    if ok_count and any("added" in (result_map.get(h).stdout if result_map.get(h) else "") for h in host_list):
        click.echo()
        click.echo("Note: Users newly added to the docker group must re-login")
        click.echo("(or run 'newgrp docker') for the change to take effect.")

    if fail_count:
        sys.exit(1)


@setup.command("fix-permissions")
@host_options
@click.option("--user", "-u", default=None, help="Target owner (default: SSH user)")
@click.option("--cache-dir", default=None, help="Cache directory (default: ~/.cache/huggingface)")
@click.option("--save-sudo", is_flag=True, default=False, help="Install sudoers entry for passwordless chown (requires sudo once)")
@dry_run_option
@click.pass_context
def setup_fix_permissions(ctx, hosts, hosts_file, cluster_name, user, cache_dir, save_sudo, dry_run):
    """Fix file ownership in HuggingFace cache on cluster hosts.

    Docker containers create files as root in ~/.cache/huggingface/,
    leaving the normal user unable to manage or clean the cache.
    This command runs chown on all target hosts to restore ownership.

    Tries non-interactive sudo first on all hosts in parallel, then
    falls back to password-based sudo for any that fail.

    Use --save-sudo to install a scoped sudoers entry so future runs
    never need a password. The entry only permits chown on the cache
    directory — no broader privileges are granted.

    Examples:

      sparkrun setup fix-permissions --hosts 10.24.11.13,10.24.11.14

      sparkrun setup fix-permissions --cluster mylab

      sparkrun setup fix-permissions --cluster mylab --cache-dir /data/hf-cache

      sparkrun setup fix-permissions --cluster mylab --save-sudo

      sparkrun setup fix-permissions --cluster mylab --dry-run
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.sudo import run_with_sudo_fallback, run_sudo_script_on_host

    config = SparkrunConfig()
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)

    # Resolve cache path
    cache_path = cache_dir  # None means use getent-based home detection

    click.echo("Fixing file permissions for user '%s' on %d host(s)..." % (user, len(host_list)))
    if cache_path:
        click.echo("Cache directory: %s" % cache_path)
    click.echo()

    sudo_password = None

    from sparkrun.scripts import read_script

    # --save-sudo: install scoped sudoers entry on each host
    if save_sudo:
        click.echo("Installing sudoers entry for passwordless chown...")
        from sparkrun.utils.shell import validate_unix_username
        import re as _re

        validate_unix_username(user)
        safe_cache_dir = cache_path or ""
        if safe_cache_dir and not _re.fullmatch(r"[/a-zA-Z0-9_.~-]+", safe_cache_dir):
            raise click.UsageError("cache_dir contains unsafe characters: %r" % safe_cache_dir)
        sudoers_script = read_script("fix_permissions_sudoers.sh").format(
            user=user,
            cache_dir=safe_cache_dir,
        )

        if dry_run:
            click.echo("  [dry-run] Would install sudoers entry on %d host(s):" % len(host_list))
            for h in host_list:
                click.echo("    %s: /etc/sudoers.d/sparkrun-chown-%s" % (h, user))
            click.echo()
        else:
            sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)
            sudoers_ok = 0
            sudoers_fail = 0
            for h in host_list:
                r = run_sudo_script_on_host(
                    h,
                    sudoers_script,
                    sudo_password,
                    ssh_kwargs=ssh_kwargs,
                    timeout=300,
                    dry_run=False,
                )
                if r.success:
                    sudoers_ok += 1
                    click.echo("  [OK]   %s: %s" % (h, r.stdout.strip()))
                else:
                    sudoers_fail += 1
                    click.echo("  [FAIL] %s: %s" % (h, r.stderr.strip()[:200]), err=True)
            click.echo("Sudoers install: %d OK, %d failed." % (sudoers_ok, sudoers_fail))
            click.echo()

    # Generate the chown script with sudo -n (non-interactive).
    # Uses getent passwd to resolve the target user's home directory,
    # avoiding tilde/HOME ambiguity when running under sudo.
    chown_script = read_script("fix_permissions.sh").format(
        user=user,
        cache_dir=cache_path or "",
    )

    # Password-based fallback script (no sudo prefix — run_remote_sudo_script runs as root)
    fallback_script = read_script("fix_permissions_fallback.sh").format(
        user=user,
        cache_dir=cache_path or "",
    )

    # Try non-interactive sudo, then password-based fallback
    if not dry_run and sudo_password is None:
        # Prompt only if parallel run produces failures (deferred below)
        pass

    result_map, still_failed = run_with_sudo_fallback(
        host_list,
        chown_script,
        fallback_script,
        ssh_kwargs,
        dry_run=dry_run,
        sudo_password=sudo_password,
    )

    # If hosts failed without a password, prompt and retry
    if still_failed and not dry_run:
        if sudo_password is None:
            sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)
            # Re-run fallback with the password for failed hosts
            result_map, still_failed = run_with_sudo_fallback(
                still_failed,
                chown_script,
                fallback_script,
                ssh_kwargs,
                dry_run=dry_run,
                sudo_password=sudo_password,
            )

        # Retry individually on per-host sudo failures
        if still_failed and sudo_password:
            click.echo()
            click.echo("Sudo authentication failed on %d host(s). Retrying individually..." % len(still_failed))
            for fhost in still_failed:
                per_host_pw = click.prompt("[sudo] password for %s @ %s" % (user, fhost), hide_input=True)
                retry_result = run_sudo_script_on_host(
                    fhost,
                    fallback_script,
                    per_host_pw,
                    ssh_kwargs=ssh_kwargs,
                    timeout=300,
                    dry_run=dry_run,
                )
                result_map[fhost] = retry_result

    # Report results
    ok_count = 0
    skip_count = 0
    fail_count = 0
    for h in host_list:
        r = result_map.get(h)
        if r is None:
            continue
        if r.success:
            if "SKIP:" in r.stdout:
                skip_count += 1
                click.echo("  [SKIP] %s: %s" % (h, r.stdout.strip()))
            else:
                ok_count += 1
                click.echo("  [OK]   %s: %s" % (h, r.stdout.strip()))
        else:
            fail_count += 1
            click.echo("  [FAIL] %s: %s" % (h, r.stderr.strip()[:200]), err=True)

    click.echo()
    parts = []
    if ok_count:
        parts.append("%d fixed" % ok_count)
    if skip_count:
        parts.append("%d skipped (no cache)" % skip_count)
    if fail_count:
        parts.append("%d failed" % fail_count)
    click.echo("Results: %s." % ", ".join(parts) if parts else "No hosts processed.")

    if fail_count:
        sys.exit(1)


@setup.command("clear-cache")
@host_options
@click.option("--user", "-u", default=None, help="Target user for sudoers entry (default: SSH user)")
@click.option("--save-sudo", is_flag=True, default=False, help="Install sudoers entry for passwordless cache clearing (requires sudo once)")
@dry_run_option
@click.pass_context
def setup_clear_cache(ctx, hosts, hosts_file, cluster_name, user, save_sudo, dry_run):
    """Drop the Linux page cache on cluster hosts.

    Runs 'sync' followed by writing 3 to /proc/sys/vm/drop_caches on
    each target host.  This frees cached file data so inference
    containers have maximum available memory on DGX Spark's unified
    CPU/GPU memory.

    Tries non-interactive sudo first on all hosts in parallel, then
    falls back to password-based sudo for any that fail.

    Use --save-sudo to install a scoped sudoers entry so future runs
    never need a password. The entry only permits writing to
    /proc/sys/vm/drop_caches — no broader privileges are granted.

    Examples:

      sparkrun setup clear-cache --hosts 10.24.11.13,10.24.11.14

      sparkrun setup clear-cache --cluster mylab

      sparkrun setup clear-cache --cluster mylab --save-sudo

      sparkrun setup clear-cache --cluster mylab --dry-run
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.sudo import run_with_sudo_fallback, run_sudo_script_on_host

    config = SparkrunConfig()
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)

    click.echo("Clearing page cache on %d host(s)..." % len(host_list))
    click.echo()

    sudo_password = None

    from sparkrun.scripts import read_script

    # --save-sudo: install scoped sudoers entry on each host
    if save_sudo:
        click.echo("Installing sudoers entry for passwordless cache clearing...")
        from sparkrun.utils.shell import validate_unix_username

        validate_unix_username(user)
        sudoers_script = read_script("clear_cache_sudoers.sh").format(user=user)

        if dry_run:
            click.echo("  [dry-run] Would install sudoers entry on %d host(s):" % len(host_list))
            for h in host_list:
                click.echo("    %s: /etc/sudoers.d/sparkrun-dropcaches-%s" % (h, user))
            click.echo()
        else:
            sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)
            sudoers_ok = 0
            sudoers_fail = 0
            for h in host_list:
                r = run_sudo_script_on_host(
                    h,
                    sudoers_script,
                    sudo_password,
                    ssh_kwargs=ssh_kwargs,
                    timeout=300,
                    dry_run=False,
                )
                if r.success:
                    sudoers_ok += 1
                    click.echo("  [OK]   %s: %s" % (h, r.stdout.strip()))
                else:
                    sudoers_fail += 1
                    click.echo("  [FAIL] %s: %s" % (h, r.stderr.strip()[:200]), err=True)
            click.echo("Sudoers install: %d OK, %d failed." % (sudoers_ok, sudoers_fail))
            click.echo()

    # Generate the drop_caches script with sudo -n (non-interactive).
    drop_script = read_script("clear_cache.sh")

    # Password-based fallback script (no sudo — run_remote_sudo_script runs as root)
    fallback_script = read_script("clear_cache_fallback.sh")

    # Try non-interactive sudo, then password-based fallback
    result_map, still_failed = run_with_sudo_fallback(
        host_list,
        drop_script,
        fallback_script,
        ssh_kwargs,
        dry_run=dry_run,
        sudo_password=sudo_password,
    )

    # If hosts failed without a password, prompt and retry
    if still_failed and not dry_run:
        if sudo_password is None:
            sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)
            result_map, still_failed = run_with_sudo_fallback(
                still_failed,
                drop_script,
                fallback_script,
                ssh_kwargs,
                dry_run=dry_run,
                sudo_password=sudo_password,
            )

        # Retry individually on per-host sudo failures
        if still_failed and sudo_password:
            click.echo()
            click.echo("Sudo authentication failed on %d host(s). Retrying individually..." % len(still_failed))
            for fhost in still_failed:
                per_host_pw = click.prompt("[sudo] password for %s @ %s" % (user, fhost), hide_input=True)
                retry_result = run_sudo_script_on_host(
                    fhost,
                    fallback_script,
                    per_host_pw,
                    ssh_kwargs=ssh_kwargs,
                    timeout=300,
                    dry_run=dry_run,
                )
                result_map[fhost] = retry_result

    # Report results
    ok_count = 0
    fail_count = 0
    for h in host_list:
        r = result_map.get(h)
        if r is None:
            continue
        if r.success:
            ok_count += 1
            click.echo("  [OK]   %s: %s" % (h, r.stdout.strip()))
        else:
            fail_count += 1
            click.echo("  [FAIL] %s: %s" % (h, r.stderr.strip()[:200]), err=True)

    click.echo()
    parts = []
    if ok_count:
        parts.append("%d cleared" % ok_count)
    if fail_count:
        parts.append("%d failed" % fail_count)
    click.echo("Results: %s." % ", ".join(parts) if parts else "No hosts processed.")

    if fail_count:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Earlyoom process patterns — used to build --prefer / --avoid arguments
# ---------------------------------------------------------------------------

# Processes that should be killed first on OOM (inference workloads).
# These are the main memory consumers on DGX Spark systems.
EARLYOOM_PREFER_PATTERNS = [
    "vllm",
    "VLLM",
    "sglang",
    "llama-server",
    "llama-cli",
    "trtllm",
    "tritonserver",
    "ray",
    "python3",
    "python",
]

# Processes to protect from OOM kill (system services).
EARLYOOM_AVOID_PATTERNS = [
    "systemd",
    "sshd",
    "dockerd",
    "containerd",
    "dbus-daemon",
    "NetworkManager",
]


def _earlyoom_summary(stdout: str) -> str:
    """Extract key status lines from earlyoom install output.

    Filters out noisy apt-get progress (Reading database ...) and
    returns only the meaningful status lines (INSTALLED/PRESENT/CONFIGURED/OK/ERROR).
    """
    keywords = ("INSTALLING:", "INSTALLED:", "PRESENT:", "CONFIGURED:", "OK:", "ERROR:")
    lines = [line.strip() for line in stdout.strip().splitlines() if any(line.strip().startswith(kw) for kw in keywords)]
    return "; ".join(lines) if lines else stdout.strip()[:100]


def _build_earlyoom_regex(patterns: list[str]) -> str:
    """Build a regex pattern string for earlyoom --prefer/--avoid.

    earlyoom uses POSIX extended regex matching against ``/proc/pid/comm``.
    Wraps patterns in ``(...)`` so matching is unanchored and can match
    anywhere in the process name.
    """
    return "(%s)" % "|".join(patterns)


@setup.command("earlyoom")
@host_options
@click.option("--user", "-u", default=None, help="SSH username (default: from config or current user)")
@click.option("--prefer", "extra_prefer", default=None, help="Additional comma-separated process patterns to prefer killing on OOM")
@click.option("--avoid", "extra_avoid", default=None, help="Additional comma-separated process patterns to avoid killing on OOM")
@dry_run_option
@click.pass_context
def setup_earlyoom(ctx, hosts, hosts_file, cluster_name, user, extra_prefer, extra_avoid, dry_run):
    """Install and configure earlyoom OOM killer on cluster hosts.

    earlyoom monitors available memory and proactively kills processes
    before the kernel OOM killer triggers. This prevents system hangs
    when large inference models exhaust memory on DGX Spark.

    By default, sparkrun configures earlyoom to prefer killing inference
    workload processes (vllm, sglang, llama-server, trtllm, python) and
    to avoid killing system services (sshd, systemd, dockerd).

    Use --prefer/--avoid to add additional process patterns.

    Requires sudo on target hosts (apt-get install, systemctl).

    Examples:

      sparkrun setup earlyoom --hosts 192.168.11.13,192.168.11.14

      sparkrun setup earlyoom --cluster mylab

      sparkrun setup earlyoom --cluster mylab --prefer "my-app,worker"

      sparkrun setup earlyoom --cluster mylab --dry-run
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.sudo import run_with_sudo_fallback, run_sudo_script_on_host

    config = SparkrunConfig()
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)

    # Build prefer/avoid pattern lists
    prefer = list(EARLYOOM_PREFER_PATTERNS)
    if extra_prefer:
        prefer.extend(p.strip() for p in extra_prefer.split(",") if p.strip())

    avoid = list(EARLYOOM_AVOID_PATTERNS)
    if extra_avoid:
        avoid.extend(p.strip() for p in extra_avoid.split(",") if p.strip())

    prefer_regex = _build_earlyoom_regex(prefer)
    avoid_regex = _build_earlyoom_regex(avoid)

    click.echo("Installing earlyoom on %d host(s)..." % len(host_list))
    click.echo("  Prefer (kill first): %s" % prefer_regex)
    click.echo("  Avoid (protect):     %s" % avoid_regex)
    click.echo()
    click.echo("Note: first install may take ~1 minute per host (apt-get update + install).")
    click.echo()

    from sparkrun.scripts import read_script

    # Generate install scripts with the prefer/avoid patterns
    install_script = read_script("earlyoom_install.sh").format(
        prefer=prefer_regex,
        avoid=avoid_regex,
    )
    fallback_script = read_script("earlyoom_install_fallback.sh").format(
        prefer=prefer_regex,
        avoid=avoid_regex,
    )

    # Try non-interactive sudo, then password-based fallback
    result_map, still_failed = run_with_sudo_fallback(
        host_list,
        install_script,
        fallback_script,
        ssh_kwargs,
        dry_run=dry_run,
    )

    # Report hosts that succeeded immediately
    for h in host_list:
        r = result_map.get(h)
        if r and r.success:
            click.echo("  [OK]   %s: %s" % (h, _earlyoom_summary(r.stdout)))

    # If hosts failed without a password, prompt and retry
    if still_failed and not dry_run:
        sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)

        # Run fallback with progress — report each host as it completes
        click.echo("Configuring %d host(s)..." % len(still_failed))
        remaining = list(still_failed)
        for h in remaining:
            click.echo("  %-20s ..." % h, nl=False)
            r = run_sudo_script_on_host(
                h,
                fallback_script,
                sudo_password,
                ssh_kwargs=ssh_kwargs,
                timeout=300,
                dry_run=dry_run,
            )
            result_map[h] = r
            if r.success:
                click.echo(" %s" % _earlyoom_summary(r.stdout))
            else:
                click.echo(" FAILED")

        still_failed = [h for h in remaining if not result_map.get(h) or not result_map[h].success]

        # Retry individually on per-host sudo failures
        if still_failed and sudo_password:
            click.echo()
            click.echo("Sudo authentication failed on %d host(s). Retrying individually..." % len(still_failed))
            for fhost in still_failed:
                per_host_pw = click.prompt("[sudo] password for %s @ %s" % (user, fhost), hide_input=True)
                click.echo("  %-20s ..." % fhost, nl=False)
                retry_result = run_sudo_script_on_host(
                    fhost,
                    fallback_script,
                    per_host_pw,
                    ssh_kwargs=ssh_kwargs,
                    timeout=300,
                    dry_run=dry_run,
                )
                result_map[fhost] = retry_result
                if retry_result.success:
                    click.echo(" %s" % _earlyoom_summary(retry_result.stdout))
                else:
                    click.echo(" FAILED")

    # Final summary
    ok_count = sum(1 for h in host_list if result_map.get(h) and result_map[h].success)
    fail_count = sum(1 for h in host_list if result_map.get(h) and not result_map[h].success)

    for h in host_list:
        r = result_map.get(h)
        if r and not r.success:
            click.echo("  [FAIL] %s: %s" % (h, r.stderr.strip()[:200]), err=True)

    click.echo()
    parts = []
    if ok_count:
        parts.append("%d configured" % ok_count)
    if fail_count:
        parts.append("%d failed" % fail_count)
    click.echo("Results: %s." % ", ".join(parts) if parts else "No hosts processed.")

    if ok_count:
        click.echo()
        click.echo("Thanks to @shahizat for posting this idea on the DGX forums!")

    if fail_count:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------


@setup.command("diagnose", hidden=True)
@host_options
@dry_run_option
@click.option(
    "-o",
    "--output",
    "output_file",
    default=None,
    type=click.Path(),
    help="Output NDJSON file path (default: spark_diag_<timestamp>.ndjson)",
)
@click.option("--json", "json_stdout", is_flag=True, help="Also print summary to stdout as JSON")
@click.option("--sudo", "use_sudo", is_flag=True, default=False, help="Also collect sudo-only diagnostics (dmidecode)")
@click.pass_context
def setup_diagnose(ctx, hosts, hosts_file, cluster_name, dry_run, output_file, json_stdout, use_sudo):
    """Collect hardware, firmware, network, and Docker diagnostics from hosts.

    Collects OS, kernel, CPU, memory, disk, GPU, network, Docker, and
    firmware device information without requiring elevated privileges.

    Use --sudo to also collect dmidecode data (BIOS, system, baseboard,
    memory details) which requires a sudo password.
    """
    import json as _json
    from datetime import datetime, timezone

    from sparkrun.diagnostics import (
        NDJSONWriter,
        collect_config_diagnostics,
        collect_spark_diagnostics,
        collect_sudo_diagnostics,
    )

    from ._common import _get_cluster_manager, _get_context, _resolve_setup_context

    sctx = _get_context(ctx)
    config = sctx.config

    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config)

    if not output_file:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = "spark_diag_%s.ndjson" % ts

    # Prompt for sudo password upfront if --sudo
    sudo_password = None
    if use_sudo and not dry_run:
        sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)

    click.echo("Collecting diagnostics from %d host(s)..." % len(host_list))
    click.echo("Output: %s" % output_file)
    click.echo()

    with NDJSONWriter(output_file) as writer:
        try:
            from sparkrun import __version__
        except Exception:
            __version__ = "unknown"

        writer.emit(
            "diag_header",
            {
                "sparkrun_version": __version__,
                "hosts": host_list,
                "command": "sparkrun setup diagnose",
                "sudo": use_sudo,
            },
        )

        # Local config: clusters and registries
        cluster_mgr = _get_cluster_manager()
        registry_mgr = config.get_registry_manager()
        collect_config_diagnostics(
            writer,
            config=config,
            cluster_mgr=cluster_mgr,
            registry_mgr=registry_mgr,
        )

        host_data = collect_spark_diagnostics(
            hosts=host_list,
            ssh_kwargs=ssh_kwargs,
            writer=writer,
            dry_run=dry_run,
        )

        # Sudo pass: dmidecode
        if use_sudo and sudo_password:
            click.echo("Collecting sudo diagnostics (dmidecode)...")
            collect_sudo_diagnostics(
                hosts=host_list,
                ssh_kwargs=ssh_kwargs,
                sudo_password=sudo_password,
                writer=writer,
                dry_run=dry_run,
            )

    # Display summary
    ok = sum(1 for v in host_data.values() if v.get("DIAG_COMPLETE") == "1")
    fail = len(host_list) - ok

    click.echo()
    click.echo("Diagnostics complete: %d/%d hosts OK" % (ok, len(host_list)))
    if fail:
        click.echo("  %d host(s) failed — see %s for details" % (fail, output_file), err=True)

    if json_stdout:
        summary = {
            "output_file": output_file,
            "total_hosts": len(host_list),
            "successful": ok,
            "failed": fail,
            "hosts": {h: bool(d.get("DIAG_COMPLETE") == "1") for h, d in host_data.items()},
        }
        click.echo(_json.dumps(summary, indent=2))

    if fail:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Wizard registration
# ---------------------------------------------------------------------------

from ._wizard import setup_wizard  # noqa: E402

setup.add_command(setup_wizard)
