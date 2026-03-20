"""sparkrun setup group and subcommands."""

from __future__ import annotations

import sys

import click

from ._common import (
    _detect_shell,
    _get_cluster_manager,
    _require_uv,
    _resolve_cluster_user,
    _resolve_setup_context,
    _shell_rc_file,
    dry_run_option,
    host_options,
)


@click.group()
@click.pass_context
def setup(ctx):
    """Setup and configuration commands."""
    pass


@setup.command("completion")
@click.option("--shell", type=click.Choice(["bash", "zsh", "fish"]), default=None,
              help="Shell type (auto-detected if not specified)")
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
@click.option("--shell", type=click.Choice(["bash", "zsh", "fish"]), default=None,
              help="Shell type (auto-detected if not specified)")
@click.option("--no-update-registries", is_flag=True,
              help="Skip updating recipe registries after installation")
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
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        click.echo("Error installing sparkrun: %s" % result.stderr.strip(), err=True)
        sys.exit(1)
    click.echo("sparkrun installed on PATH")

    # Step 2: Clean up old aliases/functions from previous installs
    if rc_file.exists():
        old_markers = [
            "alias sparkrun=", "alias sparkrun ",
            "function sparkrun", "sparkrun()",
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
@click.option("--no-update-registries", is_flag=True,
              help="Skip updating recipe registries after upgrading sparkrun")
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
        capture_output=True, text=True,
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
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        click.echo("Error updating sparkrun: %s" % result.stderr.strip(), err=True)
        sys.exit(1)

    # The running process still has the old module cached, and reload
    # won't help because uv tool installs into a separate virtualenv.
    # Ask the newly installed binary instead.
    ver_result = subprocess.run(
        ["sparkrun", "--version"],
        capture_output=True, text=True,
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


@setup.command("ssh")
@host_options
@click.option("--extra-hosts", default=None,
              help="Additional comma-separated hosts to include (e.g. control machine)")
@click.option("--include-self/--no-include-self", default=True, show_default=True,
              help="Include this machine's hostname in the mesh")
@click.option("--user", "-u", default=None, help="SSH username (default: current user)")
@click.option("--discover-ips/--no-discover-ips", default=True, show_default=True,
              help="After meshing, discover IB/CX7 IPs and distribute host keys")
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
    import subprocess

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
    cluster_user = _resolve_cluster_user(cluster_name, hosts, hosts_file, cluster_mgr)

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
    if include_self and host_list:
        self_host = local_ip_for(host_list[0])
        if self_host and self_host not in seen:
            host_list.append(self_host)
            seen.add(self_host)
            added.append("%s (this machine)" % self_host)

    if not host_list:
        click.echo("Error: No hosts specified. Use --hosts, --hosts-file, or --cluster.", err=True)
        sys.exit(1)

    if len(host_list) < 2:
        click.echo(
            "Error: SSH mesh requires at least 2 hosts (got %d)." % len(host_list),
            err=True,
        )
        sys.exit(1)

    # Default user: --user flag > cluster user > config ssh.user > OS user
    if user is None:
        user = cluster_user or config.ssh_user or os.environ.get("USER", "root")

    # === Phase 1: Mesh SSH keys ===
    from sparkrun.scripts import get_script_path
    with get_script_path("mesh_ssh_keys.sh") as script_path:
        cmd = ["bash", str(script_path), user] + host_list

        if dry_run:
            click.echo("Would run:")
            click.echo("  " + " ".join(cmd))
            if discover_ips:
                click.echo()
                click.echo("Phase 2 (discover IPs + distribute host keys) would run after mesh completes.")
            return

        click.echo("Setting up SSH mesh for user '%s' across %d hosts..." % (user, len(host_list)))
        click.echo("Cluster Hosts: %s" % ", ".join(sorted(cluster_hosts)))
        if added:
            click.echo("Added: %s" % ", ".join(added))
        click.echo()

        # Run interactively — the script prompts for passwords
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    # === Phase 2: Discover additional IPs and distribute host keys ===
    if not discover_ips:
        sys.exit(0)

    if len(cluster_hosts) < 2:
        sys.exit(0)

    click.echo()
    click.echo("Discovering additional network IPs on cluster hosts...")

    ssh_kwargs = {"ssh_user": user}
    if config.ssh_key:
        ssh_kwargs["ssh_key"] = config.ssh_key

    from sparkrun.orchestration.networking import discover_host_network_ips, distribute_host_keys
    from sparkrun.orchestration.primitives import check_tcp_reachability

    discovered = discover_host_network_ips(cluster_hosts, ssh_kwargs=ssh_kwargs)

    if not discovered:
        click.echo("  No additional network IPs found.")
        sys.exit(0)

    # Collect all unique discovered IPs
    all_discovered_ips: list[str] = []
    seen_ips: set[str] = set()
    for host, ips in sorted(discovered.items()):
        click.echo("  %s: %s" % (host, ", ".join(ips)))
        for ip in ips:
            if ip not in seen_ips:
                all_discovered_ips.append(ip)
                seen_ips.add(ip)

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

    # Distribute host keys for ALL discovered IPs — cluster nodes can
    # reach each other's IB IPs even when the control machine can't.
    click.echo()
    click.echo("Distributing host keys for %d discovered IP(s)..." % len(all_discovered_ips))
    ks_results = distribute_host_keys(
        all_discovered_ips, cluster_hosts, ssh_kwargs=ssh_kwargs,
    )
    ks_ok = sum(1 for r in ks_results if r.success)
    ks_fail = sum(1 for r in ks_results if not r.success)
    if ks_fail:
        click.echo("  Warning: keyscan failed on %d host(s)." % ks_fail, err=True)
    click.echo("  Host keys for %d IP(s) distributed to %d host(s) + local." % (len(all_discovered_ips), ks_ok))

    sys.exit(0)


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
            click.echo("    %-20s -> %s/%d  MTU %d  [%s]" % (
                a.iface_name, a.ip, plan.prefix_len, plan.mtu, status))
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
        hp.host for hp in plan.host_plans
        if hp.needs_change and len(hp.assignments) == 2
           and not detections.get(hp.host, CX7HostDetection(host="")).sudo_ok
    }
    sudo_password = None
    if sudo_hosts_needing_pw:
        click.echo("Sudo password required for %d host(s)." % len(sudo_hosts_needing_pw))
        sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)

    click.echo("Applying configuration to %d host(s)..." % needs_config)
    results = apply_cx7_plan(
        plan, ssh_kwargs=ssh_kwargs, dry_run=dry_run,
        sudo_password=sudo_password, sudo_hosts=sudo_hosts_needing_pw,
    )

    # Build a map of host -> result for easy lookup
    result_map = {r.host: r for r in results}

    # Check for sudo failures and retry with per-host passwords
    if sudo_hosts_needing_pw and not dry_run:
        failed_sudo_hosts = [
            r.host for r in results
            if not r.success and r.host in sudo_hosts_needing_pw
        ]
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
                    hp, mtu=plan.mtu, prefix_len=plan.prefix_len,
                    ssh_kwargs=ssh_kwargs, dry_run=dry_run,
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
            all_cx7_ips, host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run,
        )
        ks_ok = sum(1 for r in ks_results if r.success)
        ks_fail = sum(1 for r in ks_results if not r.success)
        if ks_fail:
            click.echo("  Warning: keyscan failed on %d host(s)." % ks_fail, err=True)
        click.echo("  Host keys for %d CX7 IPs distributed to %d host(s) + local." % (len(all_cx7_ips), ks_ok))

    if failed:
        sys.exit(1)


@setup.command("fix-permissions")
@host_options
@click.option("--user", "-u", default=None, help="Target owner (default: SSH user)")
@click.option("--cache-dir", default=None, help="Cache directory (default: ~/.cache/huggingface)")
@click.option("--save-sudo", is_flag=True, default=False,
              help="Install sudoers entry for passwordless chown (requires sudo once)")
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
        sudoers_script = read_script("fix_permissions_sudoers.sh").format(
            user=user, cache_dir=cache_path or "",
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
                    h, sudoers_script, sudo_password, ssh_kwargs=ssh_kwargs, timeout=300, dry_run=False,
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
        user=user, cache_dir=cache_path or "",
    )

    # Password-based fallback script (no sudo prefix — run_remote_sudo_script runs as root)
    fallback_script = read_script("fix_permissions_fallback.sh").format(
        user=user, cache_dir=cache_path or "",
    )

    # Try non-interactive sudo, then password-based fallback
    if not dry_run and sudo_password is None:
        # Prompt only if parallel run produces failures (deferred below)
        pass

    result_map, still_failed = run_with_sudo_fallback(
        host_list, chown_script, fallback_script, ssh_kwargs,
        dry_run=dry_run, sudo_password=sudo_password,
    )

    # If hosts failed without a password, prompt and retry
    if still_failed and not dry_run:
        if sudo_password is None:
            click.echo("Sudo password required for %d host(s)." % len(still_failed))
            sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)
            # Re-run fallback with the password for failed hosts
            result_map, still_failed = run_with_sudo_fallback(
                still_failed, chown_script, fallback_script, ssh_kwargs,
                dry_run=dry_run, sudo_password=sudo_password,
            )

        # Retry individually on per-host sudo failures
        if still_failed and sudo_password:
            click.echo()
            click.echo("Sudo authentication failed on %d host(s). Retrying individually..." % len(still_failed))
            for fhost in still_failed:
                per_host_pw = click.prompt("[sudo] password for %s @ %s" % (user, fhost), hide_input=True)
                retry_result = run_sudo_script_on_host(
                    fhost, fallback_script, per_host_pw, ssh_kwargs=ssh_kwargs, timeout=300, dry_run=dry_run,
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
@click.option("--save-sudo", is_flag=True, default=False,
              help="Install sudoers entry for passwordless cache clearing (requires sudo once)")
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
                    h, sudoers_script, sudo_password, ssh_kwargs=ssh_kwargs, timeout=300, dry_run=False,
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
        host_list, drop_script, fallback_script, ssh_kwargs,
        dry_run=dry_run, sudo_password=sudo_password,
    )

    # If hosts failed without a password, prompt and retry
    if still_failed and not dry_run:
        if sudo_password is None:
            click.echo("Sudo password required for %d host(s)." % len(still_failed))
            sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)
            result_map, still_failed = run_with_sudo_fallback(
                still_failed, drop_script, fallback_script, ssh_kwargs,
                dry_run=dry_run, sudo_password=sudo_password,
            )

        # Retry individually on per-host sudo failures
        if still_failed and sudo_password:
            click.echo()
            click.echo("Sudo authentication failed on %d host(s). Retrying individually..." % len(still_failed))
            for fhost in still_failed:
                per_host_pw = click.prompt("[sudo] password for %s @ %s" % (user, fhost), hide_input=True)
                retry_result = run_sudo_script_on_host(
                    fhost, fallback_script, per_host_pw, ssh_kwargs=ssh_kwargs, timeout=300, dry_run=dry_run,
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
    lines = [
        line.strip() for line in stdout.strip().splitlines()
        if any(line.strip().startswith(kw) for kw in keywords)
    ]
    return "; ".join(lines) if lines else stdout.strip()[:100]


def _build_earlyoom_regex(patterns: list[str]) -> str:
    """Build a regex pattern string for earlyoom --prefer/--avoid.

    earlyoom uses POSIX extended regex matching against ``/proc/pid/comm``.
    Wraps patterns in ``^(...)`` so matching is anchored to the start of
    the process name.
    """
    return "^(%s)" % "|".join(patterns)


@setup.command("earlyoom")
@host_options
@click.option("--user", "-u", default=None, help="SSH username (default: from config or current user)")
@click.option("--prefer", "extra_prefer", default=None,
              help="Additional comma-separated process patterns to prefer killing on OOM")
@click.option("--avoid", "extra_avoid", default=None,
              help="Additional comma-separated process patterns to avoid killing on OOM")
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
        prefer=prefer_regex, avoid=avoid_regex,
    )
    fallback_script = read_script("earlyoom_install_fallback.sh").format(
        prefer=prefer_regex, avoid=avoid_regex,
    )

    # Try non-interactive sudo, then password-based fallback
    result_map, still_failed = run_with_sudo_fallback(
        host_list, install_script, fallback_script, ssh_kwargs,
        dry_run=dry_run,
    )

    # Report hosts that succeeded immediately
    for h in host_list:
        r = result_map.get(h)
        if r and r.success:
            click.echo("  [OK]   %s: %s" % (h, _earlyoom_summary(r.stdout)))

    # If hosts failed without a password, prompt and retry
    if still_failed and not dry_run:
        click.echo("Sudo password required for %d host(s)." % len(still_failed))
        sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)

        # Run fallback with progress — report each host as it completes
        click.echo("Configuring %d host(s)..." % len(still_failed))
        remaining = list(still_failed)
        for h in remaining:
            click.echo("  %-20s ..." % h, nl=False)
            r = run_sudo_script_on_host(
                h, fallback_script, sudo_password, ssh_kwargs=ssh_kwargs, timeout=300, dry_run=dry_run,
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
                    fhost, fallback_script, per_host_pw, ssh_kwargs=ssh_kwargs, timeout=300, dry_run=dry_run,
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
