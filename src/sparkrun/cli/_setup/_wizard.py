"""Guided setup wizard for sparkrun."""

from __future__ import annotations

import logging

import click

logger = logging.getLogger(__name__)


@click.command("wizard")
@click.option("--hosts", "-H", default=None, help="Pre-populate host list (comma-separated)")
@click.option("--cluster", "cluster_name", default=None, help="Cluster name")
@click.option("--user", "-u", default=None, help="SSH username")
@click.option("--dry-run", "-n", is_flag=True, help="Preview without executing")
@click.option("--yes", "-y", is_flag=True, help="Accept all defaults (non-interactive)")
@click.pass_context
def setup_wizard(ctx, hosts, cluster_name, user, dry_run, yes):
    """Guided setup wizard for sparkrun.

    Walks through cluster creation, SSH mesh, CX7 configuration,
    sudoers entries, and earlyoom installation step by step.

    Auto-detects CX7 peers when running on a DGX Spark.

    \b
    Examples:
      sparkrun setup wizard
      sparkrun setup wizard --hosts 10.0.0.1,10.0.0.2 --cluster mylab
      sparkrun setup wizard --dry-run --hosts 10.0.0.1 --cluster test
      sparkrun setup wizard --yes --hosts 10.0.0.1,10.0.0.2
    """
    import os
    import shutil
    import subprocess

    from sparkrun import __version__
    from sparkrun.core.cluster_manager import ClusterError
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.networking import (
        CX7Topology,
        _group_interfaces_by_port,
        build_host_detection,
        detect_cx7_for_hosts,
        detect_topology,
        discover_cx7_peers,
        generate_cx7_detect_script,
        parse_cx7_detect_output,
        plan_cluster_cx7,
        plan_ring_cx7,
        select_subnets,
        select_subnets_for_topology,
        apply_cx7_plan,
        distribute_cx7_host_keys,
    )
    from sparkrun.orchestration.primitives import build_ssh_kwargs, local_ip_for
    from sparkrun.orchestration.sudo import dispatch_sudo_script, run_with_sudo_fallback, run_sudo_script_on_host
    from sparkrun.scripts import read_script

    from .._common import _get_cluster_manager
    from ._commands import setup_install
    from ._phases import (
        EARLYOOM_PREFER_PATTERNS,
        EARLYOOM_AVOID_PATTERNS,
        _build_earlyoom_regex,
        _DOCKER_GROUP_SCRIPT,
        _DOCKER_GROUP_FALLBACK_SCRIPT,
        _docker_group_summary,
    )
    from ._ssh import _run_ssh_mesh, _detect_and_update_mgmt_ips

    # Manifest tracking
    from sparkrun.core.setup_manifest import ManifestManager

    # Track results for summary
    results = {}
    sudo_password = None
    host_list = []
    cx7_detected_any = False
    cx7_changed_ips = False

    try:
        # ── Phase 0: Welcome + Install Check ─────────────────────────
        click.echo()
        click.echo("Welcome to sparkrun %s setup wizard!" % __version__)
        click.echo("=" * 48)
        click.echo()

        # Default user (may be overridden by interactive prompt below)
        default_user = os.environ.get("USER", "root")
        if user is None:
            user = default_user

        # Auto-install if not installed via uv tool
        uv = shutil.which("uv")
        if uv:
            try:
                check = subprocess.run(
                    [uv, "tool", "list"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if check.returncode == 0 and "sparkrun" not in check.stdout:
                    click.echo("Installing sparkrun as a uv tool...")
                    try:
                        ctx.invoke(setup_install)
                        results["install"] = "OK"
                    except SystemExit:
                        results["install"] = "failed (non-fatal)"
                    click.echo()
            except Exception:
                pass  # uv check failed, skip install
        else:
            logger.debug("uv not found, skipping tool install check")

        # ── Phase 1: Host Discovery + Cluster Creation ───────────────
        click.echo("Phase 1: Cluster Setup")
        click.echo("-" * 30)

        config = SparkrunConfig()
        cluster_mgr = _get_cluster_manager()
        manifest_mgr = ManifestManager(cluster_mgr.clusters_dir)

        # Check for existing clusters
        existing = cluster_mgr.list_clusters()
        default_name = cluster_mgr.get_default() if existing else None
        if existing and not cluster_name and not hosts:
            if default_name and not yes:
                # Default cluster exists — offer to continue with it (low friction)
                default_def = cluster_mgr.get(default_name)
                click.echo(
                    "Default cluster: %s (%d hosts: %s)"
                    % (
                        default_name,
                        len(default_def.hosts),
                        ", ".join(default_def.hosts),
                    )
                )
                use_default = click.confirm("Continue with this cluster?", default=True)
                if use_default:
                    host_list = list(default_def.hosts)
                    cluster_name = default_name
                    results["cluster"] = "%s (%d hosts, default)" % (cluster_name, len(host_list))
                    click.echo()
                # If declined, fall through to show all clusters
                elif len(existing) > 1:
                    click.echo()
                    click.echo("Available clusters:")
                    for i, c in enumerate(existing, 1):
                        marker = " (default)" if c.name == default_name else ""
                        click.echo("  %d. %s (%d hosts)%s" % (i, c.name, len(c.hosts), marker))
                    click.echo()
                    use_other = click.confirm("Use one of these?", default=False)
                    if use_other:
                        choice = click.prompt(
                            "Select cluster",
                            type=click.IntRange(1, len(existing)),
                            default=1,
                        )
                        chosen = existing[choice - 1]
                        host_list = list(chosen.hosts)
                        cluster_name = chosen.name
                        if default_name != cluster_name:
                            cluster_mgr.set_default(cluster_name)
                        results["cluster"] = "%s (%d hosts, set as default)" % (cluster_name, len(host_list))
                        click.echo("Using cluster '%s': %s" % (cluster_name, ", ".join(host_list)))
                        click.echo()
            elif not default_name and not yes:
                # No default, but clusters exist — show list
                click.echo("Existing clusters:")
                for i, c in enumerate(existing, 1):
                    click.echo("  %d. %s (%d hosts)" % (i, c.name, len(c.hosts)))
                click.echo()
                use_existing = click.confirm("Use an existing cluster?", default=False)
                if use_existing:
                    choice = click.prompt(
                        "Select cluster",
                        type=click.IntRange(1, len(existing)),
                        default=1,
                    )
                    chosen = existing[choice - 1]
                    host_list = list(chosen.hosts)
                    cluster_name = chosen.name
                    cluster_mgr.set_default(cluster_name)
                    results["cluster"] = "%s (%d hosts, set as default)" % (cluster_name, len(host_list))
                    click.echo("Using cluster '%s': %s" % (cluster_name, ", ".join(host_list)))
                    click.echo()
            elif yes and default_name:
                # --yes with a default cluster: use it automatically
                default_def = cluster_mgr.get(default_name)
                host_list = list(default_def.hosts)
                cluster_name = default_name
                results["cluster"] = "%s (%d hosts, default)" % (cluster_name, len(host_list))

        # When using an existing cluster, inherit its SSH user if the
        # wizard's --user flag wasn't explicitly provided.
        if cluster_name and user == default_user:
            try:
                _cluster_def = cluster_mgr.get(cluster_name)
                if _cluster_def.user:
                    user = _cluster_def.user
                    click.echo("Using cluster SSH user: %s" % user)
            except Exception:
                pass

        if not host_list:
            # Step 1a: Local CX7 detection
            click.echo("Detecting CX7 interfaces on this machine...")
            local_cx7 = None
            try:
                script = generate_cx7_detect_script()
                det_result = subprocess.run(
                    ["bash", "-s"],
                    input=script,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if det_result.returncode == 0:
                    raw = parse_cx7_detect_output(det_result.stdout)
                    local_cx7 = build_host_detection("localhost", raw)
            except Exception as e:
                logger.debug("Local CX7 detection failed: %s", e)

            local_is_spark = local_cx7 is not None and local_cx7.detected

            if local_is_spark:
                # Step 1b: Peer discovery on CX7 subnets
                click.echo("  CX7 detected! This machine is a DGX Spark.")
                cx7_detected_any = True

                cx7_subnets = [iface.subnet for iface in local_cx7.interfaces if iface.subnet]

                if cx7_subnets and not hosts:
                    click.echo("  Scanning CX7 subnets for peer Sparks...")
                    peers = discover_cx7_peers(cx7_subnets, timeout=20)
                    if peers:
                        click.echo("  Found %d peer(s) on CX7: %s" % (len(peers), ", ".join(peers)))
                        click.echo()
                        click.echo("  Note: These are CX7 addresses. Enter management IPs below")
                        click.echo("  if your hosts use separate management networking.")
                    else:
                        click.echo("  No peers found on CX7 subnets.")
                    click.echo()

                # Suggest local management IP as default
                if not hosts:
                    default_hosts = local_cx7.mgmt_ip or "127.0.0.1"
                    if yes:
                        hosts = default_hosts
                    else:
                        hosts = click.prompt(
                            "Enter host IPs/hostnames (comma-separated)",
                            default=default_hosts,
                        )
            else:
                # Step 1c: No CX7 on control machine
                click.echo("  No CX7 interfaces detected on this machine.")
                click.echo()

                if not hosts:
                    if yes:
                        click.echo(
                            "Error: --hosts required with --yes when no CX7 detected.",
                            err=True,
                        )
                        return
                    hosts = click.prompt("Enter DGX Spark host IPs/hostnames (comma-separated)")

            # Parse host list
            if hosts:
                host_list = [h.strip() for h in hosts.split(",") if h.strip()]

            if not host_list:
                click.echo("Error: No hosts specified.", err=True)
                return

            # Detect CX7 on remote hosts if not already known
            if not local_is_spark and not dry_run:
                click.echo("Detecting CX7 on remote hosts...")
                ssh_kwargs = build_ssh_kwargs(config)
                if user:
                    ssh_kwargs["ssh_user"] = user
                try:
                    remote_detections = detect_cx7_for_hosts(
                        host_list,
                        ssh_kwargs=ssh_kwargs,
                    )
                    cx7_detected_any = any(d.detected for d in remote_detections.values())
                except Exception as e:
                    logger.debug("Remote CX7 detection failed: %s", e)

            # Step 1d: Create cluster
            if not cluster_name:
                if yes:
                    cluster_name = "default"
                else:
                    cluster_name = click.prompt("Cluster name", default="default")

            # Prompt for SSH username (--user flag takes precedence)
            if not yes and user == default_user:
                user = click.prompt("SSH username", default=default_user)

            try:
                cluster_mgr.create(
                    name=cluster_name,
                    hosts=host_list,
                    user=user if user != default_user else None,
                )
                cluster_mgr.set_default(cluster_name)
                results["cluster"] = "%s (%d hosts, set as default)" % (cluster_name, len(host_list))
                click.echo(
                    "Created cluster '%s' with %d host(s), set as default."
                    % (
                        cluster_name,
                        len(host_list),
                    )
                )
            except ClusterError as e:
                if "already exists" in str(e):
                    if yes or click.confirm(
                        "Cluster '%s' already exists. Update it?" % cluster_name,
                        default=True,
                    ):
                        cluster_mgr.update(
                            name=cluster_name,
                            hosts=host_list,
                            user=user if user != default_user else None,
                        )
                        cluster_mgr.set_default(cluster_name)
                        results["cluster"] = "%s (%d hosts, updated)" % (cluster_name, len(host_list))
                        click.echo("Updated cluster '%s'." % cluster_name)
                    else:
                        cluster_name = click.prompt("Enter a different cluster name")
                        cluster_mgr.create(
                            name=cluster_name,
                            hosts=host_list,
                            user=user if user != default_user else None,
                        )
                        cluster_mgr.set_default(cluster_name)
                        results["cluster"] = "%s (%d hosts, set as default)" % (
                            cluster_name,
                            len(host_list),
                        )
                else:
                    raise

        click.echo()

        # Detect CX7 on cluster hosts if not already known (e.g. reusing
        # an existing cluster skips the host-discovery path above).
        if host_list and not cx7_detected_any and len(host_list) >= 2 and not dry_run:
            ssh_kwargs_probe = build_ssh_kwargs(config)
            if user:
                ssh_kwargs_probe["ssh_user"] = user
            try:
                probe = detect_cx7_for_hosts(host_list, ssh_kwargs=ssh_kwargs_probe)
                cx7_detected_any = any(d.detected for d in probe.values())
            except Exception as e:
                logger.debug("CX7 probe on existing cluster failed: %s", e)

        # ── Phase 2: SSH Mesh ────────────────────────────────────────
        if host_list:
            click.echo("Phase 2: SSH Mesh")
            click.echo("-" * 30)

            run_mesh = True
            if not yes:
                run_mesh = click.confirm(
                    "Set up SSH mesh across %d host(s) + this machine?" % len(host_list),
                    default=True,
                )

            if run_mesh:
                try:
                    # Prepare mesh list: cluster hosts + control machine
                    mesh_hosts = list(host_list)
                    seen = set(mesh_hosts)
                    self_ip = local_ip_for(host_list[0]) if host_list else None
                    local_user = os.environ.get("USER", "root")
                    cross_user = user != local_user
                    if self_ip and self_ip in seen and cross_user:
                        # Control machine was explicitly listed — keep it.
                        click.echo(
                            "Note: SSH user '%s' differs from local user '%s'. "
                            "The mesh script will handle cross-user key exchange for %s automatically."
                            % (user, local_user, self_ip)
                        )
                    elif self_ip and self_ip not in seen and not cross_user:
                        mesh_hosts.append(self_ip)
                        seen.add(self_ip)
                    elif self_ip and self_ip not in seen and cross_user:
                        click.echo(
                            "Note: Skipping control machine (%s) in mesh — user '%s' differs from "
                            "local user '%s'. Control→cluster SSH is handled automatically."
                            % (self_ip, user, local_user)
                        )

                    ok = _run_ssh_mesh(
                        mesh_hosts,
                        user,
                        cluster_hosts=host_list,
                        ssh_key=config.ssh_key,
                        discover_ips=(len(host_list) >= 2),
                        dry_run=dry_run,
                        control_is_member=(self_ip is not None and self_ip in seen),
                    )
                    results["ssh"] = "OK" if ok else "failed"
                    if ok and not dry_run and cluster_name:
                        manifest_mgr.record_phase(
                            cluster_name, user, mesh_hosts, "ssh_mesh",
                            mesh_hosts=mesh_hosts, cross_user=cross_user,
                        )
                except Exception as e:
                    results["ssh"] = "failed"
                    click.echo("SSH mesh error: %s" % e, err=True)
                    if not yes and not click.confirm("Continue?", default=True):
                        return
            else:
                results["ssh"] = "skipped"
            click.echo()

        # Build SSH kwargs once for remaining phases
        ssh_kwargs = build_ssh_kwargs(config)
        if user:
            ssh_kwargs["ssh_user"] = user

        # For sudo operations (docker group, sudoers, earlyoom), the SSH user
        # may need to differ from the cluster user.  Start with the cluster
        # user and fall back to the OS user (or a user-specified alternate)
        # if the cluster user lacks sudo access.
        sudo_ssh_kwargs = dict(ssh_kwargs)
        _indirect_sudo_user: str | None = None

        def _run_sudo_on_host(host, script, password, timeout=300):
            """Dispatch to direct or indirect sudo based on _indirect_sudo_user."""
            return dispatch_sudo_script(
                host,
                script,
                password,
                ssh_kwargs=ssh_kwargs if _indirect_sudo_user else sudo_ssh_kwargs,
                indirect_sudo_user=_indirect_sudo_user,
                timeout=timeout,
                dry_run=dry_run,
            )

        # ── Management IP normalization ──────────────────────────────
        # After SSH mesh, detect each host's management IP and update the
        # cluster definition if the user provided CX7 or other non-mgmt IPs.
        if host_list and cluster_name and results.get("ssh") == "OK":
            prev_len = len(host_list)
            _detect_and_update_mgmt_ips(
                host_list,
                cluster_name,
                cluster_mgr,
                ssh_kwargs,
                dry_run=dry_run,
            )
            # Refresh summary if hosts changed (dedup or mgmt IP correction)
            if len(host_list) != prev_len and results.get("cluster"):
                results["cluster"] = "%s (%d hosts, updated)" % (cluster_name, len(host_list))
            click.echo()

        # ── Sudo password helper (deferred collection) ───────────────
        def _ensure_sudo_password():
            nonlocal sudo_password, sudo_ssh_kwargs
            if sudo_password is not None:
                return sudo_password
            if dry_run:
                return None

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
                    test_results.extend(run_remote_scripts_parallel(
                        remote_hosts,
                        "sudo -n true",
                        quiet=True,
                        timeout=10,
                        **sudo_ssh_kwargs,
                    ))
                if all(r.success for r in test_results):
                    return None
            except Exception:
                pass

            # Prompt for sudo password
            sudo_password = click.prompt("[sudo] password for %s" % sudo_user, hide_input=True)

            # Verify sudo works with this password on at least one host
            test_host = host_list[0]
            if should_run_locally(test_host, sudo_user):
                import subprocess as _sp

                _proc = _sp.run(["sudo", "-S", "true"], input=sudo_password + "\n", capture_output=True, text=True, timeout=10)
                test_r = RemoteResult(host=test_host, returncode=_proc.returncode, stdout=_proc.stdout, stderr=_proc.stderr)
            else:
                test_r = run_sudo_script_on_host(
                    test_host,
                    "true",
                    sudo_password,
                    ssh_kwargs=sudo_ssh_kwargs,
                    timeout=10,
                )
            if not test_r.success:
                # Sudo failed for cluster user — offer alternate user
                alt_default = default_user if sudo_user != default_user else ""
                click.echo("  Sudo failed for '%s'. Specify a user with sudo access." % sudo_user)
                alt_user = click.prompt("Sudo user", default=alt_default)
                sudo_password = click.prompt("[sudo] password for %s" % alt_user, hide_input=True)

                # Use indirect sudo: SSH as cluster user, su to alt_user
                nonlocal _indirect_sudo_user
                _indirect_sudo_user = alt_user

            return sudo_password

        # ── Phase 3: CX7 Configuration ───────────────────────────────
        if cx7_detected_any and len(host_list) >= 2:
            click.echo("Phase 3: CX7 Network Configuration")
            click.echo("-" * 30)
            click.echo("Configures high-speed CX7 networking between hosts.")

            run_cx7 = yes or click.confirm("Configure CX7 networking?", default=True)

            if run_cx7:
                try:
                    detections = detect_cx7_for_hosts(
                        host_list,
                        ssh_kwargs=ssh_kwargs,
                        dry_run=dry_run,
                    )

                    # Determine topology
                    hosts_with_cx7 = {h: d for h, d in detections.items() if d.detected}
                    n_hosts = len(hosts_with_cx7)
                    has_2_ports = all(
                        len(_group_interfaces_by_port(d.interfaces)) >= 2
                        for d in hosts_with_cx7.values()
                    )
                    ring_eligible = n_hosts == 3 and has_2_ports

                    effective_topology = CX7Topology.SWITCH
                    topology_result = None

                    if ring_eligible and not yes:
                        topo_choice = click.prompt(
                            "  Topology (3 hosts with 2 ports detected)",
                            type=click.Choice(["auto", "switch", "ring"]),
                            default="auto",
                        )
                        if topo_choice == "ring":
                            effective_topology = CX7Topology.RING
                        elif topo_choice == "auto":
                            click.echo("  Detecting topology via neighbor discovery...")
                            topology_result = detect_topology(
                                detections, host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run,
                                sudo_password=_ensure_sudo_password(),
                            )
                            effective_topology = topology_result.topology
                            click.echo("  Detected topology: %s" % effective_topology.value)
                    elif ring_eligible and yes:
                        # --yes with ring-eligible: auto-detect
                        click.echo("  Detecting topology via neighbor discovery...")
                        topology_result = detect_topology(
                            detections, host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run,
                        )
                        effective_topology = topology_result.topology
                        click.echo("  Detected topology: %s" % effective_topology.value)

                    # For explicit ring without detection data, run detection
                    if effective_topology == CX7Topology.RING and topology_result is None:
                        if not dry_run:
                            click.echo("  Running topology detection for ring...")
                            topology_result = detect_topology(
                                detections, host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run,
                                sudo_password=_ensure_sudo_password(),
                            )
                        else:
                            from sparkrun.orchestration.networking import CX7TopologyResult
                            topology_result = CX7TopologyResult(topology=CX7Topology.RING)

                    click.echo("  Topology: %s" % effective_topology.value)

                    # Select subnets and plan based on topology
                    if effective_topology == CX7Topology.RING:
                        all_subnets = select_subnets_for_topology(detections, effective_topology)
                        click.echo("  Subnets: %s" % ", ".join(str(s) for s in all_subnets))
                        plan = plan_ring_cx7(detections, topology_result, all_subnets)
                    else:
                        s1, s2 = select_subnets(detections)
                        all_subnets = [s1, s2]
                        click.echo("  Subnets: %s, %s" % (s1, s2))
                        plan = plan_cluster_cx7(detections, s1, s2)

                    # Check for plan-level errors (e.g. insufficient ports)
                    if plan.errors and not plan.host_plans:
                        for e in plan.errors:
                            click.echo("  Error: %s" % e, err=True)
                        results["cx7"] = "failed"
                    else:
                        needs_config = sum(1 for hp in plan.host_plans if hp.needs_change)

                        if plan.all_valid:
                            click.echo("  All hosts already configured.")
                            results["cx7"] = "configured (%s)" % effective_topology.value
                        elif dry_run:
                            click.echo("  [dry-run] Would configure %d host(s)." % needs_config)
                            results["cx7"] = "dry-run"
                        else:
                            pw = _ensure_sudo_password()
                            sudo_hosts = {hp.host for hp in plan.host_plans if hp.needs_change}
                            apply_results = apply_cx7_plan(
                                plan,
                                ssh_kwargs=ssh_kwargs,
                                dry_run=dry_run,
                                sudo_password=pw,
                                sudo_hosts=sudo_hosts if pw else set(),
                            )
                            ok_count = sum(1 for r in apply_results if r.success)

                            # Distribute host keys for CX7 IPs
                            all_cx7_ips = [a.ip for hp in plan.host_plans for a in hp.assignments if a.ip]
                            if all_cx7_ips:
                                distribute_cx7_host_keys(
                                    all_cx7_ips,
                                    host_list,
                                    ssh_kwargs=ssh_kwargs,
                                    dry_run=dry_run,
                                )

                            results["cx7"] = "configured (%s)" % effective_topology.value if ok_count else "failed"
                            if ok_count:
                                cx7_changed_ips = True
                            if ok_count and not dry_run and cluster_name:
                                manifest_mgr.record_phase(
                                    cluster_name, user, host_list, "cx7",
                                    subnets=[str(s) for s in all_subnets],
                                    cx7_ips=all_cx7_ips,
                                    netplan_file="/etc/netplan/40-cx7.yaml",
                                    topology=effective_topology.value,
                                )

                    # Save topology to cluster definition
                    if cluster_name and effective_topology != CX7Topology.UNKNOWN:
                        try:
                            cluster_mgr.update(cluster_name, topology=effective_topology.value)
                        except Exception:
                            pass

                except Exception as e:
                    results["cx7"] = "failed"
                    click.echo("CX7 error: %s" % e, err=True)
                    if not yes and not click.confirm("Continue?", default=True):
                        return
            else:
                results["cx7"] = "skipped"
            click.echo()

        # ── Phase 3b: Re-run SSH mesh after CX7 changes ─────────────
        # CX7 configuration may add new IPs that need to be in the SSH
        # mesh.  Re-run the mesh (discover-ips phase only) so inter-node
        # SSH works over the newly configured CX7 interfaces.
        if cx7_changed_ips and host_list and len(host_list) >= 2 and results.get("ssh") == "OK":
            click.echo("Phase 3b: Re-meshing SSH after CX7 IP changes")
            click.echo("-" * 30)
            click.echo("CX7 configuration changed network IPs. Re-running SSH mesh")
            click.echo("to ensure full connectivity across all interfaces.")
            click.echo()

            try:
                mesh_hosts = list(host_list)
                seen_mesh = set(mesh_hosts)
                self_ip = local_ip_for(host_list[0]) if host_list else None
                local_user_remesh = os.environ.get("USER", "root")
                cross_user_remesh = user != local_user_remesh
                if self_ip and self_ip not in seen_mesh and not cross_user_remesh:
                    mesh_hosts.append(self_ip)
                    seen_mesh.add(self_ip)

                ok = _run_ssh_mesh(
                    mesh_hosts,
                    user,
                    cluster_hosts=host_list,
                    ssh_key=config.ssh_key,
                    discover_ips=True,
                    dry_run=dry_run,
                    control_is_member=(self_ip is not None and self_ip in seen_mesh),
                )
                results["ssh_remesh"] = "OK" if ok else "failed"
                if ok and not dry_run and cluster_name:
                    manifest_mgr.record_phase(
                        cluster_name, user, mesh_hosts, "ssh_mesh_post_cx7",
                        mesh_hosts=mesh_hosts, cross_user=cross_user_remesh,
                    )
            except Exception as e:
                results["ssh_remesh"] = "failed"
                click.echo("SSH re-mesh error: %s" % e, err=True)
                if not yes and not click.confirm("Continue?", default=True):
                    return
            click.echo()

        # ── Phase 4: Docker Group ────────────────────────────────────
        if host_list:
            click.echo("Phase 4: Docker Group Membership")
            click.echo("-" * 30)
            click.echo("Ensures user can run Docker commands without sudo.")

            run_docker = yes or click.confirm(
                "Add '%s' to the docker group on all hosts?" % user,
                default=True,
            )

            if run_docker:
                try:
                    dg_script = _DOCKER_GROUP_SCRIPT.format(user=user)
                    dg_fallback = _DOCKER_GROUP_FALLBACK_SCRIPT.format(user=user)

                    if dry_run:
                        click.echo("  [dry-run] Would ensure docker group on %d host(s)." % len(host_list))
                        results["docker"] = "dry-run"
                    else:
                        pw = _ensure_sudo_password()
                        if _indirect_sudo_user:
                            # Indirect sudo: SSH as cluster user, su to sudo user
                            dg_result_map = {}
                            for h in host_list:
                                r = _run_sudo_on_host(h, dg_fallback, pw, timeout=30)
                                dg_result_map[h] = r
                        else:
                            dg_result_map, dg_still_failed = run_with_sudo_fallback(
                                host_list,
                                dg_script,
                                dg_fallback,
                                sudo_ssh_kwargs,
                                dry_run=dry_run,
                            )
                            if dg_still_failed:
                                if pw:
                                    for h in dg_still_failed:
                                        r = _run_sudo_on_host(h, dg_fallback, pw, timeout=30)
                                        dg_result_map[h] = r

                        dg_ok = sum(1 for h in host_list if dg_result_map.get(h) and dg_result_map[h].success)
                        results["docker"] = "OK (%d/%d)" % (dg_ok, len(host_list)) if dg_ok else "failed"
                        if dg_ok and not dry_run and cluster_name:
                            manifest_mgr.record_phase(cluster_name, user, host_list, "docker_group")
                        for h in host_list:
                            r = dg_result_map.get(h)
                            if r and r.success:
                                click.echo("  %s: %s" % (h, _docker_group_summary(r.stdout, user=user)))
                except Exception as e:
                    results["docker"] = "failed"
                    click.echo("Docker group error: %s" % e, err=True)
                    if not yes and not click.confirm("Continue?", default=True):
                        return
            else:
                results["docker"] = "skipped"
            click.echo()

        # ── Phase 5: Sudoers Entries ─────────────────────────────────
        if host_list:
            click.echo("Phase 5: Sudoers Entries")
            click.echo("-" * 30)
            click.echo("Scoped sudoers for fix-permissions + clear-cache (no broad sudo).")

            run_sudoers = yes or click.confirm("Install sudoers entries?", default=True)

            if run_sudoers:
                try:
                    pw = _ensure_sudo_password()

                    from sparkrun.utils.shell import validate_unix_username

                    validate_unix_username(user)
                    sudoers_scripts = [
                        (
                            "fix-permissions",
                            read_script("fix_permissions_sudoers.sh").format(
                                user=user,
                                cache_dir="",
                            ),
                        ),
                        (
                            "clear-cache",
                            read_script("clear_cache_sudoers.sh").format(
                                user=user,
                            ),
                        ),
                    ]

                    if dry_run:
                        for label, _ in sudoers_scripts:
                            click.echo(
                                "  [dry-run] Would install %s sudoers on %d host(s)."
                                % (
                                    label,
                                    len(host_list),
                                )
                            )
                        results["sudoers"] = "dry-run"
                    else:
                        failed_any = False
                        for label, script in sudoers_scripts:
                            label_ok = 0
                            for h in host_list:
                                r = _run_sudo_on_host(h, script, pw or "", timeout=300)
                                if r.success:
                                    label_ok += 1
                                else:
                                    logger.debug(
                                        "Sudoers %s failed on %s: %s",
                                        label,
                                        h,
                                        r.stderr[:100],
                                    )
                            if label_ok < len(host_list):
                                failed_any = True
                            click.echo("  %s: %d/%d host(s)" % (label, label_ok, len(host_list)))
                        results["sudoers"] = "installed (fix-permissions, clear-cache)" if not failed_any else "partial"
                        if not dry_run and cluster_name:
                            manifest_mgr.record_phase(
                                cluster_name, user, host_list, "sudoers",
                                files=[
                                    "/etc/sudoers.d/sparkrun-chown-%s" % user,
                                    "/etc/sudoers.d/sparkrun-dropcaches-%s" % user,
                                ],
                            )
                except Exception as e:
                    results["sudoers"] = "failed"
                    click.echo("Sudoers error: %s" % e, err=True)
                    if not yes and not click.confirm("Continue?", default=True):
                        return
            else:
                results["sudoers"] = "skipped"
            click.echo()

        # ── Phase 6: earlyoom ────────────────────────────────────────
        if host_list:
            click.echo("Phase 6: earlyoom OOM Protection")
            click.echo("-" * 30)
            click.echo("Prevents system hangs by proactively managing memory pressure.")

            run_earlyoom = yes or click.confirm("Install earlyoom?", default=True)

            if run_earlyoom:
                try:
                    prefer_regex = _build_earlyoom_regex(EARLYOOM_PREFER_PATTERNS)
                    avoid_regex = _build_earlyoom_regex(EARLYOOM_AVOID_PATTERNS)

                    install_script = read_script("earlyoom_install.sh").format(
                        prefer=prefer_regex,
                        avoid=avoid_regex,
                    )
                    fallback_script = read_script("earlyoom_install_fallback.sh").format(
                        prefer=prefer_regex,
                        avoid=avoid_regex,
                    )

                    pw = _ensure_sudo_password()
                    if _indirect_sudo_user:
                        result_map = {}
                        still_failed = []
                        for h in host_list:
                            r = _run_sudo_on_host(h, fallback_script, pw, timeout=300)
                            result_map[h] = r
                            if not r.success:
                                still_failed.append(h)
                    else:
                        result_map, still_failed = run_with_sudo_fallback(
                            host_list,
                            install_script,
                            fallback_script,
                            sudo_ssh_kwargs,
                            dry_run=dry_run,
                            sudo_password=pw,
                        )
                        if still_failed and pw and not dry_run:
                            for h in still_failed:
                                r = _run_sudo_on_host(h, fallback_script, pw, timeout=300)
                                result_map[h] = r

                    ok_count = sum(1 for h in host_list if result_map.get(h) and result_map[h].success)
                    if dry_run:
                        results["earlyoom"] = "dry-run"
                    else:
                        results["earlyoom"] = "installed" if ok_count > 0 else "failed"
                        click.echo("  earlyoom configured on %d/%d host(s)." % (ok_count, len(host_list)))
                        if ok_count and cluster_name:
                            installed_pkg = any(
                                "INSTALLING:" in (result_map.get(h) and result_map[h].stdout or "")
                                for h in host_list
                            )
                            manifest_mgr.record_phase(
                                cluster_name, user, host_list, "earlyoom",
                                installed_package=installed_pkg,
                            )
                except Exception as e:
                    results["earlyoom"] = "failed"
                    click.echo("earlyoom error: %s" % e, err=True)
                    if not yes and not click.confirm("Continue?", default=True):
                        return
            else:
                results["earlyoom"] = "skipped"
            click.echo()

    except (KeyboardInterrupt, click.Abort):
        click.echo()
        click.echo()
        click.echo("Setup interrupted.")
        if results:
            click.echo("Completed before interruption:")
            for key, val in results.items():
                click.echo("  %-10s %s" % (key + ":", val))
        click.echo()
        click.echo("You can re-run 'sparkrun setup wizard' to resume, or configure")
        click.echo("individual steps via 'sparkrun setup <command>'.")
        click.echo("For manual setup instructions, see: https://sparkrun.dev")
        click.echo()
        return

    # ── Phase 7: Summary ─────────────────────────────────────────
    click.echo()
    click.echo("Setup Complete!")
    click.echo("=" * 48)
    click.echo()
    if results.get("cluster"):
        click.echo("  Cluster:    %s" % results["cluster"])
    if results.get("ssh"):
        click.echo("  SSH mesh:   %s" % results["ssh"])
    if results.get("cx7"):
        click.echo("  CX7:        %s" % results["cx7"])
    if results.get("ssh_remesh"):
        click.echo("  SSH remesh: %s" % results["ssh_remesh"])
    if results.get("docker"):
        click.echo("  Docker:     %s" % results["docker"])
    if results.get("sudoers"):
        click.echo("  Sudoers:    %s" % results["sudoers"])
    if results.get("earlyoom"):
        click.echo("  earlyoom:   %s" % results["earlyoom"])
    click.echo()
    click.echo("Next steps:")
    click.echo("  sparkrun list                            # Browse available recipes")
    click.echo("  sparkrun show qwen3-1.7b-vllm           # Recipe details + VRAM estimate")
    click.echo("  sparkrun run qwen3-1.7b-vllm --dry-run  # Preview a launch")
    click.echo("  sparkrun run qwen3-1.7b-vllm            # Launch inference")
    click.echo()
