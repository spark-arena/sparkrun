"""Guided setup wizard for sparkrun."""

from __future__ import annotations

from sparkrun.core.application_profile import render_identity_text


import logging

import click
from sparkrun.core.features import feature_gate_enabled

logger = logging.getLogger(__name__)


@click.command("wizard", hidden=not feature_gate_enabled("cli.setup.wizard"))
@click.option("--hosts", "-H", default=None, help="Pre-populate host list (comma-separated)")
@click.option("--cluster", "cluster_name", default=None, help="Cluster name")
@click.option("--user", "-u", default=None, help="SSH username")
@click.option("--dry-run", "-n", is_flag=True, help="Preview without executing")
@click.option("--yes", "-y", is_flag=True, help="Accept all defaults (non-interactive)")
@click.pass_context
def setup_wizard(ctx, hosts, cluster_name, user, dry_run, yes):
    """Guided setup wizard for {app_command}.

    Selects host setup steps from application policy, detected hardware, and
    the cluster executor. Checks prerequisites before offering changes.

    Auto-detects CX7 peers when running on a DGX Spark.

    \b
    Examples:
      {app_command} setup wizard
      {app_command} setup wizard --hosts 10.0.0.1,10.0.0.2 --cluster mylab
      {app_command} setup wizard --dry-run --hosts 10.0.0.1 --cluster test
      {app_command} setup wizard --yes --hosts 10.0.0.1,10.0.0.2
    """
    from sparkrun.core.features import is_feature_enabled
    from sparkrun.core.config import SparkrunConfig

    if not is_feature_enabled("cli.setup.wizard", config=SparkrunConfig()):
        raise click.ClickException("Setup wizard is disabled; use setup check or enable cli.setup.wizard")

    import subprocess

    from sparkrun.core.version import display_version
    from sparkrun.core.cluster_manager import ClusterError
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.networking import (
        CX7_NETPLAN_FILE,
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
        distribute_host_keys,
    )
    from sparkrun.orchestration.primitives import build_ssh_kwargs
    from sparkrun.orchestration.sudo import dispatch_sudo_script, run_sudo_script_on_host
    from sparkrun.utils.net import local_ip_for

    from .._common import _get_cluster_manager
    from ._ssh import (
        _default_ssh_user,
        _detect_and_update_mgmt_ips,
        _ensure_ssh_access,
        _report_host_key_distribution,
        _run_ssh_mesh,
    )

    # Manifest tracking
    from sparkrun.core.setup_manifest import ManifestManager

    # Track results for summary
    results = {}
    wizard_run_kind = "initial_setup"
    sudo_password = None
    host_list = []
    cx7_detected_any = False
    cx7_changed_ips = False

    from contextlib import ExitStack

    recordings = ExitStack()
    locked_clusters = set()
    try:
        # ── Phase 0: Welcome + Install Check ─────────────────────────
        click.echo()
        click.echo(render_identity_text("Welcome to {app_command} %s setup wizard!" % display_version(SparkrunConfig())))
        click.echo("=" * 48)
        click.echo()

        # Default user (may be overridden by interactive prompt below).
        # Never fabricate "root" — see _default_ssh_user.
        default_user = _default_ssh_user()
        if user is None:
            user = default_user

        click.echo(render_identity_text("Controller installation and completion: {app_command} setup install"))
        click.echo()

        # ── Phase 1: Host Discovery + Cluster Creation ───────────────
        click.echo("Phase 1: Cluster Setup")
        click.echo("-" * 30)

        config = SparkrunConfig()
        cluster_mgr = _get_cluster_manager()
        manifest_mgr = None if dry_run else ManifestManager(cluster_mgr.clusters_dir)
        from sparkrun.core.setup_steps import step_enabled, build_setup_plan
        from sparkrun.core.setup_probe import probe_setup_hosts, resolve_setup_context
        from sparkrun.core.setup_models import HostState
        from sparkrun.core.setup_actions import SetupActionContext
        from ._step_runner import run_host_steps

        cx7_enabled = step_enabled("cx7", config)

        # Control→host SSH access is a precondition for every probe the wizard
        # runs (CX7 detection, shared-cache detection, the mesh itself).  Gate
        # on it once, as the *first* network action and only after the SSH
        # username is known — otherwise a cluster that has never seen our key
        # produces a wall of "Permission denied" from each probe in turn, with
        # the username prompt arriving after the damage is done.
        ssh_access_hosts: list[str] | None = None

        def _record_cluster():
            if manifest_mgr is not None and cluster_name and cluster_name not in locked_clusters:
                # A renamed cluster replaces the earlier selection; never hold
                # multiple cluster locks while prompting for a new name.
                recordings.close()
                locked_clusters.clear()
                try:
                    recordings.enter_context(manifest_mgr.recording(cluster_name))
                except (ValueError, OSError) as exc:
                    raise click.ClickException(str(exc)) from exc
                locked_clusters.add(cluster_name)

        def _gate_ssh_access(current_user):
            """Run the access gate once; return the (possibly corrected) user."""
            nonlocal ssh_access_hosts
            _record_cluster()
            if ssh_access_hosts is not None or not host_list:
                return current_user
            outcome = _ensure_ssh_access(host_list, current_user, config, dry_run=dry_run, yes=yes)
            ssh_access_hosts = outcome.ok_hosts
            if outcome.blocked:
                results["ssh_access"] = "%d/%d hosts" % (len(outcome.ok_hosts), len(host_list))
            elif outcome.bootstrapped:
                results["ssh_access"] = "OK (key installed)"
            else:
                results["ssh_access"] = "OK"
            click.echo()
            return outcome.user

        # Check for existing clusters
        existing = cluster_mgr.list_clusters()
        default_name = cluster_mgr.get_default() if existing else None
        wizard_run_kind = "config_revision" if existing or default_name else "initial_setup"
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
                            if not dry_run:
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
                    if not dry_run:
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

        elif cluster_name and not host_list and not hosts:
            # --cluster was explicitly named; if it already exists, offer to reuse it
            try:
                named_def = cluster_mgr.get(cluster_name)
                if yes:
                    host_list = list(named_def.hosts)
                    results["cluster"] = "%s (%d hosts)" % (cluster_name, len(host_list))
                else:
                    click.echo("Cluster: %s (%d hosts: %s)" % (cluster_name, len(named_def.hosts), ", ".join(named_def.hosts)))
                    if click.confirm("Continue with this cluster?", default=True):
                        host_list = list(named_def.hosts)
                        results["cluster"] = "%s (%d hosts)" % (cluster_name, len(host_list))
                        click.echo()
                    # If user declines, fall through to host-discovery + creation/update flow
            except ClusterError:
                pass  # Cluster doesn't exist yet; fall through to normal creation flow

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
            # Identify the control host before offering hardware-specific discovery.
            if cx7_enabled and not dry_run:
                from sparkrun.core.setup_probe import local_setup_step_supported

                cx7_enabled = local_setup_step_supported("cx7", config)
            # Step 1a: Local CX7 detection
            local_cx7 = None
            if cx7_enabled and not dry_run:
                click.echo("Detecting CX7 interfaces on this machine...")
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

            if local_cx7 is not None and local_is_spark:
                # Step 1b: Peer discovery on CX7 subnets
                click.echo("  CX7 interfaces detected on this machine.")
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
                if cx7_enabled and not dry_run:
                    click.echo("  No CX7 interfaces detected on this machine.")
                click.echo()

                if not hosts:
                    if yes:
                        click.echo(
                            "Error: --hosts required with --yes when no existing cluster or discovered hosts are available.",
                            err=True,
                        )
                        return
                    hosts = click.prompt("Enter host IPs/hostnames (comma-separated)")

            # Parse host list
            if hosts:
                host_list = [h.strip() for h in hosts.split(",") if h.strip()]

            if not host_list:
                click.echo("Error: No hosts specified.", err=True)
                return

            # Step 1d: Name the cluster and settle on an SSH user.  Both
            # prompts come *before* any SSH so the first connection is made
            # with the credentials the operator actually intends to use.
            if not cluster_name:
                if yes:
                    cluster_name = "default"
                else:
                    cluster_name = click.prompt("Cluster name", default="default")

            # Prompt for SSH username (--user flag takes precedence)
            if not yes and user == default_user:
                user = click.prompt("SSH username", default=default_user or None)

            click.echo()

            # Step 1e: SSH access gate — the first network action of the run.
            user = _gate_ssh_access(user)

            # Detect a shared/NFS HF cache and offer to enable the matching
            # distribution preferences (skip the redundant per-host rsync,
            # drop perm preservation that fails under root_squash).
            dist_cfg = None
            if ssh_access_hosts and not dry_run:
                from sparkrun.core.cluster_manager import (
                    ClusterDistributionConfig,
                    ModelDistributionPrefs,
                )
                from sparkrun.models.distribute import detect_shared_cache

                ssh_kwargs_cache = build_ssh_kwargs(config)
                if user and user != default_user:
                    ssh_kwargs_cache["ssh_user"] = user
                try:
                    shared = detect_shared_cache(None, ssh_access_hosts, **ssh_kwargs_cache)
                except Exception as e:
                    logger.debug("Shared-cache detection failed: %s", e)
                    shared = False

                if shared:
                    click.echo("Detected a shared/NFS HuggingFace cache across the cluster hosts.")
                    skip_fan_out = yes or click.confirm(
                        "Skip per-host model sync (the cache is already visible everywhere)?",
                        default=True,
                    )
                    disable_perms = yes or click.confirm(
                        "Disable file-permission preservation during sync (recommended on NFS)?",
                        default=True,
                    )
                    dist_cfg = ClusterDistributionConfig(
                        model=ModelDistributionPrefs(
                            preserve_perms=not disable_perms,
                            skip_fan_out=bool(skip_fan_out),
                        )
                    )

            from sparkrun.core.scheduler import NEW_CLUSTER_DEFAULT_SCHEDULER, new_cluster_scheduler_notice

            if dry_run:
                results["cluster"] = "%s (%d hosts, preview only)" % (cluster_name, len(host_list))
                click.echo("[dry-run] Would create/update cluster '%s' and set it as default." % cluster_name)
            else:
                try:
                    cluster_mgr.create(
                        name=cluster_name,
                        hosts=host_list,
                        user=user if user != default_user else None,
                        distribution=dist_cfg,
                        scheduler=NEW_CLUSTER_DEFAULT_SCHEDULER,
                    )
                    if not dry_run:
                        cluster_mgr.set_default(cluster_name)
                    results["cluster"] = "%s (%d hosts, set as default)" % (cluster_name, len(host_list))
                    click.echo(
                        "Created cluster '%s' with %d host(s), set as default."
                        % (
                            cluster_name,
                            len(host_list),
                        )
                    )
                    click.echo(new_cluster_scheduler_notice())
                except ClusterError as e:
                    if "already exists" in str(e):
                        if yes or click.confirm(
                            "Cluster '%s' already exists. Update it?" % cluster_name,
                            default=True,
                        ):
                            # Only override distribution prefs when we actually
                            # detected a shared cache; otherwise leave the existing
                            # cluster's prefs untouched (update() defaults to _UNSET).
                            _update_kw = {}
                            if dist_cfg is not None:
                                _update_kw["distribution"] = dist_cfg
                            cluster_mgr.update(
                                name=cluster_name,
                                hosts=host_list,
                                user=user if user != default_user else None,
                                **_update_kw,
                            )
                            if not dry_run:
                                cluster_mgr.set_default(cluster_name)
                            results["cluster"] = "%s (%d hosts, updated)" % (cluster_name, len(host_list))
                            click.echo("Updated cluster '%s'." % cluster_name)
                        else:
                            cluster_name = click.prompt("Enter a different cluster name")
                            _record_cluster()
                            cluster_mgr.create(
                                name=cluster_name,
                                hosts=host_list,
                                user=user if user != default_user else None,
                                distribution=dist_cfg,
                                scheduler=NEW_CLUSTER_DEFAULT_SCHEDULER,
                            )
                            if not dry_run:
                                cluster_mgr.set_default(cluster_name)
                            click.echo(new_cluster_scheduler_notice())
                            results["cluster"] = "%s (%d hosts, set as default)" % (
                                cluster_name,
                                len(host_list),
                            )
                    else:
                        raise

        click.echo()

        # Reusing an existing cluster skips the discovery path above, so the
        # access gate has not run yet — run it here before probing.
        user = _gate_ssh_access(user)

        ssh_kwargs = build_ssh_kwargs(config)
        if user:
            ssh_kwargs["ssh_user"] = user

        def _mesh_members():
            """The wizard's SSH mesh: cluster hosts plus, when it can join, this machine.

            The control machine joins unless the SSH user differs from the
            local one (cross-user key exchange is handled separately); an
            explicitly listed control machine stays either way.
            """
            members = list(host_list)
            self_ip = local_ip_for(host_list[0]) if host_list else None
            cross_user = user != _default_ssh_user()
            if self_ip and self_ip not in members and not cross_user:
                members.append(self_ip)
            return members, self_ip, cross_user

        def _refresh_plan():
            nonlocal states, setup_context
            try:
                cluster_def = cluster_mgr.get(cluster_name) if cluster_name else None
            except ClusterError:
                cluster_def = None
            if dry_run:
                inventory = cluster_def.hosts_hardware if cluster_def else {}
                states = {h: HostState(h, hardware=inventory.get(h)) for h in host_list}
                setup_context = resolve_setup_context(states, config=config, cluster=cluster_def, cluster_name=cluster_name)
            else:
                # Never send setup probes to hosts that failed the SSH gate.
                targets = [h for h in host_list if h in (ssh_access_hosts or [])]
                # The control machine is probed as a mesh peer too, so an
                # "already configured" mesh covers every leg the wizard builds.
                states, setup_context = probe_setup_hosts(
                    targets,
                    ssh_kwargs=ssh_kwargs,
                    config=config,
                    cluster=cluster_def,
                    cluster_name=cluster_name,
                    extra_mesh_peers=[h for h in _mesh_members()[0] if h not in host_list],
                )
                for h in host_list:
                    if h not in states:
                        states[h] = HostState(h, reachable=False, error="SSH access was not established")
                setup_context.multi_host = len(host_list) > 1
                inventory = dict(cluster_def.hosts_hardware) if cluster_def else {}
                inventory.update({h: state.hardware for h, state in states.items() if state.hardware is not None})
                if inventory and cluster_def:
                    cluster_mgr.update(cluster_name, hosts_hardware=inventory)
            for host, state in states.items():
                label = ", ".join(a.model for a in state.hardware.accelerators) if state.hardware else "hardware identification required"
                click.echo("  %s: %s" % (host, label))

        states = {}
        setup_context = None
        _refresh_plan()

        def _plan_entry(key, host):
            if setup_context is None:
                raise RuntimeError("Setup plan has not been probed")
            return next(p for p in build_setup_plan(states[host], setup_context) if p.step.key == key)

        def _selected(key, host):
            entry = _plan_entry(key, host)
            return entry.selected and not entry.blocked_by

        def _satisfied(key):
            """Every host passes every check of *key* — the same verdict ``setup check`` renders.

            Unknown (unprobed, dry-run, SKIP) is not satisfied: a false
            "already configured" would skip real work, a false "needs setup"
            only costs a prompt.
            """
            return bool(host_list) and all(_plan_entry(key, h).satisfied for h in host_list)

        mesh_ready = False

        cx7_detected_any = any(state.cx7 and state.cx7.detected for state in states.values())

        # ── Phase 2: SSH Mesh ────────────────────────────────────────
        if len(host_list) > 1 and all(_selected("ssh_mesh", h) for h in host_list):
            click.echo("Phase 2: SSH Mesh")
            click.echo("-" * 30)

            run_mesh = True
            if _satisfied("ssh_mesh"):
                members = _mesh_members()[0]
                click.echo(
                    "  SSH mesh already working across %d host(s)%s — skipping."
                    % (len(host_list), " + this machine" if len(members) > len(host_list) else "")
                )
                click.echo(render_identity_text("  (Run '{app_command} setup ssh' to re-key the mesh deliberately.)"))
                results["ssh"] = "already configured"
                mesh_ready = True
                run_mesh = False
            elif not yes:
                run_mesh = click.confirm(
                    "Set up SSH mesh across %d host(s) + this machine?" % len(host_list),
                    default=True,
                )

            if run_mesh:
                try:
                    # Prepare mesh list: cluster hosts + control machine
                    mesh_hosts, self_ip, cross_user = _mesh_members()
                    local_user = _default_ssh_user()
                    if self_ip and self_ip in host_list and cross_user:
                        # Control machine was explicitly listed — keep it.
                        click.echo(
                            "Note: SSH user '%s' differs from local user '%s'. "
                            "The mesh script will handle cross-user key exchange for %s automatically." % (user, local_user, self_ip)
                        )
                    elif self_ip and self_ip not in mesh_hosts:
                        click.echo(
                            "Note: Skipping control machine (%s) in mesh — user '%s' differs from "
                            "local user '%s'. Control→cluster SSH is handled automatically." % (self_ip, user, local_user)
                        )

                    ok = _run_ssh_mesh(
                        mesh_hosts,
                        user,
                        cluster_hosts=host_list,
                        ssh_key=config.ssh_key,
                        discover_ips=(len(host_list) >= 2),
                        dry_run=dry_run,
                        control_is_member=(self_ip is not None and self_ip in mesh_hosts),
                    )
                    mesh_ready = bool(ok)
                    results["ssh"] = "OK" if ok else "failed"
                    if ok and not dry_run and cluster_name:
                        assert manifest_mgr is not None
                        manifest_mgr.record_phase(
                            cluster_name,
                            user,
                            mesh_hosts,
                            "ssh_mesh",
                            mesh_hosts=mesh_hosts,
                            cross_user=cross_user,
                        )
                except Exception as e:
                    results["ssh"] = "failed"
                    click.echo("SSH mesh error: %s" % e, err=True)
                    if not yes and not click.confirm("Continue?", default=True):
                        return
            elif not mesh_ready:
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
        if host_list and cluster_name and mesh_ready:
            previous_hosts = list(host_list)
            prev_len = len(host_list)
            _detect_and_update_mgmt_ips(
                host_list,
                cluster_name,
                cluster_mgr,
                ssh_kwargs,
                dry_run=dry_run,
            )
            if previous_hosts != host_list:
                ssh_access_hosts = list(host_list)
                _refresh_plan()
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
                try:
                    from sparkrun.utils.shell import validate_unix_username

                    validate_unix_username(alt_user)
                except ValueError as exc:
                    raise click.BadParameter(str(exc), param_hint="'Sudo user'") from exc
                sudo_password = click.prompt("[sudo] password for %s" % alt_user, hide_input=True)

                # Use indirect sudo: SSH as cluster user, su to alt_user
                nonlocal _indirect_sudo_user
                _indirect_sudo_user = alt_user

            return sudo_password

        # ── Phase 3: CX7 Configuration ───────────────────────────────
        if (
            cx7_detected_any
            and len(host_list) >= 2
            and all(
                _selected("cx7", h) and states[h].cx7 and states[h].cx7.detected and states[h].facts.get("CHECK_NETPLAN") == "1"
                for h in host_list
            )
        ):
            click.echo("Phase 3: CX7 Network Configuration")
            click.echo("-" * 30)
            click.echo("Configures high-speed CX7 networking between hosts.")

            cx7_done = _satisfied("cx7")
            if cx7_done:
                click.echo("  CX7 already configured and persisted on all host(s) — skipping.")
                click.echo(render_identity_text("  (Run '{app_command} setup cx7' to re-plan addresses deliberately.)"))
                results["cx7"] = "already configured"
            run_cx7 = not cx7_done and (yes or click.confirm("Configure CX7 networking?", default=True))

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
                    has_2_ports = all(len(_group_interfaces_by_port(d.interfaces)) >= 2 for d in hosts_with_cx7.values())
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
                                detections,
                                host_list,
                                ssh_kwargs=ssh_kwargs,
                                dry_run=dry_run,
                            )
                            effective_topology = topology_result.topology
                            click.echo("  Detected topology: %s" % effective_topology.value)
                    elif ring_eligible and yes:
                        # --yes with ring-eligible: auto-detect
                        click.echo("  Detecting topology via neighbor discovery...")
                        topology_result = detect_topology(
                            detections,
                            host_list,
                            ssh_kwargs=ssh_kwargs,
                            dry_run=dry_run,
                        )
                        effective_topology = topology_result.topology
                        click.echo("  Detected topology: %s" % effective_topology.value)

                    # For explicit ring without detection data, run detection
                    if effective_topology == CX7Topology.RING and topology_result is None:
                        if not dry_run:
                            click.echo("  Running topology detection for ring...")
                            topology_result = detect_topology(
                                detections,
                                host_list,
                                ssh_kwargs=ssh_kwargs,
                                dry_run=dry_run,
                            )
                        else:
                            from sparkrun.orchestration.networking import CX7TopologyResult

                            topology_result = CX7TopologyResult(topology=CX7Topology.RING)

                    click.echo("  Topology: %s" % effective_topology.value)

                    # Reuse any interface filter saved on the cluster (e.g. set
                    # via `setup cx7 --interfaces '*np1'`) so the wizard pins the
                    # same port pair.  See issue #203.
                    cx7_interfaces: list[str] | None = None
                    if cluster_name:
                        try:
                            cx7_interfaces = cluster_mgr.get(cluster_name).fabric_interfaces or None
                        except Exception:
                            cx7_interfaces = None
                    if cx7_interfaces:
                        click.echo("  Interface filter: %s" % ", ".join(cx7_interfaces))

                    # Select subnets and plan based on topology
                    if effective_topology == CX7Topology.RING:
                        all_subnets = select_subnets_for_topology(detections, effective_topology)
                        click.echo("  Subnets: %s" % ", ".join(str(s) for s in all_subnets))
                        if topology_result is None:
                            raise click.ClickException("Ring topology requires detected link information")
                        plan = plan_ring_cx7(detections, topology_result, all_subnets, interfaces=cx7_interfaces)
                    else:
                        s1, s2 = select_subnets(detections)
                        all_subnets = [s1, s2]
                        click.echo("  Subnets: %s, %s" % (s1, s2))
                        plan = plan_cluster_cx7(detections, s1, s2, interfaces=cx7_interfaces)

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
                                _report_host_key_distribution(
                                    distribute_host_keys(
                                        all_cx7_ips,
                                        host_list,
                                        ssh_kwargs=ssh_kwargs,
                                        dry_run=dry_run,
                                    ),
                                    len(all_cx7_ips),
                                )

                            results["cx7"] = "configured (%s)" % effective_topology.value if ok_count else "failed"
                            if ok_count:
                                cx7_changed_ips = True
                            if ok_count and not dry_run and cluster_name:
                                assert manifest_mgr is not None
                                manifest_mgr.record_phase(
                                    cluster_name,
                                    user,
                                    host_list,
                                    "cx7",
                                    subnets=[str(s) for s in all_subnets],
                                    cx7_ips=all_cx7_ips,
                                    netplan_file=CX7_NETPLAN_FILE,
                                    topology=effective_topology.value,
                                )

                    # Save topology to cluster definition
                    if not dry_run and cluster_name and effective_topology != CX7Topology.UNKNOWN:
                        try:
                            cluster_mgr.update(cluster_name, topology=effective_topology.value)
                        except Exception:
                            pass

                except Exception as e:
                    results["cx7"] = "failed"
                    click.echo("CX7 error: %s" % e, err=True)
                    if not yes and not click.confirm("Continue?", default=True):
                        return
            elif not cx7_done:
                results["cx7"] = "skipped"
            click.echo()

        # ── Phase 3b: Re-run SSH mesh after CX7 changes ─────────────
        # CX7 configuration may add new IPs that need to be in the SSH
        # mesh.  Re-run the mesh (discover-ips phase only) so inter-node
        # SSH works over the newly configured CX7 interfaces.
        if cx7_changed_ips and host_list and len(host_list) >= 2 and mesh_ready:
            click.echo("Phase 3b: Re-meshing SSH after CX7 IP changes")
            click.echo("-" * 30)
            click.echo("CX7 configuration changed network IPs. Re-running SSH mesh")
            click.echo("to ensure full connectivity across all interfaces.")
            click.echo()

            try:
                mesh_hosts, self_ip, cross_user_remesh = _mesh_members()

                ok = _run_ssh_mesh(
                    mesh_hosts,
                    user,
                    cluster_hosts=host_list,
                    ssh_key=config.ssh_key,
                    discover_ips=True,
                    dry_run=dry_run,
                    control_is_member=(self_ip is not None and self_ip in mesh_hosts),
                )
                results["ssh_remesh"] = "OK" if ok else "failed"
                if ok and not dry_run and cluster_name:
                    assert manifest_mgr is not None
                    manifest_mgr.record_phase(
                        cluster_name,
                        user,
                        mesh_hosts,
                        "ssh_mesh_post_cx7",
                        mesh_hosts=mesh_hosts,
                        cross_user=cross_user_remesh,
                    )
            except Exception as e:
                results["ssh_remesh"] = "failed"
                click.echo("SSH re-mesh error: %s" % e, err=True)
                if not yes and not click.confirm("Continue?", default=True):
                    return
            click.echo()

        action_context = SetupActionContext(user=user, ssh_kwargs=ssh_kwargs, dry_run=dry_run, dispatch=_run_sudo_on_host)
        results.update(
            run_host_steps(
                states,
                setup_context,
                action_context,
                yes=yes,
                manifest_mgr=manifest_mgr,
                cluster_name=cluster_name,
                ensure_password=_ensure_sudo_password,
            )
        )

    except (KeyboardInterrupt, click.Abort):
        click.echo()
        click.echo()
        click.echo("Setup interrupted.")
        if results:
            click.echo("Completed before interruption:")
            for key, val in results.items():
                click.echo("  %-10s %s" % (key + ":", val))
        click.echo()
        click.echo(render_identity_text("You can re-run '{app_command} setup wizard' to resume, or configure"))
        click.echo(render_identity_text("individual steps via '{app_command} setup <command>'."))
        click.echo("For manual setup instructions, see: https://sparkrun.dev")
        click.echo()
        return

    finally:
        recordings.close()

    # ── Phase 7: Summary ─────────────────────────────────────────
    click.echo()
    click.echo("Setup preview complete (no changes made)." if dry_run else "Setup finished; review the readiness results above.")
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
    from sparkrun.core.setup_steps import all_setup_steps

    for step in all_setup_steps():
        if step.key not in {"ssh_mesh", "cx7"} and step.key in results:
            click.echo("  %s: %s" % (step.label, results[step.key]))
    if "readiness" in results:
        click.echo("  Readiness: %s" % results["readiness"])

    click.echo()
    click.echo("Next steps:")
    click.echo(render_identity_text("  {app_command} setup check                 # Review remaining setup gaps"))
    click.echo(render_identity_text("  {app_command} list                        # Browse configured recipe registries"))
    click.echo("  Select a recipe and container qualified for the detected target hardware.")

    from sparkrun.telemetry import emit_setup_wizard_event

    emit_setup_wizard_event(
        config,
        wizard_run_kind=wizard_run_kind,
        results=results,
        cluster_node_count=len(host_list),
        dry_run=dry_run,
        cx7_detected=cx7_detected_any,
    )
    click.echo()

    if any(value == "failed" for value in results.values()):
        raise click.ClickException("One or more setup steps failed; see results above")
