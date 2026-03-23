"""sparkrun export command group."""

from __future__ import annotations

import logging
import os
import sys
import textwrap

import click
import yaml

from ._common import (
    RECIPE_NAME,
    TARGET,
    _apply_node_trimming,
    _apply_recipe_overrides,
    _get_config_and_registry,
    _is_cluster_id,
    _load_recipe,
    _resolve_hosts_or_exit,
    dry_run_option,
    host_options,
    recipe_override_options,
    resolve_cluster_config,
)

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def export(ctx):
    """Export recipes and running workload configurations."""
    pass


@export.command("recipe")
@click.argument("recipe_name", type=RECIPE_NAME)
@click.option("--json", "output_json", is_flag=True, help="Output normalized recipe as JSON to stdout")
@click.option("--save", "save_path", type=click.Path(), help="Save a copy of the recipe to a file")
@click.pass_context
def export_recipe(ctx, recipe_name, output_json=False, save_path=None):
    """Export normalized recipe to stdout or file."""
    config, _ = _get_config_and_registry()
    recipe, recipe_path, registry_mgr = _load_recipe(config, recipe_name)

    if save_path is None:
        click.echo(recipe.export(json=output_json))
        return

    click.echo("Recipe saved to %s" % recipe.export(path=save_path, json=output_json))


@export.command("running-recipe")
@click.argument("target", type=TARGET)
@host_options
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--save", "save_path", type=click.Path(), help="Save to a file")
@click.pass_context
def export_running(ctx, target, hosts, hosts_file, cluster_name, output_json, save_path):
    """Export the effective recipe from a running workload.

    TARGET can be a recipe name or a cluster ID (from sparkrun status output).
    The exported recipe includes all applied overrides baked into defaults.

    Examples:

      sparkrun export running-recipe e5f6a7b8

      sparkrun export running-recipe glm-4.7-flash-awq --cluster mylab

      sparkrun export running-recipe e5f6a7b8 --json

      sparkrun export running-recipe e5f6a7b8 --save effective-recipe.yaml
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import load_job_metadata, generate_cluster_id

    config, _ = _get_config_and_registry()

    # Resolve cluster_id
    if _is_cluster_id(target) is not None:
        cluster_id = _is_cluster_id(target)
    else:
        # Target is a recipe name — need hosts to generate cluster_id
        recipe, _recipe_path, _registry_mgr = _load_recipe(config, target)
        host_list, _cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)
        cluster_id = generate_cluster_id(recipe, host_list)

    # Load job metadata
    meta = load_job_metadata(cluster_id, cache_dir=str(config.cache_dir))
    if not meta:
        click.echo("Error: No job metadata found for '%s'." % target, err=True)
        click.echo("  The workload may not have been launched from this machine.", err=True)
        sys.exit(1)

    # Reconstruction strategy (priority order):
    # 1. recipe_state in metadata → full fidelity deserialization
    # 2. recipe name + overrides → load recipe, export with overrides
    # 3. recipe name + individual fields → partial override reconstruction
    recipe_state = meta.get("recipe_state")
    meta_overrides = meta.get("overrides")
    effective_image = meta.get("effective_container_image")

    if recipe_state:
        # Best path: full-fidelity reconstruction
        recipe = Recipe._deserialize(recipe_state)
        _output_export(recipe, output_json, save_path,
                        overrides=meta_overrides, container_image=effective_image)
        return

    # Fallback: load recipe by name from registries
    recipe_name = meta.get("recipe")
    if not recipe_name:
        click.echo("Error: Job metadata for '%s' has no recipe name or state." % target, err=True)
        sys.exit(1)

    try:
        recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name)
    except SystemExit:
        click.echo("Error: Could not load recipe '%s' from registries." % recipe_name, err=True)
        click.echo("  The original recipe may have been removed or renamed.", err=True)
        sys.exit(1)

    if meta_overrides:
        # Good fallback: full overrides dict available
        _output_export(recipe, output_json, save_path,
                        overrides=meta_overrides, container_image=effective_image)
        return

    # Legacy fallback: reconstruct partial overrides from individual metadata fields
    legacy_overrides = {}
    if meta.get("tensor_parallel") is not None:
        legacy_overrides["tensor_parallel"] = meta["tensor_parallel"]
    if meta.get("port") is not None:
        legacy_overrides["port"] = meta["port"]
    if meta.get("served_model_name") is not None:
        legacy_overrides["served_model_name"] = meta["served_model_name"]

    _output_export(recipe, output_json, save_path,
                    overrides=legacy_overrides or None, container_image=effective_image)


def _output_export(recipe, output_json, save_path, overrides=None, container_image=None):
    """Export recipe to stdout or file with optional overrides baked in."""
    if save_path is None:
        click.echo(recipe.export(json=output_json, overrides=overrides, container_image=container_image))
    else:
        click.echo("Recipe saved to %s" % recipe.export(
            path=save_path, json=output_json, overrides=overrides, container_image=container_image))


# ---------------------------------------------------------------------------
# systemd export helpers
# ---------------------------------------------------------------------------

_SYSTEMD_UNIT_TEMPLATE = textwrap.dedent("""\
    [Unit]
    Description=sparkrun inference: {recipe_name} ({model})
    After=network-online.target docker.service
    Wants=network-online.target
    Requires=docker.service

    [Service]
    Type=simple
    User={ssh_user}
    Group={ssh_user}
    ExecStart={sparkrun_path} run {service_dir}/recipe.yaml \\
        --cluster {cluster_name} --foreground --no-follow
    Restart=on-failure
    RestartSec=30
    TimeoutStartSec=600
    TimeoutStopSec=120
    Environment=HOME={user_home}
    Environment=PATH=/usr/local/bin:/usr/bin:/bin:{extra_path}
    WorkingDirectory={user_home}
    StandardOutput=journal
    StandardError=journal
    SyslogIdentifier=sparkrun-{slug}

    [Install]
    WantedBy=multi-user.target
""")


def _render_systemd_unit(slug, recipe, cluster_name, ssh_user, sparkrun_path, user_home):
    """Render a systemd unit file from template values."""
    extra_path = os.path.dirname(sparkrun_path) if sparkrun_path else ""
    service_dir = "%s/.config/sparkrun/services/%s" % (user_home, slug)
    return _SYSTEMD_UNIT_TEMPLATE.format(
        recipe_name=recipe.name,
        model=recipe.model,
        slug=slug,
        ssh_user=ssh_user,
        sparkrun_path=sparkrun_path,
        service_dir=service_dir,
        cluster_name=cluster_name,
        user_home=user_home,
        extra_path=extra_path,
    )


def _render_install_script(slug, recipe_yaml, cluster_yaml, cluster_name, user_home):
    """Render user-level install script (no sudo needed)."""
    service_dir = "%s/.config/sparkrun/services/%s" % (user_home, slug)
    clusters_dir = "%s/.config/sparkrun/clusters" % user_home
    # Escape single quotes in YAML content for heredoc safety
    recipe_yaml_escaped = recipe_yaml.replace("'", "'\\''")
    cluster_yaml_escaped = cluster_yaml.replace("'", "'\\''")
    return textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail

        # Create service directory
        mkdir -p '{service_dir}'
        mkdir -p '{clusters_dir}'

        # Write baked recipe
        cat > '{service_dir}/recipe.yaml' << 'SPARKRUN_RECIPE_EOF'
        {recipe_yaml}
        SPARKRUN_RECIPE_EOF

        # Write cluster reference
        cat > '{service_dir}/cluster.yaml' << 'SPARKRUN_CLUSTER_EOF'
        {cluster_yaml}
        SPARKRUN_CLUSTER_EOF

        # Write cluster definition for sparkrun run --cluster
        cp '{service_dir}/cluster.yaml' '{clusters_dir}/{cluster_name}.yaml'

        echo "Service files installed to {service_dir}"
    """).format(
        service_dir=service_dir,
        clusters_dir=clusters_dir,
        cluster_name=cluster_name,
        recipe_yaml=recipe_yaml_escaped,
        cluster_yaml=cluster_yaml_escaped,
    )


def _render_sudo_install_script(slug, unit_contents):
    """Render sudo install script (writes unit file, enables service)."""
    unit_path = "/etc/systemd/system/sparkrun-%s.service" % slug
    # Escape single quotes in unit contents for heredoc safety
    unit_escaped = unit_contents.replace("'", "'\\''")
    return textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail

        # Write systemd unit file
        cat > '{unit_path}' << 'SPARKRUN_UNIT_EOF'
        {unit_contents}
        SPARKRUN_UNIT_EOF

        # Reload systemd and enable service
        systemctl daemon-reload
        systemctl enable 'sparkrun-{slug}'

        echo "Service sparkrun-{slug} installed and enabled"
    """).format(
        unit_path=unit_path,
        slug=slug,
        unit_contents=unit_escaped,
    )


def _render_uninstall_script(slug, cluster_name, user_home):
    """Render uninstall script (sudo: stop, disable, remove unit; user: remove service dir)."""
    service_dir = "%s/.config/sparkrun/services/%s" % (user_home, slug)
    cluster_file = "%s/.config/sparkrun/clusters/%s.yaml" % (user_home, cluster_name)
    return textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail

        SERVICE_NAME='sparkrun-{slug}'
        UNIT_PATH='/etc/systemd/system/sparkrun-{slug}.service'

        # Stop and disable if active
        if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
            systemctl stop "$SERVICE_NAME"
            echo "Stopped $SERVICE_NAME"
        fi
        if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
            systemctl disable "$SERVICE_NAME"
            echo "Disabled $SERVICE_NAME"
        fi

        # Remove unit file
        if [ -f "$UNIT_PATH" ]; then
            rm -f "$UNIT_PATH"
            systemctl daemon-reload
            echo "Removed $UNIT_PATH"
        fi

        # Remove service directory and cluster definition
        rm -rf '{service_dir}'
        rm -f '{cluster_file}'
        echo "Removed service files for {slug}"
    """).format(
        slug=slug,
        service_dir=service_dir,
        cluster_file=cluster_file,
    )


def _detect_remote_sparkrun(host, ssh_kwargs, dry_run=False):
    """Detect sparkrun path and user home on a remote host.

    Returns (sparkrun_path, user_home) or exits with error.
    """
    from sparkrun.orchestration.ssh import run_remote_script

    script = textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail
        SPARKRUN_PATH=$(which sparkrun 2>/dev/null || true)
        if [ -z "$SPARKRUN_PATH" ]; then
            echo "ERROR: sparkrun not found on PATH" >&2
            exit 1
        fi
        echo "$SPARKRUN_PATH"
        echo "$HOME"
    """)

    if dry_run:
        return "/usr/local/bin/sparkrun", "/home/user"

    result = run_remote_script(host, script, **ssh_kwargs, timeout=15)
    if not result.success:
        click.echo("Error: sparkrun not found on head node '%s'." % host, err=True)
        stderr = result.stderr.strip()
        if stderr:
            click.echo("  %s" % stderr, err=True)
        click.echo("  Install sparkrun on the head node before using --install.", err=True)
        sys.exit(1)

    lines = result.stdout.strip().splitlines()
    if len(lines) < 2:
        click.echo("Error: Unexpected response from head node '%s'." % host, err=True)
        sys.exit(1)

    return lines[0].strip(), lines[1].strip()


def _resolve_recipe_for_systemd(target, config, hosts, hosts_file, cluster_name,
                                 options, tensor_parallel, pipeline_parallel,
                                 gpu_mem, max_model_len, image, port, served_model_name):
    """Resolve recipe, hosts, and overrides for the systemd command.

    Returns (recipe, overrides, host_list, effective_cluster_name).
    """
    from sparkrun.orchestration.job_metadata import load_job_metadata
    from sparkrun.core.recipe import Recipe

    # If target is a cluster_id, reconstruct from job metadata
    if _is_cluster_id(target) is not None:
        cluster_id = _is_cluster_id(target)
        meta = load_job_metadata(cluster_id, cache_dir=str(config.cache_dir))
        if not meta:
            click.echo("Error: No job metadata found for '%s'." % target, err=True)
            sys.exit(1)

        recipe_state = meta.get("recipe_state")
        meta_overrides = meta.get("overrides", {}) or {}
        effective_image = meta.get("effective_container_image")
        meta_hosts = meta.get("hosts", [])

        if recipe_state:
            recipe = Recipe._deserialize(recipe_state)
        else:
            recipe_name = meta.get("recipe")
            if not recipe_name:
                click.echo("Error: Job metadata has no recipe name or state.", err=True)
                sys.exit(1)
            recipe, _, _ = _load_recipe(config, recipe_name)

        if effective_image:
            recipe.container = effective_image

        # Use hosts from metadata if not overridden
        if not hosts and not hosts_file and not cluster_name and meta_hosts:
            host_list = meta_hosts
        else:
            host_list, _ = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)

        return recipe, meta_overrides, host_list

    # Target is a recipe name
    recipe, _, _ = _load_recipe(config, target, resolve=False)

    # Build overrides from CLI options
    extra_kw = {}
    if port is not None:
        extra_kw["port"] = port
    if served_model_name is not None:
        extra_kw["served_model_name"] = served_model_name

    recipe, overrides = _apply_recipe_overrides(
        options, tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel, gpu_mem=gpu_mem,
        max_model_len=max_model_len, image=image, recipe=recipe, **extra_kw,
    )

    host_list, _ = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)
    host_list = _apply_node_trimming(host_list, recipe, overrides=overrides)

    return recipe, overrides, host_list


def _build_cluster_yaml(cluster_name, hosts, ssh_user=None):
    """Build a cluster definition YAML string."""
    cluster_def = {
        "name": cluster_name,
        "hosts": hosts,
    }
    if ssh_user:
        cluster_def["user"] = ssh_user
    return yaml.safe_dump(cluster_def, default_flow_style=False)


@export.command("systemd")
@click.argument("target", type=TARGET)
@host_options
@recipe_override_options
@click.option("--port", type=int, default=None, help="Override service port")
@click.option("--served-model-name", default=None, help="Override served model name")
@click.option("--install", "do_install", is_flag=True, help="Deploy service to head node")
@click.option("--uninstall", "do_uninstall", is_flag=True, help="Remove service from head node")
@click.option("--start", is_flag=True, help="Start service after install (implies --install)")
@click.option("--service-name", default=None, help="Override service name (default: sparkrun-<slug>)")
@dry_run_option
@click.pass_context
def export_systemd(ctx, target, hosts, hosts_file, cluster_name,
                   tensor_parallel, pipeline_parallel, gpu_mem, max_model_len,
                   options, image, port, served_model_name,
                   do_install, do_uninstall, start, service_name, dry_run):
    """Generate a systemd service for a sparkrun inference workload.

    TARGET can be a recipe name (with optional overrides) or a cluster_id
    (from a running workload).

    By default, prints generated artifacts (unit file, scripts) for review.
    Use --install to deploy to the head node, or --uninstall to remove.

    \b
    Examples:
      sparkrun export systemd qwen3-1.7b --cluster mylab
      sparkrun export systemd qwen3-1.7b --cluster mylab --install --start
      sparkrun export systemd qwen3-1.7b --cluster mylab --uninstall
      sparkrun export systemd e5f6a7b8 --install
    """
    if do_install and do_uninstall:
        click.echo("Error: --install and --uninstall are mutually exclusive.", err=True)
        sys.exit(1)

    if start:
        do_install = True

    config, _ = _get_config_and_registry()

    # Resolve recipe, overrides, and hosts
    recipe, overrides, host_list = _resolve_recipe_for_systemd(
        target, config, hosts, hosts_file, cluster_name,
        options, tensor_parallel, pipeline_parallel,
        gpu_mem, max_model_len, image, port, served_model_name,
    )

    head_host = host_list[0]
    slug = service_name or recipe.slug
    systemd_cluster_name = "%s-systemd" % slug

    # Resolve SSH user
    from sparkrun.orchestration.primitives import build_ssh_kwargs
    from ._common import _get_cluster_manager

    cluster_mgr = _get_cluster_manager()
    cc = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr)
    ssh_user = cc.user or config.ssh_user or os.environ.get("USER", "root")
    ssh_kwargs = build_ssh_kwargs(config)
    if ssh_user:
        ssh_kwargs["ssh_user"] = ssh_user

    # Generate artifacts
    recipe_yaml = recipe.export(overrides=overrides)
    cluster_yaml = _build_cluster_yaml(systemd_cluster_name, host_list, ssh_user=ssh_user)

    if do_uninstall:
        _do_uninstall(slug, systemd_cluster_name, head_host, ssh_user, ssh_kwargs, dry_run)
        return

    # Detect sparkrun on head node (needed for unit file)
    sparkrun_path, user_home = _detect_remote_sparkrun(head_host, ssh_kwargs, dry_run=dry_run)

    unit_contents = _render_systemd_unit(
        slug, recipe, systemd_cluster_name, ssh_user, sparkrun_path, user_home,
    )
    install_script = _render_install_script(
        slug, recipe_yaml, cluster_yaml, systemd_cluster_name, user_home,
    )
    sudo_install_script = _render_sudo_install_script(slug, unit_contents)
    uninstall_script = _render_uninstall_script(slug, systemd_cluster_name, user_home)

    if not do_install:
        # Dry-run mode: display all generated artifacts
        click.echo("=" * 60)
        click.echo("systemd service: sparkrun-%s" % slug)
        click.echo("Head node: %s" % head_host)
        click.echo("Cluster hosts: %s" % ", ".join(host_list))
        click.echo("=" * 60)
        click.echo()
        click.echo("--- Unit file: /etc/systemd/system/sparkrun-%s.service ---" % slug)
        click.echo(unit_contents)
        click.echo("--- Baked recipe ---")
        click.echo(recipe_yaml)
        click.echo("--- Cluster definition: %s ---" % systemd_cluster_name)
        click.echo(cluster_yaml)
        click.echo("--- Install script (user-level) ---")
        click.echo(install_script)
        click.echo("--- Install script (sudo) ---")
        click.echo(sudo_install_script)
        click.echo("--- Uninstall script (sudo) ---")
        click.echo(uninstall_script)
        click.echo("To deploy, re-run with --install")
        return

    # --install mode
    _do_install(slug, head_host, ssh_user, ssh_kwargs,
                install_script, sudo_install_script, start, dry_run)


def _do_install(slug, head_host, ssh_user, ssh_kwargs, install_script, sudo_install_script, start, dry_run):
    """Deploy service files and systemd unit to head node."""
    from sparkrun.orchestration.ssh import run_remote_script
    from sparkrun.orchestration.sudo import run_sudo_script_on_host

    service_name = "sparkrun-%s" % slug

    # Step 1: User-level script (create dirs, write recipe + cluster YAML)
    click.echo("Installing service files on %s..." % head_host)
    r = run_remote_script(head_host, install_script, **ssh_kwargs, dry_run=dry_run)
    if not r.success:
        click.echo("Error: Failed to install service files on %s." % head_host, err=True)
        if r.stderr.strip():
            click.echo("  %s" % r.stderr.strip(), err=True)
        sys.exit(1)
    if r.stdout.strip():
        click.echo("  %s" % r.stdout.strip())

    # Step 2: Sudo script (write unit file, daemon-reload, enable)
    click.echo("Installing systemd unit (requires sudo)...")
    sudo_password = click.prompt("[sudo] password for %s" % ssh_user, hide_input=True)
    r = run_sudo_script_on_host(
        head_host, sudo_install_script, sudo_password,
        ssh_kwargs=ssh_kwargs, dry_run=dry_run,
    )
    if not r.success:
        click.echo("Error: Failed to install systemd unit on %s." % head_host, err=True)
        if r.stderr.strip():
            click.echo("  %s" % r.stderr.strip(), err=True)
        sys.exit(1)
    if r.stdout.strip():
        click.echo("  %s" % r.stdout.strip())

    # Step 3: Optionally start the service
    if start:
        click.echo("Starting %s..." % service_name)
        start_script = "systemctl start '%s'" % service_name
        r = run_sudo_script_on_host(
            head_host, start_script, sudo_password,
            ssh_kwargs=ssh_kwargs, dry_run=dry_run,
        )
        if not r.success:
            click.echo("Error: Failed to start %s." % service_name, err=True)
            if r.stderr.strip():
                click.echo("  %s" % r.stderr.strip(), err=True)
            sys.exit(1)
        click.echo("  %s started" % service_name)

    click.echo()
    click.echo("Service %s installed on %s." % (service_name, head_host))
    click.echo("  Status: systemctl status %s" % service_name)
    click.echo("  Logs:   journalctl -u %s -f" % service_name)


def _do_uninstall(slug, cluster_name, head_host, ssh_user, ssh_kwargs, dry_run):
    """Remove service from head node."""
    from sparkrun.orchestration.ssh import run_remote_command
    from sparkrun.orchestration.sudo import run_sudo_script_on_host

    service_name = "sparkrun-%s" % slug

    # Detect user home on the remote host
    if dry_run:
        user_home = "/home/%s" % ssh_user
    else:
        r = run_remote_command(head_host, "echo $HOME", **ssh_kwargs, timeout=10)
        if not r.success:
            click.echo("Error: Could not detect home directory on %s." % head_host, err=True)
            sys.exit(1)
        user_home = r.stdout.strip()

    uninstall_script = _render_uninstall_script(slug, cluster_name, user_home)

    if dry_run:
        click.echo("[dry-run] Would uninstall %s from %s" % (service_name, head_host))
        click.echo()
        click.echo("--- Uninstall script ---")
        click.echo(uninstall_script)
        return

    click.echo("Uninstalling %s from %s (requires sudo)..." % (service_name, head_host))
    sudo_password = click.prompt("[sudo] password for %s" % ssh_user, hide_input=True)
    r = run_sudo_script_on_host(
        head_host, uninstall_script, sudo_password,
        ssh_kwargs=ssh_kwargs, dry_run=dry_run,
    )
    if not r.success:
        click.echo("Error: Failed to uninstall %s on %s." % (service_name, head_host), err=True)
        if r.stderr.strip():
            click.echo("  %s" % r.stderr.strip(), err=True)
        sys.exit(1)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines():
            click.echo("  %s" % line)
    click.echo("Service %s removed from %s." % (service_name, head_host))
