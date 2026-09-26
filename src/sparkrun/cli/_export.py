"""sparkrun export command group."""

from __future__ import annotations

from sparkrun.core.application_profile import render_identity_text

from sparkrun.core.application_profile import get_application_profile, resource_name, child_environment

import logging
import os
import sys
import json
import textwrap
from pathlib import Path

import click
import yaml

from ._common import (
    RECIPE_NAME,
    TARGET,
    _apply_recipe_overrides,
    _get_config_and_registry,
    _get_context,
    _is_cluster_id,
    _load_recipe,
    _resolve_hosts_or_exit,
    dry_run_option,
    host_options,
    recipe_override_options,
    resolve_cluster_config,
    json_option,
    print_json,
)

# NOTE: _resolve_hosts_or_exit is retained in _export.py because the internal
# helpers (_resolve_recipe_for_systemd, export_running) use it conditionally
# and do not map cleanly to the @with_host_context decorator pattern.

logger = logging.getLogger(__name__)


def _apply_spark_arena_benchmarks(recipe, recipe_name: str):
    """Set metadata.spark_arena_benchmarks when the recipe comes from spark-arena."""
    from sparkrun.core.recipe import SPARK_ARENA_PREFIX

    if recipe.metadata.get("spark_arena_benchmarks"):
        return  # already set

    if recipe_name.startswith(SPARK_ARENA_PREFIX):
        from sparkrun.core.parallelism import extract_parallelism_meta

        uuid = recipe_name[len(SPARK_ARENA_PREFIX) :]
        parallelism_meta = extract_parallelism_meta(recipe.defaults or {})
        entry: dict = {"uuid": uuid}
        entry.update(parallelism_meta)
        if not parallelism_meta:
            entry["tp"] = 1
        recipe.metadata["spark_arena_benchmarks"] = [entry]


@click.group()
@click.pass_context
def export(ctx):
    """Export recipes and running workload configurations."""
    pass


@export.command("recipe")
@click.argument("recipe_name", type=RECIPE_NAME)
@json_option()
@click.option("--save", "save_path", type=click.Path(), help="Save a copy of the recipe to a file")
@click.option(
    "--keep-include",
    is_flag=True,
    help="Emit a recipe that uses include: as written, instead of the flattened, self-contained recipe",
)
@click.option(
    "--realize",
    is_flag=True,
    help="Plan against the target hosts and emit the plain recipe that launch would run (overrides resolved for this hardware)",
)
@click.option(
    "--no-pin", "no_pin", is_flag=True, help="With --realize: keep the image tag and model revision as written instead of pinning them"
)
@click.option(
    "--offline",
    is_flag=True,
    help="With --realize: pin what the hosts already have (resident image, cached model) instead of asking the registry and the Hub",
)
@host_options
@recipe_override_options
@click.pass_context
def export_recipe(
    ctx,
    recipe_name,
    output_json=False,
    save_path=None,
    keep_include=False,
    realize=False,
    no_pin=False,
    offline=False,
    hosts=None,
    hosts_file=None,
    cluster_name=None,
    options=(),
    tensor_parallel=None,
    pipeline_parallel=None,
    data_parallel=None,
    gpu_mem=None,
    max_model_len=None,
    image=None,
):
    """Export normalized recipe to stdout or file.

    A recipe built with include: is flattened into one self-contained recipe
    (its bases merged in) unless --keep-include is given.

    --realize probes the target hosts (--hosts / --cluster, or the default
    cluster) and places the workload exactly as run would, then emits an
    ordinary v2 recipe with the matching overrides: layers and any --tp / -o
    values baked in and the overrides: block removed. Nothing is launched.
    Platform defaults (e.g. DGX Spark env and memory utilization) are left to
    the platform, which applies them at launch.

    --realize also pins: the image becomes repo@sha256:... (through the
    recipe's builder, so an eugr :latest pins the nightly it would pull) and
    the model revision a commit. By default those are what the registry and
    the Hub name now; --offline pins what the hosts already hold instead.
    --no-pin keeps the references as written.

    Examples:

      {app_command} export recipe @lil/Qwen3.8-Flash-Next-NVFP4 --realize --tp 2

      {app_command} export recipe my-recipe --realize --cluster mylab --save realized.yaml
    """
    launch_flags = (hosts, hosts_file, cluster_name, tensor_parallel, pipeline_parallel, data_parallel, gpu_mem, max_model_len, image)
    if not realize and (options or any(flag is not None for flag in launch_flags)):
        raise click.UsageError("host and override options (--hosts, --cluster, --tp, -o, ...) only apply with --realize")
    if realize and keep_include:
        raise click.UsageError("--realize emits a flattened recipe; it cannot be combined with --keep-include")
    if not realize and (no_pin or offline):
        raise click.UsageError("--no-pin and --offline only apply with --realize")
    if no_pin and offline:
        raise click.UsageError("--offline chooses where pins come from; it has no effect with --no-pin")
    if realize:
        _export_realized_recipe(
            ctx,
            recipe_name,
            output_json=output_json,
            save_path=save_path,
            pin=not no_pin,
            offline=offline,
            hosts=hosts,
            hosts_file=hosts_file,
            cluster_name=cluster_name,
            options=options,
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            data_parallel=data_parallel,
            gpu_mem=gpu_mem,
            max_model_len=max_model_len,
            image=image,
        )
        return

    config, _ = _get_config_and_registry()
    recipe, recipe_path, registry_mgr = _load_recipe(config, recipe_name)

    declared = getattr(recipe, "_include_declared", None)
    if keep_include and declared is not None:
        from sparkrun.utils.yaml_helpers import LiteralBlockDumper

        text = (
            json.dumps(declared, indent=2)
            if output_json
            else yaml.dump(declared, Dumper=LiteralBlockDumper, indent=2, sort_keys=False, default_flow_style=False)
        )
        if save_path is None:
            click.echo(text)
        else:
            Path(save_path).write_text(text, encoding="utf-8")
            click.echo("Recipe saved to %s" % save_path)
        return

    # Auto-populate spark_arena_benchmarks from @spark-arena/UUID references
    _apply_spark_arena_benchmarks(recipe, recipe_name)

    if save_path is None:
        if output_json:
            print_json(recipe.to_dict())
        else:
            click.echo(recipe.export(json=False))
        return

    click.echo("Recipe saved to %s" % recipe.export(path=save_path, json=output_json))


def _export_realized_recipe(ctx, recipe_name, *, output_json, save_path, pin, offline, hosts, hosts_file, cluster_name, options, **flags):
    """``export recipe --realize``: plan like ``run``, then write the recipe that plan would launch."""
    from sparkrun import api
    from sparkrun.cli._common import resolve_host_context
    from sparkrun.utils.yaml_helpers import LiteralBlockDumper

    sctx = _get_context(ctx)
    config = sctx.config
    hctx = resolve_host_context(hosts, hosts_file, cluster_name, config, sctx.variables, sctx=sctx)
    recipe, _recipe_path, _registry_mgr = _load_recipe(config, recipe_name, resolve=False)
    recipe, overrides = _apply_recipe_overrides(options, recipe=recipe, **flags)

    local_cache_dir, remote_cache_dir, _mode, _iface = hctx.cluster.resolve_transfer_config(config)
    run_options = api.RunOptions(
        recipe=recipe,
        hosts=tuple(hctx.host_list),
        cluster=hctx.cluster_name,
        overrides=dict(overrides),
        cache_dir=remote_cache_dir,
        local_cache_dir=local_cache_dir,
    )
    try:
        realized = api.realize_recipe(run_options, pin=pin, offline=offline, sctx=sctx)
    except api.SparkrunError as exc:
        click.echo("Error: %s" % exc, err=True)
        sys.exit(1)

    info = realized.recipe["metadata"]["realized_for"]
    click.echo(
        "Realized for %s (%d node(s)); %s"
        % (info.get("platform") or info.get("accelerator") or "unknown hardware", info["nodes"], hctx.describe()),
        err=True,
    )
    if realized.plan.override_resolution is not None:
        for line in realized.plan.override_resolution.describe():
            click.echo("  %s" % line, err=True)
    if realized.dropped_keys:
        click.echo("  dropped override-only keys: %s" % ", ".join(realized.dropped_keys), err=True)
    for pinned in realized.pins:
        click.echo("  pinned %s: %s -> %s (%s)" % (pinned.field, pinned.before, pinned.after, pinned.source), err=True)
    for note in realized.unpinned:
        click.echo("  not pinned: %s" % note, err=True)

    text = (
        json.dumps(realized.recipe, indent=2)
        if output_json
        else yaml.dump(realized.recipe, Dumper=LiteralBlockDumper, indent=2, sort_keys=False, default_flow_style=False)
    )
    if save_path is None:
        click.echo(text)
    else:
        Path(save_path).write_text(text, encoding="utf-8")
        click.echo("Recipe saved to %s" % save_path, err=True)


@export.command("running-recipe")
@click.argument("target", type=TARGET)
@host_options
@json_option()
@click.option("--save", "save_path", type=click.Path(), help="Save to a file")
@click.pass_context
def export_running(ctx, target, hosts, hosts_file, cluster_name, output_json, save_path):
    """Export the effective recipe from a running workload.

    TARGET can be a recipe name or a cluster ID (from {app_command} status output).
    The exported recipe includes all applied overrides baked into defaults.

    Examples:

      {app_command} export running-recipe e5f6a7b8

      {app_command} export running-recipe glm-4.7-flash-awq --cluster mylab

      {app_command} export running-recipe e5f6a7b8 --json

      {app_command} export running-recipe e5f6a7b8 --save effective-recipe.yaml
    """
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import load_job_metadata, derive_cluster_id

    config, _ = _get_config_and_registry()

    # Resolve cluster_id
    if (cluster_id := _is_cluster_id(target)) is None:
        # Target is a recipe name — need hosts to generate cluster_id
        recipe, _recipe_path, _registry_mgr = _load_recipe(config, target)
        host_list, _cluster_mgr = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)
        cluster_id = derive_cluster_id(recipe, host_list)

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
        _output_export(recipe, output_json, save_path, overrides=meta_overrides, container_image=effective_image)
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
        _output_export(recipe, output_json, save_path, overrides=meta_overrides, container_image=effective_image)
        return

    # Legacy fallback: reconstruct partial overrides from individual metadata fields
    legacy_overrides = {}
    if meta.get("tensor_parallel") is not None:
        legacy_overrides["tensor_parallel"] = meta["tensor_parallel"]
    if meta.get("port") is not None:
        legacy_overrides["port"] = meta["port"]
    if meta.get("served_model_name") is not None:
        legacy_overrides["served_model_name"] = meta["served_model_name"]

    _output_export(recipe, output_json, save_path, overrides=legacy_overrides or None, container_image=effective_image)


def _output_export(recipe, output_json, save_path, overrides=None, container_image=None):
    """Export recipe to stdout or file with optional overrides baked in."""
    if save_path is None:
        if output_json:
            print_json(recipe.to_dict(overrides=overrides, container_image=container_image))
        else:
            click.echo(recipe.export(json=False, overrides=overrides, container_image=container_image))
    else:
        click.echo(
            "Recipe saved to %s" % recipe.export(path=save_path, json=output_json, overrides=overrides, container_image=container_image)
        )


# ---------------------------------------------------------------------------
# systemd export helpers
# ---------------------------------------------------------------------------

_SYSTEMD_UNIT_TEMPLATE = textwrap.dedent("""\
    # sparkrun.distribution={owner}
    [Unit]
    Description={app_command} inference: {recipe_name} ({model})
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
    Environment=SPARKRUN_APPLICATION_PROFILE={profile_ref}
    Environment=HOME={user_home}
    Environment=PATH=/usr/local/bin:/usr/bin:/bin:{extra_path}
    WorkingDirectory={user_home}
    StandardOutput=journal
    StandardError=journal
    SyslogIdentifier={namespace}-{slug}

    [Install]
    WantedBy=multi-user.target
""")


def _render_systemd_unit(slug, recipe, cluster_name, ssh_user, sparkrun_path, user_home):
    """Render a systemd unit file from template values."""
    extra_path = os.path.dirname(sparkrun_path) if sparkrun_path else ""
    service_dir = "%s/.config/%s/services/%s" % (user_home, get_application_profile().config_namespace, slug)
    return _SYSTEMD_UNIT_TEMPLATE.format(
        app_command=get_application_profile().command,
        namespace=get_application_profile().resource_namespace,
        owner=get_application_profile().id,
        profile_ref=child_environment()["SPARKRUN_APPLICATION_PROFILE"],
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


def _service_owner_guard(path, *, unit=False):
    """Reject foreign artifacts before installing or removing service files."""
    from sparkrun.utils.shell import quote

    legacy_check = 'grep -q "^Description=sparkrun inference:" "$_artifact"' if unit else "true"
    return """_artifact={path}
if [ -e "$_artifact" ]; then
    _owner=$(sed -n 's/^# sparkrun.distribution=//p; s/^distribution: *//p' "$_artifact" | sort -u)
    if [ "$_owner" != {owner} ]; then
        if [ -n "$_owner" ] || [ {owner} != sparkrun ] || ! {legacy}; then
            echo "Refusing to replace service artifact owned by another application: $_artifact" >&2
            exit 1
        fi
    fi
fi""".format(path=quote(path), owner=quote(get_application_profile().id), legacy=legacy_check)


def _service_artifacts(slug, cluster_name, user_home):
    root = "%s/.config/%s" % (user_home, get_application_profile().config_namespace)
    return (
        "%s/services/%s/recipe.yaml" % (root, slug),
        "%s/services/%s/cluster.yaml" % (root, slug),
        "%s/clusters/%s.yaml" % (root, cluster_name),
    )


def _service_guards(slug, cluster_name, user_home):
    unit = "/etc/systemd/system/%s.service" % resource_name("-%s" % slug)
    return "\n".join(
        [_service_owner_guard(unit, unit=True), *(_service_owner_guard(p) for p in _service_artifacts(slug, cluster_name, user_home))]
    )


def _service_lock(user_home, *, unit_path=None):
    """Lock the persistent home directory before the privileged unit lock.

    Both phases can lock this inode, and service removal cannot unlink it.
    Sharing it also serializes services referencing the same cluster file.
    """
    from sparkrun.utils.shell import quote

    script = "exec 9<%s\nflock -x 9" % quote(user_home)
    if unit_path is not None:
        script += "\nexec 8>%s\nflock -x 8" % quote(unit_path + ".lock")
    return script


def _render_install_script(slug, recipe_yaml, cluster_yaml, cluster_name, user_home):
    """Render user-level install script (no sudo needed)."""
    service_dir = "%s/.config/%s/services/%s" % (user_home, get_application_profile().config_namespace, slug)
    clusters_dir = "%s/.config/%s/clusters" % (user_home, get_application_profile().config_namespace)
    return textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail

        {lock}
        {guards}
        mkdir -p '{service_dir}' '{clusters_dir}'
        _recipe_tmp=$(mktemp '{service_dir}/.recipe.XXXXXX')
        _cluster_tmp=$(mktemp '{service_dir}/.cluster.XXXXXX')
        _definition_tmp=$(mktemp '{clusters_dir}/.cluster.XXXXXX')
        trap 'rm -f "$_recipe_tmp" "$_cluster_tmp" "$_definition_tmp"' EXIT

        # Write baked recipe
        cat > "$_recipe_tmp" << 'SPARKRUN_RECIPE_EOF'
        # sparkrun.distribution={owner}
        {recipe_yaml}
        SPARKRUN_RECIPE_EOF

        # Write cluster reference
        cat > "$_cluster_tmp" << 'SPARKRUN_CLUSTER_EOF'
        # sparkrun.distribution={owner}
        {cluster_yaml}
        SPARKRUN_CLUSTER_EOF

        # Write cluster definition for sparkrun run --cluster
        cp "$_cluster_tmp" "$_definition_tmp"
        mv -f "$_recipe_tmp" '{service_dir}/recipe.yaml'
        mv -f "$_cluster_tmp" '{service_dir}/cluster.yaml'
        mv -f "$_definition_tmp" '{clusters_dir}/{cluster_name}.yaml'

        echo "Service files installed to {service_dir}"
    """).format(
        lock=_service_lock(user_home),
        guards=_service_guards(slug, cluster_name, user_home),
        owner=get_application_profile().id,
        service_dir=service_dir,
        clusters_dir=clusters_dir,
        cluster_name=cluster_name,
        recipe_yaml=recipe_yaml,
        cluster_yaml=cluster_yaml,
    )


def _render_sudo_install_script(slug, unit_contents, *, user_home, cluster_name):
    """Render sudo install script (writes unit file, enables service)."""
    unit_path = "/etc/systemd/system/%s.service" % resource_name("-%s" % slug)
    return textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail

        {lock}
        {guards}
        {required}
        _unit_tmp=$(mktemp '{unit_path}.XXXXXX')
        trap 'rm -f "$_unit_tmp"' EXIT
        cat > "$_unit_tmp" << 'SPARKRUN_UNIT_EOF'
        {unit_contents}
        SPARKRUN_UNIT_EOF
        chmod 644 "$_unit_tmp"
        mv -f "$_unit_tmp" '{unit_path}'

        # Reload systemd and enable service
        systemctl daemon-reload
        systemctl enable '{namespace}-{slug}'

        echo "Service {namespace}-{slug} installed and enabled"
    """).format(
        namespace=get_application_profile().resource_namespace,
        unit_path=unit_path,
        lock=_service_lock(user_home, unit_path=unit_path),
        guards=_service_guards(slug, cluster_name, user_home),
        required="\n".join(
            "test -f '%s' || { echo 'Service files missing; retry installation' >&2; exit 1; }" % p
            for p in _service_artifacts(slug, cluster_name, user_home)
        ),
        slug=slug,
        unit_contents=unit_contents,
    )


def _render_uninstall_script(slug, cluster_name, user_home):
    """Render uninstall script (sudo: stop, disable, remove unit; user: remove service dir)."""
    service_dir = "%s/.config/%s/services/%s" % (user_home, get_application_profile().config_namespace, slug)
    cluster_file = "%s/.config/%s/clusters/%s.yaml" % (user_home, get_application_profile().config_namespace, cluster_name)
    unit_path = "/etc/systemd/system/%s.service" % resource_name("-%s" % slug)
    return textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail

        SERVICE_NAME='{namespace}-{slug}'
        UNIT_PATH='/etc/systemd/system/{namespace}-{slug}.service'

        {lock}
        {guards}

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
        namespace=get_application_profile().resource_namespace,
        lock=_service_lock(user_home, unit_path=unit_path),
        guards=_service_guards(slug, cluster_name, user_home),
        slug=slug,
        service_dir=service_dir,
        cluster_file=cluster_file,
    )


def _detect_remote_sparkrun(host, ssh_kwargs, dry_run=False):
    """Detect sparkrun path and user home on a remote host.

    Returns (sparkrun_path, user_home) on success, or (None, None) if not found.
    """
    from sparkrun.orchestration.ssh import run_remote_script

    script = textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
        SPARKRUN_PATH=$(which @APP_COMMAND@ 2>/dev/null || true)
        if [ -z "$SPARKRUN_PATH" ]; then
            echo "ERROR: sparkrun not found on PATH" >&2
            exit 1
        fi
        echo "$SPARKRUN_PATH"
        echo "$HOME"
    """).replace("@APP_COMMAND@", get_application_profile().command)

    if dry_run:
        return "/usr/local/bin/" + get_application_profile().command, "/home/user"

    result = run_remote_script(host, script, **ssh_kwargs, timeout=15)
    if not result.success:
        return None, None

    lines = result.stdout.strip().splitlines()
    if len(lines) < 2:
        return None, None

    return lines[0].strip(), lines[1].strip()


def _install_remote_sparkrun(host, ssh_kwargs, dry_run=False):
    """Install sparkrun on a remote host via uvx.

    Returns True on success, False on failure.
    """
    from sparkrun.orchestration.ssh import run_remote_script

    from sparkrun.core.channels import channel_requirement
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.utils.shell import quote

    requirement = channel_requirement(SparkrunConfig().self_update_channel)
    script = textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
        UV_PATH=$(which uv 2>/dev/null || true)
        if [ -z "$UV_PATH" ]; then
            echo "ERROR: uv not found — cannot auto-install sparkrun" >&2
            echo "Install uv first: https://docs.astral.sh/uv/" >&2
            exit 1
        fi
        "$UV_PATH" tool install @REQUIREMENT@ --force
    """).replace("@REQUIREMENT@", quote(requirement))

    if dry_run:
        click.echo(render_identity_text("Would install {app_command} on %s via: uvx {app_command} setup install" % host))
        return True

    result = run_remote_script(host, script, **ssh_kwargs, timeout=120)
    if result.stderr.strip():
        click.echo(result.stderr.strip(), err=True)
    return result.success


def _resolve_recipe_for_systemd(
    target,
    config,
    hosts,
    hosts_file,
    cluster_name,
    options,
    tensor_parallel,
    pipeline_parallel,
    data_parallel,
    gpu_mem,
    max_model_len,
    image,
    port,
    served_model_name,
    *,
    sctx=None,
):
    """Resolve recipe, hosts, and overrides for the systemd command.

    Returns (recipe, overrides, host_list, effective_cluster_name).
    """
    from sparkrun.orchestration.job_metadata import load_job_metadata
    from sparkrun.core.recipe import Recipe

    # If target is a cluster_id, reconstruct from job metadata
    if (cluster_id := _is_cluster_id(target)) is not None:
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
        options,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        data_parallel=data_parallel,
        gpu_mem=gpu_mem,
        max_model_len=max_model_len,
        image=image,
        recipe=recipe,
        **extra_kw,
    )

    host_list, _ = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)
    host_list = _trim_hosts_via_api(host_list, recipe, overrides, sctx=sctx)

    return recipe, overrides, host_list


def _trim_hosts_via_api(host_list, recipe, overrides, *, sctx=None):
    """Resolve ``host_list`` to the scheduler's effective hosts via ``api.schedule``.

    Mirrors the placement-driven host selection ``api.run`` performs so
    the systemd export targets the same hosts the launcher would have
    used.  Returns the original list when there is no multi-host
    parallelism to schedule.
    """
    if len(host_list) <= 1:
        return host_list

    import sparkrun.api as api
    from sparkrun.core.parallelism import extract_parallelism
    from sparkrun.core.scheduler import SchedulingRequest

    parallelism = extract_parallelism(recipe.build_config_chain(overrides))
    if not any(getattr(parallelism, k) > 1 for k in ("tensor_parallel", "pipeline_parallel", "data_parallel")):
        return host_list
    request = SchedulingRequest(
        parallelism=parallelism,
        hosts=tuple(host_list),
        host_hardware=None,
        layout=getattr(recipe, "layout", None),
        resources=None,
    )
    result = api.schedule(request, sctx=sctx)
    return list(result.assignment.hosts_used)


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
@click.option("--service-name", default=None, help="Override service name (default: {resource_namespace}-<slug>)")
@dry_run_option
@click.pass_context
def export_systemd(
    ctx,
    target,
    hosts,
    hosts_file,
    cluster_name,
    tensor_parallel,
    pipeline_parallel,
    data_parallel,
    gpu_mem,
    max_model_len,
    options,
    image,
    port,
    served_model_name,
    do_install,
    do_uninstall,
    start,
    service_name,
    dry_run,
):
    """Generate a systemd service for a {app_command} inference workload.

    TARGET can be a recipe name (with optional overrides) or a cluster_id
    (from a running workload).

    By default, prints generated artifacts (unit file, scripts) for review.
    Use --install to deploy to the head node, or --uninstall to remove.

    \b
    Examples:
      {app_command} export systemd qwen3-1.7b --cluster mylab
      {app_command} export systemd qwen3-1.7b --cluster mylab --install --start
      {app_command} export systemd qwen3-1.7b --cluster mylab --uninstall
      {app_command} export systemd e5f6a7b8 --install
    """
    if do_install and do_uninstall:
        click.echo("Error: --install and --uninstall are mutually exclusive.", err=True)
        sys.exit(1)

    if start:
        do_install = True

    config, _ = _get_config_and_registry()
    sctx = _get_context(ctx)

    # Resolve recipe, overrides, and hosts
    recipe, overrides, host_list = _resolve_recipe_for_systemd(
        target,
        config,
        hosts,
        hosts_file,
        cluster_name,
        options,
        tensor_parallel,
        pipeline_parallel,
        data_parallel,
        gpu_mem,
        max_model_len,
        image,
        port,
        served_model_name,
        sctx=sctx,
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

    # Detect sparkrun on head node (needed for unit file), auto-install if missing
    sparkrun_path, user_home = _detect_remote_sparkrun(head_host, ssh_kwargs, dry_run=dry_run)
    if sparkrun_path is None:
        click.echo(render_identity_text("{app_command} not found on %s, attempting auto-install..." % head_host))
        if not _install_remote_sparkrun(head_host, ssh_kwargs, dry_run=dry_run):
            click.echo(render_identity_text("Error: Failed to install {app_command} on '%s'." % head_host), err=True)
            click.echo(render_identity_text("  Install manually: ssh %s 'uvx {app_command} setup install'" % head_host), err=True)
            sys.exit(1)
        sparkrun_path, user_home = _detect_remote_sparkrun(head_host, ssh_kwargs, dry_run=dry_run)
        if sparkrun_path is None:
            click.echo(render_identity_text("Error: {app_command} installed but not found on PATH on '%s'." % head_host), err=True)
            sys.exit(1)
        click.echo(render_identity_text("{app_command} installed on %s" % head_host))

    unit_contents = _render_systemd_unit(
        slug,
        recipe,
        systemd_cluster_name,
        ssh_user,
        sparkrun_path,
        user_home,
    )
    install_script = _render_install_script(
        slug,
        recipe_yaml,
        cluster_yaml,
        systemd_cluster_name,
        user_home,
    )
    sudo_install_script = _render_sudo_install_script(slug, unit_contents, user_home=user_home, cluster_name=systemd_cluster_name)
    # uninstall_script = _render_uninstall_script(slug, systemd_cluster_name, user_home)

    if not do_install:
        # Dry-run mode: display all generated artifacts
        click.echo("=" * 60)
        click.echo(render_identity_text("systemd service: {resource_namespace}-%s" % slug))
        click.echo("Head node: %s" % head_host)
        click.echo("Cluster hosts: %s" % ", ".join(host_list))
        click.echo("=" * 60)
        click.echo()
        click.echo(render_identity_text("--- Unit file: /etc/systemd/system/{resource_namespace}-%s.service ---" % slug))
        click.echo(unit_contents)
        click.echo("--- Baked recipe ---")
        click.echo(recipe_yaml)
        click.echo("--- Cluster definition: %s ---" % systemd_cluster_name)
        click.echo(cluster_yaml)
        click.echo("--- Install script (user-level) ---")
        click.echo(install_script)
        click.echo("--- Install script (sudo) ---")
        click.echo(sudo_install_script)
        # click.echo("--- Uninstall script (sudo) ---")
        # click.echo(uninstall_script)
        click.echo("To deploy, re-run with --install")
        return

    # --install mode
    _do_install(slug, head_host, ssh_user, ssh_kwargs, install_script, sudo_install_script, start, dry_run)


def _do_install(slug, head_host, ssh_user, ssh_kwargs, install_script, sudo_install_script, start, dry_run):
    """Deploy service files and systemd unit to head node."""
    from sparkrun.orchestration.ssh import run_remote_script
    from sparkrun.orchestration.sudo import run_sudo_script_on_host

    service_name = resource_name("-%s" % slug)

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
        head_host,
        sudo_install_script,
        sudo_password,
        ssh_kwargs=ssh_kwargs,
        dry_run=dry_run,
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
            head_host,
            start_script,
            sudo_password,
            ssh_kwargs=ssh_kwargs,
            dry_run=dry_run,
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

    service_name = resource_name("-%s" % slug)

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
        head_host,
        uninstall_script,
        sudo_password,
        ssh_kwargs=ssh_kwargs,
        dry_run=dry_run,
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
