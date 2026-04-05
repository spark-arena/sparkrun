"""sparkrun adv group — advanced/experimental commands (hidden by default)."""

from __future__ import annotations

import click

from ._common import (
    _get_context,
    _resolve_hosts_or_exit,
    dry_run_option,
    host_options,
    json_option,
    print_json,
)


@click.group(hidden=True)
@click.pass_context
def adv(ctx):
    """Advanced and experimental commands."""
    pass


@adv.command("compare-images")
@click.argument("image")
@host_options
@dry_run_option
@json_option()
@click.pass_context
def adv_compare_images(ctx, image, hosts, hosts_file, cluster_name, dry_run, output_json):
    """Compare a container image ID across local machine and cluster hosts.

    Useful for debugging image distribution mismatches — shows the Docker
    image ID for IMAGE on the local machine and on every host.

    \b
    Examples:
      sparkrun adv compare-images myimage:latest --cluster mylab
      sparkrun adv compare-images sparkrun-eugr-vllm-tf5 --hosts 192.168.11.13
    """
    from sparkrun.containers.distribute import _check_remote_image_ids
    from sparkrun.containers.registry import get_image_id
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    sctx = _get_context(ctx)
    config = sctx.config
    host_list, _ = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)
    ssh_kwargs = build_ssh_kwargs(config)

    if dry_run:
        click.echo("[dry-run] Would compare image '%s' across local + %d host(s)" % (image, len(host_list)))
        return

    # Local image ID
    local_id = get_image_id(image)

    # Remote image IDs
    remote_ids = _check_remote_image_ids(
        image,
        host_list,
        ssh_user=ssh_kwargs.get("ssh_user"),
        ssh_key=ssh_kwargs.get("ssh_key"),
        ssh_options=ssh_kwargs.get("ssh_options"),
    )

    if output_json:
        print_json(
            {
                "image": image,
                "local": local_id,
                "hosts": {h: remote_ids.get(h) for h in host_list},
            }
        )
        return

    # Table output
    click.echo("Image: %s\n" % image)
    click.echo("  %-40s %s" % ("Host", "Image ID"))
    click.echo("  " + "-" * 110)
    click.echo("  %-40s %s" % ("(local)", local_id or "(not found)"))
    for h in host_list:
        rid = remote_ids.get(h)
        match = ""
        if rid and local_id:
            match = "  ✓ match" if rid == local_id else "  ✗ MISMATCH"
        click.echo("  %-40s %s%s" % (h, rid or "(not found)", match))
