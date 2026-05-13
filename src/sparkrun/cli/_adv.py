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
    HIDE_ADVANCED_OPTIONS,
)


@click.group(hidden=HIDE_ADVANCED_OPTIONS)
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
    from sparkrun.containers.distribute import _check_remote_image_identities, _images_match
    from sparkrun.containers.registry import get_image_identity
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    sctx = _get_context(ctx)
    config = sctx.config
    host_list, _ = _resolve_hosts_or_exit(hosts, hosts_file, cluster_name, config)
    ssh_kwargs = build_ssh_kwargs(config)

    if dry_run:
        click.echo("[dry-run] Would compare image '%s' across local + %d host(s)" % (image, len(host_list)))
        return

    # Local image identity (Id + RepoDigests)
    local_id, local_digests = get_image_identity(image)

    # Remote image identities
    remote_identities = _check_remote_image_identities(
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
                "local": {"id": local_id, "repo_digests": local_digests},
                "hosts": {
                    h: ({"id": remote_identities[h][0], "repo_digests": remote_identities[h][1]} if h in remote_identities else None)
                    for h in host_list
                },
            }
        )
        return

    # Table output
    def _short(s: str | None) -> str:
        if not s:
            return "(not found)"
        # Trim sha256: prefix and keep first 12 hex chars to keep the row compact
        return s.split(":", 1)[-1][:12] if s.startswith("sha256:") else s

    def _short_digest(digests: list[str]) -> str:
        if not digests:
            return "(none)"
        sha = digests[0].rsplit("@", 1)[-1]
        return _short(sha) if sha.startswith("sha256:") else sha

    click.echo("Image: %s\n" % image)
    click.echo("  %-32s  %-14s  %-14s  %s" % ("Host", "Image ID", "RepoDigest", "Match"))
    click.echo("  " + "-" * 80)
    click.echo("  %-32s  %-14s  %-14s  %s" % ("(local)", _short(local_id), _short_digest(local_digests), "-"))
    for h in host_list:
        r_id, r_digests = remote_identities.get(h, (None, []))
        if r_id is None and not r_digests:
            match = "(not found)"
        elif _images_match(local_id, local_digests, r_id, r_digests):
            match = "✓ match"
        else:
            match = "✗ MISMATCH"
        click.echo("  %-32s  %-14s  %-14s  %s" % (h, _short(r_id), _short_digest(r_digests), match))
