# OCI Relay image distribution

Sparkrun `develop-next` / 0.4.0 bundles the [OCI Relay](https://github.com/scitrera/oci-relay)
image-distribution adapter. It streams registry pulls or existing local images
to Docker hosts, reuses compatible layers across storage backends, and reports
transfer progress. The `plugins.oci_relay` feature defaults **on for alpha** and
**off for stable/beta**. Explicit config or environment overrides take precedence.

To test on any channel, add to your Sparkrun configuration:

```yaml
features:
  plugins.oci_relay: true
```

Then use `sparkrun run YOUR_RECIPE` normally. There is no separate plugin install
or Go build: the controller downloads checksum-pinned v0.1.0 Linux binaries and
stages them to the execution hosts. Targets need no internet access for binary
installation. An initial online acquisition is required before offline use.
Native-store access and preparation reads default on; source and transport
selection are automatic. Controller-local `:latest` refreshes stream from the
registry through OCI Relay, including when the controller already has an older
image; there is no preliminary Docker pull/import. Routine receiver and registry
progress updates use a 30-second cadence, with phase changes and completion
reported immediately. To opt out on alpha, set the feature to `false`, or use
`container_distribution_provider: builtin` for the built-in copy path.

Mac controllers should use `transfer_mode: delegated` so that the relay runs
on Linux cluster nodes. The separate macOS ARM64 binary is a standalone build;
the adapter's execution-host setup currently supports Linux amd64/arm64 only.
See upstream [platform support](https://github.com/scitrera/oci-relay/blob/main/docs/platforms.md)
and [plugin configuration](https://github.com/scitrera/oci-relay/blob/main/docs/sparkrun-plugin.md)
for transport, authentication, cache, and tuning settings. Avoid enabling both
the bundled copy and an independently installed OCI Relay plugin.

## Updating the vendor snapshot

From the Sparkrun checkout:

```sh
python scripts/vendor-oci-relay.py update --latest
python scripts/vendor-oci-relay.py verify
```

Use `--initial` only when importing into a checkout with no snapshot. `--latest`
resolves the published release's `plugin-release.json` to an immutable adapter
commit. This commit follows the engine tag and adds verified archive checksums;
the importer requires unchanged engine code since that tag. An explicit source
and full commit SHA can be supplied with `--source PATH_OR_URL --rev FULL_SHA`.

`vendor/oci-relay.lock` records file hashes, upstream revision, host API versions,
and licensing. README links are rewritten to the pinned upstream commit so
they remain usable in the imported package. CI verifies the snapshot offline. Upstream owns the adapter and
its exported contract tests; make changes there and re-vendor. OCI Relay remains
AGPL-3.0-only with its Sparkrun license exception; Sparkrun's own code remains
Apache-2.0. The vendored package includes both license documents and provenance.
