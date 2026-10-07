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
or Go build: the controller downloads checksum-pinned v0.1.1 Linux binaries and
stages them to the execution hosts. Targets need no internet access for binary
installation. Verified cached releases work offline. In automatic provider mode,
an unavailable release download (including offline mode without a cached bundle)
logs a warning and falls back to builtin image distribution. Set
`container_distribution_fallback: false` or explicitly select
`container_distribution_provider: oci-relay` to require the relay. Integrity errors
and failed transfers do not fall back; offline distribution still requires a
resident source image.
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

## ColdSnap image acquisition

ColdSnap capture staging uses Sparkrun's shared distribution path. Restore
capsule pulls also use the selected image provider through ColdSnap's manager
callbacks. With OCI Relay enabled/selected, each requested capsule is acquired
for its assigned host; different images per launch unit and multiple images on
one host are supported. ColdSnap continues to select and validate its capsules.

The artifact retains the original registry digest. Image inspection, launch,
and source tagging resolve it to the verified local Docker image ID. This also
works when a relay import has no upstream `RepoDigest` in Docker. Each callback
rechecks the provider's mapping, including activation after prepare-only;
workloads using managed images launch with `--pull never`.

Callbacks receive the operation's cluster configuration and prepared transport
session explicitly, so provider selection is preserved across controller callback
threads. Builtin/disabled-provider pulls retain the existing authenticated Docker
path. An unsupported provider falls back only under the configured automatic
fallback policy; a failed transfer is reported without a Docker retry.

Offline image requests reach only providers advertising offline support. They
never fall back to a registry pull, and workloads cannot pull implicitly. This
image policy does not establish end-to-end offline support for ColdSnap's other
asset/tool acquisition. Local-only capsules still require a resident image.
Docker build internals and image publication retain their existing paths.

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
