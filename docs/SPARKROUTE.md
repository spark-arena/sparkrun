# SparkRoute integration and vendoring

`develop-next` includes the first-party SparkRoute plugin as a vendored source
snapshot. Installing or building Sparkrun does not fetch plugin code. At first
gateway start, the plugin acquires the pinned SparkRoute executable and verifies
its release archive against the embedded SHA-256 for the controller platform.

## Channel defaults

| Active feature channel | SparkRoute | LiteLLM | Default gateway |
| --- | --- | --- | --- |
| stable | disabled | enabled | LiteLLM |
| beta | disabled | enabled | LiteLLM |
| alpha | enabled | disabled | SparkRoute |

The active channel comes from `self_update.channel` in `config.yaml`, unless
`features.channel` explicitly overrides it. Merely checking out a Git branch
does not change an existing installation's channel. To preview the alpha
defaults without changing the update channel, set:

```yaml
features:
  channel: alpha
```

Then `sparkrun proxy start` selects SparkRoute and `sparkrun proxy ui` opens its
console. `sparkrun setup features list` shows the resolved flag values.
The plugin is gated before import on stable/beta, so its bridge and recipe
extension are not registered there unless explicitly enabled.

Explicit `gateway.sparkroute` and `gateway.litellm` feature overrides in config
or environment still win. An explicit `proxy.gateway` in `proxy.yaml` or a
`--gateway` argument must name an enabled gateway. Remove a persisted gateway
pin to follow the channel default; an existing pin is not silently replaced.
If both gateways are explicitly enabled, LiteLLM retains its normal selection
priority unless SparkRoute is explicitly selected. Changing channels does not
restart a running gateway.

## Update to the latest plugin release

Maintainers need Python 3.12+, Git, and GitHub CLI. Authenticate `gh` with an
account that can read the plugin repository if it is private. From the
Sparkrun `develop-next` checkout:

```sh
python scripts/vendor-sparkroute.py update --latest
python scripts/vendor-sparkroute.py verify
uv run pytest tests/test_vendor_sparkroute.py tests/test_gateway_channels.py tests/vendor/sparkroute
git diff --stat
git add vendor/sparkroute.lock src/sparkrun/plugins/sparkroute tests/vendor/sparkroute
git commit -m "Update vendored SparkRoute plugin"
```

`--latest` resolves the canonical plugin repository's latest published GitHub
release, excludes drafts/prereleases, and imports its exact tag. It checks the
tag against the declared plugin version, preserves the license and combination
permission, and records the release tag, full Git commit/tree, and file hashes
in `vendor/sparkroute.lock` and packaged `VENDORED.toml`.

The update refuses a modified vendor snapshot unless `--force` is explicitly
supplied. Fix plugin source in its canonical repository rather than editing the
vendored copy. Review the resulting diff and run the complete host suite before
merging an update. CI and the local commit hook verify the checked-in snapshot
offline; normal builds and installations never follow `latest`.

## Initial release and development snapshots

There is no published plugin release at the time of this integration. The
initial snapshot therefore pins the reviewed development commit recorded in
the lock, without claiming a release tag. To enable the release update path:

1. Push the reviewed plugin source and run its manual **Release** qualification
   workflow. After its gates pass, tag the version declared in its
   `versions.yaml` (currently `v0.1.0`) and let the tag workflow publish the
   GitHub release.
2. Run the `update --latest` sequence above in Sparkrun and commit the resulting
   lock, source, and tests to `develop-next`.

If no release exists, the updater fails without falling back to `main` or
changing the snapshot. For a deliberate development import or rollback:

```sh
python scripts/vendor-sparkroute.py update --source /path/to/sparkrun-sparkroute-plugin --rev FULL_COMMIT_SHA
python scripts/vendor-sparkroute.py verify
```

`--source` also accepts a Git URL; without it, the canonical plugin repository
is used. Add `--initial` only when creating a snapshot in a host that has no
vendor lock yet. Use a full commit for reproducible development imports.

## License and source provenance

Sparkrun's host code remains Apache-2.0. The plugin is AGPL-3.0-only with the
sparkrun combination permission in its `LICENSE_EXCEPTION`. Both texts and its
source provenance are included in Sparkrun distributions. Plugin releases and
SparkRoute gateway releases are separate: updating the plugin snapshot carries
the gateway version and archive pins chosen by that plugin release.
