<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-License-Identifier: AGPL-3.0-only
SPDX-FileComment: The Sparkrun additional permission in LICENSE_EXCEPTION applies.
-->

# OCI Relay

OCI Relay distributes container images from a local Docker store or remote OCI
registry to multiple Docker hosts. The Go relay handles verified transfers;
the Sparkrun plugin coordinates hosts, selects sources and transports, tunes
resource limits, and reports progress.

- Stream registry pulls directly to receivers without importing on the source.
- Reuse existing layers across classic overlay2 and containerd image stores.
- Transfer over authenticated HTTP/2, directly or through SSH forwarding/stdio.
- Use multiple TCP data paths with bounded memory and concurrency.

Native access supports qualified rootful Linux Docker 29+ stores. Real-engine
qualification currently covers Linux arm64. Linux amd64, Linux arm64 and macOS
arm64 bundles are built and tested on native runners. See [platform support](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/docs/platforms.md), [storage compatibility](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/docs/storage-compatibility.md) and
[validation](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/docs/validation.md). Published binaries are available from [GitHub releases](https://github.com/scitrera/oci-relay/releases).

## Quick start with Sparkrun

Requires Bash, `uv`, Go 1.25+, Python 3.12+, a C compiler, make, and a Sparkrun `develop-next` / 0.4.0
checkout with image-distribution and pre-pull API 1. The default checkout path
is `../oss-sparkrun`; set `SPARKRUN_CHECKOUT` to use another location.

```bash
source dev.sh
sparkrun run YOUR_RECIPE
deactivate
```

The setup builds the relay and native decoder, installs both projects editably, and enables the
plugin in a private development configuration. It preserves normal user
configuration and keeps the plugin outside Sparkrun's source tree. Image-transfer
progress is visible by default. The bundled plugin defaults on for alpha and
off for other channels; this setup explicitly selects the editable adapter. See [development setup](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/docs/development.md)
for details or [plugin configuration](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/docs/sparkrun-plugin.md) for manual setup.

To build the standalone executable:

```sh
CGO_ENABLED=0 go build -trimpath -o bin/oci-relay ./cmd/oci-relay
```

## Documentation

- [Sparkrun plugin](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/docs/sparkrun-plugin.md): configuration, transports, multiple
  links, adaptive limits, progress, and optional tuning.
- [Source modes and manifests](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/docs/source-modes.md) and
  [automatic selection](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/docs/source-selection.md).
- [Registry sources](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/docs/registry-source.md): authentication, caching, and limits.
- [Bundled receiver decoding](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/docs/bundled-decoder.md): helper installation, disk budgets, and fallback.
- [Storage compatibility](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/docs/storage-compatibility.md): native access, layer
  discovery, mixed stores, and hash identities.
- [Standalone commands](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/docs/standalone.md): sessions, events, and transfer-only validation.
- [Development](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/docs/development.md): tests, CI, versions, and vendoring;
  [release procedure](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/docs/releasing.md).
- [Validation and limitations](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/docs/validation.md).

## License and contributions

Copyright 2026 Scitrera LLC. OCI Relay is [AGPL-3.0-only](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/LICENSE), with a
[Sparkrun additional permission](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/LICENSE_EXCEPTION) allowing combination and
vendoring while Sparkrun's Apache-licensed portions retain Apache-2.0.
OCI Relay remains AGPL, including applicable corresponding-source obligations.
Ship both license documents with binaries and plugin copies.

See [third-party notices](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/THIRD_PARTY_NOTICES.md) and
[dependency licenses](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/THIRD_PARTY_LICENSES.txt). Contributions require the
[CLA](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/CLA.md); see [CONTRIBUTING.md](https://github.com/scitrera/oci-relay/blob/158efd023c90d19fae1e7cb9e97f2fa481383c9b/CONTRIBUTING.md).
