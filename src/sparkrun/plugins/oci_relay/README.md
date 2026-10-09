<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-License-Identifier: Apache-2.0
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
arm64 bundles are built and tested on native runners. See [platform support](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/docs/platforms.md), [storage compatibility](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/docs/storage-compatibility.md) and
[validation](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/docs/validation.md). Published binaries are available from [GitHub releases](https://github.com/spark-arena/oci-relay/releases).

## Quick start with Sparkrun

Requires Bash, `uv`, Go 1.26.9+, Python 3.12+, a C compiler, make, and a Sparkrun `develop-next` / 0.4.0
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
off for other channels; this setup explicitly selects the editable adapter. See [development setup](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/docs/development.md)
for details or [plugin configuration](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/docs/sparkrun-plugin.md) for manual setup.

To build the standalone executable:

```sh
CGO_ENABLED=0 go build -trimpath -o bin/oci-relay ./cmd/oci-relay
```

## Documentation

- [Sparkrun plugin](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/docs/sparkrun-plugin.md): configuration, transports, multiple
  links, adaptive limits, progress, and optional tuning.
- [Source modes and manifests](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/docs/source-modes.md) and
  [automatic selection](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/docs/source-selection.md).
- [Registry sources](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/docs/registry-source.md): authentication, caching, and limits.
- [Bundled receiver decoding](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/docs/bundled-decoder.md): helper installation, disk budgets, and fallback.
- [Storage compatibility](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/docs/storage-compatibility.md): native access, layer
  discovery, mixed stores, and hash identities.
- [Standalone commands](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/docs/standalone.md): sessions, events, and transfer-only validation.
- [Development](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/docs/development.md): tests, CI, versions, and vendoring;
  [release procedure](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/docs/releasing.md).
- [Validation and limitations](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/docs/validation.md).

## License and contributions

Copyright 2026 Scitrera LLC. OCI Relay and its Sparkrun plugin are licensed
under [Apache-2.0](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/LICENSE).

See [third-party notices](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/THIRD_PARTY_NOTICES.md) and
[dependency licenses](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/THIRD_PARTY_LICENSES.txt). See
[CONTRIBUTING.md](https://github.com/spark-arena/oci-relay/blob/406e3eb3f380944c9dbe7e15f630df9c2fcbc243/CONTRIBUTING.md) for contribution guidelines.
