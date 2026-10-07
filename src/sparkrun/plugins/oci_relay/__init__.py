# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see LICENSE_EXCEPTION.

"""Sparkrun image-copy provider. Registration has no deployment side effects."""

__version__ = "0.1.1"
SPARKRUN_PLUGIN_API_VERSION = 1


def register(v):
    import sparkrun.plugins as api

    if getattr(api, "IMAGE_DISTRIBUTION_API_VERSION", None) != 1 or getattr(api, "IMAGE_PULL_API_VERSION", None) != 1:
        raise RuntimeError("OCI Relay requires Sparkrun develop-next / 0.4.0 with image-distribution API 1 and pre-pull API 1")
    from .provider import PROVIDER

    api.register_image_distribution_provider("oci-relay", PROVIDER)
