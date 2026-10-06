# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see LICENSE_EXCEPTION.
"""Small offline contract suite exported with the bundled adapter."""

import json
from pathlib import Path


def test_oci_relay_release_pins_and_host_progress_level():
    from sparkrun.plugins import oci_relay
    from sparkrun.plugins.oci_relay.progress import PROGRESS
    from sparkrun.core.progress import PROGRESS as HOST_PROGRESS

    pins = json.loads(Path(oci_relay.__file__).with_name('releases.json').read_text())[oci_relay.__version__]
    assert set(pins) == {'linux/amd64', 'linux/arm64', 'darwin/arm64'}
    for arch, pin in pins.items():
        assert pin['url'] == (f'https://github.com/scitrera/oci-relay/releases/download/v{oci_relay.__version__}/'
                              f'oci-relay_{oci_relay.__version__}_{arch.replace("/", "_")}.tar.gz')
        assert len(pin['sha256']) == 64 and all(c in '0123456789abcdef' for c in pin['sha256'])
    assert PROGRESS == HOST_PROGRESS


def test_oci_relay_defaults_keep_native_access_and_public_fallback_available():
    from sparkrun.plugins.oci_relay.source_policy import validate

    settings = validate({})
    assert settings.get('source_mode', 'auto') == 'auto'
    assert settings['allow_native_store'] is True
    assert settings['allow_preparation_read'] is True
