# SPDX-FileCopyrightText: 2026 Spark Arena
# SPDX-License-Identifier: Apache-2.0
"""Bounded upstream range settings, separate from receiver link striping."""

LIMITS = {
    'registry_range_concurrency': (1, 16),
    'registry_range_chunk_bytes': (64 << 10, 64 << 20),
    'registry_range_threshold_bytes': (64 << 10, 1 << 50),
    'registry_range_buffer_bytes': (64 << 10, 1 << 30),
}


def validate(settings):
    for key, (minimum, maximum) in LIMITS.items():
        if key in settings:
            value = settings[key]
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f'{key} must be between {minimum} and {maximum}')


def plan(settings, capabilities):
    values = {key: settings[key] for key in LIMITS if key in settings}
    validate(values)
    if 'registry-range-v1' not in capabilities:
        if values:
            raise ValueError('registry range overrides require a source binary with registry-range-v1')
        return {}
    # New binaries select conservative defaults and reserve range read-ahead
    # from max_buffer_bytes themselves. Old releases see no unknown plan keys.
    return values
