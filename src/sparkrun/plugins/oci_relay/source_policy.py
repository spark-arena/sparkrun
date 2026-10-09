# SPDX-FileCopyrightText: 2026 Spark Arena
# SPDX-License-Identifier: Apache-2.0

"""Pre-transfer source selection. Capability checks and estimates, not a speed guarantee."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re

STORE_VERSION_PATTERN = re.compile(r'v?([0-9]+)\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z][0-9A-Za-z.+~_-]*)?')
SOURCE_MODES = frozenset({'auto', 'docker', 'docker-save', 'docker-classic', 'docker-containerd', 'registry'})
LOCAL_DOCKER = ['docker', '--host', 'unix:///var/run/docker.sock']


class SourceUnavailable(RuntimeError):
    """No source permitted by policy can be selected before receiver transfer."""


@dataclass(frozen=True)
class Selection:
    mode: str
    reason: str


def validate(settings):
    """Return validated plugin settings with defaults, preserving explicit opt-outs."""
    settings = {'allow_native_store': True, 'allow_preparation_read': True, **settings}
    from .registry_ranges import validate as validate_ranges

    validate_ranges(settings)
    discovery = settings.get('cache_discovery_seconds', 10)
    if type(discovery) is not int or not 1 <= discovery <= 300:
        raise ValueError('cache_discovery_seconds must be between 1 and 300')
    if 'data_paths' in settings:
        from .paths import validate_paths

        validate_paths(settings['data_paths'])
        if settings.get('transport', 'auto') not in {'auto', 'http2-direct'}:
            raise ValueError('data_paths requires auto or http2-direct transport')
    connections = settings.get('connections_per_path', 1)
    if type(connections) is not int or not 1 <= connections <= 4:
        raise ValueError('connections_per_path must be between 1 and 4')
    if connections != 1 and 'data_paths' not in settings:
        raise ValueError('connections_per_path requires data_paths')
    threshold = settings.get('stripe_threshold_bytes', 256 << 20)
    if type(threshold) is not int or (threshold != 0 and not 8 << 20 <= threshold <= 1 << 50):
        raise ValueError('stripe_threshold_bytes must be 0 (disabled) or between 8 MiB and 1 PiB')
    stripes = settings.get('stripe_streams', 4)
    if type(stripes) is not int or not 2 <= stripes <= 8:
        raise ValueError('stripe_streams must be between 2 and 8')
    piece = settings.get('stripe_piece_bytes', 1 << 20)
    if type(piece) is not int or not 1 << 20 <= piece <= 64 << 20 or piece & (piece - 1):
        raise ValueError('stripe_piece_bytes must be a power of two between 1 and 64 MiB')
    window = settings.get('http2_stream_window_bytes', 0)
    if type(window) is not int or (window != 0 and (not 1 << 20 <= window <= 64 << 20 or window & (window - 1))):
        raise ValueError('http2_stream_window_bytes must be 0 or a power of two between 1 and 64 MiB')
    if 'containerd_content_root' in settings:
        root = settings['containerd_content_root']
        if not isinstance(root, str) or not root.startswith('/') or any(c in root for c in ',\n\r\x00'):
            raise ValueError('containerd_content_root must be an absolute bind-safe path')
    mode = settings.get('source_mode', 'auto')
    if not isinstance(mode, str) or mode not in SOURCE_MODES:
        raise ValueError('source_mode must be auto, docker, docker-save, docker-classic, docker-containerd or registry')
    for key in ('allow_native_store', 'allow_preparation_read', 'registry_source', 'registry_plain_http'):
        if key in settings and type(settings[key]) is not bool:
            raise ValueError(key + ' must be a boolean')
    cache = settings.get('registry_cache_bytes', 0)
    if type(cache) is not int or not 0 <= cache <= 1 << 50:
        raise ValueError('registry_cache_bytes must be between 0 and 1 PiB')
    if 'registry_config' in settings and (not isinstance(settings['registry_config'], str) or not settings['registry_config'].startswith('/') or '\x00' in settings['registry_config']):
        raise ValueError('registry_config must be an absolute path on the fetcher')
    if mode == 'registry' and (settings.get('manifest') or settings.get('receiver_import', 'pull') != 'pull'):
        raise ValueError('registry resolves its own manifest and requires receiver_import: pull')
    join = settings.get('source_join_milliseconds', 0)
    if type(join) is not int or not 0 <= join <= 2000:
        raise ValueError('source_join_milliseconds must be between 0 and 2000')
    procs = settings.get('relay_gomaxprocs', 0)
    if type(procs) is not int or not 0 <= procs <= 64:
        raise ValueError('relay_gomaxprocs must be between 0 and 64 (0 inherits the runtime default)')
    decoder = settings.get('receiver_decoder', 'auto')
    if decoder not in {'auto', 'none', 'unpigz'}:
        raise ValueError('receiver_decoder must be auto, none or unpigz')
    workers = settings.get('decode_workers', 8)
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('decode_workers must be between 1 and 16')
    for key, minimum, default in [('max_decode_bytes', 4 << 20, 64 << 30), ('decode_reserve_bytes', 0, 16 << 30)]:
        value = settings.get(key, default)
        if type(value) is not int or not minimum <= value <= 1 << 50:
            raise ValueError(key + ' is outside supported byte budgets')
    importer = settings.get('receiver_import', 'pull')
    if decoder == 'unpigz' and importer != 'pull':
        raise ValueError('unpigz receiver decoder requires pull import')
    if not isinstance(importer, str) or importer not in {'pull', 'load-cached', 'load'}:
        raise ValueError('receiver_import must be pull, load-cached or load')
    if importer != 'pull':
        cap = settings.get('max_import_bytes', 0)
        if type(cap) is not int or not 4 << 20 <= cap <= 1 << 50:
            raise ValueError('load import requires an explicit max_import_bytes budget')
    if 'max_spool_bytes' in settings:
        cap = settings['max_spool_bytes']
        if type(cap) is not int or not 4 << 20 <= cap <= 1 << 50:
            raise ValueError('invalid source spool budget')
    if mode in {'docker-classic', 'docker-containerd', 'docker-save'} and settings.get('manifest'):
        raise ValueError(mode + ' supplies its own uncompressed manifest; remove the manifest setting')
    if mode in {'docker-classic', 'docker-containerd'} and settings.get('allow_native_store') is not True:
        raise SourceUnavailable(mode + ' requires allow_native_store: true')
    if mode == 'docker-save' and (settings.get('allow_preparation_read') is not True or 'max_spool_bytes' not in settings):
        raise SourceUnavailable('docker-save requires allow_preparation_read: true and an explicit full-archive max_spool_bytes budget')
    if mode == 'docker' and not settings.get('manifest') and settings.get('allow_preparation_read') is not True:
        raise SourceUnavailable('docker requires a manifest or allow_preparation_read: true')
    if mode == 'auto' and not settings.get('manifest') and not any(settings.get(key) is True for key in (
        'allow_native_store', 'allow_preparation_read',
    )):
        raise SourceUnavailable('auto requires a manifest, allow_native_store: true or allow_preparation_read: true')
    return settings


def docker_facts(runner, host):
    # Never return the image config or daemon environment in selection logs.
    return json.loads(runner.execute(host, LOCAL_DOCKER + ['info', '--format',
        '{"version":{{json .ServerVersion}},"driver":{{json .Driver}},'
        '"root":{{json .DockerRootDir}},"security":{{json .SecurityOptions}},'
        '"os":{{json .OSType}},"runtimes":{{json .Runtimes}},"driver_status":{{json .DriverStatus}},'
        '"containerd_address":{{if .Containerd}}{{json .Containerd.Address}}{{else}}null{{end}}}']))


def store_version_supported(version):
    """Docker 29+ baseline; backend/layout/hash checks still determine compatibility."""
    match = STORE_VERSION_PATTERN.fullmatch(version) if isinstance(version, str) else None
    return match is not None and int(match[1]) >= 29


def classic_store_reason(facts):
    """Return a reason for rejection, or None when the read-only helper qualifies."""
    if not store_version_supported(facts.get('version')):
        return 'classic-store access requires Docker 29 or newer'
    if facts.get('os') != 'linux' or facts.get('driver') != 'overlay2':
        return 'source is not a qualified Linux classic overlay2 store'
    if not isinstance(facts.get('security'), list) or not all(isinstance(item, str) for item in facts['security']):
        return 'Docker security-mode facts are unavailable'
    if any('rootless' in item or 'userns' in item for item in facts['security']):
        return 'rootless/user-namespace stores are not qualified'
    if not isinstance(facts.get('runtimes'), dict) or 'runc' not in facts['runtimes']:
        return 'the source helper requires the runc runtime'
    root = facts.get('root')
    if not isinstance(root, str) or not root.startswith('/') or any(c in root for c in ',\n\r\x00'):
        return 'the Docker store root is unavailable or unsuitable for a read-only bind'
    return None



def containerd_store_reason(facts):
    if not store_version_supported(facts.get('version')):
        return 'containerd content access requires Docker 29 or newer'
    if facts.get('driver') != 'overlayfs' or ['driver-type', 'io.containerd.snapshotter.v1'] not in (facts.get('driver_status') or []):
        return 'source is not a qualified containerd overlayfs image store'
    # Same confinement/runtime constraints as the classic helper.
    return classic_store_reason(dict(facts, driver='overlay2'))


def native_mounts(facts, settings):
    if containerd_store_reason(facts) is None:
        default_root = facts['root'] + '/containerd/daemon/io.containerd.content.v1.content'
        if facts.get('containerd_address') in {'/run/containerd/containerd.sock', '/var/run/containerd/containerd.sock'}:
            default_root = '/var/lib/containerd/io.containerd.content.v1.content'
        root = settings.get('containerd_content_root') or default_root
        return [(root, '/oci-relay-store', True)]
    return [(facts['root'] + '/image/overlay2', '/oci-relay-store/image/overlay2', True),
            (facts['root'] + '/overlay2', '/oci-relay-store/overlay2', True)]


def archive_reason(settings, facts, route):
    # Keep the overlay2 restriction and resource checks for full-export auto-selection.
    # Explicit docker-save remains available for users qualifying another image store.
    if not store_version_supported(facts.get('version')) or facts.get('driver') != 'overlay2':
        return 'OCI archive export is not qualified for this engine/store'
    if settings.get('allow_preparation_read') is not True or 'max_spool_bytes' not in settings:
        return 'full-archive preparation is disabled or lacks an explicit budget'
    if settings.get('transport', 'auto') != 'http2-direct':
        return 'full staging is not preferred on SSH or an unnegotiated auto route'
    speed = settings.get('network_gbps', route.get('network_gbps', 0))
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed < 25:
        return 'no fast direct route hint (at least 25 Gbps)'
    size = facts.get('image_size')
    if type(size) is not int or size < 0:
        return 'source image size is unavailable'
    # Docker Size is not an exact tar length. Leave headroom for headers/config;
    # the Go archive adapter still enforces the exact byte cap while writing.
    estimate = size + max(4 << 20, (size + 3) // 4)
    if estimate > settings['max_spool_bytes']:
        return 'estimated archive plus headroom exceeds the explicit spool budget'
    # Docker itself creates an export scratch copy. Conservatively require room
    # for both copies on each filesystem, without guessing whether they overlap.
    for key in ('spool_free_bytes', 'docker_free_bytes'):
        free = facts.get(key)
        if type(free) is not int or free < 2 * estimate:
            return 'insufficient or unknown free space for archive and Docker export scratch'
    return None


def select(settings, facts, route):
    settings = validate(settings)
    requested = settings.get('source_mode', 'auto')
    if requested != 'auto':
        return Selection(requested, 'explicit source_mode override')
    if settings.get('manifest'):
        return Selection('docker', 'supplied manifest pins the exact representation')
    native_reason = 'native-store access is not enabled'
    if settings.get('allow_native_store') is True:
        native_reason = classic_store_reason(facts)
        if containerd_store_reason(facts) is None:
            return Selection('docker-containerd', 'qualified read-only containerd content; retains exact blobs without export or recompression')
        if native_reason is None:
            return Selection('docker-classic', 'qualified read-only classic store; avoids export, compression and payload staging')
    archive_rejection = archive_reason(settings, facts, route)
    if archive_rejection is None:
        return Selection('docker-save', native_reason + '; fast direct route and archive/scratch budgets fit')
    reasons = native_reason + '; ' + archive_rejection
    if settings.get('allow_preparation_read') is True:
        return Selection('docker', reasons + '; using bounded push preparation')
    raise SourceUnavailable(reasons + '; no supplied manifest or permitted preparation source')


def detect(runner, host, image, settings, route):
    """Only read-only probes; no preparation, helper creation or receiver pull."""
    settings = validate(settings)
    if settings.get('source_mode', 'auto') != 'auto' or settings.get('manifest'):
        return select(settings, {}, route)
    facts = docker_facts(runner, host)
    if settings.get('allow_native_store') is True and (classic_store_reason(facts) is None or containerd_store_reason(facts) is None):
        return select(settings, facts, route)
    # Only probe archive resources when all non-resource prerequisites hold.
    candidate = dict(facts, image_size=0, spool_free_bytes=1 << 60, docker_free_bytes=1 << 60)
    if archive_reason(settings, candidate, route) is None:
        try:
            facts['image_size'] = int(runner.execute(host, LOCAL_DOCKER + ['image', 'inspect', '--format={{.Size}}', image]))
            for key, path in [('spool_free_bytes', settings.get('spool_dir') or '/tmp'), ('docker_free_bytes', facts['root'])]:
                raw = runner.execute(host, ['df', '-P', '-B1', '--', path]).decode().splitlines()
                facts[key] = int(raw[-1].split()[3])
        except (OSError, RuntimeError, ValueError, IndexError, KeyError):
            # Unknown free space rules out auto staging. Docker probe failures
            # above still surface: they do not prove a working fallback exists.
            pass
    return select(settings, facts, route)
