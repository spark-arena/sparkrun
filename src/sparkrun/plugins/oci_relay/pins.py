# SPDX-FileCopyrightText: 2026 Spark Arena
# SPDX-License-Identifier: Apache-2.0

"""Verified registry-pin bindings in the management user's private host cache.

Docker cannot tag an imported image with an upstream RepoDigest. A receipt is
written only after relay verification, and resolves to an immutable Docker ID,
never to the mutable retention tag. The cache has the same trust boundary as
the management user and Docker access; it is not an untrusted manifest cache.
"""

import hashlib
import json
import re
import shlex
import uuid

from .host import OperationError
from .source_policy import LOCAL_DOCKER


def valid_id(value):
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def digest(image):
    if "@" not in image:
        return None
    name, pin = image.rsplit("@", 1)
    if not name or "@" in name or not valid_id(pin):
        raise OperationError("OCI Relay requires a full sha256 registry pin")
    return pin


def retention_tag(image):
    # Include the full reference, not a truncated digest or a mutable input tag.
    digest(image)
    return "oci-relay/pinned:" + hashlib.sha256(image.encode()).hexdigest()


def _path(runner, host, image):
    key = hashlib.sha256(image.encode()).hexdigest()
    return runner.cache_directory(host) + "/pins/" + key + ".json"


def _inspect(runner, host, reference):
    result = runner.connection(host).execute(
        host or "localhost", LOCAL_DOCKER + ["image", "inspect", "--format={{.Id}}|{{.Os}}|{{.Architecture}}", reference], timeout=30,
    )
    fields = result.stdout.decode().strip().split("|") if result.returncode == 0 else []
    if len(fields) != 3 or not valid_id(fields[0]) or fields[1] != "linux":
        return None
    configured = runner.settings.get("platform", "").split("/")[:2]
    expected = ["linux", runner.architecture(host)]
    if configured != [""] and configured != expected:
        return None
    return fields[0] if fields[1:] == expected else None


def resolve(runner, host, image):
    """Recover a prior verified import, or Docker's own resident pin binding."""
    pin = digest(image)
    if pin is None:
        return None
    output = runner.connection(host).execute(
        host or "localhost", ["head", "-c", "16385", "--", _path(runner, host, image)], timeout=15,
    )
    if output.returncode == 0 and len(output.stdout) <= 16384:
        try:
            receipt = json.loads(output.stdout)
        except (ValueError, UnicodeError):
            receipt = None
        if (isinstance(receipt, dict) and receipt.get("version") == 1
                and receipt.get("image") == image and receipt.get("registry_digest") == pin
                and valid_id(receipt.get("config_digest")) and valid_id(receipt.get("runtime_image"))):
            runtime_image = receipt["runtime_image"]
            if _inspect(runner, host, runtime_image) == runtime_image:
                return runtime_image
    # Do not infer a pin from a retention tag: it can be retagged by the user.
    return _inspect(runner, host, image)


def record(runner, host, image, runtime_image, config_digest):
    pin = digest(image)
    if not pin or not valid_id(runtime_image) or not valid_id(config_digest):
        raise OperationError("cannot record an unverified registry pin")
    path = _path(runner, host, image)
    directory = path.rsplit("/", 1)[0]
    temporary = path + "." + uuid.uuid4().hex
    receipt = dict(version=1, image=image, registry_digest=pin, runtime_image=runtime_image, config_digest=config_digest)
    # Atomic publication; neither a partial write nor a tag race can rebind a pin.
    script = ("umask 077; mkdir -p -- " + shlex.quote(directory) + " && cat > " + shlex.quote(temporary)
              + " && mv -f -- " + shlex.quote(temporary) + " " + shlex.quote(path))
    try:
        runner.execute(host, ["sh", "-c", script], input_data=json.dumps(receipt).encode())
    finally:
        runner.connection(host).execute(host or "localhost", ["rm", "-f", "--", temporary], timeout=10)


def preflight(runner, host, image):
    """Check metadata storage before payload work; this does not reserve disk."""
    directory = _path(runner, host, image).rsplit("/", 1)[0]
    temporary = directory + "/.preflight." + uuid.uuid4().hex
    script = "umask 077; mkdir -p -- " + shlex.quote(directory) + " && cat > " + shlex.quote(temporary)
    try:
        runner.execute(host, ["sh", "-c", script], input_data=b"\0" * 4096)
    except OperationError as error:
        raise OperationError(f"{host}: registry pin cache is not writable before transfer: {error}") from error
    finally:
        runner.connection(host).execute(host or "localhost", ["rm", "-f", "--", temporary], timeout=10)
