"""Recognise workload-log signatures whose remedy sparkrun can name.

A workload that dies during engine init leaves a traceback in its log and
nothing pointing back at the *launcher* decision that caused it.  This module
recognises one such class — a process that failed writing inside its own Python
installation — so the post-mortem seam
(:meth:`~sparkrun.orchestration.executors._base.Executor.describe_terminated`)
can attribute it instead of handing the operator a bare exception.

**Detection here, wording there.**  The signature is substrate-independent (the
same traceback appears whether the process ran in Docker, on a k8s pod, or
natively), but the fix is not: ``-o user=root`` is a Docker ``--user`` concept
and means nothing on a ``local`` job.  So this module returns *what happened*
and the executor authors *what to do about it* — the same split
:attr:`~sparkrun.core.cluster_status.TerminationInfo.investigate_hints` already
enforces for ``docker logs`` vs ``kubectl logs``.

**Narrow on purpose.**  sparkrun runs containers rootless by default
(``--user $(id -u):$(id -g)``, see
:meth:`~sparkrun.orchestration.executors.docker.DockerExecutor.apply_runtime_adjustments`),
which is right for nearly every image and wrong for the few that JIT-compile
into their own ``site-packages``.  A detector that fired on every write failure
would recommend running as root for problems root does not fix, and the
recommendation would be ignored by the time it mattered — the same
false-positive discipline the ENTRYPOINT preflight is built around, where
"non-empty ENTRYPOINT" would have flagged every working NGC image.  Two gates
keep it tight:

* the failing path must lie under a Python installation tree
  (``site-packages`` / ``dist-packages``), and
* ``ENOENT`` only counts alongside a directory-creation frame.

That second gate is what separates "could not *create* a directory in the
package" from "a packaged data file is missing", which is an image defect that
running as root would not fix.  ``ENOENT`` is in scope at all because
``Path.mkdir(parents=True)`` walks up on ``FileNotFoundError`` and re-raises
from the retry, so a blocked creation can surface as either errno depending on
which rung of the path was missing versus unwritable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Path fragments that mark a Python installation tree.  Both spellings are
#: needed: Debian/Ubuntu system interpreters use ``dist-packages`` (which is
#: what the DGX Spark vLLM images ship), everything else uses ``site-packages``.
_PY_INSTALL_MARKERS = ("/site-packages/", "/dist-packages/")

#: Frames that mean the failing call was *creating* a directory rather than
#: reading one.  ``pathlib.Path.mkdir`` shows up as ``, in mkdir``; the raw
#: ``os`` calls show up as the call text itself.
_CREATION_FRAME_MARKERS = (
    "os.mkdir(",
    "os.makedirs(",
    ", in mkdir",
    ", in makedirs",
    ", in _mkdir",
)

#: ``OSError`` renders as ``[Errno N] <message>: '<path>'``.  Only the three
#: errnos that can mean "you may not write here" are matched; the message is
#: captured for the hint so the operator sees the kernel's own words.
_OS_ERROR_RE = re.compile(r"\[Errno (?P<errno>2|13|30)\]\s+(?P<message>[A-Za-z][A-Za-z \-]*?):\s*['\"](?P<path>[^'\"]+)['\"]")

EACCES = 13
ENOENT = 2
EROFS = 30


@dataclass(frozen=True)
class InPlaceWriteFailure:
    """A process failed writing inside its own Python installation.

    Args:
        path: The path the process could not create or write.
        errno: The reported errno — :data:`EACCES`, :data:`EROFS` or
            :data:`ENOENT`.
        message: The OS message as it appeared in the log (``Permission
            denied``, ``Read-only file system``, ...).
    """

    path: str
    errno: int
    message: str


def detect_in_place_write_failure(text: str) -> InPlaceWriteFailure | None:
    """Return the first in-place write failure in *text*, or ``None``.

    "First" rather than "last" because every match that clears both gates is
    equally actionable, and a crash log routinely repeats the same traceback
    once per worker process — reporting the last one would name a different
    rank's copy of the same failure.

    Args:
        text: Decoded log output.  Bounded by the caller; this does no
            truncation of its own.

    Returns:
        The failure, or ``None`` when nothing in *text* clears both gates.
    """
    if not text:
        return None

    has_creation_frame = any(marker in text for marker in _CREATION_FRAME_MARKERS)

    for match in _OS_ERROR_RE.finditer(text):
        path = match.group("path")
        if not any(marker in path for marker in _PY_INSTALL_MARKERS):
            continue
        errno = int(match.group("errno"))
        if errno == ENOENT and not has_creation_frame:
            # A missing file under site-packages is a broken image, not a
            # permission problem; running as root would not conjure it.
            continue
        return InPlaceWriteFailure(path=path, errno=errno, message=match.group("message").strip())

    return None


__all__ = [
    "EACCES",
    "ENOENT",
    "EROFS",
    "InPlaceWriteFailure",
    "detect_in_place_write_failure",
]
