"""The one seam through which sparkrun asks the HuggingFace Hub for *metadata*.

Weight downloads (:mod:`sparkrun.models.download`) do not route through here —
a 200 GB pull legitimately takes hours and must not be budgeted.  What this
module bounds is the **advisory** metadata phase: the config / quant-config /
safetensors-size / param-count / visibility lookups that feed the VRAM estimate,
the fit table and telemetry.  Every one of those is optional; the launch degrades
to "no memory claim" without them (``api/_hosts.py`` already catches that), and
none is worth a single second of an unexplained hang (issue #278).

Three defects made that phase unbounded, and each needs a different lever:

1. **The library's shared HTTP client has no timeout at all.**
   ``huggingface_hub.utils._http.default_client_factory`` builds
   ``httpx.Client(..., timeout=None)``, so ``model_info``, ``list_repo_tree``
   and ``paginate`` block forever on a half-closed socket.  Per-call ``timeout=``
   arguments cannot fix this: ``list_repo_tree`` does not accept one.  The lever
   is :func:`configure_hub_client`, which installs a bounded client through the
   library's public ``set_client_factory`` hook — one place, every httpx call.

2. **``hf_xet`` bypasses that client entirely.**  On a Xet-backed repo,
   ``hf_hub_download`` hands the transfer to a Rust HTTP stack
   (``file_download.py:xet_get``) that honours none of the Python timeouts.
   The lever is :func:`without_xet`, applied to metadata reads only: the files
   in question are kilobytes of JSON, where Xet's chunk deduplication buys
   nothing, so declining it costs nothing and restores the bounded HTTP path
   (``file_download.py`` falls through to ``http_get`` when Xet is off).
   Weight downloads keep Xet.

3. **A failed lookup was repeated, because only *success* was memoised.**
   ``Recipe.estimate_vram`` writes detected values back into ``metadata`` and
   re-runs detection when they are absent — so an unreachable Hub is re-asked on
   every one of the three estimates a single ``sparkrun run`` performs.  The
   levers are the negative memo and the budget below.

4. **A per-request ceiling does not bound a *lookup*.**  Every call goes through
   ``utils/_http.http_backoff``, which retries connection errors, timeouts and
   5xx/429 five times with exponential backoff (1 s → 8 s).  One
   ``hf_hub_download`` against an unreachable endpoint measured ~40 s with the
   client bounded at 3 s, and none of it is reachable from sparkrun: the retry
   policy is fixed at the call sites inside ``file_download`` and ``paginate``.
   The lever is :func:`_run_with_deadline` — the lookup runs on a daemon thread
   the caller stops waiting on.  This is the one layer that is structural rather
   than cooperative, and it is what makes the guarantee below hold no matter
   what a future ``huggingface_hub`` does inside the call.

**The guarantee: the advisory phase costs at most ``hub.metadata_budget_s``.**
The budget is wall-clock spent inside Hub metadata lookups (not since process
start); each lookup is deadlined at whatever remains, and running out trips a
process-wide breaker so the dozen-odd remaining lookups are skipped rather than
each paying their own timeout.

The breaker is deliberately *not* tripped by an exception — a 404 for
``hf_quant_config.json`` is the normal outcome for most models and costs
milliseconds.  Only time counts, because only time is what the user is losing.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_HUB_METADATA_BUDGET_S",
    "DEFAULT_HUB_TIMEOUT_S",
    "configure_hub_client",
    "disable_hub_metadata",
    "hub_degraded_message",
    "hub_metadata_call",
    "hub_unreachable",
    "reset_hub_state",
    "without_xet",
]

T = TypeVar("T")

#: Per-request ceiling handed to the shared httpx client, in seconds.
#:
#: Applied to connect / read / write but **not** pool.  httpx's ``read`` bounds
#: the gap between received bytes rather than the total transfer, so this is
#: safe for a multi-hour weight download; ``pool`` is left unbounded because
#: ``hf_hub_download`` fans out over a worker pool and a bounded pool wait would
#: turn concurrency into spurious failures.
DEFAULT_HUB_TIMEOUT_S = 15.0

#: Wall-clock budget for the *whole* advisory metadata phase, in seconds.
#:
#: Sized as "two dead requests and stop": at the per-request ceiling above, a
#: Hub that is not answering burns this in two calls and the breaker skips the
#: rest.  Raising it buys more patience for a slow-but-alive Hub; ``0`` or
#: negative means unbounded, matching ``readiness.*``.
DEFAULT_HUB_METADATA_BUDGET_S = 30.0


class _HubState:
    """Process-wide state for the advisory Hub phase.

    One lock covers all of it.  Contention is irrelevant (these are seconds-long
    network calls) and the alternative — separate locks for the memo, the ledger
    and the breaker — makes "spent the budget and tripped" two observable steps.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.client_configured = False
        self.spent_s = 0.0
        self.tripped_label: str | None = None
        #: (label, model_id, revision) -> None, for lookups already known to fail.
        self.negative: set[tuple[str, str, str | None]] = set()
        self.warned = False
        #: Set by ``--no-auto-detect``: skipping is what the user asked for, so
        #: it is kept distinct from the breaker, which reports a degradation.
        self.disabled = False


_STATE = _HubState()


def reset_hub_state() -> None:
    """Forget the budget ledger, breaker, and negative memo.

    For tests.  Does **not** un-install the bounded client: ``set_client_factory``
    closes the previous session, and re-installing per test would drop connection
    reuse for no gain.
    """
    with _STATE.lock:
        _STATE.spent_s = 0.0
        _STATE.tripped_label = None
        _STATE.negative.clear()
        _STATE.warned = False
        _STATE.disabled = False


def disable_hub_metadata() -> None:
    """Skip every advisory Hub lookup for the rest of this process.

    What ``--no-auto-detect`` sets.  It lives here rather than as an
    ``auto_detect=False`` argument threaded down because the decision is
    process-wide and the call sites are not: ``estimate_vram`` runs from host
    resolution, the banner, the scheduling pass and telemetry, and a flag that
    reached three of those four would look like it worked while still hanging.
    """
    with _STATE.lock:
        _STATE.disabled = True


# ---------------------------------------------------------------------------
# Layer 1 — bound the library's shared HTTP client
# ---------------------------------------------------------------------------


def _resolve_timeout_s(config: Any = None) -> float:
    """Per-request ceiling from ``hub.timeout_s``, or the default."""
    if config is None:
        try:
            from sparkrun.core.config import SparkrunConfig

            config = SparkrunConfig()
        except Exception:  # noqa: BLE001 — an unreadable config must not un-bound the client
            return DEFAULT_HUB_TIMEOUT_S
    value = getattr(config, "hub_timeout_s", None)
    if not isinstance(value, (int, float)) or value <= 0 or math.isinf(value):
        return DEFAULT_HUB_TIMEOUT_S
    return float(value)


def configure_hub_client(config: Any = None) -> None:
    """Install a timeout-bearing HTTP client for ``huggingface_hub``.

    Idempotent and safe to call from any Hub entry point.  Failure is not fatal:
    a future ``huggingface_hub`` that moves ``default_client_factory`` leaves the
    library on its own (unbounded) client, which is exactly today's behaviour —
    so this can only ever improve on the status quo, never break a working setup.
    """
    with _STATE.lock:
        if _STATE.client_configured:
            return
        _STATE.client_configured = True  # one attempt per process, success or not

        try:
            import httpx
            from huggingface_hub import set_client_factory

            # Private, but the only way to inherit the library's own event hooks
            # and redirect policy; building a bare client would silently drop the
            # request hook that stamps the user-agent and auth telemetry headers.
            from huggingface_hub.utils._http import default_client_factory

            timeout_s = _resolve_timeout_s(config)

            def _bounded_client_factory() -> "httpx.Client":
                client = default_client_factory()
                client.timeout = httpx.Timeout(
                    connect=timeout_s,
                    read=timeout_s,
                    write=timeout_s,
                    pool=None,
                )
                return client

            set_client_factory(_bounded_client_factory)
            _align_download_timeouts(timeout_s)
            logger.debug("Bounded huggingface_hub HTTP client at %.1fs per request", timeout_s)
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not bound the huggingface_hub HTTP client: %s", e)


def _align_download_timeouts(timeout_s: float) -> None:
    """Point ``hf_hub_download``'s own ceilings at ``hub.timeout_s`` too.

    The client default only governs calls that pass no timeout of their own —
    ``model_info``, ``list_repo_tree``, ``paginate``.  ``hf_hub_download`` passes
    ``HF_HUB_ETAG_TIMEOUT`` / ``HF_HUB_DOWNLOAD_TIMEOUT`` explicitly, which
    *override* the client, so without this a configured 3 s ceiling silently
    stayed 10 s for exactly the calls sparkrun makes most.  Both are read through
    the module at call time, so assigning them takes effect.

    An explicit environment variable wins: it is the operator speaking about this
    one process, which outranks a config file, and it is also how the library
    documents the knob.
    """
    import os

    try:
        from huggingface_hub import constants
    except ImportError:  # pragma: no cover — huggingface_hub is a required dep
        return

    if not os.environ.get("HF_HUB_ETAG_TIMEOUT"):
        constants.HF_HUB_ETAG_TIMEOUT = timeout_s
    # Bounds the gap between received bytes, not the total transfer, so a long
    # weight download is unaffected.
    if not os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT"):
        constants.HF_HUB_DOWNLOAD_TIMEOUT = timeout_s


# ---------------------------------------------------------------------------
# Layer 2 — decline Xet for metadata reads
# ---------------------------------------------------------------------------


@contextmanager
def without_xet() -> Iterator[None]:
    """Route Hub transfers inside the block through plain, bounded HTTP.

    ``huggingface_hub.constants.HF_HUB_DISABLE_XET`` is read at *call* time
    (``utils/_runtime.is_xet_available``, ``file_download``), so flipping the
    module attribute takes effect for the enclosed calls and restores after.

    Like the surrounding ``disable_progress_bars()`` / ``enable_progress_bars()``
    pairs this toggles process-global library state, so it must not wrap a block
    that also performs a weight download on another thread.  Every caller is a
    single small-JSON read on the launch path, which satisfies that.
    """
    try:
        from huggingface_hub import constants
    except ImportError:  # pragma: no cover — huggingface_hub is a required dep
        yield
        return

    previous = getattr(constants, "HF_HUB_DISABLE_XET", False)
    constants.HF_HUB_DISABLE_XET = True
    try:
        yield
    finally:
        constants.HF_HUB_DISABLE_XET = previous


# ---------------------------------------------------------------------------
# Layer 3 — budget, breaker, negative memo
# ---------------------------------------------------------------------------


def _resolve_budget_s(config: Any = None) -> float:
    """Advisory-phase budget from ``hub.metadata_budget_s``, or the default."""
    if config is None:
        try:
            from sparkrun.core.config import SparkrunConfig

            config = SparkrunConfig()
        except Exception:  # noqa: BLE001
            return DEFAULT_HUB_METADATA_BUDGET_S
    value = getattr(config, "hub_metadata_budget_s", None)
    if not isinstance(value, (int, float)):
        return DEFAULT_HUB_METADATA_BUDGET_S
    if math.isinf(value) or value <= 0:
        return math.inf
    return float(value)


def hub_unreachable() -> bool:
    """Whether the advisory budget is spent and further lookups are being skipped."""
    with _STATE.lock:
        return _STATE.tripped_label is not None


def hub_degraded_message() -> str | None:
    """A one-line explanation of the skip, or ``None`` if nothing was skipped.

    Returned once per process — the caller that gets the string owns printing
    it.  Fifteen advisory lookups share one breaker, so repeating the notice per
    lookup would bury the launch banner under the same paragraph fifteen times.
    """
    with _STATE.lock:
        if _STATE.tripped_label is None or _STATE.warned:
            return None
        _STATE.warned = True
        spent = _STATE.spent_s
    return (
        "Hugging Face Hub did not respond within %.0fs (last attempt: %s); continuing "
        "without model metadata. The VRAM estimate and fit table may be incomplete; "
        "the launch itself is unaffected.\n"
        "  Set HF_HUB_OFFLINE=1 to skip Hub lookups entirely, pass --no-auto-detect, "
        "or raise hub.metadata_budget_s in config.yaml." % (spent, _STATE.tripped_label)
    )


def hub_metadata_call(
    label: str,
    model_id: str,
    revision: str | None,
    fn: Callable[[], T | None],
    *,
    config: Any = None,
) -> T | None:
    """Run one advisory Hub metadata lookup under the process-wide budget.

    Returns ``fn()``'s result, or ``None`` when the lookup was skipped (breaker
    already tripped, or this call exhausted the budget) or already known to fail.

    *label* names the lookup for the diagnostic line and, with *model_id* and
    *revision*, keys the negative memo.  ``fn`` must be side-effect-free from the
    caller's point of view — it is not guaranteed to run.

    Each lookup is deadlined at whatever remains of the budget, so the phase as a
    whole cannot exceed it — including a compound lookup like the safetensors
    size, which makes up to three Hub round trips behind one label.

    A ``None`` result is memoised regardless of *why* it was ``None``.  That is
    deliberate and covers both halves of the problem: a repo genuinely without
    ``hf_quant_config.json`` should not be re-asked three times per launch, and
    neither should an unreachable Hub.  Only negatives are memoised — successes
    are already carried forward by ``Recipe.estimate_vram``'s metadata
    write-back, which is the documented mechanism for exactly that.
    """
    configure_hub_client(config)

    key = (label, model_id, revision)
    with _STATE.lock:
        if _STATE.disabled:
            logger.debug("Skipping %s for %s: Hub auto-detection disabled", label, model_id)
            return None
        if _STATE.tripped_label is not None:
            logger.debug("Skipping %s for %s: Hub metadata budget already spent", label, model_id)
            return None
        if key in _STATE.negative:
            logger.debug("Skipping %s for %s: previously unavailable in this process", label, model_id)
            return None
        budget_s = _resolve_budget_s(config)
        remaining = budget_s - _STATE.spent_s
    if remaining <= 0:
        _trip(label)
        return None

    started = time.monotonic()
    try:
        # The Xet toggle is flipped on *this* thread, not inside the worker: an
        # abandoned worker restoring process-global library state at an arbitrary
        # later moment is exactly the kind of action a timed-out lookup must not
        # still be able to take.
        with without_xet():
            outcome = _run_with_deadline(fn, remaining, label)
    finally:
        elapsed = time.monotonic() - started
        with _STATE.lock:
            _STATE.spent_s += elapsed

    if outcome is _TIMED_OUT:
        _trip(label)
        return None

    if outcome is None:
        with _STATE.lock:
            _STATE.negative.add(key)

    with _STATE.lock:
        exhausted = _STATE.spent_s >= budget_s and _STATE.tripped_label is None
    if exhausted:
        _trip(label)

    return outcome


#: Returned by :func:`_run_with_deadline` when the lookup outlived its deadline.
#: Distinct from ``None``, which is a lookup that answered "not available".
_TIMED_OUT = object()


def _run_with_deadline(fn: Callable[[], T | None], deadline_s: float, label: str) -> Any:
    """Run *fn*, giving up after *deadline_s* seconds.

    Enforced from outside the call because it cannot be enforced from within:
    ``huggingface_hub`` retries every request five times with exponential
    backoff, and the policy is fixed at its call sites — so a lookup routinely
    outruns the per-request ceiling by an order of magnitude.

    Giving up abandons the worker rather than cancelling it; Python has no way to
    interrupt a thread blocked in a socket read.  That costs one parked daemon
    thread, which burns no CPU and cannot hold the interpreter open at exit.  The
    alternative — waiting for a call that has already proven it will not
    return — is the bug (issue #278).  It is bounded in practice by the breaker:
    the first timeout stops any further lookup from starting, so a command leaks
    at most one such thread.

    An unbounded deadline runs *fn* inline, so opting out of the budget is
    byte-identical to the behaviour before this module existed.
    """
    if math.isinf(deadline_s):
        return fn()

    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 — re-raised on the calling thread
            box["error"] = e

    worker = threading.Thread(target=_target, name="sparkrun-hub-%s" % label, daemon=True)
    worker.start()
    worker.join(deadline_s)

    if worker.is_alive():
        logger.debug("Hub lookup %r exceeded its %.1fs deadline; abandoning it", label, deadline_s)
        return _TIMED_OUT
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _trip(label: str) -> None:
    """Open the breaker, naming the lookup that spent the last of the budget."""
    with _STATE.lock:
        if _STATE.tripped_label is not None:
            return
        _STATE.tripped_label = label
        spent = _STATE.spent_s
    # INFO, not WARNING: the user-facing channel is :func:`hub_degraded_message`,
    # which says considerably more and says it once.  Logging at WARNING too
    # printed both, with the terser one first.
    logger.info(
        "Hugging Face Hub metadata budget exhausted after %.1fs (at %s); skipping remaining lookups",
        spent,
        label,
    )
