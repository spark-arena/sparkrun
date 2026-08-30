"""The HuggingFace Hub metadata seam: bounded client, budget, breaker, memo.

Regression coverage for issue #278 — ``sparkrun run`` hanging indefinitely on a
half-closed Hub connection, printing nothing.
"""

from __future__ import annotations

import math
import time
from types import SimpleNamespace

import pytest

from sparkrun.models import hub


@pytest.fixture(autouse=True)
def _fresh_hub_state():
    hub.reset_hub_state()
    yield
    hub.reset_hub_state()


def _config(timeout_s=None, budget_s=None):
    """A stand-in for SparkrunConfig exposing only the two knobs the seam reads."""
    return SimpleNamespace(
        hub_timeout_s=timeout_s if timeout_s is not None else hub.DEFAULT_HUB_TIMEOUT_S,
        hub_metadata_budget_s=budget_s if budget_s is not None else hub.DEFAULT_HUB_METADATA_BUDGET_S,
    )


# ---------------------------------------------------------------------------
# Layer 1 — the shared client must carry a timeout
# ---------------------------------------------------------------------------


def test_configure_hub_client_installs_a_bounded_client(monkeypatch):
    """huggingface_hub ships ``httpx.Client(timeout=None)``; we must replace it.

    This is the defect that made ``model_info`` / ``list_repo_tree`` able to
    block forever — neither accepts a usable per-call timeout, so the shared
    client is the only lever.
    """
    installed = {}

    def _capture(factory):
        installed["client"] = factory()

    monkeypatch.setattr("huggingface_hub.set_client_factory", _capture)
    hub._STATE.client_configured = False

    hub.configure_hub_client(_config(timeout_s=7.0))

    timeout = installed["client"].timeout
    assert timeout.connect == 7.0
    assert timeout.read == 7.0
    assert timeout.write == 7.0
    # Unbounded on purpose: hf_hub_download fans out over a worker pool, and a
    # bounded pool wait turns concurrency into spurious failures.
    assert timeout.pool is None


def test_configure_hub_client_preserves_library_event_hooks(monkeypatch):
    """Inheriting hf's own factory keeps the user-agent/auth request hook."""
    installed = {}
    monkeypatch.setattr("huggingface_hub.set_client_factory", lambda f: installed.update(client=f()))
    hub._STATE.client_configured = False

    hub.configure_hub_client(_config())

    assert "request" in installed["client"].event_hooks


def test_configure_hub_client_is_idempotent(monkeypatch):
    calls = []
    monkeypatch.setattr("huggingface_hub.set_client_factory", calls.append)
    hub._STATE.client_configured = False

    hub.configure_hub_client(_config())
    hub.configure_hub_client(_config())

    assert len(calls) == 1


def test_configure_hub_client_survives_a_library_refactor(monkeypatch):
    """A future huggingface_hub without the private factory must not break startup.

    Degrading here leaves the library on its own client, i.e. exactly today's
    behaviour — so this can only ever fail to improve things, never regress them.
    """

    def _boom(_factory):
        raise RuntimeError("set_client_factory moved")

    monkeypatch.setattr("huggingface_hub.set_client_factory", _boom)
    hub._STATE.client_configured = False

    hub.configure_hub_client(_config())  # must not raise


# ---------------------------------------------------------------------------
# Layer 2 — metadata reads decline Xet
# ---------------------------------------------------------------------------


def test_metadata_calls_run_with_xet_disabled():
    """hf_xet is a Rust HTTP stack that honours none of the Python timeouts.

    Small-JSON metadata gains nothing from Xet's chunk dedup, so declining it
    for these reads restores the bounded ``http_get`` path at no cost.
    """
    from huggingface_hub import constants

    seen = []
    hub.hub_metadata_call("probe", "org/model", None, lambda: seen.append(constants.HF_HUB_DISABLE_XET) or "ok")

    assert seen == [True]


def test_without_xet_restores_the_previous_value():
    from huggingface_hub import constants

    constants.HF_HUB_DISABLE_XET = False
    try:
        with hub.without_xet():
            assert constants.HF_HUB_DISABLE_XET is True
        assert constants.HF_HUB_DISABLE_XET is False
    finally:
        constants.HF_HUB_DISABLE_XET = False


def test_without_xet_restores_on_exception():
    from huggingface_hub import constants

    constants.HF_HUB_DISABLE_XET = False
    try:
        with pytest.raises(ValueError):
            with hub.without_xet():
                raise ValueError("boom")
        assert constants.HF_HUB_DISABLE_XET is False
    finally:
        constants.HF_HUB_DISABLE_XET = False


def test_download_timeouts_are_aligned_with_the_client(monkeypatch):
    """hf_hub_download passes its own ceilings, which override the client default.

    Without this, a configured 3s ceiling silently stayed at the library's 10s
    for exactly the calls sparkrun makes most.
    """
    from huggingface_hub import constants

    monkeypatch.delenv("HF_HUB_ETAG_TIMEOUT", raising=False)
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.setattr(constants, "HF_HUB_ETAG_TIMEOUT", 10)
    monkeypatch.setattr(constants, "HF_HUB_DOWNLOAD_TIMEOUT", 10)

    hub._align_download_timeouts(3.0)

    assert constants.HF_HUB_ETAG_TIMEOUT == 3.0
    assert constants.HF_HUB_DOWNLOAD_TIMEOUT == 3.0


def test_an_explicit_env_timeout_wins_over_config(monkeypatch):
    """The operator speaking about this one process outranks a config file."""
    from huggingface_hub import constants

    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "42")
    monkeypatch.setattr(constants, "HF_HUB_ETAG_TIMEOUT", 42)

    hub._align_download_timeouts(3.0)

    assert constants.HF_HUB_ETAG_TIMEOUT == 42


# ---------------------------------------------------------------------------
# Layer 4 — a deadline the library cannot talk us out of
# ---------------------------------------------------------------------------


def test_a_hung_lookup_is_abandoned_at_its_deadline():
    """The decisive layer: huggingface_hub retries 5x with backoff internally.

    A per-request ceiling therefore does not bound a *lookup* — one measured
    ~40s against an unreachable endpoint with the client bounded at 3s — so the
    deadline has to be imposed from outside the call.
    """
    import threading

    release = threading.Event()
    started = threading.Event()

    def _hang():
        started.set()
        release.wait(30)  # stands in for a socket read that never returns
        return "too late"

    t0 = time.monotonic()
    try:
        result = hub.hub_metadata_call("config.json", "org/m", None, _hang, config=_config(budget_s=0.3))
        elapsed = time.monotonic() - t0

        assert result is None
        assert started.is_set(), "the lookup must actually have been attempted"
        assert elapsed < 5.0, "the caller must not wait for the hung lookup"
        assert hub.hub_unreachable()
    finally:
        release.set()


def test_an_abandoned_lookup_cannot_strand_the_process():
    """Abandoning means a parked daemon thread — never a non-daemon one."""
    import threading

    release = threading.Event()
    live = []

    def _hang():
        live.append(threading.current_thread())
        release.wait(30)

    try:
        hub.hub_metadata_call("config.json", "org/m", None, _hang, config=_config(budget_s=0.3))
        assert live and live[0].daemon
    finally:
        release.set()


def test_the_breaker_bounds_the_number_of_abandoned_threads():
    """One command leaks at most one worker: the first timeout stops the rest."""
    import threading

    release = threading.Event()
    live = []

    def _hang():
        live.append(threading.current_thread())
        release.wait(30)

    cfg = _config(budget_s=0.3)
    try:
        for name in ("config.json", "hf_quant_config.json", "safetensors size", "safetensors params"):
            hub.hub_metadata_call(name, "org/m", None, _hang, config=cfg)
        assert len(live) == 1
    finally:
        release.set()


def test_an_unbounded_budget_runs_inline():
    """Opting out must be byte-identical to the behaviour before this module."""
    import threading

    seen = []
    hub.hub_metadata_call(
        "config.json",
        "org/m",
        None,
        lambda: seen.append(threading.current_thread()) or "ok",
        config=_config(budget_s=math.inf),
    )

    assert seen == [threading.current_thread()]


def test_an_exception_crosses_the_worker_thread():
    """Callers rely on their own try/except; the deadline must not swallow it."""
    with pytest.raises(RuntimeError, match="404"):
        hub.hub_metadata_call(
            "hf_quant_config.json",
            "org/m",
            None,
            lambda: (_ for _ in ()).throw(RuntimeError("404")),
            config=_config(budget_s=30.0),
        )


# ---------------------------------------------------------------------------
# Layer 3 — budget, breaker, negative memo
# ---------------------------------------------------------------------------


def test_a_slow_lookup_trips_the_breaker_and_skips_the_rest(monkeypatch):
    """One command makes ~15 advisory lookups; a dead Hub must not cost 15 timeouts."""
    clock = {"t": 0.0}
    monkeypatch.setattr(hub.time, "monotonic", lambda: clock["t"])

    def _hang():
        clock["t"] += 20.0  # a bounded-but-timed-out request
        return None

    cfg = _config(budget_s=30.0)

    assert hub.hub_metadata_call("config.json", "org/m", None, _hang, config=cfg) is None
    assert not hub.hub_unreachable()

    assert hub.hub_metadata_call("safetensors size", "org/m", None, _hang, config=cfg) is None
    assert hub.hub_unreachable()

    ran = []
    assert hub.hub_metadata_call("model visibility", "org/m", None, lambda: ran.append(1) or "public", config=cfg) is None
    assert ran == [], "breaker must skip the call, not merely discard its result"


def test_the_breaker_reports_once(monkeypatch):
    """Fifteen lookups share one breaker; the notice must not repeat per lookup."""
    clock = {"t": 0.0}
    monkeypatch.setattr(hub.time, "monotonic", lambda: clock["t"])
    cfg = _config(budget_s=1.0)

    def _hang():
        clock["t"] += 5.0
        return None

    assert hub.hub_degraded_message() is None, "nothing skipped yet"
    hub.hub_metadata_call("config.json", "org/m", None, _hang, config=cfg)

    first = hub.hub_degraded_message()
    assert first is not None
    assert "config.json" in first
    assert "--no-auto-detect" in first
    assert hub.hub_degraded_message() is None


def test_a_negative_result_is_not_re_asked():
    """The #278 amplification: three estimate_vram calls per run, each refetching.

    ``Recipe.estimate_vram`` only writes *successful* detection back into
    metadata, so an unreachable Hub failed ``needs_detection`` every time and the
    whole fetch sequence reran on all three passes.
    """
    calls = []

    def _missing():
        calls.append(1)
        return None

    for _ in range(3):
        assert hub.hub_metadata_call("hf_quant_config.json", "org/m", None, _missing) is None

    assert calls == [1]


def test_the_negative_memo_is_keyed_per_lookup_and_repo():
    calls = []

    def _missing():
        calls.append(1)
        return None

    hub.hub_metadata_call("config.json", "org/a", None, _missing)
    hub.hub_metadata_call("config.json", "org/b", None, _missing)
    hub.hub_metadata_call("safetensors size", "org/a", None, _missing)
    hub.hub_metadata_call("config.json", "org/a", "refs/pr/1", _missing)

    assert len(calls) == 4


def test_a_success_is_returned_and_not_memoised():
    """Successes are carried forward by estimate_vram's metadata write-back.

    Memoising them here too would duplicate that mechanism and hand callers a
    shared mutable dict.
    """
    calls = []

    def _ok():
        calls.append(1)
        return {"model_type": "llama"}

    assert hub.hub_metadata_call("config.json", "org/m", None, _ok) == {"model_type": "llama"}
    assert hub.hub_metadata_call("config.json", "org/m", None, _ok) == {"model_type": "llama"}
    assert len(calls) == 2


def test_an_exception_does_not_trip_the_breaker():
    """A 404 for hf_quant_config.json is the normal case for most repos.

    Only elapsed time trips the breaker, because only time is what the user is
    losing — classifying library exceptions would make a missing optional file
    look like an outage.
    """
    with pytest.raises(RuntimeError):
        hub.hub_metadata_call("hf_quant_config.json", "org/m", None, lambda: (_ for _ in ()).throw(RuntimeError("404")))

    assert not hub.hub_unreachable()
    assert hub.hub_metadata_call("config.json", "org/m", None, lambda: "ok") == "ok"


def test_disable_hub_metadata_skips_silently():
    """``--no-auto-detect`` is a request, not a degradation — no warning."""
    ran = []
    hub.disable_hub_metadata()

    assert hub.hub_metadata_call("config.json", "org/m", None, lambda: ran.append(1) or "x") is None
    assert ran == []
    assert hub.hub_degraded_message() is None


def test_a_non_positive_budget_means_unbounded(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(hub.time, "monotonic", lambda: clock["t"])
    cfg = _config(budget_s=math.inf)

    for i in range(5):
        clock["t"] += 100.0
        assert hub.hub_metadata_call("probe", "org/m%d" % i, None, lambda: "ok", config=cfg) == "ok"

    assert not hub.hub_unreachable()


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_config_knobs_read_from_the_hub_section(tmp_path):
    from sparkrun.core.config import SparkrunConfig

    path = tmp_path / "config.yaml"
    path.write_text("hub:\n  timeout_s: 3.5\n  metadata_budget_s: 12\n")
    config = SparkrunConfig(config_path=path)

    assert config.hub_timeout_s == 3.5
    assert config.hub_metadata_budget_s == 12.0


def test_config_defaults_when_unset(tmp_path):
    from sparkrun.core.config import SparkrunConfig

    path = tmp_path / "config.yaml"
    path.write_text("cache_dir: /tmp/x\n")
    config = SparkrunConfig(config_path=path)

    assert config.hub_timeout_s == hub.DEFAULT_HUB_TIMEOUT_S
    assert config.hub_metadata_budget_s == hub.DEFAULT_HUB_METADATA_BUDGET_S


def test_a_non_positive_timeout_does_not_un_bound_the_client(tmp_path):
    """Unlike ``readiness.*``, "no timeout" is the defect and has no spelling."""
    from sparkrun.core.config import SparkrunConfig

    path = tmp_path / "config.yaml"
    path.write_text("hub:\n  timeout_s: 0\n")

    assert SparkrunConfig(config_path=path).hub_timeout_s == hub.DEFAULT_HUB_TIMEOUT_S


def test_a_non_positive_budget_config_means_unbounded(tmp_path):
    from sparkrun.core.config import SparkrunConfig

    path = tmp_path / "config.yaml"
    path.write_text("hub:\n  metadata_budget_s: 0\n")

    assert SparkrunConfig(config_path=path).hub_metadata_budget_s == math.inf


# ---------------------------------------------------------------------------
# The fetchers actually route through the seam
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "name", "args"),
    [
        ("sparkrun.models.vram", "fetch_model_config", ("org/m",)),
        ("sparkrun.models.vram", "fetch_safetensors_size", ("org/m",)),
        ("sparkrun.models.vram", "fetch_safetensors_params", ("org/m",)),
        ("sparkrun.models.quantization", "fetch_hf_quant_config", ("org/m",)),
    ],
)
def test_advisory_fetchers_are_skipped_once_the_breaker_is_open(module, name, args, monkeypatch):
    import importlib

    mod = importlib.import_module(module)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **_kw: pytest.fail("no Hub call may be made once the breaker is open"),
    )
    monkeypatch.setattr(
        "huggingface_hub.model_info",
        lambda **_kw: pytest.fail("no Hub call may be made once the breaker is open"),
    )
    hub._trip("test")

    assert getattr(mod, name)(*args) is None


def test_visibility_fails_closed_when_the_breaker_is_open(monkeypatch):
    """A skipped visibility probe may only ever withhold the model name.

    That it fails closed is what makes it safe to budget at all — telemetry
    reports ``unknown`` rather than leaking a possibly-private repo id.
    """
    from sparkrun.models import vram

    monkeypatch.setattr(vram, "_VISIBILITY_MEMO", {})
    monkeypatch.setattr(
        "huggingface_hub.model_info",
        lambda **_kw: pytest.fail("no Hub call may be made once the breaker is open"),
    )
    hub._trip("test")

    assert vram.fetch_model_visibility("org/m") == vram.MODEL_VISIBILITY_UNKNOWN
