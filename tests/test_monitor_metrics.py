"""Tests for cli/_monitor_metrics.py: engine log lines -> metrics footer markup.

Fixture lines are verbatim engine output; the upstream file that emits each one
is named in its comment.  Extend the table only from a line read in that source.
"""

from __future__ import annotations

import re

import pytest

from sparkrun.cli._monitor_metrics import (
    _METRICS,
    _PAGE_TAIL,
    _SPARK_GLYPHS,
    RuntimeMetricsScraper,
    render_metrics_footer,
)

_PALETTE = tuple(f"c{i}" for i in range(11))

# (id, verbatim line, expected {metric: value}), rounded to one decimal.
_FIXTURES: tuple[tuple[str, str, dict[str, float]], ...] = (
    # vllm/v1/metrics/loggers.py: one line, optional "Engine %03d:" prefix
    (
        "vllm-v1",
        "INFO 10-17 08:12:34 [loggers.py:331] Engine 000: Avg prompt throughput: 123.4 tokens/s, "
        "Avg generation throughput: 45.6 tokens/s, Running: 3 reqs, Waiting: 1 reqs, "
        "GPU KV cache usage: 12.3%, Prefix cache hit rate: 45.6%",
        {"gen": 45.6, "run": 3.0, "kv": 12.3, "queue": 1.0, "prompt": 123.4, "hit": 45.6},
    ),
    # vllm/engine/metrics.py (v0): "Pending" wording, no prefix-hit field
    (
        "vllm-v0",
        "INFO 10-17 08:12:34 [metrics.py:386] Avg prompt throughput: 12.5 tokens/s, "
        "Avg generation throughput: 6.7 tokens/s, Running: 2 reqs, Swapped: 0 reqs, Pending: 4 reqs, "
        "GPU KV cache usage: 55.0%, CPU KV cache usage: 0.0%.",
        {"gen": 6.7, "run": 2.0, "kv": 55.0, "queue": 4.0, "prompt": 12.5},
    ),
    # vllm/v1/spec_decode/metrics.py: its own line, no engine stats
    (
        "vllm-spec",
        "INFO 10-17 08:12:34 [metrics.py:118] SpecDecoding metrics: Mean acceptance length: 2.31, "
        "Accepted throughput: 40.11 tokens/s, Drafted throughput: 55.12 tokens/s, Accepted: 8011 tokens, "
        "Drafted: 11023 tokens, Per-position acceptance rate: 0.512, 0.201, Avg Draft acceptance rate: 72.7%",
        {"accept": 72.7},
    ),
    # srt/managers/scheduler_components/metrics_reporter.py: decode line ("token
    # usage" is a pool fraction, so 0.52 -> 52%)
    (
        "sglang-decode",
        "[2026-01-12 23:01:05 TP0] INFO Decode batch, #running-req: 1000, #token: 336859, "
        "token usage: 0.52, cuda graph: False, gen throughput (token/s): 245.93, #queue-req: 625",
        {"gen": 245.9, "run": 1000.0, "kv": 52.0, "queue": 625.0},
    ),
    # same file, older spelling with the period after "batch"
    (
        "sglang-decode-old-spelling",
        "[2026-01-12 23:01:05 TP0] INFO Decode batch. #running-req: 1000, #token: 336859, "
        "token usage: 0.52, cuda graph: False, gen throughput (token/s): 245.93, #queue-req: 625",
        {"gen": 245.9, "run": 1000.0, "kv": 52.0, "queue": 625.0},
    ),
    # spec-decoding variant of the decode line ("accept rate" is 0..1)
    (
        "sglang-spec-decode",
        "[2026-01-12 23:01:05 TP0] INFO Decode batch. #running-req: 1000, #token: 336859, "
        "token usage: 0.52, cuda graph: False, accept len: 2.53, accept rate: 0.42, "
        "gen throughput (token/s): 245.93, #queue-req: 625",
        {"gen": 245.9, "run": 1000.0, "kv": 52.0, "queue": 625.0, "accept": 42.0},
    ),
    # prefill line: the hit rate is cached / (new + cached) = 2971 / 3292,
    # and the pp rate is the engine's own input throughput field, verbatim.
    (
        "sglang-prefill",
        "[2026-10-17 08:12:34 TP0] INFO Prefill batch. #new-seq: 1, #new-token: 321, #cached-token: 2971, "
        "token usage: 0.10, #running-req: 2, #queue-req: 0, cuda graph: True, input throughput (token/s): 536.67",
        {"hit": 90.2, "prompt": 536.7, "kv": 10.0, "run": 2.0, "queue": 0.0},
    ),
    # Newer SGLang splits the pool reading into "full token usage" / "swa token
    # usage" on the same line — captured verbatim from a serving container on the
    # Spark (2026-10-17). The two pools have separate capacities, so "kv" reports
    # the fuller one (here a tie); the engine does the same in
    # PoolStats.get_max_pool_usage().
    (
        "sglang-decode-full-swa",
        "[2026-10-17 07:04:26 TP0] Decode batch, #running-req: 2, #full token: 180992, full token usage: 0.02, "
        "#swa token: 1280, swa token usage: 0.02, accept len: 4.27, accept rate: 0.65, cuda graph: True, "
        "gen throughput (token/s): 96.76, #queue-req: 0",
        {"gen": 96.8, "run": 2.0, "kv": 2.0, "queue": 0.0, "accept": 65.0},
    ),
    # Same build, verbatim (Spark container, 2026-10-15): the sliding-window pool
    # is the fuller one, so kv must read 10%, not the full pool's 0%. Half the
    # samples in that container's log have swa above full.
    (
        "sglang-prefill-swa-fuller",
        "[2026-10-15 04:26:43 TP0] Prefill batch, #new-seq: 1, #new-token: 4096, #cached-token: 0, "
        "full token usage: 0.00, swa token usage: 0.10, #running-req: 0, #queue-req: 0, "
        "#pending-token: 3856, cuda graph: False, input throughput (token/s): 157.88",
        {"hit": 0.0, "prompt": 157.9, "kv": 10.0, "run": 0.0, "queue": 0.0},
    ),
    # Same build's prefill line, also verbatim from the serving container: the
    # hit rate must show as 0% (not be skipped) when nothing was cached, and
    # "#pending-token" must not reach the queue rule (only "#queue-req" does).
    (
        "sglang-prefill-full-swa",
        "[2026-10-14 22:02:55 TP0] Prefill batch, #new-seq: 1, #new-token: 193, #cached-token: 0, "
        "full token usage: 0.00, swa token usage: 0.00, #running-req: 0, #queue-req: 0, "
        "#pending-token: 0, cuda graph: False, input throughput (token/s): 7.13",
        {"hit": 0.0, "prompt": 7.1, "kv": 0.0, "run": 0.0, "queue": 0.0},
    ),
    # real server line (build 10430+, verbatim from a public log); the deployed
    # b809x images have neither this line nor print_timings_pp — see the module
    # docstring — so on those only the final block below fires
    (
        "llama-server-live-decode",
        "0.16.079.155 I slot print_timing: id  0 | task 0 | n_gen =    100, tg =  19.49 t/s, tg_3s =  19.68 t/s",
        {"gen": 19.7},
    ),
    # same line before tg_3s existed (builds b8931..b10430, verbatim): the
    # tg_3s rule cannot match it and must not swallow it either
    (
        "llama-server-live-decode-no-tg3s",
        "60.58.784.269 I slot print_timing: id  3 | task 25239 | n_decoded =    100, tg =  68.44 t/s",
        {"gen": 68.4},
    ),
    # live prompt processing, current builds (verbatim); needs >= 3 s of prompt
    # processing upstream before the engine prints it at all
    (
        "llama-server-live-prompt",
        "6.44.065.830 I slot print_timing: id 1 | task 1319 | prompt processing, n_tokens = 2048, progress = 0.05, "
        "t = 3.55 s / 576.84 tokens per second",
        {"prompt": 576.8},
    ),
    # common/sampling.cpp common_perf_print: the final pair, server wording
    (
        "llama-final-prompt-eval",
        "0.00.35.064 I common_perf_print: prompt eval time =     85.79 ms /    34 tokens (    2.52 ms per token,   396.31 tokens per second)",
        {"prompt": 396.3},
    ),
    # same pair: "runs" is the CLI wording, "tokens" the server's
    (
        "llama-final-eval-runs",
        "0.00.35.064 I common_perf_print: eval time =      45.67 ms /    16 runs   (     2.85 ms per token,   350.75 tokens per second)",
        {"gen": 350.8},
    ),
    (
        "llama-final-eval-tokens",
        "0.00.35.064 I common_perf_print: eval time =      45.67 ms /    16 tokens   (     2.85 ms per token,   350.75 tokens per second)",
        {"gen": 350.8},
    ),
    # b80xx/deployed final block: one SLT_INF call with embedded newlines, so the
    # timing lines themselves have no prefix (verbatim; build 5142 sample)
    (
        "llama-final-prompt-eval-b80xx",
        "prompt eval time =     221.09 ms /    21 tokens (   10.53 ms per token,    94.98 tokens per second)",
        {"prompt": 95.0},
    ),
    (
        "llama-final-eval-b80xx",
        "       eval time =    8471.73 ms /   261 tokens (   32.46 ms per token,    30.81 tokens per second)",
        {"gen": 30.8},
    ),
    # speculative decoding: master's INFO spelling (with a mean-len tail) and the
    # b80xx CONT-level spelling, which renders with no prefix and no level letter
    (
        "llama-draft-acceptance",
        "0.35.841.238 I slot print_timing: id  0 | task 0 | draft acceptance = 0.75000 (  291 accepted /   388 generated), "
        "mean len =  2.50",
        {"accept": 75.0},
    ),
    (
        "llama-draft-acceptance-rate-cont",
        "draft acceptance rate = 0.45213 (  170 accepted /   376 generated)",
        {"accept": 45.2},
    ),
    # llama-cli's per-turn info line (build 11706, default verbosity): the CLI's
    # eval-time lines are gated behind --perf, so this is its only default rate
    (
        "llama-cli-bracket-rates",
        "[ Prompt: 321.4 t/s | Generation: 87.9 t/s ]",
        {"prompt": 321.4, "gen": 87.9},
    ),
    # Aneda (picom dsgb10 + GLP): the engine's periodic line, read in a live log
    # 2026-11-24. tok_out/tok_in are the interval's counts and t its measured
    # seconds, so the rates divide by t; reqs_stock + reqs_steered is what the
    # rank served in the interval.
    (
        "aneda-stats",
        "stats: t=10 rank=0 tok_out=12 tok_in=4824 kv_util=0.000 kv_hit=0.000 "
        "draft_accept=0.000 queue=0 engram_hit=0.501 engram_used=0.000 reqs_stock=2 reqs_steered=2",
        {
            "gen": 1.2,
            "prompt": 482.4,
            "run": 4.0,
            "queue": 0.0,
            "kv": 0.0,
            "hit": 0.0,
            "accept": 0.0,
            "engram_hit": 50.1,
            "engram_used": 0.0,
        },
    ),
    (
        "aneda-stats-loaded",
        "stats: t=10 rank=0 tok_out=12 tok_in=362657 kv_util=0.082 kv_hit=0.994 "
        "draft_accept=0.286 queue=0 engram_hit=0.997 engram_used=0.344 reqs_stock=1 reqs_steered=0",
        {
            "gen": 1.2,
            "prompt": 36265.7,
            "run": 1.0,
            "queue": 0.0,
            "kv": 8.2,
            "hit": 99.4,
            "accept": 28.6,
            "engram_hit": 99.7,
            "engram_used": 34.4,
        },
    ),
)

# Lines that must never produce a sample.
_NOISE: tuple[str, ...] = (
    "docker: pull complete",
    "Traceback (most recent call last):",
    "INFO 10-17 08:12:34 [core.py:130] Starting vLLM server on port 8000",
    "0.00.35.064 I llama_model_load: vocab only",
    # deployed b809x build (default verbosity, 10x per long request): prompt
    # progress carries counts only, no rate, and matches no gate substring
    "slot update_slots: id  0 | task 9 | prompt processing progress, n_tokens = 2048, batch.n_tokens = 2048, progress = 0.258553",
    # the final block's tag line is emitted *before* the payload lines; on its
    # own it carries no metric at all
    "slot print_timing: id  0 | task 9 |",
    # KV capacity at init is not KV usage, and llama.cpp logs no usage figure —
    # in either direction (this pair is verbatim from the deployed build)
    "llama_kv_cache: size =  384.00 MiB ( 16384 cells,  24 layers,  2/2 seqs), K (f16):  192.00 MiB, V (f16):  192.00 MiB",
    "llama_kv_cache:      CUDA0 KV buffer size =   384.00 MiB",
    # host prompt-cache occupancy is not a hit rate
    "srv        update:  - cache state: 1 prompts, 1.643 MiB (limits: 8192.000 MiB, 32768 tokens, 698127 est)",
    # slot count at startup is static config, not a running/queued count: the
    # deployed build's spelling, then the master-era one (n_slots is otherwise
    # DEBUG-only)
    "srv    load_model: initializing slots, n_slots = 2",
    "0.05.701.511 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 131072, kv_unified = 'false'",
    # reuse signal and per-task bookkeeping: numbers, but no metric we render
    "slot get_availabl: id  0 | task -1 | selected slot by LCP similarity, sim_best = 0.999 (> 0.100 thold), f_keep = 0.975",
    "slot      release: id  0 | task 9 | stop processing: n_tokens = 8120, truncated = 0",
    "srv  log_server_r: done request: POST /completion 127.0.0.1 200",
    # Engram's n-gram hash-memory stats: its hit_rate is not a KV-cache hit rate
    # (see the boundary test below), and the underscore keeps it out of the gate.
    "[2026-10-17 21:17:43 TP0] Engram layer=1 lookups=305185 hit_rate=99.5% reads=1607 cache=2.0GiB(16-way) "
    "scales=0.0GiB threads=96 packed=True",
)


def _scrape(line: str, source: tuple[str, str] = ("h1", "c1")) -> dict[str, float]:
    scraper = RuntimeMetricsScraper()
    scraper.observe(source, line)
    scraper.commit()
    return {spec.key: round(value, 1) for spec in _METRICS if (value := scraper.latest(spec.key)) is not None}


def _plain(markup: str) -> str:
    """Markup-stripped footer text."""
    return re.sub(r"\[/?[^\]]*\]", "", markup)


@pytest.mark.parametrize(("line", "expected"), [(line, expected) for _id, line, expected in _FIXTURES], ids=[i[0] for i in _FIXTURES])
def test_scraper_extracts_metrics_from_real_runtime_lines(line: str, expected: dict[str, float]):
    assert _scrape(line) == expected


@pytest.mark.parametrize("line", _NOISE)
def test_scraper_ignores_lines_without_metric_shapes(line: str):
    assert _scrape(line) == {}


def test_scraper_aggregates_multi_source_sums_rates_and_maxes_percents():
    scraper = RuntimeMetricsScraper()
    scraper.observe(("h1", "r0"), "Avg generation throughput: 100.0 tokens/s, GPU KV cache usage: 40.0%")
    scraper.observe(("h1", "r1"), "Avg generation throughput: 50.0 tokens/s, GPU KV cache usage: 61.0%")
    scraper.commit()

    # Two ranks report the same metric: rates add up (the cluster total is what
    # a user reads), a 0-100 metric takes the worst rank (the pressure signal).
    assert scraper.latest("gen") == 150.0
    assert scraper.latest("kv") == 61.0
    assert scraper.history("gen") == [150.0]

    # A commit with nothing observed since the last one adds no duplicate point.
    scraper.commit()
    assert scraper.history("gen") == [150.0]


def test_scraper_same_source_samples_average_within_one_window():
    scraper = RuntimeMetricsScraper()
    scraper.observe(("h1", "c1"), "Avg generation throughput: 10.0 tokens/s")
    scraper.observe(("h1", "c1"), "Avg generation throughput: 20.0 tokens/s")
    scraper.commit()
    # Both prints land in one 30 s poll: the window point is their mean.
    assert scraper.latest("gen") == 15.0


def test_scraper_ring_is_bounded_and_keeps_newest():
    scraper = RuntimeMetricsScraper(history_points=3)
    for i in range(5):
        scraper.observe(("h1", "c1"), f"Avg generation throughput: {i}.0 tokens/s")
        scraper.commit()
    assert scraper.history("gen") == [2.0, 3.0, 4.0]


def test_scraper_reset_drops_every_sample():
    scraper = RuntimeMetricsScraper()
    scraper.observe(("h1", "c1"), "Avg generation throughput: 10.0 tokens/s")
    scraper.commit()
    scraper.reset()
    assert scraper.latest("gen") is None
    assert scraper.history("gen") == []
    scraper.commit()  # nothing pending after a reset
    assert scraper.history("gen") == []


def _sglang_prefill(stamp: str, new_tokens: int, throughput: float = 1.0) -> str:
    """An SGLang prefill line of the shape the serving container prints."""
    return (
        f"[{stamp} TP0] Prefill batch, #new-seq: 1, #new-token: {new_tokens}, #cached-token: 0, "
        f"full token usage: 0.00, swa token usage: 0.02, #running-req: 0, #queue-req: 0, #pending-token: 0, "
        f"cuda graph: False, input throughput (token/s): {throughput}"
    )


# Verbatim: the last two prints of one 25,344-token prompt chunked at 4096.
# Both land in the same second, so the engine divides by a millisecond gap; its
# field is the pp metric, so the footer reports it exactly as printed.
_SGLANG_PREFILL_SPIKE_PAIR: tuple[str, ...] = (
    "[2026-10-15 00:58:50 TP0] Prefill batch, #new-seq: 1, #new-token: 4096, #cached-token: 0, full token usage: 0.00, "
    "swa token usage: 0.02, #running-req: 0, #queue-req: 0, #pending-token: 603, cuda graph: False, "
    "input throughput (token/s): 10690.42",
    "[2026-10-15 00:58:50 TP0] Prefill batch, #new-seq: 1, #new-token: 603, #cached-token: 0, full token usage: 0.00, "
    "swa token usage: 0.01, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, "
    "input throughput (token/s): 88174.92",
)


def test_sglang_prefill_uses_the_engines_input_throughput_field():
    scraper = RuntimeMetricsScraper()
    for line in _SGLANG_PREFILL_SPIKE_PAIR:
        scraper.observe(("h1", "c1"), line)
        scraper.commit()

    # One point per poll, the logged pp value verbatim.
    assert scraper.history("prompt") == [10690.42, 88174.92]


def test_sglang_prefill_prompt_rate_is_per_source():
    scraper = RuntimeMetricsScraper()
    rank0, rank1 = ("h1", "r0"), ("h1", "r1")
    scraper.observe(rank0, _sglang_prefill("2026-10-15 00:58:46", 1024, 120.0))
    scraper.observe(rank0, _sglang_prefill("2026-10-15 00:58:48", 4096, 180.0))
    scraper.observe(rank1, _sglang_prefill("2026-10-15 00:58:56", 4096, 200.0))
    scraper.commit()

    # Rates add across ranks; each source's window point is its own mean.
    assert scraper.latest("prompt", sources={rank0}) == 150.0
    assert scraper.latest("prompt", sources={rank1}) == 200.0
    assert scraper.latest("prompt") == 350.0


def test_sglang_prefill_without_a_timestamp_still_reports_input_throughput():
    # The field is the only input the pp rule needs: no "[<ts> TPn]" prefix.
    scraped = _scrape(
        "Prefill batch, #new-token: 4096, token usage: 0.10, #running-req: 2, #queue-req: 0, input throughput (token/s): 500.0"
    )
    assert scraped == {"run": 2.0, "kv": 10.0, "queue": 0.0, "prompt": 500.0}


# TensorFold's per-request completion line, verbatim from the deployed
# workload's log (2026-10-17). tok/s is decode-only, prefill pure prefill
# seconds, accepted=a/b the accepted/drafted speculative tokens.
_TENSORFOLD_DONE = (
    "[tensorfold] done req-477975f9986c prompt=43 cached=0 thinking=True tokens=19 "
    "sha=e96a676a795c finish=stop tok/s=62.2 ttft=0.18s prefill=0.17s rounds=7 accepted=11/23"
)
# A one-token request: 0/0 speculative tokens and a decode rate of zero.
_TENSORFOLD_DONE_ONE_TOKEN = (
    "[tensorfold] done req-60459bc11fea prompt=52 cached=0 thinking=True tokens=1 "
    "sha=fd77ae3ed92c finish=length tok/s=0.0 ttft=0.18s prefill=0.18s rounds=0 accepted=0/0"
)
_TENSORFOLD_DONE_FASTER = (
    "[tensorfold] done req-0f5d78cbbc06 prompt=2048 cached=0 thinking=True tokens=128 "
    "sha=f9b521598a86 finish=length tok/s=64.6 ttft=2.27s prefill=2.26s rounds=53 accepted=74/123"
)


def test_tensorfold_done_reports_decode_prefill_cache_and_spec():
    scraper = RuntimeMetricsScraper()
    scraper.observe(("h1", "c1"), _TENSORFOLD_DONE)
    scraper.commit()

    assert scraper.latest("gen") == 62.2
    # Prompt rate is fresh tokens over pure prefill time: 43 / 0.17 s.
    assert scraper.latest("prompt") == pytest.approx(252.9, abs=0.05)
    assert scraper.latest("hit") == 0.0
    # The denominator includes all drafted tokens, including accepted ones.
    assert scraper.latest("accept") == pytest.approx(11 / 23 * 100)


def test_tensorfold_one_token_request_skips_acceptance():
    scraper = RuntimeMetricsScraper()
    scraper.observe(("h1", "c1"), _TENSORFOLD_DONE_ONE_TOKEN)
    scraper.commit()

    assert scraper.latest("gen") == 0.0
    assert scraper.latest("prompt") == pytest.approx(288.9, abs=0.05)
    # 0/0 drafted tokens is "no drafts", not 0% acceptance: no sample at all.
    assert scraper.latest("accept") is None


def test_tensorfold_two_requests_in_one_window_average():
    scraper = RuntimeMetricsScraper()
    for line in (_TENSORFOLD_DONE, _TENSORFOLD_DONE_FASTER):
        scraper.observe(("h1", "c1"), line)
    scraper.commit()

    # Both completions land in one 30 s poll: the window point is their mean.
    assert scraper.latest("gen") == pytest.approx(63.4)


def test_tensorfold_lines_without_metrics_are_ignored():
    scraper = RuntimeMetricsScraper()
    scraper.observe(("h1", "c1"), '10.0.4.30 "GET /v1/completions" 200 -')
    scraper.observe(("h1", "c1"), "[tensorfold] ended req-477975f9986c reason=client-left prompt=12 tokens=3 after=0.50s")
    scraper.commit()

    assert all(scraper.latest(spec.key) is None for spec in _METRICS)


def test_tensorfold_cached_prefix_is_not_counted_as_computed_prompt():
    scraper = RuntimeMetricsScraper()
    scraper.observe(
        ("h1", "c1"),
        "[tensorfold] done req-0a1b2c3d4e5f prompt=100 cached=40 thinking=False tokens=60 "
        "sha=deadbeef finish=length tok/s=50.0 ttft=0.20s prefill=0.5s rounds=1 accepted=1/3",
    )
    scraper.commit()

    assert scraper.latest("prompt") == 120.0  # (100 - 40) / 0.5, not 200
    assert scraper.latest("hit") == 40.0
    assert scraper.latest("accept") == pytest.approx(100 / 3)


def test_tensorfold_done_without_thinking_field_still_scrapes():
    """Upstream emitters log the same completion line without the thinking field.

    The field is optional in the rule for exactly this shape: the format string
    read in the emitter (``_chat_prepared``) has prompt/cached then tokens.
    """
    scraper = RuntimeMetricsScraper()
    scraper.observe(
        ("h1", "c1"),
        "[tensorfold] done req-abc123 prompt=100 cached=0 tokens=50 sha=abcd1234 "
        "finish=stop tok/s=50.0 ttft=0.20s prefill=0.25s rounds=4 accepted=6/10",
    )
    scraper.commit()

    assert scraper.latest("gen") == 50.0
    assert scraper.latest("prompt") == 400.0  # 100 / 0.25
    assert scraper.latest("accept") == 60.0  # 6 / 10


def test_tensorfold_seed_page_takes_the_last_line_not_the_mean():
    scraper = RuntimeMetricsScraper()
    # A first (seed) page is history of unknown duration: each metric keeps the
    # page's last sample, not the mean over the whole page.
    scraper.observe_page(("h1", "c1"), [_TENSORFOLD_DONE, _TENSORFOLD_DONE_FASTER, _TENSORFOLD_DONE])
    scraper.commit()

    assert scraper.history("gen") == [62.2]


def test_aneda_stats_rates_divide_by_the_lines_own_window():
    """Same counts, different window: the line's own ``t`` is the divisor.

    The engine's tick rides the request loop, so ``t`` drifts (10..59 s in the
    live log) and a fixed cadence would overstate a slow window's rates. The
    counts are the live log's loaded line (2026-11-24).
    """
    line = "stats: t={t} rank=0 tok_out=2915 tok_in=600424 kv_util=0.010 kv_hit=0.737 draft_accept=0.783 queue=0 engram_hit=0.901 engram_used=0.028 reqs_stock=5 reqs_steered=0"
    ten = RuntimeMetricsScraper()
    ten.observe(("h1", "c1"), line.format(t=10))
    ten.commit()
    assert ten.latest("gen") == 291.5
    assert ten.latest("prompt") == 60042.4

    twenty = RuntimeMetricsScraper()
    twenty.observe(("h1", "c1"), line.format(t=20))
    twenty.commit()
    assert twenty.latest("gen") == 145.75
    assert twenty.latest("prompt") == 30021.2


def test_aneda_idle_window_reports_zeros_but_no_hit_or_accept():
    """A fully idle interval: rates are real zeros, the ratios have no denominator."""
    scraper = RuntimeMetricsScraper()
    scraper.observe(
        ("h1", "c1"),
        "stats: t=10 rank=0 tok_out=0 tok_in=0 kv_util=0.000 kv_hit=0.000 draft_accept=0.000 "
        "queue=0 engram_hit=0.000 engram_used=0.000 reqs_stock=0 reqs_steered=0",
    )
    scraper.commit()

    for key in ("gen", "prompt", "run", "queue", "kv", "engram_used"):
        assert scraper.latest(key) == 0.0, key
    # No prompt tokens, no generated tokens: nothing to reuse, nothing verified.
    for key in ("hit", "accept", "engram_hit"):
        assert scraper.latest(key) is None, key


def test_aneda_front_line_yields_no_metrics():
    """The front-end's line duplicates the engine's counts and means its queue."""
    assert _scrape("stats: t=10 front queue=0.0 reqs_stock=1 reqs_steered=0") == {}


def test_aneda_engram_fields_never_land_in_hit_or_kv():
    """Engram's own counters are separate keys, not the KV hit rate or occupancy."""
    loaded = _scrape(
        "stats: t=10 rank=0 tok_out=12 tok_in=362657 kv_util=0.082 kv_hit=0.994 "
        "draft_accept=0.286 queue=0 engram_hit=0.997 engram_used=0.344 reqs_stock=1 reqs_steered=0"
    )
    assert loaded["hit"] == 99.4  # kv_hit, never engram_hit's 99.7
    assert loaded["kv"] == 8.2  # kv_util, never engram_used's 34.4
    assert loaded["engram_hit"] == 99.7
    assert loaded["engram_used"] == 34.4


def test_render_metrics_footer_hidden_until_first_commit():
    scraper = RuntimeMetricsScraper()
    assert render_metrics_footer(scraper, _PALETTE, 100) == ""
    scraper.observe(("h1", "c1"), "docker: pull complete")
    scraper.commit()
    assert render_metrics_footer(scraper, _PALETTE, 100) == ""
    assert render_metrics_footer(scraper, _PALETTE, 0) == ""


def test_render_metrics_footer_items_carry_label_glyphs_and_value():
    scraper = RuntimeMetricsScraper()
    scraper.observe(("h1", "c1"), "Avg generation throughput: 45.6 tokens/s, Running: 3 reqs")
    scraper.commit()
    out = render_metrics_footer(scraper, _PALETTE, 1000)
    plain = _plain(out)
    assert "tg" in plain and "45.6" in plain
    assert "jobs" in plain and " 3" in plain
    assert any(glyph in plain for glyph in _SPARK_GLYPHS)
    # Colors come from the palette dict, as Rich markup tags the Static resolves.
    assert "[c0]" in out and "[/c0]" in out


def test_render_metrics_footer_label_field_and_sparkline_are_five_columns():
    """Exact layout: 5-column label, 5-column sparkline, then the value."""
    scraper = RuntimeMetricsScraper()
    scraper.observe(("h1", "c1"), "Avg generation throughput: 45.6 tokens/s")
    scraper.commit()

    plain = _plain(render_metrics_footer(scraper, (), 1000))
    # "gen" padded to 5, space, a flat 5-wide sparkline padded with spaces, space, value.
    assert plain == "tg" + " " * 8 + "▄ 45.6"


def test_render_metrics_footer_value_is_the_newest_sampled_window():
    """The number is the newest window's mean, not a cross-window average."""
    scraper = RuntimeMetricsScraper()
    for value in (10.0, 20.0, 90.0):
        scraper.observe(("h1", "c1"), f"Avg generation throughput: {value} tokens/s")
        scraper.commit()
    plain = _plain(render_metrics_footer(scraper, (), 1000))
    assert "90.0" in plain
    assert "40.0" not in plain  # (10 + 20 + 90) / 3 would be 40.0

    # Two prints inside the next poll are one point: that window's mean, 40.0.
    for value in (30.0, 50.0):
        scraper.observe(("h1", "c1"), f"Avg generation throughput: {value} tokens/s")
    scraper.commit()
    plain = _plain(render_metrics_footer(scraper, (), 1000))
    assert "40.0" in plain
    assert "90.0" not in plain


def test_render_metrics_footer_without_palette_keeps_glyphs_uncolored():
    scraper = RuntimeMetricsScraper()
    scraper.observe(("h1", "c1"), "Avg generation throughput: 45.6 tokens/s")
    scraper.commit()
    out = render_metrics_footer(scraper, (), 1000)
    assert "[c" not in out
    assert any(glyph in out for glyph in _SPARK_GLYPHS)


def test_render_metrics_footer_pins_percent_metrics_to_a_0_100_scale():
    half = RuntimeMetricsScraper()
    half.observe(("h1", "c1"), "token usage: 0.5")
    half.commit()
    out = _plain(render_metrics_footer(half, (), 1000))
    assert "kv" in out and "50%" in out and "▅" in out  # mid-scale, not a flat full block

    maxed = RuntimeMetricsScraper()
    maxed.observe(("h1", "c1"), "token usage: 1.0")
    maxed.commit()
    assert "100%" in _plain(render_metrics_footer(maxed, (), 1000))
    assert "█" in _plain(render_metrics_footer(maxed, (), 1000))


def test_render_metrics_footer_drops_overflow_items_in_priority_order():
    scraper = RuntimeMetricsScraper()
    scraper.observe(
        ("h1", "c1"),
        "Avg generation throughput: 45.6 tokens/s, Running: 3 reqs, GPU KV cache usage: 12.3%, Waiting: 1 reqs, "
        "Avg prompt throughput: 123.4 tokens/s, Prefix cache hit rate: 45.6%",
    )
    scraper.observe(("h1", "c1"), "SpecDecoding metrics: Avg Draft acceptance rate: 72.7%")
    scraper.commit()

    full = _plain(render_metrics_footer(scraper, _PALETTE, 1000))
    for label in ("jobs", "queue", "pp", "tg", "kv", "hit", "spec"):
        assert label in full

    narrow = _plain(render_metrics_footer(scraper, _PALETTE, len(full) - 1))
    assert full.startswith(narrow)  # what survives is a prefix, never a rewrap
    assert "spec" not in narrow
    assert render_metrics_footer(scraper, _PALETTE, 5) == ""


def test_sglang_prefill_reread_page_is_inert():
    scraper = RuntimeMetricsScraper()
    page = [_sglang_prefill("2026-10-15 00:58:46", 4096)]
    for _ in range(2):  # the same tail page, polled twice
        scraper.observe_page(("h1", "c1"), page)
        scraper.commit()

    # Re-reading an unchanged page must not add a point.
    assert scraper.history("prompt") == [1.0]


def test_observe_page_first_read_seeds_the_last_sample():
    """A first page is history of unknown duration: its last sample is the reading.

    The prime read deliberately reaches thousands of lines back to find a real
    stat; averaging that page would report a multi-minute mean as the first number.
    """
    scraper = RuntimeMetricsScraper()
    source = ("h1", "c1")
    scraper.observe_page(source, [f"Avg generation throughput: {v} tokens/s" for v in (10.0, 20.0, 30.0)])
    scraper.commit()
    assert scraper.history("gen") == [30.0]

    scraper.observe_page(source, [f"Avg generation throughput: {v} tokens/s" for v in (10.0, 20.0, 30.0, 40.0)])
    scraper.commit()
    assert scraper.history("gen") == [30.0, 40.0]  # the appended line is a steady sample


def test_observe_page_reprime_seeds_over_a_stale_tail():
    """A re-prime after a read failure arrives wider than the tail: seed, never average.

    The stored tail sits at the end of the deep page, so positional alignment
    finds nothing; averaging the page would report minutes and re-count lines
    already committed to the rings.
    """
    scraper = RuntimeMetricsScraper()
    source = ("h1", "c1")
    scraper.observe_page(source, ["Avg generation throughput: 10.0 tokens/s", "Avg generation throughput: 20.0 tokens/s"])
    scraper.commit()

    deep = [f"Avg generation throughput: {v} tokens/s" for v in (100.0, 200.0, 300.0)]
    deep += ['INFO:     127.0.0.1:60082 - "GET /health HTTP/1.1" 200 OK'] * (_PAGE_TAIL + 10)
    scraper.observe_page(source, deep)
    scraper.commit()

    assert scraper.history("gen") == [20.0, 300.0]  # the last sample, not the page mean 200


def test_observe_page_reread_adds_no_samples():
    scraper = RuntimeMetricsScraper()
    page = ["Avg generation throughput: 10.0 tokens/s", "Avg generation throughput: 20.0 tokens/s"]
    for _ in range(2):  # the same tail page, polled twice
        scraper.observe_page(("h1", "c1"), page)
        scraper.commit()

    # Seeded from the last sample; the re-read lines add nothing.
    assert scraper.history("gen") == [20.0]


def test_observe_page_ingests_only_the_appended_suffix():
    scraper = RuntimeMetricsScraper()
    source = ("h1", "c1")
    page = ["Avg generation throughput: 10.0 tokens/s", "Avg generation throughput: 20.0 tokens/s"]
    scraper.observe_page(source, page)
    scraper.commit()
    scraper.observe_page(source, [*page, "Avg generation throughput: 90.0 tokens/s"])
    scraper.commit()

    # Only the appended line is new: the re-read 10/20 must not enter the mean.
    assert scraper.history("gen") == [20.0, 90.0]


def test_observe_page_keeps_a_genuinely_repeated_line():
    scraper = RuntimeMetricsScraper()
    source = ("h1", "c1")
    scraper.observe_page(source, ["Avg generation throughput: 10.0 tokens/s"])
    scraper.commit()
    scraper.observe_page(
        source,
        [
            "Avg generation throughput: 10.0 tokens/s",
            "Avg generation throughput: 30.0 tokens/s",
            "Avg generation throughput: 30.0 tokens/s",
        ],
    )
    scraper.commit()

    # Two identical new lines are two samples, not a deduped one: mean 30.0.
    assert scraper.history("gen") == [10.0, 30.0]


def test_observe_page_realigns_without_overlap():
    scraper = RuntimeMetricsScraper()
    source = ("h1", "c1")
    scraper.observe_page(source, ["Avg generation throughput: 10.0 tokens/s", "Avg generation throughput: 20.0 tokens/s"])
    scraper.commit()
    # Nothing in common (rotation/truncation): the whole page is new.
    scraper.observe_page(source, ["Avg generation throughput: 50.0 tokens/s", "Avg generation throughput: 90.0 tokens/s"])
    scraper.commit()

    assert scraper.history("gen") == [20.0, 70.0]


def test_observe_page_empty_page_keeps_alignment():
    scraper = RuntimeMetricsScraper()
    source = ("h1", "c1")
    page = ["Avg generation throughput: 10.0 tokens/s", "Avg generation throughput: 20.0 tokens/s"]
    scraper.observe_page(source, page)
    scraper.commit()

    scraper.observe_page(source, [])  # an empty read is not a reset
    scraper.commit()
    assert scraper.history("gen") == [20.0]

    scraper.observe_page(source, page)  # still aligned with the last real page
    scraper.commit()
    assert scraper.history("gen") == [20.0]


def test_window_means_are_per_source_then_summed():
    scraper = RuntimeMetricsScraper()
    for value in (10.0, 20.0):
        scraper.observe(("h1", "r0"), f"Avg generation throughput: {value} tokens/s")
    for value in (50.0, 100.0):
        scraper.observe(("h1", "r1"), f"Avg generation throughput: {value} tokens/s")
    scraper.commit()

    # Each rank's window mean first (15 and 75), then the rank values add up.
    assert scraper.latest("gen", sources={("h1", "r0")}) == 15.0
    assert scraper.latest("gen", sources={("h1", "r1")}) == 75.0
    assert scraper.latest("gen") == 90.0


def test_window_without_samples_adds_no_point():
    scraper = RuntimeMetricsScraper()
    scraper.observe(("h1", "c1"), "Avg generation throughput: 10.0 tokens/s")
    scraper.commit()
    scraper.commit()  # nothing observed between the two commits

    assert scraper.history("gen") == [10.0]


def test_scraper_history_aligns_a_late_source_with_its_first_poll():
    scraper = RuntimeMetricsScraper()
    scraper.observe(("h1", "r0"), "Avg generation throughput: 10.0 tokens/s")
    scraper.commit()
    scraper.observe(("h1", "r0"), "Avg generation throughput: 20.0 tokens/s")
    scraper.observe(("h1", "r1"), "Avg generation throughput: 5.0 tokens/s")
    scraper.commit()

    # r1 joined late: it contributes only to the newest point, it does not shift
    # r0's first point into a shorter series.
    assert scraper.history("gen") == [10.0, 25.0]


def test_scraper_latest_and_history_filter_by_sources():
    scraper = RuntimeMetricsScraper()
    scraper.observe(("h1", "r0"), "Avg generation throughput: 100.0 tokens/s")
    scraper.observe(("h1", "r1"), "Avg generation throughput: 50.0 tokens/s")
    scraper.commit()

    assert scraper.latest("gen") == 150.0
    assert scraper.latest("gen", sources={("h1", "r0")}) == 100.0
    assert scraper.history("gen", sources={("h1", "r0")}) == [100.0]
    assert scraper.history("gen", sources=set()) == []
    assert scraper.latest("kv") is None


def test_scraper_pct_filter_takes_max_within_sources():
    scraper = RuntimeMetricsScraper()
    scraper.observe(("h1", "r0"), "GPU KV cache usage: 40.0%")
    scraper.observe(("h1", "r1"), "GPU KV cache usage: 61.0%")
    scraper.commit()

    assert scraper.latest("kv", sources={("h1", "r1")}) == 61.0
    assert scraper.latest("kv") == 61.0


def test_render_metrics_footer_renders_only_given_sources():
    scraper = RuntimeMetricsScraper()
    scraper.observe(("h1", "r0"), "Avg generation throughput: 100.0 tokens/s")
    scraper.observe(("h1", "r1"), "Avg generation throughput: 50.0 tokens/s")
    scraper.commit()

    only_r0 = _plain(render_metrics_footer(scraper, _PALETTE, 1000, sources={("h1", "r0")}))
    assert "100.0" in only_r0 and "150.0" not in only_r0
    assert "150.0" in _plain(render_metrics_footer(scraper, _PALETTE, 1000))


def test_render_metrics_footer_rounds_jobs_to_the_nearest_integer():
    """jobs is a count: a fractional window average prints as one, halves up.

    ``format(…, ".0f")`` would render the 2.5 average below as "2" (halves to
    even), which understates a running-request count.
    """
    scraper = RuntimeMetricsScraper()
    for running in (2, 3):
        scraper.observe(("h1", "c1"), f"Avg generation throughput: 10.0 tokens/s, Running: {running} reqs, Waiting: 0 reqs")
    scraper.commit()  # one poll: the window mean is (2 + 3) / 2 = 2.5

    plain = _plain(render_metrics_footer(scraper, (), 1000))
    jobs_item = plain.split("queue")[0].rstrip()  # the jobs item precedes queue in priority order
    assert jobs_item.startswith("jobs")
    assert "2.5" not in jobs_item and "." not in jobs_item
    assert jobs_item.endswith("3")  # halves up, not to even


def test_kv_usage_reports_the_fullest_pool_and_ignores_hisparse_tiers():
    """kv is the engine's max over pools; device/host tiers of one pool are not pools.

    Provenance for the tier cases, read in the deployed fork
    (/sgl-workspace/sglang/.../scheduler_components/pool_stats_observer.py): both
    usage-msg builders append the ``gpu``/``cpu`` pair when ``is_hisparse`` and only
    then fall back to the bare pool field under ``if not parts:``, so a hisparse
    line with no hybrid flag carries the tiers and nothing else — while the engine's
    own headline (``stats.token_usage``) is ``get_max_pool_usage()``, which folds in
    full, swa and mamba only and never the hisparse fields. Tier readings are
    therefore reported as no kv rather than as the pool.
    """
    cases = {
        # both pools present, swa fuller
        "full token usage: 0.02, swa token usage: 0.24, #running-req: 1": 24.0,
        # full fuller
        "full token usage: 0.31, swa token usage: 0.05, #running-req: 1": 31.0,
        # hybrid-SSM: the engine's max includes the mamba pool
        "#full token: 100, full token usage: 0.10, mamba num: 4, mamba usage: 0.44, #running-req: 1": 44.0,
        # HiSparse breakdown rides along with the pools but must not become kv
        "full token usage: 0.20, swa token usage: 0.10, #gpu token: 900, gpu token usage: 0.90, "
        "#cpu token: 500, cpu token usage: 0.50, #running-req: 1": 20.0,
        # the pre-split spelling still reads as the pool
        "#token: 336859, token usage: 0.37, #running-req: 1": 37.0,
        # a HiSparse-only line carries no pool reading at all
        "#gpu token: 900, gpu token usage: 0.90, #cpu token: 500, cpu token usage: 0.50": None,
    }
    for line, expected in cases.items():
        scraped = _scrape(line)
        assert scraped.get("kv") == expected or (expected is None and "kv" not in scraped), f"{line!r} -> {scraped}"


def test_engram_lines_are_not_kv_cache_hits():
    """Engram's hash-memory hit rate must never be read as the KV-cache hit rate.

    Engram (srt/layers/engram.py) is a gated n-gram hash memory added to the
    residual stream; its line reports that table's own lookups. The footer's
    ``hit`` is prompt-token reuse — cached / (new + cached) — so the two share a
    word and nothing else. Its counters also reset per print, and an idle
    interval prints 100.0% with reads=0 (4508 of 8588 prints in the live SGLang
    log), which the pct aggregation (max across sources) would happily render as
    a healthy 100%: another reason it needs its own metric, not a new rule.
    """
    engram = (
        "[2026-10-17 21:17:43 TP0] Engram layer=1 lookups=305185 hit_rate=99.5% reads=1607 "
        "cache=2.0GiB(16-way) scales=0.0GiB threads=96 packed=True"
    )
    alone = RuntimeMetricsScraper()
    alone.observe(("h1", "c1"), engram)
    alone.commit()
    assert [spec.key for spec in _METRICS if alone.latest(spec.key) is not None] == []
    assert render_metrics_footer(alone, _PALETTE, 1000) == ""

    decode = (
        "[2026-10-17 21:00:00 TP0] Decode batch, #running-req: 1, full token usage: 0.31, "
        "gen throughput (token/s): 44.5, #queue-req: 0, #new-token: 100, #cached-token: 900"
    )
    mixed = RuntimeMetricsScraper()
    mixed.observe(("h1", "c1"), engram)
    mixed.observe(("h1", "c1"), decode)
    mixed.commit()
    assert mixed.latest("hit") == 90.0  # 900 / (100 + 900), never Engram's 99.5
    assert mixed.latest("kv") == 31.0  # the pool reading, not the 2.0GiB table


def test_scraper_retain_forgets_dead_sources():
    scraper = RuntimeMetricsScraper()
    rank0, rank1 = ("h1", "r0"), ("h1", "r1")
    scraper.observe(rank0, "Avg generation throughput: 100.0 tokens/s")
    scraper.observe(rank1, "Avg generation throughput: 50.0 tokens/s")
    scraper.commit()
    assert scraper.latest("gen") == 150.0

    # rank1's container instance changed: its last value must not keep counting.
    scraper.retain({rank0})
    assert scraper.latest("gen") == 100.0
    assert scraper.history("gen") == [100.0]

    scraper.observe(rank0, "Avg generation throughput: 120.0 tokens/s")
    scraper.commit()
    assert scraper.latest("gen") == 120.0


@pytest.mark.parametrize("accepted,drafted,expected", [(10, 10, 100.0), (0, 10, 0.0), (3, 4, 75.0)])
def test_tensorfold_acceptance_uses_total_drafts(accepted, drafted, expected):
    scraper = RuntimeMetricsScraper()
    scraper.observe(("h1", "c1"), _TENSORFOLD_DONE.replace("accepted=11/23", f"accepted={accepted}/{drafted}"))
    scraper.commit()
    assert scraper.latest("accept") == expected


def test_stalled_source_does_not_shift_its_old_samples_into_new_windows():
    scraper = RuntimeMetricsScraper()
    first, second = ("h1", "r0"), ("h2", "r1")
    scraper.observe(first, "Avg generation throughput: 10.0 tokens/s, GPU KV cache usage: 10.0%")
    scraper.observe(second, "Avg generation throughput: 20.0 tokens/s, GPU KV cache usage: 80.0%")
    scraper.commit()
    assert scraper.history("gen") == [30.0]
    scraper.observe(first, "Avg generation throughput: 30.0 tokens/s, GPU KV cache usage: 30.0%")
    scraper.commit()
    assert scraper.history("gen") == [30.0, 30.0]
    assert scraper.history("kv") == [80.0, 30.0]
    assert scraper.history("gen", sources={second}) == [20.0]
    scraper.commit()  # a completely idle poll adds no point
    scraper.observe(second, "Avg generation throughput: 40.0 tokens/s, GPU KV cache usage: 40.0%")
    scraper.commit()
    assert scraper.history("gen") == [30.0, 30.0, 40.0]
    assert scraper.history("kv") == [80.0, 30.0, 40.0]
    assert scraper.history("gen", sources={second}) == [20.0, 40.0]


def test_alternating_sources_keep_aggregate_history_bounded():
    scraper = RuntimeMetricsScraper(history_points=3)
    for value in range(6):
        scraper.observe(("h1", f"r{value % 2}"), f"Avg generation throughput: {value}.0 tokens/s")
        scraper.commit()
    assert scraper.history("gen") == [3.0, 4.0, 5.0]
    assert scraper.history("gen", sources={("h1", "r0")}) == [0.0, 2.0, 4.0]
