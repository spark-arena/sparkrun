"""Tests for cli/_monitor_colorize.py: colorizer classes, rule families, and dispatch."""

from __future__ import annotations

import pytest

from sparkrun.cli._monitor_colorize import (
    COLORIZERS,
    GENERIC_COLORIZER,
    RUNTIME_COLORIZERS,
    AnedaColorizer,
    AtlasColorizer,
    LlamaCppColorizer,
    SglangColorizer,
    TensorFoldColorizer,
    TrtLlmColorizer,
    VllmColorizer,
    colorize_log_spans,
    colorizer_for,
    pick_colorizer,
    unclaimed_ranges,
)


def _pairs(spans, text):
    """``[(slice text, style)]`` for spans (which are ``(start, end, style)`` tuples)."""
    return [(text[start:end], style) for start, end, style in spans]


def _styles(spans):
    return {style for _, _, style in spans}


# ---------------------------------------------------------------------------
# Rule families (through the module-level dispatch, no runtime hint)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "token", "style"),
    [
        ("INFO 10-11 08:12:34 [core] init", "INFO", "cyan"),
        ("(EngineCore pid=9527) INFO 10-11 08:12:34 [core.py:130] Worker ready", "INFO", "cyan"),
        ("[2026-10-11 08:12:34 TP0] INFO backend ready", "INFO", "cyan"),
        ("0.00.035.064 I llama_model_load: vocab only - skipping tensors", "I", "cyan"),
        ("0.00.000.000 E llama_model_load_from_file_impl: failed to load model", "E", "red"),
        ("0.12.345.678 D llama_server: verbose", "D", "dim"),
        ("[10/11/2026-08:12:34 UTC] [E] rank 1 exited", "[E]", "red"),
        ("E1011 08:12:34.123  9527 trt_ffi.cc:42] bad tensor", "E", "red"),
        ("dgx01:28971 [0] NCCL INFO Ring ALGO", "INFO", "cyan"),
        ("[2026-10-11T08:12:34Z WARN atlas::serve] oom", "WARN", "yellow"),
        ("2026-10-11T08:12:34.123Z  INFO atlas::serve: GET /v1/chat", "INFO", "cyan"),
        ("INFO:engine.core:Started engine", "INFO", "cyan"),
        ("TRACE:engine.core:tick", "TRACE", "dim"),
        ("CRITICAL:engine.core:down", "CRITICAL", "bold red"),
    ],
)
def test_level_tokens_colorize(text, token, style):
    assert (token, style) in _pairs(colorize_log_spans(text), text)


@pytest.mark.parametrize(
    ("text", "token"),
    [
        ("worker 3 failed to connect", "failed"),
        ("CUDA out of memory", "out of memory"),
        ("connection refused by peer", "refused"),
        ("request timeout after 30s", "timeout"),
        ("process killed by signal 9", "killed"),
        ("panic: index out of range", "panic"),
        ("worker died, retrying soon", "retrying"),
    ],
)
def test_severity_words_are_colored(text, token):
    pairs = _pairs(colorize_log_spans(text), text)
    assert any(token == text_slice for text_slice, _ in pairs), (text, pairs)
    assert any(style in ("red", "yellow") for _, style in pairs)


def test_whole_line_rules_end_a_crash():
    traceback = "Traceback (most recent call last):"
    assert _pairs(colorize_log_spans(traceback), traceback) == [(traceback, "bold red")]
    exc = "ValueError: boom"
    assert _pairs(colorize_log_spans(exc), exc) == [(exc, "bold red")]


def test_warning_prose_is_yellow_not_whole_line_red():
    text = "Warning: non-loopback server has no API key"
    spans = colorize_log_spans(text)
    assert ("Warning", "yellow") in _pairs(spans, text)
    assert not _styles(spans) & {"red", "bold red"}


def test_severity_and_levels_stay_word_bounded():
    # ``error_code`` is not a bare failure word, and lowercase prose is never a level.
    assert "red" not in _styles(colorize_log_spans("metric error_code_total 3"))
    assert "cyan" not in _styles(colorize_log_spans("info: nothing to see"))


@pytest.mark.parametrize(
    ("text", "token", "style"),
    [
        ("done req-1 prompt=128 tokens=64 tok/s=100.0 ttft=0.10s", "128", "yellow"),
        ("done req-1 prompt=128 tokens=64 tok/s=100.0 ttft=0.10s", "100.0", "bold yellow"),
        ("done req-1 prompt=128 tokens=64 tok/s=100.0 ttft=0.10s", "0.10s", "yellow"),
        # The count before a rate *key* stays a count: ``64 tok/s=100.0`` must not merge.
        ("done req-1 prompt=128 tokens=64 tok/s=100.0 ttft=0.10s", "64", "yellow"),
        ("finished in 1.25s", "1.25s", "yellow"),
        ("elapsed 250ms", "250ms", "yellow"),
        ('"GET /v1/chat/completions HTTP/1.1" 200 OK', "GET", "bold cyan"),
        ('"GET /v1/chat/completions HTTP/1.1" 200 OK', "/v1/chat/completions", "cyan"),
        ('"POST /v1/chat/completions HTTP/1.1" 404 Not Found', "404", "yellow"),
        ('"POST /v1/chat/completions HTTP/1.1" 503 Service Unavailable', "503", "red"),
        ('"POST /v1/chat/completions HTTP/1.1" 301 Moved', "301", "cyan"),
        ('"GET /upgrade HTTP/1.1" 101 Switching Protocols', "101", "cyan"),
        ("serving at http://127.0.0.1:8080/v1", "http://127.0.0.1:8080/v1", "underline blue"),
        ("request chatcmpl-9f8e7d6c completed", "chatcmpl-9f8e7d6c", "magenta"),
        ("trace 550e8400-e29b-41d4-a716-446655440000 done", "550e8400-e29b-41d4-a716-446655440000", "magenta"),
        ("INFO:engine.core:Started engine", "engine.core", "bold blue"),
        # Engine stat lines: rates bold, phase words cyan/green, counts and units yellow.
        ("Avg prompt throughput: 0.0 tokens/s", "Avg prompt throughput", "cyan"),
        ("Avg prompt throughput: 0.0 tokens/s", "0.0 tokens/s", "bold yellow"),
        ("Avg generation throughput: 429.5 tokens/s", "Avg generation throughput", "green"),
        ("Avg generation throughput: 429.5 tokens/s", "429.5 tokens/s", "bold yellow"),
        ("Running: 24 reqs, Waiting: 0 reqs", "24", "yellow"),
        ("Running: 24 reqs, Waiting: 0 reqs", "0", "yellow"),
        ("GPU KV cache usage: 89.0%", "89.0%", "yellow"),
        ("Available KV cache memory: 5.77 GiB", "5.77 GiB", "yellow"),
        ("GPU KV cache size: 465,920 tokens", "465,920", "yellow"),
        ("Prefill batch, #new-token: 384, #cached-token: 150144, token usage: 0.19", "Prefill", "cyan"),
        ("Prefill batch, #new-token: 384, #cached-token: 150144, token usage: 0.19", "384", "yellow"),
        ("Prefill batch, #new-token: 384, #cached-token: 150144, token usage: 0.19", "150144", "yellow"),
        ("Prefill batch, #new-token: 384, #cached-token: 150144, token usage: 0.19", "0.19", "yellow"),
        ("Decode batch. #running-req: 23, #token: 4096", "Decode", "green"),
        ("Decode batch. #running-req: 23, #token: 4096", "23", "yellow"),
        ("input throughput (token/s): 536.67", "input throughput", "cyan"),
        ("input throughput (token/s): 536.67", "536.67", "bold yellow"),
        (
            "prompt eval time = 123456.78 ms / 5 tokens (24691.36 ms per token, 0.04 tokens per second)",
            "prompt eval time",
            "cyan",
        ),
        (
            "prompt eval time = 123456.78 ms / 5 tokens (24691.36 ms per token, 0.04 tokens per second)",
            "0.04 tokens per second",
            "bold yellow",
        ),
        (
            "eval time = 654.01 ms / 100 runs (6.54 ms per token, 152.9 tokens per second)",
            "eval time",
            "green",
        ),
        (
            "eval time = 654.01 ms / 100 runs (6.54 ms per token, 152.9 tokens per second)",
            "152.9 tokens per second",
            "bold yellow",
        ),
        # Realistic sniffed engine lines: phase and data families stack on the level/process spans.
        (
            "(APIServer pid=6051) INFO 05-11 17:46:04 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s",
            "APIServer",
            "bold blue",
        ),
        (
            "(APIServer pid=6051) INFO 05-11 17:46:04 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s",
            "0.0 tokens/s",
            "bold yellow",
        ),
        (
            "[2025-10-31 00:07:31 TP0] Prefill batch, #cached-token: 150144, input throughput (token/s): 536.67",
            "536.67",
            "bold yellow",
        ),
        ("gen throughput (token/s): 152.9", "gen throughput", "green"),
        ("gen throughput (token/s): 152.9", "152.9", "bold yellow"),
    ],
)
def test_data_families_colorize(text, token, style):
    assert (token, style) in _pairs(colorize_log_spans(text), text)


@pytest.mark.parametrize(
    ("text", "token"),
    [
        ("serving at http://127.0.0.1:8080/v1", "8080"),  # URL port: the URL claims the whole span
        ("(EngineCore pid=9527) INFO 10-11 08:12:34 [core.py:130] ready", "130"),  # logger line number
        ("INFO 10-11 08:12:34 ready", "34"),  # timestamp seconds
        ("[2026-10-11 08:12:34 TP0] cuda graph: False", "False"),  # non-numeric value
        ("Engine 000: Avg prompt throughput: 0.0 tokens/s", "Avg"),  # the stats label is not a value
    ],
)
def test_colon_values_need_space_and_never_fire_on_ports_timestamps_or_false_values(text, token):
    """The colon rule requires ``: `` before a digit, so ``:130``/``:8080``/``:34`` stay put."""
    data_styles = {"yellow", "bold yellow"}
    slices = [slice_text for slice_text, style in _pairs(colorize_log_spans(text), text) if style in data_styles]
    assert token not in slices, (text, slices)


@pytest.mark.parametrize(
    ("text", "token", "style"),
    [
        ("[2026-10-11 08:12:34 TP0] INFO backend ready", "TP0", "cyan"),
        ("TP1 worker joined", "TP1", "magenta"),
        ("rank=2 world ready", "rank=2", "green"),
        ("dgx01:28971 [3] NCCL INFO Ring ALGO", "[3]", "yellow"),
    ],
)
def test_rank_tags_cycle_the_palette(text, token, style):
    assert (token, style) in _pairs(colorize_log_spans(text), text)


# ---------------------------------------------------------------------------
# Per-runtime classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("colorizer", "text", "token", "style"),
    [
        (VllmColorizer(), "(EngineCore pid=9527) INFO 10-11 08:12:34 [core.py:130] ready", "EngineCore", "bold blue"),
        (SglangColorizer(), "[2026-10-11 08:12:34 TP0] INFO backend ready", "TP0", "cyan"),
        (LlamaCppColorizer(), "0.00.035.064 I llama_model_load: vocab only", "llama_model_load", "bold blue"),
        (LlamaCppColorizer(), "0.00.035.064 I load_tensors: n_ctx: 4096", "4096", "yellow"),
        (TrtLlmColorizer(), "[10/11/2026-08:12:34 UTC] [TRT-LLM] [I] engine ready", "[TRT-LLM]", "bold blue"),
        (AtlasColorizer(), "[2026-10-11T08:12:34Z WARN atlas::serve] oom", "atlas::serve", "bold blue"),
        (AnedaColorizer(), "serve: req 7 session 0006b0500f500006: prompt 145800, reused 0 (cold)", "serve:", "bold blue"),
        (AnedaColorizer(), "serve: req 7 session 0006b0500f500006: prompt 145800", "0006b0500f500006", "magenta"),
        (AnedaColorizer(), "req 1 session 1882707761037312: prompt 5 (cached 0), completion 1, length; ttft 0.039 s", "length", "yellow"),
        (AnedaColorizer(), "serve: req 7 session 0006b0500f500006: prefill 39.3 ms, stop; store 0.0 MiB", "stop", "green"),
        (AnedaColorizer(), "[picom-launcher] rank 0: engine ready", "[picom-launcher]", "bold blue"),
        (AnedaColorizer(), "stats: t=10 rank=0 tok_out=12 tok_in=4824 queue=0", "stats: t=10 rank=0", "bold blue"),
    ],
)
def test_runtime_extras_colorize(colorizer, text, token, style):
    assert (token, style) in _pairs(colorizer.spans(text), text)


TENSORFOLD_LINES = [
    "[tensorfold] serving nemotron at http://127.0.0.1:8080/v1 (sampling: greedy)",
    "[tensorfold] done req-8f3a2c1e prompt=100 tokens=200 tok/s=100.0",
    "[tensorfold] warning: non-loopback server has no API key",
    "[tensorfold] refused cmpl-1: model not found",
    "[tensorfold] ended chatcmpl-1 reason=error (OutOfMemory) prompt=12 tokens=0 after=1.25s",
    '[tensorfold] 10.0.0.5 "POST /v1/chat/completions HTTP/1.1" 503 -',
    # Real workload log, 2026-10-17: launcher boot line and a completed request,
    # plus a max-token cut-off (the log's own finish=length line); the client
    # cancellation is log.zig's own test string for its ``ended`` helper.
    "[tensorfold-launcher] boot: node_rank=0/2 rendezvous=10.0.4.34:25000"
    " model=/cache/runtime/tensorfold/model-overlay serving=0.0.0.0:8000"
    " context=1048576 engram=/cache/huggingface/hub/dsv41-engram",
    "[tensorfold] done req-477975f9986c prompt=43 cached=0 thinking=True tokens=19"
    " sha=e96a676a795c finish=stop tok/s=62.2 ttft=0.18s prefill=0.17s rounds=7 accepted=11/23",
    "[tensorfold] done req-0f5d78cbbc06 prompt=2048 cached=0 thinking=True tokens=128"
    " sha=f9b521598a86 finish=length tok/s=64.6 ttft=2.27s prefill=2.26s rounds=53 accepted=74/123",
    "[tensorfold] ended req-1 reason=client-left prompt=12 tokens=3 after=0.50s",
    "tensorfold: cannot listen on 127.0.0.1:8080 (AddressInUse)",
]

RUNTIME_MATRIX = [
    "INFO 10-11 08:12:34 [core] init",
    "(EngineCore pid=9527) INFO 10-11 08:12:34 [core.py:130] Worker ready",
    "[2026-10-11 08:12:34 TP0] INFO backend ready",
    "0.00.035.064 I llama_model_load: vocab only - skipping tensors",
    "0.00.000.000 E llama_model_load_from_file_impl: failed to load model",
    "[10/11/2026-08:12:34 UTC] [E] rank 1 exited",
    "E1011 08:12:34.123  9527 trt_ffi.cc:42] bad tensor",
    "dgx01:28971 [0] NCCL INFO Ring ALGO",
    "[2026-10-11T08:12:34Z WARN atlas::serve] oom",
    "2026-10-11T08:12:34.123Z  INFO atlas::serve: GET /v1/chat",
    "INFO:engine.core:Started engine",
    "Traceback (most recent call last):",
    "ValueError: boom",
    '"POST /v1/chat/completions HTTP/1.1" 503 Service Unavailable',
    "plain prose with no data at all",
    "downloaded INFO.txt already",
    "(APIServer pid=6051) INFO 05-11 17:46:04 [loggers.py:271] Engine 000: Avg prompt throughput: 0.0 tokens/s,"
    " Avg generation throughput: 429.5 tokens/s, Running: 24 reqs, Waiting: 0 reqs, GPU KV cache usage: 89.0%",
    "[2025-10-31 00:07:31 TP0] Prefill batch, #new-token: 384, #cached-token: 150144, token usage: 0.19, #running-req: 1",
    "[2025-10-31 00:07:31 TP0] Decode batch. #running-req: 23, #token: 4096, token usage: 0.41",
    *TENSORFOLD_LINES,
]


@pytest.mark.parametrize("colorizer", COLORIZERS)
@pytest.mark.parametrize("text", RUNTIME_MATRIX)
def test_spans_are_sorted_disjoint_and_in_bounds(colorizer, text):
    spans = colorizer.spans(text)
    previous_end = 0
    for start, end, style in spans:
        assert start >= previous_end, (text, spans)  # sorted and disjoint
        assert start < end <= len(text), (text, spans)
        assert style, (text, spans)
        previous_end = end


ANEDA_LINES = [
    # Real workload log, 2026-11-24 (rank 0's container stdout, both emitters).
    "stats: t=10 rank=0 tok_out=12 tok_in=4824 kv_util=0.000 kv_hit=0.000 draft_accept=0.000 queue=0 engram_hit=0.501 engram_used=0.000 reqs_stock=2 reqs_steered=2",
    "stats: t=10 front queue=0.0 reqs_stock=1 reqs_steered=0",
    "serve: req 7 session 0006b0500f500006: prompt 145800, reused 0 (cold), prefill 14762.0 ms (9877 tok/s), 345 tokens in 3645.1 ms (94.4 tok/s), stop; store 104.1 MiB",
    "req 1 plan: session 1882707761037312, new: no session shares a byte",
    "serve: req 8 restart: a fork of 0006b0500f500006: common prefix 145798 ids; reused 145408 (snapshot)",
    "state: session 2 spilled to disk (9.3 MiB; 0.01 of 32.00 GiB of the file in use; host 0.20 ms)",
    "spill: 9.3 MiB to staging in 0.17 ms on the stream (58.5 GB/s; checksums before it 0.22 ms)",
    "[picom-launcher] rank 0: starting dsfront on 10.0.4.31:8000 (served name deepseek-ai/DeepSeek-V4.1-Flash-1017, max_tokens 32768, auth keyless)",
    "serve: listening on /cache/runtime/picom/run/engine.sock (context 1048576, session store 12.00 GiB)",
]


def test_aneda_colorizer_recognizes_its_own_lines_only():
    colorizer = AnedaColorizer()
    for text in ANEDA_LINES:
        assert colorizer.recognizes(text), text
    for text in RUNTIME_MATRIX:
        if text in ANEDA_LINES:
            continue
        assert not colorizer.recognizes(text), text


def test_aneda_outcome_and_drafts_keep_their_semantics():
    colorizer = AnedaColorizer()
    # The end reason follows the completion counts: stop is natural, length is the ceiling.
    line = "req 2 session 0006b0500f500001: completion 2, stop; ttft 0.041 s"
    assert ("stop", "green") in _pairs(colorizer.spans(line), line)
    # The acceptance is one data span: the generic key=value rule would take only a piece.
    batch = "drafts: 6.94 tokens and 8.79 rows per step, 433 of 585 drafts accepted; GPU ms by rows: 4: 31.10 (2)"
    assert ("433 of 585 drafts accepted", "yellow") in _pairs(colorizer.spans(batch), batch)


def test_tensorfold_colorizer_recognizes_its_own_lines_only():
    colorizer = TensorFoldColorizer()
    for text in TENSORFOLD_LINES:
        assert colorizer.recognizes(text), text
    for text in RUNTIME_MATRIX:
        if text in TENSORFOLD_LINES:
            continue
        assert not colorizer.recognizes(text), text


def test_tensorfold_fatal_is_whole_line_and_semantics_are_specific():
    fatal = "tensorfold: cannot listen on 127.0.0.1:8080 (AddressInUse)"
    assert _pairs(TensorFoldColorizer().spans(fatal), fatal) == [(fatal, "red")]

    done = "[tensorfold] done req-8f3a2c1e prompt=100"
    pairs = _pairs(TensorFoldColorizer().spans(done), done)
    assert ("done", "green") in pairs
    assert ("req-8f3a2c1e", "magenta") in pairs

    warning = "[tensorfold] warning: open"
    assert ("warning:", "yellow") in _pairs(TensorFoldColorizer().spans(warning), warning)


def test_tensorfold_launcher_and_outcome_tokens_colorize():
    colorizer = TensorFoldColorizer()
    launcher = next(text for text in TENSORFOLD_LINES if text.startswith("[tensorfold-launcher]"))
    assert colorizer.recognizes(launcher)
    assert ("[tensorfold-launcher]", "bold blue") in _pairs(colorizer.spans(launcher), launcher)

    for text in TENSORFOLD_LINES:
        pairs = _pairs(colorizer.spans(text), text)
        if "finish=stop" in text:
            assert ("finish=stop", "green") in pairs, pairs
        if "finish=length" in text:
            assert ("finish=length", "yellow") in pairs, pairs
        if "reason=client-left" in text:
            assert ("reason=client-left", "dim") in pairs, pairs

    done_stop = next(text for text in TENSORFOLD_LINES if "finish=stop" in text)
    pairs = _pairs(colorizer.spans(done_stop), done_stop)
    assert ("accepted=11/23", "yellow") in pairs  # one data span, not just the numerator
    assert ("sha=e96a676a795c", "magenta") in pairs  # hex hash: not digit-leading


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("runtime", "expected"),
    [
        ("vllm-ray", "vllm"),
        ("vllm-distributed", "vllm"),
        ("eugr-vllm", "vllm"),
        ("sglang", "sglang"),
        ("llama-cpp", "llama-cpp"),
        ("trtllm", "trtllm"),
        ("atlas", "atlas"),
        ("tensorfold", "tensorfold"),
        ("aneda", "aneda"),
        ("Aneda", "aneda"),
        ("VLLM-Ray", "vllm"),
        ("unknown-engine", None),
        ("", None),
        (None, None),
    ],
)
def test_colorizer_for_resolves_known_names(runtime, expected):
    colorizer = colorizer_for(runtime)
    assert (colorizer.name if colorizer else None) == expected


def test_every_runtime_alias_resolves_to_a_registered_colorizer():
    registered = {colorizer.name for colorizer in (*COLORIZERS, GENERIC_COLORIZER)}
    assert {colorizer.name for colorizer in RUNTIME_COLORIZERS.values()} <= registered
    for runtime in ("vllm-ray", "vllm-distributed", "eugr-vllm", "sglang", "trtllm", "atlas", "llama-cpp"):
        assert colorizer_for(runtime) is not None, runtime


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("(EngineCore pid=1) INFO 10-11 08:12:34 [core.py:1] x", "vllm"),
        ("[2026-10-11 08:12:34 TP0] INFO ready", "sglang"),
        ("0.00.035.064 I llama_model_load: vocab only", "llama-cpp"),
        ("[10/11/2026-08:12:34 UTC] [E] rank 1 exited", "trtllm"),
        ("[2026-10-11T08:12:34Z WARN atlas::serve] oom", "atlas"),
        ("[tensorfold] serving nemotron", "tensorfold"),
        ("tensorfold: cannot listen", "tensorfold"),
        ("serve: req 7 session 0006b0500f500006: prompt 145800", "aneda"),
        ("stats: t=10 rank=0 tok_out=12 tok_in=4824", "aneda"),
        ("plain line, no signature", "generic"),
    ],
)
def test_pick_colorizer_sniffs_unknown_runtimes(text, expected):
    assert pick_colorizer(text, runtime="out-of-tree-engine").name == expected


def test_named_runtime_wins_over_sniffing():
    # A vLLM workload whose line looks like an Atlas line: the name decides.
    text = "[2026-10-11T08:12:34Z WARN atlas::serve] oom"
    assert pick_colorizer(text, runtime="vllm-ray").name == "vllm"
    assert pick_colorizer(text, runtime=None).name == "atlas"


def test_colorize_log_spans_wraps_dispatch_and_is_cached():
    text = "[tensorfold] done req-abc123 prompt=1"
    first = _pairs(colorize_log_spans(text), text)
    assert ("done", "green") in first
    assert first == _pairs(colorize_log_spans(text), text)

    # The runtime argument is part of the span identity: naming another runtime
    # drops that runtime's semantics but keeps the shared data families.
    as_atlas = _pairs(colorize_log_spans(text, runtime="atlas"), text)
    assert ("done", "green") not in as_atlas
    assert ("req-abc123", "magenta") in as_atlas


# ---------------------------------------------------------------------------
# unclaimed_ranges (the dim baseline's complement)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("total", "claimed", "expected"),
    [
        (10, [], [(0, 10)]),
        (10, [(0, 10)], []),
        (10, [(0, 3)], [(3, 10)]),
        (10, [(7, 10)], [(0, 7)]),
        (10, [(3, 5)], [(0, 3), (5, 10)]),
        (10, [(3, 5), (4, 7)], [(0, 3), (7, 10)]),  # overlapping claims merge
        (10, [(2, 4), (4, 6)], [(0, 2), (6, 10)]),  # adjacent claims merge
        (10, [(-5, 2), (12, 20)], [(2, 10)]),  # out-of-range clamps
        (10, [(5, 5), (6, 6)], [(0, 10)]),  # empty claims ignored
    ],
)
def test_unclaimed_ranges(total, claimed, expected):
    assert unclaimed_ranges(total, claimed) == expected
