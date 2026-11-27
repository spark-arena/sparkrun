"""Runtime log-metric scraping, and the log pane's metrics sparkline strip.

The log pane already streams engine logs; this module reads the numbers those
lines report and turns them into a one-row sparkline footer. Formats were
verified against the engines' own sources (2026-10-17), and a rule is only added
for a line shape someone has actually read there:

* vLLM v1   ``Avg prompt throughput: 123.4 tokens/s, Avg generation throughput: …,
             Running: 3 reqs, Waiting: 1 reqs, GPU KV cache usage: 12.3% …``
             (vllm/v1/metrics/loggers.py)
* vLLM v0   same fields with ``Swapped: … reqs, Pending: … reqs`` and no
             prefix-hit field (vllm/engine/metrics.py)
* vLLM spec ``SpecDecoding metrics: … Avg Draft acceptance rate: 72.7%``
             (vllm/v1/spec_decode/metrics.py)
* SGLang    ``Decode batch, #running-req: …, token usage: 0.52, … gen throughput
             (token/s): …, #queue-req: …`` and the prefill variant with
             ``#new-token``/``#cached-token`` (srt/managers/scheduler_components/
             metrics_reporter.py). Newer builds split the pool reading into
             ``full token usage`` and ``swa token usage`` on the same line: a
             hybrid-attention model keeps full-attention and sliding-window
             pools in separate allocators (``PoolStats.full_capacity`` vs
             ``swa_capacity`` in pool_stats_observer.py), so either can bind
             first. Both are read and the fuller one is reported, matching the
             engine's own ``PoolStats.get_max_pool_usage()`` — which also folds
             in the ``mamba usage`` pool of hybrid-SSM models, so that rule
             rides along. HiSparse's ``gpu``/``cpu token usage`` are device and
             host tiers of one pool, not separate pools, and the engine's max
             never includes them, so they stay out: a line carrying only those
             (hisparse with no hybrid flag) reports no kv at all rather than a
             tier reading.
             ``token usage`` and ``accept rate`` are pool fractions (0..1), not
             percents; ``#new-token`` is the uncached portion, so the hit rate is
             ``cached / (new + cached)``.
             The prefill line's ``input throughput (token/s)`` field is the pp
             metric: the engine divides the batch's new tokens (``#new-token``,
             plus ``#replay-token``) by the wall time since the previous prefill
             print (metrics_reporter.report_prefill_stats), so it is read
             verbatim.
* llama-server every era logs behind the same slot-line prefix
             ``slot %12.*s: id %2d | task %d | `` (server-common.h / utils.hpp),
             identical at b4000, b8076/b8094 and master, so the rules below match
             on the text that follows it. The deployed pinned images
             (``scitrera/dgx-spark-llama-cpp:b8094-cu131``, b8076) have no
             ``print_timings_tg``/``print_timings_pp`` at all, so their only
             capture is the final per-task block (the two eval-time rules) plus
             ``draft acceptance rate``, which is logged at CONT level and so
             renders with no prefix and no level letter.
* master and the v0.x tags add live lines: ``n_gen = …, tg = … t/s,
             tg_3s = … t/s`` and ``prompt processing, n_tokens = …,
             progress = …, t = … s / … tokens per second``. Both are guarded
             upstream (the tg line needs ``n_gen >= 100`` and prints at most once
             per 3 s; the pp line needs >= 3 s of prompt processing), so an idle
             or short-request workload legitimately shows no live sample rather
             than a missing rule.
* The b8931–b10430 era prints that live decode line without ``tg_3s``, as
             ``n_decoded = …, tg = … t/s`` — separate rule, the ``tg_3s``
             spelling above cannot match it.
* Final per-task block: the same text is emitted by three emitters — the server
             slot (``… tokens``), the lib's ``llama_perf_context_print`` (old
             ``llama_print_timings``) and ``common_perf_print`` (``… runs``) —
             which is why the eval-time rule accepts both words. Server blocks
             may be multi-line, with the timing lines themselves unprefixed
             (b80xx emits one ``SLT_INF`` call with embedded newlines). The CLI
             family prints one ``[ Prompt: … t/s | Generation: … t/s ]`` bracket
             line per turn (tools/cli/cli-context.cpp) at default verbosity,
             which is why that pair of rules exists alongside the
             ``runs``-wording eval rule: the CLI's own ``eval time`` lines are
             gated behind ``--perf`` (default false, common/arg.cpp).
* Verified negatives: llama.cpp logs no slot-busy or request-queue count at any
             level (``n_processing_slots``/``n_idle_slots`` are DEBUG-only and
             appear only while ``/metrics`` or ``/slots`` is polled; the
             deferred count is never logged), no KV-cache usage figure (only the
             init capacity ``size = … cells`` and KV-full warnings) and no
             prefix-cache hit percentage — so ``run``, ``queue``, ``kv`` and
             ``hit`` have no llama source and their absence in the footer is
             correct, not a missed rule. Default verbosity is INFO with
             timestamps and prefixes always on (``common_init``), so no
             verbosity flag is needed to see the metric lines.

* TensorFold  ``[tensorfold] done req-477975f9986c prompt=43 cached=0
             thinking=True tokens=19 sha=… finish=stop tok/s=62.2 ttft=0.18s
             prefill=0.17s rounds=7 accepted=11/23`` — the deployed workload's
             real completion line (read in a live log, 2026-10-17; optional
             ``background preemptions=…`` and ``checkpoints=…`` tail fields only
             add text). ``tok/s`` is decode-only (``(collected - 1) /
             (finished - prefilled)``), ``prefill`` pure prefill seconds
             (``-1.00s`` when unknown), ``accepted=a/b`` the accepted/drafted
             speculative tokens (``0/0`` for a one-token request) and ``cached``
             the prefix-cache tokens of ``prompt``. Two consequences: the prompt
             rate is the fresh tokens over prefill, ``(prompt - cached) /
             prefill`` — cached tokens were served, not computed — and a ``0/0``
             acceptance means no drafts, not 0%, so it emits no sample. Access
             (``{ip} "GET {path}" {code} -``) and ``ended … reason=…`` lines
             carry no metrics.

* Aneda      ``stats: t=10 rank=0 tok_out=12 tok_in=4824 kv_util=0.000 kv_hit=0.000
             draft_accept=0.000 queue=0 engram_hit=0.501 engram_used=0.000 reqs_stock=2
             reqs_steered=2``, the picom (dsgb10) engine's periodic line, emitted on rank 0
             only, beside the front-end's ``stats: t=10 front queue=0.0 reqs_stock=1
             reqs_steered=0`` (read in a live log, 2026-11-24; the fields are defined in the
             recipe registry's ``recipes/ds4/picom-weightless-steering.patch``,
             ``Server::stats_tick`` and ``Stats::maybe_emit``). ``t`` is the interval's
             *measured* seconds (10..59: the tick rides the request loop), and
             ``tok_out``/``tok_in`` are that interval's counts, reset each print, so the
             rates are those counts over the line's own ``t`` — not a fixed cadence.
             ``reqs_stock + reqs_steered`` is what the rank **served** in the interval,
             which fills the ``run`` slot vLLM fills with its currently-running count:
             same slot, different quantity. ``kv_util`` is slot occupancy, ``kv_hit``
             prompt-token reuse, ``draft_accept`` accepted/verified, all 0..1. The Engram
             counters carry no denominator on the line (the engine prints ``0.000`` when it
             had no lookups), so a printed zero is a reading; a ratio whose denominator the
             interval never had (``tok_in`` 0, ``tok_out`` 0) is a missing measurement and
             emits no sample. The front line's request counts duplicate the engine's and its
             ``queue`` is a per-request mean, so only the engine line is scraped. The
             engine's printf hardcodes ``rank=0``, so the rule cannot and does not key on a
             rank.

Sparkline glyphs and palette live here (moved from ``_monitor_tui``) because
both the node-stats panel and the log pane's footer draw them. Nothing in this
module imports ``textual`` or ``rich``, so every rule unit-tests without a
terminal and the log stream's hot path never pays for a widget import.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
import math
import re

# ---------------------------------------------------------------------------
# Sparkline helpers (shared by the node-stats panel and the metrics footer)
# ---------------------------------------------------------------------------

# Sparklines are rendered as inline block glyphs inside the existing
# Static(markup=True) widgets — textual.widgets.Sparkline would force them into
# a widget grid for no gain.
_SPARK_GLYPHS = "▁▂▃▄▅▆▇█"
_SPARKLINE_WIDTH = 20  # ~40 s of trend at the default 2 s record cadence

# Sparkline colors come from the active theme's palette, stepped away from the
# background — lighten on dark themes, darken on light ones — so they stay
# visible either way.  Five hues x three luminance steps; the grid places
# same-hue entries far apart.  (Named ANSI colors cannot be used: Textual
# resolves only what its color table knows, so "bright_red" silently falls back
# to the default foreground — the whole reason these are theme variables.)
# Minimal ANSI themes (ansi-dark/ansi-light) collapse every shade onto its base,
# so they yield 4-5 distinct colors instead of 11; that is inherent to those
# palettes, not a bug here — the full sets come from textual-dark/light and the
# third-party themes.
_SPARK_FAMILIES = ("primary", "secondary", "accent", "success", "error")
_SPARK_STEPS_DARK = ("-lighten-1", "-lighten-2", "-lighten-3")
_SPARK_STEPS_LIGHT = ("-darken-1", "-darken-2", "-darken-3")
# (family index, step index) in cell order: GPU %, GPU Temp, GPU Power, CPU %,
# CPU Temp, GPU Clock, RAM %, Net Send, Net Recv, App, Cache
_SPARK_GRID = ((0, 0), (4, 0), (2, 0), (3, 0), (1, 0), (0, 1), (3, 1), (2, 1), (4, 1), (0, 2), (1, 1))


def _spark_palette(css: Mapping[str, object], dark: bool) -> tuple[str, ...]:
    """Eleven distinct sparkline colors, in cell order, from *css* theme variables."""
    steps = _SPARK_STEPS_DARK if dark else _SPARK_STEPS_LIGHT
    try:
        return tuple(str(css[_SPARK_FAMILIES[family] + steps[step]]) for family, step in _SPARK_GRID)
    except KeyError:  # unexpected theme shape: render uncolored rather than blow up
        return ()


def _at(palette: Sequence[str], index: int) -> str | None:
    """Palette entry *index*, or None when the palette is unset/short."""
    return palette[index] if index < len(palette) else None


def _sparkline(
    values: list[float],
    width: int = _SPARKLINE_WIDTH,
    lo: float | None = None,
    hi: float | None = None,
) -> str:
    """One block-glyph sparkline, newest right-aligned; blank (spaces) when empty.

    ``lo``/``hi`` pin the scale — percent metrics pass 0/100 so hosts stay
    comparable — and whichever end is not given uses the window min/max.  A flat
    window renders '▄' per point.
    """
    pts = list(values)[-width:]
    if not pts:
        return " " * width
    lo_eff = min(min(pts), lo) if lo is not None else min(pts)
    hi_eff = max(max(pts), hi) if hi is not None else max(pts)
    span = hi_eff - lo_eff
    if span <= 0:
        glyphs = "▄" * len(pts)
    else:
        glyphs = "".join(_SPARK_GLYPHS[min(7, max(0, round((v - lo_eff) / span * 7)))] for v in pts)
    return glyphs.rjust(width)


# ---------------------------------------------------------------------------
# Metric specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """One footer item: how to read it back and which palette cell colors it."""

    key: str  # canonical metric id, also the scraper's ring key
    label: str  # footer label
    kind: str  # "rate" | "count" | "pct"
    color_index: int  # _SPARK_GRID palette slot


# Display order is also priority order: narrow panes drop from the right; every
# label fits the footer's fixed 5-column label field, so a longer word would
# widen the whole row.
_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("run", "jobs", "count", 3),
    MetricSpec("queue", "queue", "count", 9),
    MetricSpec("prompt", "pp", "rate", 1),
    MetricSpec("gen", "tg", "rate", 0),
    MetricSpec("kv", "kv", "pct", 5),
    MetricSpec("hit", "hit", "pct", 8),
    MetricSpec("accept", "spec", "pct", 4),
    MetricSpec("engram_hit", "e-hit", "pct", 2),
    MetricSpec("engram_used", "e-use", "pct", 7),
)

# Aggregation per metric key: rates/counts add up across ranks, a 0-100 metric
# takes the worst rank (that is the pressure worth watching).
_KINDS: dict[str, str] = {spec.key: spec.kind for spec in _METRICS}


@dataclass(frozen=True)
class _MetricRule:
    """One compiled scrape rule; its provenance comment sits above the entry."""

    pattern: re.Pattern[str]
    key: str
    scale: float = 1.0
    ratio: bool = False  # two groups: value = g2 / (g1 + g2) * 100
    combine: str = "latest"  # "latest" overwrites; "max" keeps the fuller reading of one line


# Unsigned decimal (int or float) — non-capturing, so each rule has exactly one
# group except the ratio rules, which need two.
_NUM = r"\d+(?:\.\d+)?"

# TensorFold's per-request completion line (see the module docstring for the
# field meanings). The order is fixed by the emitter; only the optional
# ``background preemptions`` field floats, sitting between prefill and rounds.
# ``prefill`` may be ``-1.00s`` (unknown), which the handler treats as no sample.
_TENSORFOLD_DONE_RE = re.compile(
    r"\[tensorfold\] done req-\w+"
    r" prompt=(?P<prompt>\d+)"
    r" cached=(?P<cached>\d+)"
    r"(?: thinking=\w+)?"
    r" tokens=\d+"
    r" sha=\w+"
    r" finish=\w+"
    r" tok/s=(?P<tok_s>-?\d+(?:\.\d+)?)"
    r" ttft=-?\d+(?:\.\d+)?s"
    r" prefill=(?P<prefill>-?\d+(?:\.\d+)?)s"
    r"(?: background preemptions=\d+)?"
    r" rounds=\d+"
    r" accepted=(?P<accepted>\d+)/(?P<drafted>\d+)"
)

_ANEDA_STATS_RE = re.compile(
    r"^stats: t=(?P<t>\d+) rank=\d+ "
    r"tok_out=(?P<out>\d+) tok_in=(?P<in>\d+) "
    r"kv_util=(?P<kv>[\d.]+) kv_hit=(?P<hit>[\d.]+) draft_accept=(?P<accept>[\d.]+) "
    r"queue=(?P<queue>\d+) engram_hit=(?P<engram_hit>[\d.]+) engram_used=(?P<engram_used>[\d.]+) "
    r"reqs_stock=(?P<stock>\d+) reqs_steered=(?P<steered>\d+)"
)

_RULES: tuple[_MetricRule, ...] = (
    # vLLM v1+v0 engine stat line
    _MetricRule(re.compile(rf"Avg generation throughput: ({_NUM}) tokens/s"), "gen"),
    _MetricRule(re.compile(rf"Avg prompt throughput: ({_NUM}) tokens/s"), "prompt"),
    _MetricRule(re.compile(r"Running: (\d+) reqs"), "run"),
    _MetricRule(re.compile(r"Waiting: (\d+) reqs"), "queue"),
    _MetricRule(re.compile(r"Pending: (\d+) reqs"), "queue"),  # v0 wording
    _MetricRule(re.compile(rf"GPU KV cache usage: ({_NUM})%"), "kv"),
    _MetricRule(re.compile(rf"Prefix cache hit rate: ({_NUM})%"), "hit"),
    # vLLM spec-decode metrics line
    _MetricRule(re.compile(rf"Avg Draft acceptance rate: ({_NUM})%"), "accept"),
    # SGLang decode/prefill lines
    _MetricRule(re.compile(rf"gen throughput \(token/s\): ({_NUM})"), "gen"),
    _MetricRule(re.compile(rf"input throughput \(token/s\): ({_NUM})"), "prompt"),
    _MetricRule(re.compile(r"#running-req: (\d+)"), "run"),
    _MetricRule(re.compile(r"#queue-req: (\d+)"), "queue"),
    # KV pool usage, one rule per pool the engine itself reports, all folding
    # into "kv" as the fullest pool: a hybrid model holds full-attention and
    # sliding-window pools in separate allocators (PoolStats.full_capacity vs
    # swa_capacity), so either one can bind first, and SGLang's own headline
    # number is max(full, swa, mamba) — PoolStats.get_max_pool_usage()
    # (scheduler_components/pool_stats_observer.py). The plain spelling is
    # lookbehind-guarded so "full"/"swa"/"gpu"/"cpu" prefixes never feed it.
    _MetricRule(re.compile(rf"(?<![a-z] )token usage: ({_NUM})"), "kv", scale=100.0, combine="max"),
    _MetricRule(re.compile(rf"full token usage: ({_NUM})"), "kv", scale=100.0, combine="max"),
    _MetricRule(re.compile(rf"swa token usage: ({_NUM})"), "kv", scale=100.0, combine="max"),
    # Hybrid-SSM models carry a third pool the engine also counts in that max.
    _MetricRule(re.compile(rf"mamba usage: ({_NUM})"), "kv", scale=100.0, combine="max"),
    _MetricRule(re.compile(rf"accept rate: ({_NUM})"), "accept", scale=100.0),
    _MetricRule(re.compile(r"#new-token: (\d+), #cached-token: (\d+)"), "hit", ratio=True),
    # llama-server live decode: tg_3s is the windowed rate, steadier than tg
    _MetricRule(re.compile(rf"tg = +{_NUM} t/s, tg_3s = +({_NUM}) t/s"), "gen"),
    # b8931..b10430-era live decode: same line before tg_3s existed; the
    # tg_3s spelling above cannot match it
    _MetricRule(re.compile(rf"n_decoded = +\d+, tg = +({_NUM}) t/s"), "gen"),
    _MetricRule(
        re.compile(rf"prompt processing, n_tokens = +{_NUM}, progress = +{_NUM}, t = +{_NUM} s / +({_NUM}) tokens per second"),
        "prompt",
    ),
    # Speculative decoding's final per-task ratio: master prints it at INFO with a
    # "mean len" tail, b80xx prints the same ratio at CONT level (no prefix); the
    # value is 0..1 like vLLM's. master: "draft acceptance = %0.5f (%5d accepted / %5d generated)".
    _MetricRule(re.compile(rf"draft acceptance(?: rate)? = ({_NUM})"), "accept", scale=100.0),
    # Final timing pair; the lookbehind keeps the prompt line out of the gen metric
    _MetricRule(re.compile(rf"prompt eval time = +{_NUM} ms / +\d+ tokens \(.*?({_NUM}) tokens per second\)"), "prompt"),
    _MetricRule(re.compile(rf"(?<!prompt )eval time = +{_NUM} ms / +\d+ (?:runs|tokens)\s+\(.*?({_NUM}) tokens per second\)"), "gen"),
    # llama-cli's one rate line per turn, printed at default verbosity: its
    # ``eval time`` lines only appear with --perf (default off), so without this
    # the CLI path has no default capture at all
    _MetricRule(re.compile(rf"Prompt: ({_NUM}) t/s"), "prompt"),
    _MetricRule(re.compile(rf"Generation: ({_NUM}) t/s"), "gen"),
)

# One substring gate kills ~99.9% of non-metric lines before the
# regexes run; the pane's key-latency test (<300 ms under a 15k-line burst)
# depends on staying cheap.
_GATES: tuple[str, ...] = (
    "throughput",
    "tokens per second",
    " t/s",
    "eval time",
    "token usage",
    "hit rate",
    "accept",
    "#running-req",
    "#queue-req",
    "KV cache usage",
    "done req-",
    "stats: t=",
)


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


# How many lines of a tail page are remembered for "already ingested"
# alignment. One steady-state poll reads one page (LogTarget.page reads 500
# lines), so a page of state is all the alignment ever needs.
_PAGE_TAIL: int = 500


def _page_overlap(previous: Sequence[int], new: Sequence[int]) -> int:
    """Length of the longest suffix of *previous* that prefixes *new*, or 0.

    Consecutive tail-page reads overlap by everything the log did not append, so
    this is how many leading lines of *new* were already ingested.
    """
    for k in range(min(len(previous), len(new)), 0, -1):
        if previous[-k:] == new[:k]:
            return k
    return 0


class RuntimeMetricsScraper:
    """Ingest-scraped runtime metrics for one workload. Pure; no widgets.

    The caller polls on a fixed cadence: ``observe_page()`` ingests one tail
    page (``observe()`` still takes a single line), and ``commit()`` runs once
    per poll, so every poll contributes exactly one ring point per source: the
    mean of every value that source reported in that window. Values are held
    per source: several ranks report the same metric, and summing (for additive
    metrics) their window means is what a user reads as the total.

    Aggregation happens at read time over an optional source subset, so one
    scraper can serve several views of the same workload (the log pane filters
    to the sources in its current scope) without re-ingesting anything.
    """

    def __init__(self, history_points: int = 120) -> None:
        self._history_points = history_points
        self._window: dict[str, dict[tuple[str, str], tuple[float, int]]] = {}
        self._rings: dict[str, dict[tuple[str, str], deque[tuple[int, float]]]] = {}
        self._poll = 0
        self._dirty: set[str] = set()
        # Hashes of the last ingested page per source, for observe_page alignment.
        self._tails: dict[tuple[str, str], list[int]] = {}

    def observe(self, source: tuple[str, str], text: str) -> None:
        """Record every metric *text* reports, attributed to *source*."""
        self._observe_line(source, text, replace=False)

    def _observe_line(self, source: tuple[str, str], text: str, *, replace: bool) -> None:
        """Scrape one line into *source*'s window; *replace* seeds, else averages."""
        if not any(gate in text for gate in _GATES):
            return
        self._observe_tensorfold_done(source, text, replace=replace)
        self._observe_aneda_stats(source, text, replace=replace)
        pooled: dict[str, float] = {}
        for rule in _RULES:
            match = rule.pattern.search(text)
            if match is None:
                continue
            if rule.ratio:
                fresh = float(match.group(1))
                cached = float(match.group(2))
                total = fresh + cached
                if total <= 0:
                    continue
                value = cached / total * 100.0
            else:
                value = float(match.group(1)) * rule.scale
            if rule.combine == "max":
                # Several pools on one line, one metric: keep the fullest, which
                # is what the engine's own get_max_pool_usage() reports.
                pooled[rule.key] = max(pooled.get(rule.key, value), value)
                continue
            self._sample(rule.key, source, value, replace=replace)
        for key, value in pooled.items():
            self._sample(key, source, value, replace=replace)

    def _sample(self, key: str, source: tuple[str, str], value: float, *, replace: bool = False) -> None:
        """Add one sample to *key*'s current window for *source*.

        *replace* overwrites the source's entry instead of accumulating it: a
        seed page is history of unknown duration, so its last sample is the
        current reading and the page mean would report minutes.
        """
        window = self._window.setdefault(key, {})
        if replace:
            window[source] = (value, 1)
        else:
            total, count = window.get(source, (0.0, 0))
            window[source] = (total + value, count + 1)
        self._dirty.add(key)

    def observe_page(self, source: tuple[str, str], texts: Sequence[str]) -> None:
        """Ingest one tail page for *source*, skipping what was already observed.

        Consecutive tail pages overlap by everything the log did not append, so
        only the suffix is new to this window. The pages are aligned
        positionally over line hashes, so genuinely repeated identical lines
        still count; a page with no overlap (rotation, truncation) is
        re-ingested whole. An empty page is not a reset: it leaves the
        remembered tail untouched so the next read still aligns. The remembered
        tail is bounded to one steady-state page.

        A page wider than the alignment window (``_PAGE_TAIL``) — a source's
        first read, or the tracker's re-prime after a read failure — is history
        of unknown duration: the remembered hashes sit at its end, not its
        start, so it cannot align positionally, and each metric seeds the
        window with its last sample instead of averaging the page — the first
        point is a reading, not a mean over minutes. That also holds when a
        stale tail survives from before the gap: re-ingesting it would
        re-average lines already committed to the rings.
        """
        page = [hash(text) for text in texts]
        if not page:
            return
        previous = self._tails.get(source) or []
        seed = not previous or len(page) > _PAGE_TAIL
        skip = 0 if seed else _page_overlap(previous, page)
        for text in texts[skip:]:
            self._observe_line(source, text, replace=seed)
        self._tails[source] = page[-_PAGE_TAIL:]

    def _observe_tensorfold_done(self, source: tuple[str, str], text: str, *, replace: bool) -> None:
        """Record TensorFold's completion line: decode rate, prefill rate, cache, spec.

        ``tok/s`` is decode-only, so it is the generation rate directly. The
        prompt rate is the fresh tokens over the pure prefill time: ``cached``
        tokens were served from the prefix cache, not computed. A ``0/0``
        ``accepted`` ratio is "no drafts", not 0% acceptance, so it is skipped.
        """
        match = _TENSORFOLD_DONE_RE.search(text)
        if match is None:
            return
        prompt = int(match.group("prompt"))
        cached = int(match.group("cached"))
        self._sample("gen", source, float(match.group("tok_s")), replace=replace)
        prefill = float(match.group("prefill"))
        if prefill > 0:
            self._sample("prompt", source, max(0, prompt - cached) / prefill, replace=replace)
        if prompt > 0:
            self._sample("hit", source, cached / prompt * 100.0, replace=replace)
        accepted = int(match.group("accepted"))
        drafted = int(match.group("drafted"))
        if drafted > 0:
            self._sample("accept", source, accepted / drafted * 100.0, replace=replace)

    def _observe_aneda_stats(self, source: tuple[str, str], text: str, *, replace: bool) -> None:
        """Record aneda's periodic engine stats line (see the module docstring).

        ``t`` is the line's own measured window in seconds and ``tok_out``/``tok_in`` are
        that window's counts, so the rates are those counts over ``t``. A ratio whose
        denominator the window never had (``tok_in`` 0, ``tok_out`` 0) is a missing
        measurement, not 0%, so no sample is taken; the Engram counters carry no
        denominator on the line, so their printed ``0.000`` is a reading.
        """
        match = _ANEDA_STATS_RE.search(text)
        if match is None:
            return
        window = int(match.group("t"))
        out_tokens = int(match.group("out"))
        in_tokens = int(match.group("in"))
        if window > 0:
            self._sample("gen", source, out_tokens / window, replace=replace)
            self._sample("prompt", source, in_tokens / window, replace=replace)
        self._sample("run", source, float(int(match.group("stock")) + int(match.group("steered"))), replace=replace)
        self._sample("queue", source, float(match.group("queue")), replace=replace)
        self._sample("kv", source, float(match.group("kv")) * 100.0, replace=replace)
        if in_tokens > 0:
            self._sample("hit", source, float(match.group("hit")) * 100.0, replace=replace)
        if out_tokens > 0:
            self._sample("accept", source, float(match.group("accept")) * 100.0, replace=replace)
        if in_tokens > 0 or out_tokens > 0:
            self._sample("engram_hit", source, float(match.group("engram_hit")) * 100.0, replace=replace)
        self._sample("engram_used", source, float(match.group("engram_used")) * 100.0, replace=replace)

    def commit(self) -> None:
        """Close the window: one ring point per source per metric.

        The point is the mean of that source's samples in the window — one
        tracker poll, 30 s by default — so several engine prints inside one
        poll land as one point. A metric with no samples in the window adds no
        point; a source with no samples contributes nothing, so a gap in one
        rank never reads as a zero. An idle workload therefore adds no points
        and its number holds: a quiet window is a non-report, never a repeated
        value.
        """
        self._poll += 1
        for spec in _METRICS:
            if spec.key not in self._dirty:
                continue
            window = self._window.get(spec.key)
            if not window:
                continue
            rings = self._rings.setdefault(spec.key, {})
            for source, (total, count) in window.items():
                if count:
                    rings.setdefault(source, deque(maxlen=self._history_points)).append((self._poll, total / count))
            window.clear()
        self._dirty.clear()

    def history(self, key: str, sources: Collection[tuple[str, str]] | None = None) -> list[float]:
        """Committed aggregate series for *key*, oldest first.

        Sources are aligned by poll identity, so a late or stalled source
        contributes only to the windows in which it reported. Empty windows
        are omitted. *sources* restricts the aggregation to that subset
        (None means every source).
        """
        rings = self._rings.get(key)
        if not rings:
            return []
        keys = sorted(source for source in rings if sources is None or source in sources)
        if not keys:
            return []
        pct = _KINDS.get(key) == "pct"
        windows: dict[int, float] = {}
        for source in keys:
            for poll, value in rings[source]:
                windows[poll] = max(windows.get(poll, value), value) if pct else windows.get(poll, 0.0) + value
        polls = sorted(windows)[-self._history_points :]
        return [windows[poll] for poll in polls]

    def latest(self, key: str, sources: Collection[tuple[str, str]] | None = None) -> float | None:
        """Last committed aggregate for *key* (over *sources*), or None before the first commit."""
        series = self.history(key, sources)
        return series[-1] if series else None

    def retain(self, sources: Collection[tuple[str, str]]) -> None:
        """Forget every source outside *sources* — a rank whose container instance changed.

        Without this a dead rank's last value stays in ``_window`` and is
        summed into every later commit, inflating the total forever.
        """
        keep = set(sources)
        for key in list(self._window):
            window = self._window[key]
            for source in [s for s in window if s not in keep]:
                del window[source]
        for key in list(self._rings):
            rings = self._rings[key]
            for source in [s for s in rings if s not in keep]:
                del rings[source]
        for source in [s for s in self._tails if s not in keep]:
            del self._tails[source]

    def reset(self) -> None:
        """Drop every sample — two workloads' streams must not blend."""
        self._window.clear()
        self._rings.clear()
        self._poll = 0
        self._dirty.clear()
        self._tails.clear()


# ---------------------------------------------------------------------------
# Footer rendering
# ---------------------------------------------------------------------------


def _round_half_up(value: float) -> int:
    """Nearest integer, halves rounded up.

    Counts are whole things (running requests), so a fractional window average
    still has to print as one.  ``round()`` and ``format(…, ".0f")`` both round
    halves to even — a 2.5 average would print as 2 — which reads as an
    undercount in a footer whose whole job is to be a truthful glance.
    """
    return math.floor(value + 0.5)


def render_metrics_footer(
    scraper: RuntimeMetricsScraper,
    palette: Sequence[str],
    width: int,
    spark_width: int = 5,
    sources: Collection[tuple[str, str]] | None = None,
) -> str:
    """One row of ``label ▁▂▃ value`` items, in priority order, as Rich markup.

    A metric that never appeared in the stream is absent, and items that do not
    fit in *width* are dropped rather than wrapped — the row is a glance, not a
    table.  An empty string means there is nothing to show and the caller hides
    the row.  *sources* narrows the aggregation to that source subset (None
    means every source the scraper has seen).

    The number beside each sparkline is the average of the most recent sampled
    window (the newest committed point) — the same sample the sparkline's
    rightmost point draws.  Labels occupy a fixed 5-column field.
    """
    if width <= 0:
        return ""
    label_width = 5
    items: list[tuple[str, int]] = []
    for spec in _METRICS:
        history = scraper.history(spec.key, sources)
        if not history:
            continue
        window = history[-spark_width:]
        newest = history[-1]
        if spec.kind == "pct":
            spark = _sparkline(window, width=spark_width, lo=0.0, hi=100.0)
            shown = f"{newest:.0f}%"
        else:
            spark = _sparkline(window, width=spark_width)
            shown = f"{newest:.1f}" if spec.kind == "rate" else f"{_round_half_up(newest)}"
        color = _at(palette, spec.color_index)
        if color:
            spark = f"[{color}]{spark}[/{color}]"
        label = f"{spec.label:<{label_width}}"
        items.append((f"[dim]{label}[/dim] {spark} {shown}", label_width + spark_width + len(shown) + 2))

    kept: list[str] = []
    used = 0
    for markup, plain_len in items:
        needed = plain_len if not kept else plain_len + 2  # two-space separator
        if used + needed > width:
            break
        used += needed
        kept.append(markup)
    return "  ".join(kept)
