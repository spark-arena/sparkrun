"""Content-based colorizer classes for the monitor log pane.

The pane's ``c`` toggle renders engine logs, and this module decides which
characters of a line carry which style. Selection happens in two steps:

* A known runtime name selects that runtime's colorizer class outright. The pane
  knows its workload's runtime (``LogTarget.runtime``), so an in-tree engine gets
  its dedicated class deterministically.
* An unknown or absent name — an out-of-tree engine, such as TensorFold — falls
  back to content sniffing: the first colorizer whose signature matches the line,
  then :class:`GenericColorizer`.

Every colorizer shares one generic vocabulary, so a line no in-tree runtime ever
emitted still gets its levels, status codes, and numbers colored. Formats were
verified against the engines' own sources:

* vLLM        ``(EngineCore pid=9527) INFO 10-11 08:12:34 [core.py:130] …`` (vllm/logger.py _FORMAT)
* SGLang      ``[2026-10-11 08:12:34 TP0] INFO backend ready`` (sglang/srt/utils/common.py)
* llama.cpp   ``0.00.035.064 I llama_model_load: vocab only`` (common/log: [M.s.ms.us] L message, I/W/E/D)
* TRT-LLM     ``[10/11/2026-08:12:34 UTC] [E] rank 1 exited`` and glog ``E1011 08:12:34.123 …``
* NCCL        ``dgx01:28971 [0] NCCL INFO Ring ALGO`` (sparkrun sets NCCL_DEBUG=INFO)
* Atlas       ``[2026-10-11T08:12:34Z WARN atlas::serve] …`` (Rust env_logger) and tracing
* TensorFold  ``[tensorfold] {message}`` with NO level token (zig/src/server/log.zig):
              ``warning: …``, ``refused …``, ``engine ready``, ``serving … at http://…``,
              ``done … finish=stop|length …``, ``ended … reason=client-left|error …``, and
              access lines ``ip "GET …" 4xx/5xx -``; fatals ``tensorfold: {s}`` on stderr
              (main.zig fail()); the launcher prefixes its own lines ``[tensorfold-launcher]``.
              Shapes verified against a real workload log, 2026-10-17.

Styles are plain Rich style strings, and nothing here imports ``rich`` or ``textual``,
so every rule unit-tests without a terminal.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import re
from functools import lru_cache

# Level palette ----------------------------------------------------------------

#: Longest real prefix before a level token is SGLang's timestamp + TP tag
#: (~35 chars). 64 leaves headroom while keeping a level word buried in message
#: prose (for example "downloaded INFO.txt") from being colored. Case-sensitive:
#: runtimes emit uppercase levels, and TensorFold's lowercase ``warning:`` is
#: matched by its own prefix-anchored rule instead.
_LEVEL_PREFIX_CHARS = 64

_LEVEL_TOKEN_RE = re.compile(r"\b(TRACE|DEBUG|INFO|NOTICE|WARNING|WARN|ERROR|ERR|CRITICAL|CRIT|FATAL)\b")
_LEVEL_STYLES: dict[str, str] = {
    "TRACE": "dim",
    "DEBUG": "dim",
    "NOTICE": "cyan",
    "INFO": "cyan",
    "WARN": "yellow",
    "WARNING": "yellow",
    "ERR": "red",
    "ERROR": "red",
    "CRIT": "bold red",
    "CRITICAL": "bold red",
    "FATAL": "bold red",
}
_SHORT_LETTER_STYLES: dict[str, str] = {"I": "cyan", "W": "yellow", "E": "red", "F": "bold red", "D": "dim"}

# Data palette -----------------------------------------------------------------

#: Rank tags (``TP0``, ``rank=1``, NCCL's ``[0]``) cycle this palette by rank.
_RANK_STYLES = ("cyan", "magenta", "green", "yellow", "blue", "red")

STYLE_SEVERE = "red"
STYLE_WARN = "yellow"
STYLE_DATA = "yellow"  # numbers, durations, key=value values
STYLE_RATE = "bold yellow"  # per-second rates: 429.5 tokens/s, tok/s=100.0
STYLE_PREFILL = "cyan"  # prompt-processing phase words: Prefill, prompt eval time
STYLE_DECODE = "green"  # token-generation phase words: Decode, eval time
STYLE_METHOD = "bold cyan"
STYLE_PATH = "cyan"
STYLE_URL = "underline blue"
STYLE_PROCESS = "bold blue"  # process/module tags: (EngineCore pid=…), atlas::serve, llama_model_load
STYLE_ID = "magenta"
STYLE_OK = "green"
STYLE_TRACEBACK = "bold red"

# Rules ------------------------------------------------------------------------

#: One rule: plain text -> candidate spans. Candidates may overlap another rule's
#: candidates; :func:`_claim` keeps the first and drops the rest.
Rule = Callable[[str], list[tuple[int, int, str]]]


def _re_rule(pattern: str, style: str | Callable[[re.Match[str]], str], flags: int = 0, span_group: int = 0) -> Rule:
    """Style every non-overlapping match of *pattern*.

    ``span_group`` exists because Python's ``re`` requires fixed-width
    lookbehinds: patterns wanting a path after one of several HTTP methods, a
    Python logger name after a level token, or a llama.cpp ``key: value`` number
    capture the wanted part in a group instead.
    """
    compiled = re.compile(pattern, flags)

    def rule(plain: str) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        for m in compiled.finditer(plain):
            value: str = style if isinstance(style, str) else style(m)
            spans.append((m.start(span_group), m.end(span_group), value))
        return spans

    return rule


def _whole_line_rule(pattern: str, style: str, flags: int = 0) -> Rule:
    """Style the entire line when *pattern* matches anywhere in it."""
    compiled = re.compile(pattern, flags)

    def rule(plain: str) -> list[tuple[int, int, str]]:
        return [(0, len(plain), style)] if compiled.search(plain) else []

    return rule


def _level_head_rule() -> Rule:
    """Level tokens inside the head window ``plain[:_LEVEL_PREFIX_CHARS]``."""

    def rule(plain: str) -> list[tuple[int, int, str]]:
        head = plain[:_LEVEL_PREFIX_CHARS]
        return [(m.start(), m.end(), _LEVEL_STYLES[m.group(1)]) for m in _LEVEL_TOKEN_RE.finditer(head)]

    return rule


def _short_letter_style(m: re.Match[str]) -> str:
    return _SHORT_LETTER_STYLES[m.group(1)]


#: ``key=value`` values whose key names a per-second rate take the rate style.
_RATE_KEY_RE = re.compile(r"(?:tok(?:ens?)?/s|t/s|reqs?/s|throughput|tput)", re.IGNORECASE)


def _value_style(m: re.Match[str]) -> str:
    return STYLE_RATE if _RATE_KEY_RE.fullmatch(m.group(1).strip()) else STYLE_DATA


def _status_style(m: re.Match[str]) -> str:
    # Total over 1xx-5xx: a 101 upgrade line must color, not raise.
    return {"1": STYLE_PATH, "2": STYLE_OK, "3": STYLE_PATH, "4": STYLE_WARN, "5": STYLE_SEVERE}[m.group(1)[0]]


def _rank_style(m: re.Match[str]) -> str:
    return _RANK_STYLES[int(m.group(1)) % len(_RANK_STYLES)]


#: Traceback header, and the bare exception line that ends a crash.
_WHOLE_LINE_RULES: tuple[Rule, ...] = (
    _whole_line_rule(r"^Traceback \(most recent call last\):", STYLE_TRACEBACK),
    _whole_line_rule(r"^(?:[A-Za-z_][\w.]*\.)?[A-Za-z_]*(?:Error|Exception|Interrupt|Exit|Fault)(?::|$)", STYLE_TRACEBACK),
)

#: Every level form is its own rule, so a line carrying both an infix level token
#: and a bracket level gets both colored.
_LEVEL_RULES: tuple[Rule, ...] = (
    _level_head_rule(),
    _re_rule(r"\[([IWEF])\]", _short_letter_style),  # TRT-LLM bracket levels
    _re_rule(r"(?<![A-Za-z])([IWEF])(?=\d{4}\s)", _short_letter_style, span_group=1),  # glog E1011
    _re_rule(r"^(?:\d[\d.]*\s+|\[[\d.]+\]\s+)?([IWED])(?=\s)", _short_letter_style, span_group=1),  # llama.cpp
)

#: Failure and warning words in message prose. Word-bounded and case-insensitive,
#: so ``error_code`` and ``ValueError`` stay uncolored.
_SEVERITY_RULES: tuple[Rule, ...] = (
    _re_rule(
        r"\b(?:errors?|fail(?:ed|ing|ures?)|refused|denied|timeouts?|timed out|ooms?|out of memory|killed"
        r"|panics?|panicked|fatal|unhealthy|abort(?:ed)?|exceptions?|corrupt(?:ed)?|crashed?)\b",
        STYLE_SEVERE,
        re.IGNORECASE,
    ),
    _re_rule(r"\b(?:warns?|warned|warnings?|deprecated|retr(?:y|ies|ying)|degraded|fallback|slow)\b", STYLE_WARN, re.IGNORECASE),
)

#: Phase anchors, so a stream of interleaved prefill/decode lines scans by color:
#: prompt processing is cyan, token generation green. Word-bounded, so
#: ``prefill_step`` and ``chunked_prefill`` stay uncolored, taking the
#: engine-capitalized and lowercase spellings alike.
_PHASE_RULES: tuple[Rule, ...] = (
    _re_rule(r"\b(?:Avg )?prompt throughput\b", STYLE_PREFILL),  # vLLM stats log
    _re_rule(r"\b(?:Avg )?generation throughput\b", STYLE_DECODE),  # vLLM stats log
    _re_rule(r"\binput throughput\b", STYLE_PREFILL),  # SGLang prefill batch: "input throughput (token/s)"
    _re_rule(r"\bgen throughput\b", STYLE_DECODE),  # SGLang decode batch: "gen throughput (token/s)"
    _re_rule(r"\bprompt eval time\b", STYLE_PREFILL),  # llama.cpp prefill block
    _re_rule(r"\beval time\b", STYLE_DECODE),  # llama.cpp generation block; the prompt pair is claimed above
    _re_rule(r"\bprompt processing\b", STYLE_PREFILL),  # llama.cpp live prefill line ("prompt processing, n_tokens = …")
    _re_rule(r"\b[Pp]refill\b", STYLE_PREFILL),  # SGLang/TRT-LLM "Prefill batch"
    _re_rule(r"\b[Dd]ecode\b", STYLE_DECODE),  # SGLang "Decode batch"
)

#: Per-second rate units, slash and spelled forms (llama.cpp: "0.04 tokens per second").
_RATE_UNIT = r"(?:[Tt]okens?/s|tok/s|t/s|[Rr]eqs?/s|[Tt]okens? per second|[Rr]eqs? per second)"

#: The data every runtime's lines carry: numbers, request shapes, ids, ranks.
_DATA_RULES: tuple[Rule, ...] = (
    # Ranks first: ``rank=2`` is a rank, not just another key=value number.
    _re_rule(r"\b(?:TP|tp|Rank|rank|local_rank|world_size)[=:]?\s?(\d{1,3})\b", _rank_style),  # TP0, rank=1
    _re_rule(r"\[(\d{1,3})\](?= NCCL)", _rank_style),  # NCCL dgx01:28971 [0]
    # Rates before the rest: ``0.0 tokens/s`` is one rate, not a bare number plus prose.
    # The lookahead keeps ``tokens=200 tok/s=100.0`` honest: a unit followed by ``=``
    # is the next key's name, so the ``200`` stays a count in the key=value rule.
    _re_rule(rf"\b\d[\d._]*\s?{_RATE_UNIT}\b(?!\s*=)", STYLE_RATE),  # 429.5 tokens/s
    _re_rule(rf"\b{_RATE_UNIT}\)?\s*: (\d[\d._]*)", STYLE_RATE, span_group=1),  # input throughput (token/s): 536.67
    _re_rule(r"\b\d[\d._]*\s?%", STYLE_DATA),  # GPU KV cache usage: 89.0%
    _re_rule(r"\b\d[\d._]*\s?(?:KiB|MiB|GiB|TiB|KB|MB|GB|TB)\b", STYLE_DATA),  # Available KV cache memory: 5.77 GiB
    _re_rule(r"([a-z_][\w/.\-]*)=(\d[\d._]*(?:[a-zµ%][\w/µ%]*)?)", _value_style, re.IGNORECASE, span_group=2),  # prompt=100, tok/s=100.0
    _re_rule(r"\b\d+(?:\.\d+)?\s?(?:ms|µs|us|ns|min|s)\b", STYLE_DATA),  # bare 1.25s / 250ms
    _re_rule(r"\b(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b", STYLE_METHOD),
    _re_rule(r"\b(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/[^\s\"']*)", STYLE_PATH, span_group=1),
    _re_rule(r"https?://[^\s\"')]+", STYLE_URL),
    _re_rule(r"(?<=\" )([1-5]\d\d)(?=[ -])", _status_style),  # access lines: "GET … HTTP/1.1" 200 OK / 503 -
    _re_rule(r"\b(?:req|request|chatcmpl|cmpl|resp|response|trace|span)[-_][0-9A-Za-z]{1,}\b", STYLE_ID),
    _re_rule(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", STYLE_ID),
    # Colon-form stats: SGLang ``#new-token: 384``, vLLM ``Running: 24 reqs``.
    _re_rule(r"([A-Za-z#][\w#.\-/ ]{0,39}): (\d[\d._]*(?:,\d{3})*)", STYLE_DATA, span_group=2),
    _re_rule(r"^(?:TRACE|DEBUG|INFO|NOTICE|WARNING|WARN|ERROR|CRITICAL|FATAL):([\w.]+):", STYLE_PROCESS, span_group=1),
)


def _claim(rules: Sequence[Rule], plain: str) -> list[tuple[int, int, str]]:
    """Run *rules* in order, keeping each candidate that claims free characters."""
    claimed: list[tuple[int, int, str]] = []
    taken: list[tuple[int, int]] = []
    limit = len(plain)
    for rule in rules:
        for start, end, style in rule(plain):
            start, end = max(0, min(start, limit)), max(0, min(end, limit))
            if end <= start or any(start < t_end and end > t_start for t_start, t_end in taken):
                continue
            taken.append((start, end))
            claimed.append((start, end, style))
    claimed.sort(key=lambda span: span[0])
    return claimed


def unclaimed_ranges(total: int, claimed: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Complement of *claimed* over ``[0, total)``, merged, sorted, and clamped.

    The pane dims these ranges: highlights and native ANSI runs are claimed, so
    dimming never touches them.
    """
    merged: list[tuple[int, int]] = []
    for start, end in sorted(claimed):
        start, end = max(0, start), min(total, end)
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    gaps: list[tuple[int, int]] = []
    position = 0
    for start, end in merged:
        if start > position:
            gaps.append((position, start))
        position = max(position, end)
    if position < total:
        gaps.append((position, total))
    return gaps


# Colorizers -------------------------------------------------------------------


class LogColorizer:
    """Base colorizer: the generic vocabulary every runtime's lines share.

    Subclass to add a runtime's own shapes: set :attr:`signature` so sniffing can
    recognize it and :attr:`own_rules` for the class-specific spans. Override
    :meth:`spans` outright when a runtime needs different ordering.
    """

    name: str = "generic"
    signature: re.Pattern[str] | None = None
    own_rules: tuple[Rule, ...] = ()

    def rules(self) -> list[Rule]:
        return [*self.own_rules, *_WHOLE_LINE_RULES, *_LEVEL_RULES, *_SEVERITY_RULES, *_PHASE_RULES, *_DATA_RULES]

    def recognizes(self, plain: str) -> bool:
        return bool(self.signature and self.signature.search(plain))

    def spans(self, plain: str) -> list[tuple[int, int, str]]:
        return _claim(self.rules(), plain)


class GenericColorizer(LogColorizer):
    """Fallback for engines sparkrun does not know, and the last sniff candidate."""


class VllmColorizer(LogColorizer):
    name = "vllm"
    signature = re.compile(
        r"^\(?(?:EngineCore|Worker|APIServer|AsyncLLMEngine)[\w_]*[\):]"
        r"|\b(?:TRACE|DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL) \d\d-\d\d \d\d:\d\d:\d\d \["
    )
    own_rules = (_re_rule(r"(?<=\()(?:EngineCore|Worker|APIServer|AsyncLLMEngine)(?:_\d+)?(?=[)\s])", STYLE_PROCESS),)


class SglangColorizer(LogColorizer):
    name = "sglang"
    signature = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} TP\d+\]")


class LlamaCppColorizer(LogColorizer):
    name = "llama-cpp"
    signature = re.compile(r"^(?:\[\d[\d.]*\]\s*|\d[\d.]+\s+)[IWEDC]\s")
    own_rules = (
        _re_rule(r"\b(?:llama|common|ggml|metal|cuda|vulkan|sycl|tokeniz|model)\w*(?=:)", STYLE_PROCESS),
        _re_rule(r"\b\w{2,20}: (\d[\d.]*)", STYLE_DATA, span_group=1),  # n_ctx: 4096
    )


class TrtLlmColorizer(LogColorizer):
    name = "trtllm"
    signature = re.compile(r"^\[\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2} UTC\]|^[IWEF]\d{4} \d{2}:\d{2}:\d{2}\.\d+")
    own_rules = (_re_rule(r"\[TRT-LLM\]", STYLE_PROCESS),)


class AtlasColorizer(LogColorizer):
    name = "atlas"
    signature = re.compile(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z[\s\]]"
        r"|^\[[\d\-T:.Z]+ [A-Z]+(?: [\w:.]+)?\]"
    )
    own_rules = (_re_rule(r"\b(?:atlas|tower|axum|hyper|reqwest)(?:::\w+)+\b", STYLE_PROCESS),)


class TensorFoldColorizer(LogColorizer):
    """TensorFold (ashhart/TensorFold): server lines are ``[tensorfold] {message}`` with no
    level token (zig/src/server/log.zig), the launcher's are ``[tensorfold-launcher] …``, so
    its rules anchor to those prefixes; ``tensorfold: {s}`` startup fatals on stderr
    (main.zig) color whole-line. Outcome tokens keep their semantics: ``done``/``engine
    ready`` green, ``finish=length`` (reply hit the token limit) yellow, ``reason=client-left``
    dim (a normal client cancellation) while ``reason=error`` stays severe. Verified against a
    real workload log, 2026-10-17.
    """

    name = "tensorfold"
    signature = re.compile(r"^\[tensorfold\] |^tensorfold: |^\[tensorfold-launcher\] ")
    own_rules = (
        _whole_line_rule(r"^tensorfold: ", STYLE_SEVERE),
        _re_rule(r"^\[tensorfold-launcher\]", STYLE_PROCESS),
        _re_rule(r"(?<=\[tensorfold\] )warning:", STYLE_WARN),
        _re_rule(r"(?<=\[tensorfold\] )refused(?=\s)", STYLE_SEVERE),
        _re_rule(r"(?<=\[tensorfold\] )engine ready\b", STYLE_OK),
        _re_rule(r"(?<=\[tensorfold\] )serving(?= .* at http)", STYLE_OK),
        _re_rule(r"\breason=error\b", STYLE_SEVERE),
        _re_rule(r"\breason=client-left\b", "dim"),
        _re_rule(r"(?<=\[tensorfold\] )done(?=\s)", STYLE_OK),
        _re_rule(r"\bfinish=stop\b", STYLE_OK),
        _re_rule(r"\bfinish=length\b", STYLE_WARN),
        # One data span for the spec-decode ratio: the generic key=value rule
        # would claim only the numerator ("accepted=11" of "accepted=11/23").
        _re_rule(r"\baccepted=\d+/\d+\b", STYLE_DATA),
        # The completion hash: hex, so the digit-starting key=value rule misses it.
        _re_rule(r"\bsha=[0-9a-f]{6,}\b", STYLE_ID),
    )


_VLLM = VllmColorizer()
_SGLANG = SglangColorizer()
_LLAMA_CPP = LlamaCppColorizer()
_TRT_LLM = TrtLlmColorizer()
_ATLAS = AtlasColorizer()
_TENSORFOLD = TensorFoldColorizer()

#: Sniffing order: the first colorizer whose signature matches the line wins.
COLORIZERS: tuple[LogColorizer, ...] = (_VLLM, _SGLANG, _LLAMA_CPP, _TRT_LLM, _ATLAS, _TENSORFOLD)
GENERIC_COLORIZER = GenericColorizer()

#: Canonical runtime name -> colorizer. Keys cover the in-tree runtimes
#: (``vllm-ray``, ``vllm-distributed``, ``eugr-vllm``, ``sglang``, ``trtllm``,
#: ``atlas``, ``llama-cpp``, ``tensorfold``) plus aliases. ``modular-max`` and
#: ``tokenary`` are deliberately absent: their lines are Python-logging shaped,
#: which the generic vocabulary covers, so sniffing handles them.  The
#: DeepSeek community fork logs under the same ``[tensorfold]``/``tensorfold:``
#: prefixes, so the single runtime's alias covers it without content sniffing.
RUNTIME_COLORIZERS: dict[str, LogColorizer] = {
    "vllm": _VLLM,
    "vllm-ray": _VLLM,
    "vllm-distributed": _VLLM,
    "eugr-vllm": _VLLM,
    "sglang": _SGLANG,
    "llama-cpp": _LLAMA_CPP,
    "llamacpp": _LLAMA_CPP,
    "trtllm": _TRT_LLM,
    "trt-llm": _TRT_LLM,
    "atlas": _ATLAS,
    "tensorfold": _TENSORFOLD,
}

_BY_NAME: dict[str, LogColorizer] = {colorizer.name: colorizer for colorizer in (*COLORIZERS, GENERIC_COLORIZER)}


def colorizer_for(runtime: str | None) -> LogColorizer | None:
    """The colorizer a runtime name selects, or None when the name is unknown."""
    if not runtime:
        return None
    return RUNTIME_COLORIZERS.get(runtime.strip().lower())


def pick_colorizer(plain: str, runtime: str | None = None) -> LogColorizer:
    """The colorizer for one line: a known runtime name wins, else sniff, else generic."""
    named = colorizer_for(runtime)
    if named is not None:
        return named
    for colorizer in COLORIZERS:
        if colorizer.recognizes(plain):
            return colorizer
    return GENERIC_COLORIZER


@lru_cache(maxsize=8192)
def _spans_cached(plain: str, colorizer_name: str) -> tuple[tuple[int, int, str], ...]:
    """Bounded memo for :func:`colorize_log_spans`.

    The pane re-renders every buffered line on a resize or a wrap toggle, so the
    rule set must not be paid per line per event. Keyed by colorizer name as well
    as text, so a pane retarget can never serve spans from another runtime.
    """
    return tuple(_BY_NAME[colorizer_name].spans(plain))


def colorize_log_spans(plain: str, *, runtime: str | None = None) -> list[tuple[int, int, str]]:
    """Spans to stylize on one ANSI-stripped log line: sorted, disjoint, in bounds."""
    return list(_spans_cached(plain, pick_colorizer(plain, runtime).name))
