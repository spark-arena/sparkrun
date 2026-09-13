"""String ⇄ value parsing and formatting helpers."""

from __future__ import annotations

import logging
import re
from typing import Any

from vpd.legacy.arguments import arg_substitute

logger = logging.getLogger(__name__)

# Brace masking sentinels.  Control characters YAML 1.2 forbids in scalar
# content, so they cannot collide with anything a recipe or an override value
# can legitimately carry.
_LBRACE_SENTINEL = "\x00"
_RBRACE_SENTINEL = "\x01"

# A ``{key}`` span.  ``[^{}]*`` is what makes this safe to run over JSON: a
# brace whose span contains another brace (``{"a":{...``) cannot match, so it
# is treated as a literal rather than as the start of a placeholder.
_PLACEHOLDER_SPAN_RE = re.compile(r"\{[^{}]*\}")

# Upper bound on substitution passes.  Legitimate nesting is a short chain
# (``base_url`` -> ``port``); anything deeper than this is a cycle.
_MAX_SUBSTITUTION_PASSES = 10

# A backslash followed by trailing blanks before a newline.  Tabs are in the
# class as well as spaces: the character is invisible either way, and an
# editor that emits one (or a paste that carries one) produced a break the
# spaces-only form did not repair.
_TRAILING_BLANK_CONTINUATION_RE = re.compile(r"\\[ \t]+\n")


def sanitize_line_continuations(value: str) -> str:
    """Repair backslash line-continuations broken by trailing whitespace.

    In bash ``\\<newline>`` continues a line, but ``\\<space><newline>`` is an
    *escaped space* followed by a real newline — so the command ends there and
    every subsequent line is executed as its own command.  The failure is
    invisible in the recipe (the offending character has no glyph) and the
    error names neither sparkrun nor the recipe: the next line's first token is
    reported as a missing command (``--port: command not found``).

    Nothing legitimately ends a line with an escaped blank, so this is a repair
    rather than a guess.  Note ``\\r`` needs no handling here: YAML normalizes
    CRLF to LF inside block scalars at parse time, so a recipe edited on
    Windows has already lost its carriage returns before rendering.
    """
    return _TRAILING_BLANK_CONTINUATION_RE.sub("\\\n", value)


def uses_brace_escapes(value: str) -> bool:
    """Whether *value* is written in the doubled-brace escape convention.

    ``{{`` is the marker, and it is a reliable one: a ``{`` immediately
    followed by another ``{`` cannot occur in JSON (an object key must be a
    string), so a doubled *opening* brace is only ever a deliberate escape.

    A doubled *closing* brace carries no such signal — ``}}`` ends any nested
    JSON object (``{"a":{"b":1}}``) just as often as it escapes one literal
    brace — which is why the convention has to be read off the opening brace
    and applied to the whole string, rather than decided per ``}}``.
    """
    return "{{" in value


def mask_non_placeholder_braces(value: str, *, escapes: bool) -> str:
    """Hide every brace that is not part of a ``{key}`` placeholder.

    vpd's placeholder regex is ``\\{(.*?)\\}``, which cannot tell a placeholder
    from a brace that merely happens to sit in the text.  Any ``{`` opens a
    non-greedy match that runs to the first ``}``, so a JSON-valued flag
    swallows the placeholder nested inside it and the whole span is restored
    verbatim.  Masking first leaves only real placeholders visible to vpd.

    Scanning left to right, each position is one of:

    - ``{{`` / ``}}`` — a brace escape.  With *escapes* (recipe commands) it
      masks to a **single** sentinel, so it restores as one literal brace;
      without (hook commands), it masks to **two**, so the doubled braces
      survive untouched.  Either way the braces are invisible to vpd, so an
      escaped span can no longer eat the placeholder inside it.
    - ``{key}`` — a placeholder, passed through for vpd to resolve.
    - a lone ``{`` or ``}`` — literal, masked so it cannot open a bogus span.

    Args:
        value: Template string.
        escapes: Treat ``{{``/``}}`` as escapes that collapse to one brace.

    Returns:
        The masked string; pair with :func:`unmask_braces`.
    """
    out: list[str] = []
    i = 0
    end = len(value)
    while i < end:
        ch = value[i]
        if ch == "{":
            if value.startswith("{{", i):
                out.append(_LBRACE_SENTINEL if escapes else _LBRACE_SENTINEL * 2)
                i += 2
                continue
            match = _PLACEHOLDER_SPAN_RE.match(value, i)
            if match:
                out.append(match.group())
                i = match.end()
                continue
            out.append(_LBRACE_SENTINEL)
        elif ch == "}":
            if value.startswith("}}", i):
                out.append(_RBRACE_SENTINEL if escapes else _RBRACE_SENTINEL * 2)
                i += 2
                continue
            out.append(_RBRACE_SENTINEL)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def unmask_braces(value: str) -> str:
    """Restore :func:`mask_non_placeholder_braces` sentinels as literal braces."""
    return value.replace(_LBRACE_SENTINEL, "{").replace(_RBRACE_SENTINEL, "}")


def render_template(value: str, values: Any, *, escapes: bool = False, max_passes: int = _MAX_SUBSTITUTION_PASSES) -> str:
    """Render ``{key}`` placeholders, iterating until the text stops changing.

    Iterating is what makes nested references work — a default of
    ``http://localhost:{port}`` needs a second pass to resolve ``{port}`` once
    ``{base_url}`` has been pulled in.

    The iteration is bounded.  An unbounded fixpoint loop never terminates for
    a self-growing value (``a: "x{a}"`` renders ``x{a}`` -> ``xx{a}`` -> ...),
    turning a malformed recipe into a hang with no output.  On hitting the
    bound we log and return the last result, so the failure surfaces as a bad
    command rather than a wedged process.  A value that resolves to itself
    (``a: "{a}"``) is a fixpoint on the first pass and never reaches this.

    Args:
        value: Template string.
        values: Anything with a one-argument ``.get(key)`` — a ``dict`` or a
            SAF ``Variables`` config chain.
        escapes: Treat ``{{``/``}}`` as escapes collapsing to one brace.
        max_passes: Substitution passes before giving up.

    Returns:
        The rendered string.
    """
    rendered = mask_non_placeholder_braces(value, escapes=escapes)
    for _ in range(max_passes):
        nxt = arg_substitute(rendered, values)
        if nxt == rendered:
            return unmask_braces(rendered)
        rendered = nxt
    logger.warning(
        "Template did not stabilize after %d substitution passes — check for a placeholder whose value contains itself; using the last result",
        max_passes,
    )
    return unmask_braces(rendered)


def coerce_value(value: str):
    """Coerce a string value to int, float, or bool where possible."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_kv_output(output: str) -> dict[str, str]:
    """Parse key=value lines from script output.

    Lines starting with ``#`` are ignored. Leading/trailing whitespace
    on keys and values is stripped.

    Args:
        output: Raw stdout containing key=value lines.

    Returns:
        Dictionary of parsed key=value pairs.
    """
    result: dict[str, str] = {}
    for line in output.strip().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def parse_scoped_name(name: str) -> tuple[str | None, str]:
    """Parse ``@registry/lookup_name`` into ``(registry, lookup_name)``.

    Returns ``(None, name)`` when the input has no ``@`` prefix or
    no ``/`` separator.
    """
    if name.startswith("@") and "/" in name:
        prefix, lookup_name = name.split("/", 1)
        return prefix[1:], lookup_name  # strip leading @
    return None, name


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    Returns ``"Xs"`` for durations under 60s, ``"Xm Ys"`` for durations
    under an hour, and ``"Xh Ym Zs"`` for longer durations.
    """
    s = int(seconds)
    if s < 60:
        return "%.1fs" % seconds
    m, s = divmod(s, 60)
    if m < 60:
        return "%dm %ds" % (m, s)
    h, m = divmod(m, 60)
    return "%dh %dm %ds" % (h, m, s)
