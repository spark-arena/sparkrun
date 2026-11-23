"""Reading literal flag values out of free-form ``command:`` templates.

A recipe may pin a serve setting inside its ``command:`` instead of declaring it
in ``defaults``, which makes the value invisible to the config chain. The
consumers that must still see it (VRAM estimation, the served name, GPU
utilization, api keys, validation) read it back through :func:`command_flag_value`
as a last resort. One parser, so they agree on what a flag occurrence is.
"""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def command_flag_value(command, *flags: str, bare: str | None = None, literal_only: bool = True) -> str | None:
    """Value of the last occurrence of any of *flags* in *command*, or ``None``.

    - Flags match as **whole tokens**, case-insensitively, in either spelling
      (``--flag value`` / ``--flag=value``), so ``--port`` never matches
      ``--port-range``. *flags* are aliases of one option, and as in argparse
      the last occurrence of any of them is the one the engine uses.
    - Line continuations (a lone ``\\`` or one glued to a value) are ignored.
    - Surrounding quotes are stripped; values are single tokens.
    - A flag followed by another flag (or nothing) is *bare*: returns *bare*.
      A negative number is a value, not a flag.
    - With *literal_only*, a value containing ``{`` or ``}`` (an unrendered
      ``{placeholder}``) yields ``None``: the config chain owns that value.

    Non-string input yields ``None`` rather than raising: callers reach this via
    ``getattr(recipe, "command", None)`` on duck-typed recipes.
    """
    if not command or not isinstance(command, str) or not flags:
        return None
    wanted = {flag.lower() for flag in flags}
    tokens = [token for token in command.split() if token != "\\"]
    found = False
    value = None
    for index, token in enumerate(tokens):
        name, sep, inline = token.partition("=")
        if name.lower() not in wanted:
            continue
        found = True
        if sep:
            value = inline
        else:
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            is_flag = following.startswith("-") and not _NUMBER_RE.match(following)
            value = "" if not following or is_flag else following
    if not found:
        return None
    value = value.rstrip("\\").strip("\"'")
    if not value:
        return bare
    if literal_only and ("{" in value or "}" in value):
        return None
    return value
