"""Tests for sparkrun.utils.text brace masking and template rendering.

These lock down the scan rules shared by recipe command rendering and
lifecycle-hook rendering: which braces are placeholders, which are literal,
and how ``{{``/``}}`` behave in each mode.
"""

from __future__ import annotations

import logging

import pytest

from sparkrun.utils.text import (
    mask_non_placeholder_braces,
    render_template,
    sanitize_line_continuations,
    unmask_braces,
    uses_brace_escapes,
)


class TestUsesBraceEscapes:
    """The convention is read off ``{{``, which plain JSON cannot produce."""

    @pytest.mark.parametrize(
        "text",
        [
            '{{"a": 1}}',
            '{{"method":"mtp","n":{n}}}',
            "{{{k}",
        ],
    )
    def test_doubled_open_is_the_marker(self, text):
        assert uses_brace_escapes(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "no braces here",
            "{key}",
            '{"a": 1}',
            # The whole point: a trailing '}}' is ordinary nested JSON, never
            # on its own evidence of escaping.
            '{"a":{"b":2}}',
            '{"a":{"b":{"c":3}}}',
            "awk '{print $1}'",
        ],
    )
    def test_doubled_close_alone_is_not(self, text):
        assert uses_brace_escapes(text) is False


class TestMaskRoundTrip:
    """mask -> unmask must be lossless apart from the intended escape collapse."""

    @pytest.mark.parametrize(
        "text",
        [
            "no braces here",
            "{key}",
            '{"a": 1}',
            '{"a":{"b":2}}',
            "awk '{print $1}'",
            "${HF_HOME}",
            "{}",
        ],
    )
    def test_plain_mode_is_lossless(self, text):
        """Without escapes every brace comes back exactly as it went in."""
        assert unmask_braces(mask_non_placeholder_braces(text, escapes=False)) == text

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("{{a}}", "{a}"),
            ('{{"a": 1}}', '{"a": 1}'),
            ('{{"a": {{"b": 1}}}}', '{"a": {"b": 1}}'),
            ("{{{k}", "{{k}"),
            ("{k}}}", "{k}}"),
        ],
    )
    def test_escape_mode_collapses_doubled_braces(self, text, expected):
        """With escapes a ``{{``/``}}`` pair restores as one literal brace."""
        assert unmask_braces(mask_non_placeholder_braces(text, escapes=True)) == expected

    def test_placeholders_survive_masking_verbatim(self):
        """A ``{key}`` span is left untouched so vpd can still see it."""
        masked = mask_non_placeholder_braces('{{"n":{tokens}}}', escapes=True)

        assert "{tokens}" in masked
        assert "{{" not in masked

    def test_json_braces_are_hidden_from_substitution(self):
        """Braces that merely surround JSON must not look like placeholders."""
        masked = mask_non_placeholder_braces('{"a":{"b":{n}}}', escapes=False)

        assert masked.count("{") == 1  # only the real placeholder
        assert "{n}" in masked


class TestRenderTemplate:
    """The mask -> substitute -> unmask pipeline."""

    def test_standalone_placeholders(self):
        assert render_template("--host {host} --port {port}", {"host": "0.0.0.0", "port": 8000}) == "--host 0.0.0.0 --port 8000"

    def test_unknown_key_left_verbatim(self):
        """An unresolved placeholder is restored as-is (vpd behavior preserved)."""
        assert render_template("--x {nope}", {"port": 8000}) == "--x {nope}"

    def test_falsy_values_are_substituted(self):
        assert render_template("{a}|{b}|{c}", {"a": 0, "b": False, "c": ""}) == "0|False|"

    def test_placeholder_inside_bare_json(self):
        """A placeholder nested in single-brace JSON resolves.

        The unescaped shape: no ``{{``, just JSON with a placeholder inside it.
        vpd alone matches from the JSON's opening brace through the
        placeholder's closing brace and restores the span verbatim.
        """
        assert render_template('{"method":"mtp","n":{n}}', {"n": 2}) == '{"method":"mtp","n":2}'

    def test_placeholder_inside_nested_bare_json(self):
        assert render_template('{"a":{"b":{n}}}', {"n": 1}) == '{"a":{"b":1}}'

    def test_placeholder_inside_escaped_json(self):
        """The escaped shape: escapes collapse, inner placeholder resolves."""
        rendered = render_template('{{"method":"mtp","n":{n}}}', {"n": 1}, escapes=True)

        assert rendered == '{"method":"mtp","n":1}'

    def test_doubled_braces_preserved_without_escapes(self):
        """Without escape mode ``{{x}}`` keeps both braces and is not substituted."""
        assert render_template("{{keep}}", {"keep": "X"}) == "{{keep}}"

    def test_json_without_placeholders_untouched(self):
        text = "--config '{\"canvas_length\": 256}'"

        assert render_template(text, {"port": 8000}) == text

    def test_awk_program_untouched(self):
        text = "docker ps | awk '{print $1}'"

        assert render_template(text, {"port": 8000}) == text

    def test_shell_expansion_without_matching_key(self):
        assert render_template("echo ${HF_HOME}", {"port": 8000}) == "echo ${HF_HOME}"

    def test_nested_reference_resolves(self):
        assert render_template("{base_url}", {"base_url": "http://h:{port}", "port": 8000}) == "http://h:8000"

    def test_deep_chain_resolves(self):
        assert render_template("{a}", {"a": "{b}", "b": "{c}", "c": "end"}) == "end"

    def test_self_resolving_value_is_a_fixpoint(self):
        """``a: "{a}"`` stabilizes on the first pass — not a cycle."""
        assert render_template("{a}", {"a": "{a}"}) == "{a}"

    def test_self_growing_value_terminates(self, caplog):
        """``a: "x{a}"`` grows every pass; the loop must stop and say so.

        Without the bound this never reaches a fixpoint and the render hangs
        while the string grows without limit.
        """
        with caplog.at_level(logging.WARNING, logger="sparkrun.utils.text"):
            rendered = render_template("{a}", {"a": "x{a}"})

        assert rendered.endswith("{a}")
        assert "did not stabilize" in caplog.text

    def test_mutual_recursion_terminates(self, caplog):
        with caplog.at_level(logging.WARNING, logger="sparkrun.utils.text"):
            render_template("{a}", {"a": "1{b}", "b": "2{a}"})

        assert "did not stabilize" in caplog.text

    def test_max_passes_is_configurable(self):
        assert render_template("{a}", {"a": "x{a}"}, max_passes=3) == "xxx{a}"

    def test_no_warning_for_ordinary_templates(self, caplog):
        with caplog.at_level(logging.WARNING, logger="sparkrun.utils.text"):
            render_template("--port {port}", {"port": 8000})

        assert caplog.text == ""


# ---------------------------------------------------------------------------
# sanitize_line_continuations
# ---------------------------------------------------------------------------


class TestSanitizeLineContinuations:
    """Backslash continuations broken by an invisible trailing blank.

    The failure these repair is silent at authoring time and unattributable at
    runtime: bash ends the command at the escaped blank and reports the *next*
    line's first token as a missing command.
    """

    @pytest.mark.parametrize(
        "blanks",
        [" ", "   ", "\t", "\t\t", " \t", "\t ", " \t "],
        ids=["space", "spaces", "tab", "tabs", "space-tab", "tab-space", "mixed"],
    )
    def test_repairs_every_blank_run(self, blanks: str):
        """Tabs repair exactly as spaces do — both are invisible in an editor."""
        assert sanitize_line_continuations(f"a \\{blanks}\nb") == "a \\\nb"

    def test_clean_continuation_untouched(self):
        assert sanitize_line_continuations("a \\\nb") == "a \\\nb"

    def test_preserves_escaped_blank_mid_line(self):
        """``hello\\ world`` is an escaped space, not a broken continuation."""
        assert sanitize_line_continuations("echo hello\\ world") == "echo hello\\ world"

    def test_preserves_escaped_blank_before_more_text(self):
        """Only a blank run *terminated by a newline* is a broken continuation."""
        assert sanitize_line_continuations("echo a\\\tb\tc") == "echo a\\\tb\tc"

    def test_repairs_every_occurrence(self):
        assert sanitize_line_continuations("a \\ \nb \\\t\nc") == "a \\\nb \\\nc"

    def test_no_continuations_is_identity(self):
        assert sanitize_line_continuations("plain command --flag 1") == "plain command --flag 1"

    def test_repaired_command_survives_bash(self):
        """The repair is verified against the shell, not just the string.

        Asserting on the rendered text alone would pass for a regex that
        produced something bash still splits.
        """
        import subprocess

        broken = "echo ONE \\\t\n  TWO \\ \n  THREE"
        proc = subprocess.run(
            ["bash", "-c", sanitize_line_continuations(broken)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.split() == ["ONE", "TWO", "THREE"]
        assert "command not found" not in proc.stderr

    def test_unrepaired_command_breaks_in_bash(self):
        """The control: without the repair, bash really does break this way."""
        import subprocess

        proc = subprocess.run(
            ["bash", "-c", "echo ONE \\\t\n  --port 8000"],
            capture_output=True,
            text=True,
        )
        assert "command not found" in proc.stderr
