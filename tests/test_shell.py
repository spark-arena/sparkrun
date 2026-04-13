"""Unit tests for sparkrun.utils.shell module."""

from __future__ import annotations

import base64
import re

import pytest

from sparkrun.utils.shell import (
    assert_safe_path,
    b64_encode_cmd,
    b64_wrap_bash,
    quote,
    quote_dict,
    safe_remote_path,
    validate_unix_username,
)


def test_b64_encode_cmd():
    """Test that commands are correctly base64 encoded as utf-8."""
    cmd = "echo 'hello world'"
    encoded = b64_encode_cmd(cmd)

    # Verify we can decode it back to the original command
    decoded = base64.b64decode(encoded.encode("utf-8")).decode("utf-8")
    assert decoded == cmd


def test_b64_encode_cmd_with_unicode():
    """Test that commands with unicode/emojis are correctly base64 encoded."""
    cmd = 'echo "hello 🌍🚀"'
    encoded = b64_encode_cmd(cmd)

    decoded = base64.b64decode(encoded.encode("utf-8")).decode("utf-8")
    assert decoded == cmd


def test_b64_wrap_bash():
    """Test the bash wrapping pipeline."""
    cmd = 'vllm serve --hf-overrides \'{"rope": "yarn"}\''
    wrapped = b64_wrap_bash(cmd, quoted=False)

    assert wrapped.startswith("printf %s ")
    assert wrapped.endswith(" | base64 -d -- | bash --noprofile --norc")

    # Extract the encoded part and verify
    match = re.search(r"printf %s (\S+)", wrapped)
    assert match
    encoded = match.group(1)

    decoded = base64.b64decode(encoded.encode("utf-8")).decode("utf-8")
    assert decoded == cmd


def test_quote():
    """Test shell quoting."""
    assert quote("simple") == "simple"
    assert quote("has spaces") == "'has spaces'"
    assert quote("has'quotes") == "'has'\"'\"'quotes'"


def test_quote_dict():
    """Test quoting string values in a dictionary."""
    d = {
        "simple": "simple",
        "spaces": "has spaces",
        "number": 42,
        "nested": {"key": "val"},
    }
    quoted = quote_dict(d)

    assert quoted["simple"] == "simple"
    assert quoted["spaces"] == "'has spaces'"
    assert quoted["number"] == 42
    assert quoted["nested"] == {"key": "val"}


class TestAssertSafePath:
    """Tests for assert_safe_path() shell injection guard."""

    def test_normal_absolute_path(self):
        assert assert_safe_path("/home/user/.cache/sparkrun/mods/fix-something") == "/home/user/.cache/sparkrun/mods/fix-something"

    def test_tilde_path(self):
        assert assert_safe_path("~/.cache/sparkrun/mods/fix-something") == "~/.cache/sparkrun/mods/fix-something"

    def test_path_with_spaces(self):
        assert assert_safe_path("/home/user/my models/llama") == "/home/user/my models/llama"

    def test_path_with_at_sign(self):
        assert assert_safe_path("/cache/@eugr/recipes") == "/cache/@eugr/recipes"

    def test_path_with_plus_equals(self):
        assert assert_safe_path("/cache/model+extras/v1=final") == "/cache/model+extras/v1=final"

    def test_semicolon_rejected(self):
        with pytest.raises(ValueError, match="Unsafe character"):
            assert_safe_path("/tmp/foo; rm -rf /")

    def test_pipe_rejected(self):
        with pytest.raises(ValueError, match="Unsafe character"):
            assert_safe_path("/tmp/foo | cat /etc/passwd")

    def test_ampersand_rejected(self):
        with pytest.raises(ValueError, match="Unsafe character"):
            assert_safe_path("/tmp/foo & malicious")

    def test_dollar_rejected(self):
        with pytest.raises(ValueError, match="Unsafe character"):
            assert_safe_path("/tmp/$HOME/exploit")

    def test_backtick_rejected(self):
        with pytest.raises(ValueError, match="Unsafe character"):
            assert_safe_path("/tmp/`whoami`/data")

    def test_newline_rejected(self):
        with pytest.raises(ValueError, match="Unsafe character"):
            assert_safe_path("/tmp/foo\nrm -rf /")

    def test_backslash_rejected(self):
        with pytest.raises(ValueError, match="Unsafe character"):
            assert_safe_path("/tmp/foo\\bar")

    def test_single_quote_rejected(self):
        with pytest.raises(ValueError, match="Unsafe character"):
            assert_safe_path("/tmp/foo'bar")

    def test_double_quote_rejected(self):
        with pytest.raises(ValueError, match="Unsafe character"):
            assert_safe_path('/tmp/foo"bar')

    def test_parentheses_rejected(self):
        with pytest.raises(ValueError, match="Unsafe character"):
            assert_safe_path("/tmp/$(whoami)")

    def test_curly_braces_rejected(self):
        with pytest.raises(ValueError, match="Unsafe character"):
            assert_safe_path("/tmp/${HOME}")

    def test_exclamation_rejected(self):
        with pytest.raises(ValueError, match="Unsafe character"):
            assert_safe_path("/tmp/foo!bar")

    def test_angle_brackets_rejected(self):
        with pytest.raises(ValueError, match="Unsafe character"):
            assert_safe_path("/tmp/foo > /etc/passwd")


class TestSafeRemotePath:
    """Tests for safe_remote_path() — tilde-to-$HOME conversion + validation."""

    def test_tilde_prefix_converted(self):
        assert safe_remote_path("~/.cache/sparkrun/mods/fix") == "$HOME/.cache/sparkrun/mods/fix"

    def test_bare_tilde_converted(self):
        assert safe_remote_path("~") == "$HOME"

    def test_absolute_path_unchanged(self):
        assert safe_remote_path("/home/user/.cache/sparkrun") == "/home/user/.cache/sparkrun"

    def test_relative_path_unchanged(self):
        assert safe_remote_path("mods/fix-something") == "mods/fix-something"

    def test_tilde_mid_path_unchanged(self):
        """Tilde not at start is not a home-dir reference — leave it alone."""
        assert safe_remote_path("/tmp/~backup") == "/tmp/~backup"

    def test_injection_rejected(self):
        with pytest.raises(ValueError, match="Unsafe character"):
            safe_remote_path("~/; rm -rf /")

    def test_dollar_rejected(self):
        with pytest.raises(ValueError, match="Unsafe character"):
            safe_remote_path("$HOME/.cache")

    def test_path_with_spaces(self):
        assert safe_remote_path("~/.cache/my models") == "$HOME/.cache/my models"


def test_validate_unix_username_valid():
    """Test valid unix usernames."""
    valid_names = ["user", "user123", "user-name", "user_name", "user$"]
    for name in valid_names:
        assert validate_unix_username(name) == name


def test_validate_unix_username_invalid():
    """Test invalid unix usernames."""
    invalid_names = [
        "User",  # capital letter
        "1user",  # starts with number
        "-user",  # starts with dash
        "user name",  # spaces
        "user@host",  # invalid chars
        "user$$",  # multiple trailing $
    ]
    for name in invalid_names:
        with pytest.raises(ValueError, match="Invalid username"):
            validate_unix_username(name)
