"""Unit tests for sparkrun.utils.shell module."""

from __future__ import annotations


import base64
import re

import pytest

from sparkrun.utils.shell import (
    b64_encode_cmd,
    b64_wrap_bash,
    quote,
    quote_dict,
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
    wrapped = b64_wrap_bash(cmd)

    assert wrapped.startswith("printf '%s' '")
    assert wrapped.endswith("' | base64 -d -- | bash --noprofile --norc")

    # Extract the encoded part and verify
    match = re.search(r"printf '%s' '([^']+)'", wrapped)
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
