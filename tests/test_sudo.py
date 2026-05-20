"""Tests for sudo_user validation in run_indirect_sudo_script."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from sparkrun.orchestration.sudo import run_indirect_sudo_script
from sparkrun.orchestration.ssh import RemoteResult


@patch("sparkrun.orchestration.ssh._run_subprocess")
def test_valid_username_passes(mock_run):
    """A well-formed Unix username proceeds to SSH execution."""
    mock_run.return_value = RemoteResult(host="1.2.3.4", returncode=0, stdout="ok", stderr="")
    res = run_indirect_sudo_script(
        host="1.2.3.4",
        script="echo hello",
        sudo_user="drew",
        sudo_password="secret",
        ssh_kwargs={"ssh_user": "bob"},
    )
    assert res.success
    mock_run.assert_called_once()


def test_malicious_username_raises():
    """A username with injection payload is rejected before any SSH call."""
    with pytest.raises(ValueError, match="Invalid username"):
        run_indirect_sudo_script(
            host="1.2.3.4",
            script="echo hello",
            sudo_user="evil';os.system('rm -rf ~');('",
            sudo_password="secret",
        )


def test_username_with_semicolon_raises():
    """A username containing a semicolon is rejected."""
    with pytest.raises(ValueError, match="Invalid username"):
        run_indirect_sudo_script(
            host="1.2.3.4",
            script="true",
            sudo_user="user;bad",
            sudo_password="pw",
        )


def test_empty_username_raises():
    """An empty username is rejected."""
    with pytest.raises(ValueError, match="Invalid username"):
        run_indirect_sudo_script(
            host="1.2.3.4",
            script="true",
            sudo_user="",
            sudo_password="pw",
        )


@patch("sparkrun.orchestration.ssh._run_subprocess")
def test_dry_run_skips_ssh_but_still_validates(mock_run):
    """dry_run=True skips SSH execution but validation still runs first."""
    # Valid username: no SSH, returns dry-run result
    res = run_indirect_sudo_script(
        host="1.2.3.4",
        script="echo hello",
        sudo_user="admin",
        sudo_password="pw",
        dry_run=True,
    )
    assert res.returncode == 0
    assert "[dry-run]" in res.stdout
    mock_run.assert_not_called()

    # Invalid username: should still raise even in dry_run mode
    with pytest.raises(ValueError, match="Invalid username"):
        run_indirect_sudo_script(
            host="1.2.3.4",
            script="true",
            sudo_user="evil';bad",
            sudo_password="pw",
            dry_run=True,
        )
