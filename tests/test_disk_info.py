"""Tests for sparkrun.orchestration.disk_info — cache status probe."""

from __future__ import annotations

from unittest.mock import patch

from sparkrun.orchestration.disk_info import (
    CacheStatus,
    _parse_probe_output,
    probe_cache_status,
)
from sparkrun.orchestration.ssh import RemoteResult


def _r(host: str, stdout: str = "", returncode: int = 0, stderr: str = "") -> RemoteResult:
    return RemoteResult(host=host, returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# _parse_probe_output
# ---------------------------------------------------------------------------


def test_parse_probe_full_output():
    """Pipe-delimited key=value blob is parsed into a CacheStatus."""
    out = "sr_exists=yes|sr_du=4G|hf_exists=yes|hf_du=120G|sr_dir=/home/u/.cache/sparkrun|hf_dir=/home/u/.cache/huggingface|free_space=812G"
    status = _parse_probe_output("h1", out)
    assert status.host == "h1"
    assert status.sparkrun_exists is True
    assert status.sparkrun_size == "4G"
    assert status.hf_exists is True
    assert status.hf_size == "120G"
    assert status.sparkrun_dir == "/home/u/.cache/sparkrun"
    assert status.hf_dir == "/home/u/.cache/huggingface"
    assert status.free_space == "812G"


def test_parse_probe_missing_dirs():
    out = "sr_exists=no|sr_du=-|hf_exists=no|hf_du=-|sr_dir=/x|hf_dir=/y|free_space=-"
    status = _parse_probe_output("h1", out)
    assert status.sparkrun_exists is False
    assert status.hf_exists is False
    assert status.free_space == "-"


def test_parse_probe_only_last_line_used():
    """Real probes may emit warnings on earlier lines; only the last is the data line."""
    out = "warning: foo\n\nsr_exists=yes|sr_du=1G|hf_exists=no|hf_du=-|sr_dir=/s|hf_dir=/h|free_space=10G"
    status = _parse_probe_output("h1", out)
    assert status.sparkrun_exists is True
    assert status.hf_exists is False


def test_parse_probe_empty():
    status = _parse_probe_output("h1", "")
    assert status.host == "h1"
    # All fields revert to dataclass defaults
    assert status.sparkrun_exists is False
    assert status.free_space == "-"


# ---------------------------------------------------------------------------
# probe_cache_status
# ---------------------------------------------------------------------------


def test_probe_cache_status_returns_per_host_results():
    results = [
        _r("h1", "sr_exists=yes|sr_du=4G|hf_exists=yes|hf_du=12G|sr_dir=/s|hf_dir=/h|free_space=812G"),
        _r("h2", "sr_exists=no|sr_du=-|hf_exists=no|hf_du=-|sr_dir=/s|hf_dir=/h|free_space=910G"),
    ]
    with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=results):
        info = probe_cache_status(["h1", "h2"], hf_cache_dir="/h")
    assert info["h1"].free_space == "812G"
    assert info["h1"].sparkrun_exists is True
    assert info["h2"].sparkrun_exists is False
    assert info["h2"].free_space == "910G"


def test_probe_cache_status_failed_host_carries_error():
    """SSH failure populates the error field with stderr."""
    results = [
        _r("h1", "sr_exists=yes|sr_du=1G|hf_exists=yes|hf_du=2G|sr_dir=/s|hf_dir=/h|free_space=100G"),
        _r("h2", "", returncode=255, stderr="ssh: connect refused"),
    ]
    with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=results):
        info = probe_cache_status(["h1", "h2"], hf_cache_dir="/h")
    assert info["h1"].error is None
    assert info["h2"].error == "ssh: connect refused"


def test_probe_cache_status_empty_hosts_returns_empty():
    with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel") as mock_run:
        info = probe_cache_status([], hf_cache_dir="/h")
    assert info == {}
    mock_run.assert_not_called()


def test_probe_cache_status_dry_run_returns_placeholders():
    with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel") as mock_run:
        info = probe_cache_status(["h1", "h2"], hf_cache_dir="/h", dry_run=True)
    assert set(info) == {"h1", "h2"}
    assert info["h1"].hf_dir == "/h"
    mock_run.assert_not_called()


def test_probe_cache_status_default_sparkrun_path():
    """Without explicit sparkrun_cache_dir the probe uses $HOME/.cache/sparkrun."""
    captured: dict = {}

    def _capture(hosts, cmd, **kw):
        captured["cmd"] = cmd
        return [_r(hosts[0], "sr_exists=yes|sr_du=1G|hf_exists=yes|hf_du=2G|sr_dir=$HOME/.cache/sparkrun|hf_dir=/h|free_space=100G")]

    with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", side_effect=_capture):
        probe_cache_status(["h1"], hf_cache_dir="/h")
    assert "$HOME/.cache/sparkrun" in captured["cmd"]


# ---------------------------------------------------------------------------
# format_cache_status_table (lives in cli_formatters)
# ---------------------------------------------------------------------------


def _status(host, **kw) -> CacheStatus:
    return CacheStatus(host=host, hf_dir=kw.pop("hf_dir", "/h"), **kw)


def test_format_table_renders_columns():
    from sparkrun.utils.cli_formatters import format_cache_status_table

    host_status = {
        "h1": _status("h1", sparkrun_exists=True, sparkrun_size="4G", hf_exists=True, hf_size="12G", free_space="812G"),
    }
    table = format_cache_status_table(host_status)
    assert "h1" in table
    assert "812G" in table
    assert "SR exists" in table
    assert "Free Space" in table
    assert "HF path" in table


def test_format_table_marks_highlight_hosts_with_arrow():
    from sparkrun.utils.cli_formatters import format_cache_status_table

    host_status = {
        "h1": _status("h1", free_space="812G"),
        "h2": _status("h2", free_space="0"),
    }
    table = format_cache_status_table(host_status, highlight_hosts=["h2"])
    h2_line = next(line for line in table.splitlines() if "h2" in line)
    h1_line = next(line for line in table.splitlines() if "h1" in line and "Free" not in line)
    assert "→" in h2_line
    assert "→" not in h1_line


def test_format_table_includes_local_row_first():
    from sparkrun.utils.cli_formatters import format_cache_status_table

    host_status = {"h1": _status("h1", free_space="500G")}
    local = _status("(local)", free_space="200G", hf_dir="/local/hf")
    table = format_cache_status_table(host_status, local_status=local)
    lines = [line for line in table.splitlines() if "(local)" in line or "h1" in line]
    assert "(local)" in lines[0]
    assert "h1" in lines[1]


def test_format_table_error_host_shows_error_row():
    from sparkrun.utils.cli_formatters import format_cache_status_table

    host_status = {"h2": _status("h2", error="ssh: connection refused")}
    table = format_cache_status_table(host_status)
    assert "h2" in table
    assert "Error: ssh: connection refused" in table


def test_format_table_empty_returns_empty_string():
    from sparkrun.utils.cli_formatters import format_cache_status_table

    assert format_cache_status_table({}) == ""
