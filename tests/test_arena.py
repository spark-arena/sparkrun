"""Tests for sparkrun arena auth and upload modules."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from sparkrun.arena.auth import (
    ExchangeResult,
    generate_challenge_id,
    get_token_path,
    save_refresh_token,
    load_refresh_token,
    clear_refresh_token,
    exchange_token,
    is_logged_in,
    _can_open_browser,
)
from sparkrun.arena.upload import (
    generate_submission_id,
    upload_file,
    upload_benchmark_results,
)
from sparkrun.cli import main


# ---------------------------------------------------------------------------
# Auth: token persistence
# ---------------------------------------------------------------------------

class TestTokenPersistence:
    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sparkrun.arena.auth.get_token_path", lambda: tmp_path / "token")
        save_refresh_token("test-refresh-token-123")

        token = load_refresh_token()
        assert token == "test-refresh-token-123"

        # Check file permissions
        path = tmp_path / "token"
        assert oct(path.stat().st_mode & 0o777) == "0o600"

    def test_load_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sparkrun.arena.auth.get_token_path", lambda: tmp_path / "nonexistent")
        assert load_refresh_token() is None

    def test_load_empty(self, tmp_path, monkeypatch):
        path = tmp_path / "token"
        path.write_text("")
        monkeypatch.setattr("sparkrun.arena.auth.get_token_path", lambda: path)
        assert load_refresh_token() is None

    def test_clear(self, tmp_path, monkeypatch):
        path = tmp_path / "token"
        path.write_text("some-token")
        monkeypatch.setattr("sparkrun.arena.auth.get_token_path", lambda: path)

        clear_refresh_token()
        assert not path.exists()

    def test_clear_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sparkrun.arena.auth.get_token_path", lambda: tmp_path / "nonexistent")
        # Should not raise
        clear_refresh_token()


# ---------------------------------------------------------------------------
# Auth: challenge ID generation
# ---------------------------------------------------------------------------

class TestChallengeID:
    def test_format(self):
        cid = generate_challenge_id()
        assert len(cid) == 11  # XXXXX-XXXXX
        assert cid[5] == "-"
        assert cid[:5].isalnum()
        assert cid[6:].isalnum()

    def test_uniqueness(self):
        ids = {generate_challenge_id() for _ in range(50)}
        # With 36^10 possibilities, collisions in 50 draws are astronomically unlikely
        assert len(ids) == 50


# ---------------------------------------------------------------------------
# Auth: token exchange
# ---------------------------------------------------------------------------

class TestExchangeToken:
    def test_success(self):
        mock_response = json.dumps({
            "id_token": "id-tok-123",
            "user_id": "uid-456",
            "bucket": "spark-arena.firebasestorage.app",
        }).encode()

        with mock.patch("sparkrun.arena.auth.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = mock_response
            mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = exchange_token("refresh-tok")
            assert result.id_token == "id-tok-123"
            assert result.user_id == "uid-456"
            assert result.bucket == "spark-arena.firebasestorage.app"
            assert result.email is None
            assert result.display_name is None
            assert result.provider is None

    def test_http_error(self):
        from urllib.error import HTTPError
        with mock.patch("sparkrun.arena.auth.urlopen") as mock_urlopen:
            error = HTTPError(
                url="https://auth.sparkrun.dev/exchange",
                code=401,
                msg="Unauthorized",
                hdrs={},
                fp=mock.MagicMock(),
            )
            error.read = mock.MagicMock(return_value=b'{"error": "invalid_token"}')
            mock_urlopen.side_effect = error

            with pytest.raises(RuntimeError, match="Token exchange failed"):
                exchange_token("bad-token")

    def test_incomplete_response(self):
        mock_response = json.dumps({"id_token": "tok"}).encode()  # missing user_id, bucket

        with mock.patch("sparkrun.arena.auth.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = mock_response
            mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            with pytest.raises(RuntimeError, match="Incomplete exchange response"):
                exchange_token("tok")


# ---------------------------------------------------------------------------
# Auth: is_logged_in
# ---------------------------------------------------------------------------

class TestIsLoggedIn:
    def test_no_token(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sparkrun.arena.auth.get_token_path", lambda: tmp_path / "nonexistent")
        assert is_logged_in() is False

    def test_valid_token(self, tmp_path, monkeypatch):
        path = tmp_path / "token"
        path.write_text("valid-token")
        monkeypatch.setattr("sparkrun.arena.auth.get_token_path", lambda: path)

        with mock.patch("sparkrun.arena.auth.exchange_token",
                        return_value=ExchangeResult(id_token="id", user_id="uid", bucket="bucket")):
            assert is_logged_in() is True

    def test_invalid_token(self, tmp_path, monkeypatch):
        path = tmp_path / "token"
        path.write_text("expired-token")
        monkeypatch.setattr("sparkrun.arena.auth.get_token_path", lambda: path)

        with mock.patch("sparkrun.arena.auth.exchange_token", side_effect=RuntimeError("expired")):
            assert is_logged_in() is False


# ---------------------------------------------------------------------------
# Auth: browser detection
# ---------------------------------------------------------------------------

class TestCanOpenBrowser:
    def test_ssh_no_display(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", mock.MagicMock(isatty=mock.MagicMock(return_value=True)))
        monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 1234 5.6.7.8 22")
        monkeypatch.delenv("DISPLAY", raising=False)
        assert _can_open_browser() is False

    def test_no_tty(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", mock.MagicMock(isatty=mock.MagicMock(return_value=False)))
        assert _can_open_browser() is False

    def test_linux_no_display(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setattr("sys.stdin", mock.MagicMock(isatty=mock.MagicMock(return_value=True)))
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        assert _can_open_browser() is False


# ---------------------------------------------------------------------------
# Upload: submission ID
# ---------------------------------------------------------------------------

class TestSubmissionId:
    def test_format(self):
        sid = generate_submission_id()
        assert sid.startswith("sub")
        assert len(sid) > 5  # sub + timestamp_ms

    def test_monotonic(self):
        import time
        s1 = generate_submission_id()
        time.sleep(0.01)
        s2 = generate_submission_id()
        # Later ID should have higher numeric value
        assert int(s2[3:]) >= int(s1[3:])


# ---------------------------------------------------------------------------
# Upload: upload_file
# ---------------------------------------------------------------------------

class TestUploadFile:
    def test_success(self, tmp_path):
        test_file = tmp_path / "test.csv"
        test_file.write_text("col1,col2\n1,2\n")

        with mock.patch("sparkrun.arena.upload.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = upload_file(
                id_token="id-tok",
                bucket="spark-arena.firebasestorage.app",
                user_id="uid-123",
                submission_id="sub1234",
                file_path=test_file,
                folder="logs",
            )
            assert result is True

            # Verify the URL was constructed correctly
            call_args = mock_urlopen.call_args
            req = call_args[0][0]
            assert "spark-arena.firebasestorage.app" in req.full_url
            assert "submissions%2Fuid-123%2Fsub1234%2Flogs%2Ftest.csv" in req.full_url
            assert req.get_header("Authorization") == "Bearer id-tok"

    def test_http_error(self, tmp_path):
        test_file = tmp_path / "test.csv"
        test_file.write_text("data")

        from urllib.error import HTTPError
        with mock.patch("sparkrun.arena.upload.urlopen") as mock_urlopen:
            error = HTTPError(
                url="https://example.com",
                code=403,
                msg="Forbidden",
                hdrs={},
                fp=mock.MagicMock(),
            )
            error.read = mock.MagicMock(return_value=b"forbidden")
            mock_urlopen.side_effect = error

            result = upload_file(
                id_token="id-tok",
                bucket="bucket",
                user_id="uid",
                submission_id="sub1",
                file_path=test_file,
                folder="logs",
            )
            assert result is False


# ---------------------------------------------------------------------------
# Upload: upload_benchmark_results
# ---------------------------------------------------------------------------

class TestUploadBenchmarkResults:
    def test_orchestration(self, tmp_path):
        csv_file = tmp_path / "benchmark.csv"
        csv_file.write_text("col1,col2\n1,2\n")
        meta_file = tmp_path / "metadata.json"
        meta_file.write_text('{"key": "value"}\n')
        recipe_file = tmp_path / "recipe.yaml"
        recipe_file.write_text("model: test\n")

        upload_files = [
            (recipe_file, "recipes"),
            (csv_file, "logs"),
            (meta_file, "metadata"),
        ]

        with mock.patch("sparkrun.arena.upload.exchange_token") as mock_exchange, \
             mock.patch("sparkrun.arena.upload.upload_file", return_value=True) as mock_upload:
            mock_exchange.return_value = ExchangeResult(id_token="id-tok", user_id="uid-123", bucket="bucket-name")

            success, sub_id = upload_benchmark_results(
                refresh_token="refresh-tok",
                upload_files=upload_files,
            )

            assert success is True
            assert sub_id.startswith("sub")
            assert mock_upload.call_count == 3

    def test_explicit_submission_id(self, tmp_path):
        csv_file = tmp_path / "benchmark.csv"
        csv_file.write_text("data")

        with mock.patch("sparkrun.arena.upload.exchange_token") as mock_exchange, \
             mock.patch("sparkrun.arena.upload.upload_file", return_value=True):
            mock_exchange.return_value = ExchangeResult(id_token="id-tok", user_id="uid-123", bucket="bucket-name")

            success, sub_id = upload_benchmark_results(
                refresh_token="tok",
                upload_files=[(csv_file, "logs")],
                submission_id="sub-custom-123",
            )

            assert success is True
            assert sub_id == "sub-custom-123"

    def test_missing_files_skipped(self, tmp_path):
        existing = tmp_path / "results.csv"
        existing.write_text("data")
        missing = tmp_path / "nonexistent.csv"

        with mock.patch("sparkrun.arena.upload.exchange_token") as mock_exchange, \
             mock.patch("sparkrun.arena.upload.upload_file", return_value=True) as mock_upload:
            mock_exchange.return_value = ExchangeResult(id_token="id-tok", user_id="uid-123", bucket="bucket-name")

            success, sub_id = upload_benchmark_results(
                refresh_token="tok",
                upload_files=[(existing, "logs"), (missing, "logs")],
            )

            assert success is True
            assert mock_upload.call_count == 1  # only the existing file


# ---------------------------------------------------------------------------
# CLI: arena commands
# ---------------------------------------------------------------------------

class TestArenaCLI:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_arena_help(self, runner):
        result = runner.invoke(main, ["arena", "--help"])
        assert result.exit_code == 0
        assert "login" in result.output
        assert "logout" in result.output
        assert "status" in result.output
        assert "benchmark" in result.output

    def test_login_help(self, runner):
        result = runner.invoke(main, ["arena", "login", "--help"])
        assert result.exit_code == 0

    def test_logout_not_logged_in(self, runner, monkeypatch):
        monkeypatch.setattr("sparkrun.arena.auth.get_token_path",
                            lambda: Path("/tmp/nonexistent_sparkrun_token"))
        result = runner.invoke(main, ["arena", "logout"])
        assert result.exit_code == 0
        assert "Not logged in" in result.output

    def test_logout_clears_token(self, runner, tmp_path, monkeypatch):
        token_path = tmp_path / "token"
        token_path.write_text("some-token")
        monkeypatch.setattr("sparkrun.arena.auth.get_token_path", lambda: token_path)

        result = runner.invoke(main, ["arena", "logout"])
        assert result.exit_code == 0
        assert "Logged out" in result.output
        assert not token_path.exists()

    def test_status_not_logged_in(self, runner, monkeypatch):
        monkeypatch.setattr("sparkrun.arena.auth.get_token_path",
                            lambda: Path("/tmp/nonexistent_sparkrun_token"))
        result = runner.invoke(main, ["arena", "status"])
        assert result.exit_code == 0
        assert "Not logged in" in result.output

    def test_status_logged_in(self, runner, tmp_path, monkeypatch):
        token_path = tmp_path / "token"
        token_path.write_text("valid-token")
        monkeypatch.setattr("sparkrun.arena.auth.get_token_path", lambda: token_path)

        with mock.patch("sparkrun.arena.auth.exchange_token",
                        return_value=ExchangeResult(
                            id_token="id-tok", user_id="user-abc", bucket="bucket",
                            email="test@example.com", provider="google.com",
                        )):
            result = runner.invoke(main, ["arena", "status"])
            assert result.exit_code == 0
            assert "test@example.com" in result.output
            assert "google.com" in result.output

    def test_status_expired_token(self, runner, tmp_path, monkeypatch):
        token_path = tmp_path / "token"
        token_path.write_text("expired-token")
        monkeypatch.setattr("sparkrun.arena.auth.get_token_path", lambda: token_path)

        with mock.patch("sparkrun.arena.auth.exchange_token",
                        side_effect=RuntimeError("token expired")):
            result = runner.invoke(main, ["arena", "status"])
            assert result.exit_code == 0
            assert "invalid or expired" in result.output.lower()


# ---------------------------------------------------------------------------
# Benchmark result dataclass
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    def test_defaults(self):
        from sparkrun.benchmarking.base import BenchmarkResult
        r = BenchmarkResult()
        assert r.success is False
        assert r.results is None
        assert r.outputs is None

    def test_populated(self):
        from sparkrun.benchmarking.base import BenchmarkResult
        r = BenchmarkResult(
            success=True,
            recipe_name="test-recipe",
        )
        assert r.success is True
        assert r.recipe_name == "test-recipe"
