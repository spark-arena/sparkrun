"""Tests for sparkrun.builders._ghcr — GHCR API and build-index helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from sparkrun.builders._ghcr import (
    _ghcr_anonymous_token,
    fetch_build_index,
    ghcr_list_tags,
    ghcr_get_labels,
)

_TEST_URL = "https://example.com/build-index.json"
_TEST_CACHE_NAME = "test-build-index.json"

_TOKEN_RESPONSE = {"token": "test-bearer-token"}


def _mock_resp(data: dict | list | bytes):
    """Create a mock urllib response context manager."""
    m = mock.MagicMock()
    if isinstance(data, bytes):
        m.read.return_value = data
    else:
        m.read.return_value = json.dumps(data).encode()
    m.__enter__ = mock.Mock(return_value=m)
    m.__exit__ = mock.Mock(return_value=False)
    return m


# ---------------------------------------------------------------------------
# fetch_build_index
# ---------------------------------------------------------------------------

class TestFetchBuildIndex:

    def test_returns_list_from_network(self):
        entries = [{"tag": "2025032501", "variant": "nightly", "repo_commit": "abc"}]
        body = json.dumps(entries).encode()
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            result = fetch_build_index(_TEST_URL)
        assert result == entries

    def test_returns_empty_on_network_error(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            result = fetch_build_index(_TEST_URL)
        assert result == []

    def test_returns_empty_on_invalid_json(self):
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            result = fetch_build_index(_TEST_URL)
        assert result == []

    def test_returns_empty_when_not_array(self):
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps({"not": "array"}).encode()
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            result = fetch_build_index(_TEST_URL)
        assert result == []

    def test_uses_cache(self, tmp_path):
        entries = [{"tag": "2025032501"}]
        cache_file = tmp_path / _TEST_CACHE_NAME
        cache_file.write_text(json.dumps(entries))

        with mock.patch("urllib.request.urlopen") as mock_url:
            result = fetch_build_index(_TEST_URL, cache_dir=tmp_path, cache_name=_TEST_CACHE_NAME)
        mock_url.assert_not_called()
        assert result == entries

    def test_force_refresh_bypasses_cache(self, tmp_path):
        old = [{"tag": "old"}]
        new = [{"tag": "new"}]
        cache_file = tmp_path / _TEST_CACHE_NAME
        cache_file.write_text(json.dumps(old))

        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(new).encode()
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            result = fetch_build_index(
                _TEST_URL, cache_dir=tmp_path, cache_name=_TEST_CACHE_NAME, force_refresh=True,
            )
        assert result == new

    def test_writes_cache_on_success(self, tmp_path):
        entries = [{"tag": "2025032501"}]
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = json.dumps(entries).encode()
        mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
        mock_resp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            fetch_build_index(_TEST_URL, cache_dir=tmp_path, cache_name=_TEST_CACHE_NAME)

        cached = json.loads((tmp_path / _TEST_CACHE_NAME).read_text())
        assert cached == entries


# ---------------------------------------------------------------------------
# _ghcr_anonymous_token
# ---------------------------------------------------------------------------

class TestGhcrAnonymousToken:

    def test_returns_token_on_success(self):
        with mock.patch("urllib.request.urlopen", return_value=_mock_resp(_TOKEN_RESPONSE)):
            token = _ghcr_anonymous_token("spark-arena/dgx-vllm-eugr-nightly")
        assert token == "test-bearer-token"

    def test_returns_none_on_error(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("fail")):
            assert _ghcr_anonymous_token("spark-arena/foo") is None

    def test_returns_none_when_no_token_key(self):
        with mock.patch("urllib.request.urlopen", return_value=_mock_resp({"error": "bad"})):
            assert _ghcr_anonymous_token("spark-arena/foo") is None


# ---------------------------------------------------------------------------
# ghcr_list_tags
# ---------------------------------------------------------------------------

class TestGhcrListTags:

    def test_returns_yyyymmddnn_tags(self):
        """Token is acquired first, then tags are listed with auth."""
        data = {"tags": ["latest", "2025032501", "2025032502", "v1.0", "2025032601"]}
        responses = [_mock_resp(_TOKEN_RESPONSE), _mock_resp(data)]

        with mock.patch("urllib.request.urlopen", side_effect=responses):
            tags = ghcr_list_tags("spark-arena/dgx-vllm-eugr-nightly")
        assert tags == ["2025032501", "2025032502", "2025032601"]

    def test_returns_empty_on_error(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("fail")):
            assert ghcr_list_tags("spark-arena/foo") == []

    def test_works_without_token(self):
        """If token acquisition fails, the API call proceeds without auth."""
        data = {"tags": ["2025032501"]}
        # First call (token) fails, second call (tags) succeeds
        responses = [OSError("token fail"), _mock_resp(data)]
        # _ghcr_anonymous_token catches the exception internally, so we mock it
        with mock.patch("sparkrun.builders._ghcr._ghcr_anonymous_token", return_value=None):
            with mock.patch("urllib.request.urlopen", return_value=_mock_resp(data)):
                tags = ghcr_list_tags("spark-arena/test")
        assert tags == ["2025032501"]


# ---------------------------------------------------------------------------
# ghcr_get_labels
# ---------------------------------------------------------------------------

class TestGhcrGetLabels:

    def _mock_urlopen_sequence(self, responses):
        """Create a side_effect that returns different responses per call.

        Prepends a token response automatically.
        """
        mocks = [_mock_resp(_TOKEN_RESPONSE)]
        for data in responses:
            mocks.append(_mock_resp(data))
        return mocks

    def test_returns_labels_from_config(self):
        manifest = {
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {"digest": "sha256:abc123"},
        }
        config = {
            "config": {
                "Labels": {
                    "dev.sparkrun.repo-commit": "abc123",
                    "dev.sparkrun.vllm-hash": "def456",
                }
            }
        }
        responses = self._mock_urlopen_sequence([manifest, config])
        with mock.patch("urllib.request.urlopen", side_effect=responses):
            labels = ghcr_get_labels("spark-arena/test", "2025032501")
        assert labels["dev.sparkrun.repo-commit"] == "abc123"
        assert labels["dev.sparkrun.vllm-hash"] == "def456"

    def test_returns_empty_on_error(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("fail")):
            assert ghcr_get_labels("spark-arena/test", "tag") == {}

    def test_returns_empty_when_no_config_digest(self):
        manifest = {
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {},
        }
        responses = self._mock_urlopen_sequence([manifest])
        with mock.patch("urllib.request.urlopen", side_effect=responses):
            assert ghcr_get_labels("spark-arena/test", "tag") == {}
