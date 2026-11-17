"""Tests for surfacing the model's true context window through the proxy.

When sparkrun discovers a healthy inference endpoint it also reads the
backend's ``/v1/models`` ``max_model_len``. This must:
- be captured on the endpoint (``DiscoveredEndpoint.max_model_len``),
- be advertised in the LiteLLM config as ``model_info.max_input_tokens`` so
  LiteLLM exposes it to clients,
- **survive** a config rewrite driven by an alias change, and be part of the
  identity that decides whether a rewrite is needed at all, and
- be readable via ``proxy models --json`` (``ProxyModel.max_model_len``).

The last two are the ones that fail silently: without them the window is
either erased by the next ``proxy alias add`` or never written to an
already-running proxy in the first place.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from sparkrun.api.proxy._ops import ProxyModel
from sparkrun.proxy.discovery import DiscoveredEndpoint, _check_single_health
from sparkrun.plugins.litellm.engine import ProxyEngine, _model_keys, build_litellm_config, write_config


def _endpoint(max_model_len: int | None = None) -> DiscoveredEndpoint:
    return DiscoveredEndpoint(
        cluster_id="c",
        model="test/model",
        served_model_name=None,
        runtime="vllm",
        host="10.0.0.1",
        port=8000,
        healthy=True,
        actual_models=["test/model"],
        max_model_len=max_model_len,
    )


def _models_response(payload: dict) -> MagicMock:
    """A urlopen context manager returning *payload* with HTTP 200."""
    resp = MagicMock()
    resp.status = 200
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# Probe: reading max_model_len off the backend's /v1/models
# ---------------------------------------------------------------------------


def test_check_single_health_parses_max_model_len():
    """A backend reporting ``max_model_len`` has it captured by the probe."""
    resp = _models_response({"data": [{"id": "test/model", "max_model_len": 524288}]})

    with patch("sparkrun.proxy.discovery.urllib.request.urlopen", return_value=resp):
        healthy, models, max_model_len, task_fields = _check_single_health(_endpoint())

    assert healthy is True
    assert models == ["test/model"]
    assert max_model_len == 524288
    # A chat backend says nothing about tasks; the probe reports that faithfully.
    assert task_fields == (None, [])


def test_check_single_health_without_max_model_len():
    """A backend that omits the field stays healthy with an unknown window."""
    resp = _models_response({"data": [{"id": "test/model"}]})

    with patch("sparkrun.proxy.discovery.urllib.request.urlopen", return_value=resp):
        healthy, models, max_model_len, _ = _check_single_health(_endpoint())

    assert healthy is True
    assert models == ["test/model"]
    assert max_model_len is None


def test_check_single_health_ignores_non_positive_max_model_len():
    """Zero / non-int windows are not a context length; the first real one wins."""
    resp = _models_response(
        {
            "data": [
                {"id": "a", "max_model_len": 0},
                {"id": "b", "max_model_len": "131072"},
                {"id": "c", "max_model_len": 4096},
            ]
        }
    )

    with patch("sparkrun.proxy.discovery.urllib.request.urlopen", return_value=resp):
        _, _, max_model_len, _ = _check_single_health(_endpoint())

    assert max_model_len == 4096


def test_check_single_health_captures_task_fields():
    """A diffusion backend's ``task_type`` / ``supported_task_types`` are captured.

    The probe records the advertised strings verbatim — interpreting them is
    ``media_mode``'s job, so the probe stays a faithful reporter of whatever
    the backend said.
    """
    resp = _models_response(
        {
            "data": [
                {
                    "id": "black-forest-labs/FLUX.2-dev",
                    "max_model_len": None,
                    "task_type": "T2I",
                    "supported_task_types": ["T2I", "I2I"],
                }
            ]
        }
    )

    with patch("sparkrun.proxy.discovery.urllib.request.urlopen", return_value=resp):
        healthy, models, max_model_len, task_fields = _check_single_health(_endpoint())

    assert healthy is True
    assert models == ["black-forest-labs/FLUX.2-dev"]
    assert max_model_len is None  # a diffusion pipeline reports no context window
    assert task_fields == ("T2I", ["T2I", "I2I"])


def test_check_single_health_prefers_entry_that_advertises_tasks():
    """Task fields come from the entry that carries them, not position 0.

    ``max_model_len`` uses the same "first entry that has a real value" rule;
    a backend listing a plain entry before the media one must not read as a
    chat model.
    """
    resp = _models_response(
        {
            "data": [
                {"id": "plain"},
                {"id": "ltx-video", "task_type": "T2V"},
            ]
        }
    )

    with patch("sparkrun.proxy.discovery.urllib.request.urlopen", return_value=resp):
        _, _, _, task_fields = _check_single_health(_endpoint())

    assert task_fields == ("T2V", [])


def test_check_single_health_discards_non_string_task_values():
    """A malformed body cannot push a non-string into the config builder."""
    resp = _models_response({"data": [{"id": "m", "task_type": None, "supported_task_types": [7, "", "  ", "T2I"]}]})

    with patch("sparkrun.proxy.discovery.urllib.request.urlopen", return_value=resp):
        _, _, _, task_fields = _check_single_health(_endpoint())

    assert task_fields == (None, ["T2I"])


def test_check_single_health_unreachable_backend():
    """An unreachable backend is unhealthy with no models, window or tasks."""
    with patch("sparkrun.proxy.discovery.urllib.request.urlopen", side_effect=OSError("refused")):
        assert _check_single_health(_endpoint()) == (False, [], None, (None, []))


# ---------------------------------------------------------------------------
# Config emission
# ---------------------------------------------------------------------------


def test_build_litellm_config_includes_model_info_when_known():
    config = build_litellm_config([_endpoint(max_model_len=524288)], master_key=None)
    entry = config["model_list"][0]
    assert entry["model_name"] == "test/model"
    assert entry["model_info"]["max_input_tokens"] == 524288


def test_build_litellm_config_omits_model_info_when_unknown():
    config = build_litellm_config([_endpoint(max_model_len=None)], master_key=None)
    entry = config["model_list"][0]
    assert "model_info" not in entry


# ---------------------------------------------------------------------------
# Config lifecycle: the window must survive rewrites and force one when new
# ---------------------------------------------------------------------------


def test_model_keys_distinguishes_context_window():
    """A newly-learned window must read as a change, or no rewrite happens.

    ``apply_desired_state`` returns early when the key sets match, so if the
    window is absent from the identity an already-running proxy never gets
    ``model_info`` written at all.
    """
    without = build_litellm_config([_endpoint(max_model_len=None)], master_key=None)
    with_window = build_litellm_config([_endpoint(max_model_len=524288)], master_key=None)

    assert _model_keys(without) != _model_keys(with_window)


def test_model_keys_stable_for_identical_window():
    """A steady state still costs nothing — no spurious restart."""
    a = build_litellm_config([_endpoint(max_model_len=524288)], master_key=None)
    b = build_litellm_config([_endpoint(max_model_len=524288)], master_key=None)

    assert _model_keys(a) == _model_keys(b)


def test_endpoints_from_config_recovers_context_window(tmp_path: Path):
    """The alias rebuild path must not lose the window it reads back."""
    engine = ProxyEngine(state_dir=tmp_path)
    write_config(build_litellm_config([_endpoint(max_model_len=524288)], master_key=None), engine.config_path)

    recovered = engine._endpoints_from_config()

    assert [ep.max_model_len for ep in recovered] == [524288]


def test_alias_rebuild_preserves_model_info(tmp_path: Path):
    """``proxy alias add`` must not strip the window from every model.

    ``sync_aliases`` rebuilds the config from ``_endpoints_from_config``, so a
    field that does not round-trip is erased on the first alias change — and
    the discovery sweep will not restore it, because the model set is
    unchanged.
    """
    engine = ProxyEngine(state_dir=tmp_path)
    write_config(build_litellm_config([_endpoint(max_model_len=524288)], master_key=None), engine.config_path)

    rebuilt = build_litellm_config(engine._endpoints_from_config(), master_key=None, aliases={"fast": "test/model"})

    real = [e for e in rebuilt["model_list"] if e["model_name"] == "test/model"]
    assert real, "the real model entry disappeared from the rebuild"
    assert real[0]["model_info"]["max_input_tokens"] == 524288


# ---------------------------------------------------------------------------
# api.proxy surface
# ---------------------------------------------------------------------------


def test_proxy_model_to_dict_includes_max_model_len():
    m = ProxyModel(model_name="m", api_base="http://x/v1", max_model_len=524288)
    d = json.loads(json.dumps(m.to_dict()))
    assert d["model_name"] == "m"
    assert d["max_model_len"] == 524288


def test_proxy_model_to_dict_omits_max_model_len_when_none():
    m = ProxyModel(model_name="m", api_base="http://x/v1", max_model_len=None)
    d = m.to_dict()
    assert "max_model_len" not in d


def test_models_via_api_reads_model_info():
    """``proxy models --json`` surfaces the window LiteLLM reports back."""
    from sparkrun.plugins.litellm.engine import ProxyEngine

    engine = ProxyEngine()
    engine._api_request = MagicMock(
        return_value={
            "data": [
                {
                    "model_name": "m",
                    "litellm_params": {"api_base": "http://10.0.0.1:8000/v1"},
                    "model_info": {"max_input_tokens": 524288},
                },
                {
                    "model_name": "n",
                    "litellm_params": {"api_base": "http://10.0.0.2:8000/v1"},
                    "model_info": {"max_input_tokens": None},
                },
            ]
        }
    )

    models = engine.query_models()

    assert [(m.model_name, m.max_model_len) for m in models] == [("m", 524288), ("n", None)]
