"""Media models must reach the proxy labelled as media, chat models untouched.

A diffusion backend (SGLang Diffusion) advertises ``task_type`` /
``supported_task_types`` (T2I/I2I/T2V/I2V) on its ``/v1/models`` entry.
sparkrun's health probe now carries those onto ``DiscoveredEndpoint``, and
``build_litellm_config`` translates them into LiteLLM's ``model_info.mode``
(``image_generation`` / ``video_generation``) — without it every image/video
model is emitted as an ordinary chat deployment and the gateway's own
/v1/models misreports what it fronts.

The four properties pinned here, in order of how badly each fails silently:

* **Chat is byte-identical.** An endpoint with no task vocabulary must emit
  exactly the entry it emitted before this feature existed — same keys, same
  order, same types.  A regression here rewrites (and restarts) every running
  gateway's config for no reason.
* **Media carries a mode.** T2I/I2I -> image_generation, T2V/I2V ->
  video_generation, via ``task_type`` or via the ``supported_task_types``
  fallback.  The enum strings are pinned verbatim: they are LiteLLM's, and a
  second spelling would be invisible to capability routing.
* **Unknown task vocabulary is not media.** Unrecognised tokens emit no
  ``mode`` key at all — the entry falls back to the pre-feature shape rather
  than a guessed mode.  Never default a media-looking model to ``chat``, and
  never crash on a schema this repo has not seen.
* **The tag survives every rebuild path.** ``sync_aliases`` reconstructs the
  endpoints from the emitted config, and ``_model_keys`` decides whether a
  rewrite happens at all; drop the tag in either place and it is erased by
  the next ``proxy alias add`` or never written to a running proxy — the
  exact ``max_input_tokens`` incident the module docstring documents.
"""

from __future__ import annotations

import json
from pathlib import Path

from sparkrun.plugins.litellm.engine import ProxyEngine, _model_keys, build_litellm_config, write_config
from sparkrun.plugins.litellm.media_mode import MEDIA_MODES, media_mode
from sparkrun.proxy.discovery import DiscoveredEndpoint

CHAT_MODEL = "Qwen/Qwen3-1.7B"
IMAGE_MODEL = "black-forest-labs/FLUX.2-dev"
VIDEO_MODEL = "Lightricks/LTX-Video"


def _endpoint(
    model: str = CHAT_MODEL,
    *,
    task_type: str | None = None,
    task_types: list[str] | None = None,
    max_model_len: int | None = None,
) -> DiscoveredEndpoint:
    return DiscoveredEndpoint(
        cluster_id="c",
        model=model,
        served_model_name=None,
        runtime="vllm",
        host="10.0.0.1",
        port=8000,
        healthy=True,
        actual_models=[model],
        max_model_len=max_model_len,
        task_type=task_type,
        task_types=task_types or [],
    )


# ---------------------------------------------------------------------------
# 1. Chat models are untouched
# ---------------------------------------------------------------------------


def test_chat_endpoint_entry_is_unchanged():
    """A model with no task vocabulary emits the exact entry it emitted before.

    This is the regression guard for the byte-identity requirement: the whole
    chat fleet's config must not shift when media support is added.
    """
    config = build_litellm_config([_endpoint()], master_key=None)

    assert config["model_list"] == [
        {
            "model_name": CHAT_MODEL,
            "litellm_params": {
                "model": "openai/%s" % CHAT_MODEL,
                "api_base": "http://10.0.0.1:8000/v1",
                "api_key": "not-needed",
            },
        }
    ]


def test_chat_endpoint_with_window_is_unchanged():
    """The pre-existing ``max_input_tokens`` emission is untouched too."""
    config = build_litellm_config([_endpoint(max_model_len=40960)], master_key=None)

    assert config["model_list"] == [
        {
            "model_name": CHAT_MODEL,
            "litellm_params": {
                "model": "openai/%s" % CHAT_MODEL,
                "api_base": "http://10.0.0.1:8000/v1",
                "api_key": "not-needed",
            },
            "model_info": {"max_input_tokens": 40960},
        }
    ]


def test_chat_model_identity_is_semantically_unchanged():
    """A chat model's restart-identity still turns on name/base/window only.

    ``_model_keys`` gained a fourth element for ``mode``; for chat it is
    ``None``, so no chat config can start reading as changed and trigger a
    gratuitous gateway restart.
    """
    keys = _model_keys(build_litellm_config([_endpoint(max_model_len=40960)], master_key=None))

    assert keys == {(CHAT_MODEL, "http://10.0.0.1:8000/v1", 40960, None)}


# ---------------------------------------------------------------------------
# 2. Media models carry a mode
# ---------------------------------------------------------------------------


def test_task_type_t2i_maps_to_image_generation():
    config = build_litellm_config([_endpoint(IMAGE_MODEL, task_type="T2I")], master_key=None)

    assert config["model_list"][0]["model_info"]["mode"] == "image_generation"


def test_task_type_t2v_maps_to_video_generation():
    config = build_litellm_config([_endpoint(VIDEO_MODEL, task_type="T2V")], master_key=None)

    assert config["model_list"][0]["model_info"]["mode"] == "video_generation"


def test_supported_task_types_used_when_task_type_absent():
    """``supported_task_types`` alone is enough to decide the mode."""
    image = build_litellm_config([_endpoint(IMAGE_MODEL, task_types=["T2I", "I2I"])], master_key=None)
    video = build_litellm_config([_endpoint(VIDEO_MODEL, task_types=["T2V", "I2V"])], master_key=None)

    assert image["model_list"][0]["model_info"]["mode"] == "image_generation"
    assert video["model_list"][0]["model_info"]["mode"] == "video_generation"


def test_primary_task_type_wins_over_pool():
    """The endpoint's advertised default decides; the pool only breaks ties."""
    config = build_litellm_config([_endpoint(IMAGE_MODEL, task_type="T2I", task_types=["T2V"])], master_key=None)

    assert config["model_list"][0]["model_info"]["mode"] == "image_generation"


def test_mode_coexists_with_context_window():
    """``mode`` is added to an existing ``model_info``, not written over it."""
    config = build_litellm_config([_endpoint(IMAGE_MODEL, task_type="T2I", max_model_len=4096)], master_key=None)

    assert config["model_list"][0]["model_info"] == {
        "max_input_tokens": 4096,
        "mode": "image_generation",
    }


def test_every_served_name_of_a_media_endpoint_is_tagged():
    """``--served-model-name "concrete alias"`` -> both names carry the mode."""
    ep = _endpoint(IMAGE_MODEL, task_type="T2I")
    ep.actual_models = [IMAGE_MODEL, "flux"]

    config = build_litellm_config([ep], master_key=None)

    assert [e["model_info"]["mode"] for e in config["model_list"]] == [
        "image_generation",
        "image_generation",
    ]


def test_mode_values_are_litellms_enum_strings():
    """The pair is pinned verbatim — a third or re-spelled value is a bug."""
    assert MEDIA_MODES == ("image_generation", "video_generation")
    assert media_mode("T2I") == "image_generation"
    assert media_mode("T2V") == "video_generation"


# ---------------------------------------------------------------------------
# 3. Unknown / ambiguous vocabulary never crashes and never guesses
# ---------------------------------------------------------------------------


def test_unknown_task_type_emits_no_mode_key():
    """Unrecognised enums fall back to the pre-feature shape — no crash, no mode.

    Documented behavior: an endpoint advertising something this release does
    not know (a newer backend, a custom recipe) is emitted exactly like a chat
    model.  Guessing a mode would be a confident lie; guessing ``chat`` for a
    model that advertises *some* task vocabulary is the lie the tag exists to
    remove, so the only honest output is no mode key at all.
    """
    config = build_litellm_config([_endpoint("some/model", task_type="AUDIO_TO_TEXT")], master_key=None)

    assert "mode" not in config["model_list"][0].get("model_info", {})
    assert config["model_list"] == [
        {
            "model_name": "some/model",
            "litellm_params": {
                "model": "openai/some/model",
                "api_base": "http://10.0.0.1:8000/v1",
                "api_key": "not-needed",
            },
        }
    ]


def test_mixed_image_and_video_pool_emits_no_mode():
    """An endpoint advertising both families has no per-deployment mode."""
    config = build_litellm_config([_endpoint("omni/model", task_types=["T2I", "T2V"])], master_key=None)

    assert "model_info" not in config["model_list"][0]


def test_non_string_task_values_do_not_crash():
    """Robust against a malformed /v1/models body, not just a hostile one.

    The probe filters non-strings on the way in, but ``DiscoveredEndpoint`` is
    a plain dataclass and ``build_litellm_config`` is also called with
    endpoints recovered from a hand-edited config, so the classifier has to
    tolerate junk rather than assume the probe produced it.
    """
    ep = DiscoveredEndpoint(
        cluster_id="c",
        model="weird/model",
        served_model_name=None,
        runtime="vllm",
        host="10.0.0.1",
        port=8000,
        healthy=True,
        actual_models=["weird/model"],
        task_type=7,  # type: ignore[arg-type]
        task_types=[None, "T2I"],  # type: ignore[list-item]
    )

    config = build_litellm_config([ep], master_key=None)

    assert config["model_list"][0]["model_info"]["mode"] == "image_generation"


def test_alternative_task_shapes_do_not_crash():
    """A task_type dict or a task_types given as a str yields no mode, not a traceback."""
    for task_type, task_types in (
        ({"nested": 1}, []),
        (None, "T2I"),
        (None, {}),
        (None, [1, 2, 3]),
        (None, ["", "   "]),
    ):
        ep = DiscoveredEndpoint(
            cluster_id="c",
            model="weird/model",
            served_model_name=None,
            runtime="vllm",
            host="10.0.0.1",
            port=8000,
            healthy=True,
            actual_models=["weird/model"],
            task_type=task_type,  # type: ignore[arg-type]
            task_types=task_types,  # type: ignore[arg-type]
        )
        config = build_litellm_config([ep], master_key=None)
        assert "mode" not in config["model_list"][0].get("model_info", {}), (task_type, task_types)


def test_media_mode_returns_none_undecorated():
    """The contract function's fail-closed return is None, never 'chat'."""
    assert media_mode() is None
    assert media_mode("NOPE") is None
    assert media_mode("T2I", "T2V") is None
    assert media_mode(primary="T2I") == "image_generation"


# ---------------------------------------------------------------------------
# 4. The tag survives every rebuild path
# ---------------------------------------------------------------------------


def test_model_keys_distinguishes_the_mode():
    """A newly-learned mode must read as a change, or no rewrite happens.

    This is the failure mode the context-window work already documented:
    ``apply_desired_state`` returns early when the key sets match, so a proxy
    whose config predates the tag would keep serving it untagged forever.
    """
    untagged = build_litellm_config([_endpoint(IMAGE_MODEL)], master_key=None)
    tagged = build_litellm_config([_endpoint(IMAGE_MODEL, task_type="T2I")], master_key=None)

    assert _model_keys(untagged) != _model_keys(tagged)


def test_model_keys_stable_for_identical_mode():
    """Steady state still costs nothing — no restart storm."""
    a = build_litellm_config([_endpoint(IMAGE_MODEL, task_type="T2I")], master_key=None)
    b = build_litellm_config([_endpoint(IMAGE_MODEL, task_type="T2I")], master_key=None)

    assert _model_keys(a) == _model_keys(b)


def test_endpoints_from_config_recovers_the_mode(tmp_path: Path):
    """Config recovery re-reads the mode as task vocabulary that remaps to it."""
    engine = ProxyEngine(state_dir=tmp_path)
    write_config(
        build_litellm_config([_endpoint(IMAGE_MODEL, task_type="T2I")], master_key=None),
        engine.config_path,
    )

    recovered = engine._endpoints_from_config()

    assert [ep.task_type for ep in recovered] == ["image_generation"]
    # The recovered endpoint must rebuild to the same emission, not merely
    # carry a field with a plausible name.
    rebuilt = build_litellm_config(recovered, master_key=None)
    assert rebuilt["model_list"][0]["model_info"]["mode"] == "image_generation"


def test_alias_rebuild_preserves_the_mode(tmp_path: Path):
    """``proxy alias add`` must not strip the tag from every media model.

    ``sync_aliases`` feeds ``_endpoints_from_config()`` straight back into
    ``build_litellm_config``, so a field that does not round-trip is erased
    on the first alias change — and the discovery sweep will not restore it,
    because the model set is unchanged.
    """
    engine = ProxyEngine(state_dir=tmp_path)
    write_config(
        build_litellm_config([_endpoint(IMAGE_MODEL, task_type="T2I")], master_key=None),
        engine.config_path,
    )

    rebuilt = build_litellm_config(engine._endpoints_from_config(), master_key=None, aliases={"flux-quick": IMAGE_MODEL})

    real = [e for e in rebuilt["model_list"] if e["model_name"] == IMAGE_MODEL]
    assert real, "the real model entry disappeared from the rebuild"
    assert real[0]["model_info"]["mode"] == "image_generation"
    # The alias entry keeps pointing at the concrete model, as before.
    alias = [e for e in rebuilt["model_list"] if e["model_name"] == "flux-quick"]
    assert alias[0]["litellm_params"]["model"] == "openai/%s" % IMAGE_MODEL


def test_seed_round_trip_carries_task_fields():
    """``proxy start --auto-discover`` seeds survive the asdict round-trip.

    ``api/proxy/_ops.py`` serialises initial endpoints with ``asdict(ep)`` and
    ``discovery_service`` rebuilds them with ``DiscoveredEndpoint(**seed)``, so
    a field the dataclass carries travels for free — but only as long as it
    *is* a dataclass field.  A mode that lived outside the dataclass would be
    silently absent from every seeded media endpoint, and seeding is the only
    thing that makes a media model routable in the first sweep.
    """
    from dataclasses import asdict

    media = _endpoint(IMAGE_MODEL, task_type="T2I", task_types=["T2I", "I2I"])
    seed = {k: v for k, v in asdict(media).items() if k != "api_key"}

    rebuilt = DiscoveredEndpoint(**seed, api_key="sk-upstream")

    assert (rebuilt.task_type, rebuilt.task_types) == ("T2I", ["T2I", "I2I"])
    entry = build_litellm_config([rebuilt], master_key=None)["model_list"][0]
    assert entry["model_info"]["mode"] == "image_generation"


def test_config_round_trip_is_stable_for_media(tmp_path: Path):
    """build -> write -> recover -> build changes nothing, including chat."""
    engine = ProxyEngine(state_dir=tmp_path)
    endpoints = [
        _endpoint(CHAT_MODEL, max_model_len=40960),
        _endpoint(IMAGE_MODEL, task_type="T2I"),
        _endpoint(VIDEO_MODEL, task_type="T2V"),
    ]
    write_config(build_litellm_config(endpoints, master_key=None), engine.config_path)

    once = build_litellm_config(engine._endpoints_from_config(), master_key=None)
    write_config(once, engine.config_path)
    twice = build_litellm_config(engine._endpoints_from_config(), master_key=None)

    assert once == twice
    modes = {e["model_name"]: e.get("model_info", {}).get("mode") for e in once["model_list"]}
    assert modes == {CHAT_MODEL: None, IMAGE_MODEL: "image_generation", VIDEO_MODEL: "video_generation"}


def test_emitted_media_entry_shape():
    """The media entry is the chat entry plus exactly one ``mode`` key.

    The tag is purely additive: every key a chat deployment already emitted is
    present and unchanged, and ``mode`` is the only thing a media model adds.
    """
    config = build_litellm_config([_endpoint(IMAGE_MODEL, task_type="T2I")], master_key=None)

    assert config["model_list"][0] == {
        "model_name": IMAGE_MODEL,
        "litellm_params": {
            "model": "openai/%s" % IMAGE_MODEL,
            "api_base": "http://10.0.0.1:8000/v1",
            "api_key": "not-needed",
        },
        "model_info": {"mode": "image_generation"},
    }
    # and it is YAML-safe
    assert json.loads(json.dumps(config["model_list"][0]))["model_info"]["mode"] == "image_generation"
