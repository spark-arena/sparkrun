"""The one mapping between a backend's advertised task types and LiteLLM's ``mode``.

A diffusion backend (SGLang Diffusion) advertises what it can do on its
``/v1/models`` entry as ``task_type`` / ``supported_task_types`` — ``T2I``,
``I2I``, ``T2V``, ``I2V``.  LiteLLM's router speaks a different vocabulary:
``model_info.mode``, whose media values are ``image_generation`` and
``video_generation``.  Without that translation every media model is written
into ``model_list`` as an ordinary chat deployment, so ``/v1/models`` at the
gateway reports an image pipeline as a chat model.

This module is that translation, and it is deliberately the *only* place in
sparkrun that spells these values: the token shapes, the enum strings and the
fail-closed rules live here, and anything else that needs the same contract
goes through ``media_mode()`` rather than re-spelling strings.
``tests/test_litellm_media_mode.py`` pins the enum strings and the rules
byte-for-byte.

Three properties are load-bearing:

* **Never default to chat.** ``None`` — not ``"chat"`` — is returned when the
  tokens are unknown, absent, or contradictory.  A media model emitted without
  a mode is invisible to capability routing and reads as a chat model to every
  ``/v1/models`` consumer, which is the exact lie this tag exists to remove;
  refusing to guess keeps the entry honest instead of confidently wrong.
* **A declared primary wins.** SGLang's entry carries one ``task_type`` plus
  the full ``supported_task_types`` pool.  The primary is what the endpoint
  serves by default, so it decides; the pool only breaks a tie when there is
  no primary.
* **Mixed families are ambiguous, not image-flavoured.** An endpoint
  advertising both ``T2I`` and ``T2V`` has no single per-deployment mode, and
  picking either would mislabel half its traffic.
"""

from __future__ import annotations

#: The only media modes sparkrun emits — LiteLLM's own enum strings, not
#: invented spellings.
IMAGE_MODE = "image_generation"
VIDEO_MODE = "video_generation"

#: Stable order; consumers that list or filter media modes iterate this tuple
#: rather than re-spelling the values.
MEDIA_MODES: tuple[str, ...] = (IMAGE_MODE, VIDEO_MODE)


def _classify(token: object) -> str | None:
    """One task-type token -> media mode, or ``None`` when it says nothing.

    Matching is case-insensitive and tolerant of the enum strings themselves
    (``image_generation``) so the function round-trips its own output — the
    gateway recovers a mode from its own config and must not lose it there.
    """
    if not isinstance(token, str):
        return None
    text = token.strip().upper()
    if not text:
        return None
    if text in ("IMAGE_GENERATION", "IMAGE"):
        return IMAGE_MODE
    if text in ("VIDEO_GENERATION", "VIDEO"):
        return VIDEO_MODE
    # The shapes a diffusion backend actually advertises: anything ending 2I
    # (T2I, I2I) is image-side, anything ending 2V (T2V, I2V) is video-side.
    if text.endswith("2I"):
        return IMAGE_MODE
    if text.endswith("2V"):
        return VIDEO_MODE
    return None


def media_mode(*tokens: object, primary: object = None) -> str | None:
    """Derive LiteLLM's ``model_info.mode`` from advertised task-type tokens.

    Args:
        *tokens: The endpoint's ``supported_task_types`` pool.
        primary: The endpoint's single advertised ``task_type``.  When it maps
            to a media mode it wins outright; the pool is only consulted when
            it does not.

    Returns:
        ``"image_generation"`` / ``"video_generation"``, or ``None`` when no
        mode can be decided — nothing recognised (the ordinary chat case), or
        a pool that names both families (ambiguous for a per-deployment tag).
        Callers emit no ``mode`` key in that case rather than defaulting.
    """
    decided = _classify(primary)
    if decided is not None:
        return decided
    classified = {mode for mode in (_classify(token) for token in tokens) if mode is not None}
    if len(classified) == 1:
        return classified.pop()
    return None
