"""Tests for the content signature that makes cross-driver image caching work.

Background
----------
Image cache detection compared Docker image IDs, falling back to RepoDigest
overlap.  Each signal covers one gap, but not both at once:

* ``.Id`` is recomputed by the local storage driver, so ``overlay2`` and
  the containerd ``overlayfs`` snapshotter disagree on it for identical
  content (issue #152).
* RepoDigests are registry metadata, so ``docker save | docker load``
  strips them entirely.

A fleet that pulls on the control host and ``save | load`` fans out to a
worker on a *different* driver hits both gaps simultaneously: different
IDs, no digests anywhere on the worker.  The worker was therefore judged
stale on every launch and re-received the whole image forever.

``content_signature`` closes that: it digests the image configuration plus
the ordered layer chain, both read verbatim from the image manifest and
stored identically by every driver.

These tests use a stubbed ``docker`` so the legacy-daemon fallback paths
(which cannot be reached against a modern daemon) are exercised too.
"""

from __future__ import annotations

import json
import subprocess
from unittest import mock

import pytest

from sparkrun.containers.registry import (
    IMAGE_INSPECT_FORMAT,
    LEGACY_IMAGE_INSPECT_FORMAT,
    ImageIdentity,
    content_signature,
    get_image_identity,
    parse_image_identity,
)


def _fake_run(script):
    """Build a ``subprocess.run`` stub from a {format_string: (rc, stdout)} map.

    Keyed on the ``--format`` argument so a test can state exactly how the
    daemon answers each template, including "this template errors".
    """

    def run(cmd, *args, **kwargs):
        fmt = cmd[cmd.index("--format") + 1] if "--format" in cmd else ""
        rc, out = script[fmt]
        return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr="")

    return run


class TestContentSignature:
    def test_signature_stable_for_same_content(self):
        sig_a = content_signature("sha256:L1 sha256:L2", '{"Env":["A=1"]}')
        sig_b = content_signature("sha256:L1 sha256:L2", '{"Env":["A=1"]}')
        assert sig_a == sig_b
        assert sig_a is not None and sig_a.startswith("sha256:")

    def test_config_key_order_does_not_change_signature(self):
        """JSON key order is normalised, so driver/Go differences cannot split it."""
        one = json.dumps({"Env": ["A=1"], "WorkingDir": "/w", "Labels": {"a": "1"}})
        two = json.dumps({"WorkingDir": "/w", "Labels": {"a": "1"}, "Env": ["A=1"]})
        assert content_signature("sha256:L1", one) == content_signature("sha256:L1", two)

    def test_different_layers_change_signature(self):
        cfg = '{"Env":["A=1"]}'
        assert content_signature("sha256:L1 sha256:L2", cfg) != content_signature(
            "sha256:L1 sha256:L3", cfg
        )

    def test_layer_order_change_is_detected(self):
        """Reordered layers are different content, not the same image."""
        cfg = '{"Env":["A=1"]}'
        assert content_signature("sha256:L1 sha256:L2", cfg) != content_signature(
            "sha256:L2 sha256:L1", cfg
        )

    def test_different_config_changes_signature(self):
        layers = "sha256:L1"
        assert content_signature(layers, '{"Env":["A=1"]}') != content_signature(
            layers, '{"Env":["A=2"]}'
        )

    def test_field_boundary_is_not_ambiguous(self):
        """Layers/config must not hash identically when text crosses the boundary.

        Without a real separator, ``("a", "b")`` and ``("ab", "")`` would
        collide -- a silent false match between different images.
        """
        assert content_signature("aaa", "bbb") != content_signature("aaabbb", "")
        assert content_signature("aa", "bbb") != content_signature("a", "abbb")

    def test_both_empty_means_no_signature(self):
        assert content_signature("", "") is None
        assert content_signature("   ", "   ") is None

    def test_config_only_still_signatures(self):
        assert content_signature("", '{"Env":["A=1"]}') is not None

    def test_layers_only_still_signatures(self):
        assert content_signature("sha256:L1", "") is not None

    def test_unparsable_config_is_hashed_verbatim_not_dropped(self):
        """Bad JSON degrades to hashing the raw text rather than losing the field.

        Dropping it would let two images with different configs but equal
        layers collide, which is worse than a redundant transfer.
        """
        garbage = "{not json at all"
        sig = content_signature("sha256:L1", garbage)
        assert sig is not None
        assert sig != content_signature("sha256:L1", "")


class TestParseImageIdentity:
    def test_absent_image(self):
        identity = parse_image_identity("")
        assert identity == ImageIdentity.empty()
        assert identity.image_id is None
        assert identity.repo_digests == []
        assert identity.content_sig is None

    def test_legacy_two_field_output(self):
        identity = parse_image_identity("sha256:abc|repo@sha256:x ")
        assert identity.image_id == "sha256:abc"
        assert identity.repo_digests == ["repo@sha256:x"]
        assert identity.content_sig is None

    def test_full_four_field_output(self):
        raw = 'sha256:abc|repo@sha256:x |sha256:L1 sha256:L2 |{"Env":["A=1"]}'
        identity = parse_image_identity(raw)
        assert identity.image_id == "sha256:abc"
        assert identity.repo_digests == ["repo@sha256:x"]
        assert identity.content_sig == content_signature(
            "sha256:L1 sha256:L2", '{"Env":["A=1"]}'
        )

    def test_config_containing_separator_is_not_truncated(self):
        config = '{"Env":["PIPE=a|b|c"]}'
        identity = parse_image_identity("sha256:abc||sha256:L1 |" + config)
        assert identity.content_sig == content_signature("sha256:L1", config)

    def test_no_digests_but_content_present(self):
        """The broken-node shape: no RepoDigests, but a usable signature.

        Shape is ``<id>|<digests>|<layers>|<config>``; the digest field is
        present but empty on a ``save | load`` node.
        """
        identity = parse_image_identity('sha256:worker||sha256:L1 sha256:L2 |{"Env":[]}')
        assert identity.image_id == "sha256:worker"
        assert identity.repo_digests == []
        assert identity.content_sig is not None


class TestGetImageIdentityFallback:
    """A daemon that cannot render the rich template must degrade, not fail.

    Reporting the image *absent* when the template merely errored would
    trigger a needless multi-gigabyte re-transfer, so the code retries
    with the legacy template.
    """

    RICH_FAIL = ("no such format field", "")

    def test_rich_template_used_when_supported(self):
        out = 'sha256:abc|repo@sha256:x |sha256:L1 |{"Env":[]}'
        with mock.patch(
            "sparkrun.containers.registry.subprocess.run",
            side_effect=_fake_run({IMAGE_INSPECT_FORMAT: (0, out)}),
        ) as m:
            identity = get_image_identity("img:latest")
        assert identity.image_id == "sha256:abc"
        assert identity.content_sig is not None
        # Only one inspect: the rich template succeeded, no retry.
        assert m.call_count == 1

    def test_falls_back_to_legacy_when_rich_template_errors(self):
        """Template error (not absence) still yields the id + digests."""
        with mock.patch(
            "sparkrun.containers.registry.subprocess.run",
            side_effect=_fake_run(
                {
                    IMAGE_INSPECT_FORMAT: self.RICH_FAIL,
                    LEGACY_IMAGE_INSPECT_FORMAT: (0, "sha256:abc|repo@sha256:x "),
                }
            ),
        ):
            identity = get_image_identity("img:latest")
        assert identity.image_id == "sha256:abc"
        assert identity.repo_digests == ["repo@sha256:x"]
        assert identity.content_sig is None

    def test_genuinely_absent_image_is_empty(self):
        """Both templates fail => absent, and the caller transfers."""
        with mock.patch(
            "sparkrun.containers.registry.subprocess.run",
            side_effect=_fake_run(
                {
                    IMAGE_INSPECT_FORMAT: self.RICH_FAIL,
                    LEGACY_IMAGE_INSPECT_FORMAT: (1, ""),
                }
            ),
        ):
            identity = get_image_identity("img:missing")
        assert identity == ImageIdentity.empty()

    def test_legacy_template_is_the_pre_change_shape(self):
        """Pin the legacy template so it stays what old daemons can render."""
        assert IMAGE_INSPECT_FORMAT.startswith("{{.Id}}|")
        assert LEGACY_IMAGE_INSPECT_FORMAT == "{{.Id}}|{{range .RepoDigests}}{{.}} {{end}}"
        assert LEGACY_IMAGE_INSPECT_FORMAT in IMAGE_INSPECT_FORMAT


class TestCrossDriverScenario:
    """End-to-end of the reported failure, with no remote hosts involved.

    Simulates the exact identity triples observed on a two-node DGX Spark
    cluster: a containerd-snapshotter head and an overlay2 worker holding
    the same image after ``save | load``.
    """

    LAYERS = " ".join(f"sha256:{i:064x}" for i in range(1, 30))
    CONFIG = json.dumps(
        {"Entrypoint": ["/entrypoint.sh"], "Env": ["PATH=/bin"], "WorkingDir": "/w"}
    )

    def _head(self):
        return parse_image_identity(
            f"sha256:{'a' * 64}|eugr/img@sha256:{'a' * 64} |{self.LAYERS} |{self.CONFIG}"
        )

    def _worker(self):
        # Different local Id (different driver), no RepoDigests (save/load).
        return parse_image_identity(f"sha256:{'b' * 64}||{self.LAYERS} |{self.CONFIG}")

    def test_head_and_worker_share_content_signature(self):
        head, worker = self._head(), self._worker()
        assert head.content_sig == worker.content_sig
        assert head.content_sig is not None

    def test_ids_and_digests_disagree(self):
        """Confirms the old signals really do fail on this topology."""
        head, worker = self._head(), self._worker()
        assert head.image_id != worker.image_id
        assert head.repo_digests != []
        assert worker.repo_digests == []

    def test_different_image_does_not_share_signature(self):
        layers2 = self.LAYERS + " sha256:" + "f" * 64
        other = parse_image_identity(f"sha256:{'c' * 64}||{layers2} |{self.CONFIG}")
        assert other.content_sig != self._head().content_sig
