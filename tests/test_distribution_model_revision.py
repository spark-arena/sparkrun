"""A distribution entry's revision is per-entry and authoritative.

A launch distributes several unrelated repos: the served model, plus any
speculative draft model a runtime adds in ``prepare()``.  A commit SHA is only
meaningful in the repo it came from, so the recipe's top-level
``model_revision`` must reach *only* the served model's entry.

Distribution used to apply it to every entry lacking one of its own
(``entry.revision or model_revision``).  With a pinned recipe that meant the
draft model was fetched at the served model's SHA — a revision that repo has
never had — and the launch died on ``Revision Not Found`` *after* paying for
the served model's sync.  Both distribution paths carried the same fallback.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from sparkrun.core.recipe import Recipe

PRIMARY = "Qwen/Qwen3-1.7B"
DRAFT = "Qwen/Qwen3-1.7B-Draft"
SHA = "319f741cce68d7914884900c138a1fbb70a42f30"


class _Cfg:
    class _P:
        def __str__(self):
            return "/tmp/cache"

    cache_dir = _P()


def _recipe(model_revision=None):
    """A recipe whose runtime has added a draft model, as ``prepare()`` does."""
    d = {
        "recipe_version": "2",
        "name": "spec",
        "model": PRIMARY,
        "runtime": "vllm-distributed",
        "container": "img:latest",
    }
    if model_revision:
        d["model_revision"] = model_revision
    r = Recipe.from_dict(d)
    r.distribution_config.add_model(DRAFT)
    return r


# ---------------------------------------------------------------------------
# Single-localhost fast path
# ---------------------------------------------------------------------------


def _patch_local(monkeypatch, calls):
    monkeypatch.setattr("sparkrun.orchestration.distribution.is_local_host", lambda h: True)
    monkeypatch.setattr("sparkrun.orchestration.distribution._is_cross_user", lambda kw: False)
    monkeypatch.setattr("sparkrun.orchestration.distribution._get_hf_token", lambda: "")
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda *a, **kw: {})
    monkeypatch.setattr("sparkrun.containers.registry.ensure_image", lambda image, **kw: 0)

    def _dl(model, **kw):
        calls.append((model, kw.get("revision")))
        return 0

    monkeypatch.setattr("sparkrun.models.download.download_model", _dl)


def test_local_path_pins_only_the_served_model(monkeypatch):
    calls: list = []
    _patch_local(monkeypatch, calls)
    from sparkrun.orchestration.distribution import distribute_from_config

    distribute_from_config(_recipe(SHA), "img:latest", ["localhost"], "/tmp/cache", _Cfg(), dry_run=False)

    assert calls == [(PRIMARY, SHA), (DRAFT, None)]


def test_local_path_unpinned_recipe_pins_nothing(monkeypatch):
    calls: list = []
    _patch_local(monkeypatch, calls)
    from sparkrun.orchestration.distribution import distribute_from_config

    distribute_from_config(_recipe(), "img:latest", ["localhost"], "/tmp/cache", _Cfg(), dry_run=False)

    assert calls == [(PRIMARY, None), (DRAFT, None)]


# ---------------------------------------------------------------------------
# Multi-host cluster path
# ---------------------------------------------------------------------------


@patch("sparkrun.orchestration.distribution._distribute_single_image", return_value=[])
@patch("sparkrun.orchestration.distribution._distribute_single_model", return_value=[])
@patch("sparkrun.orchestration.distribution._is_cross_user", return_value=False)
@patch("sparkrun.orchestration.distribution.is_local_host", return_value=False)
@patch("sparkrun.orchestration.distribution.is_control_in_cluster", return_value=True)
@patch("sparkrun.orchestration.infiniband.validate_ib_connectivity", return_value={})
@patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
@patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={})
def test_cluster_path_pins_only_the_served_model(
    mock_ssh, mock_detect, mock_validate, mock_in_cluster, mock_local, mock_cross, mock_model, mock_image
):
    from sparkrun.orchestration.comm_env import ClusterCommEnv
    from sparkrun.orchestration.distribution import distribute_from_config, TransferModeResult
    from sparkrun.orchestration.infiniband import IBDetectionResult

    mock_detect.return_value = IBDetectionResult(
        comm_env=ClusterCommEnv.empty(),
        ib_ip_map={},
        mgmt_ip_map={},
    )

    distribute_from_config(
        _recipe(SHA),
        "img:latest",
        ["h1", "h2"],
        "/tmp/cache",
        MagicMock(cache_dir="/tmp/cache"),
        dry_run=False,
        pre_ib=TransferModeResult(mode="local", ib_result=None),
    )

    # _distribute_single_model(name, targets, host_list, cache_dir,
    #                          local_cache, mode, hosts, worker_hosts,
    #                          ssh_kwargs, revision, ...)
    seen = [(c.args[0], c.args[9]) for c in mock_model.call_args_list]
    assert seen == [(PRIMARY, SHA), (DRAFT, None)]
