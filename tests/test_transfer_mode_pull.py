"""``--transfer-mode pull``: every node fetches from origin itself.

The other three modes all route the bytes through *somewhere* — the control
machine (``local``/``push``) or the head (``delegated``).  ``pull`` routes them
through nobody, which is the only strategy that works when nodes need
*different* images and the fastest one when every node has good egress.  It
costs N× bandwidth and needs credentials on every node, so it is opt-in.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sparkrun.core.cluster_manager import ModelDistributionPrefs
from sparkrun.orchestration.distribution import _distribute_single_image, _distribute_single_model

HOSTS = ["h1", "h2", "h3"]


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


@patch("sparkrun.containers.sync.sync_resource_to_hosts", return_value=[])
def test_pull_fans_every_host_out_to_the_registry(mock_sync):
    """No head leg at all: one parallel sync covering every target."""
    failed = _distribute_single_image(
        "org/img:tag",
        HOSTS,
        HOSTS,
        "pull",
        None,
        None,
        {},
        dry_run=False,
        auto_delegated=False,
    )

    assert failed == []
    assert mock_sync.call_count == 1
    assert mock_sync.call_args.args[1] == HOSTS


@patch("sparkrun.containers.sync.sync_resource_to_hosts", return_value=[])
def test_rebuild_reaches_the_side_that_actually_pulls(mock_sync):
    """Under ``pull`` that side is every node, so force_pull must reach them.

    The presence check is metadata-only, so without this an image re-pushed
    under the same tag is never refreshed and ``--rebuild`` silently does
    nothing.
    """
    _distribute_single_image(
        "org/img:tag",
        HOSTS,
        HOSTS,
        "pull",
        None,
        None,
        {},
        dry_run=False,
        auto_delegated=False,
        force_pull=True,
    )

    script = mock_sync.call_args.args[0]
    assert 'FORCE_PULL="1"' in script


@patch("sparkrun.containers.distribute.distribute_image_from_local", return_value=[])
@patch("sparkrun.containers.sync.sync_image_to_hosts", return_value=["h2"])
def test_inferred_pull_falls_back_to_push_for_the_nodes_that_failed(mock_sync, mock_push):
    """Only the failed hosts are pushed to, and only when the mode was inferred."""
    failed = _distribute_single_image(
        "org/img:tag",
        HOSTS,
        HOSTS,
        "pull",
        None,
        None,
        {},
        dry_run=False,
        auto_delegated=True,
    )

    assert failed == []
    assert mock_push.call_args.args[1] == ["h2"]


@patch("sparkrun.containers.distribute.distribute_image_from_local")
@patch("sparkrun.containers.sync.sync_image_to_hosts", return_value=["h2"])
def test_explicit_pull_is_honored_literally_and_never_pushes(mock_sync, mock_push):
    """A user-named mode is not second-guessed — the rule `delegated` follows."""
    failed = _distribute_single_image(
        "org/img:tag",
        HOSTS,
        HOSTS,
        "pull",
        None,
        None,
        {},
        dry_run=False,
        auto_delegated=False,
    )

    assert failed == ["h2"]
    mock_push.assert_not_called()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@patch("sparkrun.models.distribute.distribute_model_per_node", return_value=[])
@pytest.mark.usefixtures("unavailable_model_inventory")
def test_model_pull_downloads_on_every_node(mock_per_node):
    _distribute_single_model(
        "org/model",
        HOSTS,
        HOSTS,
        "/cache",
        "/cache",
        "pull",
        None,
        None,
        {},
        revision=None,
        hf_token=None,
        dry_run=False,
        auto_delegated=False,
    )

    assert mock_per_node.call_args.args[1] == HOSTS


@patch("sparkrun.models.distribute.distribute_model_from_head", return_value=[])
@patch("sparkrun.models.distribute.distribute_model_per_node")
@pytest.mark.usefixtures("unavailable_model_inventory")
def test_a_shared_cache_overrides_pull(mock_per_node, mock_from_head):
    """N nodes writing one NFS path concurrently is waste at best.

    With ``skip_fan_out`` the workers already mount the head's copy, so the
    head downloads once and the fan-out is skipped — not N downloads racing
    into the same directory.
    """
    _distribute_single_model(
        "org/model",
        HOSTS,
        HOSTS,
        "/shared",
        "/shared",
        "pull",
        None,
        None,
        {},
        revision=None,
        hf_token=None,
        dry_run=False,
        auto_delegated=False,
        prefs=ModelDistributionPrefs(skip_fan_out=True),
    )

    mock_per_node.assert_not_called()
    assert mock_from_head.call_args.kwargs["skip_fan_out"] is True


@patch("sparkrun.models.distribute.distribute_model_from_head")
@patch("sparkrun.models.distribute.distribute_model_per_node", return_value=[])
@pytest.mark.usefixtures("unavailable_model_inventory")
def test_a_shared_cache_on_one_host_still_pulls_directly(mock_per_node, mock_from_head):
    """`skip_fan_out` is about workers mounting the head's copy — with a single
    target there are no workers, so the head indirection buys nothing."""
    _distribute_single_model(
        "org/model",
        ["h1"],
        ["h1"],
        "/shared",
        "/shared",
        "pull",
        None,
        None,
        {},
        revision=None,
        hf_token=None,
        dry_run=False,
        auto_delegated=False,
        prefs=ModelDistributionPrefs(skip_fan_out=True),
    )

    mock_from_head.assert_not_called()
    mock_per_node.assert_called_once()


def test_head_and_per_node_paths_build_the_same_fetch_script():
    """One builder for both, so GGUF handling and token injection cannot drift.

    The drift would only show on gated or quant-selected models, which is
    exactly where it would be most expensive to discover.
    """
    from sparkrun.models.distribute import _build_model_ensure_script

    plain = _build_model_ensure_script("org/model", "/cache", revision="abc123")
    assert "org/model" in plain and "abc123" in plain

    gguf = _build_model_ensure_script("org/model:Q4_K_M", "/cache")
    assert "Q4_K_M" in gguf

    gated = _build_model_ensure_script("org/model", "/cache", hf_token="hf_secret")
    assert gated.startswith("export HF_TOKEN=")


@pytest.mark.parametrize("mode,heterogeneous", [("pull", False), ("delegated", True)])
@pytest.mark.parametrize("options", [None, ["-p", "2222", "-o", "ProxyJump=bastion"]])
@pytest.mark.parametrize("dry_run,offline,force_pull", [(False, False, True), (True, True, False)])
def test_pull_forwards_configured_ssh_through_real_sync_helpers(mode, heterogeneous, options, dry_run, offline, force_pull):
    from sparkrun.orchestration.ssh import RemoteResult

    results = [RemoteResult(host=h, returncode=int(h == "h2"), stdout="", stderr="") for h in HOSTS]
    ssh = {"ssh_user": "user", "ssh_key": "/key", "ssh_options": options}
    with patch("sparkrun.orchestration.primitives.run_remote_scripts_parallel", return_value=results) as run:
        failed = _distribute_single_image(
            "org/img:v1",
            HOSTS,
            HOSTS,
            mode,
            None,
            None,
            ssh,
            dry_run=dry_run,
            auto_delegated=False,
            force_pull=force_pull,
            heterogeneous=heterogeneous,
            offline=offline,
        )
    assert failed == ["h2"]
    run.assert_called_once()
    assert run.call_args.args[0] == HOSTS
    assert run.call_args.kwargs == {**ssh, "dry_run": dry_run, "session_guard": True}
    script = run.call_args.args[1]
    assert 'FORCE_PULL="%d"' % force_pull in script
    assert 'OFFLINE="%d"' % offline in script


@pytest.mark.parametrize("helper", ["image", "resource"])
def test_sync_helpers_keep_the_existing_positional_dry_run(helper):
    from sparkrun.containers.sync import sync_image_to_hosts
    from sparkrun.orchestration.primitives import sync_resource_to_hosts

    with patch("sparkrun.orchestration.primitives.run_remote_scripts_parallel", return_value=[]) as run:
        if helper == "image":
            sync_image_to_hosts("org/img:v1", HOSTS, "user", "/key", True)
        else:
            sync_resource_to_hosts("echo sync", HOSTS, "Resource", "user", "/key", True)
    assert run.call_args.kwargs["dry_run"] is True
    assert run.call_args.kwargs["ssh_options"] is None
