"""``sparkrun run --offline``: distribution copies what exists and fetches nothing.

Offline means no internet egress.  Copies *inside* the cluster of an image or
model that already exists (control machine or head) are allowed; ``docker
pull``, Hugging Face downloads and the ensure scripts' ``uv`` /
``huggingface_hub`` installation are not.  So every leaf that would fetch must
refuse instead, and the flag has to reach every one of them through every
transfer mode — a single leaf that missed it is a silent egress.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from unittest import mock

import pytest

from sparkrun.models.distribute import _build_model_ensure_script
from sparkrun.orchestration.transfer import TransferFailure
from sparkrun.scripts import read_script

needs_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="requires a POSIX shell",
)


def _write_stub(path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


# ---------------------------------------------------------------------------
# image_sync.sh under real bash with a fake docker
# ---------------------------------------------------------------------------


@pytest.fixture
def run_image_sync(tmp_path):
    """Execute the rendered ``image_sync.sh`` against a recording ``docker`` shim.

    Returns ``run(offline, force_pull="0", *, present)`` yielding
    ``(CompletedProcess, [argv, ...])``.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "docker.log"

    def run(offline: str, force_pull: str = "0", *, present: bool):
        _write_stub(
            bin_dir / "docker",
            'echo "$@" >> "%s"\nif [ "$1" = "image" ] && [ "$2" = "inspect" ]; then exit %d; fi\nexit 0\n' % (log, 0 if present else 1),
        )
        if log.exists():
            log.unlink()
        script = read_script("image_sync.sh").format(image="img:1.0", force_pull=force_pull, offline=offline)
        env = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]))
        proc = subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, env=env, timeout=30)
        calls = log.read_text().splitlines() if log.exists() else []
        return proc, calls

    return run


def _pulled(calls: list[str]) -> bool:
    return any(c.startswith("pull") for c in calls)


@needs_bash
class TestImageSyncScriptOffline:
    def test_absent_fails_without_pulling(self, run_image_sync):
        proc, calls = run_image_sync("1", present=False)
        assert proc.returncode == 3
        assert "OFFLINE: image img:1.0 is not present on this host and offline mode does not pull" in proc.stderr
        assert not _pulled(calls)

    def test_present_succeeds_without_pulling(self, run_image_sync):
        proc, calls = run_image_sync("1", present=True)
        assert proc.returncode == 0
        assert "Image already available: img:1.0" in proc.stdout
        assert not _pulled(calls)

    @pytest.mark.parametrize("present", [True, False])
    def test_offline_outranks_force_pull(self, run_image_sync, present):
        proc, calls = run_image_sync("1", "1", present=present)
        assert not _pulled(calls)
        assert proc.returncode == (0 if present else 3)

    def test_online_absent_still_pulls(self, run_image_sync):
        proc, calls = run_image_sync("0", present=False)
        assert proc.returncode == 0
        assert any(c.startswith("pull img:1.0") for c in calls)

    def test_no_stray_format_braces(self):
        rendered = read_script("image_sync.sh").format(image="i", force_pull="0", offline="1")
        assert "{" not in rendered and "}" not in rendered


class TestImageSyncRenderSites:
    """Both places that render ``image_sync.sh`` pass the flag."""

    @mock.patch("sparkrun.containers.sync.sync_resource_to_hosts", return_value=[])
    def test_sync_image_to_hosts(self, mock_sync):
        from sparkrun.containers.sync import sync_image_to_hosts

        sync_image_to_hosts("img:1.0", ["h1"], offline=True)
        assert 'OFFLINE="1"' in mock_sync.call_args.args[0]
        sync_image_to_hosts("img:1.0", ["h1"])
        assert 'OFFLINE="0"' in mock_sync.call_args.args[0]

    @mock.patch("sparkrun.containers.distribute._check_remote_image_identities", return_value={})
    @mock.patch("sparkrun.orchestration.ssh.run_remote_script_streaming")
    def test_distribute_image_from_head(self, mock_run, mock_check):
        from sparkrun.containers.distribute import distribute_image_from_head
        from sparkrun.orchestration.ssh import RemoteResult

        mock_run.return_value = RemoteResult(host="head", returncode=0, stdout="", stderr="")
        distribute_image_from_head("img:1.0", ["head"], offline=True)
        assert 'OFFLINE="1"' in mock_run.call_args_list[0].args[1]


# ---------------------------------------------------------------------------
# model_sync.sh / model_sync_gguf.sh under real bash with network-tool shims
# ---------------------------------------------------------------------------

# Everything the miss path can reach for to download or install.
_NETWORK_TOOLS = ("huggingface-cli", "hf", "uv", "uvx", "curl", "wget", "pip", "pip3", "python3")


@pytest.fixture
def run_model_sync(tmp_path):
    """Execute a rendered model ensure script with every network tool shimmed.

    Returns ``run(script)`` yielding ``(CompletedProcess, [recorded argv])``.
    """
    bin_dir = tmp_path / "stub-bin"
    bin_dir.mkdir()
    log = tmp_path / "net.log"
    for tool in _NETWORK_TOOLS:
        _write_stub(bin_dir / tool, 'echo "%s $*" >> "%s"\nexit 0\n' % (tool, log))

    def run(script: str):
        env = {"HOME": "/nonexistent", "PATH": f"{bin_dir}:/usr/bin:/bin"}
        proc = subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, timeout=30, env=env)
        return proc, (log.read_text().splitlines() if log.exists() else [])

    return run


def _seed_snapshot(cache, repo_dir: str, filename: str):
    snap = cache / "hub" / repo_dir / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / filename).write_text("w")
    refs = cache / "hub" / repo_dir / "refs"
    refs.mkdir()
    (refs / "main").write_text("abc")


@needs_bash
class TestModelSyncScriptOffline:
    def test_missing_model_fails_before_any_fetch(self, tmp_path, run_model_sync):
        proc, calls = run_model_sync(_build_model_ensure_script("org/model", str(tmp_path), offline=True))

        assert proc.returncode == 3
        assert "OFFLINE: model org/model is not in this host's Hugging Face cache" in proc.stderr
        assert calls == []
        assert "Downloading" not in proc.stdout
        assert "installing pinned uv" not in proc.stdout

    def test_cached_model_succeeds(self, tmp_path, run_model_sync):
        _seed_snapshot(tmp_path, "models--org--model", "model.safetensors")

        proc, calls = run_model_sync(_build_model_ensure_script("org/model", str(tmp_path), offline=True))

        assert proc.returncode == 0, proc.stderr
        assert "already cached" in proc.stdout
        assert calls == []

    def test_missing_gguf_fails_before_any_fetch(self, tmp_path, run_model_sync):
        proc, calls = run_model_sync(_build_model_ensure_script("org/repo-GGUF:Q4_K_M", str(tmp_path), offline=True))

        assert proc.returncode == 3
        assert "OFFLINE: GGUF model org/repo-GGUF" in proc.stderr
        assert calls == []

    def test_cached_gguf_succeeds(self, tmp_path, run_model_sync):
        _seed_snapshot(tmp_path, "models--org--repo-GGUF", "repo-Q4_K_M.gguf")

        proc, calls = run_model_sync(_build_model_ensure_script("org/repo-GGUF:Q4_K_M", str(tmp_path), offline=True))

        assert proc.returncode == 0, proc.stderr
        assert calls == []

    def test_online_missing_still_downloads(self, tmp_path, run_model_sync):
        proc, calls = run_model_sync(_build_model_ensure_script("org/model", str(tmp_path)))

        assert proc.returncode == 0
        assert any(c.startswith("huggingface-cli download org/model") for c in calls)

    def test_rendered_default_is_online(self):
        assert 'OFFLINE="0"' in _build_model_ensure_script("org/model", "/hf")
        assert 'OFFLINE="1"' in _build_model_ensure_script("org/model", "/hf", offline=True)


class TestRemoteModelLeavesRenderOffline:
    @mock.patch("sparkrun.orchestration.primitives.sync_resource_to_hosts", return_value=["h2"])
    def test_per_node_renders_and_classifies(self, mock_sync):
        from sparkrun.models.distribute import OFFLINE_REMOTE_MODEL_NOT_CACHED_REASON, distribute_model_per_node

        failures = distribute_model_per_node("org/model", ["h1", "h2"], cache_dir="/hf", offline=True)

        assert 'OFFLINE="1"' in mock_sync.call_args.args[0]
        assert failures == [TransferFailure(host="h2", reason=OFFLINE_REMOTE_MODEL_NOT_CACHED_REASON)]

    @mock.patch("sparkrun.orchestration.distribution._distribute_from_head", return_value=["head", "w1"])
    def test_from_head_renders_and_classifies(self, mock_head):
        from sparkrun.models.distribute import OFFLINE_REMOTE_MODEL_NOT_CACHED_REASON, distribute_model_from_head

        failures = distribute_model_from_head("org/model", ["head", "w1"], cache_dir="/hf", offline=True)

        assert 'OFFLINE="1"' in mock_head.call_args.kwargs["ensure_script"]
        assert {f.reason for f in failures} == {OFFLINE_REMOTE_MODEL_NOT_CACHED_REASON}


def test_classifier_names_offline_refusal():
    from sparkrun.orchestration.ssh import RemoteResult
    from sparkrun.orchestration.transfer import classify_rsync_failure

    result = RemoteResult(host="h", returncode=3, stdout="", stderr="OFFLINE: image x is not present on this host")
    assert classify_rsync_failure(result) == "offline: not present, not pulled"


# ---------------------------------------------------------------------------
# ensure_image — the control machine's copy
# ---------------------------------------------------------------------------


class TestEnsureImageOffline:
    @mock.patch("sparkrun.containers.registry.pull_image")
    @mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=True)
    def test_present_returns_zero_without_pull(self, _exists, mock_pull):
        from sparkrun.containers.registry import ensure_image

        assert ensure_image("img:1.0", offline=True) == 0
        mock_pull.assert_not_called()

    @mock.patch("sparkrun.containers.registry.pull_image")
    @mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=True)
    def test_latest_present_skips_opportunistic_pull(self, _exists, mock_pull):
        from sparkrun.containers.registry import ensure_image

        assert ensure_image("ghcr.io/example/img:latest", offline=True) == 0
        mock_pull.assert_not_called()

    @mock.patch("sparkrun.containers.registry.pull_image")
    @mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=False)
    def test_absent_fails_without_pull(self, _exists, mock_pull, caplog):
        from sparkrun.containers.registry import ensure_image

        assert ensure_image("img:1.0", offline=True) != 0
        mock_pull.assert_not_called()
        assert "offline: image img:1.0 is not present on the control machine" in caplog.text

    @mock.patch("sparkrun.containers.registry.pull_image")
    @mock.patch("sparkrun.containers.registry.image_exists_locally", return_value=True)
    def test_force_pull_is_refused(self, _exists, mock_pull):
        from sparkrun.containers.registry import ensure_image

        assert ensure_image("img:1.0", force_pull=True, offline=True) != 0
        mock_pull.assert_not_called()

    @mock.patch("subprocess.run")
    def test_no_docker_pull_subprocess(self, mock_run):
        from sparkrun.containers.registry import ensure_image

        mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")
        ensure_image("ghcr.io/example/img:latest", offline=True)
        assert all(c.args[0][:2] != ["docker", "pull"] for c in mock_run.call_args_list)


# ---------------------------------------------------------------------------
# distribute_model_from_local — the control machine's cache
# ---------------------------------------------------------------------------


class TestModelFromLocalOffline:
    @mock.patch("sparkrun.models.distribute.run_rsync_parallel")
    @mock.patch("sparkrun.models.distribute.is_model_cached", return_value=False)
    @mock.patch("sparkrun.models.distribute.download_model")
    def test_uncached_fails_without_download(self, mock_dl, mock_cached, mock_rsync):
        from sparkrun.models.distribute import OFFLINE_MODEL_NOT_CACHED_REASON, distribute_model_from_local

        failures = distribute_model_from_local("org/model", ["h1", "h2"], cache_dir="/hf", revision="abc", offline=True)

        mock_dl.assert_not_called()
        mock_rsync.assert_not_called()
        assert failures == [TransferFailure(host=h, reason=OFFLINE_MODEL_NOT_CACHED_REASON) for h in ("h1", "h2")]
        # Same revision semantics as the presence helper.
        assert mock_cached.call_args.kwargs["revision"] == "abc"

    @mock.patch("sparkrun.models.distribute.map_transfer_failures_detailed", return_value=[])
    @mock.patch("sparkrun.models.distribute.run_rsync_parallel", return_value=[])
    @mock.patch("sparkrun.models.distribute.is_model_cached", return_value=True)
    @mock.patch("sparkrun.models.distribute.download_model")
    def test_cached_proceeds_to_rsync(self, mock_dl, _cached, mock_rsync, _map):
        from sparkrun.models.distribute import distribute_model_from_local

        failures = distribute_model_from_local("org/model", ["h1"], cache_dir="/hf", offline=True, preserve_perms=False)

        assert failures == []
        mock_dl.assert_not_called()
        mock_rsync.assert_called_once()


# ---------------------------------------------------------------------------
# Threading: every leaf in every mode receives offline=True
# ---------------------------------------------------------------------------


@pytest.fixture
def image_leaves():
    with (
        mock.patch("sparkrun.containers.distribute.distribute_image_from_local", return_value=[]) as m_local,
        mock.patch("sparkrun.containers.distribute.distribute_image_from_head", return_value=[]) as m_head,
        mock.patch("sparkrun.containers.sync.sync_image_to_hosts", return_value=[]) as m_sync,
    ):
        yield {"local": m_local, "head": m_head, "sync": m_sync}


@pytest.fixture
def model_leaves():
    with (
        mock.patch("sparkrun.models.distribute.distribute_model_from_local", return_value=[]) as m_local,
        mock.patch("sparkrun.models.distribute.distribute_model_from_head", return_value=[]) as m_head,
        mock.patch("sparkrun.models.distribute.distribute_model_per_node", return_value=[]) as m_node,
    ):
        yield {"local": m_local, "head": m_head, "node": m_node}


def _called_leaves(leaves: dict) -> list:
    return [c for m in leaves.values() for c in m.call_args_list]


def _single_image(mode, hosts=("head", "w1"), targets=None, auto_delegated=False, **kw):
    from sparkrun.orchestration.distribution import _distribute_single_image

    hosts = list(hosts)
    return _distribute_single_image("img:1.0", list(targets or hosts), hosts, mode, None, None, {}, False, auto_delegated, **kw)


def _single_model(mode, hosts=("head", "w1"), targets=None, auto_delegated=False, **kw):
    from sparkrun.orchestration.distribution import _distribute_single_model

    hosts = list(hosts)
    return _distribute_single_model(
        "org/model", list(targets or hosts), hosts, "/hf", "/hf", mode, None, None, {}, None, None, False, auto_delegated, **kw
    )


class TestImageThreading:
    @pytest.mark.parametrize("mode", ["local", "push", "delegated", "pull"])
    def test_every_leaf_offline(self, image_leaves, mode):
        _single_image(mode, offline=True)
        calls = _called_leaves(image_leaves)
        assert calls
        assert all(c.kwargs.get("offline") is True for c in calls)

    def test_subset_push(self, image_leaves):
        _single_image("push", hosts=("head", "w1", "w2"), targets=("head", "w1"), offline=True)
        calls = _called_leaves(image_leaves)
        assert len(calls) == 2 and all(c.kwargs["offline"] is True for c in calls)

    def test_auto_delegated_push_fallback(self, image_leaves):
        image_leaves["head"].side_effect = [["head", "w1"], []]
        _single_image("delegated", auto_delegated=True, offline=True)
        # delegated attempt, push to head, head→worker fan-out
        assert image_leaves["head"].call_count == 2
        assert image_leaves["local"].call_count == 1
        assert all(c.kwargs["offline"] is True for c in _called_leaves(image_leaves))

    def test_pull_fallback_and_heterogeneous(self, image_leaves):
        image_leaves["sync"].return_value = ["w1"]
        _single_image("delegated", heterogeneous=True, auto_delegated=True, offline=True)
        assert image_leaves["sync"].call_count == 1 and image_leaves["local"].call_count == 1
        assert all(c.kwargs["offline"] is True for c in _called_leaves(image_leaves))

    def test_default_is_online(self, image_leaves):
        _single_image("local")
        assert image_leaves["local"].call_args.kwargs["offline"] is False

    def test_image_plan_forwards(self, image_leaves):
        from sparkrun.orchestration.distribution import _distribute_image_plan

        plan = [("img:a", ["head"]), ("img:b", ["w1"])]
        _distribute_image_plan(plan, ["head", "w1"], "pull", None, None, {}, False, False, offline=True)
        assert image_leaves["sync"].call_count == 2
        assert all(c.kwargs["offline"] is True for c in image_leaves["sync"].call_args_list)


class TestModelThreading:
    @pytest.mark.parametrize("mode", ["local", "push", "delegated", "pull"])
    def test_every_leaf_offline(self, model_leaves, mode):
        _single_model(mode, offline=True)
        calls = _called_leaves(model_leaves)
        assert calls
        assert all(c.kwargs.get("offline") is True for c in calls)

    def test_pull_with_shared_cache(self, model_leaves):
        from sparkrun.orchestration.distribution import ModelDistributionPrefs

        _single_model("pull", prefs=ModelDistributionPrefs(skip_fan_out=True), offline=True)
        assert model_leaves["head"].call_args.kwargs["offline"] is True

    def test_auto_delegated_push_fallback(self, model_leaves):
        model_leaves["head"].side_effect = [[TransferFailure(host="head", reason="x")], []]
        _single_model("delegated", auto_delegated=True, offline=True)
        assert model_leaves["head"].call_count == 2
        assert model_leaves["local"].call_count == 1
        assert all(c.kwargs["offline"] is True for c in _called_leaves(model_leaves))

    def test_default_is_online(self, model_leaves):
        _single_model("local")
        assert model_leaves["local"].call_args.kwargs["offline"] is False


class TestPushHelpersThreading:
    def test_image_push(self, image_leaves):
        from sparkrun.orchestration.distribution import _distribute_image_push

        _distribute_image_push("img:1.0", ["head", "w1"], None, {}, False, offline=True)
        calls = _called_leaves(image_leaves)
        assert len(calls) == 2 and all(c.kwargs["offline"] is True for c in calls)

    def test_model_push(self, model_leaves):
        from sparkrun.orchestration.distribution import _distribute_model_push

        _distribute_model_push("org/model", ["head", "w1"], "/hf", None, {}, offline=True)
        calls = _called_leaves(model_leaves)
        assert len(calls) == 2 and all(c.kwargs["offline"] is True for c in calls)


# ---------------------------------------------------------------------------
# distribute_from_config reaches the leaves
# ---------------------------------------------------------------------------


def _recipe():
    from sparkrun.core.recipe import Recipe

    return Recipe.from_dict(
        {
            "recipe_version": "2",
            "model": "org/model",
            "runtime": "vllm-distributed",
            "container": "img:1.0",
        }
    )


@mock.patch("sparkrun.orchestration.distribution._is_cross_user", return_value=False)
@mock.patch("sparkrun.orchestration.distribution.is_local_host", return_value=False)
@mock.patch("sparkrun.orchestration.distribution.is_control_in_cluster", return_value=False)
@mock.patch("sparkrun.orchestration.infiniband.validate_ib_connectivity", return_value={})
@mock.patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
@mock.patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={})
@pytest.mark.parametrize("mode", ["local", "push", "delegated", "pull", "auto"])
def test_distribute_from_config_reaches_leaves(
    mock_ssh, mock_detect, _validate, _in_cluster, _local, _cross, mode, image_leaves, model_leaves, tmp_path
):
    from sparkrun.orchestration.comm_env import ClusterCommEnv
    from sparkrun.orchestration.distribution import distribute_from_config
    from sparkrun.orchestration.infiniband import IBDetectionResult

    mock_detect.return_value = IBDetectionResult(comm_env=ClusterCommEnv.empty(), ib_ip_map={}, mgmt_ip_map={})
    if mode == "auto":
        # auto resolves to delegated with the push fallback armed; make both
        # delegated attempts fail so the fallback legs run too.
        image_leaves["head"].side_effect = [["head", "w1"], []]
        model_leaves["head"].side_effect = [[TransferFailure(host="head", reason="x")], []]

    distribute_from_config(
        _recipe(),
        "img:1.0",
        ["head", "w1"],
        "/hf",
        mock.MagicMock(cache_dir=str(tmp_path)),
        dry_run=False,
        transfer_mode=mode,
        offline=True,
    )

    image_calls = _called_leaves(image_leaves)
    model_calls = _called_leaves(model_leaves)
    assert image_calls and model_calls
    assert all(c.kwargs["offline"] is True for c in image_calls + model_calls)
    if mode == "auto":
        assert image_leaves["local"].called and model_leaves["local"].called


class TestLocalFastPathOffline:
    """Single-localhost launches ensure locally; offline must not download there either."""

    def _patch(self, monkeypatch, *, cached: bool):
        seen: dict = {}
        monkeypatch.setattr("sparkrun.orchestration.distribution.is_local_host", lambda h: True)
        monkeypatch.setattr("sparkrun.orchestration.distribution._is_cross_user", lambda kw: False)
        monkeypatch.setattr("sparkrun.orchestration.distribution._get_hf_token", lambda: "")
        monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda *a, **kw: {})

        def _ensure(image, **kw):
            seen["ensure_offline"] = kw.get("offline")
            return 0

        monkeypatch.setattr("sparkrun.containers.registry.ensure_image", _ensure)
        download = mock.MagicMock(return_value=0)
        monkeypatch.setattr("sparkrun.models.download.download_model", download)
        monkeypatch.setattr("sparkrun.models.download.is_model_cached", lambda *a, **kw: cached)
        return seen, download

    def test_cached(self, monkeypatch, tmp_path):
        from sparkrun.orchestration.distribution import distribute_from_config

        seen, download = self._patch(monkeypatch, cached=True)
        distribute_from_config(
            _recipe(), "img:1.0", ["localhost"], "/hf", mock.MagicMock(cache_dir=str(tmp_path)), dry_run=False, offline=True
        )

        assert seen["ensure_offline"] is True
        download.assert_not_called()

    def test_uncached_raises(self, monkeypatch, tmp_path):
        from sparkrun.orchestration.distribution import DistributionError, distribute_from_config

        _, download = self._patch(monkeypatch, cached=False)
        with pytest.raises(DistributionError, match="offline: model org/model"):
            distribute_from_config(
                _recipe(), "img:1.0", ["localhost"], "/hf", mock.MagicMock(cache_dir=str(tmp_path)), dry_run=False, offline=True
            )
        download.assert_not_called()


class TestStagePreparedImagesForwards:
    @mock.patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={})
    @mock.patch("sparkrun.orchestration.distribution.distribute_from_config", return_value=(None, {}, {}, {}))
    def test_forwards_offline(self, mock_dist, _ssh):
        from types import SimpleNamespace

        from sparkrun.core.image_preparation import stage_prepared_images

        prepared = SimpleNamespace(
            image_plan=SimpleNamespace(head_image=lambda: "img:1.0"),
            images_by_node=("img:1.0",),
            container_distribution=None,
        )
        stage_prepared_images(prepared, _recipe(), ["h1"], "/hf", mock.MagicMock(), offline=True)
        assert mock_dist.call_args.kwargs["offline"] is True
        stage_prepared_images(prepared, _recipe(), ["h1"], "/hf", mock.MagicMock())
        assert mock_dist.call_args.kwargs["offline"] is False
