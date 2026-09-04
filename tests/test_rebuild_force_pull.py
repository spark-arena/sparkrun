"""Tests for ``sparkrun run --rebuild`` on the docker-pull (registry image) path.

``--rebuild`` sets ``recipe.builder_config["rebuild"]``, which historically only
the *eugr* builder read.  A plain v2 recipe declares no ``builder:`` at all, so
``recipe.builder`` is ``""`` and ``launch_inference`` skips the builder phase
entirely — the flag was a documented no-op for exactly the recipes that pull a
registry image.

The wiring therefore lives in the distribution layer, which is where the pull
actually happens for those recipes, and it is routed to whichever side pulls
from the registry for the transfer mode in play.  These tests pin that routing
plus the shell-level branch in ``image_sync.sh``.
"""

from __future__ import annotations

import os
import stat
import subprocess
from unittest import mock

import pytest

from sparkrun.scripts import read_script


# ---------------------------------------------------------------------------
# image_sync.sh — the head-side ensure script
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_docker(tmp_path):
    """A stub ``docker`` on PATH that records argv and reports the image present.

    Returns a callable ``run(force_pull, *, present=True)`` executing the
    rendered script and yielding ``(stdout, [argv, ...])``.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "docker.log"

    def _write_stub(present: bool) -> None:
        stub = bin_dir / "docker"
        # `docker image inspect` exit status is what the presence check reads.
        stub.write_text(
            "#!/bin/bash\n"
            'echo "$@" >> "%s"\n'
            'if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then exit %d; fi\n'
            "exit 0\n" % (log, 0 if present else 1)
        )
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    def run(force_pull: str, *, present: bool = True):
        _write_stub(present)
        if log.exists():
            log.unlink()
        script = read_script("image_sync.sh").format(image="img:latest", force_pull=force_pull)
        env = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]))
        proc = subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, env=env)
        assert proc.returncode == 0, proc.stderr
        calls = log.read_text().splitlines() if log.exists() else []
        return proc.stdout, calls

    return run


class TestImageSyncScriptForcePull:
    """The presence check is metadata-only, so --rebuild must bypass it."""

    def test_present_and_not_forced_skips_pull(self, fake_docker):
        stdout, calls = fake_docker("0", present=True)
        assert "Image already available" in stdout
        assert not any(c.startswith("pull") for c in calls)

    def test_present_but_forced_pulls_anyway(self, fake_docker):
        """The regression this exists for: a present-but-incomplete image."""
        stdout, calls = fake_docker("1", present=True)
        assert "Force pull requested" in stdout
        assert any(c.startswith("pull img:latest") for c in calls)

    def test_forced_does_not_bother_inspecting(self, fake_docker):
        """Short-circuit order: no point asking when the answer can't matter."""
        _, calls = fake_docker("1", present=True)
        assert not any(c.startswith("image inspect") for c in calls)

    def test_absent_pulls_without_force(self, fake_docker):
        stdout, calls = fake_docker("0", present=False)
        assert "Pulling image" in stdout
        assert any(c.startswith("pull img:latest") for c in calls)

    def test_script_has_no_stray_format_braces(self):
        """image_sync.sh is consumed via str.format(); literal braces break it."""
        rendered = read_script("image_sync.sh").format(image="i", force_pull="0")
        assert "{" not in rendered and "}" not in rendered


# ---------------------------------------------------------------------------
# containers.distribute — force_pull reaches the pulling side
# ---------------------------------------------------------------------------


class TestDistributeImageFromLocalForcePull:
    """The control machine is the pulling side for local/push."""

    @mock.patch("sparkrun.containers.distribute.run_pipeline_to_remotes_parallel")
    @mock.patch("sparkrun.containers.distribute._filter_hosts_needing_image")
    @mock.patch("sparkrun.containers.distribute.get_image_identity")
    @mock.patch("sparkrun.containers.distribute.ensure_image")
    def test_force_pull_forwarded(self, mock_ensure, mock_ident, mock_filter, mock_pipe):
        from sparkrun.containers.distribute import distribute_image_from_local

        mock_ensure.return_value = 0
        mock_ident.return_value = ("sha256:new", [])
        mock_filter.return_value = []

        distribute_image_from_local("img:latest", ["h1"], force_pull=True)
        assert mock_ensure.call_args.kwargs["force_pull"] is True

    @mock.patch("sparkrun.containers.distribute.run_pipeline_to_remotes_parallel")
    @mock.patch("sparkrun.containers.distribute._filter_hosts_needing_image")
    @mock.patch("sparkrun.containers.distribute.get_image_identity")
    @mock.patch("sparkrun.containers.distribute.ensure_image")
    def test_default_is_unforced(self, mock_ensure, mock_ident, mock_filter, mock_pipe):
        from sparkrun.containers.distribute import distribute_image_from_local

        mock_ensure.return_value = 0
        mock_ident.return_value = ("sha256:x", [])
        mock_filter.return_value = []

        distribute_image_from_local("img:latest", ["h1"])
        assert mock_ensure.call_args.kwargs["force_pull"] is False

    @mock.patch("sparkrun.containers.distribute.ensure_image")
    def test_failed_forced_pull_aborts_distribution(self, mock_ensure):
        """A forced pull that fails must not fall through to shipping the old image."""
        from sparkrun.containers.distribute import distribute_image_from_local

        mock_ensure.return_value = 1
        failed = distribute_image_from_local("img:latest", ["h1", "h2"], force_pull=True)
        assert failed == ["h1", "h2"]


class TestDistributeImageFromHeadForcePull:
    """The head is the pulling side for delegated."""

    @mock.patch("sparkrun.containers.distribute._check_remote_image_identities")
    @mock.patch("sparkrun.orchestration.ssh.run_remote_script_streaming")
    def test_force_pull_renders_forced_ensure_script(self, mock_run, mock_check):
        from sparkrun.containers.distribute import distribute_image_from_head
        from sparkrun.orchestration.ssh import RemoteResult

        mock_run.return_value = RemoteResult(host="head", returncode=0, stdout="", stderr="")

        distribute_image_from_head("img:latest", ["head"], force_pull=True)

        ensure_script = mock_run.call_args_list[0][0][1]
        assert 'FORCE_PULL="1"' in ensure_script

    @mock.patch("sparkrun.containers.distribute._check_remote_image_identities")
    @mock.patch("sparkrun.orchestration.ssh.run_remote_script_streaming")
    def test_force_pull_skips_the_precheck(self, mock_run, mock_check):
        """The pre-check describes the image being *replaced*.

        Honoring it would let "every host already agrees" short-circuit the very
        pull --rebuild was passed to force.
        """
        from sparkrun.containers.distribute import distribute_image_from_head
        from sparkrun.orchestration.ssh import RemoteResult

        mock_run.return_value = RemoteResult(host="head", returncode=0, stdout="", stderr="")
        # Everything already in agreement — the unforced path returns early here.
        mock_check.return_value = {h: ("sha256:same", []) for h in ("head", "w1")}

        failed = distribute_image_from_head("img:latest", ["head", "w1"], force_pull=True)

        assert failed == []
        mock_check.assert_not_called()
        # Pull on head *and* the distribute leg both ran.
        assert mock_run.call_count == 2

    @mock.patch("sparkrun.containers.distribute._check_remote_image_identities")
    @mock.patch("sparkrun.orchestration.ssh.run_remote_script_streaming")
    def test_unforced_still_short_circuits(self, mock_run, mock_check):
        """Guard the other direction: the default path keeps its early return."""
        from sparkrun.containers.distribute import distribute_image_from_head
        from sparkrun.orchestration.ssh import RemoteResult

        mock_run.return_value = RemoteResult(host="head", returncode=0, stdout="", stderr="")
        mock_check.return_value = {h: ("sha256:same", []) for h in ("head", "w1")}

        assert distribute_image_from_head("img:latest", ["head", "w1"]) == []
        mock_run.assert_not_called()

    @mock.patch("sparkrun.containers.distribute._check_remote_image_identities")
    @mock.patch("sparkrun.orchestration.ssh.run_remote_script_streaming")
    def test_default_renders_unforced_ensure_script(self, mock_run, mock_check):
        from sparkrun.containers.distribute import distribute_image_from_head
        from sparkrun.orchestration.ssh import RemoteResult

        mock_run.return_value = RemoteResult(host="head", returncode=0, stdout="", stderr="")
        mock_check.return_value = {}

        distribute_image_from_head("img:latest", ["head"])
        assert 'FORCE_PULL="0"' in mock_run.call_args_list[0][0][1]


# ---------------------------------------------------------------------------
# _distribute_single_image — routed to whichever side pulls
# ---------------------------------------------------------------------------


@pytest.fixture
def image_fns():
    """Patch both distribution primitives and hand back the mocks."""
    with (
        mock.patch("sparkrun.containers.distribute.distribute_image_from_local") as m_local,
        mock.patch("sparkrun.containers.distribute.distribute_image_from_head") as m_head,
    ):
        m_local.return_value = []
        m_head.return_value = []
        yield m_local, m_head


def _single(mode, hosts=("head", "w1"), **kw):
    from sparkrun.orchestration.distribution import _distribute_single_image

    hosts = list(hosts)
    return _distribute_single_image("img:latest", hosts, hosts, mode, None, None, {}, False, False, **kw)


class TestSingleImageForcePullRouting:
    def test_local_mode_forces_the_control_machine(self, image_fns):
        m_local, _ = image_fns
        _single("local", force_pull=True)
        assert m_local.call_args.kwargs["force_pull"] is True

    def test_delegated_mode_forces_the_head(self, image_fns):
        _, m_head = image_fns
        _single("delegated", force_pull=True)
        assert m_head.call_args.kwargs["force_pull"] is True

    def test_push_mode_forces_control_but_not_the_head_leg(self, image_fns):
        """Push exists for heads that cannot reach the registry at all.

        Forcing the head→worker leg would both duplicate the transfer and
        re-pull on a host that just received the fresh image over the wire.
        """
        m_local, m_head = image_fns
        _single("push", hosts=["head", "w1"], force_pull=True)

        assert m_local.call_args.kwargs["force_pull"] is True
        assert "force_pull" not in m_head.call_args.kwargs

    def test_unforced_by_default(self, image_fns):
        m_local, _ = image_fns
        _single("local")
        assert m_local.call_args.kwargs["force_pull"] is False


# ---------------------------------------------------------------------------
# distribute_from_config — the recipe is where the intent arrives
# ---------------------------------------------------------------------------


class TestRebuildDerivedFromRecipe:
    """``--rebuild`` reaches distribution as ``builder_config.rebuild``."""

    @pytest.mark.parametrize(
        "builder,builder_config,expected",
        [
            ("", {"rebuild": True}, True),
            ("docker-pull", {"rebuild": True}, True),
            ("coldsnap", {"rebuild": True}, False),
            ("eugr", {"rebuild": True}, False),
            ("", {"rebuild": False}, False),
            ("", {}, False),
            ("", None, False),
        ],
    )
    def test_derivation(self, builder, builder_config, expected):
        from types import SimpleNamespace

        from sparkrun.orchestration.distribution import _force_registry_pull_for_recipe

        recipe = SimpleNamespace(builder=builder, builder_config=builder_config)
        assert _force_registry_pull_for_recipe(recipe) is expected

    def test_cli_rebuild_flag_lands_on_builder_config(self):
        """The CLI writes the flag the distribution layer reads."""
        from sparkrun.core.recipe import Recipe

        recipe = Recipe(
            {
                "recipe_version": "2",
                "model": "org/model",
                "runtime": "vllm",
                "container": "ghcr.io/example/img:latest",
            }
        )
        # cli/_run.py: `if rebuild is not None: recipe.builder_config["rebuild"] = rebuild`
        recipe.builder_config["rebuild"] = True
        assert recipe.builder_config.get("rebuild") is True
        # And it survives the __getstate__/__setstate__ round trip, which is how
        # the recipe reaches distribution through the api.plan → api.run split.
        import pickle

        assert pickle.loads(pickle.dumps(recipe)).builder_config.get("rebuild") is True

    def test_plain_v2_recipe_declares_no_builder(self):
        """Why this wiring can't live in the builder: the phase never runs."""
        from sparkrun.core.recipe import Recipe

        recipe = Recipe(
            {
                "recipe_version": "2",
                "model": "org/model",
                "runtime": "vllm",
                "container": "ghcr.io/example/img:latest",
            }
        )
        assert recipe.builder == ""
