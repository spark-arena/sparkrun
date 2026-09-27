"""Offline launches: mode resolution, the cluster default, the preflight, and the process-wide half."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from sparkrun.core.cluster_manager import ClusterDefinition, ClusterError, ClusterManager
from sparkrun.core.offline import OfflineGap, OfflineUnavailableError, copy_sources, find_offline_gaps, resolve_offline

# --- resolution ------------------------------------------------------------------------------


def test_cli_beats_cluster_beats_default():
    offline_cluster = SimpleNamespace(name="lab", offline=True)
    assert resolve_offline(None, None).offline is False
    assert resolve_offline(None, None).source == "default"
    assert resolve_offline(None, offline_cluster).describe() == "yes (cluster 'lab')"
    assert resolve_offline(False, offline_cluster).describe() == "no (cli)"  # --online overrides the cluster
    assert resolve_offline(True, SimpleNamespace(name="lab", offline=None)).offline is True


# --- cluster field ---------------------------------------------------------------------------


def test_cluster_offline_round_trips_and_is_omitted_when_unset(tmp_path):
    mgr = ClusterManager(tmp_path)
    mgr.create("plain", ["h1"])
    mgr.create("air", ["h1"], offline=True)
    assert mgr.get("plain").offline is None
    assert "offline" not in mgr._cluster_path("plain").read_text()
    assert mgr.get("air").offline is True
    assert ClusterDefinition(name="x", hosts=["h"], offline=True).to_dict()["offline"] is True

    mgr.update("air", offline=None)  # `cluster update --online`
    assert mgr.get("air").offline is None
    assert "offline" not in mgr._cluster_path("air").read_text()


def test_a_non_boolean_offline_is_refused_not_read_as_online(tmp_path):
    mgr = ClusterManager(tmp_path)
    mgr.clusters_dir.mkdir(parents=True, exist_ok=True)
    mgr._cluster_path("bad").write_text(yaml.safe_dump({"name": "bad", "hosts": ["h1"], "offline": "yes"}))
    with pytest.raises(ClusterError, match="offline must be true or false"):
        mgr.get("bad")


def test_cluster_cli_create_and_update(tmp_path, monkeypatch):
    from sparkrun.cli import main

    import sparkrun.cli._cluster as cluster_cli

    mgr = ClusterManager(tmp_path)
    monkeypatch.setattr(cluster_cli, "_get_cluster_manager", lambda **kw: mgr)
    runner = CliRunner()
    assert runner.invoke(main, ["cluster", "create", "lab", "--hosts", "h1", "--offline"]).exit_code == 0
    assert mgr.get("lab").offline is True
    result = runner.invoke(main, ["cluster", "update", "lab", "--online"])  # the only change: must not be "nothing to update"
    assert result.exit_code == 0, result.output
    assert mgr.get("lab").offline is None
    assert runner.invoke(main, ["cluster", "update", "lab", "--offline"]).exit_code == 0
    assert mgr.get("lab").offline is True
    assert "Offline:" in runner.invoke(main, ["cluster", "show", "lab"]).output


# --- copy sources and gaps -------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,auto,hetero,expected",
    [
        ("local", False, False, ("control",)),
        ("push", False, False, ("control",)),
        ("delegated", False, False, ("head",)),
        ("delegated", True, False, ("head", "control")),
        ("delegated", True, True, ("control",)),  # heterogeneous delegated is a per-node pull, never via the head
        ("delegated", False, True, ()),
        ("pull", False, False, ()),
    ],
)
def test_copy_sources_mirror_distribution(mode, auto, hetero, expected):
    assert copy_sources(mode, auto_delegated=auto, heterogeneous=hetero) == expected


def _gaps(
    *,
    mode="local",
    auto=False,
    hetero=False,
    images=None,
    hosts_with_image=(),
    control_image=False,
    model="org/m",
    model_hosts=("h1", "h2"),
    hosts_with_model=(),
    control_model=False,
):
    return find_offline_gaps(
        images=images if images is not None else {"img:1": ["h1", "h2"]},
        model=model,
        model_hosts=list(model_hosts),
        head="h1",
        transfer_mode=mode,
        auto_delegated=auto,
        heterogeneous=hetero,
        image_present_on=lambda image, hosts: set(hosts_with_image) & set(hosts),
        image_on_control=lambda image: control_image,
        model_present_on=lambda hosts: set(hosts_with_model) & set(hosts),
        model_on_control=lambda: control_model,
    )


def test_everything_resident_is_no_gap():
    assert _gaps(hosts_with_image=("h1", "h2"), hosts_with_model=("h1", "h2")) == []


def test_a_control_machine_copy_covers_missing_hosts_in_local_mode():
    assert _gaps(control_image=True, control_model=True) == []


def test_nothing_anywhere_is_a_gap_per_asset():
    gaps = _gaps()
    assert [(g.kind, g.hosts) for g in gaps] == [("image", ("h1", "h2")), ("model", ("h1", "h2"))]


def test_delegated_copies_from_the_head_only():
    assert _gaps(mode="delegated", hosts_with_image=("h1",), hosts_with_model=("h1",)) == []
    gaps = _gaps(mode="delegated", control_image=True, control_model=True)  # explicit delegated never pushes from here
    assert {g.kind for g in gaps} == {"image", "model"}
    assert _gaps(mode="delegated", auto=True, control_image=True, control_model=True) == []


def test_pull_mode_needs_every_host_to_have_it():
    gaps = _gaps(mode="pull", hosts_with_image=("h1",), control_image=True, hosts_with_model=("h1", "h2"))
    assert gaps == [OfflineGap("image", "img:1", ("h2",), "none: this transfer mode copies nothing")]


def test_a_shared_model_cache_only_needs_the_head():
    assert _gaps(hosts_with_image=("h1", "h2"), model_hosts=("h1",), hosts_with_model=("h1",)) == []


def test_the_error_lists_every_gap_with_the_way_out():
    message = str(OfflineUnavailableError(_gaps()))
    assert "image img:1: missing on h1, h2" in message
    assert "model org/m: missing on h1, h2" in message
    assert "--online" in message


# --- preflight in the launcher ---------------------------------------------------------------


def _recipe(**extra):
    from sparkrun.core.recipe import Recipe

    return Recipe.from_dict({"model": "org/m", "runtime": "vllm-distributed", "container": "ghcr.io/org/img:1", **extra})


def _run_preflight(monkeypatch, recipe, *, images_present=(), model_present=(), control_image=False, mode="local", rebuild=False):
    import sparkrun.containers.distribute as distribute
    import sparkrun.containers.registry as registry
    import sparkrun.models.download as download
    import sparkrun.models.revision as revision
    from sparkrun.core import launcher
    from sparkrun.core.bootstrap import get_runtime

    seen = {}

    def identities(image, hosts, **kw):
        seen.setdefault("images", []).append(image)
        return {h: ("id", []) for h in hosts if h in images_present}

    monkeypatch.setattr(distribute, "_check_remote_image_identities", identities)
    monkeypatch.setattr(registry, "image_exists_locally", lambda image: control_image)
    monkeypatch.setattr(revision, "hosts_with_snapshot", lambda model, rev, hosts, **kw: set(model_present) & set(hosts))
    monkeypatch.setattr(download, "is_model_cached", lambda *a, **k: False)
    if rebuild:
        recipe.builder_config = {"rebuild": True}
    launcher._offline_preflight(
        recipe,
        get_runtime(recipe.runtime),
        ["h1", "h2"],
        cluster=None,
        launch_hardware=None,
        config=None,
        v=None,
        ssh_kwargs={},
        transfer_result=SimpleNamespace(mode=mode, auto_delegated=False),
        cache_dir=None,
        local_cache_dir=None,
        skip_model=False,
        skip_images=False,
        shared_model_cache=False,
    )
    return seen


def test_preflight_passes_when_everything_is_resident(monkeypatch, v):
    _run_preflight(monkeypatch, _recipe(), images_present=("h1", "h2"), model_present=("h1", "h2"))


def test_preflight_raises_before_side_effects(monkeypatch, v):
    with pytest.raises(OfflineUnavailableError) as info:
        _run_preflight(monkeypatch, _recipe(), images_present=("h1",), model_present=("h1", "h2"))
    assert [(g.kind, g.hosts) for g in info.value.gaps] == [("image", ("h2",))]


def test_preflight_checks_the_image_the_builder_would_pull(monkeypatch, v):
    recipe = _recipe(container="eugr/spark-vllm:latest", builder="eugr")
    seen = _run_preflight(monkeypatch, recipe, images_present=("h1", "h2"), model_present=("h1", "h2"))
    assert seen["images"] == ["ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest"]


def test_rebuild_is_refused_offline(monkeypatch, v):
    with pytest.raises(OfflineUnavailableError, match="--rebuild"):
        _run_preflight(monkeypatch, _recipe(), rebuild=True)


# --- the rest of the launch path -------------------------------------------------------------


def test_url_recipe_offline_uses_the_cache_without_fetching(tmp_path, monkeypatch):
    from sparkrun.core import recipe as recipe_mod

    url = "https://spark-arena.com/api/recipes/abc"
    cache = tmp_path / "cached.yaml"
    monkeypatch.setattr(recipe_mod, "_url_cache_path", lambda u: cache)
    monkeypatch.setattr("urllib.request.OpenerDirector.open", lambda *a, **k: pytest.fail("no fetch offline"))
    with pytest.raises(recipe_mod.RecipeError, match="not cached"):
        recipe_mod.fetch_and_cache_recipe(url, offline=True)
    cache.write_text("model: org/m\n")
    assert recipe_mod.fetch_and_cache_recipe(url, offline=True) == cache


def test_enter_offline_mode_applies_the_process_wide_half(monkeypatch):
    import sparkrun.models.hub as hub
    from sparkrun.cli._common import _enter_offline_mode
    from sparkrun.core.application_profile import env_name

    calls = []
    monkeypatch.setattr(hub, "disable_hub_metadata", lambda: calls.append("hub"))
    for name in (env_name("NO_TELEMETRY"), "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    import os

    assert _enter_offline_mode(None, SimpleNamespace(name="c", offline=None)) is False
    assert calls == []
    assert _enter_offline_mode(True, None) is True
    assert calls == ["hub"] and os.environ[env_name("NO_TELEMETRY")] == "1"
    assert "HF_HUB_OFFLINE" not in os.environ  # only benchmark applications get it
    assert _enter_offline_mode(None, SimpleNamespace(name="c", offline=True), benchmark_apps=True) is True
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_job_metadata_records_the_resident_digest(tmp_path):
    from sparkrun.orchestration.job_metadata import load_job_metadata, save_job_metadata

    recipe = _recipe()
    cid, cid2 = "sparkrun_%s_%s" % ("a" * 16, "b" * 12), "sparkrun_%s_%s" % ("c" * 16, "d" * 12)
    save_job_metadata(
        cid, recipe, ["h1"], cache_dir=str(tmp_path), container_image="ghcr.io/org/img:1", container_digest="sha256:" + "a" * 64
    )
    assert load_job_metadata(cid, cache_dir=str(tmp_path))["effective_container_digest"] == "sha256:" + "a" * 64
    save_job_metadata(cid2, recipe, ["h1"], cache_dir=str(tmp_path), container_image="ghcr.io/org/img:1")
    assert "effective_container_digest" not in load_job_metadata(cid2, cache_dir=str(tmp_path))


def test_plan_resolves_offline_from_the_cluster(tmp_path, v):
    import sparkrun.api as api
    from sparkrun.core.recipe import Recipe

    path = tmp_path / "r.yaml"
    path.write_text(
        yaml.safe_dump({"model": "org/m", "runtime": "vllm-distributed", "container": "img:1", "defaults": {"tensor_parallel": 1}})
    )
    cluster = ClusterDefinition(name="air", hosts=["s1"], offline=True)
    options = api.RunOptions(recipe=Recipe.load(str(path), resolve=False), cluster=cluster, hosts=("s1",), dry_run=True)
    assert api.plan(options).offline.describe() == "yes (cluster 'air')"
    from dataclasses import replace

    assert api.plan(replace(options, offline=False)).offline.describe() == "no (cli)"


# --- builders --------------------------------------------------------------------------------


def _eugr_prepare(monkeypatch, container, *, build_args=None, present=False, rebuild=False):
    import sparkrun.containers.registry as registry
    from sparkrun.builders.eugr import EugrBuilder
    from sparkrun.core.recipe import Recipe

    data = {"model": "org/m", "runtime": "vllm", "container": container}
    if build_args:
        data["build_args"] = build_args
    recipe = Recipe.from_dict(data)
    if rebuild:
        recipe.builder_config = {"rebuild": True}
    builder = EugrBuilder()
    monkeypatch.setattr(registry, "image_exists_locally", lambda image: present)
    monkeypatch.setattr(builder, "_force_pull_image", lambda *a, **k: pytest.fail("offline must not pull"))
    monkeypatch.setattr(builder, "ensure_repo", lambda *a, **k: pytest.fail("offline must not clone"))
    return builder.prepare(container, recipe, hosts=["h1"], builder_context={"offline": True})


def test_eugr_offline_leaves_a_pullable_image_to_distribution(monkeypatch):
    assert _eugr_prepare(monkeypatch, "eugr/spark-vllm:latest") == "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest"


def test_eugr_offline_uses_a_present_local_build(monkeypatch):
    assert _eugr_prepare(monkeypatch, "eugr/spark-vllm:latest", build_args=["--use-wheels"], present=True) == "sparkrun-eugr-vllm"


def test_eugr_offline_refuses_to_build(monkeypatch):
    with pytest.raises(RuntimeError, match="offline"):
        _eugr_prepare(monkeypatch, "eugr/spark-vllm:latest", build_args=["--use-wheels"])


def test_eugr_offline_refuses_rebuild(monkeypatch):
    with pytest.raises(RuntimeError, match="--rebuild"):
        _eugr_prepare(monkeypatch, "vllm/vllm-openai:v1", rebuild=True)


def _uv_venv_script(offline: bool) -> str:
    from sparkrun.builders.uv_venv import _provision_script, _resolve_spec
    from sparkrun.core.recipe import Recipe

    recipe = Recipe.from_dict(
        {
            "model": "org/m",
            "runtime": "vllm-distributed",
            "container": "img:1",
            "builder": "uv-venv",
            "builder_config": {"requirements": ["vllm==0.1"], "venv_path": "$HOME/venv"},
        }
    )
    return _provision_script(_resolve_spec(recipe), offline=offline)


def _run_uv_script(tmp_path, script: str, *, with_uv: bool):
    """Run a provisioning script under bash with recording shims for uv / pip / curl."""
    import subprocess

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "calls.log"
    shims = {"curl": "", "pip": "", "python3": ""}
    if with_uv:
        # A fake uv that records its environment's UV_OFFLINE and creates the venv python.
        shims["uv"] = 'if [ "$1" = venv ]; then mkdir -p "$2/bin"; touch "$2/bin/python"; chmod +x "$2/bin/python"; fi\n'
    for name, body in shims.items():
        shim = bin_dir / name
        shim.write_text('#!/bin/bash\necho "%s UV_OFFLINE=${UV_OFFLINE:-} $*" >> "%s"\n%s' % (name, log, body))
        shim.chmod(0o755)
    env = {"HOME": str(tmp_path), "PATH": "%s:/usr/bin:/bin" % bin_dir}
    result = subprocess.run(["bash", "-s"], input=script, text=True, capture_output=True, env=env)
    return result, (log.read_text().splitlines() if log.exists() else [])


def test_uv_venv_offline_installs_from_uvs_cache_only(tmp_path):
    result, calls = _run_uv_script(tmp_path, _uv_venv_script(offline=True), with_uv=True)
    assert result.returncode == 0, result.stderr
    assert calls and all(c.startswith("uv UV_OFFLINE=1 ") for c in calls)  # every uv call offline; no pip, no curl
    assert any(" pip install " in c for c in calls)


def test_uv_venv_offline_never_installs_uv(tmp_path):
    result, calls = _run_uv_script(tmp_path, _uv_venv_script(offline=True), with_uv=False)
    assert result.returncode == 1
    assert "offline launch does not install it" in result.stderr
    assert calls == []  # neither pip nor curl was tried


def test_uv_venv_online_script_is_unchanged_by_the_offline_option():
    online = _uv_venv_script(offline=False)
    assert "UV_OFFLINE" not in online
    assert "curl -LsSf" in online and "pip install -q" in online


def test_uv_venv_prepare_passes_offline_to_the_script(monkeypatch):
    import sparkrun.builders.uv_venv as uv_venv
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.ssh import RemoteResult

    scripts = []

    def fake_fanout(hosts, script, **kw):
        scripts.append(script)
        return [RemoteResult(host=h, returncode=0, stdout="uv-venv: provisioned", stderr="") for h in hosts]

    monkeypatch.setattr(uv_venv, "run_remote_scripts_parallel", fake_fanout)
    recipe = Recipe.from_dict(
        {"model": "org/m", "runtime": "vllm-distributed", "container": "img:1", "builder_config": {"requirements": ["vllm==0.1"]}}
    )
    assert uv_venv.UvVenvBuilder().prepare("img:1", recipe, hosts=["h1"], builder_context={"offline": True}) == "img:1"
    assert "export UV_OFFLINE=1" in scripts[0]


def test_launch_passes_offline_to_distribution_and_the_builder():
    """Source-level guard: both hand-offs exist in launch_inference."""
    import inspect

    from sparkrun.core import launcher

    source = inspect.getsource(launcher.launch_inference)
    assert 'builder_context={"offline": True} if offline else None' in source
    assert "offline=offline," in source
