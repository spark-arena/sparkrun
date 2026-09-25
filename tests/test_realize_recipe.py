"""``api.realize_recipe`` / ``sparkrun export recipe --realize``."""

from __future__ import annotations

from dataclasses import replace

import pytest
import yaml
from click.testing import CliRunner

from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.recipe import Recipe

_RECIPE = {
    "model": "Qwen/Qwen3-1.7B",
    "runtime": "vllm-distributed",
    "container": "vllm/vllm-openai:latest",
    "defaults": {"port": 8000, "tensor_parallel": 2, "max_num_seqs": 8, "speculator": "mtp"},
    "env": {"BASE": "1"},
    "overrides": [
        {"when": {"nodes": 2}, "defaults": {"max_num_seqs": 4}, "env": {"MULTI": "1"}},
        {"when": {"capability": {"ne": "unified-memory"}}, "defaults": {"gpu_memory_utilization": 0.94}},
        {"when": {"config": {"speculator": "none"}}, "defaults": {"max_num_seqs": 16}},
    ],
}
_HOSTS = ("s1", "s2")


def _detected_gb10() -> HostHardware:
    return HostHardware(
        accelerators=[AcceleratorSpec(vendor="nvidia", model="gb10", capabilities=frozenset({"cuda"}))],
        source="detected",
    )


@pytest.fixture
def planned(monkeypatch):
    """Record the options realize plans with; plan them dry so no host is probed."""
    import sparkrun.api._run as run_module

    real_plan = run_module.plan
    seen = []

    def fake_plan(options, *, sctx=None):
        seen.append(options)
        return real_plan(replace(options, dry_run=True), sctx=sctx)

    monkeypatch.setattr(run_module, "plan", fake_plan)
    return seen


def _options(tmp_path, *, overrides=None, data=None):
    import sparkrun.api as api
    from sparkrun.core.cluster_manager import ClusterDefinition

    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(data or _RECIPE))
    recipe = Recipe.load(str(path), resolve=False)
    cluster = ClusterDefinition(name="c", hosts=list(_HOSTS), hosts_hardware={h: _detected_gb10() for h in _HOSTS})
    return api.RunOptions(recipe=recipe, cluster=cluster, hosts=_HOSTS, overrides=overrides or {})


def test_realize_bakes_matched_layers_and_drops_the_rest(tmp_path, v, planned):
    import sparkrun.api as api

    realized = api.realize_recipe(_options(tmp_path))
    data = realized.recipe
    assert "overrides" not in data
    assert data["defaults"]["max_num_seqs"] == 4  # nodes=2 layer matched
    assert "gpu_memory_utilization" not in data["defaults"]  # discrete-card layer skipped on GB10
    assert data["env"] == {"BASE": "1", "MULTI": "1"}
    assert data["metadata"]["realized_for"] == {"nodes": 2, "platform": "DGX Spark", "accelerator": "nvidia/gb10"}


def test_realize_plans_for_real_so_hardware_is_probed(tmp_path, v, planned):
    import sparkrun.api as api

    options = replace(_options(tmp_path), dry_run=True)
    api.realize_recipe(options)
    assert planned[0].dry_run is False  # a dry-run plan skips the probe


def test_selector_only_keys_are_dropped(tmp_path, v, planned):
    import sparkrun.api as api

    realized = api.realize_recipe(_options(tmp_path))
    assert "speculator" not in realized.recipe["defaults"]
    assert realized.dropped_keys == ("speculator",)


def test_a_selector_key_given_on_the_cli_is_kept(tmp_path, v, planned):
    import sparkrun.api as api

    realized = api.realize_recipe(_options(tmp_path, overrides={"speculator": "mtp"}))
    assert realized.recipe["defaults"]["speculator"] == "mtp"
    assert realized.dropped_keys == ()


def test_cli_overrides_are_baked_in(tmp_path, v, planned):
    import sparkrun.api as api

    realized = api.realize_recipe(_options(tmp_path, overrides={"max_model_len": 4096}))
    assert realized.recipe["defaults"]["max_model_len"] == 4096


def test_realized_recipe_loads_with_the_same_effective_config(tmp_path, v, planned):
    import sparkrun.api as api

    realized = api.realize_recipe(_options(tmp_path))
    out = tmp_path / "realized.yaml"
    out.write_text(yaml.safe_dump(realized.recipe))
    reloaded = Recipe.load(str(out))
    assert reloaded.overrides == [] or not reloaded.overrides
    expected = {k: v for k, v in realized.plan.recipe.defaults.items() if k not in realized.dropped_keys}
    assert reloaded.defaults == expected
    assert reloaded.env == realized.plan.recipe.env


def test_cli_rejects_launch_options_without_realize(tmp_path):
    from sparkrun.cli import main

    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump(_RECIPE))
    result = CliRunner().invoke(main, ["export", "recipe", str(path), "--tp", "2"])
    assert result.exit_code != 0
    assert "only apply with --realize" in result.output

    result = CliRunner().invoke(main, ["export", "recipe", str(path), "--realize", "--keep-include"])
    assert result.exit_code != 0
    assert "--keep-include" in result.output


def test_cli_realize_writes_plain_yaml_to_stdout(tmp_path, v, planned):
    from sparkrun.cli import main

    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump(_RECIPE))
    result = CliRunner().invoke(main, ["export", "recipe", str(path), "--realize", "--hosts", ",".join(_HOSTS)])
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(result.stdout)
    assert "overrides" not in data
    assert data["metadata"]["realized_for"]["nodes"] == 2
    assert "Realized for" in result.stderr
