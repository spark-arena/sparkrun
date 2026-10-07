"""Dense FP8 tuning and installation contracts without a GPU dependency."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from sparkrun.scripts import sglang_tune_fp8 as script
from sparkrun.tuning.sglang import SglangTuner, install_sglang_fp8_configs


@pytest.mark.parametrize("m", [1, 6, 128, 4096])
@pytest.mark.parametrize("block", [(32, 32), (128, 128)])
def test_search_contains_default_and_respects_quantization_scale_boundaries(m, block):
    configs = list(script.search_configs(m, *block))
    assert script.default_config(*block) in configs and len(configs) > 1
    assert all(block[1] % c["BLOCK_SIZE_K"] == 0 for c in configs)
    assert len({tuple(c.items()) for c in configs}) == len(configs)


def make_config(root, kernel, version="3.7.1", device="NVIDIA GB10"):
    directory = script.config_directory(root, kernel, version)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / script.config_filename(1792, 5120, device, 32, 32)
    path.write_text(json.dumps({"1": script.default_config(32, 32)}))
    return path


def test_install_selects_kernel_triton_device_and_preserves_bundled_configs(tmp_path):
    kernel = tmp_path / "runtime" / "fp8_kernel.py"
    kernel.parent.mkdir()
    kernel.write_text("# pinned runtime kernel")
    root = tmp_path / "tuning"
    expected = make_config(root, kernel)
    make_config(root, kernel, version="3.6.0")
    make_config(root, kernel, device="NVIDIA H100")
    destination = kernel.parent / "configs"
    destination.mkdir()
    bundled = destination / "other-model.json"
    bundled.write_text("bundled")
    assert script.install_configs(root, kernel, "3.7.1", "NVIDIA GB10") == 1
    assert (destination / expected.name).read_bytes() == expected.read_bytes()
    assert bundled.read_text() == "bundled"
    kernel.write_text("# different runtime kernel")
    assert script.install_configs(root, kernel, "3.7.1", "NVIDIA GB10") == 0


def test_install_replaces_destination_symlink_without_writing_through(tmp_path):
    kernel = tmp_path / "fp8_kernel.py"
    kernel.write_text("kernel")
    config = make_config(tmp_path / "tuning", kernel)
    destination = tmp_path / "configs" / config.name
    destination.parent.mkdir()
    external = tmp_path / "shared.json"
    external.write_text("original shared artifact")
    destination.symlink_to(external)
    script.install_configs(tmp_path / "tuning", kernel, "3.7.1", "NVIDIA GB10")
    assert not destination.is_symlink()
    assert external.read_text() == "original shared artifact"


@pytest.mark.parametrize("value", [{}, [], {"0": {}}, {"junk": {}}, {"1": 7}, {"1": {}}, {"1": {"BLOCK_SIZE_M": "bad"}}])
def test_install_rejects_empty_or_malformed_configs(tmp_path, value):
    kernel = tmp_path / "fp8_kernel.py"
    kernel.write_text("kernel")
    config = make_config(tmp_path / "tuning", kernel)
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Invalid FP8"):
        script.install_configs(tmp_path / "tuning", kernel, "3.7.1", "NVIDIA GB10")
    assert not (tmp_path / "configs" / config.name).exists()


def test_output_is_atomic_and_separates_measurements_from_runtime_json(tmp_path):
    path = tmp_path / "fp8" / "hash" / "triton_3_7_1" / "shape.json"
    script.write_json(path, {"1": script.default_config(32, 32)})
    script.write_json(path.with_suffix(".measurements"), {"1": {"speedup": 1.2}})
    assert not list(tmp_path.rglob("*.tmp"))
    assert list(tmp_path.rglob("*.json")) == [path]
    assert set(json.loads(path.read_text())) == {"1"}


def test_installation_uses_executor_and_skips_other_executors(tmp_path, monkeypatch):
    root = tmp_path / "tuning"
    monkeypatch.setattr("sparkrun.tuning.sglang.get_sglang_tuning_dir", lambda: root)
    send = Mock(return_value=SimpleNamespace(success=True, stdout="installed", stderr=""))
    monkeypatch.setattr("sparkrun.orchestration.primitives.run_command_on_host", send)
    executor = Mock(executor_name="docker")
    install_sglang_fp8_configs([("h1", "c1")], executor, {})
    send.assert_not_called()
    root.joinpath("fp8").mkdir(parents=True)
    root.joinpath("fp8", "shape.json").write_text("{}")
    executor.executor_name = "local"
    install_sglang_fp8_configs([("h1", "c1")], executor, {})
    send.assert_not_called()
    executor.executor_name = "docker"
    install_sglang_fp8_configs([("h1", "c1"), ("h2", "c2")], executor, {"ssh_user": "tester"})
    assert send.call_count == 2
    assert send.call_args.kwargs["ssh_kwargs"] == {"ssh_user": "tester"}
    assert executor.exec_cmd.call_args.kwargs["env"] == {"PYTHONPATH": ""}
    assert executor.exec_cmd.call_args.kwargs["user"] == "0:0"


def test_runtime_installs_after_recipe_preparation(monkeypatch):
    from sparkrun.runtimes.base import RuntimePlugin
    from sparkrun.runtimes.sglang import SglangRuntime

    calls = []
    monkeypatch.setattr(RuntimePlugin, "_pre_serve", lambda *args: calls.append("recipe"))
    monkeypatch.setattr("sparkrun.tuning.sglang.install_sglang_fp8_configs", lambda *args: calls.append("tuning"))
    runtime = SglangRuntime()
    runtime.executor = Mock()
    runtime._pre_serve([("h1", "c1")], {}, False)
    assert calls == ["recipe", "tuning"]


@pytest.fixture
def recipe(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text("recipe_version: '2'\nmodel: test/model\nruntime: sglang\ncontainer: test:image\ndefaults:\n  tensor_parallel: 4\n")
    return path


@pytest.mark.parametrize(
    "options",
    [
        ["--mode", "fp8"],
        ["--shape", "1792", "5120"],
        ["--mode", "fp8", "--shape", "1792", "5120", "--block-shape", "33", "32"],
        ["--mode", "fp8", "--shape", "1792", "5120", "--tp", "2", "--tp", "4"],
        ["--mode", "fp8", "--shape", "1792", "5120", "--parallel", "2"],
    ],
)
def test_cli_rejects_ambiguous_or_invalid_fp8_requests(recipe, options):
    from sparkrun.cli import main

    result = CliRunner().invoke(main, ["tune", "sglang", str(recipe), "-H", "localhost", "-n", *options])
    assert result.exit_code == 2 and "Error:" in result.output


def test_cli_threads_fp8_shapes_and_uses_recipe_tp_once(recipe, monkeypatch):
    from sparkrun.cli import main

    factory = Mock()
    factory.return_value.run_tuning.return_value = 0
    monkeypatch.setattr("sparkrun.tuning.sglang.SglangTuner", factory)
    result = CliRunner().invoke(
        main,
        [
            "tune",
            "sglang",
            str(recipe),
            "-H",
            "localhost",
            "-n",
            "--mode",
            "fp8",
            "--shape",
            "1792",
            "5120",
            "--shape",
            "5120",
            "576",
            "--block-shape",
            "32",
            "32",
            "--batch-size",
            "6",
        ],
    )
    assert result.exit_code == 0, result.output
    assert factory.call_args.kwargs["shapes"] == ((1792, 5120), (5120, 576))
    assert factory.call_args.kwargs["block_shape"] == (32, 32)
    assert factory.call_args.kwargs["batch_sizes"] == (6,)
    factory.return_value.run_tuning.assert_called_once_with(tp_sizes=(4,), parallel=1)


def test_fp8_does_not_clone_or_skip_for_existing_moe_configs(tmp_path):
    tuner = SglangTuner(host="localhost", image="test", model="test", output_dir=str(tmp_path), mode="fp8", shapes=((1792, 5120),))
    assert tuner.skip_clone and not tuner._pre_check_tp(4, "3.7.1")


def test_tuning_sleeper_overrides_image_boot_entrypoint(tmp_path, monkeypatch):
    scripts = []

    def send(host, script, **kwargs):
        scripts.append(script)
        return SimpleNamespace(success=True, stderr="")

    monkeypatch.setattr("sparkrun.orchestration.primitives.run_script_on_host", send)
    tuner = SglangTuner(host="localhost", image="test", model="test", output_dir=str(tmp_path), dry_run=True)
    assert tuner._launch_container() == 0
    assert "--entrypoint ''" in scripts[-1]


def test_tuning_image_preparation_keeps_provider_alias(monkeypatch):
    from sparkrun.tuning._common import prepare_tuning_image

    pull = Mock(return_value=[])
    resolve = Mock(return_value="oci-relay/pinned:verified")
    monkeypatch.setattr("sparkrun.containers.distribute.distribute_image_from_head", pull)
    monkeypatch.setattr("sparkrun.core.image_distribution.resolve_distributed_image", resolve)
    assert prepare_tuning_image("upstream@sha256:pin", "host", {"ssh_user": "test"}) == "oci-relay/pinned:verified"
    pull.assert_called_once_with("upstream@sha256:pin", ["host"], ssh_user="test")
    assert prepare_tuning_image("source", "host", {}, dry_run=True) == "source"
    assert pull.call_count == 1
    pull.return_value = ["host"]
    with pytest.raises(RuntimeError, match="Failed to prepare tuning image"):
        prepare_tuning_image("source", "host", {})


def test_installation_does_not_initialize_runtime_or_cuda(tmp_path, monkeypatch):
    kernel = tmp_path / "package" / "kernels" / "ops" / "quantization" / "fp8_kernel.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("pinned kernel")
    make_config(tmp_path / "tuning", kernel)
    monkeypatch.setattr(
        script.importlib.util, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(tmp_path / "package")])
    )
    monkeypatch.setattr(script.importlib.metadata, "version", lambda name: "3.7.1")
    monkeypatch.setattr(script.subprocess, "check_output", lambda *args, **kwargs: "NVIDIA GB10\n")
    assert script.install_for_image(tmp_path / "tuning") == 1


def test_existing_experimental_registry_gains_tuning_path_without_reset():
    from sparkrun.core.registry import RegistryEntry, RegistryManager

    entry = RegistryEntry(name="experimental", url="https://github.com/spark-arena/recipe-registry.git", subpath="experimental-recipes")
    custom = RegistryEntry(name="custom", url=entry.url, subpath=entry.subpath, tuning_subpath="my-tuning", mods_subpath="my-mods")
    assert RegistryManager._backfill_default_subpaths([entry, custom])
    assert entry.tuning_subpath == "experimental-tuning"
    assert custom.tuning_subpath == "my-tuning" and custom.mods_subpath == "my-mods"
    assert not RegistryManager._backfill_default_subpaths([entry, custom])
