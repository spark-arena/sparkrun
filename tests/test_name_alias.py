from click.testing import CliRunner
from sparkrun.cli import main
from unittest.mock import MagicMock


def test_run_with_name_override(monkeypatch):
    runner = CliRunner()

    # Mock launch_inference in its original module, since _run.py imports it locally
    mock_launch = MagicMock()
    mock_result = MagicMock()
    mock_result.cluster_id = "test-cluster-id"
    mock_result.serve_command = "vllm serve"
    mock_result.runtime_info = {}
    mock_result.rc = 0
    mock_result.head_host = "localhost"
    mock_result.host_list = ["localhost"]
    mock_launch.return_value = mock_result
    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", mock_launch)

    monkeypatch.setattr("sparkrun.core.launcher.post_launch_lifecycle", MagicMock())
    monkeypatch.setattr("sparkrun.cli._run._resolve_hosts_or_exit", lambda *args, **kwargs: (["localhost"], None))
    mock_recipe = MagicMock()
    mock_recipe.runtime = "vllm"
    mock_recipe.model = "test-model"
    mock_recipe.validate.return_value = []
    mock_recipe.mode = "solo"
    mock_recipe.build_config_chain.return_value = {"port": 8000}
    monkeypatch.setattr("sparkrun.cli._run._load_recipe", lambda *args, **kwargs: (mock_recipe, "path", None))

    mock_ret = MagicMock()
    mock_ret.topology = None
    mock_ret.resolve_transfer_config.return_value = (None, None, None, None)
    monkeypatch.setattr("sparkrun.cli._run.resolve_cluster_config", lambda *args, **kwargs: mock_ret)

    mock_runtime = MagicMock()
    mock_runtime.runtime_name = "vllm"
    mock_runtime.resolve_container.return_value = "img:latest"
    mock_runtime.validate_recipe.return_value = []
    monkeypatch.setattr("sparkrun.core.bootstrap.get_runtime", lambda *args, **kwargs: mock_runtime)
    monkeypatch.setattr("sparkrun.cli._run.validate_and_prepare_hosts", lambda *args, **kwargs: (["localhost"], True))
    monkeypatch.setattr("sparkrun.cli._run._display_vram_estimate", lambda *args, **kwargs: None)

    result = runner.invoke(main, ["run", "test-recipe", "--container-name", "custom-cluster-id", "--solo"])

    assert result.exit_code == 0
    args, kwargs = mock_launch.call_args
    assert kwargs.get("cluster_id_override") == "custom-cluster-id"
