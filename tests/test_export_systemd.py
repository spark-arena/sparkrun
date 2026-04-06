"""Test systemd export correctly outputs YAML."""

from click.testing import CliRunner
from sparkrun.cli import main


def test_export_systemd_preserves_single_quotes(monkeypatch):
    """Test systemd export does not escape single quotes in YAML."""
    runner = CliRunner()

    # Mock detect_remote_sparkrun to avoid SSH
    monkeypatch.setattr(
        "sparkrun.cli._export._detect_remote_sparkrun",
        lambda host, ssh_kwargs, dry_run=False: ("/usr/local/bin/sparkrun", "/home/user"),
    )

    # Use a recipe that contains single quotes in env vars
    result = runner.invoke(main, ["export", "systemd", "@official/qwen3-coder-next-int4-autoround-vllm", "--hosts", "127.0.0.1"])

    assert result.exit_code == 0
    # Look for the env var VLLM_MARLIN_USE_ATOMIC_ADD which should have single quotes
    # Before the fix, it would be VLLM_MARLIN_USE_ATOMIC_ADD: '\''1'\''
    # After the fix, it should be VLLM_MARLIN_USE_ATOMIC_ADD: '1'
    assert "VLLM_MARLIN_USE_ATOMIC_ADD: '1'" in result.output
    assert "VLLM_MARLIN_USE_ATOMIC_ADD: '\\''1'\\''" not in result.output

    # Another test: verify the bash script structure uses << 'SPARKRUN_RECIPE_EOF'
    assert "<< 'SPARKRUN_RECIPE_EOF'" in result.output
