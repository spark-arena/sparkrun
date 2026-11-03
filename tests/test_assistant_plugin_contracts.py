"""Keep shipped assistant references aligned with the CLI without executing workflows."""

from contextlib import ExitStack
from pathlib import Path
import re
import runpy
import shlex
import json

import click
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = sorted(
    path
    for package in ("sparkrun-cc-plugin", "sparkrun-openclaw-plugin", "sparkrun-omp-plugin")
    for pattern in ("*.md", "commands/*.md", "skills/*/SKILL.md")
    for path in (ROOT / package).glob(pattern)
)


def test_generated_assistant_references_match_source():
    generator = runpy.run_path(str(ROOT / "scripts/sync-assistant-plugins.py"))
    skill_pairs = list(generator["rendered_skills"]())
    skills = {path.parent.name for path in (ROOT / "sparkrun-cc-plugin/skills").glob("*/SKILL.md")}
    packages = {"sparkrun-openclaw-plugin", "sparkrun-omp-plugin"}
    assert {(target.relative_to(ROOT).parts[0], target.parent.name) for target, _ in skill_pairs} == {
        (package, name) for package in packages for name in skills
    }
    for target, expected in skill_pairs:
        assert target.read_text() == expected, f"Regenerate {target} with scripts/sync-assistant-plugins.py"

    command_pairs = list(generator["rendered_commands"]())
    commands = {path.name for path in (ROOT / "sparkrun-cc-plugin/commands").glob("*.md")}
    assert {target.name for target, _ in command_pairs} == commands
    command_dir = ROOT / "sparkrun-omp-plugin/commands"
    for target, expected in command_pairs:
        assert target.parent == command_dir, target
        assert target.read_text() == expected, f"Regenerate {target} with scripts/sync-assistant-plugins.py"
        assert "sparkrun_exec" not in expected, target

    for target, expected in skill_pairs:
        if target.relative_to(ROOT).parts[0] == "sparkrun-omp-plugin":
            assert "`bash`" in expected
            assert "sparkrun_exec" not in expected


def test_skill_names_do_not_shadow_claude_action_commands():
    package = ROOT / "sparkrun-cc-plugin"
    commands = {path.stem for path in (package / "commands").glob("*.md")}
    names = set()
    for path in (package / "skills").glob("*/SKILL.md"):
        metadata = yaml.safe_load(path.read_text().split("---", 2)[1])
        assert metadata["name"] == path.parent.name
        assert metadata["name"] not in commands
        assert metadata["name"] not in names
        names.add(metadata["name"])
    manifest = json.loads((ROOT / "sparkrun-openclaw-plugin/openclaw.plugin.json").read_text())
    assert set(manifest["skills"]) == {f"skills/{name}" for name in names}
    omp_skills = {path.parent.name for path in (ROOT / "sparkrun-omp-plugin/skills").glob("*/SKILL.md")}
    assert omp_skills == names


def test_omp_marketplace_catalog_matches_plugin():
    catalog = json.loads((ROOT / ".omp-plugin/marketplace.json").read_text())
    assert catalog["name"] == "sparkrun"
    assert len(catalog["plugins"]) == 1
    entry = catalog["plugins"][0]
    assert entry["name"] == "sparkrun"
    assert entry["source"] == "./sparkrun-omp-plugin"
    manifest = json.loads((ROOT / "sparkrun-omp-plugin/.claude-plugin/plugin.json").read_text())
    assert entry["name"] == manifest["name"]
    assert entry["version"] == manifest["version"]


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda path: str(path.relative_to(ROOT)))
def test_documented_cli_examples_parse_without_dispatch(path, tmp_path):
    from sparkrun.cli import main

    token_file = tmp_path / "token"
    token_file.write_text("test-token")
    for block in re.findall(r"```bash\n(.*?)```", path.read_text(), re.S):
        for line in block.splitlines():
            if not line.startswith("sparkrun "):
                continue
            example = re.sub(r"<([^>]+)>", lambda match: str(token_file) if match[1] == "token_file" else "example", line)
            args = shlex.split(example, comments=True)[1:]
            if args == ["--version"]:
                continue  # Eager version rendering has no command syntax to check.
            command = main
            name = "sparkrun"
            parent = None
            with ExitStack() as stack:
                while True:
                    try:
                        ctx = stack.enter_context(command.make_context(name, args, parent=parent))
                        if not isinstance(command, click.Group):
                            break
                        remaining = [*ctx._protected_args, *ctx.args]
                        if not remaining:
                            break
                        name, command, args = command.resolve_command(ctx, remaining)
                        assert command is not None, line
                        parent = ctx
                    except click.ClickException as error:
                        pytest.fail(f"{path.relative_to(ROOT)}: {line}\n{error.format_message()}")
            # Deliberately never call invoke(): install/stop/cleanup examples only parse.
