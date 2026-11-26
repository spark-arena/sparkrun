"""Offline documentation checks collected by the normal Python CI suite."""

from __future__ import annotations

import ast
from collections import Counter
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
from textwrap import dedent
from types import ModuleType
from urllib.parse import unquote, urlsplit

import pytest
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = {"CHANGELOG.md", "docs/RELEASE_NOTES_0.3.0.md"}


def _documents(root=ROOT):
    # Bounded source paths work in checkouts and archives without borrowing a
    # parent repository's index or scanning environments/builds/ignored reports.
    patterns = (
        "*.md",
        "docs/**/*.md",
        ".github/ISSUE_TEMPLATE/*.md",
        "sparkrun-cc-plugin/*.md",
        "sparkrun-cc-plugin/commands/*.md",
        "sparkrun-cc-plugin/skills/*/*.md",
        "sparkrun-openclaw-plugin/*.md",
        "sparkrun-openclaw-plugin/skills/*/*.md",
        "sparkrun-omp-plugin/*.md",
        "sparkrun-omp-plugin/commands/*.md",
        "sparkrun-omp-plugin/skills/*/*.md",
        "src/sparkrun/plugins/*/README.md",
        "src/sparkrun/orchestration/executors/seccomp/README.md",
        "tests/fixtures/application_profiles/*.md",
    )
    return sorted({path for pattern in patterns for path in root.glob(pattern) if path.is_file()})


def _without_fences(text):
    return re.sub(r"^[ \t]*```[^\n]*\n.*?^[ \t]*```[ \t]*$", "", text, flags=re.M | re.S)


def _anchors(path):
    counts = Counter()
    anchors = set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*$", _without_fences(path.read_text()), re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = counts[slug]
        anchors.add(slug if count == 0 else f"{slug}-{count}")
        counts[slug] += 1
    return anchors


@pytest.mark.parametrize("path", _documents(), ids=lambda path: str(path.relative_to(ROOT)))
def test_local_document_links(path):
    text = _without_fences(path.read_text())
    for match in re.finditer(r'\[[^\]\n]*\]\((<[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\)', text):
        target = match[1].strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or target.startswith("//"):
            continue
        destination = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
        assert destination.exists(), f"{path.relative_to(ROOT)}: missing {target}"
        if parsed.fragment and destination.suffix == ".md":
            assert unquote(parsed.fragment) in _anchors(destination), f"{path.relative_to(ROOT)}: missing anchor {target}"


def test_current_documented_python_imports():
    checked = set()
    for path in _documents():
        if str(path.relative_to(ROOT)) in HISTORICAL:
            continue
        for block in re.finditer(r"^[ \t]*```python\n(.*?)^[ \t]*```", path.read_text(), re.M | re.S):
            # Syntax is checked, but examples are not executed against hosts or services.
            parsed = ast.parse(dedent(block[1]), filename=str(path))
            for node in ast.walk(parsed):
                if (
                    not isinstance(node, ast.ImportFrom)
                    or not node.module
                    or not (node.module == "sparkrun" or node.module.startswith("sparkrun."))
                ):
                    continue
                for name in node.names:
                    key = (node.module, name.name)
                    if key in checked or name.name == "*":
                        continue
                    module = importlib.import_module(node.module)
                    if not hasattr(module, name.name):
                        importlib.import_module(node.module + "." + name.name)
                    checked.add(key)
    assert checked, "No current Sparkrun imports found in the documentation"


def _copyable_plugin_examples():
    path = ROOT / "sparkrun-cc-plugin/skills/plugins/SKILL.md"
    examples = [dedent(match[1]) for match in re.finditer(r"^```python\n(.*?)^```", path.read_text(), re.M | re.S)]
    assert examples, "The plugin-authoring skill must contain Python examples"
    assert all("# pyright-checked copyable" in example for example in examples), (
        "Every Python example must be explicitly copyable and checked, or marked as pseudocode"
    )
    return examples


def _copyable_plugin_module():
    examples = [example for example in _copyable_plugin_examples() if "# pyright-checked copyable example" in example]
    assert len(examples) == 1, "Mark one complete plugin module for the isolated handler test"
    return examples[0]


def test_copyable_plugin_example_typechecks_with_pinned_pyright(tmp_path):
    examples = _copyable_plugin_examples()
    sources = []
    for index, example in enumerate(examples):
        source = tmp_path / f"plugin_example_{index}.py"
        source.write_text(example)
        sources.append(source)
    config = tmp_path / "pyrightconfig.json"
    settings = {
        "include": [source.name for source in sources],
        "extraPaths": [str(ROOT / "src")],
        "pythonVersion": "3.12",
        "typeCheckingMode": "basic",
    }
    if (ROOT / ".venv").is_dir():
        # A dev checkout resolves third-party imports from its venv; CI installs
        # the same pinned dev extras into the interpreter that runs the tests.
        settings |= {"venvPath": str(ROOT), "venv": ".venv"}
    config.write_text(json.dumps(settings))
    pyright = [sys.executable, "-m", "pyright"]
    version = subprocess.run([*pyright, "--version"], cwd=ROOT, capture_output=True, text=True, check=True)
    assert "1.1.414" in version.stdout, version.stdout
    result = subprocess.run([*pyright, "--project", str(config), "--outputjson"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)["summary"]
    assert summary["errorCount"] == 0, result.stdout
    assert summary["warningCount"] == 0, result.stdout


def test_copyable_plugin_example_registers_handler_and_tuple_preparation(monkeypatch):
    from sparkrun.core import recipe_items

    monkeypatch.setattr(recipe_items, "_RECIPE_ITEMS", dict(recipe_items._RECIPE_ITEMS))
    module = ModuleType("documentation_plugin_example")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(_copyable_plugin_module(), "plugin_example.py", "exec"), module.__dict__)
    module.register(None)

    registration = recipe_items.get_recipe_item("snapshot")
    assert registration is not None
    parsed = registration.handler.parse({"size": 8}, None)
    assert registration.handler.validate(parsed, None) == []
    assert registration.handler.export(parsed, None) == {"size": 8}
    assert registration.preparation_steps is not None
    steps = registration.preparation_steps(None)
    assert isinstance(steps, tuple)
    assert len(steps) == 1
    assert steps[0].action(None, {}) is None


def test_plugin_author_extension_import_advice_matches_public_facades():
    from sparkrun import plugins, transports
    from sparkrun.orchestration import telemetry

    guide = (ROOT / "sparkrun-cc-plugin/skills/plugins/SKILL.md").read_text()
    plugin_exports = set(plugins.__all__)
    for name in (
        "RuntimePlugin",
        "BuilderPlugin",
        "Executor",
        "BenchmarkingPlugin",
        "HardwarePlatformPlugin",
        "ExecutionContext",
        "PreparationStep",
        "Recipe",
        "register_recipe_item",
    ):
        assert name in plugin_exports and hasattr(plugins, name)
    assert "Transport" not in plugin_exports and "Transport" in transports.__all__
    assert "TelemetryProvider" not in plugin_exports and "TelemetryProvider" in telemetry.__all__
    assert "Scheduler" not in plugin_exports
    assert "No public base export in 0.4.3" in guide
    assert "must not add unconditional overrides" in guide
    assert "replacing that method with `return True` exposes a gated class" in guide


@pytest.mark.parametrize(
    "command",
    [
        "proxy",
        "proxy start",
        "proxy stop",
        "proxy status",
        "proxy models",
        "proxy sync",
        "proxy load",
        "proxy unload",
        "proxy alias",
        "proxy alias add",
        "proxy alias remove",
        "proxy alias list",
        "proxy ui",
        "proxy admin-token",
        "proxy admin-token get",
        "proxy admin-token set",
        "proxy admin-token clear",
        "benchmark perf",
        "benchmark performance",
        "benchmark tools",
        "benchmark resume",
        "setup features",
        "setup features enable",
    ],
)
def test_documented_commands_have_help(command):
    from sparkrun.cli import main

    result = CliRunner().invoke(main, [*command.split(), "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_document_inventory_without_git_metadata(tmp_path):
    expected = ["README.md", "docs/API.md", "src/sparkrun/plugins/k8s/README.md", "tests/fixtures/application_profiles/README.md"]
    excluded = [
        ".slop/report.md",
        "build/README.md",
        ".venv/README.md",
        "node_modules/README.md",
    ]
    # An enclosing repository must not change the inventory of an archive.
    (tmp_path / ".git").mkdir()
    source = tmp_path / "archive"
    for name in expected + excluded:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Documentation\n")
    assert [path.relative_to(source).as_posix() for path in _documents(source)] == sorted(expected)
