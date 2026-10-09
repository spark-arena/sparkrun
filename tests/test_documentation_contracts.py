"""Offline documentation checks collected by the normal Python CI suite."""

from __future__ import annotations

import ast
from collections import Counter
import importlib
from pathlib import Path
import re
from textwrap import dedent
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
