#!/usr/bin/env python3
"""
Synchronize version numbers across all sparkrun subprojects.

Reads versions.yaml as the single source of truth and updates:
  - pyproject.toml    (Python package)
  - plugin.json       (Claude Code plugin)
  - marketplace.json  (Claude Code marketplace registry)

Usage:
    python scripts/update-versions.py           # sync all versions
    python scripts/update-versions.py --check   # dry-run: exit 1 if out of sync
    python scripts/update-versions.py --verbose # show every file touched
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required.  Install it with:\n  pip install pyyaml\n  uv pip install pyyaml")

logger = logging.getLogger("update-versions")

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent  # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent  # oss-sparkrun/
VERSIONS_YAML = PROJECT_ROOT / "versions.yaml"

# ---------------------------------------------------------------------------
# Per-project update rules
#
# Each entry maps a versions.yaml key to the files that must be updated and
# the strategy used.  Adding a new subproject only requires a new entry here.
# ---------------------------------------------------------------------------

# Strategies:
#   pyproject    – rewrite `version = "X.Y.Z"` in [project] table
#   package      – rewrite `"version": "X.Y.Z"` in JSON (package.json)
#   plugin       – rewrite `"version": "X.Y.Z"` in plugin.json
#   marketplace  – rewrite `"version"` for a matching plugin entry inside
#                  a marketplace.json plugins array (matched by source path
#                  containing the project directory name)
#   init_py      – rewrite `__version__ = "X.Y.Z"` in a Python file

# (strategy, path relative to PROJECT_ROOT, *optional extra args)
UpdateRule = tuple  # (strategy, rel_path[, *extras])

PROJECT_RULES: dict[str, list[UpdateRule]] = {
    "sparkrun": [
        ("pyproject", Path("pyproject.toml")),
    ],
    "sparkrun-cc-plugin": [
        ("plugin", Path("sparkrun-cc-plugin/.claude-plugin/plugin.json")),
        ("marketplace", Path(".claude-plugin/marketplace.json"), "sparkrun-cc-plugin"),
    ],
    "sparkrun-openclaw-plugin": [
        ("package", Path("sparkrun-openclaw-plugin/package.json")),
        ("plugin", Path("sparkrun-openclaw-plugin/openclaw.plugin.json")),
    ],
}

# ---------------------------------------------------------------------------
# Updaters — each returns (changed: bool, old_version: str | None)
# ---------------------------------------------------------------------------

# Matches: version = "1.2.3"  (with optional surrounding whitespace)
_RE_PYPROJECT_VERSION = re.compile(r'^(\s*version\s*=\s*")[^"]*(")', re.MULTILINE)

# Matches: __version__ = "1.2.3"  (single or double quotes)
_RE_INIT_VERSION = re.compile(r"""^(\s*__version__\s*=\s*['"])[^'"]*(['"])""", re.MULTILINE)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def update_pyproject(path: Path, version: str, dry_run: bool) -> tuple[bool, str | None]:
    """Update version = "..." in a pyproject.toml."""
    text = _read_text(path)
    m = _RE_PYPROJECT_VERSION.search(text)
    if not m:
        logger.warning("No version field found in %s", path)
        return False, None

    old = text[m.start(1) + len(m.group(1)) : m.end(2) - len(m.group(2))]
    if old == version:
        return False, old

    new_text = _RE_PYPROJECT_VERSION.sub(rf"\g<1>{version}\2", text, count=1)
    if not dry_run:
        _write_text(path, new_text)
    return True, old


def update_init_py(path: Path, version: str, dry_run: bool) -> tuple[bool, str | None]:
    """Update __version__ = '...' in a Python __init__.py."""
    text = _read_text(path)
    m = _RE_INIT_VERSION.search(text)
    if not m:
        logger.warning("No __version__ found in %s", path)
        return False, None

    old = text[m.start(1) + len(m.group(1)) : m.end(2) - len(m.group(2))]
    if old == version:
        return False, old

    new_text = _RE_INIT_VERSION.sub(rf"\g<1>{version}\2", text, count=1)
    if not dry_run:
        _write_text(path, new_text)
    return True, old


def update_json_version(path: Path, version: str, dry_run: bool) -> tuple[bool, str | None]:
    """Update "version" in a JSON file (package.json / plugin.json)."""
    text = _read_text(path)
    data = json.loads(text)
    old = data.get("version")
    if old == version:
        return False, old

    data["version"] = version

    if not dry_run:
        # Preserve 2-space indent and trailing newline (npm/node convention)
        _write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return True, old


def update_marketplace(
    path: Path,
    version: str,
    dry_run: bool,
    project_dir: str,
) -> tuple[bool, str | None]:
    """Update a plugin's version inside a marketplace.json plugins array.

    Finds the plugin whose ``source`` field contains *project_dir* and
    updates its ``version``.
    """
    text = _read_text(path)
    data = json.loads(text)

    plugins = data.get("plugins")
    if not isinstance(plugins, list):
        logger.warning("No 'plugins' array in %s", path)
        return False, None

    target = None
    for entry in plugins:
        source = entry.get("source", "")
        if project_dir in source:
            target = entry
            break

    if target is None:
        logger.warning(
            "No plugin with source containing '%s' in %s",
            project_dir,
            path,
        )
        return False, None

    old = target.get("version")
    if old == version:
        return False, old

    target["version"] = version

    if not dry_run:
        _write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return True, old


STRATEGY_MAP = {
    "pyproject": update_pyproject,
    "init_py": update_init_py,
    "package": update_json_version,
    "plugin": update_json_version,
    "marketplace": update_marketplace,
}

# ---------------------------------------------------------------------------
# Semver validation
# ---------------------------------------------------------------------------

_RE_SEMVER = re.compile(
    r"^\d+\.\d+\.\d+"  # major.minor.patch
    r"(?:-[0-9A-Za-z.-]+)?"  # optional pre-release
    r"(?:\+[0-9A-Za-z.-]+)?$"  # optional build metadata
)


def validate_version(version: str) -> bool:
    return bool(_RE_SEMVER.match(version))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_versions() -> dict[str, str]:
    """Load and validate versions.yaml."""
    if not VERSIONS_YAML.exists():
        sys.exit(f"versions.yaml not found at {VERSIONS_YAML}")

    with open(VERSIONS_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        sys.exit(f"versions.yaml must be a YAML mapping, got {type(data).__name__}")

    versions: dict[str, str] = {}
    for key, val in data.items():
        ver = str(val)
        if not validate_version(ver):
            sys.exit(f"Invalid semver for '{key}': {ver}")
        versions[key] = ver

    return versions


def run(*, check: bool = False, verbose: bool = False) -> int:
    """
    Synchronize versions.  Returns 0 on success, 1 if --check finds drift.
    """
    versions = load_versions()
    changes: list[str] = []
    errors: list[str] = []

    for project, target_version in sorted(versions.items()):
        rules = PROJECT_RULES.get(project)
        if rules is None:
            errors.append(f"'{project}' is in versions.yaml but has no rules in update-versions.py — add an entry to PROJECT_RULES")
            continue

        for rule in rules:
            strategy, rel_path = rule[0], rule[1]
            extras = rule[2:]  # additional args for strategies that need them
            abs_path = PROJECT_ROOT / rel_path
            if not abs_path.exists():
                errors.append(f"File not found: {abs_path}")
                continue

            updater = STRATEGY_MAP[strategy]
            changed, old_version = updater(abs_path, target_version, check, *extras)

            if changed:
                msg = f"  {rel_path}: {old_version} -> {target_version}"
                changes.append(msg)
                if verbose or check:
                    logger.info(msg)
            elif verbose:
                logger.info("  %s: already %s", rel_path, target_version)

    # Report
    if errors:
        logger.error("Errors encountered:")
        for e in errors:
            logger.error("  %s", e)

    if check:
        if changes:
            logger.error("Version drift detected (%d file(s) out of sync):", len(changes))
            for c in changes:
                logger.error(c)
            return 1
        else:
            logger.info("All versions in sync.")
            return 0

    if changes:
        logger.info("Updated %d file(s):", len(changes))
        for c in changes:
            logger.info(c)
    else:
        logger.info("All versions already in sync — nothing to update.")

    if errors:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize subproject versions from versions.yaml",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Dry-run: report drift and exit 1 if any file is out of sync",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show every file inspected, not just changes",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    sys.exit(run(check=args.check, verbose=args.verbose))


if __name__ == "__main__":
    main()
