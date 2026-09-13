"""Host-side contracts for the commit-pinned SparkRoute vendor snapshot."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "vendor" / "sparkroute.lock"
PROVENANCE = ROOT / "src" / "sparkrun" / "plugins" / "sparkroute" / "VENDORED.toml"


@pytest.fixture
def vendor_module():
    path = ROOT / "scripts" / "vendor-sparkroute.py"
    spec = importlib.util.spec_from_file_location("_sparkrun_vendor_sparkroute_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def _manifest_text(*, duplicate_version: bool = False) -> str:
    version = 'version = "0.1.0"\n' if duplicate_version else ""
    return (
        "schema = 1\n"
        'name = "sparkroute"\n' + version + 'module = "sparkrun.plugins.sparkroute"\n'
        'feature = "gateway.sparkroute"\n'
        'repository = "git@github.com:sparksq/sparkrun-sparkroute-plugin.git"\n'
        'sparkrun = ">=0.3.8,<0.4"\n'
        'source = "src/sparkrun/plugins/sparkroute"\n'
        'tests = "tests/test_sparkroute_*.py"\n'
    )


def test_importer_uses_the_generated_project_version(vendor_module, monkeypatch):
    def show(_repository, *arguments, **_kwargs):
        return '[project]\nversion = "0.1.0"\n' if arguments[-1].endswith(":pyproject.toml") else _manifest_text()

    monkeypatch.setattr(vendor_module, "_run_git", show)
    manifest = vendor_module._manifest(Path("unused"), "a" * 40)

    assert manifest.version == "0.1.0"


def test_importer_rejects_a_duplicate_manifest_version(vendor_module, monkeypatch):
    def show(_repository, *arguments, **_kwargs):
        return '[project]\nversion = "0.1.0"\n' if arguments[-1].endswith(":pyproject.toml") else _manifest_text(duplicate_version=True)

    monkeypatch.setattr(vendor_module, "_run_git", show)

    with pytest.raises(vendor_module.VendorError, match="must not duplicate"):
        vendor_module._manifest(Path("unused"), "a" * 40)


def test_sparkroute_vendor_snapshot_verifies_offline():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "vendor-sparkroute.py"), "verify"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "SparkRoute vendor snapshot verified" in result.stdout


def test_packaged_provenance_matches_the_repository_lock():
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    provenance = tomllib.loads(PROVENANCE.read_text(encoding="utf-8"))

    assert lock["repository"] == "https://github.com/sparksq/sparkrun-sparkroute-plugin.git"
    assert len(lock["commit"]) == 40
    assert lock["module"] == "sparkrun.plugins.sparkroute"
    assert lock["feature"] == "gateway.sparkroute"
    for field in ("repository", "commit", "tree", "version", "content_sha256"):
        assert provenance[field] == lock[field]


def test_verifier_rejects_a_locally_modified_vendor_file(tmp_path: Path):
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "vendor").mkdir()
    shutil.copy2(ROOT / "scripts" / "vendor-sparkroute.py", checkout / "scripts" / "vendor-sparkroute.py")
    shutil.copy2(LOCK, checkout / "vendor" / "sparkroute.lock")
    shutil.copytree(
        ROOT / "src" / "sparkrun" / "plugins" / "sparkroute",
        checkout / "src" / "sparkrun" / "plugins" / "sparkroute",
    )
    shutil.copytree(ROOT / "tests" / "vendor" / "sparkroute", checkout / "tests" / "vendor" / "sparkroute")

    changed = checkout / "src" / "sparkrun" / "plugins" / "sparkroute" / "engine.py"
    changed.write_bytes(changed.read_bytes() + b"\n# local edit\n")
    result = subprocess.run(
        [sys.executable, str(checkout / "scripts" / "vendor-sparkroute.py"), "verify"],
        cwd=checkout,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "vendored file differs from its lock" in result.stderr


@pytest.fixture
def upstream(tmp_path):
    repository = tmp_path / "upstream"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    (repository / "plugin.toml").write_text(_manifest_text())
    (repository / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n')
    package = repository / "src/sparkrun/plugins/sparkroute"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0.1.0"\n')
    (package / "LICENSE").write_text("test license material\n")
    (package / "LICENSE_EXCEPTION").write_text("test permission material\n")
    (repository / "tests").mkdir()
    (repository / "tests/test_sparkroute_fixture.py").write_text("def test_fixture(): pass\n")
    _commit(repository)
    subprocess.run(["git", "-C", str(repository), "tag", "v0.1.0"], check=True)
    return repository


def _commit(repository):
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        check=True,
    )


@pytest.fixture
def import_target(vendor_module, tmp_path, monkeypatch):
    root = tmp_path / "host"
    root.mkdir()
    monkeypatch.setattr(vendor_module, "ROOT", root)
    monkeypatch.setattr(vendor_module, "LOCK_PATH", root / "vendor/sparkroute.lock")
    monkeypatch.setattr(vendor_module, "SOURCE_DESTINATION", root / "src/sparkrun/plugins/sparkroute")
    monkeypatch.setattr(vendor_module, "TEST_DESTINATION", root / "tests/vendor/sparkroute")
    monkeypatch.setattr(vendor_module, "PROVENANCE_PATH", vendor_module.SOURCE_DESTINATION / "VENDORED.toml")
    return vendor_module


def test_latest_import_pins_the_release_tag_not_newer_branch_content(import_target, upstream, monkeypatch):
    released = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    package = upstream / "src/sparkrun/plugins/sparkroute"
    (package / "unreleased.py").write_text("UNRELEASED = True\n")
    _commit(upstream)
    monkeypatch.setattr(import_target, "latest_release_tag", lambda: "v0.1.0")
    assert import_target.main(["update", "--latest", "--source", str(upstream), "--initial"]) == 0
    lock = tomllib.loads(import_target.LOCK_PATH.read_text())
    assert lock["commit"] == released
    assert lock["release_tag"] == "v0.1.0"
    assert not (import_target.SOURCE_DESTINATION / "unreleased.py").exists()
    assert (import_target.SOURCE_DESTINATION / "LICENSE_EXCEPTION").read_bytes() == (package / "LICENSE_EXCEPTION").read_bytes()
    # Verify and repeat entirely from the pinned source; builds need no release lookup.
    import_target.verify()
    before = import_target.LOCK_PATH.read_bytes()
    assert import_target.main(["update", "--latest", "--source", str(upstream)]) == 0
    assert import_target.LOCK_PATH.read_bytes() == before


def test_bad_release_does_not_replace_the_existing_snapshot(import_target, upstream, monkeypatch):
    import_target.update(source=str(upstream), revision="HEAD", initial=True, force=False)
    before = import_target.LOCK_PATH.read_bytes()
    subprocess.run(["git", "-C", str(upstream), "tag", "v0.2.0"], check=True)
    monkeypatch.setattr(import_target, "latest_release_tag", lambda: "v0.2.0")
    assert import_target.main(["update", "--latest", "--source", str(upstream)]) == 1
    assert import_target.LOCK_PATH.read_bytes() == before
    import_target.verify()


def test_missing_permission_is_rejected_before_replacement(import_target, upstream):
    import_target.update(source=str(upstream), revision="HEAD", initial=True, force=False)
    before = import_target.LOCK_PATH.read_bytes()
    (upstream / "src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION").unlink()
    _commit(upstream)
    with pytest.raises(import_target.VendorError, match="required package material"):
        import_target.update(source=str(upstream), revision="HEAD", initial=False, force=False)
    assert import_target.LOCK_PATH.read_bytes() == before
    import_target.verify()


@pytest.mark.parametrize(
    "payload",
    [
        {"tag_name": "v0.1.0", "draft": True, "prerelease": False},
        {"tag_name": "v0.1.0", "draft": False, "prerelease": True},
        {"tag_name": "main", "draft": False, "prerelease": False},
        {},
        [],
    ],
)
def test_latest_rejects_unpublished_or_invalid_releases(vendor_module, monkeypatch, payload):
    monkeypatch.setattr(vendor_module.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess([], 0, json.dumps(payload)))
    with pytest.raises(vendor_module.VendorError):
        vendor_module.latest_release_tag()


def test_latest_queries_the_canonical_plugin_repository(vendor_module, monkeypatch):
    def query(args, **kwargs):
        assert args == ["gh", "api", "repos/sparksq/sparkrun-sparkroute-plugin/releases/latest"]
        assert kwargs["timeout"] == 60
        return subprocess.CompletedProcess(args, 0, json.dumps({"tag_name": "v0.1.0", "draft": False, "prerelease": False}))

    monkeypatch.setattr(vendor_module.subprocess, "run", query)
    assert vendor_module.latest_release_tag() == "v0.1.0"


def test_no_published_release_does_not_fall_back_or_modify_the_host(import_target, monkeypatch):
    def missing(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(import_target.subprocess, "run", missing)
    assert import_target.main(["update", "--latest", "--initial"]) == 1
    assert list(import_target.ROOT.iterdir()) == []
