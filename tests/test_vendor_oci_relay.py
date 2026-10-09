"""Host-side contracts for the commit-pinned OCI Relay vendor snapshot."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "vendor" / "oci-relay.lock"
PROVENANCE = ROOT / "src" / "sparkrun" / "plugins" / "oci_relay" / "VENDORED.toml"


@pytest.fixture
def vendor_module():
    path = ROOT / "scripts" / "vendor-oci-relay.py"
    spec = importlib.util.spec_from_file_location("_sparkrun_vendor_oci_relay_test", path)
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
        'name = "oci-relay"\n' + version + 'module = "sparkrun.plugins.oci_relay"\n'
        'feature = "plugins.oci_relay"\n'
        'repository = "https://github.com/spark-arena/oci-relay.git"\n'
        'sparkrun = ">=0.3.7,<0.5"\n'
        'source = "plugin/src/sparkrun_oci_relay"\n'
        'tests = "plugin/vendor_tests/test_oci_relay_*.py"\n'
        "image_distribution_api = 1\nimage_pull_api = 1\n"
    )


def test_importer_uses_the_generated_project_version(vendor_module, monkeypatch):
    def show(_repository, *arguments, **_kwargs):
        return '[project]\nversion = "0.1.0"\nlicense = "Apache-2.0"\n' if arguments[-1].endswith(":pyproject.toml") else _manifest_text()

    monkeypatch.setattr(vendor_module, "_run_git", show)
    manifest = vendor_module._manifest(Path("unused"), "a" * 40)

    assert manifest.version == "0.1.0"


def test_importer_rejects_a_duplicate_manifest_version(vendor_module, monkeypatch):
    def show(_repository, *arguments, **_kwargs):
        return (
            '[project]\nversion = "0.1.0"\nlicense = "Apache-2.0"\n'
            if arguments[-1].endswith(":pyproject.toml")
            else _manifest_text(duplicate_version=True)
        )

    monkeypatch.setattr(vendor_module, "_run_git", show)

    with pytest.raises(vendor_module.VendorError, match="must not duplicate"):
        vendor_module._manifest(Path("unused"), "a" * 40)


def test_oci_relay_vendor_snapshot_verifies_offline():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "vendor-oci-relay.py"), "verify", "--allow-absent"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "OCI Relay" in result.stdout


@pytest.fixture
def upstream(tmp_path):
    repository = tmp_path / "upstream"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    (repository / "plugin.toml").write_text(_manifest_text())
    (repository / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\nlicense = "Apache-2.0"\n')
    package = repository / "plugin/src/sparkrun_oci_relay"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "0.1.0"\nSPARKRUN_PLUGIN_API_VERSION = 1\n')
    (package / "LICENSE").write_text("test license material\n")
    (repository / "plugin/vendor_tests").mkdir(parents=True)
    (repository / "plugin/vendor_tests/test_oci_relay_fixture.py").write_text("def test_fixture(): pass\n")
    for name in ("LICENSE", "COPYRIGHT", "README.md"):
        (repository / name).write_text("test material\n")
    (package / "releases.json").write_text(
        json.dumps(
            {
                "0.1.0": {
                    platform: {
                        "sha256": "a" * 64,
                        "url": f"https://github.com/spark-arena/oci-relay/releases/download/v0.1.0/oci-relay_0.1.0_{platform.replace('/', '_')}.tar.gz",
                    }
                    for platform in ("linux/amd64", "linux/arm64", "darwin/arm64")
                }
            }
        )
    )
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
    (root / "pyproject.toml").write_text('[project]\nversion = "0.4.0"\n')
    api = root / "src/sparkrun/core/image_distribution.py"
    api.parent.mkdir(parents=True)
    api.write_text("IMAGE_DISTRIBUTION_API_VERSION = 1\nIMAGE_PULL_API_VERSION = 1\n")
    monkeypatch.setattr(vendor_module, "ROOT", root)
    monkeypatch.setattr(vendor_module, "LOCK_PATH", root / "vendor/oci-relay.lock")
    monkeypatch.setattr(vendor_module, "SOURCE_DESTINATION", root / "src/sparkrun/plugins/oci_relay")
    monkeypatch.setattr(vendor_module, "TEST_DESTINATION", root / "tests/vendor/oci_relay")
    monkeypatch.setattr(vendor_module, "PROVENANCE_PATH", vendor_module.SOURCE_DESTINATION / "VENDORED.toml")
    return vendor_module


def test_latest_import_pins_the_release_tag_not_newer_branch_content(import_target, upstream, monkeypatch):
    released = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    package = upstream / "plugin/src/sparkrun_oci_relay"
    (package / "unreleased.py").write_text("UNRELEASED = True\n")
    _commit(upstream)
    monkeypatch.setattr(import_target, "latest_release_tag", lambda: "v0.1.0")
    monkeypatch.setattr(import_target, "latest_plugin_revision", lambda tag: released)
    assert import_target.main(["update", "--latest", "--source", str(upstream), "--initial"]) == 0
    lock = tomllib.loads(import_target.LOCK_PATH.read_text())
    assert lock["commit"] == released
    assert lock["release_tag"] == "v0.1.0"
    assert not (import_target.SOURCE_DESTINATION / "unreleased.py").exists()
    assert (import_target.SOURCE_DESTINATION / "LICENSE").read_bytes() == (upstream / "LICENSE").read_bytes()
    assert not (import_target.SOURCE_DESTINATION / "LICENSE_EXCEPTION").exists()
    # Verify and repeat entirely from the pinned source; builds need no release lookup.
    import_target.verify()
    before = import_target.LOCK_PATH.read_bytes()
    assert import_target.main(["update", "--latest", "--source", str(upstream)]) == 0
    assert import_target.LOCK_PATH.read_bytes() == before


def test_bad_release_does_not_replace_the_existing_snapshot(import_target, upstream, monkeypatch):
    import_target.update(
        source=str(upstream),
        revision=subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
        initial=True,
        force=False,
    )
    before = import_target.LOCK_PATH.read_bytes()
    subprocess.run(["git", "-C", str(upstream), "tag", "v0.2.0"], check=True)
    monkeypatch.setattr(import_target, "latest_release_tag", lambda: "v0.2.0")
    monkeypatch.setattr(
        import_target,
        "latest_plugin_revision",
        lambda tag: subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
    )
    assert import_target.main(["update", "--latest", "--source", str(upstream)]) == 1
    assert import_target.LOCK_PATH.read_bytes() == before
    import_target.verify()


def test_missing_license_is_rejected_before_replacement(import_target, upstream):
    import_target.update(
        source=str(upstream),
        revision=subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
        initial=True,
        force=False,
    )
    before = import_target.LOCK_PATH.read_bytes()
    (upstream / "LICENSE").unlink()
    _commit(upstream)
    with pytest.raises(import_target.VendorError, match="LICENSE"):
        import_target.update(
            source=str(upstream),
            revision=subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
            initial=False,
            force=False,
        )
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
        assert args == ["gh", "api", "repos/spark-arena/oci-relay/releases/latest"]
        assert kwargs["timeout"] == 60
        return subprocess.CompletedProcess(args, 0, json.dumps({"tag_name": "v0.1.0", "draft": False, "prerelease": False}))

    monkeypatch.setattr(vendor_module.subprocess, "run", query)
    assert vendor_module.latest_release_tag() == "v0.1.0"


def test_no_published_release_does_not_fall_back_or_modify_the_host(import_target, monkeypatch):
    def missing(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(import_target.subprocess, "run", missing)
    assert import_target.main(["update", "--latest", "--initial"]) == 1
    assert sorted(p.name for p in import_target.ROOT.iterdir()) == ["pyproject.toml", "src"]


@pytest.mark.parametrize("value", ["true", "0", '"1"'])
def test_importer_rejects_invalid_image_distribution_api(vendor_module, monkeypatch, value):
    def show(_repository, *arguments, **_kwargs):
        return (
            '[project]\nversion = "0.1.0"\nlicense = "Apache-2.0"\n'
            if arguments[-1].endswith(":pyproject.toml")
            else _manifest_text().replace("image_distribution_api = 1", "image_distribution_api = " + value)
        )

    monkeypatch.setattr(vendor_module, "_run_git", show)
    with pytest.raises(vendor_module.VendorError, match="image_distribution_api"):
        vendor_module._manifest(Path("unused"), "a" * 40)


def test_importer_preserves_image_distribution_api_without_importing_code(vendor_module, monkeypatch):
    def show(_repository, *arguments, **_kwargs):
        return '[project]\nversion = "0.1.0"\nlicense = "Apache-2.0"\n' if arguments[-1].endswith(":pyproject.toml") else _manifest_text()

    monkeypatch.setattr(vendor_module, "_run_git", show)
    manifest = vendor_module._manifest(Path("unused"), "a" * 40)
    assert manifest.image_distribution_api == 1
    parameters = dict(manifest=manifest, commit="a" * 40, tree="b" * 40, digest="c" * 64)
    assert tomllib.loads(vendor_module._render_lock(**parameters, files=[]))["image_distribution_api"] == 1
    assert tomllib.loads(vendor_module._render_provenance(**parameters))["image_distribution_api"] == 1


def test_optional_verify_accepts_only_absent_content(import_target):
    import_target.verify(allow_absent=True)
    with pytest.raises(import_target.VendorError, match="does not exist"):
        import_target.verify()
    cache = import_target.SOURCE_DESTINATION / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "old.cpython-312.pyc").write_bytes(b"stale bytecode")
    import_target.verify(allow_absent=True)
    (import_target.SOURCE_DESTINATION / "__init__.py").write_text("# unlocked source")
    with pytest.raises(import_target.VendorError, match="does not exist"):
        import_target.verify(allow_absent=True)


def test_import_requires_full_commit_and_explicit_initial(import_target, upstream):
    with pytest.raises(import_target.VendorError, match="full lowercase commit"):
        import_target.update(source=str(upstream), revision="HEAD", initial=True, force=False)
    rev = _head(upstream)
    with pytest.raises(import_target.VendorError, match="first import requires"):
        import_target.update(source=str(upstream), revision=rev, initial=False, force=False)
    import_target.SOURCE_DESTINATION.mkdir(parents=True)
    (import_target.SOURCE_DESTINATION / "local.py").write_text("# work in progress")
    with pytest.raises(import_target.VendorError, match="unlocked"):
        import_target.update(source=str(upstream), revision=rev, initial=True, force=True)


def _head(repository):
    return subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra", "provenance", "wrong-plugin"])
def test_verify_and_update_reject_damaged_snapshots(import_target, upstream, mutation):
    import_target.update(source=str(upstream), revision=_head(upstream), initial=True, force=False)
    package = import_target.SOURCE_DESTINATION
    if mutation == "changed":
        (package / "__init__.py").write_text("# local edit")
    elif mutation == "missing":
        (package / "LICENSE").unlink()
    elif mutation == "extra":
        (package / "extra.py").write_text("# untracked")
    elif mutation == "provenance":
        import_target.PROVENANCE_PATH.write_text('version = "99.0.0"\n')
    else:
        import_target.LOCK_PATH.write_text(import_target.LOCK_PATH.read_text().replace('name = "oci-relay"', 'name = "wrong"'))
    before = import_target.LOCK_PATH.read_bytes()
    with pytest.raises(import_target.VendorError):
        import_target.verify(allow_absent=True)
    with pytest.raises(import_target.VendorError):
        import_target.update(source=str(upstream), revision=_head(upstream), initial=False, force=False)
    assert import_target.LOCK_PATH.read_bytes() == before


@pytest.mark.parametrize("mutation", ["missing-api", "unsupported-api", "incompatible-host", "license", "symlink"])
def test_incompatible_upstream_fails_before_replacement(import_target, upstream, mutation):
    import_target.update(source=str(upstream), revision=_head(upstream), initial=True, force=False)
    before = import_target.LOCK_PATH.read_bytes()
    package = upstream / "plugin/src/sparkrun_oci_relay"
    if mutation == "missing-api":
        (package / "__init__.py").write_text("raise AssertionError('upstream must not execute')")
    elif mutation == "unsupported-api":
        (package / "__init__.py").write_text("SPARKRUN_PLUGIN_API_VERSION = True")
    elif mutation == "incompatible-host":
        manifest = upstream / "plugin.toml"
        manifest.write_text(manifest.read_text().replace(">=0.3.7,<0.5", ">=0.3.7,<0.4"))
    elif mutation == "license":
        (upstream / "LICENSE").unlink()
    else:
        (package / "redirect.py").symlink_to("__init__.py")
    _commit(upstream)
    with pytest.raises(import_target.VendorError):
        import_target.update(source=str(upstream), revision=_head(upstream), initial=False, force=False)
    assert import_target.LOCK_PATH.read_bytes() == before
    import_target.verify()


def test_local_dirty_source_is_not_imported(import_target, upstream):
    revision = _head(upstream)
    (upstream / "plugin/src/sparkrun_oci_relay/uncommitted.py").write_text("# not approved")
    import_target.update(source=str(upstream), revision=revision, initial=True, force=False)
    assert not (import_target.SOURCE_DESTINATION / "uncommitted.py").exists()
    lock = tomllib.loads(import_target.LOCK_PATH.read_text())
    provenance = tomllib.loads(import_target.PROVENANCE_PATH.read_text())
    assert lock["commit"] == revision
    assert lock["commit"] == provenance["commit"]
    assert (import_target.SOURCE_DESTINATION / "LICENSE").read_bytes() == (upstream / "LICENSE").read_bytes()
    assert not (import_target.SOURCE_DESTINATION / "LICENSE_EXCEPTION").exists()


def test_destination_symlink_is_rejected_before_replacement(import_target, upstream, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    original = outside / "preserved"
    original.write_text("keep me")
    import_target.SOURCE_DESTINATION.parent.mkdir(parents=True)
    import_target.SOURCE_DESTINATION.symlink_to(outside, target_is_directory=True)
    with pytest.raises(import_target.VendorError, match="symlink"):
        import_target.update(source=str(upstream), revision=_head(upstream), initial=True, force=True)
    assert original.read_text() == "keep me"
    with pytest.raises(import_target.VendorError, match="symlink"):
        import_target.verify(allow_absent=True)


def test_oci_relay_binding_is_enabled_on_every_channel():
    from sparkrun.core.features import get_feature
    from sparkrun.core.in_tree_plugins import plugin_feature_flag

    assert plugin_feature_flag("oci_relay") == "plugins.oci_relay"
    feature = get_feature("plugins.oci_relay")
    assert feature is not None
    assert feature.default is True
    assert all(feature.default_for_channel(channel) for channel in ("stable", "beta", "alpha"))


@pytest.mark.parametrize("change", [{"version": "9.9.9"}, {"repository": "https://example.invalid"}, {"commit": "main"}, {"schema": 2}])
def test_latest_adapter_descriptor_rejects_invalid_pins(vendor_module, monkeypatch, change):
    metadata = {"schema": 1, "version": "0.1.0", "repository": vendor_module.UPSTREAM_REPOSITORY, "commit": "a" * 40, **change}
    monkeypatch.setattr(vendor_module.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(metadata)))
    with pytest.raises(vendor_module.VendorError, match="plugin-release"):
        vendor_module.latest_plugin_revision("v0.1.0")


def test_latest_adapter_descriptor_resolves_only_published_commit(vendor_module, monkeypatch):
    metadata = {"schema": 1, "version": "0.1.0", "repository": vendor_module.UPSTREAM_REPOSITORY, "commit": "a" * 40}

    def download(command, **kwargs):
        assert command == [
            "gh",
            "release",
            "download",
            "v0.1.0",
            "--repo",
            "spark-arena/oci-relay",
            "--pattern",
            "plugin-release.json",
            "--output",
            "-",
        ]
        return subprocess.CompletedProcess(command, 0, json.dumps(metadata))

    monkeypatch.setattr(vendor_module.subprocess, "run", download)
    assert vendor_module.latest_plugin_revision("v0.1.0") == "a" * 40


@pytest.mark.parametrize("engine_changed", [False, True])
def test_post_release_adapter_pin_requires_unchanged_engine(import_target, upstream, monkeypatch, engine_changed):
    if engine_changed:
        (upstream / "cmd").mkdir()
        (upstream / "cmd/engine.go").write_text("package main\n")
    else:
        pins = upstream / "plugin/src/sparkrun_oci_relay/releases.json"
        pins.write_text(pins.read_text().replace("a" * 64, "b" * 64))
    _commit(upstream)
    revision = _head(upstream)
    monkeypatch.setattr(import_target, "latest_release_tag", lambda: "v0.1.0")
    monkeypatch.setattr(import_target, "latest_plugin_revision", lambda tag: revision)
    assert import_target.main(["update", "--latest", "--source", str(upstream), "--initial"]) == int(engine_changed)
    if not engine_changed:
        assert tomllib.loads(import_target.LOCK_PATH.read_text())["commit"] == revision
        import_target.verify()
    else:
        assert not import_target.LOCK_PATH.exists()


@pytest.mark.parametrize(
    "channel,override,enabled",
    [
        ("alpha", None, True),
        ("beta", None, True),
        ("stable", None, True),
        ("alpha", False, False),
        ("beta", False, False),
        ("stable", False, False),
        ("stable", True, True),
    ],
)
def test_bundled_oci_relay_bootstrap_gate(tmp_path, channel, override, enabled):
    if not PROVENANCE.is_file():
        pytest.skip("optional OCI Relay vendor snapshot is absent")
    features = {"channel": channel}
    if override is not None:
        features["plugins.oci_relay"] = override
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"features": features}))
    code = """
import json
import sys
from sparkrun.core.bootstrap import init_sparkrun
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.image_distribution import _PROVIDERS, has_image_distribution_provider
init_sparkrun(config=SparkrunConfig(sys.argv[1]))
print(json.dumps({
    "registered": "oci-relay" in _PROVIDERS,
    "selected": has_image_distribution_provider(),
    "imported": "sparkrun.plugins.oci_relay" in sys.modules,
}))
"""
    env = {key: value for key, value in os.environ.items() if not key.startswith("SPARKRUN_FEATURE_")}
    env.update(
        STATEFUL_ROOT=str(tmp_path / "state"),
        SPARKRUN_NO_TELEMETRY="1",
        SPARKRUN_NO_INSTALLED_PLUGINS="1",
        SPARKRUN_NO_EXTERNAL_PLUGINS="1",
    )
    result = subprocess.run([sys.executable, "-c", code, str(config)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "registered": enabled,
        "selected": enabled,
        "imported": enabled,
    }


def test_vendored_readme_links_pin_upstream_documents(vendor_module):
    commit = "a" * 40
    text = "[guide](docs/platforms.md#platforms) [license](LICENSE) [web](https://example.com) [anchor](#setup)"
    observed = vendor_module._portable_readme(text, commit)
    assert observed == (
        f"[guide](https://github.com/spark-arena/oci-relay/blob/{commit}/docs/platforms.md#platforms) "
        f"[license](https://github.com/spark-arena/oci-relay/blob/{commit}/LICENSE) "
        "[web](https://example.com) [anchor](#setup)"
    )


@pytest.mark.parametrize("license_id", [None, "AGPL-3.0-only", "MIT", ""])
def test_non_apache_source_does_not_replace_snapshot(import_target, upstream, license_id):
    import_target.update(source=str(upstream), revision=_head(upstream), initial=True, force=False)
    before = import_target.LOCK_PATH.read_bytes()
    text = '[project]\nversion = "0.1.0"\n'
    if license_id is not None:
        text += "license = " + json.dumps(license_id) + "\n"
    (upstream / "pyproject.toml").write_text(text)
    _commit(upstream)
    with pytest.raises(import_target.VendorError, match="must be Apache-2.0"):
        import_target.update(source=str(upstream), revision=_head(upstream), initial=False, force=False)
    assert import_target.LOCK_PATH.read_bytes() == before
    import_target.verify()
