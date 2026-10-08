# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/coldsnap/LICENSE_EXCEPTION.

from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from sparkrun.plugins.coldsnap import builder as module


IMAGE = "registry.example/runtime@sha256:" + "a" * 64


def settings():
    return module._resolve_settings(SimpleNamespace(builder_config={}), None)


def result(*, success=True, stdout=""):
    return SimpleNamespace(success=success, stdout=stdout, stderr="", returncode=0 if success else 1)


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ("linux/aarch64", "linux/arm64"),
        ("linux/arm64", "linux/arm64"),
        ("linux/x86_64", "linux/amd64"),
        ("linux/amd64", "linux/amd64"),
    ],
)
def test_builder_platform_comes_from_target_docker_not_controller_cpu(monkeypatch, reported, expected):
    builder = module.ColdSnapBuilder()
    monkeypatch.setattr(builder, "_run", lambda *_a, **_k: result(stdout=reported))
    assert builder._detect_docker_platform("spark-head", settings(), {}) == expected


def test_explicit_local_cross_arch_build_fails_before_pull_or_source_fetch(monkeypatch):
    builder = module.ColdSnapBuilder()
    monkeypatch.setattr(builder, "_detect_docker_platform", lambda host, *_a: "linux/amd64" if host == "localhost" else "linux/arm64")
    monkeypatch.setattr(builder, "_run_streaming", lambda *_a, **_k: pytest.fail("cross-arch build must not pull an image"))
    with pytest.raises(RuntimeError, match="delegated transfer mode"):
        builder.prepare_image(IMAGE, SimpleNamespace(builder_config={}), ["spark-head"], transfer_mode="push")


def test_delegated_build_passes_target_platform_to_every_docker_operation(monkeypatch):
    builder = module.ColdSnapBuilder()
    calls, probes = [], []

    def platform(host, *_args):
        probes.append(host)
        assert host == "spark-head"
        return "linux/arm64"

    def run(host, script, **_kwargs):
        assert host == "spark-head"
        calls.append(script)
        if "torch.cuda.get_device_capability" in script:
            return result(stdout="121")
        if "io.sparksq.coldsnap.runtime" in script:
            return result(stdout="<no value>")
        return result(success=False)

    def stream(host, script, **_kwargs):
        assert host == "spark-head"
        calls.append(script)
        return result()

    monkeypatch.setattr(builder, "_detect_docker_platform", platform)
    monkeypatch.setattr(builder, "_prepared_sources", lambda *_a: nullcontext("/tmp/test-coldsnap-sources"))
    monkeypatch.setattr(builder, "_run", run)
    monkeypatch.setattr(builder, "_run_streaming", stream)
    builder.prepare_image(
        IMAGE, SimpleNamespace(builder_config={"rebuild": True}), ["spark-head"], transfer_mode="delegated", snapshot_driver="n580"
    )
    assert probes == ["spark-head"]
    assert any("pull --platform linux/arm64" in script for script in calls)
    assert any("run --rm --network none --platform linux/arm64" in script for script in calls)
    build = next(script for script in calls if "sync_source()" in script)
    assert "COLDSNAP_DOCKER_PLATFORM=linux/arm64" in build
    assert build.count('--platform "$COLDSNAP_DOCKER_PLATFORM"') == 6


def test_rendered_git_fetch_failure_never_continues_to_checkout(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git = fake_bin / "git"
    git.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$GIT_TEST_LOG"\ncase " $* " in *" fetch "*) exit 42;; esac\n')
    git.chmod(0o755)
    log = tmp_path / "git.log"
    script = module.render_build_script(module._build_plan(IMAGE, settings(), "121", docker_platform="linux/arm64"))
    source_part = script.split("COLDSNAP_BASE_NCCL_RELEASE=", 1)[0] + '\nprintf "UNREACHABLE\\n"\n'
    executed = subprocess.run(
        ["bash"],
        input=source_part,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "GIT_TEST_LOG": str(log),
        },
    )
    assert executed.returncode != 0
    assert "failed to fetch pinned coldsnap" in executed.stderr
    assert "UNREACHABLE" not in executed.stdout
    assert "checkout" not in log.read_text() and "rev-parse" not in log.read_text()


@pytest.mark.parametrize("transfer_success", [True, False])
def test_source_staging_uses_controller_credentials_and_cleans_exact_remote_directory(monkeypatch, transfer_success):
    builder = module.ColdSnapBuilder()
    plan = module._build_plan(IMAGE, settings(), "121", docker_platform="linux/arm64")
    fetches, commands, transfers = [], [], []
    destination = "/tmp/sparkrun-coldsnap-sources.0123456789"

    def clone(path, url, ref, revision):
        fetches.append((url, ref, revision))
        path.mkdir()

    def run(_host, script, **_kwargs):
        commands.append(script)
        return result(stdout=destination if "mktemp" in script else "")

    def transfer(source, host, target, **kwargs):
        assert len(list(Path(source).iterdir())) == 3
        transfers.append((host, target, kwargs))
        return result(success=transfer_success)

    monkeypatch.setattr(module, "clone_pinned_source", clone)
    monkeypatch.setattr(module, "should_run_locally", lambda *_a: False)
    monkeypatch.setattr(module, "run_rsync", transfer)
    monkeypatch.setattr(builder, "_run", run)
    if transfer_success:
        with builder._prepared_sources(plan, "spark-head", {"ssh_user": "cluster-user"}) as prepared:
            assert prepared == destination
    else:
        with pytest.raises(RuntimeError, match="could not stage verified sources"):
            with builder._prepared_sources(plan, "spark-head", {"ssh_user": "cluster-user"}):
                pytest.fail("failed transfer must not start build")
    assert len(fetches) == 3 and len(transfers) == 1
    assert transfers[0][2]["ssh_user"] == "cluster-user"
    assert commands[-1] == "rm -rf -- " + destination
    assert not any("fetch" in script or "gh auth" in script or "github.com" in script for script in commands)


def test_private_source_access_failure_occurs_on_controller_before_remote_staging(monkeypatch):
    builder = module.ColdSnapBuilder()
    plan = module._build_plan(IMAGE, settings(), "121", docker_platform="linux/arm64")
    monkeypatch.setattr(module, "should_run_locally", lambda *_a: False)

    def denied(*_args):
        raise RuntimeError("control-node repository read access required")

    monkeypatch.setattr(module, "clone_pinned_source", denied)
    monkeypatch.setattr(builder, "_run", lambda *_a, **_k: pytest.fail("source failure must precede remote staging"))
    with pytest.raises(RuntimeError, match="control-node repository read access"):
        with builder._prepared_sources(plan, "spark-head", {}):
            pytest.fail("source failure must precede build")
