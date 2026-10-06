"""Opt-in real Docker check; set SPARKRUN_TEST_HOOK_IMAGE to a local bash image."""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from sparkrun.core.recipe import Recipe
from sparkrun.core.runtime_cache import RuntimeCacheSettings, build_runtime_cache_mounts
from sparkrun.orchestration.hooks import build_hook_launch_context, run_post_exec, run_pre_exec
from sparkrun.runtimes.sglang import SglangRuntime
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime


@pytest.mark.skipif(not os.environ.get("SPARKRUN_TEST_HOOK_IMAGE"), reason="requires an explicitly selected local Docker image")
@pytest.mark.parametrize("runtime", [VllmDistributedRuntime(), SglangRuntime()], ids=["vllm", "sglang"])
def test_real_container_hook_and_serve_share_compilation_cache(tmp_path, runtime):
    image = os.environ["SPARKRUN_TEST_HOOK_IMAGE"]
    recipe = Recipe.from_dict({"model": "fixture/model", "runtime": runtime.runtime_name})
    mounts = build_runtime_cache_mounts(
        runtime=runtime,
        recipe=recipe,
        settings=RuntimeCacheSettings(),
        root=str(tmp_path / "compilation cache"),
        image=image,
    )
    for directory in mounts.dirs:
        os.makedirs(directory, exist_ok=True)
    # Test a compiler override as well as the default cache root.
    os.makedirs(mounts.leaf + "/custom-triton", exist_ok=True)
    os.chmod(mounts.leaf + "/custom-triton", 0o777)  # root with all capabilities dropped
    container = "sparkrun-hook-env-test-" + uuid.uuid4().hex[:12]

    def docker(*args):
        return subprocess.run(["docker", *args], check=True, capture_output=True, text=True, timeout=30)

    args = [
        "run",
        "--pull=never",
        "--name",
        container,
        "--network=none",
        "--runtime=runc",
        "--memory=128m",
        "--cpus=0.5",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--read-only",
        "-d",
        "--mount",
        f"type=bind,src={mounts.leaf},dst=/cache/runtime",
        "--entrypoint=/bin/bash",
    ]
    for key, value in {**mounts.env, "TRITON_CACHE_DIR": "/cache/runtime/custom-triton", "SPARKRUN_NODE_RANK": "image-value"}.items():
        args.extend(["--env", f"{key}={value}"])
    try:
        docker(*args, image, "-c", "sleep infinity")
        context = build_hook_launch_context(
            ["localhost"],
            recipe.build_config_chain(),
            recipe=recipe,
            runtime=runtime.runtime_name,
            cluster_id=container,
            runtime_cache=mounts,
        )
        run_pre_exec(
            [("localhost", container)],
            [
                'test "$SPARKRUN_NODE_RANK" = 0 && '
                'test "$SPARKRUN_RUNTIME_CACHE_DIR" = /cache/runtime && '
                'bash -c \'printf "%s" "$SPARKRUN_NODE_RANK" > "$TRITON_CACHE_DIR/marker"\''
            ],
            {},
            trust=True,
            launch_context=context,
        )
        # Ordinary serve-like exec inherits the original image variable, not
        # the hook overlay, and sees the hook's write via its compiler path.
        observed = docker(
            "exec",
            container,
            "bash",
            "-c",
            'test "$SPARKRUN_NODE_RANK" = image-value && test -z "${SPARKRUN_HOOK:-}" && cat "$TRITON_CACHE_DIR/marker"',
        )
        assert observed.stdout == "0"
        run_post_exec(
            "localhost",
            container,
            ['test "$SPARKRUN_HOOK" = post_exec && test "$(cat "$SPARKRUN_RUNTIME_CACHE_DIR/custom-triton/marker")" = 0'],
            {},
            trust=True,
            launch_context=replace(context, head_ip="127.0.0.1"),
        )
        assert Path(mounts.leaf + "/custom-triton/marker").read_text() == "0"
    finally:
        docker("rm", "-f", container)
