"""Standalone dense block-FP8 tuner/installer, executed inside the runtime image.

Uses the installed SGLang implementation, including its default configuration,
not a benchmark checkout whose kernels can differ from the serving image.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


DEFAULT_BATCH_SIZES = (1, 2, 4, 6, 8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048, 4096)


def default_config(block_n, block_k):
    return dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k, GROUP_SIZE_M=32, num_warps=4, num_stages=3)


def search_configs(m, block_n, block_k):
    """Keep K tiles within one quantization block (including 32-element blocks)."""
    yield default_config(block_n, block_k)
    for bm, bn, warps, stages, group in itertools.product(
        (16, 32, 64) if m <= 32 else (32, 64, 128), (32, 64, 128), (4, 8), (2, 3), (1, 32)
    ):
        candidate = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=block_k, GROUP_SIZE_M=group, num_warps=warps, num_stages=stages)
        if candidate != default_config(block_n, block_k):
            yield candidate


def load_kernel():
    for name in ("sglang.kernels.ops.quantization.fp8_kernel", "sglang.srt.layers.quantization.fp8_kernel"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as error:
            if not error.name or not name.startswith(error.name):
                raise
    raise RuntimeError("This image has no supported SGLang block-FP8 kernel")


def config_directory(root, kernel_file, triton_version):
    digest = hashlib.sha256(Path(kernel_file).read_bytes()).hexdigest()
    return Path(root) / "fp8" / digest / ("triton_" + triton_version.replace(".", "_"))


def config_filename(n, k, device, block_n, block_k):
    return "N=%d,K=%d,device_name=%s,dtype=fp8_w8a8,block_shape=[%d, %d].json" % (n, k, device.replace(" ", "_"), block_n, block_k)


def write_json(path, data):
    """Never leave a truncated config if tuning is interrupted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)
    # The output bind mount belongs to the SSH user; Docker normally runs root.
    owner = path.parent
    while owner.parent != owner and owner.name != "fp8":
        owner = owner.parent
    if owner.name == "fp8" and os.geteuid() == 0:
        stat = owner.parent.stat()
        for item in (path, path.parent, path.parent.parent, owner):
            os.chown(item, stat.st_uid, stat.st_gid)


def install_configs(root, kernel_file, triton_version, device):
    source = config_directory(root, kernel_file, triton_version)
    destination = Path(kernel_file).parent / "configs"
    paths = sorted(source.glob("N=*,K=*,device_name=%s,dtype=fp8_w8a8,*.json" % device.replace(" ", "_")))
    for path in paths:
        value = json.loads(path.read_text())
        if (
            not value
            or not isinstance(value, dict)
            or not all(
                str(m).isdigit()
                and int(m) > 0
                and isinstance(c, dict)
                and all(type(c.get(key)) is int and c[key] > 0 for key in default_config(32, 32))
                for m, c in value.items()
            )
        ):
            raise ValueError("Invalid FP8 tuning configuration: %s" % path.name)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / path.name
        temporary = target.with_suffix(".json.tmp")
        shutil.copyfile(path, temporary)
        temporary.replace(target)  # Atomic; never write through a target symlink.
    print("Installed %d matching SGLang FP8 tuning config(s)" % len(paths), flush=True)
    return len(paths)


def install_for_image(root):
    """Locate configs without importing Torch or initializing CUDA at startup."""
    package = importlib.util.find_spec("sglang")
    if package is None or not package.submodule_search_locations:
        raise RuntimeError("Cannot locate the installed SGLang package")
    candidates = [
        Path(base) / relative
        for base in package.submodule_search_locations
        for relative in ("kernels/ops/quantization/fp8_kernel.py", "srt/layers/quantization/fp8_kernel.py")
    ]
    kernel_file = next((path for path in candidates if path.is_file()), None)
    if kernel_file is None:
        raise RuntimeError("This image has no supported SGLang block-FP8 kernel")
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=10)
    names = set(line.strip() for line in devices.splitlines() if line.strip())
    if not names:
        raise RuntimeError("No NVIDIA GPU available for config installation")
    version = importlib.metadata.version("triton")
    return sum(install_configs(root, kernel_file, version, name) for name in sorted(names))


def tune_shape(torch, triton, kernel, n, k, block, batches, output):
    block_n, block_k = block
    run = getattr(kernel, "w8a8_block_fp8_matmul_triton", None) or kernel.w8a8_block_fp8_matmul
    original_lookup = kernel.get_w8a8_block_fp8_configs
    baseline = default_config(*block)
    active = baseline
    kernel.get_w8a8_block_fp8_configs = lambda *_: {1: active}
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    b = torch.randn(n, k, device="cuda").clamp(-4, 4).to(torch.float8_e4m3fn)
    bs = torch.rand(triton.cdiv(n, block_n), triton.cdiv(k, block_k), device="cuda") * 0.1 + 0.01
    configs, measurements = {}, {}
    path = output / config_filename(n, k, torch.cuda.get_device_name(), *block)
    started = time.monotonic()
    try:
        for m in batches:
            active = baseline
            a = torch.randn(m, k, device="cuda").clamp(-4, 4).to(torch.float8_e4m3fn)
            scales = torch.rand(m, triton.cdiv(k, block_k), device="cuda") * 0.1 + 0.01

            def invoke(a=a, scales=scales):
                return run(a, b, scales, bs, list(block), output_dtype=torch.bfloat16)

            expected = invoke()
            # Also check the reference kernel against dense dequantized matmul.
            # Other candidates must then match the reference kernel bit-for-bit.
            dequant_a = a.float() * scales.repeat_interleave(block_k, 1)[:, :k]
            dequant_b = b.float() * bs.repeat_interleave(block_n, 0).repeat_interleave(block_k, 1)[:n, :k]
            reference = dequant_a @ dequant_b.T
            torch.testing.assert_close(expected.float(), reference, rtol=0.01, atol=0.02)
            del dequant_a, dequant_b, reference
            candidates, rejected = [], 0
            for candidate in search_configs(m, *block):
                active = candidate
                try:
                    actual = invoke()
                    if not torch.equal(actual, expected):
                        rejected += 1
                        continue
                    # Graph replay measures GPU work without Python dispatch
                    # dominating the small decode shapes (serving uses graphs).
                    latency = triton.testing.do_bench_cudagraph(invoke, rep=5)
                except triton.runtime.autotuner.OutOfResources:
                    rejected += 1
                    continue
                candidates.append((latency, candidate))
            if not candidates:
                raise RuntimeError("No valid configurations for N=%d K=%d M=%d" % (n, k, m))
            # Recheck the finalists and baseline with longer measurements.
            finalists = [candidate for _, candidate in sorted(candidates, key=lambda entry: entry[0])[:3]]
            if baseline not in finalists:
                finalists.append(baseline)
            timings = []
            for candidate in finalists:
                active = candidate
                timings.append((triton.testing.do_bench_cudagraph(invoke, rep=100), candidate))
            best_ms, best = min(timings, key=lambda entry: entry[0])
            baseline_ms = next(ms for ms, candidate in timings if candidate == baseline)
            configs[str(m)] = best
            measurements[str(m)] = dict(default_ms=baseline_ms, tuned_ms=best_ms, speedup=baseline_ms / best_ms, rejected=rejected)
            write_json(path, configs)
            write_json(path.with_suffix(".measurements"), measurements)
            print(
                "FP8 N=%d K=%d M=%d: %.4f -> %.4f ms (%.2fx), %d invalid candidates; %.1fs elapsed"
                % (n, k, m, baseline_ms, best_ms, baseline_ms / best_ms, rejected, time.monotonic() - started),
                flush=True,
            )
    finally:
        kernel.get_w8a8_block_fp8_configs = original_lookup
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=2, action="append", default=[])
    parser.add_argument("--block-shape", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--batch-size", type=int, action="append")
    parser.add_argument("--output", required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    batches = sorted(set(args.batch_size or DEFAULT_BATCH_SIZES))
    if any(n <= 0 for shape in args.shape for n in shape) or any(m <= 0 for m in batches):
        parser.error("shapes and batch sizes must be positive")
    if any(n < 16 or n & (n - 1) for n in args.block_shape):
        parser.error("quantization blocks must be powers of two, at least 16")
    if not args.install and not args.shape:
        parser.error("at least one explicit weight shape is required")
    if args.install:
        install_for_image(args.output)
        return
    torch = importlib.import_module("torch")
    triton = importlib.import_module("triton")
    importlib.import_module("triton.testing")
    kernel = load_kernel()
    output = config_directory(args.output, kernel.__file__, triton.__version__)
    print(
        json.dumps(
            dict(
                device=torch.cuda.get_device_name(),
                torch=torch.__version__,
                triton=triton.__version__,
                kernel=kernel.__file__,
                output=str(output),
            )
        ),
        flush=True,
    )
    for n, k in dict.fromkeys(tuple(shape) for shape in args.shape):
        tune_shape(torch, triton, kernel, n, k, tuple(args.block_shape), batches, output)


if __name__ == "__main__":
    main()
