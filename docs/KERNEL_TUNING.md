# SGLang kernel tuning

`sparkrun tune sglang` tunes kernels inside the recipe's image on one host.
The GPU should be idle; concurrent inference or another tuning job invalidates
performance comparisons. No model weights are loaded for dense FP8 tuning.

The existing default, `--mode moe`, tunes fused MoE Triton kernels. It does not
address warnings about missing `dtype=fp8_w8a8` dense GEMM configurations, and is
not the appropriate tuner for an image using a separate MXFP4 MoE backend.

## Dense block-FP8

Use the **per-rank** N and K dimensions and quantization block shape reported by
the running engine. Shapes are explicit because model families and custom
adapters can shard or fuse projections differently. The tuner does not divide
these dimensions by TP again. For example:

```bash
sparkrun tune sglang @experimental/deepseek-v4.1-flash-b12x-dspark-sglang-tp4 \
  -H <idle-spark> --mode fp8 --block-shape 32 32 \
  --shape 1792 5120 --shape 8192 1280 --shape 5120 2048 \
  --shape 1152 5120 --shape 5120 576 --shape 25600 6144
```

`--batch-size M` is repeatable. Omit it to tune token counts
`1, 2, 4, 6, 8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048, 4096`.
The grid includes the six-token DSpark verification case as well as decode and
chunked prefill. Use `--timeout` to set a job limit in seconds (default: none).
FP8 mode runs once, using the recipe's TP setting for reporting, and requires
`--parallel 1`. It does not clone upstream benchmarks or run recipe preparation
mods. The runtime kernel comes directly from the selected image. Image preparation
uses Sparkrun's distribution provider, including OCI Relay's verified local aliases.
The image's serving entrypoint is overridden for the tuning sleeper.

The tuner includes the default tile configuration in its search, supports
32-element quantization blocks, and measures CUDA graph replay to avoid Python
launch overhead dominating small decode kernels. It checks the default against
dense dequantized multiplication, then requires each candidate to match the
default's BF16 output bit-for-bit on the test tensors. Only valid candidates are
benchmarked; the finalists and default receive longer measurements. These are
kernel measurements, not end-to-end throughput or a model quality qualification.

## Results and distribution

The default output is `~/.cache/sparkrun/tuning/sglang`. Remote results are synced
back to the control node. `--output-dir` selects a staging directory on both
machines; copy reviewed results into the tuning cache or a registry to use them.
Dense results use this layout:

```text
fp8/<sha256-of-installed-kernel-file>/triton_<version>/
  N=...,K=...,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[32, 32].json
```

JSON files map token count M to tile configurations. Adjacent `.measurements`
files record default/tuned latency in milliseconds, the measured ratio, and
rejected-candidate counts. Configs are written atomically after each batch size,
so an interrupted run can leave a valid but incomplete M grid; check coverage
before publishing it.

A registry declares a tuning root in `.sparkrun/registry.yaml`, then places the
above tree under `<tuning-root>/sglang/`. Registry discovery copies JSON configs
into the local cache, preserving the relative path; launch distribution copies
them to all participating hosts. Existing local files take precedence over
registry copies. Local-file recipes can use configs already in the local tuning
cache; they do not automatically import configs from an unrelated registry.

Before Docker SGLang serving begins, Sparkrun installs matching dense configs
into the loaded kernel module's `configs` directory. This is needed for images
whose dense FP8 loader has no configurable search path. It copies individual
files and retains unrelated configs bundled in the image. It matches the kernel
source hash, Triton version, and GPU name; a different image/kernel or Triton
build requires separate tuning. Installation failures warn and leave the
runtime's normal fallback behavior in effect. Other executors do not perform
this Docker installation step. Existing MoE config mounting is unchanged.

The CLI's tuning-sync phase runs before serving containers exist and distributes
the files to host storage. Installation runs after container creation and recipe
preparation, when the final package files are available to fingerprint. It uses
package metadata and `nvidia-smi`, without importing Torch or initializing CUDA.

The `replicated_split` assertion in the pinned DeepSeek V4.1 adapter is separate:
its weight proxy does not slice `[N/32, K/32]` block scales along with the weight.
The adapter catches the resulting shape assertion and disables splitting for
that layer. Dense kernel tuning does not repair or enable that optimization.
