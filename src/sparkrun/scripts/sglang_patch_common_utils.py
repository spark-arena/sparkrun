"""Patch SGLang common_utils.py to handle MoE config attribute variations.

This script is piped into a running container via ``docker exec -i``
after cloning the SGLang benchmark scripts.  It fixes two classes of
upstream issues:

1. ``config.architectures`` may be ``None`` for some models (e.g.
   Qwen3.5 MoE under transformers >= 5.x), causing an immediate crash.

2. MoE config attribute names vary across model families:
   - Mixtral: ``num_local_experts``
   - Qwen: ``num_experts``
   - DeepSeek: ``n_routed_experts``

   The SGLang tuning script only checks ``num_local_experts``, which
   crashes for models using alternative names.

The fix injects a ``_normalize_moe_config()`` helper and inserts a call
to it directly before the first ``config.architectures`` access in
``get_model_config()``.
"""

import pathlib
import sys

TARGET = pathlib.Path("/tmp/sglang_src/benchmark/kernels/fused_moe_triton/common_utils.py")

NORMALIZER_FN = '''\

def _normalize_moe_config(cfg):
    """Normalize MoE config attributes across model families."""
    # Ensure architectures is set
    if not getattr(cfg, "architectures", None):
        cfg.architectures = [getattr(cfg, "model_type", "unknown")]

    # Resolve nested sub-configs (multimodal models)
    for sub_attr in ("text_config", "llm_config"):
        sub = getattr(cfg, sub_attr, None)
        if sub is not None:
            for attr in (
                "num_local_experts", "num_experts", "n_routed_experts",
                "intermediate_size", "moe_intermediate_size", "hidden_size",
            ):
                if not hasattr(cfg, attr) and hasattr(sub, attr):
                    setattr(cfg, attr, getattr(sub, attr))

    # num_local_experts aliases
    if getattr(cfg, "num_local_experts", None) is None:
        for alt in ("num_experts", "n_routed_experts"):
            v = getattr(cfg, alt, None)
            if v is not None:
                cfg.num_local_experts = v
                break

    # intermediate_size aliases
    if getattr(cfg, "intermediate_size", None) is None:
        for alt in ("moe_intermediate_size", "ffn_dim"):
            v = getattr(cfg, alt, None)
            if v is not None:
                cfg.intermediate_size = v
                break

    return cfg

'''

SENTINEL = "_normalize_moe_config"


def main() -> int:
    if not TARGET.exists():
        print("SKIP: %s not found" % TARGET, file=sys.stderr)
        return 0

    text = TARGET.read_text()

    if SENTINEL in text:
        # Already patched (e.g. --skip-clone re-run)
        return 0

    # --- Step 1: Insert normalizer function before get_model_config ---
    if "def get_model_config" in text:
        text = text.replace(
            "def get_model_config",
            NORMALIZER_FN + "def get_model_config",
        )

    # --- Step 2: Insert normalizer call before the architectures access ---
    # This is the most reliable injection point because we know the exact
    # line pattern.  The normalizer fixes architectures, num_local_experts,
    # and intermediate_size before any of them are accessed.
    target_line = "architecture = config.architectures[0]"
    if target_line in text:
        lines = text.split("\n")
        new_lines = []
        for line in lines:
            if target_line in line:
                indent = len(line) - len(line.lstrip())
                new_lines.append(" " * indent + "config = _normalize_moe_config(config)")
            new_lines.append(line)
        text = "\n".join(new_lines)

    TARGET.write_text(text)
    print("OK: patched %s" % TARGET)
    return 0


if __name__ == "__main__":
    sys.exit(main())
