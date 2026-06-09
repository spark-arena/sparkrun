---
name: sparkrun-agentic-inference
description: Deploy vLLM inference backends optimized specifically for autonomous coding agents, utilizing the "Goldilocks" configuration for massive context depths.
---

# Deploying Agentic Inference with Sparkrun

When deploying a model to serve as a backend for autonomous coding agents (like yourself!), the hardware profile fundamentally shifts away from standard high-throughput API serving.

## The Agentic "Production" Profile
- **Concurrency (Batch Size):** Extremely low (typically 1 to 5 concurrent requests from the main agent and its subagents).
- **Context Depth:** Extremely high (pushing the limits of the KV cache to store codebase files, documentation, and long conversation transcripts).
- **Performance Priority:** Ultra-fast "Time-To-First-Token" (snappy response times) and raw decode speeds. Maximum throughput across thousands of users is irrelevant.

## The "Goldilocks" Configuration: `--max-num-seqs 16`

To maximize performance for this specific profile, ALWAYS ensure `max_num_seqs: 16` is applied to the Sparkrun recipe's defaults (or passed via `-o max_num_seqs=16` when running `sparkrun arena benchmark run` or `vllm-tune`).

**Why?**
By default, vLLM compiles and caches CUDA graphs to support up to 256 simultaneous users. This pre-allocation consumes a massive, static chunk of VRAM. 

By hard-capping the sequence limit to `16` (which an agentic workflow will rarely exceed), you achieve the best of both worlds:
1. **Retain CUDA Graphs:** The server still utilizes CUDA graphs for low-latency kernel launches, completely bypassing CPU overhead and maximizing token generation speed.
2. **Maximize KV Cache:** The VRAM footprint of the graphs is drastically shrunk, freeing up gigabytes of memory that `vLLM` will automatically reallocate into the KV Cache. This allows you to fit significantly larger codebases into the context window without Out-Of-Memory (OOM) errors.

## Edge Case: When to use `--enforce-eager`
The ONLY times you should completely disable CUDA graphs using `--enforce-eager` (or `enforce_eager: true` in the recipe) are:
1. **Architectural Conflicts:** The model utilizes dynamic branching logic (like Gemma-4's Multi-Token Prediction) that natively breaks standard CUDA graph compilation.
2. **Absolute Context Limits:** You are pushing the absolute mathematical limits of the hardware (e.g., trying to stuff 150k tokens onto a single GPU) and need to reclaim the final ~1GB of VRAM that even a drastically shrunk CUDA graph would consume. In this scenario, you sacrifice CPU overhead to squeeze in the final few tokens.
