---
name: sparkrun-benchmark-workflow
description: Standard operating procedure for orchestrating large-context LLM benchmarks with sparkrun and llama-benchy. Trigger this skill whenever tasked with benchmarking a new LLM model for Spark Arena.
---

# Sparkrun Benchmarking Workflow

When orchestrating a new LLM benchmark for Spark Arena using `sparkrun`, follow this systematic process to ensure stability and accuracy.

## 1. Prepare the Recipe
* Recipes are stored in `/home/jlapenna/p/sparkstack-registry/sparkrun/`.
* Inspect the target model's `.yaml` recipe. Ensure that `max_model_len` is correctly set to fit within the `121GB` limit of the target node (usually `spark.lan.jlapenna.net`).
* **Warning for MTP Models:** If the model uses Multi-Token Prediction (MTP), the `num_speculative_tokens` (e.g. 4) will significantly increase the memory footprint of the KV Cache. Even if the vLLM memory estimator says it fits, MTP speculative heads may cause out-of-memory errors on startup. Adjust the context length or speculative tokens if needed.

## 2. Pre-Flight Node Cleanup
* **CRITICAL:** `sparkrun` currently has a bug where interrupted or manually launched Docker containers may be left running on the host, silently hogging VRAM.
* Always check the status of the target host before starting: `uv run sparkrun status --hosts spark.lan.jlapenna.net`
* Kill any running inference servers or lingering test containers: `uv run sparkrun stop --all --hosts spark.lan.jlapenna.net`

## 3. Launching the Benchmark
* Run the benchmark via the CLI:
  `cd /home/jlapenna/p/sparkrun && uv run sparkrun arena benchmark run /home/jlapenna/p/sparkstack-registry/sparkrun/<recipe>.yaml --hosts spark.lan.jlapenna.net`
* The benchmark will automatically push the results to Spark Arena via a `subXXXXXX` ID.

## 4. Recovering from Crashes and Deadlocks
* If the benchmark crashes or deadlocks during a run (e.g. vLLM freezes on a massive `100,000` context length test under concurrency), DO NOT tamper with the benchmark's internal state.yaml file. Modifying the schedule manually invalidates the benchmark results.
* Instead, kill the `sparkrun arena benchmark run` process, then forcefully kill any `llama-benchy` orphaned processes locally (`pkill -9 -f llama-benchy`) and natively remove the hung remote Docker container using `uv run sparkrun stop --all --hosts spark.lan.jlapenna.net` (see Step 2).
* Clean up the corrupted or aborted benchmark state by deleting its state directory: `rm -rf /home/jlapenna/.cache/sparkrun/benchmarks/<benchmark_id>`.
* To prevent the deadlock from recurring, modify the model's `.yaml` recipe file and reduce the `max_model_len` to a value known to be safe (e.g., `65536`).
* When you relaunch the benchmark using the updated recipe, `llama-benchy` will query the new `max_model_len` and naturally skip any tests in the official profile that exceed this length. This produces a valid, untampered benchmark submission.

## 5. Documentation
* Track your progress across sessions by maintaining the `benchmark_status.md` artifact. Update it with "Completed", "Failed", or "In Progress" statuses along with any relevant findings.
