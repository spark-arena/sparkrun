---
description: Run performance or tool-evaluation benchmarks; optionally publish to Arena
argument-hint: "[perf|tools] <recipe> [options]"
---

# /sparkrun:benchmark

Use the workloads skill. Preserve the requested category; default to performance only
when none is specified. Resolve a recipe and target hosts as for run.

```bash
sparkrun benchmark perf <recipe> --cluster <name>
sparkrun benchmark tools <recipe> --skip-run
sparkrun benchmark perf <recipe> --profile @official/spark-arena-v2
sparkrun benchmark tools <recipe> -b parallel=2 -b max_turns=8
sparkrun benchmark <recipe> --arena
sparkrun benchmark <recipe> --arena --local-test
sparkrun benchmark resume <benchmark_id>
```

Use main `benchmark --arena` for requested publication, not the hidden
`arena benchmark` alias. Choosing an Arena profile alone does not upload.
`--local-test` requires `--arena` and skips authentication/upload. Login requires
an interactive user terminal.

Performance typically uses llama-benchy; tools uses tool-eval-bench. Discover
profiles via `registry list-benchmark-profiles` rather than pinning obsolete
names/versions. `-b` changes framework args; `-o` changes recipe defaults.
`--skip-run` measures an existing workload without owning cleanup. Otherwise
benchmark stops workloads it launches unless `--no-stop` is set.

`--resume` reuses matching progress, `--fresh` replaces it, and `benchmark resume`
resumes saved work or retries pending publication. Set both tool timeout and
`--timeout` appropriately; long runs need background execution. Report actual
artifact paths, framework summary, partial/failing cases, and any benchmark ID.
Do not add `--collect-diagnostics`: that CLI option belongs to `run`.

User request: $ARGUMENTS
