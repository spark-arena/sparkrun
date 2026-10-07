# Model cache validation and distribution

Manifest preparation binds a model's endpoint, immutable commit, selected files,
checksums and runtime paths together. Core validates every required host before
launch; a download, rsync exit code or plugin success receipt is insufficient.

## Default behavior and policy

No configuration change is needed to adopt manifests. The default `legacy`
policy uses metadata validation with compatibility fallback. Optional settings
in `config.yaml`:

```yaml
model_cache_validation: legacy        # legacy (default) | metadata | checksum
model_file_selection: auto            # auto | all | safetensors
model_distribution_provider: auto     # auto | builtin | registered provider name
# model_distribution_fallback: true   # optional override of provider policy
```

Both `legacy` and `metadata` resolve authoritative inventories, select required
files, check existence/type/readability/size, and persist manifests automatically.
Manifests are stored in `.sparkrun-model-artifacts/` under the HF cache home on
the inventory source and launch targets. Lookups cover both the requested ref and
resolved commit, so a later pinned recipe can reuse an inventory learned from an
unpinned launch. Inventory is saved before acquisition/repair, so even an
interrupted or failed preparation retains the expected file list for a later
offline run. Stored manifests describe expected files; they never replace
checking the current files or certify that a download completed.

The difference is what happens when inventory is unavailable:

- `legacy` uses saved inventory if available, warning when an online mutable-ref
  refresh fails. Without inventory it warns that the cache is unverified and uses
  the previous cache/download checks. This lets existing offline caches keep
  working and migrate automatically on a later successful online preparation.
- `metadata` requires authoritative inventory; missing inventory or a failed
  online mutable-ref refresh fails preparation.
- `checksum` has the same inventory requirement and additionally reads selected
  payloads to check authoritative SHA-256 or Git blob SHA-1 hashes. Missing
  digests are unverified, never successful checksum checks.

Every policy rejects invalid manifests and requires repair or failure for known
missing/damaged files. Repair, transport and workload access failures never
downgrade to the old checks. Selecting a model provider requires the strict
inventory contract even if the configured validation policy is `legacy`.

Named clusters can override those settings without changing global configuration:

```yaml
distribution:
  model:
    validation: metadata
    file_selection: auto
    provider: builtin
    # fallback: false
```

Existing `enabled`, `preserve_perms` and `skip_fan_out` remain compatible with the
compatibility fallback. Manifest preparation reobserves each host even on shared storage;
already-complete destinations need no copy. It does not recursively chown caches.
Its snapshot copies preserve times when supported, without preserving ownership
or permissions. Permission failures fail preparation rather than triggering a
large redownload under a different identity.

## Select files without redundant weights

With inventory available, the default `auto` selection reads the root
`model.safetensors.index.json` at the resolved commit and
includes every shard named by its `weight_map`. It excludes unreferenced root
checkpoints and recognized `original/` and `metal/` alternative weight payloads.
Configuration, tokenizer, license and other ancillary files remain included.
Unfamiliar nested component trees remain included to avoid discarding separate
vision/encoder weights. No shard count is inferred from safetensors filenames.
Without a root index, the complete repository inventory is selected.

This addresses [#142](https://github.com/spark-arena/sparkrun/issues/142) without
checking for a GPT-OSS model name. At GPT-OSS-120B commit
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, metadata and the index select 15 root
shards: about **65.28 GB instead of 195.76 GB**, avoiding about 130.49 GB of
alternative weights. This measurement uses inventory sizes, not a weight download.

Use `all` for a loader requiring additional weight formats; `safetensors` requires
a root index and fails if none exists. GGUF uses its quantization selection,
checks multipart completeness, and binds its selected weight and projector from
that same revision. It never picks an arbitrary cached snapshot at launch.

## Inspect, migrate and repair

These commands print structured JSON reports; inspection/import return nonzero
when a target is incomplete or unverified. `--host` is repeatable and uses the
configured SSH credentials; `--cache-dir` is the target's HF home, not `hub/`.

```sh
# Metadata only, offline and read-only; no recursive cache traversal.
sparkrun model-cache inspect org/model --revision COMMIT --cache-dir /models/hf

# Resolve authoritative inventory online (plus a small index, at most 16 MiB).
sparkrun model-cache resolve org/model --revision COMMIT --output manifest.json

# Import trusted inventory for an existing/offline cache. Import is not repair.
sparkrun model-cache import org/model --revision COMMIT --manifest manifest.json

# Repair exactly the missing/wrong-size files and revalidate each target.
sparkrun model-cache repair org/model --revision COMMIT --host node1 --host node2

# Explicit checksum audit/repair, including same-size corruption.
sparkrun model-cache inspect org/model --revision COMMIT --checksum
sparkrun model-cache repair org/model --revision COMMIT --checksum
```

Pass the same `--file-selection` and GGUF `--projector` used for preparation when
inspecting or importing. `repair` accepts `--transfer-mode`, `--local-cache-dir`,
`--offline` and `--dry-run`. Offline preparation uses saved/imported inventory and
existing controller/cluster files; it never calls the Hub or installs tools. A
manifest-less offline cache is **unverified**, not implicitly valid or corrupt;
only `legacy` launch/distribution may proceed using the previous presence checks.
Dry-run makes no metadata requests, probes or cache writes. Origin resolution
for mutable refs is online even when the payload is already cached. A pinned
commit with saved inventory needs no origin request.

The Python surface is `sparkrun.api.model_cache`: `resolve_manifest`,
`inspect_model`, `import_manifest`, and `repair_model`, with typed requests,
manifests, reports and prepared bindings. `api.materialize()` remains read-only.

For a pre-placed directory, put the trusted manifest at
`<directory>/.sparkrun-model-manifest.json`, with inventory paths relative to that
directory. For a pre-placed GGUF file, use
`<file>.sparkrun-model-manifest.json`, paths relative to its parent, and set
`weight_path` to that filename. These inventories are checked before launch,
including under the default policy; missing inventories produce an explicit unverified
migration warning. Core does not acquire or repair pre-placed paths automatically.

## Transfer and launch guarantees

Local, push, delegated and per-node pull share preparation. Delegated acquisition
runs on the designated head and need not contact the Hub from the controller.
Each distribution entry has its own revision, including speculative draft models.
Required metadata failures are separate from advisory metadata budgets. Remote
acquisition uses the pinned `huggingface_hub==1.8.0` adapter; offline observation
requires only Bash and ordinary Linux utilities, not Python or uv on targets.

Copies enumerate selected snapshot entries and dereference them into regular
snapshot files. They exclude other revisions, live blob-directory churn, locks,
partial downloads and unrelated cache state. This representation works alongside
classic and shared-blob HF layouts, without copying both snapshots and blobs.
Repair stages one affected file and atomically replaces its snapshot entry; it
never deletes a shared canonical blob. Copied files are not skipped based only
on equal sizes when a checksum repair has identified corruption.

Core holds per-cache/repository `flock` leases across acquisition and provider
calls. An operation nonce recognizes leases reached through shared-cache aliases;
complete aliases are omitted from transfer after observation. External downloaders
and garbage collectors do not participate in these leases. A final observation is
still required, and no receipt promises protection against later external changes.

The reproduced self-copy defect underlying the PR #308 review is also guarded in
the existing local rsync wrapper using directory identity, including symlink
aliases. Vanished-source retries are bounded to one and require a still-complete
manifest source. Diagnostics from the first failure remain logged, the original
time budget remains in force, and target validation is mandatory after success.
`SPARKRUN_NO_RSYNC_RETRY=1` disables retries. The resilience idea follows
[Aisoipheo's PR #308](https://github.com/spark-arena/sparkrun/pull/308); this does
not import that PR's observed-file verifier or repo-wide success marker.

Standard vLLM and SGLang commands consume the prepared snapshot while retaining
the recipe's served model name and immutable revision. Draft revision pins remain
independent. Custom commands must consume `{model}` or the validated revision;
a literal mutable ref cannot bypass the binding. Env `launch.model_path` and
`launch.model_revision` consume the same binding. Hooks additionally receive
`SPARKRUN_MODEL_PATH` (workload path, blank for controller post-commands) and the
resolved `SPARKRUN_MODEL_REVISION`. Compilation/runtime caches remain separate
and are not distributed with model payloads.

Docker access checks run the selected image with its final UID and structured
mounts, without GPU allocation or the image entrypoint. Local checks use the
host execution identity. Unavailable or shadowed mounts fail before eviction.
Custom executors need an equivalent `model_access_command` capability; raw mount
options cannot bypass mapping checks. Delegating runtimes that own preparation
retain that responsibility. ColdSnap native restores still bypass HF preparation
through their asset policy; capture/safetensors preparation uses the normal path.

## Rollout evidence and remaining acceptance

Tiny real-rsync fixtures cover missing files, dangling shared links, aliases,
same-size repair, offline reuse and all four modes. Provider tests reject false
success, changed identities and incomplete outcomes. Workload tests cover pinned
commands, env reuse, UID options and mount shadowing.

Local warm preparation (three samples, two observations per host) measured median
41 ms / 60 ms / 186 ms for 10 / 100 / 1,000 selected files using saved pinned
inventory under the default compatibility policy. It hashes no weights
and batches metadata calls; these are local filesystem measurements, not NFS or
multi-node startup promises. Actual NFS locking/latency and large-model startup
remain deployment acceptance checks before removing the default compatibility fallback.
