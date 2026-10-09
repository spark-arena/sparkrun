# Application profiles

An application profile defines application defaults, identity, paths and update
policy. A downstream application ships its profile in a small installed Python
package that depends on an unchanged Sparkrun wheel and selected integrations. Python imports, recipe keys, extension IDs,
registry formats and protocol keys remain shared.

CLI startup and completion measurements are recorded in
[the performance report](APPLICATION_PROFILE_PERFORMANCE.md).

The public contract is **application profile/plugin API version 1**, exposed by
`sparkrun.application.APPLICATION_PROFILE_API_VERSION`. The controller still
requires Python >=3.12; inference containers have their own interpreter.

## Entry points and startup

```python
# example_product/profile.py
from sparkrun.application import ApplicationProfile

PROFILE = ApplicationProfile(
    id="example-product",
    display_name="Example Product",
    command="example-product",
    package="example-product",
    profile_ref="example_product.profile:PROFILE",
    required_integrations=("example-hardware",),
    defaults={"defaults": {"executor": "docker"}},
    registries=(),
    telemetry_enabled=False,
    hardware_fallback="require-metadata",
)
```

The launcher must select the profile before importing the CLI:

```python
def main():
    from sparkrun.application import run_cli
    from example_product.profile import PROFILE
    run_cli(profile=PROFILE)
```

Declare `example-product = "example_product:main"` in `[project.scripts]` and
pin compatible core and integration dependencies with `dependencies = [...]`
under `[project]`. The application and any reusable integrations are separate
packages maintained outside the core repository.

For console-free use, call `sparkrun.application.initialize(profile=PROFILE,
config_path=...)`. It returns a `SparkrunContext` exposing `config`, `variables`
and `application_profile`; importing the API does not import the CLI.
`api.default_sctx()` delegates to this initializer. `UpdateSource` and
`APPLICATION_PROFILE_API_VERSION` are also exported by `sparkrun.application`;
see the [supported import map](DISTRIBUTION_API_MIGRATION.md#supported-imports). Ordinary console,
`python -m sparkrun`, and implicit API initialization select built-in Sparkrun.

One profile is active per process. Repeating that selection is idempotent;
changing it raises an error. Select before reading config, paths, feature flags,
plugins or SAF. The profile recursively copies/freezes nested defaults. The test
reset is not a production profile-switch API.

## Plugin identity and shared services

A controller is the application/profile using one canonical config directory.
All config files and processes using that application and directory share one
opaque ID. Separate config directories or applications have separate identities.
There is no controller ID setting or per-process identity selection.

Plugins can read `get_application_profile()` from `sparkrun.application`, including
inside `register(v)`. Hooks receiving a `SparkrunContext` can read
`sctx.application_profile`. These expose the full immutable application policy.
Use the smaller identity descriptors when sending ownership to another service:

```python
from sparkrun.application import get_application_identity, get_controller_identity

application = get_application_identity()     # pure: no filesystem writes
controller = get_controller_identity()       # lazy, persistent controller identity
payload = controller.to_dict()
labels = controller.labels()
# {"sparkrun.distribution": "sparkrun", "sparkrun.controller": "c-..."}
```

Context equivalents are `sctx.application_identity` and
`sctx.controller_identity`; the latter uses that context's config and is cached
for the session. The first initialization pins the canonical config path, even
when initialization was implicit. Later initialization with a different path
raises an error instead of retaining plugins selected by another config.
Child processes, including gateway autodiscovery, inherit the selected config.

`ApplicationIdentity` contains `id`, `command`, `package`, and
`resource_namespace`. It excludes profile defaults, secrets and local paths.
`get_application_identity(profile)` describes any supplied profile without
selecting it. `ControllerIdentity.to_dict()` produces a versioned envelope:

```json
{
  "schema_version": 1,
  "application": {
    "id": "jetsonrun",
    "command": "jetsonrun",
    "package": "jetsonrun",
    "resource_namespace": "jetsonrun"
  },
  "controller_id": "c-4ca78d30f5b74c53a2f6df917c8e01ab"
}
```

A shared service may deserialize any number of descriptors in one process:

```python
from sparkrun.application import ControllerIdentity

owner = ControllerIdentity.from_dict(payload)
controllers[owner.scope_key] = owner  # (application.id, controller_id)
resources[(*owner.scope_key, resource_id)] = resource
assert owner.owns(owner.labels())
```

Use `(application.id, controller_id, resource_id)` to key resources and scope
reconciliation/deletion to the appropriate controller. The same controller ID
under Sparkrun and Jetsonrun is distinct. Separate config directories of the
same application also have distinct identities. `owns(labels)` requires both ownership labels; it never
claims unlabelled legacy resources. These descriptors are provenance metadata,
not authentication. A service must authorize the identity claimed by its client.

The first controller-identity access atomically creates a private random seed in
`<canonical-config-directory>/.controllers/<application-id>.id`. The public opaque
ID is derived from that seed, the application ID, and the canonical directory.
Concurrent callers and later processes agree regardless of config filename.
Copying or moving the directory to another canonical path yields a different
controller automatically, even when the seed file is copied. Symlink aliases of
the same directory agree; restoring the seed at the original path restores that
identity. The seed and local path are not exported in the ownership envelope.
There is still one derived controller ID per application/directory, with no
selection or override setting. Creation requires a writable config directory;
later reads reuse the saved seed. An invalid seed raises an error rather than
silently replacing ownership.
Reading the application descriptor, help, or ordinary profile selection does not
generate an ID. Plugin registration should request persistent identity only when
it needs it.

New job metadata includes the controller envelope under `controller`. Existing
metadata remains readable. Plugins can use the label helper in their service
requests or resource creation. Existing executor ownership checks remain scoped
to the application; this groundwork does not make every executor enforce
controller ownership. Recipe fingerprints and intent IDs remain portable.

This enables a plugin to distinguish callers of a shared external service; it
does not add multi-controller reconciliation to SparkRoute itself. Each client
process still selects one application profile. Gateway process directories are
also application-owned: a persistent `.distribution` claim prevents another
application from overwriting an existing gateway's config, logs or state when
cache roots accidentally overlap. Choose separate cache directories for separate
managed gateway processes; those clients may still connect to one external service.

## Profile fields

| Fields | Contract |
| --- | --- |
| `id`, `display_name`, `command`, `package`, `description` | Product identity; `package` owns version reporting and self-update. |
| `documentation_url`, `support_url` | Optional reviewed product links. |
| `config_namespace`, `cache_namespace`, `state_namespace`, `resource_namespace` | Default to `id`; validated names, not paths. |
| `env_prefix`, `env_aliases` | Prefix defaults to uppercase ID with underscores; aliases are explicit per-setting sequences of environment variable names. |
| `defaults` | Site defaults, including `defaults.executor`, runtime/builder settings and `plugins.<id>` settings. |
| `integrations`, `required_integrations` | Stable installed integration IDs selected by default; required failures block launch. |
| `feature_defaults` | Boolean flag defaults applying across application release channels. |
| `feature_channel_defaults` | Application release channel → flag → boolean; overrides `feature_defaults` for that channel. Keys must name a declared update channel or `default_channel`. |
| `feature_channel` | Independent core feature maturity channel (`stable` by default); `None` follows the application update channel. |
| `registries`, `bootstrap_registry_urls` | Replacement catalog and optional bootstrap sources. An empty tuple means no shipped registries. |
| `default_channel`, `update_sources` | Channel-to-`UpdateSource(requirement, strategy)` mapping. Strategy is `package` or `git`. Requirements must name the owning package. Empty sources disable installation/self-update. |
| `telemetry_enabled`, `telemetry_endpoint`, `telemetry_key` | Usage telemetry policy. Alternate profiles default off. |
| `hardware_fallback` | `require-metadata` by default; built-in Sparkrun preserves `dgx-spark`. |
| `profile_ref` | Installed `module:attribute` reference required for child execution. |

The four namespace fields and `env_prefix` are concrete, validated strings after
construction. Omit these arguments to use identity-derived defaults; an empty
string also requests the default. Explicit `None` is rejected. This does not
change the separate `None` policies for `registries` or `feature_channel`.

`integrations`, `required_integrations`, and `bootstrap_registry_urls` accept lists
or tuples of nonempty strings, copied into immutable tuples. Integration entries
must be valid stable IDs. A bare string, mapping, or malformed member raises a
field-specific error instead of being interpreted as a sequence of characters.
Each `env_aliases` value likewise requires a list/tuple of variable names.

`None` for `registries` is reserved for built-in Sparkrun's legacy catalog.
Namespace names allow lowercase ASCII letters, digits and hyphens, beginning
with a letter, at most 48 characters. Installed package names follow Python
package naming conventions.

## Settings and provenance

Existing setting-specific recipe/cluster/CLI precedence is preserved. Application profile
defaults sit below explicit user configuration at the site-default tier. An
explicit runtime or executor ID continues to identify the same implementation.
Nested nonempty mappings merge by key; lists replace; explicit empty mappings
replace; false and null remain explicit values interpreted by each setting's
schema. There is no generic list concatenation.

`SparkrunConfig.effective_data` is a fresh merged view. `set()` and `save()` operate
on user data, so changing a setting does not serialize the entire profile into
`config.yaml`. `setting_source(key)` reports `config`, `distribution` or `baseline`;
feature diagnostics also identify application profile defaults.

### Feature defaults and channels

Application policy follows the configured `self_update.channel`, falling back to
`default_channel`. It does not infer a channel from a package version or Git
checkout. A profile with `stable` and `next` update sources can declare:

```python
feature_defaults={"cli.tune": False, "gateway.sparkroute": False},
feature_channel_defaults={
    "next": {"gateway.sparkroute": True, "gateway.litellm": False},
},
feature_channel="stable",
```

Here `next` enables and automatically selects SparkRoute, while `cli.tune` stays
off and unspecified flags use upstream stable defaults. `features.channel: alpha`
changes upstream fallback behavior; it does not select the application's `next`
policy. Without a configuration object, feature resolution uses `default_channel`
for application policy. The explicit `channel=` argument to the feature resolver
continues to select only core feature maturity.

Precedence, highest first:

1. Product-specific feature environment override.
2. Explicit `features.<flag>` user configuration (literal dotted flag keys).
3. Profile default for the application release channel.
4. Profile baseline `feature_defaults`.
5. Upstream default for the core feature channel.
6. Upstream flag baseline; unknown flags default off.

Channel mappings are frozen with the profile, and false is an explicit default.
`setup features list` reports `distribution-channel` when application channel
policy wins. Its JSON retains `channel` for core maturity and adds
`application_channel` for the application's release track.

The `cli.tune` flag defaults on in Sparkrun. Disabling it hides `tune` from root
help and completion and blocks tuning commands. The `integration.arena` flag
gates the Arena commands, benchmark options and lifecycle; it also defaults on.
Application profiles can set either flag to false. Explicit user configuration
or `<command> setup features enable <flag>` overrides an application default.

`integration.k8s` loads the Kubernetes plugin. Its core defaults are off for
stable/beta and on for alpha. Only an enabled plugin registers `executor.k8s`,
`cli.setup.k8s`, and `api.run.k8s`; those child features default on within it.
Application profiles can override the loading default independently of the core
feature channel.

Installed integration selection is by stable ID:

```yaml
integrations:
  example-hardware: true
  another-integration: false
```

False persists across profile upgrades. Disabling a required integration remains
visible in diagnostics and blocks launch. Installed metadata is enumerated before
import, so an unselected integration never executes merely because it is installed.
See [PLUGINS.md](PLUGINS.md) for the registration contract.

## Paths, environment and children

Local defaults remain `~/.config/<config_namespace>` and
`~/.cache/<cache_namespace>`; state defaults to `~/.config/<state_namespace>`.
This is not an XDG migration for existing Sparkrun users. Explicit config paths
and product `*_CONFIG_DIR`, `*_CACHE_DIR`, `*_STATE_DIR` overrides are honored.
The SAF adapter uses the product identity and a product-specific stateful-root
override. Compatibility `DEFAULT_*` constants remain for older callers; new code
must call the resolvers in `core.config` or `core.application_profile` at use time.

Remote defaults use the target's home. Where an absolute path is needed,
`probe_remote_sparkrun_cache()` evaluates the product cache override and target
XDG cache environment on that host. Never substitute the controller's home into
a remote path. HuggingFace caches continue using HuggingFace's own shared policy.

Alternate products do not automatically inherit `SPARKRUN_*` overrides. The
intentional shared controls are test containment (`SPARKRUN_NO_TELEMETRY`,
`SPARKRUN_NO_EXTERNAL_PLUGINS`, `SPARKRUN_NO_INSTALLED_PLUGINS`) and the internal
child identity variables `SPARKRUN_APPLICATION_PROFILE` and
`SPARKRUN_APPLICATION_CONFIG` (the latter only for an explicitly supplied
same-controller config path). Product feature overrides
use `<PREFIX>_FEATURE_<DOTTED_NAME_WITH_UNDERSCORES>`.
Plugin-owned controls such as `SPARKROUTE_BINARY` are shared across profiles.

`child_environment()` supplies the importable profile reference. For a child on
the same controller, `child_environment(config_path=...)` also propagates a
selected configuration file. Do not pass a controller-local path into a remote
launcher image. Re-entered
Python processes restore it before resolving paths or plugins. Core gateway
children, systemd exports and Kubernetes launcher Jobs propagate this identity.
A remote installation or launcher image must contain the distribution package,
its compatible core and required integrations. A missing profile import or a
missing/incompatible required integration fails instead of becoming Sparkrun.
The internal reference is trusted local bootstrap input, not a remote manifest.

## Registry and update policy

An alternate catalog replaces Sparkrun's fallback and bootstrap catalog, including
reconciliation and migration paths. Entries use existing URL/subpath/name
validation and reserved-name rules: `official` is not available for an unrelated
recipe source. User entries and persisted remove/untrust decisions survive reload
and restoration. A changed source does not inherit trust from a previously trusted
source. A materialized user override continues to win over a newer profile.

Setting both `registries=()` and `bootstrap_registry_urls=()` supplies an empty
catalog. This also applies to `registry update` (including restoration) and
`registry revert-to-defaults`; an empty catalog stays empty. Sparkrun retains its
existing catalog. Selecting a hardware plugin does not change application policy.

To ship an application-specific catalog, declare entries in its profile:

```python
registries=(
    {
        "name": "product-recipes",
        "url": "https://git.example.org/team/product-recipes.git",  # replace with your repository
        "subpath": "recipes",
        "benchmark_subpath": "benchmarking",
        "trusted": False,
    },
),
bootstrap_registry_urls=(),
```

This **replaces**, rather than extends, the Spark catalog. Registry names must be
unique within the profile. The catalog is frozen, and entries are validated with
the regular registry name, Git URL, and subpath rules when resolved. Optional
`bootstrap_registry_urls` performs first-run discovery only from the application's
listed sources; it never adds Sparkrun's bootstrap URLs. Explicit catalog entries
are live defaults, not serialized into the user's configuration by `registry
update`. `registry show` identifies their application profile as the source.
Unfinished discovery can be retried with an explicit registry update; local
catalog configuration edits preserve that pending work without using the network.
See [catalog registry changes](CATALOG_API.md#import-lifetime-and-registry-changes)
for retry, empty-inventory, and failure behavior.

Users can still add their own registries, including a Spark registry deliberately.
An empty profile catalog does not delete those entries. Plugin-contributed
registries remain a separate overlay for enabled plugins and retain normal
trust/override rules. The example hardware plugin contributes no registry until
qualified recipes are available. With no registry-provided benchmark profiles,
use a local recipe/profile; enabling Arena alone does not install its official
benchmark catalog.

An application profile may explicitly declare reviewed trusted sources. Selecting or
installing a package does **not** elevate the registries its plugin hook declares;
those use external-plugin trust policy. Registry discovery never runs executable
recipe hooks by itself.

Self-update verifies both the exact normalized owning tool package and the
running environment against `uv tool list --show-paths`. Upgrade, install and
post-update probes target that distribution, using its configured source and
its executable inside the current environment. Channel names do not imply Git:
a downstream beta channel can use a package source. Unsupported channels cannot
fall back to Sparkrun's source. A source must use a named requirement for the
owning distribution package, including for VCS URLs. Arbitrary declared channel
names such as `stable` and `next` resolve correctly internally. The current CLI
still offers fixed `--stable`/`--beta`/`--alpha`/`--yolo` switches: updating an already
configured `next` works, but selecting it needs a generic CLI channel selector
before the full custom-channel workflow is exposed.

`setup version` displays the application package/version, profile ID, release
channel, core package/version, installed commits when available, and loaded plugin
versions with their package/source. Plugins without a declared version show
`unknown` rather than inheriting the core version. The `--json` form exposes the
same diagnostics as structured data for tooling and self-update.

`setup version --json` retains top-level `version`, `commit`, `channel` for the
running distribution and adds separate `distribution`, `core` and loaded `plugins`
records. Package sources compare versions; Git sources compare commits. An
unchanged identity is not reported as an upgrade. Downstream update channels do
not enable upstream experimental features unless `feature_channel` explicitly
chooses that maturity policy.

## Coexistence and legacy behavior

Recipe intent/fingerprints remain portable. Generated deployment names use the
resource namespace; job metadata stores `distribution` and `resource_namespace`.
Executors carry the shared `sparkrun.distribution` label or a native pidfile owner
record. Explicit owner values take precedence over names. Status exposes owned
workloads, while foreign canonical Docker workloads still consume placement slots;
physical telemetry and occupied ports remain shared host facts.

Stop and teardown check persisted ownership. Metadata listing, removal and prune
exclude foreign owners even when a cache is explicitly shared. Kubernetes objects,
queues, service accounts and RBAC defaults use the resource namespace. Systemd
units carry an owner marker, and uninstall checks it. Setup manifests carry an
owner too. Cross-distribution management is not exposed.

Missing ownership is legacy Sparkrun only when the existing job name, metadata
format/location, service description or setup-manifest format establishes that
provenance. Arbitrary unlabeled containers are not claimed. The setup wizard
uses application step defaults and measured target hardware, rather than the
application ID, to select supported changes. Alternate applications require an
owned setup manifest for uninstall; teardown never guesses which host changes
belong to them. Legacy standalone CX7/earlyoom commands retain their Spark-only
guards; the shared wizard applies its hardware and executor prerequisites.

## Hardware-aware setup

`setup wizard` and `setup check` use the shared `core.setup_steps` plan. Each host
is probed through the hardware integration API, then its executor is resolved
from that hardware and cluster/config settings. The matching hardware platform
explicitly names the core/plugin steps supported by that executor; application
features can narrow this selection but cannot add unsupported steps. Optional
probes run only after that plan is selected. Wizard runs persist successfully
identified hardware in the cluster. Unknown or unreachable targets cannot receive
host actions. Recipe-specific executor overrides still need launch-time checks.

`cli.setup.wizard` controls wizard visibility and invocation. When disabled,
bare `setup` shows help even when no default cluster exists. Step policy uses the
normal feature precedence, including application release-channel defaults:

```python
feature_defaults={
    "setup.steps.cx7": False,
    "setup.steps.rdma": False,
    "setup.steps.nvidia_cdi": False,
    "setup.steps.earlyoom": False,
    "setup.steps.sudoers": False,
},
```

The remaining built-in step IDs are `docker`, `docker_group`, `nvidia_container`,
`host_ipc`, and `ssh_mesh`. Hardware identification is mandatory. Disabled and
inapplicable steps do not count as readiness gaps; failed or disabled prerequisites
block dependent actions. Explicit enabling never bypasses hardware-plan membership
or applicability. Hardware integrations own support; application profiles own
preferences. See [SETUP_STEPS.md](SETUP_STEPS.md) for the opt-in plan declaration.

The wizard does not install the controller CLI: use `setup install` explicitly.
`--dry-run` does not probe hosts, apply actions, install software, or save cluster
and manifest changes. It uses any saved inventory and marks unknown prerequisites
for discovery on a real run. `setup check --json` is a read-only JSON report with
checks and per-host step selection/reasons.

Plugin registration and help/completion do not perform target probes. See
[Setup steps](SETUP_STEPS.md) for the console-free extension API.

## Optional gateway integrations

SparkRoute is installed and updated separately; it is not included in the
Sparkrun wheel. Alternate applications must select a plugin release that supports
their application profile API and install the application's console in the same
Python environment for gateway callbacks. Consult the
[SparkRoute plugin documentation](https://github.com/sparksq/sparkrun-sparkroute-plugin)
for supported profiles and configuration.

## Validation

The installable fixture has no fabricated partner URL, registry, update source,
image or telemetry endpoint. Offline wheel tests install the core first, hash its
installed files, then install the downstream package and shared plugin and verify
that the core files are unchanged. Fresh processes exercise both launchers,
help/version/completion, separate roots, explicit plugin selection and child API
initialization. Run with:

```bash
SPARKRUN_TEST_WHEELS=1 .venv/bin/python -m pytest tests/test_application_profile_wheels.py -q
.venv/bin/python -m pytest tests/test_application_profiles.py -q
```

`uv` and cached build/runtime dependencies are required for the offline wheel test.
The ordinary suite skips that install test unless explicitly enabled.

The wheel suite uses minimal private application/plugin fixtures under
`tests/fixtures/application_profiles`. They exercise package composition and
namespace isolation. Hardware detection and device qualification belong to the
integration that supplies that hardware support; these tests make no claim about
physical-device compatibility. See [MULTIPLATFORM.md](MULTIPLATFORM.md) for the
platform and probe extension contracts.


### Initialization failures and embedding

Call `initialize()` once before starting application workers. A successful call
pins the process to one profile and canonical configuration path. Repeating that
binding reuses the registry; independent applications/configurations use separate
processes. `variables=` is a first-initialization injection point: later calls
may omit it or pass that same instance; a different instance is rejected. This
does not promise that individual API operations are thread-safe. Sequential
operations can reuse the context without carrying the previous cluster's SSH
user forward: connection overrides live in operation-specific configuration
views, while the configured fallback account and application/controller binding
remain shared.

Plugin bootstrap can partially register process-global objects before an error.
If initialization raises, fix the configuration/plugin and start a new process.
Subsequent initialization or registry access in the failed process raises an
explicit restart-required error with the original cause. It does not silently
return an API-ready context with missing plugins.

[Benchmark API usage](BENCHMARK_API.md) covers callback-driven progress, explicit
decisions, results, and publication retries for applications with their own UI.
