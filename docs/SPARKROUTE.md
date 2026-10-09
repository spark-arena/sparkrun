# Optional SparkRoute integration

SparkRoute is not bundled with Sparkrun. Install and update the plugin separately
using the [SparkRoute plugin instructions](https://github.com/sparksq/sparkrun-sparkroute-plugin).
ColdSnap is also distributed separately through its
[plugin repository](https://github.com/sparksq/sparkrun-coldsnap-plugin).
OCI Relay remains bundled with Sparkrun.

## Choose SparkRoute

LiteLLM is the default gateway on stable, beta, and alpha. After installing a
compatible SparkRoute plugin in the Python environment running Sparkrun:

```sh
sparkrun setup features enable gateway.sparkroute
sparkrun proxy start --gateway sparkroute --host 127.0.0.1
```

Enabling a feature flag does not install a plugin. The SparkRoute flag defaults
on for alpha when the plugin is available; selecting it still requires an
explicit `--gateway sparkroute` or `proxy.gateway` setting in `proxy.yaml`.
ColdSnap's `plugins.coldsnap` flag remains disabled by default.

`--gateway` persists the choice. An existing SparkRoute pin is not silently
replaced when its plugin is absent. To select the bundled gateway instead:

```sh
sparkrun proxy start --gateway litellm
```

Use `--restart` to replace an already running gateway. Process-level status and
stop remain available for a recorded gateway whose plugin is missing; querying
its models or changing its configuration requires reinstalling the plugin.
See [proxy management](PROXY.md) for details.
