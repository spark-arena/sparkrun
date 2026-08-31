"""sparkrun proxy — unified OpenAI-compatible gateway for inference endpoints."""

DEFAULT_PROXY_PORT = 4000
DEFAULT_PROXY_HOST = "0.0.0.0"
DEFAULT_MASTER_KEY = None  # No auth by default — avoids LiteLLM DB requirement
DEFAULT_DISCOVER_INTERVAL = 30
# Consecutive sweeps an endpoint must be absent before auto-discover removes
# it. A single failed health probe is not evidence a workload is gone, and
# evicting it costs a gateway restart plus a window of 404s for a model that
# is serving fine. 1 restores the historical remove-on-first-miss behaviour.
DEFAULT_DISCOVER_REMOVAL_GRACE_SWEEPS = 2
