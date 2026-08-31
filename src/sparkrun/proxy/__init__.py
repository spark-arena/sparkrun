"""sparkrun proxy — unified OpenAI-compatible gateway for inference endpoints."""

DEFAULT_PROXY_PORT = 4000
DEFAULT_PROXY_HOST = "0.0.0.0"
DEFAULT_MASTER_KEY = None  # No auth by default — avoids LiteLLM DB requirement
DEFAULT_DISCOVER_INTERVAL = 30
