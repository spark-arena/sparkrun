import re


def default_env_hf_offline(env: dict[str, str] = None, **kwargs) -> dict[str, str]:
    return {
        # DEFAULT: disable online HF/transformers checks -- we've already copied all data locally!
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        **(env or {}),
        **kwargs,
    }


# `--api-key value` or `--api-key=value`.  Stops at whitespace, line
# continuation backslash, or shell metacharacters that wouldn't appear
# in a real key.  Both vLLM and SGLang use the same flag spelling.
_API_KEY_RE = re.compile(r"--api-key(?:=|\s+)([^\s\\]+)")


def parse_api_key_from_command(command: str | None) -> str | None:
    """Extract a literal ``--api-key <value>`` value from a serve command.

    Returns ``None`` when no flag is present, or when the value is a
    ``{placeholder}`` (the defaults path handles those).  Surrounding
    quotes are stripped.
    """
    if not command:
        return None
    match = _API_KEY_RE.search(command)
    if not match:
        return None
    val = match.group(1).strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        val = val[1:-1]
    if not val:
        return None
    if val.startswith("{") and val.endswith("}"):
        return None
    return val
