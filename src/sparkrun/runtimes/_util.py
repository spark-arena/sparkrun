import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe


def default_env_hf_offline(env: dict[str, str] = None, **kwargs) -> dict[str, str]:
    return {
        # DEFAULT: disable online HF/transformers checks -- we've already copied all data locally!
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        **(env or {}),
        **kwargs,
    }


def parse_flag_value_from_command(command: str | None, flag: str) -> str | None:
    """Extract a literal ``<flag> <value>`` or ``<flag>=<value>`` from a command.

    Matches the value up to the next whitespace or line-continuation
    backslash.  Returns ``None`` when *flag* is absent, when the value
    is empty, or when the value is a ``{placeholder}`` (the defaults
    path handles those).  Surrounding quotes are stripped.

    Used to recover an api/auth key from recipes that embed it directly
    in their ``command:`` text rather than going through ``defaults``.
    """
    if not command:
        return None
    pattern = re.escape(flag) + r"(?:=|\s+)([^\s\\]+)"
    match = re.search(pattern, command)
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


def parse_api_key_from_command(command: str | None) -> str | None:
    """Backward-compatible alias for ``parse_flag_value_from_command(command, "--api-key")``."""
    return parse_flag_value_from_command(command, "--api-key")


def resolve_api_key(
    recipe: "Recipe",
    overrides: dict | None,
    env_var: str,
    flag_name: str = "--api-key",
) -> str | None:
    """Resolve a runtime's api-key value via the shared priority chain.

    Order: CLI override (``overrides["api_key"]``) → ``recipe.defaults["api_key"]``
    → ``recipe.env[env_var]`` → literal ``flag_name`` parsed from the recipe's
    ``command`` field.  Returns ``None`` when none are set.

    Args:
        recipe: The loaded recipe.
        overrides: Optional CLI overrides dict.
        env_var: Recipe ``env`` key consulted after defaults (e.g. ``"VLLM_API_KEY"``,
            ``"SGLANG_API_KEY"``, ``"LLAMA_API_KEY"``).
        flag_name: CLI flag whose inline value is parsed from
            ``recipe.command`` as a last resort.  Defaults to ``"--api-key"``.
    """
    if overrides:
        val = overrides.get("api_key")
        if val:
            return str(val)
    val = recipe.defaults.get("api_key")
    if val:
        return str(val)
    val = recipe.env.get(env_var)
    if val:
        return str(val)
    parsed = parse_flag_value_from_command(recipe.command, flag_name)
    if parsed:
        return parsed
    return None
