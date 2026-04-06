def default_env_hf_offline(env: dict[str, str] = None, **kwargs) -> dict[str, str]:
    return {
        # DEFAULT: disable online HF/transformers checks -- we've already copied all data locally!
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        **(env or {}),
        **kwargs,
    }
