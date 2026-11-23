from typing import TYPE_CHECKING

from sparkrun.utils.command_flags import command_flag_value

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe


def ptrace_executor_config() -> dict[str, object]:
    """Return the ``default_executor_config`` fragment that keeps a hung engine inspectable.

    The one recurring way to diagnose a wedged inference server is to attach a
    stack sampler to the stuck process (``py-spy dump --pid``, ``gdb -p``,
    ``pystack``).  Every runtime that uses this is multi-process — vLLM's API
    server plus its EngineCore children, SGLang's scheduler/detokenizer/TP
    workers, TRT-LLM's ``mpirun`` ranks — so the sampler is never a
    *descendant* of its target, and Yama's default ``ptrace_scope=1`` therefore
    requires ``CAP_SYS_PTRACE``, which is not in Docker's default capability
    set.

    The capability is added to the **container's** set, so it is available to a
    ``docker exec -u root <container> py-spy dump --pid N`` even though the
    workload itself runs as an unprivileged user (``auto_user``): a non-root
    process gets no permitted capabilities from ``--cap-add`` (Docker sets no
    ambient caps), but an exec that *is* root draws from the container's set.
    That exec is how an operator actually reaches a hung engine, which is why
    this is worth carrying by default.

    Notably weaker than the other privilege keys gated by the launcher:
    ``CAP_SYS_PTRACE`` confers no access outside the container's PID namespace,
    so it does not undo the rootless hardening the way ``privileged`` or
    ``cap_add: [SYS_ADMIN]`` would.

    Returned from :meth:`~sparkrun.runtimes.base.RuntimePlugin.default_executor_config`
    (chain layer 7) rather than appended as a raw flag by
    ``get_extra_docker_opts``, so that every layer above it — cluster, recipe,
    ``-o cap_add=…`` — can override or drop it, and so it reaches any executor
    that understands :class:`~sparkrun.orchestration.executors._base.ExecutorConfig`
    rather than docker alone.  Note the list *replaces* rather than merges, so
    a recipe setting ``cap_add`` must re-list ``SYS_PTRACE`` to keep it.
    """
    return {"cap_add": ["SYS_PTRACE"]}


def default_env_hf_offline(env: dict[str, str] | None = None, **kwargs) -> dict[str, str]:
    return {
        # DEFAULT: disable online HF/transformers checks -- we've already copied all data locally!
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        **(env or {}),
        **kwargs,
    }


def parse_flag_value_from_command(command: str | None, flag: str) -> str | None:
    """Extract a literal ``<flag> <value>`` or ``<flag>=<value>`` from a command.

    ``None`` when *flag* is absent, bare, or set to a ``{placeholder}`` (the
    defaults path handles those). Used to recover an api/auth key from recipes
    that embed it directly in their ``command:`` text rather than going through
    ``defaults``. See :func:`sparkrun.utils.command_flags.command_flag_value`.
    """
    return command_flag_value(command, flag)


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
