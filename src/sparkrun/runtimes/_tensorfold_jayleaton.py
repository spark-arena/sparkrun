"""Jayleaton Zig engine variant for the TensorFold runtime.

The Jayleaton port is a different TensorFold fork from the bertholomus TP2
engine the rest of this runtime targets: it serves the EXL3 checkpoint through
two binaries (``tensorfold-dsv41 serve`` on rank 0 and ``tf-dsv41-m1 follow``
on rank 1), keeps per-rank Engram and prepared directories on local NVMe, and
takes a large ``TF_DSV41_*`` serving profile.  This module owns that engine's
command, env, validation and startup order; :class:`TensorFoldRuntime` exposes
it as ``tf_family: jayleaton``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from sparkrun.core.validation import RecipeIssue
from sparkrun.runtimes.base import RuntimePlugin, render_flag_value
from sparkrun.utils.data import integer_setting
from sparkrun.utils.shell import quote

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe

FAMILY = "jayleaton"
WORLDS = frozenset({2})
DEFAULT_COMMAND = "/opt/tensorfold/bin/tensorfold-dsv41"
DEFAULT_FOLLOWER = "/opt/tensorfold/bin/tf-dsv41-m1"
DEFAULT_TF_PORT = 29571

OWNED_KEYS = frozenset(
    {
        "command_binary",
        "follower_binary",
        "tf_port",
        "assets_dir",
        "prepared_dir",
        "state_dir",
        "cache_dir",
        "engram_dir",
        "bias_vl",
        "tf_env",
    }
)

_ENV_PATHS = {
    "TF_DSV41_ASSETS": "/assets",
    "TF_DSV41_ENGRAM_DIR": "/engram",
    "TF_DSV41_PREPARED": "/prepared",
    "TF_DSV41_STATE_DIR": "/state",
    "TF_DSV41_CALIB_DIR": "/state/calib-zig",
    "TF_DSV41_REQUEST_LOG": "/state/requests.jsonl",
    "TF_DSV41_SESSION_DISK": "/sessions",
    "TF_DSV41_PF_DENSE_TABLE": "/cache/pfdense-table.json",
}


def owned_keys() -> frozenset[str]:
    return OWNED_KEYS


def is_jayleaton(config) -> bool:
    return config.get("tf_family") == FAMILY


def _tp(config) -> int:
    value = config.get("tensor_parallel")
    if value is None:
        return 1
    try:
        return max(1, integer_setting(value, key="tensor_parallel"))
    except (TypeError, ValueError):
        return 1


def world_size(parallelism, config) -> int:
    return _tp(config) if _tp(config) > 1 else 1


def start_order(config) -> str:
    return "workers_first" if _tp(config) > 1 else "head_first"


def rendezvous_port(config, init_port: int) -> int:
    value = config.get("tf_port")
    if value is None:
        return DEFAULT_TF_PORT
    try:
        return int(value)
    except (TypeError, ValueError):
        return init_port


def rendezvous_port_span(config) -> int:
    return 1


def _model_path(config) -> str:
    model = config.get("_prepared_model_path") or config.get("resolved_model_path")
    if not model:
        raise ValueError("the jayleaton family needs a pre-placed flash17 pack path")
    return str(model)


def _path(config, key: str, default: str) -> str:
    value = config.get(key)
    return str(value) if value not in (None, "") else default


def _env(config, rank: int, head_ip: str, init_port: int) -> Iterator[str]:
    values = {
        "TF_TP_WORLD": 2,
        "TF_TP_RANK": rank,
        "TF_TP_DEVICE": 0,
        "TF_TP_MASTER": head_ip,
        "TF_TP_PORT": init_port,
        "TF_COMM_BACKEND": "roce",
        "TF_TP_KERNEL_IMAGE": "/opt/tensorfold/share/tp_mailbox.fatbin",
        **_ENV_PATHS,
    }
    for key, value in values.items():
        yield "%s=%s" % (key, quote(str(value)))
    extra = config.get("tf_env")
    if isinstance(extra, dict):
        for key, value in extra.items():
            if isinstance(key, str) and value is not None:
                yield "%s=%s" % (key, quote(str(value)))


def rank0_command(config) -> str:
    binary = config.get("command_binary") or DEFAULT_COMMAND
    model = _model_path(config)
    parts = [
        str(binary),
        "serve",
        quote(model),
        "--host",
        render_flag_value(config.get("host") or "0.0.0.0"),
        "--port",
        render_flag_value(config.get("port") or 8000),
        "--parallel",
        render_flag_value(config.get("max_num_seqs") or 4),
        "--context",
        render_flag_value(config.get("max_model_len") or 1048576),
        "--name",
        render_flag_value(config.get("served_model_name") or "deepseek-v4.1-flash"),
        "--max-tokens",
        render_flag_value(config.get("max_tokens") or 32768),
        "--no-update-check",
    ]
    return " ".join(parts)


def follower_command(config) -> str:
    binary = config.get("follower_binary") or DEFAULT_FOLLOWER
    return " ".join([str(binary), "follow", quote(_model_path(config)), "/assets"])


def generate_node_command(
    recipe: "Recipe",
    config,
    *,
    head_ip: str,
    num_nodes: int,
    node_rank: int,
    init_port: int,
) -> str:
    if num_nodes not in WORLDS:
        raise ValueError("the jayleaton family supports TP=2 only (got %d nodes)" % num_nodes)
    _model_path(config)  # validation only; the command uses the /model mount
    env = list(_env(config, node_rank, head_ip, init_port))
    command = rank0_command(config) if node_rank == 0 else follower_command(config)
    return " ".join(env + [command])


def generate_command(recipe: "Recipe", config, *, is_cluster: bool) -> str:
    if not is_cluster:
        raise ValueError("the jayleaton family is multi-node only (TP=2)")
    _model_path(config)
    return rank0_command(config)


def validate(recipe: "Recipe", config, runtime: RuntimePlugin) -> list[str | RecipeIssue]:
    issues: list[str | RecipeIssue] = list(RuntimePlugin.validate_recipe(runtime, recipe))
    dims = {}
    for key in ("tensor_parallel", "pipeline_parallel", "data_parallel", "expert_parallel"):
        value = config.get(key)
        try:
            dims[key] = max(1, integer_setting(value, key=key)) if value is not None else 1
        except (TypeError, ValueError):
            dims[key] = 1
    if dims["tensor_parallel"] != 2:
        issues.append(runtime.recipe_error("the jayleaton family is TP=2 only"))
    for key in ("pipeline_parallel", "data_parallel", "expert_parallel"):
        if dims[key] > 1:
            issues.append(runtime.recipe_error("%s is not supported by the jayleaton family" % key))
    model = config.get("_prepared_model_path") or config.get("resolved_model_path")
    if not model or not str(model).startswith("/"):
        issues.append(
            runtime.recipe_error(
                "the jayleaton family needs a pre-placed flash17 pack path (absolute model: or cluster_config.resolved_model_path)"
            )
        )
    if recipe.command:
        issues.append(runtime.recipe_error("a command: template is not supported by the jayleaton family"))
    if not (recipe.container or recipe.containers):
        issues.append(runtime.recipe_error("the jayleaton family needs an explicit container: image"))
    if recipe.min_nodes and recipe.min_nodes > 2:
        issues.append(runtime.recipe_error("min_nodes=%d exceeds TP=2" % recipe.min_nodes))
    return issues


def prepare(recipe: "Recipe", hosts: list[str], config) -> None:
    if len(hosts) != 2:
        raise RuntimeError("the jayleaton family needs exactly two hosts (got %d)" % len(hosts))
    if _tp(config) != 2:
        raise RuntimeError("the jayleaton family is TP=2 only")
    if not (config.get("_prepared_model_path") or config.get("resolved_model_path")):
        raise RuntimeError("the jayleaton family needs a pre-placed flash17 pack path")
    if recipe.command:
        raise RuntimeError("a command: template is not supported by the jayleaton family")
    if not (recipe.container or recipe.containers):
        raise RuntimeError("the jayleaton family needs an explicit container: image")


def version_commands() -> dict[str, str]:
    return {
        "tensorfold_dsv41_zig": "/opt/tensorfold/bin/tensorfold-dsv41 --version 2>/dev/null | head -1 || echo unknown",
        "tensorfold_dsv41_m1": "test -x /opt/tensorfold/bin/tf-dsv41-m1 && echo present || echo missing",
    }
