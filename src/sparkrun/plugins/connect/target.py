"""Console-free halves of ``sparkrun ssh`` / ``sparkrun exec``: targets and remote commands."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

from sparkrun.core.cluster_manager import CLUSTER_NAME_PATTERN

_NODE_INDEX = re.compile(r"[0-9]+")


class ConnectTargetError(ValueError):
    """A target that names no node."""


@dataclass(frozen=True)
class NodeTarget:
    """A parsed ``TARGET``: what it names (``None`` = the default) and a node index."""

    ref: str | None
    node: int


def split_node_suffix(target: str | None) -> NodeTarget:
    """Split ``REF[/NODE]``; all digits alone is a node of the default.

    The node is taken from the **last** ``/`` only when it is all digits, so a
    reference that contains slashes itself (``@registry/recipe``, a path) keeps
    them: ``@official/qwen/1`` is node 1 of ``@official/qwen``. A trailing
    ``/`` means node 0. All digits is always an index, so a cluster or recipe
    whose name is digits is written ``2/``.
    """
    if not target:
        return NodeTarget(None, 0)
    if _NODE_INDEX.fullmatch(target):
        return NodeTarget(None, int(target))
    head, sep, tail = target.rpartition("/")
    if sep and head and (not tail or _NODE_INDEX.fullmatch(tail)):
        return NodeTarget(head, int(tail) if tail else 0)
    return NodeTarget(target, 0)


def parse_cluster_target(target: str | None, cluster_option: str | None = None) -> NodeTarget:
    """``sparkrun ssh`` target: ``N``, ``CLUSTER``, ``CLUSTER/`` or ``CLUSTER/N``.

    Naming the cluster both ways is refused even when both agree, since one
    spelling is almost always a mistake about the other.

    Raises:
        ConnectTargetError: Not a node number or cluster name, or a cluster named twice.
    """
    parsed = split_node_suffix(target)
    if parsed.ref is None:
        return NodeTarget(cluster_option, parsed.node)
    if not CLUSTER_NAME_PATTERN.match(parsed.ref):
        raise ConnectTargetError("'%s' is not a node number or CLUSTER[/NODE]" % target)
    if cluster_option:
        raise ConnectTargetError("cluster given twice: '%s' and --cluster %s" % (target, cluster_option))
    return parsed


def remote_command(args: list[str] | tuple[str, ...]) -> str | None:
    """The shell string for the words after ``--``, or ``None`` for a shell.

    A single word is passed through untouched, which is ssh's own convention
    (``ssh host 'ps aux | grep vllm'``): pipes, ``&&`` and ``$VARS`` reach the
    remote shell. Several words are each quoted, so an argument containing a
    space survives as one argument. Write ``-- 'a | b'`` for shell syntax and
    ``-- grep "two words" file`` for literal arguments.
    """
    if not args:
        return None
    if len(args) == 1:
        return args[0]
    return shlex.join(args)
