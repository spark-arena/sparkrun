"""Default :class:`SparkrunContext` factory for the library API.

Each ``api.*`` function accepts an optional ``sctx`` argument; when
``None`` (the common one-shot case), :func:`default_sctx` builds a
fresh session bundling the SAF :class:`Variables` and a
:class:`SparkrunConfig`.  Callers that issue multiple ``api.*`` calls
in sequence can construct an :class:`SparkrunContext` once and pass
it in to share state (config, registry manager, cluster manager).

This module is the *only* place where the api layer constructs
default session state — the rest of the api forwards an explicit
``sctx`` everywhere, so there are no implicit globals leaking through
the call graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext


def default_sctx() -> "SparkrunContext":
    """Build a fresh :class:`SparkrunContext` for a one-shot API call.

    Initialises the SAF plugin registry (idempotent — uses the module
    singleton if already bootstrapped) and instantiates a
    :class:`SparkrunConfig` from the default config path.
    """
    from sparkrun.core.bootstrap import init_sparkrun
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.context import SparkrunContext

    return SparkrunContext(
        variables=init_sparkrun(),
        config=SparkrunConfig(),
        verbose=False,
        progress=None,
    )


def resolve_sctx(sctx: "SparkrunContext | None") -> "SparkrunContext":
    """Return *sctx* if non-None, else a freshly-built default."""
    return sctx if sctx is not None else default_sctx()


__all__ = ["default_sctx", "resolve_sctx"]
