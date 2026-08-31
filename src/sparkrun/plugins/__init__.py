"""First-party cross-cutting integrations, loaded as in-tree plugins.

Each subpackage here is one integration, discovered and registered at bootstrap
by :func:`sparkrun.core.in_tree_plugins.load_in_tree_plugins` — the same
registration path out-of-tree plugins take, so a first-party integration has no
capability an external one lacks.

This package is deliberately *not* the home for every first-party plugin. A
runtime belongs in ``sparkrun.runtimes``, an executor in
``sparkrun.orchestration.executors``, and so on: those map cleanly onto one
extension point and keep their own scan in :mod:`sparkrun.core.bootstrap`.
What lives here is what spans several at once and is only coherent as a single
removable unit — an integration contributing, say, a backend implementation, a
hidden CLI command, and the wire protocol that backend calls back through, none
of which is "a runtime" or "an executor" on its own.
"""
