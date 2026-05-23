"""Pluggable scheduler implementations for sparkrun.

Modules in this package are scanned by :mod:`sparkrun.core.bootstrap`
for :class:`~sparkrun.core.scheduler.Scheduler` subclasses and
registered at the :data:`~sparkrun.core.scheduler.EXT_SCHEDULER`
extension point.
"""
