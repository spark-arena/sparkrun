"""Unit tests for SAF-driven executor discovery.

Covers the workstream A3 contract:

- ``core.bootstrap.init_sparkrun`` discovers every :class:`Executor`
  subclass under ``sparkrun.orchestration.executors`` and registers it
  under :data:`EXT_EXECUTOR`.
- :func:`get_executor` / :func:`list_executors` query the SAF registry.
- :meth:`ExecutorConfig.from_chain` validates executor selectors
  against the SAF registry (not against the hardcoded
  ``_KNOWN_EXECUTORS`` set, which has been removed).
- A *custom* executor subclass, once registered via SAF, becomes
  resolvable through :func:`resolve_executor` / :func:`get_executor`
  exactly like the built-ins.
- The :class:`RuntimePlugin` ``executor`` lazy property is gone —
  selection always flows through :func:`resolve_executor`.
"""

from __future__ import annotations

import pytest

from sparkrun.core.bootstrap import init_sparkrun
from sparkrun.orchestration.executor import (
    EXT_EXECUTOR,
    DockerExecutor,
    Executor,
    ExecutorConfig,
    get_executor,
    list_executors,
    resolve_executor,
)
from sparkrun.orchestration.executors.k8s import K8sExecutor
from sparkrun.orchestration.executors.local import LocalExecutor


# ---------------------------------------------------------------------------
# Roundtrip: SAF init registers all built-in executors
# ---------------------------------------------------------------------------


class TestSAFDiscoveryRoundtrip:
    """``init_sparkrun`` registers docker, local, k8s under EXT_EXECUTOR."""

    def test_init_sparkrun_registers_docker(self):
        v = init_sparkrun()
        cls = get_executor("docker", v=v)
        assert cls is DockerExecutor
        assert isinstance(cls(), DockerExecutor)

    def test_init_sparkrun_registers_local(self):
        v = init_sparkrun()
        cls = get_executor("local", v=v)
        assert cls is LocalExecutor

    def test_init_sparkrun_registers_k8s(self):
        v = init_sparkrun()
        cls = get_executor("k8s", v=v)
        assert cls is K8sExecutor

    def test_list_executors_after_init_contains_builtins(self):
        v = init_sparkrun()
        names = list_executors(v)
        assert "docker" in names
        assert "local" in names
        assert "k8s" in names

    def test_extension_point_name_matches_constant(self):
        """Each registered plugin must advertise the canonical EXT_EXECUTOR."""
        for cls in (DockerExecutor, LocalExecutor, K8sExecutor):
            assert cls().extension_point_name(None) == EXT_EXECUTOR


# ---------------------------------------------------------------------------
# Selector validation goes through SAF (no hardcoded _KNOWN_EXECUTORS)
# ---------------------------------------------------------------------------


class TestSelectorValidationViaSAF:
    """:meth:`ExecutorConfig.from_chain` queries SAF for known names."""

    def test_known_selector_accepted(self):
        cfg = ExecutorConfig.from_chain({"executor": "local"})
        assert cfg.executor_type == "local"

    def test_unknown_selector_falls_back_to_docker(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="sparkrun.orchestration.executors._base"):
            cfg = ExecutorConfig.from_chain({"executor": "discovery_probe"})
        assert cfg.executor_type == "docker"
        # The fallback warning must reference the rejected selector.
        assert any("discovery_probe" in rec.getMessage() for rec in caplog.records)

    def test_known_executors_attribute_was_removed(self):
        """The legacy hardcoded set is gone — guard against reintroduction."""
        assert not hasattr(ExecutorConfig, "_KNOWN_EXECUTORS")


# ---------------------------------------------------------------------------
# Custom Executor subclass: register + resolve via SAF
# ---------------------------------------------------------------------------


class _DiscoveryProbeExecutor(Executor):
    """Minimal in-test executor used to verify discovery is dynamic.

    Implements the abstract command surface with no-op strings so the
    plugin can be instantiated for resolution assertions.  Not registered
    by the production package's discovery (lives in tests/), so the
    SAF registry only learns about it via :func:`register_plugin` in
    the fixture below.
    """

    executor_name = "discovery_probe"

    def run_cmd(self, image, command="", container_name=None, detach=True, env=None, volumes=None, extra_opts=None):
        return "wasm run %s" % container_name

    def exec_cmd(self, container_name, command, detach=False, env=None):
        return "wasm exec %s" % container_name

    def stop_cmd(self, container_name, force=True):
        return "wasm stop %s" % container_name

    def logs_cmd(self, container_name, follow=False, tail=None):
        return "wasm logs %s" % container_name

    def status_cmd(self, container_name):
        return "wasm status %s" % container_name

    def inspect_exists_cmd(self, image):
        return "wasm inspect %s" % image

    def pull_cmd(self, image):
        return "wasm pull %s" % image


@pytest.fixture
def custom_executor_registered():
    """Register :class:`_DiscoveryProbeExecutor` with SAF for the duration of a test."""
    from scitrera_app_framework import register_plugin

    v = init_sparkrun()
    register_plugin(_DiscoveryProbeExecutor, v=v)
    yield v
    # SAF doesn't expose unregister; the test-harness Variables instance
    # is scoped per session so leakage to unrelated tests is harmless
    # (the wasm name is unique to this module).


class TestCustomExecutorDiscovery:
    """Once registered with SAF, a custom executor is selectable end-to-end."""

    def test_get_executor_returns_custom_class(self, custom_executor_registered):
        v = custom_executor_registered
        cls = get_executor("discovery_probe", v=v)
        assert cls is _DiscoveryProbeExecutor

    def test_list_executors_includes_custom(self, custom_executor_registered):
        v = custom_executor_registered
        assert "discovery_probe" in list_executors(v)

    def test_resolve_executor_returns_custom_instance(self, custom_executor_registered):
        v = custom_executor_registered
        ex = resolve_executor(cli_overrides={"executor": "discovery_probe"}, v=v)
        assert isinstance(ex, _DiscoveryProbeExecutor)
        assert ex.config.executor_type == "discovery_probe"

    def test_from_chain_accepts_custom_when_registered(self, custom_executor_registered):
        cfg = ExecutorConfig.from_chain({"executor": "discovery_probe"})
        assert cfg.executor_type == "discovery_probe"


# ---------------------------------------------------------------------------
# RuntimePlugin no longer exposes the legacy executor property
# ---------------------------------------------------------------------------


class TestRuntimePluginExecutorRemoved:
    """The ``RuntimePlugin.executor`` lazy property has been removed."""

    def test_executor_is_plain_attribute_not_property(self):
        from sparkrun.runtimes.base import RuntimePlugin

        # In the new world ``executor`` is a class-level instance
        # attribute (class default ``None``), not a property.
        attr = vars(RuntimePlugin).get("executor")
        assert attr is None or not isinstance(attr, property)

    def test_resolve_executor_helper_is_present(self):
        from sparkrun.runtimes.base import RuntimePlugin

        assert callable(getattr(RuntimePlugin, "_resolve_executor", None))

    def test_resolve_executor_helper_returns_docker_by_default(self):
        """No explicit executor + no recipe/config defaults → DockerExecutor."""
        from sparkrun.runtimes.base import RuntimePlugin

        class _Dummy(RuntimePlugin):
            runtime_name = "dummy-runtime-for-test"

            def generate_command(self, *a, **kw):
                return ""

        rt = _Dummy()
        ex = rt._resolve_executor()
        assert isinstance(ex, DockerExecutor)
        # Cached on the instance for subsequent calls.
        assert rt._resolve_executor() is ex
