"""Provider receipts never substitute for core observation or origin policy."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from sparkrun.core import model_distribution as md
from sparkrun.models.preparation import prepare_model
from test_model_artifacts import artifact, cache, DATA


@pytest.fixture(autouse=True)
def isolated_providers(monkeypatch):
    monkeypatch.setattr(md, "_PROVIDERS", {})


def request(**kwargs):
    return md.ModelCopyRequest(artifact(), None, "/source", (md.ModelTarget("h", "/target", "ib-h", tuple(DATA)),), "operation", **kwargs)


def provider(result=None):
    return Mock(api_version=1, copy=Mock(return_value=result or md.ModelTransferResult(artifact().identity, {"h": "complete"})))


def test_selection_and_explicit_unavailable():
    assert md.try_model_transfer(request()) is None
    with pytest.raises(ValueError, match="not enabled"):
        md.try_model_transfer(request(config={"model_distribution_provider": "missing"}))
    one = provider()
    md.register_model_distribution_provider("one", one)
    assert md.try_model_transfer(request()).outcomes == {"h": "complete"}
    md.register_model_distribution_provider("two", provider())
    with pytest.raises(ValueError, match="multiple"):
        md.try_model_transfer(request())
    assert md.try_model_transfer(request(config={"model_distribution_provider": "builtin"})) is None
    assert md.try_model_transfer(request(config={"model_distribution_provider": "one"})).outcomes == {"h": "complete"}


@pytest.mark.parametrize(
    "outcomes,errors",
    [({}, {}), ({"extra": "complete"}, {}), ({"h": "magic"}, {}), ({"h": "complete"}, {"h": "error"}), ({"h": "failed"}, {})],
)
def test_result_contract(outcomes, errors):
    md.register_model_distribution_provider("test", provider(md.ModelTransferResult(artifact().identity, outcomes, errors)))
    with pytest.raises(md.ModelDistributionFailed):
        md.try_model_transfer(request())


def test_identity_cannot_change():
    md.register_model_distribution_provider("test", provider(md.ModelTransferResult("wrong", {"h": "complete"})))
    with pytest.raises(md.ModelDistributionFailed, match="identity"):
        md.try_model_transfer(request())


def test_unsupported_fallback_only_auto_before_writes():
    p = provider()
    p.copy.side_effect = md.ModelDistributionUnsupported("not installed")
    md.register_model_distribution_provider("test", p)
    assert md.try_model_transfer(request(config={"model_distribution_fallback": True})) is None
    with pytest.raises(md.ModelDistributionUnsupported):
        md.try_model_transfer(request(config={"model_distribution_provider": "test", "model_distribution_fallback": True}))
    p.copy.side_effect = RuntimeError("partial mutation")
    with pytest.raises(md.ModelDistributionFailed, match="no fallback"):
        md.try_model_transfer(request(config={"model_distribution_fallback": True}))


def test_copy_only_and_offline_pull():
    class CopyOnly:
        def copy(self, req):
            return md.ModelTransferResult(req.manifest.identity, {"h": "complete"})

    md.register_model_distribution_provider("copy", CopyOnly())
    assert md.try_model_transfer(request(), pull=True) is None
    p = provider()
    md._PROVIDERS["copy"] = p
    with pytest.raises(md.ModelDistributionUnsupported, match="offline"):
        md.try_model_transfer(request(offline=True), pull=True)
    p.pull.assert_not_called()


def test_cancel_and_dry_run_do_not_invoke_provider():
    p = provider()
    md.register_model_distribution_provider("test", p)
    with pytest.raises(md.ModelDistributionFailed, match="cancelled"):
        md.try_model_transfer(request(cancelled=lambda: True))
    md.try_model_transfer(request(dry_run=True))
    p.copy.assert_not_called()


def test_explicit_context_survives_callback_thread():
    session = Mock()

    def copy(req):
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(lambda: (req.config["setting"], req.session, req.offline)).result() == ("value", session, True)
        return md.ModelTransferResult(req.manifest.identity, {"h": "complete"})

    md.register_model_distribution_provider("test", Mock(api_version=1, copy=copy))
    md.try_model_transfer(request(config={"setting": "value"}, session=session, offline=True))
    session.close.assert_not_called()


def test_provider_claiming_success_with_no_files_fails_core(tmp_path):
    from sparkrun.api.model_cache import inspect_model
    from sparkrun.models.artifacts import ModelArtifactRequest

    manifest = artifact()
    source, target = tmp_path / "source", tmp_path / "target"
    cache(source, manifest)
    p = provider(md.ModelTransferResult(manifest.identity, {"localhost": "complete"}))
    md.register_model_distribution_provider("test", p)

    @md.model_distribution_operation
    def operation():
        with pytest.raises(ValueError, match="invalid after transfer"):
            prepare_model("org/model", ["localhost"], cache_dir=str(target), local_cache_dir=str(source), manifest=manifest, offline=True)
        assert md.prepared_models() == ()

    operation()
    # Inventory survives a failed transfer, but cannot certify missing payloads.
    report = inspect_model(ModelArtifactRequest("org/model"), cache_dir=str(target))[0]
    assert report.status == "incomplete"
    assert {failure.path for failure in report.failures} == set(DATA)
    assert not (target / manifest.relative_snapshot).exists()


def test_no_bindings_leak_between_operations(tmp_path):
    manifest = artifact()
    cache(tmp_path, manifest)

    @md.model_distribution_operation
    def operation(config=None):
        assert md.prepared_models() == ()
        prepare_model("org/model", ["localhost"], cache_dir=str(tmp_path), manifest=manifest, offline=True)
        assert md.prepared_model("org/model", "localhost").manifest == manifest

    operation()
    assert md.prepared_models() == ()
    operation()


def test_registration_version_and_rollback():
    from sparkrun.core.registration import registry_transaction

    p = provider()
    p.api_version = 99
    with pytest.raises(ValueError, match="version"):
        md.register_model_distribution_provider("bad", p)
    with pytest.raises(RuntimeError, match="registration failed"):
        with registry_transaction():
            md.register_model_distribution_provider("rolled-back", provider())
            raise RuntimeError("registration failed")
    assert not md._PROVIDERS


def test_cluster_model_settings_are_scoped_and_roundtrip(tmp_path):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.cluster_manager import ClusterDefinition, ClusterDistributionConfig

    config = SparkrunConfig(config_path=tmp_path / "missing.yaml")
    prefs = {"model": {"validation": "metadata", "file_selection": "all", "provider": "example", "fallback": False}}
    cluster = ClusterDefinition(name="models", hosts=["host"], distribution=ClusterDistributionConfig.from_dict(prefs))
    assert cluster.distribution.to_dict() == prefs
    scoped = config.for_cluster(cluster)
    assert scoped.get("model_cache_validation") == "metadata"
    assert scoped.get("model_file_selection") == "all"
    assert scoped.get("model_distribution_provider") == "example"
    assert scoped.get("model_distribution_fallback") is False
    assert config.get("model_cache_validation") is None
    assert scoped.for_cluster(None).get("model_cache_validation") is None


@pytest.mark.parametrize("paths", [("../outside",), ("config.json", "config.json"), ("not-in-manifest",)])
def test_provider_request_cannot_expand_inventory(paths):
    from dataclasses import replace

    with pytest.raises(ValueError, match="manifest"):
        replace(request(), targets=(md.ModelTarget("h", "/target", "h", paths),))


def test_malformed_receipt_is_a_contract_failure():
    md.register_model_distribution_provider("test", provider(md.ModelTransferResult(artifact().identity, {"h": []})))
    with pytest.raises(md.ModelDistributionFailed, match="malformed"):
        md.try_model_transfer(request())
