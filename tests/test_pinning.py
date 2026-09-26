"""Pinning: registry / host image digests, model commits, eugr pull refs, ``export recipe --realize``."""

from __future__ import annotations

import io
import json
import urllib.error
from dataclasses import replace

import pytest
import yaml
from click.testing import CliRunner

from sparkrun.containers import digest as digest_mod
from sparkrun.containers.digest import DigestResolutionError, pinned_ref, resolve_host_digest, resolve_registry_digest, split_registry
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.recipe import Recipe
from sparkrun.models import revision as revision_mod
from sparkrun.models.revision import RevisionResolutionError, resolve_cached_commit, resolve_hub_commit, resolve_local_cached_commit

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
COMMIT = "c" * 40
IMAGE = "ghcr.io/org/img:latest"


# --- image refs ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("ghcr.io/org/img:1", ("ghcr.io", "org/img", "1", None)),
        ("vllm/vllm-openai:latest", ("registry-1.docker.io", "vllm/vllm-openai", "latest", None)),
        ("alpine", ("registry-1.docker.io", "library/alpine", None, None)),
        ("docker.io/library/alpine:3", ("registry-1.docker.io", "library/alpine", "3", None)),
        ("myreg:5000/team/img", ("myreg:5000", "team/img", None, None)),
        ("localhost/img:dev", ("localhost", "img", "dev", None)),
    ],
)
def test_split_registry(ref, expected):
    assert split_registry(ref) == expected


def test_pinned_ref_keeps_the_spelling():
    assert pinned_ref("vllm/vllm-openai:latest", SHA_A) == "vllm/vllm-openai@" + SHA_A


# --- registry digest -------------------------------------------------------------------------


class _Response(io.BytesIO):
    def __init__(self, body=b"{}", headers=None, status=200):
        super().__init__(body)
        self.headers = headers or {}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(url, code, headers=None):
    return urllib.error.HTTPError(url, code, "err", headers or {}, None)


def test_registry_digest_follows_the_bearer_challenge(monkeypatch):
    seen = []

    def fake_urlopen(request, timeout=None):
        url, auth = request.full_url, request.headers.get("Authorization")
        seen.append((url, auth))
        if url.startswith("https://ghcr.io/token"):
            assert "scope=repository%3Aorg%2Fimg%3Apull" in url
            return _Response(json.dumps({"token": "T"}).encode())
        if auth != "Bearer T":
            raise _http_error(url, 401, {"WWW-Authenticate": 'Bearer realm="https://ghcr.io/token",service="ghcr.io"'})
        return _Response(headers={"Docker-Content-Digest": SHA_A})

    monkeypatch.setattr(digest_mod.urllib.request, "urlopen", fake_urlopen)
    assert resolve_registry_digest(IMAGE) == SHA_A
    assert seen[0][0] == "https://ghcr.io/v2/org/img/manifests/latest"


def test_registry_digest_hashes_the_body_when_the_header_is_missing(monkeypatch):
    import hashlib

    body = b'{"schemaVersion": 2}'
    monkeypatch.setattr(digest_mod.urllib.request, "urlopen", lambda request, timeout=None: _Response(body))
    assert resolve_registry_digest(IMAGE) == "sha256:" + hashlib.sha256(body).hexdigest()


def test_registry_digest_uses_inline_docker_credentials(monkeypatch, tmp_path):
    (tmp_path / ".docker").mkdir()
    (tmp_path / ".docker" / "config.json").write_text(json.dumps({"auths": {"ghcr.io": {"auth": "dXNlcjpwYXNz"}}}))
    monkeypatch.setattr(digest_mod.Path, "home", classmethod(lambda cls: tmp_path))
    token_auth = []

    def fake_urlopen(request, timeout=None):
        if request.full_url.startswith("https://ghcr.io/token"):
            token_auth.append(request.headers.get("Authorization"))
            return _Response(json.dumps({"token": "T"}).encode())
        if request.headers.get("Authorization") != "Bearer T":
            raise _http_error(request.full_url, 401, {"WWW-Authenticate": 'Bearer realm="https://ghcr.io/token"'})
        return _Response(headers={"Docker-Content-Digest": SHA_A})

    monkeypatch.setattr(digest_mod.urllib.request, "urlopen", fake_urlopen)
    assert resolve_registry_digest(IMAGE) == SHA_A
    assert token_auth == ["Basic dXNlcjpwYXNz"]


@pytest.mark.parametrize("code,match", [(404, "no tag"), (403, "refused access")])
def test_registry_digest_failures_raise(monkeypatch, code, match):
    def fake_urlopen(request, timeout=None):
        raise _http_error(request.full_url, code)

    monkeypatch.setattr(digest_mod.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(DigestResolutionError, match=match):
        resolve_registry_digest(IMAGE)


def test_an_already_pinned_ref_is_not_looked_up(monkeypatch):
    monkeypatch.setattr(digest_mod.urllib.request, "urlopen", lambda *a, **k: pytest.fail("no lookup expected"))
    assert resolve_registry_digest("ghcr.io/org/img@" + SHA_B) == SHA_B


# --- host digest -----------------------------------------------------------------------------


def _hosts_report(monkeypatch, identities, local=(None, [])):
    import sparkrun.containers.distribute as distribute
    import sparkrun.containers.registry as registry

    monkeypatch.setattr(distribute, "_check_remote_image_identities", lambda image, hosts, **kw: dict(identities))
    monkeypatch.setattr(registry, "get_image_identity", lambda image: local)


def test_host_digest_agreed_by_digest(monkeypatch):
    _hosts_report(monkeypatch, {"h1": ("id1", ["ghcr.io/org/img@" + SHA_A]), "h2": ("id2", ["ghcr.io/org/img@" + SHA_A])})
    assert resolve_host_digest(IMAGE, ["h1", "h2"]) == SHA_A


def test_host_digests_that_disagree_raise(monkeypatch):
    _hosts_report(monkeypatch, {"h1": ("id1", ["ghcr.io/org/img@" + SHA_A]), "h2": ("id2", ["ghcr.io/org/img@" + SHA_B])})
    with pytest.raises(DigestResolutionError, match="different builds"):
        resolve_host_digest(IMAGE, ["h1", "h2"])


def test_save_loaded_hosts_borrow_the_digest_of_a_matching_image_id(monkeypatch):
    # docker save | docker load keeps the image id but drops RepoDigests.
    _hosts_report(monkeypatch, {"h1": ("idX", []), "h2": ("idX", [])}, local=("idX", ["ghcr.io/org/img@" + SHA_A]))
    assert resolve_host_digest(IMAGE, ["h1", "h2"]) == SHA_A


def test_a_holder_with_a_different_image_id_is_not_trusted(monkeypatch):
    _hosts_report(monkeypatch, {"h1": ("idX", []), "h2": ("idX", [])}, local=("idY", ["ghcr.io/org/img@" + SHA_A]))
    with pytest.raises(DigestResolutionError, match="records a registry digest"):
        resolve_host_digest(IMAGE, ["h1", "h2"])


def test_a_host_without_the_image_raises(monkeypatch):
    _hosts_report(monkeypatch, {"h1": ("idX", ["ghcr.io/org/img@" + SHA_A])})
    with pytest.raises(DigestResolutionError, match="not present on h2"):
        resolve_host_digest(IMAGE, ["h1", "h2"])


# --- model revision --------------------------------------------------------------------------


def test_hub_commit_passes_a_commit_through(monkeypatch):
    monkeypatch.setattr("sparkrun.models.hub.hub_metadata_call", lambda *a, **k: pytest.fail("no lookup expected"))
    assert resolve_hub_commit("org/model", COMMIT) == COMMIT


def test_hub_commit_resolves_through_the_metadata_budget(monkeypatch):
    calls = []

    def fake_call(label, model_id, revision, fn, **kw):
        calls.append((label, model_id, revision))
        return COMMIT

    monkeypatch.setattr("sparkrun.models.hub.hub_metadata_call", fake_call)
    assert resolve_hub_commit("org/model-GGUF:Q4_K_M") == COMMIT
    assert calls == [("model revision", "org/model-GGUF", None)]  # the quant suffix is not part of the repo


def test_hub_commit_failure_raises(monkeypatch):
    monkeypatch.setattr("sparkrun.models.hub.hub_metadata_call", lambda *a, **k: None)
    with pytest.raises(RevisionResolutionError, match="--offline"):
        resolve_hub_commit("org/model")


class _Result:
    def __init__(self, host, stdout, success=True):
        self.host, self.stdout, self.success = host, stdout, success


def _cached(monkeypatch, answers):
    import sparkrun.orchestration.ssh as ssh

    scripts = []

    def fake(hosts, script, **kw):
        scripts.append(script)
        return [_Result(h, answers.get(h, "")) for h in hosts]

    monkeypatch.setattr(ssh, "run_remote_scripts_parallel", fake)
    return scripts


def test_cached_commit_reads_refs_on_every_host(monkeypatch):
    scripts = _cached(monkeypatch, {"h1": COMMIT + "\n", "h2": COMMIT})
    assert resolve_cached_commit("org/model", None, ["h1", "h2"], cache_dir="/cache/hf") == COMMIT
    assert 'cat "/cache/hf/hub/models--org--model/refs/main"' in scripts[0]


def test_cached_commits_that_disagree_raise(monkeypatch):
    _cached(monkeypatch, {"h1": COMMIT, "h2": "d" * 40})
    with pytest.raises(RevisionResolutionError, match="different commits") as info:
        resolve_cached_commit("org/model", None, ["h1", "h2"])
    assert info.value.missing == ()


def test_a_host_missing_the_model_is_reported_as_missing(monkeypatch):
    _cached(monkeypatch, {"h1": COMMIT})
    with pytest.raises(RevisionResolutionError) as info:
        resolve_cached_commit("org/model", None, ["h1", "h2"])
    assert info.value.missing == ("h2",)


def test_revision_names_are_not_shell(monkeypatch):
    _cached(monkeypatch, {})
    with pytest.raises(RevisionResolutionError, match="plain branch"):
        resolve_cached_commit("org/model", "main; rm -rf /", ["h1"])


def test_local_cached_commit(tmp_path):
    refs = tmp_path / "hub" / "models--org--model" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(COMMIT)
    assert resolve_local_cached_commit("org/model", None, str(tmp_path)) == COMMIT
    assert resolve_local_cached_commit("org/other", None, str(tmp_path)) is None


# --- eugr pull_ref ---------------------------------------------------------------------------


def _eugr_recipe(container, build_args=None):
    data = {"model": "org/m", "runtime": "vllm", "container": container}
    if build_args:
        data["build_args"] = build_args
    return Recipe.from_dict(data)


@pytest.mark.parametrize(
    "container,build_args,present,expected",
    [
        ("eugr/spark-vllm:latest", None, False, "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest"),  # sentinel -> nightly
        ("vllm/vllm-openai:v1", None, False, "vllm/vllm-openai:v1"),  # pullable ref
        ("eugr/spark-vllm:latest", ["--use-wheels"], False, None),  # local wheels build
        ("vllm-node", None, False, "ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest"),  # absent -> substitute
        ("vllm-node", None, True, None),  # present local image
    ],
)
def test_eugr_pull_ref(monkeypatch, container, build_args, present, expected):
    import sparkrun.containers.registry as registry
    from sparkrun.builders.eugr import EugrBuilder

    monkeypatch.setattr(registry, "image_exists_locally", lambda image: present)
    recipe = _eugr_recipe(container, build_args)
    assert EugrBuilder().pull_ref(container, recipe) == expected


@pytest.mark.parametrize("container", ["eugr/spark-vllm:latest", "vllm/vllm-openai:v1", "vllm-node"])
def test_eugr_pull_ref_agrees_with_prepare(monkeypatch, container):
    """The drift guard: what pull_ref reports is what prepare() hands the launch."""
    import sparkrun.containers.registry as registry
    from sparkrun.builders.eugr import EugrBuilder

    monkeypatch.setattr(registry, "image_exists_locally", lambda image: False)
    recipe = _eugr_recipe(container)
    builder = EugrBuilder()
    assert builder.pull_ref(container, recipe) == builder.prepare(container, recipe, hosts=["h1"], dry_run=True)


def test_image_builders_that_build_report_no_pull_ref():
    from sparkrun.builders.base import BuilderPlugin
    from sparkrun.builders.docker_pull import DockerPullBuilder

    recipe = _eugr_recipe("org/img:1")
    assert DockerPullBuilder().pull_ref("org/img:1", recipe) == "org/img:1"

    class Building(BuilderPlugin):
        builder_name = "building"

        def prepare(self, image, recipe, hosts, **kw):
            return image

    assert Building().pull_ref("org/img:1", recipe) is None


# --- realize + pin ---------------------------------------------------------------------------

_RECIPE = {
    "model": "org/model",
    "runtime": "vllm-distributed",
    "container": IMAGE,
    "defaults": {"port": 8000, "tensor_parallel": 2},
}
_HOSTS = ("s1", "s2")


@pytest.fixture
def dry_plan(monkeypatch):
    import sparkrun.api._run as run_module

    real_plan = run_module.plan
    monkeypatch.setattr(run_module, "plan", lambda options, *, sctx=None: real_plan(replace(options, dry_run=True), sctx=sctx))


@pytest.fixture
def resolvers(monkeypatch):
    calls = {}

    def record(name, value):
        def fn(*args, **kwargs):
            calls.setdefault(name, []).append((args, kwargs))
            return value

        return fn

    monkeypatch.setattr(digest_mod, "resolve_registry_digest", record("registry", SHA_A))
    monkeypatch.setattr(digest_mod, "resolve_host_digest", record("hosts", SHA_B))
    monkeypatch.setattr(revision_mod, "resolve_hub_commit", record("hub", COMMIT))
    monkeypatch.setattr(revision_mod, "resolve_cached_commit", record("cached", "e" * 40))
    return calls


def _options(tmp_path, data=None):
    import sparkrun.api as api
    from sparkrun.core.cluster_manager import ClusterDefinition

    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump(data or _RECIPE))
    hardware = HostHardware(
        accelerators=[AcceleratorSpec(vendor="nvidia", model="gb10", capabilities=frozenset({"cuda"}))], source="detected"
    )
    cluster = ClusterDefinition(name="c", hosts=list(_HOSTS), hosts_hardware={h: hardware for h in _HOSTS})
    return api.RunOptions(recipe=Recipe.load(str(path), resolve=False), cluster=cluster, hosts=_HOSTS)


def test_realize_pins_from_the_registry_and_the_hub_by_default(tmp_path, v, dry_plan, resolvers):
    import sparkrun.api as api

    realized = api.realize_recipe(_options(tmp_path))
    data = realized.recipe
    assert data["container"] == "ghcr.io/org/img@" + SHA_A
    assert data["model_revision"] == COMMIT
    assert list(data)[list(data).index("model") + 1] == "model_revision"
    assert data["metadata"]["pinned"]["mode"] == "online"
    assert data["metadata"]["pinned"]["from"] == {"container": IMAGE, "model_revision": "main"}
    assert {p.field: p.source for p in realized.pins} == {"container": "registry", "model_revision": "hub"}
    assert "hosts" not in resolvers and "cached" not in resolvers


def test_offline_pins_from_the_placed_hosts(tmp_path, v, dry_plan, resolvers):
    import sparkrun.api as api

    realized = api.realize_recipe(_options(tmp_path), offline=True)
    assert realized.recipe["container"] == "ghcr.io/org/img@" + SHA_B
    assert realized.recipe["model_revision"] == "e" * 40
    assert resolvers["hosts"][0][0][1] == list(_HOSTS)
    assert "registry" not in resolvers and "hub" not in resolvers


def test_offline_falls_back_to_the_control_cache_only_for_missing_hosts(tmp_path, v, dry_plan, resolvers, monkeypatch):
    import sparkrun.api as api

    def missing(*a, **k):
        raise RevisionResolutionError("absent", missing=("s2",))

    monkeypatch.setattr(revision_mod, "resolve_cached_commit", missing)
    monkeypatch.setattr(revision_mod, "resolve_local_cached_commit", lambda *a, **k: "f" * 40)
    assert api.realize_recipe(_options(tmp_path), offline=True).recipe["model_revision"] == "f" * 40

    def mismatch(*a, **k):
        raise RevisionResolutionError("different commits")

    monkeypatch.setattr(revision_mod, "resolve_cached_commit", mismatch)
    with pytest.raises(api.SparkrunError, match="different commits"):
        api.realize_recipe(_options(tmp_path), offline=True)


def test_no_pin_leaves_references_alone(tmp_path, v, dry_plan, resolvers):
    import sparkrun.api as api

    realized = api.realize_recipe(_options(tmp_path), pin=False)
    assert realized.recipe["container"] == IMAGE
    assert "model_revision" not in realized.recipe
    assert "pinned" not in realized.recipe["metadata"]
    assert resolvers == {}


def test_already_pinned_references_are_kept(tmp_path, v, dry_plan, resolvers):
    import sparkrun.api as api

    data = {**_RECIPE, "container": "ghcr.io/org/img@" + SHA_B, "model_revision": COMMIT}
    realized = api.realize_recipe(_options(tmp_path, data))
    assert realized.recipe["container"] == "ghcr.io/org/img@" + SHA_B
    assert realized.pins == ()
    assert "registry" not in resolvers


def test_a_locally_built_image_refuses_to_pin(tmp_path, v, dry_plan, resolvers, monkeypatch):
    import sparkrun.api as api
    import sparkrun.containers.registry as registry

    monkeypatch.setattr(registry, "image_exists_locally", lambda image: False)
    data = {**_RECIPE, "runtime": "vllm", "container": "eugr/spark-vllm:latest", "build_args": ["--use-wheels"]}
    with pytest.raises(api.SparkrunError, match="--no-pin"):
        api.realize_recipe(_options(tmp_path, data))


def test_a_separate_draft_model_is_reported_unpinned(tmp_path, v, dry_plan, resolvers):
    import sparkrun.api as api

    data = {**_RECIPE, "defaults": {**_RECIPE["defaults"], "speculative_config": {"method": "eagle", "model": "org/draft"}}}
    assert api.realize_recipe(_options(tmp_path, data)).unpinned == (
        "speculative_config.model org/draft (draft model revision is not pinned)",
    )


@pytest.mark.parametrize(
    "args,match",
    [
        (["--no-pin"], "only apply with --realize"),
        (["--offline"], "only apply with --realize"),
        (["--realize", "--no-pin", "--offline"], "no effect"),
    ],
)
def test_cli_pin_flag_combinations(tmp_path, args, match):
    from sparkrun.cli import main

    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump(_RECIPE))
    result = CliRunner().invoke(main, ["export", "recipe", str(path), *args])
    assert result.exit_code != 0
    assert match in result.output
