"""``sparkrun daemon help``: the operation reference and where its text comes from."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.cli._daemon_help import render_list, render_operation, type_label
from sparkrun.daemon.op_docs import argument_docs, attribute_docs, operation_docs, plain, summary


def _documented(a, b=None):
    """Make a thing, quickly. Then more prose.

    Args:
        a: The first ``value``, see :func:`sparkrun.api.run`.
            It continues here.
        b (int): Second; *emphasised*.

    Raises:
        ValueError: never mentioned as an argument.
    """


@dataclass
class _Base:
    hosts: list | None = None
    """Hosts to use.

    Not part of the summary."""


@dataclass
class _Options(_Base):
    recipe: str = ""
    """Recipe name."""
    hosts: list | None = None
    """Refined: the candidate hosts."""
    undocumented: int = 0


def test_plain_strips_markup():
    assert plain("Use ``x`` via :func:`sparkrun.api.run` on *hosts*.") == "Use x via run on hosts."


def test_summary_is_the_first_sentence():
    assert summary(_documented) == "Make a thing, quickly."
    assert summary(None) == ""


def test_argument_docs_reads_the_args_section_only():
    assert argument_docs(_documented) == {"a": "The first value, see run. It continues here.", "b": "Second; emphasised."}


def test_attribute_docs_follow_the_mro_and_take_one_paragraph():
    assert attribute_docs(_Options) == {"hosts": "Refined: the candidate hosts.", "recipe": "Recipe name."}


@pytest.mark.parametrize(
    ("schema", "label"),
    [
        ({"type": "string", "maxLength": 4096}, "string"),
        ({"type": "string", "maxLength": 64}, "string (max 64)"),
        ({"type": "integer", "minimum": 1, "maximum": 65535}, "integer 1-65535"),
        ({"type": "array", "items": {"type": "string"}}, "list of string"),
        ({"enum": ["a", "b"]}, 'one of "a"|"b"'),
        ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "string (or null)"),
        ({}, "any"),
    ],
)
def test_type_label(schema, label):
    assert type_label(schema) == label


_OP = {
    "name": "workloads.stop_all",
    "kind": "job",
    "scope": "operate",
    "description": "Stop everything on the hosts or cluster.",
    "http_path": "/v1/workloads/stop_all",
    "request_schema": {
        "type": "object",
        "properties": {
            "hosts": {"type": "array", "items": {"type": "string"}},
            "cluster": {"type": "string"},
            "dry_run": {"type": "boolean"},
        },
        "required": [],
        "anyOf": [{"required": ["hosts"]}, {"required": ["cluster"]}],
    },
    "result_schema": {"type": "object", "properties": {"jobs_stopped": {"type": "integer"}}},
}


def test_render_operation_shows_calling_contract_and_a_runnable_example():
    text = render_operation(_OP, width=100, argument_docs={"dry_run": "Preview only."})

    assert "kind:   job (returns a job ID" in text
    assert "scope:  operate" in text
    assert "HTTP:   POST /v1/workloads/stop_all (or POST /v1/ops/workloads.stop_all)" in text
    assert "At least one of: hosts, cluster" in text
    assert "Preview only." in text
    assert "Result: object with jobs_stopped" in text
    example = text.rsplit("--data ", 1)[1]
    assert example.endswith(" --wait")
    assert json.loads(example[: -len(" --wait")].strip("'")) == {"hosts": ["<host>"]}


def test_render_operation_marks_required_fields_first():
    op = {
        **_OP,
        "request_schema": {"type": "object", "properties": {"z": {"type": "string"}, "name": {"type": "string"}}, "required": ["name"]},
    }

    lines = [line for line in render_operation(op, width=100).splitlines() if line.startswith("  * ") or line.startswith("    z")]
    assert lines[0].startswith("  * name")


def test_render_list_groups_by_namespace_and_truncates():
    catalog = [
        {**_OP},
        {**_OP, "name": "plugins.k8s.status", "description": "x" * 300},
        {**_OP, "name": "catalog.search", "kind": "query", "scope": "read"},
    ]
    text = render_list(catalog, width=80, source="in this installation")

    assert text.index("\ncatalog\n") < text.index("\nplugins.k8s\n") < text.index("\nworkloads\n")
    assert all(len(line) <= 80 for line in text.splitlines() if line.startswith("  plugins.k8s"))


def test_every_core_operation_has_its_own_real_description():
    """A generic fallback ("X through the library API.") reads as noise in a reference."""
    from sparkrun.application import initialize
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.daemon_registry import operation_catalog
    from sparkrun.daemon.ops import register_core_ops

    initialize()
    register_core_ops()
    core = [op for op in operation_catalog(SparkrunConfig()) if op["owner"] in (None, "core")]
    generic = [op["name"] for op in core if "through the library API" in op["description"] or "operation-local" in op["description"]]
    assert not generic
    descriptions = [op["description"] for op in core]
    assert len(descriptions) == len(set(descriptions))


def test_options_model_operations_document_their_fields():
    _summary, docs = operation_docs("workloads.run")

    assert "Recipe" in docs["recipe"]
    assert docs["confirm"].startswith("Ask for approval")


def test_cli_lists_and_describes_operations():
    runner = CliRunner()

    listing = runner.invoke(main, ["daemon", "help"])
    assert listing.exit_code == 0, listing.output
    assert "workloads.run" in listing.output and "cluster.create" in listing.output

    detail = runner.invoke(main, ["daemon", "help", "cluster.create"], env={"COLUMNS": "120"})
    assert detail.exit_code == 0, detail.output
    assert "* name" in detail.output and "Host names or addresses, head first." in detail.output
    assert "sparkrun daemon call cluster.create --data" in detail.output

    namespace = runner.invoke(main, ["daemon", "help", "tailscale"])
    assert namespace.exit_code == 0
    assert "tailscale.join" in namespace.output and "workloads.run" not in namespace.output


def test_cli_suggests_close_names():
    result = CliRunner().invoke(main, ["daemon", "help", "workloads.strat"])

    assert result.exit_code == 1
    assert "Did you mean: workloads.status" in result.output


def test_cli_json_is_the_catalog_entry():
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.daemon_registry import operation_catalog

    result = CliRunner().invoke(main, ["daemon", "help", "cluster.create", "--json"])

    assert result.exit_code == 0, result.output
    expected = next(op for op in operation_catalog(SparkrunConfig()) if op["name"] == "cluster.create")
    assert json.loads(result.output) == json.loads(json.dumps(expected, default=str))


def test_cli_reads_a_remote_daemons_catalog(monkeypatch):
    seen = []

    def request(self, method, path, body=None, *, headers=None):
        seen.append((method, path))
        return [{**_OP, "name": "remote.only", "description": "Only on that daemon."}]

    monkeypatch.setattr("sparkrun.daemon.client.Client.request", request)
    result = CliRunner().invoke(main, ["daemon", "--url", "http://controller:8443", "--token", "t", "help"])

    assert result.exit_code == 0, result.output
    assert seen == [("GET", "/v1/ops")]
    assert "on http://controller:8443" in result.output and "remote.only" in result.output
    assert "workloads.run" not in result.output


def test_an_example_is_never_a_runnable_empty_call():
    """For cluster.default, --data '{}' would clear the default cluster."""
    op = {
        **_OP,
        "name": "cluster.default",
        "request_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": []},
    }

    assert '--data \'{"name": "<name>"}\'' in render_operation(op, width=100)


def test_an_operation_that_is_also_a_namespace_lists_its_children():
    text = render_operation({**_OP, "name": "catalog.registries"}, width=100, children=["catalog.registries.add"])

    assert "Operations under catalog.registries:\n  catalog.registries.add" in text


def test_a_long_type_gets_its_own_line_and_notes_keep_their_column():
    properties = {"mode": {"enum": ["auto", "local", "push", "delegated", "pull"]}, "a": {"type": "string"}}
    op = {**_OP, "request_schema": {"type": "object", "properties": properties, "required": []}}

    lines = render_operation(op, width=100, argument_docs={"mode": "The mode.", "a": "A field."}).splitlines()
    mode_note = next(line for line in lines if line.strip() == "The mode.")
    a_row = next(line for line in lines if line.startswith("    a "))
    assert mode_note.index("The mode.") == a_row.index("A field.")


def test_none_reads_as_null_for_wire_callers():
    assert plain("``None`` takes the default") == "null takes the default"


@pytest.mark.parametrize("body", [{"detail": "proxy page"}, [{"kind": "job"}], ["x"]])
def test_a_malformed_remote_catalog_is_a_clean_error(monkeypatch, body):
    monkeypatch.setattr("sparkrun.daemon.client.Client.request", lambda self, method, path, body_=None, **k: body)

    result = CliRunner().invoke(main, ["daemon", "--url", "http://controller:8443", "--token", "t", "help"])

    assert result.exit_code == 1
    assert "Unexpected GET /v1/ops response" in result.output
