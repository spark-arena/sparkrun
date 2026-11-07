"""``sparkrun daemon help``: a terminal reference for the daemon's operations.

Pure rendering over the operation catalog (the same list ``GET /v1/ops``
returns, from :func:`sparkrun.core.daemon_registry.operation_catalog`), so the
reference cannot describe operations the daemon does not have. Per-field notes
come from the library functions the operations call
(:mod:`sparkrun.daemon.op_docs`) when the catalog is this installation's.
"""

from __future__ import annotations

import difflib
import json
import shlex
import textwrap

_KINDS = {
    "query": "answered directly",
    "job": "returns a job ID; add --wait to follow it to the result",
    "stream": "served as a stream under /v1/streams",
}

#: maxLength values that only bound request size, not anything a caller chooses.
_NOISE_LENGTHS = {256, 1024, 4096}


def type_label(schema: dict) -> str:
    """A short human type for a JSON schema (``list of string``, ``integer 1-65535``, ...)."""
    if not isinstance(schema, dict) or not schema:
        return "any"
    if "anyOf" in schema:
        options = [type_label(option) for option in schema["anyOf"]]
        return " or ".join(dict.fromkeys(option for option in options if option != "null")) + (" (or null)" if "null" in options else "")
    if "enum" in schema:
        return "one of " + "|".join(json.dumps(value) for value in schema["enum"])
    kind = schema.get("type", "any")
    if isinstance(kind, list):
        return " or ".join(kind)
    if kind == "array":
        return "list of " + type_label(schema.get("items", {}))
    if kind in {"integer", "number"}:
        low, high = schema.get("minimum"), schema.get("maximum")
        if low is not None and high is not None:
            return "%s %s-%s" % (kind, low, high)
        if low is not None:
            return "%s >= %s" % (kind, low)
        return kind
    if kind == "string":
        limit = schema.get("maxLength")
        return "string" if limit in _NOISE_LENGTHS or limit is None else "string (max %d)" % limit
    return kind


def example_value(schema: dict, name: str):
    """A placeholder for *name* that has the right JSON shape."""
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return example_value(next((s for s in schema["anyOf"] if s.get("type") != "null"), {}), name)
    kind = schema.get("type")
    if kind == "array":
        return [example_value(schema.get("items", {}), name.rstrip("s"))]
    if kind == "boolean":
        return True
    if kind in {"integer", "number"}:
        return schema.get("minimum", 0)
    if kind == "object":
        return {}
    return "<%s>" % name


def _required_groups(schema: dict) -> tuple[list[str], list[list[str]]]:
    """``(always required, one-of groups)`` from a request schema."""
    required = list(schema.get("required", []))
    alternatives = [name for option in schema.get("anyOf", []) for name in option.get("required", [])]
    if len(alternatives) == 1:
        return required + alternatives, []
    return required, [alternatives] if alternatives else []


def render_list(catalog: list[dict], *, width: int, source: str, prefix: str | None = None) -> str:
    """All operations, grouped by namespace, one line each."""
    entries = [op for op in catalog if prefix is None or op["name"] == prefix or op["name"].startswith(prefix + ".")]
    lines = ["Daemon operations %s. Call one with:" % source, "  sparkrun daemon call <operation> --data '<json>' [--wait]", ""]
    groups: dict[str, list[dict]] = {}
    for op in entries:
        parts = op["name"].split(".")
        groups.setdefault(".".join(parts[:2]) if parts[0] == "plugins" else parts[0], []).append(op)
    for group in sorted(groups):
        lines.append(group)
        # Sized per group: one long plugin name must not squeeze every description.
        name_width = max(len(op["name"]) for op in groups[group]) + 1
        for op in sorted(groups[group], key=lambda o: o["name"]):
            head = "  %-*s %-6s %-8s" % (name_width, op["name"], op.get("kind", "?"), op.get("scope", "?"))
            room = max(20, width - len(head) - 1)
            description = op.get("description") or ""
            if len(description) > room:
                description = description[: room - 3].rstrip() + "..."
            lines.append(head + " " + description)
        lines.append("")
    lines += [
        "kind:  query = answered directly; job = returns a job ID (add --wait to follow it)",
        "scope: what a token must allow to call it (read < operate < admin)",
        "Details and request fields: sparkrun daemon help <operation>",
    ]
    return "\n".join(lines)


def render_operation(op: dict, *, width: int, argument_docs: dict[str, str] | None = None, children: list[str] | None = None) -> str:
    """One operation: what it does, how it is called, its request fields and an example to edit.

    *children* names operations under this one's namespace (``catalog.registries``
    is an operation and the prefix of ``catalog.registries.add``), which the
    detail view would otherwise hide.
    """
    argument_docs = argument_docs or {}
    schema = op.get("request_schema") or {}
    properties = schema.get("properties", {})
    required, one_of = _required_groups(schema)
    wrap = textwrap.TextWrapper(width=max(40, width), initial_indent="  ", subsequent_indent="  ")
    lines = [op["name"], *wrap.wrap(op.get("description") or "(no description)"), ""]
    kind = op.get("kind", "?")
    lines.append("  kind:   %s (%s)" % (kind, _KINDS.get(kind, kind)))
    lines.append("  scope:  %s" % op.get("scope", "?"))
    if op.get("http_path"):
        lines.append("  HTTP:   POST %s (or POST /v1/ops/%s)" % (op["http_path"], op["name"]))
    lines.append("")

    if properties:
        lines.append("Request fields (* = required):")
        name_width = max(len(name) for name in properties) + 2
        type_width = min(max(len(type_label(spec)) for spec in properties.values()), 28) + 2
        note_column = 4 + name_width + 1 + type_width + 1
        for name in sorted(properties, key=lambda n: (n not in required, n)):
            marker = "*" if name in required else " "
            label = type_label(properties[name])
            note = argument_docs.get(name, "")
            body = textwrap.wrap(note, width=max(20, width - note_column)) if note else []
            if len(label) > type_width:
                # An overlong type (a long enum) gets its own line; notes keep their column.
                lines.append("  %s %-*s %s" % (marker, name_width, name, label))
                lines.extend(" " * note_column + line for line in body)
                continue
            head = "  %s %-*s %-*s" % (marker, name_width, name, type_width, label)
            if not body:
                lines.append(head.rstrip())
                continue
            lines.append(head + " " + body[0])
            lines.extend(" " * note_column + more for more in body[1:])
        for group in one_of:
            lines.append("  At least one of: %s" % ", ".join(group))
        lines.append("")
    else:
        lines.extend(["Request fields: none", ""])

    result = op.get("result_schema") or {}
    result_fields = list((result.get("properties") or {}).keys()) or list(((result.get("items") or {}).get("properties") or {}).keys())
    if result_fields:
        shown = ", ".join(result_fields[:8]) + (", ..." if len(result_fields) > 8 else "")
        shape = "list of objects" if result.get("type") == "array" else "object"
        lines.extend(textwrap.wrap("Result: %s with %s" % (shape, shown), width=max(40, width), subsequent_indent="  "))
        lines.append("")

    example = {name: example_value(properties.get(name, {}), name) for name in required}
    if one_of and not any(name in example for name in one_of[0]):
        example[one_of[0][0]] = example_value(properties.get(one_of[0][0], {}), one_of[0][0])
    if not example and properties:
        # Never print a runnable empty call: for cluster.default, {} clears the default.
        first = sorted(properties)[0]
        example[first] = example_value(properties[first], first)
    command = "sparkrun daemon call %s --data %s" % (op["name"], shlex.quote(json.dumps(example)))
    if kind == "job":
        command += " --wait"
    lines += ["Example (replace the <placeholders>):", "  " + command]
    if children:
        lines += ["", "Operations under %s:" % op["name"], *("  " + child for child in children)]
    return "\n".join(lines)


def suggestions(catalog: list[dict], name: str) -> list[str]:
    """Operation names close to a mistyped *name* (difflib), best first."""
    names = [op["name"] for op in catalog]
    return difflib.get_close_matches(name, names, n=5, cutoff=0.5)
