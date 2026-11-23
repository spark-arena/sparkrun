"""Operation documentation drawn from the library functions the operations call.

Core daemon operations are adapters over ``sparkrun.api`` functions, so their
summaries and per-argument notes already exist as those functions' docstrings.
Reading them here keeps ``GET /v1/ops`` descriptions and ``sparkrun daemon
help`` in step with the code instead of with a second copy. Display only: a
request schema never depends on any of this.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap

_ROLE = re.compile(r":[a-z]+:`~?([^`]+)`")
_LITERAL = re.compile(r"``([^`]+)``")
_EMPHASIS = re.compile(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])")
_SECTION = re.compile(r"^(Args|Arguments|Parameters|Attributes|Returns|Raises|Yields|Examples?|Notes?|Warning|See Also):\s*$")


def plain(text: str) -> str:
    """Docstring markup as plain wire-facing text (``x``, :func:`x`, *x* -> x; None -> null)."""
    text = _ROLE.sub(lambda m: m.group(1).rsplit(".", 1)[-1], text)
    text = _LITERAL.sub(r"\1", text)
    text = _EMPHASIS.sub(r"\1", text)
    # Callers send JSON; Python's None is null there.
    return re.sub(r"\bNone\b", "null", text)


def api_target(target: str):
    """The ``sparkrun.api`` object named by an operation target (``"proxy.start"``), or ``None``."""
    from sparkrun import api

    value = api
    for part in target.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def summary(obj) -> str:
    """First sentence of *obj*'s docstring, as plain text ("" when undocumented)."""
    doc = inspect.getdoc(obj) if obj is not None else None
    if not doc:
        return ""
    first = plain(" ".join(doc.split("\n\n", 1)[0].split()))
    match = re.match(r"(.+?[.!?])(\s|$)", first)
    return match.group(1) if match else first


def argument_docs(obj) -> dict[str, str]:
    """``name -> text`` from a Google-style ``Args:`` (or ``Attributes:``) section."""
    doc = inspect.getdoc(obj) if obj is not None else None
    if not doc:
        return {}
    docs: dict[str, str] = {}
    current = None
    in_args = False
    indent = None
    for line in doc.splitlines():
        section = _SECTION.match(line.strip())
        if section and not line.startswith(" "):
            in_args = section.group(1) in {"Args", "Arguments", "Parameters", "Attributes"}
            current, indent = None, None
            continue
        if not in_args or not line.strip():
            continue
        width = len(line) - len(line.lstrip())
        entry = re.match(r"\s*\*{0,2}([A-Za-z_][A-Za-z0-9_]*)(?:\s*\([^)]*\))?:\s*(.*)$", line)
        if entry and (indent is None or width <= indent):
            indent = width
            current = entry.group(1)
            docs[current] = entry.group(2).strip()
        elif current is not None and width > (indent or 0):
            docs[current] = (docs[current] + " " + line.strip()).strip()
    return {name: plain(text) for name, text in docs.items()}


def attribute_docs(cls) -> dict[str, str]:
    """``field -> first paragraph`` of the attribute docstrings on a class and its bases.

    The convention the options models use: a string literal directly after an
    annotated field. Subclasses win over the bases they refine.
    """
    docs: dict[str, str] = {}
    for klass in reversed(getattr(cls, "__mro__", ())):
        if klass is object:
            continue
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(klass)))
        except (OSError, TypeError, SyntaxError):
            continue
        body = tree.body[0].body if tree.body and isinstance(tree.body[0], ast.ClassDef) else []
        for node, following in zip(body, body[1:], strict=False):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and isinstance(following, ast.Expr)
                and isinstance(following.value, ast.Constant)
                and isinstance(following.value.value, str)
            ):
                first = inspect.cleandoc(following.value.value).split("\n\n", 1)[0]
                docs[node.target.id] = plain(" ".join(first.split()))
    return docs


def core_targets() -> dict[str, tuple[str, str | None]]:
    """``operation -> (api target, options model or None)`` for every core operation with one."""
    from sparkrun.daemon.model_cache_ops import BINDINGS
    from sparkrun.daemon.ops import OPTION_SPECS, SPECS
    from sparkrun.daemon.special_ops import SPECIAL_TARGETS
    from sparkrun.daemon.workloads import workload_target

    targets: dict[str, tuple[str, str | None]] = {name: (target, None) for name, target, *_ in SPECS}
    targets.update({name: (target, model) for name, target, model, _scope in OPTION_SPECS})
    targets.update({name: (target, None) for name, target in SPECIAL_TARGETS.items()})
    targets.update({binding.name: (binding.target, None) for binding in BINDINGS})
    for name in ("ensure_ready", "stop", "status", "sleep", "wake", "inventory", "endpoints", "plugins", "reconcile"):
        targets["workloads." + name] = (workload_target(name), None)
    return targets


#: Fields the daemon adapter adds on top of the library call (``ops.invoke_options``,
#: ``artifacts.with_diagnostics``); no library docstring describes them.
ADAPTER_FIELD_DOCS = {
    "confirm": "Ask for approval (a job prompt) after planning, before anything launches.",
    "wait_ready": "Wait for the endpoint to answer before the job completes (default true).",
    "post_launch": "Run the recipe's post-launch hooks once ready (default true).",
    "collect_diagnostics": "Capture a diagnostics bundle around the whole operation, attached to the job.",
}


def operation_docs(name: str) -> tuple[str, dict[str, str]]:
    """``(summary, argument docs)`` for a core operation, from its library function.

    An options-model operation (``workloads.run`` takes ``RunOptions``) documents
    its fields on the model, so argument notes come from there.
    """
    entry = core_targets().get(name)
    if entry is None:
        return "", {}
    target, model = entry
    function = api_target(target)
    arguments = {**ADAPTER_FIELD_DOCS, **argument_docs(function)}
    if model is not None:
        from sparkrun import api

        container = api.proxy if model.startswith("Proxy") else api
        options = getattr(container, model, None)
        arguments = {**attribute_docs(options), **argument_docs(options), **arguments}
    return summary(function), arguments
