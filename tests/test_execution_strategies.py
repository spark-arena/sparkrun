from __future__ import annotations

from dataclasses import dataclass

import pytest

from sparkrun.core.execution import (
    ActivationResult,
    ExecutionContext,
    PreparationStep,
    PreparedExecution,
    resolve_recipe_execution,
    run_preparation_steps,
)
from sparkrun.core.recipe import Recipe
from sparkrun.core.recipe_items import register_recipe_item, unregister_recipe_item
from sparkrun.core.timing import Timeline


class _Handler:
    def parse(self, value, recipe):
        return value

    def validate(self, value, recipe):
        return []

    def export(self, value, recipe):
        return value


@dataclass
class _Strategy:
    name: str

    def preparation_steps(self, context):
        return (PreparationStep(self.name + ".prepare", lambda _ctx, _receipts: self.name),)

    def finalize_preparation(self, context, receipts):
        return PreparedExecution(self.name, receipts=receipts)

    def prepare_activation(self, context):
        return None

    def activate(self, context, receipt):
        return ActivationResult(0)


def _context(*items):
    recipe = Recipe.from_dict(
        {
            "recipe_version": "2",
            "model": "org/model",
            "runtime": "vllm-distributed",
            **{key: {} for key in items},
        }
    )
    plan = type("Plan", (), {"recipe": recipe})()
    return ExecutionContext(options=object(), plan=plan, sctx=object())


def test_recipe_local_strategy_is_selected_only_when_its_item_is_present():
    handler = _Handler()
    strategy = _Strategy("demo")
    register_recipe_item("strategy_demo", handler, owner="tests.demo", execution_strategy=strategy)
    try:
        selected, steps = resolve_recipe_execution(_context("strategy_demo"))
        absent, absent_steps = resolve_recipe_execution(_context())
    finally:
        unregister_recipe_item("strategy_demo", owner="tests.demo")

    assert selected is strategy
    assert [step.name for step in steps] == ["demo.prepare"]
    assert absent is None
    assert absent_steps == ()


def test_multiple_recipe_local_strategies_are_rejected():
    first = _Handler()
    second = _Handler()
    register_recipe_item("strategy_one", first, owner="tests.one", execution_strategy=_Strategy("one"))
    register_recipe_item("strategy_two", second, owner="tests.two", execution_strategy=_Strategy("two"))
    try:
        with pytest.raises(ValueError, match="multiple execution strategies"):
            resolve_recipe_execution(_context("strategy_one", "strategy_two"))
    finally:
        unregister_recipe_item("strategy_one", owner="tests.one")
        unregister_recipe_item("strategy_two", owner="tests.two")


def test_preparation_steps_run_in_dependency_order_and_cleanup_in_reverse():
    events = []
    context = _context()
    steps = (
        PreparationStep(
            "plugin.finish",
            lambda _ctx, receipts: events.append(("finish", receipts["plugin.start"])) or "done",
            requires=("plugin.start",),
            cleanup=lambda _ctx, receipt: events.append(("cleanup-finish", receipt)),
        ),
        PreparationStep(
            "plugin.start",
            lambda _ctx, _receipts: events.append(("start",)) or "ready",
            cleanup=lambda _ctx, receipt: events.append(("cleanup-start", receipt)),
        ),
        PreparationStep(
            "plugin.fail",
            lambda _ctx, _receipts: (_ for _ in ()).throw(RuntimeError("boom")),
            requires=("plugin.finish",),
        ),
    )

    with pytest.raises(RuntimeError, match="boom"):
        run_preparation_steps(context, steps)

    assert events == [
        ("start",),
        ("finish", "ready"),
        ("cleanup-finish", "done"),
        ("cleanup-start", "ready"),
    ]


def test_preparation_steps_record_named_spans_under_shared_parent():
    timeline = Timeline()
    parent = timeline.begin("execution.prepare", strategy="demo")
    context = _context()
    receipts = run_preparation_steps(
        context,
        (
            PreparationStep("demo.first", lambda *_: "ready"),
            PreparationStep("demo.second", lambda _ctx, values: values["demo.first"], requires=("demo.first",)),
        ),
        timeline=timeline,
        parent=parent,
    )
    timeline.end(parent)

    spans = {span["name"]: span for span in timeline.export()["spans"]}
    assert receipts == {"demo.first": "ready", "demo.second": "ready"}
    assert spans["demo.first"]["parent"] == parent
    assert spans["demo.second"]["parent"] == parent
    assert spans["demo.first"]["attrs"]["step"] == "demo.first"


def test_preparation_steps_reject_unknown_dependencies_and_cycles():
    context = _context()
    with pytest.raises(ValueError, match="unknown"):
        run_preparation_steps(context, (PreparationStep("one", lambda *_: None, requires=("missing",)),))
    with pytest.raises(ValueError, match="cycle"):
        run_preparation_steps(
            context,
            (
                PreparationStep("one", lambda *_: None, requires=("two",)),
                PreparationStep("two", lambda *_: None, requires=("one",)),
            ),
        )
