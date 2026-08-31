"""Plugin-owned top-level recipe item registry.

Cross-cutting integrations can claim a recipe key without teaching
``Recipe`` their schema.  A handler owns parsing, validation, and canonical
export for that key; the core only preserves lifecycle and round-trip order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from sparkrun.core.execution import ExecutionContext, PreparationStep, RecipeExecutionStrategy
    from sparkrun.core.recipe import Recipe


_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


class RecipeItemHandler(Protocol):
    """Lifecycle implemented by the plugin that owns a recipe key."""

    def parse(self, value: Any, recipe: "Recipe") -> Any: ...

    def validate(self, value: Any, recipe: "Recipe") -> list[str]: ...

    def export(self, value: Any, recipe: "Recipe") -> Any: ...


@dataclass(frozen=True)
class FunctionalRecipeItemHandler:
    """Convenience handler built from functions."""

    parse_item: Callable[[Any, "Recipe"], Any]
    validate_item: Callable[[Any, "Recipe"], list[str]] = lambda _value, _recipe: []
    export_item: Callable[[Any, "Recipe"], Any] = lambda value, _recipe: value

    def parse(self, value: Any, recipe: "Recipe") -> Any:
        return self.parse_item(value, recipe)

    def validate(self, value: Any, recipe: "Recipe") -> list[str]:
        return list(self.validate_item(value, recipe))

    def export(self, value: Any, recipe: "Recipe") -> Any:
        return self.export_item(value, recipe)


@dataclass(frozen=True)
class RecipeItemRegistration:
    key: str
    owner: str
    handler: RecipeItemHandler
    execution_strategy: "RecipeExecutionStrategy | None" = None
    preparation_steps: "Callable[[ExecutionContext], tuple[PreparationStep, ...]] | None" = None


_RECIPE_ITEMS: dict[str, RecipeItemRegistration] = {}


def register_recipe_item(
    key: str,
    handler: RecipeItemHandler,
    *,
    owner: str,
    execution_strategy: "RecipeExecutionStrategy | None" = None,
    preparation_steps: "Callable[[ExecutionContext], tuple[PreparationStep, ...]] | None" = None,
) -> None:
    """Claim a top-level recipe *key* for *owner*.

    Registration is idempotent for the same owner and handler object.  A
    second owner cannot silently reinterpret an existing recipe surface.
    """

    if not _KEY_PATTERN.fullmatch(key):
        raise ValueError("plugin recipe item key must use lowercase letters, digits, '_' or '-'")
    # Resolve lazily to keep this small registry independent of Recipe's much
    # larger import graph. Registration happens after core bootstrap, while
    # handlers are invoked during Recipe construction.
    from sparkrun.core.recipe import _KNOWN_KEYS

    if key in _KNOWN_KEYS:
        raise ValueError("plugin recipe item %r conflicts with a core recipe key" % key)
    if not owner.strip():
        raise ValueError("plugin recipe item owner is required")
    for method in ("parse", "validate", "export"):
        if not callable(getattr(handler, method, None)):
            raise TypeError("recipe item handler must implement %s()" % method)
    if execution_strategy is not None:
        if not getattr(execution_strategy, "name", ""):
            raise TypeError("recipe execution strategy must declare a non-empty name")
        for method in ("preparation_steps", "finalize_preparation", "prepare_activation", "activate"):
            if not callable(getattr(execution_strategy, method, None)):
                raise TypeError("recipe execution strategy must implement %s()" % method)
    if preparation_steps is not None and not callable(preparation_steps):
        raise TypeError("recipe preparation_steps must be callable")
    existing = _RECIPE_ITEMS.get(key)
    if existing is not None:
        if existing.owner == owner and existing.handler is handler:
            return
        raise ValueError("recipe item %r is already owned by %s" % (key, existing.owner))
    _RECIPE_ITEMS[key] = RecipeItemRegistration(
        key=key,
        owner=owner,
        handler=handler,
        execution_strategy=execution_strategy,
        preparation_steps=preparation_steps,
    )


def unregister_recipe_item(key: str, *, owner: str) -> None:
    """Remove an owner's registration (primarily for plugin test isolation)."""

    existing = _RECIPE_ITEMS.get(key)
    if existing is None:
        return
    if existing.owner != owner:
        raise ValueError("recipe item %r is owned by %s, not %s" % (key, existing.owner, owner))
    _RECIPE_ITEMS.pop(key)


def get_recipe_item(key: str) -> RecipeItemRegistration | None:
    return _RECIPE_ITEMS.get(key)


def registered_recipe_items() -> tuple[RecipeItemRegistration, ...]:
    return tuple(_RECIPE_ITEMS[key] for key in sorted(_RECIPE_ITEMS))


__all__ = [
    "FunctionalRecipeItemHandler",
    "RecipeItemHandler",
    "RecipeItemRegistration",
    "get_recipe_item",
    "register_recipe_item",
    "registered_recipe_items",
    "unregister_recipe_item",
]
