from sparkrun.core.recipe import Recipe
from sparkrun.runtimes._vllm_common import VllmMixin


def test_resolve_overrides_for_auto_from_overrides():
    """Test that 'auto' in overrides is preserved."""
    mixin = VllmMixin()
    recipe = Recipe.from_dict({"name": "test", "model": "m", "runtime": "vllm"})
    overrides = {"max_model_len": "auto"}
    
    new_overrides = mixin.resolve_overrides_for_auto(recipe, overrides)
    assert new_overrides["max_model_len"] == "auto"


def test_resolve_overrides_for_auto_from_defaults():
    """Test that 'auto' in recipe defaults is injected into overrides."""
    mixin = VllmMixin()
    recipe = Recipe.from_dict({
        "name": "test", 
        "model": "m", 
        "runtime": "vllm", 
        "defaults": {"max_model_len": "auto"}
    })
    
    new_overrides = mixin.resolve_overrides_for_auto(recipe, {})
    # Since it's in defaults, it gets added to the overrides dict so the 
    # downstream command builder (which looks at config chain) sees it, 
    # overriding the None conversion in estimate_vram.
    assert new_overrides["max_model_len"] == "auto"


def test_resolve_overrides_for_auto_int():
    """Test that an integer value is passed through unmodified."""
    mixin = VllmMixin()
    recipe = Recipe.from_dict({"name": "test", "model": "m", "runtime": "vllm"})
    overrides = {"max_model_len": 4096}
    
    new_overrides = mixin.resolve_overrides_for_auto(recipe, overrides)
    assert new_overrides == {"max_model_len": 4096}


def test_resolve_overrides_for_auto_none():
    """Test that missing max_model_len leaves overrides unmodified."""
    mixin = VllmMixin()
    recipe = Recipe.from_dict({"name": "test", "model": "m", "runtime": "vllm"})
    
    new_overrides = mixin.resolve_overrides_for_auto(recipe, {})
    assert new_overrides == {}
