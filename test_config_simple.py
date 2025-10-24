"""Simple test of configclass without any imports from cross_gym."""

from dataclasses import MISSING

from cross_gym.utils.configclass import configclass

print("Testing configclass implementation...")
print("=" * 60)

# Test 1: Basic usage
print("\n[Test 1] Basic configclass")


@configclass
class SimpleConfig:
    name: str = "default"
    value: int = 42


config = SimpleConfig()
print(f"✓ name={config.name}, value={config.value}")

# Test 2: Mutable defaults
print("\n[Test 2] Mutable defaults")


@configclass
class MutableConfig:
    params: dict = {}
    items: list = []


c1 = MutableConfig()
c2 = MutableConfig()
c1.params['x'] = 1
c2.params['x'] = 2
print(f"✓ c1.params={c1.params}, c2.params={c2.params}")
assert c1.params['x'] != c2.params['x'], "Should be independent!"

# Test 3: Field without default
print("\n[Test 3] Field without default (gets MISSING)")


@configclass
class RequiredConfig:
    optional: str = "default"
    required: int  # No default


config3 = RequiredConfig(required=42)
print(f"✓ optional={config3.optional}, required={config3.required}")

# Test 4: Inheritance (the critical test!)
print("\n[Test 4] Inheritance with mixed defaults")


@configclass
class ParentConfig:
    parent_with_default: str = "parent"
    parent_params: dict = {}


@configclass
class ChildConfig(ParentConfig):
    child_required: int  # No default - this was causing the error!
    child_with_default: str = "child"


config4 = ChildConfig(child_required=100)
print(f"✓ parent_with_default={config4.parent_with_default}")
print(f"  child_required={config4.child_required}")
print(f"  child_with_default={config4.child_with_default}")

# Test 5: Explicit MISSING
print("\n[Test 5] Explicit MISSING")


@configclass
class ExplicitMissingConfig:
    required: str = MISSING
    optional: str = "default"


config5 = ExplicitMissingConfig(required="provided")
print(f"✓ required={config5.required}, optional={config5.optional}")

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("=" * 60)
