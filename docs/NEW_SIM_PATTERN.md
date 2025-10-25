# New Simulator Configuration Pattern

## The Problem with Super-Sets

**Old approach (not elegant)**:

```python
@configclass
class SimulationCfg:
    simulator: SimulatorType = ISAACGYM  # Manual selection
    
    # IsaacGym/IsaacSim parameters
    physx: PhysxCfg = ...
    
    # Genesis parameters  
    rigid_options: RigidOptionsCfg = ...
    
    # All simulators' parameters mixed together! ❌
```

**Problems**:

- ❌ Super-set contains parameters that don't apply to all simulators
- ❌ Confusing - which parameters apply to which simulator?
- ❌ Error-prone - easy to configure wrong parameters
- ❌ Not type-safe - IDE can't help

---

## The Elegant Solution

**Each simulator gets its own config class** with `class_type` attribute:

```python
# IsaacGym configuration
@configclass
class IsaacGymCfg(SimCfgBase):
    class_type: type = IsaacGymContext  # Determines which context to use
    
    # ONLY IsaacGym parameters
    physx: PhysxCfg = PhysxCfg()
    substeps: int = 1
    up_axis: str = "z"
    # ...

# Genesis configuration
@configclass
class GenesisCfg(SimCfgBase):
    class_type: type = GenesisContext  # Different context!
    
    # ONLY Genesis parameters
    sim_options: GenesisSimOptionsCfg = ...
    rigid_options: GenesisRigidOptionsCfg = ...
    # ...
```

---

## Usage Example

### Define Task Configuration

```python
from cross_gym import (
    ManagerBasedRLEnvCfg,
    IsaacGymCfg,  # Simulator-specific config
    PhysxCfg,
)

@configclass
class MyTaskCfg(ManagerBasedRLEnvCfg):
    # Use IsaacGym
    sim: IsaacGymCfg = IsaacGymCfg(
        dt=0.01,
        device="cuda:0",
        headless=True,
        physx=PhysxCfg(
            solver_type=1,
            num_position_iterations=4,
        ),
    )
    
    # Scene (simulator-agnostic!)
    scene: MySceneCfg = MySceneCfg(...)
    
    # Managers (simulator-agnostic!)
    observations: ObservationManagerCfg = ...
    rewards: RewardManagerCfg = ...
    # ...
```

### Switch Simulators

**Just change the config class**:

```python
# Use Genesis instead
from cross_gym import GenesisCfg

@configclass
class MyTaskCfg(ManagerBasedRLEnvCfg):
    # Same task, different simulator!
    sim: GenesisCfg = GenesisCfg(
        dt=0.01,
        device="cuda:0",
        # Genesis-specific parameters
        rigid_options=GenesisRigidOptionsCfg(...),
    )
    
    # Everything else stays the same!
    scene: MySceneCfg = MySceneCfg(...)
    observations: ObservationManagerCfg = ...
    rewards: RewardManagerCfg = ...
```

---

## How It Works

### 1. Config Determines Simulator

```python
IsaacGymCfg(...)  # class_type = IsaacGymContext
GenesisCfg(...)   # class_type = GenesisContext
```

### 2. Environment Uses class_type

```python
# In ManagerBasedEnv.__init__
sim_class = self.cfg.sim.class_type  # Get IsaacGymContext
self.sim = sim_class(self.cfg.sim)    # Create it!
```

### 3. No Manual Switching

No need for:

```python
if cfg.sim.simulator == ISAACGYM:  # ❌ OLD
    sim = IsaacGymContext(...)
elif cfg.sim.simulator == GENESIS:
    sim = GenesisContext(...)
```

Instead:

```python
sim = cfg.sim.class_type(cfg.sim)  # ✅ NEW - Works for any simulator!
```

---

## Benefits

### 1. Type Safety ✅

IDE autocomplete shows **only** the relevant parameters:

```python
# IsaacGymCfg autocomplete shows:
cfg = IsaacGymCfg(
    dt=...,
    physx=...,  # ✓ Available
    substeps=...,  # ✓ Available
    # rigid_options NOT shown - doesn't exist for IsaacGym!
)
```

### 2. No Parameter Pollution ✅

Each config contains **only** what that simulator needs:

```python
# IsaacGymCfg
- physx, substeps, up_axis  # IsaacGym parameters

# GenesisCfg  
- sim_options, rigid_options, viewer_options  # Genesis parameters

# No overlap, no confusion!
```

### 3. Follows Existing Pattern ✅

Same as assets:

```python
# Assets
ArticulationCfg.class_type = Articulation
RigidObjectCfg.class_type = RigidObject

# Simulators
IsaacGymCfg.class_type = IsaacGymContext
GenesisCfg.class_type = GenesisContext
```

### 4. Extensible ✅

Adding a new simulator:

```python
# 1. Create config
@configclass
class NewSimCfg(SimCfgBase):
    class_type: type = NewSimContext
    # ... simulator-specific parameters

# 2. Create context
class NewSimContext(SimulationContext):
    # ... implementation

# 3. Done! No changes needed elsewhere
```

---

## Comparison

| Aspect                  | Old (Super-Set)         | New (Specific Configs)     |
|-------------------------|-------------------------|----------------------------|
| **Clarity**             | ❌ All parameters mixed  | ✅ Only relevant parameters |
| **Type Safety**         | ❌ Weak                  | ✅ Strong                   |
| **Extensibility**       | ❌ Modify central config | ✅ Add new config class     |
| **Maintainability**     | ❌ Complex               | ✅ Simple                   |
| **Pattern Consistency** | ❌ Different from assets | ✅ Same as assets           |
| **User Experience**     | ❌ Confusing             | ✅ Clear                    |

---

## Summary

✅ **Each simulator has its own config class**  
✅ **Config class has `class_type` pointing to context class**  
✅ **Environment uses `class_type` to instantiate simulator**  
✅ **No manual switch statements needed**  
✅ **Type-safe, clear, and extensible**

This is the **IsaacLab way** - elegant and maintainable! 🎯

