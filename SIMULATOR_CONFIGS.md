# Simulator-Specific Configurations

Cross-Gym uses **simulator-specific configuration classes** instead of a monolithic config. This is more elegant and follows the same `class_type` pattern used for assets.

---

## Pattern

Each simulator has its own config class:

```python
@configclass
class IsaacGymCfg(SimCfgBase):
    class_type: type = IsaacGymContext  # Points to the context class
    
    # IsaacGym-specific parameters
    physx: PhysxCfg = PhysxCfg()
    substeps: int = 1
    # ...
```

---

## Usage

### IsaacGym

```python
from cross_gym import IsaacGymCfg, PhysxCfg

# Create IsaacGym simulation config
sim_cfg = IsaacGymCfg(
    dt=0.01,
    device="cuda:0",
    headless=True,
    physx=PhysxCfg(
        solver_type=1,  # TGS
        num_position_iterations=4,
        num_velocity_iterations=1,
    ),
)

# Use in task
@configclass
class MyTaskCfg(ManagerBasedRLEnvCfg):
    sim: IsaacGymCfg = IsaacGymCfg(...)
```

### Genesis

```python
from cross_gym import GenesisCfg

# Create Genesis simulation config
sim_cfg = GenesisCfg(
    dt=0.01,
    device="cuda:0",
    headless=True,
    sim_options=GenesisSimOptionsCfg(
        substeps=1,
    ),
    rigid_options=GenesisRigidOptionsCfg(
        enable_collision=True,
        constraint_solver="Newton",
    ),
)

# Use in task
@configclass  
class MyTaskCfg(ManagerBasedRLEnvCfg):
    sim: GenesisCfg = GenesisCfg(...)
```

---

## Switching Simulators

Just change the config class used!

```python
# Option 1: IsaacGym
@configclass
class MyTaskCfg(ManagerBasedRLEnvCfg):
    sim: IsaacGymCfg = IsaacGymCfg(dt=0.01, device="cuda:0")
    # ... rest of config

# Option 2: Genesis (same task, different simulator!)
@configclass
class MyTaskCfg(ManagerBasedRLEnvCfg):
    sim: GenesisCfg = GenesisCfg(dt=0.01, device="cuda:0")
    # ... rest of config (unchanged!)
```

---

## Benefits

✅ **No Supersets**: Each simulator config contains ONLY its parameters  
✅ **Type Safety**: IDE autocomplete shows correct parameters  
✅ **Follows Pattern**: Same `class_type` pattern as ArticulationCfg → Articulation  
✅ **Extensible**: Easy to add new simulators  
✅ **Clean**: No unused parameters polluting the config

---

## Available Simulators

### IsaacGym ✅ (Implemented)

```python
from cross_gym import IsaacGymCfg, PhysxCfg

IsaacGymCfg(
    # Common
    dt=0.01,
    device="cuda:0",
    gravity=(0.0, 0.0, -9.81),
    headless=True,
    
    # IsaacGym-specific
    physx=PhysxCfg(...),
    substeps=1,
    up_axis="z",
    use_gpu_pipeline=True,
)
```

### Genesis 🚧 (Planned)

```python
from cross_gym import GenesisCfg

GenesisCfg(
    # Common
    dt=0.01,
    device="cuda:0",
    gravity=(0.0, 0.0, -9.81),
    headless=True,
    
    # Genesis-specific
    sim_options=GenesisSimOptionsCfg(...),
    rigid_options=GenesisRigidOptionsCfg(...),
    viewer_options=GenesisViewerOptionsCfg(...),
    backend="gpu",
)
```

### IsaacSim 📋 (Planned)

```python
from cross_gym import IsaacSimCfg

IsaacSimCfg(
    # Common
    dt=0.01,
    device="cuda:0",
    gravity=(0.0, 0.0, -9.81),
    headless=True,
    
    # IsaacSim-specific  
    # ... USD-specific parameters
)
```

---

## Migration from Old Pattern

### Old (Not Elegant):
```python
# One big config with all simulator parameters
sim: SimulationCfg = SimulationCfg(
    simulator=SimulatorType.ISAACGYM,  # Select simulator
    dt=0.01,
    physx=PhysxCfg(...),  # IsaacGym/IsaacSim only
    # ... lots of parameters that may not apply to all simulators
)
```

### New (Elegant):
```python
# Simulator-specific config with class_type
sim: IsaacGymCfg = IsaacGymCfg(
    dt=0.01,
    physx=PhysxCfg(...),  # Only IsaacGym parameters!
)
# class_type = IsaacGymContext is already in IsaacGymCfg
```

---

## Implementation Note

The environment automatically uses `cfg.sim.class_type` to instantiate the correct simulator:

```python
# In ManagerBasedEnv.__init__
sim_class = self.cfg.sim.class_type  # Get IsaacGymContext (or GenesisContext, etc.)
self.sim = sim_class(self.cfg.sim)    # Create the simulator
```

No manual switch statements needed! The config class itself determines which simulator to use. 🎯

