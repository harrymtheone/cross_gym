# Cross-Gym Framework Architecture

## Project Status

### ✅ Completed Components

1. **Core Simulation Layer** (`cross_gym/sim/`)
   - `SimulationContext`: Abstract base class for simulator abstraction
   - `IsaacGymContext`: Full IsaacGym implementation
   - `SimulationCfg`: Configuration system for simulation parameters
   - `SimulatorType`: Enum for selecting simulators

2. **Asset System** (`cross_gym/assets/`)
   - `AssetBase`: Base class for all assets
   - `Articulation`: Robot/articulated body implementation
   - `ArticulationData`: Data container for robot state
   - Backend view pattern for simulator-specific operations

3. **Scene Management** (`cross_gym/scene/`)
   - `InteractiveScene`: Scene manager for assets
   - `InteractiveSceneCfg`: Configuration for scene setup
   - Asset registration and initialization
   - Dictionary-style asset access

4. **Manager System** (`cross_gym/managers/`)
   - `ManagerBase`: Base class for all managers
   - `ActionManager`: Processes and applies actions
   - `ObservationManager`: Computes observations
   - `RewardManager`: Computes rewards with logging
   - `TerminationManager`: Checks termination conditions
   - `CommandManager`: Generates goal commands
   - `EventManager`: Handles randomization events

5. **Environment Classes** (`cross_gym/envs/`) ✅
   - `ManagerBasedEnv`: Base environment with managers
   - `ManagerBasedRLEnv`: RL environment with Gym interface
   - Configuration classes for both

### 📋 TODO

6. **Terrain System** - Not yet implemented
7. **Genesis Support** - Not yet implemented
8. **MDP Terms Library** - Not yet implemented
9. **Additional Examples** - More comprehensive examples needed

---

## Architecture Diagram

```
                                    USER LEVEL
┌─────────────────────────────────────────────────────────────────────┐
│                         Task Configuration                           │
│  @configclass                                                        │
│  class MyTaskCfg(ManagerBasedRLEnvCfg):                             │
│      sim = SimulationCfg(simulator=SimulatorType.ISAACGYM, ...)     │
│      scene = MySceneCfg(...)                                        │
│      actions = ActionManagerCfg(...)                                │
│      observations = ObservationManagerCfg(...)                      │
│      rewards = RewardManagerCfg(...)                                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────▼──────────┐
                    │   ManagerBasedRLEnv   │  ✅ Environment (gym.Env)
                    │   - step()            │
                    │   - reset()           │
                    │   - render()          │
                    └───────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼────────┐    ┌────────▼────────┐    ┌───────▼────────┐
│ ActionManager  │    │ ObservationMgr  │    │ RewardManager  │  ✅ Managers
│ ✅ Completed   │    │ ✅ Completed    │    │ ✅ Completed   │
└───────┬────────┘    └────────┬────────┘    └───────┬────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ InteractiveScene    │  ✅ Scene Management
                    │ ✅ Completed        │
                    │  - articulations    │
                    │  - sensors          │
                    │  - terrain          │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
┌───────▼────────┐    ┌────────▼────────┐    ┌──────▼─────────┐
│ Articulation   │    │ RigidObject     │    │ SensorBase     │  ✅ Assets
│ ✅ Completed   │    │ 📋 TODO         │    │ 📋 TODO        │
└───────┬────────┘    └────────┬────────┘    └──────┬─────────┘
        │                      │                     │
        └──────────────────────┼─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ SimulationContext   │  ✅ Simulator Abstraction
                    │ ✅ Completed        │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
┌───────▼──────────┐  ┌────────▼────────┐  ┌────────▼────────┐
│ IsaacGymContext  │  │ GenesisContext  │  │ IsaacSimContext │  Simulators
│ ✅ Completed     │  │ 📋 TODO         │  │ 📋 TODO         │
└──────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## Key Design Patterns

### 1. Simulator Abstraction

**Problem**: Different simulators have different APIs

**Solution**: Abstract base class `SimulationContext` with concrete implementations

```python
# User code - simulator agnostic!
cfg.sim.simulator = SimulatorType.ISAACGYM  # or GENESIS or ISAACSIM
env = ManagerBasedRLEnv(cfg)
```

### 2. Backend View Pattern

**Problem**: Assets need simulator-specific operations

**Solution**: Assets hold data, delegate operations to backend views

```python
class Articulation(AssetBase):
    def __init__(self, cfg):
        self.data = ArticulationData()  # Simulator-agnostic data
        self._backend = None  # Simulator-specific view
    
    def update(self, dt):
        # Read from simulator through backend
        self.data.joint_pos = self._backend.get_joint_positions()
```

### 3. Manager Pattern

**Problem**: Environment logic gets monolithic

**Solution**: Decompose into managers, each handling one aspect

- `ActionManager`: Raw actions → simulator commands
- `ObservationManager`: Simulator state → policy observations
- `RewardManager`: State → scalar reward
- `TerminationManager`: State → reset signal
- `EventManager`: Randomization/perturbations
- `CommandManager`: Goal generation

### 4. Configuration-Driven

**Problem**: Hard-coded parameters are inflexible

**Solution**: Everything configured through dataclasses

```python
@configclass
class MyTaskCfg(ManagerBasedRLEnvCfg):
    # Just declare, framework does the rest
    rewards = RewardManagerCfg()
    rewards.tracking = ManagerTermCfg(func=tracking_reward, weight=1.0)
    rewards.energy = ManagerTermCfg(func=energy_penalty, weight=-0.01)
```

---

## Data Flow

### Step Loop

```
1. User calls env.step(action)
                    │
                    ▼
2. ActionManager.process_action(action)
    ├─ Split actions for each term
    └─ Each ActionTerm processes its actions
                    │
                    ▼
3. FOR each decimation step:
    ├─ ActionManager.apply_action()
    │   └─ Write to articulation.data.applied_torques
    ├─ Scene.write_data_to_sim()
    │   └─ Backend views write to simulator
    ├─ SimulationContext.step()
    │   └─ Simulator steps physics
    └─ Scene.update(dt)
        └─ Backend views read from simulator
                    │
                    ▼
4. After decimation:
    ├─ RewardManager.compute()
    │   └─ Weighted sum of reward terms
    ├─ TerminationManager.compute()
    │   └─ Check termination conditions
    ├─ ObservationManager.compute()
    │   └─ Compute observations
    └─ CommandManager.compute()
        └─ Update commands if needed
                    │
                    ▼
5. Return (obs, reward, terminated, truncated, info)
```

### Reset Loop

```
1. User calls env.reset() or auto-reset after termination
                    │
                    ▼
2. EventManager.apply(mode="reset", env_ids=reset_ids)
    └─ Apply randomization events
                    │
                    ▼
3. Scene.reset(env_ids)
    └─ Reset each asset to initial state
                    │
                    ▼
4. Managers reset:
    ├─ ActionManager.reset()
    ├─ ObservationManager.reset()
    ├─ RewardManager.reset()  # Log episode rewards
    ├─ TerminationManager.reset()
    └─ CommandManager.reset()  # Generate new commands
                    │
                    ▼
5. Compute initial observation
                    │
                    ▼
6. Return (obs, info)
```

---

## Import Pattern (IsaacLab-Style)

**Goal**: Avoid circular imports between class and config

### Pattern:

1. **`__init__.py`**: Import class FIRST, then config
```python
from .my_class import MyClass
from .my_class_cfg import MyClassCfg
```

2. **`my_class.py`**: Use TYPE_CHECKING guard
```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import MyClassCfg

class MyClass:
    def __init__(self, cfg: MyClassCfg):
        ...
```

3. **`my_class_cfg.py`**: Import class directly
```python
from . import MyClass

@configclass
class MyClassCfg:
    class_type: type = MyClass
```

---

## Cross-Simulator Compatibility

### Supported Features Matrix

| Feature | IsaacGym | Genesis | IsaacSim |
|---------|----------|---------|----------|
| **Status** | ✅ Implemented | 📋 TODO | 📋 TODO |
| **Physics** | PhysX | Custom | PhysX |
| **GPU Accel** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Tensor API** | ✅ Native | ✅ Native | ✅ PhysX Tensors |
| **Asset Format** | URDF/MJCF | URDF | USD/URDF |
| **Terrain** | Heightfield/Trimesh | Heightfield/Trimesh | All |
| **Rendering** | Rasterization | Rasterization | Ray-tracing |

### Switching Simulators

```python
# Just change ONE line in config!
cfg.sim.simulator = SimulatorType.ISAACGYM  # Original
cfg.sim.simulator = SimulatorType.GENESIS   # Switch to Genesis
cfg.sim.simulator = SimulatorType.ISAACSIM  # Switch to IsaacSim

# Everything else stays the same!
env = ManagerBasedRLEnv(cfg)
obs, _ = env.reset()
```

---

## File Structure

```
cross_gym/
├── __init__.py                 # Main exports
├── sim/                        # Simulation layer ✅
│   ├── __init__.py
│   ├── simulation_context.py  # Abstract base
│   ├── simulation_cfg.py       # Configuration
│   ├── simulator_type.py       # Enum
│   └── isaacgym/              # IsaacGym implementation ✅
│       ├── isaacgym_context.py
│       ├── isaacgym_articulation_view.py
│       └── isaacgym_rigid_object_view.py
│
├── assets/                     # Asset system ✅
│   ├── __init__.py
│   ├── asset_base.py          # Base asset class
│   └── articulation/          # Articulation asset ✅
│       ├── __init__.py
│       ├── articulation.py
│       ├── articulation_cfg.py
│       └── articulation_data.py
│
├── scene/                      # Scene management ✅
│   ├── __init__.py
│   ├── interactive_scene.py
│   └── interactive_scene_cfg.py
│
├── managers/                   # Manager system ✅
│   ├── __init__.py
│   ├── manager_base.py        # Base classes
│   ├── manager_term_cfg.py
│   ├── action_manager.py      # ✅ Completed
│   ├── observation_manager.py # ✅ Completed
│   ├── reward_manager.py      # ✅ Completed
│   ├── termination_manager.py # ✅ Completed
│   ├── command_manager.py     # ✅ Completed
│   └── event_manager.py       # ✅ Completed
│
├── envs/                       # Environment classes ✅
│   ├── __init__.py
│   ├── common.py              # Common types
│   ├── manager_based_env.py   # ✅ Completed
│   ├── manager_based_env_cfg.py # ✅ Completed
│   ├── manager_based_rl_env.py# ✅ Completed
│   ├── manager_based_rl_env_cfg.py # ✅ Completed
│   └── mdp/                   # MDP terms library 📋
│
└── utils/                      # Utilities ✅
    ├── __init__.py
    ├── configclass.py         # Configuration decorator
    ├── dict.py                # Dictionary utilities
    ├── helpers.py             # Helper functions
    └── math.py                # Math utilities
```

---

## Next Steps

1. **Add MDP Terms Library** (High Priority)
   - Common observation functions
   - Common reward functions  
   - Common action terms
   - Common termination functions

2. **Create Working Example** (High Priority)
   - Full locomotion task with real robot URDF
   - Complete configuration example
   - Trainable with standard RL algorithms

3. **Terrain System** (Medium Priority)
   - Heightfield generation
   - Trimesh terrain
   - Integration with scene

4. **Implement Genesis Support** (Medium Priority)
   - `GenesisContext` implementation
   - Genesis-specific backend views
   - Testing and validation

5. **Sensors & Advanced Features** (Lower Priority)
   - Camera sensors
   - RayCaster (height scanners)
   - Controllers (IK, OSC)
   - More example tasks

