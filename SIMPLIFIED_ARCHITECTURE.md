# Simplified Multi-Simulator Architecture

## Key Insight: Single Abstraction Layer

**Before:** Double abstraction (package + wrappers inside)  
**After:** Single abstraction (package IS the implementation)

---

## New Architecture

### **cross_core/base/** - Abstract Interfaces Only

```python
# Minimal simulation interface (for environments)
class SimulationContext(ABC):
    @abstractmethod
    def step(self): pass
    
    @abstractmethod
    def reset(self): pass
    
    @abstractmethod
    def render(self): pass

# Scene owns everything
class InteractiveScene(ABC):
    @abstractmethod
    def get_articulation(self, name): pass
    
    @abstractmethod
    def get_sensor(self, name): pass
    
    @abstractmethod
    def step(self): pass
    
    @abstractmethod
    def reset(self): pass

# Articulation interface
class ArticulationBase(ABC):
    @abstractmethod
    def get_joint_positions(self): pass
    ...
```

### **cross_gym/** - Direct IsaacGym Implementation

**No intermediate wrappers!** Just implement the interfaces:

```python
# Scene owns and initializes everything
class IsaacGymInteractiveScene(InteractiveScene):
    def __init__(self, cfg, device):
        # Initialize IsaacGym directly
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(...)  # Direct API
        self.viewer = self.gym.create_viewer(...)
        
        # Build scene
        self._build_scene()  # Direct gym API calls
    
    def _build_scene(self):
        # Direct IsaacGym API - no wrappers!
        self.gym.add_triangle_mesh(...)  # Terrain
        asset = self.gym.load_asset(...)  # Assets
        self.gym.create_env(...)  # Envs
        self.gym.create_actor(...)  # Actors
        self.gym.prepare_sim(...)
    
    def step(self, render=True):
        self.gym.simulate(self.sim)  # Direct API
        self.gym.fetch_results(self.sim, True)

# Articulation uses direct gym handles
class IsaacGymArticulation(ArticulationBase):
    def __init__(self, cfg, actor_handles, gym, sim, device, num_envs):
        self.gym = gym  # Direct reference, no wrapper
        self.sim = sim
        self.actor_handles = actor_handles
    
    def get_joint_positions(self):
        # Direct IsaacGym API calls
        return self.gym.acquire_dof_state_tensor(self.sim)[...]
```

---

## What Was Eliminated

❌ **Removed SimulationContext wrapper** (`cross_gym/sim/isaacgym_context.py`)
  - Was just wrapping gym/sim handles
  - Scene now owns gym/sim directly

❌ **Removed ArticulationView wrapper** (`isaacgym_articulation_view.py`)
  - Was just wrapping actor handles
  - Articulation now uses actor handles directly

❌ **Removed build_scene() from context**
  - Scene owns its own building logic
  - More cohesive design

---

## What Remains

✅ **Abstract interfaces** (cross_core/base)
  - Define what backends must implement
  - Type safety and IDE support

✅ **Package-level separation**
  - `cross_gym` for IsaacGym
  - `cross_genesis` for Genesis (future)
  - Clean backend isolation

✅ **class_type pattern**
  - `scene = cfg.class_type(cfg, device)`
  - Clean instantiation

---

## Benefits

### **Simpler:**
- One abstraction layer instead of two
- Less boilerplate code
- Easier to understand

### **More Direct:**
- cross_gym code directly uses IsaacGym API
- No unnecessary indirection
- Better performance

### **Clearer Ownership:**
- Scene owns: simulation init + scene building + asset management
- Articulation owns: joint state access
- No confusion about responsibilities

---

## New Usage Pattern

```python
from cross_tasks.locomotion import T1LocomotionCfg
import torch

# Create task config
task_cfg = T1LocomotionCfg(num_envs=4096)
env_cfg = task_cfg.get_env_cfg()

# Create scene (scene owns everything)
device = torch.device("cuda:0")
scene = env_cfg.scene.class_type(env_cfg.scene, device)

# Use scene directly
robot = scene.get_articulation("robot")
positions = robot.get_joint_positions()

# Physics control through scene
scene.step()
scene.reset()
scene.render()
```

**No separate SimulationContext needed!**

---

## Configuration Structure

### **Before (Complex):**
```python
# Separate sim and scene configs
SimulationContextCfg:
    - dt, gravity, physx, etc.

InteractiveSceneCfg:
    - num_envs, env_spacing
    - robot, sensors, terrain

# Environment needs both
DirectRLEnvCfg:
    sim: SimulationContextCfg
    scene: InteractiveSceneCfg
```

### **After (Simplified):**
```python
# Scene config includes simulation params
IsaacGymSceneCfg:
    # Simulation
    - dt, gravity, physx, etc.
    
    # Scene
    - num_envs, env_spacing
    - robot, sensors, terrain

# Environment just needs scene
DirectRLEnvCfg:
    scene: InteractiveSceneCfg  # That's it!
```

---

## Package Structure (Simplified)

```
cross_core/
├── base/           # Abstract interfaces only
│   ├── sim_context_base.py      # Minimal interface
│   ├── scene_base.py             # Scene owns everything
│   ├── articulation_base.py
│   └── sensor_base.py
├── utils/          # Shared utilities
└── terrains/       # Terrain generation (shared)

cross_gym/
├── scene/          # Scene owns simulation + building
│   ├── interactive_scene.py      # Direct IsaacGym API
│   └── interactive_scene_cfg.py  # Includes sim params
├── assets/
│   └── articulation/
│       ├── articulation_cfg.py
│       └── isaacgym_articulation.py  # Direct API
├── sensors/        # Direct API
└── actuators/      # Direct API

cross_env/
└── envs/
    ├── vec_env.py
    ├── direct_rl_env.py          # Just uses scene
    └── direct_rl_env_cfg.py      # scene only

cross_tasks/
└── locomotion/
    └── t1_locomotion_cfg.py      # Returns scene cfg
```

---

## Files Removed

- ❌ `cross_gym/sim/` - Entire directory deleted
- ❌ `cross_gym/assets/articulation/isaacgym_articulation_view.py`

---

## Summary

✅ **Cleaner**: One abstraction layer (package boundaries)  
✅ **Simpler**: cross_gym directly uses IsaacGym API  
✅ **More Direct**: No wrapper classes inside backend packages  
✅ **Same Benefits**: Backend isolation, type safety, extensibility  
✅ **Better Performance**: One less indirection layer  

**The architecture is now much simpler while maintaining all the key benefits!**

