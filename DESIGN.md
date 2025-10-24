# Cross-Gym: Cross-Platform RL Framework Design

## Overview

**Cross-Gym** is a unified, cross-platform robot reinforcement learning framework inspired by IsaacLab's architecture, but designed to work seamlessly across multiple simulators (IsaacGym, Genesis, IsaacSim).

## Design Philosophy

### Key Principle: Simulator Abstraction
The core insight is that **most RL framework code is simulator-agnostic**. It defines:
- Environment structure (observations, actions, rewards, terminations)
- Training loop logic
- Data management and logging
- Task configuration

Only a small portion of code directly interacts with the simulator for:
- Physics stepping
- State reading (joint positions, velocities, contact forces)
- Command writing (joint torques, position targets)
- Scene setup (spawning robots, terrains)

**Solution**: Create a `SimulationContext` abstraction layer that provides a unified API, with simulator-specific implementations underneath.

---

## Framework Analysis

### 1. Direct Framework (`direct/`)
**Architecture:**
- Direct environment classes that inherit from base environment
- Simulator wrappers: `BaseWrapper` → `IsaacGymWrapper`, `GenesisWrapper`
- Tight coupling between environment and simulator

**Pros:**
- Simple, direct control
- Good for prototyping

**Cons:**
- Monolithic environment classes
- Hard to reuse observation/reward/action logic
- Limited modularity

### 2. Manager-Based Framework (`manager_based/`)
**Architecture:**
- Modular managers: `ActionManager`, `ObservationManager`, `RewardManager`, `TerminationManager`
- Simulator wrappers with clean separation
- Environment composes managers

**Pros:**
- Better modularity
- Easier to reuse components
- Cleaner separation of concerns

**Cons:**
- Still missing some IsaacLab features
- Terrain handling less sophisticated

### 3. IsaacLab (`isaaclab/`)
**Architecture:**
- Highly modular manager-based design
- Rich ecosystem: sensors, terrains, actuators, controllers
- Scene-based asset management
- Extensive MDP terms library

**Pros:**
- Production-ready
- Excellent modularity
- Rich feature set
- Well-documented

**Cons:**
- Tightly coupled to IsaacSim
- Uses IsaacSim-specific APIs (USD, PhysX tensors, omni.kit)

---

## Cross-Gym Architecture

### Core Design Decision
**Follow IsaacLab's architecture but replace simulator-specific code with abstraction layers.**

```
┌─────────────────────────────────────────────────────────────┐
│                         USER TASK                            │
│  (Task-specific config: observations, rewards, actions)      │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    ENVIRONMENT LAYER                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ ManagerBasedRLEnv / DirectRLEnv                        │ │
│  │  - Step loop                                           │ │
│  │  - Reset logic                                         │ │
│  │  - Gym interface                                       │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     MANAGER LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Action       │  │ Observation  │  │ Reward           │  │
│  │ Manager      │  │ Manager      │  │ Manager          │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Termination  │  │ Command      │  │ Event            │  │
│  │ Manager      │  │ Manager      │  │ Manager          │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      SCENE LAYER                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ InteractiveScene                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │ │
│  │  │ Articulation │  │ RigidObject  │  │ Sensors     │  │ │
│  │  │ (Robot)      │  │              │  │             │  │ │
│  │  └──────────────┘  └──────────────┘  └─────────────┘  │ │
│  │  ┌──────────────┐  ┌──────────────┐                   │ │
│  │  │ Terrain      │  │ Actuators    │                   │ │
│  │  └──────────────┘  └──────────────┘                   │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│           SIMULATOR ABSTRACTION LAYER                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ SimulationContext (Abstract Base Class)                │ │
│  │  - step()          - reset()                           │ │
│  │  - render()        - get_state()                       │ │
│  │  - create_articulation_view()                          │ │
│  │  - create_sensor()                                     │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
┌────────▼──────┐  ┌──────▼──────┐  ┌────▼─────────┐
│ IsaacGym      │  │ Genesis     │  │ IsaacSim     │
│ Context       │  │ Context     │  │ Context      │
│               │  │             │  │              │
│ - Native      │  │ - Native    │  │ - Native     │
│   IsaacGym    │  │   Genesis   │  │   IsaacSim   │
│   API calls   │  │   API calls │  │   API calls  │
└───────────────┘  └─────────────┘  └──────────────┘
```

---

## Key Architectural Components

### 1. Simulation Context (`cross_gym/sim/`)

**Abstract Base: `SimulationContext`**
```python
class SimulationContext(ABC):
    # Core simulation control
    @abstractmethod
    def step(self, render: bool = False)
    
    @abstractmethod
    def reset(self)
    
    @abstractmethod
    def render(self)
    
    # Asset view creation (simulator-specific)
    @abstractmethod
    def create_articulation_view(self, prim_path, num_envs)
    
    @abstractmethod
    def create_rigid_object_view(self, prim_path, num_envs)
    
    # State access
    @property
    @abstractmethod
    def device(self) -> torch.device
```

**Concrete Implementations:**
- `IsaacGymContext`: Uses isaacgym API
- `GenesisContext`: Uses genesis API  
- `IsaacSimContext`: Uses isaacsim/USD API

### 2. Asset System (`cross_gym/assets/`)

**Design Pattern:**
- Assets (Articulation, RigidObject, Sensors) are **simulator-agnostic**
- They hold data containers and configuration
- They delegate simulator interactions to **backend views**

**Backend Views:**
Each simulator implements views that handle low-level operations:
- `IsaacGymArticulationView`: Wraps IsaacGym tensor API
- `GenesisArticulationView`: Wraps Genesis entity API
- `IsaacSimArticulationView`: Wraps USD/PhysX tensor API

**Example: Articulation**
```python
class Articulation(AssetBase):
    def __init__(self, cfg: ArticulationCfg):
        self.cfg = cfg
        self.data = ArticulationData()
        self._backend = None  # Created by simulation context
    
    def update(self, dt: float):
        # Read state from backend
        self.data.joint_pos = self._backend.get_joint_positions()
        self.data.joint_vel = self._backend.get_joint_velocities()
        # ... etc
    
    def write_data_to_sim(self):
        # Write commands to backend
        self._backend.set_joint_torques(self.data.applied_torques)
```

### 3. Scene Management (`cross_gym/scene/`)

**InteractiveScene:**
- Manages all assets in the environment
- Handles environment cloning/replication
- Coordinates updates across all assets

```python
class InteractiveScene:
    def __init__(self, cfg: InteractiveSceneCfg):
        # Parse config and create assets
        self.articulations: dict[str, Articulation] = {}
        self.sensors: dict[str, SensorBase] = {}
        self.terrain: Terrain | None = None
    
    def update(self, dt: float):
        # Update all assets
        for articulation in self.articulations.values():
            articulation.update(dt)
        for sensor in self.sensors.values():
            sensor.update(dt)
```

### 4. Manager System (`cross_gym/managers/`)

Managers are **100% simulator-agnostic**. They:
- Define MDP terms (observations, rewards, actions, terminations)
- Process and combine terms
- Provide clean interfaces for environment

**Key Managers:**
- `ActionManager`: Processes raw actions → simulator commands
- `ObservationManager`: Computes observations from scene state
- `RewardManager`: Computes reward from observations/state
- `TerminationManager`: Checks termination conditions
- `CommandManager`: Generates goal commands (velocity, pose, etc.)
- `EventManager`: Handles randomization events

### 5. Environment Layer (`cross_gym/envs/`)

**Two Environment Types:**

**ManagerBasedRLEnv:**
- Modular, compositional
- Uses managers for all MDP components
- Best for complex tasks

**DirectRLEnv:**
- Monolithic, override methods directly
- Simpler for basic tasks
- Faster prototyping

### 6. MDP Terms (`cross_gym/envs/mdp/`)

Library of reusable MDP components:
- **Actions**: `JointPositionAction`, `TaskSpaceAction`, etc.
- **Observations**: `RobotState`, `TerrainScan`, etc.
- **Rewards**: `TrackingReward`, `EnergyPenalty`, etc.
- **Terminations**: `ContactTermination`, `TimeoutTermination`, etc.

All terms are simulator-agnostic!

---

## Simulator Abstraction Strategy

### Abstraction Layers:

**Layer 1: Core Simulation Control**
- Physics stepping
- Rendering
- Time management
- → Handled by `SimulationContext`

**Layer 2: Asset State Access**
- Reading joint states, body poses, contact forces
- Writing joint commands
- → Handled by Backend Views (e.g., `IsaacGymArticulationView`)

**Layer 3: Scene Setup**
- Spawning robots from URDF/USD
- Creating terrain
- Setting up environments
- → Handled by Spawner functions (simulator-specific)

### Cross-Simulator Compatibility Matrix:

| Feature | IsaacGym | Genesis | IsaacSim |
|---------|----------|---------|----------|
| **Physics** | PhysX | Custom | PhysX |
| **GPU Pipeline** | Yes | Yes | Yes |
| **Tensor API** | Native | Native | PhysX Tensors |
| **Asset Format** | URDF/MJCF | URDF | USD/URDF |
| **Terrain** | Heightfield/Trimesh | Heightfield/Trimesh | Heightfield/Trimesh |
| **Sensors** | Limited | Limited | Rich (RTX) |
| **Rendering** | Rasterization | Rasterization | Ray-tracing |

---

## Implementation Plan

### Phase 1: Core Infrastructure ✅ (Partially Done)
- [x] `SimulationContext` abstract base
- [x] `IsaacGymContext` implementation
- [x] `AssetBase` and `Articulation`
- [x] `ArticulationData`
- [x] Configuration system (`configclass`)

### Phase 2: Scene Management ✅ (Complete)
- [x] `InteractiveScene` class
- [x] `InteractiveSceneCfg`
- [x] Asset registration and cloning
- [ ] Terrain integration (pending)

### Phase 3: Managers ✅ (Complete)
- [x] `ActionManager` + action terms
- [x] `ObservationManager` + observation terms  
- [x] `RewardManager` + reward terms
- [x] `TerminationManager` + termination terms
- [x] `CommandManager` + command generators
- [x] `EventManager` + randomization events

### Phase 4: Environment ✅ (Complete)
- [x] `ManagerBasedEnv` base class
- [x] `ManagerBasedRLEnv` (with gym interface)
- [ ] `DirectRLEnv` (simpler alternative - planned)

### Phase 5: Genesis Support
- [ ] `GenesisContext` implementation
- [ ] `GenesisArticulationView`
- [ ] Genesis-specific spawners

### Phase 6: Additional Features
- [ ] Actuator models (PD controllers, motor models)
- [ ] Advanced sensors (cameras, raycasters, IMU)
- [ ] Controller library (IK, OSC, etc.)
- [ ] Terrain generators
- [ ] UI/visualization tools

---

## Example Usage

### Task Definition:
```python
from cross_gym.envs import ManagerBasedRLEnvCfg
from cross_gym.scene import InteractiveSceneCfg
from cross_gym.sim import SimulationCfg, SimulatorType

@configclass
class MySceneCfg(InteractiveSceneCfg):
    # Terrain
    terrain = TerrainImporterCfg(...)
    
    # Robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        file="path/to/robot.urdf",
    )
    
    # Height scanner
    height_scanner = RayCasterCfg(...)

@configclass  
class MyTaskCfg(ManagerBasedRLEnvCfg):
    # Simulator selection
    sim = SimulationCfg(
        simulator=SimulatorType.ISAACGYM,  # or GENESIS or ISAACSIM
        dt=0.01,
        device="cuda:0",
    )
    
    # Scene
    scene = MySceneCfg(num_envs=4096, env_spacing=4.0)
    
    # Actions
    actions = ActionManagerCfg(
        joint_pos=JointPositionActionCfg(asset_name="robot", ...),
    )
    
    # Observations
    observations = ObservationManagerCfg(
        policy=ObservationGroupCfg(
            base_lin_vel=...,
            base_ang_vel=...,
            joint_pos=...,
            ...
        )
    )
    
    # Rewards
    rewards = RewardManagerCfg(
        tracking=TrackingRewardCfg(...),
        energy_penalty=EnergyPenaltyCfg(...),
    )
```

### Training:
```python
from cross_gym.envs import ManagerBasedRLEnv

# Create environment - works with ANY simulator!
env = ManagerBasedRLEnv(cfg=MyTaskCfg())

# Standard gym interface
obs, _ = env.reset()
for _ in range(1000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
```

### Switching Simulators:
```python
# Just change one line!
cfg.sim.simulator = SimulatorType.GENESIS  # Was ISAACGYM

# Everything else stays the same!
env = ManagerBasedRLEnv(cfg=cfg)
```

---

## Benefits of This Design

1. **Cross-Platform**: Switch simulators with 1 line of config
2. **Modular**: Reuse observations, rewards, actions across tasks
3. **IsaacLab-Compatible**: Similar API, easy migration
4. **Extensible**: Add new simulators by implementing `SimulationContext`
5. **Performance**: Each simulator uses native optimizations
6. **Clean**: Separation of concerns, testable components


