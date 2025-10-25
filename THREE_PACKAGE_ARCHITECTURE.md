# Cross-Gym: Three-Package Architecture

**Clean separation of framework, training, and tasks**

---

## 🎯 Architecture Overview

Cross-Gym uses a **three-package architecture** for maximum clarity and reusability:

```
1. cross_gym/    → Environment framework (core library)
2. bridge_rl/    → RL training framework (core library)
3. tasks/        → Task definitions (user applications)
```

---

## 📦 Package Purposes

### 1. **cross_gym** - Environment Framework (Core Library)

**Purpose**: Multi-simulator robot RL environments

**Contains**:

- Simulation abstraction (IsaacGym, Genesis, IsaacSim)
- Asset system (Articulation, sensors, etc.)
- Scene management
- 6 managers (Action, Observation, Reward, Termination, Command, Event)
- MDP terms library

**Users**: Framework developers, task developers

**Stability**: Core library - changes infrequently

### 2. **bridge_rl** - RL Training Framework (Core Library)

**Purpose**: Standalone RL algorithms and training

**Contains**:

- Algorithms (PPO, future: AMP, DreamWaQ, PIE, SAC)
- Neural networks (Actor-Critic, etc.)
- Experience storage (RolloutStorage, ReplayBuffer)
- Training runners
- Logging and utilities

**Users**: RL researchers, algorithm developers

**Stability**: Core library - changes infrequently

**Note**: Can be used with **any** Gymnasium environment, not just Cross-Gym!

### 3. **tasks** - Task Definitions (User Applications)

**Purpose**: Specific task implementations

**Contains**:

- `manager_based_rl/` - Manager-based task definitions
- `direct_rl/` - Direct RL task definitions
- `task_registry.py` - Task management

**Users**: Task developers, researchers

**Stability**: User-specific - changes frequently

**Note**: This is where **your project-specific code** lives!

---

## 🏗️ Directory Structure

```
cross_gym/                         # Project root
│
├── cross_gym/                     # Package 1: Environment framework
│   ├── sim/                      # Simulation layer
│   ├── assets/                   # Assets (robots, objects)
│   ├── scene/                    # Scene management
│   ├── managers/                 # 6 managers
│   ├── envs/                     # Environment classes
│   │   └── mdp/                 # MDP terms library
│   └── utils/                    # Utilities (configclass, etc.)
│
├── bridge_rl/                     # Package 2: RL training framework
│   ├── algorithms/               # Training algorithms
│   │   └── ppo/                 # PPO implementation
│   ├── modules/                  # Network building blocks
│   ├── storage/                  # Experience buffers
│   ├── runners/                  # Training orchestration
│   └── utils/                    # Logger, statistics
│
├── tasks/                         # Package 3: Task definitions
│   ├── task_registry.py         # Task registry
│   ├── manager_based_rl/        # Manager-based tasks
│   │   ├── __init__.py
│   │   └── example_locomotion.py  # Example task
│   └── direct_rl/               # Direct RL tasks
│       └── __init__.py
│
├── examples/                      # Example scripts
│   ├── train_with_registry.py   # Using task registry
│   ├── train_ppo.py             # Direct training
│   └── ...
│
└── docs/                          # Documentation
    └── ...
```

---

## 🔄 How They Work Together

```
┌─────────────────────────────────────────────────┐
│ tasks/                                          │
│                                                 │
│ - Define task configurations                   │
│ - Register tasks                                │
│ - Task-specific logic                           │
│                                                 │
│ Uses:                                           │
│   ├─ cross_gym (for environments)              │
│   └─ bridge_rl (for training)                  │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌─────────────────┐    ┌──────────────────┐
│ cross_gym/      │    │ bridge_rl/       │
│                 │    │                  │
│ - Environments  │    │ - Algorithms     │
│ - Managers      │    │ - Networks       │
│ - MDP terms     │    │ - Runners        │
│                 │    │                  │
│ Core library    │    │ Core library     │
│ (stable)        │    │ (stable)         │
└─────────────────┘    └──────────────────┘
```

---

## ✅ Benefits of Three-Package Design

### 1. **Clear Separation of Concerns**

- **cross_gym**: Environment logic (simulation, assets, managers)
- **bridge_rl**: Training logic (algorithms, networks, logging)
- **tasks**: Application logic (your specific tasks)

### 2. **Independent Development**

- Framework developers work in `cross_gym/` and `bridge_rl/`
- Task developers work in `tasks/`
- No mixing of core library and application code

### 3. **Easy Sharing**

- Share framework: `cross_gym` + `bridge_rl` (core libraries)
- Share task: Just the specific task file from `tasks/`
- Share everything: All three packages

### 4. **Reusability**

- `cross_gym` can be used without `bridge_rl` (for other RL libraries)
- `bridge_rl` can be used without `cross_gym` (for other Gym envs)
- `tasks` are specific to your project

### 5. **Stability**

- Core packages change infrequently (stable APIs)
- Tasks package changes frequently (your experiments)
- Clear what's library vs application

---

## 📝 Workflow Examples

### As Framework Developer

```python
# Work in cross_gym/ or bridge_rl/
# Add new features to core framework
# Users import your stable library
```

### As Task Developer

```python
# Work in cross_gym_tasks/
# Import stable frameworks
from cross_gym import *
from bridge_rl import PPOCfg


# Define your task
@configclass
class MyTaskCfg(ManagerBasedRLEnvCfg):
    # Your task definition
    pass


# Register
from cross_gym_tasks import task_registry

task_registry.register("my_task", MyTaskCfg)
```

### As End User

```bash
# Just use registered cross_gym_tasks
python train.py --task locomotion --experiment_name exp001

# Or create environment for inference
from cross_gym_tasks import task_registry
env = task_registry.make_env("locomotion")
```

---

## 🎨 Design Patterns

### Package Dependencies

```
tasks/
├── depends on: cross_gym (required)
└── depends on: bridge_rl (optional, for training)

bridge_rl/
└── depends on: nothing! (standalone)

cross_gym/
└── depends on: nothing! (standalone)
```

**Both core packages are standalone!**

### Import Pattern

```python
# In cross_gym_tasks/manager_based_rl/my_task.py
from cross_gym import *  # Environment framework
from bridge_rl import PPOCfg  # Training framework
from cross_gym_tasks import task_registry  # Task registry


# Define task
@configclass
class MyTaskCfg(ManagerBasedRLEnvCfg):
    ...


# Register
task_registry.register("my_task", MyTaskCfg)
```

---

## 📊 Package Comparison

| Aspect         | cross_gym    | bridge_rl    | tasks            |
|----------------|--------------|--------------|------------------|
| **Type**       | Core library | Core library | Applications     |
| **Changes**    | Rarely       | Rarely       | Frequently       |
| **Purpose**    | Environments | Training     | Your tasks       |
| **Depends on** | Nothing      | Nothing      | Both             |
| **Used by**    | tasks, users | tasks, users | End users        |
| **Stability**  | Stable API   | Stable API   | Project-specific |

---

## 🚀 Getting Started

### 1. Install Frameworks

```bash
# Both are core libraries
pip install -e .  # Installs cross_gym and bridge_rl
```

### 2. Create Your Task

```bash
# In cross_gym_tasks/manager_based_rl/
cp example_locomotion.py my_task.py
# Edit my_task.py for your robot/objective
```

### 3. Register Task

```python
# In your task file
task_registry.register("my_task", MyTaskCfg)
```

### 4. Train

```bash
python examples/train_with_registry.py --task my_task
```

---

## ✨ Summary

**Three clean, independent packages**:

1. **cross_gym** - Environment framework ✅
2. **bridge_rl** - RL training framework ✅
3. **tasks** - Your task definitions ✅

**Benefits**:

- ✅ Clear separation
- ✅ Independent development
- ✅ Easy sharing
- ✅ Professional organization

**Cross-Gym: Framework + Training + Tasks = Complete Solution!** 🎯

