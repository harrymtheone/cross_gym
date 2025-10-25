# Tasks Package

This package contains task definitions for Cross-Gym.

---

## 📁 Structure

```
tasks/
├── __init__.py              # Task registry
├── task_registry.py         # Registry implementation
│
├── manager_based_rl/        # Manager-based tasks
│   ├── __init__.py
│   └── example_locomotion.py  # Example task
│
└── direct_rl/               # Direct RL tasks
    └── __init__.py
```

---

## 🎯 Purpose

The `tasks/` package is **separate from the core frameworks** (cross_gym, bridge_rl).

**Why?**
- ✅ Core frameworks are stable, reusable libraries
- ✅ Tasks are user-defined, project-specific
- ✅ Clean separation of framework vs application
- ✅ Easy to share tasks without sharing framework code

---

## 📝 Creating a New Task

### Manager-Based Task

1. **Create task file** in `manager_based_rl/`:

```python
# tasks/manager_based_rl/my_task.py

from cross_gym import *
from cross_gym.utils.configclass import configclass
from tasks import task_registry

@configclass
class MySceneCfg(InteractiveSceneCfg):
    num_envs: int = 4096
    robot: ArticulationCfg = ArticulationCfg(...)

@configclass
class MyTaskCfg(ManagerBasedRLEnvCfg):
    sim: IsaacGymCfg = IsaacGymCfg(...)
    scene: MySceneCfg = MySceneCfg()
    observations: ObservationManagerCfg = ...
    rewards: RewardManagerCfg = ...
    terminations: TerminationManagerCfg = ...

# Register
task_registry.register("my_task", MyTaskCfg)
```

2. **Import in `manager_based_rl/__init__.py`**:

```python
from .my_task import MyTaskCfg

__all__ = ["MyTaskCfg"]
```

3. **Use with registry**:

```python
from tasks import task_registry
from bridge_rl import OnPolicyRunnerCfg

runner = task_registry.make_runner("my_task", OnPolicyRunnerCfg, args)
runner.learn()
```

### Direct RL Task

1. **Create task file** in `direct_rl/`:

```python
# tasks/direct_rl/simple_task.py

from cross_gym.envs import DirectRLEnv, DirectRLEnvCfg
from tasks import task_registry

@configclass
class SimpleTaskCfg(DirectRLEnvCfg):
    # Direct RL configuration
    pass

# Register
task_registry.register("simple", SimpleTaskCfg, task_type="direct")
```

---

## 🔍 Task Registry API

### Register Tasks

```python
from tasks import task_registry

task_registry.register(
    name="my_task",
    task_cfg_class=MyTaskCfg,
    task_type="manager_based"  # or "direct"
)
```

### List Tasks

```python
tasks = task_registry.list_tasks()
print(f"Available tasks: {tasks}")
```

### Create Environment

```python
env = task_registry.make_env("my_task", args)
```

### Create Runner

```python
runner = task_registry.make_runner("my_task", RunnerCfg, args)
```

### Command Line Arguments

```python
parser = task_registry.get_arg_parser()
args = parser.parse_args()

# Supports:
# --task TASK_NAME
# --num_envs N
# --device cuda:0
# --headless
# --max_iterations N
# --experiment_name NAME
# --resume_path PATH
```

---

## 📚 Example Tasks

### Locomotion (Manager-Based)

See `manager_based_rl/example_locomotion.py` for a complete example of:
- Quadruped/biped forward locomotion
- Velocity tracking
- Energy efficiency
- Stability rewards

---

## 🎯 Best Practices

### Task Organization

```
tasks/manager_based_rl/
├── locomotion/
│   ├── __init__.py
│   ├── quadruped_walk.py
│   ├── biped_walk.py
│   └── rough_terrain.py
│
├── manipulation/
│   ├── __init__.py
│   ├── reach.py
│   └── grasp.py
│
└── navigation/
    └── ...
```

### Task Naming

- Use descriptive names: `"quadruped_walk"`, not `"task1"`
- Include difficulty: `"walk_flat"`, `"walk_rough"`
- Version if needed: `"walk_v1"`, `"walk_v2"`

### Configuration

- Use `MISSING` for required fields (forces users to set them)
- Provide sensible defaults for hyperparameters
- Document what each reward/observation does
- Include comments about tuning

---

## ✨ Benefits

**Separation of Concerns**:
- `cross_gym/` - Framework (stable)
- `bridge_rl/` - Training (stable)
- `tasks/` - Applications (project-specific)

**Easy Sharing**:
- Share framework: `cross_gym` + `bridge_rl`
- Share tasks: Just `tasks/` package
- Share specific task: Single file

**Clean Workflow**:
- Framework developers work in cross_gym/bridge_rl
- Task developers work in tasks/
- No mixing of concerns!

---

**Tasks package: Where your robot RL projects live!** 🤖

