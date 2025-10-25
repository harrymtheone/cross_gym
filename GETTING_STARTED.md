# Getting Started with Cross-Gym

Quick start guide for building your first robot RL task with Cross-Gym.

---

## Installation

```bash
# Install dependencies
pip install torch numpy gymnasium

# Install IsaacGym (for IsaacGym backend)
# Download from https://developer.nvidia.com/isaac-gym
cd isaacgym/python && pip install -e .

# Install Cross-Gym
cd /path/to/cross_gym
pip install -e .
```

---

## Your First Task (5 Steps)

### 1. Import Cross-Gym

```python
from cross_gym import *
from cross_gym.utils.configclass import configclass
```

### 2. Define Scene

```python
@configclass
class MySceneCfg(InteractiveSceneCfg):
    num_envs: int = 4096
    env_spacing: float = 4.0
    
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        file="path/to/your/robot.urdf",
        init_state=ArticulationCfg.InitStateCfg(
            pos=(0.0, 0.0, 0.6),
            rot=(1.0, 0.0, 0.0, 0.0),  # Identity quat (w,x,y,z)
        ),
    )
```

### 3. Define Task Configuration

```python
@configclass
class MyTaskCfg(ManagerBasedRLEnvCfg):
    # Simulator
    sim: IsaacGymCfg = IsaacGymCfg(
        dt=0.01,
        device="cuda:0",
        physx=PhysxCfg(
            solver_type=1,
            num_position_iterations=4,
        ),
    )
    
    # Scene
    scene: MySceneCfg = MySceneCfg()
    
    # Episode
    decimation: int = 2
    episode_length_s: float = 10.0
    
    # Observations - use MDP library!
    observations: ObservationManagerCfg = ObservationManagerCfg()
    observations.policy = ObservationGroupCfg()
    observations.policy.base_vel = ManagerTermCfg(
        func=mdp.observations.base_lin_vel
    )
    observations.policy.joint_pos = ManagerTermCfg(
        func=mdp.observations.joint_pos
    )
    
    # Rewards - use MDP library!
    rewards: RewardManagerCfg = RewardManagerCfg()
    rewards.alive = ManagerTermCfg(
        func=mdp.rewards.alive_reward, 
        weight=1.0
    )
    rewards.energy = ManagerTermCfg(
        func=mdp.rewards.energy_penalty, 
        weight=-0.01
    )
    
    # Terminations - use MDP library!
    terminations: TerminationManagerCfg = TerminationManagerCfg()
    terminations.time_out = ManagerTermCfg(
        func=mdp.terminations.time_out
    )
```

### 4. Create Environment

```python
# Create environment
env = ManagerBasedRLEnv(cfg=MyTaskCfg())

# Reset
obs, info = env.reset()
print(f"Observation keys: {obs.keys()}")
```

### 5. Run Training Loop

```python
for episode in range(100):
    obs, info = env.reset()
    
    for step in range(500):
        # Get action from your policy
        actions = policy(obs)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)

env.close()
```

---

## 🔄 Switching Simulators

Just change the sim config class:

```python
# IsaacGym
sim: IsaacGymCfg = IsaacGymCfg(dt=0.01, device="cuda:0")

# Genesis (when implemented)
sim: GenesisCfg = GenesisCfg(dt=0.01, device="cuda:0")
```

Everything else stays the same!

---

## 📚 Next Steps

- **Examples**: See [examples/README.md](examples/README.md)
- **MDP Library**: Check available terms in `cross_gym.envs.mdp`
- **Detailed Docs**: Explore [docs/](docs/) directory
- **Configuration**: Read [docs/SIMULATOR_CONFIGS.md](docs/SIMULATOR_CONFIGS.md)

---

## 💡 Quick Tips

### Use the MDP Library

```python
from cross_gym import mdp

# Observations
mdp.observations.base_lin_vel
mdp.observations.joint_pos

# Rewards  
mdp.rewards.alive_reward
mdp.rewards.energy_penalty

# Terminations
mdp.terminations.time_out
mdp.terminations.base_height_termination
```

### Configure Actions

```python
from cross_gym.envs.mdp.actions import JointEffortAction

actions: ActionManagerCfg = ActionManagerCfg()
actions.joint_effort = ManagerTermCfg(
    func=JointEffortAction,
    params={"asset_name": "robot", "scale": 1.0}
)
```

---

## 🔍 Common Patterns

### Required vs Optional Fields

```python
# Required (use MISSING)
from dataclasses import MISSING

sim: IsaacGymCfg = MISSING  # Must be provided

# Optional (use default value)
decimation: int = 2  # Has default

# Nullable (use None)
from typing import Optional
commands: Optional[CommandManagerCfg] = None
```

### Mutable Defaults

```python
# These work automatically!
params: dict = {}  # Auto-converted to default_factory
items: list = []   # Auto-converted to default_factory
```

---

**Ready to build robot RL tasks with Cross-Gym!** 🚀

For detailed information, see the [docs/](docs/) directory.

