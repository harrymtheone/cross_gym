# Cross-Gym Examples

This directory contains example tasks and tutorials for Cross-Gym.

## Examples

### 1. Simple Task Example (`simple_task_example.py`)

A minimal example showing how to create a basic RL task using Cross-Gym:

- Defining a scene with a robot
- Setting up observations, actions, rewards, and terminations
- Creating the environment

**Note**: This is a configuration example and won't run without a proper robot URDF and action implementation.

```bash
python examples/simple_task_example.py
```

## Creating Your Own Task

### Step 1: Define MDP Terms

Create functions for observations, rewards, and terminations:

```python
def my_observation(env) -> torch.Tensor:
    """Compute a custom observation."""
    robot = env.scene["robot"]
    return robot.data.joint_pos

def my_reward(env) -> torch.Tensor:
    """Compute a custom reward."""
    # Reward logic here
    return torch.ones(env.num_envs, device=env.device)

def my_termination(env) -> torch.Tensor:
    """Check a custom termination condition."""
    # Termination logic here
    return env.episode_length_buf >= env.max_episode_length
```

### Step 2: Create Scene Configuration

```python
from cross_gym import InteractiveSceneCfg, ArticulationCfg
from cross_gym.utils.configclass import configclass

@configclass
class MySceneCfg(InteractiveSceneCfg):
    num_envs: int = 4096
    env_spacing: float = 4.0
    
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        file="path/to/robot.urdf",
    )
```

### Step 3: Create Task Configuration

```python
from cross_gym import (
    ManagerBasedRLEnvCfg,
    IsaacGymCfg,
    PhysxCfg,
    ActionManagerCfg,
    ObservationManagerCfg,
    ObservationGroupCfg,
    RewardManagerCfg,
    TerminationManagerCfg,
    ManagerTermCfg,
)

@configclass
class MyTaskCfg(ManagerBasedRLEnvCfg):
    # Simulation - use simulator-specific config!
    sim: IsaacGymCfg = IsaacGymCfg(
        dt=0.01,
        device="cuda:0",
        physx=PhysxCfg(...),
    )
    
    # Scene
    scene: MySceneCfg = MySceneCfg()
    
    # Episode settings
    decimation: int = 2
    episode_length_s: float = 10.0
    
    # Observations
    observations: ObservationManagerCfg = ObservationManagerCfg()
    observations.policy = ObservationGroupCfg()
    observations.policy.my_obs = ManagerTermCfg(func=my_observation)
    
    # Rewards
    rewards: RewardManagerCfg = RewardManagerCfg()
    rewards.my_reward = ManagerTermCfg(func=my_reward, weight=1.0)
    
    # Terminations
    terminations: TerminationManagerCfg = TerminationManagerCfg()
    terminations.my_term = ManagerTermCfg(func=my_termination)
```

### Step 4: Create and Run Environment

```python
from cross_gym import ManagerBasedRLEnv

# Create environment
env = ManagerBasedRLEnv(cfg=MyTaskCfg())

# Reset
obs, info = env.reset()

# Run episode
for step in range(100):
    # Get actions from policy (or random)
    actions = torch.randn(env.num_envs, env.single_action_space.shape[0], device=env.device)
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(actions)

# Clean up
env.close()
```

## Switching Simulators

To switch simulators, just change the config class used:

```python
# Use IsaacGym
from cross_gym import IsaacGymCfg

@configclass
class MyTaskCfg(ManagerBasedRLEnvCfg):
    sim: IsaacGymCfg = IsaacGymCfg(dt=0.01, device="cuda:0")

# Use Genesis
from cross_gym import GenesisCfg

@configclass
class MyTaskCfg(ManagerBasedRLEnvCfg):
    sim: GenesisCfg = GenesisCfg(dt=0.01, device="cuda:0")
```

Everything else stays the same!

## Next Steps

- Check out the full documentation in `/docs`
- Read the design document in `/DESIGN.md`
- Explore the architecture in `/ARCHITECTURE.md`
- Implement your own MDP terms for custom tasks

