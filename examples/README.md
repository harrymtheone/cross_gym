# Cross-Gym Examples

Examples demonstrating how to use Cross-Gym for robot reinforcement learning.

---

## 📝 Available Examples

### `train_example.py` - Complete Training Example

**Demonstrates**: The clean TaskRegistry pattern

**Pattern**:
```python
# 1. Define complete task config (env + algorithm + runner)
task_cfg = MyTaskCfg()

# 2. Create TaskRegistry with config
task_registry = TaskRegistry(task_cfg)

# 3. Make runner (creates env → algorithm → runner)
runner = task_registry.make()

# 4. Train!
runner.learn()
```

**Run**:
```bash
python examples/train_example.py
```

---

## 🎯 TaskCfg Structure

A complete task config contains three components:

```python
@configclass
class MyTaskCfg(TaskCfg):
    # 1. Environment (simulation, scene, managers)
    env: ManagerBasedRLEnvCfg = MyEnvCfg()
    
    # 2. Algorithm (PPO hyperparameters, networks)
    algorithm: PPOCfg = PPOCfg(
        gamma=0.99,
        learning_rate=1e-3,
        ...
    )
    
    # 3. Runner (training iterations, logging)
    runner: OnPolicyRunnerCfg = OnPolicyRunnerCfg(
        max_iterations=1000,
        project_name="my_project",
        experiment_name="exp001",
    )
```

---

## 🚀 Creating Your Own Task

### Step 1: Define Environment Configuration

```python
from cross_gym import *
from cross_gym.utils.configclass import configclass

@configclass
class MySceneCfg(InteractiveSceneCfg):
    num_envs: int = 4096
    env_spacing: float = 4.0
    
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        file="path/to/your/robot.urdf",
        init_state=ArticulationCfg.InitStateCfg(
            pos=(0.0, 0.0, 0.6),
            rot=(1.0, 0.0, 0.0, 0.0),  # (w, x, y, z)
        ),
    )

@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    sim: IsaacGymCfg = IsaacGymCfg(dt=0.01, device="cuda:0")
    scene: MySceneCfg = MySceneCfg()
    decimation: int = 2
    episode_length_s: float = 10.0
    
    # Define observations, rewards, terminations using MDP library
    observations: ObservationManagerCfg = ...
    rewards: RewardManagerCfg = ...
    terminations: TerminationManagerCfg = ...
```

### Step 2: Define Complete Task

```python
from cross_gym_tasks import TaskCfg
from cross_gym_rl.algorithms.ppo import PPOCfg
from cross_gym_rl.runners import OnPolicyRunnerCfg

@configclass
class MyTaskCfg(TaskCfg):
    env: MyEnvCfg = MyEnvCfg()
    algorithm: PPOCfg = PPOCfg(gamma=0.99, ...)
    runner: OnPolicyRunnerCfg = OnPolicyRunnerCfg(max_iterations=1000, ...)
```

### Step 3: Train

```python
from cross_gym_tasks import TaskRegistry

task_registry = TaskRegistry(MyTaskCfg())
runner = task_registry.make()
runner.learn()
```

---

## 🔧 Using the MDP Library

Cross-Gym provides 20+ ready-to-use MDP terms:

### Observations
```python
from cross_gym import mdp

observations.policy.base_vel = ManagerTermCfg(func=mdp.observations.base_lin_vel)
observations.policy.joint_pos = ManagerTermCfg(func=mdp.observations.joint_pos)
observations.policy.joint_vel = ManagerTermCfg(func=mdp.observations.joint_vel)
```

### Rewards
```python
rewards.alive = ManagerTermCfg(func=mdp.rewards.alive_reward, weight=1.0)
rewards.tracking = ManagerTermCfg(
    func=mdp.rewards.lin_vel_tracking_reward,
    weight=2.0,
    params={"target_x": 1.0}
)
rewards.energy = ManagerTermCfg(func=mdp.rewards.energy_penalty, weight=-0.01)
```

### Terminations
```python
terminations.time_out = ManagerTermCfg(func=mdp.terminations.time_out)
terminations.base_height = ManagerTermCfg(
    func=mdp.terminations.base_height_termination,
    params={"min_height": 0.3}
)
```

---

## 📚 Available MDP Terms

### Observations (10 functions)
- `base_pos`, `base_quat`, `base_lin_vel`, `base_ang_vel`
- `joint_pos`, `joint_vel`, `joint_pos_normalized`
- `body_pos`, `episode_progress`

### Rewards (8 functions)
- `alive_reward`, `lin_vel_tracking_reward`, `ang_vel_tracking_reward`
- `energy_penalty`, `torque_penalty`
- `upright_reward`, `height_reward`, `joint_acc_penalty`

### Terminations (6 functions)
- `time_out`, `base_height_termination`, `base_height_range_termination`
- `base_tilt_termination`, `base_contact_termination`, `illegal_contact_termination`

### Actions (2 classes)
- `JointPositionAction` - Position control
- `JointEffortAction` - Torque control

---

## 🔄 Switching Simulators

Just change the sim config:

```python
# Use IsaacGym
sim: IsaacGymCfg = IsaacGymCfg(dt=0.01, device="cuda:0")

# Use Genesis (when implemented)
sim: GenesisCfg = GenesisCfg(dt=0.01, device="cuda:0")
```

Everything else stays the same!

---

## 📖 Next Steps

1. **Read the documentation**: See [../docs/README.md](../docs/README.md)
2. **Explore MDP library**: Check `cross_gym/envs/mdp/`
3. **Create your task**: Copy `train_example.py` and modify
4. **Implement actions**: Add action terms (currently TODO)
5. **Provide robot URDF**: Update robot.file in scene config

---

## 💡 Tips

### Debug Mode
```python
# Use fewer environments for debugging
scene.num_envs = 16

# Non-headless for visualization
sim.headless = False
```

### Hyperparameter Tuning
```python
# Adjust PPO hyperparameters
algorithm: PPOCfg = PPOCfg(
    gamma=0.99,              # Discount factor
    lam=0.95,                # GAE lambda
    clip_param=0.2,          # PPO clip
    learning_rate=1e-3,      # Learning rate
    num_learning_epochs=5,   # Epochs per update
)
```

### Custom Networks
```python
# Adjust network architecture
actor_hidden_dims=[512, 256, 128]  # Larger network
critic_hidden_dims=[512, 256, 128]
```

---

## ❓ Troubleshooting

**ImportError: No module named 'isaacgym'**
- Install IsaacGym from https://developer.nvidia.com/isaac-gym
- `cd isaacgym/python && pip install -e .`

**Missing robot URDF**
- Provide path in `robot.file` field
- Make sure URDF is valid

**Action terms not implemented**
- This is a TODO - implement action terms in your task
- Or use the MDP library action classes

---

**Ready to train robot RL policies with Cross-Gym!** 🚀
