# Cross-Gym Usage Pattern

**The cleanest way to train robot RL policies**

---

## 🎯 Design: Everything in TaskCfg

```python
TaskCfg contains:
├── env: ManagerBasedRLEnvCfg        # Environment (sim, scene, managers)
├── algorithm: PPOCfg                 # Algorithm (hyperparameters, networks)
└── runner: OnPolicyRunnerCfg         # Runner (iterations, logging)
```

**One config, everything defined!**

---

## 🚀 Ultra-Clean Training Pattern

```python
# Step 1: Define complete task config
@configclass
class MyTaskCfg(TaskCfg):
    env: MyEnvCfg = MyEnvCfg()
    algorithm: PPOCfg = PPOCfg(...)
    runner: OnPolicyRunnerCfg = OnPolicyRunnerCfg(...)


# Step 2: Create TaskRegistry with config
task_registry = TaskRegistry(MyTaskCfg())

# Step 3: Make runner (creates env → algorithm → runner)
runner = task_registry.make()

# Step 4: Train!
runner.learn()
```

**That's it! 4 lines to train.**

---

## 🏗️ What TaskRegistry.make() Does

```
task_registry.make():
│
├─ 1. Create Environment
│     env = task_cfg.env.class_type(task_cfg.env)
│
├─ 2. Create Algorithm (with env)
│     algorithm = task_cfg.algorithm.class_type(task_cfg.algorithm, env)
│
└─ 3. Create Runner (with env + algorithm)
      runner = task_cfg.runner.class_type(task_cfg.runner, env, algorithm)

Returns: runner (ready to call .learn())
```

**No duplication - each component created once and passed forward!**

---

## 📝 Complete Example

```python
from cross_gym import *
from cross_gym.utils.configclass import configclass
from cross_gym_tasks import TaskRegistry, TaskCfg


# Define environment
@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    sim: IsaacGymCfg = IsaacGymCfg(dt=0.01, device="cuda:0")
    scene: MySceneCfg = MySceneCfg()

    observations: ObservationManagerCfg = ObservationManagerCfg()
    observations.policy = ObservationGroupCfg()
    observations.policy.base_vel = ManagerTermCfg(func=mdp.observations.base_lin_vel)

    rewards: RewardManagerCfg = RewardManagerCfg()
    rewards.alive = ManagerTermCfg(func=mdp.rewards.alive_reward, weight=1.0)

    terminations: TerminationManagerCfg = TerminationManagerCfg()
    terminations.time_out = ManagerTermCfg(func=mdp.terminations.time_out)


# Define complete task
@configclass
class MyTaskCfg(TaskCfg):
    # Environment
    env: MyEnvCfg = MyEnvCfg()

    # Algorithm
    from cross_gym_rl.algorithms.ppo import PPOCfg
    algorithm: PPOCfg = PPOCfg(
        gamma=0.99,
        lam=0.95,
        learning_rate=1e-3,
    )

    # Runner
    from cross_gym_rl.runners import OnPolicyRunnerCfg
    runner: OnPolicyRunnerCfg = OnPolicyRunnerCfg(
        max_iterations=1000,
        num_steps_per_update=24,
        project_name="my_project",
        experiment_name="exp001",
    )


# Train!
task_registry = TaskRegistry(MyTaskCfg())
runner = task_registry.make()
runner.learn()
```

---

## ✅ Design Benefits

### 1. **No Duplication**

- Environment created once ✅
- Algorithm created once ✅
- Each receives what it needs ✅

### 2. **Clean Dependencies**

```
Runner receives:
├── env (created first)
└── algorithm (created second, needs env)
```

### 3. **Single Source of Truth**

Everything in `TaskCfg`:

- Environment settings
- Algorithm hyperparameters
- Training iterations
- Logging settings

### 4. **Type-Safe**

```python
task_cfg.env          # Type: ManagerBasedRLEnvCfg
task_cfg.algorithm    # Type: PPOCfg
task_cfg.runner       # Type: OnPolicyRunnerCfg
```

---

## 🎨 Pattern Comparison

### Old Pattern ❌

```python
# Duplication and confusion
env = create_env(env_cfg)
algorithm = PPO(alg_cfg, env)
runner = Runner(runner_cfg)  # Creates env again!
```

### New Pattern ✅

```python
# Clean, no duplication
task_cfg = TaskCfg()  # Contains env + algorithm + runner
task_registry = TaskRegistry(task_cfg)
runner = task_registry.make()  # Creates once, wires correctly
runner.learn()
```

---

## 📊 Configuration Hierarchy

```
TaskCfg
├── env: ManagerBasedRLEnvCfg
│   ├── sim: IsaacGymCfg
│   ├── scene: InteractiveSceneCfg
│   ├── observations: ObservationManagerCfg
│   ├── rewards: RewardManagerCfg
│   └── terminations: TerminationManagerCfg
│
├── algorithm: PPOCfg
│   ├── gamma, lam (RL params)
│   ├── clip_param (PPO params)
│   └── actor_hidden_dims, critic_hidden_dims (network)
│
└── runner: OnPolicyRunnerCfg
    ├── max_iterations
    ├── num_steps_per_update
    └── logging settings
```

**Three configs, one unified TaskCfg!**

---

## 🎯 Summary

**Ultra-clean pattern**:

1. TaskCfg contains env + algorithm + runner
2. TaskRegistry(task_cfg) receives complete config
3. make() creates env → algorithm → runner
4. runner.learn() trains

**Benefits**:

- ✅ No duplication
- ✅ Clear dependencies
- ✅ Single source of truth
- ✅ Type-safe
- ✅ Professional

**Cross-Gym: Clean, simple, powerful!** 🚀

