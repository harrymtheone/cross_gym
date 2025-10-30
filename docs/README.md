# Cross-Platform Robotics Framework

A modular framework for robot learning that supports multiple physics simulators through a clean plugin architecture.

## Architecture

The framework is organized into separate packages:

- **cross_core**: Shared utilities and abstract interfaces
- **cross_gym**: IsaacGym backend implementation
- **cross_env**: Backend-agnostic environments and managers  
- **cross_rl**: RL algorithms (PPO, DreamWAQ, etc.)
- **cross_tasks**: Task definitions with backend selection

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Key Features

✅ **Multi-Simulator Support**: Switch between simulators by changing one config parameter  
✅ **Clean Abstractions**: High-level code is completely simulator-agnostic  
✅ **Modular Design**: Each simulator backend is isolated in its own package  
✅ **Extensible**: Add new simulators without touching existing code  

## Package Structure

```
cross_gym/  (repo)
├── cross_core/         # Shared utilities and interfaces ✅
├── cross_gym/          # IsaacGym backend ✅ (partial)
├── cross_env/          # Backend-agnostic envs ⚠️ TODO
├── cross_rl/           # RL algorithms ⚠️ TODO
├── cross_tasks/        # Task definitions ⚠️ TODO
├── reference/          # Old implementations for reference
│   ├── current_cross_gym/
│   ├── current_cross_rl/
│   └── current_cross_tasks/
└── docs/               # Documentation
    ├── ARCHITECTURE.md
    └── CROSS_GYM_DESIGN.md
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For IsaacGym backend
# (Follow IsaacGym installation instructions separately)
```

### Usage Example

```python
# Define task with backend selection
from cross_tasks.locomotion import T1TaskCfg

task_cfg = T1TaskCfg(sim_backend="isaacgym")

# Create environment (automatically uses correct backend)
from cross_env import create_env

env = create_env(task_cfg)

# Train with RL algorithm (backend-agnostic)
from cross_rl.algorithms.ppo import PPO

ppo = PPO(env)
ppo.train()
```

## Backend Selection

Switch simulators by changing the config:

```python
# Use IsaacGym
task_cfg = T1TaskCfg(sim_backend="isaacgym")

# Use Genesis (when implemented)
task_cfg = T1TaskCfg(sim_backend="genesis")
```

The rest of your code stays the same!

## Implementation Status

### ✅ Completed
- Abstract base classes (cross_core)
- IsaacGym simulation context
- IsaacGym scene management
- Terrain generation
- Architecture documentation

### ⚠️ In Progress
- Sensors and actuators adaptation
- Backend-agnostic environment layer
- RL algorithms migration
- Task definitions with backend selection

### 📝 Planned
- Genesis backend
- MuJoCo backend
- Example scripts and tutorials

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md): Complete architecture documentation
- [IsaacGym Backend Design](docs/CROSS_GYM_DESIGN.md): Detailed IsaacGym implementation

## Adding a New Simulator

To add support for a new simulator:

1. Create new package `cross_<simulator>/`
2. Implement abstract interfaces from `cross_core.base`:
   - `SimulationContextBase`
   - `InteractiveSceneBase`
   - `ArticulationBase`
   - `SensorBase`
3. Add backend selection in task configs

See [docs/ARCHITECTURE.md#adding-a-new-simulator-backend](docs/ARCHITECTURE.md#adding-a-new-simulator-backend) for details.

## Migration from Old Structure

The original code has been preserved in the `reference/` folder for reference during migration:

- `reference/current_cross_gym/`: Original cross_gym package
- `reference/current_cross_rl/`: Original cross_rl package  
- `reference/current_cross_tasks/`: Original cross_tasks package

## Contributing

When contributing to backend implementations:

1. Follow the abstract interface contracts in `cross_core.base`
2. Keep backend-specific code isolated in the backend package
3. Never import simulator-specific code in cross_env or cross_rl
4. Document simulator-specific constraints

## License

[Your License Here]

## Citation

```bibtex
[Your Citation Here]
```

## Contact

[Your Contact Information]

