# Actuator Parameter Merging System

## Overview

The actuator system in Cross-Gym implements a **parameter merging pattern** where URDF/USD default values are combined with user configuration values. This follows IsaacLab's design.

## Design Pattern

### Two-Source Parameter Resolution

```
┌─────────────────────────────────────────────────────────┐
│  Articulation._process_actuators_cfg()                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Load URDF/USD parameters from backend:             │
│     ├─ stiffness = [50.0, 50.0]                        │
│     ├─ damping = [5.0, 5.0]                            │
│     ├─ effort_limit = [100.0, 100.0]                   │
│     └─ ...                                             │
│                                                         │
│  2. Create Actuator with both sources:                 │
│     actuator_cfg.class_type(                           │
│         cfg=actuator_cfg,        # Config values       │
│         stiffness=stiffness_urdf, # URDF defaults      │
│         damping=damping_urdf,                          │
│         ...                                            │
│     )                                                  │
│                                                         │
│  3. Actuator merges parameters (config priority):      │
│     ├─ If cfg.stiffness is specified → use it         │
│     └─ Else → use URDF default                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Priority Rules

**When a parameter is loaded:**

1. **Config value specified?** → Use config value
2. **Config value is None?** → Use URDF default
3. **Config value is dict?** → Use per-joint values, fallback to URDF for unspecified joints

## Parameter Types

### 1. Float (Single Value)
```python
cfg.stiffness = 100.0
# Result: all joints get 100.0
# Shape: (num_envs, num_joints)
# Value: [[100.0, 100.0, ...]]
```

### 2. Dictionary (Per-Joint)
```python
cfg.stiffness = {
    "hip_left": 100.0,
    "hip_right": 100.0,
    "knee_left": 80.0,
    "knee_right": 80.0,
}
# Result: joints get specified values
```

### 3. None (Use URDF Default)
```python
cfg.stiffness = None
# Result: values from URDF loaded by _parse_dof_parameter()
```

## Implementation: _parse_dof_parameter()

```python
def _parse_dof_parameter(
    self,
    cfg_value: float | dict[str, float] | None,
    default_value: torch.Tensor | float,
) -> torch.Tensor:
    """
    Priority: cfg_value > default_value
    
    Resolution steps:
    1. If cfg_value is None: use default_value from URDF
    2. If cfg_value is float: apply to all joints
    3. If cfg_value is dict: apply per-joint, fallback to URDF
    
    Returns: tensor with shape (num_envs, num_joints)
    """
```

## Two-Stage Command Flow

The actuator system uses two stages for commands to separate user intent from simulation reality:

### Stage 1: User Targets
```
Located in: ArticulationData
├─ dof_pos_target      # What user wants
├─ dof_vel_target      # What user wants
└─ dof_effort_target   # What user wants (feedforward)
```

### Stage 2: Simulation Targets
```
Located in: Articulation
├─ _dof_pos_target_sim     # After actuator processing
├─ _dof_vel_target_sim     # After actuator processing
└─ _dof_effort_target_sim  # After actuator processing
```

### Data Flow

```python
# User sets target
robot.set_dof_position_target([0.5, -0.5, 1.0])
    ↓
# Stored in ArticulationData
data.dof_pos_target[env_0] = [0.5, -0.5, 1.0]
    ↓
# Simulation updates
robot.write_data_to_sim()
    ├─ _apply_actuator_model()
    │   ├─ Create ActuatorCommand from user targets
    │   ├─ actuator.compute(command, pos, vel)
    │   │   └─ PD control: tau = kp*(q_target - q) + kd*(dq_target - dq)
    │   ├─ command.joint_efforts.copy_(computed_torque)
    │   └─ Assign to _dof_effort_target_sim
    │
    └─ backend.set_joint_torques(_dof_effort_target_sim)
```

## IdealPDActuator Performance

The `IdealPDActuator` uses optimized tensor operations:

### Efficient Pattern
```python
# ✅ In-place operations (no intermediate tensors)
self.computed_torque.zero_()                    # In-place zero
self.computed_torque.add_(self.stiffness * err) # In-place add
torch.clamp(..., out=self.applied_torque)       # Direct output
command.joint_efforts.copy_(self.applied_torque) # In-place copy
```

### Why In-Place Operations Matter
- `.zero_()` - Modifies tensor directly, no allocation
- `.add_()` - Accumulates in-place, no temporary tensor
- `out=` parameter - Writes result directly to buffer, no copy
- `.copy_()` - Updates reference tensor in-place

**Memory Efficiency**: Single computation = Single buffer allocation

## Configuration Example

```python
from cross_gym.actuators import IdealPDActuatorCfg
from cross_gym.assets.articulation import ArticulationCfg

@dataclass
class MyRobotCfg(ArticulationCfg):
    """Robot configuration with actuators."""
    
    actuators = {
        "legs": IdealPDActuatorCfg(
            # Match leg joints using regex
            joint_names_expr=[".*leg.*"],
            
            # Config values (override URDF)
            stiffness=100.0,        # All leg joints
            damping=10.0,
            effort_limit=None,      # Use URDF default
            velocity_limit=None,    # Use URDF default
        ),
        
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[".*arm.*"],
            
            # Per-joint configuration
            stiffness={
                "shoulder_pitch": 50.0,
                "shoulder_roll": 50.0,
                "elbow": 30.0,
            },
            damping=5.0,            # Same for all
        ),
    }
```

## Class Structure

```
ActuatorBase (Abstract)
├── Attributes:
│   ├─ cfg: ActuatorBaseCfg
│   ├─ joint_names: list[str]
│   ├─ dof_indices: torch.Tensor | slice
│   ├─ stiffness: torch.Tensor (shape: num_envs × num_joints)
│   ├─ damping: torch.Tensor
│   ├─ armature: torch.Tensor
│   ├─ friction: torch.Tensor
│   ├─ effort_limit: torch.Tensor
│   ├─ velocity_limit: torch.Tensor
│   ├─ computed_torque: torch.Tensor
│   └─ applied_torque: torch.Tensor
│
└── Methods:
    ├─ __init__(cfg, joint_names, joint_ids, num_envs, device, 
    │           stiffness, damping, armature, friction, 
    │           effort_limit, velocity_limit)
    ├─ _parse_dof_parameter(cfg_value, default_value) → Tensor
    ├─ compute(command, joint_pos, joint_vel) → ActuatorCommand (abstract)
    └─ reset(env_ids) (abstract)

IdealPDActuator(ActuatorBase)
└── Implements:
    └─ compute(): tau = kp*(q_target - q) + kd*(dq_target - dq) + tau_ff
```

## IsaacLab Alignment

✅ This implementation follows IsaacLab's design:

1. **URDF parameters loaded from backend** - Passed as defaults
2. **Config values override URDF** - Config has priority
3. **Flexible parameter specification** - Float, dict, or None
4. **Parameter resolution tracking** - Can log what values were used
5. **Efficient tensor operations** - In-place operations throughout
6. **Two-stage command flow** - Separation of user intent and simulation

## Performance Characteristics

| Operation | Cost |
|-----------|------|
| Create actuator | O(1) per joint |
| compute() call | O(n_joints) per environment |
| Parameter parsing | O(n_joints) at init time |
| Tensor copy operations | In-place, minimal overhead |

**Memory**: (num_envs × num_joints) for each parameter tensor

## Debugging

To check what parameters an actuator is using:
```python
actuator = robot.actuators["legs"]
print(f"Stiffness: {actuator.stiffness[0]}")  # First env, all joints
print(f"Damping: {actuator.damping[0]}")
print(f"Effort limit: {actuator.effort_limit[0]}")
```

## See Also

- `ActuatorBase` - `cross_gym/actuators/actuator_base.py`
- `IdealPDActuator` - `cross_gym/actuators/actuator_pd.py`
- `Articulation` - `cross_gym/assets/articulation/articulation.py`
- `ArticulationData` - `cross_gym/assets/articulation/articulation_data.py`
