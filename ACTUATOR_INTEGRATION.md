# Actuator Integration Pattern (IsaacLab Style)

How actuators integrate with articulation in Cross-Gym.

---

## 🎯 Actuator Flow in IsaacLab

### 1. **Configuration** (Define actuators)
```python
@configclass
class MyRobotCfg(ArticulationCfg):
    # Define actuator groups
    actuators: dict[str, ActuatorBaseCfg] = {
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],
            stiffness=20.0,
            damping=0.5,
            effort_limit=33.5,
        ),
        "arms": DCMotorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*"],
            stiffness=10.0,
            damping=0.2,
        ),
    }
```

### 2. **Initialization** (Create actuators)
```python
# In Articulation.__init__ or initialize()
def _process_actuators_cfg(self):
    self.actuators = {}
    
    for actuator_name, actuator_cfg in self.cfg.actuators.items():
        # Find joints matching the patterns
        joint_names = []
        joint_ids = []
        for pattern in actuator_cfg.joint_names_expr:
            for i, name in enumerate(self.dof_names):
                if re.match(pattern, name):
                    joint_names.append(name)
                    joint_ids.append(i)
        
        # Create actuator instance
        actuator = actuator_cfg.class_type(
            cfg=actuator_cfg,
            num_envs=self.num_envs,
            num_joints=len(joint_ids),
            joint_names=joint_names,
            joint_ids=joint_ids,
            stiffness=...,  # Parse from cfg
            damping=...,    # Parse from cfg
            device=self.device,
        )
        
        self.actuators[actuator_name] = actuator
```

### 3. **Set Targets** (User calls)
```python
# User sets desired joint positions
robot.set_joint_position_target(target_positions)

# This stores in articulation data
def set_joint_position_target(self, target):
    self.data.joint_pos_target[:] = target
```

### 4. **Apply Actuators** (write_data_to_sim)
```python
def write_data_to_sim(self):
    # Apply actuator models
    self._apply_actuator_model()
    
    # Write computed torques to simulation
    self.backend.set_joint_torques(self.data.applied_torques)

def _apply_actuator_model(self):
    for actuator in self.actuators.values():
        # Get targets for this actuator's joints
        command = ActuatorCommand(
            joint_positions=self.data.joint_pos_target[:, actuator.joint_indices],
            joint_velocities=self.data.joint_vel_target[:, actuator.joint_indices],
            joint_efforts=self.data.joint_effort_target[:, actuator.joint_indices],
        )
        
        # Compute torques using actuator model
        torques = actuator.compute(
            command,
            joint_pos=self.data.dof_pos[:, actuator.joint_indices],
            joint_vel=self.data.dof_vel[:, actuator.joint_indices],
        )
        
        # Store computed torques
        self.data.applied_torques[:, actuator.joint_indices] = torques
```

---

## 📊 Data Flow

```
User:
  robot.set_joint_position_target(targets)
    ↓
ArticulationData:
  joint_pos_target = targets (stored)
    ↓
write_data_to_sim():
  _apply_actuator_model()
    ↓
For each actuator group:
  command = ActuatorCommand(positions=targets[group_joints])
  torques = actuator.compute(command, current_pos, current_vel)
  applied_torques[group_joints] = torques
    ↓
  backend.set_joint_torques(applied_torques)
    ↓
Simulation:
  Applies torques to joints
```

---

## 🎨 Required Components

### 1. **ArticulationData** needs:
```python
# Target buffers (what user sets)
joint_pos_target: torch.Tensor
joint_vel_target: torch.Tensor
joint_effort_target: torch.Tensor

# Computed torques (from actuators)
computed_torque: torch.Tensor  # Before clipping
applied_torque: torch.Tensor   # After clipping
```

### 2. **Articulation** needs:
```python
# Actuator instances
actuators: dict[str, ActuatorBase]

# Methods
def _process_actuators_cfg():
    # Create actuator instances from config

def set_joint_position_target(target):
    # Store in data.joint_pos_target

def _apply_actuator_model():
    # For each actuator: compute torques

def write_data_to_sim():
    # Apply actuators, then write to sim
```

### 3. **ArticulationCfg** needs:
```python
actuators: dict[str, ActuatorBaseCfg] = {
    "group_name": ActuatorCfg(...),
}
```

---

## 🚀 Benefits

**1. Multiple Actuator Groups**
- Different PD gains for legs vs arms
- Different motor models per group

**2. Modularity**
- Actuator models are pluggable
- Easy to add delays, motor dynamics, etc.

**3. Realistic Simulation**
- Models real motor behavior
- Not just direct torque control

---

## ✅ Next Steps for Cross-Gym

1. Add actuator fields to ArticulationData
2. Add actuator processing to Articulation
3. Implement set_joint_position_target()
4. Implement _apply_actuator_model() in write_data_to_sim()
5. Add actuators field to ArticulationCfg

Should I implement this integration?

