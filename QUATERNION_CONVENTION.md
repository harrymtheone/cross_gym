# Quaternion Convention in Cross-Gym

## Format: (w, x, y, z)

Cross-Gym uses the **(w, x, y, z)** quaternion format throughout the framework, where:
- `w` is the scalar/real part
- `x, y, z` are the vector/imaginary parts

This is the **scalar-first** convention, also known as the **Hamilton convention**.

### Why (w, x, y, z)?

This format is used by:
- ✅ **scipy.spatial.transform.Rotation** (Python standard)
- ✅ **Eigen** (C++ library used in robotics)
- ✅ **ROS tf2** (Robot Operating System)
- ✅ **PyBullet** (physics simulator)
- ✅ **MuJoCo** (physics simulator)

### Alternative Formats

Other simulators use different conventions:

| Simulator/Library | Format | Notes |
|------------------|--------|-------|
| **Cross-Gym** | **(w, x, y, z)** | ← We use this! |
| Isaac Gym | (x, y, z, w) | Scalar-last |
| Isaac Sim | (x, y, z, w) | Scalar-last |
| Genesis | (w, x, y, z) | Scalar-first (same as us!) |

### Identity Quaternion

The identity quaternion (no rotation) is:
```python
identity = (1.0, 0.0, 0.0, 0.0)  # (w=1, x=0, y=0, z=0)
```

### Automatic Conversion

The backend simulator wrappers handle conversion automatically:

```python
# User code (always uses w, x, y, z)
articulation.data.root_quat_w  # (num_envs, 4) in (w, x, y, z) format

# Backend handles conversion
# IsaacGymArticulationView converts (w,x,y,z) ↔ (x,y,z,w) internally
```

### Example Usage

```python
import torch
from cross_gym import ArticulationCfg

# Define initial orientation (45° around Z-axis)
quat_wxyz = torch.tensor([0.9239, 0.0, 0.0, 0.3827])  # (w, x, y, z)

# Configure robot
robot_cfg = ArticulationCfg(
    init_state=ArticulationCfg.InitStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),  # Identity: (w, x, y, z)
    )
)

# Access robot orientation
robot = scene["robot"]
quat = robot.data.root_quat_w  # Always (w, x, y, z)
w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
```

### Math Utilities

All quaternion math functions use (w, x, y, z):

```python
from cross_gym.utils.math import quat_mul, quat_rotate, quat_conjugate

# Multiply quaternions
q1 = torch.tensor([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)
q2 = torch.tensor([0.707, 0.0, 0.0, 0.707])
q_result = quat_mul(q1, q2)

# Rotate vector
v = torch.tensor([1.0, 0.0, 0.0])
v_rotated = quat_rotate(q1, v)

# Conjugate (inverse for unit quaternions)
q_inv = quat_conjugate(q1)
```

### Migration from Isaac Gym

If you're migrating from Isaac Gym code that uses (x, y, z, w):

```python
# Isaac Gym format (x, y, z, w)
quat_xyzw = torch.tensor([0.0, 0.0, 0.0, 1.0])

# Convert to Cross-Gym format (w, x, y, z)
quat_wxyz = torch.cat([quat_xyzw[3:4], quat_xyzw[0:3]])

# Or for batched quaternions
quat_wxyz = torch.cat([quat_xyzw[:, 3:4], quat_xyzw[:, 0:3]], dim=-1)
```

### Important Notes

1. **Always use (w, x, y, z)** in your task code
2. **Backend handles conversion** - you don't need to worry about it
3. **Identity quaternion** is (1, 0, 0, 0), NOT (0, 0, 0, 1)
4. **Math utilities** expect (w, x, y, z) format

### Summary

✅ **Use (w, x, y, z)** everywhere in Cross-Gym  
✅ **Identity is (1, 0, 0, 0)**  
✅ **Backend handles simulator-specific conversion**  
✅ **Compatible with scipy, Eigen, ROS, PyBullet, MuJoCo**

