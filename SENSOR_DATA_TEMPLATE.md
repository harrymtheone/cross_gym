# Sensor Data Template Pattern

## Overview

All sensors now use a consistent data container pattern with a base class `SensorBaseData` that provides common sensor pose information.

## Architecture

```
SensorBaseData (base class)
├── pos_w: Tensor[num_envs, 3]     # Sensor position in world frame
└── quat_w: Tensor[num_envs, 4]    # Sensor orientation (w,x,y,z)
    │
    ├── RayCasterData (inherits)
    │   ├── ray_directions_w: Tensor[num_envs, num_rays, 3]
    │   ├── distances: Tensor[num_envs, num_rays]
    │   ├── hit_mask: Tensor[num_envs, num_rays]
    │   ├── hit_points_w: Tensor[num_envs, num_rays, 3]
    │   └── hit_normals_w: Tensor[num_envs, num_rays, 3]
    │
    ├── ImuData (future)
    │   ├── lin_acc_b: Tensor[num_envs, 3]
    │   ├── ang_vel_b: Tensor[num_envs, 3]
    │   └── ... (other IMU measurements)
    │
    └── ContactSensorData (future)
        ├── forces_w: Tensor[num_envs, 3]
        └── ... (other contact measurements)
```

## Implementation

### 1. SensorBaseData (Template)

**File**: `cross_gym/sensors/sensor_base_data.py`

```python
@dataclass
class SensorBaseData:
    """Base data container for all sensors.
    
    Provides common sensor data that all sensors have:
    - Sensor pose in world frame (position and orientation)
    """
    
    pos_w: torch.Tensor = None
    """Sensor position in world frame. Shape: (num_envs, 3)"""
    
    quat_w: torch.Tensor = None
    """Sensor orientation in world frame (w, x, y, z). Shape: (num_envs, 4)"""
```

### 2. Sensor-Specific Data (Inherit from Template)

**Example**: `cross_gym/sensors/ray_caster/ray_caster_data.py`

```python
@dataclass
class RayCasterData(SensorBaseData):
    """Data container for ray caster sensor.
    
    Inherits sensor pose (pos_w, quat_w) from SensorBaseData.
    """
    
    # Add sensor-specific measurements
    ray_directions_w: torch.Tensor = None
    distances: torch.Tensor = None
    hit_mask: torch.Tensor = None
    # ... more fields
```

### 3. Automatic Pose Update

The `SensorBase` class automatically copies computed pose to the data container after each update:

**File**: `cross_gym/sensors/sensor_base.py`

```python
def _update_outdated_buffers(self):
    # ... compute sensor pose ...
    self._update_sensor_pose()
    
    # ... call subclass update ...
    self._update_buffers_impl(outdated_env_ids)
    
    # Automatically copy pose to data container
    if hasattr(self._data, 'pos_w') and self._data.pos_w is not None:
        self._data.pos_w.copy_(self._pos_w)
    if hasattr(self._data, 'quat_w') and self._data.quat_w is not None:
        self._data.quat_w.copy_(self._quat_w)
```

## Benefits

### 1. **Consistency**
- All sensors have pose information in the same place
- Users always know where to find sensor pose: `sensor.data.pos_w`

### 2. **No Duplication**
- Sensor-specific data classes don't need to redefine pos_w and quat_w
- Single source of truth for sensor pose structure

### 3. **Automatic Management**
- Base class handles pose updates automatically
- Subclasses don't need to manually copy pose data

### 4. **Clean Interface**
```python
# Access sensor data
data = sensor.data  # Triggers lazy evaluation

# All sensors have pose
position = data.pos_w      # (num_envs, 3)
orientation = data.quat_w  # (num_envs, 4)

# Plus sensor-specific data
distances = data.distances  # Ray caster specific
```

## Usage Example

### Creating a New Sensor

```python
from dataclasses import dataclass
import torch
from cross_gym.sensors import SensorBaseData, SensorBase

# 1. Define data container (inherit from SensorBaseData)
@dataclass
class ImuData(SensorBaseData):
    """IMU sensor data container."""
    
    # Inherits pos_w and quat_w automatically
    
    # Add IMU-specific measurements
    lin_acc_b: torch.Tensor = None  # Linear acceleration in body frame
    ang_vel_b: torch.Tensor = None  # Angular velocity in body frame
    lin_vel_b: torch.Tensor = None  # Linear velocity in body frame

# 2. Define sensor class
class Imu(SensorBase):
    def __init__(self, cfg, articulation, sim):
        super().__init__(cfg, articulation, sim)
        
        # Create data container
        self._data = ImuData()
        
        # Initialize buffers (including pos_w and quat_w!)
        self._data.pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._data.quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._data.quat_w[:, 0] = 1.0
        
        # Initialize IMU-specific buffers
        self._data.lin_acc_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._data.ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._data.lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
    
    @property
    def data(self) -> ImuData:
        self._update_outdated_buffers()
        return self._data
    
    def _update_buffers_impl(self, env_ids):
        """Update IMU measurements.
        
        Note: Sensor pose is automatically updated by base class.
        """
        # Compute IMU measurements
        # ... compute lin_acc_b, ang_vel_b, etc ...
        
        # Pose (pos_w, quat_w) is automatically copied by base class!
```

### Using the Sensor

```python
# In your environment
def get_observations(self):
    imu_data = self.sensors["imu"].data  # Lazy evaluation
    
    # Access pose (available for all sensors)
    sensor_pos = imu_data.pos_w      # (num_envs, 3)
    sensor_quat = imu_data.quat_w    # (num_envs, 4)
    
    # Access sensor-specific data
    acceleration = imu_data.lin_acc_b  # (num_envs, 3)
    angular_vel = imu_data.ang_vel_b   # (num_envs, 3)
    
    return {
        "imu_acc": acceleration,
        "imu_gyro": angular_vel,
    }
```

## Migration Guide

### Before (Old Pattern)
```python
@dataclass
class MySensorData:
    # Every sensor defined pos_w and quat_w
    pos_w: torch.Tensor = None
    quat_w: torch.Tensor = None
    
    # Sensor-specific fields
    measurement: torch.Tensor = None

class MySensor(SensorBase):
    def _update_buffers_impl(self, env_ids):
        # Had to manually copy pose
        self._data.pos_w.copy_(self.pos_w)
        self._data.quat_w.copy_(self.quat_w)
        
        # Update measurements
        # ...
```

### After (New Pattern)
```python
@dataclass
class MySensorData(SensorBaseData):
    # pos_w and quat_w inherited automatically!
    
    # Only define sensor-specific fields
    measurement: torch.Tensor = None

class MySensor(SensorBase):
    def _update_buffers_impl(self, env_ids):
        # Pose is automatically copied by base class!
        
        # Just update measurements
        # ...
```

## Design Rationale

### Why Dataclass Inheritance?
- Python dataclasses support inheritance cleanly
- Fields from base class are automatically included
- Type hints work correctly
- Zero runtime overhead

### Why Auto-Copy in Base Class?
- Reduces boilerplate in sensor implementations
- Ensures pose is always consistent
- Single place to handle pose updates
- Subclasses can focus on sensor-specific logic

### Why Keep _pos_w and _quat_w in Base?
- Computed during `_update_sensor_pose()` before subclass update
- Subclass can access via `self.pos_w` during computation
- Then automatically copied to `self._data.pos_w` after update
- Clean separation: internal computation vs public interface

## Summary

The sensor data template pattern provides:
1. ✅ **Consistency**: All sensors have pose in the same place
2. ✅ **DRY principle**: No duplicate pos_w/quat_w definitions
3. ✅ **Automatic management**: Base class handles pose updates
4. ✅ **Clean API**: Users always know where to find pose
5. ✅ **Easy extension**: Just inherit and add fields

This makes creating new sensors simpler and more maintainable!

