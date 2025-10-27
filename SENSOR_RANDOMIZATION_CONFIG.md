# Sensor Randomization Configuration Guide

## Overview

The sensor randomization configuration has been updated to use nested config classes instead of tuples. This provides better clarity, type safety, and flexibility.

## New Configuration Structure

### Position Offset Randomization

```python
@configclass
class OffsetCfg:
    """Position offset randomization ranges per axis."""
    x: tuple[float, float] | None = None  # X-axis range (min, max) in meters
    y: tuple[float, float] | None = None  # Y-axis range (min, max) in meters
    z: tuple[float, float] | None = None  # Z-axis range (min, max) in meters
```

### Rotation Randomization

```python
@configclass
class RotationCfg:
    """Rotation randomization ranges per axis."""
    roll: tuple[float, float] | None = None   # Roll range (min, max) in degrees
    pitch: tuple[float, float] | None = None  # Pitch range (min, max) in degrees
    yaw: tuple[float, float] | None = None    # Yaw range (min, max) in degrees
```

## Usage Examples

### Example 1: Randomize All Axes

```python
from cross_gym.sensors import RayCasterCfg

sensor_cfg = RayCasterCfg(
    body_name="base",
    offset=(0.0, 0.0, 0.5),  # Nominal offset
    
    # Randomize position on all axes
    offset_range=RayCasterCfg.OffsetCfg(
        x=(-0.02, 0.02),  # ±2cm in x
        y=(-0.02, 0.02),  # ±2cm in y
        z=(-0.01, 0.01),  # ±1cm in z
    ),
    
    # Randomize rotation on all axes
    rotation_range=RayCasterCfg.RotationCfg(
        roll=(-5.0, 5.0),    # ±5 degrees
        pitch=(-5.0, 5.0),   # ±5 degrees
        yaw=(-10.0, 10.0),   # ±10 degrees
    ),
)
```

### Example 2: Randomize Specific Axes Only

```python
sensor_cfg = RayCasterCfg(
    body_name="base",
    
    # Only randomize x and y, keep z fixed
    offset_range=RayCasterCfg.OffsetCfg(
        x=(-0.03, 0.03),  # Randomize
        y=(-0.03, 0.03),  # Randomize
        z=None,           # No randomization
    ),
    
    # Only randomize yaw, keep roll and pitch fixed
    rotation_range=RayCasterCfg.RotationCfg(
        roll=None,          # No randomization
        pitch=None,         # No randomization
        yaw=(-15.0, 15.0),  # ±15 degrees
    ),
)
```

### Example 3: No Randomization

```python
sensor_cfg = RayCasterCfg(
    body_name="base",
    offset=(0.0, 0.0, 0.5),
    
    # Use default (no randomization)
    offset_range=RayCasterCfg.OffsetCfg(),
    rotation_range=RayCasterCfg.RotationCfg(),
)

# Or simply don't specify them (defaults to no randomization)
sensor_cfg = RayCasterCfg(
    body_name="base",
    offset=(0.0, 0.0, 0.5),
)
```

### Example 4: Asymmetric Ranges

```python
sensor_cfg = RayCasterCfg(
    body_name="base",
    
    # Asymmetric ranges (useful for systematic biases)
    offset_range=RayCasterCfg.OffsetCfg(
        x=(-0.01, 0.03),  # More likely to be forward
        y=(-0.02, 0.02),  # Symmetric
        z=(0.0, 0.02),    # Only positive (upward bias)
    ),
    
    rotation_range=RayCasterCfg.RotationCfg(
        roll=(-2.0, 2.0),
        pitch=(-3.0, 1.0),  # Slight downward bias
        yaw=(-10.0, 10.0),
    ),
)
```

### Example 5: IMU with Mounting Uncertainty

```python
imu_cfg = ImuCfg(
    body_name="base",
    offset=(0.0, 0.0, 0.05),  # Mounted 5cm above base
    
    # Small position uncertainty (manufacturing tolerance)
    offset_range=ImuCfg.OffsetCfg(
        x=(-0.002, 0.002),  # ±2mm
        y=(-0.002, 0.002),  # ±2mm
        z=(-0.001, 0.001),  # ±1mm
    ),
    
    # Mounting angle uncertainty
    rotation_range=ImuCfg.RotationCfg(
        roll=(-1.0, 1.0),   # ±1 degree
        pitch=(-1.0, 1.0),  # ±1 degree
        yaw=(-2.0, 2.0),    # ±2 degrees
    ),
)
```

## Migration from Old Format

### Old Format (Deprecated)
```python
# Old tuple-based format (NO LONGER SUPPORTED)
offset_range=((−0.02, 0.02), (−0.02, 0.02), (−0.01, 0.01))
rotation_range=((−5.0, 5.0), (−5.0, 5.0), (−10.0, 10.0))
```

### New Format
```python
# New nested config format
offset_range=SensorCfg.OffsetCfg(
    x=(-0.02, 0.02),
    y=(-0.02, 0.02),
    z=(-0.01, 0.01),
)
rotation_range=SensorCfg.RotationCfg(
    roll=(-5.0, 5.0),
    pitch=(-5.0, 5.0),
    yaw=(-10.0, 10.0),
)
```

## Benefits of New Format

### 1. **Clarity**
```python
# Old: Which axis is which?
offset_range=((0.01, 0.02), (-0.03, 0.01), (0.0, 0.05))

# New: Crystal clear!
offset_range=OffsetCfg(
    x=(0.01, 0.02),   # X-axis
    y=(-0.03, 0.01),  # Y-axis
    z=(0.0, 0.05),    # Z-axis
)
```

### 2. **Selective Randomization**
```python
# Old: Had to specify all axes, use (0.0, 0.0) to disable
offset_range=((0.01, 0.02), (0.0, 0.0), (0.0, 0.0))  # Only X

# New: Just omit what you don't want!
offset_range=OffsetCfg(
    x=(0.01, 0.02),  # Only X
    # y and z stay at nominal values
)
```

### 3. **Type Safety**
```python
# Old: Easy to make mistakes
offset_range=((0.01,), (-0.02, 0.02), (-0.01, 0.01))  # Oops! Wrong tuple size

# New: Type checker catches errors
offset_range=OffsetCfg(
    x=0.01,  # Error: Expected tuple[float, float], got float
)
```

### 4. **Self-Documenting**
```python
# New format is self-documenting
rotation_range=RotationCfg(
    roll=(-5.0, 5.0),    # Roll axis rotation
    pitch=None,          # No pitch randomization
    yaw=(-10.0, 10.0),   # Yaw axis rotation
)
```

## Implementation Details

### How Randomization Works

1. **Initialization**: Random offsets/rotations are sampled once when the sensor is created
2. **Per-Environment**: Each environment gets its own unique random values
3. **Fixed During Training**: Values stay constant across episodes (realistic hardware doesn't change!)
4. **Reset Behavior**: Can optionally resample on environment reset (not implemented by default)

### Internal Storage

```python
# Nominal values (from config)
self._offset_pos      # Base position offset
self._offset_quat     # Base rotation offset

# Randomized values (nominal + random)
self._offset_pos_sim  # Actual position used in simulation
self._offset_quat_sim # Actual rotation used in simulation
```

### Checking if Randomization is Enabled

```python
def _has_randomization(self) -> bool:
    """Check if any randomization is configured."""
    # Check position
    if self.cfg.offset_range is not None:
        if (cfg.offset_range.x is not None or 
            cfg.offset_range.y is not None or 
            cfg.offset_range.z is not None):
            return True
    
    # Check rotation
    if self.cfg.rotation_range is not None:
        if (cfg.rotation_range.roll is not None or 
            cfg.rotation_range.pitch is not None or 
            cfg.rotation_range.yaw is not None):
            return True
    
    return False
```

## Use Cases

### 1. **Sim-to-Real Transfer**
```python
# Add uncertainty matching real hardware tolerances
offset_range=OffsetCfg(x=(-0.005, 0.005), y=(-0.005, 0.005))
rotation_range=RotationCfg(roll=(-2.0, 2.0), pitch=(-2.0, 2.0))
```

### 2. **Domain Randomization**
```python
# Large variations for robustness
offset_range=OffsetCfg(x=(-0.05, 0.05), y=(-0.05, 0.05), z=(-0.02, 0.02))
rotation_range=RotationCfg(roll=(-10.0, 10.0), pitch=(-10.0, 10.0), yaw=(-15.0, 15.0))
```

### 3. **Ablation Studies**
```python
# Test sensitivity to specific axes
# Experiment 1: Only X-axis uncertainty
offset_range=OffsetCfg(x=(-0.03, 0.03))

# Experiment 2: Only yaw uncertainty
rotation_range=RotationCfg(yaw=(-15.0, 15.0))
```

### 4. **Manufacturing Tolerance Simulation**
```python
# Realistic manufacturing tolerances
offset_range=OffsetCfg(
    x=(-0.001, 0.001),  # ±1mm precision
    y=(-0.001, 0.001),
    z=(-0.0005, 0.0005),  # Higher precision in Z
)
rotation_range=RotationCfg(
    roll=(-0.5, 0.5),   # ±0.5 degree mounting accuracy
    pitch=(-0.5, 0.5),
    yaw=(-1.0, 1.0),
)
```

## Summary

The new nested config format provides:
- ✅ **Clarity**: Named fields instead of positional tuples
- ✅ **Flexibility**: Randomize only specific axes
- ✅ **Type Safety**: Better error detection
- ✅ **Self-Documentation**: Code is easier to understand
- ✅ **Maintainability**: Easier to extend in the future

Use this format for all sensor configurations going forward!

