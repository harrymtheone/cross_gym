"""Math utility functions."""
from __future__ import annotations

from typing import Sequence

import torch


@torch.jit.script
def torch_rand_float(lower: float, upper: float, shape: Sequence[int], device: torch.device):
    return lower + (upper - lower) * torch.rand(shape, device=device)


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions.
    
    Args:
        q1: First quaternion (w, x, y, z)
        q2: Second quaternion (w, x, y, z)
        
    Returns:
        Product quaternion (w, x, y, z)
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


@torch.jit.script
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion.
    
    Args:
        q: Quaternion (w, x, y, z)
        v: Vector (x, y, z)
        
    Returns:
        Rotated vector
    """
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:4]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion conjugate.
    
    Args:
        q: Quaternion (w, x, y, z)
        
    Returns:
        Conjugated quaternion (w, -x, -y, -z)
    """
    return torch.cat([q[:, 0:1], -q[:, 1:4]], dim=-1)


@torch.jit.script
def quat_from_euler_xyz(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles (roll, pitch, yaw) to quaternion.
    
    Args:
        euler: Euler angles in radians (roll, pitch, yaw). Shape: (..., 3)
        
    Returns:
        Quaternion (w, x, y, z). Shape: (..., 4)
    """
    roll = euler[:, 0]
    pitch = euler[:, 1]
    yaw = euler[:, 2]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=-1)
