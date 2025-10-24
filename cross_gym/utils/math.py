"""Math utility functions."""

import torch
from typing import Tuple


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions.
    
    Args:
        q1: First quaternion (x, y, z, w)
        q2: Second quaternion (x, y, z, w)
        
    Returns:
        Product quaternion
    """
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([x, y, z, w], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion.
    
    Args:
        q: Quaternion (x, y, z, w)
        v: Vector (x, y, z)
        
    Returns:
        Rotated vector
    """
    shape = q.shape
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion conjugate.
    
    Args:
        q: Quaternion (x, y, z, w)
        
    Returns:
        Conjugated quaternion
    """
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


__all__ = ["quat_mul", "quat_rotate", "quat_conjugate"]
