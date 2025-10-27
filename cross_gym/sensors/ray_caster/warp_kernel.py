# Copyright (c) 2025, CrossGym Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp operations for GPU-accelerated raycasting."""

from __future__ import annotations

import torch
import warp as wp

wp.config.quiet = True
wp.init()


def raycast_mesh(
        ray_starts: torch.Tensor,
        ray_directions: torch.Tensor,
        mesh: 'wp.Mesh',
        max_dist: float = 1e6,
        return_distance: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Perform GPU raycasting using Warp.
    
    This is a wrapper around Warp's mesh raycasting that handles PyTorch tensors.
    
    Args:
        ray_starts: Ray origins. Shape: (num_envs, num_rays, 3) or (N, 3)
        ray_directions: Ray directions (should be normalized). Shape: (num_envs, num_rays, 3) or (N, 3)
        mesh: Warp mesh to raycast against
        max_dist: Maximum ray distance
        return_distance: Whether to return distances
        
    Returns:
        ray_hits: Hit positions. Shape same as input. Contains inf for misses.
        distances: Distances to hits. Shape: (..., num_rays). Contains inf for misses. 
                   None if return_distance=False.
                 
    Raises:
        ImportError: If Warp is not available
    """
    # Store original shape
    original_shape = ray_starts.shape
    device = ray_starts.device

    # Get mesh device
    mesh_device = wp.device_to_torch(mesh.device)

    # Flatten to (N, 3)
    ray_starts_flat = ray_starts.to(mesh_device).reshape(-1, 3).contiguous()
    ray_directions_flat = ray_directions.to(mesh_device).reshape(-1, 3).contiguous()
    num_rays = ray_starts_flat.shape[0]

    # Create output tensors
    ray_hits = torch.full((num_rays, 3), float('inf'), device=mesh_device, dtype=torch.float32)

    if return_distance:
        ray_distances = torch.full((num_rays,), float('inf'), device=mesh_device, dtype=torch.float32)
    else:
        ray_distances = None

    # Convert to Warp arrays
    ray_starts_wp = wp.from_torch(ray_starts_flat, dtype=wp.vec3)
    ray_directions_wp = wp.from_torch(ray_directions_flat, dtype=wp.vec3)
    ray_hits_wp = wp.from_torch(ray_hits, dtype=wp.vec3)

    if return_distance:
        ray_distances_wp = wp.from_torch(ray_distances, dtype=wp.float32)
    else:
        ray_distances_wp = wp.empty(1, dtype=wp.float32, device=mesh.device)

    # Launch raycasting kernel
    wp.launch(
        kernel=_raycast_kernel,
        dim=num_rays,
        inputs=[
            mesh.id,
            ray_starts_wp,
            ray_directions_wp,
            ray_hits_wp,
            ray_distances_wp,
            float(max_dist),
            int(return_distance),
        ],
        device=mesh.device,
    )

    # Synchronize to ensure kernel completion
    wp.synchronize()

    # Reshape back to original shape
    ray_hits = ray_hits.to(device).reshape(original_shape)

    if return_distance:
        ray_distances = ray_distances.to(device).reshape(original_shape[:-1])

    return ray_hits, ray_distances


@wp.kernel(enable_backward=False)
def _raycast_kernel(
        mesh: wp.uint64,
        ray_starts: wp.array1d(dtype=wp.vec3),
        ray_directions: wp.array1d(dtype=wp.vec3),
        ray_hits: wp.array1d(dtype=wp.vec3),
        ray_distances: wp.array1d(dtype=wp.float32),
        max_dist: float,
        return_distance: int,
):
    """Warp kernel for raycasting against a mesh.
    
    Args:
        mesh: Warp mesh ID
        ray_starts: Ray starting positions
        ray_directions: Ray directions
        ray_hits: Output hit positions (inf for misses)
        ray_distances: Output distances (inf for misses)
        max_dist: Maximum ray distance
        return_distance: Whether to compute distances
    """
    tid = wp.tid()

    # Output parameters for mesh_query_ray
    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index

    # Perform raycast query
    hit_success = wp.mesh_query_ray(
        mesh,
        ray_starts[tid],
        ray_directions[tid],
        max_dist,
        t, u, v, sign, n, f
    )

    # Check if hit
    if hit_success:
        # Hit! Compute hit position
        ray_hits[tid] = ray_starts[tid] + t * ray_directions[tid]

        # Optionally compute distance
        if return_distance == 1:
            ray_distances[tid] = t
    # If miss, outputs are already initialized to inf
