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
        mesh: wp.Mesh,
        max_dist: float = 1e6,
        return_hit_points: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Perform GPU raycasting using Warp.
    
    This is a wrapper around Warp's mesh raycasting that handles PyTorch tensors.
    Always returns distances. Optionally returns hit positions.
    
    Args:
        ray_starts: Ray origins. Shape: (num_envs, num_rays, 3) or (N, 3)
        ray_directions: Ray directions (should be normalized). Shape: (num_envs, num_rays, 3) or (N, 3)
        mesh: Warp mesh to raycast against
        max_dist: Maximum ray distance
        return_hit_points: Whether to compute and return hit positions
        
    Returns:
        ray_hits: Hit positions. Shape same as input. Contains inf for misses.
                  None if return_hit_points=False.
        distances: Distances to hits. Shape: (..., num_rays). Contains inf for misses.
                   Always returned.
                 
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
    ray_distances = torch.full((num_rays,), float('inf'), device=mesh_device, dtype=torch.float32)
    
    if return_hit_points:
        # Use kernel that computes both distances and hit points
        ray_hits = torch.full((num_rays, 3), float('inf'), device=mesh_device, dtype=torch.float32)
        
        wp.launch(
            kernel=_raycast_kernel_with_hit_points,
            dim=num_rays,
            inputs=[
                mesh.id,
                ray_starts_flat,
                ray_directions_flat,
                ray_hits,
                ray_distances,
                float(max_dist),
            ],
            device=mesh.device,
        )
    else:
        # Use optimized kernel that only computes distances
        ray_hits = None
        
        wp.launch(
            kernel=_raycast_kernel_distance_only,
            dim=num_rays,
            inputs=[
                mesh.id,
                ray_starts_flat,
                ray_directions_flat,
                ray_distances,
                float(max_dist),
            ],
            device=mesh.device,
        )

    # Synchronize to ensure kernel completion
    wp.synchronize()

    # Reshape distances back to original shape (always returned)
    ray_distances = ray_distances.to(device).reshape(original_shape[:-1])

    # Reshape hit points if computed
    if return_hit_points:
        ray_hits = ray_hits.to(device).reshape(original_shape)

    return ray_hits, ray_distances


@wp.kernel(enable_backward=False)
def _raycast_kernel_distance_only(
        mesh_id: wp.uint64,
        ray_starts: wp.array1d(dtype=wp.vec3),
        ray_directions: wp.array1d(dtype=wp.vec3),
        ray_distances: wp.array1d(dtype=wp.float32),
        max_dist: float,
):
    """Optimized Warp kernel for raycasting - distance only.
    
    This kernel only computes distances, avoiding unnecessary computation
    of hit positions when they're not needed.
    
    Args:
        mesh_id: Warp mesh ID
        ray_starts: Ray starting positions
        ray_directions: Ray directions (should be normalized)
        ray_distances: Output distances (inf for misses)
        max_dist: Maximum ray distance
    """
    tid = wp.tid()

    # Perform raycast query
    query = wp.mesh_query_ray(
        mesh_id,
        ray_starts[tid],
        ray_directions[tid],
        max_dist
    )

    # Store distance if hit (otherwise remains inf)
    if query.result:
        ray_distances[tid] = query.t


@wp.kernel(enable_backward=False)
def _raycast_kernel_with_hit_points(
        mesh_id: wp.uint64,
        ray_starts: wp.array1d(dtype=wp.vec3),
        ray_directions: wp.array1d(dtype=wp.vec3),
        ray_hits: wp.array1d(dtype=wp.vec3),
        ray_distances: wp.array1d(dtype=wp.float32),
        max_dist: float,
):
    """Warp kernel for raycasting - distances and hit points.
    
    This kernel computes both distances and hit point positions.
    
    Args:
        mesh_id: Warp mesh ID
        ray_starts: Ray starting positions
        ray_directions: Ray directions (should be normalized)
        ray_hits: Output hit positions (inf for misses)
        ray_distances: Output distances (inf for misses)
        max_dist: Maximum ray distance
    """
    tid = wp.tid()

    # Perform raycast query
    query = wp.mesh_query_ray(
        mesh_id,
        ray_starts[tid],
        ray_directions[tid],
        max_dist
    )

    # Store results if hit (otherwise remains inf)
    if query.result:
        ray_distances[tid] = query.t
        ray_hits[tid] = ray_starts[tid] + query.t * ray_directions[tid]
