"""Utility functions for terrain generation."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import trimesh


def edge_detection(
        height_map: np.ndarray,
        horizontal_scale: float,
        slope_threshold: float,
) -> np.ndarray:
    """Detect edges in height map based on slope threshold.
    
    A point is marked as edge if its height change is greater than 
    slope_threshold in any direction.
    
    Args:
        height_map: 2D array of heights
        horizontal_scale: Distance between grid points in meters
        slope_threshold: Slope threshold for edge detection
        
    Returns:
        2D boolean array where True indicates an edge point
    """
    num_rows, num_cols = height_map.shape

    # Scale threshold by horizontal distance
    height_threshold = slope_threshold * horizontal_scale

    # Initialize edge map
    edges = np.zeros((num_rows, num_cols), dtype=bool)

    # Check x direction
    x_diff = np.abs(height_map[1:, :] - height_map[:-1, :]) > height_threshold
    edges[:-1, :] |= x_diff
    edges[1:, :] |= x_diff

    # Check y direction
    y_diff = np.abs(height_map[:, 1:] - height_map[:, :-1]) > height_threshold
    edges[:, :-1] |= y_diff
    edges[:, 1:] |= y_diff

    return edges


def trimesh_to_height_map_cuda(
        mesh: trimesh.Trimesh,
        horizontal_scale: float = 0.1,
        ray_start_height: float = 1000.0
) -> np.ndarray:
    """Generate height map from trimesh using raycasting.
    
    Uses Open3D for GPU-accelerated raycasting.
    
    Args:
        mesh: Trimesh object
        horizontal_scale: Grid resolution in meters
        ray_start_height: Height from which to cast rays downward
        
    Returns:
        2D array of heights
    """

    # Compute grid bounds
    min_bound = mesh.vertices.min(axis=0)
    max_bound = mesh.vertices.max(axis=0)

    x_vals = np.arange(min_bound[0], max_bound[0] + horizontal_scale, horizontal_scale)
    y_vals = np.arange(min_bound[1], max_bound[1] + horizontal_scale, horizontal_scale)
    nx, ny = len(x_vals), len(y_vals)

    # Create ray origins and directions
    xx, yy = np.meshgrid(x_vals, y_vals, indexing="xy")
    ray_origins = np.stack(
        [xx.ravel(), yy.ravel(), np.full(xx.size, ray_start_height)],
        axis=1
    ).astype(np.float32)
    ray_directions = np.tile(np.array([0, 0, -1.0], dtype=np.float32), (ray_origins.shape[0], 1))

    # Convert Trimesh to Open3D
    verts = o3d.core.Tensor(mesh.vertices.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    tris = o3d.core.Tensor(mesh.faces.astype(np.int32), dtype=o3d.core.Dtype.Int32)
    o3d_mesh = o3d.t.geometry.TriangleMesh(verts, tris)

    # Build raycasting scene
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d_mesh)

    # Prepare rays
    rays = np.concatenate([ray_origins, ray_directions], axis=1)
    rays_tensor = o3d.core.Tensor(rays)

    # Cast rays
    result = scene.cast_rays(rays_tensor)
    hit_t = result['t_hit'].numpy()

    # Check which rays hit (finite t values)
    valid = np.isfinite(hit_t)

    # Compute heights - use default for misses
    default_height = -1000.0
    heights = np.full(hit_t.shape, default_height, dtype=np.float32)
    heights[valid] = ray_start_height - hit_t[valid]

    # Reshape and transpose
    heights = heights.reshape(ny, nx)
    heights = heights.T

    return heights


def _trimesh_to_height_map_cpu(
        mesh: trimesh.Trimesh,
        horizontal_scale: float,
        ray_start_height: float
) -> np.ndarray:
    """CPU fallback for trimesh to height map conversion."""
    bounds = mesh.bounds
    min_x, min_y = bounds[0, 0], bounds[0, 1]
    max_x, max_y = bounds[1, 0], bounds[1, 1]

    x_samples = int(np.ceil((max_x - min_x) / horizontal_scale)) + 1
    y_samples = int(np.ceil((max_y - min_y) / horizontal_scale)) + 1

    x_coords = np.linspace(min_x, max_x, x_samples)
    y_coords = np.linspace(min_y, max_y, y_samples)

    X, Y = np.meshgrid(x_coords, y_coords)
    ray_origins = np.column_stack([
        X.flatten(),
        Y.flatten(),
        np.full(X.size, ray_start_height)
    ])

    ray_directions = np.tile([0, 0, -1], (ray_origins.shape[0], 1))

    # Perform ray casting
    locations, ray_indices, _ = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False
    )

    # Initialize height map
    height_map = np.full((x_samples, y_samples), 0.0)

    # Fill in heights
    if len(locations) > 0:
        for i, ray_idx in enumerate(ray_indices):
            y_idx = ray_idx // x_samples
            x_idx = ray_idx % x_samples
            height_map[x_idx, y_idx] = locations[i, 2]

    return height_map


def create_rectangle(
        size: tuple[float, float] = (1.0, 1.0),
        height: float = 0.0,
        transform: np.ndarray = None,
        up_left_center: bool = True
) -> trimesh.Trimesh:
    """Create a rectangular trimesh.
    
    Args:
        size: (width, length) in meters
        height: Z-height of the rectangle
        transform: Optional 4x4 transformation matrix
        up_left_center: If True, rectangle origin is at upper-left corner.
                       If False, rectangle is centered at origin.
        
    Returns:
        Trimesh rectangle
    """
    width, length = size
    half_width = width / 2
    half_length = length / 2

    # Create vertices centered at origin
    vertices = np.array([
        [-half_width, -half_length, height],  # bottom-left
        [half_width, -half_length, height],  # bottom-right
        [half_width, half_length, height],  # top-right
        [-half_width, half_length, height],  # top-left
    ], dtype=np.float32)

    # Create faces (two triangles)
    faces = np.array([
        [0, 1, 2],  # bottom-left, bottom-right, top-right
        [0, 2, 3],  # bottom-left, top-right, top-left
    ], dtype=np.uint32)

    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # If up_left_center, move so (0,0) is at upper-left corner
    if up_left_center:
        mesh.apply_transform(
            trimesh.transformations.translation_matrix([half_width, half_length, 0])
        )

    # Apply additional transformation if provided
    if transform is not None:
        mesh.apply_transform(transform)

    return mesh
