"""Terrain generator for creating procedural terrains."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import scipy.ndimage
import trimesh

from . import SubTerrain
from .utils import trimesh_to_height_map_cuda, edge_detection, create_rectangle
from cross_gym.scene import MeshRegistry

if TYPE_CHECKING:
    from . import TerrainGeneratorCfg



class TerrainGenerator:
    """Generates procedural terrains from sub-terrain configurations.
    
    The terrain generator:
    1. Creates a grid of sub-terrains (each generates a trimesh)
    2. Merges sub-terrains into a global trimesh with efficient padding
    3. Adds borders around the terrain
    4. Converts to height map for raycasting/sensors
    5. Computes environment origins for spawning
    """

    def __init__(self, cfg: TerrainGeneratorCfg):
        """Initialize terrain generator.
        
        Args:
            cfg: Terrain generator configuration
        """
        self.cfg = cfg

        # Set random seed
        if cfg.seed is not None:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)

        # Sub-terrain grid
        self._terrain_mat: dict[tuple[int, int], SubTerrain] = {}

        # Generate and merge terrains
        self._compose_sub_terrains()
        self._merge_terrains()
        self._add_border_plane()

        print(f"[TerrainGenerator] Terrain generated:")
        print(f"  Vertices: {self._global_trimesh.vertices.shape[0]}")
        print(f"  Faces: {self._global_trimesh.faces.shape[0]}")

        # Compute height map and origins
        self._compute_env_origins()
        self._compute_height_map()
        self._compute_terrain_types()

        # Register mesh with singleton registry
        mesh_registry = MeshRegistry.instance()
        
        if mesh_registry is None:
            raise RuntimeError("Mesh registry not found. Have you created an InteractiveScene?")
        
        mesh_registry.register_mesh("terrain", self._global_trimesh)
        print(f"[TerrainGenerator] Registered mesh 'terrain' in mesh registry")

    @property
    def num_rows(self) -> int:
        """Number of terrain rows."""
        return self.cfg.num_rows

    @property
    def num_cols(self) -> int:
        """Number of terrain columns."""
        return self.cfg.num_cols

    @property
    def num_terrains(self) -> int:
        """Total number of sub-terrains."""
        return self.num_rows * self.num_cols

    @property
    def mesh(self) -> trimesh.Trimesh:
        """Global terrain mesh."""
        return self._global_trimesh

    @property
    def vertices(self) -> np.ndarray:
        """Global terrain vertices."""
        return np.array(self._global_trimesh.vertices, dtype=np.float32)

    @property
    def faces(self) -> np.ndarray:
        """Global terrain faces."""
        return np.array(self._global_trimesh.faces, dtype=np.uint32)

    @property
    def sub_terrain_meshes(self) -> list[trimesh.Trimesh]:
        """List of sub-terrain meshes."""
        return self._sub_terrain_meshes

    @property
    def height_map(self) -> np.ndarray:
        """Height map for raycasting."""
        return self._height_map

    @property
    def edge_map(self) -> np.ndarray:
        """Edge detection map."""
        return self._edge_map

    @property
    def terrain_origins(self) -> np.ndarray:
        """Environment spawn origins (num_rows, num_cols, 3)."""
        return self._origins

    @property
    def terrain_type(self) -> np.ndarray:
        """Terrain type indices (num_rows, num_cols)."""
        return self._terrain_type

    @property
    def global_terrain_size(self) -> tuple[float, float]:
        """Total terrain size (width, length)."""
        return self._global_terrain_size

    def _compose_sub_terrains(self):
        """Generate individual sub-terrains."""
        # Get proportions for weighted sampling
        proportions = np.array([cfg.proportion for cfg in self.cfg.sub_terrains.values()])
        proportions = np.cumsum(proportions / np.sum(proportions))

        sub_terrain_configs = list(self.cfg.sub_terrains.values())

        for col in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                # Sample terrain type based on column
                choice = col / self.cfg.num_cols + 0.001
                terrain_idx = np.searchsorted(proportions, choice)
                sub_terrain_cfg = sub_terrain_configs[terrain_idx]

                # Compute difficulty
                if self.cfg.curriculum:
                    difficulty = row / max(1, self.cfg.num_rows - 1)
                    # Apply difficulty range
                    difficulty = self.cfg.difficulty_range[0] + difficulty * (
                            self.cfg.difficulty_range[1] - self.cfg.difficulty_range[0])
                else:
                    difficulty = np.random.uniform(*self.cfg.difficulty_range)

                # Generate sub-terrain using class_type
                sub_terrain = sub_terrain_cfg.class_type(sub_terrain_cfg, difficulty)
                self._terrain_mat[(row, col)] = sub_terrain

    def _merge_terrains(self):
        """Merge all sub-terrains into a global trimesh with efficient padding."""
        # Find max size in x direction for each row
        max_size_x = 0
        for sub_terrain in self._terrain_mat.values():
            max_size_x = max(max_size_x, sub_terrain.cfg.size[0])
        max_size_x = max_size_x * self.cfg.num_rows

        # Merge sub-terrains
        self._sub_terrain_meshes = []
        cur_y = 0.0

        for col in range(self.cfg.num_cols):
            cur_x = 0.0

            for row in range(self.cfg.num_rows):
                sub_terrain = self._terrain_mat[(row, col)]

                # Apply transformation to position the mesh
                transformation = trimesh.transformations.translation_matrix([cur_x, cur_y, 0])
                sub_terrain.mesh.apply_transform(transformation)

                # Update frame origin (bottom-left corner of sub-terrain)
                sub_terrain.frame_origin = (cur_x, cur_y, 0)

                # Update actual origin (could be different from frame_origin)
                sub_terrain.origin = (
                    cur_x + sub_terrain.origin[0],
                    cur_y + sub_terrain.origin[1],
                    sub_terrain.origin[2],
                )

                # Add to mesh list
                self._sub_terrain_meshes.append(sub_terrain.mesh)

                # Move to next row position
                cur_x += sub_terrain.cfg.size[0]

            # Efficiently pad remaining space in x direction with a single rectangle
            pad_x = max_size_x - cur_x
            if pad_x > 0:
                self._sub_terrain_meshes.append(
                    create_rectangle(
                        size=(pad_x, sub_terrain.cfg.size[1]),  # noqa
                        height=0.0,
                        transform=trimesh.transformations.translation_matrix([cur_x, cur_y, 0]),
                        up_left_center=True
                    )
                )

            # Move to next column position
            cur_y += sub_terrain.cfg.size[1]

        # Merge all sub-terrains into a single trimesh
        self._global_trimesh = trimesh.util.concatenate(self._sub_terrain_meshes)  # noqa

        # Store global terrain size
        self._global_terrain_size = (max_size_x, cur_y)

    def _add_border_plane(self):
        """Add a border plane around the terrain."""
        if self.cfg.border_width <= 0:
            return

        terrain_width, terrain_length = self._global_terrain_size
        border_width = self.cfg.border_width
        border_height = self.cfg.border_height

        # Create border rectangles (efficient - only 4 rectangles!)
        borders = [
            # Bottom border
            create_rectangle(
                size=(terrain_width + 2 * border_width, border_width),
                height=border_height,
                transform=trimesh.transformations.translation_matrix(
                    [-border_width, -border_width, 0]
                ),
                up_left_center=True
            ),
            # Top border
            create_rectangle(
                size=(terrain_width + 2 * border_width, border_width),
                height=border_height,
                transform=trimesh.transformations.translation_matrix(
                    [-border_width, terrain_length, 0]
                ),
                up_left_center=True
            ),
            # Left border
            create_rectangle(
                size=(border_width, terrain_length),
                height=border_height,
                transform=trimesh.transformations.translation_matrix(
                    [-border_width, 0, 0]
                ),
                up_left_center=True
            ),
            # Right border
            create_rectangle(
                size=(border_width, terrain_length),
                height=border_height,
                transform=trimesh.transformations.translation_matrix(
                    [terrain_width, 0, 0]
                ),
                up_left_center=True
            )
        ]

        # Merge borders with terrain
        self._global_trimesh = trimesh.util.concatenate([self._global_trimesh] + borders)  # noqa

    def _compute_env_origins(self):
        """Compute spawn origins for each environment."""
        self._origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3), dtype=float)

        for row in range(self.cfg.num_rows):
            for col in range(self.cfg.num_cols):
                self._origins[row, col] = self._terrain_mat[(row, col)].origin

    def _compute_height_map(self):
        """Convert global trimesh to height map for raycasting."""
        # Convert trimesh to height map using CUDA raycasting
        self._height_map = trimesh_to_height_map_cuda(
            self._global_trimesh,
            horizontal_scale=self.cfg.horizontal_scale,
        )

        # Compute edge map if slope threshold is provided
        self._edge_map = edge_detection(
            self._height_map,
            horizontal_scale=self.cfg.horizontal_scale,
            slope_threshold=self.cfg.slope_threshold,
        )

        # Dilate edge map to add safety margin
        if self.cfg.edge_width is not None:
            if self.cfg.edge_width <= 0:
                raise ValueError(f"edge_width must be > 0, got {self.cfg.edge_width}")

            half_edge_width = int(self.cfg.edge_width / self.cfg.horizontal_scale)
            structure = np.ones((half_edge_width * 2 + 1, half_edge_width * 2 + 1))
            self._edge_map = scipy.ndimage.binary_dilation(self._edge_map, structure=structure)

    def _compute_terrain_types(self):
        """Store terrain type for each sub-terrain."""
        self._terrain_type = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=int)

        # For now, just use row index as type
        # TODO: Proper terrain type tracking
        for row in range(self.cfg.num_rows):
            for col in range(self.cfg.num_cols):
                self._terrain_type[row, col] = row
