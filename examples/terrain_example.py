"""Example demonstrating terrain generation in Cross-Gym.

Shows how to create procedural terrains with curriculum learning
using the trimesh-based approach (efficient merging with padding).
"""

from __future__ import annotations

try:
    import isaacgym, torch
except ImportError:
    import torch

from cross_gym import terrains
from cross_gym.utils import configclass


# ============================================================================
# Terrain Configuration
# ============================================================================

@configclass
class MyTerrainCfg(terrains.TerrainGeneratorCfg):
    """Terrain configuration with flat terrain type."""

    # Grid layout
    num_rows: int = 10  # Difficulty levels
    num_cols: int = 10  # Terrain types

    # Curriculum
    curriculum: bool = True
    difficulty_range: tuple[float, float] = (0.0, 1.0)

    # Sub-terrains
    from cross_gym.terrains.trimesh_terrains import FlatTerrainCfg

    sub_terrains: dict = {
        "flat": FlatTerrainCfg(
            proportion=1.0,
            size=(8.0, 8.0),
        ),
    }

    # Mesh parameters
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float = 0.75
    edge_width: float = 0.05

    # Border
    border_width: float = 1.0
    border_height: float = 0.0


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Demonstrate terrain generation."""

    print("=" * 80)
    print("Cross-Gym Terrain Generation Example")
    print("=" * 80)

    # Create terrain configuration
    terrain_cfg = MyTerrainCfg()

    print("\nTerrain Configuration:")
    print(f"  Grid: {terrain_cfg.num_rows} rows × {terrain_cfg.num_cols} cols")
    print(f"  Curriculum: {terrain_cfg.curriculum}")
    print(f"  Sub-terrain types: {list(terrain_cfg.sub_terrains.keys())}")
    print(f"  Total terrains: {terrain_cfg.num_rows * terrain_cfg.num_cols}")
    print(f"  Horizontal scale: {terrain_cfg.horizontal_scale}m")

    # Generate terrain
    print("\nGenerating terrain...")
    terrain = terrains.TerrainGenerator(terrain_cfg)

    print(f"\nTerrain Generated:")
    print(f"  Global trimesh:")
    print(f"    Vertices: {terrain.vertices.shape}")
    print(f"    Faces: {terrain.faces.shape}")
    print(f"  Height map shape: {terrain.height_map.shape}")
    print(f"  Edge map shape: {terrain.edge_map.shape}")
    print(f"  Origins shape: {terrain.terrain_origins.shape}")
    print(f"  Global size: {terrain.global_terrain_size}")

    print("\n" + "=" * 80)
    print("Terrain generation successful!")
    print("=" * 80)
    print("\nKey features demonstrated:")
    print("  ✓ Trimesh-based terrain generation")
    print("  ✓ Efficient merging with padding rectangles")
    print("  ✓ Border added with only 4 rectangles")
    print("  ✓ Converted to height map for raycasting")
    print("  ✓ Edge detection for safety")
    print("\nNext steps:")
    print("  1. Add this terrain to your scene configuration")
    print("  2. Use terrain.terrain_origins for spawning robots")
    print("  3. Use terrain.height_map for raycasting/sensors")
    print("=" * 80)


if __name__ == "__main__":
    main()
