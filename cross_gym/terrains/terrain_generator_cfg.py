"""Configuration for terrain generator."""

from __future__ import annotations

from dataclasses import MISSING

from cross_core.utils import configclass
from . import SubTerrainBaseCfg, TerrainGenerator


@configclass
class TerrainGeneratorCfg:
    """Configuration for procedural terrain generation.
    
    The terrain generator creates a grid of sub-terrains, each generated
    from a function. Supports curriculum learning and caching.
    """

    class_type: type = TerrainGenerator

    # ========== Grid Layout ==========
    num_rows: int = 10
    """Number of rows of sub-terrains (difficulty levels if curriculum=True)."""

    num_cols: int = 10
    """Number of columns of sub-terrains (terrain types)."""

    # ========== Sub-Terrains ==========
    sub_terrains: dict[str, SubTerrainBaseCfg] = MISSING
    """Dictionary of sub-terrain configurations.
    
    Keys are terrain names, values are configurations.
    Sub-terrains are sampled based on their proportion values.
    """

    # ========== Curriculum ==========
    curriculum: bool = False
    """Whether to use curriculum learning.
    
    If True:
        - Rows represent difficulty levels (easy → hard)
        - difficulty = row_index / (num_rows - 1)
    If False:
        - Random difficulty for each sub-terrain
    """

    difficulty_range: tuple[float, float] = (0.0, 1.0)
    """Range of difficulty values [min, max]."""

    # ========== Mesh Parameters ==========
    horizontal_scale: float = 0.1
    """Grid resolution along x and y axes (meters per pixel)."""

    slope_threshold: float = 0.75
    """Slope threshold for edge detection.
    
    If None, no edge detection is applied.
    """

    edge_width: float | None = 0.05
    """Width of edge dilation in meters.
    
    If None, no dilation is applied.
    If provided but <= 0, raises ValueError during terrain generation.
    """

    # ========== Border ==========
    border_width: float = 0.0
    """Width of border around terrain (meters)."""

    border_height: float = 0.0
    """Height of border (meters). Negative values go below ground."""

    # ========== Physics ==========
    static_friction: float = 1.0
    """Static friction coefficient."""

    dynamic_friction: float = 1.0
    """Dynamic friction coefficient."""

    restitution: float = 0.0
    """Restitution coefficient (bounciness)."""

    # ========== Caching ==========
    use_cache: bool = False
    """Whether to cache generated terrains."""

    cache_dir: str = "/tmp/cross_gym/terrains"
    """Directory for terrain cache."""

    # ========== Randomization ==========
    seed: int | None = None
    """Random seed for reproducibility. If None, uses random seed."""
