# Copyright (c) 2025, CrossGym Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mesh registry for managing meshes shared between terrain and sensors."""

from __future__ import annotations

import numpy as np
import torch
import trimesh
import warp as wp


class MeshEntry:
    """Entry for a registered mesh.
    
    Tracks the mesh, its version, and cached Warp mesh.
    """

    def __init__(self, mesh: trimesh.Trimesh, name: str):
        """Initialize mesh entry.
        
        Args:
            mesh: The trimesh object
            name: Unique name for this mesh
        """
        self.name = name
        self.mesh = mesh
        self.version = 0  # Increment when mesh changes
        self.warp_mesh: wp.Mesh | None = None
        self.warp_version = -1  # Track which version is cached

    def update_mesh(self, mesh: trimesh.Trimesh):
        """Update the mesh and increment version.
        
        Args:
            mesh: New trimesh data
        """
        self.mesh = mesh
        self.version += 1
        # Warp mesh will be regenerated on next access


class MeshRegistry:
    """Registry for managing meshes shared between terrain and sensors.
    
    This allows terrain to register trimeshes and sensors to query them
    without tight coupling. Handles conversion to Warp meshes with caching.
    
    Example:
        >>> # In terrain
        >>> mesh_registry.register_mesh("terrain", trimesh_object)
        
        >>> # In sensor
        >>> warp_mesh = mesh_registry.get_warp_mesh("terrain")
        >>> raycast_mesh(rays, mesh=warp_mesh)
    """

    def __init__(self, device: torch.device):
        """Initialize mesh registry.
        
        Args:
            device: PyTorch device for tensor operations
        """
        self.device = device
        self._meshes: dict[str, MeshEntry] = {}

    def register_mesh(self, name: str, mesh: trimesh.Trimesh) -> None:
        """Register a mesh in the registry.
        
        Args:
            name: Unique name for the mesh (e.g., "terrain", "obstacles")
            mesh: The trimesh to register
            
        Raises:
            ValueError: If mesh with this name already exists
        """
        if name in self._meshes:
            raise ValueError(f"Mesh '{name}' already registered. Use update_mesh() instead.")

        self._meshes[name] = MeshEntry(mesh, name)

    def update_mesh(self, name: str, mesh: trimesh.Trimesh) -> None:
        """Update an existing mesh.
        
        Args:
            name: Name of the mesh to update
            mesh: New trimesh data
            
        Raises:
            KeyError: If mesh doesn't exist
        """
        if name not in self._meshes:
            raise KeyError(f"Mesh '{name}' not found. Register it first with register_mesh().")

        self._meshes[name].update_mesh(mesh)

    def get_trimesh(self, name: str) -> trimesh.Trimesh:
        """Get the trimesh for a registered mesh.
        
        Args:
            name: Name of the mesh
            
        Returns:
            The trimesh object
            
        Raises:
            KeyError: If mesh doesn't exist
        """
        if name not in self._meshes:
            raise KeyError(f"Mesh '{name}' not found in registry. Available: {self.list_meshes()}")

        return self._meshes[name].mesh

    def get_warp_mesh(self, name: str) -> wp.Mesh:
        """Get the Warp mesh for a registered mesh.
        
        Converts trimesh to Warp mesh and caches it. If the mesh has been
        updated since last conversion, it will be reconverted.
        
        Args:
            name: Name of the mesh
            
        Returns:
            The Warp mesh object on GPU
            
        Raises:
            KeyError: If mesh doesn't exist
            ImportError: If Warp is not available
        """
        if name not in self._meshes:
            raise KeyError(f"Mesh '{name}' not found in registry. Available: {self.list_meshes()}")

        entry = self._meshes[name]

        # Check if we need to regenerate Warp mesh
        if entry.warp_mesh is None or entry.warp_version != entry.version:
            # Convert trimesh to Warp mesh
            entry.warp_mesh = self._convert_to_warp_mesh(entry.mesh)
            entry.warp_version = entry.version

        return entry.warp_mesh

    def _convert_to_warp_mesh(self, mesh: trimesh.Trimesh) -> wp.Mesh:
        """Convert a trimesh to a Warp mesh.
        
        Args:
            mesh: The trimesh to convert
            
        Returns:
            Warp mesh on GPU
        """
        # Get vertices and faces
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.int32).flatten()

        # Determine Warp device from torch device
        if self.device.type == "cuda":
            warp_device = f"cuda:{self.device.index}" if self.device.index is not None else "cuda:0"
        else:
            warp_device = "cpu"

        # Create Warp mesh
        wp_mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=warp_device),
            indices=wp.array(faces, dtype=wp.int32, device=warp_device)
        )
        return wp_mesh

    def list_meshes(self) -> list[str]:
        """List all registered mesh names.
        
        Returns:
            List of mesh names
        """
        return list(self._meshes.keys())

    def mesh_exists(self, name: str) -> bool:
        """Check if a mesh exists in the registry.
        
        Args:
            name: Name to check
            
        Returns:
            True if mesh exists
        """
        return name in self._meshes

    def get_mesh_version(self, name: str) -> int:
        """Get the current version of a mesh.
        
        Args:
            name: Name of the mesh
            
        Returns:
            Current version number
            
        Raises:
            KeyError: If mesh doesn't exist
        """
        if name not in self._meshes:
            raise KeyError(f"Mesh '{name}' not found in registry. Available: {self.list_meshes()}")
        return self._meshes[name].version

    def clear(self):
        """Clear all registered meshes."""
        self._meshes.clear()


__all__ = ["MeshRegistry", "MeshEntry"]
