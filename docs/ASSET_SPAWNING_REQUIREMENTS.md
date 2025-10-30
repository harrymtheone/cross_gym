# Asset Spawning Requirements

## Current Issue

**Error:** `AttributeError: 'NoneType' object has no attribute 'view'`

**Root Cause:** The simulation backend doesn't spawn/load URDF assets before `ArticulationView` tries to access them.

**Flow:**
```
InteractiveScene.__init__()
  └─> _parse_cfg()  # Creates Articulation objects
  └─> _initialize_assets()
      └─> articulation.initialize()
          └─> backend = sim.create_articulation_view()  # ❌ No actor exists yet!
              └─> IsaacGymArticulationView.initialize_tensors()
                  └─> CRASH: gymtorch.wrap_tensor(None)
```

---

## What Needs to be Implemented

### **1. Asset Spawning in SimulationContext**

The simulation context needs to load and spawn assets **before** `InteractiveScene` creates views.

**Location:** `cross_gym/sim/isaacgym/isaacgym_context.py`

**Required Methods:**

```python
class IsaacGymContext(SimulationContext):
    
    def spawn_urdf(
        self,
        urdf_path: str,
        prim_path: str,
        cfg: ArticulationCfg,
        env_ids: list[int]
    ) -> int:
        """Load URDF and spawn actors in environments.
        
        Steps:
        1. Load URDF as gym asset
        2. Configure asset options (fix_base_link, collision, etc.)
        3. Create actor in each environment
        4. Return actor handle
        
        Args:
            urdf_path: Path to URDF file
            prim_path: Pattern like "/World/envs/env_.*/Robot"
            cfg: Articulation configuration
            env_ids: Environment indices to spawn in
            
        Returns:
            Actor handle
        """
        # Load URDF
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = cfg.asset_options.fix_base_link
        asset_options.collapse_fixed_joints = cfg.asset_options.collapse_fixed_joints
        # ... more options
        
        asset = self.gym.load_asset(
            self.sim,
            os.path.dirname(urdf_path),
            os.path.basename(urdf_path),
            asset_options
        )
        
        # Spawn in each environment
        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.gym.create_actor(
                env_handle,
                asset,
                initial_pose,
                prim_path,  # Actor name
                env_id,
                collision_group,
                collision_filter
            )
        
        return actor_handle
```

**Required Attributes:**
```python
self.envs = []  # List of environment handles
self.actors = {}  # Dict mapping prim_path to actor handles
```

---

### **2. Environment Creation**

Environments must be created **before** spawning assets.

**Location:** `cross_gym/sim/isaacgym/isaacgym_context.py`

**Required Method:**

```python
def create_envs(self, num_envs: int, spacing: float = 2.0):
    """Create Isaac Gym environments.
    
    Args:
        num_envs: Number of parallel environments
        spacing: Distance between environments
    """
    # Compute environment grid
    num_per_row = int(np.sqrt(num_envs))
    
    # Create environments in grid
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    
    self.envs = []
    for i in range(num_envs):
        env = self.gym.create_env(self.sim, lower, upper, num_per_row)
        self.envs.append(env)
```

**When to Call:** After `sim.create()` but before spawning assets.

---

### **3. Modified InteractiveScene Initialization Flow**

**Current Flow (Broken):**
```python
# InteractiveScene.__init__()
self._parse_cfg()  # Creates Articulation(cfg)
self._clone_environments()
self._initialize_assets()  # Calls articulation.initialize()
                            # ❌ Tries to create view before actors exist
```

**Required Flow:**
```python
# InteractiveScene.__init__()
self._parse_cfg()  # Creates Articulation(cfg) - NO initialization yet

# NEW: Spawn all assets into simulation
self._spawn_assets()  # Loads URDFs and creates actors in sim

self._clone_environments()  # (May not be needed if spawn handles it)

# NOW create views (actors exist)
self._initialize_assets()  # Creates ArticulationView successfully ✅
```

---

### **4. New Method: `InteractiveScene._spawn_assets()`**

**Location:** `cross_gym/scene/interactive_scene.py`

**Implementation:**

```python
def _spawn_assets(self):
    """Spawn all assets into simulation before creating views.
    
    Must be called after _parse_cfg() and before _initialize_assets().
    """
    # Get number of environments (may need to create them first)
    if not hasattr(self.sim, 'envs') or len(self.sim.envs) == 0:
        self.sim.create_envs(self.num_envs, spacing=2.0)
    
    # Spawn terrain (if configured)
    if self.terrain is not None:
        self.sim.spawn_terrain(self.terrain)
    
    # Spawn articulations
    for name, articulation in self.articulations.items():
        if articulation.cfg.file is not None:
            self.sim.spawn_urdf(
                urdf_path=articulation.cfg.file,
                prim_path=articulation.cfg.prim_path,
                cfg=articulation.cfg,
                env_ids=list(range(self.num_envs))
            )
    
    # Prepare simulation (required before creating views)
    self.sim.prepare_sim()  # Allocates buffers
```

---

### **5. Simulation Preparation**

After spawning all assets, Isaac Gym needs preparation.

**Location:** `cross_gym/sim/isaacgym/isaacgym_context.py`

```python
def prepare_sim(self):
    """Prepare simulation after spawning all assets.
    
    This allocates physics buffers and prepares tensor API.
    Must be called before creating any views.
    """
    # Prepare simulation
    self.gym.prepare_sim(self.sim)
    
    # Acquire tensor handles
    self.gym.refresh_actor_root_state_tensor(self.sim)
    self.gym.refresh_dof_state_tensor(self.sim)
    self.gym.refresh_net_contact_force_tensor(self.sim)
```

---

### **6. Terrain Spawning**

Terrain needs to be added to simulation as a static mesh.

**Location:** `cross_gym/sim/isaacgym/isaacgym_context.py`

```python
def spawn_terrain(self, terrain: TerrainGenerator):
    """Spawn terrain as static trimesh in all environments.
    
    Args:
        terrain: Terrain generator with trimesh
    """
    # Convert trimesh to Isaac Gym triangle mesh
    vertices = terrain.mesh.vertices
    triangles = terrain.mesh.faces
    
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = vertices.shape[0]
    tm_params.nb_triangles = triangles.shape[0]
    tm_params.transform.p = gymapi.Vec3(0, 0, 0)
    
    # Add to each environment
    for env in self.envs:
        self.gym.add_triangle_mesh(
            self.sim,
            vertices.flatten(),
            triangles.flatten(),
            tm_params
        )
```

---

## Implementation Checklist

### **IsaacGymContext (`isaacgym_context.py`)**

- [ ] `create_envs(num_envs, spacing)` - Create environment grid
- [ ] `spawn_urdf(urdf_path, prim_path, cfg, env_ids)` - Load and spawn URDF
- [ ] `spawn_terrain(terrain)` - Add terrain mesh
- [ ] `prepare_sim()` - Allocate buffers after spawning
- [ ] `self.envs: list` - Store environment handles
- [ ] `self.actors: dict` - Store actor handles by name

### **InteractiveScene (`interactive_scene.py`)**

- [ ] `_spawn_assets()` - New method to spawn before initialization
- [ ] Update `__init__` flow:
  ```python
  self._parse_cfg()      # Create objects
  self._spawn_assets()   # Spawn into sim ← NEW
  self._initialize_assets()  # Create views
  ```

### **Optional: Improved Architecture**

Consider moving spawning into `Articulation.spawn()`:

```python
# In Articulation class
def spawn(self, sim: SimulationContext, num_envs: int):
    """Spawn this articulation into simulation.
    
    Args:
        sim: Simulation context
        num_envs: Number of environments
    """
    sim.spawn_urdf(
        urdf_path=self.cfg.file,
        prim_path=self.cfg.prim_path,
        cfg=self.cfg,
        env_ids=list(range(num_envs))
    )
```

Then in `InteractiveScene`:
```python
def _spawn_assets(self):
    for articulation in self.articulations.values():
        articulation.spawn(self.sim, self.num_envs)
```

---

## Reference Implementation

Look at Isaac Gym examples for:
- `create_sim()` → `create_envs()` → `load_asset()` → `create_actor()` → `prepare_sim()`
- Asset loading options and configuration
- Actor placement and indexing

---

## Testing Strategy

1. **Start simple:** Spawn single T1 robot in 1 environment
2. **Verify views work:** Check that tensor wrapping succeeds
3. **Scale up:** Multiple environments with cloning
4. **Add terrain:** After basic spawning works
5. **Full integration:** Complete scene with terrain + robots

---

## Next Steps

1. Implement `create_envs()` first
2. Then `spawn_urdf()` for single robot
3. Test with 1 environment
4. Scale to multiple environments
5. Add terrain spawning
6. Complete the pipeline

This is the missing link between configuration and simulation! 🔧

