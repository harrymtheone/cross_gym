"""T1 robot locomotion task configuration."""

from __future__ import annotations

from cross_core.terrains import TerrainGeneratorCfg
from cross_core.terrains.trimesh_terrains import FlatTerrainCfg
from cross_core.utils import configclass
from cross_env.envs import DirectRLEnvCfg
from cross_gym.assets import GymArticulationCfg
from cross_gym.scene import IsaacGymSceneCfg, PhysXCfg, SimCfg


@configclass
class T1LocomotionCfg:
    """T1 robot locomotion task configuration."""

    num_envs: int = 4096
    decimation: int = 4
    episode_length_s: float = 20.0

    def get_env_cfg(self) -> DirectRLEnvCfg:
        """Get environment configuration."""
        return DirectRLEnvCfg(
            scene=self.get_scene_cfg(),
            decimation=self.decimation,
            episode_length_s=self.episode_length_s,
        )

    def get_scene_cfg(self):
        """Get scene configuration (contains both scene and sim params)."""
        return IsaacGymSceneCfg(
            num_envs=self.num_envs,
            env_spacing=3.0,

            sim=SimCfg(
                dt=0.005,
                substeps=1,
                use_gpu_pipeline=True,
                headless=False,
                physx=PhysXCfg(
                    solver_type=1,
                    num_position_iterations=4,
                    num_velocity_iterations=1,
                ),
            ),

            # Terrain
            terrain=TerrainGeneratorCfg(
                size=(8.0, 8.0),
                horizontal_scale=0.1,
                vertical_scale=0.005,
                border_size=20.0,
                num_rows=10,
                num_cols=10,
                curriculum=False,
                sub_terrains={
                    "flat": FlatTerrainCfg(proportion=1.0),
                },
            ),

            # Robot
            robot=GymArticulationCfg(
                prim_path="/World/envs/env_.*/T1",
                file="cross_assets/robots/T1/T1_serial.urdf",
                init_state=GymArticulationCfg.InitStateCfg(
                    pos=(0.0, 0.0, 0.42),
                    rot=(1.0, 0.0, 0.0, 0.0),
                    joint_pos={".*": 0.0},
                    joint_vel={".*": 0.0},
                ),
                asset_options=GymArticulationCfg.AssetOptionsCfg(
                    fix_base_link=False,
                    default_dof_drive_mode=3,
                    armature=0.01,
                ),
            ),
        )
