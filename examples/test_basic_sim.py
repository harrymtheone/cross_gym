"""Basic test to verify simulation context works."""

try:
    import isaacgym, torch  # noqa
except ImportError:
    import torch

import os
import sys

# Add cross_gym to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cross_gym.sim import SimulationContext
from cross_gym.sim.isaacgym import IsaacGymContext, IsaacGymCfg, PhysxCfg


def test_basic_sim_context():
    """Test basic simulation context creation and stepping."""

    print("=" * 80)
    print("Testing Cross-Gym Basic Simulation Context")
    print("=" * 80)

    # Create configuration
    cfg = IsaacGymCfg(
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dt=0.01,
        headless=True,  # No viewer for this test
        physx=PhysxCfg(
            solver_type=1,
            num_position_iterations=4,
            num_velocity_iterations=1,
        ),
    )

    print(f"\nConfiguration:")
    print(f"  Simulator: {cfg.class_type.__name__}")
    print(f"  Device: {cfg.device}")
    print(f"  Physics dt: {cfg.dt}")
    print(f"  Headless: {cfg.headless}")

    # Create simulation context
    print("\nCreating simulation context...")
    sim = cfg.class_type(cfg)  # Use class_type pattern!

    print(f"✓ Simulation context created successfully!")
    print(f"  Device: {sim.device}")
    print(f"  Physics dt: {sim.physics_dt}")

    # Add ground plane
    print("\nAdding ground plane...")
    sim.add_ground_plane()
    print("✓ Ground plane added!")

    # Reset simulation (prepare for running)
    print("\nResetting simulation...")
    sim.reset()
    print("✓ Simulation reset complete!")

    # Test stepping
    print("\nTesting physics stepping...")
    for i in range(10):
        sim.step(render=False)
        if (i + 1) % 5 == 0:
            print(f"  Stepped {i + 1} times")

    print("✓ Physics stepping works!")

    # Clean up
    print("\nCleaning up...")
    SimulationContext.clear_instance()
    print("✓ Cleanup complete!")

    print("\n" + "=" * 80)
    print("SUCCESS: All basic tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_basic_sim_context()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
