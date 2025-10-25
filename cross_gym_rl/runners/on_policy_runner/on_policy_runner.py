"""On-policy runner for PPO and similar algorithms."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from cross_gym_rl.utils.logger import Logger, EpisodeLogger

if TYPE_CHECKING:
    from cross_gym.envs import VecEnv
    from cross_gym_rl.algorithms import AlgorithmBase
    from . import OnPolicyRunnerCfg


class OnPolicyRunner:
    """Runner for on-policy RL algorithms (PPO, A2C, etc.).
    
    Orchestrates the training loop:
    1. Collect rollouts
    2. Update policy
    3. Log metrics
    4. Save checkpoints
    """

    def __init__(self, cfg: OnPolicyRunnerCfg, env: VecEnv, algorithm: AlgorithmBase):
        """Initialize runner with pre-created environment and algorithm.
        
        Args:
            cfg: Runner configuration
            env: Environment instance (created by TaskRegistry)
            algorithm: Algorithm instance (created by TaskRegistry)
        """
        self.cfg = cfg
        self.env = env
        self.algorithm = algorithm

        print(f"[OnPolicyRunner] Initialized with {env.num_envs} environments")

        # Set up logging
        self._setup_logging()

        # Timing
        self.collection_time = 0
        self.learn_time = 0
        self.tot_time = 0
        self.tot_steps = 0

        # Resume from checkpoint if specified
        self.start_iteration = 0
        if cfg.resume_path is not None:
            self._load_checkpoint(cfg.resume_path, cfg.load_optimizer)

        print("[INFO] Runner initialized successfully!")

    def _setup_logging(self):
        """Set up logging directories and loggers."""
        # Create log directory structure
        log_path = Path(self.cfg.log_dir) / self.cfg.project_name / self.algorithm.__name__ / self.cfg.experiment_name

        self.log_path = log_path
        self.model_dir = log_path / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        if self.cfg.logger_backend:
            self.logger = Logger(str(log_path), backend=self.cfg.logger_backend)
        else:
            self.logger = None

        # Create episode logger
        self.episode_logger = EpisodeLogger(self.env.num_envs)

        print(f"[INFO] Logging to: {log_path}")

    def learn(self):
        """Run the main training loop."""
        print("=" * 80)
        print(f"Starting training: {self.cfg.experiment_name}")
        print(f"  Algorithm: {self.algorithm.__name__}")
        print(f"  Max iterations: {self.cfg.max_iterations}")
        print(f"  Steps per update: {self.cfg.num_steps_per_update}")
        print("=" * 80)

        # Set algorithm to training mode
        self.algorithm.train()

        # Reset environment
        observations, infos = self.env.reset()
        self.episode_logger.reset()

        # Training loop
        for iteration in range(self.start_iteration, self.cfg.max_iterations):
            start_time = time.time()

            # ========== Collect Rollouts ==========
            with torch.inference_mode():
                for step in range(self.cfg.num_steps_per_update):
                    # Get actions from policy
                    actions = self.algorithm.act(observations)

                    # Step environment
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)

                    # Store transition
                    self.algorithm.process_env_step(
                        rewards=rewards,
                        terminated=terminated,
                        truncated=truncated,
                        infos=infos,
                        observations=observations,
                    )

                    # Update episode logger
                    self.episode_logger.step(rewards, terminated, truncated)

                    # Move to next step
                    observations = next_observations
                    self.tot_steps += self.env.num_envs

                # Compute returns after collecting all steps
                self.algorithm.compute_returns(observations)

            self.collection_time = time.time() - start_time

            # ========== Update Policy ==========
            start_time = time.time()
            update_metrics = self.algorithm.update()
            self.learn_time = time.time() - start_time

            self.tot_time = self.collection_time + self.learn_time

            # ========== Logging ==========
            if iteration % self.cfg.log_interval == 0:
                self._log_metrics(iteration, update_metrics, infos)

            # ========== Save Checkpoints ==========
            if iteration % self.cfg.save_interval == 0:
                self._save_checkpoint(iteration)

            # Always save latest
            self._save_checkpoint('latest')

        print("=" * 80)
        print("Training complete!")
        print("=" * 80)

        # Clean up
        if self.logger is not None:
            self.logger.close()
        self.env.close()

    def _log_metrics(self, iteration: int, update_metrics: dict[str, float], infos: dict[str, Any]):
        """Log training metrics.
        
        Args:
            iteration: Current iteration
            update_metrics: Metrics from algorithm update
            infos: Info from environment
        """
        # Get episode statistics
        episode_stats = self.episode_logger.get_statistics()

        # Combine all metrics
        metrics = {}
        metrics.update(update_metrics)
        metrics.update(episode_stats)

        # Add timing metrics
        fps = self.cfg.num_steps_per_update * self.env.num_envs / self.tot_time
        metrics['Time/collection_time'] = self.collection_time
        metrics['Time/learn_time'] = self.learn_time
        metrics['Time/total_time'] = self.tot_time
        metrics['Time/fps'] = fps
        metrics['Time/total_steps'] = self.tot_steps

        # Add environment metrics from info
        if 'log' in infos:
            for key, value in infos['log'].items():
                metrics[f'Env/{key}'] = value

        # Log to backend
        if self.logger is not None:
            self.logger.log_dict(metrics, iteration)

        # Print to console
        if iteration % (self.cfg.log_interval * 10) == 0:
            print(f"\n[Iteration {iteration}]")
            if 'Episode/mean_return' in metrics:
                print(f"  Episode Return: {metrics['Episode/mean_return']:.2f}")
            if 'Loss/surrogate_loss' in metrics:
                print(f"  Policy Loss: {metrics['Loss/surrogate_loss']:.4f}")
            if 'Loss/value_loss' in metrics:
                print(f"  Value Loss: {metrics['Loss/value_loss']:.4f}")
            print(f"  FPS: {fps:.0f}")

    def _save_checkpoint(self, iteration):
        """Save training checkpoint.
        
        Args:
            iteration: Current iteration (or 'latest')
        """
        if isinstance(iteration, int):
            path = self.model_dir / f"model_{iteration}.pt"
        else:
            path = self.model_dir / f"{iteration}.pt"

        checkpoint = {
            'iteration': iteration if isinstance(iteration, int) else self.start_iteration,
            'algorithm_state': self.algorithm.save(str(path)),
        }

        # Save
        torch.save(checkpoint, path)

    def _load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(path, map_location=self.algorithm.device)

        self.start_iteration = checkpoint.get('iteration', 0) + 1
        self.algorithm.load(path, load_optimizer=load_optimizer)

        print(f"[INFO] Resumed from iteration {self.start_iteration}")


__all__ = ["OnPolicyRunner"]
