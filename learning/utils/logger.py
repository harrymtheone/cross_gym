"""Logger for RL training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch


class Logger:
    """Logger for RL training metrics.
    
    Supports Tensorboard logging.
    """
    
    def __init__(
        self,
        log_dir: str,
        backend: str = "tensorboard",
    ):
        """Initialize logger.
        
        Args:
            log_dir: Directory for logs
            backend: Logging backend ('tensorboard' or None)
        """
        self.log_dir = log_dir
        self.backend = backend
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize backend
        if backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
        else:
            self.writer = None
    
    def log_scalar(self, key: str, value: float, step: int):
        """Log a scalar value.
        
        Args:
            key: Metric name
            value: Metric value
            step: Training step/iteration
        """
        if self.writer is not None:
            self.writer.add_scalar(key, value, step)
    
    def log_dict(self, metrics: Dict[str, float], step: int):
        """Log a dictionary of metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step/iteration
        """
        for key, value in metrics.items():
            self.log_scalar(key, value, step)
    
    def close(self):
        """Close logger and flush remaining data."""
        if self.writer is not None:
            self.writer.close()


class EpisodeLogger:
    """Logger for episode statistics.
    
    Tracks episode lengths and returns for each environment.
    """
    
    def __init__(self, num_envs: int):
        """Initialize episode logger.
        
        Args:
            num_envs: Number of parallel environments
        """
        self.num_envs = num_envs
        self.reset()
    
    def reset(self):
        """Reset all episode statistics."""
        self.episode_returns = torch.zeros(self.num_envs)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int)
        
        # Completed episodes tracking
        self.completed_episodes_returns = []
        self.completed_episodes_lengths = []
    
    def step(self, rewards: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor):
        """Update episode statistics after environment step.
        
        Args:
            rewards: Rewards (num_envs,)
            terminated: Termination flags (num_envs,)
            truncated: Truncation flags (num_envs,)
        """
        # Accumulate rewards
        self.episode_returns += rewards.cpu()
        self.episode_lengths += 1
        
        # Check for completed episodes
        dones = terminated | truncated
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        
        if len(done_ids) > 0:
            # Store completed episode stats
            for idx in done_ids:
                idx = idx.item()
                self.completed_episodes_returns.append(self.episode_returns[idx].item())
                self.completed_episodes_lengths.append(self.episode_lengths[idx].item())
                
                # Reset for this environment
                self.episode_returns[idx] = 0
                self.episode_lengths[idx] = 0
    
    def get_statistics(self) -> Dict[str, float]:
        """Get episode statistics.
        
        Returns:
            Dictionary of statistics (mean return, mean length, etc.)
        """
        if len(self.completed_episodes_returns) == 0:
            return {}
        
        stats = {
            'Episode/mean_return': sum(self.completed_episodes_returns) / len(self.completed_episodes_returns),
            'Episode/mean_length': sum(self.completed_episodes_lengths) / len(self.completed_episodes_lengths),
            'Episode/max_return': max(self.completed_episodes_returns),
            'Episode/min_return': min(self.completed_episodes_returns),
        }
        
        # Clear completed episodes
        self.completed_episodes_returns = []
        self.completed_episodes_lengths = []
        
        return stats


__all__ = ["Logger", "EpisodeLogger"]

