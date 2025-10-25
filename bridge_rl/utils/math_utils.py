"""Math utilities for RL algorithms."""

from __future__ import annotations

import torch
from typing import Union


def masked_mean(tensor: torch.Tensor, mask: Union[torch.Tensor, slice]) -> torch.Tensor:
    """Compute mean of tensor with optional masking.
    
    Args:
        tensor: Input tensor
        mask: Boolean mask or slice
        
    Returns:
        Masked mean
    """
    if isinstance(mask, slice):
        return tensor.mean()
    else:
        return (tensor * mask).sum() / mask.sum().clamp(min=1)


def masked_sum(tensor: torch.Tensor, mask: Union[torch.Tensor, slice]) -> torch.Tensor:
    """Compute sum of tensor with optional masking.
    
    Args:
        tensor: Input tensor
        mask: Boolean mask or slice
        
    Returns:
        Masked sum
    """
    if isinstance(mask, slice):
        return tensor.sum()
    else:
        return (tensor * mask).sum()


def masked_MSE(pred: torch.Tensor, target: torch.Tensor, mask: Union[torch.Tensor, slice]) -> torch.Tensor:
    """Compute masked mean squared error.
    
    Args:
        pred: Predictions
        target: Targets
        mask: Boolean mask or slice
        
    Returns:
        Masked MSE
    """
    squared_error = (pred - target).pow(2)
    return masked_mean(squared_error, mask)


def masked_L1(pred: torch.Tensor, target: torch.Tensor, mask: Union[torch.Tensor, slice]) -> torch.Tensor:
    """Compute masked L1 loss.
    
    Args:
        pred: Predictions
        target: Targets
        mask: Boolean mask or slice
        
    Returns:
        Masked L1 loss
    """
    abs_error = torch.abs(pred - target)
    return masked_mean(abs_error, mask)


__all__ = [
    "masked_mean",
    "masked_sum",
    "masked_MSE",
    "masked_L1",
]

