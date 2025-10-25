"""Multi-layer perceptron utilities."""

from __future__ import annotations

from typing import List

import torch.nn as nn


def get_activation(activation: str) -> nn.Module:
    """Get activation function by name.
    
    Args:
        activation: Activation function name ('elu', 'relu', 'tanh', etc.)
        
    Returns:
        Activation module
    """
    activation = activation.lower()
    
    if activation == 'elu':
        return nn.ELU()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'selu':
        return nn.SELU()
    else:
        raise ValueError(f"Unknown activation: {activation}")


def make_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = 'elu',
    output_activation: str = None,
) -> nn.Sequential:
    """Create a multi-layer perceptron.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        activation: Activation function name for hidden layers
        output_activation: Activation for output layer (None = no activation)
        
    Returns:
        Sequential MLP network
    """
    layers = []
    
    # Input layer
    prev_dim = input_dim
    
    # Hidden layers
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(get_activation(activation))
        prev_dim = hidden_dim
    
    # Output layer
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        layers.append(get_activation(output_activation))
    
    return nn.Sequential(*layers)


__all__ = ["make_mlp", "get_activation"]

