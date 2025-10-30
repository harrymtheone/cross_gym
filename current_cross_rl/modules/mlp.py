"""Multi-layer perceptron utilities."""

from __future__ import annotations

from typing import Sequence

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
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: nn.Module,
        output_activation: nn.Module | None = None,
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
    layers = nn.Sequential()

    # Input layer
    prev_dim = input_dim

    # Hidden layers
    for hidden in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden))
        layers.append(activation)
        prev_dim = hidden

    # Output layer
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)

    return layers
