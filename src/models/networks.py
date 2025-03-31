"""
Neural network models for Physics Informed Neural Networks.
"""

import torch
from torch import nn
from typing import List, Dict, Any, Optional, Union


class SimpleNN(nn.Module):
    """
    Simple feed-forward neural network for 1D inputs.
    Used for PINNs with a single input variable.
    """
    
    def __init__(self, layers: List[int], activation: str = "tanh"):
        """
        Initialize a simple feed-forward neural network.
        
        Args:
            layers: List of layer sizes (input_dim, hidden_1, ..., output_dim)
            activation: Activation function to use ('tanh', 'relu', 'sigmoid')
        """
        super(SimpleNN, self).__init__()
        
        # Choose activation function
        if activation.lower() == "tanh":
            act_fn = nn.Tanh()
        elif activation.lower() == "relu":
            act_fn = nn.ReLU()
        elif activation.lower() == "sigmoid":
            act_fn = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Build neural network layers
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:  # No activation after the last layer
                modules.append(act_fn)
        
        self.net = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        return self.net(x)


class SimpleNN2(nn.Module):
    """
    Simple feed-forward neural network for 2D inputs.
    Used for PINNs with two input variables (e.g., space and time).
    """
    
    def __init__(self, layers: List[int], activation: str = "tanh"):
        """
        Initialize a simple feed-forward neural network for 2D inputs.
        
        Args:
            layers: List of layer sizes (input_dim, hidden_1, ..., output_dim)
            activation: Activation function to use ('tanh', 'relu', 'sigmoid')
        """
        super(SimpleNN2, self).__init__()
        
        # Choose activation function
        if activation.lower() == "tanh":
            act_fn = nn.Tanh()
        elif activation.lower() == "relu":
            act_fn = nn.ReLU()
        elif activation.lower() == "sigmoid":
            act_fn = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Build neural network layers
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:  # No activation after the last layer
                modules.append(act_fn)
        
        self.net = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: First input tensor of shape (batch_size, 1)
            t: Second input tensor of shape (batch_size, 1)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Concatenate inputs along feature dimension
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create a neural network model based on configuration.
    
    Args:
        config: Configuration dictionary or Config object
        
    Returns:
        Neural network model
    """
    model_type = config.get("model.type", "simple_nn")
    layers = config.get("model.layers", [1, 16, 32, 16, 1])
    activation = config.get("model.activation", "tanh")
    
    if model_type.lower() == "simple_nn":
        return SimpleNN(layers, activation)
    elif model_type.lower() == "simple_nn2":
        return SimpleNN2(layers, activation)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")