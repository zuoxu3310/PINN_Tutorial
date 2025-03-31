"""
Automatic differentiation utilities for Physics Informed Neural Networks.
"""

import torch
from torch import nn
from typing import Optional


def gradient(output: torch.Tensor, input_var: torch.Tensor, 
             order: int = 1, create_graph: bool = True) -> torch.Tensor:
    """
    Compute gradient of output with respect to input_var using PyTorch's autograd.
    
    Args:
        output: Output tensor from the network
        input_var: Input tensor with requires_grad=True
        order: Order of the derivative
        create_graph: Whether to create a computational graph (needed for higher-order derivatives)
        
    Returns:
        Gradient tensor of same shape as input_var
    """
    grad_output = output
    for _ in range(order):
        grad_output = torch.autograd.grad(
            outputs=grad_output,
            inputs=input_var,
            grad_outputs=torch.ones_like(input_var),
            create_graph=create_graph,
            retain_graph=True,
        )[0]
    
    return grad_output


def df_dt(model: nn.Module, x: torch.Tensor, t: torch.Tensor, 
          order: int = 1, create_graph: bool = True) -> torch.Tensor:
    """
    Compute the derivative of the model output with respect to t.
    
    Args:
        model: Neural network model with forward(x, t) method
        x: Spatial input tensor
        t: Time input tensor with requires_grad=True
        order: Order of the derivative
        create_graph: Whether to create a computational graph
        
    Returns:
        Derivative tensor of shape matching t
    """
    output = model(x, t)
    return gradient(output, t, order, create_graph)


def df_dx(model: nn.Module, x: torch.Tensor, t: torch.Tensor, 
          order: int = 1, create_graph: bool = True) -> torch.Tensor:
    """
    Compute the derivative of the model output with respect to x.
    
    Args:
        model: Neural network model with forward(x, t) method
        x: Spatial input tensor with requires_grad=True
        t: Time input tensor
        order: Order of the derivative
        create_graph: Whether to create a computational graph
        
    Returns:
        Derivative tensor of shape matching x
    """
    output = model(x, t)
    return gradient(output, x, order, create_graph)


def df_single(model: nn.Module, x: torch.Tensor, 
              order: int = 1, create_graph: bool = True) -> torch.Tensor:
    """
    Compute the derivative of the model output with respect to x for 1D models.
    
    Args:
        model: Neural network model with forward(x) method
        x: Input tensor with requires_grad=True
        order: Order of the derivative
        create_graph: Whether to create a computational graph
        
    Returns:
        Derivative tensor of shape matching x
    """
    output = model(x)
    return gradient(output, x, order, create_graph)