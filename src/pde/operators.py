"""
PDE operators and boundary/initial conditions for Physics Informed Neural Networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Callable, Optional, Tuple

from src.pde.derivatives import df_dx, df_dt, df_single, gradient


class LogisticOperator:
    """
    Operator for the logistic equation: df/dt = R * f * (1 - f)
    """
    
    def __init__(self, R: float = 1.0, ft0: float = 1.0):
        """
        Initialize the logistic equation operator.
        
        Args:
            R: Maximum growth rate
            ft0: Initial value f(t=0)
        """
        self.R = R
        self.ft0 = ft0
    
    def pde_residual(self, model: nn.Module, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the residual of the PDE: df/dt - R * t * (1 - t)
        
        Args:
            model: Neural network model
            t: Input tensor (time)
            
        Returns:
            PDE residual tensor
        """
        # Compute df/dt
        df_dt_val = df_single(model, t, order=1)
        
        # Compute right-hand side of PDE
        rhs = self.R * t * (1 - t)
        
        # Return the residual
        return df_dt_val - rhs
    
    def initial_condition_residual(self, model: nn.Module, t0: torch.Tensor) -> torch.Tensor:
        """
        Compute the residual of the initial condition: f(t=0) - ft0
        
        Args:
            model: Neural network model
            t0: Tensor representing t=0
            
        Returns:
            Initial condition residual tensor
        """
        f_t0 = model(t0)
        return f_t0 - self.ft0


class WaveOperator:
    """
    Operator for the 1D wave equation: d²f/dx² = (1/C²) * d²f/dt²
    """
    
    def __init__(self, C: float = 1.0):
        """
        Initialize the wave equation operator.
        
        Args:
            C: Wave speed constant
        """
        self.C = C
    
    def pde_residual(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the residual of the PDE: d²f/dx² - (1/C²) * d²f/dt²
        
        Args:
            model: Neural network model
            x: Spatial input tensor
            t: Time input tensor
            
        Returns:
            PDE residual tensor
        """
        # Compute d²f/dx²
        d2f_dx2 = df_dx(model, x, t, order=2)
        
        # Compute d²f/dt²
        d2f_dt2 = df_dt(model, x, t, order=2)
        
        # Return the residual
        return d2f_dx2 - (1 / self.C**2) * d2f_dt2
    
    def boundary_condition_residual(self, model: nn.Module, 
                                   x0: torch.Tensor, x1: torch.Tensor, 
                                   t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the residual of the boundary conditions: f(x=0,t) and f(x=1,t)
        
        Args:
            model: Neural network model
            x0: Tensor representing x=0
            x1: Tensor representing x=1
            t: Time input tensor
            
        Returns:
            Tuple of boundary condition residual tensors
        """
        f_x0 = model(x0, t)
        f_x1 = model(x1, t)
        return f_x0, f_x1
    
    def initial_condition_residual(self, model: nn.Module, 
                                  x: torch.Tensor, t0: torch.Tensor, 
                                  f_init: Optional[Callable] = None) -> torch.Tensor:
        """
        Compute the residual of the initial condition: f(x,t=0) - f_init(x)
        
        Args:
            model: Neural network model
            x: Spatial input tensor
            t0: Tensor representing t=0
            f_init: Optional function for initial condition
            
        Returns:
            Initial condition residual tensor
        """
        f_x_t0 = model(x, t0)
        
        if f_init is not None:
            f_init_x = f_init(x)
            return f_x_t0 - f_init_x
        else:
            # Default initial condition: sin(2πx)/2
            f_init_x = torch.sin(2 * np.pi * x) * 0.5
            return f_x_t0 - f_init_x
    
    def initial_derivative_residual(self, model: nn.Module, 
                                   x: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
        """
        Compute the residual of the initial derivative condition: df/dt(x,t=0) = 0
        
        Args:
            model: Neural network model
            x: Spatial input tensor
            t0: Tensor representing t=0
            
        Returns:
            Initial derivative condition residual tensor
        """
        df_dt_x_t0 = df_dt(model, x, t0, order=1)
        return df_dt_x_t0  # The initial condition is df/dt = 0