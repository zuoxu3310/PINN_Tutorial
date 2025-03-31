"""
Specific PDE implementations for Physics Informed Neural Networks.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np

from src.pde.operators import LogisticOperator, WaveOperator
from src.pde.derivatives import df_single, df_dx, df_dt


class LogisticEquation:
    """
    Implementation of the logistic equation for population growth.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the logistic equation.
        
        Args:
            config: Configuration dictionary
        """
        self.R = config.get("pde.R", 1.0)
        self.ft0 = config.get("pde.ft0", 1.0)
        self.operator = LogisticOperator(R=self.R, ft0=self.ft0)
        
        # Loss weights
        self.loss_weights = {
            "data": config.get("pde.loss_weights.data", 1.0),
            "pde": config.get("pde.loss_weights.pde", 1.0),
            "bc": config.get("pde.loss_weights.bc", 1.0)
        }
    
    def compute_loss(self, model: nn.Module, 
                    t: torch.Tensor, 
                    t_colloc: torch.Tensor,
                    x_train: Optional[torch.Tensor] = None, 
                    y_train: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss for the logistic equation.
        
        Args:
            model: Neural network model
            t: Time points
            t_colloc: Collocation points for enforcing PDE
            x_train: Optional training inputs
            y_train: Optional training targets
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # PDE residual loss
        pde_residual = self.operator.pde_residual(model, t_colloc)
        losses["pde"] = pde_residual.pow(2).mean() * self.loss_weights["pde"]
        
        # Boundary condition (initial value)
        t0 = torch.zeros_like(t_colloc[0:1])
        t0 = t0.clone().detach()
        t0.requires_grad = True
        bc_residual = self.operator.initial_condition_residual(model, t0)
        losses["bc"] = bc_residual.pow(2).mean() * self.loss_weights["bc"]
        
        # Data loss (if provided)
        if x_train is not None and y_train is not None:
            pred = model(x_train)
            losses["data"] = torch.nn.functional.mse_loss(pred, y_train) * self.loss_weights["data"]
        
        # Total loss
        losses["total"] = sum(losses.values())
        
        return losses
    
    def get_analytical_solution(self, t: torch.Tensor) -> np.ndarray:
        """
        Get the analytical solution of the logistic equation.
        
        Args:
            t: Time points
            
        Returns:
            Analytical solution values
        """
        # Convert to numpy for scipy
        t_np = t.detach().cpu().numpy()
        
        # Solve with scipy.integrate.solve_ivp
        from scipy.integrate import solve_ivp
        
        def logistic_eq_fn(x, y):
            return self.R * x * (1 - x)
        
        domain = [t_np.min(), t_np.max()]
        solution = solve_ivp(
            logistic_eq_fn, domain, [self.ft0], t_eval=t_np.flatten()
        )
        
        return solution.y.T


class WaveEquation:
    """
    Implementation of the 1D wave equation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the wave equation.
        
        Args:
            config: Configuration dictionary
        """
        self.C = config.get("pde.C", 1.0)
        self.operator = WaveOperator(C=self.C)
        
        # Loss weights
        self.loss_weights = {
            "pde": config.get("pde.loss_weights.pde", 1.0),
            "bc": config.get("pde.loss_weights.bc", 1.0),
            "init_f": config.get("pde.loss_weights.init_f", 1.0),
            "init_df": config.get("pde.loss_weights.init_df", 1.0)
        }
    
    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the initial condition function f(x, t=0).
        
        Args:
            x: Spatial points
            
        Returns:
            Initial values at t=0
        """
        return torch.sin(2 * np.pi * x) * 0.5
    
    def compute_loss(self, model: nn.Module, 
                    x: torch.Tensor, t: torch.Tensor,
                    x_idx: torch.Tensor, t_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss for the wave equation.
        
        Args:
            model: Neural network model
            x: Spatial points for the mesh grid
            t: Time points for the mesh grid
            x_idx: Spatial points array
            t_idx: Time points array
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # PDE residual loss
        pde_residual = self.operator.pde_residual(model, x, t)
        losses["pde"] = pde_residual.pow(2).mean() * self.loss_weights["pde"]
        
        # Boundary conditions (spatial)
        x0 = torch.ones_like(t_idx) * x_idx[0]
        x0 = x0.clone().detach()
        x0.requires_grad = True
        x1 = torch.ones_like(t_idx) * x_idx[-1]
        x1 = x1.clone().detach()
        x1.requires_grad = True
        
        bc_residual_x0, bc_residual_x1 = self.operator.boundary_condition_residual(
            model, x0, x1, t_idx
        )
        losses["bc_x0"] = bc_residual_x0.pow(2).mean() * self.loss_weights["bc"]
        losses["bc_x1"] = bc_residual_x1.pow(2).mean() * self.loss_weights["bc"]
        
        # Initial condition at t=0 for f(x,t)
        t0 = torch.zeros_like(x_idx)
        t0 = t0.clone().detach()
        t0.requires_grad = True
        
        init_f_residual = self.operator.initial_condition_residual(
            model, x_idx, t0, self.initial_condition
        )
        losses["init_f"] = init_f_residual.pow(2).mean() * self.loss_weights["init_f"]
        
        # Initial condition at t=0 for df/dt(x,t)
        init_df_residual = self.operator.initial_derivative_residual(
            model, x_idx, t0
        )
        losses["init_df"] = init_df_residual.pow(2).mean() * self.loss_weights["init_df"]
        
        # Total loss
        losses["total"] = sum(losses.values())
        
        return losses


def create_equation(config: Dict[str, Any]) -> Any:
    """
    Create a PDE equation instance based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PDE equation instance
    """
    pde_name = config.get("pde.name", "").lower()
    
    if pde_name == "logistic":
        return LogisticEquation(config)
    elif pde_name == "wave":
        return WaveEquation(config)
    else:
        raise ValueError(f"Unsupported PDE type: {pde_name}")