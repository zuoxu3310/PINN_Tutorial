"""
Training utilities for Physics Informed Neural Networks.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable

from src.utils.logging import TensorBoardLogger


class Trainer:
    """Generic trainer for Physics Informed Neural Networks."""
    
    def __init__(self, 
                 model: nn.Module,
                 equation: Any,
                 config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None,
                 device: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model
            equation: PDE equation instance
            config: Configuration dictionary
            logger: Logger instance
            device: Device to use for training ('cpu', 'cuda', or None for auto-detection)
        """
        self.model = model
        self.equation = equation
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        
        # Training parameters
        self.epochs = config.get("training.epochs", 2000)
        self.lr = config.get("training.learning_rate", 1e-2)
        self.print_every = config.get("training.print_every", 200)
        self.save_every = config.get("training.save_every", 500)
        
        # Setup optimizer
        optimizer_name = config.get("training.optimizer", "Adam")
        self.optimizer = self._get_optimizer(optimizer_name)
        
        # Initialize directories
        self.output_dir = Path(config.get("output.save_dir", "outputs"))
        self.model_dir = Path(config.get("output.model_dir", "outputs/models"))
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Setup TensorBoard logger if enabled
        tensorboard_enabled = config.get("logging.tensorboard", True)
        self.tb_logger = TensorBoardLogger(
            log_dir=self.output_dir / "logs",
            enabled=tensorboard_enabled
        )
        
        # Training history
        self.history = {
            "epoch": [],
            "loss": [],
            "loss_components": {},
            "time": []
        }
    
    def _get_optimizer(self, optimizer_name: str) -> torch.optim.Optimizer:
        """
        Create an optimizer instance.
        
        Args:
            optimizer_name: Name of the optimizer
            
        Returns:
            Optimizer instance
        """
        if optimizer_name.lower() == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer_name.lower() == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif optimizer_name.lower() == "rmsprop":
            return torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            self.logger.warning(f"Unsupported optimizer: {optimizer_name}, using Adam")
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def save_model(self, path: Optional[str] = None, epoch: Optional[int] = None) -> str:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model, if None a default path is used
            epoch: Current epoch number
            
        Returns:
            Path where the model was saved
        """
        if path is None:
            if epoch is not None:
                filename = f"model_epoch_{epoch}.pt"
            else:
                filename = "model_final.pt"
            path = self.model_dir / filename
        
        # Save the model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch or 0,
            'config': self.config
        }, path)
        
        return str(path)
    
    def load_model(self, path: str) -> int:
        """
        Load a model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            The epoch number of the loaded model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        
        return epoch
    
    def save_history(self, path: Optional[str] = None) -> str:
        """
        Save training history to disk.
        
        Args:
            path: Path to save the history, if None a default path is used
            
        Returns:
            Path where the history was saved
        """
        if path is None:
            path = self.output_dir / "metrics" / "training_history.json"
            os.makedirs(path.parent, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return str(path)
    
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        This method should be implemented by subclasses.
        
        Returns:
            Training history
        """
        raise NotImplementedError("Subclasses must implement the train method")
    
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the model.
        This method should be implemented by subclasses.
        
        Returns:
            Evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement the evaluate method")


class LogisticEquationTrainer(Trainer):
    """Trainer for the logistic equation."""
    
    def train(self, 
             t_domain: List[float], 
             n_points: int,
             n_collocation: int,
             x_train: Optional[torch.Tensor] = None,
             y_train: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Train the model for the logistic equation.
        
        Args:
            t_domain: Domain for the time variable [t_min, t_max]
            n_points: Number of points for evaluation
            n_collocation: Number of collocation points for PDE enforcement
            x_train: Optional training inputs (time points)
            y_train: Optional training targets (function values)
            
        Returns:
            Training history
        """
        # Create the time domain
        t_eval = torch.linspace(t_domain[0], t_domain[1], n_points, device=self.device)
        t_eval = t_eval.reshape(-1, 1)
        t_eval.requires_grad = True
        
        # Create collocation points for PDE enforcement
        t_colloc = torch.linspace(t_domain[0], t_domain[1], n_collocation, device=self.device)
        t_colloc = t_colloc.reshape(-1, 1)
        t_colloc.requires_grad = True
        
        # Move training data to device if provided
        if x_train is not None and y_train is not None:
            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)
        
        # Training loop
        start_time = time.time()
        self.model.train()
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            losses = self.equation.compute_loss(
                model=self.model,
                t=t_eval,
                t_colloc=t_colloc,
                x_train=x_train,
                y_train=y_train
            )
            
            total_loss = losses["total"]
            
            # Backward pass and optimization
            total_loss.backward()
            self.optimizer.step()
            
            # Record history
            epoch_time = time.time() - epoch_start_time
            self.history["epoch"].append(epoch)
            self.history["loss"].append(total_loss.item())
            self.history["time"].append(epoch_time)
            
            # Record loss components
            for key, value in losses.items():
                if key != "total":
                    if key not in self.history["loss_components"]:
                        self.history["loss_components"][key] = []
                    self.history["loss_components"][key].append(value.item())
            
            # Log to TensorBoard
            self.tb_logger.log_scalar("loss/total", total_loss.item(), epoch)
            for key, value in losses.items():
                if key != "total":
                    self.tb_logger.log_scalar(f"loss/{key}", value.item(), epoch)
            
            # Print progress
            if epoch % self.print_every == 0 or epoch == self.epochs - 1:
                self.logger.info(f"Epoch {epoch}/{self.epochs} - Loss: {total_loss.item():.6f}")
                for key, value in losses.items():
                    if key != "total":
                        self.logger.info(f"  {key}: {value.item():.6f}")
            
            # Save checkpoint
            if (epoch % self.save_every == 0 and epoch > 0) or epoch == self.epochs - 1:
                self.save_model(epoch=epoch)
        
        # Save final model and history
        self.save_model()
        self.save_history()
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return self.history
    
    def evaluate(self, t_domain: List[float], n_points: int) -> Dict[str, Any]:
        """
        Evaluate the model for the logistic equation.
        
        Args:
            t_domain: Domain for the time variable [t_min, t_max]
            n_points: Number of points for evaluation
            
        Returns:
            Evaluation metrics and predictions
        """
        # Create the time domain for evaluation
        t_eval = torch.linspace(t_domain[0], t_domain[1], n_points, device=self.device)
        t_eval = t_eval.reshape(-1, 1)
        
        # Switch to evaluation mode
        self.model.eval()
        
        # Compute predictions
        with torch.no_grad():
            f_eval = self.model(t_eval)
        
        # Get analytical solution if available
        try:
            f_analytical = self.equation.get_analytical_solution(t_eval)
        except:
            f_analytical = None
        
        # Convert to numpy for plotting
        t_np = t_eval.cpu().numpy()
        f_np = f_eval.cpu().numpy()
        
        # Compute metrics if analytical solution is available
        metrics = {}
        if f_analytical is not None:
            mse = np.mean((f_np - f_analytical) ** 2)
            metrics["mse"] = mse
        
        return {
            "t": t_np,
            "prediction": f_np,
            "analytical": f_analytical,
            "metrics": metrics
        }


class WaveEquationTrainer(Trainer):
    """Trainer for the wave equation."""
    
    def train(self, 
             x_domain: List[float],
             t_domain: List[float],
             n_points_x: int,
             n_points_t: int) -> Dict[str, Any]:
        """
        Train the model for the wave equation.
        
        Args:
            x_domain: Domain for the spatial variable [x_min, x_max]
            t_domain: Domain for the time variable [t_min, t_max]
            n_points_x: Number of points in the spatial dimension
            n_points_t: Number of points in the time dimension
            
        Returns:
            Training history
        """
        # Create the spatial and time domains
        x_idx = torch.linspace(x_domain[0], x_domain[1], n_points_x, device=self.device)
        t_idx = torch.linspace(t_domain[0], t_domain[1], n_points_t, device=self.device)
        
        # Create meshgrid
        X, T = torch.meshgrid(x_idx, t_idx, indexing="ij")
        x = X.flatten().reshape(-1, 1)
        t = T.flatten().reshape(-1, 1)
        
        # Ensure gradients can be computed
        x.requires_grad = True
        t.requires_grad = True
        x_idx.requires_grad = True
        t_idx.requires_grad = True
        
        # Reshape for use in boundary conditions
        x_idx = x_idx.reshape(-1, 1)
        t_idx = t_idx.reshape(-1, 1)
        
        # Training loop
        start_time = time.time()
        self.model.train()
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            losses = self.equation.compute_loss(
                model=self.model,
                x=x,
                t=t,
                x_idx=x_idx,
                t_idx=t_idx
            )
            
            total_loss = losses["total"]
            
            # Backward pass and optimization
            total_loss.backward()
            self.optimizer.step()
            
            # Record history
            epoch_time = time.time() - epoch_start_time
            self.history["epoch"].append(epoch)
            self.history["loss"].append(total_loss.item())
            self.history["time"].append(epoch_time)
            
            # Record loss components
            for key, value in losses.items():
                if key != "total":
                    if key not in self.history["loss_components"]:
                        self.history["loss_components"][key] = []
                    self.history["loss_components"][key].append(value.item())
            
            # Log to TensorBoard
            self.tb_logger.log_scalar("loss/total", total_loss.item(), epoch)
            for key, value in losses.items():
                if key != "total":
                    self.tb_logger.log_scalar(f"loss/{key}", value.item(), epoch)
            
            # Print progress
            if epoch % self.print_every == 0 or epoch == self.epochs - 1:
                self.logger.info(f"Epoch {epoch}/{self.epochs} - Loss: {total_loss.item():.6f}")
                for key, value in losses.items():
                    if key != "total":
                        self.logger.info(f"  {key}: {value.item():.6f}")
            
            # Save checkpoint
            if (epoch % self.save_every == 0 and epoch > 0) or epoch == self.epochs - 1:
                self.save_model(epoch=epoch)
        
        # Save final model and history
        self.save_model()
        self.save_history()
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return self.history
    
    def evaluate(self, 
                x_domain: List[float],
                t_domain: List[float],
                n_points_x: int,
                n_points_t: int) -> Dict[str, Any]:
        """
        Evaluate the model for the wave equation.
        
        Args:
            x_domain: Domain for the spatial variable [x_min, x_max]
            t_domain: Domain for the time variable [t_min, t_max]
            n_points_x: Number of points in the spatial dimension
            n_points_t: Number of points in the time dimension
            
        Returns:
            Evaluation results
        """
        # Create the spatial and time domains for evaluation
        x_eval = torch.linspace(x_domain[0], x_domain[1], n_points_x, device=self.device)
        t_eval = torch.linspace(t_domain[0], t_domain[1], n_points_t, device=self.device)
        
        # Create meshgrid
        X, T = torch.meshgrid(x_eval, t_eval, indexing="ij")
        x = X.flatten().reshape(-1, 1)
        t = T.flatten().reshape(-1, 1)
        
        # Switch to evaluation mode
        self.model.eval()
        
        # Compute predictions
        with torch.no_grad():
            f_eval = self.model(x, t)
        
        # Reshape for plotting
        f_eval = f_eval.reshape(n_points_x, n_points_t)
        
        # Convert to numpy for plotting
        x_np = x_eval.cpu().numpy()
        t_np = t_eval.cpu().numpy()
        f_np = f_eval.cpu().numpy()
        
        return {
            "x": x_np,
            "t": t_np,
            "prediction": f_np
        }


def create_trainer(model: nn.Module, 
                  equation: Any, 
                  config: Dict[str, Any],
                  logger: Optional[logging.Logger] = None) -> Trainer:
    """
    Create a trainer instance based on the equation type.
    
    Args:
        model: Neural network model
        equation: PDE equation instance
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Trainer instance
    """
    pde_name = config.get("pde.name", "").lower()
    
    if pde_name == "logistic":
        return LogisticEquationTrainer(model, equation, config, logger)
    elif pde_name == "wave":
        return WaveEquationTrainer(model, equation, config, logger)
    else:
        raise ValueError(f"Unsupported PDE type: {pde_name}")