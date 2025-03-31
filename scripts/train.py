#!/usr/bin/env python
"""
Generic training script for Physics Informed Neural Networks.

This script provides a unified interface for training different types of PINNs models.
"""

import os
import sys
# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import datetime
import importlib.util

from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.models.networks import create_model
from src.pde.equations import create_equation
from src.utils.training import create_trainer


def load_module_from_path(path, module_name="custom_module"):
    """
    Load a Python module from a file path.
    
    Args:
        path: Path to the Python file
        module_name: Name to assign to the module
        
    Returns:
        Loaded module
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    """
    Main function for training a PINN model.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config.config["output"]["save_dir"] = args.output_dir
        config.set_derived_paths()
    
    # Set device
    if args.gpu or args.cpu:
        device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
        config.config["training"]["device"] = device
    else:
        device = config.get("training.device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set random seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config.get("logging.save_dir"))
    logger = setup_logger(
        name=f"pinns_train_{timestamp}",
        log_dir=log_dir,
        level=config.get("logging.level", "INFO")
    )
    
    pde_name = config.get("pde.name", "unknown")
    logger.info(f"Training PINN for '{pde_name}' equation")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Device: {device}")
    
    # Create model
    model = create_model(config)
    logger.info(f"Model created: {model.__class__.__name__}")
    
    # Create equation
    equation = create_equation(config)
    logger.info(f"Equation created: {equation.__class__.__name__}")
    
    # Create trainer
    trainer = create_trainer(model, equation, config, logger)
    logger.info(f"Trainer created: {trainer.__class__.__name__}")
    
    # Determine the type of PDE and call the appropriate training method
    if pde_name.lower() == "logistic":
        # Train for logistic equation
        t_domain = config.get("data.domain_x", [0.0, 1.5])
        n_points = config.get("data.n_points_x", 100)
        n_collocation = config.get("data.n_collocation", 10)
        
        # Get training data if available
        train_points = config.get("data.train_points", [])
        if train_points:
            x_train = torch.tensor([[p[0]] for p in train_points], dtype=torch.float32)
            y_train = torch.tensor([[p[1]] for p in train_points], dtype=torch.float32)
            logger.info(f"Training data loaded: {len(train_points)} points")
        else:
            x_train = None
            y_train = None
            logger.info("No training data provided, using only PDE constraints")
        
        logger.info(f"Starting training for logistic equation...")
        logger.info(f"Time domain: {t_domain}")
        logger.info(f"Number of points: {n_points}")
        logger.info(f"Number of collocation points: {n_collocation}")
        
        history = trainer.train(
            t_domain=t_domain,
            n_points=n_points,
            n_collocation=n_collocation,
            x_train=x_train,
            y_train=y_train
        )
        
    elif pde_name.lower() == "wave":
        # Train for wave equation
        x_domain = config.get("data.domain_x", [0.0, 1.0])
        t_domain = config.get("data.domain_t", [0.0, 1.0])
        n_points_x = config.get("data.n_points_x", 100)
        n_points_t = config.get("data.n_points_t", 150)
        
        logger.info(f"Starting training for wave equation...")
        logger.info(f"Spatial domain: {x_domain}")
        logger.info(f"Time domain: {t_domain}")
        logger.info(f"Number of points (spatial): {n_points_x}")
        logger.info(f"Number of points (time): {n_points_t}")
        
        history = trainer.train(
            x_domain=x_domain,
            t_domain=t_domain,
            n_points_x=n_points_x,
            n_points_t=n_points_t
        )
        
    else:
        # For custom equations, try to import a custom module if provided
        if args.custom_module:
            try:
                custom = load_module_from_path(args.custom_module)
                logger.info(f"Loaded custom module: {args.custom_module}")
                
                if hasattr(custom, "train_model"):
                    logger.info(f"Using custom training function")
                    history = custom.train_model(trainer, config, logger)
                else:
                    logger.error(f"Custom module does not have a 'train_model' function")
                    return 1
            except Exception as e:
                logger.error(f"Error loading custom module: {e}")
                return 1
        else:
            logger.error(f"Unsupported PDE type: {pde_name}")
            logger.error(f"Please provide a custom module with a 'train_model' function")
            return 1
    
    logger.info(f"Training completed")
    logger.info(f"Model saved to: {trainer.model_dir}")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Physics Informed Neural Network")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Output directory (overrides config)")
    parser.add_argument("--gpu", action="store_true",
                      help="Use GPU if available")
    parser.add_argument("--cpu", action="store_true",
                      help="Force CPU usage")
    parser.add_argument("--seed", type=int, default=None,
                      help="Random seed")
    parser.add_argument("--custom-module", type=str, default=None,
                      help="Path to a custom Python module for unsupported PDE types")
    
    args = parser.parse_args()
    
    sys.exit(main(args))