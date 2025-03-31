#!/usr/bin/env python
"""
Example: Solving the Logistic Equation using Physics Informed Neural Networks.

The logistic equation is a first-order ordinary differential equation
given by:
    df(t)/dt = R * f(t) * (1 - f(t))

This example demonstrates how to solve this equation using PINNs.
"""

import os
import sys
# Ensure the src directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import datetime

from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.models.networks import create_model
from src.pde.equations import create_equation
from src.utils.training import create_trainer
from src.utils.plotting import plot_logistic_equation_results, plot_training_history


def main(config_path: str, output_dir: str = None, gpu: bool = None, seed: int = None):
    """
    Main function to solve the logistic equation with PINNs.
    
    Args:
        config_path: Path to configuration file
        output_dir: Optional output directory (overrides config)
        gpu: Whether to use GPU (overrides config)
        seed: Random seed for reproducibility
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override output directory if specified
    if output_dir:
        config.config["output"]["save_dir"] = output_dir
        config.set_derived_paths()
    
    # Set device
    if gpu is not None:
        device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        config.config["training"]["device"] = device
    else:
        device = config.get("training.device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set random seed if specified
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config.get("logging.save_dir"))
    logger = setup_logger(
        name="logistic_equation",
        log_dir=log_dir,
        level=config.get("logging.level", "INFO")
    )
    
    logger.info(f"Solving Logistic Equation with PINNs")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Device: {device}")
    
    # Create model
    model = create_model(config)
    logger.info(f"Model created: {model.__class__.__name__}")
    
    # Create equation
    equation = create_equation(config)
    logger.info(f"Equation created: {equation.__class__.__name__}")
    
    # Get training data
    train_points = config.get("data.train_points", [])
    if train_points:
        x_train = torch.tensor([[p[0]] for p in train_points], dtype=torch.float32)
        y_train = torch.tensor([[p[1]] for p in train_points], dtype=torch.float32)
        logger.info(f"Training data loaded: {len(train_points)} points")
    else:
        x_train = None
        y_train = None
        logger.info("No training data provided, using only PDE constraints")
    
    # Create trainer
    trainer = create_trainer(model, equation, config, logger)
    logger.info(f"Trainer created: {trainer.__class__.__name__}")
    
    # Train the model
    t_domain = config.get("data.domain_x", [0.0, 1.5])  # Domain for t (time variable)
    n_points = config.get("data.n_points_x", 100)
    n_collocation = config.get("data.n_collocation", 10)
    
    logger.info(f"Starting training...")
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
    
    logger.info(f"Training completed")
    
    # Evaluate the model
    logger.info(f"Evaluating model...")
    results = trainer.evaluate(t_domain=t_domain, n_points=n_points)
    
    # Plot results
    figure_dir = Path(config.get("output.figure_dir"))
    
    # Create train data and collocation points for plotting
    if x_train is not None and y_train is not None:
        train_data = (x_train.cpu().numpy(), y_train.cpu().numpy())
    else:
        train_data = None
    
    t_colloc = torch.linspace(t_domain[0], t_domain[1], n_collocation)
    t_colloc = t_colloc.reshape(-1, 1).cpu().numpy()
    
    # Plot solution
    solution_fig = plot_logistic_equation_results(
        results=results,
        train_data=train_data,
        collocation_points=t_colloc,
        save_path=figure_dir / "logistic_solution.png",
        show=False
    )
    
    # Plot training history
    history_figs = plot_training_history(
        history=history,
        save_path=figure_dir / "logistic_training_history.png",
        show=False
    )
    
    logger.info(f"Figures saved to {figure_dir}")
    logger.info(f"Results:")
    if "metrics" in results and "mse" in results["metrics"]:
        logger.info(f"  MSE: {results['metrics']['mse']:.6f}")
    
    logger.info(f"Done!")
    
    # Show all figures
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve Logistic Equation with PINNs")
    parser.add_argument("--config", type=str, default="config/logistic_eq.yaml",
                      help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Output directory (overrides config)")
    parser.add_argument("--gpu", action="store_true",
                      help="Use GPU if available")
    parser.add_argument("--cpu", action="store_true",
                      help="Force CPU usage")
    parser.add_argument("--seed", type=int, default=None,
                      help="Random seed")
    
    args = parser.parse_args()
    
    # Determine GPU usage
    gpu = args.gpu or (not args.cpu)
    
    main(config_path=args.config, output_dir=args.output_dir, gpu=gpu, seed=args.seed)