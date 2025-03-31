#!/usr/bin/env python
"""
Example: Solving the 1D Wave Equation using Physics Informed Neural Networks.

The 1D wave equation is a second-order partial differential equation
given by:
    d²f/dx² = (1/C²) * d²f/dt²

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
from src.utils.plotting import plot_wave_equation_results, plot_training_history


def main(config_path: str, output_dir: str = None, gpu: bool = None, seed: int = None):
    """
    Main function to solve the 1D wave equation with PINNs.
    
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
        name="wave_equation",
        log_dir=log_dir,
        level=config.get("logging.level", "INFO")
    )
    
    logger.info(f"Solving 1D Wave Equation with PINNs")
    logger.info(f"Configuration: {config_path}")
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
    
    # Train the model
    x_domain = config.get("data.domain_x", [0.0, 1.0])  # Spatial domain
    t_domain = config.get("data.domain_t", [0.0, 1.0])  # Time domain
    n_points_x = config.get("data.n_points_x", 100)
    n_points_t = config.get("data.n_points_t", 150)
    
    logger.info(f"Starting training...")
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
    
    logger.info(f"Training completed")
    
    # Evaluate the model
    logger.info(f"Evaluating model...")
    results = trainer.evaluate(
        x_domain=x_domain,
        t_domain=t_domain,
        n_points_x=n_points_x,
        n_points_t=n_points_t
    )
    
    # Plot results
    figure_dir = Path(config.get("output.figure_dir"))
    
    # Plot solution
    if config.get("output.plot_3d_surface", True):
        solution_fig = plot_wave_equation_results(
            results=results,
            save_path=figure_dir / "wave_solution_3d.png",
            show=False
        )
    
    # Plot training history
    history_figs = plot_training_history(
        history=history,
        save_path=figure_dir / "wave_training_history.png",
        show=False
    )
    
    logger.info(f"Figures saved to {figure_dir}")
    logger.info(f"Done!")
    
    # Show all figures
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve 1D Wave Equation with PINNs")
    parser.add_argument("--config", type=str, default="config/wave_eq.yaml",
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