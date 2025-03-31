#!/usr/bin/env python
"""
Evaluation script for Physics Informed Neural Networks.

This script loads a trained model and evaluates it on the specified domain.
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
import json
from pathlib import Path
from datetime import datetime

from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.models.networks import create_model
from src.pde.equations import create_equation
from src.utils.training import create_trainer
from src.utils.plotting import (
    plot_logistic_equation_results,
    plot_wave_equation_results
)


def main(args):
    """
    Main function for evaluating a PINN model.
    
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
    
    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config.get("logging.save_dir"))
    logger = setup_logger(
        name=f"pinns_evaluate_{timestamp}",
        log_dir=log_dir,
        level=config.get("logging.level", "INFO")
    )
    
    pde_name = config.get("pde.name", "unknown")
    logger.info(f"Evaluating PINN for '{pde_name}' equation")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Device: {device}")
    
    # Create model
    model = create_model(config)
    logger.info(f"Model created: {model.__class__.__name__}")
    
    # Create equation
    equation = create_equation(config)
    logger.info(f"Equation created: {equation.__class__.__name__}")
    
    # Create trainer (for loading the model)
    trainer = create_trainer(model, equation, config, logger)
    logger.info(f"Trainer created: {trainer.__class__.__name__}")
    
    # Load model
    if not args.model_path:
        # If no model path is provided, try to find the latest model
        model_dir = Path(config.get("output.model_dir", "outputs/models"))
        model_files = list(model_dir.glob("*model_*.pt"))
        if model_files:
            # Sort by modification time
            model_path = str(sorted(model_files, key=os.path.getmtime)[-1])
            logger.info(f"Using latest model: {model_path}")
        else:
            logger.error(f"No model found in {model_dir}")
            return 1
    else:
        model_path = args.model_path
    
    try:
        epoch = trainer.load_model(model_path)
        logger.info(f"Model loaded from: {model_path} (epoch {epoch})")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return 1
    
    # Determine the type of PDE and call the appropriate evaluation method
    if pde_name.lower() == "logistic":
        # Evaluate logistic equation
        t_domain = config.get("data.domain_x", [0.0, 1.5])
        n_points = args.n_points or config.get("data.n_points_x", 100)
        
        logger.info(f"Evaluating logistic equation model...")
        logger.info(f"Time domain: {t_domain}")
        logger.info(f"Number of points: {n_points}")
        
        results = trainer.evaluate(t_domain=t_domain, n_points=n_points)
        
        # Get training data if available
        train_points = config.get("data.train_points", [])
        if train_points:
            x_train = np.array([[p[0]] for p in train_points])
            y_train = np.array([[p[1]] for p in train_points])
            train_data = (x_train, y_train)
        else:
            train_data = None
        
        # Plot results
        figure_dir = Path(config.get("output.figure_dir"))
        figure_path = figure_dir / f"logistic_evaluation_{timestamp}.png"
        
        plot_logistic_equation_results(
            results=results,
            train_data=train_data,
            save_path=figure_path,
            show=not args.no_plot
        )
        
    elif pde_name.lower() == "wave":
        # Evaluate wave equation
        x_domain = config.get("data.domain_x", [0.0, 1.0])
        t_domain = config.get("data.domain_t", [0.0, 1.0])
        n_points_x = args.n_points or config.get("data.n_points_x", 100)
        n_points_t = args.n_points or config.get("data.n_points_t", 150)
        
        logger.info(f"Evaluating wave equation model...")
        logger.info(f"Spatial domain: {x_domain}")
        logger.info(f"Time domain: {t_domain}")
        logger.info(f"Number of points (spatial): {n_points_x}")
        logger.info(f"Number of points (time): {n_points_t}")
        
        results = trainer.evaluate(
            x_domain=x_domain,
            t_domain=t_domain,
            n_points_x=n_points_x,
            n_points_t=n_points_t
        )
        
        # Plot results
        figure_dir = Path(config.get("output.figure_dir"))
        figure_path = figure_dir / f"wave_evaluation_{timestamp}.png"
        
        plot_wave_equation_results(
            results=results,
            save_path=figure_path,
            show=not args.no_plot
        )
        
    else:
        logger.error(f"Unsupported PDE type: {pde_name}")
        return 1
    
    # Save evaluation results
    metrics_dir = Path(config.get("output.metrics_dir", "outputs/metrics"))
    metrics_file = metrics_dir / f"evaluation_{pde_name}_{timestamp}.json"
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Extract metrics that can be serialized to JSON
    serializable_results = {}
    for key, value in results.items():
        if key == "metrics":
            serializable_results[key] = value
        elif isinstance(value, np.ndarray):
            # Skip large arrays for JSON serialization
            serializable_results[f"{key}_shape"] = value.shape
        else:
            try:
                json.dumps({key: value})  # Test if serializable
                serializable_results[key] = value
            except:
                pass
    
    with open(metrics_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {metrics_file}")
    logger.info(f"Evaluation completed")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Physics Informed Neural Network")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to configuration file")
    parser.add_argument("--model-path", type=str, default=None,
                      help="Path to trained model checkpoint (if not provided, uses latest)")
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Output directory (overrides config)")
    parser.add_argument("--gpu", action="store_true",
                      help="Use GPU if available")
    parser.add_argument("--cpu", action="store_true",
                      help="Force CPU usage")
    parser.add_argument("--n-points", type=int, default=None,
                      help="Number of evaluation points (overrides config)")
    parser.add_argument("--no-plot", action="store_true",
                      help="Don't show plots")
    
    args = parser.parse_args()
    
    sys.exit(main(args))