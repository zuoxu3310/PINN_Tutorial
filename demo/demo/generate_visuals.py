#!/usr/bin/env python
"""
Generate visual demonstrations of the Physics Informed Neural Networks in action.
This script creates plots showing the solution of the logistic and wave equations.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import sys

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.models.networks import create_model
from src.pde.equations import create_equation
from src.utils.training import create_trainer


def demonstrate_logistic_equation(output_dir):
    """
    Create a demonstration of the logistic equation solution.
    """
    print("Generating logistic equation demonstration...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config("config/logistic_eq.yaml")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Create equation
    equation = create_equation(config)
    
    # Create trainer
    trainer = create_trainer(model, equation, config)
    
    # Generate training data
    train_points = [
        [0.0, 1.0],
        [0.15, 1.0141],
        [0.3, 1.0456],
        [0.45, 1.0753],
        [0.7, 1.1565]
    ]
    x_train = torch.tensor([[p[0]] for p in train_points], dtype=torch.float32).to(device)
    y_train = torch.tensor([[p[1]] for p in train_points], dtype=torch.float32).to(device)
    
    # Train the model (short training for demonstration)
    t_domain = [0.0, 1.5]
    n_points = 100
    n_collocation = 10
    
    # Reduce epochs for quick demonstration
    config.config["training"]["epochs"] = 200
    config.config["training"]["print_every"] = 50
    
    print("Training logistic equation model...")
    trainer.train(
        t_domain=t_domain,
        n_points=n_points,
        n_collocation=n_collocation,
        x_train=x_train,
        y_train=y_train
    )
    
    # Evaluate the model
    print("Evaluating logistic equation model...")
    results = trainer.evaluate(t_domain=t_domain, n_points=n_points)
    
    # Create a figure for the solution
    print("Creating visualization...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot predicted solution
    t = results['t']
    f_predicted = results['prediction']
    f_analytical = results.get('analytical', None)
    
    ax.plot(t, f_predicted, label="PINN Solution", color="darkgreen", linewidth=2)
    
    # Plot analytical solution if available
    if f_analytical is not None:
        ax.plot(t, f_analytical, label="Analytical Solution", color="magenta", 
               alpha=0.75, linestyle="--", linewidth=2)
    
    # Plot training data
    train_data_np = np.array(train_points)
    ax.scatter(train_data_np[:, 0], train_data_np[:, 1], 
              label="Training Data", color="blue", s=80, zorder=5)
    
    # Create collocation points for visualization
    t_colloc = np.linspace(t_domain[0], t_domain[1], n_collocation)
    
    # Evaluate analytical solution at collocation points
    from scipy.integrate import solve_ivp
    def logistic_eq_fn(x, y):
        return equation.R * x * (1 - x)  # Assuming R=1.0
    
    sol = solve_ivp(logistic_eq_fn, t_domain, [equation.ft0], t_eval=t_colloc)
    f_colloc = sol.y.T
    
    ax.scatter(t_colloc, f_colloc, label="Collocation Points", 
              color="magenta", alpha=0.75, s=50, zorder=4, marker="x")
    
    # Set labels and title
    ax.set_xlabel("Time (t)", fontsize=12)
    ax.set_ylabel("Population (f(t))", fontsize=12)
    ax.set_title("Logistic Equation: PINN Solution vs Analytical Solution", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    # Save the figure
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "logistic_equation_demo.png"), dpi=300, bbox_inches='tight')
    
    print(f"Logistic equation demonstration saved to {output_dir}")
    return fig


def demonstrate_wave_equation(output_dir):
    """
    Create a demonstration of the wave equation solution.
    """
    print("Generating wave equation demonstration...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config("config/wave_eq.yaml")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Create equation
    equation = create_equation(config)
    
    # Create trainer
    trainer = create_trainer(model, equation, config)
    
    # Reduce epochs for quick demonstration
    config.config["training"]["epochs"] = 300
    config.config["training"]["print_every"] = 50
    
    # Train the model
    x_domain = [0.0, 1.0]
    t_domain = [0.0, 1.0]
    n_points_x = 50
    n_points_t = 50
    
    print("Training wave equation model...")
    trainer.train(
        x_domain=x_domain,
        t_domain=t_domain,
        n_points_x=n_points_x,
        n_points_t=n_points_t
    )
    
    # Evaluate the model
    print("Evaluating wave equation model...")
    results = trainer.evaluate(
        x_domain=x_domain,
        t_domain=t_domain,
        n_points_x=n_points_x,
        n_points_t=n_points_t
    )
    
    # Create a figure for the solution
    print("Creating visualization...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    x = results['x']
    t = results['t']
    f = results['prediction']
    
    # Create meshgrid for plotting
    X, T = np.meshgrid(t, x)
    
    surf = ax.plot_surface(X, T, f, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='f(x,t)')
    
    # Set labels and title
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Position (x)', fontsize=12)
    ax.set_zlabel('Amplitude f(x,t)', fontsize=12)
    ax.set_title('1D Wave Equation: PINN Solution', fontsize=14)
    
    # Save the figure
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "wave_equation_demo.png"), dpi=300, bbox_inches='tight')
    
    # Create a 2D slice at specific time points
    fig2, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    
    # Plot slices at different time points
    time_points = [0, 16, 33, 49]  # Indices for t=0, t=0.33, t=0.66, t=1.0
    for i, t_idx in enumerate(time_points):
        axs[i].plot(x, f[:, t_idx], 'b-', linewidth=2)
        axs[i].set_title(f"t = {t[t_idx]:.2f}", fontsize=12)
        axs[i].set_xlabel('Position (x)', fontsize=10)
        axs[i].set_ylabel('Amplitude f(x,t)', fontsize=10)
        axs[i].grid(alpha=0.3)
    
    fig2.suptitle('1D Wave Equation: Time Evolution', fontsize=14)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "wave_equation_time_slices.png"), dpi=300, bbox_inches='tight')
    
    print(f"Wave equation demonstration saved to {output_dir}")
    return fig, fig2


if __name__ == "__main__":
    output_dir = "demo/figures"
    
    # Run demonstrations
    logistic_fig = demonstrate_logistic_equation(output_dir)
    wave_fig, wave_slices_fig = demonstrate_wave_equation(output_dir)
    
    # Show all figures
    plt.show()