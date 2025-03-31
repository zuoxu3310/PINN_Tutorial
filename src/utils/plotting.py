"""
Plotting utilities for Physics Informed Neural Networks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import logging


def plot_logistic_equation_results(results: Dict[str, Any], 
                                  train_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                  collocation_points: Optional[np.ndarray] = None,
                                  save_path: Optional[str] = None,
                                  show: bool = True) -> plt.Figure:
    """
    Plot the results of the logistic equation solution.
    
    Args:
        results: Dictionary with t, prediction, and optionally analytical solution
        train_data: Optional tuple of (t_train, f_train) for training data
        collocation_points: Optional array of collocation points
        save_path: Optional path to save the figure
        show: Whether to show the figure
        
    Returns:
        The created figure
    """
    t = results['t']
    f_predicted = results['prediction']
    f_analytical = results.get('analytical', None)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot predicted solution
    ax.plot(t, f_predicted, label="PINN solution", color="darkgreen", linewidth=2)
    
    # Plot analytical solution if available
    if f_analytical is not None:
        ax.plot(t, f_analytical, label="Analytical solution", color="magenta", alpha=0.75, linestyle="--", linewidth=2)
    
    # Plot training data if available
    if train_data is not None:
        t_train, f_train = train_data
        ax.scatter(t_train, f_train, label="Training data", color="blue", s=50, zorder=5)
    
    # Plot collocation points if available
    if collocation_points is not None:
        if f_analytical is not None:
            # Evaluate analytical solution at collocation points for visualization
            from scipy.integrate import solve_ivp
            def logistic_eq_fn(x, y):
                return 1.0 * x * (1 - x)  # Assuming R=1.0
            
            domain = [t.min(), t.max()]
            sol = solve_ivp(logistic_eq_fn, domain, [1.0], t_eval=collocation_points.flatten())
            f_colloc = sol.y.T
            
            ax.scatter(collocation_points, f_colloc, label="Collocation points", 
                      color="magenta", alpha=0.75, s=30, zorder=4)
    
    # Set labels and title
    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel("f(t)", fontsize=12)
    ax.set_title("Logistic Equation: Population Growth Model", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Add metrics if available
    metrics = results.get('metrics', {})
    if metrics:
        metric_text = []
        for key, value in metrics.items():
            if key.lower() == 'mse':
                metric_text.append(f"MSE: {value:.6f}")
        
        if metric_text:
            ax.text(0.02, 0.02, '\n'.join(metric_text), transform=ax.transAxes,
                   fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the figure if requested
    if show:
        plt.show()
    
    return fig


def plot_wave_equation_results(results: Dict[str, Any],
                              save_path: Optional[str] = None,
                              show: bool = True) -> plt.Figure:
    """
    Plot the results of the wave equation solution as a 3D surface.
    
    Args:
        results: Dictionary with x, t, and prediction arrays
        save_path: Optional path to save the figure
        show: Whether to show the figure
        
    Returns:
        The created figure
    """
    x = results['x']
    t = results['t']
    f = results['prediction']
    
    # Create meshgrid for plotting
    X, T = np.meshgrid(t, x)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, T, f, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='f(x,t)')
    
    # Set labels and title
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('x', fontsize=12)
    ax.set_zlabel('f(x,t)', fontsize=12)
    ax.set_title('1D Wave Equation Solution', fontsize=14)
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the figure if requested
    if show:
        plt.show()
    
    return fig


def plot_training_history(history: Dict[str, Any],
                         save_path: Optional[str] = None,
                         show: bool = True) -> List[plt.Figure]:
    """
    Plot training history metrics.
    
    Args:
        history: Training history dictionary
        save_path: Optional base path to save the figures
        show: Whether to show the figures
        
    Returns:
        List of created figures
    """
    figures = []
    
    # Plot total loss
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    ax_loss.plot(history["epoch"], history["loss"], label="Total Loss")
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.set_title("Training Loss Evolution", fontsize=14)
    ax_loss.set_yscale('log')
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()
    
    figures.append(fig_loss)
    
    # Plot loss components
    loss_components = history.get("loss_components", {})
    if loss_components:
        fig_components, ax_components = plt.subplots(figsize=(10, 6))
        
        for component, values in loss_components.items():
            ax_components.plot(history["epoch"][:len(values)], values, label=f"{component}")
        
        ax_components.set_xlabel("Epoch", fontsize=12)
        ax_components.set_ylabel("Loss", fontsize=12)
        ax_components.set_title("Loss Components Evolution", fontsize=14)
        ax_components.set_yscale('log')
        ax_components.grid(alpha=0.3)
        ax_components.legend()
        
        figures.append(fig_components)
    
    # Plot computation time
    if "time" in history:
        fig_time, ax_time = plt.subplots(figsize=(10, 6))
        ax_time.plot(history["epoch"], history["time"])
        ax_time.set_xlabel("Epoch", fontsize=12)
        ax_time.set_ylabel("Time (seconds)", fontsize=12)
        ax_time.set_title("Computation Time per Epoch", fontsize=14)
        ax_time.grid(alpha=0.3)
        
        figures.append(fig_time)
    
    # Save figures if requested
    if save_path:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract base filename without extension
        base_path = Path(save_path).with_suffix('')
        
        # Save each figure with a suffix
        for i, fig in enumerate(figures):
            if i == 0:
                fig_path = f"{base_path}_total_loss.png"
            elif i == 1:
                fig_path = f"{base_path}_loss_components.png"
            elif i == 2:
                fig_path = f"{base_path}_computation_time.png"
            else:
                fig_path = f"{base_path}_{i}.png"
            
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    # Show figures if requested
    if show:
        plt.show()
    
    return figures