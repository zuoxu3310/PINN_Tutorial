# Physics Informed Neural Networks (PINNs)

This repository contains a modular and extensible implementation of Physics Informed Neural Networks for solving partial differential equations (PDEs).

## Overview

Physics Informed Neural Networks (PINNs) combine deep learning with physical knowledge to solve partial differential equations. They train neural networks to approximate solutions to PDEs by incorporating the PDE itself into the loss function.

Main features of this implementation:
- Modular architecture for easy extension to new types of PDEs
- Configurable via YAML files
- Comprehensive logging and visualization
- Supports both 1D and 2D problems

## Project Structure

```
pinns/
│
├── config/                 # Configuration files
│   ├── default.yaml        # Default configuration parameters
│   ├── logistic_eq.yaml    # Config for logistic equation example
│   └── wave_eq.yaml        # Config for wave equation example
│
├── src/                    # Source code
│   ├── models/             # Neural network architecture definitions
│   ├── pde/                # PDE operators and equations
│   └── utils/              # Utility functions
│
├── examples/               # Example scripts for different equations
│   ├── logistic_equation.py
│   └── wave_equation.py
│
├── scripts/                # Generic scripts for training and evaluation
│   ├── train.py
│   └── evaluate.py
│
└── outputs/                # Directory for outputs
    ├── models/             # Saved model checkpoints
    ├── figures/            # Generated plots and visualizations
    ├── logs/               # Training logs
    └── metrics/            # Training and evaluation metrics
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/zuoxu3310/PINN_Tutorial.git
   cd PINN_Turorial
   ```

2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Supported PDEs

Currently, the following PDEs are implemented:

1. **Logistic Equation** (1st order ODE):
   ```
   df(t)/dt = R * f(t) * (1 - f(t))
   ```

2. **1D Wave Equation** (2nd order PDE):
   ```
   d²f/dx² = (1/C²) * d²f/dt²
   ```

## Usage

### Training a Model

To train a model for the logistic equation:

```bash
python examples/logistic_equation.py --config config/logistic_eq.yaml --gpu
```

To train a model for the wave equation:

```bash
python examples/wave_equation.py --config config/wave_eq.yaml --gpu
```

Or use the generic training script:

```bash
python scripts/train.py --config config/logistic_eq.yaml --gpu
```

### Evaluating a Model

```bash
python scripts/evaluate.py --config config/logistic_eq.yaml --model-path outputs/models/model_final.pt
```

### Configuration

Models and training parameters are configured via YAML files. See the `config/` directory for examples.

Key configuration parameters:

- `model`: Neural network architecture settings
- `pde`: PDE-specific parameters (type, coefficients, etc.)
- `training`: Training parameters (epochs, learning rate, etc.)
- `data`: Domain and discretization parameters
- `output`: Output settings (directories, plots, etc.)

## Extending to New PDEs

To add a new PDE:

1. Create a new PDE operator in `src/pde/operators.py`
2. Implement the equation in `src/pde/equations.py`
3. Add a trainer in `src/utils/training.py`
4. Create a configuration file in `config/`
5. Either create a dedicated example or use the generic training script with a custom module

## Results

The framework outputs various visualizations and metrics:

- Solution plots showing the PINN approximation vs. analytical solutions (when available)
- Training history plots (loss evolution, computation time, etc.)
- Metrics saved as JSON files for further analysis

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

2. Nascimento, R. G., Fricke, K., & Viana, F. A. (2020). A tutorial on solving ordinary differential equations using Python and hybrid physics-informed neural network. Engineering Applications of Artificial Intelligence, 96, 103996.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
