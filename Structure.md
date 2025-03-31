```
pinns/
│
├── config/
│   ├── default.yaml        # Default configuration parameters
│   ├── logistic_eq.yaml    # Config for logistic equation example
│   └── wave_eq.yaml        # Config for wave equation example
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── networks.py     # Neural network architecture definitions
│   │
│   ├── pde/
│   │   ├── __init__.py
│   │   ├── derivatives.py  # Automatic differentiation utilities
│   │   ├── operators.py    # PDE operators and conditions
│   │   └── equations.py    # Specific equation implementations
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py       # Configuration handling
│       ├── logging.py      # Logging utilities
│       ├── plotting.py     # Visualization functions
│       └── training.py     # Training loops and utilities
│
├── examples/
│   ├── logistic_equation.py  # Example for solving logistic equation
│   └── wave_equation.py      # Example for solving 1D wave equation
│
├── scripts/
│   ├── train.py            # Main training script with CLI arguments
│   └── evaluate.py         # Model evaluation script
│
├── tests/                  # Unit tests
│
└── outputs/                # Directory for outputs (git-ignored)
    ├── models/             # Saved model checkpoints
    ├── figures/            # Generated plots and visualizations
    ├── logs/               # Training logs
    └── metrics/            # Training and evaluation metrics
```
