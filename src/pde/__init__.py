"""
PDE operators and solvers for Physics Informed Neural Networks.
"""

from src.pde.derivatives import df_dt, df_dx, df_single, gradient
from src.pde.operators import LogisticOperator, WaveOperator
from src.pde.equations import LogisticEquation, WaveEquation, create_equation