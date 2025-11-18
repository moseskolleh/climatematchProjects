"""
Uncertainty quantification and Bayesian integration.

This module provides tools for:
- Bayesian integration of multiple constraints
- Dependency modeling between constraints
- Uncertainty decomposition (aleatory vs epistemic)
- Sensitivity analysis
"""

from .bayesian_integration import BayesianIntegrator, MultiConstraintPosterior
from .dependency_modeling import ConstraintDependencyAnalyzer
from .uncertainty_decomposition import UncertaintyDecomposer

__all__ = [
    "BayesianIntegrator",
    "MultiConstraintPosterior",
    "ConstraintDependencyAnalyzer",
    "UncertaintyDecomposer",
]
