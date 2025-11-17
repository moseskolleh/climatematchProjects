"""
Hybrid Physics-ML Framework for Flood Prediction
=================================================

This package implements a hybrid framework combining simplified physics-based
hydrological models with machine learning for flood prediction in West African rivers.

Modules:
--------
- physics_modules: Differentiable physics components (infiltration, routing, ET)
- ml_modules: Machine learning components (downscaling, parameter estimation, correction)
- hybrid_models: Integrated hybrid models combining physics and ML
- data_processing: Data pipelines and preprocessing
- validation: Validation frameworks and metrics
"""

__version__ = "0.1.0"
__author__ = "Moses - Digital Society School, Amsterdam University of Applied Sciences"
__email__ = "moses@example.com"

from . import physics_modules
from . import ml_modules
from . import hybrid_models
from . import data_processing
from . import validation

__all__ = [
    "physics_modules",
    "ml_modules",
    "hybrid_models",
    "data_processing",
    "validation",
]
