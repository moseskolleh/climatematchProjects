"""
Multi-Constraint Framework for Climate Sensitivity

This package implements a comprehensive framework for constraining equilibrium
climate sensitivity (ECS) using multiple lines of evidence: paleoclimate records,
historical observations, and process-based understanding.

Key modules:
- constraints: Individual constraint implementations
- uncertainty: Uncertainty quantification and Bayesian integration
- validation: Validation framework including perfect model tests
- models: Climate model interfaces and feedback decomposition
- data_processing: Data ingestion and quality control
- utils: Utility functions and helpers
"""

__version__ = "0.1.0"
__author__ = "Moses Kolleh"
__email__ = "moses@example.com"

from . import constraints
from . import uncertainty
from . import validation
from . import models
from . import data_processing
from . import utils

__all__ = [
    "constraints",
    "uncertainty",
    "validation",
    "models",
    "data_processing",
    "utils",
]
