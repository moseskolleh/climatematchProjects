"""
Multi-Constraint Framework for Climate Sensitivity

This package provides tools for estimating Equilibrium Climate Sensitivity (ECS)
using multiple independent constraints from paleoclimate, observational, and
process-based evidence.

Modules:
--------
- paleoclimate: Constraints from past climate periods (LGM, mPWP, LIG)
- observational: Constraints from historical observations
- process_based: Constraints from physical understanding of feedbacks
- integration: Multi-constraint Bayesian framework
- uncertainty: Uncertainty quantification methods
- validation: Perfect model tests and cross-validation
- utils: Utility functions

Example Usage:
--------------
>>> from src.paleoclimate import LGMConstraint
>>> from src.observational import HistoricalWarmingConstraint
>>> from src.integration import MultiConstraintFramework
>>>
>>> # Initialize constraints
>>> lgm = LGMConstraint()
>>> hist = HistoricalWarmingConstraint()
>>>
>>> # Combine constraints
>>> mcf = MultiConstraintFramework([lgm, hist])
>>> ecs_distribution = mcf.integrate_constraints()
>>> print(f"ECS median: {ecs_distribution.median():.2f} K")
"""

__version__ = "1.0.0"
__author__ = "Moses - Environmental Scientist & Sustainable AI Researcher"
__email__ = "moses@climatematch.io"

from . import paleoclimate
from . import observational
from . import process_based
from . import integration
from . import uncertainty
from . import validation
from . import utils

__all__ = [
    'paleoclimate',
    'observational',
    'process_based',
    'integration',
    'uncertainty',
    'validation',
    'utils'
]
