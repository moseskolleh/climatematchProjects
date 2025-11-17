"""
Multi-Constraint Framework for Climate Sensitivity

This package provides tools for constraining equilibrium climate sensitivity (ECS)
using multiple lines of evidence from paleoclimate, observations, and process understanding.
"""

__version__ = "0.1.0"
__author__ = "Moses"
__email__ = "moses@digitalsocietyschool.nl"

from . import constraints
from . import validation
from . import uncertainty
from . import data_processing
from . import utils

__all__ = [
    'constraints',
    'validation',
    'uncertainty',
    'data_processing',
    'utils'
]
