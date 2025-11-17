"""
Integrated Coastal Risk Framework for African Cities

This package provides tools for assessing compound coastal hazards and
vulnerability under deep uncertainty.

Main modules:
- bayesian_network: Compound hazard dependency modeling
- agent_based_model: Household-level vulnerability simulation
- data_processing: Data ingestion and quality control
- uncertainty: Deep uncertainty and robust decision making
- validation: Model validation and performance metrics
"""

__version__ = "0.1.0"
__author__ = "Moses"
__email__ = "moses@example.com"

from . import bayesian_network
from . import agent_based_model
from . import data_processing
from . import uncertainty
from . import validation
from . import utils

__all__ = [
    'bayesian_network',
    'agent_based_model',
    'data_processing',
    'uncertainty',
    'validation',
    'utils'
]
