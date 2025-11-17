"""
Adaptive Multi-Scale Drought Early Warning System

A comprehensive drought early warning system for Africa that combines
ensemble machine learning with local knowledge systems.
"""

__version__ = "0.1.0"
__author__ = "Moses - Digital Society School, Amsterdam University of Applied Sciences"

from . import data_ingestion
from . import data_processing
from . import models
from . import api
from . import utils

__all__ = [
    "data_ingestion",
    "data_processing",
    "models",
    "api",
    "utils",
]
