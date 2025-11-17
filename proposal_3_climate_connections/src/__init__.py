"""
Theory-Guided Discovery of Climate System Connections

This package provides tools for discovering and validating climate teleconnections
using theory-guided machine learning approaches.
"""

__version__ = "0.1.0"
__author__ = "Moses Kolleh"
__email__ = "moses@example.com"

from . import data_processing
from . import discovery
from . import validation
from . import visualization

__all__ = [
    "data_processing",
    "discovery",
    "validation",
    "visualization",
]
