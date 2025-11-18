"""
Observational Constraints Module

This module provides tools for constraining climate sensitivity using
historical observational data.

Classes:
--------
- HistoricalWarmingConstraint: Energy budget constraint from historical warming
- PatternConstraint: Constraint from warming patterns
"""

from .historical_warming import HistoricalWarmingConstraint
from .pattern_constraint import PatternConstraint

__all__ = [
    'HistoricalWarmingConstraint',
    'PatternConstraint'
]
