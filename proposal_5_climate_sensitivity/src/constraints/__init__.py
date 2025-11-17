"""
Constraints Module

Implements various constraints on climate sensitivity from paleoclimate,
observations, and process understanding.
"""

from .base_constraint import BaseConstraint
from .multi_constraint_framework import MultiConstraintFramework

# Import specific constraint types
from .paleoclimate import LGMConstraint, MPWPConstraint
from .observational import HistoricalWarmingConstraint, EnergyBudgetConstraint
from .process_based import CloudFeedbackConstraint, WaterVaporConstraint

__all__ = [
    'BaseConstraint',
    'MultiConstraintFramework',
    'LGMConstraint',
    'MPWPConstraint',
    'HistoricalWarmingConstraint',
    'EnergyBudgetConstraint',
    'CloudFeedbackConstraint',
    'WaterVaporConstraint'
]
