"""
Multi-Constraint Integration Module

Bayesian framework for combining multiple independent constraints.
"""

from .multi_constraint import MultiConstraintFramework
from .independence_test import IndependenceTest

__all__ = ['MultiConstraintFramework', 'IndependenceTest']
