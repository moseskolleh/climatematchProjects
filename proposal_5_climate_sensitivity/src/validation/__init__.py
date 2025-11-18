"""
Validation framework for climate sensitivity constraints.

This module provides tools for:
- Perfect model experiments
- Cross-validation across model generations
- Assessment of constraint independence
- Validation metrics and diagnostics
"""

from .perfect_model import PerfectModelValidator
from .cross_validation import CrossGenerationValidator
from .constraint_independence import ConstraintIndependenceAnalyzer

__all__ = [
    "PerfectModelValidator",
    "CrossGenerationValidator",
    "ConstraintIndependenceAnalyzer",
]
