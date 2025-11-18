"""
Constraint implementations for climate sensitivity estimation.

This module provides implementations of different constraint types:
- Paleoclimate constraints (LGM, Pliocene, Last Interglacial)
- Observational constraints (historical temperature, energy budget)
- Process-based constraints (cloud feedbacks, regional patterns)
"""

from .paleoclimate import PaleoclimateConstraint, LGMConstraint, PlioceneConstraint
from .observational import ObservationalConstraint, EnergyBudgetConstraint, EmergentConstraint
from .process_based import ProcessBasedConstraint, CloudFeedbackConstraint

__all__ = [
    "PaleoclimateConstraint",
    "LGMConstraint",
    "PlioceneConstraint",
    "ObservationalConstraint",
    "EnergyBudgetConstraint",
    "EmergentConstraint",
    "ProcessBasedConstraint",
    "CloudFeedbackConstraint",
]
