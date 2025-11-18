"""
Paleoclimate Constraints Module

This module provides tools for constraining climate sensitivity using
paleoclimate evidence from various past climate periods.

Classes:
--------
- LGMConstraint: Last Glacial Maximum constraint
- mPWPConstraint: Mid-Pliocene Warm Period constraint
- LIGConstraint: Last Interglacial constraint
- PETMConstraint: Paleocene-Eocene Thermal Maximum constraint
"""

from .lgm_constraint import LGMConstraint
from .mpwp_constraint import mPWPConstraint
from .lig_constraint import LIGConstraint

__all__ = [
    'LGMConstraint',
    'mPWPConstraint',
    'LIGConstraint'
]
