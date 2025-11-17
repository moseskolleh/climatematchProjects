"""
Physics Modules
===============

Differentiable physics-based components for hydrological modeling.

Modules:
--------
- infiltration: Green-Ampt infiltration model
- evapotranspiration: Penman-Monteith ET calculation
- routing: Kinematic wave routing for river networks
"""

from .infiltration import GreenAmptInfiltration
from .evapotranspiration import PenmanMonteithET
from .routing import KinematicWaveRouting

__all__ = [
    "GreenAmptInfiltration",
    "PenmanMonteithET",
    "KinematicWaveRouting",
]
