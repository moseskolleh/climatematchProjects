"""
Discovery module for identifying climate teleconnections using theory-guided methods.
"""

from .causal_discovery import (
    CausalDiscoveryEngine,
    GrangerCausality,
    ConvergentCrossMapping,
    TransferEntropy,
    StructuralCausalModel
)

from .physical_constraints import (
    PhysicalConstraintEngine,
    RossbyWaveValidator,
    EnergyConservationValidator,
    TimescaleValidator
)

from .discovery_engine import TeleconnectionDiscoveryEngine

__all__ = [
    "CausalDiscoveryEngine",
    "GrangerCausality",
    "ConvergentCrossMapping",
    "TransferEntropy",
    "StructuralCausalModel",
    "PhysicalConstraintEngine",
    "RossbyWaveValidator",
    "EnergyConservationValidator",
    "TimescaleValidator",
    "TeleconnectionDiscoveryEngine",
]
