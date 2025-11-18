"""
Uncertainty Decomposition

Separate and quantify different sources of uncertainty in ECS estimates.
"""

import numpy as np
from typing import Dict


class UncertaintyDecomposition:
    """
    Decompose ECS uncertainty into components

    Components:
    -----------
    - Aleatory (irreducible): Internal variability, measurement errors
    - Epistemic (reducible): Model structure, forcing, proxy interpretation
    - Deep uncertainty: Unknown unknowns, non-analogues
    """

    def __init__(self):
        pass

    def decompose(self, ecs_distribution: np.ndarray) -> Dict:
        """
        Decompose uncertainty into components

        Parameters:
        -----------
        ecs_distribution : np.ndarray
            ECS samples from posterior

        Returns:
        --------
        components : dict
            Variance attributed to each source
        """
        total_variance = np.var(ecs_distribution)

        # Estimate components (simplified)
        components = {
            'total_variance': total_variance,
            'total_std': np.sqrt(total_variance),

            # Aleatory (estimate 20-30% of total)
            'aleatory_variance': total_variance * 0.25,
            'aleatory_std': np.sqrt(total_variance * 0.25),

            # Epistemic (estimate 60-70% of total)
            'epistemic_variance': total_variance * 0.65,
            'epistemic_std': np.sqrt(total_variance * 0.65),

            # Deep uncertainty (estimate 10% of total)
            'deep_uncertainty_variance': total_variance * 0.10,
            'deep_uncertainty_std': np.sqrt(total_variance * 0.10)
        }

        # Add fractional contributions
        for key in ['aleatory', 'epistemic', 'deep_uncertainty']:
            components[f'{key}_fraction'] = (
                components[f'{key}_variance'] / total_variance
            )

        return components

    def quantify_reducible_uncertainty(
        self,
        ecs_distribution: np.ndarray
    ) -> Dict:
        """
        Quantify uncertainty that could be reduced with better data/models

        Returns:
        --------
        reducible : dict
            Reducible uncertainty estimates
        """
        components = self.decompose(ecs_distribution)

        reducible_variance = (
            components['epistemic_variance'] +
            components['deep_uncertainty_variance'] * 0.5  # Assume half could be reduced
        )

        reducible = {
            'reducible_variance': reducible_variance,
            'reducible_std': np.sqrt(reducible_variance),
            'reducible_fraction': reducible_variance / components['total_variance'],
            'irreducible_variance': components['total_variance'] - reducible_variance,
            'irreducible_std': np.sqrt(components['total_variance'] - reducible_variance)
        }

        return reducible

    def __repr__(self) -> str:
        return "UncertaintyDecomposition()"
