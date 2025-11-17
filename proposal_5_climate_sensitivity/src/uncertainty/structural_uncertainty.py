"""
Structural Uncertainty Analysis

Analyzes and decomposes structural uncertainties in ECS estimates.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class StructuralUncertaintyAnalysis:
    """
    Analyze structural uncertainties in climate sensitivity estimates.

    Structural uncertainty arises from:
    - Model structural differences
    - Different constraint implementations
    - Methodological choices
    """

    def __init__(self):
        """Initialize structural uncertainty analysis."""
        pass

    def decompose(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Decompose total uncertainty into components.

        Args:
            results: Results from multi-constraint framework

        Returns:
            Dictionary with uncertainty decomposition
        """
        logger.info("Decomposing structural uncertainty")

        individual_constraints = results.get('individual_constraints', [])

        if not individual_constraints:
            logger.warning("No individual constraint results to analyze")
            return {}

        # Extract posterior distributions
        constraint_means = []
        constraint_stds = []

        for result in individual_constraints:
            samples = result['result']['ecs_constrained']
            constraint_means.append(np.mean(samples))
            constraint_stds.append(np.std(samples))

        constraint_means = np.array(constraint_means)
        constraint_stds = np.array(constraint_stds)

        # Within-constraint variance (average of individual variances)
        within_variance = np.mean(constraint_stds ** 2)

        # Between-constraint variance (variance of means)
        between_variance = np.var(constraint_means)

        # Total variance
        total_variance = within_variance + between_variance

        # Fraction of uncertainty from each source
        within_fraction = within_variance / total_variance if total_variance > 0 else 0
        between_fraction = between_variance / total_variance if total_variance > 0 else 0

        decomposition = {
            'within_constraint_variance': float(within_variance),
            'between_constraint_variance': float(between_variance),
            'total_variance': float(total_variance),
            'within_constraint_fraction': float(within_fraction),
            'between_constraint_fraction': float(between_fraction),
            'constraint_means': constraint_means.tolist(),
            'constraint_stds': constraint_stds.tolist()
        }

        logger.info(
            f"Uncertainty decomposition: {within_fraction*100:.1f}% within-constraint, "
            f"{between_fraction*100:.1f}% between-constraint"
        )

        return decomposition
