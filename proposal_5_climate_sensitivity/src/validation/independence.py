"""
Constraint Independence Testing

Tests whether constraints are truly independent.
"""

import numpy as np
from typing import List, Dict, Any
import xarray as xr
from scipy.stats import pearsonr
import logging

logger = logging.getLogger(__name__)


class ConstraintIndependenceTest:
    """
    Test independence between constraints.

    Constraints should be independent to justify multiplying their likelihoods
    in Bayesian combination. This class tests for independence.
    """

    def __init__(self):
        """Initialize independence test."""
        pass

    def assess(
        self,
        constraint_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Assess independence between constraints.

        Args:
            constraint_results: Results from multiple constraints

        Returns:
            Dictionary with independence metrics
        """
        logger.info("Assessing constraint independence")

        n_constraints = len(constraint_results)

        # Extract model weights from each constraint
        weights_matrix = []
        constraint_names = []

        for result in constraint_results:
            if 'weights' in result['result']:
                weights_matrix.append(result['result']['weights'])
                constraint_names.append(result['name'])

        weights_matrix = np.array(weights_matrix)

        # Calculate pairwise correlations
        n = len(weights_matrix)
        correlations = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    corr, _ = pearsonr(weights_matrix[i], weights_matrix[j])
                    correlations[i, j] = corr
                else:
                    correlations[i, j] = 1.0

        # Find highly correlated pairs (potential non-independence)
        threshold = 0.7
        high_corr_pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                if abs(correlations[i, j]) > threshold:
                    high_corr_pairs.append({
                        'constraint_1': constraint_names[i],
                        'constraint_2': constraint_names[j],
                        'correlation': float(correlations[i, j])
                    })

        # Calculate mean absolute correlation (excluding diagonal)
        off_diag = correlations[~np.eye(n, dtype=bool)]
        mean_abs_corr = float(np.mean(np.abs(off_diag))) if len(off_diag) > 0 else 0.0

        independence_assessment = {
            'correlation_matrix': correlations.tolist(),
            'constraint_names': constraint_names,
            'mean_absolute_correlation': mean_abs_corr,
            'high_correlation_pairs': high_corr_pairs,
            'independent': mean_abs_corr < 0.5,  # Arbitrary threshold
            'n_constraints': n
        }

        if high_corr_pairs:
            logger.warning(
                f"Found {len(high_corr_pairs)} pairs of highly correlated constraints. "
                f"Independence assumption may be violated."
            )
        else:
            logger.info("No high correlations found. Constraints appear independent.")

        return independence_assessment
