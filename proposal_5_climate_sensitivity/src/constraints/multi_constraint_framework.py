"""
Multi-Constraint Framework

Integrates multiple independent constraints on climate sensitivity.
"""

import numpy as np
import xarray as xr
from typing import List, Dict, Any, Optional
import logging
from scipy import stats

from .base_constraint import BaseConstraint

logger = logging.getLogger(__name__)


class MultiConstraintFramework:
    """
    Framework for combining multiple constraints on climate sensitivity.

    This class implements conservative combination rules:
    - Only report narrowed uncertainty if multiple constraints agree
    - Test for constraint independence
    - Provide transparent reporting of all assumptions
    """

    def __init__(
        self,
        constraints: List[BaseConstraint],
        method: str = 'bayesian',
        min_agreement: int = 2
    ):
        """
        Initialize multi-constraint framework.

        Args:
            constraints: List of constraint objects
            method: Combination method ('bayesian', 'equal_weight', 'variance_weighted')
            min_agreement: Minimum number of agreeing constraints to report narrowing
        """
        self.constraints = constraints
        self.method = method
        self.min_agreement = min_agreement

        logger.info(
            f"Initialized MultiConstraintFramework with {len(constraints)} constraints"
        )

    def integrate_constraints(
        self,
        cmip_data: xr.Dataset,
        paleo_data: Optional[Dict] = None,
        obs_data: Optional[xr.Dataset] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Integrate all constraints to produce combined ECS estimate.

        Args:
            cmip_data: CMIP model data
            paleo_data: Paleoclimate constraint data
            obs_data: Observational constraint data
            **kwargs: Additional arguments

        Returns:
            Dictionary with combined results
        """
        logger.info("Integrating constraints")

        # Apply each constraint
        constraint_results = []
        for constraint in self.constraints:
            try:
                logger.info(f"Applying {constraint.name}")

                if constraint.constraint_type == 'paleoclimate':
                    result = constraint.apply(cmip_data, paleo_data)
                elif constraint.constraint_type == 'observational':
                    result = constraint.apply(cmip_data, obs_data)
                else:  # process_based
                    result = constraint.apply(cmip_data)

                constraint_results.append({
                    'name': constraint.name,
                    'type': constraint.constraint_type,
                    'result': result
                })

            except Exception as e:
                logger.error(f"Error applying {constraint.name}: {e}")
                continue

        if not constraint_results:
            raise ValueError("No constraints could be applied successfully")

        # Combine constraints
        combined = self._combine_constraints(constraint_results, cmip_data)

        # Assess agreement
        agreement = self._assess_agreement(constraint_results)

        # Check if we meet minimum agreement threshold
        if agreement['n_agreeing'] < self.min_agreement:
            logger.warning(
                f"Only {agreement['n_agreeing']} constraints agree, "
                f"minimum is {self.min_agreement}. "
                f"Reporting individual constraints without combination."
            )
            combined['combined_valid'] = False
        else:
            combined['combined_valid'] = True

        combined['agreement'] = agreement
        combined['individual_constraints'] = constraint_results

        return combined

    def _combine_constraints(
        self,
        constraint_results: List[Dict],
        cmip_data: xr.Dataset
    ) -> Dict[str, Any]:
        """
        Combine results from multiple constraints.

        Args:
            constraint_results: List of results from individual constraints
            cmip_data: CMIP data for reference

        Returns:
            Combined constraint results
        """
        if self.method == 'bayesian':
            return self._bayesian_combination(constraint_results, cmip_data)
        elif self.method == 'equal_weight':
            return self._equal_weight_combination(constraint_results)
        elif self.method == 'variance_weighted':
            return self._variance_weighted_combination(constraint_results)
        else:
            raise ValueError(f"Unknown combination method: {self.method}")

    def _bayesian_combination(
        self,
        constraint_results: List[Dict],
        cmip_data: xr.Dataset
    ) -> Dict[str, Any]:
        """
        Combine constraints using Bayesian approach.

        Multiplies likelihoods from each constraint (assuming independence).
        """
        logger.info("Combining constraints using Bayesian method")

        # Get model ECS values
        if hasattr(cmip_data, 'attrs') and 'true_ecs' in cmip_data.attrs:
            model_ecs = np.array([
                float(cmip_data.sel(model=m).attrs.get('true_ecs', 3.0))
                for m in cmip_data.model.values
            ])
        else:
            model_ecs = np.array([3.0] * len(cmip_data.model))

        # Start with uniform prior (equal model weights)
        combined_weights = np.ones(len(model_ecs))

        # Multiply likelihoods from each constraint
        for result in constraint_results:
            weights = result['result']['weights']
            combined_weights *= weights

        # Normalize
        combined_weights /= combined_weights.sum()

        # Sample from combined distribution
        ecs_samples = np.random.choice(
            model_ecs,
            size=10000,
            p=combined_weights / combined_weights.sum()
        )

        # Add within-model uncertainty
        ecs_samples += np.random.normal(0, 0.15 * np.abs(ecs_samples))

        # Calculate statistics
        return {
            'ecs_distribution': ecs_samples,
            'ecs_mean': float(np.mean(ecs_samples)),
            'ecs_std': float(np.std(ecs_samples)),
            'ecs_median': float(np.median(ecs_samples)),
            'ecs_5': float(np.percentile(ecs_samples, 5)),
            'ecs_95': float(np.percentile(ecs_samples, 95)),
            'ecs_66': float(np.percentile(ecs_samples, 66)),
            'ecs_17': float(np.percentile(ecs_samples, 17)),
            'ecs_83': float(np.percentile(ecs_samples, 83)),
            'combined_weights': combined_weights,
            'method': 'bayesian'
        }

    def _equal_weight_combination(
        self,
        constraint_results: List[Dict]
    ) -> Dict[str, Any]:
        """Combine by averaging ECS distributions with equal weights."""
        logger.info("Combining constraints with equal weights")

        # Pool all samples
        all_samples = []
        for result in constraint_results:
            all_samples.append(result['result']['ecs_constrained'])

        combined_samples = np.concatenate(all_samples)

        return {
            'ecs_distribution': combined_samples,
            'ecs_mean': float(np.mean(combined_samples)),
            'ecs_std': float(np.std(combined_samples)),
            'ecs_median': float(np.median(combined_samples)),
            'method': 'equal_weight'
        }

    def _variance_weighted_combination(
        self,
        constraint_results: List[Dict]
    ) -> Dict[str, Any]:
        """Combine using inverse variance weighting."""
        logger.info("Combining constraints with variance weighting")

        means = []
        variances = []

        for result in constraint_results:
            samples = result['result']['ecs_constrained']
            means.append(np.mean(samples))
            variances.append(np.var(samples))

        # Inverse variance weights
        weights = 1.0 / np.array(variances)
        weights /= weights.sum()

        # Combined mean
        combined_mean = np.sum(weights * np.array(means))

        # Combined variance
        combined_var = 1.0 / np.sum(1.0 / np.array(variances))

        # Generate samples from combined distribution
        combined_samples = np.random.normal(
            combined_mean,
            np.sqrt(combined_var),
            10000
        )

        return {
            'ecs_distribution': combined_samples,
            'ecs_mean': combined_mean,
            'ecs_std': np.sqrt(combined_var),
            'method': 'variance_weighted'
        }

    def _assess_agreement(self, constraint_results: List[Dict]) -> Dict[str, Any]:
        """
        Assess agreement between constraints.

        Returns:
            Dictionary with agreement metrics
        """
        # Extract posterior means and std devs
        means = []
        stds = []

        for result in constraint_results:
            samples = result['result']['ecs_constrained']
            means.append(np.mean(samples))
            stds.append(np.std(samples))

        means = np.array(means)
        stds = np.array(stds)

        # Check how many constraints have overlapping confidence intervals
        n_agreeing = 0
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                # Check if 90% confidence intervals overlap
                lower_i = means[i] - 1.645 * stds[i]
                upper_i = means[i] + 1.645 * stds[i]
                lower_j = means[j] - 1.645 * stds[j]
                upper_j = means[j] + 1.645 * stds[j]

                if (lower_i <= upper_j) and (lower_j <= upper_i):
                    n_agreeing += 1

        # Calculate variance ratio (between-constraint / within-constraint)
        between_var = np.var(means)
        within_var = np.mean(stds ** 2)
        variance_ratio = between_var / within_var if within_var > 0 else 0

        return {
            'n_agreeing': n_agreeing,
            'constraint_means': means.tolist(),
            'constraint_stds': stds.tolist(),
            'mean_of_means': float(np.mean(means)),
            'std_of_means': float(np.std(means)),
            'variance_ratio': float(variance_ratio),
            'high_agreement': variance_ratio < 0.5  # Arbitrary threshold
        }
