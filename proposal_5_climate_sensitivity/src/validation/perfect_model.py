"""
Perfect Model Testing

Tests constraint methodology by treating one model as "truth".
"""

import numpy as np
import xarray as xr
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PerfectModelTest:
    """
    Perfect model test for validating constraint methodology.

    In a perfect model test, we pretend one CMIP model is the "true" climate
    and see if our constraint methodology can recover its properties (ECS)
    when using pseudo-observations generated from that model.
    """

    def __init__(self):
        """Initialize perfect model test."""
        self.results = {}

    def run(
        self,
        constraints: List,
        cmip_data: xr.Dataset,
        test_models: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run perfect model test.

        Args:
            constraints: List of constraint objects to test
            cmip_data: CMIP model ensemble
            test_models: Models to use as "truth" (None = all models)

        Returns:
            Dictionary with test results
        """
        logger.info("Running perfect model tests")

        if test_models is None:
            test_models = cmip_data.model.values

        results = {}

        for test_model in test_models:
            logger.info(f"Testing with {test_model} as truth")

            # Get "true" ECS for this model
            if hasattr(cmip_data.sel(model=test_model), 'attrs'):
                true_ecs = cmip_data.sel(model=test_model).attrs.get('true_ecs', 3.0)
            else:
                true_ecs = 3.0

            # Create pseudo-observations from this model
            pseudo_obs = self._create_pseudo_observations(
                cmip_data.sel(model=test_model)
            )

            # Apply constraints using all OTHER models
            other_models = [m for m in cmip_data.model.values if m != test_model]
            reduced_data = cmip_data.sel(model=other_models)

            constraint_results = []
            for constraint in constraints:
                try:
                    result = constraint.apply(reduced_data, pseudo_obs)

                    # Get constrained ECS distribution
                    ecs_samples = result['ecs_constrained']

                    # Check if true value is within credible interval
                    ecs_5 = np.percentile(ecs_samples, 5)
                    ecs_95 = np.percentile(ecs_samples, 95)
                    within_ci = (ecs_5 <= true_ecs <= ecs_95)

                    # Calculate bias and RMSE
                    bias = np.mean(ecs_samples) - true_ecs
                    rmse = np.sqrt(np.mean((ecs_samples - true_ecs) ** 2))

                    constraint_results.append({
                        'constraint_name': constraint.name,
                        'true_ecs': true_ecs,
                        'estimated_ecs_mean': np.mean(ecs_samples),
                        'estimated_ecs_std': np.std(ecs_samples),
                        'within_90_ci': within_ci,
                        'bias': bias,
                        'rmse': rmse
                    })

                except Exception as e:
                    logger.error(f"Error testing {constraint.name}: {e}")

            results[str(test_model)] = constraint_results

        # Calculate summary statistics
        summary = self._calculate_summary_statistics(results)

        return {
            'individual_results': results,
            'summary': summary
        }

    def _create_pseudo_observations(self, model_data: xr.Dataset) -> Dict[str, Any]:
        """
        Create pseudo-observations from a model.

        Extracts relevant quantities from the model that would be
        used as constraints.
        """
        pseudo_obs = {}

        # Temperature-related pseudo-observations
        if 'tas' in model_data.data_vars:
            # Global mean temperature
            pseudo_obs['global_temp'] = float(
                model_data['tas'].mean(dim=['lat', 'lon', 'time'])
            )

        # Add noise to simulate observational uncertainty
        # (This makes the test more realistic)
        for key in pseudo_obs:
            pseudo_obs[key] += np.random.normal(0, 0.1)

        return pseudo_obs

    def _calculate_summary_statistics(
        self,
        results: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """Calculate summary statistics across all perfect model tests."""
        all_biases = []
        all_rmses = []
        all_within_ci = []

        for model_results in results.values():
            for constraint_result in model_results:
                all_biases.append(constraint_result['bias'])
                all_rmses.append(constraint_result['rmse'])
                all_within_ci.append(constraint_result['within_90_ci'])

        summary = {
            'mean_bias': float(np.mean(all_biases)),
            'mean_rmse': float(np.mean(all_rmses)),
            'fraction_within_90_ci': float(np.mean(all_within_ci)),
            'n_tests': len(all_biases)
        }

        logger.info(f"Perfect model test summary: bias={summary['mean_bias']:.2f}, "
                   f"RMSE={summary['mean_rmse']:.2f}, "
                   f"fraction in CI={summary['fraction_within_90_ci']:.2f}")

        return summary
