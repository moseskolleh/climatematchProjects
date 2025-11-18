"""
Perfect Model Testing

Validate constraint methods using climate models as truth.
"""

import numpy as np
from typing import Dict, List


class PerfectModelTest:
    """
    Perfect model framework for validating constraints

    For each model in ensemble:
    1. Treat as "truth"
    2. Apply constraint method using other models
    3. Compare constrained estimate to true model ECS
    4. Calculate performance metrics
    """

    def __init__(self, model_ensemble: List[str] = None):
        self.model_ensemble = model_ensemble or []
        self.results = []

    def validate_constraint(
        self,
        constraint_method,
        model_ecs_values: np.ndarray,
        model_names: List[str]
    ) -> Dict:
        """
        Perform leave-one-out validation

        Parameters:
        -----------
        constraint_method : callable
            Method to apply constraint
        model_ecs_values : np.ndarray
            True ECS for each model
        model_names : list
            Model names

        Returns:
        --------
        validation : dict
            Validation statistics
        """
        results = []

        for i, (true_ecs, model_name) in enumerate(zip(model_ecs_values, model_names)):
            # Training set (all models except i)
            train_indices = np.delete(np.arange(len(model_ecs_values)), i)
            train_ecs = model_ecs_values[train_indices]

            # Apply constraint method (simplified)
            # In practice, would use actual model data
            constrained_dist = self._apply_constraint(constraint_method, train_ecs)

            # Metrics
            constrained_median = np.median(constrained_dist)
            constrained_ci = np.percentile(constrained_dist, [5, 95])

            bias = constrained_median - true_ecs
            coverage = (true_ecs >= constrained_ci[0]) and (true_ecs <= constrained_ci[1])
            sharpness = constrained_ci[1] - constrained_ci[0]

            results.append({
                'model': model_name,
                'true_ecs': true_ecs,
                'constrained_median': constrained_median,
                'constrained_ci': constrained_ci,
                'bias': bias,
                'coverage': coverage,
                'sharpness': sharpness
            })

        # Aggregate performance
        validation = {
            'mean_bias': np.mean([r['bias'] for r in results]),
            'rmse': np.sqrt(np.mean([r['bias']**2 for r in results])),
            'coverage_rate': np.mean([r['coverage'] for r in results]),
            'mean_sharpness': np.mean([r['sharpness'] for r in results]),
            'individual_results': results
        }

        return validation

    def _apply_constraint(self, method, train_ecs):
        """Helper to apply constraint (simplified)"""
        # In real implementation, would use actual constraint method
        # For now, return samples around training mean
        mean_ecs = np.mean(train_ecs)
        std_ecs = np.std(train_ecs)
        return np.random.normal(mean_ecs, std_ecs, 10000)

    def calculate_skill_scores(self, validation_results: Dict) -> Dict:
        """
        Calculate skill scores for constraint method

        Parameters:
        -----------
        validation_results : dict
            Results from validate_constraint

        Returns:
        --------
        skill : dict
            Skill scores
        """
        # Coverage skill (should be ~0.90 for 90% CI)
        coverage_rate = validation_results['coverage_rate']
        coverage_skill = 1 - abs(coverage_rate - 0.90) / 0.90

        # Bias skill (lower is better)
        mean_bias = abs(validation_results['mean_bias'])
        bias_skill = max(0, 1 - mean_bias / 1.0)  # Normalize by 1 K

        # RMSE skill (lower is better)
        rmse = validation_results['rmse']
        rmse_skill = max(0, 1 - rmse / 2.0)  # Normalize by 2 K

        # Sharpness skill (narrower CI is better, but not at expense of coverage)
        sharpness = validation_results['mean_sharpness']
        sharpness_skill = max(0, 1 - sharpness / 4.0)  # Normalize by 4 K

        # Overall skill (weighted combination)
        overall_skill = (
            coverage_skill * 0.4 +
            bias_skill * 0.3 +
            rmse_skill * 0.2 +
            sharpness_skill * 0.1
        )

        skill = {
            'coverage_skill': coverage_skill,
            'bias_skill': bias_skill,
            'rmse_skill': rmse_skill,
            'sharpness_skill': sharpness_skill,
            'overall_skill': overall_skill
        }

        return skill

    def __repr__(self) -> str:
        return f"PerfectModelTest(n_models={len(self.model_ensemble)})"
