"""
Perfect model validation framework.

Tests constraint methodology by:
1. Using one climate model as "truth"
2. Applying constraints using other models
3. Checking if true ECS is recovered within uncertainty bounds
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PerfectModelResult:
    """Results from perfect model validation experiment."""
    true_ecs: float
    estimated_ecs_mean: float
    estimated_ecs_std: float
    estimated_ecs_samples: np.ndarray
    coverage: bool
    bias: float
    percentile_rank: float
    model_name: str
    metadata: Dict


class PerfectModelValidator:
    """
    Perfect model validation for climate sensitivity constraints.

    This validator tests whether constraint methods can recover known
    values of ECS when one model is treated as "truth" and others are
    used to build the constraint.
    """

    def __init__(self, confidence_level: float = 0.90):
        """
        Initialize perfect model validator.

        Args:
            confidence_level: Confidence level for coverage test (default: 90%)
        """
        self.confidence_level = confidence_level

    def validate_constraint(
        self,
        model_ensemble: Dict[str, float],
        constraint_function: callable,
        constraint_data: Dict[str, any],
        leave_out_models: Optional[List[str]] = None
    ) -> List[PerfectModelResult]:
        """
        Perform leave-one-out perfect model validation.

        Args:
            model_ensemble: Dict of {model_name: true_ecs}
            constraint_function: Function that applies constraint and returns samples
            constraint_data: Additional data needed by constraint function
            leave_out_models: Specific models to test (default: all)

        Returns:
            List of PerfectModelResult for each tested model
        """
        if leave_out_models is None:
            leave_out_models = list(model_ensemble.keys())

        results = []

        for test_model in leave_out_models:
            # True ECS for test model
            true_ecs = model_ensemble[test_model]

            # Training ensemble (all models except test model)
            training_ensemble = {
                k: v for k, v in model_ensemble.items() if k != test_model
            }

            # Apply constraint using training ensemble
            estimated_samples = constraint_function(
                training_ensemble, constraint_data
            )

            # Calculate statistics
            estimated_mean = np.mean(estimated_samples)
            estimated_std = np.std(estimated_samples)

            # Check coverage: is true value within confidence interval?
            alpha = 1 - self.confidence_level
            lower = np.percentile(estimated_samples, 100 * alpha / 2)
            upper = np.percentile(estimated_samples, 100 * (1 - alpha / 2))
            coverage = (true_ecs >= lower) and (true_ecs <= upper)

            # Calculate bias
            bias = estimated_mean - true_ecs

            # Percentile rank of true value in estimated distribution
            percentile_rank = np.mean(estimated_samples <= true_ecs) * 100

            results.append(PerfectModelResult(
                true_ecs=true_ecs,
                estimated_ecs_mean=estimated_mean,
                estimated_ecs_std=estimated_std,
                estimated_ecs_samples=estimated_samples,
                coverage=coverage,
                bias=bias,
                percentile_rank=percentile_rank,
                model_name=test_model,
                metadata={
                    "confidence_level": self.confidence_level,
                    "n_training_models": len(training_ensemble)
                }
            ))

        return results

    def calculate_validation_metrics(
        self,
        results: List[PerfectModelResult]
    ) -> Dict[str, float]:
        """
        Calculate summary validation metrics.

        Args:
            results: List of PerfectModelResult from validation

        Returns:
            Dict of validation metrics
        """
        n_models = len(results)

        # Coverage rate: fraction of times true value is within CI
        coverage_rate = np.mean([r.coverage for r in results])

        # Mean bias
        mean_bias = np.mean([r.bias for r in results])

        # RMSE
        rmse = np.sqrt(np.mean([r.bias**2 for r in results]))

        # Mean absolute error
        mae = np.mean([abs(r.bias) for r in results])

        # Calibration: are percentile ranks uniformly distributed?
        # Use Kolmogorov-Smirnov test against uniform distribution
        percentile_ranks = np.array([r.percentile_rank for r in results])
        ks_statistic = self._kolmogorov_smirnov_uniform(percentile_ranks / 100)

        # Sharpness: average uncertainty width
        uncertainty_widths = [r.estimated_ecs_std * 2 for r in results]
        mean_sharpness = np.mean(uncertainty_widths)

        return {
            "n_models": n_models,
            "coverage_rate": coverage_rate,
            "target_coverage": self.confidence_level,
            "mean_bias": mean_bias,
            "rmse": rmse,
            "mae": mae,
            "ks_statistic": ks_statistic,
            "mean_sharpness": mean_sharpness,
            "calibrated": (coverage_rate >= self.confidence_level - 0.1) and (ks_statistic < 0.3)
        }

    def _kolmogorov_smirnov_uniform(self, data: np.ndarray) -> float:
        """
        Calculate KS statistic for test against uniform [0,1] distribution.

        Args:
            data: Array of values in [0,1]

        Returns:
            KS statistic
        """
        data_sorted = np.sort(data)
        n = len(data)

        # Empirical CDF
        ecdf = np.arange(1, n + 1) / n

        # Theoretical CDF (uniform)
        theoretical = data_sorted

        # KS statistic: maximum absolute difference
        ks_stat = np.max(np.abs(ecdf - theoretical))

        return ks_stat

    def plot_validation_results(
        self,
        results: List[PerfectModelResult],
        save_path: Optional[str] = None
    ):
        """
        Plot validation results (requires matplotlib).

        Args:
            results: List of validation results
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: True vs Estimated ECS
        ax = axes[0, 0]
        true_ecs = [r.true_ecs for r in results]
        est_ecs = [r.estimated_ecs_mean for r in results]
        est_std = [r.estimated_ecs_std for r in results]

        ax.errorbar(true_ecs, est_ecs, yerr=est_std, fmt='o', capsize=5)
        ax.plot([2, 6], [2, 6], 'k--', label='Perfect agreement')
        ax.set_xlabel('True ECS (K)')
        ax.set_ylabel('Estimated ECS (K)')
        ax.set_title('Perfect Model Validation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Coverage indicators
        ax = axes[0, 1]
        model_names = [r.model_name[:15] for r in results]  # Truncate names
        colors = ['green' if r.coverage else 'red' for r in results]
        y_pos = np.arange(len(results))
        ax.barh(y_pos, [r.estimated_ecs_mean for r in results], color=colors, alpha=0.6)
        ax.scatter(true_ecs, y_pos, color='black', marker='x', s=100, label='True ECS')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names)
        ax.set_xlabel('ECS (K)')
        ax.set_title('Coverage: Green=Within CI, Red=Outside CI')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        # Plot 3: Bias distribution
        ax = axes[1, 0]
        biases = [r.bias for r in results]
        ax.hist(biases, bins=10, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', label='Zero bias')
        ax.axvline(np.mean(biases), color='blue', linestyle='--', label=f'Mean bias: {np.mean(biases):.2f}')
        ax.set_xlabel('Bias (Estimated - True) (K)')
        ax.set_ylabel('Frequency')
        ax.set_title('Bias Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Percentile ranks (should be uniform)
        ax = axes[1, 1]
        percentiles = [r.percentile_rank for r in results]
        ax.hist(percentiles, bins=np.linspace(0, 100, 11), edgecolor='black', alpha=0.7)
        expected_count = len(results) / 10
        ax.axhline(expected_count, color='red', linestyle='--', label='Expected (uniform)')
        ax.set_xlabel('Percentile Rank of True Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Calibration Check (should be uniform)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


def run_synthetic_validation(
    n_models: int = 20,
    true_ecs_range: Tuple[float, float] = (2.0, 5.0),
    constraint_noise: float = 0.5
) -> Tuple[List[PerfectModelResult], Dict[str, float]]:
    """
    Run synthetic validation experiment with known ground truth.

    Args:
        n_models: Number of synthetic models
        true_ecs_range: Range of true ECS values
        constraint_noise: Noise level in constraint

    Returns:
        Tuple of (validation results, metrics)
    """
    # Generate synthetic model ensemble
    true_ecs_values = np.linspace(true_ecs_range[0], true_ecs_range[1], n_models)
    model_ensemble = {f"Model_{i:02d}": ecs for i, ecs in enumerate(true_ecs_values)}

    # Define a simple constraint function that adds noise
    def synthetic_constraint(training_ensemble, constraint_data):
        # Use mean of training ensemble with added noise
        training_mean = np.mean(list(training_ensemble.values()))
        samples = np.random.normal(training_mean, constraint_noise, 1000)
        return samples

    # Run validation
    validator = PerfectModelValidator(confidence_level=0.90)
    results = validator.validate_constraint(
        model_ensemble=model_ensemble,
        constraint_function=synthetic_constraint,
        constraint_data={}
    )

    metrics = validator.calculate_validation_metrics(results)

    return results, metrics
