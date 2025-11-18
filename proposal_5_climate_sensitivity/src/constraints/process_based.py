"""
Process-based constraints on climate sensitivity.

This module implements constraints derived from understanding of physical processes:
- Cloud feedbacks (low, high, altitude)
- Water vapor and lapse rate feedbacks
- Surface albedo feedbacks
- Regional pattern scaling
"""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class ProcessConstraintResult:
    """Results from a process-based constraint analysis."""
    ecs_mean: float
    ecs_std: float
    ecs_samples: np.ndarray
    feedback_components: Dict[str, float]
    total_feedback: float
    metadata: Dict


class ProcessBasedConstraint:
    """
    Base class for process-based constraints.

    Climate sensitivity is determined by climate feedback parameter λ:
        ECS = F_2xCO2 / λ

    where λ = λ_planck + λ_wv + λ_lr + λ_cloud + λ_albedo

    Component feedbacks:
    - Planck: Direct radiative response (~3.2 W/m²/K, stabilizing)
    - Water vapor: Increased atmospheric moisture (~1.8 W/m²/K, amplifying)
    - Lapse rate: Atmospheric temperature profile changes (~-0.8 W/m²/K, stabilizing)
    - Cloud: Changes in cloud cover/properties (~0.4 W/m²/K, amplifying, uncertain!)
    - Albedo: Snow/ice retreat (~0.4 W/m²/K, amplifying)
    """

    F_2XCO2 = 3.7  # W/m²

    def __init__(self, n_samples: int = 10000):
        """
        Initialize process-based constraint.

        Args:
            n_samples: Number of Monte Carlo samples
        """
        self.n_samples = n_samples

    def estimate_ecs_from_feedbacks(
        self,
        feedback_components: Dict[str, tuple],
        correlations: Optional[Dict[tuple, float]] = None
    ) -> ProcessConstraintResult:
        """
        Estimate ECS from feedback components.

        Args:
            feedback_components: Dict of {name: (mean, std)} for each feedback
            correlations: Optional dict of {(name1, name2): correlation}

        Returns:
            ProcessConstraintResult with ECS estimates
        """
        if correlations is None:
            correlations = {}

        # Extract feedback names, means, and stds
        names = list(feedback_components.keys())
        means = np.array([feedback_components[n][0] for n in names])
        stds = np.array([feedback_components[n][1] for n in names])

        # Build covariance matrix
        n_feedbacks = len(names)
        cov = np.diag(stds**2)

        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i < j:  # Upper triangle
                    corr = correlations.get((name_i, name_j), 0.0)
                    cov[i, j] = corr * stds[i] * stds[j]
                    cov[j, i] = cov[i, j]  # Symmetric

        # Generate correlated samples
        feedback_samples = np.random.multivariate_normal(
            means, cov, size=self.n_samples
        )

        # Total feedback is sum of components
        total_feedback_samples = np.sum(feedback_samples, axis=1)

        # ECS = F_2xCO2 / λ
        # Note: By convention, stabilizing feedbacks are negative
        # Total feedback should be negative for stable climate
        valid_mask = total_feedback_samples < -0.5  # Stability constraint
        total_feedback_samples = total_feedback_samples[valid_mask]

        ecs_samples = self.F_2XCO2 / (-total_feedback_samples)

        # Physical constraints
        physical_mask = (ecs_samples > 0) & (ecs_samples < 12)
        ecs_samples = ecs_samples[physical_mask]

        # Calculate mean feedback components for reporting
        mean_feedbacks = {
            name: np.mean(feedback_samples[:len(ecs_samples), i])
            for i, name in enumerate(names)
        }

        return ProcessConstraintResult(
            ecs_mean=np.mean(ecs_samples),
            ecs_std=np.std(ecs_samples),
            ecs_samples=ecs_samples,
            feedback_components=mean_feedbacks,
            total_feedback=np.mean(total_feedback_samples),
            metadata={
                "method": "Process-Based Feedbacks",
                "n_samples": len(ecs_samples),
                "feedback_names": names
            }
        )


class CloudFeedbackConstraint(ProcessBasedConstraint):
    """
    Cloud feedback constraint using process understanding.

    Cloud feedbacks are the largest source of uncertainty in climate sensitivity.
    This class decomposes cloud feedbacks into components:
    - Low cloud amount (subtropical stratocumulus regions)
    - High cloud altitude (tropical anvil clouds)
    - Cloud optical properties (liquid/ice water path)
    - Middle latitude clouds
    """

    def __init__(self, n_samples: int = 10000):
        super().__init__(n_samples)

    def estimate_from_cmip6_diagnostics(
        self,
        model_diagnostics: Dict[str, np.ndarray],
        observed_diagnostics: Dict[str, tuple]
    ) -> ProcessConstraintResult:
        """
        Estimate ECS using cloud feedback diagnostics from CMIP6.

        This implements a simplified version of emergent constraints on cloud feedbacks.

        Args:
            model_diagnostics: Dict of {model_name: [low_cloud_fb, high_cloud_fb, ...]}
            observed_diagnostics: Dict of {diagnostic_name: (observed_value, uncertainty)}

        Returns:
            ProcessConstraintResult with constrained ECS
        """
        # This is a placeholder for a more sophisticated implementation
        # In practice, would use actual CMIP6 cloud radiative kernel diagnostics

        # Typical CMIP6 cloud feedback: ~0.4 ± 0.35 W/m²/K
        cloud_feedback_mean = 0.42
        cloud_feedback_std = 0.35

        # Other well-constrained feedbacks
        feedbacks = {
            "planck": (-3.2, 0.05),  # Well understood, small uncertainty
            "water_vapor": (1.80, 0.15),  # Constrained by observations
            "lapse_rate": (-0.80, 0.20),  # Anti-correlated with water vapor
            "cloud": (cloud_feedback_mean, cloud_feedback_std),  # Large uncertainty!
            "albedo": (0.35, 0.10)  # Moderate uncertainty
        }

        # Water vapor and lapse rate are anti-correlated
        correlations = {
            ("water_vapor", "lapse_rate"): -0.5
        }

        return self.estimate_ecs_from_feedbacks(feedbacks, correlations)

    def decompose_cloud_feedback(
        self,
        low_cloud: tuple = (0.30, 0.25),
        high_cloud: tuple = (0.10, 0.15),
        middle_cloud: tuple = (0.02, 0.10)
    ) -> Dict[str, tuple]:
        """
        Decompose cloud feedback into components.

        Args:
            low_cloud: (mean, std) for low cloud feedback (W/m²/K)
            high_cloud: (mean, std) for high cloud feedback
            middle_cloud: (mean, std) for middle cloud feedback

        Returns:
            Dict of feedback components with uncertainties
        """
        # Combine cloud components
        cloud_total_mean = low_cloud[0] + high_cloud[0] + middle_cloud[0]

        # Assume partial independence: total variance is less than sum of variances
        correlation = 0.3  # Moderate correlation between cloud types
        cloud_total_var = (
            low_cloud[1]**2 + high_cloud[1]**2 + middle_cloud[1]**2 +
            2 * correlation * (
                low_cloud[1] * high_cloud[1] +
                low_cloud[1] * middle_cloud[1] +
                high_cloud[1] * middle_cloud[1]
            )
        )
        cloud_total_std = np.sqrt(cloud_total_var)

        return {
            "cloud_low": low_cloud,
            "cloud_high": high_cloud,
            "cloud_middle": middle_cloud,
            "cloud_total": (cloud_total_mean, cloud_total_std)
        }


class RegionalPatternConstraint(ProcessBasedConstraint):
    """
    Constraint based on regional warming patterns.

    Different models produce different spatial patterns of warming, which
    affect global feedback strength (pattern effect). Observations of
    historical patterns can constrain future sensitivity.
    """

    def __init__(self, n_samples: int = 10000):
        super().__init__(n_samples)

    def estimate_pattern_effect(
        self,
        tropical_pacific_warming: float,
        tropical_pacific_uncertainty: float,
        pattern_sensitivity: float = -0.3
    ) -> float:
        """
        Estimate adjustment to ECS based on warming pattern.

        Args:
            tropical_pacific_warming: Observed tropical Pacific warming trend
            tropical_pacific_uncertainty: Uncertainty in trend
            pattern_sensitivity: Sensitivity of feedback to pattern (W/m²/K per K)

        Returns:
            Adjustment factor for ECS
        """
        # Models with stronger tropical Pacific warming tend to have lower ECS
        # This is because warming in subsidence regions reduces low cloud cover
        # more than warming in ascending regions

        # Sample warming patterns
        pattern_samples = np.random.normal(
            tropical_pacific_warming,
            tropical_pacific_uncertainty,
            self.n_samples
        )

        # Calculate adjustment to feedback parameter
        feedback_adjustment = pattern_sensitivity * pattern_samples

        # This translates to an adjustment in ECS
        # Δλ = pattern_sensitivity * ΔT_pattern
        # ΔECS = ECS² / F_2xCO2 * Δλ

        # For now, return mean adjustment
        return np.mean(feedback_adjustment)


def combine_process_constraints(
    constraints: List[ProcessConstraintResult],
    account_for_dependencies: bool = True
) -> ProcessConstraintResult:
    """
    Combine multiple process-based constraints.

    Args:
        constraints: List of ProcessConstraintResult objects
        account_for_dependencies: Whether to account for shared physical processes

    Returns:
        Combined ProcessConstraintResult
    """
    # Combine samples from all constraints
    all_samples = np.concatenate([c.ecs_samples for c in constraints])

    # Resample to standard size
    combined_samples = np.random.choice(
        all_samples, size=10000, replace=True
    )

    # Aggregate feedback components (take mean across constraints)
    all_feedback_names = set()
    for c in constraints:
        all_feedback_names.update(c.feedback_components.keys())

    combined_feedbacks = {}
    for name in all_feedback_names:
        values = [
            c.feedback_components.get(name, np.nan)
            for c in constraints
        ]
        combined_feedbacks[name] = np.nanmean(values)

    return ProcessConstraintResult(
        ecs_mean=np.mean(combined_samples),
        ecs_std=np.std(combined_samples),
        ecs_samples=combined_samples,
        feedback_components=combined_feedbacks,
        total_feedback=np.mean([c.total_feedback for c in constraints]),
        metadata={
            "method": "Combined Process-Based",
            "n_constraints": len(constraints)
        }
    )
