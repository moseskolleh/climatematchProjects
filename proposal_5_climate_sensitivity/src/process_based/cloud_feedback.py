"""
Cloud Feedback Constraint

Uses observations of cloud properties to constrain cloud feedback
and thereby climate sensitivity.
"""

import numpy as np
from typing import Dict


class CloudFeedbackConstraint:
    """
    Climate sensitivity constraint from cloud feedback analysis

    Uses emergent constraints relating present-day cloud properties
    to future cloud feedback.
    """

    def __init__(self):
        self.F_2xCO2 = 3.7

    def calculate_cloud_feedback_distribution(
        self,
        n_samples: int = 100000
    ) -> np.ndarray:
        """
        Calculate cloud feedback distribution from observations

        Returns:
        --------
        feedback_samples : np.ndarray
            Cloud feedback samples (W/m²/K)
        """
        # Based on literature (e.g., Zelinka et al., Sherwood et al.)
        # Cloud feedback: +0.5 W/m²/K ± 0.3 W/m²/K
        feedback_samples = np.random.normal(0.5, 0.3, n_samples)

        return feedback_samples

    def feedback_to_ecs(self, cloud_feedback: float) -> float:
        """
        Convert cloud feedback to ECS estimate

        Uses Planck feedback and other well-constrained feedbacks

        Parameters:
        -----------
        cloud_feedback : float
            Cloud feedback (W/m²/K)

        Returns:
        --------
        ecs : float
            ECS estimate (K)
        """
        # Standard feedback decomposition
        planck_feedback = -3.2  # W/m²/K (always negative, stabilizing)
        wv_lr_feedback = 1.3  # Water vapor + lapse rate (W/m²/K)
        albedo_feedback = 0.4  # Surface albedo (W/m²/K)

        # Total feedback
        total_feedback = (planck_feedback + wv_lr_feedback +
                         albedo_feedback + cloud_feedback)

        # ECS = F_2×CO₂ / (-total_feedback)
        # Negative sign because negative feedback reduces temperature
        ecs = self.F_2xCO2 / (-total_feedback)

        return ecs

    def calculate_ecs_distribution(self, n_samples: int = 100000) -> np.ndarray:
        """
        Calculate ECS distribution from cloud feedback constraint

        Returns:
        --------
        ecs_samples : np.ndarray
            ECS samples (K)
        """
        cloud_feedback_samples = self.calculate_cloud_feedback_distribution(n_samples)

        ecs_samples = np.array([
            self.feedback_to_ecs(cf) for cf in cloud_feedback_samples
        ])

        # Physical constraints
        ecs_samples = ecs_samples[(ecs_samples > 0) & (ecs_samples < 10)]

        return ecs_samples

    def get_likelihood_function(self) -> callable:
        """Return likelihood function"""
        ecs_samples = self.calculate_ecs_distribution()

        def likelihood(ecs_values):
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(ecs_samples)
            return kde(ecs_values)

        return likelihood

    def calculate_constraint(self) -> Dict:
        """Calculate full constraint"""
        ecs_samples = self.calculate_ecs_distribution()
        cloud_fb = self.calculate_cloud_feedback_distribution()

        return {
            'name': 'Cloud',
            'ecs_samples': ecs_samples,
            'likelihood': self.get_likelihood_function(),
            'median': np.median(ecs_samples),
            'ci_90': tuple(np.percentile(ecs_samples, [5, 95])),
            'metadata': {
                'method': 'Cloud feedback emergent constraint',
                'cloud_feedback_median': np.median(cloud_fb),
                'cloud_feedback_ci_90': tuple(np.percentile(cloud_fb, [5, 95])),
                'n_samples': len(ecs_samples)
            }
        }

    def __repr__(self) -> str:
        return "CloudFeedbackConstraint()"
