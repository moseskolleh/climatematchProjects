"""
Last Interglacial (LIG) Climate Sensitivity Constraint
"""

import numpy as np
import xarray as xr
from typing import Dict, Tuple, Optional


class LIGConstraint:
    """
    Climate sensitivity constraint from Last Interglacial

    The LIG (~129-116 ka) provides a constraint on ECS through
    relatively small temperature changes with orbital forcing.
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.F_2xCO2 = 3.7

    def calculate_global_mean_temperature(self) -> Tuple[float, float, float]:
        """
        Calculate LIG temperature anomaly

        Returns:
        --------
        median, lower, upper : float
            Temperature estimates (K)
        """
        # Literature: +1.5°C ± 0.8°C
        sigma = 0.8 / 1.645
        samples = np.random.normal(1.5, sigma, 10000)
        return np.median(samples), np.percentile(samples, [5, 95])[0], np.percentile(samples, [5, 95])[1]

    def calculate_orbital_forcing(self) -> Tuple[float, float]:
        """
        Calculate orbital forcing at LIG

        Returns:
        --------
        forcing, uncertainty : float
            Effective global forcing (W/m²)
        """
        # Simplified: peak northern summer insolation
        # Global mean effect small but tropical amplification important
        forcing = 0.5  # W/m² effective
        uncertainty = 0.3

        return forcing, uncertainty

    def calculate_ecs_distribution(self, n_samples: int = 100000) -> np.ndarray:
        """Calculate ECS distribution"""
        T_median, T_lower, T_upper = self.calculate_global_mean_temperature()
        T_sigma = (T_upper - T_lower) / (2 * 1.645)
        T_samples = np.random.normal(T_median, T_sigma, n_samples)

        F_orb, F_orb_unc = self.calculate_orbital_forcing()
        F_samples = np.random.normal(F_orb, F_orb_unc, n_samples)

        # ECS from feedback analysis (more complex than simple forcing/response)
        # Use feedback decomposition approach
        # For simplicity, use approximate relationship
        ecs_samples = (T_samples / (F_samples / self.F_2xCO2)) * 0.7  # Correction factor

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

        return {
            'name': 'LIG',
            'ecs_samples': ecs_samples,
            'likelihood': self.get_likelihood_function(),
            'median': np.median(ecs_samples),
            'ci_90': tuple(np.percentile(ecs_samples, [5, 95])),
            'metadata': {
                'period': 'Last Interglacial',
                'age': '129-116 ka',
                'temperature_anomaly': self.calculate_global_mean_temperature(),
                'n_samples': len(ecs_samples)
            }
        }

    def __repr__(self) -> str:
        return "LIGConstraint()"
