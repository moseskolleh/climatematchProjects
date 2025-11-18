"""
Mid-Pliocene Warm Period (mPWP) Climate Sensitivity Constraint

This module implements methods to constrain climate sensitivity using
Mid-Pliocene Warm Period (~3.3-3.0 Ma) paleoclimate evidence.
"""

import numpy as np
import xarray as xr
from typing import Dict, Tuple, Optional


class mPWPConstraint:
    """
    Climate sensitivity constraint from Mid-Pliocene Warm Period

    The mPWP provides a constraint on Earth System Sensitivity (ESS) and
    ECS through observed temperatures under warmth with CO2 similar to present.

    Parameters:
    -----------
    data_path : str, optional
        Path to processed mPWP data
    co2_mpwp : float, optional
        mPWP CO2 concentration in ppm (default: 400)
    co2_pi : float, optional
        Pre-industrial CO2 in ppm (default: 280)
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        co2_mpwp: float = 400.0,
        co2_pi: float = 280.0
    ):
        self.data_path = data_path
        self.co2_mpwp = co2_mpwp
        self.co2_pi = co2_pi
        self.F_2xCO2 = 3.7  # W/m²

        self.proxy_temps = None
        self.model_temps = None

    def load_data(self, proxy_file: str = None, model_file: str = None) -> None:
        """Load mPWP data"""
        if proxy_file:
            self.proxy_temps = xr.open_dataset(proxy_file)
        if model_file:
            self.model_temps = xr.open_dataset(model_file)

    def calculate_co2_forcing(self) -> float:
        """Calculate CO2 radiative forcing"""
        F_CO2 = 5.35 * np.log(self.co2_mpwp / self.co2_pi)
        return F_CO2

    def calculate_slow_feedbacks(self) -> Tuple[float, float]:
        """
        Estimate contribution of slow feedbacks (ice sheets, vegetation)

        Returns:
        --------
        slow_feedback_forcing : float
            Additional forcing from slow feedbacks (W/m²)
        uncertainty : float
            Uncertainty (W/m²)
        """
        # Ice sheet reduction (mainly Greenland, some West Antarctic)
        # Contributes ~0.5-1.0 W/m² equivalent forcing
        slow_feedback_forcing = 0.75
        uncertainty = 0.25

        return slow_feedback_forcing, uncertainty

    def calculate_global_mean_temperature(
        self,
        n_bootstrap: int = 10000
    ) -> Tuple[float, float, float]:
        """
        Calculate global mean temperature anomaly

        Returns:
        --------
        median, lower, upper : float
            Temperature estimates (K)
        """
        # Literature consensus: +3.0°C ± 1.0°C
        sigma = 1.0 / 1.645
        samples = np.random.normal(3.0, sigma, n_bootstrap)

        return np.median(samples), np.percentile(samples, [5, 95])[0], np.percentile(samples, [5, 95])[1]

    def calculate_ess_distribution(self, n_samples: int = 100000) -> np.ndarray:
        """
        Calculate Earth System Sensitivity distribution

        ESS includes slow feedbacks (ice sheets)
        """
        T_median, T_lower, T_upper = self.calculate_global_mean_temperature()
        T_sigma = (T_upper - T_lower) / (2 * 1.645)
        T_samples = np.random.normal(T_median, T_sigma, n_samples)

        F_CO2 = self.calculate_co2_forcing()
        slow_fb, slow_fb_unc = self.calculate_slow_feedbacks()

        F_total_samples = np.random.normal(F_CO2 + slow_fb, slow_fb_unc, n_samples)

        ess_samples = T_samples / (F_total_samples / self.F_2xCO2)
        ess_samples = ess_samples[(ess_samples > 0) & (ess_samples < 15)]

        return ess_samples

    def calculate_ecs_distribution(self, n_samples: int = 100000) -> np.ndarray:
        """
        Calculate ECS distribution (removing slow feedbacks)

        Returns:
        --------
        ecs_samples : np.ndarray
            ECS samples (K)
        """
        ess_samples = self.calculate_ess_distribution(n_samples)

        # ESS typically 50% higher than ECS due to slow feedbacks
        # ECS ≈ ESS / 1.5 (approximate)
        ecs_samples = ess_samples / 1.5

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
        ess_samples = self.calculate_ess_distribution()

        return {
            'name': 'mPWP',
            'ecs_samples': ecs_samples,
            'likelihood': self.get_likelihood_function(),
            'median': np.median(ecs_samples),
            'ci_90': tuple(np.percentile(ecs_samples, [5, 95])),
            'metadata': {
                'period': 'Mid-Pliocene Warm Period',
                'age': '3.3-3.0 Ma',
                'temperature_anomaly': self.calculate_global_mean_temperature(),
                'ess_median': np.median(ess_samples),
                'n_samples': len(ecs_samples)
            }
        }

    def __repr__(self) -> str:
        return f"mPWPConstraint(CO2_mPWP={self.co2_mpwp} ppm)"
