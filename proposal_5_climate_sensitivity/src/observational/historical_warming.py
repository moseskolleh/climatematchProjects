"""
Historical Warming Climate Sensitivity Constraint

Energy budget approach using observed warming and radiative forcing.
"""

import numpy as np
import xarray as xr
from typing import Dict, Tuple, Optional


class HistoricalWarmingConstraint:
    """
    Climate sensitivity constraint from historical energy budget

    Uses observed warming, radiative forcing, and ocean heat uptake
    to constrain ECS through energy balance.

    Parameters:
    -----------
    baseline_period : tuple
        Period for baseline (default: (1850, 1900))
    recent_period : tuple
        Recent period for comparison (default: (2010, 2020))
    """

    def __init__(
        self,
        baseline_period: Tuple[int, int] = (1850, 1900),
        recent_period: Tuple[int, int] = (2010, 2020)
    ):
        self.baseline_period = baseline_period
        self.recent_period = recent_period
        self.F_2xCO2 = 3.7  # W/m²

    def calculate_observed_warming(self) -> Tuple[float, float]:
        """
        Calculate observed warming from multiple datasets

        Returns:
        --------
        warming : float
            Best estimate of warming (K)
        uncertainty : float
            Uncertainty (K)
        """
        # Multi-dataset average
        # HadCRUT5, GISTEMP, NOAAGlobalTemp, Berkeley Earth
        # Literature values for 1850-1900 to 2010-2020
        warming_estimates = [1.09, 1.12, 1.10, 1.13]  # K

        warming = np.mean(warming_estimates)
        uncertainty = np.std(warming_estimates) + 0.1  # Add structural uncertainty

        return warming, uncertainty

    def calculate_co2_forcing(self) -> Tuple[float, float]:
        """
        Calculate CO2 radiative forcing change

        Returns:
        --------
        forcing : float
            CO2 forcing (W/m²)
        uncertainty : float
            Uncertainty (W/m²)
        """
        # 1850: ~285 ppm, 2020: ~415 ppm
        co2_1850 = 285
        co2_2020 = 415

        forcing = 5.35 * np.log(co2_2020 / co2_1850)
        uncertainty = forcing * 0.1  # 10% uncertainty

        return forcing, uncertainty

    def calculate_other_ghg_forcing(self) -> Tuple[float, float]:
        """
        Calculate forcing from other greenhouse gases

        Returns:
        --------
        forcing : float
            Other GHG forcing (W/m²)
        uncertainty : float
            Uncertainty (W/m²)
        """
        # CH4, N2O, halocarbons
        # Based on IPCC AR6
        forcing = 1.0  # W/m²
        uncertainty = 0.2

        return forcing, uncertainty

    def calculate_aerosol_forcing(self) -> Tuple[float, float]:
        """
        Calculate aerosol radiative forcing

        This is the largest uncertainty in forcing estimates

        Returns:
        --------
        forcing : float
            Aerosol forcing (W/m²)
        uncertainty : float
            Uncertainty (W/m²)
        """
        # IPCC AR6: -1.0 W/m² with large uncertainty
        forcing = -1.0
        uncertainty = 0.5  # 5-95% range: -1.5 to -0.5

        return forcing, uncertainty

    def calculate_other_forcing(self) -> Tuple[float, float]:
        """
        Calculate other forcing components

        Returns:
        --------
        forcing : float
            Other forcing (W/m²)
        uncertainty : float
            Uncertainty (W/m²)
        """
        # Land use, solar, ozone, stratospheric water vapor
        forcing = 0.1
        uncertainty = 0.2

        return forcing, uncertainty

    def calculate_total_forcing(self, n_samples: int = 100000) -> np.ndarray:
        """
        Calculate total effective radiative forcing with uncertainties

        Parameters:
        -----------
        n_samples : int
            Number of Monte Carlo samples

        Returns:
        --------
        forcing_samples : np.ndarray
            Samples of total forcing (W/m²)
        """
        # Sample each component
        F_co2, F_co2_unc = self.calculate_co2_forcing()
        F_co2_samples = np.random.normal(F_co2, F_co2_unc, n_samples)

        F_ghg, F_ghg_unc = self.calculate_other_ghg_forcing()
        F_ghg_samples = np.random.normal(F_ghg, F_ghg_unc, n_samples)

        F_aer, F_aer_unc = self.calculate_aerosol_forcing()
        F_aer_samples = np.random.normal(F_aer, F_aer_unc, n_samples)

        F_other, F_other_unc = self.calculate_other_forcing()
        F_other_samples = np.random.normal(F_other, F_other_unc, n_samples)

        # Total forcing
        forcing_samples = F_co2_samples + F_ghg_samples + F_aer_samples + F_other_samples

        return forcing_samples

    def calculate_energy_imbalance(self) -> Tuple[float, float]:
        """
        Calculate Earth's energy imbalance (ocean heat uptake)

        Returns:
        --------
        imbalance : float
            Energy imbalance (W/m²)
        uncertainty : float
            Uncertainty (W/m²)
        """
        # From CERES and Argo: ~0.7 W/m² for recent decade
        imbalance = 0.7
        uncertainty = 0.3

        return imbalance, uncertainty

    def calculate_pattern_effect(self) -> Tuple[float, float]:
        """
        Calculate pattern effect correction

        The spatial pattern of warming affects effective climate sensitivity.
        Historical pattern may differ from equilibrium pattern.

        Returns:
        --------
        correction_factor : float
            Multiplicative correction to ECS
        uncertainty : float
            Uncertainty in correction
        """
        # Based on model studies
        # Historical ECS typically underestimates equilibrium ECS by ~10-20%
        correction_factor = 1.15  # 15% upward adjustment
        uncertainty = 0.1  # ±10%

        return correction_factor, uncertainty

    def calculate_ecs_distribution(self, n_samples: int = 100000) -> np.ndarray:
        """
        Calculate ECS distribution from energy budget

        ECS = (ΔT / (ΔF - ΔN)) × F_2×CO₂ × pattern_correction

        Parameters:
        -----------
        n_samples : int
            Number of Monte Carlo samples

        Returns:
        --------
        ecs_samples : np.ndarray
            ECS samples (K)
        """
        # Sample observed warming
        T_obs, T_obs_unc = self.calculate_observed_warming()
        T_samples = np.random.normal(T_obs, T_obs_unc, n_samples)

        # Sample total forcing
        F_samples = self.calculate_total_forcing(n_samples)

        # Sample energy imbalance
        N, N_unc = self.calculate_energy_imbalance()
        N_samples = np.random.normal(N, N_unc, n_samples)

        # Sample pattern effect
        pattern, pattern_unc = self.calculate_pattern_effect()
        pattern_samples = np.random.normal(pattern, pattern_unc, n_samples)

        # Calculate ECS
        ecs_samples = (T_samples / (F_samples - N_samples)) * self.F_2xCO2 * pattern_samples

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

        # Calculate forcing breakdown
        F_co2, _ = self.calculate_co2_forcing()
        F_ghg, _ = self.calculate_other_ghg_forcing()
        F_aer, F_aer_unc = self.calculate_aerosol_forcing()
        F_other, _ = self.calculate_other_forcing()

        return {
            'name': 'Historical',
            'ecs_samples': ecs_samples,
            'likelihood': self.get_likelihood_function(),
            'median': np.median(ecs_samples),
            'ci_90': tuple(np.percentile(ecs_samples, [5, 95])),
            'metadata': {
                'period': f'{self.baseline_period} to {self.recent_period}',
                'observed_warming': self.calculate_observed_warming(),
                'total_forcing': (F_co2 + F_ghg + F_aer + F_other,
                                 F_aer_unc),  # Aerosol dominates uncertainty
                'energy_imbalance': self.calculate_energy_imbalance(),
                'pattern_effect': self.calculate_pattern_effect(),
                'n_samples': len(ecs_samples)
            }
        }

    def sensitivity_to_aerosols(self) -> Dict:
        """
        Analyze sensitivity to aerosol forcing uncertainty

        Returns:
        --------
        results : dict
            ECS estimates for different aerosol forcing values
        """
        aerosol_forcings = np.linspace(-1.5, -0.5, 11)  # W/m²
        results = {}

        for F_aer in aerosol_forcings:
            # Recalculate with fixed aerosol forcing
            T_obs, _ = self.calculate_observed_warming()
            F_co2, _ = self.calculate_co2_forcing()
            F_ghg, _ = self.calculate_other_ghg_forcing()
            F_other, _ = self.calculate_other_forcing()
            N, _ = self.calculate_energy_imbalance()
            pattern, _ = self.calculate_pattern_effect()

            F_total = F_co2 + F_ghg + F_aer + F_other

            ecs = (T_obs / (F_total - N)) * self.F_2xCO2 * pattern

            results[F_aer] = ecs

        return results

    def __repr__(self) -> str:
        return f"HistoricalWarmingConstraint({self.baseline_period} to {self.recent_period})"
