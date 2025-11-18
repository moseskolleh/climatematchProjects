"""
Last Glacial Maximum (LGM) Climate Sensitivity Constraint

This module implements methods to constrain climate sensitivity using
Last Glacial Maximum (21,000 years ago) paleoclimate evidence.
"""

import numpy as np
import xarray as xr
from scipy import stats
from typing import Dict, Tuple, Optional
import warnings


class LGMConstraint:
    """
    Climate sensitivity constraint from Last Glacial Maximum

    The LGM provides a constraint on ECS through the observed temperature
    response to known forcings during a cold climate state.

    Parameters:
    -----------
    data_path : str, optional
        Path to processed LGM data
    co2_lgm : float, optional
        LGM CO2 concentration in ppm (default: 190)
    co2_pi : float, optional
        Pre-industrial CO2 in ppm (default: 280)
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        co2_lgm: float = 190.0,
        co2_pi: float = 280.0
    ):
        self.data_path = data_path
        self.co2_lgm = co2_lgm
        self.co2_pi = co2_pi
        self.F_2xCO2 = 3.7  # W/m² (IPCC AR6)

        # Will be populated by load_data()
        self.proxy_temps = None
        self.model_temps = None
        self.forcings = None

    def load_data(self, proxy_file: str = None, model_file: str = None) -> None:
        """
        Load LGM temperature reconstructions and model data

        Parameters:
        -----------
        proxy_file : str, optional
            Path to proxy temperature data
        model_file : str, optional
            Path to PMIP model simulations
        """
        if proxy_file:
            self.proxy_temps = xr.open_dataset(proxy_file)

        if model_file:
            self.model_temps = xr.open_dataset(model_file)

    def calculate_co2_forcing(self) -> float:
        """
        Calculate radiative forcing from CO2 change

        Returns:
        --------
        F_CO2 : float
            Radiative forcing from CO2 (W/m²)
        """
        F_CO2 = 5.35 * np.log(self.co2_lgm / self.co2_pi)
        return F_CO2

    def calculate_ice_sheet_forcing(
        self,
        ice_area: float = 45e12,  # m²
        albedo_change: float = 0.3
    ) -> Tuple[float, float]:
        """
        Calculate radiative forcing from ice sheet albedo change

        Parameters:
        -----------
        ice_area : float
            Additional ice sheet area at LGM (m²)
        albedo_change : float
            Change in surface albedo

        Returns:
        --------
        F_ice : float
            Radiative forcing from ice sheets (W/m²)
        uncertainty : float
            Uncertainty in forcing (W/m²)
        """
        # Global mean insolation
        S0 = 1361  # W/m²
        global_mean_insolation = S0 / 4  # W/m²

        # Earth surface area
        earth_area = 5.1e14  # m²

        # Forcing calculation
        F_ice = (-global_mean_insolation * albedo_change *
                 ice_area / earth_area)

        # Uncertainty (±30%)
        uncertainty = abs(F_ice) * 0.3

        return F_ice, uncertainty

    def calculate_vegetation_forcing(self) -> Tuple[float, float]:
        """
        Calculate radiative forcing from vegetation changes

        Returns:
        --------
        F_veg : float
            Radiative forcing from vegetation (W/m²)
        uncertainty : float
            Uncertainty (W/m²)
        """
        # Based on PMIP model analyses
        F_veg = -1.0  # W/m² (approximate)
        uncertainty = 0.5  # Large uncertainty

        return F_veg, uncertainty

    def calculate_dust_forcing(self) -> Tuple[float, float]:
        """
        Calculate radiative forcing from dust aerosols

        Returns:
        --------
        F_dust : float
            Radiative forcing from dust (W/m²)
        uncertainty : float
            Uncertainty (W/m²)
        """
        # Based on ice core dust records and model studies
        F_dust = -1.0  # W/m²
        uncertainty = 0.7  # Very uncertain

        return F_dust, uncertainty

    def calculate_total_forcing(self) -> Dict[str, Tuple[float, float]]:
        """
        Calculate total radiative forcing at LGM with uncertainties

        Returns:
        --------
        forcings : dict
            Dictionary of forcing components and uncertainties
        """
        forcings = {}

        # CO2 forcing (well constrained)
        forcings['co2'] = (self.calculate_co2_forcing(), 0.1)

        # Ice sheet forcing
        forcings['ice'] = self.calculate_ice_sheet_forcing()

        # Vegetation forcing
        forcings['vegetation'] = self.calculate_vegetation_forcing()

        # Dust forcing
        forcings['dust'] = self.calculate_dust_forcing()

        # Total forcing with uncertainty propagation
        total = sum(f[0] for f in forcings.values())
        uncertainty = np.sqrt(sum(f[1]**2 for f in forcings.values()))

        forcings['total'] = (total, uncertainty)

        return forcings

    def calculate_global_mean_temperature(
        self,
        n_bootstrap: int = 10000
    ) -> Tuple[float, float, float]:
        """
        Calculate global mean temperature anomaly from proxies

        Uses bootstrap resampling to quantify uncertainty

        Parameters:
        -----------
        n_bootstrap : int
            Number of bootstrap samples

        Returns:
        --------
        median : float
            Median temperature anomaly (K)
        lower : float
            5th percentile (K)
        upper : float
            95th percentile (K)
        """
        # For demonstration, use literature values
        # In practice, would process actual proxy data

        # Literature consensus: -6°C ± 1.5°C (5-95%)
        # Generate samples from this distribution
        sigma = 1.5 / 1.645  # Convert 90% CI to std dev
        samples = np.random.normal(-6.0, sigma, n_bootstrap)

        median = np.median(samples)
        lower = np.percentile(samples, 5)
        upper = np.percentile(samples, 95)

        return median, lower, upper

    def calculate_ecs_distribution(
        self,
        n_samples: int = 100000,
        state_dependence_factor: float = 1.0
    ) -> np.ndarray:
        """
        Calculate ECS distribution from LGM constraint

        Parameters:
        -----------
        n_samples : int
            Number of Monte Carlo samples
        state_dependence_factor : float
            Adjustment for state-dependence (default: 1.0 = no adjustment)

        Returns:
        --------
        ecs_samples : np.ndarray
            Array of ECS samples (K)
        """
        # Sample temperature anomaly
        T_median, T_lower, T_upper = self.calculate_global_mean_temperature()
        T_sigma = (T_upper - T_lower) / (2 * 1.645)
        T_samples = np.random.normal(T_median, T_sigma, n_samples)

        # Sample total forcing
        forcings = self.calculate_total_forcing()
        F_total, F_uncertainty = forcings['total']
        F_samples = np.random.normal(F_total, F_uncertainty, n_samples)

        # Calculate ECS
        # ECS = (ΔT_LGM / (ΔF_LGM / F_2×CO₂)) * state_dependence_factor
        ecs_samples = (T_samples / (F_samples / self.F_2xCO2)) * state_dependence_factor

        # Apply physical constraints (positive ECS)
        ecs_samples = ecs_samples[ecs_samples > 0]

        # Remove extreme outliers (>10 K unrealistic)
        ecs_samples = ecs_samples[ecs_samples < 10]

        return ecs_samples

    def get_likelihood_function(self) -> callable:
        """
        Return likelihood function for Bayesian integration

        Returns:
        --------
        likelihood : callable
            Function that takes ECS values and returns likelihood
        """
        ecs_samples = self.calculate_ecs_distribution()

        def likelihood(ecs_values):
            """
            Calculate likelihood for given ECS values

            Uses kernel density estimation from samples
            """
            from scipy.stats import gaussian_kde

            # Fit KDE to samples
            kde = gaussian_kde(ecs_samples)

            # Evaluate at given ECS values
            return kde(ecs_values)

        return likelihood

    def calculate_constraint(self) -> Dict:
        """
        Calculate full constraint with metadata

        Returns:
        --------
        constraint : dict
            Dictionary containing:
            - name: Constraint name
            - ecs_samples: ECS distribution samples
            - likelihood: Likelihood function
            - median: Median ECS
            - ci_90: 90% confidence interval
            - metadata: Additional information
        """
        ecs_samples = self.calculate_ecs_distribution()

        constraint = {
            'name': 'LGM',
            'ecs_samples': ecs_samples,
            'likelihood': self.get_likelihood_function(),
            'median': np.median(ecs_samples),
            'ci_90': tuple(np.percentile(ecs_samples, [5, 95])),
            'metadata': {
                'period': 'Last Glacial Maximum',
                'age': '21,000 years ago',
                'temperature_anomaly': self.calculate_global_mean_temperature(),
                'forcings': self.calculate_total_forcing(),
                'n_samples': len(ecs_samples)
            }
        }

        return constraint

    def validate_against_models(self, model_data: xr.Dataset) -> Dict:
        """
        Validate constraint against PMIP model ensemble

        Parameters:
        -----------
        model_data : xr.Dataset
            PMIP model simulations

        Returns:
        --------
        validation : dict
            Validation statistics
        """
        # Extract model ECS and LGM temperature
        model_ecs = model_data['ecs'].values
        model_lgm_temp = model_data['lgm_temperature_anomaly'].values

        # Calculate ECS from our constraint for each model's LGM temp
        predicted_ecs = []
        forcings = self.calculate_total_forcing()
        F_total = forcings['total'][0]

        for T_lgm in model_lgm_temp:
            ecs_pred = (T_lgm / (F_total / self.F_2xCO2))
            predicted_ecs.append(ecs_pred)

        predicted_ecs = np.array(predicted_ecs)

        # Calculate validation metrics
        bias = np.mean(predicted_ecs - model_ecs)
        rmse = np.sqrt(np.mean((predicted_ecs - model_ecs)**2))
        correlation = np.corrcoef(predicted_ecs, model_ecs)[0, 1]

        validation = {
            'bias': bias,
            'rmse': rmse,
            'correlation': correlation,
            'n_models': len(model_ecs)
        }

        return validation

    def __repr__(self) -> str:
        return f"LGMConstraint(CO2_LGM={self.co2_lgm} ppm, CO2_PI={self.co2_pi} ppm)"
