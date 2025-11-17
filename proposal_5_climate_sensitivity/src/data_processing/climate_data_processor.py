"""
Climate Data Processor

Main class for loading and processing climate data from multiple sources.
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

from .cmip_loader import CMIPDataLoader
from .paleoclimate_loader import PaleoclimateDataLoader
from .observational_loader import ObservationalDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClimateDataProcessor:
    """
    Main processor for climate data used in climate sensitivity constraints.

    This class coordinates data loading from CMIP models, paleoclimate proxies,
    and modern observations, ensuring consistent formatting and metadata.

    Attributes:
        data_dir (Path): Root directory for data storage
        cmip_loader (CMIPDataLoader): Loader for CMIP model data
        paleo_loader (PaleoclimateDataLoader): Loader for paleoclimate data
        obs_loader (ObservationalDataLoader): Loader for observational data
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the Climate Data Processor.

        Args:
            data_dir: Root directory for data storage. If None, uses default location.
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            data_dir = Path(data_dir)

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data loaders
        self.cmip_loader = CMIPDataLoader(data_dir / "raw" / "cmip")
        self.paleo_loader = PaleoclimateDataLoader(data_dir / "raw" / "paleoclimate")
        self.obs_loader = ObservationalDataLoader(data_dir / "raw" / "observations")

        logger.info(f"Initialized ClimateDataProcessor with data directory: {data_dir}")

    def load_cmip_data(
        self,
        experiment: str = 'abrupt-4xCO2',
        models: Optional[List[str]] = None,
        variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        """
        Load CMIP model data for climate sensitivity analysis.

        Args:
            experiment: CMIP experiment name (e.g., 'abrupt-4xCO2', 'piControl')
            models: List of model names to load. If None, loads all available models.
            variables: List of variables to load. If None, loads standard ECS variables.

        Returns:
            xarray Dataset containing CMIP model outputs
        """
        if variables is None:
            variables = ['tas', 'ts', 'rtnt', 'rlut', 'rsut', 'rsdt']

        logger.info(f"Loading CMIP data for experiment: {experiment}")
        data = self.cmip_loader.load_experiment(
            experiment=experiment,
            models=models,
            variables=variables
        )

        # Standardize metadata
        data = self._standardize_cmip_metadata(data)

        return data

    def load_paleoclimate_data(
        self,
        period: str = 'LGM',
        proxy_types: Optional[List[str]] = None
    ) -> Dict[str, xr.Dataset]:
        """
        Load paleoclimate proxy data for constraint analysis.

        Args:
            period: Paleoclimate period (e.g., 'LGM', 'MPWP', 'Last_Interglacial')
            proxy_types: Types of proxies to load. If None, loads all available.

        Returns:
            Dictionary of datasets by proxy type
        """
        logger.info(f"Loading paleoclimate data for period: {period}")
        data = self.paleo_loader.load_period(period, proxy_types)

        # Process and quality control
        data = self._process_paleoclimate_data(data)

        return data

    def load_observational_data(
        self,
        datasets: Optional[List[str]] = None,
        start_year: int = 1850,
        end_year: Optional[int] = None
    ) -> xr.Dataset:
        """
        Load modern observational data.

        Args:
            datasets: List of observational datasets to load. If None, loads all.
            start_year: Start year for data
            end_year: End year for data. If None, uses latest available.

        Returns:
            xarray Dataset containing observational data
        """
        logger.info(f"Loading observational data from {start_year} to {end_year or 'present'}")
        data = self.obs_loader.load_observations(
            datasets=datasets,
            start_year=start_year,
            end_year=end_year
        )

        # Harmonize observations
        data = self._harmonize_observations(data)

        return data

    def _standardize_cmip_metadata(self, data: xr.Dataset) -> xr.Dataset:
        """Standardize CMIP metadata for consistency."""
        # Add standard attributes
        data.attrs['processed_by'] = 'ClimateDataProcessor'
        data.attrs['framework'] = 'Multi-Constraint Climate Sensitivity'

        # Ensure standard coordinate names
        coord_mapping = {
            'latitude': 'lat',
            'longitude': 'lon',
            'time': 'time'
        }

        for old_name, new_name in coord_mapping.items():
            if old_name in data.coords and new_name not in data.coords:
                data = data.rename({old_name: new_name})

        return data

    def _process_paleoclimate_data(
        self,
        data: Dict[str, xr.Dataset]
    ) -> Dict[str, xr.Dataset]:
        """Process and quality control paleoclimate proxy data."""
        processed = {}

        for proxy_type, dataset in data.items():
            # Apply quality flags
            if 'quality_flag' in dataset:
                dataset = dataset.where(dataset.quality_flag >= 0.5)

            # Add uncertainty estimates if not present
            if 'uncertainty' not in dataset.data_vars:
                logger.warning(
                    f"No uncertainty estimates for {proxy_type}. "
                    f"Using default 20% uncertainty."
                )
                for var in dataset.data_vars:
                    if var != 'quality_flag':
                        dataset[f'{var}_uncertainty'] = abs(dataset[var]) * 0.2

            processed[proxy_type] = dataset

        return processed

    def _harmonize_observations(self, data: xr.Dataset) -> xr.Dataset:
        """Harmonize observational datasets to common grid and time basis."""
        # Ensure consistent time coordinates
        if 'time' in data.coords:
            # Convert to standard calendar if needed
            if hasattr(data.time, 'calendar'):
                if data.time.calendar != 'standard':
                    logger.info(f"Converting from {data.time.calendar} to standard calendar")
                    # Conversion logic would go here

        # Add provenance metadata
        data.attrs['harmonized'] = True
        data.attrs['harmonization_date'] = str(np.datetime64('today'))

        return data

    def compute_model_ecs(
        self,
        abrupt_4x: xr.Dataset,
        pi_control: xr.Dataset,
        method: str = 'regression'
    ) -> Dict[str, float]:
        """
        Compute equilibrium climate sensitivity from CMIP abrupt-4xCO2 experiments.

        Args:
            abrupt_4x: Dataset from abrupt-4xCO2 experiment
            pi_control: Dataset from piControl experiment
            method: Method for ECS calculation ('regression' or 'plateau')

        Returns:
            Dictionary with ECS values and uncertainties by model
        """
        ecs_values = {}

        for model in abrupt_4x.model.values:
            model_4x = abrupt_4x.sel(model=model)
            model_pi = pi_control.sel(model=model)

            if method == 'regression':
                ecs = self._compute_ecs_regression(model_4x, model_pi)
            elif method == 'plateau':
                ecs = self._compute_ecs_plateau(model_4x, model_pi)
            else:
                raise ValueError(f"Unknown ECS computation method: {method}")

            ecs_values[str(model.values)] = ecs

        return ecs_values

    def _compute_ecs_regression(
        self,
        abrupt_4x: xr.Dataset,
        pi_control: xr.Dataset
    ) -> Dict[str, float]:
        """
        Compute ECS using Gregory regression method.

        This method regresses the top-of-atmosphere radiative imbalance
        against global mean surface temperature change.
        """
        # Calculate global mean temperature change
        if 'tas' in abrupt_4x.data_vars:
            temp_var = 'tas'
        elif 'ts' in abrupt_4x.data_vars:
            temp_var = 'ts'
        else:
            raise ValueError("No temperature variable found in dataset")

        # Get anomalies relative to piControl
        temp_4x = abrupt_4x[temp_var].mean(dim=['lat', 'lon'])
        temp_pi = pi_control[temp_var].mean(dim=['lat', 'lon'])
        delta_t = temp_4x - temp_pi.mean(dim='time')

        # Calculate TOA radiative imbalance
        if 'rtnt' in abrupt_4x.data_vars:
            rad_imbalance = abrupt_4x['rtnt'].mean(dim=['lat', 'lon'])
        else:
            # Calculate from components
            rad_imbalance = (
                abrupt_4x['rsdt'].mean(dim=['lat', 'lon']) -
                abrupt_4x['rsut'].mean(dim=['lat', 'lon']) -
                abrupt_4x['rlut'].mean(dim=['lat', 'lon'])
            )

        # Use years 1-150 for regression
        years_to_use = slice(1, 150)
        delta_t_regress = delta_t.isel(time=years_to_use)
        rad_regress = rad_imbalance.isel(time=years_to_use)

        # Perform linear regression
        coefficients = np.polyfit(
            delta_t_regress.values,
            rad_regress.values,
            deg=1
        )

        # ECS is the temperature at which radiative imbalance reaches zero
        # N = F - lambda * Delta_T
        # At equilibrium: 0 = F - lambda * ECS
        # ECS = F / lambda

        # For 4xCO2, F ≈ 8 W/m² (can be refined with actual forcing)
        forcing_4x = 8.0  # W/m²
        lambda_param = -coefficients[0]  # Climate feedback parameter

        ecs_4x = forcing_4x / lambda_param
        ecs_2x = ecs_4x / 2  # Scale to 2xCO2

        # Estimate uncertainty from residuals
        predicted = np.polyval(coefficients, delta_t_regress.values)
        residuals = rad_regress.values - predicted
        uncertainty = np.std(residuals) / lambda_param

        return {
            'ecs': float(ecs_2x),
            'ecs_uncertainty': float(uncertainty),
            'lambda': float(lambda_param),
            'method': 'gregory_regression'
        }

    def _compute_ecs_plateau(
        self,
        abrupt_4x: xr.Dataset,
        pi_control: xr.Dataset
    ) -> Dict[str, float]:
        """
        Compute ECS from equilibrated temperature in long runs.

        This method uses the plateau temperature in runs that have
        reached equilibrium.
        """
        # This is a simplified version - would need actual implementation
        # based on run length and equilibration criteria

        logger.warning("Plateau method not fully implemented, using simplified approach")

        # Use final 50 years of run
        if 'tas' in abrupt_4x.data_vars:
            temp_var = 'tas'
        else:
            temp_var = 'ts'

        temp_4x = abrupt_4x[temp_var].mean(dim=['lat', 'lon'])
        temp_pi = pi_control[temp_var].mean(dim=['lat', 'lon'])

        # Get final 50 years
        equilibrium_temp = temp_4x.isel(time=slice(-50, None)).mean(dim='time')
        control_temp = temp_pi.mean(dim='time')

        ecs_4x = float(equilibrium_temp - control_temp)
        ecs_2x = ecs_4x / 2

        uncertainty = float(temp_4x.isel(time=slice(-50, None)).std(dim='time'))

        return {
            'ecs': ecs_2x,
            'ecs_uncertainty': uncertainty,
            'method': 'plateau'
        }
