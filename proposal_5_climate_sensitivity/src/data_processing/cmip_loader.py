"""
CMIP Data Loader

Loads and processes CMIP5/CMIP6 model outputs for climate sensitivity analysis.
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class CMIPDataLoader:
    """
    Loader for CMIP model data.

    Handles loading of CMIP5 and CMIP6 model outputs from standard
    experiments used in climate sensitivity analysis.
    """

    # Standard CMIP experiments for ECS analysis
    STANDARD_EXPERIMENTS = [
        'abrupt-4xCO2',
        'piControl',
        'historical',
        'abrupt-2xCO2',
        '1pctCO2'
    ]

    # Standard variables for ECS calculation
    STANDARD_VARIABLES = [
        'tas',   # Near-surface air temperature
        'ts',    # Surface temperature
        'rtnt',  # TOA net radiation
        'rlut',  # TOA outgoing longwave radiation
        'rsut',  # TOA outgoing shortwave radiation
        'rsdt',  # TOA incoming shortwave radiation
        'hfds',  # Downward heat flux at surface (ocean heat uptake)
    ]

    def __init__(self, data_dir: Path):
        """
        Initialize CMIP data loader.

        Args:
            data_dir: Directory containing CMIP data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Model metadata
        self.model_metadata = {}

    def load_experiment(
        self,
        experiment: str,
        models: Optional[List[str]] = None,
        variables: Optional[List[str]] = None,
        variant: str = 'r1i1p1f1'
    ) -> xr.Dataset:
        """
        Load data from a CMIP experiment.

        Args:
            experiment: Experiment name (e.g., 'abrupt-4xCO2')
            models: List of model names to load. If None, loads all available.
            variables: List of variables to load. If None, uses standard variables.
            variant: Variant label for ensemble members

        Returns:
            xarray Dataset with multi-model data
        """
        if experiment not in self.STANDARD_EXPERIMENTS:
            logger.warning(
                f"Experiment '{experiment}' not in standard list. "
                f"Proceeding anyway, but results may be unexpected."
            )

        if variables is None:
            variables = self.STANDARD_VARIABLES

        # Check if data exists, otherwise create synthetic data for demonstration
        experiment_dir = self.data_dir / experiment
        if not experiment_dir.exists():
            logger.warning(
                f"No data found for experiment '{experiment}'. "
                f"Generating synthetic data for demonstration."
            )
            return self._generate_synthetic_cmip_data(
                experiment=experiment,
                models=models or ['CESM2', 'GFDL-CM4', 'IPSL-CM6A', 'UKESM1'],
                variables=variables
            )

        # Load actual CMIP data
        datasets = []
        available_models = self._find_available_models(experiment_dir)

        if models is None:
            models = available_models
        else:
            models = [m for m in models if m in available_models]

        for model in models:
            model_data = self._load_model_data(
                experiment_dir / model,
                variables,
                variant
            )
            if model_data is not None:
                datasets.append(model_data.expand_dims(model=[model]))

        if not datasets:
            raise ValueError(f"No data found for experiment '{experiment}'")

        # Combine all models
        combined = xr.concat(datasets, dim='model')

        return combined

    def _find_available_models(self, experiment_dir: Path) -> List[str]:
        """Find available models in an experiment directory."""
        if not experiment_dir.exists():
            return []

        models = []
        for path in experiment_dir.iterdir():
            if path.is_dir():
                models.append(path.name)

        return sorted(models)

    def _load_model_data(
        self,
        model_dir: Path,
        variables: List[str],
        variant: str
    ) -> Optional[xr.Dataset]:
        """Load data for a single model."""
        try:
            datasets = {}

            for var in variables:
                var_file = model_dir / f"{var}_{variant}.nc"

                if var_file.exists():
                    datasets[var] = xr.open_dataset(var_file)
                else:
                    logger.debug(f"Variable {var} not found for {model_dir.name}")

            if not datasets:
                return None

            # Merge all variables
            combined = xr.merge(datasets.values())

            return combined

        except Exception as e:
            logger.error(f"Error loading {model_dir.name}: {e}")
            return None

    def _generate_synthetic_cmip_data(
        self,
        experiment: str,
        models: List[str],
        variables: List[str]
    ) -> xr.Dataset:
        """
        Generate synthetic CMIP data for demonstration purposes.

        This creates realistic-looking data with appropriate dimensions,
        coordinates, and realistic ECS values for each model.
        """
        logger.info(f"Generating synthetic data for experiment: {experiment}")

        n_models = len(models)
        n_years = 150 if experiment == 'abrupt-4xCO2' else 500
        n_lat = 64
        n_lon = 128

        # Create coordinates
        time = np.arange(n_years)
        lat = np.linspace(-89.375, 89.375, n_lat)
        lon = np.linspace(0, 357.1875, n_lon)

        # Model-specific ECS values (realistic range: 1.5 - 5.5 K)
        model_ecs = {
            'CESM2': 5.1,
            'GFDL-CM4': 3.9,
            'IPSL-CM6A': 4.5,
            'UKESM1': 5.3,
            'CanESM5': 5.6,
            'MIROC6': 2.6,
            'MRI-ESM2': 3.1,
            'NorESM2': 3.0
        }

        datasets = []

        for model in models:
            # Get or assign ECS for this model
            ecs = model_ecs.get(model, np.random.uniform(2.5, 4.5))

            # Generate temperature response
            if experiment == 'abrupt-4xCO2':
                # Exponential approach to equilibrium
                # ΔT = ECS * 2 * (1 - exp(-t/tau))
                tau = np.random.uniform(40, 80)  # Response timescale (years)
                temp_response = 2 * ecs * (1 - np.exp(-time / tau))

                # Add spatial pattern
                # Simplified: More warming at high latitudes
                lat_factor = 1 + 0.5 * (np.abs(lat) / 90) ** 2

                # Add noise
                noise_scale = 0.1
                temp_response_3d = (
                    temp_response[:, np.newaxis, np.newaxis] *
                    lat_factor[np.newaxis, :, np.newaxis] +
                    noise_scale * np.random.randn(n_years, n_lat, n_lon)
                )

                # TOA radiative imbalance
                # N = F - lambda * ΔT, where lambda = F / ECS
                forcing = 8.0  # W/m² for 4xCO2
                lambda_param = forcing / (2 * ecs)
                rad_imbalance = forcing - lambda_param * temp_response

                rad_imbalance_3d = (
                    rad_imbalance[:, np.newaxis, np.newaxis] +
                    0.5 * np.random.randn(n_years, n_lat, n_lon)
                )

            else:  # piControl
                # Small random fluctuations around zero
                temp_response_3d = 0.2 * np.random.randn(n_years, n_lat, n_lon)
                rad_imbalance_3d = 0.5 * np.random.randn(n_years, n_lat, n_lon)

            # Create dataset
            data_vars = {}

            if 'tas' in variables:
                data_vars['tas'] = (
                    ('time', 'lat', 'lon'),
                    temp_response_3d,
                    {'long_name': 'Near-Surface Air Temperature', 'units': 'K'}
                )

            if 'rtnt' in variables:
                data_vars['rtnt'] = (
                    ('time', 'lat', 'lon'),
                    rad_imbalance_3d,
                    {'long_name': 'TOA Net Radiation', 'units': 'W m-2'}
                )

            # Additional variables can be added as needed
            for var in ['rlut', 'rsut', 'rsdt']:
                if var in variables:
                    # Generate placeholder data
                    if var == 'rsdt':
                        # Incoming solar radiation (roughly constant)
                        data_vars[var] = (
                            ('time', 'lat', 'lon'),
                            340 + 5 * np.random.randn(n_years, n_lat, n_lon),
                            {'long_name': 'TOA Incident Shortwave Radiation', 'units': 'W m-2'}
                        )
                    else:
                        data_vars[var] = (
                            ('time', 'lat', 'lon'),
                            100 + 20 * np.random.randn(n_years, n_lat, n_lon),
                            {'long_name': f'TOA {var}', 'units': 'W m-2'}
                        )

            ds = xr.Dataset(
                data_vars,
                coords={
                    'time': time,
                    'lat': lat,
                    'lon': lon
                },
                attrs={
                    'model': model,
                    'experiment': experiment,
                    'true_ecs': ecs,
                    'synthetic': True
                }
            )

            datasets.append(ds.expand_dims(model=[model]))

        # Combine all models
        combined = xr.concat(datasets, dim='model')

        return combined

    def get_model_metadata(self, model_name: str) -> Dict:
        """
        Get metadata for a specific model.

        Args:
            model_name: Name of the CMIP model

        Returns:
            Dictionary with model metadata
        """
        # This would load actual metadata from CMIP archives
        # For now, return basic information

        metadata = {
            'model_name': model_name,
            'institution': 'Unknown',
            'resolution': 'Unknown',
            'physics_version': 'Unknown'
        }

        return metadata

    def list_available_models(self, experiment: str) -> List[str]:
        """
        List all available models for a given experiment.

        Args:
            experiment: CMIP experiment name

        Returns:
            List of available model names
        """
        experiment_dir = self.data_dir / experiment

        if not experiment_dir.exists():
            # Return common CMIP6 models for reference
            return [
                'CESM2', 'GFDL-CM4', 'IPSL-CM6A', 'UKESM1',
                'CanESM5', 'MIROC6', 'MRI-ESM2', 'NorESM2',
                'ACCESS-CM2', 'CNRM-CM6', 'EC-Earth3'
            ]

        return self._find_available_models(experiment_dir)
