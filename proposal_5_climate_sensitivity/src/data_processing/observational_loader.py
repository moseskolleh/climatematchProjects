"""
Observational Data Loader

Loads modern observational data for climate sensitivity constraints.
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class ObservationalDataLoader:
    """
    Loader for modern observational climate data.

    Handles historical temperature observations, satellite data,
    ocean heat content, and other constraint-relevant observations.
    """

    # Available observational datasets
    DATASETS = {
        'HadCRUT5': {
            'name': 'HadCRUT5 Temperature',
            'variable': 'temperature',
            'start_year': 1850,
            'spatial_resolution': '5x5 degrees'
        },
        'GISTEMP': {
            'name': 'GISS Surface Temperature Analysis',
            'variable': 'temperature',
            'start_year': 1880,
            'spatial_resolution': '2x2 degrees'
        },
        'CERES': {
            'name': 'CERES Energy Balance',
            'variable': 'radiation',
            'start_year': 2000,
            'spatial_resolution': '1x1 degrees'
        },
        'ARGO': {
            'name': 'Argo Ocean Heat Content',
            'variable': 'ocean_heat',
            'start_year': 2005,
            'spatial_resolution': 'irregular'
        },
    }

    def __init__(self, data_dir: Path):
        """
        Initialize observational data loader.

        Args:
            data_dir: Directory containing observational data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_observations(
        self,
        datasets: Optional[List[str]] = None,
        start_year: int = 1850,
        end_year: Optional[int] = None
    ) -> xr.Dataset:
        """
        Load observational data.

        Args:
            datasets: List of dataset names to load
            start_year: Start year for data
            end_year: End year for data (None = latest available)

        Returns:
            xarray Dataset with observational data
        """
        if datasets is None:
            datasets = list(self.DATASETS.keys())

        if end_year is None:
            end_year = 2023  # Current year - adjust as needed

        logger.info(
            f"Loading observational data from {start_year} to {end_year}"
        )

        # Check if data exists
        data_available = any(
            (self.data_dir / f"{ds}.nc").exists() for ds in datasets
        )

        if not data_available:
            logger.warning(
                "No observational data found. "
                "Generating synthetic data for demonstration."
            )
            return self._generate_synthetic_observations(
                datasets, start_year, end_year
            )

        # Load actual data
        loaded_data = []
        for dataset_name in datasets:
            data_file = self.data_dir / f"{dataset_name}.nc"

            if data_file.exists():
                ds = xr.open_dataset(data_file)
                # Filter by time range
                ds = ds.sel(time=slice(str(start_year), str(end_year)))
                loaded_data.append(ds)
            else:
                logger.debug(f"Dataset {dataset_name} not found")

        if not loaded_data:
            raise ValueError("No observational data could be loaded")

        # Merge datasets
        combined = xr.merge(loaded_data)

        return combined

    def _generate_synthetic_observations(
        self,
        datasets: List[str],
        start_year: int,
        end_year: int
    ) -> xr.Dataset:
        """
        Generate synthetic observational data for demonstration.

        Creates realistic historical warming trends with appropriate
        spatial patterns and uncertainties.
        """
        logger.info("Generating synthetic observational data")

        n_years = end_year - start_year + 1
        years = np.arange(start_year, end_year + 1)

        # Generate realistic warming trend
        # Observed warming: ~0.07°C/decade over 1850-2020
        # With acceleration in recent decades

        # Base trend
        trend = 0.007 * (years - start_year)  # °C per year

        # Add multi-decadal variability (ENSO, PDO, AMO)
        # AMO-like oscillation (60-80 year period)
        amo = 0.1 * np.sin(2 * np.pi * (years - start_year) / 70)

        # PDO-like oscillation (20-30 year period)
        pdo = 0.08 * np.sin(2 * np.pi * (years - start_year) / 25)

        # Volcanic forcing (simplified)
        volcanic = np.zeros(n_years)
        # Major eruptions: Krakatoa (1883), Pinatubo (1991)
        if 1883 in years:
            idx = np.where(years == 1883)[0][0]
            volcanic[idx:idx+3] = -0.3  # Cooling effect for 3 years
        if 1991 in years:
            idx = np.where(years == 1991)[0][0]
            volcanic[idx:idx+3] = -0.4

        # Combine signals
        global_temp_anomaly = trend + amo + pdo + volcanic

        # Add observational noise
        obs_noise = np.random.normal(0, 0.05, n_years)
        global_temp_anomaly += obs_noise

        # Create spatial pattern (for gridded data)
        n_lat = 36  # 5-degree resolution
        n_lon = 72
        lat = np.linspace(-87.5, 87.5, n_lat)
        lon = np.linspace(-177.5, 177.5, n_lon)

        # Amplification factors: more warming at high latitudes
        lat_amplification = 1 + 1.5 * (np.abs(lat) / 90) ** 2

        # Land-ocean contrast (more warming over land)
        # Simplified: assume more land at mid-latitudes
        land_ocean = 1 + 0.3 * np.exp(-((lat - 45) / 30) ** 2)

        # Create 3D temperature field
        temp_3d = (
            global_temp_anomaly[:, np.newaxis, np.newaxis] *
            lat_amplification[np.newaxis, :, np.newaxis] *
            land_ocean[np.newaxis, :, np.newaxis] +
            0.1 * np.random.randn(n_years, n_lat, n_lon)
        )

        # Calculate uncertainty (larger in early period)
        uncertainty = 0.15 / (1 + (years - start_year) / 50)

        # Build dataset
        data_vars = {
            'temperature': (
                ['time', 'lat', 'lon'],
                temp_3d,
                {
                    'long_name': 'Surface temperature anomaly',
                    'units': 'K',
                    'reference_period': '1850-1900'
                }
            ),
            'temperature_uncertainty': (
                ['time'],
                uncertainty,
                {
                    'long_name': 'Temperature uncertainty',
                    'units': 'K'
                }
            ),
            'global_mean_temperature': (
                ['time'],
                global_temp_anomaly,
                {
                    'long_name': 'Global mean temperature anomaly',
                    'units': 'K',
                    'reference_period': '1850-1900'
                }
            ),
        }

        # Add TOA radiation if CERES is requested
        if 'CERES' in datasets and start_year <= 2000:
            # CERES data available from 2000
            ceres_start = max(0, 2000 - start_year)
            n_ceres = n_years - ceres_start

            if n_ceres > 0:
                # Current Earth energy imbalance: ~0.7 W/m²
                # Increasing trend due to greenhouse forcing
                toa_imbalance = 0.5 + 0.01 * np.arange(n_ceres)
                toa_imbalance += np.random.normal(0, 0.1, n_ceres)

                # Pad with NaN for pre-CERES period
                toa_full = np.full(n_years, np.nan)
                toa_full[ceres_start:] = toa_imbalance

                data_vars['toa_net_flux'] = (
                    ['time'],
                    toa_full,
                    {
                        'long_name': 'TOA net radiative flux',
                        'units': 'W m-2',
                        'description': 'Positive = Earth energy gain'
                    }
                )

        # Add ocean heat content if ARGO is requested
        if 'ARGO' in datasets and start_year <= 2005:
            argo_start = max(0, 2005 - start_year)
            n_argo = n_years - argo_start

            if n_argo > 0:
                # Ocean heat content increasing trend
                # Roughly 0.7 W/m² * Earth surface area accumulated
                ohc_rate = 0.7 * 5.1e14  # W (J/s)
                ohc = np.cumsum(np.full(n_argo, ohc_rate * 365.25 * 24 * 3600))  # Joules

                # Add variability
                ohc += np.random.normal(0, 1e22, n_argo)

                # Pad with NaN for pre-ARGO period
                ohc_full = np.full(n_years, np.nan)
                ohc_full[argo_start:] = ohc

                data_vars['ocean_heat_content'] = (
                    ['time'],
                    ohc_full,
                    {
                        'long_name': 'Ocean heat content anomaly',
                        'units': 'J',
                        'reference_year': '2005'
                    }
                )

        # Create dataset
        ds = xr.Dataset(
            data_vars,
            coords={
                'time': years,
                'lat': lat,
                'lon': lon
            },
            attrs={
                'description': 'Synthetic observational climate data',
                'start_year': start_year,
                'end_year': end_year,
                'synthetic': True,
                'note': 'Generated for demonstration purposes'
            }
        )

        return ds

    def get_dataset_info(self, dataset_name: str) -> dict:
        """
        Get information about a specific observational dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with dataset information
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(self.DATASETS.keys())}"
            )

        return self.DATASETS[dataset_name].copy()

    def list_available_datasets(self) -> List[str]:
        """List all available observational datasets."""
        return list(self.DATASETS.keys())

    def calculate_warming_rate(
        self,
        data: xr.Dataset,
        start_year: int,
        end_year: int
    ) -> float:
        """
        Calculate linear warming rate over a specified period.

        Args:
            data: Dataset containing temperature data
            start_year: Start year for trend calculation
            end_year: End year for trend calculation

        Returns:
            Warming rate in °C per decade
        """
        # Select time period
        temp = data['global_mean_temperature'].sel(
            time=slice(start_year, end_year)
        )

        # Linear regression
        time_numeric = temp.time.values - temp.time.values[0]
        coefficients = np.polyfit(time_numeric, temp.values, deg=1)

        # Convert to °C per decade
        warming_rate = coefficients[0] * 10  # slope * 10 years

        return float(warming_rate)
