"""
Paleoclimate Data Loader

Loads proxy data from various paleoclimate periods for climate sensitivity constraints.
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PaleoclimateDataLoader:
    """
    Loader for paleoclimate proxy data.

    Handles data from various paleoclimate periods including LGM, MPWP,
    Last Interglacial, and PETM.
    """

    # Supported paleoclimate periods
    PERIODS = {
        'LGM': {
            'name': 'Last Glacial Maximum',
            'age': '~21 ka',
            'co2': 190,  # ppm
            'delta_temp': -5.0  # °C relative to pre-industrial
        },
        'MPWP': {
            'name': 'Mid-Pliocene Warm Period',
            'age': '~3.3-3.0 Ma',
            'co2': 400,
            'delta_temp': 2.5
        },
        'Last_Interglacial': {
            'name': 'Last Interglacial',
            'age': '~130-115 ka',
            'co2': 280,
            'delta_temp': 0.5
        },
        'PETM': {
            'name': 'Paleocene-Eocene Thermal Maximum',
            'age': '~56 Ma',
            'co2': 1000,
            'delta_temp': 5.0
        }
    }

    # Types of proxy data
    PROXY_TYPES = [
        'foram',  # Foraminiferal assemblages
        'ice_core',  # Ice core data
        'pollen',  # Pollen assemblages
        'alkenone',  # Alkenone-based temperature
        'mg_ca',  # Mg/Ca ratios
        'tex86',  # TEX86 paleotemperature proxy
    ]

    def __init__(self, data_dir: Path):
        """
        Initialize paleoclimate data loader.

        Args:
            data_dir: Directory containing paleoclimate data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_period(
        self,
        period: str,
        proxy_types: Optional[List[str]] = None
    ) -> Dict[str, xr.Dataset]:
        """
        Load proxy data for a specific paleoclimate period.

        Args:
            period: Period name (e.g., 'LGM', 'MPWP')
            proxy_types: Types of proxies to load

        Returns:
            Dictionary of datasets by proxy type
        """
        if period not in self.PERIODS:
            raise ValueError(
                f"Unknown period: {period}. "
                f"Available periods: {list(self.PERIODS.keys())}"
            )

        if proxy_types is None:
            proxy_types = self.PROXY_TYPES

        logger.info(f"Loading {period} data")

        period_dir = self.data_dir / period

        if not period_dir.exists():
            logger.warning(
                f"No data found for period '{period}'. "
                f"Generating synthetic proxy data for demonstration."
            )
            return self._generate_synthetic_proxy_data(period, proxy_types)

        # Load actual proxy data
        datasets = {}
        for proxy_type in proxy_types:
            proxy_file = period_dir / f"{proxy_type}.nc"

            if proxy_file.exists():
                datasets[proxy_type] = xr.open_dataset(proxy_file)
            else:
                logger.debug(f"Proxy type {proxy_type} not found for {period}")

        return datasets

    def _generate_synthetic_proxy_data(
        self,
        period: str,
        proxy_types: List[str]
    ) -> Dict[str, xr.Dataset]:
        """
        Generate synthetic proxy data for demonstration.

        Creates realistic proxy data with appropriate spatial distribution
        and uncertainties.
        """
        logger.info(f"Generating synthetic proxy data for {period}")

        period_info = self.PERIODS[period]
        datasets = {}

        for proxy_type in proxy_types:
            # Number of proxy sites (varies by type and period)
            if period == 'LGM':
                n_sites = np.random.randint(100, 200)
            elif period == 'MPWP':
                n_sites = np.random.randint(50, 100)
            else:
                n_sites = np.random.randint(30, 80)

            # Random geographic distribution
            lat = np.random.uniform(-80, 80, n_sites)
            lon = np.random.uniform(-180, 180, n_sites)

            # Temperature anomaly relative to pre-industrial
            # Base value from period + spatial variability + measurement uncertainty
            temp_anomaly = (
                period_info['delta_temp'] +
                np.random.normal(0, 1.5, n_sites) +  # Spatial variability
                np.random.normal(0, 0.5, n_sites)    # Measurement uncertainty
            )

            # Proxy-specific uncertainty
            if proxy_type in ['alkenone', 'tex86']:
                uncertainty = np.full(n_sites, 1.5)  # °C
            elif proxy_type == 'mg_ca':
                uncertainty = np.full(n_sites, 2.0)
            elif proxy_type == 'foram':
                uncertainty = np.full(n_sites, 1.0)
            else:
                uncertainty = np.full(n_sites, 2.5)

            # Quality flags (0-1, with 1 being highest quality)
            quality = np.random.beta(5, 2, n_sites)

            # Create dataset
            ds = xr.Dataset(
                {
                    'temperature_anomaly': (
                        ['site'],
                        temp_anomaly,
                        {
                            'long_name': 'Temperature anomaly relative to pre-industrial',
                            'units': 'K'
                        }
                    ),
                    'uncertainty': (
                        ['site'],
                        uncertainty,
                        {
                            'long_name': 'Temperature reconstruction uncertainty',
                            'units': 'K'
                        }
                    ),
                    'quality_flag': (
                        ['site'],
                        quality,
                        {
                            'long_name': 'Data quality flag',
                            'description': '0=poor, 1=excellent'
                        }
                    ),
                    'lat': (
                        ['site'],
                        lat,
                        {'long_name': 'Latitude', 'units': 'degrees_north'}
                    ),
                    'lon': (
                        ['site'],
                        lon,
                        {'long_name': 'Longitude', 'units': 'degrees_east'}
                    ),
                },
                coords={
                    'site': np.arange(n_sites)
                },
                attrs={
                    'period': period,
                    'period_name': period_info['name'],
                    'period_age': period_info['age'],
                    'co2_concentration': period_info['co2'],
                    'proxy_type': proxy_type,
                    'synthetic': True
                }
            )

            datasets[proxy_type] = ds

        return datasets

    def get_period_info(self, period: str) -> Dict:
        """
        Get information about a paleoclimate period.

        Args:
            period: Period name

        Returns:
            Dictionary with period information
        """
        if period not in self.PERIODS:
            raise ValueError(f"Unknown period: {period}")

        return self.PERIODS[period].copy()

    def list_available_periods(self) -> List[str]:
        """List all available paleoclimate periods."""
        return list(self.PERIODS.keys())

    def calculate_forcing(self, co2_paleo: float, co2_reference: float = 280) -> float:
        """
        Calculate radiative forcing from CO2 change.

        Uses standard formula: ΔF = 5.35 * ln(CO2/CO2_ref)

        Args:
            co2_paleo: CO2 concentration in paleoclimate period (ppm)
            co2_reference: Reference CO2 concentration (ppm), default is pre-industrial

        Returns:
            Radiative forcing in W/m²
        """
        forcing = 5.35 * np.log(co2_paleo / co2_reference)
        return forcing
