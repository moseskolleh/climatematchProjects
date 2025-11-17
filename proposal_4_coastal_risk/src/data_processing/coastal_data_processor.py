"""
Coastal Data Processor

Main class for processing coastal hazard and vulnerability data with
hierarchical approach (global priors + local refinement).
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings


class CoastalDataProcessor:
    """
    Process coastal hazard and vulnerability data for risk assessment.

    Implements hierarchical data strategy:
    - Tier 1: Global datasets (satellite, reanalysis)
    - Tier 2: Regional datasets (tide gauges, local observations)
    - Tier 3: Community-reported data
    - Tier 4: Synthetic data when observations unavailable

    Parameters
    ----------
    city : str
        Target coastal city
    data_dir : str or Path
        Directory containing raw data
    """

    def __init__(self, city: str, data_dir: Optional[str] = None):
        self.city = city
        self.data_dir = Path(data_dir) if data_dir else Path('data/raw')

        # City coordinates (replace with database)
        self.city_coords = self._get_city_coordinates(city)

        # Data availability flags
        self.has_tide_gauge = False
        self.has_local_dem = False
        self.data_quality_score = 0.0

    def _get_city_coordinates(self, city: str) -> Tuple[float, float]:
        """Get latitude, longitude for city."""
        coords = {
            'Lagos': (6.4541, 3.3947),
            'Mombasa': (-4.0435, 39.6682),
            'Dakar': (14.7167, -17.4677),
            'Maputo': (-25.9655, 32.5832)
        }
        return coords.get(city, (0.0, 0.0))

    def load_hazard_data(
        self,
        start_year: int = 2000,
        end_year: int = 2023,
        variables: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load historical hazard data with quality control.

        Parameters
        ----------
        start_year : int
            Start year for data
        end_year : int
            End year for data
        variables : List[str], optional
            Specific variables to load

        Returns
        -------
        pd.DataFrame
            Processed hazard data with quality flags
        """
        if variables is None:
            variables = ['sea_level', 'storm_surge', 'precipitation', 'wave_height']

        # Try to load local observations first
        hazard_data = self._load_local_observations(start_year, end_year, variables)

        if hazard_data.empty:
            # Fallback to global datasets
            warnings.warn(f"No local data for {self.city}, using global datasets")
            hazard_data = self._load_global_datasets(start_year, end_year, variables)

        # Quality control
        hazard_data = self._quality_control(hazard_data)

        # Fill gaps
        hazard_data = self._fill_data_gaps(hazard_data, variables)

        # Calculate quality score
        self.data_quality_score = self._assess_data_quality(hazard_data)

        return hazard_data

    def _load_local_observations(
        self,
        start_year: int,
        end_year: int,
        variables: List[str]
    ) -> pd.DataFrame:
        """Load data from local tide gauges and weather stations."""
        # Check for tide gauge data
        tide_gauge_file = self.data_dir / f'{self.city.lower()}_tide_gauge.csv'

        if tide_gauge_file.exists():
            self.has_tide_gauge = True
            data = pd.read_csv(tide_gauge_file, parse_dates=['date'])
            data = data[(data['date'].dt.year >= start_year) &
                       (data['date'].dt.year <= end_year)]
            return data
        else:
            return pd.DataFrame()

    def _load_global_datasets(
        self,
        start_year: int,
        end_year: int,
        variables: List[str]
    ) -> pd.DataFrame:
        """
        Load global datasets (satellite altimetry, reanalysis).

        Uses:
        - ERA5 for atmospheric variables
        - CMEMS for sea level and waves
        - CHIRPS for precipitation
        """
        lat, lon = self.city_coords
        data_dict = {'date': pd.date_range(f'{start_year}-01-01',
                                           f'{end_year}-12-31',
                                           freq='D')}

        # Simulate global dataset extraction
        # In practice, use actual API calls to Copernicus, NASA, etc.
        n_days = len(data_dict['date'])

        if 'sea_level' in variables:
            # Simulate sea level trend + variability
            trend = np.linspace(0, 50, n_days)  # 50mm rise over period
            seasonal = 50 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
            noise = np.random.normal(0, 20, n_days)
            data_dict['sea_level'] = trend + seasonal + noise

        if 'storm_surge' in variables:
            # Simulate storm surge events
            base = np.random.gamma(0.5, 0.1, n_days)
            # Add extreme events
            n_storms = int(n_days / 365.25 * 3)  # 3 per year
            storm_days = np.random.choice(n_days, n_storms, replace=False)
            data_dict['storm_surge'] = base
            data_dict['storm_surge'][storm_days] += np.random.gamma(2, 0.5, n_storms)

        if 'precipitation' in variables:
            # Simulate precipitation
            seasonal = 100 * (0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n_days) / 365.25))
            daily = np.random.gamma(2, seasonal / 10, n_days)
            data_dict['precipitation'] = daily

        if 'wave_height' in variables:
            # Simulate wave height
            seasonal = 1.5 + 0.5 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
            daily = np.random.gamma(2, seasonal / 2, n_days)
            data_dict['wave_height'] = daily

        return pd.DataFrame(data_dict)

    def _quality_control(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply quality control checks.

        - Remove outliers
        - Flag suspicious values
        - Check for consistency
        """
        data = data.copy()

        # Add quality flags
        data['quality_flag'] = 'good'

        # Check for outliers (> 4 sigma)
        for col in data.select_dtypes(include=[np.number]).columns:
            if col == 'date':
                continue

            mean = data[col].mean()
            std = data[col].std()

            outliers = np.abs(data[col] - mean) > 4 * std
            data.loc[outliers, 'quality_flag'] = 'outlier'
            data.loc[outliers, col] = np.nan

        # Check for physically impossible values
        if 'precipitation' in data.columns:
            data.loc[data['precipitation'] < 0, 'quality_flag'] = 'invalid'
            data.loc[data['precipitation'] < 0, 'precipitation'] = np.nan

        if 'wave_height' in data.columns:
            data.loc[data['wave_height'] < 0, 'quality_flag'] = 'invalid'
            data.loc[data['wave_height'] < 0, 'wave_height'] = np.nan
            data.loc[data['wave_height'] > 20, 'quality_flag'] = 'outlier'
            data.loc[data['wave_height'] > 20, 'wave_height'] = np.nan

        return data

    def _fill_data_gaps(
        self,
        data: pd.DataFrame,
        variables: List[str]
    ) -> pd.DataFrame:
        """
        Fill data gaps using interpolation and statistical methods.
        """
        data = data.copy()

        for var in variables:
            if var in data.columns:
                # Linear interpolation for short gaps (< 7 days)
                data[var] = data[var].interpolate(
                    method='linear',
                    limit=7,
                    limit_direction='both'
                )

                # For longer gaps, use climatological mean
                if data[var].isna().sum() > 0:
                    if 'date' in data.columns:
                        data['month'] = pd.to_datetime(data['date']).dt.month
                        monthly_mean = data.groupby('month')[var].transform('mean')
                        data[var] = data[var].fillna(monthly_mean)
                        data = data.drop('month', axis=1)

        return data

    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """
        Calculate overall data quality score (0-1).

        Based on:
        - Completeness
        - Temporal coverage
        - Source reliability
        """
        scores = []

        # Completeness score
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        completeness = 1 - data[numeric_cols].isna().sum().sum() / (len(data) * len(numeric_cols))
        scores.append(completeness)

        # Temporal coverage score
        if 'date' in data.columns:
            date_range = (pd.to_datetime(data['date'].max()) -
                         pd.to_datetime(data['date'].min())).days
            expected_range = 365.25 * 20  # 20 years
            coverage = min(date_range / expected_range, 1.0)
            scores.append(coverage)

        # Source reliability
        if self.has_tide_gauge:
            scores.append(0.9)  # High quality
        else:
            scores.append(0.6)  # Global dataset quality

        return np.mean(scores)

    def load_vulnerability_data(self) -> pd.DataFrame:
        """
        Load vulnerability and exposure data.

        Includes:
        - Population distribution
        - Building stock
        - Infrastructure
        - Socioeconomic indicators
        """
        # Simulate vulnerability data
        # In practice, load from OpenStreetMap, census, etc.

        data = {
            'zone_id': range(100),
            'population': np.random.lognormal(7, 1, 100),
            'building_count': np.random.poisson(50, 100),
            'median_income': np.random.lognormal(6, 0.8, 100),
            'infrastructure_value': np.random.lognormal(10, 1.5, 100)
        }

        return pd.DataFrame(data)

    def load_elevation_data(
        self,
        resolution: float = 30
    ) -> np.ndarray:
        """
        Load digital elevation model.

        Parameters
        ----------
        resolution : float
            Spatial resolution in meters

        Returns
        -------
        np.ndarray
            Elevation data
        """
        # Check for local DEM
        dem_file = self.data_dir / f'{self.city.lower()}_dem.tif'

        if dem_file.exists():
            import rasterio
            with rasterio.open(dem_file) as src:
                elevation = src.read(1)
            self.has_local_dem = True
            return elevation
        else:
            # Use global DEM (SRTM, ASTER, etc.)
            warnings.warn(f"No local DEM for {self.city}, using global DEM")
            return self._load_global_dem(resolution)

    def _load_global_dem(self, resolution: float) -> np.ndarray:
        """Load global DEM data (placeholder)."""
        # Simulate DEM with coastal slope
        size = int(10000 / resolution)  # 10km x 10km
        x = np.linspace(0, size, size)
        y = np.linspace(0, size, size)
        X, Y = np.meshgrid(x, y)

        # Coastal slope
        elevation = (X / size) * 20 - 2  # -2m to 18m

        # Add terrain features
        elevation += np.random.normal(0, 1, (size, size))

        return np.maximum(elevation, -2)  # Minimum -2m

    def get_data_report(self) -> Dict:
        """
        Generate data availability and quality report.

        Returns
        -------
        Dict
            Report with data sources, coverage, and quality metrics
        """
        report = {
            'city': self.city,
            'coordinates': self.city_coords,
            'data_quality_score': self.data_quality_score,
            'has_tide_gauge': self.has_tide_gauge,
            'has_local_dem': self.has_local_dem,
            'recommended_approach': self._recommend_approach()
        }

        return report

    def _recommend_approach(self) -> str:
        """Recommend analysis approach based on data availability."""
        if self.data_quality_score > 0.8:
            return "Full Bayesian analysis with local refinement"
        elif self.data_quality_score > 0.5:
            return "Hybrid approach with global priors and limited local data"
        else:
            return "Synthetic event generation with deep uncertainty analysis"
