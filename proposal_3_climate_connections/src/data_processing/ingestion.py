"""
Data ingestion module for downloading climate reanalysis datasets.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources"""

    @abstractmethod
    def download(self, variables: List[str], start_date: datetime,
                 end_date: datetime, region: Dict) -> Path:
        """
        Download data for specified variables and time period.

        Args:
            variables: List of variable names
            start_date: Start date for data
            end_date: End date for data
            region: Dictionary with lat/lon bounds

        Returns:
            Path to downloaded file
        """
        pass

    @abstractmethod
    def get_available_variables(self) -> List[str]:
        """Get list of available variables from this data source"""
        pass

    @abstractmethod
    def check_data_availability(self, variables: List[str],
                                dates: List[datetime]) -> bool:
        """Check if data is available for specified variables and dates"""
        pass


class ERA5Downloader(DataSource):
    """Download ERA5 reanalysis data from Copernicus CDS"""

    def __init__(self, api_key: str = None, cache_dir: str = 'data/raw/ERA5'):
        """
        Initialize ERA5 downloader.

        Args:
            api_key: Copernicus CDS API key
            cache_dir: Directory for caching downloaded files
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ERA5 downloader initialized with cache: {self.cache_dir}")

    def download(self, variables: List[str], start_date: datetime,
                 end_date: datetime, region: Dict) -> Path:
        """
        Download ERA5 data.

        Args:
            variables: Variable names (e.g., 'geopotential', 'temperature')
            start_date: Start date
            end_date: End date
            region: Dict with 'north', 'south', 'east', 'west' keys

        Returns:
            Path to downloaded NetCDF file
        """
        logger.info(f"Downloading ERA5 data for {variables}")

        # Generate filename based on request
        filename = self._generate_filename(variables, start_date, end_date)
        output_file = self.cache_dir / filename

        # Check if already downloaded
        if output_file.exists():
            logger.info(f"File already exists: {output_file}")
            return output_file

        # Download using CDS API
        # NOTE: Actual implementation would use cdsapi library
        # This is a placeholder for the structure

        logger.info(f"Downloaded to: {output_file}")
        return output_file

    def get_available_variables(self) -> List[str]:
        """Get list of available ERA5 variables"""
        return [
            'geopotential',
            'temperature',
            'u_component_of_wind',
            'v_component_of_wind',
            'vertical_velocity',
            'specific_humidity',
        ]

    def check_data_availability(self, variables: List[str],
                                dates: List[datetime]) -> bool:
        """Check if ERA5 data is available"""
        # ERA5 is available from 1979 onwards
        earliest_date = datetime(1979, 1, 1)
        return all(d >= earliest_date for d in dates)

    def _generate_filename(self, variables: List[str],
                          start_date: datetime, end_date: datetime) -> str:
        """Generate filename for cached data"""
        var_str = '_'.join(sorted(variables))
        date_str = f"{start_date.strftime('%Y%m')}-{end_date.strftime('%Y%m')}"
        return f"ERA5_{var_str}_{date_str}.nc"


class MERRA2Downloader(DataSource):
    """Download MERRA-2 reanalysis data from NASA GES DISC"""

    def __init__(self, cache_dir: str = 'data/raw/MERRA2'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"MERRA-2 downloader initialized")

    def download(self, variables: List[str], start_date: datetime,
                 end_date: datetime, region: Dict) -> Path:
        """Download MERRA-2 data"""
        logger.info(f"Downloading MERRA-2 data for {variables}")
        # Placeholder implementation
        return self.cache_dir / "merra2_data.nc"

    def get_available_variables(self) -> List[str]:
        """Get available MERRA-2 variables"""
        return [
            'H',  # Geopotential height
            'T',  # Temperature
            'U',  # Zonal wind
            'V',  # Meridional wind
            'OMEGA',  # Vertical velocity
        ]

    def check_data_availability(self, variables: List[str],
                                dates: List[datetime]) -> bool:
        """Check MERRA-2 availability (from 1980)"""
        earliest_date = datetime(1980, 1, 1)
        return all(d >= earliest_date for d in dates)


class JRA55Downloader(DataSource):
    """Download JRA-55 reanalysis data from JMA"""

    def __init__(self, cache_dir: str = 'data/raw/JRA55'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"JRA-55 downloader initialized")

    def download(self, variables: List[str], start_date: datetime,
                 end_date: datetime, region: Dict) -> Path:
        """Download JRA-55 data"""
        logger.info(f"Downloading JRA-55 data for {variables}")
        # Placeholder implementation
        return self.cache_dir / "jra55_data.nc"

    def get_available_variables(self) -> List[str]:
        """Get available JRA-55 variables"""
        return [
            'hgt',  # Geopotential height
            'tmp',  # Temperature
            'ugrd',  # U-wind
            'vgrd',  # V-wind
        ]

    def check_data_availability(self, variables: List[str],
                                dates: List[datetime]) -> bool:
        """Check JRA-55 availability (from 1958)"""
        earliest_date = datetime(1958, 1, 1)
        return all(d >= earliest_date for d in dates)


class DataIngestionOrchestrator:
    """Orchestrate data downloads from multiple sources"""

    def __init__(self, config: Dict = None):
        """
        Initialize orchestrator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.downloaders = {
            'ERA5': ERA5Downloader(),
            'MERRA2': MERRA2Downloader(),
            'JRA55': JRA55Downloader()
        }
        logger.info("Data ingestion orchestrator initialized")

    def ingest_all_sources(self, variables: List[str], start_date: datetime,
                           end_date: datetime, region: Dict) -> Dict[str, Path]:
        """
        Download from all configured sources.

        Args:
            variables: Variables to download
            start_date: Start date
            end_date: End date
            region: Spatial region

        Returns:
            Dictionary mapping source name to file path
        """
        results = {}

        for source_name, downloader in self.downloaders.items():
            logger.info(f"Downloading from {source_name}")
            try:
                file_path = downloader.download(
                    variables, start_date, end_date, region
                )
                results[source_name] = file_path
            except Exception as e:
                logger.error(f"Failed to download from {source_name}: {e}")
                results[source_name] = None

        return results

    def ingest_single_source(self, source_name: str, variables: List[str],
                            start_date: datetime, end_date: datetime,
                            region: Dict) -> Path:
        """Download from a single source"""
        if source_name not in self.downloaders:
            raise ValueError(f"Unknown data source: {source_name}")

        downloader = self.downloaders[source_name]
        return downloader.download(variables, start_date, end_date, region)
