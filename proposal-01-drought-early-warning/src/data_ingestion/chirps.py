"""
CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data) Downloader

This module handles downloading CHIRPS precipitation data from UCSB servers.
"""

import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import logging

import rasterio
import numpy as np
import xarray as xr


logger = logging.getLogger(__name__)


class CHIRPSDownloader:
    """Download and process CHIRPS precipitation data."""

    BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0"

    def __init__(self, output_dir: str = "./data/raw/chirps"):
        """
        Initialize CHIRPS downloader.

        Parameters
        ----------
        output_dir : str
            Directory to save downloaded files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_daily(
        self,
        start_date: datetime,
        end_date: datetime,
        region: str = "africa",
        resolution: str = "p05"
    ) -> List[Path]:
        """
        Download daily CHIRPS data for a date range.

        Parameters
        ----------
        start_date : datetime
            Start date for download
        end_date : datetime
            End date for download
        region : str
            Region to download ('africa', 'global')
        resolution : str
            Spatial resolution ('p05' for 0.05°, 'p25' for 0.25°)

        Returns
        -------
        List[Path]
            List of downloaded file paths
        """
        downloaded_files = []
        current_date = start_date

        while current_date <= end_date:
            try:
                file_path = self._download_single_day(current_date, region, resolution)
                if file_path:
                    downloaded_files.append(file_path)
                    logger.info(f"Downloaded CHIRPS for {current_date.strftime('%Y-%m-%d')}")
            except Exception as e:
                logger.error(f"Failed to download CHIRPS for {current_date}: {e}")

            current_date += timedelta(days=1)

        return downloaded_files

    def _download_single_day(
        self,
        date: datetime,
        region: str,
        resolution: str
    ) -> Optional[Path]:
        """
        Download CHIRPS data for a single day.

        Parameters
        ----------
        date : datetime
            Date to download
        region : str
            Region to download
        resolution : str
            Spatial resolution

        Returns
        -------
        Optional[Path]
            Path to downloaded file, or None if failed
        """
        year = date.year
        file_name = f"chirps-v2.0.{date.strftime('%Y.%m.%d')}.tif"
        url = f"{self.BASE_URL}/{region}_daily/tifs/{resolution}/{year}/{file_name}"

        output_path = self.output_dir / str(year) / file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file already exists
        if output_path.exists():
            logger.debug(f"File already exists: {output_path}")
            return output_path

        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)

            # Verify the downloaded file
            if self._verify_tif(output_path):
                return output_path
            else:
                logger.error(f"Downloaded file is invalid: {output_path}")
                output_path.unlink()  # Delete invalid file
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed for {date}: {e}")
            return None

    def _verify_tif(self, file_path: Path) -> bool:
        """
        Verify that a downloaded GeoTIFF file is valid.

        Parameters
        ----------
        file_path : Path
            Path to the file to verify

        Returns
        -------
        bool
            True if valid, False otherwise
        """
        try:
            with rasterio.open(file_path) as src:
                # Check basic properties
                if src.count < 1:
                    return False
                # Try reading a small sample
                _ = src.read(1, window=((0, 10), (0, 10)))
            return True
        except Exception as e:
            logger.error(f"File verification failed: {e}")
            return False

    def load_to_xarray(
        self,
        files: List[Path],
        bbox: Optional[tuple] = None
    ) -> xr.Dataset:
        """
        Load downloaded CHIRPS files into an xarray Dataset.

        Parameters
        ----------
        files : List[Path]
            List of file paths to load
        bbox : Optional[tuple]
            Bounding box (min_lon, min_lat, max_lon, max_lat) to subset

        Returns
        -------
        xr.Dataset
            Dataset with precipitation data
        """
        data_arrays = []

        for file_path in sorted(files):
            # Extract date from filename
            date_str = file_path.stem.split('.')[1:4]
            date = datetime.strptime('.'.join(date_str), '%Y.%m.%d')

            with rasterio.open(file_path) as src:
                data = src.read(1)
                transform = src.transform
                crs = src.crs

                # Get coordinates
                height, width = data.shape
                x = np.arange(width) * transform.a + transform.c
                y = np.arange(height) * transform.e + transform.f

                # Create DataArray
                da = xr.DataArray(
                    data,
                    coords={'y': y, 'x': x, 'time': date},
                    dims=['y', 'x']
                )
                data_arrays.append(da)

        # Combine along time dimension
        ds = xr.concat(data_arrays, dim='time')

        # Create Dataset
        dataset = xr.Dataset({
            'precipitation': ds
        })

        # Add metadata
        dataset.attrs['source'] = 'CHIRPS v2.0'
        dataset.attrs['units'] = 'mm/day'
        dataset.attrs['crs'] = str(crs)

        # Subset by bounding box if provided
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            dataset = dataset.sel(
                x=slice(min_lon, max_lon),
                y=slice(max_lat, min_lat)  # y is typically descending
            )

        return dataset

    def calculate_anomaly(
        self,
        data: xr.Dataset,
        baseline_period: tuple = (1981, 2010)
    ) -> xr.Dataset:
        """
        Calculate precipitation anomaly relative to baseline.

        Parameters
        ----------
        data : xr.Dataset
            Dataset with precipitation data
        baseline_period : tuple
            Start and end years for baseline period

        Returns
        -------
        xr.Dataset
            Dataset with anomaly added
        """
        # Extract baseline period
        baseline = data.sel(
            time=slice(f"{baseline_period[0]}-01-01", f"{baseline_period[1]}-12-31")
        )

        # Calculate climatology (mean for each day of year)
        climatology = baseline.groupby('time.dayofyear').mean('time')

        # Calculate anomaly
        anomaly = data.groupby('time.dayofyear') - climatology

        # Add to dataset
        data['precipitation_anomaly'] = anomaly['precipitation']
        data['precipitation_anomaly'].attrs['units'] = 'mm/day'
        data['precipitation_anomaly'].attrs['description'] = 'Anomaly from climatology'

        return data


def main():
    """Example usage of CHIRPSDownloader."""
    # Initialize downloader
    downloader = CHIRPSDownloader(output_dir="./data/raw/chirps")

    # Download last 7 days
    end_date = datetime.now() - timedelta(days=3)  # CHIRPS has ~3 day latency
    start_date = end_date - timedelta(days=7)

    logger.info(f"Downloading CHIRPS from {start_date} to {end_date}")
    files = downloader.download_daily(start_date, end_date)

    if files:
        # Load to xarray
        logger.info("Loading data to xarray...")
        # East Africa bounding box
        bbox = (33.0, -5.0, 42.0, 15.0)  # (min_lon, min_lat, max_lon, max_lat)
        ds = downloader.load_to_xarray(files, bbox=bbox)

        # Save to NetCDF
        output_file = "./data/processed/chirps_recent.nc"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        ds.to_netcdf(output_file)
        logger.info(f"Saved to {output_file}")

        # Print summary
        print(ds)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
