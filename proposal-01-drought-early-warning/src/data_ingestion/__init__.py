"""
Data ingestion module for drought early warning system.

This module handles downloading and initial processing of data from various sources:
- Satellite products (CHIRPS, MODIS, GRACE, ERA5)
- National meteorological stations
- Community observers
- Alternative data sources
"""

from .chirps import CHIRPSDownloader
from .modis import MODISDownloader
from .era5 import ERA5Downloader
from .stations import StationDataCollector
from .quality_control import QualityController

__all__ = [
    "CHIRPSDownloader",
    "MODISDownloader",
    "ERA5Downloader",
    "StationDataCollector",
    "QualityController",
]
