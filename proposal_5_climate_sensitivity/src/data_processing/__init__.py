"""
Data Processing Module

Handles ingestion, preprocessing, and harmonization of climate data from various sources.
"""

from .climate_data_processor import ClimateDataProcessor
from .cmip_loader import CMIPDataLoader
from .paleoclimate_loader import PaleoclimateDataLoader
from .observational_loader import ObservationalDataLoader

__all__ = [
    'ClimateDataProcessor',
    'CMIPDataLoader',
    'PaleoclimateDataLoader',
    'ObservationalDataLoader'
]
