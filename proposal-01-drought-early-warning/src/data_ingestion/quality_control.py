"""
Quality Control Module

Implements automated quality control checks for incoming data.
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class QualityFlag(Enum):
    """Quality flags for data."""
    MISSING = 0
    POOR = 1
    MODERATE = 2
    GOOD = 3
    EXCELLENT = 4


@dataclass
class QualityReport:
    """Container for quality control results."""
    flags: np.ndarray
    issues: List[str]
    pass_rate: float
    statistics: Dict[str, float]


class QualityController:
    """Automated quality control for climate data."""

    def __init__(self):
        """Initialize quality controller with default parameters."""
        # Physical bounds for different variables
        self.bounds = {
            'precipitation': (0, 500),  # mm/day
            'temperature': (-50, 60),    # Celsius
            'temperature_min': (-60, 50),
            'temperature_max': (-40, 60),
            'humidity': (0, 100),        # %
            'wind_speed': (0, 50),       # m/s
        }

        # Temporal consistency thresholds (max change per day)
        self.max_daily_change = {
            'precipitation': None,  # Can change dramatically
            'temperature': 20,       # Celsius
            'humidity': 50,          # %
        }

    def validate_data(
        self,
        data: xr.Dataset,
        variable: str,
        checks: Optional[List[str]] = None
    ) -> QualityReport:
        """
        Run quality control checks on data.

        Parameters
        ----------
        data : xr.Dataset
            Dataset to validate
        variable : str
            Variable name to check
        checks : Optional[List[str]]
            List of checks to run. If None, run all checks.

        Returns
        -------
        QualityReport
            Quality control results
        """
        if checks is None:
            checks = [
                'missing',
                'range',
                'temporal_consistency',
                'spatial_coherence'
            ]

        # Initialize flags as GOOD
        flags = np.full(data[variable].shape, QualityFlag.GOOD.value, dtype=int)
        issues = []
        statistics = {}

        # Run each check
        if 'missing' in checks:
            flags, issue = self._check_missing(data[variable], flags)
            if issue:
                issues.append(issue)

        if 'range' in checks:
            flags, issue = self._check_range(data[variable], variable, flags)
            if issue:
                issues.append(issue)

        if 'temporal_consistency' in checks:
            flags, issue = self._check_temporal_consistency(
                data[variable], variable, flags
            )
            if issue:
                issues.append(issue)

        if 'spatial_coherence' in checks:
            flags, issue = self._check_spatial_coherence(data[variable], flags)
            if issue:
                issues.append(issue)

        # Calculate pass rate (GOOD or EXCELLENT)
        pass_rate = np.sum(flags >= QualityFlag.MODERATE.value) / flags.size

        # Calculate statistics
        statistics = {
            'missing_pct': np.sum(flags == QualityFlag.MISSING.value) / flags.size * 100,
            'poor_pct': np.sum(flags == QualityFlag.POOR.value) / flags.size * 100,
            'moderate_pct': np.sum(flags == QualityFlag.MODERATE.value) / flags.size * 100,
            'good_pct': np.sum(flags == QualityFlag.GOOD.value) / flags.size * 100,
        }

        return QualityReport(
            flags=flags,
            issues=issues,
            pass_rate=pass_rate,
            statistics=statistics
        )

    def _check_missing(
        self,
        data: xr.DataArray,
        flags: np.ndarray
    ) -> Tuple[np.ndarray, Optional[str]]:
        """Check for missing data."""
        missing_mask = np.isnan(data.values)
        missing_count = np.sum(missing_mask)

        if missing_count > 0:
            flags[missing_mask] = QualityFlag.MISSING.value
            missing_pct = missing_count / data.size * 100
            issue = f"Missing data: {missing_count} values ({missing_pct:.2f}%)"
            logger.warning(issue)
            return flags, issue

        return flags, None

    def _check_range(
        self,
        data: xr.DataArray,
        variable: str,
        flags: np.ndarray
    ) -> Tuple[np.ndarray, Optional[str]]:
        """Check if values are within physical bounds."""
        if variable not in self.bounds:
            logger.warning(f"No bounds defined for {variable}, skipping range check")
            return flags, None

        min_val, max_val = self.bounds[variable]
        out_of_range = (data.values < min_val) | (data.values > max_val)
        out_of_range = out_of_range & ~np.isnan(data.values)  # Exclude missing

        if np.any(out_of_range):
            flags[out_of_range] = QualityFlag.POOR.value
            count = np.sum(out_of_range)
            issue = f"Out of range: {count} values outside [{min_val}, {max_val}]"
            logger.warning(issue)
            return flags, issue

        return flags, None

    def _check_temporal_consistency(
        self,
        data: xr.DataArray,
        variable: str,
        flags: np.ndarray
    ) -> Tuple[np.ndarray, Optional[str]]:
        """Check for unrealistic temporal changes."""
        if 'time' not in data.dims:
            return flags, None

        if variable not in self.max_daily_change or self.max_daily_change[variable] is None:
            return flags, None

        max_change = self.max_daily_change[variable]

        # Calculate temporal differences
        diff = np.abs(np.diff(data.values, axis=data.dims.index('time')))

        # Find large changes
        large_changes = diff > max_change

        if np.any(large_changes):
            # Mark both the value and the next value as suspect
            for i in range(large_changes.shape[0]):
                flags[i][large_changes[i]] = min(
                    flags[i][large_changes[i]],
                    QualityFlag.MODERATE.value
                )
                if i + 1 < flags.shape[0]:
                    flags[i+1][large_changes[i]] = min(
                        flags[i+1][large_changes[i]],
                        QualityFlag.MODERATE.value
                    )

            count = np.sum(large_changes)
            issue = f"Large temporal changes: {count} instances > {max_change}"
            logger.warning(issue)
            return flags, issue

        return flags, None

    def _check_spatial_coherence(
        self,
        data: xr.DataArray,
        flags: np.ndarray
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Check for spatial outliers.

        Uses a simple approach: compare each point to its neighbors.
        """
        if len(data.dims) < 2:
            return flags, None

        # Get spatial dimensions (assume last two dimensions)
        spatial_dims = data.dims[-2:]

        # Calculate local mean (3x3 window)
        from scipy.ndimage import uniform_filter

        values = data.values
        if len(values.shape) > 2:
            # Handle time dimension
            outliers = np.zeros(values.shape, dtype=bool)
            for t in range(values.shape[0]):
                local_mean = uniform_filter(values[t], size=3, mode='constant', cval=np.nan)
                local_std = self._local_std(values[t], size=3)

                # Flag values more than 3 std from local mean
                outliers[t] = np.abs(values[t] - local_mean) > 3 * local_std
        else:
            local_mean = uniform_filter(values, size=3, mode='constant', cval=np.nan)
            local_std = self._local_std(values, size=3)
            outliers = np.abs(values - local_mean) > 3 * local_std

        outliers = outliers & ~np.isnan(values)  # Exclude missing

        if np.any(outliers):
            flags[outliers] = min(flags[outliers], QualityFlag.MODERATE.value)
            count = np.sum(outliers)
            issue = f"Spatial outliers: {count} values differ significantly from neighbors"
            logger.info(issue)
            return flags, issue

        return flags, None

    def _local_std(self, data: np.ndarray, size: int = 3) -> np.ndarray:
        """Calculate local standard deviation."""
        from scipy.ndimage import uniform_filter

        # Local variance = E[X^2] - E[X]^2
        local_mean = uniform_filter(data, size=size, mode='constant', cval=np.nan)
        local_mean_sq = uniform_filter(data**2, size=size, mode='constant', cval=np.nan)
        local_var = local_mean_sq - local_mean**2
        local_var[local_var < 0] = 0  # Handle numerical errors

        return np.sqrt(local_var)

    def apply_flags(
        self,
        data: xr.Dataset,
        variable: str,
        flags: np.ndarray,
        min_quality: QualityFlag = QualityFlag.MODERATE
    ) -> xr.Dataset:
        """
        Apply quality flags to data.

        Parameters
        ----------
        data : xr.Dataset
            Dataset to filter
        variable : str
            Variable to filter
        flags : np.ndarray
            Quality flags
        min_quality : QualityFlag
            Minimum acceptable quality level

        Returns
        -------
        xr.Dataset
            Filtered dataset with quality flags added
        """
        # Create a copy
        filtered = data.copy()

        # Mask low-quality data
        mask = flags < min_quality.value
        filtered[variable] = filtered[variable].where(~mask)

        # Add quality flags to dataset
        flags_da = xr.DataArray(
            flags,
            coords=data[variable].coords,
            dims=data[variable].dims,
            name=f'{variable}_quality'
        )
        filtered[f'{variable}_quality'] = flags_da

        return filtered


def main():
    """Example usage of QualityController."""
    # Create sample data
    np.random.seed(42)

    time = np.arange(30)
    lat = np.arange(50)
    lon = np.arange(50)

    # Realistic precipitation data with some issues
    data = np.random.gamma(2, 2, size=(30, 50, 50))  # Gamma distribution for precip

    # Introduce issues
    data[5, 10, 10] = 600  # Out of range
    data[15, :, :] = np.nan  # Missing day
    data[20, 25, 25] = 100  # Spatial outlier

    # Create dataset
    ds = xr.Dataset({
        'precipitation': (['time', 'lat', 'lon'], data)
    }, coords={
        'time': time,
        'lat': lat,
        'lon': lon
    })

    # Run quality control
    qc = QualityController()
    report = qc.validate_data(ds, 'precipitation')

    print("Quality Control Report")
    print("=" * 50)
    print(f"Pass rate: {report.pass_rate:.2%}")
    print(f"\nIssues found:")
    for issue in report.issues:
        print(f"  - {issue}")

    print(f"\nStatistics:")
    for key, value in report.statistics.items():
        print(f"  {key}: {value:.2f}%")

    # Apply filters
    filtered_ds = qc.apply_flags(ds, 'precipitation', report.flags)
    print(f"\nOriginal data size: {ds.precipitation.size}")
    print(f"Filtered data valid count: {np.sum(~np.isnan(filtered_ds.precipitation.values))}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
