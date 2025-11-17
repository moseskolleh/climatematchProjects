# Technical Architecture: Climate Teleconnection Discovery System

## 1. System Overview

This document describes the technical architecture for the Theory-Guided Discovery of Climate System Connections project. The system is designed to be scalable, maintainable, and operational, following software engineering best practices for scientific computing.

### 1.1 Design Principles

- **Modularity**: Clear separation of concerns with well-defined interfaces
- **Reproducibility**: All analyses fully reproducible with version control
- **Scalability**: Can handle increasing data volumes and computational demands
- **Maintainability**: Clean code, comprehensive documentation, automated testing
- **Operationalization**: Transition from research to operational system

### 1.2 Technology Stack

```
Data Layer:          NetCDF4, Zarr, HDF5
Processing:          Python 3.10+, NumPy, SciPy, pandas, xarray
Analysis:            scikit-learn, statsmodels, PyTorch
Visualization:       Matplotlib, Cartopy, Plotly, Dash
Cloud Computing:     AWS S3, EC2, or Google Cloud Platform
Orchestration:       Apache Airflow
Version Control:     Git, GitHub
Testing:             pytest, unittest
Documentation:       Sphinx, Read the Docs
Deployment:          Docker, Kubernetes
Monitoring:          Prometheus, Grafana
```

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Web Dashboard│  │ API Endpoints│  │ CLI Tools    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Discovery   │  │  Validation  │  │Interpretation│         │
│  │   Engine     │  │   Framework  │  │   Toolkit    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────────┐
│                       Core Libraries                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Causal     │  │   Physical   │  │  Statistical │         │
│  │  Discovery   │  │  Constraints │  │   Methods    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Data Ingest  │  │ Preprocessing│  │    Storage   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Detailed Component Architecture

## 3. Data Layer

### 3.1 Data Ingestion Module

**Purpose**: Acquire climate data from multiple sources

**Components**:
- `ReanalysisDownloader`: Downloads ERA5, MERRA-2, JRA-55
- `CMIPDownloader`: Acquires climate model data
- `ObservationalDataFetcher`: Gets station and satellite observations

**Key Classes**:

```python
# src/data_processing/ingestion.py

class DataSource(ABC):
    """Abstract base class for data sources"""

    @abstractmethod
    def download(self, variables, start_date, end_date, region):
        pass

    @abstractmethod
    def get_available_variables(self):
        pass

    @abstractmethod
    def check_data_availability(self, variables, dates):
        pass


class ERA5Downloader(DataSource):
    """Download ERA5 reanalysis data from Copernicus CDS"""

    def __init__(self, api_key):
        self.client = cdsapi.Client(key=api_key)
        self.cache_dir = Path('data/raw/ERA5')

    def download(self, variables, start_date, end_date, region):
        """
        Download ERA5 data

        Args:
            variables: List of variable names
            start_date: Start date (datetime)
            end_date: End date (datetime)
            region: Dict with lat/lon bounds

        Returns:
            Path to downloaded file
        """
        request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variables,
            'year': list(range(start_date.year, end_date.year + 1)),
            'month': list(range(1, 13)),
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'area': [region['north'], region['west'],
                    region['south'], region['east']],
        }

        output_file = self.cache_dir / self._generate_filename(request)

        if not output_file.exists():
            self.client.retrieve('reanalysis-era5-pressure-levels',
                               request, output_file)

        return output_file


class DataIngestionOrchestrator:
    """Orchestrate data downloads from multiple sources"""

    def __init__(self):
        self.downloaders = {
            'ERA5': ERA5Downloader(),
            'MERRA2': MERRA2Downloader(),
            'JRA55': JRA55Downloader()
        }

    def ingest_all_sources(self, config):
        """Download from all configured sources"""
        for source_name, downloader in self.downloaders.items():
            logger.info(f"Downloading from {source_name}")
            downloader.download(**config)
```

### 3.2 Preprocessing Module

**Purpose**: Quality control, gap filling, standardization

**Key Components**:

```python
# src/data_processing/preprocessing.py

class PreprocessingPipeline:
    """Complete preprocessing pipeline for climate data"""

    def __init__(self, config):
        self.qc = QualityControl(config.qc_params)
        self.interpolator = GapFiller(config.interpolation_method)
        self.standardizer = DataStandardizer()

    def process(self, raw_data):
        """
        Apply full preprocessing pipeline

        Args:
            raw_data: xarray.Dataset with raw climate data

        Returns:
            Preprocessed xarray.Dataset
        """
        # Quality control
        qc_data = self.qc.apply(raw_data)

        # Gap filling
        filled_data = self.interpolator.fill_gaps(qc_data)

        # Standardization
        standardized = self.standardizer.standardize(filled_data)

        # Detrending
        detrended = self.detrend(standardized)

        # Deseasonalization
        anomalies = self.remove_seasonal_cycle(detrended)

        return anomalies

    def detrend(self, data):
        """Remove linear trend and low-frequency variability"""
        from scipy import signal

        detrended = xr.apply_ufunc(
            signal.detrend,
            data,
            input_core_dims=[['time']],
            output_core_dims=[['time']],
            vectorize=True,
            dask='parallelized'
        )

        return detrended

    def remove_seasonal_cycle(self, data, baseline_period=('1981', '2010')):
        """Calculate and remove climatological seasonal cycle"""
        # Calculate climatology
        climatology = (
            data.sel(time=slice(*baseline_period))
            .groupby('time.month')
            .mean('time')
        )

        # Remove seasonal cycle
        anomalies = data.groupby('time.month') - climatology

        return anomalies


class QualityControl:
    """Quality control checks for climate data"""

    def apply(self, data):
        """Run all quality control checks"""
        # Check for missing values
        self.check_missing_values(data)

        # Check for physically unrealistic values
        self.check_physical_bounds(data)

        # Check temporal continuity
        self.check_temporal_continuity(data)

        # Check spatial consistency
        self.check_spatial_consistency(data)

        return data

    def check_physical_bounds(self, data):
        """Verify values are within physical limits"""
        bounds = {
            'temperature': (150, 350),  # Kelvin
            'pressure': (0, 110000),    # Pa
            'geopotential': (-1000, 35000),  # m
            'wind': (-100, 100),        # m/s
        }

        for var in data.data_vars:
            if var in bounds:
                lower, upper = bounds[var]
                invalid = (data[var] < lower) | (data[var] > upper)

                if invalid.any():
                    logger.warning(
                        f"{invalid.sum().values} invalid values in {var}"
                    )
                    # Replace with NaN for gap filling
                    data[var] = data[var].where(~invalid)
```

### 3.3 Data Storage

**Purpose**: Efficient storage and retrieval of processed data

**Strategy**:
- **Raw data**: NetCDF files on S3/Cloud Storage
- **Processed data**: Zarr format for chunked, compressed storage
- **Results**: HDF5 for structured output
- **Metadata**: PostgreSQL database

```python
# src/data_processing/storage.py

class DataStore:
    """Manage storage and retrieval of climate data"""

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.raw_dir = self.base_path / 'raw'
        self.processed_dir = self.base_path / 'processed'
        self.results_dir = self.base_path / 'results'

    def save_processed(self, data, name):
        """Save processed data in Zarr format"""
        output_path = self.processed_dir / f"{name}.zarr"

        # Configure chunking for efficient access
        chunks = {
            'time': 12,  # Monthly chunks
            'lat': 50,
            'lon': 50
        }

        data.chunk(chunks).to_zarr(
            output_path,
            mode='w',
            consolidated=True
        )

        # Update metadata database
        self.update_metadata(name, output_path, data)

    def load_processed(self, name):
        """Load processed data from Zarr"""
        path = self.processed_dir / f"{name}.zarr"
        return xr.open_zarr(path, consolidated=True)
```

## 4. Core Libraries Layer

### 4.1 Physical Constraints Module

**Purpose**: Enforce physical consistency in discovered patterns

```python
# src/discovery/physical_constraints.py

class PhysicalConstraintEngine:
    """Validate patterns against physical principles"""

    def __init__(self):
        self.validators = [
            RossbyWaveValidator(),
            EnergyConservationValidator(),
            TimescaleValidator(),
            MomentumConservationValidator()
        ]

    def validate(self, pattern):
        """
        Check if pattern satisfies all physical constraints

        Args:
            pattern: TeleconnectionPattern object

        Returns:
            ValidationResult with pass/fail and diagnostics
        """
        results = []

        for validator in self.validators:
            result = validator.validate(pattern)
            results.append(result)

        overall_pass = all(r.passed for r in results)

        return ValidationResult(
            passed=overall_pass,
            individual_results=results,
            diagnostics=self._compile_diagnostics(results)
        )


class RossbyWaveValidator:
    """Validate consistency with Rossby wave dynamics"""

    def validate(self, pattern):
        """Check Rossby wave characteristics"""
        # Calculate observed phase speed
        phase_speed = self.calculate_phase_speed(pattern)

        # Calculate theoretical Rossby wave speed
        theoretical_speed = self.theoretical_rossby_speed(
            pattern.wavelength,
            pattern.latitude
        )

        # Check agreement (allow 50% tolerance)
        relative_error = abs(
            (phase_speed - theoretical_speed) / theoretical_speed
        )

        passed = relative_error < 0.5

        return ConstraintResult(
            constraint='rossby_wave',
            passed=passed,
            observed_value=phase_speed,
            expected_value=theoretical_speed,
            relative_error=relative_error
        )

    def theoretical_rossby_speed(self, wavelength, latitude):
        """Calculate theoretical Rossby wave phase speed"""
        beta = 2 * OMEGA * np.cos(np.radians(latitude)) / EARTH_RADIUS
        k = 2 * np.pi / wavelength

        return -beta / k**2


class EnergyConservationValidator:
    """Validate energy budget closure"""

    def validate(self, pattern):
        """Check if pattern conserves energy"""
        # Calculate energy budget components
        ke_tendency = self.kinetic_energy_tendency(pattern)
        ape_tendency = self.available_potential_energy_tendency(pattern)
        conversions = self.energy_conversions(pattern)
        boundary_fluxes = self.boundary_fluxes(pattern)

        # Budget equation: dKE/dt + dAPE/dt = conversions + fluxes
        lhs = ke_tendency + ape_tendency
        rhs = conversions + boundary_fluxes
        residual = lhs - rhs

        # Check closure (residual < 10% of total)
        total = abs(lhs) + abs(rhs)
        relative_residual = abs(residual) / total

        passed = relative_residual < 0.10

        return ConstraintResult(
            constraint='energy_conservation',
            passed=passed,
            residual=residual,
            relative_residual=relative_residual
        )
```

### 4.2 Causal Discovery Module

**Purpose**: Implement causal discovery algorithms

```python
# src/discovery/causal_discovery.py

class CausalDiscoveryEngine:
    """Apply multiple causal discovery methods"""

    def __init__(self, config):
        self.methods = {
            'granger': GrangerCausality(config.granger),
            'ccm': ConvergentCrosMapping(config.ccm),
            'transfer_entropy': TransferEntropy(config.te),
            'structural': StructuralCausalModel(config.scm)
        }

    def discover(self, source_data, target_data):
        """
        Apply all causal discovery methods

        Args:
            source_data: Time series of potential cause
            target_data: Time series of potential effect

        Returns:
            CausalityResults with findings from each method
        """
        results = {}

        for method_name, method in self.methods.items():
            results[method_name] = method.test_causality(
                source_data,
                target_data
            )

        # Aggregate results
        consensus = self.check_consensus(results)

        return CausalityResults(
            individual_results=results,
            consensus=consensus
        )

    def check_consensus(self, results):
        """Check if multiple methods agree"""
        significant = [
            r.significant for r in results.values()
        ]

        # Require agreement from ≥3 methods
        consensus = sum(significant) >= 3

        return consensus


class GrangerCausality:
    """Granger causality testing via VAR models"""

    def test_causality(self, X, Y, max_lag=12):
        """
        Test if X Granger-causes Y

        Args:
            X: Source time series
            Y: Target time series
            max_lag: Maximum lag to test

        Returns:
            GrangerResult
        """
        from statsmodels.tsa.api import VAR
        from statsmodels.tsa.stattools import grangercausalitytests

        # Prepare data
        data = pd.DataFrame({'Y': Y, 'X': X})

        # Test for each lag
        test_results = grangercausalitytests(
            data,
            max_lag,
            verbose=False
        )

        # Extract p-values
        p_values = [
            test_results[lag][0]['ssr_ftest'][1]
            for lag in range(1, max_lag + 1)
        ]

        # Optimal lag
        optimal_lag = np.argmin(p_values) + 1
        min_p_value = min(p_values)

        return GrangerResult(
            significant=min_p_value < 0.01,
            p_value=min_p_value,
            optimal_lag=optimal_lag,
            all_p_values=p_values
        )
```

### 4.3 Statistical Validation Module

**Purpose**: Rigorous statistical testing and validation

```python
# src/validation/statistical_tests.py

class StatisticalValidationFramework:
    """Comprehensive statistical validation"""

    def __init__(self):
        self.bootstrap = BootstrapTester()
        self.fdr = FDRControl()
        self.cross_validator = CrossValidator()

    def validate_discovery(self, discovery):
        """
        Full validation of a discovered teleconnection

        Args:
            discovery: TeleconnectionDiscovery object

        Returns:
            ValidationReport
        """
        # Bootstrap confidence intervals
        ci = self.bootstrap.confidence_interval(
            discovery.source,
            discovery.target,
            statistic=discovery.statistic
        )

        # Temporal stability
        stability = self.cross_validator.temporal_stability(
            discovery.source,
            discovery.target
        )

        # Cross-reanalysis validation
        reanalysis = self.cross_validator.cross_reanalysis(
            discovery,
            datasets=['ERA5', 'MERRA2', 'JRA55']
        )

        # Overall validation decision
        validated = (
            ci.excludes_zero and
            stability.is_stable and
            reanalysis.is_robust
        )

        return ValidationReport(
            validated=validated,
            confidence_interval=ci,
            stability=stability,
            reanalysis_validation=reanalysis
        )


class FDRControl:
    """False Discovery Rate control"""

    def control(self, p_values, alpha=0.01):
        """
        Apply Benjamini-Hochberg FDR control

        Args:
            p_values: Array of p-values
            alpha: FDR level

        Returns:
            Array of booleans indicating discoveries
        """
        m = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        # Critical values
        critical = (np.arange(1, m + 1) / m) * alpha

        # Find discoveries
        discoveries = sorted_p <= critical

        if discoveries.any():
            threshold_idx = np.where(discoveries)[0][-1]
            threshold = sorted_p[threshold_idx]
        else:
            threshold = 0

        # Create boolean array
        is_discovery = p_values <= threshold

        return FDRResult(
            discoveries=is_discovery,
            threshold=threshold,
            n_discoveries=is_discovery.sum()
        )
```

## 5. Application Layer

### 5.1 Discovery Engine

**Purpose**: Orchestrate teleconnection discovery workflow

```python
# src/discovery/discovery_engine.py

class TeleconnectionDiscoveryEngine:
    """Main engine for discovering teleconnections"""

    def __init__(self, config):
        self.data_store = DataStore(config.data_path)
        self.preprocessor = PreprocessingPipeline(config.preprocessing)
        self.constraints = PhysicalConstraintEngine()
        self.causal_discovery = CausalDiscoveryEngine(config.causal)
        self.validator = StatisticalValidationFramework()

    def run_discovery_campaign(self, source_regions, target_regions):
        """
        Run systematic discovery campaign

        Args:
            source_regions: List of source region definitions
            target_regions: List of target region definitions

        Returns:
            TeleconnectionCatalog
        """
        discoveries = []

        for source in source_regions:
            for target in target_regions:
                # Extract regional data
                source_data = self.extract_regional_index(source)
                target_data = self.extract_regional_index(target)

                # Test causality
                causality = self.causal_discovery.discover(
                    source_data,
                    target_data
                )

                if not causality.consensus:
                    continue

                # Create candidate pattern
                pattern = self.create_pattern(source, target, causality)

                # Check physical constraints
                physical_check = self.constraints.validate(pattern)

                if not physical_check.passed:
                    continue

                # Statistical validation
                validation = self.validator.validate_discovery(pattern)

                if validation.validated:
                    discoveries.append(pattern)
                    logger.info(f"Discovered: {pattern.name}")

        return TeleconnectionCatalog(discoveries)
```

### 5.2 Interpretation Toolkit

**Purpose**: Tools for physical interpretation

```python
# src/discovery/interpretation.py

class InterpretationToolkit:
    """Tools for mechanistic understanding"""

    def composite_analysis(self, index, fields, threshold=1.0):
        """Create composite maps for high/low index states"""
        # Implementation from methodology.md
        pass

    def energy_budget(self, pattern):
        """Diagnose energy pathways"""
        # Implementation from methodology.md
        pass

    def wave_activity_flux(self, pattern):
        """Calculate wave activity propagation"""
        # Takaya-Nakamura wave activity flux
        pass

    def moisture_transport(self, pattern):
        """Calculate moisture transport anomalies"""
        pass
```

## 6. Operational System

### 6.1 Real-Time Monitoring

**Purpose**: Monitor teleconnection indices in real-time

```python
# src/operational/monitoring.py

class TeleconnectionMonitor:
    """Monitor teleconnection indices in real-time"""

    def __init__(self, catalog):
        self.catalog = catalog
        self.data_fetcher = RealTimeDataFetcher()
        self.alert_system = AlertSystem()

    def update(self):
        """Update all indices with latest data"""
        # Fetch latest data
        latest_data = self.data_fetcher.get_latest()

        # Calculate indices
        for tc in self.catalog.teleconnections:
            current_value = tc.calculate_index(latest_data)

            # Check for significant anomalies
            if abs(current_value) > 1.5:  # 1.5 standard deviations
                self.alert_system.send_alert(tc, current_value)
```

### 6.2 Forecasting System

**Purpose**: Generate forecasts using teleconnections

```python
# src/operational/forecasting.py

class TeleconnectionForecaster:
    """Generate forecasts based on teleconnections"""

    def forecast(self, teleconnection, lead_time):
        """
        Generate forecast using teleconnection

        Args:
            teleconnection: Teleconnection object
            lead_time: Forecast lead time (months)

        Returns:
            Forecast object with mean, uncertainty
        """
        # Get current index value
        current_index = teleconnection.current_value()

        # Apply regression model
        forecast_mean = teleconnection.regression_model.predict(
            current_index,
            lead_time
        )

        # Uncertainty from validation
        forecast_std = teleconnection.validation_error[lead_time]

        return Forecast(
            mean=forecast_mean,
            std=forecast_std,
            lead_time=lead_time
        )
```

## 7. User Interface Layer

### 7.1 Web Dashboard

**Technology**: Plotly Dash

**Features**:
- Interactive maps of teleconnection patterns
- Time series of indices
- Forecasts with uncertainty
- Composite analysis viewer

### 7.2 API

**Technology**: FastAPI

**Endpoints**:
- `GET /teleconnections`: List all discovered teleconnections
- `GET /teleconnections/{id}`: Get details of specific teleconnection
- `GET /indices/current`: Current values of all indices
- `POST /forecast`: Generate forecast
- `GET /validation/{id}`: Get validation report

### 7.3 Command-Line Interface

**Technology**: Click

**Commands**:
```bash
# Discovery
tc-discover --source tropical_pacific --target sahel

# Validation
tc-validate --teleconnection enso_sahel --method bootstrap

# Monitoring
tc-monitor --update

# Forecasting
tc-forecast --teleconnection enso_sahel --lead-time 3
```

## 8. Testing Strategy

### 8.1 Unit Tests

Test individual components in isolation

```python
# tests/test_physical_constraints.py

def test_rossby_wave_validator():
    """Test Rossby wave speed calculation"""
    validator = RossbyWaveValidator()

    # Create synthetic wave with known speed
    pattern = create_synthetic_rossby_wave(
        wavelength=5000e3,  # 5000 km
        latitude=45
    )

    result = validator.validate(pattern)

    assert result.passed
    assert result.relative_error < 0.1
```

### 8.2 Integration Tests

Test component interactions

```python
# tests/test_integration.py

def test_full_discovery_pipeline():
    """Test complete discovery workflow"""
    # Load test data
    source_data, target_data = load_test_data()

    # Run discovery
    engine = TeleconnectionDiscoveryEngine(test_config)
    discoveries = engine.run_discovery_campaign(
        [test_source_region],
        [test_target_region]
    )

    # Should discover known ENSO-Sahel connection
    assert len(discoveries) >= 1
    assert 'enso_sahel' in [d.name for d in discoveries]
```

### 8.3 Validation Tests

Test against known results

```python
# tests/test_validation.py

def test_enso_sahel_reproduction():
    """Verify we can reproduce known ENSO-Sahel connection"""
    # Load historical data
    data = load_historical_data('1979', '2020')

    # Extract indices
    nino34 = calculate_nino34(data)
    sahel_rainfall = calculate_sahel_rainfall(data)

    # Test correlation
    corr, p_val = scipy.stats.pearsonr(nino34, sahel_rainfall)

    # Should match literature
    assert corr < -0.3  # Negative correlation
    assert p_val < 0.01  # Significant
```

## 9. Deployment

### 9.1 Docker Containerization

```dockerfile
# Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libnetcdf-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY config/ ./config/

# Run application
CMD ["python", "src/operational/monitoring.py"]
```

### 9.2 Kubernetes Orchestration

```yaml
# kubernetes/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: tc-monitoring
spec:
  replicas: 2
  selector:
    matchLabels:
      app: tc-monitoring
  template:
    metadata:
      labels:
        app: tc-monitoring
    spec:
      containers:
      - name: tc-monitoring
        image: climate-tc:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

## 10. Monitoring and Observability

### 10.1 Metrics

Using Prometheus to track:
- Data processing latency
- Discovery success rate
- Validation pass rate
- API response times
- System resource usage

### 10.2 Logging

Structured logging with levels:
- DEBUG: Detailed diagnostic information
- INFO: General operational messages
- WARNING: Potential issues
- ERROR: Errors requiring attention
- CRITICAL: System failures

```python
import logging
import structlog

logger = structlog.get_logger()

logger.info(
    "teleconnection_discovered",
    name="tropical_atlantic_sahel",
    correlation=0.65,
    p_value=0.001,
    validation_status="passed"
)
```

## 11. Security

### 11.1 Data Access Control
- API authentication via JWT tokens
- Role-based access control
- Data encryption at rest and in transit

### 11.2 Code Security
- Regular dependency updates
- Security scanning (Snyk, Dependabot)
- Code review required for all changes

## 12. Documentation

### 12.1 Code Documentation
- Docstrings for all functions/classes (NumPy style)
- Type hints throughout
- Sphinx for API documentation

### 12.2 User Documentation
- User guide
- API reference
- Tutorials and examples
- FAQs

### 12.3 Developer Documentation
- Architecture overview (this document)
- Development setup guide
- Contributing guidelines
- Testing procedures

---

*This technical architecture provides a solid foundation for building a scalable, maintainable system for climate teleconnection discovery and operational forecasting.*
