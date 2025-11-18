# Data Requirements: Multi-Constraint Framework for Climate Sensitivity

## Overview

This document outlines the data requirements for implementing the multi-constraint framework for climate sensitivity estimation. The framework integrates multiple data types across different timescales and sources.

## Data Categories

### 1. Paleoclimate Data

#### 1.1 Last Glacial Maximum (LGM)

**Temperature Proxies:**
- **Ice Core Data**
  - Source: EPICA Dome C, Vostok, NGRIP
  - Variables: δD, δ18O, noble gas ratios
  - Temporal coverage: Last 800,000 years
  - Resolution: Decadal to centennial
  - Format: NetCDF, ASCII
  - Access: NOAA Paleoclimatology, PANGAEA

- **Marine Sediment Cores**
  - Source: MARGO database, CLIMAP
  - Variables: δ18O (forams), Mg/Ca, alkenones
  - Spatial coverage: Global ocean
  - Format: NetCDF, CSV
  - Access: PANGAEA, NOAA NCEI

- **Pollen Records**
  - Source: BIOME 6000, European Pollen Database
  - Variables: Pollen assemblages, climate reconstructions
  - Spatial coverage: Primarily Northern Hemisphere
  - Format: CSV, database format
  - Access: NEOTOMA, EPD

**Forcing Data:**
- **Ice Sheet Reconstructions**
  - Source: ICE-6G, GLAC-1D
  - Variables: Ice thickness, extent, topography
  - Resolution: 1° spatial
  - Format: NetCDF
  - Access: PMIP data repository

- **Greenhouse Gas Concentrations**
  - Source: Ice core measurements
  - Variables: CO₂, CH₄, N₂O
  - Resolution: Centennial
  - Format: ASCII, NetCDF
  - Access: NOAA ESRL, IPCC data

- **Vegetation Reconstructions**
  - Source: BIOME model, PMIP vegetation datasets
  - Variables: Vegetation type, fractional coverage
  - Resolution: 0.5° to 2°
  - Format: NetCDF
  - Access: PMIP data centers

**Model Simulations:**
- **PMIP4 LGM Experiments**
  - Models: 13+ ESMs
  - Variables: tas, pr, ts, tos, rlut, rsut, and many more
  - Experiments: lgm, piControl
  - Format: NetCDF (CF conventions)
  - Access: ESGF nodes
  - Total size: ~10 TB

#### 1.2 Mid-Pliocene Warm Period (mPWP)

**Temperature Proxies:**
- **Marine Proxies**
  - Source: PRISM4 dataset
  - Variables: SST reconstructions (alkenones, Mg/Ca)
  - Spatial coverage: Global
  - Format: CSV, NetCDF
  - Access: USGS PRISM project

- **Terrestrial Proxies**
  - Source: Pollen, leaf physiognomy
  - Variables: Mean annual temperature, precipitation
  - Format: CSV
  - Access: PRISM project, published compilations

**Model Simulations:**
- **PlioMIP2 Experiments**
  - Models: 16+ ESMs
  - Variables: Full 3D climate state
  - Experiments: midPliocene-eoi400, piControl
  - Format: NetCDF
  - Access: ESGF
  - Total size: ~15 TB

#### 1.3 Last Interglacial (LIG)

**Temperature Proxies:**
- **Marine and Terrestrial**
  - Source: PAGES2k, specific LIG compilations
  - Variables: Temperature anomalies
  - Coverage: Global
  - Format: CSV, NetCDF
  - Access: PAGES, published databases

**Model Simulations:**
- **PMIP4 lig127k Experiments**
  - Models: 10+ ESMs
  - Variables: Full climate state
  - Format: NetCDF
  - Access: ESGF

### 2. Observational Data

#### 2.1 Surface Temperature

**Datasets:**
1. **HadCRUT5**
   - Coverage: 1850-present
   - Resolution: 5° × 5°
   - Format: NetCDF
   - Update frequency: Monthly
   - Access: Met Office Hadley Centre
   - Size: ~2 GB

2. **GISTEMP v4**
   - Coverage: 1880-present
   - Resolution: 2° × 2°
   - Format: NetCDF, ASCII
   - Update frequency: Monthly
   - Access: NASA GISS
   - Size: ~1 GB

3. **NOAAGlobalTemp v5**
   - Coverage: 1850-present
   - Resolution: 5° × 5°
   - Format: NetCDF
   - Update frequency: Monthly
   - Access: NOAA NCEI
   - Size: ~1 GB

4. **Berkeley Earth**
   - Coverage: 1850-present
   - Resolution: 1° × 1°
   - Format: NetCDF
   - Update frequency: Monthly
   - Access: Berkeley Earth website
   - Size: ~5 GB

#### 2.2 Top-of-Atmosphere Radiation

**CERES EBAF**
- Variables: TOA net flux, shortwave, longwave
- Coverage: 2000-present
- Resolution: 1° × 1° monthly
- Format: NetCDF
- Access: NASA Langley ASDC
- Size: ~10 GB

**ERBE (Historical)**
- Coverage: 1985-1999
- Format: NetCDF
- Access: NASA

#### 2.3 Ocean Heat Content

**Argo Float Data**
- Variables: Temperature, salinity profiles
- Coverage: 2004-present, near-global
- Depth: 0-2000m
- Format: NetCDF
- Access: Argo GDAC
- Size: ~100 GB

**Historical OHC Estimates**
- Source: WOD, EN4, IAP
- Coverage: 1950-present
- Format: NetCDF
- Access: NOAA, Met Office, IAP

#### 2.4 Radiative Forcing

**Greenhouse Gases:**
- **NOAA Annual Greenhouse Gas Index (AGGI)**
  - Variables: CO₂, CH₄, N₂O, CFCs
  - Coverage: 1750-present
  - Format: ASCII, CSV
  - Access: NOAA ESRL

**Aerosols:**
- **AERONET**
  - Variables: AOD, single scattering albedo
  - Coverage: 1993-present, station data
  - Format: ASCII
  - Access: AERONET website

- **Satellite AOD**
  - Sources: MODIS, MISR, AVHRR
  - Coverage: 1981-present (various)
  - Format: NetCDF
  - Access: NASA

**Solar Irradiance:**
- **SORCE/SOLARIS-HEPPA**
  - Variables: Total solar irradiance
  - Coverage: 1610-present (reconstructions)
  - Format: ASCII
  - Access: NOAA NCEI

**Volcanic Forcing:**
- **VolMIP Forcing Dataset**
  - Variables: Stratospheric AOD
  - Coverage: 1850-present
  - Format: NetCDF
  - Access: ESGF

#### 2.5 Cloud Observations

**Satellite Data:**
1. **ISCCP**
   - Variables: Cloud fraction, type, properties
   - Coverage: 1983-2009
   - Resolution: 2.5° monthly
   - Format: NetCDF
   - Size: ~50 GB

2. **MODIS**
   - Variables: Cloud properties, AOD
   - Coverage: 2000-present
   - Resolution: 1° daily/monthly
   - Format: HDF, NetCDF
   - Size: ~200 GB (subset)

3. **CERES-MODIS**
   - Variables: Cloud properties matched to radiation
   - Coverage: 2000-present
   - Format: NetCDF
   - Size: ~100 GB

**Reanalysis:**
- ERA5, MERRA-2 cloud fields for evaluation

### 3. Climate Model Data

#### 3.1 CMIP5

**Models:** 56 models from 25+ modeling centers

**Experiments Required:**
- piControl (pre-industrial control)
- abrupt4xCO2 (for ECS calculation)
- 1pctCO2 (for TCR calculation)
- historical (for validation)
- sstClim, sstClim4xCO2 (for feedback analysis)

**Variables:**
- Basic: tas, ts, tos, pr, psl
- Radiation: rlut, rsut, rlutcs, rsutcs
- Clouds: clt, clivi, clwvi
- Circulation: ua, va, ta, hus
- Ocean: thetao, so, hfds

**Total Size:** ~50 TB (full archive), ~5 TB (essential variables)

**Access:** ESGF data nodes

#### 3.2 CMIP6

**Models:** 100+ models from 49 modeling centers

**Experiments Required:**
- piControl
- abrupt-4xCO2
- 1pctCO2
- historical
- abrupt-2xCO2 (new in CMIP6)
- sstClim, sstClim4xCO2

**Variables:** Same as CMIP5 plus:
- Feedback analysis: cllivi, clhgh, etc.
- Extended diagnostics: CFMIP variables

**Total Size:** ~100 TB (full), ~10 TB (essential)

**Access:** ESGF

#### 3.3 Perturbed Parameter Ensembles

**Sources:**
- climateprediction.net (CPDN)
- HadCM3 PPE
- CESM PPE

**Purpose:** Explore parameter space within single model

**Size:** ~5 TB

### 4. Reanalysis Data

**ERA5:**
- Variables: ta, hus, ua, va, omega
- Coverage: 1940-present
- Resolution: 0.25° hourly
- Format: GRIB, NetCDF
- Size: ~5 PB (full), ~100 GB (subset needed)
- Access: Copernicus Climate Data Store

**MERRA-2:**
- Coverage: 1980-present
- Resolution: 0.5° × 0.625°
- Size: Similar subset to ERA5
- Access: NASA GES DISC

**Use Cases:**
- Evaluate model circulation patterns
- Process-based constraints
- Observation uncertainty context

## Data Processing Requirements

### Storage Requirements

**Total Data Volume:**
- Raw data: ~30 TB
- Processed/intermediate: ~10 TB
- Results/output: ~1 TB
- **Total: ~40-50 TB**

**Recommended Storage:**
- High-performance shared filesystem (Lustre, GPFS)
- Or cloud object storage (S3, Azure Blob)

### Computational Requirements

**For Data Processing:**
- Memory: 64-128 GB RAM
- Cores: 16-32 cores
- Storage I/O: High-bandwidth parallel filesystem

**For Analysis:**
- Memory: 128-256 GB RAM for large model ensembles
- GPU: Optional, for deep learning components
- Python environment with scientific stack

### Data Processing Pipeline

```python
# Example pipeline structure
data_pipeline = {
    'paleoclimate': {
        'inputs': ['PMIP4_models', 'proxy_databases'],
        'processing': [
            'quality_control',
            'spatial_interpolation',
            'uncertainty_estimation',
            'global_mean_calculation'
        ],
        'outputs': ['processed_lgm_data.nc', 'processed_mpwp_data.nc']
    },

    'observational': {
        'inputs': ['temperature_datasets', 'CERES', 'Argo'],
        'processing': [
            'dataset_harmonization',
            'baseline_calculation',
            'trend_estimation',
            'uncertainty_quantification'
        ],
        'outputs': ['obs_warming.nc', 'energy_balance.nc']
    },

    'model_archive': {
        'inputs': ['CMIP5_ESGF', 'CMIP6_ESGF'],
        'processing': [
            'model_downloading',
            'ecs_calculation',
            'feedback_decomposition',
            'pattern_extraction'
        ],
        'outputs': ['model_ecs_database.csv', 'feedback_patterns.nc']
    }
}
```

## Data Quality Control

### Quality Checks

1. **Completeness**
   - Check for missing values
   - Verify temporal coverage
   - Assess spatial coverage

2. **Consistency**
   - Cross-validate overlapping datasets
   - Check physical plausibility
   - Verify metadata

3. **Accuracy**
   - Compare with independent sources
   - Check calibration procedures
   - Assess uncertainty estimates

### Quality Control Code

```python
def quality_control(dataset, checks):
    """
    Perform quality control on input data

    Parameters:
    -----------
    dataset : xarray.Dataset
        Input dataset
    checks : list
        List of QC checks to perform

    Returns:
    --------
    qc_flags : xarray.Dataset
        Quality control flags
    """
    qc_flags = {}

    if 'missing_values' in checks:
        qc_flags['missing_fraction'] = dataset.isnull().mean()

    if 'range_check' in checks:
        qc_flags['out_of_range'] = (
            (dataset < valid_range[0]) |
            (dataset > valid_range[1])
        )

    if 'temporal_continuity' in checks:
        qc_flags['temporal_gaps'] = check_temporal_gaps(dataset)

    if 'spatial_coverage' in checks:
        qc_flags['spatial_coverage'] = calculate_spatial_coverage(dataset)

    return qc_flags
```

## Data Access and Download

### Automated Download Scripts

**CMIP Data:**
```bash
# Using ESGF PyClient
wget https://raw.githubusercontent.com/ESGF/esgf-pyclient/master/scripts/esgf-pyclient
python esgf-pyclient \
    --project CMIP6 \
    --experiment abrupt-4xCO2 \
    --variable tas \
    --frequency mon \
    --download
```

**Observational Data:**
```python
# Example for HadCRUT5
import urllib.request

url = "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.1.0/analysis/diagnostics/HadCRUT.5.0.1.0.analysis.summary_series.global.monthly.nc"
urllib.request.urlretrieve(url, "data/raw/HadCRUT5_global.nc")
```

### Data Catalogs

Maintain intake catalogs for reproducibility:

```yaml
# catalog.yml
sources:
  hadcrut5:
    driver: netcdf
    args:
      urlpath: 's3://climate-data/observations/HadCRUT5/*.nc'
      xarray_kwargs:
        combine: by_coords

  cmip6_abrupt4xco2:
    driver: netcdf
    args:
      urlpath: 's3://climate-data/CMIP6/abrupt-4xCO2/*/tas/*.nc'
      xarray_kwargs:
        combine: nested
        concat_dim: model
```

## Documentation and Metadata

Each dataset must include:

1. **Provenance**
   - Source
   - Download date
   - Version number
   - DOI (if available)

2. **Processing History**
   - Processing steps
   - Software versions
   - Parameter settings

3. **Quality Flags**
   - QC results
   - Known issues
   - Limitations

### Metadata Template

```python
metadata_template = {
    'dataset_name': 'HadCRUT5',
    'version': '5.0.1.0',
    'source_url': 'https://www.metoffice.gov.uk/hadobs/hadcrut5/',
    'download_date': '2025-01-15',
    'doi': '10.1029/2019JD032361',
    'variables': ['temperature_anomaly'],
    'temporal_coverage': '1850-01 to 2024-12',
    'spatial_coverage': 'global',
    'resolution': '5° × 5°',
    'processing': [
        'baseline: 1961-1990',
        'regridding: none',
        'quality_control: passed'
    ],
    'known_issues': ['Limited coverage pre-1880', 'Sparse Southern Ocean'],
    'contact': 'Moses - Environmental Scientist'
}
```

## Data Sharing and Reproducibility

### Zenodo Archive

Create Zenodo deposits for:
- Processed datasets
- Analysis-ready data
- Model ensemble statistics

### Code-Data Integration

Use version-controlled data references:

```python
# data_references.py
DATA_VERSIONS = {
    'hadcrut5': 'v5.0.1.0',
    'cmip6': 'v20210101',
    'pmip4': 'v20200901'
}

def get_data_path(dataset, version=None):
    if version is None:
        version = DATA_VERSIONS[dataset]
    return f"data/{dataset}/{version}/"
```

## References

- Taylor, K. E., et al. (2012). An overview of CMIP5 and the experiment design. Bull. Amer. Meteor. Soc.
- Eyring, V., et al. (2016). Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6). Geosci. Model Dev.
- Haywood, A. M., et al. (2020). The Pliocene Model Intercomparison Project Phase 2. Climate of the Past.
