# Data Requirements

## Overview

This document specifies the data requirements for the climate sensitivity constraint framework, including sources, formats, processing needs, and quality control procedures.

## Data Categories

### 1. Historical Observations

#### 1.1 Surface Temperature

**Primary Dataset: HadCRUT5**
- **Source:** Met Office Hadley Centre
- **URL:** https://www.metoffice.gov.uk/hadobs/hadcrut5/
- **Version:** 5.0.1.0 or later
- **Coverage:** 1850-present
- **Resolution:** 5° × 5° latitude-longitude grid
- **Format:** NetCDF
- **Variables:** tas_mean, tas_lower, tas_upper (with uncertainty bounds)
- **Download size:** ~2 GB

**Secondary Dataset: Berkeley Earth**
- **Source:** Berkeley Earth Surface Temperature
- **URL:** http://berkeleyearth.org/data/
- **Coverage:** 1850-present
- **Resolution:** 1° × 1° grid
- **Format:** NetCDF or text files
- **Use:** Cross-validation and uncertainty assessment

**Processing:**
```python
import xarray as xr

# Load data
ds = xr.open_dataset('HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc')

# Calculate global mean
weights = np.cos(np.deg2rad(ds.latitude))
global_mean = ds.tas_mean.weighted(weights).mean(dim=['latitude', 'longitude'])

# Compute trend for reference period
ref_period = global_mean.sel(time=slice('1850', '1900')).mean()
recent_period = global_mean.sel(time=slice('2010', '2020')).mean()
warming = recent_period - ref_period
```

#### 1.2 Ocean Heat Content

**Primary Dataset: Argo**
- **Source:** Argo Program
- **URL:** https://www.nodc.noaa.gov/argo/
- **Coverage:** 2005-present
- **Depth:** 0-2000m
- **Format:** NetCDF (gridded products)
- **Download size:** ~10 GB

**Derived Product: Ocean Heat Uptake Rate**
- Calculate from ocean heat content time series
- Units: W/m² (globally averaged)
- Method: dQ/dt over Earth surface area

**Processing:**
```python
# Load OHC data (example)
ohc = xr.open_dataset('argo_ohc_0-2000m.nc')

# Calculate rate of change
ohc_trend = ohc.polyfit(dim='time', deg=1)
heat_uptake_rate = ohc_trend.polyfit_coefficients[0] * (time_conversion)

# Convert to W/m²
earth_surface_area = 5.1e14  # m²
heat_uptake_wm2 = heat_uptake_rate / earth_surface_area
```

#### 1.3 Top-of-Atmosphere Radiation

**Dataset: CERES EBAF**
- **Source:** NASA Langley Research Center
- **URL:** https://ceres.larc.nasa.gov/data/
- **Version:** Ed4.1 or later
- **Coverage:** 2000-present
- **Resolution:** 1° × 1° monthly means
- **Format:** NetCDF
- **Variables:** toa_sw_all_mon, toa_lw_all_mon, solar_mon
- **Download size:** ~5 GB

**Use:**
- TOA energy imbalance
- Validation of ocean heat uptake
- Forcing attribution

### 2. Radiative Forcing

**Dataset: IPCC AR6 Effective Radiative Forcing**
- **Source:** IPCC AR6 WG1
- **URL:** https://catalogue.ceda.ac.uk/uuid/9c6f410ecd7a4f6fb0f9c82b4e6f4e95
- **Coverage:** 1750-2019
- **Format:** CSV or NetCDF
- **Components:**
  - CO₂: Well-mixed greenhouse gases
  - CH₄, N₂O, halocarbons
  - Aerosols (direct + indirect)
  - Ozone (stratospheric + tropospheric)
  - Land use / albedo changes
  - Solar irradiance
  - Volcanic aerosols

**Processing:**
```python
# Load forcing data
forcing = pd.read_csv('AR6_ERF_1750-2019.csv')

# Calculate total anthropogenic forcing
anthro_forcing = forcing[['CO2', 'CH4', 'N2O', 'Aerosols', 'Ozone']].sum(axis=1)

# Get forcing change for period
ref_forcing = anthro_forcing.loc[forcing.year.between(1850, 1900)].mean()
recent_forcing = anthro_forcing.loc[forcing.year.between(2010, 2020)].mean()
delta_forcing = recent_forcing - ref_forcing
```

### 3. Paleoclimate Data

#### 3.1 Last Glacial Maximum

**Temperature:**
- **Source:** Tierney et al. (2020) data compilation
- **DOI:** 10.1038/s41586-020-2617-x
- **Value:** -6.1 ± 0.4 K (global mean cooling)
- **Format:** Published values + uncertainty

**Forcing:**
- **Source:** PMIP4 model ensemble
- **URL:** https://pmip4.lsce.ipsl.fr/
- **Components:**
  - CO₂: 190 ppm (vs 280 ppm preindustrial)
  - Ice sheets: Albedo + topography effects
  - Vegetation: Changes in land cover
  - Orbital: Different insolation pattern
- **Value:** -7.5 ± 1.5 W/m² (ensemble estimate)
- **Format:** Model output (NetCDF)

**PMIP4 Models to Download:**
- AWI-ESM-1-1-LR
- CESM2
- CNRM-CM6-1
- GISS-E2-1-G
- IPSL-CM6A-LR
- MIROC-ES2L
- MPI-ESM1-2-LR
- UKESM1-0-LL

**Variables needed:**
- tas: Surface air temperature
- rlut: Outgoing longwave radiation
- rsut: Outgoing shortwave radiation
- pr: Precipitation (for validation)

#### 3.2 Mid-Pliocene Warm Period

**Temperature:**
- **Source:** Haywood et al. (2020) PlioMIP2
- **DOI:** 10.5194/cp-16-2095-2020
- **Value:** +3.2 ± 1.2 K (warming above preindustrial)
- **Format:** Model-data synthesis

**CO₂:**
- **Source:** Ice core and proxy reconstructions
- **Value:** 400 ± 50 ppm
- **Reference:** Pagani et al. (2010), Martínez-Botí et al. (2015)

**PlioMIP2 Models:**
- Available via ESGF
- Experiments: Eoi400 (400 ppm CO₂ Pliocene)
- Format: NetCDF via CMIP6 data node

### 4. CMIP6 Model Output

**Experiments Required:**
- **piControl:** Pre-industrial control (climate drift correction)
- **abrupt-4xCO2:** Instantaneous CO₂ quadrupling (ECS calculation)
- **1pctCO2:** 1% per year CO₂ increase (TCR calculation)
- **historical:** Historical simulations (validation)

**Variables:**
- tas: Surface air temperature
- rlut, rsut, rsdt: TOA radiation (feedback calculation)
- tos: Sea surface temperature (pattern effects)
- clt: Cloud fraction (cloud feedback)

**Models (minimum ensemble):**
- ACCESS-CM2
- CanESM5
- CESM2
- CNRM-CM6-1
- EC-Earth3
- GFDL-CM4
- GISS-E2-1-G
- IPSL-CM6A-LR
- MIROC6
- MPI-ESM1-2-LR
- MRI-ESM2-0
- NorESM2-LM
- UKESM1-0-LL

**Download:**
```bash
# Example using wget (requires ESGF credentials)
wget --user=USERNAME --password=PASSWORD \
  "http://esgf-node.llnl.gov/esg-search/wget?project=CMIP6&experiment_id=abrupt-4xCO2&variable=tas"
```

**Storage requirement:** ~1-2 TB for full ensemble

### 5. Cloud Feedback Diagnostics

**Dataset: Cloud Radiative Kernels**
- **Source:** Zelinka et al. (2020)
- **URL:** https://github.com/mzelinka/cmip56_forcing_feedback_ecs
- **Format:** Python scripts + NetCDF kernels
- **Use:** Decompose cloud feedbacks into components

**Components:**
- Low cloud amount feedback
- High cloud altitude feedback
- Cloud optical depth feedback
- Cloud masking effects

## Data Processing Pipeline

### 1. Quality Control

**Temperature data:**
- Check for missing values (flag if > 30% missing)
- Detect outliers (> 5σ from local mean)
- Cross-validate between datasets (HadCRUT vs Berkeley Earth)

**Forcing data:**
- Verify internal consistency (energy conservation)
- Compare to alternative estimates (sensitivity test)
- Check for version updates

**Model data:**
- Verify grid orientation and units
- Check for simulation drift (use piControl)
- Validate against observations for historical period

### 2. Standardization

**Spatial processing:**
```python
def regrid_to_common(dataset, target_grid='1deg'):
    """Regrid to common resolution."""
    import xesmf as xe

    ds_out = xe.util.grid_global(1.0, 1.0)  # 1° × 1° grid
    regridder = xe.Regridder(dataset, ds_out, 'bilinear')
    return regridder(dataset)
```

**Temporal processing:**
```python
def calculate_anomalies(dataset, reference_period=(1850, 1900)):
    """Calculate anomalies relative to reference period."""
    ref = dataset.sel(time=slice(*reference_period)).mean(dim='time')
    return dataset - ref
```

### 3. Uncertainty Propagation

Track uncertainties through all processing steps:

```python
from uncertainties import ufloat

# Example: Temperature with uncertainty
temp_mean = ufloat(1.09, 0.15)  # K ± 1σ
forcing_mean = ufloat(2.72, 0.30)  # W/m² ± 1σ
ohu_mean = ufloat(0.56, 0.15)  # W/m² ± 1σ

# Propagate through calculation
feedback_param = (forcing_mean - ohu_mean) / temp_mean
ecs = 3.7 / feedback_param

print(f"ECS: {ecs}")  # Automatically includes propagated uncertainty
```

## Data Storage and Organization

### Directory Structure

```
data/
├── raw/                          # Original downloaded data (not in git)
│   ├── observations/
│   │   ├── hadcrut5/
│   │   ├── berkeley_earth/
│   │   ├── argo/
│   │   └── ceres/
│   ├── forcing/
│   │   └── ar6_erf/
│   ├── paleoclimate/
│   │   ├── lgm/
│   │   └── pliocene/
│   └── cmip6/
│       ├── tas/
│       ├── rlut/
│       └── rsut/
├── processed/                    # Cleaned and standardized data
│   ├── global_mean_temperature.nc
│   ├── effective_forcing.nc
│   ├── ocean_heat_uptake.nc
│   └── cmip6_ensemble_ecs.nc
└── interim/                      # Intermediate processing steps
    └── regridded/
```

### Metadata Standards

All processed datasets include:
- Processing date and version
- Source data references (DOI, URL)
- Processing steps applied
- Uncertainty estimates
- Units and conventions
- Contact information

### Data Versioning

Use data versioning to track changes:
```bash
git lfs track "data/processed/*.nc"
git add data/processed/*.nc
git commit -m "Add processed temperature data v1.0"
```

## Data Access and Download Scripts

Automated download scripts provided in `src/data_processing/`:
- `download_observations.py`: Get observational datasets
- `download_cmip6.py`: Fetch CMIP6 model output
- `download_forcing.py`: Get forcing datasets

Example:
```bash
python src/data_processing/download_observations.py --dataset hadcrut5
```

## References

Data citations and acknowledgments in separate file: `docs/data_citations.md`
