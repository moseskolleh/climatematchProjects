# Data Sources for Drought Early Warning System

## Overview

This document catalogs all data sources used in the Adaptive Multi-Scale Drought Early Warning System, including access methods, update frequencies, and quality considerations.

## Tier 1: Satellite Products (Primary Data)

### 1. CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)

**Description**: High-resolution precipitation dataset combining satellite imagery with station data

**Specifications**:
- **Spatial Resolution**: 0.05° (~5.5 km at equator)
- **Temporal Resolution**: Daily, pentadal, dekadal, monthly
- **Temporal Coverage**: 1981-present
- **Geographic Coverage**: 50°S-50°N (all of Africa)
- **Latency**: ~3 days for daily, ~2 weeks for final version

**Access**:
- **URL**: https://data.chc.ucsb.edu/products/CHIRPS-2.0/
- **Protocol**: FTP, HTTP
- **Format**: GeoTIFF, NetCDF, BIL
- **License**: Public domain

**Quality**:
- **Strengths**: Long record, high resolution, validated
- **Limitations**: Systematic biases in mountainous regions
- **Uncertainty**: Available in CHIRPS-GEFS ensemble

**Usage in Project**:
- Primary precipitation input for all models
- Drought index calculation (SPI, SPEI)
- Validation of precipitation forecasts

**Code Example**:
```python
import requests
from datetime import datetime

def download_chirps(date, region='africa'):
    base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0"
    year = date.year
    file_name = f"chirps-v2.0.{date.strftime('%Y.%m.%d')}.tif"
    url = f"{base_url}/africa_daily/tifs/p05/{year}/{file_name}"

    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(response.content)
        return file_name
    else:
        raise Exception(f"Failed to download CHIRPS data: {response.status_code}")
```

### 2. MODIS (Moderate Resolution Imaging Spectroradiometer)

**Products Used**:

#### MOD13Q1: Vegetation Indices (16-day, 250m)
- **Variables**: NDVI, EVI
- **Use**: Vegetation health monitoring
- **Quality Flags**: Pixel reliability, aerosol quantity

#### MOD11A2: Land Surface Temperature (8-day, 1km)
- **Variables**: Day/night LST
- **Use**: Heat stress detection
- **Quality**: Emissivity accuracy

#### MOD16A2: Evapotranspiration (8-day, 500m)
- **Variables**: ET, PET, LE
- **Use**: Water stress indicators

**Access**:
- **Portal**: NASA EarthData
- **API**: NASA CMR API, AppEEARS
- **Format**: HDF, GeoTIFF
- **License**: Open (registration required)

**Code Example**:
```python
from pymodis import downmodis

def download_modis_ndvi(date, tiles=['h20v08', 'h20v09']):
    """Download MODIS NDVI for East African tiles"""
    modis_down = downmodis.downModis(
        destinationFolder='/data/modis',
        user='username',
        password='password',
        product='MOD13Q1.061',
        tiles=','.join(tiles),
        today=date.strftime('%Y-%m-%d'),
        delta=16
    )
    modis_down.connect()
    modis_down.downloadsAllDay()
```

### 3. GRACE/GRACE-FO (Gravity Recovery and Climate Experiment)

**Description**: Satellite measurements of terrestrial water storage anomalies

**Specifications**:
- **Spatial Resolution**: ~300 km
- **Temporal Resolution**: Monthly
- **Temporal Coverage**: 2002-2017 (GRACE), 2018-present (GRACE-FO)
- **Variables**: Total Water Storage Anomaly, Groundwater Anomaly

**Access**:
- **URL**: https://grace.jpl.nasa.gov/
- **Providers**: JPL, CSR, GFZ
- **Format**: NetCDF
- **License**: Open

**Usage**:
- Deep drought indicator (groundwater depletion)
- Validation of hydrological models
- Long-term water availability trends

### 4. ERA5 (ECMWF Reanalysis v5)

**Description**: Global atmospheric reanalysis

**Specifications**:
- **Spatial Resolution**: 0.25° (~31 km)
- **Temporal Resolution**: Hourly
- **Temporal Coverage**: 1950-present (lagged ~5 days)
- **Variables**: 100+ atmospheric and land surface variables

**Key Variables**:
- 2m temperature
- Total precipitation
- Soil moisture (4 levels)
- Evaporation
- Surface pressure
- Wind components

**Access**:
- **Portal**: Copernicus Climate Data Store (CDS)
- **API**: CDS API (Python client available)
- **Format**: GRIB, NetCDF
- **License**: Open

**Code Example**:
```python
import cdsapi

def download_era5(variables, area, date_range):
    """
    Download ERA5 data

    area: [North, West, South, East]
    """
    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variables,
            'year': date_range['year'],
            'month': date_range['month'],
            'day': date_range['day'],
            'time': '00:00',
            'area': area,  # [60, -10, -35, 55] for Africa
        },
        'era5_download.nc'
    )
```

## Tier 2: National Meteorological Data

### Station Networks

**Countries** (Initial Phase):
- Kenya: KMD (Kenya Meteorological Department)
- Ethiopia: EMI (Ethiopian Meteorological Institute)
- Tanzania: TMA (Tanzania Meteorological Agency)

**Data Types**:
- Daily precipitation
- Temperature (min, max, mean)
- Humidity
- Wind speed
- Sunshine hours

**Access Methods**:
1. Direct API (where available)
2. Data sharing agreements
3. Historical data digitization

**Quality Control**:
- Duplicate removal
- Outlier detection
- Homogeneity testing
- Gap filling

**Challenges**:
- Sparse station coverage
- Data gaps
- Delayed reporting
- Quality inconsistencies

### TAHMO (Trans-African Hydro-Meteorological Observatory)

**Description**: Network of 600+ automated weather stations across Africa

**Specifications**:
- **Temporal Resolution**: 5-minute observations
- **Variables**: Precipitation, temperature, humidity, pressure, solar radiation
- **Data Quality**: Automated quality control

**Access**:
- **API**: TAHMO API (registration required)
- **Format**: JSON, CSV
- **Latency**: Near real-time

## Tier 3: Crowd-Sourced Observations

### Community Observer Network

**Platform**: Custom mobile application

**Data Collected**:
1. **Indigenous Indicators**:
   - Animal behavior changes
   - Plant phenology observations
   - Water source status
   - Soil condition

2. **Impact Observations**:
   - Crop condition
   - Livestock condition
   - Water availability
   - Food prices

3. **Georeferenced Photos**:
   - Vegetation state
   - Water bodies
   - Crop fields

**Validation**:
- Observer training and certification
- Cross-validation between observers
- Comparison with satellite data
- Expert review of flagged observations

**Privacy**:
- Anonymized location (aggregated to district)
- No personal information stored
- Opt-in participation

### Citizen Science Platforms

**Sources**:
- iNaturalist: Biodiversity observations
- Global Mosquito Alert: Disease vector presence
- CoCoRaHS: Precipitation measurements

## Tier 4: Alternative Data Sources

### Social Media

**Platforms**:
- Twitter/X: Drought-related keywords
- Facebook: Agricultural groups
- WhatsApp: Farmer networks (with permission)

**Methods**:
- NLP for sentiment analysis
- Keyword tracking
- Geographic tagging
- Trend analysis

**Limitations**:
- Selection bias
- Reliability concerns
- Language barriers
- Privacy considerations

### Market Prices

**Sources**:
- FAO GIEWS: Global food prices
- FEWS NET: Market monitoring
- National agricultural market boards

**Variables**:
- Cereal prices (maize, wheat, sorghum)
- Livestock prices
- Water prices

**Usage**:
- Early drought signal (price spikes)
- Impact assessment
- Economic vulnerability

### Satellite-Based Indicators

**FLDAS (Famine Early Warning Systems Network Land Data Assimilation System)**:
- Soil moisture
- Evapotranspiration
- Runoff

**SMAP (Soil Moisture Active Passive)**:
- Surface soil moisture (0-5 cm)
- 9 km resolution
- 2-3 day revisit

**Sentinel-1/2**:
- High-resolution optical/radar
- Crop monitoring
- Water body extent

## Data Integration Strategy

### Spatial Harmonization

**Target Grid**: 0.05° (~5.5 km) to match CHIRPS

**Resampling Methods**:
- Precipitation: Conservative regridding
- Temperature: Bilinear interpolation + elevation correction
- Vegetation: Majority resampling

### Temporal Alignment

**Reference**: Daily time steps (00:00 UTC)

**Aggregation**:
- Sub-daily → Daily: Mean, sum, min, max as appropriate
- 8-day, 16-day → Daily: Linear interpolation
- Monthly → Daily: Spline interpolation with uncertainty

### Quality Flags

**Unified Quality System**:
- 0: Missing
- 1: Poor quality (use with caution)
- 2: Moderate quality (acceptable)
- 3: High quality (reliable)

**Metadata**:
- Data source
- Processing date
- Quality score
- Uncertainty estimate

## Data Storage

### Structure

```
/data/
├── raw/                    # Original downloaded data
│   ├── chirps/
│   ├── modis/
│   ├── era5/
│   └── stations/
├── processed/              # Quality-controlled, harmonized
│   ├── precipitation/
│   ├── temperature/
│   ├── vegetation/
│   └── soil_moisture/
├── fused/                  # Multi-source fusion products
│   └── daily_gridded/
└── indices/                # Derived drought indices
    ├── spi/
    ├── spei/
    └── vci/
```

### Formats

- **Gridded Data**: NetCDF-4 with CF conventions
- **Tabular Data**: Parquet for efficiency
- **Metadata**: JSON-LD for semantic web compatibility

### Retention Policy

- **Raw Data**: 2 years local, 10 years archive
- **Processed Data**: 5 years local, permanent archive
- **Fusion Products**: 10 years local, permanent archive

## Data Quality Monitoring

### Automated Checks

1. **Completeness**: % of expected data received
2. **Timeliness**: Latency from observation to ingestion
3. **Validity**: Range and consistency checks
4. **Comparison**: Cross-validation between sources

### Reporting

- Daily data quality dashboard
- Weekly data quality report
- Monthly comprehensive review
- Annual data source evaluation

## Data Access Policy

### Internal Use
- Full access to all processed data
- Version control for reproducibility
- Data provenance tracking

### External Sharing
- Open access to fusion products (after publication)
- Restricted access to station data (partner agreements)
- Anonymized community data available on request

## Budget Considerations

### Data Costs

| Source | Annual Cost | Notes |
|--------|-------------|-------|
| Satellite Data | €0 | Open access |
| Computing/Storage | €15,000 | Cloud services |
| Station Data | €10,000 | Data sharing agreements |
| Mobile App | €5,000 | Hosting and SMS |
| **Total** | **€30,000** | Year 1 estimate |

## References

1. Funk, C., et al. (2015). The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes. Scientific Data, 2(1), 1-21.

2. Didan, K. (2015). MOD13Q1 MODIS/Terra Vegetation Indices 16-Day L3 Global 250m SIN Grid V006. NASA EOSDIS Land Processes DAAC.

3. Hersbach, H., et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049.

4. Landerer, F. W., & Swenson, S. C. (2012). Accuracy of scaled GRACE terrestrial water storage estimates. Water Resources Research, 48(4).
