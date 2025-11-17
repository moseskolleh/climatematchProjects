# Data Requirements

## Overview

This document outlines the data requirements for the Integrated Coastal Risk Framework, organized by priority tier following the hierarchical data strategy.

## Tier 1: Essential Global Datasets (Always Required)

### 1.1 Satellite Altimetry (Sea Level)

**Source**: CMEMS, NASA, NOAA
**Variables**:
- Sea surface height anomaly
- Absolute dynamic topography
**Resolution**:
- Spatial: ~25 km along-track
- Temporal: Daily
**Period**: 1993-present (TOPEX/Poseidon, Jason series)
**Access**:
- CMEMS: https://marine.copernicus.eu/
- NASA PO.DAAC: https://podaac.jpl.nasa.gov/

### 1.2 Reanalysis Data (Atmospheric)

**Source**: ERA5 (ECMWF)
**Variables**:
- Mean sea level pressure
- 10m wind speed (u, v components)
- Total precipitation
- Significant wave height
**Resolution**:
- Spatial: 0.25° (~25 km)
- Temporal: Hourly
**Period**: 1979-present
**Access**: Copernicus Climate Data Store

### 1.3 Digital Elevation Model

**Source**: SRTM, ASTER GDEM, or local LIDAR
**Variables**:
- Elevation above mean sea level
**Resolution**:
- SRTM: 30m (preferred) or 90m
- ASTER: 30m
- LIDAR: 1-5m (if available)
**Coverage**: Coastal zone (0-10km inland)
**Access**: USGS Earth Explorer, NASA

### 1.4 Population and Demographics

**Source**: WorldPop, GPW, national census
**Variables**:
- Population density
- Age distribution
- Household size
**Resolution**: 100m to 1km
**Period**: Latest available (updated every 1-5 years)
**Access**: WorldPop, SEDAC

## Tier 2: High-Priority Local Data (Strongly Recommended)

### 2.1 Tide Gauge Records

**Source**: National meteorological services, PSMSL
**Variables**:
- Sea level (hourly or higher frequency)
- Quality flags
**Period**: Ideally >20 years for trend analysis
**Format**: CSV or NetCDF
**Critical Metadata**:
- Gauge datum
- Vertical reference system
- Instrument changes

### 2.2 Historical Flood Records

**Source**: National disaster databases, news archives, reports
**Variables**:
- Date of event
- Affected area (polygon or extent)
- Estimated damages
- Casualties
- Flood depth (if available)
**Period**: As far back as available
**Format**: Spreadsheet or database

### 2.3 Building Footprints and Infrastructure

**Source**: OpenStreetMap, local GIS databases
**Variables**:
- Building polygons
- Building type (residential, commercial, etc.)
- Number of floors
- Construction material
- Critical infrastructure locations
**Format**: Shapefile or GeoJSON
**Tools**: OSM Overpass API

### 2.4 Socioeconomic Data

**Source**: National statistical offices, surveys
**Variables**:
- Median income by zone
- Education levels
- Employment statistics
- Property values
**Resolution**: District or neighborhood level
**Period**: Latest census or survey

## Tier 3: Valuable Supplementary Data (Nice to Have)

### 3.1 High-Resolution Bathymetry

**Source**: Local surveys, GEBCO, EMODnet
**Variables**:
- Depth below mean sea level
**Resolution**: <100m preferred
**Coverage**: Nearshore and harbor areas

### 3.2 Land Use / Land Cover

**Source**: ESA CCI, national mapping agencies
**Variables**:
- Land cover classification
- Impervious surface fraction
**Resolution**: 10-30m
**Period**: Recent (last 5 years)

### 3.3 Weather Station Data

**Source**: National meteorological services
**Variables**:
- Precipitation (daily)
- Wind speed and direction
- Temperature
**Period**: >10 years
**Format**: CSV with quality flags

### 3.4 Coastal Defense Infrastructure

**Source**: Local government, engineering reports
**Variables**:
- Seawall locations and heights
- Breakwater positions
- Drainage system capacity
- Pump station locations
**Format**: Shapefile or database

### 3.5 Ecological Data

**Source**: Environmental agencies, research institutions
**Variables**:
- Mangrove/marsh extent
- Coral reef locations
- Beach width and composition
**Format**: Shapefile or raster

## Tier 4: Community and Real-Time Data (Emerging)

### 4.1 Crowdsourced Flood Reports

**Source**: Mobile apps, social media
**Variables**:
- Flood occurrence (yes/no)
- Estimated depth
- Location (GPS)
- Photo evidence
**Platform**: Custom app or existing (e.g., mWater)

### 4.2 Social Media Data

**Source**: Twitter, Facebook, local platforms
**Variables**:
- Event mentions
- Sentiment analysis
- Impact indicators
**Method**: Keyword extraction, geolocation

## Data Quality Requirements

### Minimum Quality Standards:

1. **Completeness**:
   - Temporal: >70% coverage over period
   - Spatial: Cover entire study area

2. **Accuracy**:
   - Elevation: ±0.5m for coastal zone
   - Sea level: ±0.05m for trends
   - Population: ±20% at neighborhood level

3. **Metadata**:
   - Source clearly documented
   - Processing steps recorded
   - Uncertainties quantified

4. **Format**:
   - Machine-readable (not PDF/images)
   - Standard coordinate systems (WGS84 or local)
   - Consistent temporal reference (UTC)

## Data Processing Pipeline

### 1. Ingestion
```python
from src.data_processing import CoastalDataProcessor

processor = CoastalDataProcessor(city='Lagos')
hazard_data = processor.load_hazard_data(start_year=2000, end_year=2023)
```

### 2. Quality Control
- Outlier detection (>4σ)
- Physical consistency checks
- Temporal consistency
- Spatial consistency

### 3. Gap Filling
- Linear interpolation for short gaps (<7 days)
- Climatological mean for longer gaps
- Satellite-station merging where applicable

### 4. Harmonization
- Standardize units
- Align temporal resolution
- Reproject to common grid
- Create uncertainty flags

## Data Sharing and Privacy

### Open Data:
- All environmental hazard data
- Aggregated socioeconomic statistics
- Model outputs and scenarios

### Protected Data:
- Individual household information
- Detailed property values
- Personal identifiers

### Data Sharing Platform:
- Results uploaded to Zenodo
- Code on GitHub
- Data on figshare or institutional repository
- DOIs for all datasets

## Estimated Storage Requirements

**Minimum Configuration** (1 city, basic analysis):
- 5-10 GB storage
- 8 GB RAM
- Standard CPU

**Full Configuration** (4 cities, high resolution):
- 50-100 GB storage
- 32 GB RAM
- Multi-core CPU or GPU recommended

## Data Update Frequency

- **Real-time**: Tide gauges, weather stations (for operational system)
- **Weekly**: Satellite altimetry
- **Monthly**: Reanalysis data updates
- **Annually**: Socioeconomic data, population
- **Every 5 years**: Census, detailed surveys
- **As available**: Historical event documentation

## Data Access Challenges

### Common Issues in African Cities:

1. **Limited Historical Records**
   - Solution: Use global datasets, regional analogs

2. **Inconsistent Formats**
   - Solution: Automated parsing, manual curation

3. **Access Restrictions**
   - Solution: Partnerships with national agencies

4. **Quality Concerns**
   - Solution: Multiple source validation, uncertainty quantification

5. **Cost Barriers**
   - Solution: Focus on open data sources, seek institutional access

## Recommended Data Partnerships

- **IGAD Climate Prediction and Applications Centre (ICPAC)**: East Africa
- **AGRHYMET**: West Africa
- **African Centre of Meteorological Applications for Development (ACMAD)**
- **Regional Centre for Mapping of Resources for Development (RCMRD)**
- **National Meteorological Services**: Country-specific data

## Sample Data Locations

We provide sample datasets for testing:
- `data/sample/lagos_tide_gauge.csv`: Sample tide gauge data
- `data/sample/elevation_grid.tif`: Sample DEM
- `data/sample/building_footprints.geojson`: Sample infrastructure

## Citation Requirements

When using data, cite appropriately:
- ERA5: Hersbach et al. (2020)
- SRTM: NASA/USGS
- WorldPop: Tatem (2017)
- Individual datasets: Check license and citation requirements
