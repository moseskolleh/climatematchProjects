# Data Requirements and Sources

## Overview

This document specifies all data requirements for the Hybrid Physics-ML Framework for Flood Prediction, including sources, formats, quality requirements, and acquisition strategies.

## 1. Hydrometeorological Data

### 1.1 Precipitation

#### Satellite Products (Primary)

**CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)**
- **Spatial Resolution**: 0.05° (~5.5 km)
- **Temporal Resolution**: Daily
- **Coverage**: 1981-present, quasi-global (50°S-50°N)
- **Access**: https://www.chc.ucsb.edu/data/chirps
- **Format**: GeoTIFF, NetCDF
- **Quality**: Good for African regions, incorporates station data
- **Use**: Primary precipitation input for model training and operations

**GPM IMERG (Global Precipitation Measurement)**
- **Spatial Resolution**: 0.1° (~11 km)
- **Temporal Resolution**: 30-minute, daily
- **Coverage**: 2000-present, global (60°S-60°N)
- **Access**: https://gpm.nasa.gov/data/directory
- **Format**: HDF5, NetCDF
- **Quality**: High temporal resolution, useful for flash floods
- **Use**: High-resolution precipitation for event analysis

**TAMSAT (Tropical Applications of Meteorology using SATellite)**
- **Spatial Resolution**: 0.0375° (~4 km)
- **Temporal Resolution**: Daily, dekadal
- **Coverage**: 1983-present, Africa-focused
- **Access**: https://www.tamsat.org.uk/data
- **Format**: NetCDF
- **Quality**: Optimized for Africa, includes uncertainty estimates
- **Use**: Africa-specific validation and ensemble member

#### Ground Station Data

**National Meteorological Services**
- **Countries**: Nigeria, Ghana, Burkina Faso, Senegal, Mali, Benin
- **Access**: Through partnership agreements with NMHS
- **Temporal Resolution**: Hourly to daily
- **Quality**: Variable, requires extensive QC
- **Use**: Ground truth for satellite validation, model training

**GSOD (Global Summary of the Day)**
- **Access**: NOAA NCEI (https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00516)
- **Coverage**: Global, 1929-present
- **Temporal Resolution**: Daily
- **Quality**: Variable coverage in West Africa
- **Use**: Supplementary station data

### 1.2 Temperature and Other Meteorological Variables

**ERA5 Reanalysis**
- **Provider**: ECMWF (European Centre for Medium-Range Weather Forecasts)
- **Variables**: Temperature (2m), humidity, wind speed, radiation, pressure
- **Spatial Resolution**: 0.25° (~31 km)
- **Temporal Resolution**: Hourly
- **Coverage**: 1940-present, global
- **Access**: Copernicus Climate Data Store (https://cds.climate.copernicus.eu/)
- **Format**: GRIB, NetCDF
- **Use**: Evapotranspiration calculation, energy balance

**MERRA-2 (Modern-Era Retrospective analysis for Research and Applications)**
- **Provider**: NASA GMAO
- **Variables**: Similar to ERA5
- **Spatial Resolution**: 0.5° × 0.625°
- **Access**: https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/
- **Use**: Alternative reanalysis for ensemble uncertainty

### 1.3 Streamflow (Discharge) Data

#### In-situ Gauging Stations

**Primary Sources**:
- **National Hydrological Services**: Niger Basin Authority, Volta Basin Authority, Senegal River Basin Development Organization
- **Access**: Partnerships, data sharing agreements
- **Temporal Resolution**: Daily (some hourly)
- **Variables**: Water level, discharge
- **Quality Issues**:
  - Missing data periods
  - Rating curve uncertainties
  - Irregular maintenance
  - Delays in data availability

**Data Requirements**:
- Minimum 10 years of daily data for training basins
- At least 5 years for validation basins
- Metadata: Datum, rating curves, gauge maintenance records

#### Alternative Streamflow Data

**GloFAS (Global Flood Awareness System) Reanalysis**
- **Provider**: ECMWF/Copernicus
- **Coverage**: 1984-present, global river network
- **Resolution**: ~0.05-0.1° river grid
- **Access**: https://www.globalfloods.eu/
- **Use**: Initial conditions, ungauged basin estimates
- **Limitation**: Known biases in Africa, use with caution

**GRDC (Global Runoff Data Centre)**
- **Provider**: WMO/Germany
- **Access**: https://www.bafg.de/GRDC/
- **Coverage**: Historical data for major rivers
- **Use**: Additional historical records for major basins

## 2. Basin Characteristics and Geospatial Data

### 2.1 Digital Elevation Models (DEM)

**SRTM (Shuttle Radar Topography Mission)**
- **Resolution**: 30m (~1 arc-second)
- **Coverage**: Global (60°S-60°N)
- **Access**: USGS EarthExplorer (https://earthexplorer.usgs.gov/)
- **Format**: GeoTIFF, HGT
- **Use**: Basin delineation, slope calculation, flow routing

**ASTER GDEM**
- **Resolution**: 30m
- **Coverage**: Global (83°S-83°N)
- **Access**: NASA Earthdata (https://earthdata.nasa.gov/)
- **Use**: Alternative DEM for comparison, gap-filling

**Derived Products**:
- Basin boundaries (watershed delineation)
- Stream network (D8 or D-infinity algorithms)
- Slope maps
- Flow accumulation grids
- Topographic wetness index

### 2.2 Land Cover and Vegetation

**ESA WorldCover**
- **Resolution**: 10m
- **Coverage**: Global, 2020-present
- **Classes**: 11 land cover classes
- **Access**: https://worldcover2020.esa.int/
- **Format**: GeoTIFF
- **Use**: Parameter estimation, runoff coefficient assignment

**MODIS Land Cover (MCD12Q1)**
- **Resolution**: 500m
- **Temporal Coverage**: Annual, 2001-present
- **Access**: NASA Earthdata
- **Use**: Temporal land cover changes

**NDVI (Normalized Difference Vegetation Index)**
- **Source**: MODIS (MOD13Q1)
- **Resolution**: 250m
- **Temporal**: 16-day composite
- **Use**: Irrigation detection, vegetation dynamics

### 2.3 Soil Properties

**SoilGrids**
- **Provider**: ISRIC World Soil Information
- **Resolution**: 250m
- **Depth**: 6 standard depths (0-200 cm)
- **Variables**:
  - Soil texture (sand, silt, clay fractions)
  - Bulk density
  - Organic carbon content
  - pH, cation exchange capacity
- **Access**: https://soilgrids.org/
- **Format**: GeoTIFF
- **Use**: Infiltration parameters, water holding capacity

**FAO Soil Map**
- **Resolution**: 1:5,000,000 scale
- **Coverage**: Global
- **Access**: FAO (http://www.fao.org/soils-portal/)
- **Use**: Soil type classification for parameter regionalization

### 2.4 Water Infrastructure

**Global Reservoir and Dam Database (GRanD)**
- **Provider**: Global Water System Project
- **Records**: >7,000 reservoirs
- **Attributes**: Capacity, purpose, year of construction
- **Access**: http://globaldamwatch.org/grand/
- **Use**: Dam location and characteristics

**National Data**:
- Request from basin authorities
- Operating rules, release schedules
- Real-time reservoir levels (if available)

### 2.5 Population and Settlement Data

**WorldPop**
- **Resolution**: 100m
- **Coverage**: Global, annual
- **Access**: https://www.worldpop.org/
- **Use**: Exposure assessment, prioritization of forecast locations

**GHSL (Global Human Settlement Layer)**
- **Provider**: European Commission JRC
- **Variables**: Built-up areas, population density
- **Access**: https://ghsl.jrc.ec.europa.eu/
- **Use**: Urban extent mapping, vulnerability assessment

## 3. Climate Model Data (For Future Projections)

### CMIP6 (Coupled Model Intercomparison Project Phase 6)

**Variables**: Precipitation, temperature
**Scenarios**: SSP1-2.6, SSP2-4.5, SSP5-8.5
**Models**: Multi-model ensemble (10-15 models)
**Access**: ESGF (Earth System Grid Federation)
**Use**: Future flood risk projections under climate change

## 4. Data Acquisition Strategy

### Year 1: Data Collection Phase

#### Months 1-3: Data Source Identification and Access
- Establish partnerships with national meteorological/hydrological services
- Submit data requests to international archives
- Set up API access for real-time data sources
- Create data sharing agreements

#### Months 4-6: Historical Data Assembly
- Download satellite products for 2000-2020
- Acquire streamflow records from partner agencies
- Collect basin characteristic datasets
- Organize data in standardized format

#### Months 7-9: Quality Control and Preprocessing
- Implement automated QC algorithms
- Flag suspicious/erroneous data points
- Fill minor gaps using interpolation
- Document data quality issues

#### Months 10-12: Data Pipeline Development
- Create automated download scripts
- Develop preprocessing workflows
- Set up database for efficient storage and retrieval
- Test real-time data ingestion

### Data Storage Requirements

**Total Storage Estimate**:
- Satellite precipitation (20 years, 5 basins): ~500 GB
- Reanalysis data (20 years): ~200 GB
- Streamflow and station data: ~5 GB
- Geospatial datasets: ~50 GB
- Model outputs and results: ~200 GB
- **Total**: ~1 TB

**Storage Solution**:
- Cloud storage (AWS S3, Google Cloud Storage) for raw data
- Local server with fast SSD for processed data and model training
- Backup on institutional storage

## 5. Data Quality Standards

### Minimum Data Requirements for Basin Selection

**Training Basins** (3-5 basins):
- ≥10 years of daily discharge data
- <20% missing values
- Documented rating curves
- No major infrastructure changes during period
- Area: 1,000-50,000 km²

**Validation Basins** (2-3 basins):
- ≥5 years of daily discharge data
- <30% missing values
- Overlapping time period with training data
- Different geographic location/characteristics

**Transfer Learning Basins** (5-10 basins):
- Any amount of data (including ungauged)
- Will use transferred parameters
- Priority: basins with population centers downstream

### Data Completeness Thresholds

| Data Type | Minimum Completeness | Preferred |
|-----------|---------------------|-----------|
| Precipitation (satellite) | 95% | 99% |
| Temperature/Met variables | 90% | 95% |
| Streamflow | 70% | 85% |
| DEM/Geospatial | 100% | 100% |

### Data Latency Requirements (Operational Phase)

| Data Type | Maximum Acceptable Latency |
|-----------|---------------------------|
| Satellite precipitation | 6 hours |
| Meteorological forecasts | 12 hours |
| Upstream gauge readings | 3 hours |
| Reservoir levels | 24 hours |

## 6. Data Sharing and Management

### Data Management Plan

**Storage**:
- Raw data: Immutable, with version control
- Processed data: Documented processing steps
- Outputs: Linked to model version and configuration

**Metadata**:
- Data source and download date
- Processing steps applied
- Quality flags and issues
- Coordinate reference system
- Temporal coverage

**Access Control**:
- Public data: Open sharing after publication
- Partner data: Respect data sharing agreements
- Sensitive data: Anonymize if necessary

**Long-term Preservation**:
- Deposit final datasets in trusted repository (Zenodo, Pangaea)
- Assign DOI for citability
- Ensure data outlives project

### Ethical Considerations

- **Acknowledge data providers** in publications
- **Respect data sharing restrictions** from national services
- **Ensure data security**, especially for infrastructure-sensitive information
- **Give back to data providers**: Share improved datasets, forecasts

## 7. Data Gaps and Mitigation Strategies

### Known Gaps

1. **Limited ground station coverage**: Use satellite products as primary, validate with available stations
2. **Missing discharge data periods**: Use GloFAS for gap-filling with bias correction
3. **Uncertain dam operations**: Learn patterns from historical data, request schedules from operators
4. **Limited soil data in Africa**: Use global products, validate with field surveys if budget allows

### Adaptive Data Strategy

If high-quality data unavailable:
- **Relax selection criteria** for basins
- **Use ensemble of satellite products** to quantify uncertainty
- **Transfer from data-rich regions** more heavily
- **Focus on relative skill** (improvement over baseline) rather than absolute accuracy

---

*This data requirements document will be updated as data acquisition progresses and new sources become available.*
