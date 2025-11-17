# ClimateMatch Academy Data Download Tools

Comprehensive tools to download all datasets used in [ClimateMatch Academy](https://comptools.climatematch.io/tutorials/intro.html) Computational Tools for Climate Science tutorials.

## Overview

This repository contains tools to download **43 unique datasets** from the ClimateMatch Academy tutorials, including:

- **38 datasets** from OSF (Open Science Framework)
- **5 datasets** from NOAA and GitHub repositories

### Dataset Categories

The tutorials cover various climate science domains:

1. **Ocean-Atmosphere Reanalysis** (ERA5 datasets)
2. **Remote Sensing** (Satellite data, SST, precipitation)
3. **Paleoclimate** (Ice core, proxy data)
4. **Climate Modeling** (CMIP6, PMIP3)
5. **Climate Extremes** (Heat stress, sea level, precipitation extremes)
6. **AI and Climate Change** (ClimateBench training data)
7. **Socioeconomics** (Various scenario data)

## Quick Start

### Method 1: Python Script (Recommended)

```bash
# Install required package
pip install tqdm

# Download all datasets
python download_climatematch_data.py --parallel

# Download only OSF datasets
python download_climatematch_data.py --type OSF

# Dry run (see what would be downloaded)
python download_climatematch_data.py --dry-run
```

### Method 2: Bash Script

```bash
# Download all datasets using bash
./download_all_data.sh
```

## Tools Included

### 1. `extract_data_urls.py`

Extracts all data URLs from ClimateMatch Jupyter notebooks.

**Features:**
- Scans all 326 tutorial notebooks
- Extracts URL and filename pairs
- Organizes data by source type
- Generates JSON catalog

**Usage:**
```bash
python extract_data_urls.py
```

**Output:**
- `climatematch_data_urls.json` - Complete catalog of datasets
- `urls_osf.txt` - OSF dataset URLs
- `urls_unknown.txt` - Other dataset URLs

### 2. `download_climatematch_data.py`

Python-based downloader with advanced features.

**Features:**
- ✅ Sequential or parallel downloading
- ✅ Progress bars with `tqdm`
- ✅ Automatic directory organization
- ✅ Skip already downloaded files
- ✅ Error handling and retry
- ✅ Download summary report

**Usage:**
```bash
# Basic usage
python download_climatematch_data.py

# Parallel download (faster)
python download_climatematch_data.py --parallel

# Custom output directory
python download_climatematch_data.py --output-dir my_data

# Filter by type
python download_climatematch_data.py --type OSF

# Adjust parallel workers
python download_climatematch_data.py --parallel --max-workers 10

# Dry run
python download_climatematch_data.py --dry-run
```

**Options:**
- `--parallel` - Enable parallel downloading
- `--type TYPE` - Download specific type (OSF, unknown, all)
- `--output-dir DIR` - Set output directory (default: climatematch_data)
- `--max-workers N` - Number of parallel workers (default: 5)
- `--dry-run` - Show what would be downloaded

### 3. `download_all_data.sh`

Bash script for simple downloading without Python dependencies.

**Features:**
- ✅ Works with `wget` or `curl`
- ✅ Color-coded output
- ✅ Skip existing files
- ✅ Download summary

**Usage:**
```bash
./download_all_data.sh
```

## Data Catalog

### Complete Dataset List

| Dataset | Type | Size | Source | Tutorial |
|---------|------|------|--------|----------|
| ERA5_5vars_032018_hourly_NE-US.nc | OSF | Large | OSF | W1D2_Tutorial2 |
| wind_evel_monthly_2016.nc | OSF | Medium | OSF | W1D2_Tutorial4 |
| wind_nvel_monthly_2016.nc | OSF | Medium | OSF | W1D2_Tutorial4 |
| data_Climatebench_train_val.nc | OSF | Large | OSF | W2D4_Tutorial3 |
| data_SSP126.nc | OSF | Medium | OSF | W2D3_Tutorial8 |
| data_SSP245.nc | OSF | Medium | OSF | W2D3_Tutorial8 |
| data_SSP585.nc | OSF | Medium | OSF | W2D3_Tutorial8 |
| ... | ... | ... | ... | ... |

*See `climatematch_data_urls.json` for complete list*

### Data Sources

#### OSF (Open Science Framework) - 38 datasets
Files hosted at https://osf.io for the ClimateMatch course:
- ERA5 reanalysis data
- Ocean current data
- Climate model outputs (CMIP6, PMIP3)
- ClimateBench datasets
- Wet Bulb Globe Temperature (WBGT) data
- SSP scenario data

#### NOAA - 4 datasets
National Oceanic and Atmospheric Administration datasets:
- Ice core composite data (Antarctica 2015)
- EPICA Dome C temperature data
- Last Glacial Maximum Reanalysis
- Coral proxy data (Cobb 2013)

#### GitHub - 1 dataset
- Sanbao cave speleothem composite data

## Output Structure

```
climatematch_data/
├── osf/
│   ├── ERA5_5vars_032018_hourly_NE-US.nc
│   ├── wind_evel_monthly_2016.nc
│   ├── data_Climatebench_train_val.nc
│   └── ... (38 files)
├── unknown/
│   ├── data_antarctica2015.txt
│   ├── data_edc3deuttemp2007.txt
│   ├── data_Sanbao_composite.csv
│   ├── data_LGMR_SAT_climo.nc
│   └── data_cobb2013.txt
└── download_summary.json
```

## Dataset Details

### Ocean-Atmosphere Reanalysis (Week 1, Day 2)
- ERA5 5-variable hourly data (March 2018, NE-US)
- Wind velocity components (u, v)
- Ocean surface currents
- Sea surface temperature
- Monthly mean data

### Remote Sensing (Week 1, Day 3)
- Sea surface temperature (SST) data
- NINO index data
- Precipitation data (GPCP)

### Paleoclimate (Week 1, Day 4)
- Ice core CO2 composite (Antarctica 2015)
- Temperature reconstructions (EPICA Dome C)
- Marine sediment core data
- Speleothem records (Sanbao Cave)
- Coral proxy data

### Climate Modeling (Week 1, Day 5)
- PMIP3 model outputs
- CMIP6 model data
- Insolation data

### Climate Extremes (Week 2, Day 3)
- Precipitation extremes (Germany)
- Sea level data (Washington DC)
- Wet Bulb Globe Temperature (WBGT)
- Heat stress indicators
- SSP scenario projections (126, 245, 585)

### AI and Climate (Week 2, Day 4)
- ClimateBench training/validation data
- Spatial test data
- Scenario test data

## Requirements

### Python Method
- Python 3.6+
- `tqdm` package: `pip install tqdm`

### Bash Method
- `wget` or `curl`
- `bash` shell
- (Optional) `jq` for JSON parsing

## Estimated Download Size

- **Total estimated size**: ~2-5 GB (depends on datasets)
- **OSF datasets**: Varies (50 MB - 500 MB per file)
- **Largest files**: ERA5 reanalysis, ClimateBench training data

## Notes

1. **Internet Connection**: Downloading all datasets requires a stable internet connection
2. **Storage**: Ensure you have at least 10 GB of free disk space
3. **Time**: Complete download may take 30-60 minutes depending on connection speed
4. **Parallel Downloads**: Use `--parallel` for faster downloads but be mindful of server load

## Troubleshooting

### Connection Errors
```bash
# Retry failed downloads
python download_climatematch_data.py --parallel
```

The script automatically skips already downloaded files.

### Slow Downloads
```bash
# Increase parallel workers
python download_climatematch_data.py --parallel --max-workers 10
```

### Permission Errors
```bash
# Make scripts executable
chmod +x download_all_data.sh
chmod +x download_climatematch_data.py
```

## Data Usage

These datasets are used in ClimateMatch Academy tutorials for:
- Climate data analysis with `xarray`
- Time series analysis
- Spatial pattern detection
- Climate model evaluation
- Machine learning applications
- Paleoclimate reconstruction
- Extreme event analysis

## License

The data downloading tools are provided under the BSD 3-Clause License.

Individual datasets have their own licenses:
- OSF datasets: Check individual OSF pages
- NOAA datasets: Public domain (US Government)
- GitHub datasets: Check repository licenses

## References

- **ClimateMatch Academy**: https://comptools.climatematch.io
- **Course Repository**: https://github.com/neuromatch/climate-course-content
- **OSF Storage**: https://osf.io
- **NOAA Paleoclimatology**: https://www.ncei.noaa.gov/products/paleoclimatology

## Contributing

Found a missing dataset or bug? Please open an issue or submit a pull request!

## Acknowledgments

Data curated and organized by [ClimateMatch Academy](https://academy.climatematch.io/) for educational purposes.

---

**Last Updated**: November 2025
**Total Datasets**: 43 unique files
**Repository**: https://github.com/neuromatch/climate-course-content
