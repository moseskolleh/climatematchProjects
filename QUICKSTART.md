# Quick Start Guide

## Download All ClimateMatch Data in 3 Steps

### Option 1: Python (Recommended - Fastest)

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Download all data (parallel mode)
python download_climatematch_data.py --parallel

# Step 3: Check the climatematch_data/ folder
ls -lh climatematch_data/
```

### Option 2: Bash Script (No Python Required)

```bash
# Step 1: Make script executable
chmod +x download_all_data.sh

# Step 2: Run the script
./download_all_data.sh

# Step 3: Check the climatematch_data/ folder
ls -lh climatematch_data/
```

## What Gets Downloaded?

- **43 unique datasets** used in ClimateMatch Academy tutorials
- **~2-5 GB** total download size
- Organized into folders by source:
  - `climatematch_data/osf/` - 38 files from Open Science Framework
  - `climatematch_data/unknown/` - 5 files from NOAA and GitHub

## Dataset Categories

1. **ERA5 Reanalysis** - Weather and climate reanalysis data
2. **CMIP6 Models** - Climate model outputs
3. **Paleoclimate Data** - Ice cores, proxies, reconstructions
4. **Remote Sensing** - Satellite observations
5. **Climate Extremes** - Heat stress, precipitation, sea level
6. **AI/ML Datasets** - ClimateBench training data

## Common Use Cases

### Download Only OSF Datasets
```bash
python download_climatematch_data.py --type OSF --parallel
```

### Preview What Will Be Downloaded
```bash
python download_climatematch_data.py --dry-run
```

### Download to Custom Directory
```bash
python download_climatematch_data.py --output-dir /path/to/my/data --parallel
```

### Resume Interrupted Download
```bash
# Just run again - already downloaded files are skipped
python download_climatematch_data.py --parallel
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'tqdm'"
```bash
pip install tqdm
```

### "Permission denied"
```bash
chmod +x download_all_data.sh
chmod +x download_climatematch_data.py
```

### "Connection error"
- Check your internet connection
- Try running again (script will skip already downloaded files)
- Use sequential mode if parallel has issues: `python download_climatematch_data.py`

## Next Steps

After downloading, you can:
1. Load data with `xarray`: `xr.open_dataset('climatematch_data/osf/ERA5_5vars_032018_hourly_NE-US.nc')`
2. Follow ClimateMatch tutorials: https://comptools.climatematch.io
3. Explore the data structure: `ls -R climatematch_data/`

## Need More Info?

See `README_DATA_DOWNLOAD.md` for complete documentation.
