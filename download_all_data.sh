#!/bin/bash
# ClimateMatch Data Downloader - Bash Script
# Downloads all datasets used in ClimateMatch Academy tutorials

set -e

# Configuration
OUTPUT_DIR="climatematch_data"
URLS_FILE="climatematch_data_urls.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "ClimateMatch Academy Data Downloader"
echo "================================================"

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}Warning: jq is not installed. Using alternative method.${NC}"
    USE_JQ=false
else
    USE_JQ=true
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"/{osf,unknown}

echo -e "\n${GREEN}Created output directories${NC}"

# Function to download with progress
download_file() {
    local url=$1
    local output=$2
    local filename=$(basename "$output")

    if [ -f "$output" ]; then
        echo -e "${YELLOW}Skipping (already exists): $filename${NC}"
        return 0
    fi

    echo -e "Downloading: ${GREEN}$filename${NC}"

    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$output" "$url" 2>&1
    elif command -v curl &> /dev/null; then
        curl -# -L -o "$output" "$url"
    else
        echo -e "${RED}Error: Neither wget nor curl is available${NC}"
        return 1
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Downloaded: $filename${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed: $filename${NC}"
        return 1
    fi
}

# Download all OSF files
echo -e "\n${GREEN}Downloading OSF datasets...${NC}"

# OSF URLs (extracted from the data)
declare -A OSF_FILES=(
    ["https://osf.io/7kcwn/download"]="ERA5_5vars_032018_hourly_NE-US.nc"
    ["https://osf.io/ke9yp/download"]="wind_evel_monthly_2016.nc"
    ["https://osf.io/9zkgd/download"]="wind_nvel_monthly_2016.nc"
    ["https://osf.io/tyfbv/download"]="evel_monthly_2016.nc"
    ["https://osf.io/vzdn4/download"]="nvel_monthly_2016.nc"
    ["https://osf.io/98ksr/download"]="data_fname_surface_temp.nc"
    ["https://osf.io/3q4vs/download"]="data_era5_mm.nc"
    ["https://osf.io/aufs2/download"]="data_salt.nc"
    ["https://osf.io/c8wqt/download"]="data_theta_annual.nc"
    ["https://osf.io/kmy5w/download"]="data_Shakun2015_SST.nc"
    ["https://osf.io/45fev/download"]="data_antarctica2015co2composite_cleaned.nc"
    ["https://osf.io/p8tx3/download"]="data_tang.nc"
    ["https://osf.io/gw2m5/download"]="data_PMIP3.nc"
    ["https://osf.io/sujvp/download"]="data_tang2.nc"
    ["https://osf.io/gm2v9/download"]="data_aden.nc"
    ["https://osf.io/mr7d9/download"]="data_Bosumtwi.nc"
    ["https://osf.io/k6e3a/download"]="data_GC27.nc"
    ["https://osf.io/6pgc2/download"]="data_sst.nc"
    ["https://osf.io/8rwxb/download"]="data_nino.nc"
    ["https://osf.io/vhdcg/download"]="data_precip.nc"
    ["https://osf.io/y2pq7/download"]="data_Climatebench_train_val.nc"
    ["https://osf.io/7tr49/download"]="data_spatial_test_data.nc"
    ["https://osf.io/pkbwx/download"]="data_scenario_test_data.nc"
    ["https://osf.io/4zynp/download"]="data_WashingtonDCSSH1.nc"
    ["https://osf.io/xs7h6/download"]="data_precipitationGermany.nc"
    ["https://osf.io/ngafk/download"]="data_cmip6_data.nc"
    ["https://osf.io/69ms8/download"]="data_WBGT_day.nc"
    ["https://osf.io/67b8m/download"]="data_SSP126.nc"
    ["https://osf.io/fsx5y/download"]="data_SSP245.nc"
    ["https://osf.io/pr456/download"]="data_SSP585.nc"
    ["https://osf.io/dakv3/download"]="wbgt_hist_raw_runmean7_gev.nc"
    ["https://osf.io/ef9pv/download"]="wbgt_126_raw_runmean7_gev_2071-2100.nc"
    ["https://osf.io/j4hfc/download"]="wbgt_245_raw_runmean7_gev_2071-2100.nc"
    ["https://osf.io/y6edw/download"]="data_bgt_585.nc"
    ["https://osf.io/zqd86/download"]="data_area_mpi.nc"
    ["https://osf.io/dxq98/download"]="data_area_land_mpi.nc"
    ["https://osf.io/c6q4j/download"]="data_sq.nc"
    ["https://osf.io/w6cd5/download"]="data_ncep_air.nc"
)

SUCCESS=0
FAILED=0

for url in "${!OSF_FILES[@]}"; do
    filename="${OSF_FILES[$url]}"
    output="$OUTPUT_DIR/osf/$filename"

    if download_file "$url" "$output"; then
        ((SUCCESS++))
    else
        ((FAILED++))
    fi
done

# Download other datasets
echo -e "\n${GREEN}Downloading other datasets...${NC}"

declare -A OTHER_FILES=(
    ["https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/antarctica2015co2composite.txt"]="data_antarctica2015.txt"
    ["https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/epica_domec/edc3deuttemp2007.txt"]="data_edc3deuttemp2007.txt"
    ["https://raw.githubusercontent.com/LinkedEarth/paleoHackathon/main/data/Orbital_records/Sanbao_composite.csv"]="data_Sanbao_composite.csv"
    ["https://www.ncei.noaa.gov/pub/data/paleo/reconstructions/osman2021/LGMR_SAT_climo.nc"]="data_LGMR_SAT_climo.nc"
    ["https://www.ncei.noaa.gov/pub/data/paleo/coral/east_pacific/cobb2013-fan-modsplice-noaa.txt"]="data_cobb2013.txt"
)

for url in "${!OTHER_FILES[@]}"; do
    filename="${OTHER_FILES[$url]}"
    output="$OUTPUT_DIR/unknown/$filename"

    if download_file "$url" "$output"; then
        ((SUCCESS++))
    else
        ((FAILED++))
    fi
done

# Summary
echo -e "\n================================================"
echo -e "${GREEN}DOWNLOAD SUMMARY${NC}"
echo -e "================================================"
echo -e "Successfully downloaded: ${GREEN}$SUCCESS${NC} files"
echo -e "Failed downloads: ${RED}$FAILED${NC} files"
echo -e "Data saved to: ${GREEN}$(pwd)/$OUTPUT_DIR${NC}"
echo -e "================================================"
