#!/usr/bin/env python3
"""
ClimateMatch Data Downloader
Downloads all datasets used in ClimateMatch Academy tutorials

Usage:
    python download_climatematch_data.py [--parallel] [--type TYPE] [--output-dir DIR]

Options:
    --parallel      Download files in parallel (faster)
    --type TYPE     Download only specific type (OSF, unknown, all)
    --output-dir    Output directory for downloaded files (default: climatematch_data)
    --dry-run       Show what would be downloaded without downloading
"""

import json
import os
import sys
import argparse
from pathlib import Path
from urllib.request import urlretrieve
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import hashlib

class ClimateMatchDownloader:
    def __init__(self, data_file='climatematch_data_urls.json', output_dir='climatematch_data'):
        """Initialize the downloader."""
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.data = None
        self.downloaded = []
        self.failed = []

    def load_data_urls(self):
        """Load the data URLs from JSON file."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file {self.data_file} not found. Run extract_data_urls.py first.")

        with open(self.data_file, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {self.data['total_urls']} data URLs")
        for data_type, count in self.data['by_type'].items():
            print(f"  - {data_type}: {count} files")

    def create_directories(self):
        """Create output directory structure."""
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Create subdirectories by type
        for data_type in self.data['by_type'].keys():
            subdir = self.output_dir / data_type.lower().replace(' ', '_')
            subdir.mkdir(exist_ok=True)

        print(f"Created output directory: {self.output_dir}")

    def download_file(self, item, progress_bar=None):
        """Download a single file."""
        url = item['url']
        filename = item['filename']
        data_type = item['type']

        # Determine output path
        subdir = self.output_dir / data_type.lower().replace(' ', '_')
        output_path = subdir / filename

        # Skip if already exists
        if output_path.exists():
            if progress_bar:
                progress_bar.update(1)
            return {'status': 'skipped', 'file': filename, 'path': output_path}

        try:
            # Download file
            urlretrieve(url, output_path)

            # Verify file was downloaded
            if output_path.exists() and output_path.stat().st_size > 0:
                if progress_bar:
                    progress_bar.update(1)
                return {'status': 'success', 'file': filename, 'path': output_path, 'size': output_path.stat().st_size}
            else:
                if progress_bar:
                    progress_bar.update(1)
                return {'status': 'failed', 'file': filename, 'url': url, 'error': 'Empty file'}

        except Exception as e:
            if progress_bar:
                progress_bar.update(1)
            return {'status': 'failed', 'file': filename, 'url': url, 'error': str(e)}

    def download_sequential(self, items):
        """Download files sequentially."""
        print(f"\nDownloading {len(items)} files sequentially...")

        with tqdm(total=len(items), desc="Downloading", unit="file") as pbar:
            for item in items:
                result = self.download_file(item, pbar)

                if result['status'] == 'success':
                    self.downloaded.append(result)
                elif result['status'] == 'failed':
                    self.failed.append(result)

    def download_parallel(self, items, max_workers=5):
        """Download files in parallel."""
        print(f"\nDownloading {len(items)} files in parallel (max {max_workers} workers)...")

        with tqdm(total=len(items), desc="Downloading", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.download_file, item, pbar): item for item in items}

                for future in as_completed(futures):
                    result = future.result()

                    if result['status'] == 'success':
                        self.downloaded.append(result)
                    elif result['status'] == 'failed':
                        self.failed.append(result)

    def print_summary(self):
        """Print download summary."""
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)

        total_size = sum(r['size'] for r in self.downloaded if 'size' in r)
        print(f"Successfully downloaded: {len(self.downloaded)} files ({total_size / 1024 / 1024:.2f} MB)")

        if self.failed:
            print(f"\nFailed downloads: {len(self.failed)} files")
            for f in self.failed:
                print(f"  - {f['file']}: {f['error']}")

        print(f"\nData saved to: {self.output_dir.absolute()}")

        # Create summary file
        summary = {
            'downloaded': self.downloaded,
            'failed': self.failed,
            'total_files': len(self.downloaded) + len(self.failed),
            'success_count': len(self.downloaded),
            'failed_count': len(self.failed),
            'total_size_mb': total_size / 1024 / 1024
        }

        summary_file = self.output_dir / 'download_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Download ClimateMatch tutorial datasets')
    parser.add_argument('--parallel', action='store_true', help='Download files in parallel')
    parser.add_argument('--type', type=str, default='all', help='Type of data to download (OSF, unknown, all)')
    parser.add_argument('--output-dir', type=str, default='climatematch_data', help='Output directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be downloaded')
    parser.add_argument('--max-workers', type=int, default=5, help='Max parallel workers (default: 5)')

    args = parser.parse_args()

    # Initialize downloader
    downloader = ClimateMatchDownloader(output_dir=args.output_dir)

    # Load data URLs
    downloader.load_data_urls()

    # Filter by type if specified
    items = downloader.data['data']
    if args.type != 'all':
        items = [item for item in items if item['type'].lower() == args.type.lower()]
        print(f"Filtered to {len(items)} files of type '{args.type}'")

    if args.dry_run:
        print("\nDRY RUN - Would download:")
        for item in items:
            print(f"  - {item['filename']} from {item['url']}")
        return

    # Create directories
    downloader.create_directories()

    # Download files
    if args.parallel:
        downloader.download_parallel(items, max_workers=args.max_workers)
    else:
        downloader.download_sequential(items)

    # Print summary
    downloader.print_summary()

if __name__ == '__main__':
    try:
        import tqdm
    except ImportError:
        print("Error: tqdm package is required. Install it with: pip install tqdm")
        sys.exit(1)

    main()
