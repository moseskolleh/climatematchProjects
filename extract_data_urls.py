#!/usr/bin/env python3
"""
Extract all data URLs and filenames from ClimateMatch tutorials
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict

def extract_urls_from_notebook(notebook_path):
    """Extract URL and filename pairs from a Jupyter notebook."""
    urls_data = []

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        # Process each cell
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))

                # Find URL patterns
                url_pattern = r'url_(\w+)\s*=\s*["\']([^"\']+)["\']'
                fname_pattern = r'fname_(\w+)\s*=\s*["\']([^"\']+)["\']'

                urls = dict(re.findall(url_pattern, source))
                fnames = dict(re.findall(fname_pattern, source))

                # Match URLs with filenames
                for key in urls:
                    url = urls[key]
                    fname = fnames.get(key, f"data_{key}.nc")

                    # Determine data type based on URL or filename
                    data_type = "unknown"
                    if "osf.io" in url:
                        data_type = "OSF"
                    elif "pangeo" in url:
                        data_type = "Pangeo"
                    elif "s3" in url or "amazonaws" in url:
                        data_type = "AWS S3"

                    urls_data.append({
                        'url': url,
                        'filename': fname,
                        'variable_name': key,
                        'type': data_type,
                        'notebook': str(notebook_path)
                    })

                # Also capture standalone pooch_load calls
                pooch_pattern = r'pooch_load\(["\']([^"\']+)["\'],\s*["\']([^"\']+)["\']\)'
                pooch_matches = re.findall(pooch_pattern, source)
                for url, fname in pooch_matches:
                    data_type = "unknown"
                    if "osf.io" in url:
                        data_type = "OSF"
                    elif "pangeo" in url:
                        data_type = "Pangeo"

                    urls_data.append({
                        'url': url,
                        'filename': fname,
                        'variable_name': 'pooch_direct',
                        'type': data_type,
                        'notebook': str(notebook_path)
                    })

    except Exception as e:
        print(f"Error processing {notebook_path}: {e}")

    return urls_data

def main():
    """Main function to extract all URLs from tutorials."""
    repo_path = Path('climate-course-content')
    tutorials_path = repo_path / 'tutorials'

    all_data = []
    url_set = set()  # Track unique URLs

    # Find all notebooks
    notebooks = list(tutorials_path.rglob('*.ipynb'))
    print(f"Found {len(notebooks)} notebooks")

    # Process each notebook
    for notebook in notebooks:
        data = extract_urls_from_notebook(notebook)
        for item in data:
            # Use URL as unique identifier
            url_key = item['url']
            if url_key not in url_set:
                url_set.add(url_key)
                all_data.append(item)

    # Organize by type
    by_type = defaultdict(list)
    for item in all_data:
        by_type[item['type']].append(item)

    # Save results
    output = {
        'total_urls': len(all_data),
        'by_type': {k: len(v) for k, v in by_type.items()},
        'data': all_data
    }

    with open('climatematch_data_urls.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Create organized summary
    print("\n=== Data URL Summary ===")
    print(f"Total unique data URLs: {len(all_data)}")
    for data_type, items in by_type.items():
        print(f"\n{data_type}: {len(items)} URLs")
        print("Sample URLs:")
        for item in items[:3]:
            print(f"  - {item['filename']}: {item['url']}")

    # Save URLs by type
    for data_type, items in by_type.items():
        filename = f"urls_{data_type.lower().replace(' ', '_')}.txt"
        with open(filename, 'w') as f:
            for item in items:
                f.write(f"{item['url']}\t{item['filename']}\n")
        print(f"\nSaved {data_type} URLs to {filename}")

if __name__ == '__main__':
    main()
