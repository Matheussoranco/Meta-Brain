#!/usr/bin/env python3
"""
Download MaleCNS connectome data from Janelia.

This script downloads the v1.0 dataset files needed for Meta-Brain.
Files are large (several GB), so this may take a while.

Data source: https://male-cns.janelia.org/download/
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def check_gsutil():
    """Check if gsutil is available."""
    try:
        subprocess.run(['gsutil', 'version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_with_gsutil(source: str, dest: Path):
    """Download using gsutil."""
    print(f"Downloading {source} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    result = subprocess.run([
        'gsutil', '-m', 'cp', '-r', source, str(dest)
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    print("Download complete")
    return True


def download_with_wget(url: str, dest: Path):
    """Download using wget."""
    print(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    result = subprocess.run([
        'wget', '-c', '-O', str(dest), url
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    print("Download complete")
    return True


def main():
    parser = argparse.ArgumentParser(description='Download MaleCNS connectome data')
    parser.add_argument('--data-dir', default='data/connectome', 
                        help='Directory to store data (default: data/connectome)')
    parser.add_argument('--version', default='v1.0', choices=['v0.9', 'v1.0'],
                        help='Dataset version (default: v1.0)')
    parser.add_argument('--files', nargs='+', 
                        choices=['meta', 'edgelist', 'split_edgelist', 'synapses', 
                                 'skeletons_malecns', 'skeletons_banc', 'obj', 'all'],
                        default=['meta', 'edgelist'],
                        help='Files to download (default: meta, edgelist)')
    parser.add_argument('--method', choices=['gsutil', 'wget'], default='gsutil',
                        help='Download method (default: gsutil)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Base URL for Google Cloud Storage
    base_url = f"gs://flyem-male-cns/{args.version}/connectome-data/flat-connectome"
    
    # File mappings
    file_map = {
        'meta': f"{base_url}/malecns_{args.version.replace('.', '_')}_meta.feather",
        'edgelist': f"{base_url}/malecns_{args.version.replace('.', '_')}_simple_edgelist.feather",
        'split_edgelist': f"{base_url}/malecns_{args.version.replace('.', '_')}_split_edgelist.feather",
        'synapses': f"{base_url}/malecns_{args.version.replace('.', '_')}_synapses.parquet",
        'skeletons_malecns': f"gs://flyem-male-cns/{args.version}/connectome-data/skeletons/malecns_malecns_space_swc/",
        'skeletons_banc': f"gs://flyem-male-cns/{args.version}/connectome-data/skeletons/malecns_banc_space_swc/",
        'obj': f"gs://flyem-male-cns/{args.version}/connectome-data/meshes/obj/",
    }
    
    # Check download method
    if args.method == 'gsutil':
        if not check_gsutil():
            print("gsutil not found. Install Google Cloud SDK: https://cloud.google.com/sdk")
            print("Falling back to wget...")
            args.method = 'wget'
    
    # Determine files to download
    if 'all' in args.files:
        files_to_download = list(file_map.keys())
    else:
        files_to_download = args.files
    
    print(f"Downloading MaleCNS {args.version} data to {data_dir}")
    print(f"Files: {files_to_download}")
    print(f"Method: {args.method}")
    print()
    
    success = True
    for file_key in files_to_download:
        if file_key not in file_map:
            print(f"Unknown file: {file_key}")
            continue
        
        source = file_map[file_key]
        dest = data_dir / file_key
        
        if args.method == 'gsutil':
            ok = download_with_gsutil(source, dest)
        else:
            # For wget, need HTTP URLs
            http_url = source.replace('gs://', 'https://storage.googleapis.com/')
            ok = download_with_wget(http_url, dest)
        
        if not ok:
            success = False
    
    if success:
        print("\n✓ All downloads completed successfully!")
        print(f"Data saved to: {data_dir}")
        print("\nNext steps:")
        print("1. Run: meta-brain --mode baseline")
        print("2. Then: meta-brain --mode plasticity")
    else:
        print("\n✗ Some downloads failed")
        sys.exit(1)


if __name__ == '__main__':
    main()