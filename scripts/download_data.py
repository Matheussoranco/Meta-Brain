#!/usr/bin/env python3
"""
Download MaleCNS connectome data from Janelia.

This script downloads the v1.0 dataset files needed for Meta-Brain.
Files are large (several GB), so this may take a while.

Data source: https://male-cns.janelia.org/download/
Official v1.0 release files:
- body-annotations-male-cns-v1.0-minconf-0.5.feather (neuron annotations)
- body-neurotransmitters-male-cns-v1.0.feather (neurotransmitter predictions)
- connectome-weights-male-cns-v1.0-minconf-0.5.feather (full weighted connectome)
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
    parser.add_argument('--version', default='v1.0', choices=['v1.0'],
                        help='Dataset version (default: v1.0)')
    parser.add_argument('--files', nargs='+',
                        choices=['annotations', 'neurotransmitters', 'weights', 'all'],
                        default=['annotations', 'neurotransmitters', 'weights'],
                        help='Files to download (default: all three v1.0 files)')
    parser.add_argument('--method', choices=['gsutil', 'wget', 'fsspec'], default='gsutil',
                        help='Download method (default: gsutil)')

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Official MaleCNS v1.0 file names from https://male-cns.janelia.org/download/
    base_gcs = "gs://flyem-male-cns/v1.0/connectome-data/flat-connectome"
    base_https = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

    file_map = {
        'annotations': {
            'gcs': f"{base_gcs}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
            'https': f"{base_https}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
            'filename': "body-annotations-male-cns-v1.0-minconf-0.5.feather",
            'size_mb': 13,
        },
        'neurotransmitters': {
            'gcs': f"{base_gcs}/body-neurotransmitters-male-cns-v1.0.feather",
            'https': f"{base_https}/body-neurotransmitters-male-cns-v1.0.feather",
            'filename': "body-neurotransmitters-male-cns-v1.0.feather",
            'size_mb': 42,
        },
        'weights': {
            'gcs': f"{base_gcs}/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
            'https': f"{base_https}/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
            'filename': "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
            'size_mb': 1100,  # ~1.1 GB
        },
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
    total_size = sum(file_map[f]['size_mb'] for f in files_to_download)
    print(f"Files: {files_to_download} (~{total_size} MB total)")
    print(f"Method: {args.method}")
    print()

    success = True
    for file_key in files_to_download:
        if file_key not in file_map:
            print(f"Unknown file: {file_key}")
            continue

        info = file_map[file_key]
        source = info['gcs'] if args.method == 'gsutil' else info['https']
        dest = data_dir / info['filename']

        print(f"[{file_key}] {info['filename']} (~{info['size_mb']} MB)")

        if args.method == 'gsutil':
            ok = download_with_gsutil(source, dest)
        elif args.method == 'fsspec':
            # Use fsspec for streaming download (good for Colab)
            try:
                import fsspec
                fs = fsspec.filesystem("gcs", token="anon")
                gcs_path = source.replace("gs://", "")
                with fs.open(gcs_path, "rb") as src, open(dest, "wb") as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)
                print(f"  Downloaded via fsspec")
                ok = True
            except Exception as e:
                print(f"  fsspec failed: {e}")
                ok = False
        else:
            # wget with HTTPS URL
            ok = download_with_wget(info['https'], dest)

        if not ok:
            success = False

    if success:
        print("\n✓ All downloads completed successfully!")
        print(f"Data saved to: {data_dir}")
        print("\nNext steps:")
        print("1. Run the Colab notebook or local training script")
        print("2. The notebook uses these exact file names")
    else:
        print("\n✗ Some downloads failed")
        sys.exit(1)


if __name__ == '__main__':
    main()