"""
export_index.py
Converts a precomputed retrieval index (.pkl) to browser-readable format.

Usage:
    python export_index.py --index index.pkl --out_dir ./browser_index

Outputs:
    index.bin        — raw float32 descriptor matrix, shape (N, dim)
    index_names.json — list of plan filenames, length N

Place both output files alongside retrieval_tool.html and load them in the browser.
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Export retrieval index for browser tool')
    parser.add_argument('--index',   required=True, help='Path to .pkl index file')
    parser.add_argument('--out_dir', default='.',   help='Output directory (default: current dir)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading index: {args.index}")
    with open(args.index, 'rb') as f:
        idx = pickle.load(f)

    names = idx['names']
    descs = np.array(idx['descs'], dtype=np.float32)

    print(f"  Plans:          {len(names)}")
    print(f"  Descriptor dim: {descs.shape[1]}")
    print(f"  Matrix size:    {descs.nbytes / 1e6:.1f} MB")

    bin_path   = out_dir / 'index.bin'
    names_path = out_dir / 'index_names.json'

    descs.tofile(str(bin_path))
    with open(names_path, 'w') as f:
        json.dump(names, f)

    print(f"\nExported:")
    print(f"  {bin_path}   ({bin_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  {names_path} ({names_path.stat().st_size / 1e3:.1f} KB)")
    print(f"\nPlace these files alongside retrieval_tool.html and load them in the browser.")


if __name__ == '__main__':
    main()