"""
build_index_ent.py
Build a browser-compatible retrieval index using the entrance-aware descriptor
from geo_ent_num_retrieval.py.

Descriptor layout (63 floats per plan)
───────────────────────────────────────
  [  0 :  48 ]  shape    — L2-normalised centroid-distance Fourier (48 dims)
  [ 48 :  52 ]  entrance — door position + wall normal (4 dims)
  [ 52 :  63 ]  rooms    — per-type room counts, normalised (11 dims)

Outputs
───────
  index.bin        — raw float32 matrix, shape (N, 63), written to --out_dir
  index_names.json — list of plan PNG names, length N

Usage
─────
  python build_index_ent.py --data_dir test_data --out_dir browser_files
  python build_index_ent.py --data_dir test_data --out_dir browser_files \\
                            --name_json browser_files/index_names.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from geo_ent_num_retrieval import (
    shape_desc, entrance_desc, room_count_desc,
    frontdoor_center,
    N_FREQS, ENT_DIM, ROOM_DIM,
)

SHAPE_DIM = N_FREQS          # 48
TOTAL_DIM = SHAPE_DIM + ENT_DIM + ROOM_DIM   # 63


def build(data_dir: Path, name_json=None):
    """Return (names, descriptor_matrix) for all valid plans in data_dir."""
    if name_json and Path(name_json).exists():
        with open(name_json) as fh:
            meta = json.load(fh)
        # meta can be a list or a {idx: name} dict
        if isinstance(meta, dict):
            ordered = [meta[k] for k in sorted(meta, key=lambda x: int(x))]
        else:
            ordered = meta
    else:
        ordered = [
            f.replace('.json', '.png')
            for f in sorted(os.listdir(data_dir))
            if f.endswith('.json')
        ]

    names, rows = [], []
    skipped_missing = 0
    skipped_invalid = 0

    for png_name in ordered:
        stem = Path(png_name).stem
        path = data_dir / (stem + '.json')
        if not path.exists():
            skipped_missing += 1
            continue
        with open(path) as fh:
            data = json.load(fh)
        bdry = data.get('boundary')
        if not bdry or frontdoor_center(data) is None:
            skipped_invalid += 1
            continue

        s = shape_desc(bdry)
        e = entrance_desc(data)
        r = room_count_desc(data)

        row = np.concatenate([s, e, r])           # (63,)
        names.append(png_name)
        rows.append(row)

    if skipped_missing:
        print(f"  Skipped {skipped_missing} — JSON not found")
    if skipped_invalid:
        print(f"  Skipped {skipped_invalid} — no boundary or no front door")

    if not rows:
        raise ValueError("No valid plans found.")

    matrix = np.stack(rows, axis=0).astype(np.float32)   # (N, 63)
    return names, matrix


def main():
    parser = argparse.ArgumentParser(
        description="Build entrance-aware retrieval index for the browser tool.")
    parser.add_argument('--data_dir',  default='test_data',
                        help='Directory containing plan JSON files (default: test_data)')
    parser.add_argument('--out_dir',   default='browser_files',
                        help='Output directory (default: browser_files)')
    parser.add_argument('--name_json', default=None,
                        help='Optional existing index_names.json to preserve ordering')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning: {data_dir}/")
    names, matrix = build(data_dir, args.name_json)
    N, dim = matrix.shape
    print(f"  {N} plans indexed  |  dim = {dim}  "
          f"(shape:{SHAPE_DIM} + entrance:{ENT_DIM} + rooms:{ROOM_DIM})")
    print(f"  Matrix size: {matrix.nbytes / 1e6:.1f} MB")

    bin_path   = out_dir / 'index.bin'
    names_path = out_dir / 'index_names.json'

    matrix.tofile(str(bin_path))
    with open(names_path, 'w') as fh:
        json.dump(names, fh)

    print(f"\nExported:")
    print(f"  {bin_path}   ({bin_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  {names_path} ({names_path.stat().st_size / 1e3:.1f} KB)")
    print(f"\nPlace these files alongside retrieval_tool.html (or in browser_files/)")
    print(f"and load them in the browser. Descriptor dim = {dim}.")


if __name__ == '__main__':
    main()
