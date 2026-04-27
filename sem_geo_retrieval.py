"""
Combined boundary-shape + room-semantic retrieval for 2D floor plans.

Combined descriptor  (shape_w × shape_part) ⊕ (sem_w × sem_part)
  shape_part  (48-dim, L2-normalised Fourier):
    - sample 128 pts uniformly along perimeter
    - centroid-distance profile, normalise by mean  → scale invariant
    - FFT magnitude of first 48 coefficients        → phase invariant

  sem_part  (22-dim, [0,1]-scaled):
    For each of 10 room groups (bedrooms merged):
      [count / max_count,  total_area / 256²]
    Plus: [total_room_count / 15,  boundary_area / 256²]

    Room groups:
      bedroom  = {1 Master, 5 Child, 7 Second}  ← all treated as bedrooms
      living   = {0}   kitchen  = {2}
      bathroom = {3}   dining   = {4}
      study    = {6}   guest    = {8}
      balcony  = {9}   entrance = {10}   storage = {11}

Boundary fit (--fit):
    Uniformly scales each plan so its boundary bounding box fills
    [0, SCALE]², making shape descriptors truly scale-invariant.

Augmentation (--augment):
    Encodes 8 geometric variants per plan into the index:
      r000 r090 r180 r270  (CCW rotations)
      m000 m090 m180 m270  (horizontal mirror × rotations)
    Each variant is stored as a separate named entry (e.g. '42_r090').

Usage:
  python3 sem_geo_retrieval.py --query 1.json --data_dir dataset/ --topn 5
  python3 sem_geo_retrieval.py --query 1.json --data_dir dataset/ \\
      --fit --augment --build_index db.bin
  python3 sem_geo_retrieval.py --query 1.json --load_index db.bin --topn 8 \\
      --shape_w 1.0 --sem_w 0.5 --out result.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import numpy as np

# ── constants ────────────────────────────────────────────────────────────────
SCALE    = 256.0
N_SAMP   = 128
N_FREQS  = 48

SEMANTICS = {
    0: 'Living',   1: 'Master',   2: 'Kitchen',
    3: 'Bathroom', 4: 'Dining',   5: 'Child',
    6: 'Study',    7: 'Second',   8: 'Guest',
    9: 'Balcony',  10: 'Entrance',11: 'Storage',
    12: 'Wall-in', 13: 'External',14: 'Ext.wall',
    15: 'FrontDoor',16: 'Int.wall',17: 'Int.door',
}

ROOM_COLORS = [
    '#E6B8A2', '#B8D4E8', '#B8E8B8', '#E8E8B8', '#D4B8E8',
    '#E8B8D4', '#B8E8E8', '#E8D4B8', '#C8E8C8', '#B8C8E8',
    '#E8C8B8', '#D8D8D8', '#AAAAAA', '#CCCCCC', '#888888',
    '#FFAAAA', '#BBBBBB', '#DDDDDD',
]

# ── semantic groups ──────────────────────────────────────────────────────────
# (name, label_set, max_expected_count)
ROOM_GROUPS = [
    ('bedroom',  {1, 5, 7}, 5),
    ('living',   {0},        2),
    ('kitchen',  {2},        2),
    ('bathroom', {3},        4),
    ('dining',   {4},        2),
    ('study',    {6},        2),
    ('guest',    {8},        2),
    ('balcony',  {9},        3),
    ('entrance', {10},       2),
    ('storage',  {11},       3),
]
# abbreviated labels for the room-summary annotation
GROUP_ABBR = {
    'bedroom': 'Bed', 'living': 'Liv', 'kitchen': 'Kit',
    'bathroom': 'Bath', 'dining': 'Din', 'study': 'Stu',
    'guest': 'Gst', 'balcony': 'Bal', 'entrance': 'Ent', 'storage': 'Sto',
}


# ── geometry helpers ─────────────────────────────────────────────────────────

def segs_to_verts(segments):
    return [(float(s[0]), float(s[1])) for s in segments]


def polygon_area_segs(segments):
    """Shoelace area from ordered [x1,y1,x2,y2] segment list."""
    xs = [s[0] for s in segments]
    ys = [s[1] for s in segments]
    n  = len(xs)
    a  = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(a) / 2.0


def sample_perimeter(verts, n):
    pts = np.array(verts + [verts[0]], dtype=float)
    diffs   = np.diff(pts, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    cum     = np.concatenate([[0.0], np.cumsum(seg_len)])
    total   = cum[-1]
    if total < 1e-9:
        return np.zeros((n, 2))
    positions = np.linspace(0, total, n, endpoint=False)
    result = np.empty((n, 2))
    for i, pos in enumerate(positions):
        idx = int(np.searchsorted(cum, pos, side='right')) - 1
        idx = min(idx, len(seg_len) - 1)
        t   = (pos - cum[idx]) / (seg_len[idx] + 1e-12)
        result[i] = pts[idx] + t * diffs[idx]
    return result


def boundary_poly_arr(boundary_segs):
    v = segs_to_verts(boundary_segs)
    v.append(v[0])
    return np.array(v)



# ── geometric augmentation ────────────────────────────────────────────────────

# 8-fold symmetry: 4 rotations × (identity + horizontal mirror)
# All transforms operate on coordinates already in [0, SCALE]² and keep
# them in [0, SCALE]² by re-centring after each transform.

def _apply_transform(pts, M):
    """Apply 2×2 matrix M to a list of [x,y] points, then shift back so
    the bounding box starts at (0,0) and is scaled to fill SCALE."""
    arr = np.array(pts, dtype=float)
    arr = arr @ M.T
    # re-centre into [0, SCALE]²
    arr -= arr.min(axis=0)
    span = arr.max(axis=0)
    span = np.where(span < 1e-9, 1.0, span)
    arr  = arr / span * SCALE
    return arr.tolist()


# rotation matrices (CCW) and mirror for square grid
_AUG_TRANSFORMS = [
    ('r000',    np.array([[ 1,  0], [ 0,  1]], dtype=float)),
    ('r090',    np.array([[ 0, -1], [ 1,  0]], dtype=float)),
    ('r180',    np.array([[-1,  0], [ 0, -1]], dtype=float)),
    ('r270',    np.array([[ 0,  1], [-1,  0]], dtype=float)),
    ('m000',    np.array([[-1,  0], [ 0,  1]], dtype=float)),   # H-mirror
    ('m090',    np.array([[ 0,  1], [ 1,  0]], dtype=float)),   # H-mirror + r090
    ('m180',    np.array([[ 1,  0], [ 0, -1]], dtype=float)),   # H-mirror + r180
    ('m270',    np.array([[ 0, -1], [-1,  0]], dtype=float)),   # H-mirror + r270
]


def augment_plan(data):
    """Return a list of 8 data dicts, one per augmentation variant.
    Each dict has 'name' suffixed with the variant tag (e.g. 'plan_r090').
    The 'boundary' and 'coord' fields are transformed; 'labels' is unchanged.
    """
    variants = []
    base_name = data.get('name', 'plan')
    boundary  = data.get('boundary', [])
    coord     = data.get('coord', [])

    for tag, M in _AUG_TRANSFORMS:
        new_bdry  = _apply_transform(boundary, M) if boundary else []
        new_coord = [_apply_transform(room, M) for room in coord]
        variants.append({
            **data,
            'name':     f'{base_name}_{tag}',
            'boundary': new_bdry,
            'coord':    new_coord,
        })
    return variants


# ── v2.1.0 adapters ──────────────────────────────────────────────────────────

def boundary_v2_to_segs(boundary_v2):
    """Convert v2.1.0 boundary vertex list to legacy segment format.
    Extracts [x,y] from each vertex, scales by 256, returns as list
    of [x,y] pairs (one per vertex, closing vertex excluded)."""
    return [[v[0] * SCALE, v[1] * SCALE] for v in boundary_v2]


def rooms_v2_to_legacy(rooms_v2):
    """Convert v2.1.0 rooms list to legacy coord/labels format.
    Returns (coord_list, labels_list) where:
      coord_list  — list of [[x*256,y*256],...] per room (polygon vertices)
      labels_list — list of source_type integers (original RPLAN types)
    Rooms with fewer than 3 polygon vertices are skipped."""
    coord_list, labels_list = [], []
    for room in rooms_v2:
        poly = room.get('polygon', [])
        if len(poly) < 3:
            continue
        coord_list.append([[v[0] * SCALE, v[1] * SCALE] for v in poly])
        labels_list.append(room['source_type'])
    return coord_list, labels_list


def load_plan_v2(path):
    """Load a v2.1.0 JSON and return a dict in the legacy internal format
    that all descriptor functions expect:
      {
        'name':     id string,
        'boundary': legacy segment list (scaled to 256),
        'coord':    legacy coord list (scaled to 256),
        'labels':   legacy labels list (source_type integers),
      }
    Raises ValueError if boundary is missing or has fewer than 3 vertices.
    """
    with open(path) as f:
        data = json.load(f)

    bdry_raw  = data.get('boundary', [])
    rooms_raw = data.get('rooms', [])

    # Detect v2.1.0: 'rooms' present, boundary entries are 4-element lists,
    # and rooms have 'source_type'
    is_v2 = (
        rooms_raw
        and bdry_raw
        and isinstance(bdry_raw[0], list)
        and len(bdry_raw[0]) == 4
        and 'source_type' in rooms_raw[0]
    )

    if is_v2:
        if len(bdry_raw) < 3:
            raise ValueError(f"Boundary has fewer than 3 vertices: {path}")
        boundary = boundary_v2_to_segs(bdry_raw)
        coord, labels = rooms_v2_to_legacy(rooms_raw)
        return {
            'name':     data.get('id', os.path.splitext(os.path.basename(path))[0]),
            'boundary': boundary,
            'coord':    coord,
            'labels':   labels,
        }
    else:
        # Legacy format — load directly without conversion
        if not bdry_raw:
            raise ValueError(f"Boundary missing or empty: {path}")
        data.setdefault('name', os.path.splitext(os.path.basename(path))[0])
        return data


# ── shape descriptor ─────────────────────────────────────────────────────────

def shape_desc(boundary_segs, n_samp=N_SAMP, n_freqs=N_FREQS):
    """Centroid-distance Fourier descriptor, L2-normalised. (n_freqs,) float32."""
    verts = segs_to_verts(boundary_segs)
    if len(verts) < 3:
        return np.zeros(n_freqs, dtype=np.float32)

    pts      = sample_perimeter(verts, n_samp)
    centroid = pts.mean(axis=0)
    dists    = np.linalg.norm(pts - centroid, axis=1)
    mean_d   = dists.mean()
    if mean_d < 1e-9:
        return np.zeros(n_freqs, dtype=np.float32)

    dists /= mean_d
    fft = np.fft.rfft(dists)
    mag = np.abs(fft)[1: n_freqs + 1]
    if len(mag) < n_freqs:
        mag = np.pad(mag, (0, n_freqs - len(mag)))

    # L2-normalise so shape_w / sem_w are on equal footing
    norm = np.linalg.norm(mag)
    if norm > 1e-9:
        mag = mag / norm
    return mag.astype(np.float32)


# ── semantic descriptor ───────────────────────────────────────────────────────

def sem_desc(data):
    """
    22-dim semantic feature in [0, 1].
    Layout: for each group [count/max, area/256²]  (20-dim)
            then [total_rooms/15,  boundary_area/256²]  (2-dim)
    """
    coords   = data.get('coord', [])
    labels   = data.get('labels', [])
    boundary = data.get('boundary', [])

    # per-group accumulators
    group_count = {g: 0   for g, _, _ in ROOM_GROUPS}
    group_area  = {g: 0.0 for g, _, _ in ROOM_GROUPS}

    for room_segs, lbl in zip(coords, labels):
        for gname, gset, _ in ROOM_GROUPS:
            if lbl in gset:
                group_count[gname] += 1
                group_area[gname]  += polygon_area_segs(room_segs)
                break

    feat = []
    for gname, _, max_cnt in ROOM_GROUPS:
        feat.append(min(group_count[gname] / max_cnt, 1.0))
        feat.append(min(group_area[gname]  / (SCALE ** 2), 1.0))

    total_rooms   = len(labels)
    bdry_area     = polygon_area_segs(boundary) if boundary else 0.0
    feat.append(min(total_rooms / 15.0, 1.0))
    feat.append(min(bdry_area   / (SCALE ** 2), 1.0))

    return np.array(feat, dtype=np.float32)   # (22,)


# ── combined descriptor ───────────────────────────────────────────────────────

def requirements_sem_hint(requirements, boundary_area=0.0):
    """
    Build a partial 22-dim semantic vector from a requirements dict
    {bedrooms, bathrooms, living, kitchen, …}.  Unknown groups stay 0.
    Used to make boundary-only queries semantically meaningful.

    requirements keys recognised (all optional, integers):
        bedrooms, bathrooms, living, kitchen, dining, study,
        guest, balcony, entrance, storage
    """
    req = requirements or {}
    # map friendly names → group names used in ROOM_GROUPS
    name_map = {
        'bedrooms':  'bedroom',
        'bathrooms': 'bathroom',
        'living':    'living',
        'kitchen':   'kitchen',
        'dining':    'dining',
        'study':     'study',
        'guest':     'guest',
        'balcony':   'balcony',
        'entrance':  'entrance',
        'storage':   'storage',
    }
    group_count = {g: 0 for g, _, _ in ROOM_GROUPS}
    for req_key, gname in name_map.items():
        v = req.get(req_key)
        if v is not None:
            group_count[gname] = int(v)

    # estimate area per room as a typical fraction of boundary area
    # (use median area fractions observed across dataset: ~0.12 per room)
    AREA_PER_ROOM = 0.12 * boundary_area

    feat = []
    total_rooms = 0
    for gname, _, max_cnt in ROOM_GROUPS:
        cnt = group_count[gname]
        total_rooms += cnt
        feat.append(min(cnt / max_cnt, 1.0))
        feat.append(min(cnt * AREA_PER_ROOM / (SCALE ** 2), 1.0))

    feat.append(min(total_rooms / 15.0, 1.0))
    feat.append(min(boundary_area / (SCALE ** 2), 1.0))
    return np.array(feat, dtype=np.float32)   # (22,)


def combined_desc(data, shape_w=1.0, sem_w=0.5, requirements=None):
    """Concatenate weighted shape and semantic parts.

    If requirements is provided (and data has no rooms), the semantic
    part is built from requirements instead of from empty room lists.
    """
    bdry = data.get('boundary', [])
    sd   = shape_desc(bdry) * shape_w if bdry else np.zeros(N_FREQS, np.float32)

    has_rooms = bool(data.get('coord') or data.get('labels'))
    if not has_rooms and requirements:
        bdry_area = polygon_area_segs(bdry) if bdry else 0.0
        semd = requirements_sem_hint(requirements, bdry_area) * sem_w
    else:
        semd = sem_desc(data) * sem_w

    return np.concatenate([sd, semd])


# ── room-summary string for annotation ───────────────────────────────────────

def room_summary(data):
    coords = data.get('coord', [])
    labels = data.get('labels', [])
    counts = {g: 0 for g, _, _ in ROOM_GROUPS}
    for _, lbl in zip(coords, labels):
        for gname, gset, _ in ROOM_GROUPS:
            if lbl in gset:
                counts[gname] += 1
                break
    parts = [f"{GROUP_ABBR[g]}:{counts[g]}"
             for g, _, _ in ROOM_GROUPS if counts[g] > 0]
    return '  '.join(parts)


# ── database ─────────────────────────────────────────────────────────────────

def build_db(data_dir, name_json=None, shape_w=1.0, sem_w=0.5,
             augment=False):
    if name_json is not None:
        with open(name_json) as f:
            meta = json.load(f)
        ordered = [meta[k] for k in sorted(meta, key=lambda x: int(x))]
    else:
        ordered = [f.replace('.json', '.png')
                   for f in sorted(os.listdir(data_dir)) if f.endswith('.json')]

    names, descs, raws = [], [], []
    missing = 0
    for png_name in ordered:
        stem = os.path.splitext(png_name)[0]
        path = os.path.join(data_dir, stem + '.json')
        if not os.path.exists(path):
            missing += 1
            continue
        try:
            data = load_plan_v2(path)
        except (ValueError, KeyError, IndexError, json.JSONDecodeError):
            missing += 1
            continue

        variants = augment_plan(data) if augment else [data]
        for v in variants:
            names.append(v['name'])
            descs.append(combined_desc(v, shape_w, sem_w))
            raws.append(v)

    if missing:
        print(f"  Skipped {missing} files")
    return names, np.stack(descs), raws


# ── retrieval ─────────────────────────────────────────────────────────────────

def retrieve(q_desc, db_descs, topn):
    dists = np.linalg.norm(db_descs - q_desc[None], axis=1)
    idx   = np.argsort(dists)[:topn]
    return idx, dists[idx]


# ── visualisation ─────────────────────────────────────────────────────────────

def draw_plan(ax, data, title='', dist=None, shape_d=None, sem_d=None):
    coords   = data.get('coord', [])
    labels   = data.get('labels', [])
    boundary = data.get('boundary', [])

    ax.set_aspect('equal')
    ax.axis('off')

    # filled rooms
    patch_list, color_list = [], []
    for room_segs, lbl in zip(coords, labels):
        verts = segs_to_verts(room_segs)
        if len(verts) < 3:
            continue
        patch_list.append(MplPolygon(verts, closed=True))
        color_list.append(ROOM_COLORS[lbl % len(ROOM_COLORS)])

    if patch_list:
        pc = PatchCollection(patch_list, facecolors=color_list,
                             edgecolors='none', alpha=0.65, zorder=1)
        ax.add_collection(pc)

    # boundary
    if boundary:
        bp = boundary_poly_arr(boundary)
        ax.plot(bp[:, 0], bp[:, 1], color='#222222', linewidth=1.8, zorder=2)
        xs, ys = bp[:, 0], bp[:, 1]
        pad = max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.06 + 1
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)

    # title
    lines = [title]
    if dist is not None:
        score_line = f'dist={dist:.4f}'
        if shape_d is not None and sem_d is not None:
            score_line += f'  (shp={shape_d:.3f} sem={sem_d:.3f})'
        lines.append(score_line)
    ax.set_title('\n'.join(lines), fontsize=6.5, pad=2)

    # room summary annotation below
    summary = room_summary(data)
    ax.text(0.5, -0.02, summary, transform=ax.transAxes,
            ha='center', va='top', fontsize=5.5, color='#333333',
            wrap=True)


def visualize(query_data, result_data, result_names, result_dists,
              shape_dists, sem_dists, topn, out=None):
    ncols = topn + 1
    fig, axes = plt.subplots(1, ncols,
                             figsize=(3.2 * ncols, 4.2),
                             gridspec_kw={'wspace': 0.05})
    if ncols == 1:
        axes = [axes]

    draw_plan(axes[0], query_data,
              title=f"Query  {query_data.get('name', '')}")

    for i in range(topn):
        draw_plan(axes[i + 1], result_data[i],
                  title=f'#{i+1}  {result_names[i]}',
                  dist=result_dists[i],
                  shape_d=shape_dists[i],
                  sem_d=sem_dists[i])

    # room-type legend
    handles = [
        mpatches.Patch(facecolor=ROOM_COLORS[t],
                       edgecolor='#999', linewidth=0.4,
                       label=f'{t}: {SEMANTICS[t]}')
        for t in SEMANTICS if t < len(ROOM_COLORS)
    ]
    fig.legend(handles=handles, loc='lower center', ncol=9,
               fontsize=5.5, frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.suptitle('Boundary Shape + Room Semantic Retrieval',
                 fontsize=10, y=1.01)

    if out:
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved → {out}")
    else:
        plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Boundary-shape + semantic retrieval for floor plans.')
    parser.add_argument('--query',       required=True)
    parser.add_argument('--data_dir',    default=None)
    parser.add_argument('--name_json',   default=None)
    parser.add_argument('--build_index', default=None, metavar='PATH',
                        help='Build database and save descriptor index to PATH')
    parser.add_argument('--load_index',  default=None, metavar='PATH',
                        help='Load precomputed descriptor index from PATH')
    parser.add_argument('--topn',      type=int,   default=5)
    parser.add_argument('--shape_w',   type=float, default=1.0,
                        help='Weight for boundary shape part (default 1.0)')
    parser.add_argument('--sem_w',     type=float, default=0.5,
                        help='Weight for semantic part (default 0.5)')
    parser.add_argument('--out',       default=None)
    parser.add_argument('--augment', action='store_true',
                        help='Store 8 augmented variants per plan (4 rotations × H-mirror) in the index')
    args = parser.parse_args()

    if args.load_index is None and args.data_dir is None:
        parser.error('--data_dir is required unless --load_index is provided')

    # ── query ──
    query_data  = load_plan_v2(args.query)
    q_desc      = combined_desc(query_data, args.shape_w, args.sem_w)
    q_shape     = shape_desc(query_data['boundary']) * args.shape_w
    q_sem       = sem_desc(query_data) * args.sem_w

    print(f"Query : {query_data.get('name', args.query)}")
    print(f"  descriptor  shape={N_FREQS}  sem=22  "
          f"weights shape_w={args.shape_w}  sem_w={args.sem_w}")
    print(f"  rooms: {room_summary(query_data)}")

    # ── database ──
    if args.load_index is not None:
        import pickle
        print(f"\nLoading index from: {args.load_index}")
        with open(args.load_index, 'rb') as f:
            _idx = pickle.load(f)
        db_names = _idx['names']
        db_descs = _idx['descs']
        db_raws  = _idx['raws']
        print(f"  {len(db_names)} plans loaded  "
              f"|  desc dim: {db_descs.shape[1]}")
    else:
        print(f"\nBuilding database from: {args.data_dir}")
        db_names, db_descs, db_raws = build_db(
            args.data_dir, args.name_json, args.shape_w, args.sem_w,
            augment=args.augment)
        print(f"  {len(db_names)} plans indexed  "
              f"|  desc dim: {db_descs.shape[1]}")

    if args.build_index is not None:
        import pickle
        print(f"\nSaving index to: {args.build_index}")
        with open(args.build_index, 'wb') as f:
            pickle.dump({'names': db_names, 'descs': db_descs, 'raws': db_raws}, f)
        print("  Index saved.")

    # ── retrieve ──
    idx, dists = retrieve(q_desc, db_descs, args.topn)

    # recompute per-part distances for display
    db_shape = np.stack([shape_desc(db_raws[i]['boundary']) * args.shape_w
                         for i in idx])
    db_sem   = np.stack([sem_desc(db_raws[i]) * args.sem_w for i in idx])
    shape_d  = np.linalg.norm(db_shape - q_shape[None], axis=1)
    sem_d    = np.linalg.norm(db_sem   - q_sem[None],   axis=1)

    print(f"\nTop-{args.topn} results (combined / shape / semantic distance):")
    for rank, (i, d, sd, semd) in enumerate(
            zip(idx, dists, shape_d, sem_d), 1):
        print(f"  #{rank:2d}  {db_names[i]:20s}  "
              f"combined={d:.4f}  shp={sd:.4f}  sem={semd:.4f}"
              f"  [{room_summary(db_raws[i])}]")

    # ── visualise ──
    visualize(
        query_data,
        [db_raws[i]  for i in idx],
        [db_names[i] for i in idx],
        dists, shape_d, sem_d,
        args.topn,
        out=args.out,
    )


if __name__ == '__main__':
    main()
