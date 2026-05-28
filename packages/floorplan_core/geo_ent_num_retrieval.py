"""
Boundary-shape + front-door position + room-count retrieval for floor plans.

Descriptor parts
────────────────
shape  (48-dim, ×shape_w) — L2-normalised Fourier on centroid-distance signal
entrance (4-dim, ×ent_w)  — front door relative to boundary (rotation-aware)
  ├ rel_x, rel_y : door snap on boundary, normalised by mean boundary radius
  └ cos_θ, sin_θ : outward wall normal at the door edge
rooms  (11-dim, ×room_w)  — per-type room counts, normalised by expected max
  living(0) master(1) kitchen(2) bathroom(3) dining(4) child(5) study(6)
  second(7) balcony(9) storage(11) + total_rooms

Shape and room-count descriptors are fully rotation-invariant.
Entrance distance is minimised analytically over all rotations.

Usage
─────
  python geo_ent_num_retrieval.py \\
      --query    dataset_gt_label_doors/0.json \\
      --data_dir dataset_gt_label_doors/ --topn 5

  python geo_ent_num_retrieval.py \\
      --query    dataset_gt_label_doors/0.json \\
      --data_dir dataset_gt_label_doors/ \\
      --shape_w 1.0 --ent_w 2.0 --room_w 1.0 --topn 8 --out result.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────
N_SAMP  = 256
N_FREQS = 48

SEMANTICS = {
    0: 'Living',   1: 'Master',    2: 'Kitchen',
    3: 'Bathroom', 4: 'Dining',    5: 'Child',
    6: 'Study',    7: 'Second',    8: 'Guest',
    9: 'Balcony',  10: 'Entrance', 11: 'Storage',
    12: 'Wall-in', 13: 'External', 14: 'Ext.wall',
    15: 'FrontDoor', 16: 'Int.wall', 17: 'Int.door',
}
ROOM_COLORS = [
    '#E6B8A2', '#B8D4E8', '#B8E8B8', '#E8E8B8', '#D4B8E8',
    '#E8B8D4', '#B8E8E8', '#E8D4B8', '#C8E8C8', '#B8C8E8',
    '#E8C8B8', '#D8D8D8', '#AAAAAA', '#CCCCCC', '#888888',
    '#FFAAAA', '#BBBBBB', '#DDDDDD',
]


# ── geometry helpers ──────────────────────────────────────────────────────────

def segs_to_verts(segs):
    return [(float(s[0]), float(s[1])) for s in segs]


def normalise_winding(verts):
    """Return verts in CW order (negative signed area). Reverses if CCW."""
    n = len(verts)
    area = sum(verts[i][0] * verts[(i+1) % n][1] -
               verts[(i+1) % n][0] * verts[i][1]
               for i in range(n))
    return verts[::-1] if area > 0 else verts


def boundary_poly_arr(bdry):
    v = segs_to_verts(bdry); v.append(v[0])
    return np.array(v)


def sample_perimeter(verts, n):
    pts     = np.array(verts + [verts[0]], dtype=float)
    diffs   = np.diff(pts, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    cum     = np.concatenate([[0.0], np.cumsum(seg_len)])
    total   = cum[-1]
    if total < 1e-9:
        return np.zeros((n, 2))
    positions = np.linspace(0, total, n, endpoint=False)
    result = np.empty((n, 2))
    for i, pos in enumerate(positions):
        idx = min(int(np.searchsorted(cum, pos, side='right')) - 1,
                  len(seg_len) - 1)
        t = (pos - cum[idx]) / (seg_len[idx] + 1e-12)
        result[i] = pts[idx] + t * diffs[idx]
    return result


def perimeter_cumlen(verts):
    """Cumulative arc-lengths and total length for a closed polygon."""
    pts = np.array(verts + [verts[0]], dtype=float)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    return np.concatenate([[0.0], np.cumsum(seg)]), seg.sum()


# ── shape descriptor ──────────────────────────────────────────────────────────

def shape_desc(bdry_segs, n_samp=N_SAMP, n_freqs=N_FREQS):
    """Centroid-distance Fourier, L2-normalised. (n_freqs,) float32."""
    verts = normalise_winding(segs_to_verts(bdry_segs))
    if len(verts) < 3:
        return np.zeros(n_freqs, dtype=np.float32)
    pts      = sample_perimeter(verts, n_samp)
    centroid = pts.mean(axis=0)
    dists    = np.linalg.norm(pts - centroid, axis=1)
    mean_d   = dists.mean()
    if mean_d < 1e-9:
        return np.zeros(n_freqs, dtype=np.float32)
    dists /= mean_d
    mag = np.abs(np.fft.rfft(dists))[1: n_freqs + 1]
    if len(mag) < n_freqs:
        mag = np.pad(mag, (0, n_freqs - len(mag)))
    norm = np.linalg.norm(mag)
    if norm > 1e-9:
        mag /= norm
    return mag.astype(np.float32)


# ── entrance descriptor ───────────────────────────────────────────────────────

def frontdoor_center(data):
    """Return (cx, cy) of the front door from door_rects, entrance bbox, or None.

    Handles both the old (door_rects) and v2 (entrance bbox) data formats.
    """
    # v2 format: entrance is a [x1, y1, x2, y2] bounding box
    ent = data.get('entrance')
    if ent and len(ent) == 4:
        return ((ent[0] + ent[2]) / 2.0, (ent[1] + ent[3]) / 2.0)

    # old format: door_rects list
    for r in data.get('door_rects', []):
        if r.get('label') == 15:
            pts = r['rect']
            return (sum(p[0] for p in pts) / len(pts),
                    sum(p[1] for p in pts) / len(pts))
    # fallback: FrontDoor room centroid (old coord/labels format)
    for segs, lbl in zip(data.get('coord', []), data.get('labels', [])):
        if lbl == 15 and segs:
            xs = [float(s[0]) for s in segs] + [float(s[2]) for s in segs]
            ys = [float(s[1]) for s in segs] + [float(s[3]) for s in segs]
            return sum(xs) / len(xs), sum(ys) / len(ys)
    return None


def entrance_desc(data):
    """
    4-dim entrance descriptor: [rel_x, rel_y, cos_θ, sin_θ]

    The door position is expressed *relative to the boundary shape*:

      rel_x, rel_y — 2-D vector from the boundary centroid to the closest
                     point on the boundary to the door, normalised by the
                     mean centroid-to-boundary distance (the same radius
                     scale used by the Fourier shape descriptor).  This is
                     invariant to which vertex is listed first, and places
                     the door geometrically within the shape rather than
                     as a raw perimeter fraction.

      cos_θ, sin_θ — outward normal of the closest boundary edge, encoding
                     which face of the building the door is on.

    Returns zeros if no front door or boundary is found.
    """
    dc = frontdoor_center(data)
    if dc is None:
        return np.zeros(4, dtype=np.float32)
    dx, dy = float(dc[0]), float(dc[1])

    bdry = data.get('boundary', [])
    if not bdry:
        return np.zeros(4, dtype=np.float32)

    verts = normalise_winding(segs_to_verts(bdry))
    n = len(verts)

    # boundary centroid and mean radius (from N_SAMP perimeter samples)
    pts      = sample_perimeter(verts, N_SAMP)
    centroid = pts.mean(axis=0)
    mean_r   = np.linalg.norm(pts - centroid, axis=1).mean()
    if mean_r < 1e-9:
        return np.zeros(4, dtype=np.float32)

    # find closest point on each boundary edge to the door
    best_dist = float('inf')
    best_px   = dx
    best_py   = dy
    best_edge = 0

    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        seg_l  = np.hypot(ex, ey)
        if seg_l < 1e-9:
            continue
        t  = max(0.0, min(1.0, ((dx - ax)*ex + (dy - ay)*ey) / seg_l**2))
        px, py = ax + t*ex, ay + t*ey
        d = np.hypot(dx - px, dy - py)
        if d < best_dist:
            best_dist = d
            best_px, best_py = px, py
            best_edge = i

    # relative position: from centroid to door's boundary snap, normalised
    rel_x = (best_px - centroid[0]) / mean_r
    rel_y = (best_py - centroid[1]) / mean_r

    # outward normal: rotate edge 90°, then ensure it points away from centroid
    ax, ay = verts[best_edge]
    bx, by = verts[(best_edge + 1) % n]
    ex, ey = bx - ax, by - ay
    seg_l  = np.hypot(ex, ey)
    nx, ny = ey / seg_l, -ex / seg_l
    # flip if pointing toward centroid instead of away
    if nx * (best_px - centroid[0]) + ny * (best_py - centroid[1]) < 0:
        nx, ny = -nx, -ny

    return np.array([rel_x, rel_y, nx, ny], dtype=np.float32)


def entrance_desc_from_point(door_xy, bdry_segs):
    """
    Compute a 4-dim entrance descriptor from an explicit door position.

    Parameters
    ----------
    door_xy   : (x, y) — door centre in the same coordinate space as the boundary
    bdry_segs : boundary segment list [[x, y, ...], ...]  (first two values used)

    Returns
    -------
    np.ndarray shape (4,) float32  — same [rel_x, rel_y, cos_θ, sin_θ] as entrance_desc()
    """
    if door_xy is None or not bdry_segs:
        return np.zeros(4, dtype=np.float32)

    dx, dy = float(door_xy[0]), float(door_xy[1])
    verts   = normalise_winding(segs_to_verts(bdry_segs))
    n       = len(verts)
    if n < 2:
        return np.zeros(4, dtype=np.float32)

    pts      = sample_perimeter(verts, N_SAMP)
    centroid = pts.mean(axis=0)
    mean_r   = np.linalg.norm(pts - centroid, axis=1).mean()
    if mean_r < 1e-9:
        return np.zeros(4, dtype=np.float32)

    best_dist = float('inf')
    best_px   = dx
    best_py   = dy
    best_edge = 0

    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        seg_l  = np.hypot(ex, ey)
        if seg_l < 1e-9:
            continue
        t  = max(0.0, min(1.0, ((dx - ax)*ex + (dy - ay)*ey) / seg_l**2))
        px, py = ax + t*ex, ay + t*ey
        d = np.hypot(dx - px, dy - py)
        if d < best_dist:
            best_dist = d
            best_px, best_py = px, py
            best_edge = i

    rel_x = (best_px - centroid[0]) / mean_r
    rel_y = (best_py - centroid[1]) / mean_r

    ax, ay = verts[best_edge]
    bx, by = verts[(best_edge + 1) % n]
    ex, ey = bx - ax, by - ay
    seg_l  = np.hypot(ex, ey)
    nx, ny = ey / seg_l, -ex / seg_l
    # flip if pointing toward centroid instead of away
    if nx * (best_px - centroid[0]) + ny * (best_py - centroid[1]) < 0:
        nx, ny = -nx, -ny

    return np.array([rel_x, rel_y, nx, ny], dtype=np.float32)


# ── room-count descriptor ─────────────────────────────────────────────────────

# (label_set, max_expected_count, display_name)
ROOM_TYPES = [
    ({0},       2, 'Living'),
    ({1},       1, 'Master'),
    ({2},       1, 'Kitchen'),
    ({3},       3, 'Bathroom'),
    ({4},       1, 'Dining'),
    ({5},       2, 'Child'),
    ({6},       1, 'Study'),
    ({7},       2, 'Second'),
    ({9},       2, 'Balcony'),
    ({11},      2, 'Storage'),
]
ROOM_DIM = len(ROOM_TYPES) + 1   # per-type counts + total room count = 11


def room_count_desc(data):
    """
    11-dim room-count descriptor: [count_0/max_0, …, count_9/max_9, total/15]

    Handles both:
    - old format: data['labels'] as a flat list of integer room types
    - v2 format:  data['rooms'] as a list of dicts with 'type' or 'source_type'

    Each element is clamped to [0, 1].  Rotation/translation invariant.
    """
    # v2 format
    if data.get('rooms'):
        labels = [r.get('source_type', r.get('type', -1)) for r in data['rooms']]
    else:
        labels = data.get('labels', [])

    feat = []
    for lbl_set, max_cnt, _ in ROOM_TYPES:
        cnt = sum(1 for l in labels if l in lbl_set)
        feat.append(min(cnt / max_cnt, 1.0))
    total = sum(1 for l in labels if l <= 11)
    feat.append(min(total / 15.0, 1.0))
    return np.array(feat, dtype=np.float32)


def room_summary(data):
    labels = data.get('labels', [])
    parts = []
    for lbl_set, _, name in ROOM_TYPES:
        cnt = sum(1 for l in labels if l in lbl_set)
        if cnt:
            parts.append(f'{name[:3]}:{cnt}')
    return '  '.join(parts)


ENT_DIM = 4   # [rel_x, rel_y, cos_θ, sin_θ]


# ── database ──────────────────────────────────────────────────────────────────

def build_db(data_dir, name_json=None):
    """Return (names, shape_descs, ent_descs, room_descs, raws)."""
    if name_json:
        with open(name_json) as f: meta = json.load(f)
        ordered = [meta[k] for k in sorted(meta, key=lambda x: int(x))]
    else:
        ordered = [f.replace('.json', '.png')
                   for f in sorted(os.listdir(data_dir)) if f.endswith('.json')]

    names, shape_descs, ent_descs, room_descs, raws = [], [], [], [], []
    skipped = 0
    for png_name in ordered:
        stem = os.path.splitext(png_name)[0]
        path = os.path.join(data_dir, stem + '.json')
        if not os.path.exists(path):
            skipped += 1; continue
        with open(path) as f: data = json.load(f)
        if not data.get('boundary') or frontdoor_center(data) is None:
            skipped += 1; continue
        names.append(png_name)
        shape_descs.append(shape_desc(data['boundary']))
        ent_descs.append(entrance_desc(data))
        room_descs.append(room_count_desc(data))
        raws.append(data)

    if skipped:
        print(f"  Skipped {skipped} (missing / no boundary / no front door)")
    return names, np.stack(shape_descs), np.stack(ent_descs), np.stack(room_descs), raws


# ── rotation-aware retrieval ──────────────────────────────────────────────────

def ent_dist_min_rotation(q_ent, db_ents):
    """
    Minimum entrance distance over all continuous rotations α ∈ [0, 2π).

    The entrance descriptor is [rel_x, rel_y, cos_θ, sin_θ].
    Rotation by α acts as a 2-D rotation on (rel_x, rel_y) and on (cos_θ, sin_θ):
        R(α) · [rx, ry, cx, cy] = [rx·cosα - ry·sinα,
                                    rx·sinα + ry·cosα,
                                    cx·cosα - cy·sinα,
                                    cx·sinα + cy·cosα]

    Expanding ||q - R(α)·db||²:
        = ||q||² + ||db||² - 2(A·cosα + B·sinα)
    where
        A = q_rx·db_rx + q_ry·db_ry + q_cx·db_cx + q_cy·db_cy   (sum of dot products)
        B = q_ry·db_rx - q_rx·db_ry + q_cy·db_cx - q_cx·db_cy   (sum of cross products)

    Minimum over α is achieved at α* = atan2(B, A), giving:
        min_α ||q - R(α)·db||² = ||q||² + ||db||² - 2·√(A² + B²)

    db_ents : (N, 4) array — all database entrance descriptors (unscaled)
    q_ent   : (4,)   array — query entrance descriptor (unscaled)
    Returns : (N,) distances (unscaled; caller applies ent_w after)
    """
    q_rx, q_ry = q_ent[0], q_ent[1]
    q_cx, q_cy = q_ent[2], q_ent[3]

    d_rx = db_ents[:, 0];  d_ry = db_ents[:, 1]
    d_cx = db_ents[:, 2];  d_cy = db_ents[:, 3]

    A = q_rx*d_rx + q_ry*d_ry + q_cx*d_cx + q_cy*d_cy
    B = q_ry*d_rx - q_rx*d_ry + q_cy*d_cx - q_cx*d_cy

    q_sq = q_rx**2 + q_ry**2 + q_cx**2 + q_cy**2
    d_sq = d_rx**2 + d_ry**2 + d_cx**2 + d_cy**2

    min_sq = q_sq + d_sq - 2.0 * np.sqrt(A**2 + B**2)
    return np.sqrt(np.maximum(min_sq, 0.0))


def _snap_to_half_pi(angles):
    """Snap each angle (radians) to the nearest multiple of π/2."""
    return np.round(angles / (np.pi / 2)) * (np.pi / 2)


def retrieve(q_shape, q_ent, q_room,
             db_shapes, db_ents, db_rooms,
             shape_w, ent_w, room_w, topn):
    """
    Combined rotation-aware distance considering 8 orientations
    (4 rotations × 2 mirror states).

    Mirroring flips x components of the entrance descriptor:
      mirror([rx, ry, nx, ny]) = [-rx, ry, -nx, ny]

    Returns best_angles and best_mirror (bool array) snapped to π/2 multiples.
    """
    shape_dists = np.linalg.norm(db_shapes - q_shape[None], axis=1) * shape_w
    room_dists  = np.linalg.norm(db_rooms  - q_room[None],  axis=1) * room_w

    q_rx, q_ry = q_ent[0], q_ent[1]
    q_cx, q_cy = q_ent[2], q_ent[3]

    # Normal orientation
    A  = q_rx*db_ents[:,0] + q_ry*db_ents[:,1] + q_cx*db_ents[:,2] + q_cy*db_ents[:,3]
    B  = q_ry*db_ents[:,0] - q_rx*db_ents[:,1] + q_cy*db_ents[:,2] - q_cx*db_ents[:,3]
    # Mirrored orientation: flip x components → [-d_rx, d_ry, -d_cx, d_cy]
    Am = -q_rx*db_ents[:,0] + q_ry*db_ents[:,1] - q_cx*db_ents[:,2] + q_cy*db_ents[:,3]
    Bm = -q_ry*db_ents[:,0] - q_rx*db_ents[:,1] - q_cy*db_ents[:,2] - q_cx*db_ents[:,3]

    q_sq = q_rx**2 + q_ry**2 + q_cx**2 + q_cy**2
    d_sq = (db_ents**2).sum(axis=1)

    # Minimum continuous-rotation entrance distance for each orientation
    ed_normal = np.sqrt(np.maximum(q_sq + d_sq - 2.0*np.sqrt(A**2  + B**2),  0.0))
    ed_mirror = np.sqrt(np.maximum(q_sq + d_sq - 2.0*np.sqrt(Am**2 + Bm**2), 0.0))

    use_mirror  = ed_mirror < ed_normal          # shape (N,) bool
    raw_angles  = np.where(use_mirror, np.arctan2(Bm, Am), np.arctan2(B, A))
    best_angles = _snap_to_half_pi(raw_angles)

    # Recompute entrance distance at the snapped angle (for the chosen orientation)
    cos_a = np.cos(best_angles);  sin_a = np.sin(best_angles)

    # Effective db entrance after mirror (if needed)
    eff_rx = np.where(use_mirror, -db_ents[:,0], db_ents[:,0])
    eff_ry = db_ents[:,1]
    eff_cx = np.where(use_mirror, -db_ents[:,2], db_ents[:,2])
    eff_cy = db_ents[:,3]

    rot_rx = eff_rx*cos_a - eff_ry*sin_a
    rot_ry = eff_rx*sin_a + eff_ry*cos_a
    rot_cx = eff_cx*cos_a - eff_cy*sin_a
    rot_cy = eff_cx*sin_a + eff_cy*cos_a
    d_ent  = np.stack([rot_rx - q_rx, rot_ry - q_ry,
                       rot_cx - q_cx, rot_cy - q_cy], axis=1)
    ent_dists = np.linalg.norm(d_ent, axis=1) * ent_w

    combined    = shape_dists + ent_dists + room_dists
    idx         = np.argsort(combined)[:topn]
    return idx, combined[idx], shape_dists[idx], ent_dists[idx], room_dists[idx], best_angles[idx], use_mirror[idx]


# ── visualisation ─────────────────────────────────────────────────────────────

def _draw_door(ax, data, color='red', arrow_len=8.0):
    """Red dot at front door + inward arrow along wall normal."""
    dc = frontdoor_center(data)
    if dc is None:
        return
    ed = entrance_desc(data)          # [rel_x, rel_y, cos_θ, sin_θ]
    nx, ny = float(ed[2]), float(ed[3])
    # inward direction = -normal
    ax.annotate('', xy=(dc[0] - nx*arrow_len, dc[1] - ny*arrow_len),
                xytext=(dc[0], dc[1]),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    ax.plot(dc[0], dc[1], 'o', ms=5, color=color, zorder=6)


def draw_plan(ax, data, title='', dist=None,
              shape_d=None, ent_d=None, room_d=None):
    coords   = data.get('coord', [])
    labels   = data.get('labels', [])
    boundary = data.get('boundary', [])

    ax.set_aspect('equal'); ax.axis('off')

    patches, colors = [], []
    for segs, lbl in zip(coords, labels):
        verts = segs_to_verts(segs)
        if len(verts) < 3: continue
        patches.append(MplPolygon(verts, closed=True))
        colors.append(ROOM_COLORS[lbl % len(ROOM_COLORS)])
    if patches:
        ax.add_collection(PatchCollection(patches, facecolors=colors,
                                          edgecolors='none', alpha=0.65, zorder=1))

    if boundary:
        bp = boundary_poly_arr(boundary)
        ax.plot(bp[:, 0], bp[:, 1], color='#111111', lw=1.8, zorder=3)
        xs, ys = bp[:, 0], bp[:, 1]
        pad = max(xs.ptp(), ys.ptp()) * 0.08 + 3
        ax.set_xlim(xs.min()-pad, xs.max()+pad)
        ax.set_ylim(ys.min()-pad, ys.max()+pad)

    _draw_door(ax, data)

    lines = [title]
    if dist is not None:
        lines.append(f'dist={dist:.4f}')
        parts = []
        if shape_d is not None: parts.append(f'shp={shape_d:.3f}')
        if ent_d   is not None: parts.append(f'ent={ent_d:.3f}')
        if room_d  is not None: parts.append(f'rm={room_d:.3f}')
        if parts: lines.append('  '.join(parts))
    ax.set_title('\n'.join(lines), fontsize=6.5, pad=2)
    ax.text(0.5, -0.02, room_summary(data), transform=ax.transAxes,
            ha='center', va='top', fontsize=5, color='#333')


def visualize(q_data, res_data, res_names, res_dists,
              shape_dists, ent_dists, room_dists, topn, out=None):
    ncols = topn + 1
    fig, axes = plt.subplots(1, ncols, figsize=(3.2*ncols, 5.0),
                             gridspec_kw={'wspace': 0.05})
    if ncols == 1: axes = [axes]

    draw_plan(axes[0], q_data, title=f"Query  {q_data.get('name','')}")
    for i in range(topn):
        draw_plan(axes[i+1], res_data[i],
                  title=f'#{i+1}  {res_names[i]}',
                  dist=res_dists[i], shape_d=shape_dists[i],
                  ent_d=ent_dists[i], room_d=room_dists[i])

    handles = [mpatches.Patch(facecolor=ROOM_COLORS[t], edgecolor='#999',
                               linewidth=0.4, label=f'{t}: {SEMANTICS[t]}')
               for t in SEMANTICS if t < len(ROOM_COLORS)]
    fig.legend(handles=handles, loc='lower center', ncol=9,
               fontsize=5.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, 1.01,
             'Shape + entrance (rotation-aware) + room count retrieval  '
             '(red dot = door, arrow = inward normal)',
             ha='center', fontsize=9)
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved → {out}")
    else:
        plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Shape + entrance (rotation-aware) + room-count retrieval.')
    parser.add_argument('--query',     required=True)
    parser.add_argument('--data_dir',  required=True)
    parser.add_argument('--name_json', default=None)
    parser.add_argument('--topn',      type=int,   default=5)
    parser.add_argument('--shape_w',   type=float, default=1.0,
                        help='Boundary shape weight (default 1.0)')
    parser.add_argument('--ent_w',     type=float, default=2.0,
                        help='Entrance position weight (default 2.0)')
    parser.add_argument('--room_w',    type=float, default=1.0,
                        help='Room-count weight (default 1.0)')
    parser.add_argument('--out',       default=None)
    args = parser.parse_args()

    with open(args.query) as f: q_data = json.load(f)
    if not q_data.get('boundary'):
        raise ValueError(f"Query has no boundary: {args.query}")

    q_shape = shape_desc(q_data['boundary'])
    q_ent   = entrance_desc(q_data)
    q_room  = room_count_desc(q_data)

    dc = frontdoor_center(q_data)
    print(f"Query : {q_data.get('name', args.query)}")
    print(f"  desc  shape={N_FREQS}-dim  entrance={ENT_DIM}-dim  rooms={ROOM_DIM}-dim")
    print(f"  weights  shape_w={args.shape_w}  ent_w={args.ent_w}  room_w={args.room_w}")
    print(f"  front door: center={dc}  rel=({q_ent[0]:.3f},{q_ent[1]:.3f})  "
          f"normal=({q_ent[2]:.3f},{q_ent[3]:.3f})")
    print(f"  rooms: {room_summary(q_data)}")

    print(f"\nBuilding database from: {args.data_dir}")
    db_names, db_shapes, db_ents, db_rooms, db_raws = build_db(
        args.data_dir, args.name_json)
    print(f"  {len(db_names)} plans indexed")

    idx, dists, shape_d, ent_d, room_d = retrieve(
        q_shape, q_ent, q_room,
        db_shapes, db_ents, db_rooms,
        args.shape_w, args.ent_w, args.room_w, args.topn)

    print(f"\nTop-{args.topn} results:")
    print(f"  {'name':25s}  {'combined':>8}  {'shape':>7}  {'ent(rot)':>8}  {'rooms':>7}  rooms")
    for rank, (i, d, sd, ed_, rd) in enumerate(
            zip(idx, dists, shape_d, ent_d, room_d), 1):
        e = db_ents[i]; q = q_ent
        A = q[0]*e[0]+q[1]*e[1]+q[2]*e[2]+q[3]*e[3]
        B = q[1]*e[0]-q[0]*e[1]+q[3]*e[2]-q[2]*e[3]
        best_deg = np.degrees(np.arctan2(B, A)) % 360
        print(f"  #{rank:2d} {db_names[i]:25s}  {d:8.4f}  {sd:7.4f}  "
              f"{ed_:8.4f}  {rd:7.4f}  α={best_deg:.0f}°  [{room_summary(db_raws[i])}]")

    visualize(q_data,
              [db_raws[i] for i in idx],
              [db_names[i] for i in idx],
              dists, shape_d, ent_d, room_d,
              args.topn, out=args.out)


if __name__ == '__main__':
    main()
