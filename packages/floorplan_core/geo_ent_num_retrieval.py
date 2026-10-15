"""
Boundary-shape + front-door position + room-count retrieval for floor plans.

Descriptor parts
────────────────
shape  (48-dim, ×shape_w) — L2-normalised Fourier on centroid-distance signal
entrance (5-dim, ×ent_w)  — front door orientation relative to boundary (rotation-aware)
  ├ cos_φ, sin_φ : unit direction from boundary centroid to door snap point
  ├ cos_θ, sin_θ : outward wall normal at the door edge
  └ r/mean_r     : normalised radial distance (rotation-invariant secondary signal)
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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

# ── constants ─────────────────────────────────────────────────────────────────
N_SAMP = 256
N_FREQS = 48
DENSE_SAMP = 2048  # perimeter pre-resampling density for shape_desc_complex (see 2026-08-11 fix)

SEMANTICS = {
    0: "Living",
    1: "Master",
    2: "Kitchen",
    3: "Bathroom",
    4: "Dining",
    5: "Child",
    6: "Study",
    7: "Second",
    8: "Guest",
    9: "Balcony",
    10: "Entrance",
    11: "Storage",
    12: "Wall-in",
    13: "External",
    14: "Ext.wall",
    15: "FrontDoor",
    16: "Int.wall",
    17: "Int.door",
}
ROOM_COLORS = [
    "#E6B8A2",
    "#B8D4E8",
    "#B8E8B8",
    "#E8E8B8",
    "#D4B8E8",
    "#E8B8D4",
    "#B8E8E8",
    "#E8D4B8",
    "#C8E8C8",
    "#B8C8E8",
    "#E8C8B8",
    "#D8D8D8",
    "#AAAAAA",
    "#CCCCCC",
    "#888888",
    "#FFAAAA",
    "#BBBBBB",
    "#DDDDDD",
]


# ── geometry helpers ──────────────────────────────────────────────────────────


def segs_to_verts(segs):
    return [(float(s[0]), float(s[1])) for s in segs]


def normalise_winding(verts):
    """Return verts in CW order (negative signed area). Reverses if CCW."""
    n = len(verts)
    area = sum(
        verts[i][0] * verts[(i + 1) % n][1] - verts[(i + 1) % n][0] * verts[i][1]
        for i in range(n)
    )
    return verts[::-1] if area > 0 else verts


def boundary_poly_arr(bdry):
    v = segs_to_verts(bdry)
    v.append(v[0])
    return np.array(v)


def sample_perimeter(verts, n):
    pts = np.array(verts + [verts[0]], dtype=float)
    diffs = np.diff(pts, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 1e-9:
        return np.zeros((n, 2))
    positions = np.linspace(0, total, n, endpoint=False)
    result = np.empty((n, 2))
    for i, pos in enumerate(positions):
        idx = min(int(np.searchsorted(cum, pos, side="right")) - 1, len(seg_len) - 1)
        t = (pos - cum[idx]) / (seg_len[idx] + 1e-12)
        result[i] = pts[idx] + t * diffs[idx]
    return result


def perimeter_cumlen(verts):
    """Cumulative arc-lengths and total length for a closed polygon."""
    pts = np.array(verts + [verts[0]], dtype=float)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    return np.concatenate([[0.0], np.cumsum(seg)]), seg.sum()


def polygon_centroid(verts):
    """Proper area-weighted polygon centroid (shoelace formula).

    Fixed 2026-08-11: a naive vertex-average (`pts.mean(axis=0)`) is biased by
    however many vertices happen to be listed on each edge. Every RPlan
    corpus entry's stored boundary carries two extra vertices marking the
    door position along whichever wall it's on — collinear, so they don't
    change the polygon's shape, but they do drag a naive vertex-average
    centroid toward the door edge. A hand-drawn query boundary carries no
    such extra vertices, so it doesn't get that pull — the naive centroid is
    systematically different between query and database for the identical
    shape. Confirmed: on a real query/match pair, this alone inflated shape
    distance from ~0.01 (correct) to ~0.74. The area-weighted centroid below
    is exactly invariant to extra collinear vertices, fixing this for both
    sides symmetrically.
    """
    n = len(verts)
    a_sum = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a_sum += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area = a_sum / 2.0
    if abs(area) < 1e-12:
        arr = np.array(verts, dtype=float)
        return arr.mean(axis=0)
    return np.array([cx / (6.0 * area), cy / (6.0 * area)], dtype=float)


# ── shape descriptor ──────────────────────────────────────────────────────────


def shape_desc(bdry_segs, n_samp=N_SAMP, n_freqs=N_FREQS):
    """Centroid-distance Fourier, L2-normalised. (n_freqs,) float32."""
    verts = normalise_winding(segs_to_verts(bdry_segs))
    if len(verts) < 3:
        return np.zeros(n_freqs, dtype=np.float32)
    pts = sample_perimeter(verts, n_samp)
    centroid = polygon_centroid(verts)
    dists = np.linalg.norm(pts - centroid, axis=1)
    mean_d = dists.mean()
    if mean_d < 1e-9:
        return np.zeros(n_freqs, dtype=np.float32)
    dists /= mean_d
    mag = np.abs(np.fft.rfft(dists))[1 : n_freqs + 1]
    if len(mag) < n_freqs:
        mag = np.pad(mag, (0, n_freqs - len(mag)))
    norm = np.linalg.norm(mag)
    if norm > 1e-9:
        mag /= norm
    return mag.astype(np.float32)


def shape_desc_complex(bdry_segs, n_samp=N_SAMP, n_freqs=N_FREQS):
    """
    Angle-from-centroid radial Fourier, complex coefficients. (n_freqs,) complex64.

    Samples boundary radius at n_samp uniform angular intervals from the centroid.
    Stores complex FFT coefficients k=1..n_freqs (DC skipped).

    Phase encodes orientation: rotating the shape by α multiplies F[k] by e^{-ikα}.
    This allows rotation-aware shape comparison in retrieve() via phase shift,
    rather than the rotation-invariant magnitude-only approach of shape_desc().

    Fixed 2026-08-11: previously sampled angle/radius directly from the raw
    vertex list. RPlan corpus boundaries carry two extra collinear vertices
    marking the door position along whichever wall it's on — harmless to the
    polygon's shape, but they used to inject spurious points into the
    angle-sorted interpolation (on top of biasing the centroid, fixed
    separately above), distorting the high-frequency coefficients. Now
    resamples the perimeter densely and uniformly by arc length first (same
    approach shape_desc() already used), so the result is invariant to
    however many vertices the input happens to list.
    """
    verts = normalise_winding(segs_to_verts(bdry_segs))
    if len(verts) < 3:
        return np.zeros(n_freqs, dtype=np.complex64)

    centroid = polygon_centroid(verts)
    dense = sample_perimeter(verts, DENSE_SAMP)
    dx = dense[:, 0] - centroid[0]
    dy = dense[:, 1] - centroid[1]
    angles = np.arctan2(dy, dx)   # in [-π, π]
    radii = np.hypot(dx, dy)

    # Sort by angle and extend by 2π on both sides for wrap-around interp
    sort_idx = np.argsort(angles)
    sorted_angles = angles[sort_idx]
    sorted_radii = radii[sort_idx]
    ext_angles = np.concatenate([sorted_angles - 2 * np.pi, sorted_angles, sorted_angles + 2 * np.pi])
    ext_radii = np.concatenate([sorted_radii, sorted_radii, sorted_radii])

    target = np.linspace(-np.pi, np.pi, n_samp, endpoint=False)
    sampled = np.interp(target, ext_angles, ext_radii)

    mean_r = sampled.mean()
    if mean_r < 1e-9:
        return np.zeros(n_freqs, dtype=np.complex64)
    sampled /= mean_r

    # Full complex FFT — keeps phase information
    fft_full = np.fft.fft(sampled)
    coeffs = fft_full[1 : n_freqs + 1].astype(np.complex64)

    # L2-normalise in complex space
    norm = np.linalg.norm(coeffs)
    if norm > 1e-9:
        coeffs /= norm

    return coeffs


# ── entrance descriptor ───────────────────────────────────────────────────────


def frontdoor_center(data):
    """Return (cx, cy) of the front door from door_rects, entrance bbox, or None.

    Handles both the old (door_rects) and v2 (entrance bbox) data formats.
    """
    # v2 format: entrance is a [x1, y1, x2, y2] bounding box
    ent = data.get("entrance")
    if ent and len(ent) == 4:
        return ((ent[0] + ent[2]) / 2.0, (ent[1] + ent[3]) / 2.0)

    # old format: door_rects list
    for r in data.get("door_rects", []):
        if r.get("label") == 15:
            pts = r["rect"]
            return (
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts),
            )
    # fallback: FrontDoor room centroid (old coord/labels format)
    for segs, lbl in zip(data.get("coord", []), data.get("labels", [])):
        if lbl == 15 and segs:
            xs = [float(s[0]) for s in segs] + [float(s[2]) for s in segs]
            ys = [float(s[1]) for s in segs] + [float(s[3]) for s in segs]
            return sum(xs) / len(xs), sum(ys) / len(ys)
    return None


def entrance_desc(data):
    """
    5-dim entrance descriptor: [cos_φ, sin_φ, cos_θ, sin_θ, r/mean_r]

      cos_φ, sin_φ — unit direction from boundary centroid to the door's
                     boundary snap point.  Encodes which side of the building
                     the door is on without penalising for absolute distance.

      cos_θ, sin_θ — outward wall normal at that edge.

      r/mean_r     — normalised radial distance from centroid to door snap.
                     Rotation-invariant secondary signal: penalises plans where
                     the door is much closer to or further from the centroid
                     than in the query.

    Returns zeros if no front door or boundary is found.
    """
    dc = frontdoor_center(data)
    if dc is None:
        return np.zeros(5, dtype=np.float32)
    dx, dy = float(dc[0]), float(dc[1])

    bdry = data.get("boundary", [])
    if not bdry:
        return np.zeros(5, dtype=np.float32)

    verts = normalise_winding(segs_to_verts(bdry))
    n = len(verts)

    pts = sample_perimeter(verts, N_SAMP)
    centroid = polygon_centroid(verts)
    mean_r = np.linalg.norm(pts - centroid, axis=1).mean()
    if mean_r < 1e-9:
        return np.zeros(5, dtype=np.float32)

    # find closest boundary edge to the door
    best_dist = float("inf")
    best_px = dx
    best_py = dy
    best_edge = 0

    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        seg_l = np.hypot(ex, ey)
        if seg_l < 1e-9:
            continue
        t = max(0.0, min(1.0, ((dx - ax) * ex + (dy - ay) * ey) / seg_l**2))
        px, py = ax + t * ex, ay + t * ey
        d = np.hypot(dx - px, dy - py)
        if d < best_dist:
            best_dist = d
            best_px, best_py = px, py
            best_edge = i

    # unit direction + normalised radius from centroid to door snap
    ddx, ddy = best_px - centroid[0], best_py - centroid[1]
    r = np.hypot(ddx, ddy)
    if r < 1e-9:
        return np.zeros(5, dtype=np.float32)
    cphi, sphi = ddx / r, ddy / r
    r_norm = r / mean_r

    # outward normal of the closest edge
    ax, ay = verts[best_edge]
    bx, by = verts[(best_edge + 1) % n]
    ex, ey = bx - ax, by - ay
    seg_l = np.hypot(ex, ey)
    nx, ny = ey / seg_l, -ex / seg_l
    if nx * (best_px - centroid[0]) + ny * (best_py - centroid[1]) < 0:
        nx, ny = -nx, -ny

    return np.array([cphi, sphi, nx, ny, r_norm], dtype=np.float32)


def entrance_desc_from_point(door_xy, bdry_segs):
    """
    Compute a 5-dim entrance descriptor from an explicit door position.

    Returns np.ndarray shape (5,) float32 — [cos_φ, sin_φ, cos_θ, sin_θ, r/mean_r].
    """
    if door_xy is None or not bdry_segs:
        return np.zeros(5, dtype=np.float32)

    dx, dy = float(door_xy[0]), float(door_xy[1])
    verts = normalise_winding(segs_to_verts(bdry_segs))
    n = len(verts)
    if n < 2:
        return np.zeros(5, dtype=np.float32)

    pts = sample_perimeter(verts, N_SAMP)
    centroid = polygon_centroid(verts)
    mean_r = np.linalg.norm(pts - centroid, axis=1).mean()
    if mean_r < 1e-9:
        return np.zeros(5, dtype=np.float32)

    best_dist = float("inf")
    best_px = dx
    best_py = dy
    best_edge = 0

    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        seg_l = np.hypot(ex, ey)
        if seg_l < 1e-9:
            continue
        t = max(0.0, min(1.0, ((dx - ax) * ex + (dy - ay) * ey) / seg_l**2))
        px, py = ax + t * ex, ay + t * ey
        d = np.hypot(dx - px, dy - py)
        if d < best_dist:
            best_dist = d
            best_px, best_py = px, py
            best_edge = i

    ddx, ddy = best_px - centroid[0], best_py - centroid[1]
    r = np.hypot(ddx, ddy)
    if r < 1e-9:
        return np.zeros(5, dtype=np.float32)
    cphi, sphi = ddx / r, ddy / r
    r_norm = r / mean_r

    ax, ay = verts[best_edge]
    bx, by = verts[(best_edge + 1) % n]
    ex, ey = bx - ax, by - ay
    seg_l = np.hypot(ex, ey)
    nx, ny = ey / seg_l, -ex / seg_l
    if nx * (best_px - centroid[0]) + ny * (best_py - centroid[1]) < 0:
        nx, ny = -nx, -ny

    return np.array([cphi, sphi, nx, ny, r_norm], dtype=np.float32)


# ── room-count descriptor ─────────────────────────────────────────────────────

# (label_set, max_expected_count, display_name)
ROOM_TYPES = [
    ({0}, 2, "Living"),
    ({1}, 1, "Master"),
    ({2}, 1, "Kitchen"),
    ({3}, 3, "Bathroom"),
    ({4}, 1, "Dining"),
    ({5}, 2, "Child"),
    ({6}, 1, "Study"),
    ({7}, 2, "Second"),
    ({9}, 2, "Balcony"),
    ({11}, 2, "Storage"),
]
ROOM_DIM = len(ROOM_TYPES) + 1  # per-type counts + total room count = 11


def room_count_desc(data):
    """
    11-dim room-count descriptor: [count_0/max_0, …, count_9/max_9, total/15]

    Handles both:
    - old format: data['labels'] as a flat list of integer room types
    - v2 format:  data['rooms'] as a list of dicts with 'type' or 'source_type'

    Each element is clamped to [0, 1].  Rotation/translation invariant.
    """
    # v2 format
    if data.get("rooms"):
        labels = [r.get("source_type", r.get("type", -1)) for r in data["rooms"]]
    else:
        labels = data.get("labels", [])

    feat = []
    for lbl_set, max_cnt, _ in ROOM_TYPES:
        cnt = sum(1 for l in labels if l in lbl_set)
        feat.append(min(cnt / max_cnt, 1.0))
    total = sum(1 for l in labels if l <= 11)
    feat.append(min(total / 15.0, 1.0))
    return np.array(feat, dtype=np.float32)


def room_summary(data):
    if data.get("rooms"):
        labels = [r.get("source_type", r.get("type", -1)) for r in data["rooms"]]
    else:
        labels = data.get("labels", [])
    parts = []
    for lbl_set, _, name in ROOM_TYPES:
        cnt = sum(1 for l in labels if l in lbl_set)
        if cnt:
            parts.append(f"{name[:3]}:{cnt}")
    return "  ".join(parts)


ENT_DIM = 5  # [cos_φ, sin_φ, cos_θ, sin_θ, r/mean_r]
ENT_RADIAL_W = 0.3  # weight of the radial component relative to ent_w


# ── database ──────────────────────────────────────────────────────────────────


def build_db(data_dir, name_json=None):
    """Return (names, shape_descs, ent_descs, room_descs, raws)."""
    if name_json:
        with open(name_json) as f:
            meta = json.load(f)
        ordered = [meta[k] for k in sorted(meta, key=lambda x: int(x))]
    else:
        ordered = [
            f.replace(".json", ".png")
            for f in sorted(os.listdir(data_dir))
            if f.endswith(".json")
        ]

    names, shape_descs, ent_descs, room_descs, raws = [], [], [], [], []
    skipped = 0
    for png_name in ordered:
        stem = os.path.splitext(png_name)[0]
        path = os.path.join(data_dir, stem + ".json")
        if not os.path.exists(path):
            skipped += 1
            continue
        with open(path) as f:
            data = json.load(f)
        if not data.get("boundary") or frontdoor_center(data) is None:
            skipped += 1
            continue
        names.append(png_name)
        shape_descs.append(shape_desc(data["boundary"]))
        ent_descs.append(entrance_desc(data))
        room_descs.append(room_count_desc(data))
        raws.append(data)

    if skipped:
        print(f"  Skipped {skipped} (missing / no boundary / no front door)")
    return names, np.stack(shape_descs), np.stack(ent_descs), np.stack(room_descs), raws


# ── index build / load ───────────────────────────────────────────────────────


def _iter_descriptors(data_dir, name_json=None):
    """Yield (name, shape_desc, ent_desc, room_desc) for every valid plan in data_dir.

    Avoids loading all raw JSON into memory — suitable for large datasets.
    Prints progress every 5000 plans.
    """
    if name_json:
        with open(name_json) as f:
            meta = json.load(f)
        ordered = [meta[k] for k in sorted(meta, key=lambda x: int(x))]
    else:
        ordered = [
            f.replace(".json", ".png")
            for f in sorted(os.listdir(data_dir))
            if f.endswith(".json")
        ]

    skipped = 0
    for i, png_name in enumerate(ordered):
        stem = os.path.splitext(png_name)[0]
        path = os.path.join(data_dir, stem + ".json")
        if not os.path.exists(path):
            skipped += 1
            continue
        with open(path) as f:
            data = json.load(f)
        if not data.get("boundary") or frontdoor_center(data) is None:
            skipped += 1
            continue
        yield (
            png_name,
            shape_desc_complex(data["boundary"]),
            entrance_desc(data),
            room_count_desc(data),
        )
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(ordered)} processed  ({skipped} skipped so far)")

    if skipped:
        print(f"  Skipped {skipped} (missing / no boundary / no front door)")


def build_index(data_dir, out_path, name_json=None):
    """Compute descriptors for all plans in data_dir and save to out_path (.npz).

    The index contains:
      names        — 1-D str array of filenames (e.g. '0.png')
      shape_descs  — (N, 48) complex64  angle-from-centroid Fourier, complex coefficients
      ent_descs    — (N, 4)  float32
      room_descs   — (N, 11) float32
    """
    print(f"Building index from: {data_dir}")
    names, shapes, ents, rooms = [], [], [], []
    for name, sd, ed, rd in _iter_descriptors(data_dir, name_json):
        names.append(name)
        shapes.append(sd)
        ents.append(ed)
        rooms.append(rd)

    np.savez(
        out_path,
        names=np.array(names),
        shape_descs=np.stack(shapes),
        ent_descs=np.stack(ents),
        room_descs=np.stack(rooms),
    )
    print(f"Index saved → {out_path}  ({len(names)} plans indexed)")


def load_index(index_path):
    """Load a pre-built descriptor index from a .npz file.

    Returns (names, shape_descs, ent_descs, room_descs).
    names is a plain Python list of filename strings.
    """
    data = np.load(index_path, allow_pickle=True)
    names = list(data["names"])
    print(f"Index loaded: {len(names)} plans from {index_path}")
    return names, data["shape_descs"], data["ent_descs"], data["room_descs"]


# ── rotation-aware retrieval ──────────────────────────────────────────────────


def ent_dist_min_rotation(q_ent, db_ents):
    """
    Minimum entrance distance over all continuous rotations α ∈ [0, 2π).

    Descriptor is [cos_φ, sin_φ, cos_θ, sin_θ, r/mean_r].
    The first four components rotate together under α; the fifth is rotation-invariant.
    Returns the 4D rotation-optimal distance only — radial term added separately in retrieve().

    db_ents : (N, 5) array
    q_ent   : (5,)   array
    Returns : (N,) distances (caller applies ent_w after)
    """
    q_px, q_py = q_ent[0], q_ent[1]
    q_cx, q_cy = q_ent[2], q_ent[3]
    d_px = db_ents[:, 0]; d_py = db_ents[:, 1]
    d_cx = db_ents[:, 2]; d_cy = db_ents[:, 3]

    A = q_px * d_px + q_py * d_py + q_cx * d_cx + q_cy * d_cy
    B = q_py * d_px - q_px * d_py + q_cy * d_cx - q_cx * d_cy

    q_sq = q_px**2 + q_py**2 + q_cx**2 + q_cy**2
    d_sq = d_px**2 + d_py**2 + d_cx**2 + d_cy**2

    min_sq = q_sq + d_sq - 2.0 * np.sqrt(A**2 + B**2)
    return np.sqrt(np.maximum(min_sq, 0.0))


def _snap_to_half_pi(angles):
    """Snap each angle (radians) to the nearest multiple of π/2."""
    return np.round(angles / (np.pi / 2)) * (np.pi / 2)


def retrieve(
    q_shape, q_ent, q_room, db_shapes, db_ents, db_rooms, shape_w, ent_w, room_w, topn
):
    """
    Rotation-aware retrieval with Option B shape matching.

    Shape distance is computed AFTER entrance alignment: the db descriptor is
    phase-shifted by the retrieval rotation α before comparing with the query,
    so both terms are evaluated in the same rotated frame.

      F_db_rotated[k] = F_db[k] · e^{-ikα}   (rotation by α)
      F_db_mirrored[k] = conj(F_db[k]) · e^{-ikα}  (mirror then rotate)

    Mirror approximation: conj(F[k]) corresponds to r'(θ) = r(−θ), which is
    exact for x-axis reflection. For mostly-rectangular plans the distinction
    from y-axis reflection is negligible.
    """
    room_dists = np.linalg.norm(db_rooms - q_room[None], axis=1) * room_w

    q_px, q_py = q_ent[0], q_ent[1]   # unit direction from centroid to door
    q_cx, q_cy = q_ent[2], q_ent[3]   # outward wall normal

    # Normal orientation
    A = (
        q_px * db_ents[:, 0] + q_py * db_ents[:, 1]
        + q_cx * db_ents[:, 2] + q_cy * db_ents[:, 3]
    )
    B = (
        q_py * db_ents[:, 0] - q_px * db_ents[:, 1]
        + q_cy * db_ents[:, 2] - q_cx * db_ents[:, 3]
    )
    # Mirrored orientation: flip x components → [-d_px, d_py, -d_cx, d_cy]
    Am = (
        -q_px * db_ents[:, 0] + q_py * db_ents[:, 1]
        - q_cx * db_ents[:, 2] + q_cy * db_ents[:, 3]
    )
    Bm = (
        -q_py * db_ents[:, 0] - q_px * db_ents[:, 1]
        - q_cy * db_ents[:, 2] - q_cx * db_ents[:, 3]
    )

    q_sq = q_px**2 + q_py**2 + q_cx**2 + q_cy**2
    d_sq = (db_ents**2).sum(axis=1)

    ed_normal = np.sqrt(np.maximum(q_sq + d_sq - 2.0 * np.sqrt(A**2 + B**2), 0.0))
    ed_mirror = np.sqrt(np.maximum(q_sq + d_sq - 2.0 * np.sqrt(Am**2 + Bm**2), 0.0))

    use_mirror = ed_mirror < ed_normal
    raw_angles = np.where(use_mirror, np.arctan2(Bm, Am), np.arctan2(B, A))
    best_angles = _snap_to_half_pi(raw_angles)   # display only
    cont_angles = raw_angles                      # used for distance computation

    # Entrance distance at the continuous optimal angle
    cos_a = np.cos(cont_angles)
    sin_a = np.sin(cont_angles)
    eff_px = np.where(use_mirror, -db_ents[:, 0], db_ents[:, 0])
    eff_py = db_ents[:, 1]
    eff_cx = np.where(use_mirror, -db_ents[:, 2], db_ents[:, 2])
    eff_cy = db_ents[:, 3]
    rot_px = eff_px * cos_a - eff_py * sin_a
    rot_py = eff_px * sin_a + eff_py * cos_a
    rot_cx = eff_cx * cos_a - eff_cy * sin_a
    rot_cy = eff_cx * sin_a + eff_cy * cos_a
    d_ent = np.stack(
        [rot_px - q_px, rot_py - q_py, rot_cx - q_cx, rot_cy - q_cy], axis=1
    )
    radial_penalty = np.abs(db_ents[:, 4] - q_ent[4]) * ENT_RADIAL_W
    ent_dists = (np.linalg.norm(d_ent, axis=1) + radial_penalty) * ent_w

    # Option B: rotation-aware shape distance
    # k_vec[i] = i+1 (matches the coefficients stored at index 0..N_FREQS-1 → k=1..N_FREQS)
    k_vec = np.arange(1, db_shapes.shape[1] + 1, dtype=float)
    phase = np.exp(-1j * k_vec[None, :] * cont_angles[:, None])   # (N, N_FREQS)
    db_rot = db_shapes * phase
    db_mirror_rot = np.conj(db_shapes) * phase
    db_eff = np.where(use_mirror[:, None], db_mirror_rot, db_rot)  # (N, N_FREQS) complex
    shape_dists = np.linalg.norm(q_shape[None, :] - db_eff, axis=1).real * shape_w

    combined = shape_dists + ent_dists + room_dists
    idx = np.argsort(combined)[:topn]
    return (
        idx,
        combined[idx],
        shape_dists[idx],
        ent_dists[idx],
        room_dists[idx],
        best_angles[idx],
        raw_angles[idx],
        use_mirror[idx],
    )


# ── visualisation ─────────────────────────────────────────────────────────────


def _draw_door(ax, data, color="red", arrow_len=8.0):
    """Red dot at front door + inward arrow along wall normal."""
    dc = frontdoor_center(data)
    if dc is None:
        return
    ed = entrance_desc(data)  # [cos_φ, sin_φ, cos_θ, sin_θ]
    nx, ny = float(ed[2]), float(ed[3])
    # inward direction = -normal
    ax.annotate(
        "",
        xy=(dc[0] - nx * arrow_len, dc[1] - ny * arrow_len),
        xytext=(dc[0], dc[1]),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
    )
    ax.plot(dc[0], dc[1], "o", ms=5, color=color, zorder=6)


def draw_plan(ax, data, title="", dist=None, shape_d=None, ent_d=None, room_d=None, angle=None, mirror=False):
    coords = data.get("coord", [])
    labels = data.get("labels", [])
    boundary = data.get("boundary", [])

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    patches, colors = [], []
    for segs, lbl in zip(coords, labels):
        verts = segs_to_verts(segs)
        if len(verts) < 3:
            continue
        patches.append(MplPolygon(verts, closed=True))
        colors.append(ROOM_COLORS[lbl % len(ROOM_COLORS)])
    if patches:
        ax.add_collection(
            PatchCollection(
                patches, facecolors=colors, edgecolors="none", alpha=0.65, zorder=1
            )
        )

    if boundary:
        bp = boundary_poly_arr(boundary)
        ax.plot(bp[:, 0], bp[:, 1], color="#111111", lw=1.8, zorder=3)
        xs, ys = bp[:, 0], bp[:, 1]
        pad = max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.08 + 3
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_ylim(ys.min() - pad, ys.max() + pad)

    _draw_door(ax, data)

    lines = [title]
    if angle is not None:
        orient_parts = [f"α={np.degrees(angle):.0f}°"]
        if mirror:
            orient_parts.append("↔ mirror")
        lines.append("  ".join(orient_parts))
    if dist is not None:
        lines.append(f"dist={dist:.4f}")
        parts = []
        if shape_d is not None:
            parts.append(f"shp={shape_d:.3f}")
        if ent_d is not None:
            parts.append(f"ent={ent_d:.3f}")
        if room_d is not None:
            parts.append(f"rm={room_d:.3f}")
        if parts:
            lines.append("  ".join(parts))
    ax.set_title("\n".join(lines), fontsize=9, pad=3)
    ax.text(
        0.5,
        -0.02,
        room_summary(data),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color="#333",
    )


def visualize(
    q_data,
    res_data,
    res_names,
    res_dists,
    shape_dists,
    ent_dists,
    room_dists,
    topn,
    out=None,
    angles=None,
    mirrors=None,
):
    ncols = topn + 1
    fig, axes = plt.subplots(
        1, ncols, figsize=(5.0 * ncols, 8.0), gridspec_kw={"wspace": 0.08}
    )
    if ncols == 1:
        axes = [axes]

    draw_plan(axes[0], q_data, title=f"Query  {q_data.get('name', '')}")
    for i in range(topn):
        draw_plan(
            axes[i + 1],
            res_data[i],
            title=f"#{i + 1}  {res_names[i]}",
            dist=res_dists[i],
            shape_d=shape_dists[i],
            ent_d=ent_dists[i],
            room_d=room_dists[i],
            angle=angles[i] if angles is not None else None,
            mirror=mirrors[i] if mirrors is not None else False,
        )

    handles = [
        mpatches.Patch(
            facecolor=ROOM_COLORS[t],
            edgecolor="#999",
            linewidth=0.4,
            label=f"{t}: {SEMANTICS[t]}",
        )
        for t in SEMANTICS
        if t < len(ROOM_COLORS)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=9,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.text(
        0.5,
        1.01,
        "Shape + entrance (rotation-aware) + room count retrieval  "
        "(red dot = door, arrow = inward normal)",
        ha="center",
        fontsize=12,
    )
    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Shape + entrance (rotation-aware) + room-count retrieval."
    )
    parser.add_argument("--query", default=None, help="Path to query JSON (omit with --build_index)")
    parser.add_argument("--data_dir", required=True, help="Directory of plan JSON files")
    parser.add_argument("--name_json", default=None)
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--shape_w", type=float, default=1.0)
    parser.add_argument("--ent_w",   type=float, default=1.0)
    parser.add_argument("--room_w",  type=float, default=1.0)
    parser.add_argument("--out", default=None, help="Output image path")
    parser.add_argument(
        "--build_index", default=None, metavar="PATH",
        help="Compute descriptors for all plans in data_dir and save index to PATH (.npz). "
             "Exits after saving — do not combine with --query.",
    )
    parser.add_argument(
        "--index", default=None, metavar="PATH",
        help="Path to a pre-built .npz index (skips scanning data_dir at query time).",
    )
    args = parser.parse_args()

    # ── index build mode ──────────────────────────────────────────────────────
    if args.build_index:
        build_index(args.data_dir, args.build_index, args.name_json)
        return

    # ── query mode ────────────────────────────────────────────────────────────
    if not args.query:
        parser.error("--query is required unless --build_index is set")

    with open(args.query) as f:
        q_data = json.load(f)
    if not q_data.get("boundary"):
        raise ValueError(f"Query has no boundary: {args.query}")

    q_shape = shape_desc_complex(q_data["boundary"])
    q_ent   = entrance_desc(q_data)
    q_room  = room_count_desc(q_data)

    dc = frontdoor_center(q_data)
    print(f"Query : {q_data.get('name', args.query)}")
    print(f"  desc  shape={N_FREQS}-dim  entrance={ENT_DIM}-dim  rooms={ROOM_DIM}-dim")
    print(f"  weights  shape_w={args.shape_w}  ent_w={args.ent_w}  room_w={args.room_w}")
    print(
        f"  front door: center={dc}  "
        f"dir=({q_ent[0]:.3f},{q_ent[1]:.3f})  normal=({q_ent[2]:.3f},{q_ent[3]:.3f})  "
        f"r/mean_r={q_ent[4]:.3f}"
    )
    print(f"  rooms: {room_summary(q_data)}")

    # ── load or build descriptor database ────────────────────────────────────
    if args.index:
        db_names, db_shapes, db_ents, db_rooms = load_index(args.index)
        db_raws = None  # loaded on demand below
    else:
        print(f"\nBuilding database from: {args.data_dir}")
        db_names, db_shapes, db_ents, db_rooms, db_raws = build_db(
            args.data_dir, args.name_json
        )
        print(f"  {len(db_names)} plans indexed")

    idx, dists, shape_d, ent_d, room_d, best_angles, raw_angles_top, use_mirror = retrieve(
        q_shape, q_ent, q_room,
        db_shapes, db_ents, db_rooms,
        args.shape_w, args.ent_w, args.room_w, args.topn,
    )

    # load raw JSON for top-N results (needed for display / downstream use)
    def _load_raw(name):
        stem = os.path.splitext(name)[0]
        path = os.path.join(args.data_dir, stem + ".json")
        with open(path) as f:
            return json.load(f)

    top_raws = (
        [db_raws[i] for i in idx]
        if db_raws is not None
        else [_load_raw(db_names[i]) for i in idx]
    )

    print(f"\nTop-{args.topn} results:")
    print(f"  {'name':25s}  {'combined':>8}  {'shape':>7}  {'ent(rot)':>8}  {'rooms':>7}  α(snap)  α(raw)  rooms")
    for rank, (i, d, sd, ed_, rd, angle, raw_a, mirror) in enumerate(
        zip(idx, dists, shape_d, ent_d, room_d, best_angles, raw_angles_top, use_mirror), 1
    ):
        mirror_str = " (mirror)" if mirror else ""
        snap_deg = np.degrees(angle)
        raw_deg = np.degrees(raw_a)
        print(
            f"  #{rank:2d} {db_names[i]:25s}  {d:8.4f}  {sd:7.4f}  "
            f"{ed_:8.4f}  {rd:7.4f}  {snap_deg:5.0f}°  {raw_deg:6.1f}°{mirror_str}  "
            f"[{room_summary(top_raws[rank-1])}]"
        )

    visualize(
        q_data, top_raws, [db_names[i] for i in idx],
        dists, shape_d, ent_d, room_d, args.topn, out=args.out,
        angles=best_angles, mirrors=use_mirror,
    )


if __name__ == "__main__":
    main()
