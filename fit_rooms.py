"""
fit_rooms.py
------------
Stage-1 floor plan fitting: translate rooms into the query boundary using
a sequential edge-push algorithm.

Processing order
----------------
1. Bedrooms  (types 1 Master, 5 Child, 7 Second) — placed first as anchors.
2. Bathrooms (type 3)
3. Kitchen   (type 2)
4. Others    (types 6 Study, 8 Guest, 9 Balcony, 10 Entrance, 11 Storage)
5. Living / Dining (types 0, 4) — placed last to fill remaining space.

For each room (in order):
  a. Inherit starting translation from the nearest already-placed room
     (by centroid distance in the original plan space). This keeps
     adjacent rooms together — e.g. bathroom follows its bedroom.
  b. Iterative edge-push: find the boundary half-plane most violated by the
     (translated) room and slide the room along that half-plane's inward
     normal until the violation is resolved. Repeat until fully inside.

The edge-push behaviour for a rectangular room against a parallel boundary
edge is identical to aligning the offending room edge to the boundary edge,
as described in the algorithm spec.

Usage (standalone):
    python fit_rooms.py plan.json --boundary boundary.json --out fitted.json
"""

import copy
import json
import sys

import numpy as np

# ── room processing order ─────────────────────────────────────────────────────

ROOM_PROC_ORDER = [
    frozenset([1, 5, 7]),           # bedrooms  (anchor group)
    frozenset([3]),                  # bathrooms
    frozenset([2]),                  # kitchen
    frozenset([6, 8, 9, 10, 11]),   # study, guest, balcony, entrance, storage
    frozenset([0, 4]),               # living, dining — fill last
]


# ── geometry helpers ───────────────────────────────────────────────────────────

def _poly_centroid(poly: np.ndarray) -> np.ndarray:
    return poly.mean(axis=0)


def _bdry_centroid(verts: np.ndarray) -> np.ndarray:
    return verts.mean(axis=0)


def _build_half_planes(boundary_verts: np.ndarray):
    """
    Return list of (nx, ny, c) for each boundary edge.
    The half-plane condition for a point p to be inside: nx*p[0] + ny*p[1] >= c
    (nx, ny) is the inward normal.
    """
    n = len(boundary_verts)
    cen = _bdry_centroid(boundary_verts)
    planes = []
    for i in range(n):
        a = boundary_verts[i]
        b = boundary_verts[(i + 1) % n]
        d = b - a
        length = np.hypot(d[0], d[1])
        if length < 1e-10:
            continue
        nx, ny = d[1] / length, -d[0] / length
        # Ensure inward (points toward centroid)
        if nx * (cen[0] - a[0]) + ny * (cen[1] - a[1]) < 0:
            nx, ny = -nx, -ny
        c = nx * a[0] + ny * a[1]
        planes.append((nx, ny, c))
    return planes


def _edge_push(poly: np.ndarray, half_planes, start_tx: float = 0.0,
               start_ty: float = 0.0, max_iter: int = 40):
    """
    Translate *poly* until all half-plane constraints are satisfied.

    Starting from (start_tx, start_ty), iteratively fix the worst boundary
    violation by sliding the room along the violating edge's inward normal.

    Returns (tx, ty) — total translation applied.
    """
    tx, ty = start_tx, start_ty

    for _ in range(max_iter):
        worst = 0.0
        wnx = wny = 0.0
        moved = poly + np.array([tx, ty])
        for nx, ny, c in half_planes:
            viols = c - (nx * moved[:, 0] + ny * moved[:, 1])
            v_max = float(viols.max())
            if v_max > worst:
                worst = v_max
                wnx, wny = nx, ny
        if worst < 1e-8:
            break
        tx += wnx * worst
        ty += wny * worst

    return tx, ty


# ── public API ─────────────────────────────────────────────────────────────────

def fit_plan(fp: dict, query_boundary) -> dict:
    """
    Fit a floor plan's rooms into *query_boundary* using the sequential
    edge-push algorithm.

    Parameters
    ----------
    fp             : v2.1.0 floor plan dict. Rooms must have 'polygon': [[x,y]…].
    query_boundary : [[x,y]…] or [[x,y,dir,…]…] — first two values used.

    Returns
    -------
    Deep copy of *fp* with updated room polygons.  Each processed room gets:
      'fitted'     : bool
      'fit_offset' : [tx, ty]  — only if the room was actually moved
    '_fit_stats'   : {'moved': int, 'skipped': int, 'no_fit': int}
    """
    bdry_xy = np.array([[float(v[0]), float(v[1])] for v in query_boundary],
                       dtype=float)
    if len(bdry_xy) < 3:
        return fp

    result = copy.deepcopy(fp)
    rooms = result.get('rooms', [])
    stats = {'moved': 0, 'skipped': 0, 'no_fit': 0}

    # Original polygon arrays (before any modification)
    orig_polys = []
    for r in rooms:
        p = r.get('polygon', [])
        if len(p) >= 3:
            orig_polys.append(np.array(p, dtype=float))
        else:
            orig_polys.append(None)

    # placed[i] = (tx, ty) once processed, else None
    placed = [None] * len(rooms)

    half_planes = _build_half_planes(bdry_xy)

    for type_set in ROOM_PROC_ORDER:
        indices = [
            i for i, r in enumerate(rooms)
            if r.get('source_type', r.get('type', -1)) in type_set
        ]

        for idx in indices:
            room = rooms[idx]
            orig = orig_polys[idx]
            if orig is None:
                room['fitted'] = False
                room['fit_reason'] = 'no_poly'
                continue

            # Inherit translation from nearest already-placed room
            start_tx, start_ty = 0.0, 0.0
            best_dist = float('inf')
            cen_i = _poly_centroid(orig)
            for j, pl in enumerate(placed):
                if pl is None or orig_polys[j] is None:
                    continue
                cen_j = _poly_centroid(orig_polys[j])
                d = float(np.hypot(cen_i[0] - cen_j[0], cen_i[1] - cen_j[1]))
                if d < best_dist:
                    best_dist = d
                    start_tx, start_ty = pl

            # Edge-push to satisfy all boundary constraints
            tx, ty = _edge_push(orig, half_planes, start_tx, start_ty)

            if abs(tx) > 1e-8 or abs(ty) > 1e-8:
                room['polygon'] = [[float(v[0] + tx), float(v[1] + ty)]
                                   for v in orig.tolist()]
                room['fitted'] = True
                room['fit_offset'] = [float(tx), float(ty)]
                stats['moved'] += 1
            else:
                room['fitted'] = True

            placed[idx] = (tx, ty)

    result['_fit_stats'] = stats
    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fit floor plan rooms into boundary')
    parser.add_argument('plan', help='Path to floor plan JSON (v2.1.0)')
    parser.add_argument('--boundary', default='self',
                        help='Boundary JSON file ([[x,y]…]) or "self" to use plan boundary')
    parser.add_argument('--out', default='fitted.json', help='Output path')
    args = parser.parse_args()

    with open(args.plan) as f:
        fp = json.load(f)

    if args.boundary == 'self':
        bdry_raw = fp.get('boundary', [])
    else:
        with open(args.boundary) as f:
            bdry_raw = json.load(f)

    fitted = fit_plan(fp, bdry_raw)
    stats = fitted.pop('_fit_stats', {})
    with open(args.out, 'w') as f:
        json.dump(fitted, f, indent=2)
    print(f"Wrote {args.out}  |  moved={stats.get('moved', 0)}  "
          f"skipped={stats.get('skipped', 0)}  no_fit={stats.get('no_fit', 0)}")
