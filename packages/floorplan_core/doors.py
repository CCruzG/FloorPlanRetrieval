"""Doors derived from a generated plan (decision 2026-11-04). Brain, pure.

Doors are not stated in the Spec. This function reads the generated rooms and the
plan's edges and places them:

  * For each edge with connection 1 between two generated rooms: find the straight
    pieces of their shared wall; take the longest. A piece shorter than
    `door_min_wall_mm` (1000) gets no door: the edge is listed as unplaced.
  * The door is `door_width_mm` (900) wide. Its near jamb is `door_jamb_mm` (100)
    from one end of the piece: the end nearer the centroid of the less private
    room (hallway, then living, ...; ties by room id), ties by coordinate.
  * Connection 2 (open) and 0 (wall) edges get no door.
  * Entrance door: `entrance_door_mm` (900) on the boundary edge nearest the
    entrance point, centred on the point, moved inward until both jambs are at
    least `door_jamb_mm` from the edge's ends. A boundary edge shorter than
    `door_min_wall_mm` but at least `entrance_min_edge_mm` (700) gets the entrance
    door centred on the edge, width min(`entrance_door_mm`, edge - 2 x 50 mm).
    A shorter edge gets none (listed as unplaced, v = "entrance").

Parameters live in `options` (spec.build_options); same inputs give the same
output. Coordinates are plan units (boundary units); one unit = `mm_per_unit` mm.
"""
import math

from shapely.geometry import Polygon as SPoly
from shapely.ops import linemerge

DEFAULTS = {
    "door_width_mm": 900.0,
    "door_jamb_mm": 100.0,
    "door_min_wall_mm": 1000.0,
    "entrance_door_mm": 900.0,
    "entrance_min_edge_mm": 700.0,
}

# Lower = more public. Unlisted types rank 4 (between kitchen and the private rooms).
_PRIVACY = {"hallway": 0, "living": 1, "living_room": 1, "dining": 2, "kitchen": 3,
            "bedroom": 5, "master_bedroom": 5, "study": 5, "bathroom": 5, "storage": 5}


def _privacy(room):
    return _PRIVACY.get(str(room.get("type") or "").lower(), 4)


def _straight_pieces(geom):
    """Straight runs of a (multi)line geometry, as [(a, b)] with a < b by coordinate."""
    if geom is None or geom.is_empty:
        return []
    lines = []
    if geom.geom_type == "LineString":
        lines = [geom]
    elif geom.geom_type == "MultiLineString":
        lines = list(geom.geoms)
    elif geom.geom_type == "GeometryCollection":
        lines = [g for g in geom.geoms if g.geom_type in ("LineString", "MultiLineString")]
    merged = []
    for ln in lines:
        m = linemerge(ln) if ln.geom_type == "MultiLineString" else ln
        merged.extend(list(m.geoms) if m.geom_type == "MultiLineString" else [m])
    out = []
    for ln in merged:
        pts = list(ln.coords)
        if len(pts) < 2:
            continue
        run = [pts[0], pts[1]]
        for p in pts[2:]:
            a, b = run[0], run[-1]
            cross = (b[0] - a[0]) * (p[1] - b[1]) - (b[1] - a[1]) * (p[0] - b[0])
            dot = (b[0] - a[0]) * (p[0] - b[0]) + (b[1] - a[1]) * (p[1] - b[1])
            if abs(cross) <= 1e-9 and dot > 0:
                run[-1] = p
            else:
                out.append((run[0], run[-1]))
                run = [b, p]
        out.append((run[0], run[-1]))
    rnd = lambda q: (round(float(q[0]), 6), round(float(q[1]), 6))   # 1e-6 unit snaps float noise
    return [tuple(sorted((rnd(a), rnd(b)))) for a, b in out]


def _door_on_piece(a, b, toward, width_u, jamb_u):
    """Centre of a door on piece a-b, near jamb `jamb_u` from the end nearer `toward`."""
    da = math.hypot(a[0] - toward[0], a[1] - toward[1])
    db = math.hypot(b[0] - toward[0], b[1] - toward[1])
    start, end = (a, b) if da <= db + 1e-5 else (b, a)   # a < b by coordinate: ties go to a
    L = math.hypot(end[0] - start[0], end[1] - start[1])
    ux, uy = (end[0] - start[0]) / L, (end[1] - start[1]) / L
    d = jamb_u + width_u / 2.0
    return [start[0] + ux * d, start[1] + uy * d]


def derive_doors(rooms, edges, boundary, entrance, mm_per_unit, options=None):
    """Return (doors, unplaced). See the module docstring.

    doors: [{u, v, kind ("internal" | "entrance"), centre, width, width_mm,
             segment [a, b], segment_mm, wall [a, b]}]; for an internal door the
             segment is the shared piece and the wall's endpoints are its ends; for
             the entrance door the wall is the boundary edge. Plan units.
    unplaced: [{u, v, segment_mm, reason}] (segment_mm None when no shared wall).
    """
    opt = {**DEFAULTS, **{k: v for k, v in (options or {}).items() if k in DEFAULTS and v is not None}}
    doors, unplaced = [], []
    door_edges = []
    seen = set()
    for e in sorted(edges or [], key=lambda e: (str(e.get("u")), str(e.get("v")))):
        if e.get("connection") != 1:
            continue
        key = frozenset((str(e.get("u")), str(e.get("v"))))
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        door_edges.append(e)
    if not mm_per_unit or mm_per_unit <= 0:
        for e in door_edges:
            unplaced.append({"u": e["u"], "v": e["v"], "segment_mm": None, "reason": "no mm_per_unit"})
        return doors, unplaced

    mm = float(mm_per_unit)
    width_u = opt["door_width_mm"] / mm
    jamb_u = opt["door_jamb_mm"] / mm
    tol = 1e-7
    by_id, polys = {}, {}
    for r in rooms or []:
        poly = r.get("polygon") or []
        if len(poly) >= 3:
            g = SPoly(poly)
            if not g.is_valid:
                g = g.buffer(0)
            if not g.is_empty and g.geom_type == "Polygon":
                by_id[str(r["id"])] = r
                polys[str(r["id"])] = g

    for e in door_edges:
        u, v = e["u"], e["v"]
        su, sv = str(u), str(v)
        if su not in polys or sv not in polys:
            unplaced.append({"u": u, "v": v, "segment_mm": None, "reason": "room not generated"})
            continue
        shared = polys[su].exterior.intersection(polys[sv].exterior.buffer(tol))
        pieces = _straight_pieces(shared)
        if not pieces:
            unplaced.append({"u": u, "v": v, "segment_mm": None, "reason": "no shared wall"})
            continue
        # longest piece; ties by coordinate
        a, b = max(pieces, key=lambda p: (round(math.hypot(p[1][0] - p[0][0], p[1][1] - p[0][1]), 9),
                                          tuple(-c for c in p[0])))
        seg_mm = math.hypot(b[0] - a[0], b[1] - a[1]) * mm
        if seg_mm < opt["door_min_wall_mm"] - 1e-6:
            unplaced.append({"u": u, "v": v, "segment_mm": round(seg_mm, 1),
                             "reason": f"shared wall shorter than {opt['door_min_wall_mm']:.0f} mm"})
            continue
        ru, rv = by_id[su], by_id[sv]
        less = min((ru, rv), key=lambda r: (_privacy(r), str(r["id"])))
        c = polys[str(less["id"])].centroid
        centre = _door_on_piece(a, b, (c.x, c.y), width_u, jamb_u)
        doors.append({"u": u, "v": v, "kind": "internal", "centre": centre, "width": width_u,
                      "width_mm": opt["door_width_mm"], "segment": [list(a), list(b)],
                      "segment_mm": round(seg_mm, 1), "wall": [list(a), list(b)]})

    # entrance door on the unit boundary
    if entrance and len(entrance) >= 4 and boundary and len(boundary) >= 3:
        px, py = (entrance[0] + entrance[2]) / 2.0, (entrance[1] + entrance[3]) / 2.0
        best = None
        n = len(boundary)
        for i in range(n):
            a, b = boundary[i], boundary[(i + 1) % n]
            dx, dy = b[0] - a[0], b[1] - a[1]
            L2 = dx * dx + dy * dy
            if L2 == 0:
                continue
            t = max(0.0, min(1.0, ((px - a[0]) * dx + (py - a[1]) * dy) / L2))
            d = math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy))
            if best is None or d < best[0] - 1e-12:
                best = (d, i, a, b)
        if best is not None:
            _, _, a, b = best
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            L_mm = L * mm
            if L_mm < opt["entrance_min_edge_mm"] - 1e-6:
                unplaced.append({"u": "entrance", "v": None, "segment_mm": round(L_mm, 1),
                                 "reason": f"boundary edge shorter than {opt['entrance_min_edge_mm']:.0f} mm"})
            elif L_mm < opt["door_min_wall_mm"] - 1e-6:
                w_mm = min(opt["entrance_door_mm"], L_mm - 100.0)
                ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
                doors.append({"u": "entrance", "v": None, "kind": "entrance",
                              "centre": [a[0] + ux * L / 2.0, a[1] + uy * L / 2.0],
                              "width": w_mm / mm, "width_mm": w_mm,
                              "segment": [list(map(float, a)), list(map(float, b))],
                              "segment_mm": round(L_mm, 1),
                              "wall": [list(map(float, a)), list(map(float, b))]})
            else:
                ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
                ew = opt["entrance_door_mm"] / mm
                s = (px - a[0]) * ux + (py - a[1]) * uy
                lo, hi = jamb_u + ew / 2.0, L - jamb_u - ew / 2.0
                s = lo if hi < lo else max(lo, min(hi, s))
                doors.append({"u": "entrance", "v": None, "kind": "entrance",
                              "centre": [a[0] + ux * s, a[1] + uy * s], "width": ew,
                              "width_mm": opt["entrance_door_mm"],
                              "segment": [list(map(float, a)), list(map(float, b))],
                              "segment_mm": round(L_mm, 1),
                              "wall": [list(map(float, a)), list(map(float, b))]})
    return doors, unplaced
