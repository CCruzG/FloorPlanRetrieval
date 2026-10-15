"""
fit_transform.py — Coordinate frame conversion between mm (floor plate) and
normalized [0,1] plan space.

Floor plate unit polygons are in mm.  Retrieved floor plans use normalised
[0, 1] coordinates (1 unit ≈ 18 m).  This module converts between the two
so the agent can work in plan space during fitting and convert back to mm
for the canvas.

Uniform scale is used (max dimension → 1.0) so rooms preserve their aspect
ratio.  The smaller axis will span [0, w/h] or [0, h/w] inside the unit.
"""

import copy
import math

import shapely.geometry as _sg


# ── plan-to-boundary scale ─────────────────────────────────────────────────

def scale_plan_to_boundary(plan: dict, boundary: list) -> dict:
    """
    Uniformly scale all room polygons so the plan approximately fills the
    query boundary, centred on the boundary's polygon centroid.

    Using the polygon centroid (not the bounding-box midpoint) is critical for
    non-convex boundaries: the bbox midpoint can land in a notch (outside the
    polygon), which causes rooms to cluster against the concave vertex after
    clipping.  Shapely's centroid is always inside the polygon.

    Scale factor = min(bdry_w/plan_w, bdry_h/plan_h) so rooms fit within the
    bbox.  Common-area rooms scale with the others (fill_residual replaces them).

    Returns a deep copy of plan with updated room polygons.
    """
    all_verts = []
    for room in plan.get("rooms", []):
        all_verts.extend(room.get("polygon") or [])
    if len(all_verts) < 2:
        return copy.deepcopy(plan)

    xs = [v[0] for v in all_verts]
    ys = [v[1] for v in all_verts]
    plan_minx, plan_maxx = min(xs), max(xs)
    plan_miny, plan_maxy = min(ys), max(ys)
    plan_w = plan_maxx - plan_minx
    plan_h = plan_maxy - plan_miny

    # Use the centroid of the plan's room union as the pivot point.  For a
    # well-formed plan this is close to the bbox center, but more accurate.
    try:
        room_polys = [
            _sg.Polygon(r["polygon"])
            for r in plan.get("rooms", [])
            if len(r.get("polygon") or []) >= 3
        ]
        room_union = _sg.unary_union(room_polys)
        plan_cx = room_union.centroid.x
        plan_cy = room_union.centroid.y
    except Exception:
        plan_cx = (plan_minx + plan_maxx) / 2
        plan_cy = (plan_miny + plan_maxy) / 2

    bxs = [v[0] for v in boundary]
    bys = [v[1] for v in boundary]
    bdry_minx, bdry_maxx = min(bxs), max(bxs)
    bdry_miny, bdry_maxy = min(bys), max(bys)
    bdry_w = bdry_maxx - bdry_minx
    bdry_h = bdry_maxy - bdry_miny

    # Polygon centroid: guaranteed inside the polygon, unlike bbox midpoint.
    try:
        bdry_centroid = _sg.Polygon(boundary).centroid
        bdry_cx = bdry_centroid.x
        bdry_cy = bdry_centroid.y
    except Exception:
        bdry_cx = (bdry_minx + bdry_maxx) / 2
        bdry_cy = (bdry_miny + bdry_maxy) / 2

    if plan_w < 1e-8 or plan_h < 1e-8:
        return copy.deepcopy(plan)

    scale = min(bdry_w / plan_w, bdry_h / plan_h)

    result = copy.deepcopy(plan)
    for room in result.get("rooms", []):
        poly = room.get("polygon") or []
        if not poly:
            continue
        new_poly = [
            [(p[0] - plan_cx) * scale + bdry_cx,
             (p[1] - plan_cy) * scale + bdry_cy]
            for p in poly
        ]
        room["polygon"] = new_poly
        if new_poly:
            nxs = [p[0] for p in new_poly]
            nys = [p[1] for p in new_poly]
            room["bbox"] = [min(nxs), min(nys), max(nxs), max(nys)]
    return result


# ── forward transform: mm → normalized ────────────────────────────────────

def normalize_unit_to_plan_frame(unit_polygon_mm: list) -> dict:
    """
    Convert a unit polygon from mm to the normalized [0, 1] plan space.

    Returns
    -------
    {
        boundary           : [[x, y], ...]  normalized polygon
        origin_mm          : [min_x, min_y] bounding box origin in mm
        scale_mm_per_unit  : float           mm per normalized unit
        bbox_mm            : [minx, miny, maxx, maxy]
        size_mm            : [width, height]
    }
    Pass boundary as the target_boundary when fitting.
    Keep origin_mm and scale_mm_per_unit to convert back with rooms_to_mm().
    """
    if len(unit_polygon_mm) < 3:
        raise ValueError("unit_polygon_mm must have at least 3 vertices")

    xs = [p[0] for p in unit_polygon_mm]
    ys = [p[1] for p in unit_polygon_mm]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max_x - min_x
    h = max_y - min_y

    scale = max(w, h)
    if scale < 1e-6:
        raise ValueError("Unit polygon is degenerate (zero or near-zero area)")

    boundary = [
        [(p[0] - min_x) / scale, (p[1] - min_y) / scale]
        for p in unit_polygon_mm
    ]

    return {
        "boundary": boundary,
        "origin_mm": [min_x, min_y],
        "scale_mm_per_unit": scale,
        "bbox_mm": [min_x, min_y, max_x, max_y],
        "size_mm": [w, h],
    }


# ── inverse transform: normalized → mm ────────────────────────────────────

def rooms_to_mm(plan: dict, origin_mm: list, scale_mm_per_unit: float) -> dict:
    """
    Convert all room polygons, bboxes, boundary, and entrance in plan from
    normalized [0, 1] to mm.  Does not modify plan in place.

    Parameters
    ----------
    plan               : v2.1.0 floor plan dict (rooms in normalized space)
    origin_mm          : [ox, oy] from normalize_unit_to_plan_frame
    scale_mm_per_unit  : scale from normalize_unit_to_plan_frame
    """
    result = copy.deepcopy(plan)
    ox, oy = float(origin_mm[0]), float(origin_mm[1])
    s = float(scale_mm_per_unit)

    def _n2mm(x, y):
        return x * s + ox, y * s + oy

    # Boundary
    new_bdry = []
    for v in result.get("boundary", []):
        nx, ny = _n2mm(v[0], v[1])
        if len(v) >= 4:
            new_bdry.append([nx, ny, v[2], v[3]])
        else:
            new_bdry.append([nx, ny])
    result["boundary"] = new_bdry

    # Rooms
    for room in result.get("rooms", []):
        poly = room.get("polygon") or []
        if poly:
            room["polygon"] = [[*_n2mm(p[0], p[1])] for p in poly]

        bbox = room.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1 = _n2mm(bbox[0], bbox[1])
            x2, y2 = _n2mm(bbox[2], bbox[3])
            room["bbox"] = [x1, y1, x2, y2]

        bi = room.get("bbox_interior")
        if bi and len(bi) == 4:
            x1, y1 = _n2mm(bi[0], bi[1])
            x2, y2 = _n2mm(bi[2], bi[3])
            room["bbox_interior"] = [x1, y1, x2, y2]

    # Entrance
    ent = result.get("entrance")
    if ent and len(ent) == 4:
        x1, y1 = _n2mm(ent[0], ent[1])
        x2, y2 = _n2mm(ent[2], ent[3])
        result["entrance"] = [x1, y1, x2, y2]
    elif ent and len(ent) == 2:
        x, y = _n2mm(ent[0], ent[1])
        result["entrance"] = [x, y]

    result["_coordinate_space"] = "mm"
    result["_transform"] = {
        "origin_mm": [ox, oy],
        "scale_mm_per_unit": s,
    }
    return result
