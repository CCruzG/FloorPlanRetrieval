"""
fit_plan.py — Geometric fitting algorithm for floor plan adaptation.

Takes a retrieved floor plan (schema v2.1.0) and a user-drawn target
boundary and maps each room to its proportionally equivalent position
in the target boundary. Room sizes are preserved unchanged.

Public interface
----------------
    fit_plan(source_plan, target) -> dict

Internal pipeline
-----------------
    fit_plan()
      ├── _analyse(source_plan)
      └── _map_centroids(analysis, target)
"""

import copy
import json

from shapely.geometry import Polygon
from shapely.validation import make_valid

# ── constants ──────────────────────────────────────────────────────────────────

EPSILON = 0.005   # normalised units
NORM    = 18.0    # metres per normalised unit

COMMON_TYPE = 0


# ── public interface ───────────────────────────────────────────────────────────

def fit_plan(source_plan: dict, target: dict) -> dict:
    """
    Fit a retrieved floor plan to a target boundary by proportional
    centroid mapping. Room sizes are unchanged.

    Parameters
    ----------
    source_plan : dict
        Complete floor plan in schema v2.1.0.
    target : dict
        {"boundary": [[x, y, direction, is_door_vertex], ...],
         "entrance": [x, y]}

    Returns
    -------
    dict
        Floor plan in schema v2.1.0 with rooms repositioned to their
        proportionally equivalent positions in the target boundary.
        fit_result.status is always "ok" with no flags.
    """
    analysis = _analyse(source_plan)
    mapped   = _map_centroids(analysis, target)
    return _build_output(source_plan, target, mapped)


# ── Step 1 ─────────────────────────────────────────────────────────────────────

def _analyse(source_plan: dict) -> dict:
    """Extract geometry from source plan."""
    bdry_raw   = source_plan.get("boundary", [])
    bdry_verts = [[v[0], v[1]] for v in bdry_raw]
    src_poly   = make_valid(Polygon(bdry_verts)) if len(bdry_verts) >= 3 \
                 else Polygon()
    src_bbox   = src_poly.bounds  # (minx, miny, maxx, maxy)

    rooms_data = {}
    primary_common_id = None
    common_candidates = []

    for room in source_plan.get("rooms", []):
        st = room.get("source_type", room.get("type", -1))
        if st == COMMON_TYPE:
            poly_verts = room.get("polygon", [])
            area = Polygon(poly_verts).area if len(poly_verts) >= 3 else 0.0
            common_candidates.append((area, room["id"]))

    if common_candidates:
        common_candidates.sort(reverse=True)
        primary_common_id = common_candidates[0][1]

    for room in source_plan.get("rooms", []):
        rid  = room["id"]
        st   = room.get("source_type", room.get("type", -1))
        poly_verts = room.get("polygon", [])

        if len(poly_verts) >= 3:
            rpoly = make_valid(Polygon(poly_verts))
            rbbox = rpoly.bounds  # (minx, miny, maxx, maxy)
        else:
            rpoly = Polygon()
            rbbox = (0, 0, 0, 0)

        is_common = (st == COMMON_TYPE and rid == primary_common_id)

        rooms_data[rid] = {
            "id":          rid,
            "source_type": st,
            "bbox":        rbbox,
            "is_common":   is_common,
        }

    return {
        "src_bbox": src_bbox,
        "rooms":    rooms_data,
    }


# ── Step 2 ─────────────────────────────────────────────────────────────────────

def _map_centroids(analysis: dict, target: dict) -> dict:
    """
    For each room (excluding common area), compute its centroid's
    normalised position within the source boundary bounding box,
    then place it at the same normalised position within the target
    boundary bounding box. Room width and height are scaled by the
    same x/y scale factors used for the centroid mapping.

    source normalised position:
        norm_cx = (room_cx - src_bbox.x1) / src_bbox.width
        norm_cy = (room_cy - src_bbox.y1) / src_bbox.height

    target centroid:
        new_cx = tgt_bbox.x1 + norm_cx * tgt_bbox.width
        new_cy = tgt_bbox.y1 + norm_cy * tgt_bbox.height

    new size:
        scale_x = tgt_bbox.width  / src_bbox.width
        scale_y = tgt_bbox.height / src_bbox.height
        new_w   = src_room_width  * scale_x
        new_h   = src_room_height * scale_y
    """
    bdry_verts = [[v[0], v[1]] for v in target.get("boundary", [])]
    tgt_poly   = make_valid(Polygon(bdry_verts)) if len(bdry_verts) >= 3 \
                 else Polygon()
    tgt_bbox   = tgt_poly.bounds  # (tgt_x1, tgt_y1, tgt_x2, tgt_y2)

    src_bbox = analysis["src_bbox"]
    src_x1, src_y1, src_x2, src_y2 = src_bbox
    src_w = src_x2 - src_x1
    src_h = src_y2 - src_y1

    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = tgt_bbox
    tgt_w = tgt_x2 - tgt_x1
    tgt_h = tgt_y2 - tgt_y1

    scale_x = tgt_w / src_w if src_w > 1e-8 else 1.0
    scale_y = tgt_h / src_h if src_h > 1e-8 else 1.0

    mapped = {}
    for rid, room in analysis["rooms"].items():
        if room["is_common"]:
            continue

        bx1, by1, bx2, by2 = room["bbox"]

        room_cx = (bx1 + bx2) / 2
        room_cy = (by1 + by2) / 2

        norm_cx = (room_cx - src_x1) / src_w if src_w > 1e-8 else 0.5
        norm_cy = (room_cy - src_y1) / src_h if src_h > 1e-8 else 0.5

        new_cx = tgt_x1 + norm_cx * tgt_w
        new_cy = tgt_y1 + norm_cy * tgt_h

        new_w = (bx2 - bx1) * scale_x
        new_h = (by2 - by1) * scale_y

        new_bbox = (new_cx - new_w / 2, new_cy - new_h / 2,
                    new_cx + new_w / 2, new_cy + new_h / 2)
        mapped[rid] = new_bbox

    return mapped


# ── Step 3 ─────────────────────────────────────────────────────────────────────

def _build_output(source_plan: dict, target: dict, mapped: dict) -> dict:
    """Assemble final schema v2.1.0 output."""
    result = copy.deepcopy(source_plan)

    # Replace boundary with target values
    result["boundary"] = _annotate_boundary(target.get("boundary", []))

    # Replace entrance
    tgt_ent = target.get("entrance")
    if tgt_ent and len(tgt_ent) == 2:
        delta = EPSILON
        result["entrance"] = [tgt_ent[0] - delta, tgt_ent[1] - delta,
                               tgt_ent[0] + delta, tgt_ent[1] + delta]
    elif tgt_ent and len(tgt_ent) == 4:
        result["entrance"] = tgt_ent

    # Replace rooms
    rooms_out = []
    for room in source_plan.get("rooms", []):
        rid = room["id"]
        st  = room.get("source_type", room.get("type", -1))
        r   = copy.deepcopy(room)

        if st == COMMON_TYPE:
            r["polygon"] = []
            r["bbox"] = [0, 0, 0, 0]
            r["bbox_interior"] = [0, 0, 0, 0]
        elif rid in mapped:
            bbox = mapped[rid]
            x1, y1, x2, y2 = bbox
            r["bbox"] = list(bbox)
            r["bbox_interior"] = _inset_bbox(bbox)
            r["polygon"] = [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ]

        rooms_out.append(r)

    result["rooms"] = rooms_out
    result["fit_result"] = {"status": "ok", "flags": []}
    return result


# ── helpers ────────────────────────────────────────────────────────────────────

def _annotate_boundary(boundary_raw):
    """Ensure boundary vertices have [x, y, direction, is_door_vertex] format."""
    out = []
    for v in boundary_raw:
        if len(v) >= 4:
            out.append(list(v[:4]))
        elif len(v) == 2:
            out.append([v[0], v[1], 0, 0])
    n = len(out)
    for i in range(n):
        x1, y1 = out[i][0], out[i][1]
        x2, y2 = out[(i + 1) % n][0], out[(i + 1) % n][1]
        dx, dy  = x2 - x1, y2 - y1
        if abs(dx) >= abs(dy):
            direction = 1 if dx > 0 else 0
        else:
            direction = 3 if dy < 0 else 2
        out[i][2] = direction
    return out


def _inset_bbox(bbox, amount=EPSILON / 2):
    x1, y1, x2, y2 = bbox
    return [x1 + amount, y1 + amount, x2 - amount, y2 - amount]


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit a floor plan to a target boundary (proportional centroid mapping)"
    )
    parser.add_argument("source",   help="Source floor plan JSON (v2.1.0)")
    parser.add_argument("target",   help='Target JSON: {"boundary": [...], "entrance": [...]}')
    parser.add_argument("--out",    default="fitted.json", help="Output path")
    parser.add_argument("--pretty", action="store_true",   help="Pretty-print output")
    args = parser.parse_args()

    with open(args.source) as f:
        src = json.load(f)
    with open(args.target) as f:
        tgt = json.load(f)

    result = fit_plan(src, tgt)
    fr = result.get("fit_result", {})
    print(f"status: {fr.get('status')}  flags: {len(fr.get('flags', []))}")

    indent = 2 if args.pretty else None
    with open(args.out, "w") as f:
        json.dump(result, f, indent=indent)
    print(f"Wrote {args.out}")
