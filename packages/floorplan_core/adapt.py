"""
adapt.py — Floor plan adaptation: fit retrieved candidates to a new query boundary.

Stage in pipeline:  retrieve → adapt → evaluate → present

Public API
----------
adapt_candidate(query_fp, candidate_fp) -> dict
adapt_all(query_fp, candidates)         -> list[dict]
"""

import copy
import json
import logging
import math
import argparse
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.errors import TopologicalError

logger = logging.getLogger(__name__)


# ── Low-level geometry helpers ────────────────────────────────────────────────


def polygon_centroid(vertices):
    """Signed area and centroid via shoelace. vertices: [[x, y], ...]"""
    n = len(vertices)
    if n < 3:
        raise ValueError("Need at least 3 vertices for centroid")
    A = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = vertices[i][0], vertices[i][1]
        x1, y1 = vertices[(i + 1) % n][0], vertices[(i + 1) % n][1]
        cross = x0 * y1 - x1 * y0
        A += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    A *= 0.5
    if abs(A) < 1e-12:
        # Degenerate polygon: fall back to vertex mean
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        return sum(xs) / n, sum(ys) / n
    cx /= 6.0 * A
    cy /= 6.0 * A
    return cx, cy


def polygon_area(vertices):
    """Shoelace area (unsigned). vertices: [[x, y], ...]"""
    n = len(vertices)
    A = 0.0
    for i in range(n):
        x0, y0 = vertices[i][0], vertices[i][1]
        x1, y1 = vertices[(i + 1) % n][0], vertices[(i + 1) % n][1]
        A += x0 * y1 - x1 * y0
    return abs(A) * 0.5


def translate_polygon(vertices, dx, dy):
    return [[v[0] + dx, v[1] + dy] for v in vertices]


def scale_polygon(vertices, scale, cx, cy):
    return [[cx + scale * (v[0] - cx), cy + scale * (v[1] - cy)] for v in vertices]


def rotate_polygon(vertices, angle_deg, cx, cy):
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    result = []
    for v in vertices:
        dx, dy = v[0] - cx, v[1] - cy
        result.append([cx + cos_a * dx - sin_a * dy,
                        cy + sin_a * dx + cos_a * dy])
    return result


def mirror_polygon(vertices, cx):
    """Mirror about vertical axis through cx."""
    return [[2.0 * cx - v[0], v[1]] for v in vertices]


def _mirror_direction(d):
    """Swap left (3) and right (1) directions when mirroring horizontally."""
    if d == 1:
        return 3
    if d == 3:
        return 1
    return d


def transform_boundary(boundary, dx, dy, scale, angle_deg, mirror, cx, cy):
    """Apply full transform sequence to a v2.1.0 boundary vertex list.

    Sequence: translate → scale → rotate → mirror (all about (cx, cy) after
    translation).  direction and is_door_vertex are preserved; if mirrored,
    left/right directions are swapped (RPLAN: 0=up 1=right 2=down 3=left,
    mirroring swaps 1 ↔ 3).

    Returns a new boundary list [[x, y, direction, is_door_vertex], ...].
    """
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    result = []
    for vertex in boundary:
        x, y = vertex[0], vertex[1]
        direction = vertex[2]
        is_door = vertex[3]

        # 1. Translate
        x += dx
        y += dy
        # 2. Scale about (cx, cy)
        x = cx + scale * (x - cx)
        y = cy + scale * (y - cy)
        # 3. Rotate about (cx, cy)
        if angle_deg != 0:
            ddx, ddy = x - cx, y - cy
            x = cx + cos_a * ddx - sin_a * ddy
            y = cy + sin_a * ddx + cos_a * ddy
        # 4. Mirror about vertical axis through cx
        if mirror:
            x = 2.0 * cx - x
            direction = _mirror_direction(direction)

        result.append([x, y, direction, is_door])
    return result


def compute_iou(poly_a_verts, poly_b_verts):
    """IoU using shapely. Returns float in [0, 1]."""
    try:
        pa = ShapelyPolygon(poly_a_verts)
        pb = ShapelyPolygon(poly_b_verts)
        if not pa.is_valid:
            pa = pa.buffer(0)
        if not pb.is_valid:
            pb = pb.buffer(0)
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        return inter / union if union > 0 else 0.0
    except (TopologicalError, Exception) as exc:
        logger.debug("compute_iou error: %s", exc)
        return 0.0


def recompute_bbox(polygon):
    """Returns [x1, y1, x2, y2] from polygon vertices."""
    xs = [v[0] for v in polygon]
    ys = [v[1] for v in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def _transform_point(x, y, dx, dy, scale, angle_deg, mirror, cx, cy):
    """Apply translate → scale → rotate → mirror to a single (x, y) point."""
    x += dx
    y += dy
    x = cx + scale * (x - cx)
    y = cy + scale * (y - cy)
    if angle_deg != 0:
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        ddx, ddy = x - cx, y - cy
        x = cx + cos_a * ddx - sin_a * ddy
        y = cy + sin_a * ddx + cos_a * ddy
    if mirror:
        x = 2.0 * cx - x
    return x, y


# ── Internal helpers ──────────────────────────────────────────────────────────


def _boundary_xy(boundary):
    """Extract [[x, y], ...] from a v2.1.0 boundary list."""
    return [[v[0], v[1]] for v in boundary]


def _largest_polygon(geom):
    """Return the largest Polygon from any shapely geometry type."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    # MultiPolygon or GeometryCollection
    polys = [g for g in geom.geoms
             if g.geom_type == "Polygon" and not g.is_empty]
    return max(polys, key=lambda p: p.area) if polys else None


# ── Main adaptation logic ─────────────────────────────────────────────────────


def adapt_candidate(query_fp: dict, candidate_fp: dict) -> dict:
    """Adapt a single candidate floor plan to the query boundary.

    query_fp:     v2.1.0 dict with boundary and entrance (rooms not required)
    candidate_fp: v2.1.0 dict with full room layout
    Returns:      adapted v2.1.0 dict
    """
    cand = copy.deepcopy(candidate_fp)
    q_boundary = query_fp["boundary"]
    q_xy = _boundary_xy(q_boundary)

    # ── Validate inputs ───────────────────────────────────────────────────────
    if len(q_xy) < 3:
        raise ValueError("Query boundary has fewer than 3 vertices")
    q_area = polygon_area(q_xy)
    if q_area < 1e-12:
        raise ValueError("Query boundary has zero area")

    c_boundary = cand["boundary"]
    c_xy = _boundary_xy(c_boundary)
    if len(c_xy) < 3:
        raise ValueError(
            f"Candidate '{cand.get('id')}' boundary has fewer than 3 vertices"
        )
    c_area = polygon_area(c_xy)
    if c_area < 1e-12:
        raise ValueError(f"Candidate '{cand.get('id')}' boundary has zero area")

    # ── Step 1 — Centroid alignment ───────────────────────────────────────────
    qcx, qcy = polygon_centroid(q_xy)
    ccx, ccy = polygon_centroid(c_xy)
    tdx = qcx - ccx
    tdy = qcy - ccy

    cand_bxy = translate_polygon(c_xy, tdx, tdy)
    rooms = cand.get("rooms") or []
    for room in rooms:
        room["polygon"] = translate_polygon(room["polygon"], tdx, tdy)

    # ── Step 2 — Area-proportional uniform scaling ────────────────────────────
    c_area_t = polygon_area(cand_bxy)
    scale_factor = math.sqrt(q_area / c_area_t) if c_area_t > 1e-12 else 1.0

    cand_bxy = scale_polygon(cand_bxy, scale_factor, qcx, qcy)
    for room in rooms:
        room["polygon"] = scale_polygon(room["polygon"], scale_factor, qcx, qcy)

    # ── Step 3 — Best-of-8 orientation matching ───────────────────────────────
    best_iou = -1.0
    best_orientation = 0
    best_mirrored = False
    best_bxy = cand_bxy
    best_room_polys = {r["id"]: r["polygon"] for r in rooms}

    for angle in (0, 90, 180, 270):
        for mirror in (False, True):
            if angle == 0 and not mirror:
                bxy = cand_bxy
                rpolys = {r["id"]: r["polygon"] for r in rooms}
            else:
                bxy = rotate_polygon(cand_bxy, angle, qcx, qcy)
                rpolys = {r["id"]: rotate_polygon(r["polygon"], angle, qcx, qcy)
                          for r in rooms}
                if mirror:
                    bxy = mirror_polygon(bxy, qcx)
                    rpolys = {rid: mirror_polygon(p, qcx)
                              for rid, p in rpolys.items()}

            iou = compute_iou(q_xy, bxy)
            if iou > best_iou:
                best_iou = iou
                best_orientation = angle
                best_mirrored = mirror
                best_bxy = bxy
                best_room_polys = rpolys

    # If all orientations produced zero overlap, keep identity transform
    if best_iou <= 0:
        best_orientation = 0
        best_mirrored = False
        best_bxy = cand_bxy          # noqa: F841 (kept for clarity)
        best_room_polys = {r["id"]: r["polygon"] for r in rooms}

    # Apply best room polygons
    for room in rooms:
        room["polygon"] = best_room_polys[room["id"]]

    # ── Transform candidate entrance (used when query has no entrance) ────────
    q_entrance = query_fp.get("entrance")
    if q_entrance is not None:
        adapted_entrance = q_entrance
    else:
        c_entrance = candidate_fp.get("entrance")
        if c_entrance and len(c_entrance) == 4:
            ax1, ay1 = _transform_point(
                c_entrance[0], c_entrance[1],
                tdx, tdy, scale_factor, best_orientation, best_mirrored, qcx, qcy)
            ax2, ay2 = _transform_point(
                c_entrance[2], c_entrance[3],
                tdx, tdy, scale_factor, best_orientation, best_mirrored, qcx, qcy)
            adapted_entrance = [ax1, ay1, ax2, ay2]
        else:
            adapted_entrance = None

    # ── Step 4 — Room clipping and flagging ───────────────────────────────────
    q_shapely = ShapelyPolygon(q_xy)
    if not q_shapely.is_valid:
        q_shapely = q_shapely.buffer(0)

    surviving_rooms = []
    removed_ids: set = set()
    rooms_removed = 0

    for room in rooms:
        poly_verts = room["polygon"]
        if len(poly_verts) < 3:
            removed_ids.add(room["id"])
            rooms_removed += 1
            continue
        try:
            room_poly = ShapelyPolygon(poly_verts)
            if not room_poly.is_valid:
                room_poly = room_poly.buffer(0)
            orig_area = room_poly.area
            if orig_area < 1e-12:
                removed_ids.add(room["id"])
                rooms_removed += 1
                continue

            intersection = q_shapely.intersection(room_poly)
            inter_area = intersection.area
            ratio = inter_area / orig_area

            if ratio >= 0.95:
                # Unclipped — keep polygon as-is
                room["clipped"] = False
                room["bbox"] = recompute_bbox(poly_verts)
                # bbox_interior: copy from source (already present in deep copy)
                surviving_rooms.append(room)

            elif ratio >= 0.10:
                # Partially outside — keep full original polygon, mark as clipped
                room["clipped"] = True
                room["bbox"] = recompute_bbox(poly_verts)
                # bbox_interior: keep from source (already present in deep copy)
                surviving_rooms.append(room)

            else:
                # Effectively outside — discard
                removed_ids.add(room["id"])
                rooms_removed += 1

        except (TopologicalError, Exception) as exc:
            logger.warning("Shapely error clipping room '%s': %s",
                           room.get("id"), exc)
            removed_ids.add(room["id"])
            rooms_removed += 1

    # Remove edges that reference any removed room
    surviving_edges = [
        e for e in (cand.get("edges") or [])
        if e.get("u") not in removed_ids and e.get("v") not in removed_ids
    ]

    # ── Step 5 — Output assembly ──────────────────────────────────────────────
    return {
        "schema_version": "2.1.0",
        "id": f"{candidate_fp['id']}__adapted",
        "source": "adapted",
        "boundary": q_boundary,
        "entrance": adapted_entrance,
        "rooms": surviving_rooms,
        "edges": surviving_edges,
        "doors": candidate_fp.get("doors"),
        "windows": candidate_fp.get("windows"),
        "walls": None,
        "requirements": query_fp.get("requirements"),
        "retrieval": {
            "source_id": candidate_fp["id"],
            "iou_score": round(best_iou, 6),
            "orientation": best_orientation,
            "mirrored": best_mirrored,
            "scale_factor": round(scale_factor, 6),
            "rooms_removed": rooms_removed,
        },
    }


def adapt_all(query_fp: dict, candidates: list) -> list:
    """Adapt all candidates.

    Returns list of adapted dicts in the same order.  Candidates that fail
    adaptation are returned with source="adapt_failed" and empty rooms list
    rather than crashing the pipeline.
    """
    results = []
    for cand in candidates:
        cand_id = cand.get("id", "unknown")
        try:
            results.append(adapt_candidate(query_fp, cand))
        except Exception as exc:
            logger.warning("adapt_candidate failed for '%s': %s", cand_id, exc)
            results.append({
                "schema_version": "2.1.0",
                "id": f"{cand_id}__adapted",
                "source": "adapt_failed",
                "boundary": query_fp.get("boundary"),
                "entrance": query_fp.get("entrance"),
                "rooms": [],
                "edges": [],
                "walls": None,
                "requirements": query_fp.get("requirements"),
                "retrieval": None,
                "error": str(exc),
            })
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser():
    p = argparse.ArgumentParser(
        description="Adapt retrieved floor plan candidates to a query boundary"
    )
    p.add_argument("--query", required=True,
                   help="Path to query floor plan JSON")
    p.add_argument("--candidates", nargs="+",
                   help="Paths to individual candidate JSON files")
    p.add_argument("--candidates_dir",
                   help="Directory of candidate JSON files (*.json)")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for adapted JSONs")
    return p


def main():
    args = _build_parser().parse_args()

    with open(args.query) as fh:
        query_fp = json.load(fh)

    if args.candidates:
        candidate_paths = [Path(p) for p in args.candidates]
    elif args.candidates_dir:
        candidate_paths = sorted(Path(args.candidates_dir).glob("*.json"))
    else:
        _build_parser().error("Provide --candidates or --candidates_dir")

    candidates = []
    for cp in candidate_paths:
        with open(cp) as fh:
            candidates.append(json.load(fh))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapted_list = adapt_all(query_fp, candidates)

    for adapted in adapted_list:
        ret = adapted.get("retrieval") or {}
        print(
            f"{adapted['id']}  "
            f"iou={ret.get('iou_score', 0.0):.3f}  "
            f"rot={ret.get('orientation', 0)}°  "
            f"mirror={ret.get('mirrored', False)}  "
            f"scale={ret.get('scale_factor', 1.0):.2f}  "
            f"removed={ret.get('rooms_removed', 0)}"
        )
        with open(out_dir / f"{adapted['id']}.json", "w") as fh:
            json.dump(adapted, fh, indent=2)

    print(f"\nWrote {len(adapted_list)} adapted floor plan(s) to {out_dir}/")


# ── Self-test __main__ block ──────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        logging.basicConfig(level=logging.WARNING)
        main()
    else:
        # ── Self-test ──────────────────────────────────────────────────────────
        logging.basicConfig(level=logging.INFO)

        # 1. Simple square query boundary [0.1,0.1] → [0.9,0.9]
        query_fp = {
            "schema_version": "2.1.0",
            "id": "query_test",
            "source": "user",
            "boundary": [
                [0.1, 0.1, 0, 0],
                [0.9, 0.1, 1, 0],
                [0.9, 0.9, 2, 0],
                [0.1, 0.9, 3, 0],
            ],
            "entrance": [0.1, 0.1, 0.2, 0.1],
            "rooms": [],
            "edges": [],
            "walls": None,
            "requirements": None,
            "retrieval": None,
        }

        # 2. Candidate with 2 rectangular rooms offset from the query
        candidate_fp = {
            "schema_version": "2.1.0",
            "id": "cand_test",
            "source": "rplan",
            "boundary": [
                [0.3, 0.3, 0, 0],
                [0.8, 0.3, 1, 0],
                [0.8, 0.7, 2, 0],
                [0.3, 0.7, 3, 0],
            ],
            "entrance": [0.3, 0.3, 0.4, 0.3],
            "rooms": [
                {
                    "id": "r_0",
                    "type": 0,
                    "source_type": 0,
                    "label": "LivingRoom",
                    "polygon": [
                        [0.3, 0.3], [0.55, 0.3],
                        [0.55, 0.7], [0.3, 0.7],
                    ],
                    "bbox": [0.3, 0.3, 0.55, 0.7],
                    "bbox_interior": [0.3, 0.3, 0.55, 0.7],
                },
                {
                    "id": "r_1",
                    "type": 2,
                    "source_type": 7,
                    "label": "Bedroom",
                    "polygon": [
                        [0.55, 0.3], [0.8, 0.3],
                        [0.8, 0.7], [0.55, 0.7],
                    ],
                    "bbox": [0.55, 0.3, 0.8, 0.7],
                    "bbox_interior": [0.55, 0.3, 0.8, 0.7],
                },
            ],
            "edges": [
                {"u": "r_0", "v": "r_1", "relation": 1, "connection": 1},
            ],
            "walls": None,
            "requirements": None,
            "retrieval": None,
        }

        # 3. Run adapt_candidate
        result = adapt_candidate(query_fp, candidate_fp)
        iou_score = result["retrieval"]["iou_score"]
        room_count = len(result["rooms"])
        print(f"iou_score  = {iou_score}")
        print(f"room_count = {room_count}")

        # 4. Assert expectations
        assert iou_score > 0, f"Expected iou_score > 0, got {iou_score}"
        assert room_count > 0, f"Expected room_count > 0, got {room_count}"
        print("Self-test passed.")
