"""
fit_boolean.py — Boolean polygon operations for floor plan fitting.

subtract_room      : donor − cutter  (resolve overlaps)
clip_room_to_boundary : room ∩ boundary  (trim protrusions)
fill_residual      : common_area = unit − union(all other rooms)  (100% coverage)

All functions return a deep copy of the plan with the modification applied.
For multi-polygon results (e.g. after subtraction), the largest polygon is kept.
"""

import copy

from shapely.geometry import Polygon as SPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid


# ── internal helpers ───────────────────────────────────────────────────────

_INSET = 0.005 / 2


def _room_geom(plan: dict, room_id: str) -> SPolygon:
    for r in plan.get("rooms", []):
        if r["id"] == room_id:
            verts = r.get("polygon") or []
            if len(verts) >= 3:
                return make_valid(SPolygon(verts))
            return SPolygon()
    raise KeyError(f"Room {room_id!r} not found in plan")


def _shapely_to_coords(geom) -> list:
    """Largest simple polygon exterior as [[x, y], ...] (no repeated closing vertex)."""
    if geom.is_empty:
        return []
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type not in ("Polygon",):
        geom = geom.convex_hull
    return [[c[0], c[1]] for c in list(geom.exterior.coords)[:-1]]


def _set_room_from_shapely(plan: dict, room_id: str, geom) -> dict:
    result = copy.deepcopy(plan)
    coords = _shapely_to_coords(geom)
    for r in result.get("rooms", []):
        if r["id"] == room_id:
            r["polygon"] = coords
            if coords:
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                r["bbox"] = [x1, y1, x2, y2]
                r["bbox_interior"] = [
                    x1 + _INSET, y1 + _INSET,
                    x2 - _INSET, y2 - _INSET,
                ]
            else:
                r["bbox"] = [0, 0, 0, 0]
                r["bbox_interior"] = [0, 0, 0, 0]
            return result
    raise KeyError(f"Room {room_id!r} not found in plan")


# ── public API ─────────────────────────────────────────────────────────────

def subtract_room(plan: dict, donor_id: str, cutter_id: str) -> dict:
    """
    Replace donor_id polygon with donor_polygon − cutter_polygon.

    Use to cleanly resolve an overlap: the area claimed by cutter_id is
    carved out of donor_id.  cutter_id is not modified.
    """
    donor_geom = _room_geom(plan, donor_id)
    cutter_geom = _room_geom(plan, cutter_id)
    result_geom = make_valid(donor_geom.difference(cutter_geom))
    return _set_room_from_shapely(plan, donor_id, result_geom)


def clip_room_to_boundary(plan: dict, room_id: str, boundary: list) -> dict:
    """
    Clip room_id polygon to the intersection with boundary [[x, y], ...].

    Use after edge moves that may push a room outside the unit boundary.
    """
    if len(boundary) < 3:
        raise ValueError("boundary must have at least 3 vertices")
    room_geom = _room_geom(plan, room_id)
    bdry_geom = make_valid(SPolygon(boundary))
    result_geom = make_valid(room_geom.intersection(bdry_geom))
    return _set_room_from_shapely(plan, room_id, result_geom)


def fill_residual(plan: dict, room_id: str, unit_polygon: list) -> dict:
    """
    Set room_id polygon = unit_polygon − union(all other room polygons).

    This is the final step that closes 100% coverage.  Always call on the
    common area room (source_type 0 or 4) after all other rooms are placed.

    unit_polygon : [[x, y], ...] in the same coordinate space as plan polygons.
    """
    if len(unit_polygon) < 3:
        raise ValueError("unit_polygon must have at least 3 vertices")

    unit_geom = make_valid(SPolygon(unit_polygon))

    others = []
    for r in plan.get("rooms", []):
        if r["id"] == room_id:
            continue
        verts = r.get("polygon") or []
        if len(verts) >= 3:
            g = make_valid(SPolygon(verts))
            if not g.is_empty:
                others.append(g)

    if others:
        occupied = make_valid(unary_union(others))
        residual = make_valid(unit_geom.difference(occupied))
    else:
        residual = unit_geom

    return _set_room_from_shapely(plan, room_id, residual)
