"""
fit_geometry.py — Room polygon editing: translation, edge movement, snapping.

All operations are coordinate-space agnostic (work in both normalized and mm).
Each function returns a deep copy of the plan with the modification applied.

Edge indices refer to the edges of a room's polygon list:
  edge 0 = vertex[0] → vertex[1]
  edge i = vertex[i] → vertex[(i+1) % n]

Use describe_plan_geometry or render_plan to discover edge indices visually.
"""

import copy
import math


# ── internal helpers ───────────────────────────────────────────────────────

_INSET = 0.005 / 2  # matches fit_plan.py EPSILON / 2


def _get_room(plan: dict, room_id: str) -> dict:
    for r in plan.get("rooms", []):
        if r["id"] == room_id:
            return r
    raise KeyError(f"Room {room_id!r} not found in plan")


def _update_room_polygon(plan: dict, room_id: str, polygon: list) -> dict:
    result = copy.deepcopy(plan)
    for r in result.get("rooms", []):
        if r["id"] == room_id:
            r["polygon"] = polygon
            if len(polygon) >= 3:
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                r["bbox"] = [x1, y1, x2, y2]
                r["bbox_interior"] = [
                    x1 + _INSET, y1 + _INSET,
                    x2 - _INSET, y2 - _INSET,
                ]
            return result
    raise KeyError(f"Room {room_id!r} not found in plan")


def _poly_centroid(poly: list) -> tuple:
    n = len(poly)
    return sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n


def edge_outward_normal(poly: list, edge_index: int) -> tuple:
    """
    Return the unit outward normal of edge edge_index of polygon poly.
    Outward = pointing away from the polygon centroid.
    """
    n = len(poly)
    a = poly[edge_index]
    b = poly[(edge_index + 1) % n]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return 0.0, 0.0
    # Candidate perpendicular
    nx, ny = -dy / length, dx / length
    # Flip if pointing inward (toward centroid)
    cx, cy = _poly_centroid(poly)
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    if nx * (mx - cx) + ny * (my - cy) < 0:
        nx, ny = -nx, -ny
    return nx, ny


# ── public API ─────────────────────────────────────────────────────────────

def translate_room(plan: dict, room_id: str, dx: float, dy: float) -> dict:
    """
    Translate all vertices of room room_id by (dx, dy).
    Use for coarse initial positioning before edge editing.
    """
    room = _get_room(plan, room_id)
    poly = room.get("polygon") or []
    if not poly:
        return copy.deepcopy(plan)
    new_poly = [[p[0] + dx, p[1] + dy] for p in poly]
    return _update_room_polygon(plan, room_id, new_poly)


def move_room_edge(plan: dict, room_id: str, edge_index: int, delta: float) -> dict:
    """
    Move edge edge_index of room room_id along its outward normal by delta.

    Positive delta grows the room (pushes edge outward).
    Negative delta shrinks the room (pulls edge inward).

    The two vertices of the selected edge translate; all other vertices remain
    fixed.  Adjacent edges stretch to follow (like pulling a wall of a room).
    """
    room = _get_room(plan, room_id)
    poly = [list(p) for p in (room.get("polygon") or [])]
    n = len(poly)
    if n < 3:
        raise ValueError(f"Room {room_id!r} has fewer than 3 vertices")
    if not (0 <= edge_index < n):
        raise ValueError(f"edge_index {edge_index} out of range for {n}-vertex polygon")

    nx, ny = edge_outward_normal(poly, edge_index)
    i = edge_index
    j = (edge_index + 1) % n
    poly[i][0] += nx * delta
    poly[i][1] += ny * delta
    poly[j][0] += nx * delta
    poly[j][1] += ny * delta

    return _update_room_polygon(plan, room_id, poly)


def snap_room_edge_to(plan: dict, room_id: str, edge_index: int,
                      coord: float, axis: str) -> dict:
    """
    Snap both vertices of edge edge_index to coord on the given axis.

    axis : "x" to set x-coordinate, "y" to set y-coordinate.

    Useful for aligning a room edge flush with the unit boundary or with
    another room's edge.
    """
    axis = axis.lower()
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")

    room = _get_room(plan, room_id)
    poly = [list(p) for p in (room.get("polygon") or [])]
    n = len(poly)
    if n < 3:
        raise ValueError(f"Room {room_id!r} has fewer than 3 vertices")
    if not (0 <= edge_index < n):
        raise ValueError(f"edge_index {edge_index} out of range for {n}-vertex polygon")

    axis_idx = 0 if axis == "x" else 1
    i = edge_index
    j = (edge_index + 1) % n
    poly[i][axis_idx] = coord
    poly[j][axis_idx] = coord

    return _update_room_polygon(plan, room_id, poly)


def align_edges(plan: dict, room_a_id: str, edge_a_index: int,
                room_b_id: str, edge_b_index: int) -> dict:
    """
    Project room_a's edge_a_index onto the line of room_b's edge_b_index.

    Both vertices of edge_a are moved to lie on the same infinite line as
    edge_b.  This eliminates gaps and overlaps along a shared room interface.
    room_b is not modified.
    """
    room_a = _get_room(plan, room_a_id)
    room_b = _get_room(plan, room_b_id)
    poly_a = [list(p) for p in (room_a.get("polygon") or [])]
    poly_b = room_b.get("polygon") or []
    n_a, n_b = len(poly_a), len(poly_b)

    if n_a < 3 or n_b < 3:
        raise ValueError("Both rooms must have at least 3 vertices")
    if not (0 <= edge_a_index < n_a):
        raise ValueError(f"edge_a_index {edge_a_index} out of range")
    if not (0 <= edge_b_index < n_b):
        raise ValueError(f"edge_b_index {edge_b_index} out of range")

    # Normal of edge_b's line (unit vector perpendicular to the edge)
    b0 = poly_b[edge_b_index]
    b1 = poly_b[(edge_b_index + 1) % n_b]
    dx, dy = b1[0] - b0[0], b1[1] - b0[1]
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return copy.deepcopy(plan)

    nx, ny = -dy / length, dx / length

    # Constant for the line equation: nx*x + ny*y = c
    c = nx * b0[0] + ny * b0[1]

    # Project each vertex of edge_a onto edge_b's line
    i = edge_a_index
    j = (edge_a_index + 1) % n_a
    for idx in (i, j):
        d = nx * poly_a[idx][0] + ny * poly_a[idx][1] - c
        poly_a[idx][0] -= nx * d
        poly_a[idx][1] -= ny * d

    return _update_room_polygon(plan, room_a_id, poly_a)
