"""
mcp_fit_server.py — MCP server for agent-driven floor plan fitting.

Exposes tools that let a Claude agent iteratively fit retrieved floor plan
rooms into a floor plate unit polygon using edge editing and boolean ops.

Typical agent workflow
──────────────────────
1. load_floor_plan("207")
2. normalize_unit_to_plan_frame(unit_polygon_mm)
   → keeps boundary_normalized, origin_mm, scale_mm_per_unit
3. render_plan(plan, boundary_normalized)          ← visual inspection
4. describe_plan_geometry(plan)                    ← discover edge indices
5. translate_room / move_room_edge / snap_room_edge_to / align_edges
   (repeat for each room until rooms tile neatly inside boundary_normalized)
6. subtract_room / clip_room_to_boundary           (resolve any overlaps / protrusions)
7. fill_residual(plan, common_room_id, boundary_normalized)   ← 100% coverage
8. run_invariants(plan, boundary_normalized)       ← verify G1–G5
9. rooms_to_mm(plan, origin_mm, scale_mm_per_unit) ← convert back for canvas

All fitting tools operate in normalized [0,1] plan space.
Convert to mm only as the final step.
"""

import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.floorplan_core.fit_transform import (
    normalize_unit_to_plan_frame as _normalize,
    rooms_to_mm as _rooms_to_mm,
    scale_plan_to_boundary as _scale_plan_to_boundary,
)
from packages.floorplan_core.fit_geometry import (
    translate_room as _translate_room,
    move_room_edge as _move_room_edge,
    snap_room_edge_to as _snap_room_edge_to,
    align_edges as _align_edges,
)
from packages.floorplan_core.fit_boolean import (
    subtract_room as _subtract_room,
    clip_room_to_boundary as _clip_room_to_boundary,
    fill_residual as _fill_residual,
)
from packages.floorplan_core.fit_invariants import (
    check_containment as _check_containment,
    check_overlaps as _check_overlaps,
    compute_coverage as _compute_coverage,
    run_invariants as _run_invariants,
)
from packages.floorplan_core.fit_planner import (
    plan_room_expansion as _plan_room_expansion,
    check_plan_compliance as _check_plan_compliance,
    PLAN_MM_PER_UNIT,
)
from packages.floorplan_core.fit_svg import render_plan_png

DATA_DIR = REPO_ROOT / "data" / "fixtures" / "plans" / "test_data"

mcp = FastMCP("floor-plan-fitter")


# ── Inspection ──────────────────────────────────────────────────────────────

@mcp.tool()
def load_floor_plan(plan_id: str) -> dict:
    """
    Load a floor plan JSON by numeric ID (e.g. "207") or absolute file path.
    Returns the complete plan in schema v2.1.0 with normalized [0,1] coordinates.
    Room polygons are under each room's "polygon" key as [[x, y], ...].
    """
    p = Path(plan_id)
    if not p.exists():
        p = DATA_DIR / f"{plan_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"Plan not found: {plan_id}")
    return json.loads(p.read_text())


@mcp.tool()
def describe_plan_geometry(plan: dict) -> dict:
    """
    Summarize plan geometry as a text-readable structure.

    Returns for each room:
      id, source_type, label, area, centroid, vertex_count, edges (with edge_index,
      from/to vertices, length, midpoint).

    Use edge_index values from here when calling move_room_edge,
    snap_room_edge_to, or align_edges.
    Also returns the boundary polygon and current coordinate space.
    """
    from shapely.geometry import Polygon as SPoly
    import math

    rooms_info = []
    for room in plan.get("rooms", []):
        poly = room.get("polygon") or []
        area = 0.0
        cx = cy = 0.0
        edges = []
        n = len(poly)
        if n >= 3:
            try:
                area = float(SPoly(poly).area)
            except Exception:
                pass
            cx = sum(p[0] for p in poly) / n
            cy = sum(p[1] for p in poly) / n
            for ei in range(n):
                a = poly[ei]
                b = poly[(ei + 1) % n]
                length = math.hypot(b[0] - a[0], b[1] - a[1])
                edges.append({
                    "edge_index": ei,
                    "from": [round(a[0], 5), round(a[1], 5)],
                    "to":   [round(b[0], 5), round(b[1], 5)],
                    "length": round(length, 5),
                    "midpoint": [
                        round((a[0] + b[0]) / 2, 5),
                        round((a[1] + b[1]) / 2, 5),
                    ],
                })
        rooms_info.append({
            "id": room["id"],
            "source_type": room.get("source_type", -1),
            "label": room.get("label", ""),
            "area": round(area, 6),
            "centroid": [round(cx, 5), round(cy, 5)],
            "vertex_count": n,
            "edges": edges,
        })

    return {
        "boundary": plan.get("boundary", []),
        "rooms": rooms_info,
        "coordinate_space": plan.get("_coordinate_space", "normalized"),
    }


@mcp.tool()
def describe_unit(unit_polygon_mm: list) -> dict:
    """
    Summarize a unit polygon in mm coordinates.

    Returns area_m2, perimeter_mm, bbox_mm, width_mm, height_mm, aspect_ratio,
    and vertex_count.  Use before normalizing to understand the unit dimensions.
    """
    from shapely.geometry import Polygon as SPoly
    geom = SPoly(unit_polygon_mm)
    b = geom.bounds
    w, h = b[2] - b[0], b[3] - b[1]
    return {
        "area_m2": round(geom.area / 1e6, 3),
        "perimeter_mm": round(geom.exterior.length, 1),
        "bbox_mm": [round(v, 1) for v in b],
        "width_mm": round(w, 1),
        "height_mm": round(h, 1),
        "aspect_ratio": round(max(w, h) / max(min(w, h), 1e-3), 3),
        "vertex_count": len(unit_polygon_mm),
    }


# ── Frame transforms ────────────────────────────────────────────────────────

@mcp.tool()
def normalize_unit_to_plan_frame(unit_polygon_mm: list) -> dict:
    """
    Convert a unit polygon from mm to normalized [0, 1] plan space.

    Returns:
      boundary           [[x, y], ...]  pass this as unit_polygon to all fitting
                                        and validation tools
      origin_mm          [x, y]         save for rooms_to_mm at the end
      scale_mm_per_unit  float          save for rooms_to_mm at the end
      bbox_mm, size_mm                  informational

    Uniform scale (max dimension → 1.0) so room aspect ratios are preserved.
    """
    return _normalize(unit_polygon_mm)


@mcp.tool()
def scale_plan_to_boundary(plan: dict, boundary: list) -> dict:
    """
    Uniformly scale all room polygons so the plan's combined bounding box fits
    inside the boundary's bounding box, centred on the boundary centroid.

    Call this BEFORE normalize_unit_to_plan_frame and fit_rooms_in_boundary so
    rooms start approximately filling the query boundary rather than clustering.

    Scale factor = min(bdry_w/plan_w, bdry_h/plan_h) — rooms are guaranteed to
    fit inside the boundary after scaling, no protrusion.

    Returns the scaled plan.
    """
    return _scale_plan_to_boundary(plan, boundary)


@mcp.tool()
def rooms_to_mm(plan: dict, origin_mm: list, scale_mm_per_unit: float) -> dict:
    """
    Convert all room polygons and boundary in a fitted plan from normalized
    [0, 1] space back to mm coordinates.

    Use the origin_mm and scale_mm_per_unit returned by normalize_unit_to_plan_frame.
    Call this only after run_invariants confirms a clean fit.
    Returns a copy of plan with _coordinate_space set to "mm".
    """
    return _rooms_to_mm(plan, origin_mm, scale_mm_per_unit)


# ── Room placement ──────────────────────────────────────────────────────────

@mcp.tool()
def translate_room(plan: dict, room_id: str, dx: float, dy: float) -> dict:
    """
    Translate all vertices of room room_id by (dx, dy).

    Use for coarse initial positioning.  Prefer move_room_edge for precise
    fitting once the room is roughly in the right position.
    Returns the updated plan.
    """
    return _translate_room(plan, room_id, dx, dy)


# ── Edge editing ────────────────────────────────────────────────────────────

@mcp.tool()
def move_room_edge(plan: dict, room_id: str, edge_index: int, delta: float) -> dict:
    """
    Push or pull edge edge_index of room room_id along its outward normal by delta.

    Positive delta → edge moves outward (room grows).
    Negative delta → edge moves inward (room shrinks).

    The two vertices of the selected edge translate; all other vertices remain
    fixed — like pushing one wall of a room.  Use render_plan or
    describe_plan_geometry to find edge indices.
    Returns the updated plan.
    """
    return _move_room_edge(plan, room_id, edge_index, delta)


@mcp.tool()
def snap_room_edge_to(plan: dict, room_id: str, edge_index: int,
                      coord: float, axis: str) -> dict:
    """
    Snap both vertices of edge edge_index of room room_id to coord on axis.

    axis : "x" or "y"

    Example: snap_room_edge_to(plan, "r_1", 2, 0.35, "x") aligns edge 2 of
    room r_1 to x=0.35.  Use to align rooms flush with the unit boundary or
    with a neighbouring room's edge.
    Returns the updated plan.
    """
    return _snap_room_edge_to(plan, room_id, edge_index, coord, axis)


@mcp.tool()
def align_edges(plan: dict, room_a_id: str, edge_a_index: int,
                room_b_id: str, edge_b_index: int) -> dict:
    """
    Make room_a's edge_a_index coplanar with room_b's edge_b_index.

    Projects the two vertices of edge_a onto the infinite line of edge_b.
    Eliminates gaps and overlaps along a shared room interface in one call.
    room_b is not modified.
    Returns the updated plan.
    """
    return _align_edges(plan, room_a_id, edge_a_index, room_b_id, edge_b_index)


# ── Boolean operations ──────────────────────────────────────────────────────

@mcp.tool()
def subtract_room(plan: dict, donor_id: str, cutter_id: str) -> dict:
    """
    Replace donor_id polygon with donor_polygon − cutter_polygon.

    Use when two rooms overlap and you want to award the disputed area to the
    cutter and trim it from the donor.  cutter_id is not modified.
    Returns the updated plan.
    """
    return _subtract_room(plan, donor_id, cutter_id)


@mcp.tool()
def clip_room_to_boundary(plan: dict, room_id: str, boundary: list) -> dict:
    """
    Clip room_id polygon to boundary [[x, y], ...].

    Use after edge edits that may push a room outside the unit boundary.
    boundary should be the normalized_unit_boundary from
    normalize_unit_to_plan_frame.
    Returns the updated plan.
    """
    return _clip_room_to_boundary(plan, room_id, boundary)


@mcp.tool()
def fill_residual(plan: dict, room_id: str, unit_polygon: list) -> dict:
    """
    Set room_id polygon = unit_polygon − union(all other room polygons).

    This is the final step that achieves 100% coverage.
    Always call on the common area room (source_type 0 or 4, typically "r_0")
    after all other rooms are correctly placed.
    unit_polygon is the normalized boundary from normalize_unit_to_plan_frame.
    Returns the updated plan.
    """
    return _fill_residual(plan, room_id, unit_polygon)


# ── Validation ──────────────────────────────────────────────────────────────

@mcp.tool()
def check_containment(plan: dict, unit_polygon: list) -> dict:
    """
    Check whether each non-common room lies inside unit_polygon.

    Returns dict keyed by room_id:
      inside      bool
      excess_area float  (< 0.0001 is acceptable float tolerance)
      room_type   int
      label       str
    """
    return _check_containment(plan, unit_polygon)


@mcp.tool()
def check_overlaps(plan: dict) -> list:
    """
    Check pairwise overlaps between all non-common rooms.

    Returns list of {room_a, room_b, overlap_area}.
    overlap_area < 0.0001 is float tolerance; anything larger needs fixing.
    """
    return _check_overlaps(plan)


@mcp.tool()
def compute_coverage(plan: dict, unit_polygon: list) -> dict:
    """
    Compute what fraction of unit_polygon is covered by all rooms combined.

    Returns covered_fraction, uncovered_area, common_area, total_area.
    Goal: covered_fraction = 1.0 after fill_residual.
    """
    return _compute_coverage(plan, unit_polygon)


@mcp.tool()
def run_invariants(
    plan: dict,
    unit_polygon: list,
    scale_mm_per_unit: float = None,
) -> dict:
    """
    Run all geometric invariant checks (G1–G6) and return a structured report.

    G1: all non-common rooms inside unit boundary
    G2: no pairwise room overlaps
    G3: all bedrooms touch the unit perimeter
    G4: 100% coverage (checked via uncovered_area)
    G5: common area polygon area ≥ 0
    G6: rooms meet architectural standards (min area, min edge, aspect ratio)
        — only checked when scale_mm_per_unit is provided

    scale_mm_per_unit: mm per normalised unit from normalize_unit_to_plan_frame.
    Omit to skip G6 (G1–G5 only).

    Returns {pass, failures, warnings, coverage, containment, overlap_count, compliance}.
    """
    return _run_invariants(plan, unit_polygon, scale_mm_per_unit=scale_mm_per_unit)


@mcp.tool()
def check_compliance(plan: dict, scale_mm_per_unit: float = None) -> dict:
    """
    Check all rooms against architectural standards (min area, min edge, aspect ratio).

    Uses scale_mm_per_unit from normalize_unit_to_plan_frame to convert normalised
    coordinates to mm.  Omit scale_mm_per_unit to use the default plan scale
    (18 000 mm/unit = 18 m/unit).

    Returns {all_compliant, rooms, violations}.
    Rooms without a standards entry (LivingRoom, DiningRoom, Balcony, etc.) are
    skipped and reported as compliant.
    """
    scale = scale_mm_per_unit if scale_mm_per_unit is not None else PLAN_MM_PER_UNIT
    return _check_plan_compliance(plan, scale)


@mcp.tool()
def plan_room_expansion(
    plan: dict,
    room_id: str,
    unit_polygon: list,
    scale_mm_per_unit: float = None,
) -> dict:
    """
    Evaluate all possible edge pushes to fix a non-compliant room.

    For each interior edge of room_id (edges shared with another room):
      - Computes the delta required to satisfy each failing constraint
        (min area, min edge, aspect ratio)
      - Simulates the target room's polygon and each neighbor's polygon after the move
      - Scores the candidate: +10 × compliance gain, −8 per neighbor made
        non-compliant, −small penalty for move size
    Returns candidates sorted best-first, plus a recommended best_move index.

    Typical agent workflow:
      1. check_compliance(plan, scale_mm_per_unit) → find failing rooms
      2. plan_room_expansion(plan, failing_room_id, unit_polygon, scale_mm_per_unit)
      3. Inspect candidates; apply best with move_room_edge(plan, room_id, edge_index, delta)
      4. clip_room_to_boundary to trim any protrusion
      5. fill_residual to restore 100% coverage
      6. run_invariants to verify

    scale_mm_per_unit: from normalize_unit_to_plan_frame (default: 18 000).
    """
    scale = scale_mm_per_unit if scale_mm_per_unit is not None else PLAN_MM_PER_UNIT
    return _plan_room_expansion(plan, room_id, unit_polygon, scale)


# ── Rendering ────────────────────────────────────────────────────────────────

@mcp.tool()
def render_plan(plan: dict, unit_polygon: list) -> Image:
    """
    Render the current plan state as a colour-coded PNG image.

    Rooms are filled with a colour per source_type.
    Edge index numbers are shown in red at each edge midpoint — use these
    as edge_index values in move_room_edge, snap_room_edge_to, align_edges.
    Room labels (abbreviation + id) are shown at centroids.
    The unit boundary is drawn as a thick dark outline.

    unit_polygon must be in the same coordinate space as plan polygons
    (normalized during fitting; mm after rooms_to_mm).
    """
    png_bytes = render_plan_png(plan, unit_polygon)
    return Image(data=png_bytes, format="png")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
