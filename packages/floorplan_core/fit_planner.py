"""
fit_planner.py — Room standards compliance checking and edge-expansion planning.

check_room_compliance  : single room vs its type's architectural standards
check_plan_compliance  : all rooms in a plan
plan_room_expansion    : for a non-compliant room, evaluate every possible edge
                         push and its consequences on adjacent rooms, return
                         ranked candidates
apply_best_expansion   : apply the top-scoring candidate and return the updated plan

All functions operate in normalized [0, 1] plan space.
PLAN_MM_PER_UNIT is the fixed scale (18 000 mm/unit) for the retrieval coordinate
system.  Pass scale_mm_per_unit explicitly when working in a unit-normalised
frame produced by normalize_unit_to_plan_frame.
"""

import json
import math
from pathlib import Path

from shapely.geometry import LineString, Polygon as SPolygon
from shapely.validation import make_valid

from packages.floorplan_core.fit_geometry import (
    move_room_edge as _move_room_edge,
    edge_outward_normal,
)

# ── constants ──────────────────────────────────────────────────────────────────

PLAN_MM_PER_UNIT = 18_000   # 1 normalised plan unit = 18 m

_STANDARDS_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "room_standards.json"
)
_standards_cache: dict | None = None


# ── standards loading ──────────────────────────────────────────────────────────

def load_standards() -> dict:
    """Return room standards dict keyed by str(source_type_id) (cached)."""
    global _standards_cache
    if _standards_cache is None:
        _standards_cache = json.loads(_STANDARDS_PATH.read_text())["by_type_id"]
    return _standards_cache


# ── single-room compliance ─────────────────────────────────────────────────────

def _room_bbox_metrics(poly: list, scale_mm_per_unit: float) -> dict | None:
    """Compute area_m2, w_mm, h_mm, short_mm, elongation from a polygon."""
    if len(poly) < 3:
        return None
    geom = make_valid(SPolygon(poly))
    if geom.is_empty:
        return None
    minx, miny, maxx, maxy = geom.bounds
    w_mm = (maxx - minx) * scale_mm_per_unit
    h_mm = (maxy - miny) * scale_mm_per_unit
    short_mm = min(w_mm, h_mm)
    long_mm  = max(w_mm, h_mm)
    return {
        "area_m2":    geom.area * (scale_mm_per_unit / 1000) ** 2,
        "area_norm":  geom.area,
        "w_mm": w_mm, "h_mm": h_mm,
        "short_mm":   short_mm,
        "long_mm":    long_mm,
        "elongation": long_mm / max(short_mm, 1e-3),
    }


def check_room_compliance(
    room: dict,
    scale_mm_per_unit: float = PLAN_MM_PER_UNIT,
    standards: dict = None,
) -> dict:
    """
    Check a single room against its type's architectural standards.

    Returns
    -------
    has_standards  bool    False → this room type has no entry in room_standards.json
    compliant      bool
    area_m2        float
    area_ok        bool
    short_edge_mm  float   shorter bbox dimension
    edge_ok        bool
    elongation     float   long / short
    aspect_ok      bool    elongation ≤ max_elongation
    standards      dict    the applicable standard entry (if any)
    """
    if standards is None:
        standards = load_standards()
    st  = room.get("source_type", -1)
    std = standards.get(str(st))
    if std is None:
        return {"has_standards": False, "compliant": True}

    m = _room_bbox_metrics(room.get("polygon", []), scale_mm_per_unit)
    if m is None:
        return {
            "has_standards": True, "room_type": st, "compliant": False,
            "reason": "no valid polygon",
            "area_m2": 0.0, "area_ok": False,
            "short_edge_mm": 0.0, "edge_ok": False,
            "elongation": 0.0, "aspect_ok": False,
            "standards": std,
        }

    # 1mm / 0.01m² tolerance absorbs float rounding from coordinate transforms
    area_ok   = m["area_m2"]   >= std["min_area_m2"]   - 0.01
    edge_ok   = m["short_mm"]  >= std["min_edge_mm"]   - 1.0
    aspect_ok = m["elongation"] <= std["max_elongation"] + 0.01

    return {
        "has_standards":  True,
        "room_type":      st,
        "compliant":      area_ok and edge_ok and aspect_ok,
        "area_m2":        round(m["area_m2"], 3),
        "area_ok":        area_ok,
        "short_edge_mm":  round(m["short_mm"], 1),
        "edge_ok":        edge_ok,
        "elongation":     round(m["elongation"], 3),
        "aspect_ok":      aspect_ok,
        "standards":      std,
    }


# ── plan-level compliance ──────────────────────────────────────────────────────

def check_plan_compliance(
    plan: dict,
    scale_mm_per_unit: float = PLAN_MM_PER_UNIT,
    standards: dict = None,
) -> dict:
    """
    Run compliance checks for all rooms in the plan.

    Returns
    -------
    all_compliant  bool
    rooms          dict keyed by room_id → check_room_compliance result
    violations     list[str]  human-readable violation descriptions
    """
    if standards is None:
        standards = load_standards()
    rooms_out  = {}
    violations = []
    for room in plan.get("rooms", []):
        result = check_room_compliance(room, scale_mm_per_unit, standards)
        rooms_out[room["id"]] = result
        if result.get("has_standards") and not result.get("compliant"):
            parts = []
            if not result.get("area_ok"):
                parts.append(
                    f"area {result['area_m2']}m² < {result['standards']['min_area_m2']}m²"
                )
            if not result.get("edge_ok"):
                parts.append(
                    f"short edge {result['short_edge_mm']}mm < {result['standards']['min_edge_mm']}mm"
                )
            if not result.get("aspect_ok"):
                parts.append(
                    f"elongation {result['elongation']} > {result['standards']['max_elongation']}"
                )
            violations.append(
                f"{room['id']} ({result['standards']['name']}): {', '.join(parts)}"
            )
    return {
        "all_compliant": len(violations) == 0,
        "rooms":         rooms_out,
        "violations":    violations,
    }


# ── internal geometry helpers ──────────────────────────────────────────────────

def _get_room(plan: dict, room_id: str) -> dict:
    for r in plan.get("rooms", []):
        if r["id"] == room_id:
            return r
    raise KeyError(f"Room {room_id!r} not found in plan")


def _edge_axis_dir(poly: list, ei: int) -> str:
    """Classify edge ei's outward normal as 'x+', 'x-', 'y+', or 'y-'."""
    nx, ny = edge_outward_normal(poly, ei)
    if abs(nx) >= abs(ny):
        return "x+" if nx >= 0 else "x-"
    return "y+" if ny >= 0 else "y-"


def _find_neighbors_for_edge(plan: dict, room_id: str, edge_idx: int) -> list:
    """
    Return rooms sharing the given edge of room_id.

    Each entry: {room, shared_edge_idx, shared_length}.
    shared_edge_idx is the edge in the neighbor that faces the target edge;
    pushing the target edge outward by delta should move this neighbor edge
    inward by delta (use move_room_edge with -delta).
    """
    target = _get_room(plan, room_id)
    poly = target["polygon"]
    n = len(poly)
    a, b  = poly[edge_idx], poly[(edge_idx + 1) % n]
    edge_line = LineString([a, b])
    mid_x = (a[0] + b[0]) / 2
    mid_y = (a[1] + b[1]) / 2

    neighbors = []
    for other in plan.get("rooms", []):
        if other["id"] == room_id:
            continue
        other_poly = other.get("polygon", [])
        m = len(other_poly)
        if m < 3:
            continue
        shared = edge_line.intersection(make_valid(SPolygon(other_poly)).boundary)
        if shared.is_empty or shared.length < 1e-5:
            continue
        # Closest edge midpoint in the neighbor polygon
        best_j, best_d = 0, float("inf")
        for j in range(m):
            ea, eb = other_poly[j], other_poly[(j + 1) % m]
            mx, my = (ea[0] + eb[0]) / 2, (ea[1] + eb[1]) / 2
            d = math.hypot(mx - mid_x, my - mid_y)
            if d < best_d:
                best_d, best_j = d, j
        neighbors.append({
            "room":            other,
            "shared_edge_idx": best_j,
            "shared_length":   round(shared.length, 6),
        })
    return neighbors


# ── per-edge delta computation ─────────────────────────────────────────────────

def _deltas_for_axis(
    room: dict, scale_mm_per_unit: float, std: dict, axis_dir: str
) -> list:
    """
    Return [(objective_label, delta_normalized)] for constraints that pushing
    in axis_dir would help satisfy.  Only failing constraints are included.
    """
    m = _room_bbox_metrics(room.get("polygon", []), scale_mm_per_unit)
    if m is None:
        return []
    is_x     = axis_dir in ("x+", "x-")
    edge_len = m["h_mm"] / scale_mm_per_unit if is_x else m["w_mm"] / scale_mm_per_unit
    results  = []

    # Area: pushing any edge by delta grows area ≈ edge_length * delta
    if m["area_m2"] < std["min_area_m2"]:
        target_area_norm = std["min_area_m2"] / (scale_mm_per_unit / 1000) ** 2
        d = (target_area_norm - m["area_norm"]) / max(edge_len, 1e-8)
        results.append(("area", d))

    # Min edge: only for edges perpendicular to the short dimension
    short_is_x = m["w_mm"] <= m["h_mm"]
    if m["short_mm"] < std["min_edge_mm"] and is_x == short_is_x:
        d = (std["min_edge_mm"] - m["short_mm"]) / scale_mm_per_unit
        results.append(("min_edge", d))

    # Aspect ratio: push short dimension
    if m["elongation"] > std["max_elongation"] and is_x == short_is_x:
        target_short_norm = (m["long_mm"] / std["max_elongation"]) / scale_mm_per_unit
        current_short_norm = m["w_mm"] / scale_mm_per_unit if short_is_x else m["h_mm"] / scale_mm_per_unit
        d = target_short_norm - current_short_norm
        if d > 1e-6:
            results.append(("aspect_ratio", d))

    return [(obj, d) for obj, d in results if 1e-6 < d < 0.5]


# ── compliance score helper ────────────────────────────────────────────────────

def _compliance_score(status: dict) -> float:
    """Fraction of the three constraints that pass (0.0–1.0)."""
    if not status.get("has_standards"):
        return 1.0
    return sum(status.get(k, True) for k in ("area_ok", "edge_ok", "aspect_ok")) / 3.0


# ── main planner ───────────────────────────────────────────────────────────────

def plan_room_expansion(
    plan: dict,
    room_id: str,
    boundary: list,
    scale_mm_per_unit: float = PLAN_MM_PER_UNIT,
    standards: dict = None,
) -> dict:
    """
    Evaluate all possible edge pushes to fix a non-compliant room.

    For each interior edge (edges with at least one neighbor room):
      - Compute the delta required to satisfy each failing objective via that edge
      - Simulate the target room's polygon after the move
      - Simulate each neighbor room's polygon after the counter-move (−delta)
      - Score: +10 × compliance gain; −8 per neighbor made non-compliant;
               −small penalty proportional to move size
      - Return candidates sorted by score, best first

    Parameters
    ----------
    plan              floor plan dict (normalised [0, 1] plan space)
    room_id           id of the room to fix
    boundary          unit boundary polygon (same coordinate space, informational)
    scale_mm_per_unit mm per normalised unit (default PLAN_MM_PER_UNIT = 18 000)
    standards         optional override; loaded from configs if None

    Returns
    -------
    current_status   compliance result before any move
    candidates       list of dicts, best first:
      edge_index       int
      delta            float  (normalised units)
      delta_mm         float
      objective        str
      axis_direction   str  (x+/x-/y+/y-)
      target_after     compliance dict after move
      neighbors        list of consequence dicts
      score            float
      recommendation   str
    best_move        0 if any candidates, else None
    message          str
    """
    if standards is None:
        standards = load_standards()

    target_room = _get_room(plan, room_id)
    st  = target_room.get("source_type", -1)
    std = standards.get(str(st))

    current_status = check_room_compliance(target_room, scale_mm_per_unit, standards)
    if not current_status.get("has_standards") or current_status.get("compliant"):
        return {
            "current_status": current_status,
            "candidates":     [],
            "best_move":      None,
            "message":        "Room is already compliant or has no standards",
        }

    poly = target_room.get("polygon", [])
    n    = len(poly)
    current_score = _compliance_score(current_status)
    candidates = []

    for ei in range(n):
        axis_dir = _edge_axis_dir(poly, ei)
        neighbors_info = _find_neighbors_for_edge(plan, room_id, ei)
        if not neighbors_info:
            # Boundary-facing edge — any expansion would be clipped away
            continue

        for obj, delta in _deltas_for_axis(target_room, scale_mm_per_unit, std, axis_dir):
            # Simulate target edge push
            try:
                sim_plan    = _move_room_edge(plan, room_id, ei, delta)
                new_target  = _get_room(sim_plan, room_id)
                target_after = check_room_compliance(new_target, scale_mm_per_unit, standards)
            except Exception:
                continue

            # Simulate each neighbor contraction
            neighbor_results = []
            for nb_info in neighbors_info:
                nb     = nb_info["room"]
                nb_id  = nb["id"]
                nb_ei  = nb_info["shared_edge_idx"]
                nb_before = check_room_compliance(nb, scale_mm_per_unit, standards)
                try:
                    nb_sim  = _move_room_edge(plan, nb_id, nb_ei, -delta)
                    new_nb  = _get_room(nb_sim, nb_id)
                    nb_after = check_room_compliance(new_nb, scale_mm_per_unit, standards)
                except Exception:
                    continue
                nb_geom_b = make_valid(SPolygon(nb.get("polygon", [])))
                nb_geom_a = make_valid(SPolygon(new_nb.get("polygon", [])))
                delta_area_pct = (
                    (nb_geom_a.area - nb_geom_b.area) / nb_geom_b.area * 100
                    if nb_geom_b.area > 0 else 0.0
                )
                neighbor_results.append({
                    "room_id":         nb_id,
                    "label":           nb.get("label", ""),
                    "room_type":       nb.get("source_type", -1),
                    "compliant_before": nb_before.get("compliant", True),
                    "compliant_after":  nb_after.get("compliant", True),
                    "delta_area_pct":   round(delta_area_pct, 1),
                    "area_m2_before":   nb_before.get("area_m2"),
                    "area_m2_after":    nb_after.get("area_m2"),
                })

            gain = _compliance_score(target_after) - current_score
            cost = sum(
                1 for nb in neighbor_results
                if nb["compliant_before"] and not nb["compliant_after"]
            )
            score = gain * 10 - cost * 8 - delta * 10

            # Human-readable recommendation
            rec_parts = []
            if target_after.get("compliant"):
                rec_parts.append("target becomes compliant")
            else:
                fixed = [
                    k.replace("_ok", "")
                    for k in ("area_ok", "edge_ok", "aspect_ok")
                    if not current_status.get(k) and target_after.get(k)
                ]
                rec_parts.append(f"fixes {', '.join(fixed)}" if fixed else "partial improvement")
            if cost > 0:
                failing_ids = [
                    nb["room_id"] for nb in neighbor_results
                    if nb["compliant_before"] and not nb["compliant_after"]
                ]
                rec_parts.append(
                    f"causes {cost} neighbor(s) to fail: {', '.join(failing_ids)}"
                )

            candidates.append({
                "edge_index":     ei,
                "delta":          round(delta, 6),
                "delta_mm":       round(delta * scale_mm_per_unit, 1),
                "objective":      obj,
                "axis_direction": axis_dir,
                "target_after":   target_after,
                "neighbors":      neighbor_results,
                "score":          round(score, 3),
                "recommendation": "; ".join(rec_parts),
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return {
        "current_status": current_status,
        "candidates":     candidates,
        "best_move":      0 if candidates else None,
        "message":        f"{len(candidates)} candidate moves evaluated",
    }


# ── auto-apply ─────────────────────────────────────────────────────────────────

def apply_best_expansion(
    plan: dict,
    room_id: str,
    boundary: list,
    scale_mm_per_unit: float = PLAN_MM_PER_UNIT,
    standards: dict = None,
) -> tuple:
    """
    Run plan_room_expansion and apply the top-scoring candidate (if score > 0).

    Applies both the target-room edge push AND the counter-move (−delta) to
    every neighbor sharing that edge, so no overlap is introduced.

    Returns (updated_plan, applied_move_dict | None).
    updated_plan is the original plan unchanged when no move is applied.
    """
    result = plan_room_expansion(plan, room_id, boundary, scale_mm_per_unit, standards)
    candidates = result.get("candidates", [])
    if not candidates or candidates[0]["score"] <= 0:
        return plan, None
    best = candidates[0]

    updated_plan = _move_room_edge(plan, room_id, best["edge_index"], best["delta"])

    # Push each neighbor's shared edge inward by the same delta so the physical
    # wall moves consistently and no new overlap is created.
    for nb_info in _find_neighbors_for_edge(plan, room_id, best["edge_index"]):
        nb_id = nb_info["room"]["id"]
        nb_ei = nb_info["shared_edge_idx"]
        try:
            updated_plan = _move_room_edge(updated_plan, nb_id, nb_ei, -best["delta"])
        except Exception:
            pass

    return updated_plan, best
