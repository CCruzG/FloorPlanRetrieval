"""
fit_invariants.py — Geometric invariant checks for fitted floor plans.

Goals checked
─────────────
G1  All non-common rooms contained within the unit polygon
G2  No pairwise room overlaps (non-common rooms)
G3  All bedrooms touch the unit perimeter
G4  100% coverage (checked via uncovered_area)
G5  Common area area ≥ 0 after fill_residual
G6  All rooms with defined standards meet min area, min edge, and aspect ratio

Works in any coordinate space (normalized or mm); unit_polygon and plan
polygons must be in the same space.  G6 requires scale_mm_per_unit to convert
normalised areas/lengths to mm.
"""

from shapely.geometry import Polygon as SPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid


BEDROOM_TYPES  = {1, 5, 7, 8}
COMMON_TYPES   = {0, 4}
TOLERANCE      = 1e-4   # area tolerance for G1/G2 (same units as polygon coords²)
TOUCH_TOLERANCE = 0.005  # distance tolerance for G3


def _room_geom(room: dict) -> SPolygon:
    verts = room.get("polygon") or []
    if len(verts) >= 3:
        return make_valid(SPolygon(verts))
    return SPolygon()


# ── individual checks ──────────────────────────────────────────────────────

def check_containment(plan: dict, unit_polygon: list) -> dict:
    """
    For each non-common room, check whether it lies inside unit_polygon.

    Returns dict keyed by room_id:
        inside      : bool
        excess_area : float  (area outside unit; < TOLERANCE is acceptable)
        room_type   : int
        label       : str
    """
    unit_geom = make_valid(SPolygon(unit_polygon))
    results = {}
    for room in plan.get("rooms", []):
        st = room.get("source_type", -1)
        if st in COMMON_TYPES:
            continue
        geom = _room_geom(room)
        if geom.is_empty:
            results[room["id"]] = {
                "inside": True, "excess_area": 0.0,
                "room_type": st, "label": room.get("label", ""),
            }
            continue
        excess = float(make_valid(geom.difference(unit_geom)).area)
        results[room["id"]] = {
            "inside": excess < TOLERANCE,
            "excess_area": round(excess, 8),
            "room_type": st,
            "label": room.get("label", ""),
        }
    return results


def check_overlaps(plan: dict) -> list:
    """
    Check pairwise overlaps between all non-common rooms.

    Returns list of {room_a, room_b, overlap_area}.
    Pairs with overlap_area < TOLERANCE are floating-point noise.
    """
    non_common = [
        r for r in plan.get("rooms", [])
        if r.get("source_type", -1) not in COMMON_TYPES
    ]
    overlaps = []
    for i, ra in enumerate(non_common):
        ga = _room_geom(ra)
        if ga.is_empty:
            continue
        for rb in non_common[i + 1:]:
            gb = _room_geom(rb)
            if gb.is_empty:
                continue
            area = float(ga.intersection(gb).area)
            if area > 1e-10:
                overlaps.append({
                    "room_a": ra["id"],
                    "room_b": rb["id"],
                    "overlap_area": round(area, 8),
                })
    return overlaps


def compute_coverage(plan: dict, unit_polygon: list) -> dict:
    """
    Compute how much of unit_polygon is covered by all rooms.

    Returns
    -------
    covered_fraction  : float  (1.0 = 100% coverage)
    uncovered_area    : float  (area not assigned to any room)
    common_area       : float  (area of common area rooms)
    total_area        : float  (area of unit_polygon)
    """
    unit_geom = make_valid(SPolygon(unit_polygon))
    total_area = float(unit_geom.area)
    if total_area < 1e-12:
        return {
            "covered_fraction": 0.0, "uncovered_area": 0.0,
            "common_area": 0.0, "total_area": 0.0,
        }

    all_geoms = []
    common_area = 0.0
    for room in plan.get("rooms", []):
        geom = _room_geom(room)
        if not geom.is_empty:
            all_geoms.append(geom)
            if room.get("source_type", -1) in COMMON_TYPES:
                common_area += float(geom.area)

    covered = 0.0
    if all_geoms:
        union = make_valid(unary_union(all_geoms))
        covered = float(union.intersection(unit_geom).area)

    return {
        "covered_fraction": round(covered / total_area, 6),
        "uncovered_area": round(max(0.0, total_area - covered), 8),
        "common_area": round(common_area, 8),
        "total_area": round(total_area, 8),
    }


def check_standards_compliance(
    plan: dict, scale_mm_per_unit: float, standards: dict = None
) -> dict:
    """
    G6: verify rooms with defined standards meet min area, min edge, and aspect ratio.

    Delegates to fit_planner.check_plan_compliance.

    Returns
    -------
    all_compliant  bool
    rooms          dict keyed by room_id → per-room compliance result
    violations     list[str]  human-readable violation descriptions
    """
    from packages.floorplan_core.fit_planner import check_plan_compliance
    return check_plan_compliance(plan, scale_mm_per_unit, standards)


def run_invariants(
    plan: dict,
    unit_polygon: list,
    scale_mm_per_unit: float = None,
    standards: dict = None,
) -> dict:
    """
    Run all invariant checks and return a structured report.

    Parameters
    ----------
    plan               fitted floor plan dict
    unit_polygon       target boundary [[x, y], ...]
    scale_mm_per_unit  mm per normalised unit; enables G6 when provided
    standards          optional room standards override for G6

    Returns
    -------
    pass          bool
    failures      list[str]   hard failures — must fix
    warnings      list[str]   soft issues — worth noting
    coverage      dict        from compute_coverage
    containment   dict        from check_containment
    overlap_count int
    compliance    dict        from check_standards_compliance; empty if no scale given
    """
    unit_geom = make_valid(SPolygon(unit_polygon))
    failures: list = []
    warnings: list = []

    # G1: containment
    containment = check_containment(plan, unit_polygon)
    for rid, info in containment.items():
        if not info["inside"]:
            failures.append(
                f"G1: {rid} ({info['label']}) protrudes boundary "
                f"(excess={info['excess_area']:.6f})"
            )

    # G2: overlaps
    overlaps = check_overlaps(plan)
    for o in overlaps:
        msg = f"G2: {o['room_a']} ∩ {o['room_b']} overlap={o['overlap_area']:.6f}"
        if o["overlap_area"] > TOLERANCE:
            failures.append(msg)
        else:
            warnings.append(f"{msg} (float noise)")

    # G3: bedrooms touch boundary
    bdry_ext = unit_geom.exterior
    for room in plan.get("rooms", []):
        if room.get("source_type", -1) in BEDROOM_TYPES:
            geom = _room_geom(room)
            if not geom.is_empty:
                dist = float(bdry_ext.distance(geom))
                if dist > TOUCH_TOLERANCE:
                    failures.append(
                        f"G3: bedroom {room['id']} not touching unit boundary "
                        f"(dist={dist:.5f})"
                    )

    # G4: coverage
    coverage = compute_coverage(plan, unit_polygon)
    if coverage["uncovered_area"] > TOLERANCE:
        warnings.append(
            f"G4: {coverage['uncovered_area']:.6f} units² uncovered "
            f"({(1 - coverage['covered_fraction']) * 100:.2f}%)"
        )

    # G5: common area non-negative
    if coverage["common_area"] < 0:
        failures.append("G5: common area polygon has negative area")

    # G6: architectural standards compliance (requires scale_mm_per_unit)
    compliance = {}
    if scale_mm_per_unit is not None:
        compliance = check_standards_compliance(plan, scale_mm_per_unit, standards)
        for violation in compliance.get("violations", []):
            failures.append(f"G6: {violation}")

    return {
        "pass":         len(failures) == 0,
        "failures":     failures,
        "warnings":     warnings,
        "coverage":     coverage,
        "containment":  containment,
        "overlap_count": len(overlaps),
        "compliance":   compliance,
    }
