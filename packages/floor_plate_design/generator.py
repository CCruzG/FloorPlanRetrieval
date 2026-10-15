"""
generate_initial_layout — Main entry point for floor plate generation.

Takes a boundary polygon and unit mix, returns a complete FloorPlateState
with all elements placed (cores, corridors, units), typed edges, and
compliance snapshot.

Stages (10 total):
  1. Boundary analysis
  2. Typology inference
  3. Core specification
  4. Core placement
  5. Corridor derivation
  6. Apartment zone computation
  7. Unit bay subdivision
  8. Typed edge assignment
  9. (Structural grid generation — skipped for initial implementation)
  10. Compliance snapshot
"""

from typing import Optional
import random
from math import atan2, degrees, pi, ceil
from shapely import ops as shapely_ops
from shapely.geometry import LineString as ShapelyLineString
from .types import (
    FloorPlateState,
    Typology,
    TypologyType,
    SlabSubType,
    CoreType,
    ConfidenceLevel,
    Core,
    Corridor,
    Unit,
    TypedEdge,
    EdgeType,
    CORE_DIMENSIONS,
)
from .geometry import (
    polygon_area,
    polygon_centroid,
    polygon_bbox,
    polygon_aspect_ratio,
    normalize_polygon_winding,
    point_distance,
    polygon_to_shapely,
    shapely_to_polygon,
    split_polygon,
    Polygon,
)
from .compliance import compute_compliance_snapshot


MIN_UNIT_AREA_M2_BY_TYPE = {
    "studio": 25.0,
    "0bed": 25.0,
    "1bed": 36.0,
    "2bed": 50.0,
    "3bed": 65.0,
}

DEFAULT_TARGET_AREA_M2_BY_TYPE = {
    "studio": 35.0,
    "0bed": 35.0,
    "1bed": 55.0,
    "2bed": 70.0,
    "3bed": 90.0,
}

DEFAULT_SCORING = {
    "placed_unit_reward": 10.0,
    "type_priority": {},
    "shortfall_penalty": 8.0,
    "residual_area_penalty": 2.0,
    "residual_fragmentation_penalty": 3.0,
    "area_deviation_penalty": 1.0,
    "compactness_penalty": 0.5,
}


# ── Stage 1: Boundary Analysis ────────────────────────────────────────────────

def analyze_boundary(boundary: Polygon) -> dict:
    """
    Stage 1: Analyze boundary geometry.
    
    Returns:
        Dict with keys: area, centroid, bbox, long_dimension, short_dimension,
                        aspect_ratio, long_axis, long_edges, short_edges
    """
    area = polygon_area(boundary)
    centroid = polygon_centroid(boundary)
    bbox = polygon_bbox(boundary)
    
    width = bbox["width"]
    height = bbox["height"]
    
    if width >= height:
        long_dim = width
        short_dim = height
        long_axis = "horizontal"
    else:
        long_dim = height
        short_dim = width
        long_axis = "vertical"
    
    aspect = long_dim / short_dim if short_dim > 0 else 1.0
    
    # Classify edges by orientation
    long_edges = []
    short_edges = []
    
    for i in range(len(boundary)):
        p1 = boundary[i]
        p2 = boundary[(i + 1) % len(boundary)]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Angle from horizontal (degrees)
        angle = degrees(atan2(dy, dx))
        # Normalize to [-90, 90]
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        
        # Within ±15° of long axis?
        angle_abs = abs(angle)
        if long_axis == "horizontal":
            if angle_abs <= 15 or angle_abs >= 165:
                long_edges.append((p1, p2))
            else:
                short_edges.append((p1, p2))
        else:  # vertical
            if angle_abs >= 75:  # ±15° from vertical = 75°-105° from horizontal
                long_edges.append((p1, p2))
            else:
                short_edges.append((p1, p2))
    
    decomposition: list[Polygon] = [boundary]
    try:
        decomposition = split_polygon(boundary)
    except Exception:
        decomposition = [boundary]

    is_rectangular = len(boundary) == 4 and len(decomposition) == 1

    return {
        "boundary": boundary,
        "area": area,
        "centroid": centroid,
        "bbox": bbox,
        "long_dimension": long_dim,
        "short_dimension": short_dim,
        "aspect_ratio": aspect,
        "long_axis": long_axis,
        "long_edges": long_edges,
        "short_edges": short_edges,
        "decomposition": decomposition,
        "decomposition_count": len(decomposition),
        "is_rectangular": is_rectangular,
    }


# ── Stage 2: Typology Inference ───────────────────────────────────────────────

def _min_area_for_unit_type(unit_type: str) -> float:
    key = str(unit_type or "").strip().lower()
    return MIN_UNIT_AREA_M2_BY_TYPE.get(key, 55.0)


def _estimate_accommodable_unit_count(analysis: dict, unit_mix: Optional[list[dict]]) -> int:
    """Estimate capacity using a simple area-only check against minimum unit area per type."""
    area_m2 = analysis["area"] / 1e6
    requested = unit_mix or []
    total_count = sum(max(0, int(item.get("count", 0))) for item in requested)

    weighted_min_area = 0.0
    for item in requested:
        count = max(0, int(item.get("count", 0)))
        weighted_min_area += count * _min_area_for_unit_type(item.get("type", ""))

    if total_count > 0:
        avg_min_area = max(20.0, weighted_min_area / total_count)
    else:
        avg_min_area = 36.0

    by_area = int(area_m2 // avg_min_area)

    if total_count > 0:
        return min(by_area, total_count)
    return by_area


def compute_unit_distribution_breakdown(
    analysis: dict,
    unit_mix: Optional[list[dict]],
    min_unit_depth_mm: float,
) -> dict:
    """Derive distribution strategy before typology/core/corridor placement."""
    accommodated_units = _estimate_accommodable_unit_count(analysis, unit_mix)
    short_dim = analysis["short_dimension"]
    long_axis = analysis["long_axis"]
    rectangular = analysis.get("is_rectangular", False)

    if accommodated_units <= 2:
        return {
            "accommodated_units": accommodated_units,
            "scenario": "two_unit_point_tower",
            "recommended_typology": TypologyType.POINT_BLOCK.value,
            "recommended_sub_type": "central_core",
            "split_strategy": "split_in_half_along_long_axis",
            "rectangular": rectangular,
            "notes": "Only two units fit by area minimum; use compact point tower logic.",
        }

    if rectangular:
        if short_dim > min_unit_depth_mm and short_dim < (2 * min_unit_depth_mm):
            return {
                "accommodated_units": accommodated_units,
                "scenario": "rectangular_single_loaded",
                "recommended_typology": TypologyType.SLAB.value,
                "recommended_sub_type": SlabSubType.SINGLE_LOADED.value,
                "split_strategy": "split_along_long_side_units_span_long_to_long",
                "rectangular": True,
                "notes": "Width supports one unit band only.",
            }
        if short_dim >= (2 * min_unit_depth_mm):
            return {
                "accommodated_units": accommodated_units,
                "scenario": "rectangular_double_loaded",
                "recommended_typology": TypologyType.SLAB.value,
                "recommended_sub_type": SlabSubType.DOUBLE_LOADED.value,
                "split_strategy": "split_short_side_then_subdivide_long_side",
                "rectangular": True,
                "notes": "Width supports two unit bands around circulation.",
            }

        return {
            "accommodated_units": accommodated_units,
            "scenario": "rectangular_narrow",
            "recommended_typology": TypologyType.POINT_BLOCK.value,
            "recommended_sub_type": "central_core",
            "split_strategy": "core_compact_landing",
            "rectangular": True,
            "notes": "Width below minimum unit depth threshold; prefer compact core strategy.",
        }

    section_assessment = []
    for idx, section in enumerate(analysis.get("decomposition", [])):
        section_bbox = polygon_bbox(section)
        width = min(section_bbox["width"], section_bbox["height"])
        if width >= (2 * min_unit_depth_mm):
            section_strategy = "double_loaded"
        elif width > min_unit_depth_mm:
            section_strategy = "single_loaded"
        else:
            section_strategy = "compact_core_edge"
        section_assessment.append({
            "section_id": idx,
            "width": width,
            "strategy": section_strategy,
        })

    return {
        "accommodated_units": accommodated_units,
        "scenario": "non_rectangular_sectioned",
        "recommended_typology": TypologyType.SLAB.value,
        "recommended_sub_type": SlabSubType.SINGLE_LOADED.value,
        "split_strategy": "decompose_and_assess_sections",
        "rectangular": False,
        "section_assessment": section_assessment,
        "notes": "Non-rectangular plate decomposed into sections for local strategy assessment.",
    }


def _expand_unit_mix(unit_mix: list[dict]) -> list[dict]:
    expanded = []
    for item in unit_mix or []:
        count = max(0, int(item.get("count", 0)))
        for _ in range(count):
            expanded.append({
                "type": item.get("type", "unit"),
                "target_area": float(item.get("target_area", 55) or 55),
            })
    return expanded


def _partition_polygon_by_weights(polygon: Polygon, weights: list[float]) -> list[Polygon]:
    if not weights:
        return []
    if len(weights) == 1:
        return [polygon]

    bbox = polygon_bbox(polygon)
    width = bbox["width"]
    height = bbox["height"]
    if width <= 0 or height <= 0:
        return [polygon for _ in weights]

    total_weight = sum(max(0.0, float(weight)) for weight in weights)
    if total_weight <= 0:
        total_weight = float(len(weights))
        weights = [1.0 for _ in weights]

    split_index = max(1, len(weights) // 2)
    left_weights = weights[:split_index]
    right_weights = weights[split_index:]
    left_weight = sum(left_weights)
    ratio = left_weight / total_weight

    split_vertical = width >= height
    if split_vertical:
        cut = bbox["min_x"] + width * ratio
        split_line = ShapelyLineString([
            (cut, bbox["min_y"] - max(width, height) - 1000),
            (cut, bbox["max_y"] + max(width, height) + 1000),
        ])
    else:
        cut = bbox["min_y"] + height * ratio
        split_line = ShapelyLineString([
            (bbox["min_x"] - max(width, height) - 1000, cut),
            (bbox["max_x"] + max(width, height) + 1000, cut),
        ])

    shp = polygon_to_shapely(polygon)
    try:
        split_result = shapely_ops.split(shp, split_line)
    except Exception:
        return [polygon for _ in weights]

    parts = [geom for geom in split_result.geoms if not geom.is_empty and geom.area > 1e-6]
    if len(parts) < 2:
        return [polygon for _ in weights]

    def sort_key(geom):
        centroid = geom.centroid
        return centroid.x if split_vertical else centroid.y

    parts.sort(key=sort_key)

    if len(parts) != 2:
        # Keep the two extreme groups to preserve a stable recursive split.
        if split_vertical:
            left_parts = [geom for geom in parts if geom.centroid.x <= cut]
            right_parts = [geom for geom in parts if geom.centroid.x > cut]
        else:
            left_parts = [geom for geom in parts if geom.centroid.y <= cut]
            right_parts = [geom for geom in parts if geom.centroid.y > cut]

        if not left_parts or not right_parts:
            return [polygon for _ in weights]

        left_poly = max(left_parts, key=lambda geom: geom.area)
        right_poly = max(right_parts, key=lambda geom: geom.area)
    else:
        left_poly, right_poly = parts[0], parts[1]

    left_polys = _partition_polygon_by_weights(shapely_to_polygon(left_poly), left_weights)
    right_polys = _partition_polygon_by_weights(shapely_to_polygon(right_poly), right_weights)
    return left_polys + right_polys


def _partition_rectangular_polygon(boundary: Polygon, weights: list[float], long_axis: str) -> list[Polygon]:
    if not weights:
        return []

    bbox = polygon_bbox(boundary)
    shp = polygon_to_shapely(boundary)
    total_weight = sum(max(0.0, float(weight)) for weight in weights)
    if total_weight <= 0:
        total_weight = float(len(weights))
        weights = [1.0 for _ in weights]

    parts: list[Polygon] = []
    eps = 1e-6
    norm_weights = [max(0.0, float(w)) for w in weights]

    # Build cumulative cut positions from the ORIGINAL total so each strip
    # gets weight/total_weight of the full span — not a progressively growing share.
    cuts: list[float] = []
    cumulative = 0.0
    for w in norm_weights[:-1]:  # last piece gets the remainder via max_x / max_y
        cumulative += w
        cuts.append(cumulative / total_weight)

    if long_axis == "horizontal":
        prev_x = bbox["min_x"]
        for i, ratio in enumerate(cuts):
            cut_x = bbox["min_x"] + bbox["width"] * ratio
            if cut_x <= prev_x + eps:
                continue
            strip = polygon_to_shapely([
                (prev_x, bbox["min_y"] - bbox["height"] - 1000),
                (cut_x, bbox["min_y"] - bbox["height"] - 1000),
                (cut_x, bbox["max_y"] + bbox["height"] + 1000),
                (prev_x, bbox["max_y"] + bbox["height"] + 1000),
            ])
            clipped = shp.intersection(strip)
            if not clipped.is_empty:
                if clipped.geom_type == "MultiPolygon":
                    clipped = max(clipped.geoms, key=lambda geom: geom.area)
                if clipped.geom_type == "Polygon":
                    parts.append(shapely_to_polygon(clipped))
            prev_x = cut_x
        # Last strip: from last cut to max_x
        if bbox["max_x"] > prev_x + eps:
            strip = polygon_to_shapely([
                (prev_x, bbox["min_y"] - bbox["height"] - 1000),
                (bbox["max_x"], bbox["min_y"] - bbox["height"] - 1000),
                (bbox["max_x"], bbox["max_y"] + bbox["height"] + 1000),
                (prev_x, bbox["max_y"] + bbox["height"] + 1000),
            ])
            clipped = shp.intersection(strip)
            if not clipped.is_empty:
                if clipped.geom_type == "MultiPolygon":
                    clipped = max(clipped.geoms, key=lambda geom: geom.area)
                if clipped.geom_type == "Polygon":
                    parts.append(shapely_to_polygon(clipped))
    else:
        prev_y = bbox["min_y"]
        for i, ratio in enumerate(cuts):
            cut_y = bbox["min_y"] + bbox["height"] * ratio
            if cut_y <= prev_y + eps:
                continue
            strip = polygon_to_shapely([
                (bbox["min_x"] - bbox["width"] - 1000, prev_y),
                (bbox["max_x"] + bbox["width"] + 1000, prev_y),
                (bbox["max_x"] + bbox["width"] + 1000, cut_y),
                (bbox["min_x"] - bbox["width"] - 1000, cut_y),
            ])
            clipped = shp.intersection(strip)
            if not clipped.is_empty:
                if clipped.geom_type == "MultiPolygon":
                    clipped = max(clipped.geoms, key=lambda geom: geom.area)
                if clipped.geom_type == "Polygon":
                    parts.append(shapely_to_polygon(clipped))
            prev_y = cut_y
        # Last strip: from last cut to max_y
        if bbox["max_y"] > prev_y + eps:
            strip = polygon_to_shapely([
                (bbox["min_x"] - bbox["width"] - 1000, prev_y),
                (bbox["max_x"] + bbox["width"] + 1000, prev_y),
                (bbox["max_x"] + bbox["width"] + 1000, bbox["max_y"]),
                (bbox["min_x"] - bbox["width"] - 1000, bbox["max_y"]),
            ])
            clipped = shp.intersection(strip)
            if not clipped.is_empty:
                if clipped.geom_type == "MultiPolygon":
                    clipped = max(clipped.geoms, key=lambda geom: geom.area)
                if clipped.geom_type == "Polygon":
                    parts.append(shapely_to_polygon(clipped))

    return parts


def _clip_rect_to_boundary(shp, min_x: float, min_y: float, max_x: float, max_y: float):
    if max_x <= min_x or max_y <= min_y:
        return None
    clip_rect = polygon_to_shapely([
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ])
    clipped = shp.intersection(clip_rect)
    if clipped.is_empty:
        return None
    if clipped.geom_type == "MultiPolygon":
        clipped = max(clipped.geoms, key=lambda geom: geom.area)
    if clipped.geom_type != "Polygon":
        return None
    return shapely_to_polygon(clipped)


def _partition_compact_rectangular_polygon(boundary: Polygon, weights: list[float]) -> list[Polygon]:
    """Partition a near-square plate into compact pieces using a squarified layout."""
    if not weights:
        return []

    bbox = polygon_bbox(boundary)
    width = max(1e-6, bbox["width"])
    height = max(1e-6, bbox["height"])
    area = width * height
    shp = polygon_to_shapely(boundary)

    raw = [max(0.0, float(weight)) for weight in weights]
    total = sum(raw)
    if total <= 0:
        raw = [1.0 for _ in weights]
        total = float(len(raw))
    norm = [(weight / total) * area for weight in raw]

    def _worst_ratio(row: list[float], short_side: float) -> float:
        if not row:
            return float("inf")
        s = sum(row)
        if s <= 0 or short_side <= 0:
            return float("inf")
        mx = max(row)
        mn = min(row)
        if mn <= 0:
            return float("inf")
        return max((short_side * short_side * mx) / (s * s), (s * s) / (short_side * short_side * mn))

    def _layout_row(row: list[float], x: float, y: float, w: float, h: float):
        rects = []
        s = sum(row)
        if s <= 0:
            return rects, (x, y, w, h)

        if w >= h:
            row_h = s / max(w, 1e-9)
            cx = x
            for cell in row:
                cw = cell / max(row_h, 1e-9)
                rects.append((cx, y, cx + cw, y + row_h))
                cx += cw
            return rects, (x, y + row_h, w, max(0.0, h - row_h))

        row_w = s / max(h, 1e-9)
        cy = y
        for cell in row:
            ch = cell / max(row_w, 1e-9)
            rects.append((x, cy, x + row_w, cy + ch))
            cy += ch
        return rects, (x + row_w, y, max(0.0, w - row_w), h)

    rects = []
    remaining = list(norm)
    x = bbox["min_x"]
    y = bbox["min_y"]
    w = width
    h = height
    row: list[float] = []

    while remaining:
        candidate = remaining[0]
        short_side = min(w, h)
        if (not row) or (_worst_ratio(row + [candidate], short_side) <= _worst_ratio(row, short_side)):
            row.append(candidate)
            remaining.pop(0)
            continue

        row_rects, (x, y, w, h) = _layout_row(row, x, y, w, h)
        rects.extend(row_rects)
        row = []

    if row:
        row_rects, (_x, _y, _w, _h) = _layout_row(row, x, y, w, h)
        rects.extend(row_rects)

    parts = []
    for min_x, min_y, max_x, max_y in rects:
        clipped = _clip_rect_to_boundary(shp, min_x, min_y, max_x, max_y)
        if clipped:
            parts.append(clipped)

    return parts


def _partition_rectangular_grid_2x2(boundary: Polygon, weights: list[float]) -> list[Polygon]:
    """Partition a plate into a weighted 2x2 grid (4 pieces), clipped to the boundary."""
    if len(weights) != 4:
        return []

    bbox = polygon_bbox(boundary)
    width = max(1e-6, bbox["width"])
    height = max(1e-6, bbox["height"])
    shp = polygon_to_shapely(boundary)

    raw = [max(0.0, float(weight)) for weight in weights]
    total = sum(raw)
    if total <= 0:
        raw = [1.0, 1.0, 1.0, 1.0]
        total = 4.0

    # Row-major pairing: [0,1] top row and [2,3] bottom row.
    top_sum = raw[0] + raw[1]
    bottom_sum = raw[2] + raw[3]
    if top_sum <= 0 or bottom_sum <= 0:
        return []

    top_height = height * (top_sum / total)
    y_top = bbox["max_y"]
    y_mid = y_top - top_height
    y_bottom = bbox["min_y"]

    top_left_width = width * (raw[0] / top_sum)
    bottom_left_width = width * (raw[2] / bottom_sum)

    x_left = bbox["min_x"]
    x_right = bbox["max_x"]
    x_mid_top = x_left + top_left_width
    x_mid_bottom = x_left + bottom_left_width

    rects = [
        (x_left, y_mid, x_mid_top, y_top),      # top-left
        (x_mid_top, y_mid, x_right, y_top),     # top-right
        (x_left, y_bottom, x_mid_bottom, y_mid),# bottom-left
        (x_mid_bottom, y_bottom, x_right, y_mid),# bottom-right
    ]

    parts = []
    for min_x, min_y, max_x, max_y in rects:
        clipped = _clip_rect_to_boundary(shp, min_x, min_y, max_x, max_y)
        if clipped:
            parts.append(clipped)
    return parts


def _split_piece_in_half(piece: Polygon) -> list[Polygon]:
    bbox = polygon_bbox(piece)
    width = bbox["width"]
    height = bbox["height"]
    if width <= 1e-6 or height <= 1e-6:
        return [piece]

    split_vertical = width >= height
    span = max(width, height) + 1000.0
    if split_vertical:
        cut = bbox["min_x"] + (width * 0.5)
        split_line = ShapelyLineString([
            (cut, bbox["min_y"] - span),
            (cut, bbox["max_y"] + span),
        ])
    else:
        cut = bbox["min_y"] + (height * 0.5)
        split_line = ShapelyLineString([
            (bbox["min_x"] - span, cut),
            (bbox["max_x"] + span, cut),
        ])

    shp = polygon_to_shapely(piece)
    try:
        result = shapely_ops.split(shp, split_line)
    except Exception:
        return [piece]

    parts = [geom for geom in result.geoms if not geom.is_empty and geom.area > 1e-6 and geom.geom_type == "Polygon"]
    if len(parts) < 2:
        return [piece]
    return [shapely_to_polygon(part) for part in parts[:2]]


def _ensure_exact_piece_count(boundary: Polygon, pieces: list[Polygon], count: int) -> list[Polygon]:
    if count <= 0:
        return []
    if not pieces:
        pieces = [boundary]

    exact = list(pieces)
    while len(exact) < count:
        idx = max(range(len(exact)), key=lambda i: polygon_area(exact[i]))
        candidate = exact.pop(idx)
        split_parts = _split_piece_in_half(candidate)
        if len(split_parts) < 2:
            exact.insert(idx, candidate)
            break
        exact.extend(split_parts)

    if len(exact) > count:
        # Keep largest pieces first; this should be rare because primary strategies target exact count.
        exact = sorted(exact, key=polygon_area, reverse=True)[:count]

    return exact


def build_subdivision_units_from_plate(
    boundary: Polygon,
    unit_mix: list[dict],
    analysis: dict,
    gen_log: list,
) -> tuple[list[Unit], list[Polygon]]:
    expanded = _expand_unit_mix(unit_mix)
    if not expanded:
        return [], []

    boundary_area_m2 = polygon_area(boundary) / 1e6
    target_weights = [max(1.0, float(item.get("target_area", 55) or 55)) for item in expanded]
    total_target = sum(target_weights)
    if total_target <= 0:
        total_target = float(len(target_weights))
        target_weights = [1.0 for _ in target_weights]

    if analysis.get("is_rectangular"):
        parts = _partition_rectangular_polygon(boundary, target_weights, analysis["long_axis"])
    else:
        scaled_weights = [weight * (boundary_area_m2 / total_target) for weight in target_weights]
        parts = _partition_polygon_by_weights(boundary, scaled_weights)
    if len(parts) != len(expanded):
        gen_log.append(
            f"Subdivision: partition count mismatch (expected {len(expanded)}, got {len(parts)}); using fallback coverage"
        )
        if len(parts) < len(expanded):
            # Fallback to equal-weight partitioning across the requested unit count.
            parts = _partition_polygon_by_weights(boundary, [1.0 for _ in expanded])

    units = []
    unresolved = []
    long_axis = analysis["long_axis"]
    for idx, (unit_spec, poly) in enumerate(zip(expanded, parts)):
        unit_area = polygon_area(poly) / 1e6
        bbox = polygon_bbox(poly)
        unit_depth = bbox["height"] if long_axis == "horizontal" else bbox["width"]
        units.append(Unit(
            id=f"unit-{idx}",
            type=unit_spec.get("type", "unit"),
            polygon=poly,
            typed_edges=[],
            entrance_point=(0, 0),
            depth=unit_depth,
            area=unit_area,
        ))

    coverage_area_m2 = sum(unit.area for unit in units)
    if coverage_area_m2 < boundary_area_m2 * 0.995:
        gen_log.append(
            f"Subdivision: coverage gap detected ({boundary_area_m2 - coverage_area_m2:.2f} m²); floor plate should be fully partitioned"
        )

    return units, unresolved


def _normalize_subdivision_unit_mix(unit_mix) -> list[dict]:
    if isinstance(unit_mix, dict):
        items = list(unit_mix.items())
    elif isinstance(unit_mix, list):
        items = []
        for item in unit_mix:
            if not isinstance(item, dict):
                continue
            items.append((item.get("type", "unit"), int(item.get("count", 0) or 0)))
    else:
        items = []

    normalized = []
    for unit_type, count in items:
        count = max(0, int(count or 0))
        if count <= 0:
            continue
        normalized.append({"type": str(unit_type), "count": count})
    return normalized


def _unit_spec_for_type(unit_type: str, constraints: dict) -> dict:
    unit_specs = constraints.get("unit_specs") or {}
    override = unit_specs.get(unit_type, {}) if isinstance(unit_specs, dict) else {}
    min_depth_m = float(override.get("min_depth_m", constraints.get("min_depth_m", 5.5)) or 5.5)
    min_area_m2 = float(override.get("min_area_m2", constraints.get("min_area_m2", 35.0)) or 35.0)
    target_area_m2 = float(
        override.get(
            "target_area_m2",
            DEFAULT_TARGET_AREA_M2_BY_TYPE.get(unit_type, max(min_area_m2 * 1.15, min_area_m2)),
        )
        or DEFAULT_TARGET_AREA_M2_BY_TYPE.get(unit_type, max(min_area_m2 * 1.15, min_area_m2))
    )
    return {
        "type": unit_type,
        "target_area_m2": target_area_m2,
        "min_area_m2": min_area_m2,
        "min_depth_m": min_depth_m,
    }


def _expand_subdivision_request(unit_mix, constraints: dict) -> list[dict]:
    expanded = []
    for item in _normalize_subdivision_unit_mix(unit_mix):
        spec = _unit_spec_for_type(item["type"], constraints)
        for _ in range(item["count"]):
            expanded.append(spec.copy())
    return expanded


def _allocate_counts_by_ratio(target_total: int, ratio_items: list[dict]) -> dict:
    """Allocate integer counts by ratio using largest remainder, preserving deterministic order."""
    if target_total <= 0 or not ratio_items:
        return {}

    ratio_sum = sum(max(0, int(item.get("count", 0))) for item in ratio_items)
    if ratio_sum <= 0:
        return {}

    provisional = []
    allocated = 0
    for idx, item in enumerate(ratio_items):
        ratio_count = max(0, int(item.get("count", 0)))
        exact = (target_total * ratio_count) / ratio_sum
        base = int(exact)
        allocated += base
        provisional.append({
            "idx": idx,
            "type": str(item.get("type", "unit")),
            "base": base,
            "remainder": exact - base,
        })

    remaining = max(0, target_total - allocated)
    provisional.sort(key=lambda row: (-row["remainder"], row["idx"]))
    for i in range(remaining):
        provisional[i % len(provisional)]["base"] += 1

    provisional.sort(key=lambda row: row["idx"])
    out = {}
    for row in provisional:
        if row["base"] > 0:
            out[row["type"]] = row["base"]
    return out


def _adjust_unit_mix_to_capacity(boundary: Polygon, unit_mix, constraints: dict) -> tuple[dict, dict, int, int, list[str]]:
    """Adapt requested unit counts to floor-plate capacity while preserving ratio by type.

    Returns: (adjusted_mix, requested_mix, requested_total, target_total, warnings)
    """
    requested_items = _normalize_subdivision_unit_mix(unit_mix)
    requested_mix = {item["type"]: item["count"] for item in requested_items}
    requested_total = sum(requested_mix.values())
    if requested_total <= 0:
        return {}, requested_mix, 0, 0, []

    boundary_area_m2 = polygon_area(boundary) / 1e6
    weighted_target_area = 0.0
    weighted_min_area = 0.0
    for item in requested_items:
        spec = _unit_spec_for_type(item["type"], constraints)
        weighted_target_area += max(spec["target_area_m2"], 1.0) * item["count"]
        weighted_min_area += max(spec["min_area_m2"], 1.0) * item["count"]

    avg_target_area = max(1.0, weighted_target_area / requested_total)
    avg_min_area = max(1.0, weighted_min_area / requested_total)

    # Geometry-aware capacity: use the same slab classification as
    # compute_unit_distribution_breakdown (short_dim vs 2×min_depth), not aspect ratio.
    corridor_width_m = float(constraints.get("corridor_width", 0) or 0) / 1000.0
    _min_depth_m = float(
        constraints.get("min_depth_m")
        or (float(constraints.get("min_unit_depth", 0) or 0) / 1000.0)
        or 7.0
    )
    if corridor_width_m > 0:
        _ab = analyze_boundary(boundary)
        _bbox = _ab["bbox"]
        _slab_len_m = max(_bbox["width"], _bbox["height"]) / 1000.0
        _slab_dep_m = min(_bbox["width"], _bbox["height"]) / 1000.0
        _aspect = _slab_len_m / max(_slab_dep_m, 0.001)
        if _aspect >= 1.4 and _slab_dep_m >= 2.0 * _min_depth_m:
            # Double-loaded slab: each placed unit yields two half-depth pieces after
            # corridor subtraction. Optimal count = units fitting on one side at target size.
            _half_dep = max((_slab_dep_m - corridor_width_m) / 2.0, _slab_dep_m * 0.25)
            _per_side_m2 = _slab_len_m * _half_dep
            optimal_total  = max(1, int(_per_side_m2 // avg_target_area))
            capacity_total = max(1, int(_per_side_m2 // avg_min_area))
        elif _aspect >= 1.4:
            # Single-loaded slab: units fill the full boundary; corridor overlaps rather than restricts.
            _usable_dep = max(_slab_dep_m - corridor_width_m, _slab_dep_m * 0.5)
            optimal_total  = max(1, int((_slab_len_m * _usable_dep) // avg_target_area))
            capacity_total = max(1, int(boundary_area_m2 // avg_min_area))
        else:
            # Near-square (point block): corridor is not a linear slab corridor, use total area.
            optimal_total  = max(1, int(boundary_area_m2 // avg_target_area))
            capacity_total = max(1, int(boundary_area_m2 // avg_min_area))
    else:
        optimal_total  = max(1, int(boundary_area_m2 // avg_target_area))
        capacity_total = max(1, int(boundary_area_m2 // avg_min_area))

    # Allow optional hard bounds from constraints while keeping adaptive behavior on by default.
    min_total_units = max(1, int(constraints.get("min_total_units", 1) or 1))
    max_total_units = int(constraints.get("max_total_units", 0) or 0)
    # Expand to fill when the plate has room (up to optimal); keep requested count when
    # it already exceeds optimal (user is packing more units in than the target size allows).
    target_total = max(min_total_units, max(requested_total, optimal_total))
    target_total = min(target_total, capacity_total)
    if max_total_units > 0:
        target_total = min(target_total, max_total_units)

    adjusted_mix = _allocate_counts_by_ratio(target_total, requested_items)
    adjusted_total = sum(adjusted_mix.values())

    if adjusted_total <= 0:
        adjusted_mix = requested_mix.copy()
        adjusted_total = requested_total

    warnings = []
    if adjusted_total < requested_total:
        warnings.append(
            f"Requested {requested_total} units but estimated feasible capacity is {adjusted_total}; returning reduced mix preserving ratio."
        )
    elif adjusted_total > requested_total:
        warnings.append(
            f"Requested {requested_total} units but estimated feasible capacity is {adjusted_total}; returning expanded mix preserving ratio."
        )

    return adjusted_mix, requested_mix, requested_total, adjusted_total, warnings


def _polygon_perimeter_mm(polygon: Polygon) -> float:
    if len(polygon) < 2:
        return 0.0
    total = 0.0
    for i in range(len(polygon)):
        total += point_distance(polygon[i], polygon[(i + 1) % len(polygon)])
    return total


def _geometry_compactness(polygon: Polygon) -> float:
    area = polygon_area(polygon)
    if area <= 0:
        return float("inf")
    perimeter = _polygon_perimeter_mm(polygon)
    return (perimeter * perimeter) / area


def _candidate_orderings(expanded_units: list[dict], type_priority: dict, seed: Optional[int]) -> list[list[dict]]:
    orders = [
        list(expanded_units),
        sorted(expanded_units, key=lambda item: item["target_area_m2"], reverse=True),
        sorted(expanded_units, key=lambda item: item["target_area_m2"]),
    ]

    if type_priority:
        orders.append(sorted(
            expanded_units,
            key=lambda item: (-float(type_priority.get(item["type"], 0.0)), -item["target_area_m2"]),
        ))

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(orders)

    unique_orders = []
    seen = set()
    for order in orders:
        signature = tuple((item["type"], round(item["target_area_m2"], 3)) for item in order)
        if signature in seen:
            continue
        seen.add(signature)
        unique_orders.append(order)
    return unique_orders


def _is_non_rectangular(boundary: Polygon) -> bool:
    """Return True when the polygon is non-rectangular (L-shape, T-shape, etc.)."""
    if len(boundary) > 4:
        return True
    bbox = polygon_bbox(boundary)
    bbox_area = max(1e-6, bbox["width"] * bbox["height"])
    poly_area = polygon_area(boundary)
    return (poly_area / bbox_area) < 0.95


def _partition_non_rectangular(boundary: Polygon, weights: list[float], count: int) -> list[Polygon]:
    """Partition a non-rectangular polygon by decomposing it into convex sub-regions first,
    allocating units to each sub-region proportionally, then partitioning each sub-region."""
    try:
        sub_regions = split_polygon(boundary)
    except Exception:
        sub_regions = [boundary]

    if len(sub_regions) <= 1:
        # Fallback to weight-based bisection when decomposition yields only one piece.
        total_w = sum(max(0.0, float(w)) for w in weights)
        if total_w <= 0:
            total_w = float(len(weights))
        scaled = [max(0.0, float(w)) / total_w for w in weights]
        return _ensure_exact_piece_count(boundary, _partition_polygon_by_weights(boundary, scaled), count)

    # Compute area for each sub-region.
    sub_areas = [polygon_area(r) for r in sub_regions]
    total_area = max(1e-6, sum(sub_areas))

    # Allocate unit count to each sub-region proportional to its area fraction.
    total_w = sum(max(0.0, float(w)) for w in weights)
    if total_w <= 0:
        total_w = float(len(weights))

    # Calculate how many units (float) each sub-region should receive.
    sub_unit_counts_float = [(a / total_area) * count for a in sub_areas]

    # Round using largest-remainder to get exact integers summing to count.
    sub_unit_counts = [int(x) for x in sub_unit_counts_float]
    remainders = [(sub_unit_counts_float[i] - sub_unit_counts[i], i) for i in range(len(sub_regions))]
    deficit = count - sum(sub_unit_counts)
    for _, idx in sorted(remainders, reverse=True)[:max(0, deficit)]:
        sub_unit_counts[idx] += 1

    # Ensure every sub-region with area gets at least 1 unit if count allows.
    for i in range(len(sub_regions)):
        if sub_unit_counts[i] == 0 and sub_areas[i] > 1e-6:
            # Find the sub-region with the most units and take one.
            donor = max(range(len(sub_regions)), key=lambda j: sub_unit_counts[j])
            if sub_unit_counts[donor] > 1:
                sub_unit_counts[donor] -= 1
                sub_unit_counts[i] = 1

    # Allocate weights to sub-regions: sort weights by size and assign largest to largest sub-region.
    sorted_weight_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
    sorted_sub_indices = sorted(range(len(sub_regions)), key=lambda i: sub_areas[i], reverse=True)

    sub_weights: list[list[float]] = [[] for _ in sub_regions]
    weight_cursor = 0
    for sub_idx in sorted_sub_indices:
        for _ in range(sub_unit_counts[sub_idx]):
            if weight_cursor < len(sorted_weight_indices):
                sub_weights[sub_idx].append(weights[sorted_weight_indices[weight_cursor]])
                weight_cursor += 1

    # Partition each sub-region with its assigned weights.
    all_pieces: list[Polygon] = []
    for i, sub_region in enumerate(sub_regions):
        n = sub_unit_counts[i]
        if n <= 0:
            continue
        w = sub_weights[i]
        if not w:
            w = [1.0] * n
        if n == 1:
            all_pieces.append(sub_region)
        else:
            sub_bbox = polygon_bbox(sub_region)
            sub_long_axis = "horizontal" if sub_bbox["width"] >= sub_bbox["height"] else "vertical"
            sub_parts = _partition_rectangular_polygon(sub_region, w, sub_long_axis)
            sub_parts = _ensure_exact_piece_count(sub_region, sub_parts, n)
            all_pieces.extend(sub_parts)

    return _ensure_exact_piece_count(boundary, all_pieces, count)


def _build_partition_pieces(boundary: Polygon, weights: list[float], orientation: str, count: int) -> list[Polygon]:
    if count <= 0:
        return []
    bbox = polygon_bbox(boundary)
    long_axis = "horizontal" if bbox["width"] >= bbox["height"] else "vertical"
    short_dim = max(1e-6, min(bbox["width"], bbox["height"]))
    long_dim = max(short_dim, max(bbox["width"], bbox["height"]))
    aspect = long_dim / short_dim

    non_rectangular = _is_non_rectangular(boundary)

    # Non-rectangular polygons (L, T, U shapes) must be decomposed into
    # convex sub-regions first, with units allocated proportionally per sub-region.
    if non_rectangular:
        return _partition_non_rectangular(boundary, weights, count)

    if orientation == "auto" and count == 4 and aspect <= 1.45:
        grid_parts = _partition_rectangular_grid_2x2(boundary, weights)
        if len(grid_parts) == 4:
            return grid_parts

    if orientation == "auto" and 3 <= count <= 6 and aspect <= 1.35:
        compact_parts = _partition_compact_rectangular_polygon(boundary, weights)
        if len(compact_parts) == count:
            return compact_parts

    use_axis = long_axis if orientation == "auto" else orientation
    if use_axis == "horizontal":
        base_parts = _partition_rectangular_polygon(boundary, weights, "horizontal")
    else:
        base_parts = _partition_rectangular_polygon(boundary, weights, "vertical")

    return _ensure_exact_piece_count(boundary, base_parts, count)


def _pick_residual_reason(piece: Polygon, unit_specs: list[dict]) -> tuple[str, Optional[str]]:
    area_m2 = polygon_area(piece) / 1e6
    bbox = polygon_bbox(piece)
    depth_m = min(bbox["width"], bbox["height"]) / 1000.0
    eligible = []
    for spec in unit_specs:
        if area_m2 >= spec["min_area_m2"] and depth_m >= spec["min_depth_m"]:
            eligible.append(spec["type"])
    if eligible:
        return "mix_constraint", eligible[0]
    if area_m2 < min(spec["min_area_m2"] for spec in unit_specs):
        return "too_small", None
    if depth_m < min(spec["min_depth_m"] for spec in unit_specs):
        return "too_narrow", None
    return "geometry_invalid", None


def _piece_metrics(piece: Polygon) -> dict:
    piece_area_m2 = polygon_area(piece) / 1e6
    bbox = polygon_bbox(piece)
    piece_depth_m = min(bbox["width"], bbox["height"]) / 1000.0
    width_m = max(bbox["width"], bbox["height"]) / 1000.0
    return {
        "polygon": piece,
        "area_m2": piece_area_m2,
        "depth_m": piece_depth_m,
        "width_m": width_m,
    }


def _assign_specs_to_pieces(unit_specs: list[dict], pieces: list[Polygon]) -> list[tuple[int, dict, dict]]:
    """Match requested unit specs to pieces so larger/fitter pieces receive larger unit types first."""
    if not unit_specs or not pieces:
        return []

    metrics = [_piece_metrics(piece) for piece in pieces]
    unassigned_piece_indices = set(range(len(metrics)))
    assignments: list[tuple[int, dict, dict]] = []

    spec_indices = list(range(len(unit_specs)))
    spec_indices.sort(
        key=lambda idx: (
            float(unit_specs[idx].get("min_area_m2", 0.0)),
            float(unit_specs[idx].get("target_area_m2", 0.0)),
            float(unit_specs[idx].get("min_depth_m", 0.0)),
        ),
        reverse=True,
    )

    for spec_idx in spec_indices:
        spec = unit_specs[spec_idx]
        best_piece_idx = None
        best_score = float("-inf")

        for piece_idx in unassigned_piece_indices:
            metric = metrics[piece_idx]
            area_m2 = metric["area_m2"]
            depth_m = metric["depth_m"]
            min_area_m2 = float(spec.get("min_area_m2", 0.0))
            min_depth_m = float(spec.get("min_depth_m", 0.0))
            target_area_m2 = float(spec.get("target_area_m2", min_area_m2))

            feasible = area_m2 >= min_area_m2 and depth_m >= min_depth_m
            area_shortfall = max(0.0, min_area_m2 - area_m2)
            depth_shortfall = max(0.0, min_depth_m - depth_m)
            area_deviation = abs(area_m2 - target_area_m2)

            score = (
                (1_000_000.0 if feasible else 0.0)
                - (50_000.0 * area_shortfall)
                - (50_000.0 * depth_shortfall)
                - (100.0 * area_deviation)
                + area_m2
            )

            if score > best_score:
                best_score = score
                best_piece_idx = piece_idx

        if best_piece_idx is None:
            continue

        unassigned_piece_indices.remove(best_piece_idx)
        assignments.append((best_piece_idx, spec, metrics[best_piece_idx]))

    assignments.sort(key=lambda item: item[0])
    return assignments


def _evaluate_contract_candidate(
    boundary: Polygon,
    unit_mix,
    constraints: dict,
    scoring: dict,
    seed: Optional[int],
    orientation: str,
    residual_slots: int,
) -> dict:
    unit_specs = _expand_subdivision_request(unit_mix, constraints)
    requested_total = len(unit_specs)
    if requested_total == 0:
        pieces = [boundary]
        residual_slots = 1
        unit_specs = []
    else:
        residual_slots = 0
        piece_count = requested_total
        target_weights = [max(1.0, spec["target_area_m2"]) for spec in unit_specs]
        pieces = _build_partition_pieces(boundary, target_weights, orientation, piece_count)

    if not pieces:
        pieces = [boundary]

    placed_units = []
    residual_zones = []
    shortfall = {}
    warnings = []
    coverage_pieces = []

    for piece in pieces:
        coverage_pieces.append(piece)

    assignments = _assign_specs_to_pieces(unit_specs, pieces)
    assigned_piece_indices = {piece_idx for piece_idx, _spec, _metric in assignments}

    for piece_idx, spec, metric in assignments:
        piece_area_m2 = metric["area_m2"]
        piece_depth_m = metric["depth_m"]
        width_m = metric["width_m"]
        target_area = max(spec["target_area_m2"], spec["min_area_m2"])
        deviation_pct = ((piece_area_m2 - target_area) / target_area) * 100.0 if target_area > 0 else 0.0
        meets_constraints = piece_area_m2 >= spec["min_area_m2"] and piece_depth_m >= spec["min_depth_m"]

        placed_unit = {
            "type": spec["type"],
            "polygon": metric["polygon"],
            "area_m2": piece_area_m2,
            "depth_m": round(piece_depth_m, 4),
            "width_m": round(width_m, 4),
            "area_deviation_pct": deviation_pct,
        }

        if not meets_constraints:
            shortfall[spec["type"]] = shortfall.get(spec["type"], 0) + 1
            warnings.append(
                f"{spec['type']} placed with constraint deviation (area={piece_area_m2:.2f}m², depth={piece_depth_m:.2f}m; min_area={spec['min_area_m2']:.2f}m², min_depth={spec['min_depth_m']:.2f}m)"
            )
            placed_unit["constraint_violation"] = {
                "min_area_m2": spec["min_area_m2"],
                "min_depth_m": spec["min_depth_m"],
            }

        placed_units.append(placed_unit)

    # Fallback safety: if assignment missed any pieces, attach them to the smallest requested type.
    if len(assigned_piece_indices) < len(pieces):
        fallback_type = min(
            unit_specs,
            key=lambda item: (float(item.get("min_area_m2", 0.0)), float(item.get("target_area_m2", 0.0))),
        )["type"] if unit_specs else "unit"
        for idx, piece in enumerate(pieces):
            if idx in assigned_piece_indices:
                continue
            metric = _piece_metrics(piece)
            placed_units.append({
                "type": fallback_type,
                "polygon": metric["polygon"],
                "area_m2": metric["area_m2"],
                "depth_m": round(metric["depth_m"], 4),
                "width_m": round(metric["width_m"], 4),
                "area_deviation_pct": 0.0,
                "constraint_violation": {"reason": "unmatched_piece"},
            })
            warnings.append("Unmatched partition piece assigned to fallback unit type.")

    output_area_m2 = sum(item["area_m2"] for item in placed_units) + sum(item["area_m2"] for item in residual_zones)
    input_area_m2 = polygon_area(boundary) / 1e6
    discrepancy_m2 = abs(output_area_m2 - input_area_m2)
    area_tolerance = float(constraints.get("area_tolerance_m2", 0.01) or 0.01)

    area_check = {
        "input_area_m2": input_area_m2,
        "output_area_m2": output_area_m2,
        "discrepancy_m2": discrepancy_m2,
        "passed": discrepancy_m2 <= area_tolerance,
    }

    shortfall_counts = {key: int(value) for key, value in shortfall.items()}
    if requested_total > len(placed_units):
        warnings.append("Target mix not fully achieved; returning best achievable subdivision result.")

    total_shortfall = sum(shortfall_counts.values())
    residual_area_m2 = sum(zone["area_m2"] for zone in residual_zones)
    fragmentation = len(residual_zones)
    compactness = sum(_geometry_compactness(unit["polygon"]) for unit in placed_units) if placed_units else 0.0
    type_priority = scoring.get("type_priority", {}) if isinstance(scoring, dict) else {}
    placed_reward = float(scoring.get("placed_unit_reward", DEFAULT_SCORING["placed_unit_reward"]))
    shortfall_penalty = float(scoring.get("shortfall_penalty", DEFAULT_SCORING["shortfall_penalty"]))
    residual_area_penalty = float(scoring.get("residual_area_penalty", DEFAULT_SCORING["residual_area_penalty"]))
    residual_fragmentation_penalty = float(scoring.get("residual_fragmentation_penalty", DEFAULT_SCORING["residual_fragmentation_penalty"]))
    area_deviation_penalty = float(scoring.get("area_deviation_penalty", DEFAULT_SCORING["area_deviation_penalty"]))
    compactness_penalty = float(scoring.get("compactness_penalty", DEFAULT_SCORING["compactness_penalty"]))

    priority_reward = 0.0
    for unit in placed_units:
        priority_reward += float(type_priority.get(unit["type"], 0.0))

    area_deviation = sum(abs(unit.get("area_deviation_pct", 0.0)) for unit in placed_units)
    score = (
        placed_reward * len(placed_units)
        + priority_reward
        - shortfall_penalty * total_shortfall
        - residual_area_penalty * residual_area_m2
        - residual_fragmentation_penalty * fragmentation
        - area_deviation_penalty * area_deviation
        - compactness_penalty * compactness
    )

    count_match = (requested_total == 0) or (len(placed_units) == requested_total)
    if requested_total > 0 and not count_match:
        warnings.append(
            f"Requested {requested_total} units but produced {len(placed_units)}; candidate rejected for count mismatch."
        )
    hard_valid = area_check["passed"] and len(residual_zones) == 0 and count_match
    return {
        "mode": "single",
        "placed_units": placed_units,
        "residual_zones": residual_zones,
        "shortfall": shortfall_counts,
        "warnings": warnings,
        "score": score,
        "seed": seed,
        "area_check": area_check,
        "hard_valid": hard_valid,
        "candidate_meta": {
            "orientation": orientation,
            "residual_slots": residual_slots,
            "requested_total": requested_total,
        },
    }


def _compute_circulation_for_subdivision(
    boundary: Polygon,
    unit_mix,
    constraints: dict,
) -> tuple:
    """Run stages 1–5 to produce cores, corridors, and apartment zones for a subdivision request.

    Returns (cores, corridors, apartment_zones, gen_log_entries).
    On any error falls back to ([], [], [boundary], [warning_msg]) so the caller
    can proceed with the full boundary as the subdivision domain.
    """
    gen_log: list[str] = []
    try:
        min_unit_depth_mm = float(
            constraints.get("min_unit_depth") or
            (float(constraints.get("min_depth_m") or 0) * 1000) or
            7000.0
        )
        if min_unit_depth_mm <= 0:
            min_unit_depth_mm = 7000.0

        corridor_width_mm = float(constraints.get("corridor_width") or 1500)

        core_type_str = str(constraints.get("core_type") or "stair_lift")
        try:
            core_type = CoreType(core_type_str)
        except ValueError:
            core_type = CoreType.STAIR_LIFT

        analysis = analyze_boundary(boundary)
        normalized_mix = _normalize_subdivision_unit_mix(unit_mix)

        distribution_breakdown = compute_unit_distribution_breakdown(
            analysis,
            normalized_mix,
            min_unit_depth_mm,
        )

        typology = infer_typology(analysis, normalized_mix, distribution_breakdown)
        sub_type_str = typology.sub_type.value if hasattr(typology.sub_type, "value") else str(typology.sub_type)
        gen_log.append(
            f"Circulation: typology={typology.type.value}/{sub_type_str} "
            f"corridor_width={corridor_width_mm:.0f}mm core={core_type_str}"
        )

        cores = place_cores(boundary, analysis, typology, core_type, gen_log)
        corridors = derive_corridor(
            boundary, analysis, typology, cores, corridor_width_mm, distribution_breakdown, gen_log
        )
        apartment_zones = compute_apartment_zones(
            boundary, analysis, typology, cores, corridors, corridor_width_mm,
            distribution_breakdown, gen_log,
        )

        if not apartment_zones:
            gen_log.append("Circulation: no apartment zones derived; using full boundary")
            apartment_zones = [boundary]

        return cores, corridors, apartment_zones, gen_log
    except Exception as exc:
        gen_log.append(f"Circulation: compute failed ({exc}); falling back to full boundary")
        return [], [], [boundary], gen_log


def _evaluate_over_zones(
    apartment_zones: list,
    boundary: Polygon,
    unit_mix,
    constraints: dict,
    scoring: dict,
    seed: Optional[int],
    orientation: str,
    residual_slots: int,
) -> dict:
    """Evaluate a subdivision candidate distributed across pre-computed apartment zones.

    Single zone: delegates directly to _evaluate_contract_candidate.
    Multiple zones: splits the unit mix proportionally by zone area, evaluates
    each zone independently, and merges the results.
    """
    if len(apartment_zones) == 1:
        return _evaluate_contract_candidate(
            apartment_zones[0], unit_mix, constraints, scoring, seed, orientation, residual_slots
        )

    zone_areas = [max(0.0, polygon_area(z)) for z in apartment_zones]
    total_zone_area = sum(zone_areas)
    if total_zone_area <= 0:
        return _evaluate_contract_candidate(
            boundary, unit_mix, constraints, scoring, seed, orientation, residual_slots
        )

    unit_specs = _expand_subdivision_request(unit_mix, constraints)
    n = len(unit_specs)
    if n == 0:
        return _evaluate_contract_candidate(
            boundary, unit_mix, constraints, scoring, seed, orientation, residual_slots
        )

    # Distribute unit count to zones using largest-remainder, proportional to zone area.
    zone_counts_float = [a / total_zone_area * n for a in zone_areas]
    zone_counts = [int(x) for x in zone_counts_float]
    deficit = n - sum(zone_counts)
    rank = sorted(
        range(len(apartment_zones)),
        key=lambda i: zone_counts_float[i] - zone_counts[i],
        reverse=True,
    )
    for i in range(max(0, deficit)):
        zone_counts[rank[i]] += 1

    # Assign largest unit specs to largest zones.
    sorted_spec_idx = sorted(range(n), key=lambda i: unit_specs[i]["target_area_m2"], reverse=True)
    sorted_zone_order = sorted(range(len(apartment_zones)), key=lambda i: zone_areas[i], reverse=True)

    zone_spec_lists: list[list[dict]] = [[] for _ in apartment_zones]
    spec_cursor = 0
    for zone_idx in sorted_zone_order:
        for _ in range(zone_counts[zone_idx]):
            if spec_cursor < n:
                zone_spec_lists[zone_idx].append(unit_specs[sorted_spec_idx[spec_cursor]])
                spec_cursor += 1

    combined_placed: list[dict] = []
    combined_residual: list[dict] = []
    combined_warnings: list[str] = []
    combined_shortfall: dict = {}
    total_score = 0.0
    all_hard_valid = True
    area_out = 0.0

    for zone, specs in zip(apartment_zones, zone_spec_lists):
        if not specs:
            continue
        zone_mix: dict = {}
        for spec in specs:
            zone_mix[spec["type"]] = zone_mix.get(spec["type"], 0) + 1

        candidate = _evaluate_contract_candidate(
            zone, zone_mix, constraints, scoring, seed, orientation, residual_slots
        )
        combined_placed.extend(candidate["placed_units"])
        combined_residual.extend(candidate["residual_zones"])
        combined_warnings.extend(candidate["warnings"])
        for k, v in candidate.get("shortfall", {}).items():
            combined_shortfall[k] = combined_shortfall.get(k, 0) + v
        total_score += candidate["score"]
        area_out += candidate["area_check"].get("output_area_m2", 0.0)
        if not candidate["hard_valid"]:
            all_hard_valid = False

    area_in = polygon_area(boundary) / 1e6
    return {
        "mode": "single",
        "placed_units": combined_placed,
        "residual_zones": combined_residual,
        "shortfall": combined_shortfall,
        "warnings": combined_warnings,
        "score": total_score,
        "seed": seed,
        "area_check": {
            "input_area_m2": area_in,
            "output_area_m2": area_out,
            "discrepancy_m2": abs(area_out - area_in),
            "passed": all_hard_valid,
        },
        "hard_valid": all_hard_valid,
        "candidate_meta": {
            "orientation": orientation,
            "residual_slots": residual_slots,
            "zone_count": len(apartment_zones),
        },
    }


def _compute_corridor_from_placed_units(
    placed_units: list,
    analysis: dict,
    typology,
    corridor_width: float,
) -> list:
    """Compute the smallest corridor rectangle such that every placed unit has
    at least one wall coinciding with a corridor edge.

    The corridor polygon overlaps the inner face of the unit band; a boolean
    subtraction (done by the caller/frontend) yields the net unit areas.

    Coordinate space: mm (same as boundary / placed_unit polygons).
    """
    unit_polys = [u["polygon"] for u in placed_units if u.get("polygon")]
    if not unit_polys:
        return []

    long_axis = analysis["long_axis"]
    bbox = analysis["bbox"]

    def _x_span(polys):
        xs = [pt[0] for poly in polys for pt in poly]
        return min(xs), max(xs)

    def _y_span(polys):
        ys = [pt[1] for poly in polys for pt in poly]
        return min(ys), max(ys)

    if typology.type == TypologyType.SLAB:
        if long_axis == "horizontal":
            if typology.sub_type == SlabSubType.DOUBLE_LOADED:
                # Corridor at mid-depth, between inner walls of end columns.
                # End columns are split by a party wall (not the corridor); see
                # _split_endunits_at_corridor_midline called after subtraction.
                center_y = (bbox["min_y"] + bbox["max_y"]) / 2.0
                y_start = center_y - corridor_width / 2.0
                y_end   = center_y + corridor_width / 2.0
                if len(unit_polys) >= 3:
                    xs_min = [min(pt[0] for pt in p) for p in unit_polys]
                    xs_max = [max(pt[0] for pt in p) for p in unit_polys]
                    left_end  = unit_polys[xs_min.index(min(xs_min))]
                    right_end = unit_polys[xs_max.index(max(xs_max))]
                    x_start = max(pt[0] for pt in left_end)
                    x_end   = min(pt[0] for pt in right_end)
                else:
                    x_start, x_end = _x_span(unit_polys)
            else:  # SINGLE_LOADED
                # Corridor at inner face (max_y), spanning only between the inner
                # walls of the two end units so those end units keep full depth.
                inner_y = min(max(pt[1] for pt in poly) for poly in unit_polys)
                y_end   = inner_y
                y_start = inner_y - corridor_width
                touching = [p for p in unit_polys if max(pt[1] for pt in p) >= inner_y - 1.0]
                polys = touching if touching else unit_polys
                if len(polys) >= 3:
                    # Corridor runs between the inner (party) walls of the two end units.
                    # End units are left untouched by the corridor; they retain full depth.
                    xs_min = [min(pt[0] for pt in p) for p in polys]
                    xs_max = [max(pt[0] for pt in p) for p in polys]
                    left_end  = polys[xs_min.index(min(xs_min))]
                    right_end = polys[xs_max.index(max(xs_max))]
                    x_start = max(pt[0] for pt in left_end)
                    x_end   = min(pt[0] for pt in right_end)
                else:
                    x_start, x_end = _x_span(polys)
            centreline = [(x_start, (y_start + y_end) / 2.0), (x_end, (y_start + y_end) / 2.0)]
        else:  # vertical long axis
            if typology.sub_type == SlabSubType.DOUBLE_LOADED:
                # Corridor at mid-width, between inner walls of end rows.
                center_x = (bbox["min_x"] + bbox["max_x"]) / 2.0
                x_start = center_x - corridor_width / 2.0
                x_end   = center_x + corridor_width / 2.0
                if len(unit_polys) >= 3:
                    ys_min = [min(pt[1] for pt in p) for p in unit_polys]
                    ys_max = [max(pt[1] for pt in p) for p in unit_polys]
                    bot_end = unit_polys[ys_min.index(min(ys_min))]
                    top_end = unit_polys[ys_max.index(max(ys_max))]
                    y_start = max(pt[1] for pt in bot_end)
                    y_end   = min(pt[1] for pt in top_end)
                else:
                    y_start, y_end = _y_span(unit_polys)
            else:  # SINGLE_LOADED
                inner_x = min(max(pt[0] for pt in poly) for poly in unit_polys)
                x_end   = inner_x
                x_start = inner_x - corridor_width
                touching = [p for p in unit_polys if max(pt[0] for pt in p) >= inner_x - 1.0]
                polys = touching if touching else unit_polys
                if len(polys) >= 3:
                    ys_min = [min(pt[1] for pt in p) for p in polys]
                    ys_max = [max(pt[1] for pt in p) for p in polys]
                    bottom_end = polys[ys_min.index(min(ys_min))]
                    top_end    = polys[ys_max.index(max(ys_max))]
                    y_start = max(pt[1] for pt in bottom_end)
                    y_end   = min(pt[1] for pt in top_end)
                else:
                    y_start, y_end = _y_span(polys)
            centreline = [((x_start + x_end) / 2.0, y_start), ((x_start + x_end) / 2.0, y_end)]
    else:  # POINT_BLOCK — full-span corridor across the plate (no bookend; all units get access)
        if long_axis == "horizontal":
            center_y = (bbox["min_y"] + bbox["max_y"]) / 2.0
            y_start = center_y - corridor_width / 2.0
            y_end   = center_y + corridor_width / 2.0
            x_start, x_end = _x_span(unit_polys)
            centreline = [(x_start, (y_start + y_end) / 2.0), (x_end, (y_start + y_end) / 2.0)]
        else:
            center_x = (bbox["min_x"] + bbox["max_x"]) / 2.0
            x_start = center_x - corridor_width / 2.0
            x_end   = center_x + corridor_width / 2.0
            y_start, y_end = _y_span(unit_polys)
            centreline = [((x_start + x_end) / 2.0, y_start), ((x_start + x_end) / 2.0, y_end)]

    corridor_poly = [
        (x_start, y_start), (x_end, y_start),
        (x_end, y_end),     (x_start, y_end),
    ]
    return [Corridor(
        id="corr-0",
        polygon=corridor_poly,
        width=corridor_width,
        centreline=centreline,
    )]


def _compute_unit_entrances(placed_units: list, corridors: list) -> list:
    """Set entrance_point and typed_edges on each placed unit.

    The entrance is the midpoint of the unit edge that is geometrically closest to the
    corridor polygon.  That edge is also tagged as 'facing_corridor' in typed_edges.
    """
    if not corridors or not placed_units:
        return placed_units

    from shapely.geometry import Point as ShapelyPoint

    corr_shp = polygon_to_shapely(corridors[0].polygon)

    result = []
    for unit in placed_units:
        poly = unit.get("polygon")
        if not poly or len(poly) < 3:
            result.append(unit)
            continue

        min_dist = float("inf")
        best_idx = 0
        best_mid: tuple[float, float] = (poly[0][0], poly[0][1])

        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            mx = (p1[0] + p2[0]) / 2.0
            my = (p1[1] + p2[1]) / 2.0
            dist = corr_shp.distance(ShapelyPoint(mx, my))
            if dist < min_dist:
                min_dist = dist
                best_idx = i
                best_mid = (mx, my)

        result.append({
            **unit,
            "entrance_point": list(best_mid),
            "typed_edges": [{"edge_index": best_idx, "type": "facing_corridor"}],
        })
    return result


def _reassign_types_by_area(placed_units: list, constraints: dict) -> list:
    """Re-label unit types after corridor subtraction so that larger pieces get larger-target types.

    After bisection the pre-subtraction type labels may no longer match the resulting areas.
    This function re-sorts all pieces by area descending and distributes the existing type
    pool (preserving counts) from largest-target-first to largest-area-first.
    """
    if not placed_units:
        return placed_units

    unit_specs_cfg = constraints.get("unit_specs") or {}

    # Collect the type pool from the current labels.
    type_counts: dict[str, int] = {}
    for u in placed_units:
        t = u.get("type", "unit")
        type_counts[t] = type_counts.get(t, 0) + 1

    # Build a flat ordered list of type slots: larger-target types first.
    def _target(t: str) -> float:
        override = unit_specs_cfg.get(t, {}) if isinstance(unit_specs_cfg, dict) else {}
        return float(override.get("target_area_m2") or
                     DEFAULT_TARGET_AREA_M2_BY_TYPE.get(t, 60.0))

    type_order = sorted(type_counts.keys(), key=_target, reverse=True)
    type_slots: list[str] = []
    for t in type_order:
        type_slots.extend([t] * type_counts[t])

    # Sort pieces by area descending and assign slots in order.
    sorted_units = sorted(placed_units, key=lambda u: u.get("area_m2", 0.0), reverse=True)
    result = []
    for idx, unit in enumerate(sorted_units):
        assigned_type = type_slots[idx] if idx < len(type_slots) else unit.get("type", "unit")
        result.append({**unit, "type": assigned_type})
    return result


def _split_endunits_at_corridor_midline(placed_units: list, corridors: list) -> list:
    """Split end units (outside the bookend corridor) at the corridor centreline.

    In a double-loaded bookend layout the corridor spans only between the inner walls
    of the end columns; those end-column units are untouched by the corridor subtraction
    but must still be halved at the same midline so every unit on the floor plate is
    split into a top half and a bottom half (or left/right for a vertical slab).
    """
    if not corridors or not placed_units:
        return placed_units

    from shapely.geometry import LineString
    from shapely.ops import split as shapely_split

    corr = corridors[0]
    c_xs = [p[0] for p in corr.polygon]
    c_ys = [p[1] for p in corr.polygon]
    corr_min_x, corr_max_x = min(c_xs), max(c_xs)
    corr_min_y, corr_max_y = min(c_ys), max(c_ys)
    corr_cx = (corr_min_x + corr_max_x) / 2.0
    corr_cy = (corr_min_y + corr_max_y) / 2.0
    horiz_corridor = (corr_max_x - corr_min_x) >= (corr_max_y - corr_min_y)

    result = []
    for unit in placed_units:
        poly = unit.get("polygon")
        if not poly:
            result.append(unit)
            continue

        u_xs = [p[0] for p in poly]
        u_ys = [p[1] for p in poly]
        u_min_x, u_max_x = min(u_xs), max(u_xs)
        u_min_y, u_max_y = min(u_ys), max(u_ys)

        # Determine whether this unit sits outside the corridor span (end unit).
        if horiz_corridor:
            # An end unit's x range lies entirely outside the corridor's x span.
            is_end_unit = u_max_x <= corr_min_x + 1.0 or u_min_x >= corr_max_x - 1.0
            # Only split if the unit straddles the corridor midline vertically.
            straddles = u_min_y < corr_cy < u_max_y
        else:
            is_end_unit = u_max_y <= corr_min_y + 1.0 or u_min_y >= corr_max_y - 1.0
            straddles = u_min_x < corr_cx < u_max_x

        if not (is_end_unit and straddles):
            result.append(unit)
            continue

        try:
            unit_shp = polygon_to_shapely(poly)
            if horiz_corridor:
                cut = LineString([(u_min_x - 1, corr_cy), (u_max_x + 1, corr_cy)])
            else:
                cut = LineString([(corr_cx, u_min_y - 1), (corr_cx, u_max_y + 1)])

            parts_geom = shapely_split(unit_shp, cut)
            parts = [g.buffer(0) for g in parts_geom.geoms
                     if g.area > 1e6 and g.geom_type == "Polygon"]
        except Exception:
            result.append(unit)
            continue

        if len(parts) < 2:
            result.append(unit)
            continue

        base_id = unit.get("id") or "u"
        for idx, part in enumerate(parts):
            new_poly = shapely_to_polygon(part)
            result.append({**unit,
                           "id": base_id + f"_{chr(97 + idx)}",
                           "polygon": new_poly,
                           "area_m2": polygon_area(new_poly) / 1e6})
    return result


def _subtract_corridors_from_placed_units(placed_units: list, corridors: list) -> list:
    """Subtract corridor polygons from each placed unit polygon (Shapely).
    Returns a new list of unit dicts with updated polygon and area.
    Units fully consumed by corridors are dropped.
    """
    if not corridors or not placed_units:
        return placed_units

    blocker_union = None
    for corr in corridors:
        shp = polygon_to_shapely(corr.polygon)
        blocker_union = shp if blocker_union is None else blocker_union.union(shp)

    if blocker_union is None or blocker_union.is_empty:
        return placed_units

    result = []
    for unit in placed_units:
        poly = unit.get("polygon")
        if not poly:
            result.append(unit)
            continue
        try:
            unit_shp = polygon_to_shapely(poly)
            clipped = unit_shp.difference(blocker_union)
            if clipped.is_empty:
                continue
            if clipped.geom_type == "MultiPolygon":
                # Keep all significant pieces — this handles double-loaded slabs where
                # the corridor bisects each unit into a top and bottom portion.
                MIN_PIECE_M2 = 5e6  # 5 m² in mm²
                pieces = [g.buffer(0) for g in clipped.geoms
                          if g.area >= MIN_PIECE_M2 and g.geom_type == "Polygon"]
                if not pieces:
                    continue
                for idx, piece in enumerate(pieces):
                    new_poly = shapely_to_polygon(piece)
                    suffix = f"_{chr(97 + idx)}"  # _a, _b, ...
                    new_id = (unit.get("id") or "u") + suffix
                    result.append({**unit, "id": new_id,
                                   "polygon": new_poly,
                                   "area_m2": polygon_area(new_poly) / 1e6})
                continue
            if clipped.geom_type == "Polygon" and not clipped.is_empty:
                new_poly = shapely_to_polygon(clipped.buffer(0))
                unit = {**unit, "polygon": new_poly, "area_m2": polygon_area(new_poly) / 1e6}
        except Exception:
            pass
        result.append(unit)
    return result


def _assign_typed_edges_to_placed_units(
    placed_units: list[dict],
    corridors: list[Corridor],
    boundary: Polygon,
    min_overlap: float = 1200.0,
) -> list[dict]:
    """Give every placed unit a COMPLETE set of typed edges.

    This path previously emitted only the single corridor-facing edge, set as
    a side effect of _compute_unit_entrances. So `external` and `shared_wall`
    never appeared anywhere in the subdivision path: the renderer's edge-type
    colouring drew one stroke per unit and nothing else, and the intertenancy
    case — a habitable room whose only boundary wall is a party wall — was not
    representable at all. Note this is a different gap from the one in
    generate_initial_layout, which does call assign_typed_edges but relied on a
    broken _is_edge_external. Both found 2026-08-18 by testing through the app;
    see projects/project-44/docs/adaptation-process-model.md.

    Corridor-facing edges already identified upstream are preserved verbatim
    rather than re-derived, so the classification cannot drift out of step with
    the entrance_point computed from that same edge.
    """
    out = []
    for unit in placed_units:
        poly = unit.get("polygon") or []
        if len(poly) < 3:
            out.append(unit)
            continue

        corridor_edges = {
            e.get("edge_index")
            for e in (unit.get("typed_edges") or [])
            if e.get("type") == EdgeType.FACING_CORRIDOR.value
        }

        classified = assign_typed_edges_for_polygon(poly, corridors, boundary, min_overlap)
        out.append({
            **unit,
            "typed_edges": [
                {
                    "edge_index": e.edge_index,
                    "type": (EdgeType.FACING_CORRIDOR.value
                             if e.edge_index in corridor_edges else e.type.value),
                }
                for e in classified
            ],
        })
    return out


def generate_subdivision_response(
    boundary: Polygon,
    unit_mix,
    constraints: Optional[dict] = None,
    scoring: Optional[dict] = None,
    mode: str = "single",
    seed: Optional[int] = None,
) -> dict:
    constraints = constraints or {}
    scoring = {**DEFAULT_SCORING, **(scoring or {})}
    boundary = normalize_polygon_winding(boundary, clockwise=False)
    bbox = polygon_bbox(boundary)
    if bbox["width"] <= 0 or bbox["height"] <= 0 or polygon_area(boundary) <= 0:
        return {
            "mode": mode,
            "placed_units": [],
            "residual_zones": [],
            "shortfall": {},
            "warnings": ["Invalid or degenerate boundary polygon."],
            "score": float("-inf"),
            "seed": seed,
            "cores": [],
            "corridors": [],
            "area_check": {
                "input_area_m2": 0.0,
                "output_area_m2": 0.0,
                "discrepancy_m2": 0.0,
                "passed": False,
            },
        }

    adjusted_mix, requested_mix, requested_total, adjusted_total, mix_warnings = _adjust_unit_mix_to_capacity(
        boundary,
        unit_mix,
        constraints,
    )

    expanded_units = _expand_subdivision_request(adjusted_mix, constraints)
    if not expanded_units:
        boundary_area_m2 = polygon_area(boundary) / 1e6
        return {
            "mode": mode,
            "placed_units": [],
            "residual_zones": [{
                "polygon": boundary,
                "area_m2": boundary_area_m2,
                "reason": "mix_constraint",
            }],
            "shortfall": {},
            "warnings": ["No requested units supplied; returning full boundary as residual."],
            "score": -boundary_area_m2,
            "seed": seed,
            "cores": [],
            "corridors": [],
            "area_check": {
                "input_area_m2": boundary_area_m2,
                "output_area_m2": boundary_area_m2,
                "discrepancy_m2": 0.0,
                "passed": True,
            },
            "mix_adjustment": {
                "requested_mix": requested_mix,
                "effective_mix": adjusted_mix,
                "requested_total": requested_total,
                "effective_total": adjusted_total,
            },
        }

    # Always subdivide the full boundary.
    # Corridor (when requested) is computed AFTER placement so it overlaps the
    # inner face of the placed units; boolean subtraction gives net unit areas.
    # Cores are not generated here — deferred to a later stage.
    cores, corridors, apartment_zones, circ_log = [], [], [boundary], []
    cores_out = []  # cores deferred; corridors_out built after corridor computation below

    type_priority = scoring.get("type_priority", {}) if isinstance(scoring, dict) else {}
    orders = _candidate_orderings(expanded_units, type_priority, seed)
    orientations = ["auto", "horizontal", "vertical"]
    residual_slots_options = [0]
    candidates = []
    tried = 0
    discarded = 0
    max_candidates = int(constraints.get("max_candidates", 16) or 16)

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(orientations)
        rng.shuffle(residual_slots_options)

    for order in orders:
        ordered_mix = {}
        # Rebuild the requested mix preserving order and counts for the candidate.
        for spec in order:
            ordered_mix.setdefault(spec["type"], 0)
            ordered_mix[spec["type"]] += 1
        for orientation in orientations:
            for residual_slots in residual_slots_options:
                tried += 1
                candidate = _evaluate_over_zones(
                    apartment_zones=apartment_zones,
                    boundary=boundary,
                    unit_mix=ordered_mix,
                    constraints=constraints,
                    scoring=scoring,
                    seed=seed,
                    orientation=orientation,
                    residual_slots=residual_slots,
                )
                if not candidate["hard_valid"]:
                    discarded += 1
                    continue
                candidates.append(candidate)
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    candidates.sort(key=lambda item: item["score"], reverse=True)
    top_candidates = candidates[:3]
    best = top_candidates[0] if top_candidates else _evaluate_over_zones(
        apartment_zones=apartment_zones,
        boundary=boundary,
        unit_mix=expanded_units,
        constraints=constraints,
        scoring=scoring,
        seed=seed,
        orientation="auto",
        residual_slots=0,
    )

    # Compute corridor from placed units (post-subdivision, overlaps units).
    typology = None
    if constraints.get("corridor_width"):
        try:
            corridor_width_mm = float(constraints["corridor_width"])
            analysis = analyze_boundary(boundary)
            normalized_mix = _normalize_subdivision_unit_mix(unit_mix)
            distribution_breakdown = compute_unit_distribution_breakdown(
                analysis, normalized_mix,
                float(constraints.get("min_unit_depth") or
                      (float(constraints.get("min_depth_m") or 0) * 1000) or 7000.0),
            )
            typology = infer_typology(analysis, normalized_mix, distribution_breakdown)
            corridors = _compute_corridor_from_placed_units(
                best.get("placed_units", []), analysis, typology, corridor_width_mm,
            )
            sub_type_str = typology.sub_type.value if hasattr(typology.sub_type, "value") else str(typology.sub_type)
            circ_log.append(
                f"Corridor (post-subdivision): typology={typology.type.value}/{sub_type_str}, "
                f"width={corridor_width_mm:.0f}mm, units={len(best.get('placed_units', []))}"
            )
        except Exception as exc:
            circ_log.append(f"Corridor post-subdivision: failed ({exc})")

    # Step 5: boolean subtraction — remove corridor footprint from each placed unit.
    if corridors:
        best["placed_units"] = _subtract_corridors_from_placed_units(
            best.get("placed_units", []), corridors
        )
        # For double-loaded bookend, end units are not touched by the corridor;
        # split them at the corridor midline so every unit is halved.
        if typology is not None and typology.sub_type == SlabSubType.DOUBLE_LOADED:
            best["placed_units"] = _split_endunits_at_corridor_midline(
                best.get("placed_units", []), corridors
            )
        # Re-assign type labels so larger pieces get larger-target types,
        # since bisection changes the area distribution vs. pre-corridor assignment.
        best["placed_units"] = _reassign_types_by_area(
            best.get("placed_units", []), constraints
        )
        # Compute entrance point (corridor-facing edge midpoint) for each unit.
        best["placed_units"] = _compute_unit_entrances(
            best.get("placed_units", []), corridors
        )
        # Complete the edge typing. _compute_unit_entrances marks only the
        # corridor-facing edge, so without this every other edge — facade and
        # party wall alike — is left unclassified and indistinguishable.
        best["placed_units"] = _assign_typed_edges_to_placed_units(
            best.get("placed_units", []), corridors, boundary
        )

    corridors_out = [
        {
            "id": c.id,
            "polygon": c.polygon,
            "width": c.width,
            "centreline": [list(p) for p in c.centreline],
        }
        for c in corridors
    ]

    warnings = list(best.get("warnings", []))
    warnings.extend(mix_warnings)
    warnings.extend(circ_log)
    if not best["area_check"]["passed"]:
        warnings.append("Area check failed beyond tolerance.")

    if mode == "harness":
        return {
            "mode": "harness",
            "candidates": top_candidates,
            "best": best,
            "parameter_sets_tried": tried,
            "parameter_sets_discarded": discarded,
            "warnings": warnings,
            "cores": cores_out,
            "corridors": corridors_out,
            "mix_adjustment": {
                "requested_mix": requested_mix,
                "effective_mix": adjusted_mix,
                "requested_total": requested_total,
                "effective_total": adjusted_total,
            },
        }

    return {
        "mode": "single",
        **{k: v for k, v in best.items() if k not in {"hard_valid", "candidate_meta"}},
        "warnings": warnings,
        "cores": cores_out,
        "corridors": corridors_out,
        "mix_adjustment": {
            "requested_mix": requested_mix,
            "effective_mix": adjusted_mix,
            "requested_total": requested_total,
            "effective_total": adjusted_total,
        },
    }


def infer_typology(
    analysis: dict,
    unit_mix: Optional[list[dict]] = None,
    distribution_breakdown: Optional[dict] = None,
) -> Typology:
    """
    Stage 2: Determine building typology from boundary analysis.
    
    Args:
        analysis: Result from analyze_boundary().
    
    Returns:
        Typology object with type, sub_type, confidence, reasoning.
    """
    aspect = analysis["aspect_ratio"]
    short_dim = analysis["short_dimension"]

    breakdown = distribution_breakdown or {}
    accommodated_units = breakdown.get("accommodated_units", _estimate_accommodable_unit_count(analysis, unit_mix))

    if breakdown:
        rec_type = breakdown.get("recommended_typology")
        rec_sub = breakdown.get("recommended_sub_type")
        if rec_type == TypologyType.POINT_BLOCK.value:
            return Typology(
                type=TypologyType.POINT_BLOCK,
                sub_type="central_core",
                confidence=ConfidenceLevel.HIGH,
                reasoning=f"Distribution strategy: {breakdown.get('scenario')} ({breakdown.get('notes', '')})",
            )
        if rec_type == TypologyType.SLAB.value:
            if rec_sub == SlabSubType.DOUBLE_LOADED.value:
                sub_type = SlabSubType.DOUBLE_LOADED
            else:
                sub_type = SlabSubType.SINGLE_LOADED
            return Typology(
                type=TypologyType.SLAB,
                sub_type=sub_type,
                confidence=ConfidenceLevel.HIGH,
                reasoning=f"Distribution strategy: {breakdown.get('scenario')} ({breakdown.get('notes', '')})",
            )

    if accommodated_units <= 2:
        typology_type = TypologyType.POINT_BLOCK
        sub_type = "central_core"
        confidence = ConfidenceLevel.HIGH
        reasoning = (
            f"Capacity-driven override: estimated max {accommodated_units} unit(s) "
            "→ point tower with compact circulation"
        )
    elif aspect >= 2.5:
        typology_type = TypologyType.SLAB
        
        if short_dim >= 14000:
            sub_type = SlabSubType.DOUBLE_LOADED
            confidence = ConfidenceLevel.HIGH
        elif short_dim >= 8000:
            sub_type = SlabSubType.SINGLE_LOADED
            confidence = ConfidenceLevel.HIGH
        else:
            sub_type = SlabSubType.SINGLE_LOADED
            confidence = ConfidenceLevel.LOW
            reasoning = f"Aspect {aspect:.1f}:1; width {short_dim:.0f}mm < 8000mm (too narrow for standard unit depth)"
    
    elif aspect >= 1.4:
        typology_type = TypologyType.SLAB
        
        if short_dim >= 14000:
            sub_type = SlabSubType.DOUBLE_LOADED
            confidence = ConfidenceLevel.MEDIUM
            reasoning = f"Aspect {aspect:.1f}:1 (borderline); width {short_dim:.0f}mm supports double-loaded"
        else:
            sub_type = SlabSubType.SINGLE_LOADED
            confidence = ConfidenceLevel.MEDIUM
            reasoning = f"Aspect {aspect:.1f}:1 (borderline); width {short_dim:.0f}mm narrows single-loaded"
    
    else:  # aspect < 1.4
        typology_type = TypologyType.POINT_BLOCK
        sub_type = "central_core"
        confidence = ConfidenceLevel.HIGH if aspect < 1.2 else ConfidenceLevel.MEDIUM
        reasoning = f"Aspect {aspect:.1f}:1 (near-square) → point block typology"
    
    # Fallback reasoning for standard cases
    if "reasoning" not in locals():
        reasoning = f"Aspect {aspect:.1f}:1, width {short_dim:.0f}mm"
    
    return Typology(
        type=typology_type,
        sub_type=sub_type,
        confidence=confidence,
        reasoning=reasoning,
    )


# ── Stage 3: Core Specification ───────────────────────────────────────────────

def specify_core(core_type: CoreType) -> dict:
    """
    Stage 3: Look up core dimensions by type.
    
    Args:
        core_type: CoreType enum value.
    
    Returns:
        Dict with width, depth, min_area.
    """
    return CORE_DIMENSIONS[core_type]


# ── Stage 4: Core Placement ───────────────────────────────────────────────────

def place_cores(
    boundary: Polygon,
    analysis: dict,
    typology: Typology,
    core_type: CoreType,
    gen_log: list,
) -> list[Core]:
    """
    Stage 4: Position cores within the boundary.
    
    For slabs: distribute along long axis.
    For point blocks: center on boundary centroid.
    
    Args:
        boundary: Polygon vertices.
        analysis: Result from analyze_boundary().
        typology: Result from infer_typology().
        core_type: Core type to use.
        gen_log: Generation log list (updated in-place).
    
    Returns:
        List of Core objects.
    """
    cores = []
    spec = specify_core(core_type)
    width, depth = spec["width"], spec["depth"]
    
    bbox = analysis["bbox"]
    long_dim = analysis["long_dimension"]
    short_dim = analysis["short_dimension"]
    long_axis = analysis["long_axis"]
    centroid = analysis["centroid"]
    
    if typology.type == TypologyType.SLAB:
        # Core count based on long dimension
        core_count = max(1, ceil(long_dim / 60000))
        
        if core_count == 1:
            # Single core centered on long axis
            if long_axis == "horizontal":
                core_x = centroid[0]
                core_y = centroid[1] - depth / 2
            else:
                core_x = centroid[0] - depth / 2
                core_y = centroid[1]
            
            core = Core(
                id="core-0",
                polygon=_rectangle_polygon(core_x, core_y, width, depth),
                type=core_type,
                centroid=(core_x + width / 2, core_y + depth / 2),
                width=width,
                depth=depth,
            )
            cores.append(core)
            gen_log.append(f"Core: 1 core placed at ({core_x:.0f}, {core_y:.0f})")
        else:
            # Multiple cores distributed along long axis
            for i in range(core_count):
                t = (i + 1) / (core_count + 1)  # Position factor
                
                if long_axis == "horizontal":
                    core_x = bbox["min_x"] + t * long_dim
                    core_y = centroid[1] - depth / 2
                else:
                    core_x = centroid[0] - depth / 2
                    core_y = bbox["min_y"] + t * long_dim
                
                core = Core(
                    id=f"core-{i}",
                    polygon=_rectangle_polygon(core_x, core_y, width, depth),
                    type=core_type,
                    centroid=(core_x + width / 2, core_y + depth / 2),
                    width=width,
                    depth=depth,
                )
                cores.append(core)
            
            gen_log.append(f"Core: {core_count} cores distributed along {long_axis} axis")
    
    else:  # POINT_BLOCK
        # Single centered core
        core_x = centroid[0] - width / 2
        core_y = centroid[1] - depth / 2
        
        core = Core(
            id="core-0",
            polygon=_rectangle_polygon(core_x, core_y, width, depth),
            type=core_type,
            centroid=centroid,
            width=width,
            depth=depth,
        )
        cores.append(core)
        gen_log.append(f"Core: Single centered core at ({centroid[0]:.0f}, {centroid[1]:.0f})")
    
    return cores


# ── Stage 5: Corridor Derivation ──────────────────────────────────────────────

def derive_corridor(
    boundary: Polygon,
    analysis: dict,
    typology: Typology,
    cores: list[Core],
    corridor_width: float,
    distribution_breakdown: Optional[dict],
    gen_log: list,
) -> list[Corridor]:
    """
    Stage 5: Create corridor polygon(s).
    
    For double-loaded slab: corridor strip down center.
    For single-loaded slab: corridor strip along one edge.
    For point block: landing zone around core (deferred).
    
    Args:
        boundary: Polygon vertices.
        analysis: Result from analyze_boundary().
        typology: Result from infer_typology().
        corridor_width: Width in mm.
        gen_log: Generation log list (updated in-place).
    
    Returns:
        List of Corridor objects.
    """
    corridors = []
    bbox = analysis["bbox"]
    long_dim = analysis["long_dimension"]
    short_dim = analysis["short_dimension"]
    long_axis = analysis["long_axis"]
    
    if typology.type == TypologyType.SLAB:
        # Corridor runs along long axis
        core_bbox = polygon_bbox(cores[0].polygon) if cores else None

        if typology.sub_type == SlabSubType.DOUBLE_LOADED:
            # Corridor is tangential to core: adjacent edge contact, no overlap.
            if long_axis == "horizontal":
                x_start = bbox["min_x"]
                if core_bbox and (core_bbox["max_y"] + corridor_width) <= bbox["max_y"]:
                    y_start = core_bbox["max_y"]
                elif core_bbox and (core_bbox["min_y"] - corridor_width) >= bbox["min_y"]:
                    y_start = core_bbox["min_y"] - corridor_width
                else:
                    y_start = bbox["min_y"] + (short_dim / 2) - (corridor_width / 2)
                x_end = bbox["max_x"]
                y_end = y_start + corridor_width
            else:
                if core_bbox and (core_bbox["max_x"] + corridor_width) <= bbox["max_x"]:
                    x_start = core_bbox["max_x"]
                elif core_bbox and (core_bbox["min_x"] - corridor_width) >= bbox["min_x"]:
                    x_start = core_bbox["min_x"] - corridor_width
                else:
                    x_start = bbox["min_x"] + (short_dim / 2) - (corridor_width / 2)
                y_start = bbox["min_y"]
                x_end = x_start + corridor_width
                y_end = bbox["max_y"]
            
            corridor_poly = [
                (x_start, y_start),
                (x_end, y_start),
                (x_end, y_end),
                (x_start, y_end),
            ]
            
            centx = (x_start + x_end) / 2
            centy = (y_start + y_end) / 2
            
            if long_axis == "horizontal":
                centreline = [(x_start, centy), (x_end, centy)]
            else:
                centreline = [(centx, y_start), (centx, y_end)]
        
        else:  # SINGLE_LOADED
            # Corridor at the inner (back) face, spanning the full slab length.
            # Units occupy the zone between the exterior edge and the corridor.
            # Tangential adjustment when a core occupies the default inner position.
            if long_axis == "horizontal":
                x_start = bbox["min_x"]
                x_end = bbox["max_x"]
                # Default: corridor flush against the inner (max_y) edge.
                y_start = bbox["max_y"] - corridor_width
                if core_bbox:
                    if (core_bbox["min_y"] - corridor_width) >= bbox["min_y"]:
                        y_start = core_bbox["min_y"] - corridor_width
                    elif (core_bbox["max_y"] + corridor_width) <= bbox["max_y"]:
                        y_start = core_bbox["max_y"]
                y_end = y_start + corridor_width
            else:
                y_start = bbox["min_y"]
                y_end = bbox["max_y"]
                # Default: corridor flush against the inner (max_x) edge.
                x_start = bbox["max_x"] - corridor_width
                if core_bbox:
                    if (core_bbox["min_x"] - corridor_width) >= bbox["min_x"]:
                        x_start = core_bbox["min_x"] - corridor_width
                    elif (core_bbox["max_x"] + corridor_width) <= bbox["max_x"]:
                        x_start = core_bbox["max_x"]
                x_end = x_start + corridor_width
            
            corridor_poly = [
                (x_start, y_start),
                (x_end, y_start),
                (x_end, y_end),
                (x_start, y_end),
            ]
            
            centx = (x_start + x_end) / 2
            centy = (y_start + y_end) / 2
            
            if long_axis == "horizontal":
                centreline = [(x_start, centy), (x_end, centy)]
            else:
                centreline = [(centx, y_start), (centx, y_end)]
        
        corridor = Corridor(
            id="corr-0",
            polygon=corridor_poly,
            width=corridor_width,
            centreline=centreline,
        )
        corridors.append(corridor)
        gen_log.append(
            f"Corridor: {long_axis} strip, {corridor_width}mm wide, {typology.sub_type} "
            "(tangential to core where feasible)"
        )
    
    else:  # POINT_BLOCK
        core_bbox = polygon_bbox(cores[0].polygon) if cores else None
        if core_bbox is None:
            gen_log.append("Corridor: Deferred for point block (no core available)")
            return corridors

        if distribution_breakdown and distribution_breakdown.get("scenario") == "two_unit_point_tower":
            corridor_width = min(corridor_width, 1200)

        # Create a compact landing tangential to the core so circulation remains local.
        if long_axis == "horizontal":
            y_start = max(bbox["min_y"], core_bbox["min_y"] - corridor_width)
            if y_start + corridor_width > core_bbox["min_y"]:
                y_start = min(core_bbox["max_y"], bbox["max_y"] - corridor_width)
            x_start = core_bbox["min_x"]
            x_end = core_bbox["max_x"]
            corridor_poly = [
                (x_start, y_start),
                (x_end, y_start),
                (x_end, y_start + corridor_width),
                (x_start, y_start + corridor_width),
            ]
            centreline = [(x_start, y_start + corridor_width / 2), (x_end, y_start + corridor_width / 2)]
        else:
            x_start = max(bbox["min_x"], core_bbox["min_x"] - corridor_width)
            if x_start + corridor_width > core_bbox["min_x"]:
                x_start = min(core_bbox["max_x"], bbox["max_x"] - corridor_width)
            y_start = core_bbox["min_y"]
            y_end = core_bbox["max_y"]
            corridor_poly = [
                (x_start, y_start),
                (x_start + corridor_width, y_start),
                (x_start + corridor_width, y_end),
                (x_start, y_end),
            ]
            centreline = [(x_start + corridor_width / 2, y_start), (x_start + corridor_width / 2, y_end)]

        corridor = Corridor(
            id="corr-0",
            polygon=corridor_poly,
            width=corridor_width,
            centreline=centreline,
        )
        corridors.append(corridor)
        gen_log.append("Corridor: compact point-block landing placed tangentially to core")
    
    return corridors


# ── Stage 6: Apartment Zone Computation ───────────────────────────────────────

def compute_apartment_zones(
    boundary: Polygon,
    analysis: dict,
    typology: Typology,
    cores: list[Core],
    corridors: list[Corridor],
    corridor_width: float,
    distribution_breakdown: Optional[dict],
    gen_log: list,
) -> list[Polygon]:
    """
    Stage 6: Compute regions available for unit placement.
    
    For double-loaded: zones north and south of corridor.
    For single-loaded: zone opposite the corridor.
    For point block: zones around landing perimeter (deferred).
    
    Args:
        boundary: Polygon vertices.
        analysis: Result from analyze_boundary().
        typology: Result from infer_typology().
        corridors: Corridor objects from Stage 5.
        corridor_width: Width in mm.
        gen_log: Generation log list (updated in-place).
    
    Returns:
        List of apartment zone polygons.
    """
    zones = []
    bbox = analysis["bbox"]
    long_dim = analysis["long_dimension"]
    short_dim = analysis["short_dimension"]
    long_axis = analysis["long_axis"]
    
    if distribution_breakdown and distribution_breakdown.get("scenario") == "two_unit_point_tower":
        # Force a 2-zone split along the long axis of the floor plate.
        if long_axis == "horizontal":
            split_x = bbox["min_x"] + (bbox["width"] / 2)
            zone_a = [
                (bbox["min_x"], bbox["min_y"]),
                (split_x, bbox["min_y"]),
                (split_x, bbox["max_y"]),
                (bbox["min_x"], bbox["max_y"]),
            ]
            zone_b = [
                (split_x, bbox["min_y"]),
                (bbox["max_x"], bbox["min_y"]),
                (bbox["max_x"], bbox["max_y"]),
                (split_x, bbox["max_y"]),
            ]
        else:
            split_y = bbox["min_y"] + (bbox["height"] / 2)
            zone_a = [
                (bbox["min_x"], bbox["min_y"]),
                (bbox["max_x"], bbox["min_y"]),
                (bbox["max_x"], split_y),
                (bbox["min_x"], split_y),
            ]
            zone_b = [
                (bbox["min_x"], split_y),
                (bbox["max_x"], split_y),
                (bbox["max_x"], bbox["max_y"]),
                (bbox["min_x"], bbox["max_y"]),
            ]
        zones.extend([zone_a, zone_b])
        gen_log.append("Zones: two-unit point tower split in half along long axis")
        return zones

    if typology.type == TypologyType.SLAB:
        corridor = corridors[0]
        corridor_bbox = polygon_bbox(corridor.polygon)
        
        if typology.sub_type == SlabSubType.DOUBLE_LOADED:
            # Zone A: above corridor
            if long_axis == "horizontal":
                zone_a = [
                    (bbox["min_x"], bbox["min_y"]),
                    (bbox["max_x"], bbox["min_y"]),
                    (bbox["max_x"], corridor_bbox["min_y"]),
                    (bbox["min_x"], corridor_bbox["min_y"]),
                ]
                zone_b = [
                    (bbox["min_x"], corridor_bbox["max_y"]),
                    (bbox["max_x"], corridor_bbox["max_y"]),
                    (bbox["max_x"], bbox["max_y"]),
                    (bbox["min_x"], bbox["max_y"]),
                ]
            else:
                zone_a = [
                    (bbox["min_x"], bbox["min_y"]),
                    (corridor_bbox["min_x"], bbox["min_y"]),
                    (corridor_bbox["min_x"], bbox["max_y"]),
                    (bbox["min_x"], bbox["max_y"]),
                ]
                zone_b = [
                    (corridor_bbox["max_x"], bbox["min_y"]),
                    (bbox["max_x"], bbox["min_y"]),
                    (bbox["max_x"], bbox["max_y"]),
                    (corridor_bbox["max_x"], bbox["max_y"]),
                ]
            
            zones.append(zone_a)
            zones.append(zone_b)
            
            zone_a_depth = polygon_bbox(zone_a)[("height" if long_axis == "horizontal" else "width")]
            zone_b_depth = polygon_bbox(zone_b)[("height" if long_axis == "horizontal" else "width")]
            gen_log.append(f"Zones: double-loaded, depth A={zone_a_depth:.0f}mm, depth B={zone_b_depth:.0f}mm")
        
        else:  # SINGLE_LOADED
            # Unit zone is on the exterior (near) side of the corridor.
            if long_axis == "horizontal":
                zone = [
                    (bbox["min_x"], bbox["min_y"]),
                    (bbox["max_x"], bbox["min_y"]),
                    (bbox["max_x"], corridor_bbox["min_y"]),
                    (bbox["min_x"], corridor_bbox["min_y"]),
                ]
            else:
                zone = [
                    (bbox["min_x"], bbox["min_y"]),
                    (corridor_bbox["min_x"], bbox["min_y"]),
                    (corridor_bbox["min_x"], bbox["max_y"]),
                    (bbox["min_x"], bbox["max_y"]),
                ]

            zones.append(zone)
            zone_depth = polygon_bbox(zone)[("height" if long_axis == "horizontal" else "width")]
            gen_log.append(f"Zones: single-loaded, depth={zone_depth:.0f}mm")
    
    else:  # POINT_BLOCK
        core_bbox = polygon_bbox(cores[0].polygon) if cores else None
        corridor_bbox = polygon_bbox(corridors[0].polygon) if corridors else None

        if core_bbox is None or corridor_bbox is None:
            gen_log.append("Zones: Deferred for point block")
            return zones

        # For point towers, keep circulation compact and place units on the opposite sides
        # of the central core/landing block rather than edge-to-edge.
        if long_axis == "horizontal":
            top_limit = min(corridor_bbox["min_y"], core_bbox["min_y"])
            bottom_limit = max(corridor_bbox["max_y"], core_bbox["max_y"])

            top_zone = [
                (bbox["min_x"], bbox["min_y"]),
                (bbox["max_x"], bbox["min_y"]),
                (bbox["max_x"], top_limit),
                (bbox["min_x"], top_limit),
            ]
            bottom_zone = [
                (bbox["min_x"], bottom_limit),
                (bbox["max_x"], bottom_limit),
                (bbox["max_x"], bbox["max_y"]),
                (bbox["min_x"], bbox["max_y"]),
            ]
            zones.extend([top_zone, bottom_zone])
        else:
            left_limit = min(corridor_bbox["min_x"], core_bbox["min_x"])
            right_limit = max(corridor_bbox["max_x"], core_bbox["max_x"])

            left_zone = [
                (bbox["min_x"], bbox["min_y"]),
                (left_limit, bbox["min_y"]),
                (left_limit, bbox["max_y"]),
                (bbox["min_x"], bbox["max_y"]),
            ]
            right_zone = [
                (right_limit, bbox["min_y"]),
                (bbox["max_x"], bbox["min_y"]),
                (bbox["max_x"], bbox["max_y"]),
                (right_limit, bbox["max_y"]),
            ]
            zones.extend([left_zone, right_zone])

        # Order zones so larger zones are tried first.
        zones.sort(key=lambda poly: polygon_area(poly), reverse=True)
        zone_desc = ", ".join(f"{polygon_bbox(z)['width']:.0f}x{polygon_bbox(z)['height']:.0f}mm" for z in zones)
        gen_log.append(f"Zones: point-block compact zones = {zone_desc}")
    
    return zones


# ── Helper: Rectangle polygon ─────────────────────────────────────────────────

def _rectangle_polygon(x: float, y: float, w: float, h: float) -> Polygon:
    """Create a rectangle polygon at (x, y) with width w and height h."""
    return [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h),
    ]


# ── Stage 7: Unit Bay Subdivision ─────────────────────────────────────────────

def compute_unit_bay_widths(
    unit_mix: list[dict],
    zone_depth: float,
) -> list[dict]:
    """
    Compute target bay width for each unit based on target area and zone depth.
    
    Args:
        unit_mix: List of unit dicts with 'type', 'count', 'target_area'.
        zone_depth: Depth of the zone in mm.
    
    Returns:
        List of {'type', 'count', 'target_width', 'min_width', 'max_width'}.
    """
    MIN_BAY_WIDTH = 4500  # mm
    MAX_BAY_WIDTH = 9000  # mm
    
    result = []
    
    for unit in unit_mix:
        unit_type = unit.get('type', 'unit')
        count = unit.get('count', 0)
        target_area = unit.get('target_area', 60)  # m²
        
        # Convert area from m² to mm²
        target_area_mm2 = target_area * 1e6
        
        # Compute bay width: area / depth
        bay_width = target_area_mm2 / zone_depth if zone_depth > 0 else 0
        
        # Clamp to min/max
        bay_width = max(MIN_BAY_WIDTH, min(bay_width, MAX_BAY_WIDTH))
        
        result.append({
            'type': unit_type,
            'count': count,
            'target_width': bay_width,
            'min_width': MIN_BAY_WIDTH,
            'max_width': MAX_BAY_WIDTH,
        })
    
    return result


def place_units_in_zone(
    zone: Polygon,
    unit_specs: list[dict],
    analysis: dict,
    corridors: list[Corridor],
    gen_log: list,
) -> tuple[list[Unit], list[Polygon]]:
    """
    Place units sequentially in an apartment zone.
    
    Args:
        zone: Zone polygon (rectangular for initial implementation).
        unit_specs: Output from compute_unit_bay_widths().
        analysis: Result from analyze_boundary().
        corridors: Corridor objects (for edge classification later).
        gen_log: Generation log list (updated in-place).
    
    Returns:
        (list of Unit objects, list of unresolved region polygons)
    """
    units = []
    unresolved = []
    
    zone_bbox = polygon_bbox(zone)
    zone_depth = zone_bbox[("height" if analysis["long_axis"] == "horizontal" else "width")]
    zone_width = zone_bbox[("width" if analysis["long_axis"] == "horizontal" else "height")]
    
    # Start placement at zone corner
    x_cursor = zone_bbox["min_x"] if analysis["long_axis"] == "horizontal" else zone_bbox["min_y"]
    
    unit_id_counter = 0
    total_units = sum(spec['count'] for spec in unit_specs)
    
    for spec in unit_specs:
        unit_type = spec['type']
        count = spec['count']
        bay_width = spec['target_width']
        
        for _ in range(count):
            # Check if unit fits
            if x_cursor + bay_width > (zone_bbox["max_x"] if analysis["long_axis"] == "horizontal" else zone_bbox["max_y"]):
                # Unit doesn't fit; add to unresolved
                remaining_width = (zone_bbox["max_x"] if analysis["long_axis"] == "horizontal" else zone_bbox["max_y"]) - x_cursor
                if remaining_width > 500:
                    unresolved_poly = _rectangle_polygon_in_zone(
                        zone_bbox, x_cursor, remaining_width, zone_depth, analysis["long_axis"]
                    )
                    unresolved.append(unresolved_poly)
                break
            
            # Create unit polygon
            if analysis["long_axis"] == "horizontal":
                unit_poly = _rectangle_polygon(x_cursor, zone_bbox["min_y"], bay_width, zone_depth)
                x_cursor += bay_width
            else:
                unit_poly = _rectangle_polygon(zone_bbox["min_x"], x_cursor, zone_depth, bay_width)
                x_cursor += bay_width
            
            # Compute unit area
            unit_area = polygon_area(unit_poly) / 1e6  # Convert to m²
            
            # Create unit (without typed edges for now - done in Stage 8)
            unit = Unit(
                id=f"unit-{unit_id_counter}",
                type=unit_type,
                polygon=unit_poly,
                typed_edges=[],
                entrance_point=(0, 0),  # Will be set in Stage 8
                depth=zone_depth,
                area=unit_area,
            )
            units.append(unit)
            unit_id_counter += 1
    
    # Handle residual space
    if analysis["long_axis"] == "horizontal":
        residual = zone_bbox["max_x"] - x_cursor
    else:
        residual = zone_bbox["max_y"] - x_cursor
    
    if residual > 500 and residual < 4500:  # Too small for a unit, distribute to existing units
        if units and residual > 10:
            # Add residual proportionally to each unit
            per_unit = residual / len(units)
            for unit in units:
                # Widen each unit
                if analysis["long_axis"] == "horizontal":
                    unit.polygon = [
                        unit.polygon[0],
                        (unit.polygon[1][0] + per_unit, unit.polygon[1][1]),
                        (unit.polygon[2][0] + per_unit, unit.polygon[2][1]),
                        unit.polygon[3],
                    ]
                else:
                    unit.polygon = [
                        unit.polygon[0],
                        unit.polygon[1],
                        (unit.polygon[2][0], unit.polygon[2][1] + per_unit),
                        (unit.polygon[3][0], unit.polygon[3][1] + per_unit),
                    ]
    elif residual >= 4500:  # Large enough to be a unit
        if analysis["long_axis"] == "horizontal":
            residual_poly = _rectangle_polygon(x_cursor, zone_bbox["min_y"], residual, zone_depth)
        else:
            residual_poly = _rectangle_polygon(zone_bbox["min_x"], x_cursor, zone_depth, residual)
        unresolved.append(residual_poly)
        gen_log.append(f"Unresolved: residual space {residual:.0f}mm × {zone_depth:.0f}mm")
    
    return units, unresolved


def _rectangle_polygon_in_zone(
    zone_bbox: dict,
    x_or_y: float,
    width: float,
    depth: float,
    long_axis: str,
) -> Polygon:
    """Helper to create a rectangle within a zone (for residual space)."""
    if long_axis == "horizontal":
        return _rectangle_polygon(x_or_y, zone_bbox["min_y"], width, depth)
    else:
        return _rectangle_polygon(zone_bbox["min_x"], x_or_y, depth, width)


# ── Stage 8: Typed Edge Assignment ────────────────────────────────────────────

def assign_typed_edges(
    unit: Unit,
    corridors: list[Corridor],
    boundary: Polygon,
    min_overlap: float = 1200.0,
) -> list[TypedEdge]:
    """
    Classify unit edges as facing_corridor, external, or shared_wall.

    Thin wrapper over assign_typed_edges_for_polygon — the subdivision path
    (generate_subdivision_response) carries units as plain dicts rather than
    Unit dataclasses, so the classification logic is factored onto the polygon.
    """
    return assign_typed_edges_for_polygon(unit.polygon, corridors, boundary, min_overlap)


def assign_typed_edges_for_polygon(
    polygon: Polygon,
    corridors: list[Corridor],
    boundary: Polygon,
    min_overlap: float = 1200.0,
) -> list[TypedEdge]:
    """
    Classify a unit polygon's edges as facing_corridor, external, or shared_wall.

    Args:
        polygon: Unit polygon (mm).
        corridors: Corridor objects (for overlap detection).
        boundary: Building boundary polygon (mm).
        min_overlap: Minimum overlap to be considered facing_corridor (mm).

    Returns:
        List of TypedEdge objects with classifications.
    """
    typed_edges = []

    for edge_idx in range(len(polygon)):
        p1 = polygon[edge_idx]
        p2 = polygon[(edge_idx + 1) % len(polygon)]
        
        # Create a thin rectangle for the edge (perpendicular thickness of 10mm)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = (dx ** 2 + dy ** 2) ** 0.5
        
        if length > 0:
            # Perpendicular direction (rotated 90 degrees)
            perp_x = -dy / length * 5
            perp_y = dx / length * 5
            
            edge_poly = [
                (p1[0] - perp_x, p1[1] - perp_y),
                (p2[0] - perp_x, p2[1] - perp_y),
                (p2[0] + perp_x, p2[1] + perp_y),
                (p1[0] + perp_x, p1[1] + perp_y),
            ]
        else:
            # Degenerate edge, skip
            typed_edges.append(TypedEdge(edge_index=edge_idx, type=EdgeType.SHARED_WALL))
            continue
        
        # Check overlap with any corridor
        max_corridor_overlap = max(
            (edge_intersection_length(edge_poly, corr.polygon) for corr in corridors),
            default=0,
        )
        
        if max_corridor_overlap >= min_overlap:
            edge_type = EdgeType.FACING_CORRIDOR
        elif _is_edge_external(p1, p2, boundary, tolerance=50):
            edge_type = EdgeType.EXTERNAL
        else:
            edge_type = EdgeType.SHARED_WALL
        
        typed_edges.append(TypedEdge(edge_index=edge_idx, type=edge_type))
    
    return typed_edges


def _is_edge_external(p1: tuple, p2: tuple, boundary: Polygon, tolerance: float = 50) -> bool:
    """
    Check whether an edge LIES ALONG the building boundary.

    Samples interior points of the edge and requires every one to sit within
    tolerance of the boundary.

    This previously returned True if EITHER ENDPOINT was within tolerance,
    which misclassified every intertenancy wall in a slab: a party wall runs
    from the corridor to the facade, so one of its endpoints always lands on
    the building perimeter. The consequence was that no unit anywhere ever
    received a shared_wall edge — assign_typed_edges treats shared_wall as the
    fallback branch, so it was unreachable — and the one wall that cannot carry
    a window was labelled as the one that can. Found 2026-08-18; see
    projects/project-44/docs/adaptation-process-model.md.

    Endpoints are deliberately excluded from the sample: they are shared with
    the neighbouring edges, so they carry no information about which edge this
    actually is.
    """
    if not boundary:
        return False

    samples = [
        (p1[0] + (p2[0] - p1[0]) * f, p1[1] + (p2[1] - p1[1]) * f)
        for f in (0.25, 0.5, 0.75)
    ]

    for pt in samples:
        min_dist = min(
            _point_to_segment_distance(pt, boundary[i], boundary[(i + 1) % len(boundary)])
            for i in range(len(boundary))
        )
        if min_dist > tolerance:
            return False

    return True


def _point_to_segment_distance(pt: tuple, seg_start: tuple, seg_end: tuple) -> float:
    """
    Minimum distance from point to line segment.
    """
    x, y = pt
    x1, y1 = seg_start
    x2, y2 = seg_end
    
    # Vector from seg_start to seg_end
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return point_distance(pt, seg_start)
    
    # Parameter t of closest point on line
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))  # Clamp to segment
    
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return point_distance(pt, (closest_x, closest_y))


def assign_entrance_points(units: list[Unit]) -> None:
    """
    Assign entrance points at the midpoint of facing_corridor edges.
    
    Modifies units in-place.
    """
    for unit in units:
        corridor_edge_idx = None
        for typed_edge in unit.typed_edges:
            if typed_edge.type == EdgeType.FACING_CORRIDOR:
                corridor_edge_idx = typed_edge.edge_index
                break
        
        if corridor_edge_idx is not None:
            p1 = unit.polygon[corridor_edge_idx]
            p2 = unit.polygon[(corridor_edge_idx + 1) % len(unit.polygon)]
            entrance = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            unit.entrance_point = entrance


def _subtract_common_and_core_from_units(
    units: list[Unit],
    cores: list[Core],
    corridors: list[Corridor],
    gen_log: list,
) -> list[Unit]:
    """Final cleanup: subtract core/corridor footprints from apartment polygons."""
    blockers = [core.polygon for core in cores] + [corr.polygon for corr in corridors]
    if not blockers:
        return units

    blocker_union = None
    for poly in blockers:
        shp = polygon_to_shapely(poly)
        blocker_union = shp if blocker_union is None else blocker_union.union(shp)

    refined: list[Unit] = []
    for unit in units:
        try:
            unit_shp = polygon_to_shapely(unit.polygon)
            clipped = unit_shp.difference(blocker_union)
            if clipped.is_empty:
                gen_log.append(f"Unit cleanup: {unit.id} removed (fully consumed by common/core subtraction)")
                continue

            if clipped.geom_type == "MultiPolygon":
                clipped = max(clipped.geoms, key=lambda g: g.area)

            clipped = clipped.buffer(0)
            if clipped.is_empty or clipped.geom_type != "Polygon":
                gen_log.append(f"Unit cleanup: {unit.id} kept original polygon (non-polygon subtraction result)")
                refined.append(unit)
                continue

            unit.polygon = shapely_to_polygon(clipped)
            unit.area = polygon_area(unit.polygon) / 1e6
            refined.append(unit)
        except Exception:
            refined.append(unit)

    gen_log.append(f"Unit cleanup: boolean subtraction applied to {len(refined)} units")
    return refined


# ── Import needed for stage 8 ────────────────────────────────────────────────

from .geometry import edge_intersection_length


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_initial_layout(
    boundary: Polygon,
    unit_mix: list[dict],
    constraints: dict,
) -> FloorPlateState:
    """
    Generate an initial floor plate layout from boundary and unit mix.
    
    Args:
        boundary: List of (x, y) vertices defining the site boundary.
                  Will be normalized to CCW winding and translated to origin.
        unit_mix: List of unit type dicts: 
                  [{"type": "1bed", "count": 2, "min_area": 45, "target_area": 55}, ...]
                  Ordered by preference (place preferred types first).
        constraints: Dict with keys:
                     - corridor_width (mm, default 1500)
                     - max_unit_depth (mm, default 8000)
                     - min_entrance_overlap (mm, default 1200)
                     - min_common_area (m², default 20)
                     - max_travel_distance (mm, default 30000)
                     - core_type (str, default "stair_lift")
    
    Returns:
        FloorPlateState: Complete, internally consistent layout
                         (may have violations in compliance_snapshot).
    """
    
    # ── Stage 0: Normalize input ──────────────────────────────────────────────
    gen_log = ["Generating initial layout..."]
    
    # Ensure CCW winding
    boundary = normalize_polygon_winding(boundary, clockwise=False)
    
    # Translate to origin
    bbox_orig = polygon_bbox(boundary)
    boundary = [
        (x - bbox_orig["min_x"], y - bbox_orig["min_y"])
        for x, y in boundary
    ]
    
    gen_log.append(f"Boundary normalized: {len(boundary)} vertices, area {polygon_area(boundary) / 1e6:.1f}m²")
    
    # Extract constraints with defaults
    corridor_width = constraints.get("corridor_width", 1500)
    max_unit_depth = constraints.get("max_unit_depth", 8000)
    min_entrance_overlap = constraints.get("min_entrance_overlap", 1200)
    min_common_area = constraints.get("min_common_area", 20)
    max_travel_distance = constraints.get("max_travel_distance", 30000)
    core_type_str = constraints.get("core_type", "stair_lift")
    subdivision_only = bool(constraints.get("subdivision_only", False))
    
    # Convert core_type string to enum
    core_type = CoreType(core_type_str)
    
    # ── STAGES 1-6 ────────────────────────────────────────────────────────────
    
    # Stage 1: Boundary analysis
    analysis = analyze_boundary(boundary)
    gen_log.append(
        f"Boundary: aspect {analysis['aspect_ratio']:.2f}:1, "
        f"long {analysis['long_axis']} {analysis['long_dimension']:.0f}mm"
    )
    gen_log.append(f"Boundary decomposition: {analysis['decomposition_count']} sub-polygon(s)")
    
    # Stage 2: Typology inference
    min_unit_depth = constraints.get("min_unit_depth", 7000)
    distribution_breakdown = compute_unit_distribution_breakdown(
        analysis,
        unit_mix,
        min_unit_depth_mm=min_unit_depth,
    )
    gen_log.append(
        f"Distribution: {distribution_breakdown.get('scenario')} | "
        f"accommodated_units={distribution_breakdown.get('accommodated_units')}"
    )

    typology = infer_typology(
        analysis,
        unit_mix=unit_mix,
        distribution_breakdown=distribution_breakdown,
    )
    gen_log.append(f"Typology: {typology.type.value}/{typology.sub_type} ({typology.confidence.value})")
    
    if subdivision_only:
        cores = []
        corridors = []
        gen_log.append(
            "Subdivision-only mode: skipped core/corridor/common-space generation; "
            "partitioning the full floor plate into unit polygons"
        )
    else:
        # Stage 3: Core specification (just lookup)
        core_spec = specify_core(core_type)

        # Stage 4: Core placement
        cores = place_cores(boundary, analysis, typology, core_type, gen_log)

        # Stage 5: Corridor derivation
        corridors = derive_corridor(
            boundary,
            analysis,
            typology,
            cores,
            corridor_width,
            distribution_breakdown,
            gen_log,
        )

        # Stage 6: Apartment zone computation
        apartment_zones = compute_apartment_zones(
            boundary,
            analysis,
            typology,
            cores,
            corridors,
            corridor_width,
            distribution_breakdown,
            gen_log,
        )
    
    
    # ── STAGES 7-9: Unit subdivision and edge assignment ──────────────────────
    
    # Stage 7: Unit subdivision
    if subdivision_only:
        units, all_unresolved = build_subdivision_units_from_plate(
            boundary,
            unit_mix,
            analysis,
            gen_log,
        )
        gen_log.append(f"Partitioned full plate into {len(units)} unit polygon(s)")
    else:
        units = []
        all_unresolved = []
        for zone_idx, zone in enumerate(apartment_zones):
            zone_bbox = polygon_bbox(zone)
            zone_depth = zone_bbox["height" if analysis["long_axis"] == "horizontal" else "width"]
            unit_specs = compute_unit_bay_widths(unit_mix, zone_depth)
            zone_units, unresolved_in_zone = place_units_in_zone(
                zone, unit_specs, analysis, corridors, gen_log
            )
            units.extend(zone_units)
            all_unresolved.extend(unresolved_in_zone)

        gen_log.append(f"Placed {len(units)} units across {len(apartment_zones)} zones")

        # Final geometry cleanup before edge typing.
        units = _subtract_common_and_core_from_units(units, cores, corridors, gen_log)
    
    # Stage 8: Typed edge assignment
    for unit in units:
        unit.typed_edges = assign_typed_edges(
            unit, corridors, boundary, min_entrance_overlap
        )
        if not subdivision_only:
            # Validate: must have exactly 1 corridor-facing edge
            corridor_edges = [
                edge for edge in unit.typed_edges
                if edge.type == EdgeType.FACING_CORRIDOR
            ]
            if len(corridor_edges) != 1:
                gen_log.append(
                    f"Warning: {unit.id} has {len(corridor_edges)} corridor-facing edges "
                    f"(expected 1)"
                )
    
    # Assign entrance points
    if not subdivision_only:
        assign_entrance_points(units)
        gen_log.append("Assigned entrance points to all units")
    else:
        gen_log.append("Subdivision-only mode: skipped corridor-based entrance assignment")
    
    # Stage 9: Structural grid generation (skipped)
    
    # ── Stage 10: Compliance snapshot ─────────────────────────────────────────
    
    floor_plate = FloorPlateState(
        boundary=boundary,
        typology=typology,
        cores=cores,
        corridors=corridors,
        units=units,
        unresolved_regions=all_unresolved,
        analysis={
            "area_mm2": analysis["area"],
            "aspect_ratio": analysis["aspect_ratio"],
            "long_axis": analysis["long_axis"],
            "long_dimension": analysis["long_dimension"],
            "short_dimension": analysis["short_dimension"],
            "boundary_decomposition": analysis.get("decomposition", []),
            "boundary_decomposition_count": analysis.get("decomposition_count", 0),
            "unit_distribution_breakdown": distribution_breakdown,
        },
        generation_log=gen_log,
    )
    
    # Compute compliance snapshot
    compliance_constraints = {
        "min_common_area": min_common_area,
        "max_travel_distance": max_travel_distance,
        "min_entrance_overlap": min_entrance_overlap,
        "max_unit_depth": max_unit_depth,
        "core_min_areas": {ct: CORE_DIMENSIONS[ct]["min_area"] for ct in CoreType},
    }
    
    floor_plate.compliance_snapshot = compute_compliance_snapshot(
        floor_plate,
        compliance_constraints,
    )
    
    return floor_plate
