"""
Geometric operations for polygon manipulation and analysis.

All dimensions in millimetres.
Uses numpy and shapely for robustness.
Refactored from experimental notebook code.
"""

from typing import Optional
import math
from shapely import (
    Polygon as ShapelyPolygon,
    Point as ShapelyPoint,
    LineString as ShapelyLineString,
    ops as shapely_ops,
    distance,
    area as shapely_area,
    length as shapely_length,
)

Polygon = list[tuple[float, float]]


# ── Basic geometry helpers ────────────────────────────────────────────────────

def polygon_area(polygon: Polygon) -> float:
    """
    Compute the area of a polygon using the shoelace formula.
    
    Args:
        polygon: List of (x, y) vertices in order.
    
    Returns:
        Area in square mm.
    """
    if len(polygon) < 3:
        return 0.0
    
    total = 0.0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    
    return abs(total) / 2.0


def polygon_centroid(polygon: Polygon) -> tuple:
    """
    Compute the centroid of a polygon.
    
    Args:
        polygon: List of (x, y) vertices.
    
    Returns:
        (cx, cy) tuple.
    """
    if len(polygon) < 3:
        return (0, 0)
    
    area = 0.0
    cx = 0.0
    cy = 0.0
    
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        cross = x1 * y2 - x2 * y1
        area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    
    if area == 0:
        # Fallback: average of vertices
        return (sum(p[0] for p in polygon) / len(polygon),
                sum(p[1] for p in polygon) / len(polygon))
    
    return (cx / (3 * area), cy / (3 * area))


def polygon_bbox(polygon: Polygon) -> dict:
    """
    Compute bounding box of a polygon.
    
    Args:
        polygon: List of (x, y) vertices.
    
    Returns:
        {"min_x", "max_x", "min_y", "max_y", "width", "height"}
    """
    if not polygon:
        return {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "width": 0, "height": 0}
    
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "width": max_x - min_x,
        "height": max_y - min_y,
    }


def polygon_aspect_ratio(polygon: Polygon) -> float:
    """
    Compute aspect ratio (long dimension / short dimension) of bounding box.
    
    Args:
        polygon: List of (x, y) vertices.
    
    Returns:
        Aspect ratio (>= 1.0).
    """
    bbox = polygon_bbox(polygon)
    long_dim = max(bbox["width"], bbox["height"])
    short_dim = min(bbox["width"], bbox["height"])
    
    if short_dim == 0:
        return 1.0
    
    return long_dim / short_dim


def signed_area(polygon: Polygon) -> float:
    """
    Compute signed area (shoelace formula).
    Positive if CCW, negative if CW (in screen coords, y-down).
    
    Args:
        polygon: List of (x, y) vertices.
    
    Returns:
        Signed area in square mm.
    """
    if len(polygon) < 3:
        return 0.0
    
    total = 0.0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    
    return total / 2.0


def is_clockwise(polygon: Polygon) -> bool:
    """
    Check if polygon is clockwise (in screen coords, y-down).
    
    Args:
        polygon: List of (x, y) vertices.
    
    Returns:
        True if clockwise, False if counter-clockwise.
    """
    return signed_area(polygon) < 0


def normalize_polygon_winding(polygon: Polygon, clockwise: bool = False) -> Polygon:
    """
    Ensure polygon has requested winding order.
    
    Args:
        polygon: List of (x, y) vertices.
        clockwise: If True, ensure CW; if False, ensure CCW.
    
    Returns:
        Polygon with corrected winding.
    """
    if is_clockwise(polygon) == clockwise:
        return polygon  # Already correct
    else:
        return list(reversed(polygon))


# ── Line/edge operations ──────────────────────────────────────────────────────

def lines_are_orthogonal(
    line1_start: tuple,
    line1_end: tuple,
    line2_start: tuple,
    line2_end: tuple,
    tolerance: float = 1e-6,
) -> bool:
    """
    Check if two line segments are orthogonal (perpendicular).
    
    Args:
        line1_start, line1_end: (x, y) endpoints of line 1.
        line2_start, line2_end: (x, y) endpoints of line 2.
        tolerance: Numerical tolerance for slope comparison.
    
    Returns:
        True if lines are orthogonal.
    """
    x1, y1 = line1_start
    x2, y2 = line1_end
    x3, y3 = line2_start
    x4, y4 = line2_end
    
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x4 - x3, y4 - y3
    
    # Check dot product: orthogonal if dot product ≈ 0
    dot_product = dx1 * dx2 + dy1 * dy2
    
    return abs(dot_product) < tolerance


def line_intersection(
    p1: tuple,
    p2: tuple,
    p3: tuple,
    p4: tuple,
) -> Optional[tuple]:
    """
    Find intersection point of two lines (infinite, not segments).
    
    Args:
        p1, p2: Points defining line 1.
        p3, p4: Points defining line 2.
    
    Returns:
        (x, y) intersection point, or None if parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        return None  # Parallel
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return (x, y)


def point_distance(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# ── Polygon operations (using shapely for robustness) ───────────────────────

def polygon_to_shapely(polygon: Polygon) -> ShapelyPolygon:
    """Convert list-of-tuples polygon to shapely Polygon."""
    if len(polygon) < 3:
        raise ValueError("Polygon must have at least 3 vertices")
    return ShapelyPolygon(polygon)


def shapely_to_polygon(shp: ShapelyPolygon) -> Polygon:
    """Convert shapely Polygon to list-of-tuples."""
    return list(shp.exterior.coords[:-1])  # Exclude closing vertex


def polygon_contains_point(polygon: Polygon, point: tuple, strict: bool = False) -> bool:
    """
    Check if point is inside polygon.
    
    Args:
        polygon: List of (x, y) vertices.
        point: (x, y) tuple.
        strict: If True, point must be strictly inside (not on boundary).
    
    Returns:
        True if point is inside.
    """
    shp = polygon_to_shapely(polygon)
    pt = ShapelyPoint(point)
    
    if strict:
        return shp.contains_properly(pt)
    else:
        return shp.contains(pt) or shp.touches(pt)


def edge_intersection_length(
    polygon1: Polygon,
    polygon2: Polygon,
) -> float:
    """
    Compute length of intersection between two polygon edges.
    Uses shapely for robust geometry.
    
    Args:
        polygon1, polygon2: Polygons to check.
    
    Returns:
        Length of intersection in mm.
    """
    shp1 = polygon_to_shapely(polygon1)
    shp2 = polygon_to_shapely(polygon2)
    
    intersection = shp1.intersection(shp2)
    
    if intersection.is_empty:
        return 0.0
    
    return float(shapely_length(intersection))


# ── Polygon splitting (refactored from notebook) ──────────────────────────────

def get_interior_points(polygon: Polygon) -> list[tuple]:
    """
    Find points of a polygon that lie strictly inside its convex hull.
    Helps identify concave regions needing splitting.
    
    Args:
        polygon: List of (x, y) vertices.
    
    Returns:
        List of interior vertices.
    """
    shp = polygon_to_shapely(polygon)
    bbox = shp.convex_hull
    
    interior = []
    for vertex in polygon:
        pt = ShapelyPoint(vertex)
        if bbox.contains_properly(pt):
            interior.append(vertex)
    
    return interior


def polygon_edges(polygon: Polygon) -> list[tuple]:
    """
    Get list of edge tuples (start_pt, end_pt).
    
    Args:
        polygon: List of (x, y) vertices.
    
    Returns:
        List of ((x1, y1), (x2, y2)) edges.
    """
    edges = []
    for i in range(len(polygon)):
        start = polygon[i]
        end = polygon[(i + 1) % len(polygon)]
        edges.append((start, end))
    
    return edges


def split_polygon(polygon: Polygon, max_iterations: int = 10) -> list[Polygon]:
    """
    Recursively split a polygon into convex sub-polygons.
    
    Algorithm:
    1. Find interior points (vertices inside convex hull).
    2. If none, polygon is convex → return it.
    3. For each interior point, find orthogonal line to nearest edge.
    4. Split polygon along this line.
    5. Recursively split resulting sub-polygons.
    
    Args:
        polygon: List of (x, y) vertices (CCW assumed).
        max_iterations: Recursion depth limit (safety).
    
    Returns:
        List of convex polygons.
    """
    completed = []
    
    def _split_recursive(poly, depth):
        if depth > max_iterations:
            # Safety limit reached; return as-is
            completed.append(poly)
            return
        
        interior_pts = get_interior_points(poly)
        
        if not interior_pts:
            # No interior points → convex
            completed.append(poly)
            return
        
        shp = polygon_to_shapely(poly)
        edges = polygon_edges(poly)
        
        best_split_line = None
        best_split_len = float('inf')
        
        # Try each interior point as a candidate for splitting
        for int_pt in interior_pts:
            int_pt_shp = ShapelyPoint(int_pt)
            
            # Find nearest edge and construct orthogonal line
            for edge_start, edge_end in edges:
                edge_line = ShapelyLineString([edge_start, edge_end])
                
                # Shortest line from point to edge
                shortest = ShapelyLineString([int_pt_shp, edge_line.interpolate(
                    edge_line.project(int_pt_shp)
                )])
                
                # Check if orthogonal
                if lines_are_orthogonal(
                    int_pt,
                    (shortest.coords[1][0], shortest.coords[1][1]),
                    edge_start,
                    edge_end,
                ):
                    # Check containment: offset line slightly to stay inside
                    sp = shortest.interpolate(0.1, True)
                    ep = shortest.interpolate(0.9, True)
                    offset_line = ShapelyLineString([sp, ep])
                    
                    if shp.contains_properly(offset_line):
                        slen = shapely_length(shortest)
                        if slen < best_split_len:
                            best_split_line = shortest
                            best_split_len = slen
            
        if best_split_line is None:
            # No valid split found; return as-is
            completed.append(poly)
            return
        
        # Split polygon
        split_result = shapely_ops.split(shp, best_split_line)
        
        for sub_poly in split_result.geoms:
            sub_poly_list = shapely_to_polygon(sub_poly)
            _split_recursive(sub_poly_list, depth + 1)
    
    _split_recursive(polygon, 0)
    return completed


def split_along_longest_axis(polygon: Polygon) -> list[Polygon]:
    """
    Split a polygon by cutting along its longest axis (minimum rotated rectangle).
    Useful for rectangular rooms.
    
    Args:
        polygon: List of (x, y) vertices.
    
    Returns:
        List of two polygons (left and right halves).
    """
    shp = polygon_to_shapely(polygon)
    mrr = shp.minimum_rotated_rectangle
    
    coords = list(mrr.exterior.coords)
    
    # MRR has 5 coords (closed rectangle); find midpoints of long edges
    edges = [
        (coords[0], coords[1]),  # Edge 0-1
        (coords[1], coords[2]),  # Edge 1-2
        (coords[2], coords[3]),  # Edge 2-3
        (coords[3], coords[0]),  # Edge 3-0
    ]
    
    edge_lengths = [point_distance(e[0], e[1]) for e in edges]
    long_edge_idx = edge_lengths.index(max(edge_lengths))
    
    # Split perpendicular to long edge
    if long_edge_idx in [0, 2]:  # Horizontal-ish edges
        # Use vertical edges
        sp = ShapelyLineString([coords[0], coords[3]]).interpolate(0.5, True)
        ep = ShapelyLineString([coords[1], coords[2]]).interpolate(0.5, True)
    else:  # Vertical-ish edges
        # Use horizontal edges
        sp = ShapelyLineString([coords[0], coords[1]]).interpolate(0.5, True)
        ep = ShapelyLineString([coords[3], coords[2]]).interpolate(0.5, True)
    
    split_line = ShapelyLineString([sp, ep])
    result = shapely_ops.split(shp, split_line)
    
    return [shapely_to_polygon(p) for p in result.geoms]
