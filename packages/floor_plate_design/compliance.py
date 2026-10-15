"""
Compliance checking functions for floor plates.

All functions are pure (no side effects) and can be called independently
or as part of a compliance snapshot. Designed for reuse in both
initial layout generation and TAO refinement loop.
"""

from typing import Optional
from .types import (
    FloorPlateState,
    Unit,
    Core,
    Corridor,
    ComplianceCheck,
    ComplianceSnapshot,
)
from .geometry import (
    polygon_area,
    polygon_centroid,
    point_distance,
    edge_intersection_length,
)


# ── Individual compliance checks ──────────────────────────────────────────────

def check_common_area_size(
    corridors: list[Corridor],
    cores: list[Core],
    min_area: float = 20.0,
) -> ComplianceCheck:
    """
    Check if total common area (corridors + cores) meets minimum.
    
    Args:
        corridors: List of Corridor objects.
        cores: List of Core objects.
        min_area: Minimum required area in m².
    
    Returns:
        ComplianceCheck with pass/fail and measurement.
    """
    total_area_mm2 = sum(polygon_area(c.polygon) for c in corridors)
    total_area_mm2 += sum(polygon_area(c.polygon) for c in cores)
    
    # Convert from mm² to m² (1 m² = 1e6 mm²)
    total_area_m2 = total_area_mm2 / 1e6
    
    pass_ = total_area_m2 >= min_area
    
    return ComplianceCheck(
        check_name="common_area_size",
        pass_=pass_,
        value=total_area_m2,
        limit=min_area,
        message=f"Common area: {total_area_m2:.1f} m² (min: {min_area} m²)",
    )


def check_travel_distances(
    units: list[Unit],
    cores: list[Core],
    max_distance: float = 30000.0,
) -> list[dict]:
    """
    Check travel distance from each unit entrance to nearest core.
    
    Args:
        units: List of Unit objects.
        cores: List of Core objects.
        max_distance: Maximum allowed distance in mm.
    
    Returns:
        List of {unit_id, distance, pass, maximum}.
    """
    results = []
    
    for unit in units:
        # Distance from unit entrance to nearest core centroid
        distances = [
            point_distance(unit.entrance_point, core.centroid)
            for core in cores
        ]
        
        min_distance = min(distances) if distances else 0
        pass_ = min_distance <= max_distance
        
        results.append({
            "unit_id": unit.id,
            "distance": min_distance,
            "pass": pass_,
            "maximum": max_distance,
        })
    
    return results


def check_entrance_overlaps(
    units: list[Unit],
    corridors: list[Corridor],
    min_overlap: float = 1200.0,
) -> list[dict]:
    """
    Check that each unit has sufficient corridor-facing edge length.
    
    Args:
        units: List of Unit objects.
        corridors: List of Corridor objects.
        min_overlap: Minimum overlap length in mm.
    
    Returns:
        List of {unit_id, overlap, pass, minimum}.
    """
    results = []
    
    for unit in units:
        # Find the corridor-facing edge and measure overlap
        overlap_lengths = []
        
        for edge in unit.typed_edges:
            if edge.type.value == "facing_corridor":
                # Get this edge of the unit polygon
                start_idx = edge.edge_index
                end_idx = (start_idx + 1) % len(unit.polygon)
                p1 = unit.polygon[start_idx]
                p2 = unit.polygon[end_idx]
                
                # Create a thin rectangle for the edge (perpendicular thickness of 10mm)
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = (dx ** 2 + dy ** 2) ** 0.5
                
                if length > 0:
                    # Perpendicular direction (rotated 90 degrees)
                    perp_x = -dy / length * 5
                    perp_y = dx / length * 5
                    
                    edge_polygon = [
                        (p1[0] - perp_x, p1[1] - perp_y),
                        (p2[0] - perp_x, p2[1] - perp_y),
                        (p2[0] + perp_x, p2[1] + perp_y),
                        (p1[0] + perp_x, p1[1] + perp_y),
                    ]
                else:
                    continue
                
                # Measure overlap with each corridor
                for corridor in corridors:
                    overlap = edge_intersection_length(edge_polygon, corridor.polygon)
                    overlap_lengths.append(overlap)
        
        max_overlap = max(overlap_lengths) if overlap_lengths else 0
        pass_ = max_overlap >= min_overlap
        
        results.append({
            "unit_id": unit.id,
            "overlap": max_overlap,
            "pass": pass_,
            "minimum": min_overlap,
        })
    
    return results


def check_window_access(units: list[Unit]) -> list[dict]:
    """
    Check that each unit has at least one external edge.
    
    Args:
        units: List of Unit objects.
    
    Returns:
        List of {unit_id, pass, external_edge_count}.
    """
    results = []
    
    for unit in units:
        external_count = sum(
            1 for edge in unit.typed_edges
            if edge.type.value == "external"
        )
        
        pass_ = external_count >= 1
        
        results.append({
            "unit_id": unit.id,
            "pass": pass_,
            "external_edge_count": external_count,
        })
    
    return results


def check_unit_depths(
    units: list[Unit],
    max_depth: float = 8000.0,
) -> list[dict]:
    """
    Check that each unit depth is within limits.
    
    Args:
        units: List of Unit objects.
        max_depth: Maximum unit depth in mm.
    
    Returns:
        List of {unit_id, depth, pass, maximum}.
    """
    results = []
    
    for unit in units:
        pass_ = unit.depth <= max_depth
        
        results.append({
            "unit_id": unit.id,
            "depth": unit.depth,
            "pass": pass_,
            "maximum": max_depth,
        })
    
    return results


def check_core_dimensions(cores: list[Core], min_areas: dict) -> list[dict]:
    """
    Check that each core meets minimum area requirements.
    
    Args:
        cores: List of Core objects.
        min_areas: Dict mapping CoreType to minimum area in m².
    
    Returns:
        List of {core_id, pass, area}.
    """
    results = []
    
    for core in cores:
        area_mm2 = polygon_area(core.polygon)
        area_m2 = area_mm2 / 1e6
        
        min_area = min_areas.get(core.type, 0)
        pass_ = area_m2 >= min_area
        
        results.append({
            "core_id": core.id,
            "pass": pass_,
            "area": area_m2,
            "minimum": min_area,
        })
    
    return results


# ── Aggregate compliance snapshot ──────────────────────────────────────────────

def compute_compliance_snapshot(
    floor_plate: FloorPlateState,
    constraints: dict,
) -> ComplianceSnapshot:
    """
    Run all compliance checks and assemble snapshot.
    
    Args:
        floor_plate: Complete FloorPlateState.
        constraints: Dict with keys: min_common_area, max_travel_distance,
                     min_entrance_overlap, max_unit_depth, core_min_areas.
    
    Returns:
        ComplianceSnapshot with all results.
    """
    # Extract constraint values
    min_common_area = constraints.get("min_common_area", 20.0)
    max_travel_distance = constraints.get("max_travel_distance", 30000.0)
    min_entrance_overlap = constraints.get("min_entrance_overlap", 1200.0)
    max_unit_depth = constraints.get("max_unit_depth", 8000.0)
    core_min_areas = constraints.get("core_min_areas", {})
    
    # Run checks
    common_area_check = check_common_area_size(
        floor_plate.corridors,
        floor_plate.cores,
        min_common_area,
    )
    
    travel_distances = check_travel_distances(
        floor_plate.units,
        floor_plate.cores,
        max_travel_distance,
    )
    
    entrance_overlaps = check_entrance_overlaps(
        floor_plate.units,
        floor_plate.corridors,
        min_entrance_overlap,
    )
    
    window_access = check_window_access(floor_plate.units)
    
    unit_depths = check_unit_depths(floor_plate.units, max_unit_depth)
    
    core_dimensions = check_core_dimensions(floor_plate.cores, core_min_areas)
    
    # Collect violations and warnings
    violations = []
    warnings = []
    
    if not common_area_check.pass_:
        violations.append(f"Common area insufficient: {common_area_check.message}")
    
    violations.extend([
        f"Unit {td['unit_id']}: travel distance {td['distance']:.0f}mm > {td['maximum']:.0f}mm"
        for td in travel_distances if not td['pass']
    ])
    
    violations.extend([
        f"Unit {eo['unit_id']}: entrance overlap {eo['overlap']:.0f}mm < {eo['minimum']:.0f}mm"
        for eo in entrance_overlaps if not eo['pass']
    ])
    
    violations.extend([
        f"Unit {wa['unit_id']}: no external edge access"
        for wa in window_access if not wa['pass']
    ])
    
    violations.extend([
        f"Unit {ud['unit_id']}: depth {ud['depth']:.0f}mm > {ud['maximum']:.0f}mm"
        for ud in unit_depths if not ud['pass']
    ])
    
    violations.extend([
        f"Core {cd['core_id']}: area {cd['area']:.1f}m² < {cd['minimum']:.1f}m²"
        for cd in core_dimensions if not cd['pass']
    ])
    
    # Warnings for near-limit conditions (within 10% of limit)
    for td in travel_distances:
        if td['pass'] and td['distance'] > td['maximum'] * 0.9:
            warnings.append(f"Unit {td['unit_id']}: travel distance approaching limit")
    
    return ComplianceSnapshot(
        common_area_size=common_area_check,
        travel_distances=travel_distances,
        entrance_overlaps=entrance_overlaps,
        window_access=window_access,
        unit_depths=unit_depths,
        core_dimensions=core_dimensions,
        violations=violations,
        warnings=warnings,
    )
