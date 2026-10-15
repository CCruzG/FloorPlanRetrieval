"""
test_generator.py — Unit tests for the floor plate generation pipeline.

Tests all stages 1-10 of layout generation with various building types.
"""

import pytest
from packages.floor_plate_design.generator import generate_initial_layout, generate_subdivision_response
from packages.floor_plate_design.types import (
    TypologyType,
    SlabSubType,
    EdgeType,
)
from packages.floor_plate_design.geometry import polygon_to_shapely


class TestGeneratorBasic:
    """Test basic generation pipeline functionality."""

    def test_double_loaded_slab_generation(self):
        """Test double-loaded slab generation."""
        boundary = [(0, 0), (14000, 0), (14000, 20000), (0, 20000)]
        unit_mix = [{"type": "1bed", "count": 6, "target_area": 55}]
        constraints = {
            "corridor_width": 1500,
            "max_unit_depth": 8000,
            "min_entrance_overlap": 1200,
            "min_common_area": 20,
            "max_travel_distance": 30000,
            "core_type": "stair_lift",
        }

        result = generate_initial_layout(boundary, unit_mix, constraints)

        # Check typology
        assert result.typology.type == TypologyType.SLAB
        assert result.typology.sub_type == SlabSubType.DOUBLE_LOADED

        # Check cores
        assert len(result.cores) == 1

        # Check corridors
        assert len(result.corridors) == 1

        # Check units placed
        assert len(result.units) > 0

        # Check units have properties
        for unit in result.units:
            assert len(unit.polygon) >= 4
            assert len(unit.typed_edges) == len(unit.polygon)
            assert unit.entrance_point != (0, 0)
            assert unit.area > 0
            assert unit.depth > 0

        # Check at least one unit has facing_corridor edge
        has_corridor_edge = any(
            any(edge.type == EdgeType.FACING_CORRIDOR for edge in unit.typed_edges)
            for unit in result.units
        )
        assert has_corridor_edge

        # Check compliance
        assert result.compliance_snapshot is not None

    def test_single_loaded_slab_generation(self):
        """Test single-loaded slab generation."""
        boundary = [(0, 0), (10000, 0), (10000, 25000), (0, 25000)]
        unit_mix = [{"type": "1bed", "count": 6, "target_area": 55}]
        constraints = {
            "corridor_width": 1500,
            "max_unit_depth": 8000,
            "min_entrance_overlap": 1200,
            "min_common_area": 20,
            "max_travel_distance": 30000,
            "core_type": "stair_lift",
        }

        result = generate_initial_layout(boundary, unit_mix, constraints)

        # Check typology
        assert result.typology.type == TypologyType.SLAB
        assert result.typology.sub_type == SlabSubType.SINGLE_LOADED

        # Check cores
        assert len(result.cores) == 1

        # Check corridors
        assert len(result.corridors) == 1

        # Check units placed
        assert len(result.units) > 0

    def test_long_slab_multi_core(self):
        """Test long slab with multiple cores."""
        boundary = [(0, 0), (14000, 0), (14000, 80000), (0, 80000)]
        unit_mix = [{"type": "1bed", "count": 8, "target_area": 55}]
        constraints = {
            "corridor_width": 1500,
            "max_unit_depth": 8000,
            "min_entrance_overlap": 1200,
            "min_common_area": 20,
            "max_travel_distance": 30000,
            "core_type": "stair_lift",
        }

        result = generate_initial_layout(boundary, unit_mix, constraints)

        # Check typology
        assert result.typology.type == TypologyType.SLAB
        assert result.typology.sub_type == SlabSubType.DOUBLE_LOADED

        # Check cores (should have 2 cores for 80km long dimension)
        assert len(result.cores) == 2

        # Check corridors
        assert len(result.corridors) == 1

        # Check units placed
        assert len(result.units) > 0

    def test_rectangular_distribution_prefers_slab_when_width_supports_two_bands(self):
        """Rectangular plates with sufficient width should prefer slab distribution."""
        boundary = [(0, 0), (18000, 0), (18000, 20000), (0, 20000)]
        unit_mix = [{"type": "1bed", "count": 6, "target_area": 55}]
        constraints = {
            "corridor_width": 1500,
            "max_unit_depth": 8000,
            "min_entrance_overlap": 1200,
            "min_common_area": 20,
            "max_travel_distance": 30000,
            "core_type": "stair_lift",
        }

        result = generate_initial_layout(boundary, unit_mix, constraints)

        assert result.typology.type == TypologyType.SLAB
        assert result.typology.sub_type == SlabSubType.DOUBLE_LOADED

        assert len(result.cores) == 1
        assert len(result.corridors) == 1
        assert len(result.units) > 0

    def test_subdivision_only_mode_skips_core_and_corridor_generation(self):
        """Subdivision-only mode should focus on apartment subdivision without common/core geometry."""
        boundary = [(0, 0), (22000, 0), (22000, 18000), (0, 18000)]
        unit_mix = [
            {"type": "1bed", "count": 2, "target_area": 55},
            {"type": "2bed", "count": 2, "target_area": 70},
        ]
        constraints = {
            "corridor_width": 1500,
            "max_unit_depth": 8000,
            "min_entrance_overlap": 1200,
            "min_common_area": 20,
            "max_travel_distance": 30000,
            "core_type": "stair_lift",
            "subdivision_only": True,
        }

        result = generate_initial_layout(boundary, unit_mix, constraints)

        assert len(result.cores) == 0
        assert len(result.corridors) == 0
        assert len(result.units) > 0
        assert all(unit.area > 20 for unit in result.units)

    def test_subdivision_only_covers_full_floor_plate(self):
        """Subdivision-only mode should partition the entire floor plate area."""
        boundary = [(0, 0), (19200, 0), (19200, 9900), (0, 9900)]
        unit_mix = [{"type": "1bed", "count": 2, "target_area": 55}]
        constraints = {
            "corridor_width": 1500,
            "max_unit_depth": 8000,
            "min_entrance_overlap": 1200,
            "min_common_area": 20,
            "max_travel_distance": 30000,
            "core_type": "stair_lift",
            "subdivision_only": True,
        }

        result = generate_initial_layout(boundary, unit_mix, constraints)

        plate_area = (19200 * 9900) / 1_000_000
        units_area = sum(unit.area for unit in result.units)

        assert len(result.units) == 2
        assert abs(units_area - plate_area) < 0.5
        assert all(unit.area > 0 for unit in result.units)

    def test_subdivision_response_single_contract(self):
        """The subdivision contract should return full-area subdivision with no residual polygons."""
        boundary = [(0, 0), (19200, 0), (19200, 9900), (0, 9900)]
        unit_mix = {"1bed": 2, "2bed": 1}
        constraints = {
            "min_depth_m": 5.5,
            "min_area_m2": 35,
            "area_tolerance_m2": 0.01,
            "max_candidates": 8,
            "max_total_units": 3,
            "unit_specs": {
                "1bed": {"target_area_m2": 55, "min_area_m2": 35, "min_depth_m": 5.5},
                "2bed": {"target_area_m2": 70, "min_area_m2": 50, "min_depth_m": 5.5},
            },
        }

        result = generate_subdivision_response(boundary, unit_mix, constraints, mode="single", seed=7)

        assert result["mode"] == "single"
        assert result["area_check"]["passed"] is True
        assert abs(result["area_check"]["discrepancy_m2"]) < 0.01
        assert len(result["placed_units"]) == 3
        assert len(result["residual_zones"]) == 0
        assert isinstance(result["warnings"], list)
        assert "score" in result

    def test_subdivision_response_harness_contract(self):
        """Harness mode should return ranked candidates and the best candidate."""
        boundary = [(0, 0), (19200, 0), (19200, 9900), (0, 9900)]
        unit_mix = {"1bed": 2, "2bed": 1}
        constraints = {
            "min_depth_m": 5.5,
            "min_area_m2": 35,
            "area_tolerance_m2": 0.01,
            "max_candidates": 8,
            "max_total_units": 3,
            "unit_specs": {
                "1bed": {"target_area_m2": 55, "min_area_m2": 35, "min_depth_m": 5.5},
                "2bed": {"target_area_m2": 70, "min_area_m2": 50, "min_depth_m": 5.5},
            },
        }

        result = generate_subdivision_response(boundary, unit_mix, constraints, mode="harness", seed=7)

        assert result["mode"] == "harness"
        assert "best" in result
        assert len(result["candidates"]) <= 3
        assert result["parameter_sets_tried"] >= len(result["candidates"])
        assert result["best"]["area_check"]["passed"] is True
        assert len(result["best"]["residual_zones"]) == 0
        assert all(len(candidate["residual_zones"]) == 0 for candidate in result["candidates"])

    def test_subdivision_point_tower_prefers_compact_pieces(self):
        """Near-square plates should avoid long-strip partitions in auto mode."""
        boundary = [(0, 0), (16700, 0), (16700, 16200), (0, 16200)]
        unit_mix = {"1bed": 1, "2bed": 2}
        constraints = {
            "min_depth_m": 5.5,
            "min_area_m2": 35,
            "area_tolerance_m2": 0.01,
            "max_candidates": 8,
            "max_total_units": 4,
            "unit_specs": {
                "1bed": {"target_area_m2": 55, "min_area_m2": 35, "min_depth_m": 5.5},
                "2bed": {"target_area_m2": 70, "min_area_m2": 50, "min_depth_m": 5.5},
            },
        }

        result = generate_subdivision_response(boundary, unit_mix, constraints, mode="single", seed=7)
        mix_adj = result["mix_adjustment"]

        assert result["area_check"]["passed"] is True
        assert len(result["residual_zones"]) == 0
        assert len(result["placed_units"]) == mix_adj["effective_total"]
        assert mix_adj["effective_total"] >= 3

        ratios = []
        for unit in result["placed_units"]:
            poly = unit["polygon"]
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            short_side = max(1e-6, min(w, h))
            long_side = max(w, h)
            ratio = long_side / short_side
            ratios.append(ratio)

        # Regression target: avoid the previous "three long strips" output pattern.
        assert sum(1 for ratio in ratios if ratio >= 2.6) <= 1
        assert sum(1 for ratio in ratios if ratio <= 1.8) >= 2

    def test_subdivision_point_tower_four_units_prefers_grid_like_split(self):
        """Near-square 4-unit requests should avoid full-width strip outcomes."""
        boundary = [(0, 0), (17400, 0), (17400, 15800), (0, 15800)]
        unit_mix = {"1bed": 2, "2bed": 2}
        constraints = {
            "min_depth_m": 5.5,
            "min_area_m2": 35,
            "area_tolerance_m2": 0.01,
            "max_candidates": 8,
            "max_total_units": 4,
            "unit_specs": {
                "1bed": {"target_area_m2": 55, "min_area_m2": 35, "min_depth_m": 5.5},
                "2bed": {"target_area_m2": 70, "min_area_m2": 50, "min_depth_m": 5.5},
            },
        }

        result = generate_subdivision_response(boundary, unit_mix, constraints, mode="single", seed=7)
        assert result["area_check"]["passed"] is True
        assert len(result["residual_zones"]) == 0
        assert len(result["placed_units"]) == 4

        plate_w = 17400.0
        plate_h = 15800.0
        wide_spanning = 0
        tall_spanning = 0
        for unit in result["placed_units"]:
            poly = unit["polygon"]
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            if w >= 0.9 * plate_w:
                wide_spanning += 1
            if h >= 0.9 * plate_h:
                tall_spanning += 1

        # A strip outcome would have multiple units spanning almost the full plate width or height.
        assert wide_spanning <= 1
        assert tall_spanning <= 1

    def test_subdivision_adapts_down_when_capacity_is_lower(self):
        """If area cannot hold requested units, result should reduce count preserving ratio."""
        boundary = [(0, 0), (14000, 0), (14000, 12000), (0, 12000)]
        unit_mix = {"2bed": 2, "3bed": 2}
        constraints = {
            "min_depth_m": 5.5,
            "min_area_m2": 35,
            "area_tolerance_m2": 0.01,
            "max_candidates": 8,
            "unit_specs": {
                "2bed": {"target_area_m2": 70, "min_area_m2": 50, "min_depth_m": 5.5},
                "3bed": {"target_area_m2": 90, "min_area_m2": 65, "min_depth_m": 5.5},
            },
        }

        result = generate_subdivision_response(boundary, unit_mix, constraints, mode="single", seed=7)
        mix_adj = result["mix_adjustment"]

        assert result["area_check"]["passed"] is True
        assert len(result["residual_zones"]) == 0
        assert mix_adj["requested_total"] == 4
        assert mix_adj["effective_total"] == 2
        assert mix_adj["effective_mix"] == {"2bed": 1, "3bed": 1}
        assert len(result["placed_units"]) == 2

    def test_subdivision_auto_fills_large_plate_to_optimal_count(self):
        """On a large boundary, unit count should expand to fill the space at target sizes,
        but never beyond the hard min-area capacity."""
        boundary = [(0, 0), (24000, 0), (24000, 18000), (0, 18000)]
        unit_mix = {"2bed": 2, "3bed": 2}
        constraints = {
            "min_depth_m": 5.5,
            "min_area_m2": 35,
            "area_tolerance_m2": 0.01,
            "max_candidates": 8,
            "unit_specs": {
                "2bed": {"target_area_m2": 70, "min_area_m2": 50, "min_depth_m": 5.5},
                "3bed": {"target_area_m2": 90, "min_area_m2": 65, "min_depth_m": 5.5},
            },
        }

        result = generate_subdivision_response(boundary, unit_mix, constraints, mode="single", seed=7)
        mix_adj = result["mix_adjustment"]
        capacity = int(24 * 18 / ((50 * 2 + 65 * 2) / 4))  # boundary_area / avg_min_area

        assert result["area_check"]["passed"] is True
        assert mix_adj["requested_total"] == 4
        assert mix_adj["effective_total"] >= mix_adj["requested_total"]
        assert mix_adj["effective_total"] <= capacity
        assert sum(mix_adj["effective_mix"].values()) == mix_adj["effective_total"]

    def test_subdivision_small_plate_does_not_collapse_to_one_unit_for_mixed_request(self):
        """Small plates with mixed requests should not collapse to a single oversized unit."""
        boundary = [(0, 0), (7800, 0), (7800, 15500), (0, 15500)]
        unit_mix = {"1bed": 2, "2bed": 2}
        constraints = {
            "min_depth_m": 5.5,
            "min_area_m2": 35,
            "area_tolerance_m2": 0.01,
            "max_candidates": 8,
            "unit_specs": {
                "1bed": {"target_area_m2": 55, "min_area_m2": 35, "min_depth_m": 5.5},
                "2bed": {"target_area_m2": 70, "min_area_m2": 50, "min_depth_m": 5.5},
            },
        }

        result = generate_subdivision_response(boundary, unit_mix, constraints, mode="single", seed=7)
        mix_adj = result["mix_adjustment"]

        assert result["area_check"]["passed"] is True
        assert len(result["residual_zones"]) == 0
        assert mix_adj["requested_total"] == 4
        assert mix_adj["effective_total"] >= 2
        assert len(result["placed_units"]) == mix_adj["effective_total"]

    def test_subdivision_assigns_larger_type_to_larger_piece_when_possible(self):
        """Larger unit types should not be assigned to clearly smaller pieces when a better fit exists."""
        boundary = [(0, 0), (20000, 0), (20000, 6400), (0, 6400)]
        unit_mix = {"1bed": 2, "2bed": 1}
        constraints = {
            "min_depth_m": 5.5,
            "min_area_m2": 35,
            "area_tolerance_m2": 0.01,
            "max_candidates": 8,
            "max_total_units": 3,
            "unit_specs": {
                "1bed": {"target_area_m2": 55, "min_area_m2": 35, "min_depth_m": 5.5},
                "2bed": {"target_area_m2": 70, "min_area_m2": 50, "min_depth_m": 5.5},
            },
        }

        result = generate_subdivision_response(boundary, unit_mix, constraints, mode="single", seed=7)
        assert len(result["placed_units"]) == 3

        areas_by_type = {}
        for unit in result["placed_units"]:
            areas_by_type.setdefault(unit["type"], []).append(unit["area_m2"])

        assert "2bed" in areas_by_type and "1bed" in areas_by_type
        assert max(areas_by_type["2bed"]) >= max(areas_by_type["1bed"])

    def test_boundary_normalization(self):
        """Test that boundary is normalized to CCW and origin."""
        # CW boundary (wrong winding)
        boundary = [(0, 0), (0, 20000), (14000, 20000), (14000, 0)]
        unit_mix = [{"type": "1bed", "count": 6, "target_area": 55}]
        constraints = {
            "corridor_width": 1500,
            "max_unit_depth": 8000,
            "min_entrance_overlap": 1200,
            "min_common_area": 20,
            "max_travel_distance": 30000,
            "core_type": "stair_lift",
        }

        result = generate_initial_layout(boundary, unit_mix, constraints)

        # Check boundary was normalized to start at origin
        assert result.boundary[0] == (0, 0) or all(
            v[0] >= 0 and v[1] >= 0 for v in result.boundary
        )

    def test_unit_edges_have_correct_types(self):
        """Test that unit edges are correctly classified."""
        boundary = [(0, 0), (14000, 0), (14000, 20000), (0, 20000)]
        unit_mix = [{"type": "1bed", "count": 6, "target_area": 55}]
        constraints = {
            "corridor_width": 1500,
            "max_unit_depth": 8000,
            "min_entrance_overlap": 1200,
            "min_common_area": 20,
            "max_travel_distance": 30000,
            "core_type": "stair_lift",
        }

        result = generate_initial_layout(boundary, unit_mix, constraints)

        # Each unit should have edges with valid types
        for unit in result.units:
            for edge in unit.typed_edges:
                assert edge.type in [
                    EdgeType.FACING_CORRIDOR,
                    EdgeType.EXTERNAL,
                    EdgeType.SHARED_WALL,
                ]

    def test_generation_log_populated(self):
        """Test that generation log is populated with diagnostic info."""
        boundary = [(0, 0), (14000, 0), (14000, 20000), (0, 20000)]
        unit_mix = [{"type": "1bed", "count": 6, "target_area": 55}]
        constraints = {
            "corridor_width": 1500,
            "max_unit_depth": 8000,
            "min_entrance_overlap": 1200,
            "min_common_area": 20,
            "max_travel_distance": 30000,
            "core_type": "stair_lift",
        }

        result = generate_initial_layout(boundary, unit_mix, constraints)

        # Check log contains entries
        assert len(result.generation_log) > 0

        # Check for key stage messages
        log_text = " ".join(result.generation_log)
        assert "Boundary" in log_text or "aspect" in log_text
        assert "Typology" in log_text or "units" in log_text

    def test_compliance_snapshot_created(self):
        """Test that compliance snapshot is computed."""
        boundary = [(0, 0), (14000, 0), (14000, 20000), (0, 20000)]
        unit_mix = [{"type": "1bed", "count": 6, "target_area": 55}]
        constraints = {
            "corridor_width": 1500,
            "max_unit_depth": 8000,
            "min_entrance_overlap": 1200,
            "min_common_area": 20,
            "max_travel_distance": 30000,
            "core_type": "stair_lift",
        }

        result = generate_initial_layout(boundary, unit_mix, constraints)

        # Check compliance snapshot exists
        assert result.compliance_snapshot is not None
        assert hasattr(result.compliance_snapshot, "violations")
        assert hasattr(result.compliance_snapshot, "warnings")
        assert hasattr(result.compliance_snapshot, "common_area_size")

    def test_capacity_driven_two_unit_override_to_point_tower(self):
        """When only two units are accommodable, typology should fall back to point tower."""
        boundary = [(0, 0), (14000, 0), (14000, 20000), (0, 20000)]
        unit_mix = [{"type": "3bed", "count": 2, "target_area": 110}]
        constraints = {
            "corridor_width": 1500,
            "max_unit_depth": 8000,
            "min_entrance_overlap": 1200,
            "min_common_area": 20,
            "max_travel_distance": 30000,
            "core_type": "stair_lift",
        }

        result = generate_initial_layout(boundary, unit_mix, constraints)
        assert result.typology.type == TypologyType.POINT_BLOCK
        assert len(result.corridors) == 1
        assert len(result.units) > 0

    def test_corridor_is_tangential_to_core(self):
        """Core and corridor should be adjacent with shared edge contact and zero area overlap."""
        boundary = [(0, 0), (14000, 0), (14000, 20000), (0, 20000)]
        unit_mix = [{"type": "1bed", "count": 4, "target_area": 55}]
        constraints = {
            "corridor_width": 1500,
            "max_unit_depth": 8000,
            "min_entrance_overlap": 1200,
            "min_common_area": 20,
            "max_travel_distance": 30000,
            "core_type": "stair_lift",
        }

        result = generate_initial_layout(boundary, unit_mix, constraints)
        assert result.cores
        assert result.corridors

        core_shp = polygon_to_shapely(result.cores[0].polygon)
        corridor_shp = polygon_to_shapely(result.corridors[0].polygon)

        # Shared boundary (touch) is allowed; positive overlap area is not.
        assert core_shp.intersection(corridor_shp).area == pytest.approx(0.0)
        assert core_shp.touches(corridor_shp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
