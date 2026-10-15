"""
Unit tests for floor plate design geometry module.

Run with: python -m pytest packages/floor_plate_design/tests/test_geometry.py
"""

import pytest
from math import isclose
from ..geometry import (
    polygon_area,
    polygon_centroid,
    polygon_bbox,
    polygon_aspect_ratio,
    signed_area,
    is_clockwise,
    normalize_polygon_winding,
    point_distance,
    lines_are_orthogonal,
)


class TestPolygonArea:
    """Test polygon_area function."""
    
    def test_unit_square(self):
        """Unit square at origin."""
        polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert polygon_area(polygon) == 1.0
    
    def test_unit_square_scaled(self):
        """Square scaled to 1000mm."""
        polygon = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
        assert polygon_area(polygon) == 1e6  # mm²
    
    def test_triangle(self):
        """Right triangle with legs 3, 4."""
        polygon = [(0, 0), (3, 0), (0, 4)]
        assert polygon_area(polygon) == 6.0
    
    def test_empty(self):
        """Empty polygon."""
        assert polygon_area([]) == 0.0
        assert polygon_area([(0, 0)]) == 0.0
        assert polygon_area([(0, 0), (1, 1)]) == 0.0


class TestPolygonCentroid:
    """Test polygon_centroid function."""
    
    def test_unit_square_centroid(self):
        """Centroid of unit square."""
        polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
        cx, cy = polygon_centroid(polygon)
        assert isclose(cx, 0.5, abs_tol=0.01)
        assert isclose(cy, 0.5, abs_tol=0.01)
    
    def test_triangle_centroid(self):
        """Centroid of triangle."""
        polygon = [(0, 0), (3, 0), (0, 3)]
        cx, cy = polygon_centroid(polygon)
        # Centroid of triangle is average of vertices
        expected_cx = (0 + 3 + 0) / 3
        expected_cy = (0 + 0 + 3) / 3
        assert isclose(cx, expected_cx, abs_tol=0.01)
        assert isclose(cy, expected_cy, abs_tol=0.01)


class TestPolygonBBox:
    """Test polygon_bbox function."""
    
    def test_unit_square_bbox(self):
        """Bounding box of unit square."""
        polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
        bbox = polygon_bbox(polygon)
        assert bbox["min_x"] == 0
        assert bbox["max_x"] == 1
        assert bbox["min_y"] == 0
        assert bbox["max_y"] == 1
        assert bbox["width"] == 1
        assert bbox["height"] == 1
    
    def test_offset_square_bbox(self):
        """Bounding box of offset square."""
        polygon = [(10, 20), (15, 20), (15, 25), (10, 25)]
        bbox = polygon_bbox(polygon)
        assert bbox["min_x"] == 10
        assert bbox["max_x"] == 15
        assert bbox["min_y"] == 20
        assert bbox["max_y"] == 25
        assert bbox["width"] == 5
        assert bbox["height"] == 5


class TestAspectRatio:
    """Test polygon_aspect_ratio function."""
    
    def test_square_aspect_ratio(self):
        """Square has aspect ratio 1."""
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert polygon_aspect_ratio(polygon) == 1.0
    
    def test_rectangle_aspect_ratio(self):
        """Rectangle 20x10 has aspect ratio 2."""
        polygon = [(0, 0), (20, 0), (20, 10), (0, 10)]
        assert isclose(polygon_aspect_ratio(polygon), 2.0)
    
    def test_long_rectangle(self):
        """Long rectangle."""
        polygon = [(0, 0), (100, 0), (100, 10), (0, 10)]
        assert isclose(polygon_aspect_ratio(polygon), 10.0)


class TestSignedArea:
    """Test signed_area function."""
    
    def test_ccw_square(self):
        """CCW square has positive signed area."""
        polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
        sa = signed_area(polygon)
        assert sa > 0
    
    def test_cw_square(self):
        """CW square has negative signed area."""
        polygon = [(0, 0), (0, 1), (1, 1), (1, 0)]
        sa = signed_area(polygon)
        assert sa < 0


class TestIsClockwise:
    """Test is_clockwise function."""
    
    def test_ccw_polygon(self):
        """CCW polygon."""
        polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert is_clockwise(polygon) is False
    
    def test_cw_polygon(self):
        """CW polygon."""
        polygon = [(0, 0), (0, 1), (1, 1), (1, 0)]
        assert is_clockwise(polygon) is True


class TestNormalizeWinding:
    """Test normalize_polygon_winding function."""
    
    def test_ccw_to_ccw(self):
        """CCW polygon stays CCW."""
        polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = normalize_polygon_winding(polygon, clockwise=False)
        assert result == polygon
    
    def test_cw_to_ccw(self):
        """CW polygon becomes CCW."""
        polygon = [(0, 0), (0, 1), (1, 1), (1, 0)]
        result = normalize_polygon_winding(polygon, clockwise=False)
        assert is_clockwise(result) is False
    
    def test_ccw_to_cw(self):
        """CCW polygon becomes CW."""
        polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = normalize_polygon_winding(polygon, clockwise=True)
        assert is_clockwise(result) is True


class TestPointDistance:
    """Test point_distance function."""
    
    def test_unit_distance(self):
        """Distance between (0,0) and (1,0)."""
        assert point_distance((0, 0), (1, 0)) == 1.0
    
    def test_pythagorean_distance(self):
        """3-4-5 triangle."""
        assert point_distance((0, 0), (3, 4)) == 5.0
    
    def test_same_point(self):
        """Distance from point to itself."""
        assert point_distance((5, 5), (5, 5)) == 0.0


class TestLinesAreOrthogonal:
    """Test lines_are_orthogonal function."""
    
    def test_perpendicular_lines(self):
        """Horizontal and vertical lines."""
        # Horizontal line
        h_start, h_end = (0, 0), (10, 0)
        # Vertical line
        v_start, v_end = (5, -5), (5, 5)
        assert lines_are_orthogonal(h_start, h_end, v_start, v_end)
    
    def test_parallel_lines(self):
        """Parallel lines are not orthogonal."""
        line1_start, line1_end = (0, 0), (10, 0)
        line2_start, line2_end = (0, 5), (10, 5)
        assert lines_are_orthogonal(line1_start, line1_end, line2_start, line2_end) is False
    
    def test_45_degree_lines(self):
        """45-degree lines are not orthogonal."""
        line1_start, line1_end = (0, 0), (10, 10)
        line2_start, line2_end = (0, 0), (10, -10)
        assert lines_are_orthogonal(line1_start, line1_end, line2_start, line2_end)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
