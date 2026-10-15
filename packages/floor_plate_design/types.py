"""
Data structures for floor plate design.

All dimensions in millimetres unless otherwise noted.
Areas in square metres.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum


class TypologyType(str, Enum):
    """Building typology classification."""
    SLAB = "slab"
    POINT_BLOCK = "point_block"


class SlabSubType(str, Enum):
    """Slab building sub-type."""
    SINGLE_LOADED = "single_loaded"
    DOUBLE_LOADED = "double_loaded"


class EdgeType(str, Enum):
    """Unit edge classification."""
    FACING_CORRIDOR = "facing_corridor"
    EXTERNAL = "external"
    SHARED_WALL = "shared_wall"


class CoreType(str, Enum):
    """Core (circulation) type."""
    STAIR_ONLY = "stair_only"
    STAIR_LIFT = "stair_lift"
    DOUBLE_STAIR_LIFT = "double_stair_lift"


class ConfidenceLevel(str, Enum):
    """Confidence level for inferred properties."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Point:
    """2D point in mm."""
    x: float
    y: float

    def to_tuple(self) -> tuple:
        return (self.x, self.y)

    @staticmethod
    def from_tuple(t: tuple) -> "Point":
        return Point(t[0], t[1])


Polygon = list[tuple[float, float]]  # List of (x, y) vertices


@dataclass
class TypedEdge:
    """Edge with classification."""
    edge_index: int
    type: EdgeType


@dataclass
class Unit:
    """Residential unit."""
    id: str
    type: str  # "1bed", "2bed", "3bed", etc.
    polygon: Polygon
    typed_edges: list[TypedEdge]
    entrance_point: tuple
    depth: float  # mm, perpendicular distance from corridor to external edge
    area: float  # m²


@dataclass
class Core:
    """Circulation core (stairs, lift, etc.)."""
    id: str
    polygon: Polygon
    type: CoreType
    centroid: tuple
    width: float  # mm
    depth: float  # mm


@dataclass
class Corridor:
    """Shared circulation space."""
    id: str
    polygon: Polygon
    width: float  # mm
    centreline: list[tuple]  # Polyline defining centerline


@dataclass
class Typology:
    """Building typology inference."""
    type: TypologyType
    sub_type: Optional[SlabSubType | Literal["central_core"]]
    confidence: ConfidenceLevel
    reasoning: str


@dataclass
class ComplianceCheck:
    """Single compliance check result."""
    check_name: str
    pass_: bool
    value: Optional[float | int] = None
    limit: Optional[float | int] = None
    message: str = ""


@dataclass
class ComplianceSnapshot:
    """Complete compliance assessment."""
    common_area_size: ComplianceCheck
    travel_distances: list[dict]  # [{unit_id, distance, pass, maximum}]
    entrance_overlaps: list[dict]  # [{unit_id, overlap, pass, minimum}]
    window_access: list[dict]  # [{unit_id, pass, external_edge_count}]
    unit_depths: list[dict]  # [{unit_id, depth, pass, maximum}]
    core_dimensions: list[dict]  # [{core_id, pass, area}]
    violations: list[str]
    warnings: list[str]


@dataclass
class FloorPlateState:
    """Complete floor plate design state."""
    boundary: Polygon
    typology: Typology
    cores: list[Core]
    corridors: list[Corridor]
    units: list[Unit]
    unresolved_regions: list[Polygon] = field(default_factory=list)
    analysis: Optional[dict] = None
    compliance_snapshot: Optional[ComplianceSnapshot] = None
    generation_log: list[str] = field(default_factory=list)


# Core dimensions by type (in mm)
CORE_DIMENSIONS = {
    CoreType.STAIR_ONLY: {"width": 3000, "depth": 4000, "min_area": 12.0},
    CoreType.STAIR_LIFT: {"width": 4000, "depth": 5000, "min_area": 20.0},
    CoreType.DOUBLE_STAIR_LIFT: {"width": 5000, "depth": 7000, "min_area": 35.0},
}
