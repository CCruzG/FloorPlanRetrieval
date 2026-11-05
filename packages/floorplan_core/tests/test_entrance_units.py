"""Direct tests for the 2026-11-04 entrance code (gate suite, item 7).

Covers: generate._entrance_cell, generate._repair_boundary_contacts(first=),
layout_cpsat._add_entrance_front (door width), node_graph.entry_room_of.
"""
import pytest
from ortools.sat.python import cp_model
from shapely.geometry import Point

from packages.floorplan_core import layout_cpsat
from packages.floorplan_core.generate import (
    _edge_border_cells, _entrance_cell, _num_components, _repair_boundary_contacts)
from packages.floorplan_core.node_graph import entry_room_of

# ── _entrance_cell on a non-convex boundary ──────────────────────────────────

# L shape, edges: 0 bottom, 1 right, 2 inner step (y=5, x 10->5), 3 inner wall
# (x=5, y 5->10), 4 top, 5 left. Cells with centres in the missing quadrant
# have no edge.
L_BOUNDARY = [[0, 0], [10, 0], [10, 5], [5, 5], [5, 10], [0, 10]]
XS = YS = [0, 2.5, 5, 7.5, 10]


def _cells():
    return _edge_border_cells(L_BOUNDARY, XS, YS, 4, 4)


def test_entrance_cell_on_inner_step_picks_cell_whose_span_holds_the_entrance():
    # Entrance on edge 2 (the step into the notch) at x = 8: cells (2,1) and (3,1)
    # front it; the span containing x = 8 is column 3.
    cell = _entrance_cell(L_BOUNDARY, [8, 5, 8, 5], XS, YS, 4, 4, _cells())
    assert cell == (3, 1)


def test_entrance_cell_on_inner_wall_stays_in_the_arm():
    # Entrance on edge 3 (x = 5, y 5..10) at y = 8.5: only the left arm fronts it.
    cell = _entrance_cell(L_BOUNDARY, [5, 8.5, 5, 8.5], XS, YS, 4, 4, _cells())
    assert cell == (1, 3)
    assert cell in _cells()[3]


def test_entrance_cell_never_lies_in_the_notch():
    for ex, ey in ((8, 5), (5, 8.5), (7.4, 5), (5, 5.1)):
        cell = _entrance_cell(L_BOUNDARY, [ex, ey, ex, ey], XS, YS, 4, 4, _cells())
        assert cell is not None
        i, j = cell
        centre = Point((XS[i] + XS[i + 1]) / 2, (YS[j] + YS[j + 1]) / 2)
        from shapely.geometry import Polygon
        assert Polygon(L_BOUNDARY).contains(centre), (ex, ey, cell)


def test_entrance_cell_none_without_entrance():
    assert _entrance_cell(L_BOUNDARY, None, XS, YS, 4, 4, _cells()) is None


# ── _repair_boundary_contacts(first=) ────────────────────────────────────────

# row 0: A D Z ; row 1: D D D. Cell (1,0) is wanted by A (edge 0) and by Z
# (the "entrance" pseudo-edge). D owns it and can spare it.
def _contested():
    owner = {(0, 0): "A", (1, 0): "D", (2, 0): "Z",
             (0, 1): "D", (1, 1): "D", (2, 1): "D"}
    edges = {0: {(1, 0)}, "entrance": {(1, 0)}}
    demands = {"A": [0], "Z": ["entrance"]}
    return owner, demands, edges


def _intact(owner, ref):
    assert set(owner) == set(ref)
    for rid in set(ref.values()):
        cells = [c for c, r in owner.items() if r == rid]
        assert cells and _num_components(cells) == 1, rid


def test_first_gives_the_entry_room_the_contested_cell():
    owner, demands, edges = _contested()
    out, rep = _repair_boundary_contacts(owner, demands, edges, 3, 2, first="Z")
    assert out[(1, 0)] == "Z"
    assert ["Z", "entrance"] in rep["satisfied"]
    assert ["A", 0] in rep["unmet"]
    _intact(out, owner)


def test_without_first_the_other_room_wins_the_cell():
    owner, demands, edges = _contested()
    out, rep = _repair_boundary_contacts(owner, demands, edges, 3, 2)
    assert out[(1, 0)] == "A"
    assert ["Z", "entrance"] in rep["unmet"]


def test_donor_that_cannot_give_the_cell_is_unmet_and_plan_unchanged():
    # D owns one cell only; giving it would empty D.
    owner = {(0, 0): "Z", (1, 0): "D"}
    edges = {"entrance": {(1, 0)}}
    before = dict(owner)
    out, rep = _repair_boundary_contacts(owner, {"Z": ["entrance"]}, edges, 2, 1, first="Z")
    assert ["Z", "entrance"] in rep["unmet"]
    assert out == before
    assert rep["transfers"] == []


def test_donor_keeping_its_own_front_blocks_the_transfer():
    # D fronts edge 1 only through (1,0); Z may not take that cell.
    owner = {(0, 0): "Z", (1, 0): "D", (2, 0): "D"}
    edges = {"entrance": {(1, 0)}, 1: {(1, 0)}}
    out, rep = _repair_boundary_contacts(
        owner, {"Z": ["entrance"], "D": [1]}, edges, 3, 1, first="Z")
    assert out[(1, 0)] == "D"
    assert ["Z", "entrance"] in rep["unmet"]
    assert ["D", 1] in rep["satisfied"]


# ── CP-SAT entrance door width ───────────────────────────────────────────────

MM = 8000.0                       # one boundary unit = 8 m
SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1]]


def _entrance_model(x0_fixed=None, x1_fixed=None):
    """One room on the bottom edge; entrance point at the edge middle (x = 4.0 m)."""
    from shapely.geometry import Polygon
    geoms = layout_cpsat._edge_geometry(SQUARE, Polygon(SQUARE), ["external"] * 4, MM)
    g = geoms[0]
    model = cp_model.CpModel()
    r = {k: model.NewIntVar(0, 80, k) for k in ("x0", "x1", "y0", "y1")}
    model.Add(r["x1"] - r["x0"] >= 25)
    model.Add(r["y1"] - r["y0"] >= 25)
    layout_cpsat._add_entrance_front(model, r, g, [0.5, 0.0, 0.5, 0.0], MM, 8.0)
    if x0_fixed is not None:
        model.Add(r["x0"] == x0_fixed)
    if x1_fixed is not None:
        model.Add(r["x1"] == x1_fixed)
    s = cp_model.CpSolver()
    return s, s.Solve(model), r


def test_room_ending_at_the_jamb_does_not_satisfy_the_entrance():
    # Entrance point at 40 cells. A room spanning 10..40 reaches the point but
    # ends at the jamb: the door (0.9 m = 9 cells wide) is not inside it.
    _, st, _ = _entrance_model(x1_fixed=40)
    assert st == cp_model.INFEASIBLE


def test_room_starting_at_the_entrance_point_is_rejected():
    _, st, _ = _entrance_model(x0_fixed=40)
    assert st == cp_model.INFEASIBLE


def test_room_covering_the_door_width_is_accepted():
    _, st, r = _entrance_model(x0_fixed=30, x1_fixed=55)
    assert st in (cp_model.OPTIMAL, cp_model.FEASIBLE)


# ── node_graph.entry_room_of ─────────────────────────────────────────────────

def _sq(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _bathroom_nearest_plan():
    # Entrance box on the living room's bottom wall, 0.06 over the living room and
    # 0.04 over the bathroom: the door is on the living room, but the bathroom's
    # centre is nearer the entrance centre.
    return {
        "rooms": [{"id": "r_0", "label": "LivingRoom", "type": 0, "polygon": _sq(.2, .2, .5, .5)},
                  {"id": "r_1", "label": "Bathroom", "type": 6, "polygon": _sq(.5, .2, .6, .5)}],
        "edges": [{"u": "r_0", "v": "r_1", "connection": 0}],
        "doors": [],
        "entrance": [.44, .198, .54, .202],
        "boundary": [[.2, .2, 0, 0], [.6, .2, 0, 1], [.6, .5, 0, 2], [.2, .5, 0, 3]],
    }


def test_entry_room_is_the_door_room_not_the_nearest_room():
    plan = _bathroom_nearest_plan()
    ent = Point(.49, .2)
    from shapely.geometry import Polygon
    dist = {r["id"]: Polygon(r["polygon"]).centroid.distance(ent) for r in plan["rooms"]}
    assert dist["r_1"] < dist["r_0"]          # fixture: the bathroom is nearer
    assert entry_room_of(plan) == "r_0"


def test_entry_room_of_expanded_plan_reads_entry_node():
    assert entry_room_of({"node_graph_expanded": True, "entry_node": "r_3/p1"}) == "r_3/p1"
