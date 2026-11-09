"""Door derivation (decision 2026-11-04): doors are derived from the generated geometry."""
import copy
import math

from packages.floorplan_core.doors import derive_doors
from packages.floorplan_core.generate import generate
from packages.floorplan_core.tests.test_spec import _good_spec

MM = 10000.0


def _rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _rooms():
    # liv (left, 5 m x 4 m) | bed (right); hall below liv; 5 m shared wall liv|bed
    return [{"id": "liv", "type": "living", "polygon": _rect(0.0, 0.0, 0.5, 0.4)},
            {"id": "bed", "type": "bedroom", "polygon": _rect(0.5, 0.0, 1.0, 0.4)},
            {"id": "hall", "type": "hallway", "polygon": _rect(0.0, 0.4, 1.0, 0.45)}]


def test_door_edge_gets_door_900_with_jamb_100():
    rooms = _rooms()
    doors, unplaced = derive_doors(rooms, [{"u": "liv", "v": "bed", "connection": 1}],
                                   _rect(0, 0, 1, 0.45), None, MM)
    assert unplaced == [] and len(doors) == 1
    d = doors[0]
    assert d["width_mm"] == 900 and abs(d["width"] * MM - 900) < 1e-6
    a, b = d["segment"]
    assert abs(a[0] - 0.5) < 1e-9 and abs(b[0] - 0.5) < 1e-9
    # near jamb 100 mm from the end nearer the living centroid (0.25, 0.2): y=0.0 and y=0.4 tie
    # -> lexicographically smaller end (y=0.0)
    jamb = d["centre"][1] - 0.5 * 900 / MM
    assert abs(jamb * MM - 100) < 1e-6
    assert abs(d["segment_mm"] - 4000) < 1


def test_door_goes_to_end_nearer_less_private_room():
    # shared wall x=0.5, y 0..1.0; the hallway reaches only y 0.6..1.0 of it
    rooms = [{"id": "hall", "type": "hallway", "polygon": [[0.2, 0.6], [0.5, 0.6], [0.5, 1.0], [0.2, 1.0]]},
             {"id": "bed", "type": "bedroom", "polygon": _rect(0.5, 0.0, 0.9, 1.0)}]
    d, _ = derive_doors(rooms, [{"u": "bed", "v": "hall", "connection": 1}], _rect(0.2, 0, 0.9, 1.0), None, MM)
    # shared piece is y 0.6..1.0 (4000 mm); hallway centroid y=0.8 ties -> smaller end, y=0.6
    assert abs(d[0]["centre"][1] - (0.6 + 550 / MM)) < 1e-9
    # swap roles: bedroom side private, living nearer the top end
    rooms[0] = {"id": "hall", "type": "living", "polygon": [[0.2, 0.0], [0.5, 0.0], [0.5, 1.0], [0.2, 1.0]]}
    rooms[1] = {"id": "bed", "type": "bedroom", "polygon": _rect(0.5, 0.0, 0.9, 1.0)}
    rooms.append({"id": "k", "type": "kitchen", "polygon": _rect(0.5, 1.0, 0.9, 1.2)})
    d, _ = derive_doors(rooms, [{"u": "k", "v": "hall", "connection": 1}], _rect(0.2, 0, 0.9, 1.2), None, MM)
    # shared wall of kitchen and living: y=1.0, x 0.5..0.5 is a point -> no wall at all
    assert d == []


def test_short_wall_is_unplaced():
    rooms = [{"id": "a", "type": "living", "polygon": _rect(0.0, 0.0, 0.5, 0.5)},
             {"id": "b", "type": "bedroom", "polygon": _rect(0.5, 0.0, 1.0, 0.0995)}]  # 995 mm
    doors, unplaced = derive_doors(rooms, [{"u": "a", "v": "b", "connection": 1}], _rect(0, 0, 1, 0.5), None, MM)
    assert doors == [] and len(unplaced) == 1
    assert unplaced[0]["u"] == "a" and unplaced[0]["v"] == "b"
    assert 990 < unplaced[0]["segment_mm"] < 1000


def test_open_and_wall_edges_get_no_door():
    rooms = _rooms()
    edges = [{"u": "liv", "v": "bed", "connection": 2}, {"u": "liv", "v": "hall", "connection": 0}]
    doors, unplaced = derive_doors(rooms, edges, _rect(0, 0, 1, 0.45), None, MM)
    assert doors == [] and unplaced == []


def test_entrance_door_on_boundary_edge():
    rooms = _rooms()
    doors, unplaced = derive_doors(rooms, [], _rect(0, 0, 1, 0.45), [0.5, 0.45, 0.5, 0.45], MM)
    assert unplaced == [] and len(doors) == 1 and doors[0]["kind"] == "entrance"
    assert abs(doors[0]["centre"][1] - 0.45) < 1e-9 and abs(doors[0]["centre"][0] - 0.5) < 1e-9


def test_entrance_near_corner_is_moved_inward():
    doors, _ = derive_doors(_rooms(), [], _rect(0, 0, 1, 0.45), [0.0, 0.45, 0.0, 0.45], MM)
    # centre at least 100 + 450 mm from the end
    assert doors[0]["centre"][0] * MM >= 550 - 1e-6


def test_missing_room_is_listed():
    doors, unplaced = derive_doors(_rooms(), [{"u": "liv", "v": "ghost", "connection": 1}],
                                   _rect(0, 0, 1, 0.45), None, MM)
    assert doors == [] and unplaced[0]["reason"] == "room not generated"


def test_deterministic():
    r, e = _rooms(), [{"u": "liv", "v": "bed", "connection": 1}, {"u": "hall", "v": "bed", "connection": 1}]
    a = derive_doors(copy.deepcopy(r), e, _rect(0, 0, 1, 0.45), [0.5, 0.45, 0.5, 0.45], MM)
    b = derive_doors(copy.deepcopy(r), e, _rect(0, 0, 1, 0.45), [0.5, 0.45, 0.5, 0.45], MM)
    assert a == b


def test_generate_accounts_for_every_door_edge():
    spec = _good_spec()
    p = generate(spec)
    door_pairs = {frozenset((e["u"], e["v"])) for e in p["edges"] if e.get("connection") == 1}
    placed = {frozenset((d["u"], d["v"])) for d in p["doors"] if d["kind"] == "internal"}
    unplaced = {frozenset((d["u"], d["v"])) for d in p["report"]["doors_unplaced"] if d["u"] != "entrance"}
    assert door_pairs and door_pairs == placed | unplaced
    assert p["frame"]["mm_per_unit"] == spec["mm_per_unit"]
    assert any(d["kind"] == "entrance" for d in p["doors"])
    # same Spec + options give the same plan, doors included
    assert generate(copy.deepcopy(spec))["doors"] == p["doors"]
    for d in p["doors"]:
        assert math.isclose(d["width"] * spec["mm_per_unit"], d["width_mm"], rel_tol=1e-9)
