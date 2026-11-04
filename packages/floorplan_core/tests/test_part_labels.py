import json
from pathlib import Path

import pytest

from packages.floorplan_core.part_labels import label_parts
from packages.floorplan_core.rect_decompose import decompose_plan

S = 18.0
RPLAN = Path("/Users/camilo/Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json/19679.json")


def _l_plan(door=None):
    # L room: big block 0..0.4 x 0..0.4 plus arm 0.4..0.55 x 0..0.1 (arm 2.7 m wide... short 0.1*18=1.8 m)
    poly = [[0, 0], [0.55, 0], [0.55, 0.1], [0.4, 0.1], [0.4, 0.4], [0, 0.4]]
    plan = {"rooms": [{"id": "r_0", "label": "LivingRoom", "type": 3, "polygon": poly},
                      {"id": "r_1", "label": "Bedroom", "type": 1,
                       "polygon": [[0.55, 0], [0.7, 0], [0.7, 0.1], [0.55, 0.1]]}],
            "doors": [], "entrance": None, "edges": []}
    if door:
        plan["doors"] = [door]
    return plan


def _run(plan, **kw):
    dec = decompose_plan(plan, sliver_max_m=0.2, merge_min_contact_m=None)
    return dec, label_parts(plan, dec, **kw)


def _roles(res, rid="r_0"):
    room = next(r for r in res["rooms"] if r["id"] == rid)
    return room, {p["index"]: p["role"] for p in room["parts"]}


def _arm_index(dec):
    parts = next(r for r in dec["rooms"] if r["id"] == "r_0")["parts"]
    assert len(parts) == 2
    return min(range(2), key=lambda i: Polygon_area(parts[i]))


def Polygon_area(part):
    from shapely.geometry import Polygon
    return Polygon(part["polygon"]).area


DOOR = {"id": "d_0", "bbox": [0.55 - 0.002, 0.02, 0.55 + 0.002, 0.08], "orientation": "vertical", "room_refs": ["r_1"]}


def test_arm_with_door_is_circulation():
    dec, res = _run(_l_plan(DOOR))
    arm = _arm_index(dec)
    room, roles = _roles(res)
    assert roles[arm] == "circulation" and roles[1 - arm] == "primary"
    assert room["parts"][arm]["door_ids"] == ["d_0"]
    assert res["connections"] == [{"door_id": "d_0", "kind": "door", "a": "r_1", "b": "r_0/circ0"}]  # v1 adds kind


def test_arm_without_door_on_facade_is_nook():
    # v1: was "storage" in v0. Arm is 1.8 m wide, no door, on the plan outline -> nook.
    dec, res = _run(_l_plan())
    assert _roles(res)[1][_arm_index(dec)] == "nook"


def test_storage_cap_param_makes_arm_a_jog():
    # v1: replaces the v0 "residual over cap" test. Cap 2.0 m puts the 1.8 m facade arm below it -> jog.
    dec, res = _run(_l_plan(), storage_max_short_m=2.0)
    assert _roles(res)[1][_arm_index(dec)] == "jog"
    assert res["params"]["storage_max_short_m"] == 2.0


def test_two_adjacent_circulation_parts_merge():
    # big block + two small parts sharing a 1.8 m edge, each with its own door.
    poly = [[0, 0], [0.4, 0], [0.4, 0.4], [0, 0.4], [0, 0.3], [-0.1, 0.3], [-0.1, 0.2],
            [-0.2, 0.2], [-0.2, 0.1], [-0.1, 0.1], [-0.1, 0.0]]
    plan = {"rooms": [{"id": "r_0", "label": "LivingRoom", "type": 3, "polygon": poly}],
            "doors": [{"id": "d_0", "bbox": [-0.1 - 0.002, 0.22, -0.1 + 0.002, 0.28], "room_refs": ["r_0"]},
                      {"id": "d_1", "bbox": [-0.2 - 0.002, 0.12, -0.2 + 0.002, 0.18], "room_refs": ["r_0"]}],
            "entrance": None, "edges": []}
    dec = decompose_plan(plan, sliver_max_m=0.2, merge_min_contact_m=None)
    assert len(dec["rooms"][0]["parts"]) == 3
    room = label_parts(plan, dec)["rooms"][0]
    circ = [p for p in room["parts"] if p["role"] == "circulation"]
    assert len(circ) == 2 and circ[0]["region"] == circ[1]["region"]
    assert len(room["regions"]) == 1


@pytest.mark.skipif(not RPLAN.exists(), reason="RPlan 19679 not available")
def test_rplan_19679():
    plan = json.loads(RPLAN.read_text())
    dec = decompose_plan(plan)
    res = label_parts(plan, dec)
    room = next(r for r in res["rooms"] if r["id"] == "r_4")
    drooms = next(r for r in dec["rooms"] if r["id"] == "r_4")
    assert len(room["parts"]) == 3
    by_size = {(round(p["short_m"], 2), round(p["long_m"], 2)): i for i, p in enumerate(drooms["parts"])}
    big = by_size[(4.92, 5.84)]
    roles = {p["index"]: p["role"] for p in room["parts"]}
    assert roles[big] == "primary"
    assert sorted(roles.values()) == ["circulation", "circulation", "primary"]
    assert len(room["regions"]) == 1 and room["pass_through"] is False
    d2 = [p["index"] for p in room["parts"] if "d_2" in p["door_ids"]]
    assert len(d2) == 1
    circ_regions = {p["region"] for p in room["parts"] if p["role"] == "circulation"}
    print("r_4 regions:", room["regions"], "distinct regions:", len(circ_regions))
    print("connections:", res["connections"])



# ---------------------------------------------------------------- rules v1
@pytest.mark.skipif(not RPLAN.exists(), reason="RPlan data not available")
def test_rplan_1789_absorption_and_pass_through():
    plan = json.loads((RPLAN.parent / "1789.json").read_text())
    dec = decompose_plan(plan)
    res = label_parts(plan, dec)
    room = next(r for r in res["rooms"] if r["id"] == "r_6")
    p = {x["index"]: x for x in room["parts"]}
    assert p[3]["role"] == "absorbed" and p[3]["absorbed_into"] == 4
    assert p[1]["role"] == "absorbed" and p[1]["absorbed_into"] == 4
    assert {"d_1", "d_2"} <= set(p[4]["door_ids"]) and p[1]["door_ids"] == []
    assert p[0]["role"] == "circulation" and p[4]["role"] == "circulation"
    assert p[0]["region"] != p[4]["region"] and len(room["regions"]) == 2
    assert p[2]["role"] == "primary" and p[2]["pass_through"] is True and room["pass_through"] is True
    assert p[3]["node"] == p[4]["node"] == "r_6/circ1"


def _block_plan(blocks, boundary, doors=(), entrance=None, extra_rooms=(), edges=()):
    """Hand-made decomposition: r_0 = primary (0..0.4 square) + ``blocks`` [(x0,y0,x1,y1)]."""
    def rect(b):
        return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
    parts = [{"polygon": rect((0, 0, 0.4, 0.4)), "sliver": False}] + [{"polygon": rect(b), "sliver": False} for b in blocks]
    plan = {"rooms": [{"id": "r_0", "label": "LivingRoom", "type": 3, "polygon": parts[0]["polygon"]}] + list(extra_rooms),
            "boundary": [[x, y, 0, 0] for x, y in boundary], "doors": list(doors), "entrance": entrance,
            "edges": list(edges)}
    dec = {"rooms": [{"id": "r_0", "label": "LivingRoom", "parts": parts}]}
    for r in extra_rooms:
        dec["rooms"].append({"id": r["id"], "label": r["label"], "parts": [{"polygon": r["polygon"], "sliver": False}]})
    return plan, dec


FAR = [(-0.5, -0.5), (1.5, -0.5), (1.5, 1.5), (-0.5, 1.5)]  # boundary touching nothing


def _door(x, y0, y1, i="d_0"):
    return {"id": i, "bbox": [x - 0.002, y0, x + 0.002, y1], "room_refs": ["r_9"]}


def test_v1_zone_door_facing_wide_block():
    blk = (0.4, 0.0, 0.4 + 2.5 / 18, 0.4)
    plan, dec = _block_plan([blk], FAR, doors=[_door(blk[2], 0.1, 0.2)])
    res = label_parts(plan, dec)
    assert res["rooms"][0]["parts"][1]["role"] == "zone"
    assert res["rooms"][0]["parts"][1]["node"] == "r_0/primary"


def test_v1_jog_storage_nook_alcove():
    w = lambda m: m / 18
    facade = [(0, 0), (0.4 + w(1.6), 0), (0.4 + w(1.6), 0.4), (0, 0.4)]
    cases = [(0.8, "jog", True), (0.8, "storage", False), (1.6, "nook", True), (1.2, "alcove", False)]
    for width, want, on_facade in cases:
        blk = (0.4, 0.1, 0.4 + w(width), 0.3)
        bnd = [(0, 0), (0.4 + w(width), 0), (0.4 + w(width), 0.4), (0, 0.4)] if on_facade else FAR
        plan, dec = _block_plan([blk], bnd)
        part = label_parts(plan, dec)["rooms"][0]["parts"][1]
        assert part["role"] == want, (width, on_facade, part)
    assert label_parts(*_block_plan([(0.4, 0.1, 0.4 + w(0.8), 0.3)], FAR))["rooms"][0]["parts"][1]["node"] == "r_0/stor0"


def test_v1_thin_strip_absorbed_into_circulation():
    circ = (0.4, 0.0, 0.6, 0.1)
    strip = (0.4, 0.1, 0.6, 0.1 + 0.3 / 18)
    top = (0.4, 0.1 + 0.3 / 18, 0.6, 0.4)
    plan, dec = _block_plan([circ, strip, top], FAR, doors=[_door(0.6, 0.02, 0.08)])
    res = label_parts(plan, dec)
    p = res["rooms"][0]["parts"]
    assert p[1]["role"] == "circulation"
    assert p[2]["role"] == "absorbed" and p[2]["absorbed_into"] == 1 and p[2]["node"] == p[1]["node"]
    assert p[3]["role"] == "alcove"


def test_v1_open_edge_resolves_to_node():
    circ = (0.4, 0.0, 0.5, 0.1)
    r1 = {"id": "r_1", "label": "Kitchen", "type": 2,
          "polygon": [[0.5, 0.0], [0.7, 0.0], [0.7, 0.1], [0.5, 0.1]]}
    plan, dec = _block_plan([circ], FAR, entrance=[0.42, -0.002, 0.48, 0.002], extra_rooms=[r1],
                            edges=[{"u": "r_0", "v": "r_1", "connection": 2}])
    res = label_parts(plan, dec)
    op = [c for c in res["connections"] if c["kind"] == "open"]
    assert op == [{"door_id": None, "kind": "open", "edge": ["r_0", "r_1"], "a": "r_0/circ0", "b": "r_1"}]
    assert res["summary"]["n_open_unresolved"] == 0
    # wall contacts: kitchen touches the circulation node, not the primary -> not openable
    assert res["rooms"][0]["kitchen_openable"] is False
    assert any(c["room_id"] == "r_1" for c in res["rooms"][0]["wall_contacts"])


def test_empty_room_polygon_does_not_crash():
    """RPlan 67697 has a room (r_1) with an empty polygon; labelling must skip it."""
    import json
    from packages.floorplan_core.rect_decompose import decompose_plan
    from packages.floorplan_core.part_labels import label_parts
    p = json.load(open("/Users/camilo/Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json/67697.json"))
    out = label_parts(p, decompose_plan(p))
    json.dumps(out)
    assert any(r["id"] == "r_7" for r in out["rooms"])


def test_45549_dining_candidate():
    """Zone facing the kitchen door -> dining?; the zone that only touches it stays a zone."""
    import json
    from packages.floorplan_core.rect_decompose import decompose_plan
    from packages.floorplan_core.part_labels import label_parts
    p = json.load(open("/Users/camilo/Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json/45549.json"))
    out = label_parts(p, decompose_plan(p))
    r5 = next(r for r in out["rooms"] if r["id"] == "r_5")
    by = {q["index"]: q for q in r5["parts"]}
    assert by[1]["role"] == "zone" and by[1]["dining_candidate"] and by[1]["label"] == "dining?"
    assert by[0]["role"] == "zone" and not by[0]["dining_candidate"]
    assert by[1]["node"] == by[2]["node"] == "r_5/primary"


def _part_v12(pid, rid, idx, **kw):
    import json as _json
    from packages.floorplan_core.rect_decompose import decompose_plan as _dp
    from packages.floorplan_core.part_labels import label_parts as _lp
    plan = _json.load(open(RPLAN.parent / f"{pid}.json"))
    lab = _lp(plan, _dp(plan), **kw)
    return next(r for r in lab["rooms"] if r["id"] == rid)["parts"][idx]


def test_v12_partial_projection_is_jog():
    # RPlan 14417 r_2 p0: 0.28 m bay on a master bedroom, cover 0.28
    p = _part_v12("14417", "r_2", 0)
    assert p["role"] == "jog" and p["proj_cover"] < 0.5
    assert _part_v12("14417", "r_2", 0, jog_min_m=None)["role"] == "absorbed"


def test_v12_full_length_strip_is_absorbed():
    # RPlan 12556 r_4 p2: 0.49 m strip along the room's whole wall, room cover 1.00
    p = _part_v12("12556", "r_4", 2)
    assert p["role"] == "absorbed" and p["proj_cover"] >= 0.99


def test_v12_cover_is_against_the_whole_room():
    # RPlan 17683 r_3 p1: 0.28 m strip in front of a lobby; the wall continues into the
    # living room, so room cover is 0.43 (neighbour-part cover was 1.00) -> jog
    p = _part_v12("17683", "r_3", 1)
    assert p["role"] == "jog" and p["proj_cover"] < 0.5


def test_v12_raster_floor():
    # RPlan 34971 r_6 p0: 0.18 m strip, below the 0.25 m floor -> absorbed, no cover computed
    p = _part_v12("34971", "r_6", 0)
    assert p["role"] == "absorbed" and "proj_cover" not in p
