import copy
import json
from pathlib import Path

import pytest
from shapely.geometry import Point, Polygon

from packages.floorplan_core import layout_cpsat
from packages.floorplan_core.generate import generate
from packages.floorplan_core.node_graph import build_node_graph, expand_plan
from packages.floorplan_core.spec import (
    CPSAT_SEED, CPSAT_TIME_S, CPSAT_WORKERS, build_options, describe, fit, spec_problems, validate_spec,
)

DATA = Path("/Users/camilo/Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json")
BOUNDARY = [[0, 0], [1, 0], [1, 1], [0, 1]]


def _good_spec():
    return {
        "boundary": copy.deepcopy(BOUNDARY),
        "edge_types": ["external", "external", "facing_corridor", "external"],
        "mm_per_unit": 10000.0,
        "nodes": [
            {"id": "bed", "room_id": "bed", "kind": "whole", "label": "bedroom", "pos": [0.25, 0.25]},
            {"id": "liv", "room_id": "liv", "kind": "whole", "label": "living", "pos": [0.75, 0.75]},
            {"id": "hall", "room_id": "hall", "kind": "whole", "label": "hallway", "pos": [0.75, 0.25]},
        ],
        "adjacencies": [{"u": "bed", "v": "hall", "connection": 1},
                        {"u": "liv", "v": "hall", "connection": 2}],
        "demands": {
            "bed": {"min_edge": 2400.0, "min_area": 9e6, "aspect_max": 2.5, "area_frac": 0.4},
            "liv": {"min_edge": 3000.0, "min_area": 1.2e7, "aspect_max": 2.5, "area_frac": 0.4},
            "hall": {"min_edge": 900.0, "min_area": 0.0, "aspect_max": 6.0,
                     "max_short": 1500.0, "area_frac": 0.2},
        },
        "boundary_demands": {"liv": [0]},
        "entrance": [0.5, 0.0, 0.5, 0.0],
        "entry_room": "hall",
        "precedence": [],
    }


def test_good_spec_is_valid():
    assert spec_problems(_good_spec()) == []
    validate_spec(_good_spec())


def _breach_demand_id(s):
    s["demands"]["ghost"] = {"min_edge": 1.0}


def _breach_adjacency_id(s):
    s["adjacencies"].append({"u": "bed", "v": "ghost", "connection": 0})


def _breach_boundary_demand_id(s):
    s["boundary_demands"]["ghost"] = [0]


def _breach_entry_room(s):
    s["entry_room"] = "ghost"


def _breach_precedence(s):
    s["precedence"] = [["bed", "ghost"]]


def _breach_edge_types(s):
    s["edge_types"] = ["external"]


def _breach_edge_index(s):
    s["boundary_demands"]["liv"] = [9]


def _breach_connection(s):
    s["adjacencies"][0]["connection"] = 3


def _breach_max_short(s):
    s["demands"]["hall"]["max_short"] = 800.0


def _breach_max_area(s):
    s["demands"]["bed"]["max_area"] = 1.0


def _breach_entrance(s):
    s["entrance"] = [0.5, 0.5, 0.5, 0.5]


def _breach_area_frac(s):
    s["demands"]["bed"]["area_frac"] = 0.9


def _breach_kind(s):
    s["nodes"][0]["kind"] = "blob"


def _breach_duplicate(s):
    s["nodes"][1]["id"] = "bed"


@pytest.mark.parametrize("breach", [
    _breach_demand_id, _breach_adjacency_id, _breach_boundary_demand_id,
    _breach_entry_room, _breach_precedence, _breach_edge_types, _breach_edge_index,
    _breach_connection, _breach_max_short, _breach_max_area, _breach_entrance,
    _breach_area_frac, _breach_kind, _breach_duplicate,
])
def test_validate_rejects_breach(breach):
    s = _good_spec()
    breach(s)
    assert spec_problems(s)
    with pytest.raises(ValueError):
        validate_spec(s)
    with pytest.raises(ValueError):
        generate(s)


def test_entrance_none_is_allowed():
    s = _good_spec()
    s["entrance"] = None
    validate_spec(s)


def test_generate_does_not_change_spec_or_options():
    s = _good_spec()
    s["precedence"] = [["bed", "liv"]]
    opts = build_options()
    s0, o0 = copy.deepcopy(s), copy.deepcopy(opts)
    plan = generate(s, opts)
    assert plan["rooms"]
    assert s == s0
    assert opts == o0
    generate(s)                       # options=None builds defaults
    assert s == s0


def test_build_options_defaults_match_solver():
    o = build_options()
    assert (o["cpsat_seed"], o["cpsat_workers"], o["cpsat_time_s"]) == (
        layout_cpsat.RANDOM_SEED, layout_cpsat.NUM_WORKERS, layout_cpsat.TIME_LIMIT_S)
    assert (CPSAT_SEED, CPSAT_WORKERS, CPSAT_TIME_S) == (2026, 8, 5.0)
    assert o["core_placement"] is False and o["core_block"] is True and o["area_lift"] is True


def test_aspect_pairs():
    assert layout_cpsat._aspect_ratio(2.5) == (5, 2)
    assert layout_cpsat._aspect_ratio(6.0) == (6, 1)


def _plain_plan():
    sq = lambda x0, y0, x1, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]  # noqa: E731
    return {
        "rooms": [
            {"id": "r_0", "label": "Bedroom", "polygon": sq(0, 0, .5, .5)},
            {"id": "r_1", "label": "Living", "polygon": sq(.5, 0, 1, .5)},
            {"id": "r_2", "label": "Hallway", "polygon": sq(0, .5, 1, 1)},
        ],
        "edges": [{"u": "r_0", "v": "r_2", "connection": 1, "relation": 2},
                  {"u": "r_1", "v": "r_2", "connection": 0, "open": True},
                  {"u": "r_0", "v": "r_1", "type": "door"},
                  {"u": "r_0", "v": "r_9", "connection": 1}],
    }


def test_describe_fit_plain_rooms_are_whole():
    spec = fit(describe(_plain_plan()), BOUNDARY, mm_per_unit=10000.0,
               entrance=[0.5, 0.0, 0.5, 0.0])
    assert all(n["room_id"] == n["id"] and n["kind"] == "whole" for n in spec["nodes"])
    assert "node_assignment" not in spec
    hall = spec["demands"]["r_2"]
    assert hall["aspect_max"] == 6.0 and hall["max_short"] == 1500.0
    assert spec["demands"]["r_0"]["aspect_max"] == 2.5 and "max_short" not in spec["demands"]["r_0"]
    validate_spec(spec)


def test_describe_fit_adjacencies_hold_connection_only():
    spec = fit(describe(_plain_plan()), BOUNDARY)
    for e in spec["adjacencies"]:
        assert set(e) == {"u", "v", "connection"}
        assert e["connection"] in (0, 1, 2)
    by = {frozenset((e["u"], e["v"])): e["connection"] for e in spec["adjacencies"]}
    assert by[frozenset(("r_0", "r_2"))] == 1
    assert by[frozenset(("r_1", "r_2"))] == 2          # open: True
    assert by[frozenset(("r_0", "r_1"))] == 1          # type: door
    assert frozenset(("r_0", "r_9")) not in by         # r_9 is not a node


@pytest.mark.skipif(not DATA.exists(), reason="RPlan data not available")
@pytest.mark.parametrize("pid", ["1789", "19679", "45549", "6079"])
def test_describe_fit_expanded_plan_has_room_id_and_kind(pid):
    plan = json.loads((DATA / f"{pid}.json").read_text())
    exp = expand_plan(plan, build_node_graph(plan))
    by_id = {r["id"]: r for r in exp["rooms"]}
    spec = fit(describe(exp), BOUNDARY)
    assert {n["id"] for n in spec["nodes"]} == set(by_id)
    kinds = set()
    for n in spec["nodes"]:
        assert n["room_id"] == by_id[n["id"]]["room_id"]
        assert n["kind"] == by_id[n["id"]]["node_kind"]
        kinds.add(n["kind"])
        if n["kind"] == "whole":
            assert n["room_id"] == n["id"]
        else:
            assert n["id"].startswith(n["room_id"] + "/")
    for e in spec["adjacencies"]:
        assert set(e) == {"u", "v", "connection"}
    # circulation nodes carry the hallway limits
    for n in spec["nodes"]:
        if n["label"] == "hallway":
            d = spec["demands"][n["id"]]
            assert d["aspect_max"] == 6.0 and d["max_short"] == 1500.0
    assert abs(sum(d["area_frac"] for d in spec["demands"].values()) - 1.0) < 1e-9


# ── describe / fit ───────────────────────────────────────────────────────────

@pytest.mark.skipif(not DATA.exists(), reason="RPlan data not available")
@pytest.mark.parametrize("pid", ["1789", "19679", "45549", "6079", "26135"])
def test_describe_is_a_valid_spec_of_the_plan_itself(pid):
    plan = json.loads((DATA / f"{pid}.json").read_text())
    for p in (plan, expand_plan(plan, build_node_graph(plan))):
        spec = describe(p, mm_per_unit=18000.0)
        validate_spec(spec)
        assert spec["boundary"] == [[float(q[0]), float(q[1])] for q in p["boundary"]]
        polys = {r["id"]: Polygon(r["polygon"]) for r in p["rooms"]}
        for n in spec["nodes"]:
            assert polys[n["id"]].buffer(1e-9).contains(Point(n["pos"])), n["id"]
        assert spec["boundary_demands"]


@pytest.mark.skipif(not DATA.exists(), reason="RPlan data not available")
def test_fit_replaces_given_fields_keeps_the_rest_and_is_pure():
    plan = json.loads((DATA / "26135.json").read_text())
    src = describe(plan, mm_per_unit=18000.0)
    before = copy.deepcopy(src)
    rid = str(src["nodes"][0]["id"])
    target = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    topo = {"room_edges": {rid: [{"target_edge_index": 2}]}, "entry_room": "ignored"}
    out = fit(src, target, edge_types=["external"] * 4, entrance=[0.5, 0.2, 0.5, 0.2],
              mm_per_unit=18000.0, node_pos={rid: [0.3, 0.3]}, topo=topo)
    assert src == before
    assert out["boundary"] == target and out["entrance"] == [0.5, 0.2, 0.5, 0.2]
    assert out["boundary_demands"] == {rid: [2]}
    assert out["nodes"][0]["pos"] == [0.3, 0.3]
    assert out["nodes"][1]["pos"] == src["nodes"][1]["pos"]
    assert out["adjacencies"] == src["adjacencies"] and out["demands"] == src["demands"]
    assert out["entry_room"] == src["entry_room"] != "ignored"
