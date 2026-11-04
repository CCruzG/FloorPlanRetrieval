import gzip
import json
from pathlib import Path

import pytest
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from packages.floorplan_core.node_graph import (anchor_node, build_node_graph, current_meta, display_graph, entry_node, entry_room_of, expand_plan, is_stale, load_sidecar,
                                                sidecar_meta)

DATA = Path("/Users/camilo/Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json")
GATE = ("1789", "19679", "45549", "6079")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="RPlan data not available")


def _plan(pid):
    return json.loads((DATA / f"{pid}.json").read_text())


@pytest.mark.parametrize("pid", GATE)
def test_nodes_tile_rooms(pid):
    plan = _plan(pid)
    g = build_node_graph(plan)
    json.dumps(g)
    for room in plan["rooms"]:
        ns = [n for n in g["nodes"] if n["room_id"] == room["id"]]
        assert ns, room["id"]
        u = unary_union([Polygon(r) for n in ns for r in n["polygon"]])
        src = Polygon(room["polygon"])
        assert u.symmetric_difference(src).area <= 1e-6 * max(src.area, 1e-9)
        assert all(n["id"] == room["id"] or n["id"].startswith(room["id"] + "/") for n in ns)


@pytest.mark.parametrize("pid", GATE)
def test_edges_reference_nodes(pid):
    g = build_node_graph(_plan(pid))
    ids = {n["id"] for n in g["nodes"]} | {"exterior"}
    for e in g["edges"]:
        assert e["a"] in ids and e["b"] in ids, e
        assert e["kind"] in ("door", "open", "block", "wall")
    assert any(e["b"] == "exterior" or e["a"] == "exterior" for e in g["edges"])


def test_skipped_rooms_are_whole():
    g = build_node_graph(_plan("16744"))
    for n in g["nodes"]:
        if n["room_label"] in ("Balcony", "Kitchen", "Bathroom"):
            assert n["kind"] == "whole"


def test_sidecar_roundtrip(tmp_path):
    p = tmp_path / "ng.jsonl.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(json.dumps(sidecar_meta()) + "\n")
        for pid in GATE:
            fh.write(json.dumps(build_node_graph(_plan(pid))) + "\n")
    meta, graphs = load_sidecar(str(p))
    assert set(graphs) == set(GATE) and not is_stale(meta)
    assert is_stale({**meta, "label_params": {**meta["label_params"], "jog_max_cover": 0.85}})
    assert meta["rules_version"] == current_meta()["rules_version"]


@pytest.mark.parametrize("pid", GATE)
def test_distribution_ratio(pid):
    m = build_node_graph(_plan(pid))["metrics"]
    r = m["distribution_ratio"]
    assert r is None or 0.0 <= r <= 1.0
    assert m["living_reach_via_circ"] <= m["living_reach"]


@pytest.mark.parametrize("pid", GATE + ("26135",))
def test_display_graph(pid):
    plan = _plan(pid)
    g = display_graph(plan, build_node_graph(plan))
    json.dumps(g)
    ids = {n["id"] for n in g["nodes"]}
    assert {n["room_id"] for n in g["nodes"]} == {r["id"] for r in plan["rooms"]}
    for e in g["edges"]:
        assert e["a"] in ids and e["b"] in ids
    for n in g["nodes"]:
        u = unary_union([Polygon(r) for r in n["polygon"]])
        assert u.buffer(1e-6).contains(Point(n["pos"])), n["id"]


def test_display_graph_splits_circulation_into_arms():
    plan = _plan("26135")
    g = display_graph(plan, build_node_graph(plan))
    arms = [n for n in g["nodes"] if n["kind"] == "circ"]
    assert len(arms) >= 2 and all(n["id"][-1].isalpha() for n in arms)
    arm_ids = {n["id"] for n in arms}
    assert any(e["kind"] == "block" and e["a"] in arm_ids and e["b"] in arm_ids for e in g["edges"])


def test_anchor_node():
    plan = _plan("26135")
    ex = expand_plan(plan, build_node_graph(plan))
    living = next(r["id"] for r in plan["rooms"] if r["label"] == "LivingRoom")
    assert anchor_node(ex, living) == f"{living}/primary"
    whole = next(r["id"] for r in ex["rooms"] if r["node_kind"] == "whole")
    assert anchor_node(ex, whole) == whole
    assert anchor_node(ex, "r_99") == "r_99"


def test_entry_node_is_kept():
    plan = _plan("26135")
    g = build_node_graph(plan)
    assert entry_node(g) == next(n["id"] for n in g["nodes"] if n["kind"] == "primary")
    assert expand_plan(plan, g)["entry_node"] == entry_node(g)
    assert display_graph(plan, g)["entry_node"] == entry_node(g)
    assert all("exterior" not in (e["a"], e["b"]) for e in display_graph(plan, g)["edges"])


def test_entry_room_of():
    plan = _plan("26135")
    g = build_node_graph(plan)
    primary = entry_node(g)
    assert entry_room_of(plan) == primary.split("/")[0]
    assert entry_room_of(expand_plan(plan, g)) == primary
