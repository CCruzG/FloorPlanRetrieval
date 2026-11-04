"""Tests for packages.floorplan_core.rect_decompose (metre-scale synthetic cases)."""
import json
from pathlib import Path

import pytest
from shapely.geometry import Polygon
from shapely.ops import unary_union

from packages.floorplan_core.rect_decompose import decompose, decompose_plan

RPLAN_DIR = Path("/Users/camilo/Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json")

SL = dict(sliver_max=0.2, merge_min_contact=1.2)

# plus: shortest-cut greedy gives 4 parts (arm, arm, arm, then 2 left); minimum is 3.
# name -> (coords, expected parts, expected status, kwargs)
CASES = {
    "rectangle": ([(0, 0), (4, 0), (4, 3), (0, 3)], 1, "rectangle", {}),
    "L": ([(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)], 2, "ok", {}),
    "T": ([(0, 2), (3, 2), (3, 0), (5, 0), (5, 2), (8, 2), (8, 4), (0, 4)], 3, "ok", {}),
    "U": ([(0, 0), (6, 0), (6, 4), (4, 4), (4, 2), (2, 2), (2, 4), (0, 4)], 3, "ok", {}),
    "Z": ([(0, 0), (4, 0), (4, 2), (6, 2), (6, 4), (2, 4), (2, 2), (0, 2)], 3, "ok", {}),
    "plus": ([(2, 0), (4, 0), (4, 2), (6, 2), (6, 4), (4, 4), (4, 6), (2, 6), (2, 4), (0, 4), (0, 2), (2, 2)], 4, "ok", {}),
    "staircase": ([(0, 0), (6, 0), (6, 1), (4, 1), (4, 2), (2, 2), (2, 3), (0, 3)], 3, "ok", {}),
    "collinear_vertex": ([(0, 0), (2, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)], 2, "ok", {}),
    "repeated_vertex": ([(0, 0), (4, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)], 2, "ok", {}),
    "jog_merged": ([(0, 0), (4.2, 0), (4.2, 3), (4, 3), (4, 4), (0, 4)], 1, "ok", SL),
    "jog_kept": ([(0, 0), (4.2, 0), (4.2, 1), (4, 1), (4, 3), (0, 3)], 2, "ok", SL),
    "jog_sliver_off": ([(0, 0), (4.2, 0), (4.2, 3), (4, 3), (4, 4), (0, 4)], 2, "ok", {}),
}


def _poly(coords):
    return Polygon(coords)


def check_invariants(coords, res):
    inp = _poly(coords)
    polys = [Polygon(p["polygon"]) for p in res["parts"]]
    uni = unary_union(polys)
    assert uni.symmetric_difference(inp).area <= 1e-9 * inp.area
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            assert polys[i].intersection(polys[j]).area <= 1e-9 * inp.area
    for p, g in zip(res["parts"], polys):
        if p["is_rect"]:
            assert len(g.exterior.coords) - 1 == 4
            assert abs(g.area - g.envelope.area) <= 1e-9 * g.area
        assert p["polygon"][0] != p["polygon"][-1]
    json.dumps(res)


@pytest.mark.parametrize("name", list(CASES))
def test_case(name):
    coords, n, status, kw = CASES[name]
    res = decompose(coords, **kw)
    assert res["status"] == status
    assert len(res["parts"]) == n
    if status == "ok":
        check_invariants(coords, res)
        if not res["reversed_cuts"]:
            assert len(res["cuts"]) == len(res["parts"]) - 1
    if status == "rectangle":
        assert res["parts"][0]["is_rect"]


@pytest.mark.parametrize("name", ["L", "T", "U", "Z", "plus", "staircase"])
def test_orientation(name):
    coords = CASES[name][0]
    a = decompose(coords)
    b = decompose(coords[::-1])
    assert b["status"] == "ok" and len(b["parts"]) == len(a["parts"])
    check_invariants(coords, b)
    assert sorted(map(str, a["parts"])) == sorted(map(str, b["parts"]))


def test_accepts_shapely_polygon():
    res = decompose(Polygon(CASES["L"][0]))
    assert res["status"] == "ok" and len(res["parts"]) == 2


def test_sloped_edge():
    res = decompose([(0, 0), (4, 0), (4, 2), (2, 3), (0, 2)])
    assert res["status"] == "not_rectilinear"
    assert len(res["parts"]) == 1 and not res["parts"][0]["is_rect"]


def test_self_intersecting():
    res = decompose([(0, 0), (4, 4), (4, 0), (0, 4)])
    assert res["status"] == "invalid"
    assert len(res["parts"]) == 1 and not res["parts"][0]["is_rect"]


def test_hole_invalid():
    p = Polygon([(0, 0), (6, 0), (6, 6), (0, 6)], [[(2, 2), (4, 2), (4, 4), (2, 4)]])
    assert decompose(p)["status"] == "invalid"


def test_degenerate_does_not_raise():
    assert decompose([])["status"] == "invalid"
    assert decompose([(0, 0), (1, 1)])["status"] == "invalid"


def test_jog_merged_details():
    res = decompose(*[CASES["jog_merged"][0]], **SL)
    assert len(res["reversed_cuts"]) == 1 and res["cuts"] == []
    assert not res["parts"][0]["is_rect"]
    assert res["parts"][0]["sliver"] is False


def test_jog_kept_details():
    res = decompose(CASES["jog_kept"][0], **SL)
    assert res["reversed_cuts"] == []
    assert len(res["cuts"]) == 1
    assert sum(p["sliver"] for p in res["parts"]) == 1


def test_sliver_disabled():
    res = decompose(CASES["jog_sliver_off"][0])
    assert res["reversed_cuts"] == [] and not any(p["sliver"] for p in res["parts"])


@pytest.mark.skipif(not RPLAN_DIR.exists(), reason="RPlan json absent")
def test_decompose_plan_rplan():
    for pid in ("0", "1", "10"):
        plan = json.load(open(RPLAN_DIR / f"{pid}.json"))
        out = decompose_plan(plan)
        json.dumps(out)
        assert out["summary"]["n_rooms"] == len(plan["rooms"])
        for room, src in zip(out["rooms"], plan["rooms"]):
            if room["status"] in ("ok", "rectangle"):
                inp = Polygon(src["polygon"])
                uni = unary_union([Polygon(p["polygon"]) for p in room["parts"]])
                assert uni.symmetric_difference(inp).area <= 1e-9 * inp.area
            for p in room["parts"]:
                assert "short_m" in p and "long_m" in p


def test_decompose_plan_skips_balcony():
    # RPlan 16744 r_5: stepped balcony (two 0.35/0.42 m flanks) -- excluded 2026-11-03
    plan = json.load(open(RPLAN_DIR / "16744.json"))
    out = decompose_plan(plan)
    bal = next(r for r in out["rooms"] if r["id"] == "r_5")
    assert bal["label"] == "Balcony" and bal["status"] == "skipped" and len(bal["parts"]) == 1
    assert Polygon(bal["parts"][0]["polygon"]).equals(Polygon(plan["rooms"][5]["polygon"]))
    for r in out["rooms"]:
        if r["label"] in ("Kitchen", "Bathroom"):
            assert r["status"] == "skipped" and len(r["parts"]) == 1
    full = decompose_plan(plan, skip_labels=())
    assert len(next(r for r in full["rooms"] if r["id"] == "r_5")["parts"]) == 3
