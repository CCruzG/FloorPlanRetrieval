"""Massing: storeys, heights, prisms, windows, ground-floor entrance, core, fallback."""
import sys
from pathlib import Path

import pytest
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.floor_plate_design import compose as C  # noqa: E402
from packages.floor_plate_design import massing as M  # noqa: E402

K = C.METRES_PER_UNIT


def rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def app(poly):
    return [[x / K, y / K] for x, y in poly]


def plan_for(x0, y0, x1, y1, ok=True):
    """Two rooms split at the middle of x and one window on the south wall (y = y0) of room a."""
    xm = (x0 + x1) / 2
    a, b = rect(x0, y0, xm, y1), rect(xm, y0, x1, y1)
    return {
        "rooms": [{"id": "a", "type": "living", "source_type": 3, "polygon": app(a)},
                  {"id": "b", "type": "bedroom", "source_type": 1, "polygon": app(b)}],
        "edges": [], "node_pos": {}, "doors": [],
        "windows": [{"room": "a", "edge_index": 0, "centre": [(x0 + xm) / 2 / K, y0 / K],
                     "wall": [[x0 / K, y0 / K], [xm / K, y0 / K]], "width_mm": 1800, "sill_mm": 0,
                     "head_mm": 2400, "kind": "window"}],
        "feasible": True, "report": {},
    }


def assembler():
    return {
        "boundary": rect(0, 0, 20, 10),
        "cores": [{"id": "core-1", "polygon": rect(16, 0, 20, 10), "service_polygon": rect(16, 4, 20, 10),
                   "lobby_polygon": rect(16, 0, 20, 4)}],
        "corridors": [{"id": "h1", "polygon": rect(8, 0, 9, 10)}],
        "units": [
            {"id": "u1", "type": "1B1b", "polygon": rect(0, 0, 8, 10), "area": 80},
            {"id": "u2", "type": "2B1b", "polygon": rect(9, 0, 16, 10), "area": 70},
        ],
    }


def composed(fail_u2=False):
    c = C.new_composed(assembler())
    s1, s2 = c.slots["u1"], c.slots["u2"]
    s1.plan, s1.status = plan_for(0, 0, 8, 10), C.GENERATED
    if fail_u2:
        s2.status, s2.error = C.FAILED, "x"
    else:
        s2.plan, s2.status = plan_for(9, 0, 16, 10), C.GENERATED
    return c


def stems(b, storey):
    return {p["stem"]: p["polygon"] for p in b["prisms"] if p["storey"] == storey}


def test_storey_count_follows_input():
    for n in (1, 3, 5):
        b = M.building_from_composed(composed(), n, 3.1, main_entrance_m=[4, 0])
        assert b["storeys"] == n and b["storeys_requested"] == n and len(b["heights"]) == n
    assert b["heights"][0]["z1"] - b["heights"][0]["z0"] == pytest.approx(3.6)
    assert b["heights"][1]["z1"] - b["heights"][1]["z0"] == pytest.approx(3.1)
    assert b["total_height_m"] == pytest.approx(3.6 + 4 * 3.1 + M.SLAB_M)


def test_max_height_caps_storeys_and_reports_both():
    b = M.building_from_composed(composed(), 10, 3.1, max_height_m=14.0)
    assert b["storeys_requested"] == 10 and b["storeys"] == 4
    assert b["total_height_m"] <= 14.0
    assert b["report"]["storeys_requested"] == 10 and b["report"]["storeys_achieved"] == b["storeys"]
    assert b["report"]["capped_by_max_height"] is True
    low = M.building_from_composed(composed(), 4, 3.1, max_height_m=1.0)
    assert low["storeys"] == 1


def test_max_height_exact_numbers():
    # 1 storey 3.85, 2 storeys 6.95, 3 storeys 10.05, 4 storeys 13.15, 5 storeys 16.25
    assert M.building_from_composed(composed(), 10, 3.1, max_height_m=13.2)["storeys"] == 4
    assert M.building_from_composed(composed(), 10, 3.1, max_height_m=13.1)["storeys"] == 3
    assert M.building_from_composed(composed(), 2, 3.1, max_height_m=99)["storeys"] == 2


def test_typical_storeys_repeat_xy_and_window_walls():
    b = M.building_from_composed(composed(), 4, 3.1, main_entrance_m=[4, 0])
    ref = stems(b, 1)
    assert any(s.endswith("/a") for s in ref) and "slab" in ref
    for i in (2, 3):
        assert stems(b, i) == ref
    walls = {i: sorted((w["unit_id"], w["wall"]) for w in b["windows"] if w["storey"] == i) for i in (1, 2, 3)}
    assert walls[1] == walls[2] == walls[3] and walls[1]


def test_no_z_overlap_in_a_column():
    b = M.building_from_composed(composed(), 4, 3.1, main_entrance_m=[4, 0])
    ps = [p for p in b["prisms"] if p["role"] != "core"]
    for i, p in enumerate(ps):
        for q in ps[i + 1:]:
            if Polygon(p["polygon"]).intersection(Polygon(q["polygon"])).area > 1e-6:
                assert p["z1"] <= q["z0"] + 1e-9 or q["z1"] <= p["z0"] + 1e-9, (p["id"], q["id"])
    slabs = sorted((p for p in ps if p["role"] in ("slab", "roof")), key=lambda p: p["z0"])
    assert len(slabs) == 5 and slabs[-1]["role"] == "roof"
    for lower, upper in zip(slabs, slabs[1:]):
        assert any(r["z0"] >= lower["z1"] - 1e-9 and r["z1"] <= upper["z0"] + 1e-9
                   for r in ps if r["role"] == "room")


def test_window_z_inside_storey_room_height():
    b = M.building_from_composed(composed(), 4, 3.1, main_entrance_m=[4, 0])
    assert b["windows"]
    for w in b["windows"]:
        h = b["heights"][w["storey"]]
        assert h["z0"] + M.SLAB_M - 1e-9 <= w["z0"] < w["z1"] <= h["z1"] + 1e-9
        assert w["role"] == "window" and w["width_m"] == pytest.approx(1.8)


def test_ground_floor_entrance_lobby():
    c = composed()
    b = M.building_from_composed(c, 3, 3.1, main_entrance_m=[4, 0])  # inside u1
    g = [p for p in b["prisms"] if p["storey"] == 0]
    assert [p["stem"] for p in g if p["role"] == "entrance_lobby"] == ["u1/entrance_lobby"]
    assert not any(p["stem"].startswith("u1/") and p["role"] == "room" for p in g)
    assert not any(w["unit_id"] == "u1" and w["storey"] == 0 for w in b["windows"])
    assert any(w["unit_id"] == "u2" and w["storey"] == 0 for w in b["windows"])
    assert any(p["stem"].startswith("u1/") and p["role"] == "room" for p in b["prisms"] if p["storey"] == 1)
    assert b["report"]["entrance_unit"] == "u1"
    # nearest unit when the point is outside every unit
    b2 = M.building_from_composed(c, 2, 3.1, main_entrance_m=[15.9, -2])
    assert b2["report"]["entrance_unit"] == "u2"


def test_no_entrance_means_no_lobby_prism():
    b = M.building_from_composed(composed(), 3, 3.1)
    assert not any(p["role"] == "entrance_lobby" for p in b["prisms"])
    assert "equals a typical storey" in b["report"]["entrance"]
    assert stems(b, 0).keys() == stems(b, 1).keys()


def test_entrance_read_from_options():
    c = composed()
    c.options["main_entrance_m"] = [4, 0]
    b = M.building_from_composed(c, 2, 3.1)
    assert sum(p["role"] == "entrance_lobby" for p in b["prisms"]) == 1


def test_core_spans_full_height_plus_overrun_once():
    b = M.building_from_composed(composed(), 4, 3.1, main_entrance_m=[4, 0])
    cores = [p for p in b["prisms"] if p["role"] == "core"]
    assert len(cores) == 1
    assert cores[0]["z0"] == 0 and cores[0]["z1"] == pytest.approx(b["total_height_m"] + M.CORE_OVERRUN_M)


def test_failed_slot_emits_fallback_without_windows():
    b = M.building_from_composed(composed(fail_u2=True), 3, 3.1)
    fb = [p for p in b["prisms"] if p["role"] == "unit_fallback"]
    assert len(fb) == 3 and {p["stem"] for p in fb} == {"u2/unit_fallback"}
    assert not any(w["unit_id"] == "u2" for w in b["windows"])
    assert b["report"]["unit_fallbacks"] == ["u2"]


def test_prism_count_formula():
    b = M.building_from_composed(composed(), 4, 3.1, main_entrance_m=[4, 0])
    typical = len(stems(b, 1))
    ground = len(stems(b, 0))
    assert len(b["prisms"]) == ground + 3 * typical + 1 + 1  # roof, core


def test_bad_input():
    with pytest.raises(ValueError):
        M.building_from_composed(composed(), 0, 3.1)
    with pytest.raises(ValueError):
        M.building_from_composed(composed(), 3, 0.1)


def _slab_polys(b):
    return [Polygon(p["polygon"], p.get("holes") or []) for p in b["prisms"] if p["role"] in ("slab", "roof")]


def test_no_slab_intersects_core_footprint():
    b = M.building_from_composed(composed(), 4, 3.1, main_entrance_m=[4, 0])
    core = Polygon(assembler()["cores"][0]["service_polygon"])
    slabs = [p for p in b["prisms"] if p["role"] in ("slab", "roof")]
    assert len(slabs) == 5
    for p in slabs:
        assert Polygon(p["polygon"], p.get("holes") or []).intersection(core).area < 1e-6, p["id"]


def test_slab_with_interior_core_gets_a_hole():
    a = assembler()
    a["cores"][0]["service_polygon"] = rect(16, 3, 19, 7)  # interior: slab keeps a hole
    c = C.new_composed(a)
    b = M.building_from_composed(c, 2, 3.1, ground_slab_offset_m=0)
    roof = [p for p in b["prisms"] if p["role"] == "roof"]
    assert len(roof) == 1 and len(roof[0]["holes"]) == 1
    assert Polygon(roof[0]["polygon"], roof[0]["holes"]).area == pytest.approx(200 - 12)


def test_slab_split_by_core_gives_one_prism_per_part():
    a = assembler()
    a["cores"][0]["service_polygon"] = rect(9, -1, 11, 11)  # cuts the 20 x 10 outline in two
    b = M.building_from_composed(C.new_composed(a), 2, 3.1, ground_slab_offset_m=0)
    roofs = [p for p in b["prisms"] if p["role"] == "roof"]
    assert len(roofs) == 2 and len({p["id"] for p in roofs}) == 2
    assert sum(Polygon(p["polygon"]).area for p in roofs) == pytest.approx(180)


def test_living_dining_sill0_windows_carry_transom():
    b = M.building_from_composed(composed(), 3, 3.1, main_entrance_m=[4, 0])
    assert b["windows"] and all(w["room"] == "a" for w in b["windows"])
    assert all(w["transom_m"] == 0.9 for w in b["windows"])
    # bedroom window (sill 900) and a window in a bedroom with sill 0 get none
    c = composed()
    pl = c.slots["u2"].plan
    pl["windows"][0]["room"] = "b"
    pl["windows"].append(dict(pl["windows"][0], room="b", sill_mm=900))
    b2 = M.building_from_composed(c, 2, 3.1)
    assert not any("transom_m" in w for w in b2["windows"] if w["unit_id"] == "u2")
    assert all("transom_m" in w for w in b2["windows"] if w["unit_id"] == "u1")
    b3 = M.building_from_composed(composed(), 2, 3.1, window_transom_m=1.0)
    assert all(w["transom_m"] == 1.0 for w in b3["windows"])
    b4 = M.building_from_composed(composed(), 2, 3.1, window_transom_m=0)
    assert not any("transom_m" in w for w in b4["windows"])


def test_ground_slab_is_offset_outline_with_mitred_corners():
    b = M.building_from_composed(composed(), 3, 3.1, main_entrance_m=[4, 0])
    g = [p for p in b["prisms"] if p["role"] == "slab" and p["storey"] == 0]
    up = [p for p in b["prisms"] if p["role"] == "slab" and p["storey"] == 1]
    assert len(g) == 1 and len(up) == 1
    gp, up_p = Polygon(g[0]["polygon"], g[0].get("holes") or []), Polygon(up[0]["polygon"], up[0].get("holes") or [])
    assert gp.area > up_p.area
    x0, y0, x1, y1 = Polygon(g[0]["polygon"]).bounds
    assert (x0, y0, x1, y1) == pytest.approx((-1.5, -1.5, 21.5, 11.5))
    # mitred, not rounded: full rectangle corner present
    assert any(abs(x + 1.5) < 1e-6 and abs(y + 1.5) < 1e-6 for x, y in g[0]["polygon"])
    assert Polygon(up[0]["polygon"]).bounds == pytest.approx((0, 0, 20, 10))
    roof = [p for p in b["prisms"] if p["role"] == "roof"][0]
    assert Polygon(roof["polygon"]).bounds == pytest.approx((0, 0, 20, 10))
    # core subtracted from the ground slab too
    assert gp.intersection(Polygon(assembler()["cores"][0]["service_polygon"])).area < 1e-6
    b2 = M.building_from_composed(composed(), 2, 3.1, ground_slab_offset_m=3.0)
    assert Polygon([p for p in b2["prisms"] if p["role"] == "slab" and p["storey"] == 0][0]["polygon"]).bounds == pytest.approx((-3, -3, 23, 13))
