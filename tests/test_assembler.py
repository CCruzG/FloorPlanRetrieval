"""Plate assembler tests (ARCHITECTURE.md, "Plate assembler" and "Second pass"). Library data is a temp copy."""
import json
import math
import os
import shutil
import sys
from pathlib import Path

import pytest
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from packages.floor_plate_design import library as lib  # noqa: E402
from packages.floor_plate_design.assembler import PARAMS, TYPES, assemble_plate, verify_result  # noqa: E402

MIXES = {"1100": {"1B1b": 1, "2B1b": 1}, "1010": {"1B1b": 1, "2B2b": 1}, "1101": {"1B1b": 1, "2B1b": 1, "3B2b": 1}}
RECT36 = [[0, 0], [36, 0], [36, 14], [0, 14]]


@pytest.fixture(scope="module")
def plates(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lib") / "library.json"
    shutil.copy(REPO / "data" / "floor_plates" / "library.json", tmp)
    old = os.environ.get(lib.ENV_VAR)
    os.environ[lib.ENV_VAR] = str(tmp)
    try:
        yield lib.list_plates()["plates"]
    finally:
        if old is None:
            os.environ.pop(lib.ENV_VAR)
        else:
            os.environ[lib.ENV_VAR] = old


@pytest.fixture(scope="module")
def results(plates):
    """assemble every library plate with every mix once (the plate's own street edges and north)."""
    out = {}
    for p in plates:
        for m in MIXES:
            out[(p["id"], m)] = assemble_plate(p["boundary_m"], MIXES[m], p["street_edges"], p["north_deg"])
    return out


def outline_edges(boundary):
    n = len(boundary)
    return [(boundary[i], boundary[(i + 1) % n]) for i in range(n)]


def facade_bearing(boundary, i, north_deg):
    """Compass bearing (clockwise from north) of the outward normal of edge i. Independent of the assembler."""
    pts = [tuple(p) for p in boundary]
    area2 = sum(pts[k][0] * pts[(k + 1) % len(pts)][1] - pts[(k + 1) % len(pts)][0] * pts[k][1] for k in range(len(pts)))
    a, b = pts[i], pts[(i + 1) % len(pts)]
    dx, dy = b[0] - a[0], b[1] - a[1]
    ln = math.hypot(dx, dy)
    s = 1 if area2 > 0 else -1
    nx, ny = dy / ln * s, -dx / ln * s
    th = math.radians(north_deg)
    north, east = (math.sin(th), -math.cos(th)), (math.cos(th), math.sin(th))
    return math.degrees(math.atan2(nx * east[0] + ny * east[1], nx * north[0] + ny * north[1])) % 360


def core_facades(res, north_deg):
    """(edge index, bearing, contact length) for every outline edge the core block touches."""
    B = res["boundary"]
    core = Polygon(res["cores"][0]["polygon"])
    out = []
    for i, (a, b) in enumerate(outline_edges(B)):
        L = LineString([a, b]).intersection(core.boundary).length
        if L > 1e-6:
            out.append((i, facade_bearing(B, i, north_deg), L))
    return out


def sun_side(bearing):
    d = abs(((bearing - PARAMS["sun_bearing_deg"] + 180) % 360) - 180)
    return d < PARAMS["sun_half_angle_deg"] - 1e-9


def door_commons(res):
    return unary_union([Polygon(c["lobby_polygon"]) for c in res["cores"]] + [Polygon(c["polygon"]) for c in res["corridors"]])


def check_geometry(res):
    """Independent of the assembler's own verify: hard rules and typed edges on the output."""
    B = Polygon(res["boundary"])
    cores = [Polygon(c["polygon"]) for c in res["cores"]]
    services = [Polygon(c["service_polygon"]) for c in res["cores"]]
    lobbies = [Polygon(c["lobby_polygon"]) for c in res["cores"]]
    halls = [Polygon(c["polygon"]) for c in res["corridors"]]
    commons = door_commons(res)
    units = [Polygon(u["polygon"]) for u in res["units"]]
    resid = [Polygon(r) for r in res["residual"]]
    assert len(cores) == 1
    assert abs(services[0].area + lobbies[0].area - cores[0].area) < 1e-6  # block = two parts
    assert services[0].intersection(lobbies[0]).area < 1e-9
    for g in cores + halls + units + resid:
        assert B.buffer(1e-6).contains(g)
    parts = units + cores + halls + resid
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            assert parts[i].intersection(parts[j]).area < 1e-6
    assert abs(sum(g.area for g in parts) - B.area) < 1e-3  # nothing lost, nothing double counted
    for u, g in zip(res["units"], units):
        spec = PARAMS["types"][u["type"]]
        assert spec["min_area"] - 1e-6 <= g.area <= PARAMS["max_area_factor"] * spec["target_area"] + 1e-6
        assert abs(g.area - u["area"]) < 1e-2
        assert g.intersection(commons).length >= PARAMS["door_contact"] - 1e-6  # door off lobby or hall
        assert g.intersection(B.boundary).length >= PARAMS["facade_min"] - 1e-6
        assert u["depth"] >= PARAMS["min_depth"] - 1e-6
        pts = u["polygon"]
        assert len(u["typed_edges"]) == len(pts)
        for te in u["typed_edges"]:
            a, b = pts[te["edge_index"]], pts[(te["edge_index"] + 1) % len(pts)]
            seg = LineString([a, b])
            on_out = seg.intersection(B.boundary).length
            on_com = seg.intersection(commons.boundary).length
            if te["type"] == "external":
                assert on_out > seg.length - 1e-6
            elif te["type"] == "facing_corridor":
                assert on_com > seg.length - 1e-6 and on_out < 1e-6
            else:
                assert te["type"] == "shared_wall" and on_out < 1e-6 and on_com < 1e-6
        e = Point(u["entrance_point"])
        assert g.boundary.distance(e) < 1e-6 and commons.distance(e) < 1e-6  # on lobby or hall, never service
        assert services[0].distance(e) > 1e-6 or lobbies[0].distance(e) < 1e-6 or any(h.distance(e) < 1e-6 for h in halls)
        ring = list(g.simplify(0).exterior.coords)[:-1]
        assert len(ring) in (4, 6)  # rectangle or one-step L
        assert u["shape"] == ("L" if len(ring) == 6 else "rect")
        if len(ring) == 4:
            assert abs(g.area - Polygon(g.envelope).area) < 1e-6
    # every commons piece serves a door
    for h in halls:
        assert any(g.intersection(h).length >= PARAMS["door_contact"] - 1e-6 for g in units)
    assert any(g.intersection(lobbies[0]).length >= PARAMS["door_contact"] - 1e-6 for g in units)
    # types follow size
    order = sorted(res["units"], key=lambda u: u["area"])
    idx = [TYPES.index(u["type"]) for u in order]
    assert idx == sorted(idx)


def test_library_plates_hard_rules(plates, results):
    assert len(plates) == 3
    for (pid, m), res in results.items():
        assert res["status"] == "ok", (pid, m, res["reason"], res["rejects"])
        assert sum(verify_result(res).values()) == 0
        check_geometry(res)
        mt = res["metrics"]
        for k in ("commons_pct", "max_walk_m", "residual_m2", "unit_areas_m2", "candidates_tried",
                  "candidates_survived", "requested_mix", "achieved_counts", "mix_error_per_type",
                  "core_facades", "l_units"):
            assert k in mt
        assert mt["units"] == len(res["units"]) == len(mt["unit_areas_m2"])
        assert sum(mt["achieved_counts"].values()) == mt["units"]
        assert mt["candidates_survived"] > 0 and mt["candidates_tried"] >= mt["candidates_survived"]
        assert set(res["rejects"]) >= {"unit_no_door", "unit_no_facade", "unit_below_min_area", "unit_not_rect_or_L"}


def test_rectangle_36x14():
    for m in MIXES:
        res = assemble_plate(RECT36, MIXES[m], [], 0)
        assert res["status"] == "ok", res["reason"]
        assert sum(verify_result(res).values()) == 0
        check_geometry(res)
        assert res["typology"][0]["access"] == "double_loaded"


def test_core_avoids_sun_side_facade_when_an_alternative_exists(plates, results):
    # 36 x 14 m, north up: the core must not touch the top (sun-side) facade; it can touch the others.
    for m in MIXES:
        res = assemble_plate(RECT36, MIXES[m], [], 0)
        facs = core_facades(res, 0)
        assert not any(sun_side(b) for _, b, _ in facs), facs
        assert res["metrics"]["core_sun_frontage_m"] == 0
    # library plates: the assembler's own report and an independent check agree
    for p in plates:
        for m in MIXES:
            res = results[(p["id"], m)]
            facs = core_facades(res, p["north_deg"])
            sun_len = sum(L for _, b, L in facs if sun_side(b))
            assert abs(sun_len - res["metrics"]["core_sun_frontage_m"]) < 1e-2
            if p["id"] != "test-plate-002":  # its only host is a 7.5 m strip: a full-depth core touches both long facades
                assert sun_len == 0, (p["id"], m, facs)


def test_plate_001_north_minus_35_puts_no_core_on_the_top_facade(plates, results):
    p = next(q for q in plates if q["id"] == "test-plate-001")
    assert p["north_deg"] == -35
    ys = [pt[1] for pt in p["boundary_m"]]
    top = min(ys)
    for m in MIXES:
        res = results[(p["id"], m)]
        core = Polygon(res["cores"][0]["polygon"])
        assert core.bounds[1] > top + 1e-6, (m, core.bounds, top)


def test_turning_north_by_180_moves_the_core():
    a = assemble_plate(RECT36, MIXES["1100"], [], 0)
    b = assemble_plate(RECT36, MIXES["1100"], [], 180)
    ya = Polygon(a["cores"][0]["polygon"]).centroid.y
    yb = Polygon(b["cores"][0]["polygon"]).centroid.y
    assert ya > 7 > yb, (ya, yb)  # north up: core in the lower half; north down: upper half
    assert not any(sun_side(bg) for _, bg, _ in core_facades(b, 180))


def test_no_hall_reaches_a_facade_unless_a_door_needs_it(results):
    eps = 1e-6
    for res in list(results.values()):
        B = Polygon(res["boundary"])
        units = [Polygon(u["polygon"]) for u in res["units"]]
        for hc in res["corridors"]:
            h = Polygon(hc["polygon"])
            x0, y0, x1, y1 = h.bounds
            horiz = (x1 - x0) >= (y1 - y0)
            ends = [LineString([(x0, y0), (x0, y1)]), LineString([(x1, y0), (x1, y1)])] if horiz else \
                   [LineString([(x0, y0), (x1, y0)]), LineString([(x0, y1), (x1, y1)])]
            for end in ends:
                if end.intersection(B.boundary).length < eps:
                    continue
                # the hall end is on a facade: a unit must have no door except through that end stretch
                need = False
                for g in units:
                    shared = g.intersection(h)
                    if shared.length >= 0.9 - eps and shared.distance(end) < eps:
                        others = unary_union([Polygon(c["lobby_polygon"]) for c in res["cores"]]
                                             + [Polygon(c["polygon"]) for c in res["corridors"] if c is not hc])
                        # cutting 1.5 m of hall must strand this unit (door contact below 0.9 m)
                        if g.intersection(others).length + shared.length - PARAMS["hall_w"] < 0.9 - eps:
                            need = True
                assert need, hc["id"]


def test_types_follow_size(results):
    for res in results.values():
        order = sorted(res["units"], key=lambda u: u["area"])
        idx = [TYPES.index(u["type"]) for u in order]
        assert idx == sorted(idx)


def test_every_commons_piece_serves_a_door(results):
    for res in results.values():
        units = [Polygon(u["polygon"]) for u in res["units"]]
        for hc in res["corridors"]:
            h = Polygon(hc["polygon"])
            assert any(g.intersection(h).length >= 0.9 - 1e-6 for g in units), hc["id"]
        lob = Polygon(res["cores"][0]["lobby_polygon"])
        assert any(g.intersection(lob).length >= 0.9 - 1e-6 for g in units)
        assert [c["kind"] for c in res["corridors"]] == ["hall"] * len(res["corridors"])  # no dead commons strips


def test_typed_edges_correct_on_an_l(results):
    ls = [(k, u) for k, res in results.items() for u in res["units"] if u["shape"] == "L"]
    assert ls, "no L unit in any library result"
    for k, u in ls:
        res = results[k]
        B = Polygon(res["boundary"])
        commons = door_commons(res)
        pts = u["polygon"]
        g = Polygon(pts)
        assert len(list(g.simplify(0).exterior.coords)[:-1]) == 6
        assert abs(g.area - u["area"]) < 1e-2
        kinds = set()
        for te in u["typed_edges"]:
            a, b = pts[te["edge_index"]], pts[(te["edge_index"] + 1) % len(pts)]
            seg = LineString([a, b])
            kinds.add(te["type"])
            if te["type"] == "external":
                assert seg.intersection(B.boundary).length > seg.length - 1e-6
            elif te["type"] == "facing_corridor":
                assert seg.intersection(commons.boundary).length > seg.length - 1e-6
            else:
                assert seg.intersection(B.boundary).length < 1e-6 and seg.intersection(commons.boundary).length < 1e-6
        assert {"external", "facing_corridor"} <= kinds
        # every part of the outline and of the commons boundary that the unit touches is typed as such
        assert abs(sum(LineString([pts[t["edge_index"]], pts[(t["edge_index"] + 1) % len(pts)]]).length
                       for t in u["typed_edges"] if t["type"] == "external")
                   - g.boundary.intersection(B.boundary).length) < 1e-3
        assert abs(sum(LineString([pts[t["edge_index"]], pts[(t["edge_index"] + 1) % len(pts)]]).length
                       for t in u["typed_edges"] if t["type"] == "facing_corridor")
                   - g.boundary.intersection(commons.boundary).length) < 1e-3
        e = Point(u["entrance_point"])
        assert commons.distance(e) < 1e-6 and g.boundary.distance(e) < 1e-6


def test_determinism(plates):
    for p in plates:
        a = assemble_plate(p["boundary_m"], MIXES["1100"], p["street_edges"], p["north_deg"])
        b = assemble_plate(p["boundary_m"], MIXES["1100"], p["street_edges"], p["north_deg"])
        assert json.dumps(a) == json.dumps(b)
    assert json.dumps(assemble_plate(RECT36, MIXES["1101"], [], 0)) == json.dumps(assemble_plate(RECT36, MIXES["1101"], [], 0))


def test_non_orthogonal_is_unsupported():
    res = assemble_plate([[0, 0], [30, 0], [30, 12], [10, 15], [0, 12]], MIXES["1100"])
    assert res["status"] == "unsupported"
    assert res["units"] == [] and "axis-aligned" in res["reason"]


def test_impossible_input_is_infeasible_with_rejects():
    res = assemble_plate([[0, 0], [12, 0], [12, 4], [0, 4]], MIXES["1100"])
    assert res["status"] == "infeasible"
    assert res["units"] == [] and res["metrics"]["candidates_survived"] == 0
    assert set(res["rejects"]) >= {"unit_no_door", "unit_no_facade", "unit_below_min_area", "unit_below_min_depth"}


def test_infeasible_with_candidates_reports_reject_counts():
    res = assemble_plate([[0, 0], [7, 0], [7, 7], [0, 7]], MIXES["1100"], [], 0)
    assert res["status"] == "infeasible"
    assert res["metrics"]["candidates_tried"] > 0 and res["metrics"]["candidates_survived"] == 0
    assert res["rejects"]["unit_below_min_area"] == res["metrics"]["candidates_tried"]
    assert set(res["rejects"]) >= {"unit_no_door", "unit_no_facade", "unit_below_min_area", "unit_below_min_depth"}


def test_mix_is_never_changed_silently(plates):
    p = plates[0]
    res = assemble_plate(p["boundary_m"], {"1B1b": 1, "3B2b": 3}, p["street_edges"], p["north_deg"])
    mt = res["metrics"]
    assert abs(sum(mt["requested_mix"].values()) - 1) < 1e-3
    assert mt["mix_error_per_type"] == {t: mt["achieved_counts"][t] - mt["requested_counts"][t] for t in TYPES}


def test_bad_mix():
    with pytest.raises(ValueError):
        assemble_plate(RECT36, {"1B1b": 0})
    with pytest.raises(ValueError):
        assemble_plate(RECT36, {"4B": 1})
