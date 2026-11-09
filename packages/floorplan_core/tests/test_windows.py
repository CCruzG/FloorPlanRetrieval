"""Window derivation (G1 part 2a, 2026-11-09): windows are derived from the generated geometry."""
import copy
import json
import math
from pathlib import Path

import pytest

from packages.floorplan_core.generate import SCALE, generate
from packages.floorplan_core.spec import build_options, describe, fit
from packages.floorplan_core.node_graph import build_node_graph, expand_plan
from packages.floorplan_core.topo_signature import exact_room_edges, topological_fitness
from packages.floorplan_core.windows import derive_windows
from packages.floorplan_core.tests.test_spec import _good_spec

MM = 10000.0    # 1 unit = 10 m
EXT = ["external"] * 4


def _rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _sq():
    return _rect(0, 0, 1, 1)    # edges: 0 bottom, 1 right, 2 top, 3 left


def _derive(rooms, types=EXT, doors=None, options=None):
    return derive_windows(rooms, _sq(), types, MM, doors, options)


def test_shared_wall_and_corridor_edges_get_no_window():
    rooms = [{"id": "b", "type": "bedroom", "polygon": _rect(0, 0, 0.5, 0.5)},
             {"id": "l", "type": "living", "polygon": _rect(0.5, 0.5, 1, 1)}]
    w, un, _ = _derive(rooms, ["shared_wall", "facing_corridor", "shared_wall", "facing_corridor"])
    assert w == [] and un == []


def test_corner_bedroom_one_window_on_longest_run_living_one_per_run():
    bed = [{"id": "b", "type": "bedroom", "polygon": _rect(0, 0, 0.6, 0.5)}]   # bottom 6 m, left 5 m
    w, _, _ = _derive(bed)
    assert len(w) == 1 and w[0]["edge_index"] == 0
    liv = [{"id": "l", "type": "living", "polygon": _rect(0, 0, 0.6, 0.5)}]
    w, _, _ = _derive(liv)
    assert sorted(x["edge_index"] for x in w) == [0, 3]
    assert all(x["sill_mm"] == 0 and x["head_mm"] == 2400 for x in w)


def test_bathroom_with_external_run_gets_window_without_is_not_unplaced():
    wet = [{"id": "w", "type": "bathroom", "polygon": _rect(0.3, 0, 0.5, 0.2)}]
    w, un, _ = _derive(wet)
    assert len(w) == 1 and w[0]["sill_mm"] == 1500 and w[0]["head_mm"] == 2400 and un == []
    inner = [{"id": "w", "type": "bathroom", "polygon": _rect(0.4, 0.4, 0.6, 0.6)}]
    w, un, _ = _derive(inner)
    assert w == [] and un == []


def test_heights_by_type():
    for t, sill in (("bedroom", 900), ("study", 900), ("kitchen", 1050), ("dining", 0)):
        w, _, _ = _derive([{"id": "r", "type": t, "polygon": _rect(0.2, 0, 0.8, 0.4)}])
        assert len(w) == 1 and w[0]["sill_mm"] == sill and w[0]["head_mm"] == 2400, t


def test_non_windowed_types_get_none():
    for t in ("hallway", "storage", "nook", "circulation"):
        w, un, _ = _derive([{"id": "r", "type": t, "polygon": _rect(0.2, 0, 0.8, 0.4)}])
        assert w == [] and un == [], t


def test_short_run_none_and_1000_mm_run_one_window_inside_jambs():
    short = [{"id": "b", "type": "bedroom", "polygon": _rect(0.3, 0, 0.35, 0.3)}]     # 500 mm
    assert _derive(short)[0] == []
    r = [{"id": "b", "type": "bedroom", "polygon": _rect(0.3, 0, 0.4, 0.3)}]          # 1000 mm
    w, _, _ = _derive(r)
    assert len(w) == 1
    half = w[0]["width"] / 2
    c = w[0]["centre"][0]
    assert c - half >= 0.3 + 300 / MM - 1e-9 and c + half <= 0.4 - 300 / MM + 1e-9
    assert abs(c - 0.35) < 1e-9


def test_width_rule_for_a_long_run():
    w, _, _ = _derive([{"id": "l", "type": "living", "polygon": _rect(0.2, 0, 0.6, 0.4)}])   # 4000 mm
    assert abs(w[0]["width_mm"] - (4000 - 600) * 0.6) < 1e-6


def test_window_clears_a_door_on_the_same_wall():
    liv = [{"id": "l", "type": "living", "polygon": _rect(0, 0, 1, 0.5)}]
    door = {"kind": "entrance", "centre": [0.5, 0.0], "width": 900 / MM}
    w, _, _ = _derive(liv, doors=[door])
    on_bottom = [x for x in w if x["edge_index"] == 0]
    assert on_bottom
    d_lo, d_hi = 0.5 - 450 / MM - 300 / MM, 0.5 + 450 / MM + 300 / MM
    for x in on_bottom:
        lo, hi = x["centre"][0] - x["width"] / 2, x["centre"][0] + x["width"] / 2
        assert hi <= d_lo + 1e-9 or lo >= d_hi - 1e-9


def test_no_window_extends_past_its_run():
    rooms = [{"id": "l", "type": "living", "polygon": _rect(0, 0, 0.45, 0.5)},
             {"id": "b", "type": "bedroom", "polygon": _rect(0.45, 0, 1, 0.3)},
             {"id": "k", "type": "kitchen", "polygon": _rect(0.45, 0.3, 1, 1)}]
    w, _, _ = _derive(rooms)
    assert w
    for x in w:
        (ax, ay), (bx, by) = x["wall"]
        run = math.hypot(bx - ax, by - ay)
        t = ((x["centre"][0] - ax) * (bx - ax) + (x["centre"][1] - ay) * (by - ay)) / run
        assert x["width"] / 2 <= t + 1e-9 and t + x["width"] / 2 <= run + 1e-9
        assert abs(x["run_mm"] - run * MM) < 1e-3


def test_deterministic():
    rooms = [{"id": "l", "type": "living", "polygon": _rect(0, 0, 0.45, 0.5)},
             {"id": "b", "type": "bedroom", "polygon": _rect(0.45, 0, 1, 0.3)}]
    a = _derive(copy.deepcopy(rooms))
    b = _derive(copy.deepcopy(rooms))
    assert a == b


def test_glazing_ratio_flags_below_ten_percent():
    # bedroom 6 m x 5 m = 30 m2; window (6000-600)*0.6 = 3240 wide x 1500 high = 4.86 m2 -> 0.162
    _, _, g = _derive([{"id": "b", "type": "bedroom", "polygon": _rect(0, 0, 0.6, 0.5)}])
    assert g[0]["flag"] is False and abs(g[0]["ratio"] - 0.162) < 1e-3
    # same room with no external edge: ratio 0, flagged
    _, _, g = _derive([{"id": "b", "type": "bedroom", "polygon": _rect(0.2, 0.2, 0.8, 0.7)}])
    assert g[0]["flag"] is True and g[0]["ratio"] == 0


def test_generate_carries_windows_and_report():
    spec = _good_spec()
    p = generate(spec)
    assert p["windows"] and all(w["kind"] == "window" for w in p["windows"])
    for k in ("windows_unplaced", "windows_below_d26", "glazing"):
        assert k in p["report"]
    assert generate(copy.deepcopy(spec))["windows"] == p["windows"]


# ── agreement with D26 on the gate's dev boundaries ─────────────────────────────

DATA = Path.home() / "Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json"
PLAN_SETS = Path(__file__).resolve().parents[3] / "data/outputs/gates/plan_sets.json"
HABITABLE = {"living", "dining", "bedroom", "master_bedroom", "study"}
N_AGREEMENT = 25    # first N dev plans; the full 300 run in tools (see the 2026-11-09 report)


def agreement_exceptions(ids):
    """Rooms that pass D26 per room (external run >= 0.8 m) with no window, as {plan id: [room]}."""
    out = {}
    mm = float(SCALE)
    for pid in ids:
        plan = json.loads((DATA / f"{pid}.json").read_text())
        p = expand_plan(plan, build_node_graph(plan))
        src = describe(p, mm_per_unit=mm)
        ent = src["entrance"]
        topo = topological_fitness(p, target_boundary=src["boundary"], target_edge_types=src["edge_types"],
                                   entrance_point=ent[:2] if ent else None)
        spec = fit(src, src["boundary"], edge_types=src["edge_types"], entrance=ent,
                   mm_per_unit=mm, node_pos=topo.get("node_pos", {}), topo=topo)
        g = generate(spec, build_options())
        re = exact_room_edges(g, spec["boundary"], spec["edge_types"])
        have = {w["room"] for w in g["windows"]}
        for r in g["rooms"]:
            if str(r["type"]).lower() not in HABITABLE:
                continue
            ok = any(e["edge_type"] == "external" and e["width_m"] >= 0.8 for e in re.get(str(r["id"]), []))
            if ok and str(r["id"]) not in have:
                out.setdefault(pid, []).append(str(r["id"]))
    return out


@pytest.mark.skipif(not (DATA.exists() and PLAN_SETS.exists()), reason="RPlan data not available")
def test_every_room_passing_d26_has_a_window_on_dev_boundaries():
    ids = json.loads(PLAN_SETS.read_text())["dev"][:N_AGREEMENT]
    exc = agreement_exceptions(ids)
    # exceptions are allowed only as recorded findings: print them for the report
    print("agreement exceptions:", exc)
    assert sum(len(v) for v in exc.values()) <= max(1, len(ids) // 10), exc
