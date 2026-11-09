"""Reachability sensor: unreachable rooms, and non-private rooms reachable only
through a private room. Plans go through run_sensors (real ctx builder)."""
from packages.floorplan_core.topo_sensors import run_sensors


def _sq(x, y, w=3, h=3):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _plan(rooms, edges):
    # rooms: {id: (label, x)}; entrance sits on r_0 (x=0).
    return {
        "rooms": [{"id": i, "label": lab, "polygon": _sq(x, 0)} for i, (lab, x) in rooms.items()],
        "edges": [{"u": u, "v": v, "connection": c} for u, v, c in edges],
        "entrance": [0, 1, 0, 2],
    }


def _reach(plan):
    res = run_sensors(plan, {}, layer="topology")
    return next(c for c in res["checks"] if c["sensor"] == "reachability")


def test_kitchen_without_door_is_unreachable():
    p = _plan({"r_0": ("living_room", 0), "r_1": ("kitchen", 3)}, [("r_0", "r_1", 0)])
    c = _reach(p)
    assert not c["ok"] and c["failing_rooms"] == ["r_1"]
    assert "unreachable from entry r_0" in c["detail"]
    assert "private room" not in c["detail"]


def test_kitchen_only_through_bedroom_fails():
    p = _plan({"r_0": ("living_room", 0), "r_1": ("bedroom", 3), "r_2": ("kitchen", 6)},
              [("r_0", "r_1", 1), ("r_1", "r_2", 1)])
    c = _reach(p)
    assert not c["ok"] and c["failing_rooms"] == ["r_2"]
    assert "reachable only through a private room: r_2 (kitchen) via r_1 (bedroom)" in c["detail"]
    assert "unreachable" not in c["detail"]


def test_ensuite_through_master_bedroom_passes():
    p = _plan({"r_0": ("living_room", 0), "r_1": ("master_bedroom", 3), "r_2": ("bathroom", 6)},
              [("r_0", "r_1", 1), ("r_1", "r_2", 1)])
    c = _reach(p)
    assert c["ok"] and c["failing_rooms"] == [] and c["margin"] == 0


def test_balcony_through_bedroom_passes():
    # 2026-11-09: only shared rooms (TRANSIT) are held to the rule; a balcony off a
    # bedroom is normal (626 of 2000 RPlan source plans).
    p = _plan({"r_0": ("living_room", 0), "r_1": ("master_bedroom", 3), "r_2": ("balcony", 6)},
              [("r_0", "r_1", 1), ("r_1", "r_2", 1)])
    assert _reach(p)["ok"]


def test_normal_plan_passes():
    p = _plan({"r_0": ("living_room", 0), "r_1": ("kitchen", 3), "r_2": ("dining", 6),
               "r_3": ("bedroom", 9)},
              [("r_0", "r_1", 1), ("r_0", "r_2", 2), ("r_0", "r_3", 1)])
    assert _reach(p)["ok"]
