"""Plate composition record: requirements, slot status transitions, export round trip."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.floor_plate_design import compose as C  # noqa: E402

SQ = [[0, 0], [9, 0], [9, 9], [0, 9]]  # plate metres


def unit(uid="u1", typ="2B1b"):
    return {"id": uid, "type": typ, "polygon": SQ, "entrance_point": [4.5, 0.0],
            "typed_edges": [{"edge_index": 0, "type": "facing_corridor"}], "area": 81.0}


def server_plan(pid=7):
    return {"id": pid, "retrieval": {"dist": 0.25, "mirror": False},
            "rooms": [{"id": "r_1", "type": 3, "source_type": 3, "label": "living", "polygon": SQ}]}


def gen_result(ok=True):
    return {
        "rooms": [{"id": "r_1/primary", "type": "living", "polygon": [[0, 0], [0.5, 0], [0.5, 0.5]]}],
        "edges": [{"u": "r_1/primary", "v": "r_2", "connection": 1}],
        "node_pos": {"r_1/primary": [0.25, 0.1]}, "room_edges": {}, "feasible": True, "report": {"coverage": 1.0},
        "sensors": {"summary": {"gates_pass": ok, "passed": 5 if ok else 3, "total": 7, "soft_score": 0.0},
                    "checks": [{"sensor": "reachability", "kind": "hard", "ok": ok, "detail": "x"},
                               {"sensor": "node_in_room", "kind": "soft", "ok": True, "detail": "y"}]},
    }


@pytest.mark.parametrize("typ,beds", [("1B1b", 1), ("2B1b", 2), ("2B2b", 2), ("3B2b", 3)])
def test_requirements_per_type(typ, beds):
    assert C.requirements_for({"type": typ}) == {"bedrooms": beds, "bathrooms": 1, "balcony": False}


def test_requirements_unknown_type_defaults_to_one_bedroom():
    assert C.requirements_for({"type": "studio"})["bedrooms"] == 1
    assert C.requirements_for({})["balcony"] is False


def test_status_transitions():
    s = C.new_slot(unit())
    assert s.status == C.PENDING
    with pytest.raises(ValueError):
        C.mark_edited(s, {"rank": 2, "mirror": False}, {}, {}, None)  # pending -> edited not allowed
    C.mark_generated(s, [C.candidate_record(server_plan(), 1)], {"rank": 1, "mirror": False},
                     C.plan_record(gen_result(), server_plan()), C.DEFAULT_OPTIONS, gen_result()["sensors"])
    assert s.status == C.GENERATED and s.error is None
    with pytest.raises(ValueError):
        C.mark_failed(s, "late")  # generated -> failed not allowed
    C.mark_edited(s, {"rank": 2, "mirror": False}, s.plan, C.DEFAULT_OPTIONS, gen_result(False)["sensors"])
    assert s.status == C.EDITED and s.chosen["rank"] == 2
    C.mark_edited(s, {"rank": 3, "mirror": False}, s.plan, C.DEFAULT_OPTIONS, None)
    assert s.status == C.EDITED
    f = C.new_slot(unit("u2"))
    C.mark_failed(f, "boom")
    assert f.status == C.FAILED and f.error == "boom"
    C.mark_edited(f, {"rank": 1, "mirror": False}, {}, {}, None)  # failed -> edited allowed
    assert f.status == C.EDITED and f.error is None


def test_sensors_record_keeps_failing_only():
    rec = C.sensors_record(gen_result(False)["sensors"])
    assert [c["sensor"] for c in rec["failing"]] == ["reachability"]
    assert C.hard_fail(gen_result(False)["sensors"]) and not C.hard_fail(gen_result(True)["sensors"])


def test_source_type_tagging_follows_node_ids():
    rooms = C.tag_source_types([{"id": "r_1/primary", "type": "living"}, {"id": "zz", "type": "bath"}],
                               server_plan())
    assert rooms[0]["source_type"] == 3 and rooms[1]["source_type"] == "bath"


def test_compose_plate_continues_after_a_failure():
    asm = {"units": [unit("u1"), unit("u2", "1B1b")], "boundary": SQ}
    seen = []

    def retrieve(b, e, t, req, topn):
        seen.append((b, e, req, topn))
        if len(seen) == 2:
            raise RuntimeError("no index")
        return [server_plan(7), server_plan(8)]

    def generate(fp, b, e, t, req, opts):
        assert opts["source_grid"] and opts["expand_nodes"] and opts["core_placement"] is False
        return gen_result()

    comp = C.compose_plate(asm, retrieve, generate)
    assert comp.slots["u1"].status == C.GENERATED and comp.slots["u2"].status == C.FAILED
    assert comp.slots["u2"].error == "no index"
    assert seen[0][0][1] == [0.5, 0.0] and seen[0][1] == [0.25, 0.0] and seen[0][3] == 5
    assert seen[0][2] == {"bedrooms": 2, "bathrooms": 1, "balcony": False}
    assert [c["rank"] for c in comp.slots["u1"].candidates] == [1, 2]
    assert comp.slots["u1"].chosen == {"rank": 1, "mirror": False}


def test_export_round_trip_and_metres():
    asm = {"units": [unit("u1")], "boundary": SQ}
    comp = C.compose_plate(asm, lambda *a: [server_plan()], lambda *a: gen_result())
    out = C.to_export(comp)
    back = json.loads(json.dumps(out))
    assert back == out
    room = back["slots"]["u1"]["plan"]["rooms"][0]
    assert room["polygon_m"][1] == [9.0, 0.0]  # 0.5 x 18
    assert back["slots"]["u1"]["plan"]["node_pos_m"]["r_1/primary"] == [4.5, 1.8]
    assert back["slots"]["u1"]["candidates"] == [{"plan_id": 7, "rank": 1, "dist": 0.25}]
    assert back["plate"]["units"][0]["id"] == "u1" and back["slots"]["u1"]["status"] == "generated"


def test_record_round_trip_keeps_state():
    asm = {"units": [unit("u1")], "boundary": SQ}
    comp = C.compose_plate(asm, lambda *a: [server_plan()], lambda *a: gen_result())
    rec = json.loads(json.dumps(C.to_record(comp)))
    again = C.from_record(rec)
    assert again.slots["u1"].status == C.GENERATED
    assert C.to_export(again) == C.to_export(comp)
