"""
gate_set_connection.py — Stage-2 gate: does SetConnection's predict_effect match
reality?

Reuses tools/diag_min_size.py's in-process app setup (load the index into
server.INDEX_STORE, test_client, the two fixed test boundaries, top 3 retrieved
plans each) and reproduces server.py's /generate route body (spec_from_plan +
boundary_demands + re-derived room_edges) so this tool can run sensors, propose
SetConnection candidates, and apply one without a second HTTP round trip.

For each candidate edit the gate checks TWO things predict_effect claims are
exact (moves-catalogue.md: "exact — geometry is unchanged"):
  1. predicted sensors (SetConnectionMove.predict_effect on the generated plan)
     == actual sensors (apply the edit to the Spec, generate() again, re-derive
     room_edges, run_sensors) — per sensor: same ok, same failing_rooms, margins
     within 1e-6.
  2. after apply + generate, the edited edge is present in the regenerated
     plan['edges'] carrying the edited connection.

Also reports, for the same six plans, whether any sensor's output differs before
vs after the topo_sensors.py edge-semantics fix (part A) — run once with the
edited module (this process), once with a frozen pre-fix snapshot of the module
(saved before part A's edits, since the working tree already carried uncommitted
Stage-1 changes with no commit boundary to diff against) — so a silent behaviour
change on UNEDITED plans does not go unreported.

Read-only / additive: writes data/outputs/diagnostics/gate_set_connection_<suffix>.json.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]          # 02_Retrieval_v02/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── fixed inputs — same convention as diag_min_size.py ───────────────────────
BOUNDARY_LIB_PATH = REPO_ROOT.parent / "test-boundaries" / "boundary_library.json"
TARGET_NAMES = ["test_boundary_001b", "test_boundary_005b"]
TOPN = 3
MM_PER_UNIT = 18000.0   # schema.SCALE; see diag_min_size.py's note on the gap
                        # between this and what the live frontend actually sends

OUT_DIR = REPO_ROOT / "data" / "outputs" / "diagnostics"
OUT_PATH = OUT_DIR / ("gate_set_connection_2026-10-30"
                      + (f"_{sys.argv[1]}" if len(sys.argv) > 1 else "") + ".json")

MARGIN_TOL = 1e-6

# Frozen copy of topo_sensors.py as it stood before part A's edge-semantics fix
# (saved by hand from the pre-edit Read — the working tree had no commit boundary
# to diff against). Throwaway comparison fixture, not imported by anything else.
PRE_FIX_SNAPSHOT = Path("/Users/camilo/.claude/jobs/bcc8ab15/tmp/topo_sensors_pre_A.py")


def load_targets():
    lib = json.loads(BOUNDARY_LIB_PATH.read_text())
    by_name = {e["name"]: e for e in lib}
    missing = [n for n in TARGET_NAMES if n not in by_name]
    if missing:
        raise SystemExit(f"boundary(ies) not found in {BOUNDARY_LIB_PATH}: {missing}")
    return [by_name[n] for n in TARGET_NAMES]


def entry_to_request_parts(entry):
    boundary = [[v["x"], v["y"]] for v in entry["vertices"]]
    entrance_point = [entry["entrance"]["x"], entry["entrance"]["y"]]
    typed_edges = [{"edge_index": i, "type": t}
                   for i, t in enumerate(entry.get("edgeTypes") or []) if t]
    requirements = entry.get("requirements") or {}
    return boundary, entrance_point, typed_edges, requirements


def build_generation(server_mod, fp, boundary, entrance_point, typed_edges, requirements,
                      mm_per_unit=MM_PER_UNIT):
    """Reproduce apps/api/server.py's /generate route body in-process: resolve
    edge types, compute the topology layer, build the Spec, thread
    boundary_demands, generate(), re-derive room_edges on the generated geometry,
    run sensors. Returns (spec, plan, gen_room_edges, sensors) — everything a
    candidate's apply()/predict_effect() needs."""
    from packages.floorplan_core.generate import spec_from_plan, generate
    from packages.floorplan_core.topo_signature import (
        default_edge_types, topological_fitness, exact_room_edges,
    )
    from packages.floorplan_core.topo_sensors import run_sensors

    resolved_types = default_edge_types(boundary, entrance_point)
    for e in typed_edges or []:
        i = e.get("edge_index")
        if i is not None and 0 <= i < len(boundary) and e.get("type"):
            resolved_types[i] = e["type"]
    topo = topological_fitness(
        fp, target_boundary=boundary, target_edge_types=resolved_types,
        entrance_point=entrance_point, requirements=requirements,
    )
    fp_with_nodes = dict(fp)
    fp_with_nodes["node_pos"] = topo.get("node_pos", {})

    drawn_entrance = [entrance_point[0], entrance_point[1],
                      entrance_point[0], entrance_point[1]]
    spec = spec_from_plan(
        fp_with_nodes, boundary, edge_types=resolved_types,
        entrance=drawn_entrance, mm_per_unit=mm_per_unit,
    )
    boundary_demands = {}
    for rid, contacts in (topo.get("room_edges") or {}).items():
        idxs = sorted({c.get("target_edge_index") for c in contacts
                       if c.get("target_edge_index") is not None})
        if idxs:
            boundary_demands[str(rid)] = idxs
    spec["boundary_demands"] = boundary_demands
    spec["area_lift"] = True

    plan = generate(spec)
    gen_room_edges = exact_room_edges(plan, boundary, resolved_types)
    sensors = run_sensors(plan, gen_room_edges, requirements=requirements,
                          target_boundary=boundary, spec=spec, layer="geometry")
    return spec, plan, gen_room_edges, resolved_types, sensors


def _checks_by_name(sensors):
    return {c["sensor"]: c for c in sensors.get("checks", [])}


def compare_sensors(predicted, actual):
    """Per-sensor: ok and failing_rooms must match; margins within MARGIN_TOL.
    Returns (match: bool, diffs: [str])."""
    pc, ac = _checks_by_name(predicted), _checks_by_name(actual)
    diffs = []
    for name in sorted(set(pc) | set(ac)):
        p, a = pc.get(name), ac.get(name)
        if p is None or a is None:
            diffs.append(f"{name}: present in only one of predicted/actual")
            continue
        if bool(p["ok"]) != bool(a["ok"]):
            diffs.append(f"{name}: ok predicted={p['ok']} actual={a['ok']}")
        if sorted(map(str, p["failing_rooms"])) != sorted(map(str, a["failing_rooms"])):
            diffs.append(f"{name}: failing_rooms predicted={p['failing_rooms']} "
                         f"actual={a['failing_rooms']}")
        if abs(float(p["margin"]) - float(a["margin"])) > MARGIN_TOL:
            diffs.append(f"{name}: margin predicted={p['margin']} actual={a['margin']} "
                         f"(Δ={abs(p['margin'] - a['margin']):.6g})")
    return (not diffs), diffs


def edge_after_apply(plan, u, v, expected_connection):
    for e in plan.get("edges") or []:
        if {e.get("u"), e.get("v")} == {u, v}:
            return e.get("connection") == expected_connection, e
    return False, None


def main():
    from apps.api import server as server_mod
    from packages.floorplan_core.index_store import build_index_store_from_npz
    from packages.floorplan_core.moves import MOVES, _TYPE_TO_CONNECTION

    npz_path = REPO_ROOT / "data" / "index" / "rplan_index_v5.npz"
    print(f"loading index: {npz_path} ...", file=sys.stderr)
    server_mod.INDEX_STORE = build_index_store_from_npz(
        npz_path=str(npz_path),
        data_dir=str(server_mod.RPLAN_DATASET_DIR),
        logger=server_mod.logger,
    )
    client = server_mod.app.test_client()
    set_connection = next(m for m in MOVES if m.name == "SetConnection")

    targets = load_targets()
    out = {"generated_at": "2026-10-30", "mm_per_unit_used": MM_PER_UNIT,
          "boundaries": []}
    totals = {"plans": 0, "candidates_proposed": 0, "predicted_eq_actual": 0,
              "mismatches": []}
    cleared = {}   # plan_key -> {sensor: [reasons cleared]} (predicted)
    before_after_a = []   # unedited-plan sensor diffs, pre- vs post- part-A fix

    for entry in targets:
        name = entry["name"]
        boundary, entrance_point, typed_edges, requirements = entry_to_request_parts(entry)
        retrieve_body = {"boundary": boundary, "entrance_point": entrance_point,
                         "requirements": requirements, "topn": TOPN,
                         "typed_edges": typed_edges}
        rv = client.post("/retrieve", json=retrieve_body)
        rdata = rv.get_json()
        if rv.status_code != 200 or rdata.get("error"):
            out["boundaries"].append({"name": name, "error": f"/retrieve failed: {rdata}"})
            continue
        plans = (rdata.get("plans") or [])[:TOPN]

        b_record = {"name": name, "results": []}
        for rank, fp in enumerate(plans, start=1):
            plan_key = f"{name}#{rank}"
            spec, plan, room_edges, resolved_types, sensors = build_generation(
                server_mod, fp, boundary, entrance_point, typed_edges, requirements)
            totals["plans"] += 1
            ctx_inputs = {"room_edges": room_edges, "requirements": requirements,
                          "target_boundary": boundary, "spec": spec}

            candidates = set_connection.propose(plan, sensors, ctx_inputs) \
                if set_connection.applicable(plan, sensors) else []
            totals["candidates_proposed"] += len(candidates)

            result = {"rank": rank, "source_id": fp.get("retrieval", {}).get("source_id"),
                      "applicable": set_connection.applicable(plan, sensors),
                      "n_candidates": len(candidates), "candidates": []}

            cleared_here = []
            for edit in candidates:
                predicted = set_connection.predict_effect(plan, edit, ctx_inputs)

                new_spec = set_connection.apply(spec, edit)
                actual_plan = None
                try:
                    from packages.floorplan_core.generate import generate
                    from packages.floorplan_core.topo_signature import exact_room_edges
                    actual_plan = generate(new_spec)
                    actual_room_edges = exact_room_edges(actual_plan, boundary, resolved_types)
                    from packages.floorplan_core.topo_sensors import run_sensors
                    actual = run_sensors(actual_plan, actual_room_edges,
                                         requirements=requirements,
                                         target_boundary=boundary, spec=new_spec, layer="geometry")
                except Exception as exc:
                    result["candidates"].append({"edit": edit, "error": f"apply/generate failed: {exc}"})
                    continue

                ok_match, diffs = compare_sensors(predicted, actual)
                edge_ok, edge_rec = edge_after_apply(
                    actual_plan, edit["u"], edit["v"], _TYPE_TO_CONNECTION[edit["type"]])
                rec = {"edit": edit, "predicted_eq_actual": ok_match,
                      "edge_present_with_edited_connection": edge_ok,
                      "diffs": diffs}
                if ok_match and edge_ok:
                    totals["predicted_eq_actual"] += 1
                else:
                    totals["mismatches"].append({"plan": plan_key, "edit": edit,
                                                 "diffs": diffs, "edge_ok": edge_ok})
                result["candidates"].append(rec)

                pred_checks = _checks_by_name(predicted)
                for sname in ("access_typed", "reachability", "daylight", "room_depth"):
                    before = _checks_by_name(sensors).get(sname)
                    after = pred_checks.get(sname)
                    if before and after and (not before["ok"]) and after["ok"]:
                        cleared_here.append({"sensor": sname, "edit": edit})

            result["predicted_clears"] = cleared_here
            b_record["results"].append(result)
            cleared[plan_key] = cleared_here

        out["boundaries"].append(b_record)

    out["totals"] = totals
    out["predicted_clears_by_plan"] = cleared

    # ── part-A before/after: does the edge-semantics fix change any sensor
    # output on the SAME six plans, for plans that carry no SetConnection edit
    # (i.e. the sensors run on generate()'s own output, unedited)? Compare the
    # current (post-fix) topo_sensors against the pre-fix version via git show.
    if PRE_FIX_SNAPSHOT.exists():
        pre_fix_src = PRE_FIX_SNAPSHOT.read_text()
    else:
        pre_fix_src = None
        out["before_after_a_error"] = f"pre-fix snapshot not found: {PRE_FIX_SNAPSHOT}"

    if pre_fix_src is not None:
        import types
        pre_mod = types.ModuleType("topo_sensors_pre_fix")
        pre_mod.__dict__["__file__"] = str(REPO_ROOT / "packages/floorplan_core/topo_sensors.py")
        exec(compile(pre_fix_src, "topo_sensors_pre_fix", "exec"), pre_mod.__dict__)

        for entry in targets:
            name = entry["name"]
            boundary, entrance_point, typed_edges, requirements = entry_to_request_parts(entry)
            retrieve_body = {"boundary": boundary, "entrance_point": entrance_point,
                             "requirements": requirements, "topn": TOPN,
                             "typed_edges": typed_edges}
            rv = client.post("/retrieve", json=retrieve_body)
            plans = (rv.get_json().get("plans") or [])[:TOPN]
            for rank, fp in enumerate(plans, start=1):
                plan_key = f"{name}#{rank}"
                spec, plan, room_edges, resolved_types, sensors_after = build_generation(
                    server_mod, fp, boundary, entrance_point, typed_edges, requirements)
                sensors_before = pre_mod.run_sensors(plan, room_edges, requirements=requirements,
                                                     target_boundary=boundary, spec=spec)
                match, diffs = compare_sensors(sensors_before, sensors_after)
                if not match:
                    before_after_a.append({"plan": plan_key, "diffs": diffs})

    out["before_after_a_diffs"] = before_after_a

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT_PATH}", file=sys.stderr)
    print(json.dumps({
        "plans": totals["plans"],
        "candidates_proposed": totals["candidates_proposed"],
        "predicted_eq_actual": totals["predicted_eq_actual"],
        "mismatches": len(totals["mismatches"]),
        "before_after_a_diffs_on_unedited_plans": len(before_after_a),
    }, indent=2))


if __name__ == "__main__":
    main()
