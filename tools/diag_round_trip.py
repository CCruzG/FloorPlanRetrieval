"""Round trip: does a plan's own Spec give the plan back? (2026-11-04)

Separates generator loss from fit loss, with no input boundary involved:
  describe  plan -> describe(plan) -> generate -> compare with the plan
  identity  plan -> fit(describe(plan), its own boundary, topology layer) -> generate -> compare
Run for rooms and nodes (node_graph.expand_plan), placement off and cpsat.

Compared with the source plan:
  feasible; coverage; iou (room polygon, source vs generated, mean and worst per
  plan); area_err; lost_door_open (door/open source edges dropped by the Generator);
  unmet_fronts; entrance_met; sensors passed on the generated plan against the same
  sensors on the source plan (a sensor the source itself fails is not generator loss).

Usage: tools/diag_round_trip.py [--n 300] [--n-cpsat 30] [--seed 4] [--tag name]
         [--plan-set dev|holdout] [--workers N] [--det-time T] [--out PATH]
         [--source-grid off|on|flex]
--source-grid (2026-11-10): generate with options["source_grid"], the plan's own rooms
as source_rooms (same frame in a round trip). Default off, as the gate runs it.
--plan-set draws the ids from data/outputs/gates/plan_sets.json (in file order, no
GATE_IDS prefix); cpsat uses the first --n-cpsat ids of the set. --workers and
--det-time set the CP-SAT workers and deterministic time (gate runs: 1 and 1.0).
Output: data/outputs/diagnostics/round_trip_<date>[_tag].json, per-mode totals on stdout.
"""
import argparse
import datetime
import json
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shapely.geometry import Polygon  # noqa: E402

from packages.floorplan_core.generate import SCALE, generate  # noqa: E402
from packages.floorplan_core.node_graph import build_node_graph, expand_plan  # noqa: E402
from packages.floorplan_core.spec import build_options, describe, fit, spec_problems  # noqa: E402
from packages.floorplan_core.topo_sensors import run_sensors  # noqa: E402
from packages.floorplan_core.topo_signature import exact_room_edges, topological_fitness  # noqa: E402

DATA = Path.home() / "Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json"
GATE_IDS = ("26135", "78127", "51270", "1789", "19679", "45549", "6079")
MM = float(SCALE)


def _poly(pts):
    g = Polygon(pts)
    return g if g.is_valid else g.buffer(0)


def _sensors(plan, spec):
    edges = exact_room_edges(plan, spec["boundary"], spec["edge_types"])
    s = run_sensors(plan, edges, requirements=None, target_boundary=spec["boundary"],
                    spec=spec, layer="geometry")
    return {c["sensor"]: bool(c["ok"]) for c in s["checks"]}


def compare(source, spec, options):
    g = generate(spec, options)
    rep = g["report"]
    src = {str(r["id"]): r["polygon"] for r in source["rooms"] if len(r.get("polygon") or []) >= 3}
    gen = {str(r["id"]): r["polygon"] for r in g["rooms"]}
    ious, area_errs = [], []
    for rid, sp in src.items():
        a = _poly(sp)
        if rid not in gen:
            ious.append(0.0)
            continue
        b = _poly(gen[rid])
        union = a.union(b).area
        ious.append(a.intersection(b).area / union if union else 0.0)
        area_errs.append(abs(b.area - a.area) / (a.area or 1.0))
    src_s, gen_s = _sensors(source, spec), _sensors(g, spec)
    bc = rep.get("boundary_contacts") or {}
    return {
        "feasible": bool(g["feasible"]),
        "coverage": rep.get("coverage"),
        "iou_mean": round(statistics.mean(ious), 4) if ious else 0.0,
        "iou_worst": round(min(ious), 4) if ious else 0.0,
        "area_err": round(statistics.mean(area_errs), 4) if area_errs else None,
        "lost_door_open": len((rep.get("edge_reconciliation") or {}).get("dropped_connections") or []),
        "unmet_fronts": len(bc.get("unmet") or []),
        "entrance_met": (bc.get("entrance") or {}).get("met"),
        "src_passed": sum(src_s.values()), "gen_passed": sum(gen_s.values()), "total": len(gen_s),
        "regressed": sorted(k for k, v in src_s.items() if v and not gen_s.get(k)),
        "src_failed": sorted(k for k, v in src_s.items() if not v),
    }


def specs_for(plan):
    src = describe(plan, mm_per_unit=MM)
    yield "describe", src
    ent = src["entrance"]
    topo = topological_fitness(plan, target_boundary=src["boundary"], target_edge_types=src["edge_types"],
                               entrance_point=ent[:2] if ent else None)
    yield "identity", fit(src, src["boundary"], edge_types=src["edge_types"], entrance=ent,
                          mm_per_unit=MM, node_pos=topo.get("node_pos", {}), topo=topo)


def summarise(rows):
    n = len(rows) or 1
    reg = {}
    for r in rows:
        for k in r["regressed"]:
            reg[k] = reg.get(k, 0) + 1
    med = lambda k: round(statistics.median(r[k] for r in rows), 3) if rows else None
    return {
        "plans": len(rows),
        "feasible": sum(r["feasible"] for r in rows),
        "iou_median": med("iou_mean"), "iou_worst_median": med("iou_worst"),
        "iou_ge_0.8": sum(r["iou_mean"] >= 0.8 for r in rows),
        "plans_losing_door": sum(r["lost_door_open"] > 0 for r in rows),
        "lost_door_open": sum(r["lost_door_open"] for r in rows),
        "plans_unmet_fronts": sum(r["unmet_fronts"] > 0 for r in rows),
        "entrance_missed": sum(r["entrance_met"] is False for r in rows),
        "src_passed_mean": round(sum(r["src_passed"] for r in rows) / n, 2),
        "gen_passed_mean": round(sum(r["gen_passed"] for r in rows) / n, 2),
        "plans_regressed": sum(bool(r["regressed"]) for r in rows),
        "regressed_by_sensor": dict(sorted(reg.items(), key=lambda kv: -kv[1])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-cpsat", type=int, default=30)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--plan-set", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--det-time", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--source-grid", choices=("off", "on", "flex"), default="off")
    a = ap.parse_args()
    sg = {"off": False, "on": True, "flex": "flex"}[a.source_grid]
    if a.plan_set:
        ids = json.loads((REPO / "data/outputs/gates/plan_sets.json").read_text())[a.plan_set]
    else:
        ids = sorted(p.stem for p in DATA.glob("*.json"))
        random.Random(a.seed).shuffle(ids)
        ids = [i for i in GATE_IDS if (DATA / f"{i}.json").exists()] + ids
    solver_kw = {}
    if a.workers is not None:
        solver_kw["cpsat_workers"] = a.workers
    if a.det_time is not None:
        solver_kw["cpsat_det_time"] = a.det_time
    out, errors, invalid = {}, [], {}
    for placement, n in ((False, a.n), ("cpsat", a.n_cpsat)):
        options = build_options(core_placement=placement, **solver_kw)
        for pid in ids[:n]:
            plan = json.loads((DATA / f"{pid}.json").read_text())
            try:
                variants = (("rooms", plan), ("nodes", expand_plan(plan, build_node_graph(plan))))
                for vname, p in variants:
                    for kind, spec in specs_for(p):
                        key = f"{vname}/{kind}/{'cpsat' if placement else 'off'}"
                        probs = spec_problems(spec)
                        if probs:
                            invalid[key] = invalid.get(key, 0) + 1
                            continue
                        opts = options
                        if sg:
                            opts = dict(options, source_grid=sg, source_rooms={
                                str(rm["id"]): rm["polygon"] for rm in p["rooms"]
                                if len(rm.get("polygon") or []) >= 3})
                        r = compare(p, spec, opts)
                        r["id"] = pid
                        out.setdefault(key, []).append(r)
            except Exception as exc:  # report and go on: a crash is a finding, not a stop
                errors.append({"id": pid, "placement": str(placement), "error": repr(exc)[:200]})
    totals = {k: summarise(v) for k, v in out.items()}
    stamp = datetime.date.today().isoformat()
    path = Path(a.out) if a.out else (
        REPO / "data/outputs/diagnostics" / f"round_trip_{stamp}{'_' + a.tag if a.tag else ''}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"generated_at": stamp, "n": a.n, "n_cpsat": a.n_cpsat, "seed": a.seed,
                                "plan_set": a.plan_set, "solver": solver_kw, "source_grid": a.source_grid,
                                "totals": totals, "invalid_specs": invalid, "errors": errors,
                                "plans": out}, indent=1))
    print(json.dumps({"totals": totals, "invalid_specs": invalid, "n_errors": len(errors),
                      "errors": errors[:5]}, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
