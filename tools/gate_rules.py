"""Gate rules (2026-11-04). Written before any comparison numbers were seen.

Two modes.

DEFAULT (no change file): behaviour must match the baseline.
  - every tracked metric equals its baseline value;
  - every deterministic placement run equals its baseline on sensors_ok (all
    sensors) and passed, per plan;
  - round-trip totals equal the baseline.
  Any difference is a FAIL, better or worse. An intended change goes through
  --change or --update-baseline.

CHANGE (--change FILE): FILE is JSON in data/outputs/gates/changes/:
    {"name": str, "written_before_code": true,
     "may_move":     {metric: {"direction": "worse"|"better", "max": number|null}},
     "must_improve": {metric: min_delta}}
  - must_improve: the metric must move in its good direction by at least min_delta.
  - may_move: the metric may move in the declared direction by at most max
    (null = unbounded). A declared "worse" metric may also improve.
  - Any metric not named must not get worse. Equal is fine.
  - A metric not named that improves is reported as "undeclared move" and does
    not fail.
  - Per-plan flips are listed for reading; the totals decide.
  - written_before_code must be true, or the file is rejected.

Metric keys are "stage/run/metric". In a change file, a bare metric name
("sensors_passed") matches that metric in every run; a glob containing "/"
("placement/nodes+cpsat/entrance_fail") matches exactly. The longer pattern wins.

Metrics tracked per placement run: sensors_passed, entrance_fail,
reachability_fail, d24_failing, lost_source_door_open_edges, infeasible,
order_violated. Per round-trip run: feasible, iou_median, plans_losing_door,
lost_door_open, entrance_missed, gen_passed_mean, plans_regressed.

The placement script's own "gate.pass" verdict (cpsat_dim against off and cpsat)
is reported and is not a rule here.
"""
import fnmatch
import json

PLACEMENT_METRICS = ["sensors_passed", "entrance_fail", "reachability_fail", "d24_failing",
                     "lost_source_door_open_edges", "infeasible", "order_violated"]
ROUND_TRIP_METRICS = ["feasible", "iou_median", "plans_losing_door", "lost_door_open",
                      "entrance_missed", "gen_passed_mean", "plans_regressed"]
HIGHER_BETTER = {"sensors_passed", "feasible", "iou_median", "gen_passed_mean"}
EPS = 1e-9


def good_direction(metric_key):
    """+1 when a larger value is better, -1 when a smaller one is."""
    return 1 if metric_key.rsplit("/", 1)[-1] in HIGHER_BETTER else -1


# ── extraction ──────────────────────────────────────────────────────────────

def extract_placement(doc):
    """(metrics, per_plan) from a diag_placement_topology.py output."""
    metrics, per_plan = {}, {}
    for run, t in doc["totals"].items():
        for m in PLACEMENT_METRICS:
            metrics[f"placement/{run}/{m}"] = t[m]
    for b in doc["boundaries"]:
        for r in b["ranks"]:
            for run in doc["modes"]:
                e = r.get(run)
                if e is None:
                    continue
                per_plan[f"placement/{run}/{b['name']}#{r['rank']}"] = {
                    "sensors_ok": e["sensors_ok"], "passed": e["passed"]}
    return metrics, per_plan


def extract_round_trip(doc, stage="round_trip_dev"):
    """(metrics, per_plan) from a diag_round_trip.py output."""
    metrics, per_plan = {}, {}
    for run, t in doc["totals"].items():
        for m in ROUND_TRIP_METRICS:
            metrics[f"{stage}/{run}/{m}"] = t[m]
    metrics[f"{stage}/all/errors"] = len(doc.get("errors") or [])
    for run, rows in doc["plans"].items():
        for r in rows:
            per_plan[f"{stage}/{run}/{r['id']}"] = {"gen_passed": r["gen_passed"], "feasible": r["feasible"]}
    return metrics, per_plan


# ── judging ─────────────────────────────────────────────────────────────────

def load_change(path):
    c = json.loads(open(path).read())
    problems = []
    if not c.get("name"):
        problems.append("missing name")
    if c.get("written_before_code") is not True:
        problems.append("written_before_code must be true")
    for m, spec in (c.get("may_move") or {}).items():
        if spec.get("direction") not in ("worse", "better"):
            problems.append(f"may_move[{m}].direction must be worse or better")
    for m, d in (c.get("must_improve") or {}).items():
        if not isinstance(d, (int, float)) or d <= 0:
            problems.append(f"must_improve[{m}] must be a positive number")
    if problems:
        raise ValueError("; ".join(problems))
    return c


def _lookup(table, key):
    """Longest matching pattern in table for key, or None."""
    best = None
    for pat in table:
        if "/" in pat:
            hit = fnmatch.fnmatchcase(key, pat)
        else:
            hit = key.rsplit("/", 1)[-1] == pat
        if hit and (best is None or ("/" in pat and "/" not in best)):
            best = pat
    return best


def judge(base_m, new_m, base_p, new_p, change=None):
    """Compare a run with the baseline. Returns a dict:
    verdict PASS|FAIL, failures [str], undeclared [str], plan_diffs [str], rows [dict]."""
    failures, undeclared, rows = [], [], []
    for key in sorted(set(base_m) | set(new_m)):
        if key not in base_m or key not in new_m:
            failures.append(f"{key}: missing in {'baseline' if key not in base_m else 'run'}")
            continue
        b, n = base_m[key], new_m[key]
        delta = n - b
        if abs(delta) <= EPS:
            continue
        better = delta * good_direction(key) > 0
        row = {"metric": key, "baseline": b, "run": n, "delta": round(delta, 4),
               "direction": "better" if better else "worse", "verdict": "ok"}
        if change is None:
            row["verdict"] = "FAIL"
            failures.append(f"{key}: {b} -> {n} (default mode: any difference fails)")
        else:
            mi = _lookup(change.get("must_improve") or {}, key)
            mm = _lookup(change.get("may_move") or {}, key)
            if mi is not None:
                need = change["must_improve"][mi]
                if not better:
                    row["verdict"] = "FAIL"
                    failures.append(f"{key}: {b} -> {n}, must improve by {need}")
                elif abs(delta) + EPS < need:
                    row["verdict"] = "FAIL"
                    failures.append(f"{key}: improved by {abs(delta)}, need {need}")
            elif mm is not None:
                spec = change["may_move"][mm]
                allowed = spec["direction"]
                mx = spec.get("max")
                if better and allowed == "worse":
                    pass                                    # improving a "may get worse" metric is fine
                elif (allowed == "better") != better:
                    row["verdict"] = "FAIL"
                    failures.append(f"{key}: {b} -> {n}, declared to move {allowed} only")
                elif mx is not None and abs(delta) > mx + EPS:
                    row["verdict"] = "FAIL"
                    failures.append(f"{key}: moved {abs(delta)} {row['direction']}, max {mx}")
            elif better:
                row["verdict"] = "undeclared move"
                undeclared.append(f"{key}: {b} -> {n}")
            else:
                row["verdict"] = "FAIL"
                failures.append(f"{key}: {b} -> {n} (worse, not declared)")
        rows.append(row)
    if change is not None:
        for pat, need in (change.get("must_improve") or {}).items():
            if not any(_lookup({pat: 1}, k) for k in new_m):
                failures.append(f"must_improve pattern matches no metric: {pat}")
    plan_diffs = []
    for key in sorted(set(base_p) | set(new_p)):
        if base_p.get(key) != new_p.get(key):
            plan_diffs.append(key)
    if change is None and plan_diffs:
        failures.append(f"{len(plan_diffs)} per-plan results differ from baseline "
                        f"(sensors_ok / passed): {', '.join(plan_diffs[:6])}"
                        + (" ..." if len(plan_diffs) > 6 else ""))
    return {"verdict": "FAIL" if failures else "PASS", "failures": failures,
            "undeclared": undeclared, "plan_diffs": plan_diffs, "rows": rows}
