"""Repeatable gate suite for project-44 (2026-11-04).

    .venv/bin/python tools/gate_all.py                       # all stages, default rules
    .venv/bin/python tools/gate_all.py --fast                # skip placement, round_trip_dev, e2e
    .venv/bin/python tools/gate_all.py --stage placement     # one stage
    .venv/bin/python tools/gate_all.py --change data/outputs/gates/changes/<name>.json
    .venv/bin/python tools/gate_all.py --update-baseline --reason "why"
    .venv/bin/python tools/gate_all.py --stage round_trip_holdout   # totals only, logged

Stages, in order: unit, selftest, moves, determinism, placement, round_trip_dev, e2e.
(round_trip_holdout runs only when named; it never passes or fails.)
Rules for placement and round_trip_dev: tools/gate_rules.py.
Every subprocess runs with PYTHONHASHSEED=0 (the Generator output depends on string
hashing across processes; see ARCHITECTURE.md "Gates") and CP-SAT at 1 worker, fixed seed,
deterministic time 1.0.
Output: data/outputs/gates/report_<date>_<hash>.md and .json. Exit 1 on any FAIL.
"""
import argparse
import datetime
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import gate_rules as rules  # noqa: E402

GATES = REPO / "data/outputs/gates"
BASELINES = GATES / "baselines"
PY = sys.executable
WORKERS, DET_TIME = 1, 1.0
E2E_API_PORT, E2E_WEB_PORT = 5012, 5183
ORDER = ["unit", "selftest", "moves", "determinism", "placement", "round_trip_dev", "e2e"]
SLOW = {"placement", "round_trip_dev", "e2e"}
ENV = {**os.environ, "PYTHONHASHSEED": "0"}


def sh(args, cwd=REPO, env=None, timeout=3600):
    p = subprocess.run(args, cwd=cwd, env=env or ENV, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def last_json(text):
    """The last top-level JSON object printed in text."""
    end = text.rstrip().rfind("}")
    if end < 0:
        return None
    depth = 0
    for i in range(end, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i:end + 1])
                except ValueError:
                    return None
    return None


# ── stages ───────────────────────────────────────────────────────────────────

def stage_unit(ctx):
    rc, out, err = sh([PY, "-m", "pytest", "packages/floorplan_core/tests", "-q", "-p", "no:cacheprovider"])
    tail = (out.strip().splitlines() or [""])[-1]
    return {"status": "PASS" if rc == 0 else "FAIL", "detail": tail,
            "failures": [] if rc == 0 else [l for l in out.splitlines() if l.startswith(("FAILED", "ERROR"))][:12] or [tail]}


def stage_selftest(ctx):
    rc, out, err = sh([PY, "packages/floorplan_core/generate.py"], env={**ENV, "PYTHONPATH": "."})
    return {"status": "PASS" if rc == 0 else "FAIL", "detail": f"exit {rc}",
            "failures": [] if rc == 0 else (out + err).strip().splitlines()[-8:]}


def stage_moves(ctx):
    fails, parts = [], []
    rc, out, err = sh([PY, "tools/gate_set_connection.py", "gate"])
    j = last_json(out + err)
    if rc != 0 or j is None:
        fails.append(f"gate_set_connection: exit {rc}, no result")
    else:
        parts.append(f"SetConnection {j['predicted_eq_actual']}/{j['candidates_proposed']} predicted=actual, mismatches {j['mismatches']}")
        if j["mismatches"] != 0:
            fails.append(f"SetConnection mismatches {j['mismatches']}")
    rc, out, err = sh([PY, "tools/gate_set_precedence.py"])
    j = last_json(out + err)
    if rc != 0 or j is None:
        fails.append(f"gate_set_precedence: exit {rc}, no result")
    else:
        parts.append(f"SetPrecedence {j['predicted_eq_actual']}/{j['candidates']} predicted=actual, mismatches {j['mismatches']}")
        if j["mismatches"] != 0:
            fails.append(f"SetPrecedence mismatches {j['mismatches']}")
    return {"status": "FAIL" if fails else "PASS", "detail": "; ".join(parts), "failures": fails}


def stage_determinism(ctx):
    hs = []
    for _ in range(2):
        rc, out, err = sh([PY, "tools/check_determinism.py", "--det-time", str(DET_TIME)])
        j = last_json(out)
        if j is None:
            return {"status": "FAIL", "detail": f"exit {rc}, no result", "failures": (out + err).strip().splitlines()[-6:]}
        if not j["all_identical"]:
            return {"status": "FAIL", "detail": f"{j['identical_pairs']}/{j['cases']} pairs identical",
                    "failures": [f"differ: {j['differ']}"]}
        hs.append(j["hashes"])
    if hs[0] != hs[1]:
        d = [k for k in hs[0] if hs[0][k] != hs[1][k]]
        return {"status": "FAIL", "detail": "plans differ between two processes", "failures": [f"differ: {d}"]}
    return {"status": "PASS", "detail": f"{j['cases']} cases (3 plans x rooms/nodes x cpsat/cpsat_dim), byte-identical twice in-process and across 2 processes"}


def _baseline_path(ctx):
    return BASELINES / f"{ctx['baseline']}.json"


def _load_baseline(ctx):
    p = _baseline_path(ctx)
    return json.loads(p.read_text()) if p.exists() else None


def _judge(ctx, stage, metrics, per_plan, extra=None):
    res = {"metrics": metrics, "per_plan": per_plan, **(extra or {})}
    if ctx["update_baseline"]:
        res.update(status="BASELINE", detail="recorded as baseline (not compared)")
        return res
    base = _load_baseline(ctx)
    if base is None or stage not in base["stages"]:
        res.update(status="FAIL", detail="no baseline", failures=[
            f"baseline '{ctx['baseline']}' has no {stage}; run with --update-baseline --reason ..."])
        return res
    b = base["stages"][stage]
    v = rules.judge(b["metrics"], metrics, b["per_plan"], per_plan, ctx["change"])
    res.update(status=v["verdict"], failures=v["failures"], undeclared=v["undeclared"],
               plan_diffs=v["plan_diffs"], rows=v["rows"], baseline_hash=base["git_hash"],
               detail=(f"vs baseline {base['git_hash']}: " +
                       (f"{len(v['rows'])} metric(s) moved" if v["rows"] else "all metrics identical")
                       + (f", {len(v['plan_diffs'])} per-plan diffs" if v["plan_diffs"] else "")
                       + (f", change '{ctx['change']['name']}'" if ctx["change"] else "")))
    return res


def stage_placement(ctx):
    out = ctx["run_dir"] / "placement.json"
    rc, so, se = sh([PY, "tools/diag_placement_topology.py", "--workers", str(WORKERS), "--det-time", str(DET_TIME),
                     "--out", str(out)])
    if rc != 0 or not out.exists():
        return {"status": "FAIL", "detail": f"exit {rc}", "failures": (so + se).strip().splitlines()[-8:]}
    doc = json.loads(out.read_text())
    metrics, per_plan = rules.extract_placement(doc)
    gate = doc.get("gate", {})
    extra = {"placement_gate_pass": gate.get("pass"),
             "placement_gate_failed": [c["name"] for c in gate.get("criteria", []) if not c["pass"]],
             "totals": {k: {m: v[m] for m in rules.PLACEMENT_METRICS} for k, v in doc["totals"].items()}}
    return _judge(ctx, "placement", metrics, per_plan, extra)


def _round_trip(ctx, plan_set, stage, tag):
    out = ctx["run_dir"] / f"{tag}.json"
    rc, so, se = sh([PY, "tools/diag_round_trip.py", "--plan-set", plan_set, "--n", "300", "--n-cpsat", "30",
                     "--workers", str(WORKERS), "--det-time", str(DET_TIME), "--out", str(out)])
    if rc != 0 or not out.exists():
        return None, {"status": "FAIL", "detail": f"exit {rc}", "failures": (so + se).strip().splitlines()[-8:]}
    return json.loads(out.read_text()), None


def stage_round_trip_dev(ctx):
    doc, bad = _round_trip(ctx, "dev", "round_trip_dev", "round_trip_dev")
    if bad:
        return bad
    metrics, per_plan = rules.extract_round_trip(doc, "round_trip_dev")
    return _judge(ctx, "round_trip_dev", metrics, per_plan,
                  {"totals": {k: {m: v[m] for m in rules.ROUND_TRIP_METRICS} for k, v in doc["totals"].items()}})


def stage_round_trip_holdout(ctx):
    doc, bad = _round_trip(ctx, "holdout", "round_trip_holdout", "round_trip_holdout")
    if bad:
        return bad
    totals = {k: {m: v[m] for m in rules.ROUND_TRIP_METRICS} for k, v in doc["totals"].items()}
    log = GATES / "holdout_log.md"
    if not log.exists():
        log.write_text("# Hold-out log\n\nEach line is one hold-out run: date, git hash. The hold-out is not used "
                       "for tuning; read totals here only, never change code to move them.\n\n")
    with log.open("a") as f:
        f.write(f"- {datetime.datetime.now().isoformat(timespec='seconds')} | {ctx['hash']}{'-dirty' if ctx['dirty'] else ''}\n")
    return {"status": "INFO", "detail": "totals only, no pass/fail; logged in holdout_log.md", "totals": totals}


def _port_open(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait_http(url, secs):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < secs:
        try:
            urllib.request.urlopen(url, timeout=2).read(10)
            return True
        except Exception:
            time.sleep(1)
    return False


def stage_e2e(ctx):
    node = subprocess.run(["which", "node"], capture_output=True, text=True).stdout.strip()
    if not node:
        return {"status": "FAIL", "detail": "node not found", "failures": ["node is not on PATH"]}
    for port in (E2E_API_PORT, E2E_WEB_PORT):
        if _port_open(port):
            return {"status": "FAIL", "detail": f"port {port} busy", "failures": [
                f"port {port} is in use; the gate does not reuse or stop a process it did not start"]}
    shots = GATES / "shots" / ctx["hash"]
    shots.mkdir(parents=True, exist_ok=True)
    logs = ctx["run_dir"]
    procs = []
    try:
        be = subprocess.Popen([PY, "tools/e2e/run_backend.py", "--port", str(E2E_API_PORT), "--workers", str(WORKERS),
                               "--det-time", str(DET_TIME)], cwd=REPO, env=ENV, start_new_session=True,
                              stdout=open(logs / "e2e_backend.log", "w"), stderr=subprocess.STDOUT)
        procs.append(be)
        vite = subprocess.Popen(["./node_modules/.bin/vite", "--port", str(E2E_WEB_PORT), "--strictPort", "--host", "127.0.0.1"],
                                cwd=REPO / "apps/web", env={**ENV, "GATE_API_PORT": str(E2E_API_PORT)},
                                start_new_session=True, stdout=open(logs / "e2e_vite.log", "w"), stderr=subprocess.STDOUT)
        procs.append(vite)
        if not _wait_http(f"http://127.0.0.1:{E2E_API_PORT}/config", 180):
            return {"status": "FAIL", "detail": "backend did not start", "failures": ["see e2e_backend.log"]}
        if not _wait_http(f"http://127.0.0.1:{E2E_WEB_PORT}/", 60):
            return {"status": "FAIL", "detail": "vite did not start", "failures": ["see e2e_vite.log"]}
        result = logs / "e2e_result.json"
        env = {**ENV, "APP_URL": f"http://127.0.0.1:{E2E_WEB_PORT}", "API_URL": f"http://127.0.0.1:{E2E_API_PORT}",
               "OUT_DIR": str(shots), "LIB": str((REPO.parent / "test-boundaries/boundary_library.json").resolve()),
               "RESULT": str(result)}
        rc, so, se = sh([node, "tools/e2e/app_flow.cjs"], env=env, timeout=900)
        res = json.loads(result.read_text()) if result.exists() else None
        shot_list = sorted(str(p.relative_to(REPO)) for p in shots.glob("*.png"))
        if rc == 3:
            return {"status": "FAIL", "detail": "Playwright not found", "failures": [(so + se).strip().splitlines()[-1]]}
        if res is None:
            return {"status": "FAIL", "detail": f"exit {rc}, no result file", "failures": (so + se).strip().splitlines()[-6:]}
        bad = [f"{c['name']} :: {c['detail']}" for c in res["checks"] if not c["ok"]]
        return {"status": "PASS" if (rc == 0 and not bad) else "FAIL",
                "detail": f"{len(res['checks']) - len(bad)}/{len(res['checks'])} checks", "failures": bad,
                "shots": shot_list}
    finally:
        for p in procs:   # only the processes started here
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


STAGES = {"unit": stage_unit, "selftest": stage_selftest, "moves": stage_moves, "determinism": stage_determinism,
          "placement": stage_placement, "round_trip_dev": stage_round_trip_dev, "e2e": stage_e2e,
          "round_trip_holdout": stage_round_trip_holdout}


# ── baseline and report ──────────────────────────────────────────────────────

def write_baseline(ctx, results, reason):
    BASELINES.mkdir(parents=True, exist_ok=True)
    path = _baseline_path(ctx)
    old = json.loads(path.read_text()) if path.exists() else {"stages": {}}
    stages = dict(old["stages"])
    for name in ("placement", "round_trip_dev"):
        r = results.get(name)
        if r and "metrics" in r:
            stages[name] = {"metrics": r["metrics"], "per_plan": r["per_plan"]}
    doc = {"name": ctx["baseline"], "git_hash": ctx["hash"], "dirty": ctx["dirty"], "date": ctx["date"],
           "reason": reason, "solver": {"workers": WORKERS, "det_time": DET_TIME, "seed": "spec default"},
           "stages": stages}
    path.write_text(json.dumps(doc, indent=1))
    log = GATES / "baseline_log.md"
    if not log.exists():
        log.write_text("# Baseline log\n\nBaselines change only through `tools/gate_all.py --update-baseline --reason ...`.\n\n")
    updated = [n for n in ("placement", "round_trip_dev") if results.get(n) and "metrics" in results[n]]
    with log.open("a") as f:
        f.write(f"- {ctx['date']} | baseline `{ctx['baseline']}` | {ctx['hash']}{'-dirty' if ctx['dirty'] else ''} | "
                f"stages {', '.join(updated)} | reason: {reason}\n")


def render_md(ctx, order, results, secs):
    L = [f"# Gate report {ctx['date']} {ctx['hash']}{'-dirty' if ctx['dirty'] else ''}", "",
         f"- Command: `{' '.join(sys.argv)}`",
         f"- Baseline: `{ctx['baseline']}`" + (f" ({results.get('_baseline_hash', '?')})" if results.get("_baseline_hash") else ""),
         f"- Solver: {WORKERS} worker, seed from spec default, deterministic time {DET_TIME}",
         f"- Rules: {'change file ' + ctx['change']['name'] if ctx['change'] else 'default (behaviour must match baseline)'}",
         f"- Overall: **{'FAIL' if ctx['failed'] else 'PASS'}**", "", "## Stages", "",
         "| Stage | Result | Time | Detail |", "|---|---|---|---|"]
    for n in order:
        r = results[n]
        L.append(f"| {n} | {r['status']} | {secs[n]:.0f}s | {r.get('detail', '')} |")
    for n in order:
        r = results[n]
        if r.get("failures"):
            L += ["", f"## {n}: failures", ""] + [f"- {f}" for f in r["failures"]]
        if r.get("undeclared"):
            L += ["", f"## {n}: undeclared moves (reported, not failed)", ""] + [f"- {f}" for f in r["undeclared"]]
        if r.get("rows"):
            L += ["", f"## {n}: metric moves against baseline", "", "| Metric | Baseline | Run | Delta | Direction | Verdict |", "|---|---|---|---|---|---|"]
            L += [f"| {x['metric']} | {x['baseline']} | {x['run']} | {x['delta']} | {x['direction']} | {x['verdict']} |" for x in r["rows"]]
        if r.get("plan_diffs"):
            L += ["", f"## {n}: per-plan differences ({len(r['plan_diffs'])})", ""] + [f"- {k}" for k in r["plan_diffs"][:40]]
        if "placement_gate_pass" in r:
            L += ["", f"Placement script verdict `gate.pass` (cpsat_dim against off and cpsat; not a gate rule): "
                  f"**{r['placement_gate_pass']}**" + (f"; failed criteria: {', '.join(r['placement_gate_failed'])}" if r["placement_gate_failed"] else "")]
        if r.get("totals"):
            L += ["", f"## {n}: totals", "", "```json", json.dumps(r["totals"], indent=1), "```"]
    shots = results.get("e2e", {}).get("shots")
    if shots:
        L += ["", "## View by hand", "", "A gate pass does not replace looking. Open these:", ""] + [f"- `{s}`" for s in shots]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="skip placement, round_trip_dev, e2e")
    ap.add_argument("--stage", choices=sorted(STAGES), help="run one stage")
    ap.add_argument("--change", help="change file (JSON) in data/outputs/gates/changes/")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--reason", help="required with --update-baseline")
    ap.add_argument("--baseline", default="main", help="baseline name (default main)")
    a = ap.parse_args()
    if a.update_baseline and not a.reason:
        ap.error("--update-baseline needs --reason")
    if a.update_baseline and a.change:
        ap.error("--update-baseline and --change do not combine")
    change = None
    if a.change:
        try:
            change = rules.load_change(a.change)
        except (ValueError, OSError) as e:
            print(f"FAIL change file rejected: {e}")
            sys.exit(1)
    h = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    date = datetime.date.today().isoformat()
    run_dir = GATES / "runs" / f"{date}_{h}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = {"hash": h, "dirty": dirty, "date": date, "baseline": a.baseline, "change": change,
           "update_baseline": a.update_baseline, "run_dir": run_dir, "failed": False}
    if a.stage:
        order = [a.stage]
    else:
        order = [s for s in ORDER if not (a.fast and s in SLOW)]
    results, secs = {}, {}
    for name in order:
        t0 = time.time()
        try:
            r = STAGES[name](ctx)
        except Exception as exc:   # a crashed stage is a FAIL, never a pass
            r = {"status": "FAIL", "detail": f"stage crashed: {exc!r}", "failures": [repr(exc)]}
        secs[name] = time.time() - t0
        results[name] = r
        print(f"{r['status']:8s} {name:20s} {secs[name]:6.0f}s  {r.get('detail', '')}", flush=True)
        for f in r.get("failures", [])[:6]:
            print(f"           - {f}", flush=True)
        if r.get("undeclared"):
            print(f"           undeclared moves: {len(r['undeclared'])}", flush=True)
        if "placement_gate_pass" in r:
            print(f"           placement gate.pass (reported, not a rule): {r['placement_gate_pass']}", flush=True)
        if name == "round_trip_holdout":
            print(json.dumps(r["totals"], indent=1))
    if any(r.get("baseline_hash") for r in results.values()):
        results["_baseline_hash"] = next(r["baseline_hash"] for r in results.values() if r.get("baseline_hash"))
    ctx["failed"] = any(r["status"] == "FAIL" for n, r in results.items() if not n.startswith("_"))
    if a.update_baseline:
        if ctx["failed"]:
            print("baseline NOT updated: a stage failed")
        else:
            write_baseline(ctx, results, a.reason)
            print(f"baseline '{a.baseline}' written at {h}{'-dirty' if dirty else ''}; logged in baseline_log.md")
    GATES.mkdir(parents=True, exist_ok=True)
    stem = GATES / f"report_{date}_{h}{'-dirty' if dirty else ''}"
    if a.stage:
        stem = Path(str(stem) + f"_{a.stage}")
    Path(str(stem) + ".md").write_text(render_md(ctx, order, results, secs))
    slim = {n: {k: v for k, v in r.items() if k not in ("per_plan", "metrics")} for n, r in results.items() if not n.startswith("_")}
    Path(str(stem) + ".json").write_text(json.dumps({
        "date": date, "git_hash": h, "dirty": dirty, "argv": sys.argv[1:], "baseline": a.baseline,
        "change": change, "overall": "FAIL" if ctx["failed"] else "PASS", "seconds": secs, "stages": slim}, indent=1))
    print(f"{'FAIL' if ctx['failed'] else 'PASS'} overall; report {Path(str(stem) + '.md').relative_to(REPO)}")
    sys.exit(1 if ctx["failed"] else 0)


if __name__ == "__main__":
    main()
