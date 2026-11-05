"""Determinism check (2026-11-04): same Spec + options -> byte-identical plan.

For 3 real plans, rooms and nodes, placement cpsat and cpsat_dim, with the gate's
solver settings (1 worker, fixed seed, deterministic time). Each case is generated
twice in this process; the sha256 of the canonical JSON is compared. Wall-clock
fields (report.placement.solve_time_s) are removed first: they are measurements,
not part of the plan. Prints one JSON object; exit 1 if any pair differs.
Run it in two separate processes and compare "hashes" for the cross-process check.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from packages.floorplan_core.generate import SCALE, generate  # noqa: E402
from packages.floorplan_core.node_graph import build_node_graph, expand_plan  # noqa: E402
from packages.floorplan_core.spec import build_options, describe  # noqa: E402

DATA = Path.home() / "Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json"
IDS = ("26135", "78127", "51270")
DET_TIME = 1.0


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k != "solve_time_s"}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def _hash(plan):
    blob = json.dumps(_strip(plan), sort_keys=True, default=float).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-time", type=float, default=DET_TIME)
    ap.add_argument("--modes", default="cpsat,cpsat_dim")
    a = ap.parse_args()
    hashes, same = {}, {}
    for pid in IDS:
        plan = json.loads((DATA / f"{pid}.json").read_text())
        for vname, p in (("rooms", plan), ("nodes", expand_plan(plan, build_node_graph(plan)))):
            spec = describe(p, mm_per_unit=float(SCALE))
            for mode in a.modes.split(","):
                opt = build_options(core_placement=mode, cpsat_workers=1, cpsat_det_time=a.det_time)
                h = [_hash(generate(spec, opt)) for _ in range(2)]
                key = f"{pid}/{vname}/{mode}"
                hashes[key] = h[0]
                same[key] = h[0] == h[1]
    out = {"det_time": a.det_time, "identical_pairs": sum(same.values()), "cases": len(same),
           "all_identical": all(same.values()), "differ": [k for k, v in same.items() if not v],
           "hashes": hashes}
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["all_identical"] else 1)


if __name__ == "__main__":
    main()
