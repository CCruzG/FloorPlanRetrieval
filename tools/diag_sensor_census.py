"""Sensor census on source plans, by unit type (G0, 2026-11-05).

What do the geometry-layer sensors say about RPlan source plans, per sensor and per
unit type? Which room labels fail min size? Does a different scale change the answer?
No generation is involved. The hold-out is excluded.

Usage: tools/diag_sensor_census.py [--n 300] [--seed 5] [--tag name]
Output: data/outputs/diagnostics/sensor_census_<date>[_tag].json, tables on stdout.
"""
import argparse
import collections
import datetime
import json
import random
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shapely.geometry import Polygon  # noqa: E402

from packages.floorplan_core.generate import SCALE  # noqa: E402
from packages.floorplan_core.spec import describe  # noqa: E402
from packages.floorplan_core.topo_sensors import MINIMUM_SIZES, run_sensors  # noqa: E402
from packages.floorplan_core.topo_signature import exact_room_edges  # noqa: E402

DATA = Path.home() / "Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json"
TYPES = ("1B1b", "2B1b", "2B2b", "3B2b")
FACTORS = (1.0, 1.05, 1.1, 1.2)
KEY = {"LivingRoom": "living", "MasterBedroom": "master_bedroom", "Bedroom": "bedroom",
       "Kitchen": "kitchen", "DiningRoom": "dining", "Bathroom": "bathroom",
       "Balcony": "balcony", "Storage": "storage"}


def unit_type(plan):
    c = collections.Counter(r.get("label") for r in plan["rooms"])
    return f"{c['MasterBedroom'] + c['Bedroom']}B{c['Bathroom']}b"


def _scan(path):
    with open(path) as fh:
        plan = json.load(fh)
    return str(plan["id"]), unit_type(plan)


def _census(pid):
    with open(DATA / f"{pid}.json") as fh:
        plan = json.load(fh)
    out = {"id": pid, "type": unit_type(plan), "factors": {}, "rooms": []}
    labels = {str(r["id"]): r.get("label") for r in plan["rooms"]}
    for f in FACTORS:
        mm = float(SCALE) * f
        spec = describe(plan, mm_per_unit=mm)
        edges = exact_room_edges(plan, spec["boundary"], spec["edge_types"])
        s = run_sensors(plan, edges, requirements=None, target_boundary=spec["boundary"],
                        spec=spec, layer="geometry")
        out["factors"][str(f)] = {c["sensor"]: bool(c["ok"]) for c in s["checks"]}
        if f == 1.0:
            for c in s["checks"]:
                if c["sensor"] == "min_room_size":
                    out["min_size_failing"] = [labels.get(str(r), str(r)) for r in c["failing_rooms"]]
                    out["min_size_detail"] = str(c.get("detail", ""))[:400]
    for r in plan["rooms"]:
        limit = MINIMUM_SIZES.get(KEY.get(r.get("label")))
        if not limit or len(r.get("polygon") or []) < 3:
            continue
        g = Polygon(r["polygon"])
        g = g if g.is_valid else g.buffer(0)
        out["rooms"].append({"label": r["label"], "m2": round(g.area * (SCALE / 1000) ** 2, 2),
                             "min_m2": limit["area"] / 1e6})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    with open(REPO / "data/outputs/gates/plan_sets.json") as fh:
        sets = json.load(fh)
    holdout = {str(i) for i in sets.get("holdout", [])}
    with Pool(8) as p:
        scanned = p.map(_scan, sorted(DATA.glob("*.json")), chunksize=500)
    counts = collections.Counter(t for _, t in scanned)
    print("plan_sets keys:", list(sets.keys()))
    print("corpus counts:", {t: counts[t] for t in TYPES}, "holdout excluded:", len(holdout))
    rng = random.Random(a.seed)
    ids = []
    for t in TYPES:
        pool = sorted(i for i, u in scanned if u == t and i not in holdout)
        ids += rng.sample(pool, min(a.n, len(pool)))
    with Pool(8) as p:
        rows = p.map(_census, ids, chunksize=10)

    sensors = sorted(rows[0]["factors"]["1.0"])
    print("\nsource plans passing, % (scale 1.0)")
    print(f"{'sensor':26}" + "".join(f"{t:>7}" for t in TYPES))
    per = {}
    for s in sensors:
        vals = []
        for t in TYPES:
            sub = [r for r in rows if r["type"] == t]
            vals.append(round(100 * sum(r["factors"]["1.0"][s] for r in sub) / len(sub), 1))
        per[s] = dict(zip(TYPES, vals))
        print(f"{s:26}" + "".join(f"{v:>7}" for v in vals))

    print("\nmin_room_size pass %, by scale factor")
    scale = {}
    for f in FACTORS:
        vals = []
        for t in TYPES:
            sub = [r for r in rows if r["type"] == t]
            vals.append(round(100 * sum(r["factors"][str(f)]["min_room_size"] for r in sub) / len(sub), 1))
        scale[str(f)] = dict(zip(TYPES, vals))
        print(f"x{f:<5}" + "".join(f"{v:>7}" for v in vals))

    print("\nplans where a label fails min_room_size, % of plans with that label")
    has, fail = collections.Counter(), collections.Counter()
    for r in rows:
        for lb in {x["label"] for x in r["rooms"]}:
            has[lb] += 1
        for lb in set(r.get("min_size_failing", [])):
            fail[lb] += 1
    bylabel = {lb: round(100 * fail[lb] / has[lb], 1) for lb in has}
    for lb, v in sorted(bylabel.items(), key=lambda kv: -kv[1]):
        print(f"{lb:16}{v:>7}  of {has[lb]}")

    print("\nraw room area, m2 (label: min | p25 median p75 | % at or above min)")
    area = {}
    for lb in has:
        v = sorted(x["m2"] for r in rows for x in r["rooms"] if x["label"] == lb)
        mn = next(x["min_m2"] for r in rows for x in r["rooms"] if x["label"] == lb)
        q = statistics.quantiles(v, n=4)
        area[lb] = {"min": mn, "p25": round(q[0], 1), "median": round(q[1], 1), "p75": round(q[2], 1),
                    "pct_ok": round(100 * sum(x >= mn for x in v) / len(v), 1)}
        print(f"{lb:16}{mn:>6} | {area[lb]['p25']:>5} {area[lb]['median']:>5} {area[lb]['p75']:>5} | {area[lb]['pct_ok']:>5}")

    tag = f"_{a.tag}" if a.tag else ""
    out = REPO / f"data/outputs/diagnostics/sensor_census_{datetime.date.today()}{tag}.json"
    with open(out, "w") as fh:
        json.dump({"n": a.n, "seed": a.seed, "corpus_counts": dict(counts), "per_sensor": per, "scale": scale,
                   "min_size_by_label": bylabel, "area": area, "rows": rows}, fh)
    print("\nwrote", out)
    for r in rows[:3]:
        print(r["type"], r.get("min_size_failing"), r.get("min_size_detail", "")[:260])


if __name__ == "__main__":
    main()
