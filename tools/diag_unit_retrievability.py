"""Does retrieval reach each unit type at unit shapes? (G0, 2026-11-05)

Queries are rectangular units at assumed areas and depths, with the entrance at the
middle of the corridor side. For each query, reports: type matches in the top 5 and in
the pool; the first exact match's rank over the whole corpus; shape and room distance;
target area against source area. Repeats the pool count with the hardcoded dining term
of the room descriptor set to 0, without changing retrieval.py.

Usage: tools/diag_unit_retrievability.py [--tag name]
Output: data/outputs/diagnostics/unit_retrievability_<date>[_tag].json, table on stdout.
"""
import argparse
import collections
import datetime
import json
import logging
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from packages.floorplan_core.geo_ent_num_retrieval import retrieve as geo_retrieve  # noqa: E402
from packages.floorplan_core.geo_ent_num_retrieval import (  # noqa: E402
    entrance_desc_from_point, shape_desc_complex)
from packages.floorplan_core.index_store import build_index_store_from_npz  # noqa: E402
from packages.floorplan_core.retrieval import _req_to_room_desc, run_retrieval  # noqa: E402

DATA = Path.home() / "Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json"
INDEX = REPO / "data/index/rplan_index_v5.npz"
M = 18.0
# Assumed target areas in m2. The demo boundaries are not chosen yet.
TYPES = {"1B1b": (50, 1, 1), "2B1b": (65, 2, 1), "2B2b": (80, 2, 2), "3B2b": (100, 3, 2)}
DEPTHS = (8.0, 10.0, 12.0)
DINING_SLOT = 4


def poly_area(pts):
    return 0.5 * abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                         for i in range(len(pts))))


def unit_type(plan):
    c = collections.Counter(r.get("label") for r in plan["rooms"])
    return f"{c['MasterBedroom'] + c['Bedroom']}B{c['Bathroom']}b"


def rectangle(area, depth):
    """Normalised rectangle, corridor side at y0. Returns boundary and entrance."""
    w, d = area / depth / M, depth / M
    x0, y0 = 0.5 - w / 2, 0.5 - d / 2
    return [[x0, y0], [x0 + w, y0], [x0 + w, y0 + d], [x0, y0 + d]], [0.5, y0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    log = logging.getLogger("retrievability")
    store = build_index_store_from_npz(npz_path=str(INDEX), data_dir=str(DATA), logger=log)
    names = [Path(n).stem for n in store.names]
    type_of, area_of = {}, {}
    for n in names:
        p = store.plan_cache.get(n)
        if p:
            type_of[n] = unit_type(p)
            area_of[n] = poly_area([v[:2] for v in p["boundary"]]) * M * M

    print("corpus boundary area by type, m2 (p10 median p90, n)")
    corpus = {}
    for t in TYPES:
        v = sorted(area_of[n] for n in names if type_of.get(n) == t)
        dec = statistics.quantiles(v, n=10)
        corpus[t] = {"p10": round(dec[0], 1), "median": round(statistics.median(v), 1),
                     "p90": round(dec[8], 1), "min": round(v[0], 1), "n": len(v)}
        print(f"{t}  {corpus[t]['p10']:>6} {corpus[t]['median']:>6} {corpus[t]['p90']:>6}  min {corpus[t]['min']:>5}  n {len(v)}")

    rows = []
    print("\ntype depth target | match top5 pool60 | first rank | src m2 top1 | area ratio | shape room | pool60 no-dining")
    for t, (area, beds, baths) in TYPES.items():
        for depth in DEPTHS:
            boundary, ent = rectangle(area, depth)
            req = {"bedrooms": beds, "bathrooms": baths, "balcony": True}
            pay = run_retrieval(store, boundary, entrance_point=ent, requirements=req, topn=60, pool_n=60)
            plans = pay["plans"]
            ids = [Path(str(p["retrieval"]["source_id"])).stem for p in plans]
            top5 = sum(type_of.get(i) == t for i in ids[:5])
            pool = sum(type_of.get(i) == t for i in ids)
            segs = [list(v) for v in boundary][::-1] if sum(
                boundary[i][0] * boundary[(i + 1) % 4][1] - boundary[(i + 1) % 4][0] * boundary[i][1]
                for i in range(4)) > 0 else [list(v) for v in boundary]
            qs, qe, qr = shape_desc_complex(segs), entrance_desc_from_point(ent, segs), _req_to_room_desc(req)
            order = geo_retrieve(qs, qe, qr, store.shapes, store.ents, store.rooms, 1.0, 2.0, 1.0, len(names))[0]
            first = next((r for r, i in enumerate(order) if type_of.get(names[int(i)]) == t), None)
            qr0 = qr.copy()
            qr0[DINING_SLOT] = 0.0
            order0 = geo_retrieve(qs, qe, qr0, store.shapes, store.ents, store.rooms, 1.0, 2.0, 1.0, 60)[0]
            pool0 = sum(type_of.get(names[int(i)]) == t for i in order0)
            top = plans[0]["retrieval"] if plans else {}
            src = area_of.get(ids[0]) if ids else None
            matches = [area_of[i] for i in ids[:5] if type_of.get(i) == t]
            row = {"type": t, "depth": depth, "target_m2": area, "top5": top5, "pool60": pool, "first_rank": first,
                   "top1_id": ids[0] if ids else None, "top1_type": type_of.get(ids[0]) if ids else None,
                   "top1_src_m2": round(src, 1) if src else None,
                   "area_ratio_top1": round(area / src, 2) if src else None,
                   "match_src_m2": round(statistics.median(matches), 1) if matches else None,
                   "shape_dist": round(top.get("shape_dist") or 0, 3), "room_dist": round(top.get("room_dist") or 0, 3),
                   "pool60_no_dining": pool0, "top5_types": [type_of.get(i) for i in ids[:5]]}
            rows.append(row)
            print(f"{t} {depth:>4} {area:>4} | {top5:>2} {pool:>3} | {str(first):>6} | {str(row['top1_src_m2']):>6} {row['top1_type']} | "
                  f"{row['area_ratio_top1']} | {row['shape_dist']} {row['room_dist']} | {pool0}")

    tag = f"_{a.tag}" if a.tag else ""
    out = REPO / f"data/outputs/diagnostics/unit_retrievability_{datetime.date.today()}{tag}.json"
    with open(out, "w") as fh:
        json.dump({"corpus": corpus, "rows": rows}, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
