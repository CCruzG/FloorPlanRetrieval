"""Does a grid cut at the source's walls realise a retrieved plan better? (experiment, 2026-11-05)

Retrieves the top 3 plans on the boundary library and on the unit shapes, then generates
each in nodes mode, placement off: off = today; flex = source walls and ownership with
the slack solve; on = the same with the source-width solve. Compares with
the source mapped into the target frame, so this tests adaptation at a NEW boundary.
Hallway area and room overlap measure correspondence. Sensors and doors measure cost.

Usage: tools/diag_source_grid.py [--case FILE] [--png FILE] [--tag name] [--topn 3] [--modes off,flex,on]
--topn, --modes (2026-11-10): the A3 rank test runs --topn 5 --modes on. A rank table
then reports, per mode, sensors passed by retrieval rank and how often rank 1 is best.
--case: a JSON with B, ent, te, req, plans. Draws its rank 1 both ways to --png.
Output: data/outputs/diagnostics/source_grid_<date>[_tag].json, table on stdout.
"""
import argparse
import datetime
import json
import logging
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shapely.geometry import Polygon  # noqa: E402

from apps.api import server as S  # noqa: E402
from packages.floorplan_core import spec as spec_mod  # noqa: E402
from packages.floorplan_core.index_store import build_index_store_from_npz  # noqa: E402
from packages.floorplan_core.node_graph import build_node_graph, expand_plan  # noqa: E402
from packages.floorplan_core.topo_signature import default_edge_types, map_rooms_to_target, topological_fitness  # noqa: E402

DATA = Path.home() / "Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json"
M = 18.0
UNITS = {"1B1b": (50, 1, 1), "2B1b": (65, 2, 1), "2B2b": (80, 2, 2), "3B2b": (100, 3, 2)}
_real_build = spec_mod.build_options
MODE = {"source_grid": False}


def _patched(*a, **k):
    o = _real_build(*a, **k)
    o["source_grid"] = MODE["source_grid"]
    return o


spec_mod.build_options = _patched


def poly(p):
    g = Polygon(p)
    return g if g.is_valid else g.buffer(0)


def library():
    """The placement gate's six boundaries, parsed as that gate parses them."""
    with open(REPO.parent / "test-boundaries" / "boundary_library.json") as fh:
        lib = {e["name"]: e for e in json.load(fh)}
    out = []
    for name in ("test_boundary_001", "test_boundary_002", "test_boundary_003",
                 "test_boundary_004", "test_boundary_001b", "test_boundary_005b"):
        e = lib[name]
        out.append({"name": name, "B": [[v["x"], v["y"]] for v in e["vertices"]],
                    "ent": [e["entrance"]["x"], e["entrance"]["y"]],
                    "te": [{"edge_index": i, "type": t} for i, t in enumerate(e.get("edgeTypes") or []) if t],
                    "req": e.get("requirements") or {}})
    return out


def unit_shapes():
    out = []
    for t, (area, beds, baths) in UNITS.items():
        for depth in (8.0, 10.0, 12.0):
            w, d = area / depth / M, depth / M
            x, y = 0.5 - w / 2, 0.5 - d / 2
            b = [[x, y], [x + w, y], [x + w, y + d], [x, y + d]]
            out.append({"name": f"{t}_d{int(depth)}", "B": b, "ent": [0.5, y],
                        "te": [{"edge_index": 0, "type": "facing_corridor"}, {"edge_index": 1, "type": "shared_wall"},
                               {"edge_index": 2, "type": "external"}, {"edge_index": 3, "type": "shared_wall"}],
                        "req": {"bedrooms": beds, "bathrooms": baths}})
    return out


def run(client, e, fp, on):
    MODE["source_grid"] = on
    return client.post("/generate", json={"fp": fp, "boundary": e["B"], "entrance_point": e["ent"], "typed_edges": e["te"],
                                          "requirements": e["req"], "core_placement": False, "expand_nodes": True}).get_json()


def measure(e, fp, g):
    ex = expand_plan(fp, build_node_graph(fp))
    types = default_edge_types(e["B"], e["ent"])
    for t in e["te"] or []:
        types[t["edge_index"]] = t["type"]
    topo = topological_fitness(ex, target_boundary=e["B"], target_edge_types=types, entrance_point=e["ent"], requirements=e["req"])
    src = map_rooms_to_target(ex, e["B"], e["ent"], topo.get("mirror", False))
    gen = {str(r["id"]): r for r in g.get("rooms") or []}
    ious, ha_src, ha_gen, sliv = [], 0.0, 0.0, 0
    hall = {str(r["id"]) for r in ex["rooms"] if r.get("label") == "Hallway"}
    for rid, pts in src.items():
        a = poly(pts)
        b = poly(gen[rid]["polygon"]) if rid in gen and len(gen[rid].get("polygon") or []) >= 3 else None
        u = a.union(b).area if b is not None else a.area
        ious.append(a.intersection(b).area / u if (b is not None and u) else 0.0)
        if rid in hall:
            ha_src += a.area
            ha_gen += b.area if b is not None else 0.0
    # Notch count: a generated room that is not close to its bounding rectangle.
    for r in gen.values():
        if len(r.get("polygon") or []) >= 3:
            p = poly(r["polygon"])
            x0, y0, x1, y1 = p.bounds
            if (x1 - x0) * (y1 - y0) > 0 and p.area / ((x1 - x0) * (y1 - y0)) < 0.92:
                sliv += 1
    s = (g.get("sensors") or {}).get("summary") or {}
    return {"feasible": bool(g.get("feasible")), "iou": statistics.mean(ious) if ious else 0.0,
            "hall_src": ha_src * M * M, "hall_gen": ha_gen * M * M, "notched": sliv, "passed": s.get("passed", 0),
            "total": s.get("total", 0),
            "lost": len(((g.get("report") or {}).get("edge_reconciliation") or {}).get("dropped_connections") or [])}


def draw(case, fp, off, on, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ex = expand_plan(fp, build_node_graph(fp))
    types = default_edge_types(case["B"], case["ent"])
    for t in case["te"]:
        types[t["edge_index"]] = t["type"]
    topo = topological_fitness(ex, target_boundary=case["B"], target_edge_types=types, entrance_point=case["ent"], requirements=case["req"])
    src = map_rooms_to_target(ex, case["B"], case["ent"], topo.get("mirror", False))
    labels = {str(r["id"]): r.get("label", "") for r in ex["rooms"]}
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    panels = [("source, mapped to target", {k: v for k, v in src.items()}), ("generated: today", {str(r["id"]): r["polygon"] for r in off["rooms"]}),
              ("generated: source grid", {str(r["id"]): r["polygon"] for r in on["rooms"]})]
    colours = plt.cm.tab10.colors
    order = sorted(src)
    for ax, (title, rooms) in zip(axs, panels):
        bx, by = zip(*(case["B"] + case["B"][:1]))
        ax.plot([x * M for x in bx], [y * M for y in by], "k", lw=2)
        for rid, pts in rooms.items():
            if len(pts) < 3:
                continue
            g = poly(pts)
            c = colours[order.index(rid) % 10] if rid in order else "0.7"
            for part in getattr(g, "geoms", [g]):
                x, y = part.exterior.xy
                ax.fill([v * M for v in x], [v * M for v in y], color=c, alpha=0.45, ec="k", lw=0.8)
            x0, y0, x1, y1 = g.bounds
            ax.text(g.representative_point().x * M, g.representative_point().y * M,
                    f"{labels.get(rid, rid)[:8]}\n{(x1 - x0) * M:.2f}x{(y1 - y0) * M:.2f}\n{g.area * M * M:.1f} m2", ha="center", va="center", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def rank_table(rows, keys, topn):
    """A3 rank test: does retrieval rank predict sensors passed after generation?
    Per group and mode: mean sensors passed per rank (infeasible or error counts 0);
    queries where rank 1 is best (ties count as best) and where it is strictly best;
    mean gain of picking the best of the top n over rank 1."""
    out = []
    print(f"\nA3 rank test | group | mode | queries | passed by rank 1..{topn} | rank 1 best (tie) | strictly | best-of-n gain")
    for group in ("library", "unit shapes"):
        for key in keys:
            q = {}
            for r in rows:
                if r["group"] != group:
                    continue
                m = r.get(key) or {}
                v = 0 if ("error" in m or not m.get("feasible")) else m.get("passed", 0)
                q.setdefault(r["name"], {})[r["rank"]] = v
            q = {n: v for n, v in q.items() if 1 in v}
            if not q:
                continue
            by_rank = [round(statistics.mean(v[k] for v in q.values() if k in v), 2)
                       if any(k in v for v in q.values()) else None for k in range(1, topn + 1)]
            best = sum(v[1] >= max(v.values()) for v in q.values())
            strict = sum(v[1] > max([x for k, x in v.items() if k != 1] or [-1]) for v in q.values())
            gain = round(statistics.mean(max(v.values()) - v[1] for v in q.values()), 2)
            rec = {"group": group, "mode": key, "queries": len(q), "passed_by_rank": by_rank,
                   "rank1_best": best, "rank1_strict": strict, "best_of_n_gain": gain,
                   "per_query": q}
            out.append(rec)
            print(f"{group:12} | {key:4} | {len(q)} | {by_rank} | {best} | {strict} | +{gain}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    ap.add_argument("--png")
    ap.add_argument("--tag", default="")
    ap.add_argument("--topn", type=int, default=3)
    ap.add_argument("--modes", default="off,flex,on")
    a = ap.parse_args()
    MODES = {"off": False, "flex": "flex", "on": True}
    modes = [(k, MODES[k]) for k in a.modes.split(",")]
    S.INDEX_STORE = build_index_store_from_npz(npz_path=str(REPO / "data/index/rplan_index_v5.npz"), data_dir=str(DATA), logger=logging.getLogger("x"))
    client = S.app.test_client()
    if a.case:
        with open(a.case) as fh:
            case = json.load(fh)
        fp = case["plans"][0]
        off, on = run(client, case, fp, False), run(client, case, fp, True)
        print("flex", measure(case, fp, run(client, case, fp, "flex")))
        print("case rank 1", fp["id"])
        for name, g in (("today", off), ("source grid", on)):
            print(name, measure(case, fp, g))
        if a.png:
            draw(case, fp, off, on, a.png)
            print("drew", a.png)

    rows = []
    for group, entries in (("library", library()), ("unit shapes", unit_shapes())):
        for e in entries:
            r = client.post("/retrieve", json={"boundary": e["B"], "entrance_point": e["ent"], "requirements": e["req"], "topn": a.topn, "typed_edges": e["te"]}).get_json()
            for rank, fp in enumerate(r.get("plans", [])[:a.topn], 1):
                rec = {"group": group, "name": e["name"], "rank": rank, "id": fp["id"]}
                for key, on in modes:
                    try:
                        rec[key] = measure(e, fp, run(client, e, fp, on))
                    except Exception as exc:  # reported, never hidden
                        rec[key] = {"error": str(exc)[:160]}
                rows.append(rec)

    print("\ngroup plans | mode | errors infeasible | iou | hall m2 src gen ratio | notched rooms per plan | sensors | plans losing a door %")
    summary = []
    for group in ("library", "unit shapes"):
        sub = [r for r in rows if r["group"] == group]
        for key, _ in modes:
            ok = [r[key] for r in sub if "error" not in r[key]]
            if not ok:
                continue
            ratio = [x["hall_gen"] / x["hall_src"] for x in ok if x["hall_src"] > 0]
            rec = {"group": group, "mode": key, "plans": len(sub), "errors": len(sub) - len(ok), "infeasible": sum(not x["feasible"] for x in ok),
                   "iou": round(statistics.mean(x["iou"] for x in ok), 3), "hall_src": round(statistics.median(x["hall_src"] for x in ok), 2),
                   "hall_gen": round(statistics.median(x["hall_gen"] for x in ok), 2), "hall_ratio": round(statistics.median(ratio), 2) if ratio else None,
                   "notched": round(statistics.mean(x["notched"] for x in ok), 2), "sensors": round(statistics.mean(x["passed"] for x in ok), 2),
                   "total": ok[0]["total"], "lost_pct": round(100 * sum(x["lost"] > 0 for x in ok) / len(ok), 1)}
            summary.append(rec)
            print(f"{group:12} {rec['plans']:>3} | {key:4} | {rec['errors']} {rec['infeasible']} | {rec['iou']} | {rec['hall_src']} {rec['hall_gen']} x{rec['hall_ratio']} | {rec['notched']} | {rec['sensors']}/{rec['total']} | {rec['lost_pct']}")
    ranks = rank_table(rows, [k for k, _ in modes], a.topn)
    tag = f"_{a.tag}" if a.tag else ""
    path = REPO / f"data/outputs/diagnostics/source_grid_{datetime.date.today()}{tag}.json"
    with open(path, "w") as fh:
        json.dump({"topn": a.topn, "modes": a.modes, "summary": summary, "ranks": ranks, "rows": rows}, fh)
    print("wrote", path)
    errs = [r for r in rows if "error" in r.get("on", {}) or "error" in r.get("off", {})][:3]
    if errs:
        print("errors:", errs)


if __name__ == "__main__":
    main()
