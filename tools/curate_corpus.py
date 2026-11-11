"""curate_corpus.py — private-access curation of the RPlan corpus (rules c1, 2026-11-11).

Checks every plan's private rooms against the access rule and writes an exclusion
list. It does not change the corpus or the index; retrieval applies the list.

Access basis: the drawn doors. Each door's two rooms are read from its position:
a point just either side of the door's midpoint, and of its quarter points, is
located in a room polygon; the majority pair wins. A door with outside on one side
(front door) gives no pair. Graph edges are not used for doors: RPlan's graph
records 2,251 doors between rooms that touch only at a corner, where the drawn door
opens into a third room (often a master bedroom, i.e. an ensuite). Open connections
(edge connection 2) are kept from the graph.
A private room (bedroom, master bedroom, study, bathroom, storage) is served when
it has a door to a transit room (living, dining, kitchen, hallway, corridor,
entrance, foyer).

An unserved room is classified:
  ensuite       bathroom with a door to a bedroom. Accepted when the plan has
                1 bedroom (1B1B: becomes a studio at refinement) or more than
                1 bathroom. Rejected otherwise (2B1B, 3B1B, ...: the only
                bathroom sits inside a bedroom).
  walk_in       storage with a door to a bedroom. Accepted (walk-in robe);
                --reject-walk-in rejects it.
  no_access     anything else: a bedroom through a bedroom, a study through a
                bedroom, a room with no door to transit. Rejected.

A plan with no bathroom is rejected (no_bathroom).

Bedrooms = master_bedroom + bedroom (study is not counted as a bedroom).
Also reported, not rejected: no_shared_bathroom (more than 1 bathroom, all of
them ensuite, so some bedroom has no bathroom off transit).

Usage (repo root, repo venv):
  .venv/bin/python tools/curate_corpus.py [--data DIR] [--out data/curation] [--reject-walk-in] [--sheet]
Outputs in --out: curation_c1.jsonl (one verdict per plan), excluded_c1.txt (ids),
summary_c1.json (rules, counts by configuration and reason), and with --sheet
rejected_c1.png (sample drawings per reason, for a visual check).
"""
import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from shapely.geometry import Point, Polygon  # noqa: E402
from packages.floorplan_core.topo_signature import _label, PRIVATE, TRANSIT  # noqa: E402

RULES = "c1"
ACCESS = {1, 2}          # door, open
BEDROOM = {"master_bedroom", "bedroom"}


OFFSET = 0.012   # normalised plan units (~0.2 m): past the wall, into each room


def door_pairs(plan):
    """[(room_a, room_b)] for each drawn door with a room on both sides."""
    polys = []
    for r in plan.get("rooms") or []:
        try:
            g = Polygon(r["polygon"])
            if g.is_valid and g.area > 0:
                polys.append((r.get("id"), g))
        except Exception:
            pass
    def room_at(x, y):
        pt = Point(x, y)
        for rid, g in polys:
            if g.contains(pt):
                return rid
        return None
    pairs = []
    for d in plan.get("doors") or []:
        x0, y0, x1, y1 = d["bbox"]
        horiz = (x1 - x0) >= (y1 - y0)
        votes = {}
        for f in (0.25, 0.5, 0.75):
            if horiz:
                x, y = x0 + f * (x1 - x0), (y0 + y1) / 2
                a, b = room_at(x, y - OFFSET), room_at(x, y + OFFSET)
            else:
                x, y = (x0 + x1) / 2, y0 + f * (y1 - y0)
                a, b = room_at(x - OFFSET, y), room_at(x + OFFSET, y)
            if a and b and a != b:
                k = tuple(sorted((a, b)))
                votes[k] = votes.get(k, 0) + 1
        if votes:
            pairs.append(max(votes, key=votes.get))
    return pairs


def classify(plan, reject_walk_in=False):
    rooms = plan.get("rooms") or []
    lab = {r.get("id"): _label(r) for r in rooms}
    adj = {}
    links = door_pairs(plan) + [(e.get("u"), e.get("v")) for e in plan.get("edges") or []
                                if e.get("connection") == 2]
    for u, v in links:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    n_bed = sum(1 for x in lab.values() if x in BEDROOM)
    n_bath = sum(1 for x in lab.values() if x == "bathroom")
    reasons, notes = [], []
    ensuites = 0
    for rid, x in lab.items():
        if x not in PRIVATE:
            continue
        nbs = adj.get(rid, set())
        if any(lab.get(n) in TRANSIT for n in nbs):
            continue
        via_bed = any(lab.get(n) in BEDROOM for n in nbs)
        if x == "bathroom" and via_bed:
            ensuites += 1
            if not (n_bed == 1 or n_bath > 1):
                reasons.append("ensuite")
        elif x == "storage" and via_bed:
            if reject_walk_in:
                reasons.append("walk_in")
            else:
                notes.append("walk_in")
        else:
            reasons.append(f"no_access:{x}")
    if n_bath == 0:
        reasons.append("no_bathroom")
    if n_bath > 1 and ensuites == n_bath:
        notes.append("no_shared_bathroom")
    return {
        "id": str(plan.get("id")), "bedrooms": n_bed, "bathrooms": n_bath,
        "keep": not reasons, "reasons": sorted(set(reasons)), "notes": sorted(set(notes)),
    }


def config_key(v):
    b = v["bathrooms"]
    return f"{v['bedrooms']}B{b}b"


def draw_sheet(data_dir, rows, path, per_reason=6, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPoly
    groups = collections.defaultdict(list)
    for v in rows:
        if not v["keep"]:
            key = "ensuite" if v["reasons"] == ["ensuite"] else \
                  ("no_access" if any(r.startswith("no_access") for r in v["reasons"]) else v["reasons"][0])
            groups[key].append(v)
    rng = random.Random(seed)
    keys = sorted(groups)
    fig, axes = plt.subplots(len(keys), per_reason, figsize=(per_reason * 2.6, len(keys) * 2.9))
    if len(keys) == 1:
        axes = [axes]
    for gi, k in enumerate(keys):
        sample = rng.sample(groups[k], min(per_reason, len(groups[k])))
        for j in range(per_reason):
            ax = axes[gi][j]; ax.set_axis_off(); ax.set_aspect("equal"); ax.invert_yaxis()
            if j >= len(sample):
                continue
            v = sample[j]
            plan = json.load(open(Path(data_dir) / f"{v['id']}.json"))
            bad = set()
            for r in plan.get("rooms") or []:
                poly = r.get("polygon") or []
                if len(poly) < 3:
                    continue
                ax.add_patch(MPoly(poly, closed=True, fill=False, lw=0.6, ec="#161616"))
                cx = sum(p[0] for p in poly) / len(poly); cy = sum(p[1] for p in poly) / len(poly)
                ax.text(cx, cy, _label(r)[:8], fontsize=5, ha="center", va="center")
            for d in plan.get("doors") or []:
                x0, y0, x1, y1 = d["bbox"]
                ax.plot([x0, x1], [y0, y1], lw=2.2, color="#EC008C")
            ax.autoscale_view()
            ax.set_title(f"{v['id']} · {config_key(v)}\n{', '.join(v['reasons'])}", fontsize=6)
        axes[gi][0].text(-0.15, 0.5, k, transform=axes[gi][0].transAxes, rotation=90, va="center", fontsize=9)
    fig.suptitle(f"Rejected by curation {RULES}: random sample per reason (doors in magenta)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json"))
    ap.add_argument("--out", default=str(REPO / "data" / "curation"))
    ap.add_argument("--reject-walk-in", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for fn in sorted(os.listdir(a.data), key=lambda s: (len(s), s)):
        if not fn.endswith(".json"):
            continue
        try:
            plan = json.load(open(os.path.join(a.data, fn)))
        except Exception:
            continue
        rows.append(classify(plan, a.reject_walk_in))

    by_cfg = collections.defaultdict(lambda: collections.Counter())
    reasons, notes = collections.Counter(), collections.Counter()
    for v in rows:
        c = by_cfg[config_key(v)]
        c["plans"] += 1; c["kept" if v["keep"] else "excluded"] += 1
        for r in v["reasons"]:
            reasons[r] += 1
        for n in v["notes"]:
            notes[n] += 1
        if v["reasons"] == ["ensuite"]:
            c["excluded_ensuite_only"] += 1
    excl = [v["id"] for v in rows if not v["keep"]]
    with open(out / f"curation_{RULES}.jsonl", "w") as fh:
        for v in rows:
            fh.write(json.dumps(v) + "\n")
    (out / f"excluded_{RULES}.txt").write_text("\n".join(excl) + "\n")
    summary = {
        "rules": RULES, "reject_walk_in": a.reject_walk_in, "data": a.data,
        "plans": len(rows), "excluded": len(excl), "kept": len(rows) - len(excl),
        "reasons": dict(reasons.most_common()), "notes": dict(notes.most_common()),
        "by_config": {k: dict(v) for k, v in sorted(by_cfg.items(), key=lambda kv: -kv[1]["plans"])},
    }
    (out / f"summary_{RULES}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("plans", "excluded", "kept", "reasons", "notes")}, indent=1))
    print("by configuration (top 12):")
    for k, v in list(summary["by_config"].items())[:12]:
        print(f"  {k:6s} plans {v.get('plans',0):6d}  excluded {v.get('excluded',0):5d}  (ensuite only {v.get('excluded_ensuite_only',0)})")
    if a.sheet:
        draw_sheet(a.data, rows, out / f"rejected_{RULES}.png")
        print("sheet:", out / f"rejected_{RULES}.png")


if __name__ == "__main__":
    main()
