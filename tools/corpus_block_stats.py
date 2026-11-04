"""Corpus statistics on decomposed room blocks (rect_decompose + part_labels rules v1.2).

Runs over a random sample (--n) or the whole RPlan corpus (--n 0), in parallel.
Outputs (in --out):
  blocks.csv          one row per part of every split room (primary included)
  rooms.csv           one row per split room (pass-through, regions, kitchen_openable, hub counts)
  labels.jsonl.gz     compact per-plan labelling (nodes, roles, connections) for re-indexing
  summary.md          tables: split rates, roles, short side per role, threshold margins,
                      hub test, kitchen relation, v1.1 flags
  hist_short_side.png / hist_area.png
"""
import argparse, csv, gzip, json, multiprocessing as mp, os, random, sys, time
from collections import Counter, defaultdict
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "packages"))
from floorplan_core.rect_decompose import decompose_plan
from floorplan_core.part_labels import label_parts

DATA = "/Users/camilo/Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json"
S = 18.0
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)
BINS = [(0.2, 0.6), (0.6, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 1e9)]
BIN_NAMES = ["0.2-0.6", "0.6-1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", ">3.0"]
ROLES = ["primary", "circulation", "zone", "jog", "alcove", "storage", "nook", "absorbed", "residual"]
# v1.1 thresholds tested for margin sensitivity: (param, value, which parts it separates)
THRESHOLDS = [("absorb_nodoor_max_m", 0.5, "no door"), ("absorb_door_max_m", 0.85, "door"),
              ("storage_max_short_m", 1.0, "no door"), ("nook_min_short_m", 1.5, "no door"),
              ("circ_max_short_m", 2.0, "door")]
MARGIN = 0.1
LABEL_KW = {}  # label_parts overrides (--param), set per worker by _init


def _init(kw):
    LABEL_KW.update(kw)


def bin_of(v):
    for i, (a, b) in enumerate(BINS):
        if a <= v < b:
            return i
    return 0 if v < 0.2 else len(BINS) - 1


def shared(a, b):
    g = a.boundary.intersection(b.boundary)
    return (g.length if g is not None else 0.0) * S


def process(plan):
    """Return per-plan stats. Pure function so it can run in a worker."""
    pid = plan["id"]
    dec = decompose_plan(plan)
    lab = label_parts(plan, dec, **LABEL_KW)
    lab_by = {r["id"]: r for r in lab["rooms"]}
    rlab = {r["id"]: r["label"] for r in dec["rooms"]}
    parts_poly = {(r["id"], i): Polygon(pt["polygon"]) for r in dec["rooms"] for i, pt in enumerate(r["parts"])}
    room_poly = {}
    for r in dec["rooms"]:
        ps = [parts_poly[(r["id"], i)] for i in range(len(r["parts"]))]
        if ps:
            room_poly[r["id"]] = unary_union(ps)

    # node-level connections: node -> set of other rooms reached by door / open edge
    reach = defaultdict(set)
    for c in lab["connections"]:
        a, b = c["a"], c["b"]
        ra, rb = a.split("/")[0], b.split("/")[0]
        if ra != rb:
            reach[a].add(rb)
            if b != "exterior":
                reach[b].add(ra)
    open_pairs = defaultdict(set)
    for e in plan.get("edges", []):
        if e.get("connection") == 2:
            open_pairs[e["u"]].add(e["v"]); open_pairs[e["v"]].add(e["u"])

    rows, room_rows, kit = [], [], []
    for r in dec["rooms"]:
        n = len(r["parts"])
        if n <= 1:
            continue
        rid = r["id"]
        lr = lab_by[rid]
        lp = lr["parts"]
        prim = next(p for p in lp if p["role"] == "primary")
        pp = r["parts"][prim["index"]]
        pnode = prim["node"]
        circ_nodes = [g["node"] for g in lr["regions"]]
        room_reach = set().union(*[reach[nd] for nd in {p["node"] for p in lp}] or [set()])
        prim_reach = reach[pnode]
        circ_reach = set().union(*[reach[nd] for nd in circ_nodes] or [set()])
        room_rows.append(dict(
            plan_id=pid, room_id=rid, room_label=r["label"], n_parts=n,
            primary_short_m=pp["short_m"], primary_long_m=pp["long_m"],
            n_regions=len(lr["regions"]), pass_through=lr["pass_through"],
            n_isolated=len(lr["isolated_circulation"]), kitchen_openable=lr["kitchen_openable"],
            n_dining=sum(bool(p.get("dining_candidate")) for p in lp),
            reach_room=len(room_reach - {"exterior"}), reach_primary=len(prim_reach - {"exterior"}),
            reach_circ=len(circ_reach - {"exterior"}),
            reach_circ_only=len((circ_reach - prim_reach) - {"exterior"}),
            entrance_on=("primary" if "exterior" in prim_reach else "circ" if "exterior" in circ_reach else
                         "other" if "exterior" in room_reach else "none")))
        for i, pt in enumerate(r["parts"]):
            poly = parts_poly[(rid, i)]
            L = lp[i]
            touches = defaultdict(float)
            kc = 0.0
            tl = set()
            for oid, op in room_poly.items():
                if oid == rid:
                    continue
                sl = shared(poly, op)
                if sl >= 0.05:
                    touches[rlab[oid]] += sl
                    tl.add(oid)
                    if rlab[oid] == "Kitchen":
                        kc += sl
            cand = sorted({rlab[o] for o in open_pairs.get(rid, ()) if o in tl and o in rlab})
            # doors as the rules saw them (before absorption moved doors onto absorbers)
            moved_in = {d for q in lp if q["absorbed_into"] == i for d in q.get("door_ids_transferred", [])}
            own_doors = [d for d in L["door_ids"] if d not in moved_in] + L.get("door_ids_transferred", [])
            rows.append(dict(
                plan_id=pid, room_id=rid, room_label=r["label"], part=i, role=L["role"], label=L["label"],
                node=L["node"], absorbed_into=L["absorbed_into"],
                short_m=pt["short_m"], long_m=pt["long_m"], area_m2=poly.area * S * S,
                aspect=pt["long_m"] / pt["short_m"] if pt["short_m"] > 0 else float("inf"),
                has_door=bool(own_doors),
                n_doors_facing=len([d for d in own_doors if d != "entrance"]),
                entrance_facing="entrance" in own_doors, external_contact_m=L["ext_m"],
                touches=json.dumps({k: round(v, 3) for k, v in touches.items()}),
                kitchen_contact_m=kc, open_edge_candidate=json.dumps(cand),
                dining_candidate=bool(L.get("dining_candidate")), proj_cover=L.get("proj_cover")))
        # kitchen relation (as in v0)
        for oid, op in room_poly.items():
            if oid == rid or rlab[oid] != "Kitchen":
                continue
            if not any(shared(parts_poly[(rid, i)], op) >= 0.05 for i in range(n)):
                continue
            kdoors = set()
            for p in lab_by[oid]["parts"]:
                kdoors |= set(p["door_ids"])
            # node level (v1.1): primary node = primary + zone/jog/alcove + parts absorbed into them
            circ_d = set().union(*[set(p["door_ids"]) for p in lp if p["node"] in circ_nodes] or [set()])
            prim_d = set().union(*[set(p["door_ids"]) for p in lp if p["node"] == pnode] or [set()])
            kit.append(dict(plan_id=pid, room_id=rid, kitchen=oid,
                            door_circ=bool(kdoors & circ_d), door_prim=bool(kdoors & prim_d),
                            prim_wall=shared(parts_poly[(rid, prim["index"])], op)))

    compact = {"id": pid,
               "rooms": [{"id": ro["id"], "label": ro["label"], "pass_through": ro["pass_through"],
                          "kitchen_openable": ro["kitchen_openable"],
                          "parts": [[p["index"], p["role"], p["label"], p["node"], round(p["short_m"], 3),
                                     round(p["long_m"], 3)] for p in ro["parts"]],
                          "block_edges": [[e["a"], e["b"], e["kind"], round(e["shared_m"], 3)]
                                          for e in ro["block_edges"]]}
                         for ro in lab["rooms"]],
               "connections": [[c["kind"], c["a"], c["b"]] for c in lab["connections"]],
               "summary": lab["summary"]}
    labels = Counter(rr["label"] for rr in plan["rooms"])
    return dict(rows=rows, room_rows=room_rows, kit=kit, labels=labels, summary=lab["summary"], compact=compact)


def work(path):
    try:
        with open(path) as fh:
            plan = json.load(fh)
        return ("ok", os.path.basename(path), process(plan))
    except Exception as e:
        return ("err", os.path.basename(path), f"{type(e).__name__}: {e}")


def pct(xs, qs=(5, 25, 50, 75, 95)):
    return np.percentile(xs, qs) if len(xs) else [float("nan")] * len(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="sample size; 0 = whole corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--param", action="append", default=[], help="label_parts override, e.g. absorb_nodoor_max_m=0.45")
    a = ap.parse_args()
    kw = {k: float(v) for k, v in (x.split("=", 1) for x in a.param)}
    tag = ("full" if a.n == 0 else f"n{a.n}") + "".join(f"_{k}{v}" for k, v in kw.items())
    out = a.out or f"data/outputs/diagnostics/block_stats_v1.1_{time.strftime('%Y-%m-%d')}_{tag}"
    out = out if os.path.isabs(out) else os.path.join(ROOT, out)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    files = sorted(f for f in os.listdir(DATA) if f.endswith(".json"))
    if a.n:
        random.Random(a.seed).shuffle(files)
        files = files[:a.n]
    paths = [os.path.join(DATA, f) for f in files]

    rows, room_rows, kit = [], [], []
    errs = Counter(); err_ex = {}
    all_rooms = Counter(); sumtot = Counter(); n_ok = 0
    gz = gzip.open(os.path.join(out, "labels.jsonl.gz"), "wt")
    with mp.Pool(a.workers, initializer=_init, initargs=(kw,)) as pool:
        for k, (st, f, res) in enumerate(pool.imap_unordered(work, paths, chunksize=32), 1):
            if st == "err":
                name = res.split(":")[0]
                errs[name] += 1
                err_ex.setdefault(name, f"{f}: {res}")
            else:
                n_ok += 1
                rows += res["rows"]; room_rows += res["room_rows"]; kit += res["kit"]
                all_rooms.update(res["labels"]); sumtot.update(res["summary"])
                gz.write(json.dumps(res["compact"], separators=(",", ":")) + "\n")
            if k % 5000 == 0:
                print(f"  {k}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)
    gz.close()

    cols = ["plan_id", "room_id", "room_label", "part", "role", "label", "node", "absorbed_into", "short_m",
            "long_m", "area_m2", "aspect", "has_door", "n_doors_facing", "entrance_facing", "external_contact_m",
            "touches", "kitchen_contact_m", "open_edge_candidate", "dining_candidate", "proj_cover"]
    rows.sort(key=lambda r: (str(r["plan_id"]), r["room_id"], r["part"]))
    with open(os.path.join(out, "blocks.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: (round(r[c], 4) if isinstance(r[c], float) else r[c]) for c in cols})
    rcols = list(room_rows[0].keys()) if room_rows else []
    room_rows.sort(key=lambda r: (str(r["plan_id"]), r["room_id"]))
    with open(os.path.join(out, "rooms.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rcols); w.writeheader()
        for r in room_rows:
            w.writerow({c: (round(r[c], 4) if isinstance(r[c], float) else r[c]) for c in rcols})

    split_rooms = Counter(r["room_label"] for r in room_rows)
    np_rows = [r for r in rows if r["role"] != "primary"]
    for r in np_rows:
        r["facing"] = r["has_door"]
        r["ext"] = r["external_contact_m"] >= 0.9

    md = [f"# Block stats, rules v1.2 ({'full corpus' if a.n == 0 else f'n={a.n}, seed={a.seed}'})"
          + (f"; overrides {kw}" if kw else "") + "\n"]
    md.append(f"- Plans: {len(files)}; processed OK: {n_ok}; failed: {sum(errs.values())}; "
              f"time {time.time()-t0:.0f}s on {a.workers} workers")
    for k, v in errs.items():
        md.append(f"  - {k}: {v} (e.g. {err_ex[k]})")
    tot = sum(all_rooms.values()); sp = sum(split_rooms.values())
    md.append(f"- Rooms total: {tot}; rooms split: {sp} ({100*sp/max(tot,1):.1f}%)\n")

    md.append("## Split rate by room label\n\n| label | rooms | split | rate % |\n|---|---|---|---|")
    for l, c in sorted(all_rooms.items(), key=lambda kv: -split_rooms[kv[0]] / kv[1]):
        md.append(f"| {l} | {c} | {split_rooms[l]} | {100*split_rooms[l]/c:.1f} |")

    md.append("\n## Labeller summary (summed over plans)\n\n| key | value |\n|---|---|")
    for k in sorted(sumtot):
        md.append(f"| {k} | {sumtot[k]} |")

    md.append("\n## Non-primary parts by role: short side (m) and area (m2)\n")
    md.append("| role | count | short P5 | P25 | median | P75 | P95 | area median | ext >= 0.9 m % |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for role in ROLES[1:]:
        rr = [r for r in np_rows if r["role"] == role]
        if not rr:
            continue
        q = pct([r["short_m"] for r in rr])
        am = np.median([r["area_m2"] for r in rr])
        ex = 100 * sum(r["ext"] for r in rr) / len(rr)
        md.append(f"| {role} | {len(rr)} | " + " | ".join(f"{v:.2f}" for v in q) + f" | {am:.2f} | {ex:.1f} |")
    md.append(f"\n- Zones labelled dining?: {sum(r['dining_candidate'] for r in np_rows)} "
              f"of {sum(r['role']=='zone' for r in np_rows)} zones")

    md.append(f"\n## Threshold margins (non-primary parts within ±{MARGIN} m of each v1.1 threshold)\n")
    md.append("A large share near a threshold means the split there is sensitive to the value.\n")
    md.append("| threshold | value | applies to | parts in group | below (v-m, v) | above [v, v+m) | share in band % |")
    md.append("|---|---|---|---|---|---|---|")
    for name, v, grp in THRESHOLDS:
        g = [r for r in np_rows if r["has_door"] == (grp == "door")]
        lo = sum(v - MARGIN < r["short_m"] < v for r in g)
        hi = sum(v <= r["short_m"] < v + MARGIN for r in g)
        md.append(f"| {name} | {v} | {grp} | {len(g)} | {lo} | {hi} | {100*(lo+hi)/max(len(g),1):.1f} |")

    md.append("\n## Short side x door x external contact >= 0.9 m (non-primary parts)\n")
    md.append("| short side (m) | door y, ext y | door y, ext n | door n, ext y | door n, ext n |\n|---|---|---|---|---|")
    cells = Counter((bin_of(r["short_m"]), r["facing"], r["ext"]) for r in np_rows)
    for b in range(len(BINS)):
        md.append(f"| {BIN_NAMES[b]} | " + " | ".join(str(cells[(b, fd, ex)]) for fd in (True, False)
                                                       for ex in (True, False)) + " |")

    md.append("\n## Hub test: split living rooms\n")
    md.append("Reach = distinct other rooms connected by a door or an open edge (exterior excluded). "
              "Room reach is the v0 hub view; primary reach is what the living room keeps once "
              "circulation regions are their own nodes.\n")
    lv = [r for r in room_rows if r["room_label"] == "LivingRoom"]
    if lv:
        n = len(lv)
        for key, nm in (("reach_room", "room (all nodes)"), ("reach_primary", "primary node"),
                        ("reach_circ", "circulation nodes"), ("reach_circ_only", "circulation only, not primary")):
            xs = [r[key] for r in lv]
            md.append(f"- {nm}: mean {np.mean(xs):.2f}, median {np.median(xs):.0f}")
        drop = [r["reach_room"] - r["reach_primary"] for r in lv]
        md.append(f"- Living rooms where the primary loses >= 1 room to circulation: "
                  f"{sum(d >= 1 for d in drop)} ({100*sum(d >= 1 for d in drop)/n:.1f}%); >= half its reach: "
                  f"{sum(r['reach_primary'] <= r['reach_room']/2 for r in lv if r['reach_room'])} "
                  f"({100*sum(r['reach_primary'] <= r['reach_room']/2 for r in lv if r['reach_room'])/n:.1f}%)")
        ent = Counter(r["entrance_on"] for r in lv)
        md.append("- Entrance lands on: " + ", ".join(f"{k} {v} ({100*v/n:.1f}%)" for k, v in ent.most_common()))
        md.append(f"- Pass-through: {sum(r['pass_through'] for r in lv)} ({100*sum(r['pass_through'] for r in lv)/n:.1f}%); "
                  f"regions per room: " + ", ".join(f"{k}: {v}" for k, v in
                                                     sorted(Counter(r['n_regions'] for r in lv).items())))
        md.append(f"- Isolated circulation regions: {sum(r['n_isolated'] for r in lv)} in "
                  f"{sum(r['n_isolated'] > 0 for r in lv)} rooms")
        q = pct([r["primary_short_m"] for r in lv], (0, 25, 50, 75, 100))
        md.append(f"- Primary short side: min {q[0]:.2f}, Q1 {q[1]:.2f}, median {q[2]:.2f}, Q3 {q[3]:.2f}, max {q[4]:.2f}")

    md.append("\n## Pass-through and kitchen_openable by room label (split rooms)\n")
    md.append("| label | split | pass-through % | kitchen_openable true | false | n/a |\n|---|---|---|---|---|---|")
    for l, c in split_rooms.most_common():
        rr = [r for r in room_rows if r["room_label"] == l]
        ko = Counter(r["kitchen_openable"] for r in rr)
        md.append(f"| {l} | {c} | {100*sum(r['pass_through'] for r in rr)/c:.1f} | {ko[True]} | {ko[False]} | {ko[None]} |")

    md.append("\n## Kitchen relation (split rooms touching a kitchen)\n")
    nk = len(kit)
    if nk:
        dc = sum(k["door_circ"] for k in kit); dp = sum(k["door_prim"] for k in kit)
        md.append(f"- Split-room/kitchen pairs: {nk}")
        md.append(f"- Kitchen door faces a circulation node: {dc} ({100*dc/nk:.1f}%)")
        md.append(f"- Kitchen door faces the primary node (incl. zones): {dp} ({100*dp/nk:.1f}%)")
        md.append(f"- Circulation only: {sum(k['door_circ'] and not k['door_prim'] for k in kit)}; primary only: "
                  f"{sum(k['door_prim'] and not k['door_circ'] for k in kit)}; neither: "
                  f"{sum(not k['door_prim'] and not k['door_circ'] for k in kit)}")
        pw = sum(k["prim_wall"] >= 1.2 for k in kit)
        md.append(f"- Kitchen shares >= 1.2 m wall with primary: {pw} ({100*pw/nk:.1f}%)")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(md) + "\n")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shown = [r for r in np_rows if r["role"] != "absorbed"]
    for key, hi, step, name, xl in (("short_m", 4, 0.05, "hist_short_side.png", "short side (m)"),
                                    ("area_m2", 15, 0.25, "hist_area.png", "area (m2)")):
        bins = np.arange(0, hi + 1e-9, step)
        roles = [ro for ro in ROLES[1:] if ro != "absorbed" and any(r["role"] == ro for r in shown)]
        data = [[min(r[key], hi) for r in shown if r["role"] == ro] for ro in roles]
        plt.figure(figsize=(9, 4.5))
        plt.hist(data, bins=bins, stacked=True, label=roles)
        if key == "short_m":
            for _, v, _g in THRESHOLDS:
                plt.axvline(v, color="k", lw=0.6, ls="--")
        plt.xlabel(xl); plt.ylabel("non-primary, non-absorbed parts"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out, name), dpi=130); plt.close()
    print(f"done in {time.time()-t0:.1f}s; ok={n_ok} err={sum(errs.values())}; out={out}")


if __name__ == "__main__":
    main()
