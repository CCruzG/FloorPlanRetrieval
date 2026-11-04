"""Node-type adjacency over the RPlan corpus, read from the node-graph sidecar (rules v1.2).

Node-level version of corpus_adjacency_stats_v2 (same thresholds, nulls and
bootstrap). Types: a whole room or a primary node keeps its room type
(living, master_bedroom, ...); circulation nodes are "circulation"; storage
blocks "closet"; nooks "nook"; residual nodes are dropped.

Pairs of nodes of the SAME room are excluded (their adjacency is structural:
the decomposition makes them touch). Adjacency = shared boundary >= 0.3 m.
Connection = door if a door edge joins the two nodes, else wall.

Nulls (as v2): 'all' shuffles every node's type within the plan; 'living_fixed'
keeps nodes of type living fixed and shuffles the rest (circulation included).
Lift CI: percentile bootstrap over plans of the observed rate / pooled null rate.
Also writes a hub comparison: living vs circulation against the room-level baseline.

Usage:
    .venv/bin/python tools/corpus_node_adjacency_stats.py [--sidecar PATH] [--baseline JSON]
"""
import argparse, json, multiprocessing as mp, os, random, sys, time
from collections import defaultdict
from itertools import combinations

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "packages"))
from floorplan_core.node_graph import is_stale, load_sidecar

SIDECAR = "/Users/camilo/Documents/Work_FBI/Artefacts/project-44/data/RPlan-node-graph/node_graph_v1.2.jsonl.gz"
BASELINE = os.path.join(ROOT, "data/outputs/diagnostics/corpus_adjacency_2026-11-03_v2-typesfixed.json")
S = 18.0
WALL_MIN_M = 0.3
SEED, SAMPLE_SIZE, N_SHUFFLES, N_BOOTSTRAP = 44, 10_000, 20, 2000
LABEL = {"LivingRoom": "living", "MasterBedroom": "master_bedroom", "Bedroom": "bedroom", "Kitchen": "kitchen",
         "Bathroom": "bathroom", "DiningRoom": "dining", "StudyRoom": "study", "Storage": "storage",
         "Balcony": "balcony", "Entrance": "entrance"}
KIND = {"circ": "circulation", "stor": "closet", "nook": "nook"}


def tp(a, b):
    return (a, b) if a <= b else (b, a)


def parse(g):
    """-> (types, rooms, polys, door_pairs) for the usable nodes of one plan graph."""
    types, rooms, polys = [], [], []
    for n in g["nodes"]:
        if n["kind"] == "residual":
            continue
        t = KIND.get(n["kind"]) or LABEL.get(n["room_label"])
        if t is None:
            continue
        p = unary_union([Polygon(r) for r in n["polygon"] if len(r) >= 3])
        if p.is_empty:
            continue
        types.append(t); rooms.append(n["room_id"]); polys.append(p)
    ids = [n["id"] for n in g["nodes"] if n["kind"] != "residual"]
    doors = {frozenset((e["a"], e["b"])) for e in g["edges"] if e["kind"] == "door"}
    return types, rooms, polys, ids, doors


def plan_pairs(g):
    types, rooms, polys, ids, doors = parse(g)
    idx = {nid: i for i, nid in enumerate(ids)}
    pairs, adj = [], {}
    n = len(types)
    for i, j in combinations(range(n), 2):
        if rooms[i] == rooms[j]:
            continue
        pairs.append((i, j))
        a, b = polys[i], polys[j]
        if not a.envelope.buffer(1e-4).intersects(b.envelope):
            continue
        inter = a.boundary.intersection(b.boundary)
        if not inter.is_empty and inter.length * S >= WALL_MIN_M:
            adj[(i, j)] = 1
    door_ij = set()
    for d in doors:
        u = [idx.get(x) for x in d]
        if len(u) == 2 and None not in u:
            door_ij.add(tuple(sorted(u)))
    return types, pairs, adj, door_ij


def observed(item):
    pid, g = item
    types, pairs, adj, door_ij = plan_pairs(g)
    tot, ad, conn = defaultdict(int), defaultdict(int), defaultdict(lambda: [0, 0])
    for i, j in pairs:
        k = tp(types[i], types[j])
        tot[k] += 1
        if (i, j) in adj:
            ad[k] += 1
            conn[k][0 if (i, j) in door_ij else 1] += 1
    # hub view: distinct other-room node types adjacent to each living / circulation node
    hub = []
    for i, t in enumerate(types):
        if t in ("living", "circulation"):
            nb = [j for (a, b) in adj for j in ((b,) if a == i else (a,) if b == i else ())]
            hub.append((t, len(nb), sum(1 for j in nb if (min(i, j), max(i, j)) in door_ij)))
    return pid, dict(tot), dict(ad), {k: tuple(v) for k, v in conn.items()}, hub


def null(item):
    pid, g = item
    types, pairs, adj, _ = plan_pairs(g)
    out = {}
    for mode in ("all", "living_fixed"):
        rng = random.Random(f"{SEED}-{mode}-{pid}")
        movable = [i for i, t in enumerate(types) if mode == "all" or t != "living"]
        tot, ad = defaultdict(int), defaultdict(int)
        for _ in range(N_SHUFFLES):
            sh = types[:]
            sub = [types[i] for i in movable]
            rng.shuffle(sub)
            for p, i in enumerate(movable):
                sh[i] = sub[p]
            for i, j in pairs:
                k = tp(sh[i], sh[j])
                tot[k] += 1
                ad[k] += (i, j) in adj
        out[mode] = (dict(tot), dict(ad))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", default=SIDECAR)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--tag", default=time.strftime("%Y-%m-%d"))
    a = ap.parse_args()
    t0 = time.time()
    meta, graphs = load_sidecar(a.sidecar)
    if is_stale(meta):
        print("WARNING: sidecar built with other rules/parameters than the current code", file=sys.stderr)
    items = list(graphs.items())
    workers = max(1, (os.cpu_count() or 4) - 1)
    tot, ad = defaultdict(int), defaultdict(int)
    conn = defaultdict(lambda: [0, 0])
    per_plan = defaultdict(list)  # type pair -> [(tot, adj)] per plan
    hub = defaultdict(list)
    with mp.Pool(workers) as pool:
        for pid, t_, a_, c_, h_ in pool.imap_unordered(observed, items, chunksize=200):
            for k, v in t_.items():
                tot[k] += v
                per_plan[k].append((v, a_.get(k, 0)))
            for k, v in a_.items():
                ad[k] += v
            for k, (d, w) in c_.items():
                conn[k][0] += d; conn[k][1] += w
            for t, nb, nd in h_:
                hub[t].append((nb, nd))
        sample = random.Random(SEED).sample(items, min(SAMPLE_SIZE, len(items)))
        ntot = {m: defaultdict(int) for m in ("all", "living_fixed")}
        nad = {m: defaultdict(int) for m in ("all", "living_fixed")}
        for res in pool.imap_unordered(null, sample, chunksize=100):
            for m, (t_, a_) in res.items():
                for k, v in t_.items():
                    ntot[m][k] += v
                for k, v in a_.items():
                    nad[m][k] += v
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for k in tot:
        obs = ad.get(k, 0) / tot[k]
        r = {"type_a": k[0], "type_b": k[1], "plans": len(per_plan[k]), "pairs": tot[k], "adjacent": ad.get(k, 0),
             "obs_rate": obs, "door_rate": conn[k][0] / ad[k] if ad.get(k) else None}
        arr = np.array(per_plan[k], dtype=float)
        boots = None
        if len(arr) >= 5:
            idx = rng.integers(0, len(arr), size=(N_BOOTSTRAP, len(arr)))
            tsum = arr[:, 0][idx].sum(1); asum = arr[:, 1][idx].sum(1)
            boots = np.sort(asum[tsum > 0] / tsum[tsum > 0])
        for m in ("all", "living_fixed"):
            nr = nad[m].get(k, 0) / ntot[m][k] if ntot[m].get(k) else None
            r[f"null_{m}"] = nr
            r[f"lift_{m}"] = obs / nr if nr else None
            if nr and boots is not None and len(boots):
                r[f"ci_{m}"] = [float(boots[int(0.025 * len(boots))] / nr),
                                float(boots[min(len(boots) - 1, int(0.975 * len(boots)))] / nr)]
            else:
                r[f"ci_{m}"] = None
        rows.append(r)
    rows.sort(key=lambda r: -(r["lift_living_fixed"] or -1))

    base = {}
    if os.path.exists(a.baseline):
        for r in json.load(open(a.baseline))["type_pairs"]:
            base[tp(r["type_a"], r["type_b"])] = r
    out = os.path.join(ROOT, "data/outputs/diagnostics", f"corpus_node_adjacency_{a.tag}")
    json.dump({"meta": {"sidecar": a.sidecar, "sidecar_meta": meta, "n_plans": len(items),
                        "wall_min_m": WALL_MIN_M, "seed": SEED, "sample": len(sample), "shuffles": N_SHUFFLES,
                        "bootstrap": N_BOOTSTRAP, "runtime_s": round(time.time() - t0, 1)},
               "type_pairs": rows}, open(out + ".json", "w"), indent=1)

    f = lambda v, d=2: "n/a" if v is None else f"{v:.{d}f}"
    ci = lambda c: "n/a" if not c else f"[{c[0]:.2f}, {c[1]:.2f}]"
    L = [f"# RPlan node-type adjacency (node graph, rules {meta.get('rules_version')})\n",
         f"Plans {len(items)}; runtime {time.time()-t0:.0f}s. Same-room node pairs excluded. "
         f"Adjacency: shared boundary >= {WALL_MIN_M} m. Nulls: 'all' and 'living_fixed' (living nodes fixed, "
         f"circulation shuffled), {len(sample)} plans x {N_SHUFFLES} shuffles, seed {SEED}. "
         f"Room-level baseline: `{os.path.basename(a.baseline)}`.\n",
         "## Hub comparison\n",
         "Distinct other-room nodes adjacent (>= 0.3 m) to each living or circulation node.\n",
         "| node type | nodes | mean adjacent | mean via door |", "|---|---|---|---|"]
    for t in ("living", "circulation"):
        h = np.array(hub[t]) if hub[t] else np.zeros((0, 2))
        if len(h):
            L.append(f"| {t} | {len(h)} | {h[:, 0].mean():.2f} | {h[:, 1].mean():.2f} |")
    L.append("\nLiving and circulation rows against the room-level baseline (obs rate = share of node pairs "
             "that touch; door = share of touching pairs joined by a door):\n")
    L.append("| pair | room-level obs | room-level door | node obs | node door | node lift_lf |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        if "living" in (r["type_a"], r["type_b"]) or "circulation" in (r["type_a"], r["type_b"]):
            k = (r["type_a"], r["type_b"])
            bk = tp(*[("living" if x == "circulation" else x) for x in k])
            b = base.get(bk, {})
            L.append(f"| {k[0]}–{k[1]} | {f(b.get('observed_rate'), 3)} | {f(b.get('door_rate_given_adjacent'))} | "
                     f"{f(r['obs_rate'], 3)} | {f(r['door_rate'])} | {f(r['lift_living_fixed'])} |")
    L.append("\n(For circulation rows the room-level columns show the living-room pair, since circulation was "
             "part of the living room before decomposition.)\n")
    L.append("## All node-type pairs (sorted by lift_living_fixed)\n")
    L.append("| type A | type B | plans | pairs | adj | obs | null_all | lift_all | CI_all | null_lf | lift_lf | CI_lf | door |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['type_a']} | {r['type_b']} | {r['plans']} | {r['pairs']} | {r['adjacent']} | {f(r['obs_rate'], 3)} | "
                 f"{f(r['null_all'], 3)} | {f(r['lift_all'])} | {ci(r['ci_all'])} | {f(r['null_living_fixed'], 3)} | "
                 f"{f(r['lift_living_fixed'])} | {ci(r['ci_living_fixed'])} | {f(r['door_rate'])} |")
    open(out + ".md", "w").write("\n".join(L) + "\n")
    print(f"done in {time.time()-t0:.0f}s; out {out}.md")


if __name__ == "__main__":
    main()
