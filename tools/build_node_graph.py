"""Build the node-graph sidecar over the RPlan corpus (rules v1.2).

Output: one gzipped JSON line per plan, keyed by plan id, after a header line
with the rules version and effective parameters (see floorplan_core.node_graph).
Default output: <data>/RPlan-node-graph/node_graph_v<rules>.jsonl.gz
"""
import argparse, gzip, json, multiprocessing as mp, os, sys, time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "packages"))
from floorplan_core.node_graph import RULES_VERSION, build_node_graph, sidecar_meta

DATA = "/Users/camilo/Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json"
OUT_DIR = "/Users/camilo/Documents/Work_FBI/Artefacts/project-44/data/RPlan-node-graph"


def work(path):
    try:
        with open(path) as fh:
            plan = json.load(fh)
        return ("ok", os.path.basename(path), build_node_graph(plan))
    except Exception as e:
        return ("err", os.path.basename(path), f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="first n plans (sorted); 0 = whole corpus")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, f"node_graph_v{RULES_VERSION}.jsonl.gz"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    files = sorted(f for f in os.listdir(DATA) if f.endswith(".json"))
    if a.n:
        files = files[:a.n]
    t0 = time.time()
    errs, kinds, ekinds = Counter(), Counter(), Counter()
    graphs = {}
    with mp.Pool(a.workers) as pool:
        for st, f, res in pool.imap_unordered(work, [os.path.join(DATA, f) for f in files], chunksize=32):
            if st == "err":
                errs[res.split(":")[0]] += 1
                print("ERR", f, res, file=sys.stderr)
                continue
            graphs[str(res["id"])] = res
            kinds.update(n["kind"] for n in res["nodes"])
            ekinds.update(e["kind"] for e in res["edges"])
    tmp = a.out + ".tmp"
    with gzip.open(tmp, "wt") as fh:
        fh.write(json.dumps(sidecar_meta(source=DATA)) + "\n")
        for pid in sorted(graphs, key=lambda s: (len(s), s)):
            fh.write(json.dumps(graphs[pid], separators=(",", ":")) + "\n")
    os.replace(tmp, a.out)
    print(f"done in {time.time()-t0:.0f}s; plans {len(graphs)}; errors {sum(errs.values())} {dict(errs)}")
    print(f"nodes {sum(kinds.values())} {dict(kinds)}")
    print(f"edges {sum(ekinds.values())} {dict(ekinds)}")
    print(f"out {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
