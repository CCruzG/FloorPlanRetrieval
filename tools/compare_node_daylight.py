"""Compare retrieval with room-level vs node-level daylight scoring (2026-11-03).

Runs run_retrieval in-process for each boundary in test-boundaries/boundary_library.json,
once with room-level daylight and once with the node-graph sidecar, over the full
pool (topn = pool_n = 60). Reports top-5 overlap, rank changes and how many pool
candidates changed topo score.

Usage: .venv/bin/python tools/compare_node_daylight.py
"""
import json, os, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from packages.floorplan_core.index_store import build_index_store_from_npz
from packages.floorplan_core.node_graph import is_stale, load_sidecar
from packages.floorplan_core.retrieval import run_retrieval

LIB = REPO_ROOT.parent / "test-boundaries" / "boundary_library.json"
SIDECAR = Path.home() / "Documents/Work_FBI/Artefacts/project-44/data/RPlan-node-graph/node_graph_v1.2.jsonl.gz"
DATA = Path.home() / "Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json"
POOL, TOP = 60, 5


class _Log:
    def warning(self, *a, **k): pass
    info = debug = error = exception = warning


def run(store, entry, graphs):
    boundary = [[v["x"], v["y"]] for v in entry["vertices"]]
    ent = [entry["entrance"]["x"], entry["entrance"]["y"]] if entry.get("entrance") else None
    typed = [{"edge_index": i, "type": t} for i, t in enumerate(entry.get("edgeTypes") or []) if t]
    out = run_retrieval(store, boundary, entrance_point=ent, requirements=entry.get("requirements") or {},
                        topn=POOL, pool_n=POOL, typed_edges=typed, node_graphs=graphs, logger=_Log())
    return [(Path(p["retrieval"]["source_id"]).stem, p["retrieval"]["topo"]["score"],
             p["retrieval"]["topo"].get("components", {}).get("daylight_fit",
                                                               p["retrieval"]["topo"].get("daylight_fit")))
            for p in out["plans"]]


def main():
    t0 = time.time()
    store = build_index_store_from_npz(npz_path=str(REPO_ROOT / "data/index/rplan_index_v5.npz"),
                                       data_dir=str(DATA), logger=_Log())
    meta, graphs = load_sidecar(str(SIDECAR))
    print(f"loaded index + sidecar in {time.time()-t0:.0f}s; sidecar stale={is_stale(meta)}", file=sys.stderr)
    lib = json.loads(LIB.read_text())
    L = ["# Retrieval: room-level vs node-level daylight\n",
         f"Pool {POOL} by shape, re-ranked by topo score; top {TOP} compared. Sidecar `{SIDECAR.name}`.\n",
         "| boundary | top-5 overlap | top-5 room | top-5 node | pool scores changed | mean |Δscore| | max rank move |",
         "|---|---|---|---|---|---|---|"]
    detail = []
    for e in lib:
        a = run(store, e, None)
        b = run(store, e, graphs)
        ra = {pid: i for i, (pid, _, _) in enumerate(a)}
        rb = {pid: i for i, (pid, _, _) in enumerate(b)}
        sa = {pid: s for pid, s, _ in a}
        sb = {pid: s for pid, s, _ in b}
        common = set(sa) & set(sb)
        changed = [pid for pid in common if abs(sa[pid] - sb[pid]) > 1e-9]
        dmean = sum(abs(sa[p] - sb[p]) for p in common) / max(len(common), 1)
        moves = [abs(ra[p] - rb[p]) for p in common]
        ta = [pid for pid, _, _ in a[:TOP]]
        tb = [pid for pid, _, _ in b[:TOP]]
        L.append(f"| {e['name']} | {len(set(ta) & set(tb))}/{TOP} | {', '.join(ta)} | {', '.join(tb)} | "
                 f"{len(changed)}/{len(common)} | {dmean:.3f} | {max(moves) if moves else 0} |")
        detail.append({"boundary": e["name"], "room": a, "node": b})
    out = REPO_ROOT / "data/outputs/diagnostics" / f"compare_node_daylight_{time.strftime('%Y-%m-%d')}"
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    out.with_suffix(".json").write_text(json.dumps(detail, indent=1))
    print("\n".join(L))
    print(f"done in {time.time()-t0:.0f}s; out {out}.md", file=sys.stderr)


if __name__ == "__main__":
    main()
