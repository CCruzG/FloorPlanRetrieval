"""Draw the gate plan sets (2026-11-04): dev (300) and holdout (300) from RPlan.

Fixed seed. The sets do not overlap each other or GATE_IDS in diag_round_trip.py.
Plans retrieved by the 18-plan placement gate are removed from the holdout (and
refilled from the same shuffled order). Writes data/outputs/gates/plan_sets.json.
Run once; the file is the record. Re-running with the same seed gives the same file
as long as the corpus and the retrieval are unchanged.
"""
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

SEED = 20261104
N = 300
OUT = REPO / "data/outputs/gates/plan_sets.json"


def retrieved_ids():
    import diag_placement_topology as d
    from apps.api import server as server_mod
    from packages.floorplan_core.index_store import build_index_store_from_npz
    server_mod.INDEX_STORE = build_index_store_from_npz(
        npz_path=str(REPO / "data/index/rplan_index_v5.npz"),
        data_dir=str(server_mod.RPLAN_DATASET_DIR), logger=server_mod.logger)
    client = server_mod.app.test_client()
    ids = {}
    for entry in d.load_targets():
        b, ep, te, rq = d.entry_to_request_parts(entry)
        if not d._orthogonal(b):
            continue
        plans = client.post("/retrieve", json={"boundary": b, "entrance_point": ep, "requirements": rq,
                                               "topn": d.TOPN, "typed_edges": te}).get_json()["plans"][:d.TOPN]
        ids[entry["name"]] = [str(p.get("id") or p.get("plan_id") or p.get("name")) for p in plans]
    return ids


def main():
    import diag_round_trip as rt
    all_ids = sorted(p.stem for p in rt.DATA.glob("*.json"))
    gate = set(rt.GATE_IDS)
    pool = [i for i in all_ids if i not in gate]
    random.Random(SEED).shuffle(pool)
    dev = pool[:N]
    rest = pool[N:]
    hold_raw = rest[:N]
    ret = retrieved_ids()
    ret_flat = {i for v in ret.values() for i in v}
    overlap = sorted(set(hold_raw) & ret_flat)
    hold = [i for i in rest if i not in ret_flat][:N]
    assert not set(dev) & set(hold) and not (set(dev) | set(hold)) & gate
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "seed": SEED, "corpus_size": len(all_ids), "gate_ids_excluded": list(rt.GATE_IDS),
        "gate18_retrieved": ret, "holdout_overlap_with_gate18_before_removal": overlap,
        "dev": dev, "holdout": hold}, indent=1))
    print("dev", len(dev), "holdout", len(hold), "gate18 retrieved", len(ret_flat),
          "overlap removed", overlap)


if __name__ == "__main__":
    main()
