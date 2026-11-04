"""
gate_set_precedence.py — gate the SetPrecedence move (Stage 2).

For the fixed test boundaries (diag_min_size.TARGET_NAMES, top 3 each), replays
retrieve -> generate -> /moves/propose through the Flask app in-process, then for
every SetPrecedence candidate re-runs /generate with that edit applied and checks
the move's predicted failing rooms (a rollout) equal the regenerated ones, sensor
by sensor. Also counts what the move buys: candidates realised by step 2c, and
those that reduce min_room_size (D24) failures without breaking another sensor.

Read-only. Output: data/outputs/diagnostics/gate_set_precedence_<date>.json.
"""
import json
import logging
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import diag_min_size as diag  # noqa: E402  (shared targets + request parts)

OUT_PATH = (REPO_ROOT / "data" / "outputs" / "diagnostics"
            / f"gate_set_precedence_{date.today().isoformat()}.json")


def _failing(checks):
    return {c["sensor"]: sorted(str(r) for r in c.get("failing_rooms") or [])
            for c in checks if not c.get("ok", True)}


def main():
    logging.disable(logging.CRITICAL)
    from apps.api import server as S
    from packages.floorplan_core.index_store import build_index_store_from_npz
    S.INDEX_STORE = build_index_store_from_npz(
        npz_path=str(REPO_ROOT / "data" / "index" / "rplan_index_v5.npz"),
        data_dir=str(S.RPLAN_DATASET_DIR), logger=S.logger)
    client = S.app.test_client()

    out = {"generated_at": date.today().isoformat(), "plans": []}
    n = {"plans": 0, "candidates": 0, "realised": 0, "predicted_eq_actual": 0,
         "mismatches": 0, "reduce_d24_no_break": 0}
    for entry in diag.load_targets():
        boundary, ep, te, rq = diag.entry_to_request_parts(entry)
        plans = client.post("/retrieve", json=dict(
            boundary=boundary, entrance_point=ep, requirements=rq,
            topn=diag.TOPN, typed_edges=te)).get_json().get("plans") or []
        for rank, plan in enumerate(plans[:diag.TOPN], start=1):
            base_body = dict(fp=plan, boundary=boundary, entrance_point=ep,
                             typed_edges=te, requirements=rq,
                             mm_per_unit=diag.MM_PER_UNIT)
            prop = client.post("/moves/propose", json=base_body).get_json()
            base_fail = _failing((prop.get("sensors") or {}).get("checks") or [])
            n["plans"] += 1
            rec = {"boundary": entry["name"], "rank": rank,
                   "base_failing": base_fail, "candidates": []}
            for cand in prop.get("candidates") or []:
                edit = cand["edit"]
                if edit.get("move") != "SetPrecedence":
                    continue
                n["candidates"] += 1
                body = dict(base_body, precedence=[
                    {"winner": edit["winner"], "loser": edit["loser"]}])
                gen = client.post("/generate", json=body).get_json()
                actual = _failing((gen.get("sensors") or {}).get("checks") or [])
                predicted = {k: sorted(str(r) for r in v)
                             for k, v in (cand["predicted"].get("failing") or {}).items()}
                same = predicted == actual
                n["predicted_eq_actual" if same else "mismatches"] += 1
                realised = bool(cand.get("realised", True))
                n["realised"] += int(realised)
                d24_before = len(base_fail.get("min_room_size", []))
                d24_after = len(actual.get("min_room_size", []))
                broke = any(set(actual.get(s, [])) - set(base_fail.get(s, []))
                            for s in actual if s != "min_room_size")
                gain = d24_after < d24_before and not broke
                n["reduce_d24_no_break"] += int(gain)
                rec["candidates"].append({
                    "winner": edit["winner"], "loser": edit["loser"],
                    "realised": realised, "predicted_eq_actual": same,
                    "d24": [d24_before, d24_after], "breaks_other": broke,
                    "actual_failing": actual,
                    **({} if same else {"predicted_failing": predicted})})
            out["plans"].append(rec)
    out["counts"] = n
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH}", file=sys.stderr)
    print(json.dumps(n, indent=2))


if __name__ == "__main__":
    main()
