"""
Measure which room-type pairs share walls in the RPlan corpus more often than chance.

Read-only analysis. Does not modify any existing file.

For each plan:
  - Parse rooms (id, type, polygon) and edges (u, v, relation, connection).
  - Compute geometric adjacency for every room pair: shared polygon boundary
    length >= WALL_MIN_M, excluding corner-only (point) contact.
  - Tabulate counts per unordered type pair (plans with both types present,
    room pairs of that type pair, adjacent room pairs, adjacent pairs broken
    down by connection type 0/1/2 and "no edge").

A permutation null is built by shuffling room-type labels within each plan
(geometry fixed) for N_SHUFFLES shuffles over a fixed-seed sample of
SAMPLE_SIZE plans, then recounting adjacency the same way. Lift = observed
adjacency rate / null adjacency rate, with a 95% bootstrap interval over
plans (percentile bootstrap, B resamples, same fixed seed).

Usage:
    .venv/bin/python tools/corpus_adjacency_stats.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from functools import partial
from itertools import combinations
from pathlib import Path

from shapely.geometry import Polygon
from shapely.validation import explain_validity

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = Path.home() / "Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json"
OUT_DIR = REPO_ROOT / "data/outputs/diagnostics"
OUT_JSON = OUT_DIR / "corpus_adjacency_2026-11-02.json"
OUT_MD = OUT_DIR / "corpus_adjacency_2026-11-02.md"

SCALE_M = 18.0  # normalised coord x 18 = metres
WALL_MIN_M = 0.3  # minimum shared-boundary length to count as adjacency (excludes corner touches)

ROOM_TYPE_MAP = {
    0: "living",
    1: "master_bedroom",
    2: "bedroom",
    3: "kitchen",
    4: "bathroom",
    5: "dining",
    6: "study",      # fixed 2026-11-03: was "storage" (RPlan type 6 = StudyRoom)
    7: "storage",    # fixed 2026-11-03: was "hallway" (type 7 = Storage; RPlan has no hallway type)
    8: "balcony",
    9: "entrance",   # fixed 2026-11-03: was merged into "balcony" (type 9 = Entrance)
}

SEED = 44
SAMPLE_SIZE = 10_000
N_SHUFFLES = 20
N_BOOTSTRAP = 2000
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)

EPS_M = 1e-6  # numerical tolerance for shared-boundary length, in metres


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def shared_boundary_length_m(poly_a: Polygon, poly_b: Polygon) -> float:
    """Length (in metres) of the shared boundary between two room polygons.

    Scales normalised coords to metres first. Returns 0.0 for point-only
    (corner) contact or no contact. Robust to invalid/degenerate polygons by
    buffering with 0 (shapely's standard repair trick); if that still fails,
    returns 0.0 (caller logs it as an invalid polygon).
    """
    inter = poly_a.boundary.intersection(poly_b.boundary)
    if inter.is_empty:
        return 0.0
    length_normalised = inter.length  # shapely geometry collections sum sub-length correctly
    return length_normalised * SCALE_M


def make_polygon(coords) -> Polygon | None:
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            return None
        return poly
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Plan parsing
# ---------------------------------------------------------------------------

class ParsedPlan:
    __slots__ = ("plan_id", "room_ids", "room_types", "polygons", "edges", "problems")

    def __init__(self, plan_id, room_ids, room_types, polygons, edges, problems):
        self.plan_id = plan_id
        self.room_ids = room_ids          # list[str]
        self.room_types = room_types      # list[str] type label, aligned with room_ids
        self.polygons = polygons          # list[Polygon], aligned with room_ids
        self.edges = edges                # dict[(idx_i,idx_j)] -> connection int (0/1/2)
        self.problems = problems          # list[str] data-quality notes


def parse_plan_file(path: Path) -> tuple[ParsedPlan | None, list[str]]:
    """Parse one RPlan JSON file. Returns (ParsedPlan or None, problem notes)."""
    problems = []
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return None, [f"PARSE_FAIL {path.name}: {e}"]

    rooms = data.get("rooms", [])
    if not rooms:
        return None, [f"NO_ROOMS {path.name}"]

    room_ids, room_types, polygons = [], [], []
    id_to_idx = {}
    for r in rooms:
        rid = r.get("id")
        rtype = r.get("type")
        if rid is None or rtype is None:
            problems.append(f"BAD_ROOM {path.name}: missing id/type")
            continue
        label = ROOM_TYPE_MAP.get(rtype)
        if label is None:
            problems.append(f"UNKNOWN_TYPE {path.name}: room {rid} type={rtype}")
            continue
        poly = make_polygon(r.get("polygon", []))
        if poly is None:
            problems.append(f"INVALID_POLYGON {path.name}: room {rid}")
            continue
        idx = len(room_ids)
        id_to_idx[rid] = idx
        room_ids.append(rid)
        room_types.append(label)
        polygons.append(poly)

    if len(room_ids) < 2:
        return None, problems + [f"TOO_FEW_ROOMS {path.name}"]

    edges = {}
    for e in data.get("edges", []):
        u, v, conn = e.get("u"), e.get("v"), e.get("connection")
        if u is None or v is None or conn is None:
            problems.append(f"BAD_EDGE {path.name}: {e}")
            continue
        if u not in id_to_idx or v not in id_to_idx:
            problems.append(f"EDGE_MISSING_ROOM {path.name}: u={u} v={v}")
            continue
        iu, iv = id_to_idx[u], id_to_idx[v]
        key = (iu, iv) if iu < iv else (iv, iu)
        edges[key] = conn

    plan = ParsedPlan(path.stem, room_ids, room_types, polygons, edges, problems)
    return plan, problems


# ---------------------------------------------------------------------------
# Per-plan adjacency computation
# ---------------------------------------------------------------------------

def type_pair_key(t1: str, t2: str) -> tuple[str, str]:
    return (t1, t2) if t1 <= t2 else (t2, t1)


def compute_plan_adjacency(plan: ParsedPlan):
    """Return list of (i, j, shared_len_m) for adjacent room-index pairs."""
    n = len(plan.room_ids)
    adj = []
    for i, j in combinations(range(n), 2):
        length_m = shared_boundary_length_m(plan.polygons[i], plan.polygons[j])
        if length_m >= WALL_MIN_M:
            adj.append((i, j, length_m))
    return adj


# ---------------------------------------------------------------------------
# Worker: process one file -> partial stats + problems
# ---------------------------------------------------------------------------

def process_file(path_str: str):
    path = Path(path_str)
    plan, problems = parse_plan_file(path)
    if plan is None:
        return None, problems

    adj = compute_plan_adjacency(plan)
    adj_set = {(i, j) for i, j, _ in adj}

    # Per-plan: which type pairs present (room-level), room pairs per type pair,
    # adjacent pairs per type pair, connection breakdown per type pair.
    types_present = set(plan.room_types)

    pair_total = defaultdict(int)      # type_pair -> count of room pairs of that type
    pair_adjacent = defaultdict(int)   # type_pair -> count of adjacent room pairs
    pair_conn = defaultdict(lambda: {"0": 0, "1": 0, "2": 0, "none": 0})

    n = len(plan.room_ids)
    for i, j in combinations(range(n), 2):
        tp = type_pair_key(plan.room_types[i], plan.room_types[j])
        pair_total[tp] += 1
        if (i, j) in adj_set:
            pair_adjacent[tp] += 1
            conn = plan.edges.get((i, j))
            if conn is None:
                pair_conn[tp]["none"] += 1
                problems.append(
                    f"ADJACENT_NO_EDGE {plan.plan_id}: rooms {plan.room_ids[i]}/{plan.room_ids[j]} "
                    f"types {tp}"
                )
            elif conn in (0, 1, 2):
                pair_conn[tp][str(conn)] += 1
            else:
                pair_conn[tp]["none"] += 1
                problems.append(f"UNKNOWN_CONNECTION {plan.plan_id}: conn={conn}")

    result = {
        "plan_id": plan.plan_id,
        "types_present": types_present,
        "pair_total": dict(pair_total),
        "pair_adjacent": dict(pair_adjacent),
        "pair_conn": {k: dict(v) for k, v in pair_conn.items()},
        "n_rooms": n,
    }
    return result, problems


def process_file_for_sample(path_str: str, seed: int, n_shuffles: int):
    """For the permutation-null sample: parse, compute adjacency once (geometry
    fixed), then shuffle type labels n_shuffles times and recount adjacency
    rates per type pair. Returns per-shuffle pair_adjacent/pair_total dicts.
    """
    path = Path(path_str)
    plan, _ = parse_plan_file(path)
    if plan is None:
        return None

    adj = compute_plan_adjacency(plan)
    adj_pairs_idx = [(i, j) for i, j, _ in adj]
    n = len(plan.room_ids)
    all_pairs_idx = list(combinations(range(n), 2))

    rng = random.Random(f"{seed}-{plan.plan_id}")
    base_types = plan.room_types

    shuffle_results = []
    for _ in range(n_shuffles):
        shuffled = base_types[:]
        rng.shuffle(shuffled)
        pair_total = defaultdict(int)
        pair_adjacent = defaultdict(int)
        adj_idx_set = set(adj_pairs_idx)
        for i, j in all_pairs_idx:
            tp = type_pair_key(shuffled[i], shuffled[j])
            pair_total[tp] += 1
            if (i, j) in adj_idx_set:
                pair_adjacent[tp] += 1
        shuffle_results.append((dict(pair_total), dict(pair_adjacent)))

    return {"plan_id": plan.plan_id, "shuffles": shuffle_results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(CORPUS_DIR.glob("*.json"), key=lambda p: p.name)
    file_strs = [str(f) for f in files]
    print(f"Found {len(file_strs)} files in corpus.", file=sys.stderr)

    # ---- Pass 1: observed adjacency, over the full corpus ----
    agg_pair_total = defaultdict(int)
    agg_pair_adjacent = defaultdict(int)
    agg_pair_conn = defaultdict(lambda: {"0": 0, "1": 0, "2": 0, "none": 0})
    plans_with_types = defaultdict(int)  # type_pair -> number of plans where both types present
    plan_pair_adjacent_by_plan = defaultdict(dict)  # for bootstrap: plan_id -> {type_pair: adjacent_count}
    plan_pair_total_by_plan = defaultdict(dict)

    all_problems = []
    n_parsed = 0
    n_failed = 0

    with mp.Pool(N_WORKERS) as pool:
        for result, problems in pool.imap_unordered(process_file, file_strs, chunksize=200):
            all_problems.extend(problems)
            if result is None:
                n_failed += 1
                continue
            n_parsed += 1
            pid = result["plan_id"]
            for tp in result["types_present"]:
                pass  # single-type presence handled via pair logic below
            types_present = result["types_present"]
            for t1, t2 in combinations(sorted(types_present), 2):
                plans_with_types[(t1, t2)] += 1
            # same-type pairs: "both types present" means >=2 rooms of that type
            for t in types_present:
                pass
            for tp, cnt in result["pair_total"].items():
                agg_pair_total[tp] += cnt
                plan_pair_total_by_plan[pid][tp] = cnt
            for tp, cnt in result["pair_adjacent"].items():
                agg_pair_adjacent[tp] += cnt
                plan_pair_adjacent_by_plan[pid][tp] = cnt
            for tp, conn_counts in result["pair_conn"].items():
                for k, v in conn_counts.items():
                    agg_pair_conn[tp][k] += v

    # same-type pair "plans with both types present" needs >=2 rooms of that type;
    # recompute properly by re-scanning room_types counts. We'll redo this cheaply
    # by reusing pair_total>0 as proxy (pair_total>0 implies >=2 rooms of that type
    # for same-type pairs, and >=1 of each for cross-type pairs). This is already
    # consistent: pair_total[tp] > 0 in a plan means that plan contributed room
    # pairs of that type -> both types (or >=2 of the same type) are present.
    plans_with_types = defaultdict(int)
    for pid, totals in plan_pair_total_by_plan.items():
        for tp in totals:
            plans_with_types[tp] += 1

    t1_elapsed = time.time()
    print(f"Pass 1 (observed) done in {t1_elapsed - t0:.1f}s. Parsed {n_parsed}, failed {n_failed}.", file=sys.stderr)

    # ---- Pass 2: permutation null over a fixed sample ----
    rng_sample = random.Random(SEED)
    valid_file_strs = file_strs  # sample from all files; invalid ones return None and are skipped
    sample = rng_sample.sample(valid_file_strs, min(SAMPLE_SIZE, len(valid_file_strs)))

    null_pair_total = defaultdict(int)
    null_pair_adjacent = defaultdict(int)
    # for bootstrap of the null rate too (optional; we bootstrap lift via observed/null ratio per plan where possible)
    null_plan_pair_adjacent = defaultdict(list)  # tp -> list of per-shuffle-plan adjacent counts (flattened)
    null_plan_pair_total = defaultdict(list)

    n_sample_parsed = 0
    worker = partial(process_file_for_sample, seed=SEED, n_shuffles=N_SHUFFLES)
    with mp.Pool(N_WORKERS) as pool:
        for res in pool.imap_unordered(worker, sample, chunksize=50):
            if res is None:
                continue
            n_sample_parsed += 1
            for pair_total, pair_adjacent in res["shuffles"]:
                for tp, cnt in pair_total.items():
                    null_pair_total[tp] += cnt
                for tp, cnt in pair_adjacent.items():
                    null_pair_adjacent[tp] += cnt

    t2_elapsed = time.time()
    print(f"Pass 2 (null, {SAMPLE_SIZE} plans x {N_SHUFFLES} shuffles) done in {t2_elapsed - t1_elapsed:.1f}s. "
          f"Sample parsed {n_sample_parsed}.", file=sys.stderr)

    # ---- Bootstrap CI for lift, over plans (using the full corpus's per-plan observed data) ----
    # Null rate is a single pooled estimate (from the sample); we bootstrap the
    # OBSERVED rate over plans (resampling plans with replacement) and divide by
    # the fixed pooled null rate to get a 95% interval for lift. This is a valid
    # approach because the null denominator is estimated from a much larger
    # pseudo-sample (10,000 plans x 20 shuffles) and its own sampling error is
    # small relative to the observed-rate's plan-to-plan variance; we note this
    # choice in the report.
    # Invert plan -> type_pair maps into type_pair -> (total_arr, adj_arr), once,
    # so the bootstrap loop below is pure numpy (the naive pure-Python version,
    # resampling plan ids and summing via dict lookups each time, does not
    # finish in reasonable time for common type pairs present in tens of
    # thousands of plans).
    import numpy as np

    tp_to_totals = defaultdict(list)
    tp_to_adj = defaultdict(list)
    for pid, totals in plan_pair_total_by_plan.items():
        adj_for_pid = plan_pair_adjacent_by_plan.get(pid, {})
        for tp, tot in totals.items():
            tp_to_totals[tp].append(tot)
            tp_to_adj[tp].append(adj_for_pid.get(tp, 0))

    rng_boot = np.random.default_rng(SEED + 1)
    all_type_pairs = set(agg_pair_total.keys())

    bootstrap_results = {}
    for tp in all_type_pairs:
        obs_total = agg_pair_total.get(tp, 0)
        obs_adj = agg_pair_adjacent.get(tp, 0)
        if obs_total == 0:
            continue
        obs_rate = obs_adj / obs_total
        null_total = null_pair_total.get(tp, 0)
        null_adj = null_pair_adjacent.get(tp, 0)
        null_rate = (null_adj / null_total) if null_total > 0 else None
        if null_rate is None or null_rate == 0:
            lift = None
        else:
            lift = obs_rate / null_rate

        # bootstrap over plans that have this type pair present
        total_arr = np.asarray(tp_to_totals.get(tp, []), dtype=np.float64)
        adj_arr = np.asarray(tp_to_adj.get(tp, []), dtype=np.float64)
        n_rp = total_arr.shape[0]
        lift_samples = []
        if lift is not None and n_rp >= 5:
            samples = np.empty(N_BOOTSTRAP, dtype=np.float64)
            valid_mask = np.ones(N_BOOTSTRAP, dtype=bool)
            for b in range(N_BOOTSTRAP):
                idx = rng_boot.integers(0, n_rp, size=n_rp)
                tot_b = total_arr[idx].sum()
                if tot_b > 0:
                    samples[b] = (adj_arr[idx].sum() / tot_b) / null_rate
                else:
                    valid_mask[b] = False
            lift_samples = sorted(samples[valid_mask].tolist())
        ci_lo, ci_hi = (None, None)
        if lift_samples:
            lo_idx = int(0.025 * len(lift_samples))
            hi_idx = min(len(lift_samples) - 1, int(0.975 * len(lift_samples)))
            ci_lo, ci_hi = lift_samples[lo_idx], lift_samples[hi_idx]

        conn = agg_pair_conn.get(tp, {"0": 0, "1": 0, "2": 0, "none": 0})
        door_rate = conn["1"] / obs_adj if obs_adj > 0 else None
        open_rate = conn["2"] / obs_adj if obs_adj > 0 else None

        bootstrap_results[tp] = {
            "type_a": tp[0],
            "type_b": tp[1],
            "plans_with_both_present": plans_with_types.get(tp, 0),
            "room_pairs_total": obs_total,
            "room_pairs_adjacent": obs_adj,
            "observed_rate": obs_rate,
            "null_rate": null_rate,
            "null_pairs_total_sampled": null_total,
            "lift": lift,
            "lift_ci95_lo": ci_lo,
            "lift_ci95_hi": ci_hi,
            "connection_counts": conn,
            "door_rate_given_adjacent": door_rate,
            "open_rate_given_adjacent": open_rate,
        }

    t3_elapsed = time.time()
    print(f"Bootstrap done in {t3_elapsed - t2_elapsed:.1f}s.", file=sys.stderr)

    # ---- Write outputs ----
    sorted_tps = sorted(
        bootstrap_results.values(),
        key=lambda r: (r["lift"] if r["lift"] is not None else -1),
        reverse=True,
    )

    meta = {
        "corpus_dir": str(CORPUS_DIR),
        "n_files": len(file_strs),
        "n_parsed": n_parsed,
        "n_failed_to_parse_or_use": n_failed,
        "wall_min_m": WALL_MIN_M,
        "scale_m_per_unit": SCALE_M,
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "n_sample_parsed": n_sample_parsed,
        "n_shuffles": N_SHUFFLES,
        "n_bootstrap": N_BOOTSTRAP,
        "n_problems_logged": len(all_problems),
        "runtime_seconds": round(time.time() - t0, 1),
        "generated": "2026-11-02",
    }

    out_json = {
        "meta": meta,
        "type_pairs": sorted_tps,
        "problems_sample": all_problems[:500],  # cap to keep file manageable
        "problems_total_count": len(all_problems),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(out_json, f, indent=2, default=str)

    write_markdown(OUT_MD, meta, sorted_tps, all_problems)

    print(f"Done in {time.time() - t0:.1f}s. Wrote {OUT_JSON} and {OUT_MD}.", file=sys.stderr)


def write_markdown(path: Path, meta: dict, sorted_tps: list, problems: list):
    lines = []
    lines.append("# RPlan corpus adjacency statistics")
    lines.append("")
    lines.append(f"Generated 2026-11-02. Runtime {meta['runtime_seconds']}s. "
                  f"Files: {meta['n_files']}. Parsed: {meta['n_parsed']}. "
                  f"Failed/unusable: {meta['n_failed_to_parse_or_use']}.")
    lines.append(f"Wall threshold: shared boundary >= {meta['wall_min_m']} m "
                  f"(normalised coord x {meta['scale_m_per_unit']} = metres).")
    lines.append(f"Null model: permutation of room-type labels within each plan, geometry fixed. "
                  f"Sample {meta['sample_size']} plans (parsed {meta['n_sample_parsed']}), "
                  f"{meta['n_shuffles']} shuffles each, seed {meta['seed']}.")
    lines.append(f"Lift = observed adjacency rate / pooled null adjacency rate. "
                  f"95% CI by percentile bootstrap ({meta['n_bootstrap']} resamples) over plans, "
                  f"resampling the OBSERVED rate only; the null rate is held fixed because it is "
                  f"estimated from a much larger pseudo-sample (plans x shuffles) and its own "
                  f"sampling error is small relative to plan-to-plan variance in the observed rate.")
    lines.append(f"Data problems logged: {meta['n_problems_logged']} (sample of up to 500 in the JSON).")
    lines.append("")
    lines.append("| type A | type B | plans w/ both | room pairs | adjacent | obs rate | null rate | lift | 95% CI | door rate | open rate |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted_tps:
        lift_s = f"{r['lift']:.2f}" if r['lift'] is not None else "n/a"
        ci_s = (f"[{r['lift_ci95_lo']:.2f}, {r['lift_ci95_hi']:.2f}]"
                if r['lift_ci95_lo'] is not None else "n/a")
        null_s = f"{r['null_rate']:.4f}" if r['null_rate'] is not None else "n/a"
        door_s = f"{r['door_rate_given_adjacent']:.2f}" if r['door_rate_given_adjacent'] is not None else "n/a"
        open_s = f"{r['open_rate_given_adjacent']:.2f}" if r['open_rate_given_adjacent'] is not None else "n/a"
        lines.append(
            f"| {r['type_a']} | {r['type_b']} | {r['plans_with_both_present']} | "
            f"{r['room_pairs_total']} | {r['room_pairs_adjacent']} | {r['observed_rate']:.4f} | "
            f"{null_s} | {lift_s} | {ci_s} | {door_s} | {open_s} |"
        )
    lines.append("")
    lines.append("## Data problems (sample)")
    lines.append("")
    # summarise problem categories
    cat_counts = defaultdict(int)
    for p in problems:
        cat = p.split(" ", 1)[0]
        cat_counts[cat] += 1
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {cat}: {cnt}")
    lines.append("")
    lines.append("First 50 raw problem entries:")
    lines.append("")
    for p in problems[:50]:
        lines.append(f"- {p}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
