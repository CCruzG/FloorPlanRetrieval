"""
Measure which room-type pairs share walls in the RPlan corpus more often than
chance -- v2: adds a second, stricter null model.

Read-only analysis. Does not modify any existing file, including v1's script
and outputs (corpus_adjacency_2026-11-02.{json,md}).

Why v2: v1's null shuffled ALL room-type labels within a plan, including the
living room's. The living room is the corpus hub (geometrically adjacent to
most other rooms in a plan). Shuffling its label moves the hub geometry onto
whichever type lands there, which inflates the null adjacency rate for
non-living pairs (any type can randomly inherit hub-like adjacency counts).

v2 adds a "living_fixed" null: shuffle room-type labels only among the
NON-living rooms in a plan; living rooms keep their real label and real
position. This holds the hub geometry fixed and asks "how often would type X
land next to this (real, fixed) living room by chance, if we only scrambled
the non-living labels?" -- a more conservative null for any pair touching the
hub, and a null that no longer lets other types borrow the hub's adjacency
footprint.

Both nulls are reported side by side: null_rate_all / lift_all (same
definition as v1) and null_rate_living_fixed / lift_living_fixed (new). The
observed adjacency counts and the bootstrap method are unchanged from v1;
only the null is recomputed. Also reports: count of edges in the whole
corpus by connection value (0/1/2 and any other value found), answering
whether connection == 2 (open) occurs anywhere.

Usage:
    .venv/bin/python tools/corpus_adjacency_stats_v2.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import random
import sys
import time
from collections import Counter, defaultdict
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np

# Reuse parsing/geometry/config from v1 -- no side effects on import (v1 only
# runs main() under its own __main__ guard).
import corpus_adjacency_stats as v1

CORPUS_DIR = v1.CORPUS_DIR
OUT_DIR = v1.OUT_DIR
import os as _os
_TAG = _os.environ.get("ADJ_TAG", "2026-11-02_v2")  # 2026-11-03: set ADJ_TAG to keep earlier outputs
OUT_JSON = OUT_DIR / f"corpus_adjacency_{_TAG}.json"
OUT_MD = OUT_DIR / f"corpus_adjacency_{_TAG}.md"

SCALE_M = v1.SCALE_M
WALL_MIN_M = v1.WALL_MIN_M
SEED = v1.SEED
SAMPLE_SIZE = v1.SAMPLE_SIZE
N_SHUFFLES = v1.N_SHUFFLES
N_BOOTSTRAP = v1.N_BOOTSTRAP
N_WORKERS = v1.N_WORKERS

parse_plan_file = v1.parse_plan_file
compute_plan_adjacency = v1.compute_plan_adjacency
type_pair_key = v1.type_pair_key


# ---------------------------------------------------------------------------
# Pass 1: observed adjacency (identical logic to v1's process_file) plus a
# raw connection-value histogram over every edge in the file (not just
# geometrically-adjacent pairs) -- answers "does connection==2 occur at all".
# ---------------------------------------------------------------------------

def process_file_observed(path_str: str):
    path = Path(path_str)
    plan, problems = parse_plan_file(path)

    # Raw connection-value histogram, independent of whether the plan parsed
    # cleanly for adjacency purposes -- read directly off the JSON edges list
    # so a room-parsing problem elsewhere doesn't hide an edge's connection
    # value.
    conn_hist = Counter()
    try:
        import json as _json
        with open(path, "r") as f:
            data = _json.load(f)
        for e in data.get("edges", []):
            conn_hist[e.get("connection")] += 1
    except Exception:
        pass

    if plan is None:
        return None, problems, conn_hist

    adj = compute_plan_adjacency(plan)
    adj_set = {(i, j) for i, j, _ in adj}
    types_present = set(plan.room_types)

    pair_total = defaultdict(int)
    pair_adjacent = defaultdict(int)
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
            elif conn in (0, 1, 2):
                pair_conn[tp][str(conn)] += 1
            else:
                pair_conn[tp]["none"] += 1

    result = {
        "plan_id": plan.plan_id,
        "types_present": types_present,
        "pair_total": dict(pair_total),
        "pair_adjacent": dict(pair_adjacent),
        "pair_conn": {k: dict(v) for k, v in pair_conn.items()},
    }
    return result, problems, conn_hist


# ---------------------------------------------------------------------------
# Pass 2a: null "all" -- identical to v1 (shuffle every room's type label).
# Reused directly from v1 (v1.process_file_for_sample).
# ---------------------------------------------------------------------------

process_file_for_sample_all = v1.process_file_for_sample


# ---------------------------------------------------------------------------
# Pass 2b: null "living_fixed" -- shuffle type labels only among non-living
# rooms; living rooms keep their real label AND real position (geometry and
# identity both held fixed for the hub).
# ---------------------------------------------------------------------------

def process_file_for_sample_living_fixed(path_str: str, seed: int, n_shuffles: int):
    path = Path(path_str)
    plan, _ = parse_plan_file(path)
    if plan is None:
        return None

    adj = compute_plan_adjacency(plan)
    adj_idx_set = {(i, j) for i, j, _ in adj}
    n = len(plan.room_ids)
    all_pairs_idx = list(combinations(range(n), 2))
    base_types = plan.room_types
    non_living_idx = [i for i, t in enumerate(base_types) if t != "living"]

    rng = random.Random(f"{seed}-livingfixed-{plan.plan_id}")

    shuffle_results = []
    for _ in range(n_shuffles):
        shuffled = base_types[:]
        sub = [base_types[i] for i in non_living_idx]
        rng.shuffle(sub)
        for pos, i in enumerate(non_living_idx):
            shuffled[i] = sub[pos]

        pair_total = defaultdict(int)
        pair_adjacent = defaultdict(int)
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

def run_null_pass(sample, worker_fn):
    """Run a null-model pass over `sample` file paths with `worker_fn` and
    return pooled (pair_total, pair_adjacent) dicts plus n_parsed."""
    pair_total = defaultdict(int)
    pair_adjacent = defaultdict(int)
    n_parsed = 0
    with mp.Pool(N_WORKERS) as pool:
        for res in pool.imap_unordered(worker_fn, sample, chunksize=50):
            if res is None:
                continue
            n_parsed += 1
            for tot, adj in res["shuffles"]:
                for tp, cnt in tot.items():
                    pair_total[tp] += cnt
                for tp, cnt in adj.items():
                    pair_adjacent[tp] += cnt
    return pair_total, pair_adjacent, n_parsed


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(CORPUS_DIR.glob("*.json"), key=lambda p: p.name)
    file_strs = [str(f) for f in files]
    print(f"Found {len(file_strs)} files in corpus.", file=sys.stderr)

    # ---- Pass 1: observed adjacency + raw connection-value histogram ----
    agg_pair_total = defaultdict(int)
    agg_pair_adjacent = defaultdict(int)
    agg_pair_conn = defaultdict(lambda: {"0": 0, "1": 0, "2": 0, "none": 0})
    plan_pair_total_by_plan = defaultdict(dict)
    plan_pair_adjacent_by_plan = defaultdict(dict)
    conn_value_counts = Counter()  # over every edge in the corpus, any connection value

    all_problems = []
    n_parsed = 0
    n_failed = 0

    with mp.Pool(N_WORKERS) as pool:
        for result, problems, conn_hist in pool.imap_unordered(process_file_observed, file_strs, chunksize=200):
            all_problems.extend(problems)
            conn_value_counts.update(conn_hist)
            if result is None:
                n_failed += 1
                continue
            n_parsed += 1
            pid = result["plan_id"]
            for tp, cnt in result["pair_total"].items():
                agg_pair_total[tp] += cnt
                plan_pair_total_by_plan[pid][tp] = cnt
            for tp, cnt in result["pair_adjacent"].items():
                agg_pair_adjacent[tp] += cnt
                plan_pair_adjacent_by_plan[pid][tp] = cnt
            for tp, conn_counts in result["pair_conn"].items():
                for k, v in conn_counts.items():
                    agg_pair_conn[tp][k] += v

    plans_with_types = defaultdict(int)
    for pid, totals in plan_pair_total_by_plan.items():
        for tp in totals:
            plans_with_types[tp] += 1

    t1 = time.time()
    print(f"Pass 1 (observed + connection histogram) done in {t1 - t0:.1f}s. "
          f"Parsed {n_parsed}, failed {n_failed}.", file=sys.stderr)
    print(f"Connection value counts (ALL edges in corpus, raw): {dict(conn_value_counts)}", file=sys.stderr)

    # ---- Pass 2: two null models over the same fixed-seed sample ----
    rng_sample = random.Random(SEED)
    sample = rng_sample.sample(file_strs, min(SAMPLE_SIZE, len(file_strs)))

    worker_all = partial(process_file_for_sample_all, seed=SEED, n_shuffles=N_SHUFFLES)
    null_total_all, null_adj_all, n_sample_parsed_all = run_null_pass(sample, worker_all)
    t2 = time.time()
    print(f"Pass 2a (null 'all', {len(sample)} plans x {N_SHUFFLES} shuffles) done in {t2 - t1:.1f}s. "
          f"Parsed {n_sample_parsed_all}.", file=sys.stderr)

    worker_living_fixed = partial(process_file_for_sample_living_fixed, seed=SEED, n_shuffles=N_SHUFFLES)
    null_total_lf, null_adj_lf, n_sample_parsed_lf = run_null_pass(sample, worker_living_fixed)
    t3 = time.time()
    print(f"Pass 2b (null 'living_fixed', {len(sample)} plans x {N_SHUFFLES} shuffles) done in {t3 - t2:.1f}s. "
          f"Parsed {n_sample_parsed_lf}.", file=sys.stderr)

    # ---- Build per-type-pair arrays for bootstrap (observed rate only; same
    # resampled observed-rate distribution is reused for both nulls' CIs) ----
    tp_to_totals = defaultdict(list)
    tp_to_adj = defaultdict(list)
    for pid, totals in plan_pair_total_by_plan.items():
        adj_for_pid = plan_pair_adjacent_by_plan.get(pid, {})
        for tp, tot in totals.items():
            tp_to_totals[tp].append(tot)
            tp_to_adj[tp].append(adj_for_pid.get(tp, 0))

    rng_boot = np.random.default_rng(SEED + 1)
    all_type_pairs = set(agg_pair_total.keys())

    results = {}
    for tp in all_type_pairs:
        obs_total = agg_pair_total.get(tp, 0)
        obs_adj = agg_pair_adjacent.get(tp, 0)
        if obs_total == 0:
            continue
        obs_rate = obs_adj / obs_total

        nt_all = null_total_all.get(tp, 0)
        na_all = null_adj_all.get(tp, 0)
        null_rate_all = (na_all / nt_all) if nt_all > 0 else None
        lift_all = (obs_rate / null_rate_all) if null_rate_all else None

        nt_lf = null_total_lf.get(tp, 0)
        na_lf = null_adj_lf.get(tp, 0)
        null_rate_lf = (na_lf / nt_lf) if nt_lf > 0 else None
        lift_lf = (obs_rate / null_rate_lf) if null_rate_lf else None

        total_arr = np.asarray(tp_to_totals.get(tp, []), dtype=np.float64)
        adj_arr = np.asarray(tp_to_adj.get(tp, []), dtype=np.float64)
        n_rp = total_arr.shape[0]

        rate_samples = []
        if n_rp >= 5 and (lift_all is not None or lift_lf is not None):
            samples = np.empty(N_BOOTSTRAP, dtype=np.float64)
            valid = np.ones(N_BOOTSTRAP, dtype=bool)
            for b in range(N_BOOTSTRAP):
                idx = rng_boot.integers(0, n_rp, size=n_rp)
                tot_b = total_arr[idx].sum()
                if tot_b > 0:
                    samples[b] = adj_arr[idx].sum() / tot_b
                else:
                    valid[b] = False
            rate_samples = samples[valid]

        def ci_for(null_rate):
            if null_rate is None or len(rate_samples) == 0:
                return None, None
            lift_samples = np.sort(rate_samples / null_rate)
            lo_idx = int(0.025 * len(lift_samples))
            hi_idx = min(len(lift_samples) - 1, int(0.975 * len(lift_samples)))
            return float(lift_samples[lo_idx]), float(lift_samples[hi_idx])

        ci_lo_all, ci_hi_all = ci_for(null_rate_all)
        ci_lo_lf, ci_hi_lf = ci_for(null_rate_lf)

        conn = agg_pair_conn.get(tp, {"0": 0, "1": 0, "2": 0, "none": 0})
        door_rate = conn["1"] / obs_adj if obs_adj > 0 else None
        open_rate = conn["2"] / obs_adj if obs_adj > 0 else None

        results[tp] = {
            "type_a": tp[0],
            "type_b": tp[1],
            "plans_with_both_present": plans_with_types.get(tp, 0),
            "room_pairs_total": obs_total,
            "room_pairs_adjacent": obs_adj,
            "observed_rate": obs_rate,
            "null_rate_all": null_rate_all,
            "lift_all": lift_all,
            "lift_all_ci95_lo": ci_lo_all,
            "lift_all_ci95_hi": ci_hi_all,
            "null_rate_living_fixed": null_rate_lf,
            "lift_living_fixed": lift_lf,
            "lift_living_fixed_ci95_lo": ci_lo_lf,
            "lift_living_fixed_ci95_hi": ci_hi_lf,
            "connection_counts": conn,
            "door_rate_given_adjacent": door_rate,
            "open_rate_given_adjacent": open_rate,
        }

    t4 = time.time()
    print(f"Bootstrap (both nulls) done in {t4 - t3:.1f}s.", file=sys.stderr)

    sorted_results = sorted(
        results.values(),
        key=lambda r: (r["lift_living_fixed"] if r["lift_living_fixed"] is not None else -1),
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
        "sample_size": len(sample),
        "n_sample_parsed_all": n_sample_parsed_all,
        "n_sample_parsed_living_fixed": n_sample_parsed_lf,
        "n_shuffles": N_SHUFFLES,
        "n_bootstrap": N_BOOTSTRAP,
        "n_problems_logged": len(all_problems),
        "connection_value_counts_all_edges": {str(k): v for k, v in conn_value_counts.items()},
        "runtime_seconds": round(time.time() - t0, 1),
        "generated": "2026-11-02",
        "null_models": {
            "all": "shuffle every room's type label within the plan (v1 null, kept for comparison)",
            "living_fixed": "shuffle type labels only among non-living rooms; living rooms keep "
                             "their real label and real position, so the hub geometry is held fixed",
        },
    }

    out_json = {
        "meta": meta,
        "type_pairs": sorted_results,
        "problems_sample": all_problems[:500],
        "problems_total_count": len(all_problems),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out_json, f, indent=2, default=str)

    write_markdown(OUT_MD, meta, sorted_results, all_problems, conn_value_counts)

    print(f"Done in {time.time() - t0:.1f}s. Wrote {OUT_JSON} and {OUT_MD}.", file=sys.stderr)


def write_markdown(path: Path, meta: dict, sorted_results: list, problems: list, conn_value_counts: Counter):
    lines = []
    lines.append("# RPlan corpus adjacency statistics -- v2 (two null models)")
    lines.append("")
    lines.append(f"Generated {time.strftime('%Y-%m-%d')} (tag {_TAG}). Type map fixed 2026-11-03 (6 study, 7 storage, 9 entrance). Runtime {meta['runtime_seconds']}s. "
                 f"Files: {meta['n_files']}. Parsed: {meta['n_parsed']}. "
                 f"Failed/unusable: {meta['n_failed_to_parse_or_use']}.")
    lines.append(f"Wall threshold: shared boundary >= {meta['wall_min_m']} m "
                 f"(normalised coord x {meta['scale_m_per_unit']} = metres).")
    lines.append(f"Sample for both nulls: {meta['sample_size']} plans (seed {meta['seed']}), "
                 f"{meta['n_shuffles']} shuffles each. Parsed: all={meta['n_sample_parsed_all']}, "
                 f"living_fixed={meta['n_sample_parsed_living_fixed']}.")
    lines.append("")
    lines.append("Null 'all' (v1): shuffle every room's type label within the plan, geometry fixed.")
    lines.append("Null 'living_fixed' (v2, new): shuffle type labels only among NON-living rooms; "
                 "living rooms keep their real label and real position. This holds the living-room "
                 "hub geometry fixed so other types can no longer inherit its adjacency footprint by "
                 "chance, which v1's null allowed.")
    lines.append("")
    lines.append(f"95% CI by percentile bootstrap ({meta['n_bootstrap']} resamples) over plans, "
                 f"resampling the OBSERVED rate only and dividing by each null's fixed pooled rate "
                 f"(same resampled-rate draws reused for both CIs).")
    lines.append("")
    lines.append(f"Connection value counts over ALL edges in the corpus (raw, incl. edges whose "
                 f"rooms failed to parse): {dict(conn_value_counts)}. "
                 f"connection==2 (open) " +
                 ("occurs" if conn_value_counts.get(2, 0) > 0 else "does NOT occur") +
                 f" in the corpus ({conn_value_counts.get(2, 0)} edges).")
    lines.append("")
    lines.append(f"Data problems logged: {meta['n_problems_logged']} (sample of up to 500 in the JSON; "
                 f"identical to v1's -- geometry and parsing are unchanged).")
    lines.append("")
    lines.append("Sorted by lift_living_fixed, descending.")
    lines.append("")
    lines.append("| type A | type B | plans w/both | pairs | adj | obs rate | "
                 "null_all | lift_all | CI_all | null_lf | lift_lf | CI_lf | door | open |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted_results:
        def f4(x):
            return f"{x:.4f}" if x is not None else "n/a"
        def f2(x):
            return f"{x:.2f}" if x is not None else "n/a"
        ci_all = (f"[{r['lift_all_ci95_lo']:.2f}, {r['lift_all_ci95_hi']:.2f}]"
                  if r['lift_all_ci95_lo'] is not None else "n/a")
        ci_lf = (f"[{r['lift_living_fixed_ci95_lo']:.2f}, {r['lift_living_fixed_ci95_hi']:.2f}]"
                 if r['lift_living_fixed_ci95_lo'] is not None else "n/a")
        lines.append(
            f"| {r['type_a']} | {r['type_b']} | {r['plans_with_both_present']} | "
            f"{r['room_pairs_total']} | {r['room_pairs_adjacent']} | {r['observed_rate']:.4f} | "
            f"{f4(r['null_rate_all'])} | {f2(r['lift_all'])} | {ci_all} | "
            f"{f4(r['null_rate_living_fixed'])} | {f2(r['lift_living_fixed'])} | {ci_lf} | "
            f"{f2(r['door_rate_given_adjacent'])} | {f2(r['open_rate_given_adjacent'])} |"
        )
    lines.append("")
    lines.append("## Data problems (sample) -- unchanged from v1")
    lines.append("")
    cat_counts = defaultdict(int)
    for p in problems:
        cat = p.split(" ", 1)[0]
        cat_counts[cat] += 1
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {cat}: {cnt}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
