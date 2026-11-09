# Gate report 2026-11-09 2b833f0-dirty

- Command: `tools/gate_all.py --stage round_trip_dev --change data/outputs/gates/changes/reachability_private_transit.json`
- Baseline: `main` (de0d24a)
- Solver: 1 worker, seed from spec default, deterministic time 1.0
- Rules: change file reachability sensor also fails when a shared room (TRANSIT) is reachable only through a private room
- Overall: **PASS**

## Stages

| Stage | Result | Time | Detail |
|---|---|---|---|
| round_trip_dev | PASS | 161s | vs baseline de0d24a: 4 metric(s) moved, 1 per-plan diffs, change 'reachability sensor also fails when a shared room (TRANSIT) is reachable only through a private room' |

## round_trip_dev: metric moves against baseline

| Metric | Baseline | Run | Delta | Direction | Verdict |
|---|---|---|---|---|---|
| round_trip_dev/nodes/describe/cpsat/plans_regressed | 9 | 8 | -1 | better | ok |
| round_trip_dev/nodes/describe/off/plans_regressed | 154 | 153 | -1 | better | ok |
| round_trip_dev/nodes/identity/cpsat/plans_regressed | 11 | 9 | -2 | better | ok |
| round_trip_dev/nodes/identity/off/plans_regressed | 165 | 164 | -1 | better | ok |

## round_trip_dev: per-plan differences (1)

- round_trip_dev/nodes/identity/off/14274

## round_trip_dev: totals

```json
{
 "rooms/describe/off": {
  "feasible": 281,
  "iou_median": 0.648,
  "plans_losing_door": 184,
  "lost_door_open": 255,
  "entrance_missed": 46,
  "gen_passed_mean": 4.51,
  "plans_regressed": 205
 },
 "rooms/identity/off": {
  "feasible": 267,
  "iou_median": 0.621,
  "plans_losing_door": 160,
  "lost_door_open": 209,
  "entrance_missed": 35,
  "gen_passed_mean": 4.65,
  "plans_regressed": 196
 },
 "nodes/describe/off": {
  "feasible": 293,
  "iou_median": 0.654,
  "plans_losing_door": 137,
  "lost_door_open": 187,
  "entrance_missed": 21,
  "gen_passed_mean": 4.81,
  "plans_regressed": 153
 },
 "nodes/identity/off": {
  "feasible": 278,
  "iou_median": 0.624,
  "plans_losing_door": 138,
  "lost_door_open": 183,
  "entrance_missed": 17,
  "gen_passed_mean": 4.76,
  "plans_regressed": 164
 },
 "rooms/describe/cpsat": {
  "feasible": 23,
  "iou_median": 0.383,
  "plans_losing_door": 5,
  "lost_door_open": 5,
  "entrance_missed": 13,
  "gen_passed_mean": 5.93,
  "plans_regressed": 12
 },
 "rooms/identity/cpsat": {
  "feasible": 23,
  "iou_median": 0.318,
  "plans_losing_door": 5,
  "lost_door_open": 5,
  "entrance_missed": 12,
  "gen_passed_mean": 5.87,
  "plans_regressed": 14
 },
 "nodes/describe/cpsat": {
  "feasible": 20,
  "iou_median": 0.39,
  "plans_losing_door": 11,
  "lost_door_open": 13,
  "entrance_missed": 7,
  "gen_passed_mean": 5.9,
  "plans_regressed": 8
 },
 "nodes/identity/cpsat": {
  "feasible": 19,
  "iou_median": 0.406,
  "plans_losing_door": 11,
  "lost_door_open": 14,
  "entrance_missed": 6,
  "gen_passed_mean": 5.77,
  "plans_regressed": 9
 }
}
```
