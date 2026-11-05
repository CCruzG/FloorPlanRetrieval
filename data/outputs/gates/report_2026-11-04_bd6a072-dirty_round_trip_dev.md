# Gate report 2026-11-04 bd6a072-dirty

- Command: `tools/gate_all.py --stage round_trip_dev`
- Baseline: `main` (bd6a072)
- Solver: 1 worker, seed from spec default, deterministic time 1.0
- Rules: default (behaviour must match baseline)
- Overall: **FAIL**

## Stages

| Stage | Result | Time | Detail |
|---|---|---|---|
| round_trip_dev | FAIL | 163s | vs baseline bd6a072: 22 metric(s) moved, 51 per-plan diffs |

## round_trip_dev: failures

- round_trip_dev/nodes/describe/cpsat/entrance_missed: 7 -> 30 (default mode: any difference fails)
- round_trip_dev/nodes/describe/cpsat/feasible: 20 -> 19 (default mode: any difference fails)
- round_trip_dev/nodes/describe/cpsat/gen_passed_mean: 5.9 -> 5.4 (default mode: any difference fails)
- round_trip_dev/nodes/describe/cpsat/iou_median: 0.39 -> 0.232 (default mode: any difference fails)
- round_trip_dev/nodes/describe/cpsat/lost_door_open: 13 -> 14 (default mode: any difference fails)
- round_trip_dev/nodes/describe/cpsat/plans_regressed: 9 -> 17 (default mode: any difference fails)
- round_trip_dev/nodes/describe/off/entrance_missed: 21 -> 300 (default mode: any difference fails)
- round_trip_dev/nodes/describe/off/gen_passed_mean: 4.81 -> 4.79 (default mode: any difference fails)
- round_trip_dev/nodes/describe/off/lost_door_open: 187 -> 188 (default mode: any difference fails)
- round_trip_dev/nodes/describe/off/plans_regressed: 154 -> 158 (default mode: any difference fails)
- round_trip_dev/nodes/identity/cpsat/gen_passed_mean: 5.77 -> 5.8 (default mode: any difference fails)
- round_trip_dev/nodes/identity/cpsat/plans_regressed: 11 -> 10 (default mode: any difference fails)
- round_trip_dev/nodes/identity/off/entrance_missed: 17 -> 16 (default mode: any difference fails)
- round_trip_dev/nodes/identity/off/lost_door_open: 183 -> 184 (default mode: any difference fails)
- round_trip_dev/rooms/describe/cpsat/entrance_missed: 13 -> 30 (default mode: any difference fails)
- round_trip_dev/rooms/describe/cpsat/feasible: 23 -> 24 (default mode: any difference fails)
- round_trip_dev/rooms/describe/cpsat/gen_passed_mean: 5.93 -> 5.67 (default mode: any difference fails)
- round_trip_dev/rooms/describe/cpsat/iou_median: 0.383 -> 0.415 (default mode: any difference fails)
- round_trip_dev/rooms/describe/cpsat/plans_regressed: 12 -> 17 (default mode: any difference fails)
- round_trip_dev/rooms/describe/off/entrance_missed: 46 -> 300 (default mode: any difference fails)
- round_trip_dev/rooms/describe/off/gen_passed_mean: 4.51 -> 4.49 (default mode: any difference fails)
- round_trip_dev/rooms/describe/off/plans_regressed: 205 -> 209 (default mode: any difference fails)
- 51 per-plan results differ from baseline (sensors_ok / passed): round_trip_dev/nodes/describe/cpsat/12897, round_trip_dev/nodes/describe/cpsat/14274, round_trip_dev/nodes/describe/cpsat/16510, round_trip_dev/nodes/describe/cpsat/21922, round_trip_dev/nodes/describe/cpsat/24861, round_trip_dev/nodes/describe/cpsat/35896 ...

## round_trip_dev: metric moves against baseline

| Metric | Baseline | Run | Delta | Direction | Verdict |
|---|---|---|---|---|---|
| round_trip_dev/nodes/describe/cpsat/entrance_missed | 7 | 30 | 23 | worse | FAIL |
| round_trip_dev/nodes/describe/cpsat/feasible | 20 | 19 | -1 | worse | FAIL |
| round_trip_dev/nodes/describe/cpsat/gen_passed_mean | 5.9 | 5.4 | -0.5 | worse | FAIL |
| round_trip_dev/nodes/describe/cpsat/iou_median | 0.39 | 0.232 | -0.158 | worse | FAIL |
| round_trip_dev/nodes/describe/cpsat/lost_door_open | 13 | 14 | 1 | worse | FAIL |
| round_trip_dev/nodes/describe/cpsat/plans_regressed | 9 | 17 | 8 | worse | FAIL |
| round_trip_dev/nodes/describe/off/entrance_missed | 21 | 300 | 279 | worse | FAIL |
| round_trip_dev/nodes/describe/off/gen_passed_mean | 4.81 | 4.79 | -0.02 | worse | FAIL |
| round_trip_dev/nodes/describe/off/lost_door_open | 187 | 188 | 1 | worse | FAIL |
| round_trip_dev/nodes/describe/off/plans_regressed | 154 | 158 | 4 | worse | FAIL |
| round_trip_dev/nodes/identity/cpsat/gen_passed_mean | 5.77 | 5.8 | 0.03 | better | FAIL |
| round_trip_dev/nodes/identity/cpsat/plans_regressed | 11 | 10 | -1 | better | FAIL |
| round_trip_dev/nodes/identity/off/entrance_missed | 17 | 16 | -1 | better | FAIL |
| round_trip_dev/nodes/identity/off/lost_door_open | 183 | 184 | 1 | worse | FAIL |
| round_trip_dev/rooms/describe/cpsat/entrance_missed | 13 | 30 | 17 | worse | FAIL |
| round_trip_dev/rooms/describe/cpsat/feasible | 23 | 24 | 1 | better | FAIL |
| round_trip_dev/rooms/describe/cpsat/gen_passed_mean | 5.93 | 5.67 | -0.26 | worse | FAIL |
| round_trip_dev/rooms/describe/cpsat/iou_median | 0.383 | 0.415 | 0.032 | better | FAIL |
| round_trip_dev/rooms/describe/cpsat/plans_regressed | 12 | 17 | 5 | worse | FAIL |
| round_trip_dev/rooms/describe/off/entrance_missed | 46 | 300 | 254 | worse | FAIL |
| round_trip_dev/rooms/describe/off/gen_passed_mean | 4.51 | 4.49 | -0.02 | worse | FAIL |
| round_trip_dev/rooms/describe/off/plans_regressed | 205 | 209 | 4 | worse | FAIL |

## round_trip_dev: per-plan differences (51)

- round_trip_dev/nodes/describe/cpsat/12897
- round_trip_dev/nodes/describe/cpsat/14274
- round_trip_dev/nodes/describe/cpsat/16510
- round_trip_dev/nodes/describe/cpsat/21922
- round_trip_dev/nodes/describe/cpsat/24861
- round_trip_dev/nodes/describe/cpsat/35896
- round_trip_dev/nodes/describe/cpsat/38209
- round_trip_dev/nodes/describe/cpsat/4039
- round_trip_dev/nodes/describe/cpsat/43042
- round_trip_dev/nodes/describe/cpsat/44478
- round_trip_dev/nodes/describe/cpsat/59241
- round_trip_dev/nodes/describe/cpsat/62007
- round_trip_dev/nodes/describe/cpsat/63467
- round_trip_dev/nodes/describe/cpsat/65900
- round_trip_dev/nodes/describe/cpsat/6609
- round_trip_dev/nodes/describe/cpsat/72600
- round_trip_dev/nodes/describe/cpsat/72664
- round_trip_dev/nodes/describe/cpsat/75410
- round_trip_dev/nodes/describe/cpsat/80515
- round_trip_dev/nodes/describe/off/11639
- round_trip_dev/nodes/describe/off/14593
- round_trip_dev/nodes/describe/off/26855
- round_trip_dev/nodes/describe/off/60154
- round_trip_dev/nodes/describe/off/62639
- round_trip_dev/nodes/describe/off/66009
- round_trip_dev/nodes/describe/off/73278
- round_trip_dev/nodes/identity/cpsat/21922
- round_trip_dev/nodes/identity/off/11639
- round_trip_dev/nodes/identity/off/21922
- round_trip_dev/rooms/describe/cpsat/12897
- round_trip_dev/rooms/describe/cpsat/13626
- round_trip_dev/rooms/describe/cpsat/16510
- round_trip_dev/rooms/describe/cpsat/24861
- round_trip_dev/rooms/describe/cpsat/4039
- round_trip_dev/rooms/describe/cpsat/44478
- round_trip_dev/rooms/describe/cpsat/59241
- round_trip_dev/rooms/describe/cpsat/62007
- round_trip_dev/rooms/describe/cpsat/65900
- round_trip_dev/rooms/describe/cpsat/6609
- round_trip_dev/rooms/describe/cpsat/69442

## round_trip_dev: totals

```json
{
 "rooms/describe/off": {
  "feasible": 281,
  "iou_median": 0.648,
  "plans_losing_door": 184,
  "lost_door_open": 255,
  "entrance_missed": 300,
  "gen_passed_mean": 4.49,
  "plans_regressed": 209
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
  "lost_door_open": 188,
  "entrance_missed": 300,
  "gen_passed_mean": 4.79,
  "plans_regressed": 158
 },
 "nodes/identity/off": {
  "feasible": 278,
  "iou_median": 0.624,
  "plans_losing_door": 138,
  "lost_door_open": 184,
  "entrance_missed": 16,
  "gen_passed_mean": 4.76,
  "plans_regressed": 165
 },
 "rooms/describe/cpsat": {
  "feasible": 24,
  "iou_median": 0.415,
  "plans_losing_door": 5,
  "lost_door_open": 5,
  "entrance_missed": 30,
  "gen_passed_mean": 5.67,
  "plans_regressed": 17
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
  "feasible": 19,
  "iou_median": 0.232,
  "plans_losing_door": 11,
  "lost_door_open": 14,
  "entrance_missed": 30,
  "gen_passed_mean": 5.4,
  "plans_regressed": 17
 },
 "nodes/identity/cpsat": {
  "feasible": 19,
  "iou_median": 0.406,
  "plans_losing_door": 11,
  "lost_door_open": 14,
  "entrance_missed": 6,
  "gen_passed_mean": 5.8,
  "plans_regressed": 10
 }
}
```
