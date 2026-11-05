# Gate report 2026-11-05 bd6a072-dirty

- Command: `tools/gate_all.py`
- Baseline: `main` (bd6a072)
- Solver: 1 worker, seed from spec default, deterministic time 1.0
- Rules: default (behaviour must match baseline)
- Overall: **FAIL**

## Stages

| Stage | Result | Time | Detail |
|---|---|---|---|
| unit | PASS | 2s | 113 passed, 1 warning in 2.22s |
| selftest | PASS | 0s | exit 0 |
| moves | PASS | 82s | SetConnection 14/14 predicted=actual, mismatches 0; SetPrecedence 12/12 predicted=actual, mismatches 0 |
| determinism | PASS | 30s | 12 cases (3 plans x rooms/nodes x cpsat/cpsat_dim), byte-identical twice in-process and across 2 processes |
| placement | FAIL | 109s | vs baseline bd6a072: 34 metric(s) moved, 91 per-plan diffs |
| round_trip_dev | PASS | 160s | vs baseline bd6a072: all metrics identical |
| e2e | FAIL | 43s | 30/32 checks |

## placement: failures

- placement/cpsat/d24_failing: 3 -> 11 (default mode: any difference fails)
- placement/cpsat/lost_source_door_open_edges: 0 -> 5 (default mode: any difference fails)
- placement/cpsat/order_violated: 59 -> 22 (default mode: any difference fails)
- placement/cpsat/reachability_fail: 1 -> 0 (default mode: any difference fails)
- placement/cpsat/sensors_passed: 106 -> 101 (default mode: any difference fails)
- placement/cpsat_dim/d24_failing: 18 -> 11 (default mode: any difference fails)
- placement/cpsat_dim/entrance_fail: 0 -> 1 (default mode: any difference fails)
- placement/cpsat_dim/lost_source_door_open_edges: 4 -> 10 (default mode: any difference fails)
- placement/cpsat_dim/order_violated: 50 -> 14 (default mode: any difference fails)
- placement/cpsat_dim/reachability_fail: 5 -> 6 (default mode: any difference fails)
- placement/nodes+cpsat/d24_failing: 10 -> 8 (default mode: any difference fails)
- placement/nodes+cpsat/infeasible: 2 -> 3 (default mode: any difference fails)
- placement/nodes+cpsat/lost_source_door_open_edges: 4 -> 1 (default mode: any difference fails)
- placement/nodes+cpsat/order_violated: 45 -> 21 (default mode: any difference fails)
- placement/nodes+cpsat/reachability_fail: 3 -> 0 (default mode: any difference fails)
- placement/nodes+cpsat/sensors_passed: 97 -> 100 (default mode: any difference fails)
- placement/nodes+cpsat_dim/d24_failing: 21 -> 10 (default mode: any difference fails)
- placement/nodes+cpsat_dim/infeasible: 3 -> 4 (default mode: any difference fails)
- placement/nodes+cpsat_dim/lost_source_door_open_edges: 9 -> 3 (default mode: any difference fails)
- placement/nodes+cpsat_dim/order_violated: 40 -> 11 (default mode: any difference fails)
- placement/nodes+cpsat_dim/reachability_fail: 8 -> 1 (default mode: any difference fails)
- placement/nodes+cpsat_dim/sensors_passed: 87 -> 98 (default mode: any difference fails)
- placement/nodes+off/d24_failing: 30 -> 15 (default mode: any difference fails)
- placement/nodes+off/infeasible: 0 -> 2 (default mode: any difference fails)
- placement/nodes+off/lost_source_door_open_edges: 11 -> 5 (default mode: any difference fails)
- placement/nodes+off/order_violated: 5 -> 0 (default mode: any difference fails)
- placement/nodes+off/reachability_fail: 11 -> 6 (default mode: any difference fails)
- placement/nodes+off/sensors_passed: 69 -> 85 (default mode: any difference fails)
- placement/off/d24_failing: 27 -> 12 (default mode: any difference fails)
- placement/off/entrance_fail: 1 -> 2 (default mode: any difference fails)
- placement/off/infeasible: 1 -> 0 (default mode: any difference fails)
- placement/off/lost_source_door_open_edges: 9 -> 15 (default mode: any difference fails)
- placement/off/order_violated: 5 -> 0 (default mode: any difference fails)
- placement/off/reachability_fail: 10 -> 14 (default mode: any difference fails)
- 91 per-plan results differ from baseline (sensors_ok / passed): placement/cpsat/test_boundary_001#1, placement/cpsat/test_boundary_001#2, placement/cpsat/test_boundary_001#3, placement/cpsat/test_boundary_001b#1, placement/cpsat/test_boundary_001b#3, placement/cpsat/test_boundary_002#1 ...

## placement: metric moves against baseline

| Metric | Baseline | Run | Delta | Direction | Verdict |
|---|---|---|---|---|---|
| placement/cpsat/d24_failing | 3 | 11 | 8 | worse | FAIL |
| placement/cpsat/lost_source_door_open_edges | 0 | 5 | 5 | worse | FAIL |
| placement/cpsat/order_violated | 59 | 22 | -37 | better | FAIL |
| placement/cpsat/reachability_fail | 1 | 0 | -1 | better | FAIL |
| placement/cpsat/sensors_passed | 106 | 101 | -5 | worse | FAIL |
| placement/cpsat_dim/d24_failing | 18 | 11 | -7 | better | FAIL |
| placement/cpsat_dim/entrance_fail | 0 | 1 | 1 | worse | FAIL |
| placement/cpsat_dim/lost_source_door_open_edges | 4 | 10 | 6 | worse | FAIL |
| placement/cpsat_dim/order_violated | 50 | 14 | -36 | better | FAIL |
| placement/cpsat_dim/reachability_fail | 5 | 6 | 1 | worse | FAIL |
| placement/nodes+cpsat/d24_failing | 10 | 8 | -2 | better | FAIL |
| placement/nodes+cpsat/infeasible | 2 | 3 | 1 | worse | FAIL |
| placement/nodes+cpsat/lost_source_door_open_edges | 4 | 1 | -3 | better | FAIL |
| placement/nodes+cpsat/order_violated | 45 | 21 | -24 | better | FAIL |
| placement/nodes+cpsat/reachability_fail | 3 | 0 | -3 | better | FAIL |
| placement/nodes+cpsat/sensors_passed | 97 | 100 | 3 | better | FAIL |
| placement/nodes+cpsat_dim/d24_failing | 21 | 10 | -11 | better | FAIL |
| placement/nodes+cpsat_dim/infeasible | 3 | 4 | 1 | worse | FAIL |
| placement/nodes+cpsat_dim/lost_source_door_open_edges | 9 | 3 | -6 | better | FAIL |
| placement/nodes+cpsat_dim/order_violated | 40 | 11 | -29 | better | FAIL |
| placement/nodes+cpsat_dim/reachability_fail | 8 | 1 | -7 | better | FAIL |
| placement/nodes+cpsat_dim/sensors_passed | 87 | 98 | 11 | better | FAIL |
| placement/nodes+off/d24_failing | 30 | 15 | -15 | better | FAIL |
| placement/nodes+off/infeasible | 0 | 2 | 2 | worse | FAIL |
| placement/nodes+off/lost_source_door_open_edges | 11 | 5 | -6 | better | FAIL |
| placement/nodes+off/order_violated | 5 | 0 | -5 | better | FAIL |
| placement/nodes+off/reachability_fail | 11 | 6 | -5 | better | FAIL |
| placement/nodes+off/sensors_passed | 69 | 85 | 16 | better | FAIL |
| placement/off/d24_failing | 27 | 12 | -15 | better | FAIL |
| placement/off/entrance_fail | 1 | 2 | 1 | worse | FAIL |
| placement/off/infeasible | 1 | 0 | -1 | better | FAIL |
| placement/off/lost_source_door_open_edges | 9 | 15 | 6 | worse | FAIL |
| placement/off/order_violated | 5 | 0 | -5 | better | FAIL |
| placement/off/reachability_fail | 10 | 14 | 4 | worse | FAIL |

## placement: per-plan differences (91)

- placement/cpsat/test_boundary_001#1
- placement/cpsat/test_boundary_001#2
- placement/cpsat/test_boundary_001#3
- placement/cpsat/test_boundary_001b#1
- placement/cpsat/test_boundary_001b#3
- placement/cpsat/test_boundary_002#1
- placement/cpsat/test_boundary_002#2
- placement/cpsat/test_boundary_002#3
- placement/cpsat/test_boundary_003#1
- placement/cpsat/test_boundary_003#2
- placement/cpsat/test_boundary_003#3
- placement/cpsat/test_boundary_004#1
- placement/cpsat/test_boundary_004#2
- placement/cpsat/test_boundary_004#3
- placement/cpsat_dim/test_boundary_001#1
- placement/cpsat_dim/test_boundary_001#2
- placement/cpsat_dim/test_boundary_001#3
- placement/cpsat_dim/test_boundary_001b#1
- placement/cpsat_dim/test_boundary_001b#2
- placement/cpsat_dim/test_boundary_001b#3
- placement/cpsat_dim/test_boundary_002#1
- placement/cpsat_dim/test_boundary_002#2
- placement/cpsat_dim/test_boundary_002#3
- placement/cpsat_dim/test_boundary_003#2
- placement/cpsat_dim/test_boundary_003#3
- placement/cpsat_dim/test_boundary_004#1
- placement/cpsat_dim/test_boundary_004#2
- placement/cpsat_dim/test_boundary_004#3
- placement/nodes+cpsat/test_boundary_001#1
- placement/nodes+cpsat/test_boundary_001#2
- placement/nodes+cpsat/test_boundary_001#3
- placement/nodes+cpsat/test_boundary_001b#1
- placement/nodes+cpsat/test_boundary_001b#2
- placement/nodes+cpsat/test_boundary_001b#3
- placement/nodes+cpsat/test_boundary_002#1
- placement/nodes+cpsat/test_boundary_002#2
- placement/nodes+cpsat/test_boundary_002#3
- placement/nodes+cpsat/test_boundary_003#1
- placement/nodes+cpsat/test_boundary_003#2
- placement/nodes+cpsat/test_boundary_004#1

Placement script verdict `gate.pass` (cpsat_dim against off and cpsat; not a gate rule): **False**; failed criteria: lost_source_edges <= off, order_violated == 0, sensors_passed >= max(off, cpsat)

## placement: totals

```json
{
 "off": {
  "sensors_passed": 74,
  "entrance_fail": 2,
  "reachability_fail": 14,
  "d24_failing": 12,
  "lost_source_door_open_edges": 15,
  "infeasible": 0,
  "order_violated": 0
 },
 "cpsat": {
  "sensors_passed": 101,
  "entrance_fail": 1,
  "reachability_fail": 0,
  "d24_failing": 11,
  "lost_source_door_open_edges": 5,
  "infeasible": 0,
  "order_violated": 22
 },
 "cpsat_dim": {
  "sensors_passed": 93,
  "entrance_fail": 1,
  "reachability_fail": 6,
  "d24_failing": 11,
  "lost_source_door_open_edges": 10,
  "infeasible": 0,
  "order_violated": 14
 },
 "nodes+off": {
  "sensors_passed": 85,
  "entrance_fail": 1,
  "reachability_fail": 6,
  "d24_failing": 15,
  "lost_source_door_open_edges": 5,
  "infeasible": 2,
  "order_violated": 0
 },
 "nodes+cpsat": {
  "sensors_passed": 100,
  "entrance_fail": 0,
  "reachability_fail": 0,
  "d24_failing": 8,
  "lost_source_door_open_edges": 1,
  "infeasible": 3,
  "order_violated": 21
 },
 "nodes+cpsat_dim": {
  "sensors_passed": 98,
  "entrance_fail": 0,
  "reachability_fail": 1,
  "d24_failing": 10,
  "lost_source_door_open_edges": 3,
  "infeasible": 4,
  "order_violated": 11
 }
}
```

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
  "plans_regressed": 154
 },
 "nodes/identity/off": {
  "feasible": 278,
  "iou_median": 0.624,
  "plans_losing_door": 138,
  "lost_door_open": 183,
  "entrance_missed": 17,
  "gen_passed_mean": 4.76,
  "plans_regressed": 165
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
  "plans_regressed": 9
 },
 "nodes/identity/cpsat": {
  "feasible": 19,
  "iou_median": 0.406,
  "plans_losing_door": 11,
  "lost_door_open": 14,
  "entrance_missed": 6,
  "gen_passed_mean": 5.77,
  "plans_regressed": 11
 }
}
```

## e2e: failures

- b: entry_node is r_2/primary :: r_3/primary
- c(cpsat): boundary_contacts.entrance.met is true :: {"applied":true,"entry_room":"r_3/primary","met":false}

## View by hand

A gate pass does not replace looking. Open these:

- `data/outputs/gates/shots/bd6a072/01_005b_retrieved.png`
- `data/outputs/gates/shots/bd6a072/02_generate_nodes_off.png`
- `data/outputs/gates/shots/bd6a072/03_generate_nodes_cpsat.png`
- `data/outputs/gates/shots/bd6a072/04_after_reset.png`
- `data/outputs/gates/shots/bd6a072/05_generate_swap.png`
- `data/outputs/gates/shots/bd6a072/06_generate_rooms_mode.png`
