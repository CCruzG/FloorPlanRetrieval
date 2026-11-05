# Gate report 2026-11-05 1c053de-dirty

- Command: `tools/gate_all.py`
- Baseline: `main` (de0d24a)
- Solver: 1 worker, seed from spec default, deterministic time 1.0
- Rules: default (behaviour must match baseline)
- Overall: **FAIL**

## Stages

| Stage | Result | Time | Detail |
|---|---|---|---|
| unit | PASS | 1s | 113 passed, 1 warning in 1.12s |
| selftest | PASS | 0s | exit 0 |
| moves | PASS | 81s | SetConnection 14/14 predicted=actual, mismatches 0; SetPrecedence 12/12 predicted=actual, mismatches 0 |
| determinism | PASS | 31s | 12 cases (3 plans x rooms/nodes x cpsat/cpsat_dim), byte-identical twice in-process and across 2 processes |
| placement | PASS | 112s | vs baseline de0d24a: all metrics identical |
| round_trip_dev | PASS | 160s | vs baseline de0d24a: all metrics identical |
| e2e | FAIL | 43s | 31/32 checks |

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

- c(cpsat): boundary_contacts.entrance.met is true :: {"applied":true,"entry_room":"r_3/primary","met":false}

## View by hand

A gate pass does not replace looking. Open these:

- `data/outputs/gates/shots/1c053de/01_005b_retrieved.png`
- `data/outputs/gates/shots/1c053de/02_generate_nodes_off.png`
- `data/outputs/gates/shots/1c053de/03_generate_nodes_cpsat.png`
- `data/outputs/gates/shots/1c053de/04_after_reset.png`
- `data/outputs/gates/shots/1c053de/05_generate_swap.png`
- `data/outputs/gates/shots/1c053de/06_generate_rooms_mode.png`
