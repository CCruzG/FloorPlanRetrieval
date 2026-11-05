# Gate report 2026-11-04 bd6a072-dirty

- Command: `tools/gate_all.py --stage placement`
- Baseline: `main` (bd6a072)
- Solver: 1 worker, seed from spec default, deterministic time 1.0
- Rules: default (behaviour must match baseline)
- Overall: **PASS**

## Stages

| Stage | Result | Time | Detail |
|---|---|---|---|
| placement | PASS | 100s | vs baseline bd6a072: all metrics identical |

Placement script verdict `gate.pass` (cpsat_dim against off and cpsat; not a gate rule): **False**; failed criteria: lost_source_edges <= off, order_violated == 0, sensors_passed >= max(off, cpsat)

## placement: totals

```json
{
 "off": {
  "sensors_passed": 74,
  "entrance_fail": 1,
  "reachability_fail": 10,
  "d24_failing": 27,
  "lost_source_door_open_edges": 9,
  "infeasible": 1,
  "order_violated": 5
 },
 "cpsat": {
  "sensors_passed": 106,
  "entrance_fail": 1,
  "reachability_fail": 1,
  "d24_failing": 3,
  "lost_source_door_open_edges": 0,
  "infeasible": 0,
  "order_violated": 59
 },
 "cpsat_dim": {
  "sensors_passed": 93,
  "entrance_fail": 0,
  "reachability_fail": 5,
  "d24_failing": 18,
  "lost_source_door_open_edges": 4,
  "infeasible": 0,
  "order_violated": 50
 },
 "nodes+off": {
  "sensors_passed": 69,
  "entrance_fail": 1,
  "reachability_fail": 11,
  "d24_failing": 30,
  "lost_source_door_open_edges": 11,
  "infeasible": 0,
  "order_violated": 5
 },
 "nodes+cpsat": {
  "sensors_passed": 97,
  "entrance_fail": 0,
  "reachability_fail": 3,
  "d24_failing": 10,
  "lost_source_door_open_edges": 4,
  "infeasible": 2,
  "order_violated": 45
 },
 "nodes+cpsat_dim": {
  "sensors_passed": 87,
  "entrance_fail": 0,
  "reachability_fail": 8,
  "d24_failing": 21,
  "lost_source_door_open_edges": 9,
  "infeasible": 3,
  "order_violated": 40
 }
}
```
