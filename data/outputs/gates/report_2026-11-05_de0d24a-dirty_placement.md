# Gate report 2026-11-05 de0d24a-dirty

- Command: `tools/gate_all.py --stage placement`
- Baseline: `main` (de0d24a)
- Solver: 1 worker, seed from spec default, deterministic time 1.0
- Rules: default (behaviour must match baseline)
- Overall: **PASS**

## Stages

| Stage | Result | Time | Detail |
|---|---|---|---|
| placement | PASS | 110s | vs baseline de0d24a: all metrics identical |

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
