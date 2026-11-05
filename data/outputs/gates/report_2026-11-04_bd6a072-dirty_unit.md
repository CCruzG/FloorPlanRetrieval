# Gate report 2026-11-04 bd6a072-dirty

- Command: `tools/gate_all.py --stage unit`
- Baseline: `main`
- Solver: 1 worker, seed from spec default, deterministic time 1.0
- Rules: default (behaviour must match baseline)
- Overall: **FAIL**

## Stages

| Stage | Result | Time | Detail |
|---|---|---|---|
| unit | FAIL | 1s | 4 failed, 107 passed, 1 warning in 0.81s |

## unit: failures

- FAILED packages/floorplan_core/tests/test_entrance_units.py::test_entry_room_is_the_door_room_not_the_nearest_room
- FAILED packages/floorplan_core/tests/test_entrance_units.py::test_entry_room_of_expanded_plan_reads_entry_node
- FAILED packages/floorplan_core/tests/test_node_graph.py::test_entry_room_of
- FAILED packages/floorplan_core/tests/test_spec.py::test_fit_replaces_given_fields_keeps_the_rest_and_is_pure
