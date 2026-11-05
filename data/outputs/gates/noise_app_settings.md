# Placement noise at app settings (2026-11-04, bd6a072 plus gate tooling)

Command: `tools/diag_placement_topology.py` with no solver flags (8 workers, 5 s wall limit, seed 2026), run 3 times in a row. 18 plans, 6 runs each, 108 plan-runs. Each run took about 188 s. Raw outputs: `data/outputs/gates/runs/noise/app_settings_{1,2,3}.json`.

Per-run totals (run 1 / run 2 / run 3). Bold marks a total that moved.

| Run | sensors_passed | entrance_fail | reachability_fail | d24_failing | lost_source_door_open_edges | infeasible | order_violated |
|---|---|---|---|---|---|---|---|
| off | 74 / 74 / 74 | 1 / 1 / 1 | 10 / 10 / 10 | 27 / 27 / 27 | 9 / 9 / 9 | 1 / 1 / 1 | 5 / 5 / 5 |
| cpsat | **106 / 107 / 107** | 0 / 0 / 0 | 1 / 1 / 1 | 3 / 3 / 3 | 0 / 0 / 0 | 0 / 0 / 0 | 52 / 52 / 52 |
| cpsat_dim | 92 / 92 / 92 | 0 / 0 / 0 | 5 / 5 / 5 | 19 / 19 / 19 | 4 / 4 / 4 | 0 / 0 / 0 | 46 / 46 / 46 |
| nodes+off | 69 / 69 / 69 | 1 / 1 / 1 | 11 / 11 / 11 | 30 / 30 / 30 | 11 / 11 / 11 | 0 / 0 / 0 | 5 / 5 / 5 |
| nodes+cpsat | **95 / 96 / 95** | 0 / 0 / 0 | 3 / 3 / 3 | 14 / 14 / 14 | 7 / 7 / 7 | **4 / 3 / 4** | 38 / 38 / 38 |
| nodes+cpsat_dim | **83 / 82 / 83** | 0 / 0 / 0 | 8 / 8 / 8 | **25 / 27 / 25** | 11 / 11 / 11 | 4 / 4 / 4 | **37 / 36 / 37** |

Per-plan: 4 of 108 plan-runs differ in sensors_ok or passed between runs (cpsat test_boundary_001 rank 3 and test_boundary_004 rank 2; nodes+cpsat test_boundary_004 rank 3; nodes+cpsat_dim test_boundary_001 rank 3). Placement off does not move.

Reading: the noise is 0 to 2 in sensors_passed per run and 0 to 2 in d24_failing, only in the CP-SAT modes. A gate on the app settings would need a tolerance of about 2; the gate suite instead uses 1 worker and a deterministic time limit, where repeated runs are identical.
