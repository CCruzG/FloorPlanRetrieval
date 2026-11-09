# Gate report 2026-11-06 38def56-dirty

- Command: `tools/gate_all.py --fast`
- Baseline: `main`
- Solver: 1 worker, seed from spec default, deterministic time 1.0
- Rules: default (behaviour must match baseline)
- Overall: **PASS**

## Stages

| Stage | Result | Time | Detail |
|---|---|---|---|
| unit | PASS | 2s | 113 passed, 1 warning in 1.43s |
| selftest | PASS | 0s | exit 0 |
| moves | PASS | 81s | SetConnection 14/14 predicted=actual, mismatches 0; SetPrecedence 12/12 predicted=actual, mismatches 0 |
| determinism | PASS | 31s | 12 cases (3 plans x rooms/nodes x cpsat/cpsat_dim), byte-identical twice in-process and across 2 processes |
