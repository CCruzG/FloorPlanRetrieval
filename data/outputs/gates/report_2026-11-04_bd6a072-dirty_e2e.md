# Gate report 2026-11-04 bd6a072-dirty

- Command: `tools/gate_all.py --stage e2e`
- Baseline: `main`
- Solver: 1 worker, seed from spec default, deterministic time 1.0
- Rules: default (behaviour must match baseline)
- Overall: **PASS**

## Stages

| Stage | Result | Time | Detail |
|---|---|---|---|
| e2e | PASS | 42s | 32/32 checks |

## View by hand

A gate pass does not replace looking. Open these:

- `data/outputs/gates/shots/bd6a072/01_005b_retrieved.png`
- `data/outputs/gates/shots/bd6a072/02_generate_nodes_off.png`
- `data/outputs/gates/shots/bd6a072/03_generate_nodes_cpsat.png`
- `data/outputs/gates/shots/bd6a072/04_after_reset.png`
- `data/outputs/gates/shots/bd6a072/05_generate_swap.png`
- `data/outputs/gates/shots/bd6a072/06_generate_rooms_mode.png`
