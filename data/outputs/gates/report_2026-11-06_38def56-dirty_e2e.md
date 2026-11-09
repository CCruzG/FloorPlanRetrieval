# Gate report 2026-11-06 38def56-dirty

- Command: `tools/gate_all.py --stage e2e`
- Baseline: `main`
- Solver: 1 worker, seed from spec default, deterministic time 1.0
- Rules: default (behaviour must match baseline)
- Overall: **FAIL**

## Stages

| Stage | Result | Time | Detail |
|---|---|---|---|
| e2e | FAIL | 43s | 31/32 checks |

## e2e: failures

- c(cpsat): boundary_contacts.entrance.met is true :: {"applied":true,"entry_room":"r_3/primary","met":false}

## View by hand

A gate pass does not replace looking. Open these:

- `data/outputs/gates/shots/38def56/01_005b_retrieved.png`
- `data/outputs/gates/shots/38def56/02_generate_nodes_off.png`
- `data/outputs/gates/shots/38def56/03_generate_nodes_cpsat.png`
- `data/outputs/gates/shots/38def56/04_after_reset.png`
- `data/outputs/gates/shots/38def56/05_generate_swap.png`
- `data/outputs/gates/shots/38def56/06_generate_rooms_mode.png`
