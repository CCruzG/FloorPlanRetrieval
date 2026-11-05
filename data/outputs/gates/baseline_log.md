# Baseline log

Baselines change only through `tools/gate_all.py --update-baseline --reason ...`.

- 2026-11-04 | baseline `main` | bd6a072-dirty | stages placement, round_trip_dev | reason: first baseline at bd6a072 plus gate tooling
- 2026-11-05 | baseline `main` | de0d24a-dirty | stages placement | reason: dining term off by default (2026-11-05): retrieval no longer favours the 1.6% of plans with a dining room, so placement runs on different plans. Reviewed and accepted by Camilo.
