# Tools

This directory contains offline and developer-only scripts.

Canonical script entrypoints now live here, while selected root-level script
paths are kept as compatibility shims for legacy workflows.

Long-term placement guidance:
- Keep CLI scripts in this directory.
- Keep reusable domain logic in `packages/floorplan_core`.
- Let root-level scripts delegate only, and avoid duplicating business logic.
