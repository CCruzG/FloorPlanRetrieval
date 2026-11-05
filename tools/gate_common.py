"""Shared helpers for the gate tools (2026-11-04). Tooling only; no product code."""
import functools


def patch_solver(workers=None, det_time=None):
    """Make spec.build_options default to the given CP-SAT workers and
    deterministic time. server.py imports build_options at call time, so the
    patch also reaches /generate through the Flask test client or a backend
    started by tools/e2e/run_backend.py. Returns the original function."""
    from packages.floorplan_core import spec as spec_mod
    orig = spec_mod.build_options
    if workers is None and det_time is None:
        return orig

    @functools.wraps(orig)
    def wrapped(*a, **k):
        if workers is not None:
            k.setdefault("cpsat_workers", workers)
        if det_time is not None:
            k.setdefault("cpsat_det_time", det_time)
        return orig(*a, **k)

    spec_mod.build_options = wrapped
    return orig
