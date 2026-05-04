"""
test_fit_plan.py — Test harness for fit_plan.py

Usage
-----
    python test_fit_plan.py                      # run TC-01 (requires test_data/)
    python test_fit_plan.py --plan test_data/207.json
"""

import argparse
import json
import sys
from pathlib import Path

from shapely.geometry import Polygon

from fit_plan import fit_plan, MIN_DIMS, NORM, EPSILON


def _run_invariants(result, target_boundary):
    """Assert geometric invariants on a fit_plan result. Returns list of failures."""
    failures = []

    tgt_poly = Polygon([[v[0], v[1]] for v in target_boundary])
    placed_polys = []
    placed_rooms = []
    for room in result.get("rooms", []):
        st = room.get("source_type", room.get("type", -1))
        if st == 0:
            continue
        poly_verts = room.get("polygon", [])
        if len(poly_verts) >= 3:
            p = Polygon(poly_verts)
            placed_polys.append(p)
            placed_rooms.append(room)

    # Invariant 1: all placed rooms are inside the target boundary
    for room, poly in zip(placed_rooms, placed_polys):
        if not (tgt_poly.contains(poly) or tgt_poly.equals(poly)):
            # allow small floating point exceedance
            excess = poly.difference(tgt_poly).area
            if excess > EPSILON ** 2:
                failures.append(
                    f"INV1 FAIL — Room {room['id']} outside boundary "
                    f"(excess area={excess:.6f})"
                )

    # Invariant 2: no two placed rooms overlap (tolerance: epsilon^2)
    for i, (ra, pa) in enumerate(zip(placed_rooms, placed_polys)):
        for rb, pb in zip(placed_rooms[i + 1:], placed_polys[i + 1:]):
            overlap_area = pa.intersection(pb).area
            if overlap_area > EPSILON ** 2:
                failures.append(
                    f"INV2 FAIL — Rooms {ra['id']} and {rb['id']} overlap "
                    f"(area={overlap_area:.6f})"
                )

    # Invariant 3: all bedrooms touch the boundary
    for room in result.get("rooms", []):
        st = room.get("source_type", room.get("type", -1))
        if st in {1, 5, 7, 8}:
            poly_verts = room.get("polygon", [])
            if len(poly_verts) >= 3:
                poly = Polygon(poly_verts)
                dist = tgt_poly.exterior.distance(poly)
                if dist > EPSILON:
                    failures.append(
                        f"INV3 FAIL — Bedroom {room['id']} does not touch boundary "
                        f"(dist={dist:.6f})"
                    )

    # Invariant 4: no legislated minimum violated without an ERROR flag.
    # Uses orientation-independent comparison (sorted dims vs sorted spec)
    # to match _enforce_min_dims — a rotated room satisfying the spec in
    # either orientation is not flagged.
    error_rooms = {f["room_id"] for f in result.get("fit_result", {}).get("flags", [])
                   if f["level"] == "ERROR"}
    for room in result.get("rooms", []):
        st = room.get("source_type", room.get("type", -1))
        spec = MIN_DIMS.get(st)
        if spec is None or spec[0] is None or spec[2] != "ERROR":
            continue
        min_w_m, min_d_m, _ = spec
        bbox = room.get("bbox")
        if not bbox:
            continue
        w = (bbox[2] - bbox[0]) * NORM
        h = (bbox[3] - bbox[1]) * NORM
        actual_sorted = sorted([w, h])
        spec_sorted   = sorted([min_w_m, min_d_m])
        if actual_sorted[0] < spec_sorted[0] or actual_sorted[1] < spec_sorted[1]:
            if room["id"] not in error_rooms:
                failures.append(
                    f"INV4 FAIL — Room {room['id']} (type {st}) violates legislated "
                    f"minimum ({w:.2f}m × {h:.2f}m, spec {min_w_m}×{min_d_m}) "
                    f"but has no ERROR flag"
                )

    # Invariant 5: status reflects highest flag level
    flags = result.get("fit_result", {}).get("flags", [])
    levels = [f["level"] for f in flags]
    status = result.get("fit_result", {}).get("status", "ok")
    if "ERROR" in levels and status != "error":
        failures.append(f"INV5 FAIL — flags contain ERROR but status='{status}'")
    elif "WARN" in levels and "ERROR" not in levels and status not in ("warn", "error"):
        failures.append(f"INV5 FAIL — flags contain WARN but status='{status}'")

    return failures


def test_fit_plan(source_path, target_boundary, target_entrance,
                  case_name="", expected_status=None):
    """Run fit_plan and assert geometric invariants."""
    source_plan = json.loads(Path(source_path).read_text())
    target = {"boundary": target_boundary, "entrance": target_entrance}
    result = fit_plan(source_plan, target)

    failures = _run_invariants(result, target_boundary)

    status   = result.get("fit_result", {}).get("status", "unknown")
    n_flags  = len(result.get("fit_result", {}).get("flags", []))

    label = f"[{case_name}]" if case_name else ""
    if failures:
        print(f"FAIL {label} — status: {status}, flags: {n_flags}")
        for f in failures:
            print(f"  {f}")
        return False
    else:
        status_check = ""
        if expected_status and status != expected_status:
            status_check = f" (expected '{expected_status}', got '{status}')"
        print(f"PASS {label} — status: {status}{status_check}, flags: {n_flags}")
        for f in result.get("fit_result", {}).get("flags", []):
            print(f"  [{f['level']}] {f.get('room_id','—')} / {f['rule']}: {f['message']}")
        return True


# ── Test cases ─────────────────────────────────────────────────────────────────

def _get_plan_boundary(source_path):
    """Extract boundary from a plan file as target boundary."""
    plan = json.loads(Path(source_path).read_text())
    return plan.get("boundary", [])


def _get_plan_entrance(source_path):
    """Extract entrance centroid from a plan file."""
    plan = json.loads(Path(source_path).read_text())
    ent = plan.get("entrance")
    if ent and len(ent) == 4:
        return [(ent[0] + ent[2]) / 2, (ent[1] + ent[3]) / 2]
    if ent and len(ent) == 2:
        return ent
    # fallback: bottom-centre of boundary
    bdry = plan.get("boundary", [])
    if bdry:
        xs = [v[0] for v in bdry]
        ys = [v[1] for v in bdry]
        return [sum(xs) / len(xs), max(ys)]
    return [0.5, 1.0]


def _scale_boundary(boundary, scale_x, scale_y, anchor=(0, 0)):
    """Scale boundary vertices around an anchor point."""
    ax, ay = anchor
    result = []
    for v in boundary:
        nx = ax + (v[0] - ax) * scale_x
        ny = ay + (v[1] - ay) * scale_y
        entry = [nx, ny] + list(v[2:])
        result.append(entry)
    return result


def run_tc01(plan_path):
    """TC-01: Identity fit — same boundary as source."""
    boundary = _get_plan_boundary(plan_path)
    entrance = _get_plan_entrance(plan_path)
    return test_fit_plan(plan_path, boundary, entrance,
                         case_name="TC-01 identity", expected_status="ok")


def run_tc02(plan_path):
    """TC-02: Slightly smaller boundary."""
    boundary = _get_plan_boundary(plan_path)
    entrance = _get_plan_entrance(plan_path)
    # Shrink boundary by 10% around its centroid
    xs = [v[0] for v in boundary]
    ys = [v[1] for v in boundary]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    small_boundary = _scale_boundary(boundary, 0.90, 0.90, anchor=(cx, cy))
    return test_fit_plan(plan_path, small_boundary, entrance,
                         case_name="TC-02 smaller boundary", expected_status=None)


def run_tc05(plan_path):
    """TC-05: Graceful handling of a plan (no errors expected from absence of a type)."""
    boundary = _get_plan_boundary(plan_path)
    entrance = _get_plan_entrance(plan_path)
    return test_fit_plan(plan_path, boundary, entrance,
                         case_name="TC-05 graceful types", expected_status=None)


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run fit_plan test cases")
    parser.add_argument("--plan", default=None,
                        help="Path to source plan JSON. "
                             "If omitted, looks for test_data/207.json")
    parser.add_argument("--tc", default="01,02,05",
                        help="Comma-separated test cases to run (01,02,05)")
    args = parser.parse_args()

    # Resolve plan path
    if args.plan:
        plan_path = args.plan
    else:
        candidates = sorted(Path("test_data").glob("*.json"))
        if not candidates:
            print("No plan files found in test_data/. "
                  "Pass --plan <path> to specify one.")
            sys.exit(1)
        # prefer 207.json if available, else first file
        preferred = Path("test_data/207.json")
        plan_path = str(preferred if preferred.exists() else candidates[0])

    print(f"Using plan: {plan_path}")
    tcs = [t.strip() for t in args.tc.split(",")]

    results = []
    for tc in tcs:
        if tc == "01":
            results.append(run_tc01(plan_path))
        elif tc == "02":
            results.append(run_tc02(plan_path))
        elif tc == "05":
            results.append(run_tc05(plan_path))
        else:
            print(f"Unknown test case: {tc}")

    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
