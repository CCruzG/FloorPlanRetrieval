"""
Task 0 gate — can a floor-plate UNIT boundary survive the canonical ARAP
pipeline?

Background (docs/adaptation-process-model.md, 2026-08-18): the app has two
disconnected flows. The floor-plate flow (draw plate -> generate units ->
batch retrieve) fits with `fitPlanLocal` in apps/web/src/retrieval.js, a
client-side edge-push placeholder that only TRANSLATES rooms — never scales
or deforms them — and which pipeline-architecture.md decision 1 declared
superseded on 2026-08-06. The canonical ARAP pipeline is wired only to the
single-drawn-boundary flow. Task 0 connects them.

The risk this test exists to retire: every boundary ARAP has been validated
against so far is a hand-drawn test boundary. Floor-plate units are small,
mostly-rectangular, corridor-fed polygons — a different shape population
entirely. If ARAP falls over on them, task 0 is a bigger job than a call-site
swap, and better to know that before editing the frontend.

Run:  .venv/bin/python tools/test_floorplate_arap.py
"""

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shapely.geometry import Polygon as SPoly  # noqa: E402

from packages.floorplan_core.arap_bridge import run_fill_consolidate_canvas  # noqa: E402
from packages.floorplan_core.index_store import build_index_store_from_npz  # noqa: E402
from packages.floorplan_core.retrieval import run_retrieval  # noqa: E402
from packages.floor_plate_design.generator import generate_initial_layout  # noqa: E402

MM_PER_UNIT = 18_000  # constants.js METRES_PER_UNIT=18; matches schema.SCALE
INDEX_PATH = REPO_ROOT / "data" / "index" / "rplan_index_v5.npz"
# Shared with fitting-prototype/retrieval-prototype, not duplicated into this
# repo's data/ — same default server.py uses (RPLAN_DATASET_DIR).
RPLAN_DIR = Path(
    "/Users/camilo/Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json"
)

# A plausible slab plate: 46 m x 16 m, in mm. Big enough for a corridor and
# several units, which is the arrangement that produces shared_wall edges.
PLATE = [(0.0, 0.0), (46_000.0, 0.0), (46_000.0, 16_000.0), (0.0, 16_000.0)]
UNIT_MIX = [
    {"type": "1bed", "count": 2, "target_area": 50},
    {"type": "2bed", "count": 2, "target_area": 70},
]


def mirror_plan_y(fp: dict) -> dict:
    """Port of main.js:mirrorPlanY — reflect about a horizontal line through
    the plan's own boundary centroid. Retrieval's mirror convention is a
    y-flip (conj() = angle negation); the fitting side only rotates in 90deg
    steps and has no reflection of its own, so the mirror has to be baked
    into the geometry before the ARAP side ever sees it. Flow A currently
    skips this entirely — a pre-existing bug that only becomes visible once
    the units are routed through ARAP."""
    bdry = fp.get("boundary") or []
    if not bdry:
        return fp
    cy = sum(p[1] for p in bdry) / len(bdry)
    my = lambda y: 2 * cy - y  # noqa: E731
    out = dict(fp)
    out["boundary"] = [[p[0], my(p[1]), *p[2:]] for p in bdry]
    out["rooms"] = [
        {**r, "polygon": [[x, my(y)] for x, y in (r.get("polygon") or [])]}
        for r in (fp.get("rooms") or [])
    ]
    ent = fp.get("entrance")
    if ent:
        out["entrance"] = [ent[0], my(ent[1]), ent[2], my(ent[3])]
    return out


def bedrooms_for(unit_type: str) -> int:
    return 3 if unit_type.startswith("3") else 2 if unit_type.startswith("2") else 1


def check_rooms(rooms: list, boundary_norm: list) -> list:
    """Invariant assertions on one adapted unit. Returns a list of failures."""
    problems = []
    if not rooms:
        return ["no rooms returned"]

    bpoly = SPoly(boundary_norm)
    polys = []
    for r in rooms:
        poly = r.get("polygon") or []
        if len(poly) < 3:
            problems.append(f"room {r.get('id')} has < 3 vertices")
            continue
        p = SPoly(poly)
        if not p.is_valid:
            problems.append(f"room {r.get('id')} polygon is invalid (self-intersecting)")
            continue
        if p.area <= 0:
            problems.append(f"room {r.get('id')} has non-positive area")
            continue
        polys.append((r.get("id"), p))

    # Containment: nothing meaningfully outside the unit boundary.
    for rid, p in polys:
        outside = p.difference(bpoly).area
        if outside / max(p.area, 1e-12) > 0.02:  # 2% slack for float noise
            problems.append(f"room {rid} sits {outside / p.area * 100:.1f}% outside the unit boundary")

    # No meaningful overlap between rooms.
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            ov = polys[i][1].intersection(polys[j][1]).area
            if ov / max(min(polys[i][1].area, polys[j][1].area), 1e-12) > 0.02:
                problems.append(f"rooms {polys[i][0]} and {polys[j][0]} overlap")

    return problems


def main() -> int:
    if not INDEX_PATH.exists():
        print(f"FAIL — index not found at {INDEX_PATH}")
        return 1

    print("Generating floor plate…")
    plate = generate_initial_layout(
        PLATE,
        UNIT_MIX,
        {
            "min_entrance_overlap": 1200,
            "min_common_area": 20,
            "max_travel_distance": 30_000,
            "core_type": "stair_lift",
            "max_unit_depth": 8000.0,
        },
    )
    units = plate.units
    print(f"  {len(units)} unit(s), typology {plate.typology.type.value}/{plate.typology.sub_type}")

    edge_types = {}
    for u in units:
        for e in u.typed_edges:
            t = e.type.value if hasattr(e.type, "value") else str(e.type)
            edge_types[t] = edge_types.get(t, 0) + 1
    print(f"  unit edge types across the plate: {edge_types}")
    if "shared_wall" not in edge_types:
        print("  NOTE — no shared_wall edges generated; the intertenancy case is not exercised")

    print("Loading retrieval index…")
    logger = logging.getLogger("floorplate-arap-gate")
    index_store = build_index_store_from_npz(
        npz_path=str(INDEX_PATH),
        data_dir=str(RPLAN_DIR),
        logger=logger,
    )

    failures, results = [], []
    for unit in units:
        poly_mm = unit.polygon
        if not poly_mm or len(poly_mm) < 3:
            failures.append(f"{unit.id}: bad polygon from the generator")
            continue

        boundary_norm = [(x / MM_PER_UNIT, y / MM_PER_UNIT) for x, y in poly_mm]
        ent = unit.entrance_point
        entrance_point = [ent[0] / MM_PER_UNIT, ent[1] / MM_PER_UNIT]

        try:
            payload = run_retrieval(
                index_store=index_store,
                boundary=[[x, y, 0, 0] for x, y in boundary_norm],
                entrance_point=entrance_point,
                requirements={"bedrooms": bedrooms_for(unit.type), "bathrooms": 1},
                topn=3,
                shape_w=1.0,
                ent_w=2.0,
                room_w=1.0,
                logger=None,
            )
        except Exception as exc:
            failures.append(f"{unit.id}: retrieval raised {type(exc).__name__}: {exc}")
            continue

        plans = payload.get("plans") or []
        if not plans:
            failures.append(f"{unit.id}: retrieval returned 0 plans")
            continue

        fp = plans[0]
        retrieval_meta = fp.get("retrieval") or {}
        if retrieval_meta.get("mirror"):
            fp = mirror_plan_y(fp)

        try:
            out = run_fill_consolidate_canvas(fp, boundary_norm, entrance_point)
        except Exception as exc:
            failures.append(f"{unit.id}: ARAP raised {type(exc).__name__}: {exc}")
            continue

        rooms = out.get("rooms") or []
        problems = check_rooms(rooms, boundary_norm)
        area_m2 = SPoly(poly_mm).area / 1e6
        results.append((unit.id, unit.type, area_m2, retrieval_meta.get("dist"),
                        bool(retrieval_meta.get("mirror")), len(rooms), problems))
        failures.extend(f"{unit.id}: {p}" for p in problems)

    print("\nPer-unit results")
    print(f"  {'unit':<10} {'type':<6} {'area m2':>8} {'dist':>7} {'mir':>4} {'rooms':>6}  status")
    for uid, utype, area, dist, mirror, nrooms, problems in results:
        status = "ok" if not problems else f"FAIL ({len(problems)})"
        dist_s = f"{dist:.3f}" if isinstance(dist, (int, float)) else "n/a"
        print(f"  {uid:<10} {utype:<6} {area:8.1f} {dist_s:>7} {str(mirror):>4} {nrooms:6d}  {status}")

    print()
    if failures:
        print(f"GATE FAILED — {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"GATE PASSED — {len(results)} floor-plate unit(s) adapted through the "
          f"canonical ARAP pipeline, all rooms valid, contained and non-overlapping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
