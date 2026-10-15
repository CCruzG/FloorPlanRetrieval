"""
Bridges the app's v2.1.0 retrieval/plan dict format to fitting-prototype's
canonical schema.py FloorPlan/Boundary contract, and runs the schema.py-native
adaptation chain (phase1 -> phase2 -> phase25 preprocess) against a live
retrieved plan and the boundary the user drew.

fitting-prototype is a separate, non-package research repo living in
Artefacts/, not part of this repo — see docs/pipeline-architecture.md
decisions 1/2. Its modules are imported directly via sys.path rather than
vendored, so that stays the single source of truth for the schema.py-native
pipeline and this module doesn't drift from it.

The conversion logic in boundary_to_schema/plan_to_schema mirrors
fitting-prototype/pipeline.py:load_case, generalised from file paths to
in-memory dicts/points — the app has no file to read, only a POSTed
retrieval result and a boundary drawn in the browser.

Runs the full chain including phase25's LLM-agent repair loop (diagnose_layout
-> run_phase25_agent -> execute_phase25_plan, up to MAX_PASSES rounds) — the
same loop pipeline.py's __main__ runs for the validated schema_fixtures.
run_phase25_agent calls a local Ollama server (qwen3.5:9b), not a remote API.
Text-only (include_image=False), matching the CLI default — no per-pass
render/encode overhead.
"""

import math
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
# fit_rooms_to_boundary (phase2.py) unconditionally writes a debug PNG via
# render.py's matplotlib.pyplot, which defaults to the interactive MacOSX
# backend. Flask serves requests off the main thread, and that backend can't
# create a GUI FigureManager there ("Cannot create a GUI FigureManager
# outside the main thread"). Must be set before phase2 (-> render) is
# imported below.
matplotlib.use("Agg")

FITTING_PROTOTYPE_DIR = Path(
    "/Users/camilo/Documents/Work_FBI/Artefacts/project-44/code/fitting-prototype"
)
if str(FITTING_PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(FITTING_PROTOTYPE_DIR))

from geometry import centroid, find_entrance_edge_idx, find_shared_wall, point_on_segment  # noqa: E402
from phase1 import place_initial_layout  # noqa: E402
from phase2 import fit_rooms_to_boundary  # noqa: E402
from phase25 import (  # noqa: E402
    diagnose_layout,
    execute_phase25_plan,
    preprocess_layout,
    run_phase25_agent,
)
from pipeline import _is_clean, _stagnated  # noqa: E402
from schema import (  # noqa: E402
    SCALE,
    SCHEMA_VERSION,
    Boundary,
    Edge,
    FloorPlan,
    ROOM_TYPE_MAP,
    Room,
)
from validator import validate  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "outputs" / "fit_pipeline"
MAX_AGENT_PASSES = 5  # matches pipeline.py's MAX_PASSES


def boundary_to_schema(boundary_pts, entrance_point, typed_edges=None) -> Boundary:
    """boundary_pts, entrance_point: normalised [0,1] coords, in whatever
    space the caller already put them in (the app transforms the drawn
    boundary into the retrieved plan's own space before calling this —
    see main.js:boundaryInPlanSpace).

    typed_edges (optional): the boundary's own edge classification, as
    [{edge_index|index, type}] with type in BOUNDARY_EDGE_TYPES. A floor-plate
    unit carries this from the generator; a freehand boundary does not. When
    supplied, it takes precedence over the synthesised default — without that
    an interior unit's party wall (a `shared_wall`, the one edge that CANNOT
    carry a window) is silently relabelled `external`, the one edge that can.
    See docs/adaptation-process-model.md, "overwritten at the handoff"."""
    bp = [(x * SCALE, y * SCALE) for x, y in boundary_pts]
    ep = (entrance_point[0] * SCALE, entrance_point[1] * SCALE)
    entrance_edge_idx = find_entrance_edge_idx(bp, ep)
    n = len(bp)
    supplied_by_index = {}
    for te in (typed_edges or []):
        idx = te.get("edge_index", te.get("index"))
        if idx is not None and te.get("type"):
            supplied_by_index[int(idx) % n] = te["type"]
    typed = [
        {
            "index": i,
            "start": bp[i],
            "end": bp[(i + 1) % n],
            "type": supplied_by_index.get(
                i,
                "facing_corridor" if i == entrance_edge_idx else "external",
            ),
        }
        for i in range(n)
    ]
    return Boundary(
        polygon=bp,
        typed_edges=typed,
        entrance_point=ep,
        entrance_edge_index=entrance_edge_idx,
    )


def plan_to_schema(data: dict) -> FloorPlan:
    """data: a v2.1.0 plan dict, as returned in /retrieve's `plans` list."""
    rooms: dict[str, Room] = {}
    for r in data["rooms"]:
        poly = [(pt[0] * SCALE, pt[1] * SCALE) for pt in r["polygon"]]
        rtype = ROOM_TYPE_MAP.get(r.get("type", r.get("source_type")), "unknown")
        rooms[r["id"]] = Room(id=r["id"], type=rtype, polygon=poly)

    plan_entrance = data["entrance"]
    entrance_mid = (
        (plan_entrance[0] + plan_entrance[2]) / 2 * SCALE,
        (plan_entrance[1] + plan_entrance[3]) / 2 * SCALE,
    )
    best_rid = None
    for rid, room in rooms.items():
        poly = room.polygon
        n = len(poly)
        for i in range(n):
            if point_on_segment(entrance_mid, poly[i], poly[(i + 1) % n], tol=100):
                best_rid = rid
                break
        if best_rid:
            break
    if best_rid is None:
        best_dist = float("inf")
        for rid, room in rooms.items():
            cx, cy = centroid(room.polygon)
            d = math.dist((cx, cy), entrance_mid)
            if d < best_dist:
                best_dist, best_rid = d, rid
    if best_rid:
        rooms[best_rid].is_entrance = True

    adjacency: dict[str, list[str]] = {rid: [] for rid in rooms}
    connections: dict[str, list[str]] = {rid: [] for rid in rooms}
    edges: dict[str, Edge] = {}

    for i, e in enumerate(data["edges"]):
        u, v = e["u"], e["v"]
        has_door = e.get("connection", 0) == 1
        relation = e.get("relation")

        shared = find_shared_wall(rooms[u].polygon, rooms[v].polygon)
        if shared:
            start, end = shared
        else:
            start = centroid(rooms[u].polygon)
            end = centroid(rooms[v].polygon)

        eid = f"e_{i}"
        edges[eid] = Edge(
            id=eid,
            room_a=u,
            room_b=v,
            start=start,
            end=end,
            type="internal",
            has_door=has_door,
            relation=relation,
        )
        rooms[u].edges.append(eid)
        rooms[v].edges.append(eid)
        adjacency[u].append(v)
        adjacency[v].append(u)
        if has_door:
            connections[u].append(v)
            connections[v].append(u)

    def _pts_near(a, b, tol: float = 1.0) -> bool:
        return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol

    for rid, room in rooms.items():
        poly = room.polygon
        n = len(poly)
        for i in range(n):
            seg_s, seg_e = poly[i], poly[(i + 1) % n]
            covered = any(
                (
                    (_pts_near(e.start, seg_s) and _pts_near(e.end, seg_e))
                    or (_pts_near(e.start, seg_e) and _pts_near(e.end, seg_s))
                )
                for e in edges.values()
                if e.room_b is not None
            )
            if not covered:
                ext_eid = f"ext_{rid}_{i}"
                edges[ext_eid] = Edge(
                    id=ext_eid,
                    room_a=rid,
                    room_b=None,
                    start=seg_s,
                    end=seg_e,
                    type="external",
                    has_door=False,
                    relation=None,
                )
                room.edges.append(ext_eid)

    return FloorPlan(
        rooms=rooms,
        edges=edges,
        adjacency=adjacency,
        connections=connections,
        entrance_mid=entrance_mid,
    )


def run_fit_pipeline(fp: dict, boundary_pts, entrance_point, label: str = "live") -> dict:
    """Runs the full schema.py-native chain against a live retrieved plan and
    the boundary the user drew — place_initial_layout -> fit_rooms_to_boundary
    -> preprocess_layout, then up to MAX_AGENT_PASSES rounds of phase25's
    diagnose -> agent -> execute repair loop, same as pipeline.py's __main__.
    Returns the same {schema_version, label, boundary, plan, violations}
    shape as the static schema_fixtures the plan-drawing panel already
    renders, plus agent_passes for visibility into how much repair ran."""
    boundary = boundary_to_schema(boundary_pts, entrance_point)
    plan = plan_to_schema(fp)

    plan = place_initial_layout(plan, boundary)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fit_rooms_to_boundary(plan, boundary, output_dir=OUTPUT_DIR, label=label)
    preprocess_layout(plan, boundary)

    diagnosis = diagnose_layout(plan, boundary)
    agent_passes = 0
    for _ in range(MAX_AGENT_PASSES):
        if _is_clean(diagnosis):
            break
        render_path = OUTPUT_DIR / f"phase25_{label}_pass{agent_passes}.png"
        actions = run_phase25_agent(label, plan, boundary, diagnosis, render_path, include_image=False)
        if not actions:
            break
        execute_phase25_plan(plan, actions)
        agent_passes += 1
        prev_diagnosis = diagnosis
        diagnosis = diagnose_layout(plan, boundary)
        if _stagnated(prev_diagnosis, diagnosis):
            break

    violations = validate(plan, boundary)

    return {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "boundary": asdict(boundary),
        "plan": asdict(plan),
        "violations": [asdict(v) for v in violations],
        "agent_passes": agent_passes,
    }
