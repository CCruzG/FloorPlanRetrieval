"""
Bridges the app's v2.1.0 retrieval/plan dict format to fitting-prototype's
arap_adapt.py — the actual As-Rigid-As-Possible mesh-deformation pipeline
(distinct from schema_bridge.py's phase1/phase2/phase25 chain, which does not
call arap_adapt.py at all — see docs/pipeline-architecture.md and the
2026-08-11 finding that those two pathways are disconnected in the codebase).

Exposes the pipeline as four discrete, independently callable steps —
align_and_solve, rectify, reconstruct, refine — each recomputing from scratch
up to that point (stateless; every step here is fast deterministic numpy/scipy,
not an LLM call, so redoing prior steps costs nothing and avoids threading
fragile state between requests). Each returns a schema.py-shaped fixture
({schema_version, label, boundary, plan, violations}) so the same
plan-drawing-panel modal used elsewhere can render every intermediate stage —
built explicitly for step-by-step visual debugging of the fit itself, ahead
of any concern for architectural representation (doors, wall thickness, etc.
are not derived here).
"""

import sys
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

FITTING_PROTOTYPE_DIR = Path(
    "/Users/camilo/Documents/Work_FBI/Artefacts/project-44/code/fitting-prototype"
)
if str(FITTING_PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(FITTING_PROTOTYPE_DIR))

from arap_adapt import (  # noqa: E402
    arap_solve,
    build_mesh,
    compute_handles_arclength,
    consolidate_living,
    corner_loop_vertex_ids,
    fill_boundary_with_living,
    reconstruct_rooms,
    rectify_rooms,
    score_adaptation,
    similarity_align,
)
from geometry import find_shared_wall  # noqa: E402
from schema import SCALE, SCHEMA_VERSION, ROOM_TYPE_MAP, MINIMUM_SIZES, Boundary, Edge, FloorPlan, Room  # noqa: E402
from validator import validate  # noqa: E402
from vlm_refine import (  # noqa: E402
    orchestrate_refinement,
    surface_defects,
    viability_score,
)

ARAP_ITERS = 20  # matches arap_adapt.py's CLI default

# This app's own repo root (packages/floorplan_core/arap_bridge.py -> repo
# root), distinct from FITTING_PROTOTYPE_DIR above — the refinement agent
# (orchestrate_refinement) needs somewhere to write its before/after debug
# renders, since the VLM step sends real PNGs to Ollama, not geometry.
APP_REPO_ROOT = Path(__file__).resolve().parents[2]
REFINE_DEBUG_DIR = APP_REPO_ROOT / "data" / "refine_debug"


# ──────────────────────────────────────────────────────────────────
# v2.1.0 <-> arap_adapt's own dict shapes (mirrors arap_adapt.load_plan /
# load_boundary, generalised from file paths to in-memory dicts/points)
# ──────────────────────────────────────────────────────────────────

def _v21_to_arap_plan(fp: dict) -> dict:
    rooms = []
    for r in fp["rooms"]:
        rtype = r.get("type", r.get("source_type"))
        rooms.append({
            "id": r["id"],
            "type": rtype,
            "label": ROOM_TYPE_MAP.get(rtype, "unknown"),
            "polygon": [(x * SCALE, y * SCALE) for x, y in r["polygon"]],
        })
    ex1, ey1, ex2, ey2 = [v * SCALE for v in fp["entrance"]]
    # fp["boundary"] points are [x, y, dir, is_door] (v2.1.0) — only x,y matter here.
    boundary = [(p[0] * SCALE, p[1] * SCALE) for p in fp["boundary"]]
    return {
        "id": fp.get("id", ""),
        "rooms": rooms,
        "graph_edges": fp.get("edges", []),
        "boundary": boundary,
        "entrance_a": (ex1, ey1),
        "entrance_b": (ex2, ey2),
        "entrance_mid": ((ex1 + ex2) / 2, (ey1 + ey2) / 2),
    }


def _query_to_arap_boundary(boundary_pts, entrance_point, typed_edges=None) -> dict:
    return {
        "name": "live",
        "polygon": [(x * SCALE, y * SCALE) for x, y in boundary_pts],
        "entrance": (entrance_point[0] * SCALE, entrance_point[1] * SCALE),
        # The boundary's own edge classification (external/shared_wall/
        # facing_corridor), carried from a floor-plate unit. ARAP itself needs
        # only geometry, so this is passed through for the compliance layer
        # downstream (the oracle's per-room daylight check) rather than used
        # here — see docs/arrangement-refactor-build-plan.md Step 1/2.
        "typed_edges": typed_edges or [],
    }


def _resolved_edge_types(boundary_pts, entrance_point, typed_edges) -> list:
    """The per-edge types the boundary actually resolves to once supplied
    unit types are merged over the freehand default — returned to the app so
    the classification can be inspected (and, from Step 2, drawn). Mirrors
    what boundary_to_schema hands the compliance layer, in the same edge order
    as the boundary the app sent."""
    from packages.floorplan_core.schema_bridge import boundary_to_schema
    b = boundary_to_schema(boundary_pts, entrance_point, typed_edges)
    return [{"index": te["index"], "type": te["type"]} for te in b.typed_edges]


# ──────────────────────────────────────────────────────────────────
# arap room-list -> schema.py FloorPlan, for rendering at any step
# ──────────────────────────────────────────────────────────────────

def _arap_rooms_to_schema_plan(rooms: list, entrance_mid) -> FloorPlan:
    """Builds a FloorPlan purely from room geometry, not a fixed graph_edges
    structure — rectify_rooms/fill_boundary_with_living/consolidate_living can
    add, drop, or merge rooms relative to the original room-id set, so a
    graph_edges-based approach (matching plan_to_schema in schema_bridge.py)
    would go stale by step 4. Internal edges: every room pair tested for a
    shared wall. External edges: whatever's left uncovered on each room's own
    polygon. Doors aren't derived here — see module docstring."""
    schema_rooms: dict[str, Room] = {
        r["id"]: Room(id=r["id"], type=r["label"], polygon=[tuple(p) for p in r["polygon"]])
        for r in rooms
    }

    edges: dict[str, Edge] = {}
    adjacency: dict[str, list[str]] = {rid: [] for rid in schema_rooms}
    connections: dict[str, list[str]] = {rid: [] for rid in schema_rooms}

    ids = list(schema_rooms.keys())
    eidx = 0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            u, v = ids[a], ids[b]
            shared = find_shared_wall(schema_rooms[u].polygon, schema_rooms[v].polygon)
            if shared:
                start, end = shared
                eid = f"e_{eidx}"
                eidx += 1
                edges[eid] = Edge(id=eid, room_a=u, room_b=v, start=start, end=end, type="internal")
                schema_rooms[u].edges.append(eid)
                schema_rooms[v].edges.append(eid)
                adjacency[u].append(v)
                adjacency[v].append(u)

    def _near(p, q, tol=1.0):
        return abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol

    for rid, room in schema_rooms.items():
        poly = room.polygon
        n = len(poly)
        for i in range(n):
            seg_s, seg_e = poly[i], poly[(i + 1) % n]
            covered = any(
                ((_near(e.start, seg_s) and _near(e.end, seg_e)) or (_near(e.start, seg_e) and _near(e.end, seg_s)))
                for e in edges.values()
                if e.room_b is not None
            )
            if not covered:
                ext_eid = f"ext_{rid}_{i}"
                edges[ext_eid] = Edge(id=ext_eid, room_a=rid, room_b=None, start=seg_s, end=seg_e, type="external")
                room.edges.append(ext_eid)

    return FloorPlan(
        rooms=schema_rooms, edges=edges, adjacency=adjacency, connections=connections,
        entrance_mid=tuple(entrance_mid),
    )


def _fixture(rooms: list, boundary_pts, entrance_point, plan_entrance_mid, label: str, score: dict | None = None) -> dict:
    # Qualified import (not a bare `from schema_bridge import ...`): this module
    # is imported as packages.floorplan_core.arap_bridge, and only REPO_ROOT
    # (not this package's own directory) is guaranteed to be on sys.path.
    from packages.floorplan_core.schema_bridge import boundary_to_schema
    boundary = boundary_to_schema(boundary_pts, entrance_point)
    plan = _arap_rooms_to_schema_plan(rooms, plan_entrance_mid)

    try:
        violations = [asdict(v) for v in validate(plan, boundary)]
    except Exception:
        violations = []  # mid-pipeline geometry (esp. steps 1-3) can be self-intersecting by design

    out = {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "boundary": asdict(boundary),
        "plan": asdict(plan),
        "violations": violations,
    }
    if score is not None:
        out["score"] = score
    return out


# ──────────────────────────────────────────────────────────────────
# The four steps
# ──────────────────────────────────────────────────────────────────

def _corner_handles(V: np.ndarray, V_aligned: np.ndarray, plan: dict, boundary: dict) -> dict:
    """Arc-length handle assignment (2026-08-13 fix, revived from
    files-in/ws/arap_3.py — see arap_adapt.compute_handles_arclength),
    shared by every step that needs handles: pins only the source plan's
    TRUE boundary corners to the target boundary's continuous arc-length
    parametrisation, at the single best-fit rigid rotational offset. This
    replaces compute_handles' corner-forcing DP, which let on-edge/
    T-junction vertices outcompete real corners for target-corner slots on
    a weak retrieval match.

    V must be the pre-alignment mesh (build_mesh's own output, never
    door-extended). V_aligned may have extra rows appended after it (e.g.
    the door vertices in _with_door_vertices) — corner_loop_vertex_ids only
    ever returns indices into the first len(V) rows, so the extension is
    safe to index with regardless."""
    cvids = corner_loop_vertex_ids(V, plan["boundary"])
    return compute_handles_arclength(V_aligned[cvids], cvids, boundary["polygon"])


def _rectify_nonliving(plan: dict, room_vids: list, boundary: dict,
                        V_aligned: np.ndarray, V_def: np.ndarray) -> list:
    """Nonliving rooms rectangle-fitted straight off ARAP's raw output, with
    no rectify_mesh in between (2026-08-13 fix — rectify_mesh's global
    cross-room class-averaging was independently collapsing rooms into
    slivers, on top of the handle-assignment bug; rectify_rooms's own
    per-room bbox/grid-snap/overlap-resolution/grow-to-boundary handles
    this better, validated across all four test boundaries in
    pipeline_batch_test.py). reference_rooms anchors each room's rectangle
    SIZE to its pre-ARAP aligned bbox so ARAP's boundary-pull distortion
    doesn't inflate it — see rectify_rooms' own docstring.

    V_aligned/V_def must already be truncated to the room-mesh vertex count
    (no door-vertex extension) — callers with an extended mesh pass V[:n]."""
    aligned_all = reconstruct_rooms(plan["rooms"], room_vids, V_aligned)
    aligned_nonliving = [r for r in aligned_all if r["label"] != "living"]
    all_rooms = reconstruct_rooms(plan["rooms"], room_vids, V_def)
    nonliving = [r for r in all_rooms if r["label"] != "living"]
    return rectify_rooms(nonliving, boundary["polygon"], reference_rooms=aligned_nonliving)


def _mesh_state(fp: dict, boundary_pts, entrance_point):
    """Shared setup through build_mesh + similarity_align + handles + arap_solve
    — every step needs at least this much."""
    plan = _v21_to_arap_plan(fp)
    boundary = _query_to_arap_boundary(boundary_pts, entrance_point)
    V, edges, room_vids = build_mesh(plan["rooms"])
    V_aligned = similarity_align(V, plan, boundary)
    handles = _corner_handles(V, V_aligned, plan, boundary)
    V_def = arap_solve(V_aligned, edges, handles, n_iters=ARAP_ITERS)
    return plan, boundary, V, edges, room_vids, V_aligned, V_def


def run_align_and_solve(fp: dict, boundary_pts, entrance_point) -> dict:
    plan, boundary, V, edges, room_vids, V_aligned, V_def = _mesh_state(fp, boundary_pts, entrance_point)
    rooms = reconstruct_rooms(plan["rooms"], room_vids, V_def)
    return _fixture(rooms, boundary_pts, entrance_point, plan["entrance_mid"], "1_align_arap")


def run_rectify(fp: dict, boundary_pts, entrance_point) -> dict:
    plan, boundary, V, edges, room_vids, V_aligned, V_def = _mesh_state(fp, boundary_pts, entrance_point)
    nonliving = _rectify_nonliving(plan, room_vids, boundary, V_aligned, V_def)
    return _fixture(nonliving, boundary_pts, entrance_point, plan["entrance_mid"], "2_rectify")


def run_reconstruct(fp: dict, boundary_pts, entrance_point) -> dict:
    plan, boundary, V, edges, room_vids, V_aligned, V_def = _mesh_state(fp, boundary_pts, entrance_point)
    nonliving = _rectify_nonliving(plan, room_vids, boundary, V_aligned, V_def)
    return _fixture(nonliving, boundary_pts, entrance_point, plan["entrance_mid"], "3_reconstruct")


def run_refine(fp: dict, boundary_pts, entrance_point) -> dict:
    plan, boundary, V, edges, room_vids, V_aligned, V_def = _mesh_state(fp, boundary_pts, entrance_point)
    nonliving = _rectify_nonliving(plan, room_vids, boundary, V_aligned, V_def)
    adapted = consolidate_living(
        fill_boundary_with_living(nonliving, boundary["polygon"]),
        boundary["polygon"],
    )
    score = score_adaptation(adapted, boundary["polygon"], plan["entrance_mid"], plan["rooms"])
    return _fixture(adapted, boundary_pts, entrance_point, plan["entrance_mid"], "4_refine", score=score)


STEPS = {
    "align": run_align_and_solve,
    "rectify": run_rectify,
    "reconstruct": run_reconstruct,
    "refine": run_refine,
}


# ──────────────────────────────────────────────────────────────────
# Canvas-native debug view (2026-08-11) — the four schema-fixture STEPS
# above render through a second conversion (arap rooms -> schema.py
# FloorPlan -> plan-drawing-renderer.js) that adds a layer between the
# geometry and what gets looked at, making it hard to tell a real fitting
# bug from a rendering-path bug. This renders directly into the same v2.1.0
# room-list shape (int type, [0,1]-normalised polygon) the app's own
# canvas-renderer.js already draws via drawFP — no new renderer involved.
# Starting with similarity_align only (build_mesh + similarity_align), per
# request: verify the SiGraDi-paper alignment step in isolation before
# trusting anything built on top of it (ARAP solve, rectify, etc.).
# ──────────────────────────────────────────────────────────────────

def _source_type_by_id(fp: dict) -> dict:
    """RPlan-native type per room id, read from the raw v2.1.0 input — the
    numbering apps/web/src/constants.js (RCOLS, ROOM_ABBR) expects on the
    canvas. Distinct from arap_adapt.py's own `type`/ROOM_TYPE_MAP, which is
    fitting-prototype's internal enumeration (bedroom subtypes consolidated
    to one slot, so Kitchen/Bathroom/Dining sit one integer higher than in
    RPlan's scheme) — correct for ARAP's own label-driven logic (rectify_
    rooms, fill_boundary_with_living, room-size standards), wrong if sent to
    the browser as-is, which is what produced the 2026-08-12 mislabelling:
    two consistent enumerations colliding on the same wire field."""
    return {r["id"]: r.get("source_type", r.get("type")) for r in fp["rooms"]}


def _rooms_to_canvas(rooms: list, door=None, source_type_by_id: dict | None = None) -> dict:
    out = {
        "rooms": [
            {
                "id": r["id"],
                "type": (source_type_by_id or {}).get(r["id"], r["type"]),
                "polygon": [[x / SCALE, y / SCALE] for x, y in r["polygon"]],
            }
            for r in rooms
        ],
    }
    if door is not None:
        out["door"] = [v / SCALE for v in door]
    return out


def _nearest_vertex(V: np.ndarray, pt) -> int:
    d = np.hypot(V[:, 0] - pt[0], V[:, 1] - pt[1])
    return int(np.argmin(d))


def _with_door_vertices(V: np.ndarray, edges: list, plan: dict):
    """Appends the plan's own entrance-rectangle corners as two extra mesh
    vertices, each connected by a single edge to its nearest real vertex, so
    the door rides through similarity_align/arap_solve via the same transform
    as everything else rather than being drawn at its stale pre-transform
    position (2026-08-11 fix — the door marker wasn't tracked through the
    pipeline at all before this). n = original vertex count; the two new
    vertices are indices n and n+1. corner_loop_vertex_ids/compute_handles_arclength
    must be called on the ORIGINAL V (n rows only) so these two never get
    treated as boundary corners — they should follow their neighbour's local
    rigidity, not get pinned to the target boundary themselves."""
    n = len(V)
    ea = np.array(plan["entrance_a"], dtype=float)
    eb = np.array(plan["entrance_b"], dtype=float)
    V_ext = np.vstack([V, ea, eb])
    edges_ext = edges + [(_nearest_vertex(V, ea), n), (_nearest_vertex(V, eb), n + 1)]
    return V_ext, edges_ext, n


def run_align_canvas(fp: dict, boundary_pts, entrance_point, typed_edges=None) -> dict:
    plan = _v21_to_arap_plan(fp)
    boundary = _query_to_arap_boundary(boundary_pts, entrance_point)
    V, edges, room_vids = build_mesh(plan["rooms"])
    V_ext, _, n = _with_door_vertices(V, edges, plan)
    V_aligned = similarity_align(V_ext, plan, boundary)
    rooms = reconstruct_rooms(plan["rooms"], room_vids, V_aligned[:n])
    door = [*V_aligned[n], *V_aligned[n + 1]]
    return _rooms_to_canvas(rooms, door, source_type_by_id=_source_type_by_id(fp))


def run_arap_solve_canvas(fp: dict, boundary_pts, entrance_point, typed_edges=None) -> dict:
    """align (build_mesh + similarity_align) -> corner_loop_vertex_ids ->
    compute_handles_arclength -> arap_solve. The frontend gates this behind
    align having already been run for the selected result — not enforced
    here, since arap_solve always needs an aligned mesh as its starting
    point regardless (this just makes that dependency an explicit UI step
    rather than something silently re-run every time)."""
    plan = _v21_to_arap_plan(fp)
    boundary = _query_to_arap_boundary(boundary_pts, entrance_point)
    V, edges, room_vids = build_mesh(plan["rooms"])
    V_ext, edges_ext, n = _with_door_vertices(V, edges, plan)
    V_aligned = similarity_align(V_ext, plan, boundary)
    # corner_loop_vertex_ids/compute_handles_arclength run on the original
    # (pre-door-extension) arrays — the two door vertices must never be
    # treated as handles themselves, only follow their neighbour via the
    # one connecting edge (see _with_door_vertices docstring).
    handles = _corner_handles(V, V_aligned, plan, boundary)
    V_def = arap_solve(V_aligned, edges_ext, handles, n_iters=ARAP_ITERS)
    rooms = reconstruct_rooms(plan["rooms"], room_vids, V_def[:n])
    door = [*V_def[n], *V_def[n + 1]]
    return _rooms_to_canvas(rooms, door, source_type_by_id=_source_type_by_id(fp))


def run_rectify_canvas(fp: dict, boundary_pts, entrance_point, typed_edges=None) -> dict:
    """align -> arap_solve -> rectify_rooms, straight off ARAP's raw output
    with no rectify_mesh in between (2026-08-13 fix — see
    _rectify_nonliving). The frontend gates this behind arap_solve having
    already been run for the selected result, same as solve is gated
    behind align.

    Living room is excluded here, matching arap_adapt.py's own main(): it's
    filled as the residual space by fill_boundary_with_living, a step this
    one doesn't reach — so the debug view at this stage genuinely shows
    every other room rectified, with living deliberately absent rather than
    a rendering gap."""
    plan = _v21_to_arap_plan(fp)
    boundary = _query_to_arap_boundary(boundary_pts, entrance_point)
    V, edges, room_vids = build_mesh(plan["rooms"])
    V_ext, edges_ext, n = _with_door_vertices(V, edges, plan)
    V_aligned = similarity_align(V_ext, plan, boundary)
    handles = _corner_handles(V, V_aligned, plan, boundary)
    V_def = arap_solve(V_aligned, edges_ext, handles, n_iters=ARAP_ITERS)

    nonliving = _rectify_nonliving(plan, room_vids, boundary, V_aligned[:n], V_def[:n])

    door = [*V_def[n], *V_def[n + 1]]
    return _rooms_to_canvas(nonliving, door, source_type_by_id=_source_type_by_id(fp))


def _fill_consolidated_canvas_state(fp: dict, boundary_pts, entrance_point, typed_edges=None):
    """align -> arap_solve -> rectify_rooms -> fill_boundary_with_living ->
    consolidate_living, no rectify_mesh (see run_rectify_canvas /
    _rectify_nonliving). This is the last step arap_adapt.py's own main()
    runs before scoring — living room now fills whatever space the
    rectified rooms left, so the result should cover the whole boundary
    with no gaps. Shared by run_fill_consolidate_canvas and the refinement
    stage below (run_oracle_canvas, run_agent_refine_canvas), which both
    need this exact fully-tessellated room list — it's what vlm_refine.py
    was built and validated against."""
    plan = _v21_to_arap_plan(fp)
    boundary = _query_to_arap_boundary(boundary_pts, entrance_point, typed_edges)
    V, edges, room_vids = build_mesh(plan["rooms"])
    V_ext, edges_ext, n = _with_door_vertices(V, edges, plan)
    V_aligned = similarity_align(V_ext, plan, boundary)
    handles = _corner_handles(V, V_aligned, plan, boundary)
    V_def = arap_solve(V_aligned, edges_ext, handles, n_iters=ARAP_ITERS)

    nonliving = _rectify_nonliving(plan, room_vids, boundary, V_aligned[:n], V_def[:n])
    adapted = consolidate_living(
        fill_boundary_with_living(nonliving, boundary["polygon"]),
        boundary["polygon"],
    )

    door = [*V_def[n], *V_def[n + 1]]
    return plan, boundary, adapted, door


def run_fill_consolidate_canvas(fp: dict, boundary_pts, entrance_point, typed_edges=None) -> dict:
    """The frontend gates this behind rectify having already been run for
    the selected result, same as rectify is gated behind solve. See
    _fill_consolidated_canvas_state for the pipeline itself."""
    _plan, _boundary, adapted, door = _fill_consolidated_canvas_state(
        fp, boundary_pts, entrance_point, typed_edges
    )
    out = _rooms_to_canvas(adapted, door, source_type_by_id=_source_type_by_id(fp))
    out["boundary_typed_edges"] = _resolved_edge_types(boundary_pts, entrance_point, typed_edges)
    return out


def _json_safe(obj):
    """vlm_refine.py's defect/log dicts are built for printing, not JSON —
    (idx, label) and (x, y) tuples, occasional numpy scalars from shapely/
    numpy geometry ops. Recursively convert tuples to lists and numpy
    scalars to native types so the response serialises cleanly."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def run_oracle_canvas(fp: dict, boundary_pts, entrance_point, typed_edges=None) -> dict:
    """align -> ... -> fill -> consolidate, then the geometry oracle
    (vlm_refine.surface_defects, docs/fitting-prototype/vlm-refinement-
    build-plan.md Phase A Steps 1-2) measures the result and ranks
    candidate defects — compliance, habitability slivers, filler-hidden
    gaps, wall alignment. No changes are made here; this is the 'what does
    the oracle see' debug view ahead of the agent step below. The frontend
    gates this behind fill having already been run for the selected
    result, same as fill is gated behind rectify."""
    plan, boundary, adapted, door = _fill_consolidated_canvas_state(
        fp, boundary_pts, entrance_point, typed_edges
    )
    # Step 2 (daylight/access) needs the boundary's resolved per-edge types with
    # their coordinates, in the same mm space as the rooms. Only pass them when a
    # unit actually supplied types — a freehand boundary is left to the geometry
    # checks, so the single-unit debug oracle keeps its current behaviour.
    resolved_edges = None
    if typed_edges:
        from packages.floorplan_core.schema_bridge import boundary_to_schema
        resolved_edges = boundary_to_schema(boundary_pts, entrance_point, typed_edges).typed_edges
    defects = surface_defects(adapted, boundary["polygon"], typed_edges=resolved_edges)
    # Step 3: one arrangement-level viability scalar + breakdown, computed from
    # the same defects so it never disagrees with the per-room flags above.
    score = viability_score(adapted, boundary["polygon"], plan.get("entrance_mid"), defects=defects)
    out = _rooms_to_canvas(adapted, door, source_type_by_id=_source_type_by_id(fp))
    out["defects"] = _json_safe(defects)
    out["score"] = _json_safe(score)
    out["boundary_typed_edges"] = _resolved_edge_types(boundary_pts, entrance_point, typed_edges)
    return out


def run_agent_refine_canvas(fp: dict, boundary_pts, entrance_point, typed_edges=None) -> dict:
    """align -> ... -> fill -> consolidate -> orchestrate_refinement
    (vlm_refine.py Phase C, build-complete 2026-07-14): the oracle surfaces
    candidate defects, a deterministic layer closes gaps under an
    invariant guard (containment/no-overlap/topology — regressions ~0 by
    construction, validated at 0/60 in the paper's batch eval), and a VLM
    paired-preference gate keeps or reverts each guarded fix by comparing
    before/after renders two ways (order-robust). Bounded to 8 iterations.
    Requires Ollama serving qwen3-vl:8b at localhost:11434 — true on this
    machine, NOT true of the Nectar-hosted deployment, which is a separate
    open question (see thread.md) rather than something this function
    works around. Debug renders + the orchestration log land in
    REFINE_DEBUG_DIR/<query_id>/ — needed by the VLM round-trip itself
    (it has to look at real PNGs), and kept afterward so what the agent
    did and why is inspectable, same spirit as the oracle_verify.py /
    phase_c_demo.py harnesses this was validated against. The frontend
    gates this behind fill (or oracle) having already been run."""
    plan, boundary, adapted, door = _fill_consolidated_canvas_state(
        fp, boundary_pts, entrance_point, typed_edges
    )
    query_id = uuid.uuid4().hex[:8]
    out_dir = REFINE_DEBUG_DIR / query_id
    result = orchestrate_refinement(
        adapted, boundary["polygon"], plan["entrance_mid"],
        out_dir=str(out_dir), query_id=query_id,
    )
    out = _rooms_to_canvas(result["rooms_refined"], door, source_type_by_id=_source_type_by_id(fp))
    out["verified"] = _json_safe(result["verified"])
    out["log"] = _json_safe(result["log"])
    out["query_id"] = query_id
    return out


def rerank_unit(variants: list, boundary_pts, entrance_point, typed_edges=None) -> dict:
    """Step 4 — arrangement-level viability rerank. Given a unit's top retrieved
    candidates, each in both mirror states (variants = [{fp, tag, candidate,
    mirror}]), adapt every one against the boundary and score it with the Step 3
    viability score, then return the most viable arrangement plus the full scored
    table. This is the agent's first arrangement-level judgement: the blind stage-1
    commitment — take retrieval's rank-1 in its descriptor-optimal orientation —
    is replaced by an explicit scored choice made BEFORE anything irreversible, so
    a mirrored or lower-ranked candidate carrying fewer daylight/access flags can
    win. Descriptor distance chose the orientation; viability overrides it. See
    docs/arrangement-refactor-build-plan.md Step 4."""
    resolved_edges = None
    if typed_edges:
        from packages.floorplan_core.schema_bridge import boundary_to_schema
        resolved_edges = boundary_to_schema(boundary_pts, entrance_point, typed_edges).typed_edges

    table, best = [], None
    for v in variants or []:
        fp = v.get("fp")
        if not fp or not fp.get("rooms"):
            continue
        plan, boundary, adapted, door = _fill_consolidated_canvas_state(
            fp, boundary_pts, entrance_point, typed_edges
        )
        defects = surface_defects(adapted, boundary["polygon"], typed_edges=resolved_edges)
        score = viability_score(
            adapted, boundary["polygon"], plan.get("entrance_mid"), defects=defects)
        row = {
            "tag":       v.get("tag"),
            "candidate": v.get("candidate"),
            "mirror":    v.get("mirror"),
            "score":     score["score"],
            "breakdown": score["breakdown"],
        }
        table.append(row)
        if best is None or score["score"] > best["score"]:
            best = {"score": score["score"], "row": row, "fp": fp,
                    "adapted": adapted, "door": door, "defects": defects}

    if best is None:
        return {"error": "no scorable candidates"}

    table.sort(key=lambda r: r["score"], reverse=True)
    out = _rooms_to_canvas(best["adapted"], best["door"],
                           source_type_by_id=_source_type_by_id(best["fp"]))
    out["defects"] = _json_safe(best["defects"])
    out["winner"] = _json_safe(best["row"])
    out["winner_tag"] = best["row"]["tag"]
    out["table"] = _json_safe(table)
    out["boundary_typed_edges"] = _resolved_edge_types(boundary_pts, entrance_point, typed_edges)
    return out


def run_arrange_canvas(fp: dict, boundary_pts, entrance_point, typed_edges=None) -> dict:
    """Step 5 — arrangement-level placement by rectangular dualization
    (docs/arrangement-refactor-build-plan.md). The one step that does NOT run
    the ARAP chain: instead of warping the retrieved mesh to the boundary, it
    re-dimensions the retrieved plan's topology as a fresh axis-aligned
    partition of the target boundary — retrieval chose the arrangement, this
    realises it. Independent of align/solve/rectify/fill; reads the raw
    retrieved rooms + relation-coded edges straight off the v2.1.0 input.

    Returns the partition in the same canvas room shape plus a report:
    feasibility, grid size, cells absorbed into the flexible (living/dining)
    residual, per-room area vs. minimum, and any source adjacency the
    partition could not realise. See rectangular_arrange.arrange."""
    from packages.floorplan_core.rectangular_arrange import arrange
    plan = _v21_to_arap_plan(fp)
    boundary = _query_to_arap_boundary(boundary_pts, entrance_point, typed_edges)
    # Resolve the boundary's edge types (supplied types win, entrance edge
    # defaults to facing_corridor, the rest to external) and hand arrange a
    # per-edge-index type list so its daylight check has real edges to read.
    resolved = _resolved_edge_types(boundary_pts, entrance_point, typed_edges)
    n_edges = len(boundary["polygon"])
    edge_types_list = [None] * n_edges
    for e in resolved:
        idx = e.get("index")
        if idx is not None and 0 <= idx < n_edges:
            edge_types_list[idx] = e.get("type")
    result = arrange(
        plan["rooms"], boundary["polygon"],
        source_edges=plan.get("graph_edges"), entrance_mid=plan.get("entrance_mid"),
        target_entrance=boundary.get("entrance"), edge_types=edge_types_list,
    )
    out = _rooms_to_canvas(result["rooms"], source_type_by_id=_source_type_by_id(fp))
    out["feasible"] = bool(result["feasible"])
    out["mode"] = result["mode"]
    out["report"] = _json_safe(result["report"])
    out["boundary_typed_edges"] = resolved
    return out


def _program_from_req(req: dict) -> list:
    """The room program implied by the requirements. Bedrooms are one master +
    the rest ordinary; a study is sized as a bedroom (it needs a window too).
    Balconies are external and excluded from the interior feasibility check."""
    n_bed = int(req.get("bedrooms", 0) or 0)
    n_bath = int(req.get("bathrooms", 0) or 0)
    prog = ["living", "kitchen"]
    if n_bed >= 1:
        prog.append("master_bedroom")
        prog += ["bedroom"] * (n_bed - 1)
    prog += ["bathroom"] * n_bath
    if req.get("study"):
        prog.append("bedroom")
    if req.get("storage"):
        prog.append("storage")
    return prog


def feasibility_check(boundary_pts, entrance_point, req, typed_edges=None) -> dict:
    """Stage 0 pre-analysis — can the requested program even be placed in this
    boundary, before retrieval? Two cheap deterministic checks against
    MINIMUM_SIZES, advisory rather than gating:

      - AREA: Σ minimum room areas vs boundary area. A hard lower bound —
        passing does not guarantee a fit (no circulation allowance), but failing
        guarantees the boundary is too small for the program.
      - FRONTAGE: Σ bedroom window-widths + one living width vs the total length
        of EXTERNAL edges. Each bedroom needs its own window and they cannot
        share a span; the common-area cluster needs one direct light source
        (secondary daylight covers the rest). This is the too-few-external-faces
        failure — e.g. two bedrooms onto a single short frontage — caught before
        a plan is ever retrieved."""
    from shapely.geometry import Polygon as SPoly
    poly_mm = [(x * SCALE, y * SCALE) for x, y in boundary_pts]
    bpoly = SPoly(poly_mm)
    avail_area = bpoly.area

    program = _program_from_req(req)
    req_area = sum(MINIMUM_SIZES.get(l, {}).get("area", 0) for l in program)

    resolved = _resolved_edge_types(boundary_pts, entrance_point, typed_edges)
    n = len(poly_mm)
    ext_len = 0.0
    for e in resolved:
        if e.get("type") == "external":
            i = e["index"]
            a, b = poly_mm[i], poly_mm[(i + 1) % n]
            ext_len += ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5

    strict = [l for l in program if l in ("bedroom", "master_bedroom")]
    req_frontage = sum(MINIMUM_SIZES.get(l, {}).get("min_edge", 0) for l in strict)
    req_frontage += MINIMUM_SIZES.get("living", {}).get("min_edge", 0)

    area_ok = req_area <= avail_area
    frontage_ok = req_frontage <= ext_len + 1.0
    return {
        "feasible": bool(area_ok and frontage_ok),
        "program": program,
        "area": {
            "ok": bool(area_ok),
            "required_m2": round(req_area / 1e6, 1),
            "available_m2": round(avail_area / 1e6, 1),
        },
        "frontage": {
            "ok": bool(frontage_ok),
            "required_m": round(req_frontage / 1000, 1),
            "available_m": round(ext_len / 1000, 1),
        },
    }


CANVAS_STEPS = {
    "align": run_align_canvas,
    "solve": run_arap_solve_canvas,
    "rectify": run_rectify_canvas,
    "fill": run_fill_consolidate_canvas,
    "oracle": run_oracle_canvas,
    "agent": run_agent_refine_canvas,
    "arrange": run_arrange_canvas,
}
