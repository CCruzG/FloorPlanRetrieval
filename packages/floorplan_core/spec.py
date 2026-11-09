"""
spec.py — the Spec: one module for its field table, describe, fit, options and checker.

The Spec is the formal description of a floor plan (ARCHITECTURE.md, "The Spec").
Fed to the Generator it produces the plan; moves edit only the Spec. Flat dict:

    boundary, edge_types, mm_per_unit            given
    nodes[]            {id, room_id, kind, label, pos}
    adjacencies[]      {u, v, connection}        connection: 0 wall, 1 door, 2 open
    demands{id}        {min_edge, min_area, aspect_max, area_frac,
                        max_short?, max_area?}
    boundary_demands   {id: [edge index]}
    entrance, entry_room, precedence

Public API:
    FIELDS                    field table (name, kind, strength, moves)
    describe(plan, mm_per_unit) -> Spec     a plan as it stands, in its own boundary
    fit(source_spec, boundary, ...) -> Spec  the source Spec fitted to a new boundary
    build_options(...)        -> options dict (solver and Generator switches)
    spec_problems(spec)       -> [problem str]; empty = valid
    validate_spec(spec)       -> None; raises ValueError listing every problem
    connection_of(edge)       -> 0 | 1 | 2 for any plan-edge dict
"""

import copy
import math

from shapely.geometry import LineString, Point, Polygon

from packages.floorplan_core.topo_signature import _label
import packages.floorplan_core.rectangular_arrange  # noqa: F401  (puts schema/ on sys.path)
from schema import MINIMUM_SIZES  # noqa: E402  (path set by rectangular_arrange)

# ── field table ──────────────────────────────────────────────────────────────
# (name, kind, strength, moves that may edit it). Mirrors ARCHITECTURE.md.
FIELDS = [
    ("boundary",             "given",  "must",   ()),
    ("edge_types",           "given",  "must",   ()),
    ("mm_per_unit",          "given",  "must",   ()),
    ("nodes[].id",           "design", "must",   ()),
    ("nodes[].room_id",      "design", "must",   ()),
    ("nodes[].kind",         "design", "must",   ()),
    ("nodes[].label",        "design", "must",   ("Swap",)),
    ("nodes[].pos",          "design", "guide",  ()),
    ("adjacencies[]",        "design", "should", ("SetConnection",)),
    ("demands[].min_edge",   "design", "should", ("label",)),
    ("demands[].min_area",   "design", "should", ("label",)),
    ("demands[].max_short",  "design", "should", ("label",)),
    ("demands[].max_area",   "design", "should", ("label",)),
    ("demands[].aspect_max", "design", "should", ("label",)),
    ("demands[].area_frac",  "design", "guide",  ("Reallocate", "label")),
    ("boundary_demands",     "design", "should", ()),
    ("entrance",             "design", "must",   ("SlideEntrance",)),
    ("entry_room",           "design", "should", ("SlideEntrance",)),
    ("precedence",           "design", "should", ("SetPrecedence",)),
]

NODE_KINDS = ("whole", "primary", "circ", "stor", "nook")
# node_graph._kind can also return "residual" for an unnamed part; the Spec
# carries it rather than rejecting the plan (reported, not in the ARCHITECTURE enum).
_EXTRA_KINDS = ("residual",)

CIRCULATION_LABELS = ("hallway", "corridor")
ASPECT_MAX_DEFAULT = 2.5        # habitable rooms
ASPECT_MAX_CIRCULATION = 6.0    # hallway, corridor
MAX_SHORT_CIRCULATION_MM = 1500.0   # decision 2026-11-04; not enforced yet

# validate_spec tolerances
ENTRANCE_TOL_FRAC = 0.01        # entrance endpoint within 1 % of the bbox diagonal of the boundary
AREA_FRAC_TOL = 0.02            # |sum(area_frac) - 1|

# Solver defaults, equal to layout_cpsat's TIME_LIMIT_S / NUM_WORKERS / RANDOM_SEED.
CPSAT_SEED = 2026
CPSAT_WORKERS = 8
CPSAT_TIME_S = 5.0


# ── edge encoding ────────────────────────────────────────────────────────────

_SEP_TO_CONN = {"wall": 0, "door": 1, "open": 2}


def connection_of(edge):
    """Connection code of a plan-edge dict: 0 wall, 1 door, 2 open.
    `separation`/`wall_type` wins (SetConnection wrote it on plan edges), then
    `connection` 1/2, then `open: True` and `type` door/open (the carry rule of
    rectangular_arrange.reconcile_edges)."""
    t = edge.get("separation") or edge.get("wall_type")
    if t in _SEP_TO_CONN:
        return _SEP_TO_CONN[t]
    c = edge.get("connection")
    if c in (1, 2):
        return c
    if edge.get("open") is True:
        return 2
    t = edge.get("type")
    if t in ("door", "open"):
        return _SEP_TO_CONN[t]
    return 0


# ── builder ──────────────────────────────────────────────────────────────────

def _poly_area(poly):
    n = len(poly)
    return abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
                   for i in range(n))) / 2.0


def _label_point(poly):
    """A point inside the polygon: its centroid, else a representative point (L/T rooms)."""
    g = Polygon(poly)
    if not g.is_valid:
        g = g.buffer(0)
    pt = g.centroid
    if not g.contains(pt):
        pt = g.representative_point()
    return [float(pt.x), float(pt.y)]


def describe(plan, mm_per_unit=None):
    """The Spec of a plan as it stands: its own boundary, entrance and positions.

    Fed to the Generator, this Spec should give back something close to the plan
    (tools/diag_round_trip.py measures how close). Nothing here refers to a target.

    From the plan: nodes, adjacencies, area_frac, entry_room (the labelled door,
    node_graph.entry_room_of). From norms by label: min_edge, min_area, aspect_max,
    max_short. From the plan's geometry: pos (a point inside each room),
    boundary_demands (boundary edges each room really fronts), entrance (as a point
    on the boundary, the form a drawn entrance takes). edge_types: the plan records
    none, so the default applies (entrance edge corridor, others external).

    A node-expanded plan (node_graph.expand_plan) carries `room_id` and `node_kind`
    on each room; a plain room is its own room: room_id = id, kind = "whole". The
    size rule for an open-plan space (decision 2026-11-03) reads those fields, not
    the id text: the primary keeps `min_edge` and takes its area share of
    `min_area`; hallway arms keep their width and no area; other members (nook,
    storage) carry no demand.
    """
    nodes, demands, src_area = [], {}, {}
    for r in plan.get("rooms") or []:
        rid = r.get("id")
        poly = r.get("polygon")
        if not poly or len(poly) < 3:
            continue
        label = _label(r)
        nodes.append({"id": rid,
                      "room_id": r.get("room_id", rid),
                      "kind": r.get("node_kind") or "whole",
                      "label": label,
                      "pos": _label_point(poly)})
        ms = MINIMUM_SIZES.get(label, {})
        circ = str(label).lower() in CIRCULATION_LABELS
        d = {"min_edge": ms.get("min_edge", 0.0),
             "min_area": ms.get("area", 0.0),
             "aspect_max": ASPECT_MAX_CIRCULATION if circ else ASPECT_MAX_DEFAULT}
        if circ:
            d["max_short"] = MAX_SHORT_CIRCULATION_MM
        demands[rid] = d
        src_area[rid] = _poly_area(poly)

    # open-plan groups: nodes of one room that are not "whole"
    groups = {}
    for n in nodes:
        if n["kind"] != "whole":
            groups.setdefault(n["room_id"], []).append(n)
    for members in groups.values():
        prim = next((m["id"] for m in members if m["kind"] == "primary"), None)
        g_area = sum(src_area[m["id"]] for m in members) or 1.0
        for m in members:
            d = demands[m["id"]]
            if m["id"] == prim:
                d["min_area"] *= src_area[m["id"]] / g_area
            elif m["label"] in CIRCULATION_LABELS:
                d["min_area"] = 0.0
            else:
                d["min_edge"] = 0.0
                d["min_area"] = 0.0
    total_area = sum(src_area.values()) or 1.0
    for rid, d in demands.items():
        d["area_frac"] = src_area[rid] / total_area

    # adjacencies: connection only (0 wall, 1 door, 2 open); edges to a room that
    # is not a node (no polygon) or a self-loop are dropped
    ids = {str(n["id"]) for n in nodes}
    adjacencies = []
    for e in plan.get("edges") or []:
        u, v = e.get("u"), e.get("v")
        if u is None or v is None or u == v or str(u) not in ids or str(v) not in ids:
            continue
        adjacencies.append({"u": u, "v": v, "connection": connection_of(e)})

    from packages.floorplan_core.node_graph import entry_room_of
    from packages.floorplan_core.topo_signature import default_edge_types, exact_room_edges

    boundary = [[float(q[0]), float(q[1])] for q in plan.get("boundary") or []]
    entrance = plan.get("entrance")
    edge_types, boundary_demands = None, {}
    if len(boundary) >= 3:
        if entrance and len(entrance) >= 4:
            mid = LineString(boundary + boundary[:1]).interpolate(
                LineString(boundary + boundary[:1]).project(
                    Point((entrance[0] + entrance[2]) / 2.0, (entrance[1] + entrance[3]) / 2.0)))
            entrance = [mid.x, mid.y, mid.x, mid.y]
        edge_types = default_edge_types(boundary, entrance[:2] if entrance else None)
        for rid, contacts in exact_room_edges(plan, boundary, edge_types).items():
            idxs = sorted({c.get("target_edge_index") for c in contacts
                           if c.get("target_edge_index") is not None})
            if idxs:
                boundary_demands[str(rid)] = idxs

    return {
        "boundary": boundary,
        "edge_types": edge_types,
        "mm_per_unit": mm_per_unit,
        "nodes": nodes,
        "adjacencies": adjacencies,
        "demands": demands,
        "boundary_demands": boundary_demands,
        "entrance": entrance,
        "entry_room": entry_room_of(plan),
        "precedence": [],
    }


def fit(source_spec, boundary, edge_types=None, entrance=None, mm_per_unit=None,
        node_pos=None, topo=None):
    """The source Spec fitted to a new boundary. Pure: source_spec is not changed.

    Replaced with the input: boundary, edge_types, entrance, mm_per_unit.
    Re-derived for the input: pos (node_pos, source positions in the target frame;
    a node without one keeps its source pos) and boundary_demands (from
    topo["room_edges"], the perimeter alignment). Kept: nodes' identity and label,
    adjacencies, demands, entry_room (topo["entry_room"], the nearest room, is only
    the fallback), precedence.
    """
    spec = copy.deepcopy(source_spec)
    spec["boundary"] = [[float(x), float(y)] for x, y in boundary]
    spec["edge_types"] = edge_types
    spec["mm_per_unit"] = mm_per_unit
    if entrance is not None:
        spec["entrance"] = entrance
    node_pos = node_pos or {}
    for n in spec["nodes"]:
        pos = node_pos.get(str(n["id"]))
        if pos is not None:
            n["pos"] = [float(pos[0]), float(pos[1])]
    boundary_demands = {}
    for rid, contacts in ((topo or {}).get("room_edges") or {}).items():
        idxs = sorted({c.get("target_edge_index") for c in contacts
                       if c.get("target_edge_index") is not None})
        if idxs:
            boundary_demands[str(rid)] = idxs
    spec["boundary_demands"] = boundary_demands
    if spec.get("entry_room") is None:
        spec["entry_room"] = (topo or {}).get("entry_room")
    return spec


def build_options(core_placement=False, core_block=True, area_lift=True,
                  source_rooms=None, pair_weights=None, cpsat_seed=CPSAT_SEED,
                  cpsat_workers=CPSAT_WORKERS, cpsat_time_s=CPSAT_TIME_S,
                  cpsat_det_time=None, source_grid=False, door_width_mm=900.0,
                  door_jamb_mm=100.0, door_min_wall_mm=1000.0, entrance_door_mm=900.0,
                  window_min_mm=600.0, window_jamb_mm=300.0, window_fill=0.6):
    """Generator and solver switches. Not part of the Spec: no move edits them.

    core_placement : False|True|"greedy"|"cpsat"|"cpsat_dim" (step 0)
    core_block     : step-3 min_edge on a per-room core block
    area_lift      : step-3 min_area lift
    source_rooms   : {str(id): polygon} source arrangement in the target frame
    pair_weights   : {(a, b): weight} soft wall adjacency
    cpsat_*        : CP-SAT seed, worker count, time limit (s)
    cpsat_det_time : None, or a deterministic-time limit that replaces the
                     wall-clock limit (gate runs use it)
    source_grid    : experiment 2026-11-05. True: grid lines at the source_rooms
                     walls, cells owned by the source room, gap widths kept close
                     to the source. "flex": the same with the slack-by-flex solve.
    door_*_mm      : door derivation (floorplan_core/doors.py): width, near-jamb offset
                     from the wall end, shortest wall that takes a door, entrance width
    window_*       : window derivation (floorplan_core/windows.py): shortest run that
                     takes a window, clear distance to run ends and doors, share of
                     the free run that is glazed
    """
    return {
        "core_placement": core_placement,
        "core_block": core_block is not False,
        "area_lift": area_lift is not False,
        "source_rooms": source_rooms,
        "pair_weights": pair_weights,
        "cpsat_seed": cpsat_seed,
        "cpsat_workers": cpsat_workers,
        "cpsat_time_s": cpsat_time_s,
        "cpsat_det_time": cpsat_det_time,
        "source_grid": source_grid,
        "door_width_mm": door_width_mm,
        "door_jamb_mm": door_jamb_mm,
        "door_min_wall_mm": door_min_wall_mm,
        "entrance_door_mm": entrance_door_mm,
        "window_min_mm": window_min_mm,
        "window_jamb_mm": window_jamb_mm,
        "window_fill": window_fill,
    }


# ── validation ───────────────────────────────────────────────────────────────

def _dist_point_segment(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def _dist_to_boundary(p, boundary):
    n = len(boundary)
    return min(_dist_point_segment(p, boundary[i], boundary[(i + 1) % n]) for i in range(n))


def spec_problems(spec):
    """Check the invariants of ARCHITECTURE.md. Returns a list of problem strings
    (empty = valid). Invariants: every id in demands, adjacencies,
    boundary_demands, entry_room, precedence exists in nodes; one edge type per
    boundary edge and indices in range; connection in 0/1/2; each minimum <= its
    maximum; entrance on the boundary (None allowed); area_frac totals about 1."""
    problems = []
    nodes = spec.get("nodes")
    if not isinstance(nodes, list):
        return ["nodes: missing or not a list"]
    ids = [str(n.get("id")) for n in nodes]
    idset = set(ids)
    if len(idset) != len(ids):
        problems.append("nodes: duplicate id")
    for n in nodes:
        for k in ("id", "room_id", "kind", "label", "pos"):
            if k not in n:
                problems.append(f"nodes[{n.get('id')}]: missing {k}")
        if n.get("kind") is not None and n["kind"] not in NODE_KINDS + _EXTRA_KINDS:
            problems.append(f"nodes[{n.get('id')}]: unknown kind {n['kind']!r}")

    boundary = spec.get("boundary") or []
    nb = len(boundary)
    et = spec.get("edge_types")
    if et is not None and len(et) != nb:
        problems.append(f"edge_types: {len(et)} types for {nb} boundary edges")

    demands = spec.get("demands") or {}
    for rid, d in demands.items():
        if str(rid) not in idset:
            problems.append(f"demands: unknown id {rid!r}")
        me, ms = d.get("min_edge"), d.get("max_short")
        if me is not None and ms is not None and me > ms:
            problems.append(f"demands[{rid}]: min_edge {me} > max_short {ms}")
        ma, mx = d.get("min_area"), d.get("max_area")
        if ma is not None and mx is not None and ma > mx:
            problems.append(f"demands[{rid}]: min_area {ma} > max_area {mx}")
        asp = d.get("aspect_max")
        if asp is not None and asp < 1.0:
            problems.append(f"demands[{rid}]: aspect_max {asp} < 1")
    fracs = [d["area_frac"] for d in demands.values() if d.get("area_frac") is not None]
    if fracs and abs(sum(fracs) - 1.0) > AREA_FRAC_TOL:
        problems.append(f"demands: area_frac totals {sum(fracs):.4f}, not about 1")

    for e in spec.get("adjacencies") or []:
        for end in ("u", "v"):
            if str(e.get(end)) not in idset:
                problems.append(f"adjacencies: unknown id {e.get(end)!r}")
        if e.get("connection") not in (0, 1, 2):
            problems.append(f"adjacencies {e.get('u')}-{e.get('v')}: connection "
                            f"{e.get('connection')!r} not in 0, 1, 2")

    for rid, idxs in (spec.get("boundary_demands") or {}).items():
        if str(rid) not in idset:
            problems.append(f"boundary_demands: unknown id {rid!r}")
        for i in idxs:
            if not (isinstance(i, int) and 0 <= i < nb):
                problems.append(f"boundary_demands[{rid}]: edge index {i!r} out of range")

    er = spec.get("entry_room")
    if er is not None and str(er) not in idset:
        problems.append(f"entry_room: unknown id {er!r}")
    for pair in spec.get("precedence") or []:
        for x in pair:
            if str(x) not in idset:
                problems.append(f"precedence: unknown id {x!r}")

    ent = spec.get("entrance")
    if ent is not None and nb >= 3:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        tol = ENTRANCE_TOL_FRAC * math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        pts = [(ent[i], ent[i + 1]) for i in range(0, len(ent) - 1, 2)]
        for p in pts:
            dist = _dist_to_boundary(p, boundary)
            if dist > tol:
                problems.append(f"entrance: point {list(p)} is {dist:.4g} from the "
                                f"boundary (tolerance {tol:.4g})")
    return problems


def validate_spec(spec):
    """Raise ValueError listing every invariant breach; return None if valid."""
    problems = spec_problems(spec)
    if problems:
        raise ValueError("invalid Spec: " + "; ".join(problems))
