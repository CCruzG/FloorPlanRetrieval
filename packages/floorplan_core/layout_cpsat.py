"""
layout_cpsat.py — CP-SAT layout solver (2026-11-02): a drop-in alternative to
core_placement.place_cores' greedy step-0 core placement.

Same job as place_cores: size and place one axis-aligned core rectangle per
room with a `min_edge` demand, inside the boundary, and move each room's node
into its rectangle. Same seam: generate.py step 0 pins the returned rectangle
edges as grid lines and seeds the core cells to the room (same `{cores, nodes,
report}` shape as place_cores — see that module's docstring for the seam).

Why a second solver: place_cores is a one-pass greedy placement with no
backtracking — it cannot jointly satisfy door adjacency, boundary fronts and
min sizes (finding, 2026-11-02: door constraints raised D24 failures 13 -> 19
over 9 plans). This module poses the same placement as one CP-SAT model so the
solver can trade all constraints against each other at once.

Grid: integer units of 100 mm. `mm` (mm_per_unit) converts a boundary-unit
coordinate to mm (`boundary_units * mm_per_unit = mm`); divide by 100 for grid
cells. Orthogonal boundaries only — a non-axis-aligned edge is unsupported and
the function returns immediately with no rooms placed.

Model, per room with a `min_edge` demand (rooms without one are not modelled,
same as place_cores):
  - one rectangle: x0, x1, y0, y1 (IntVar) + width/height/area (IntVar, lower
    bound IS the hard min_edge / min_area constraint — see _add_room).
  - aspect ratio <= ASPECT_MAX both ways (hard).
  - kept inside the boundary: the room rectangle and one fixed rectangle per
    "outside the boundary but inside its bbox" grid cell all go into one
    AddNoOverlap2D (_forbidden_rects decomposes bbox-minus-boundary along the
    boundary's own vertex grid lines — valid for an orthogonal polygon).
  - door/open adjacency (connection 1/2 or separation door/open): the two
    rooms must share >= DOOR_MIN_M of wall — hard, encoded as "touches on one
    of its four sides with enough overlap", one of which must hold
    (_add_touch_or + AddBoolOr).
  - boundary-front demands: each demanded edge must be fronted with >= its
    edge-type floor. Tried hard first; if that model is INFEASIBLE, re-solved
    with fronts soft (a shortfall penalty in the objective) — report says
    which mode solved.
  - soft wall adjacency: pairs in `pair_weights` get a reward (or, if the
    weight is negative, a penalty) for touching with >= SOFT_WALL_MIN_M of
    wall. Default (`pair_weights=None`): weight 1 for every source room<->room
    edge with connection 0 (a plain wall) between two modelled rooms — the
    hook a future authored zoning table plugs into.

Objective (module constants below, all integer — CP-SAT needs an integer
objective): maximise covered area (dominant: OBJ_AREA_W), minus the L1
distance from each rectangle's centre to its source node (OBJ_NODE_W), plus
the soft wall-adjacency term (OBJ_WALL_W * pair weight), minus the soft-front
shortfall when fronts were relaxed (OBJ_FRONT_SOFT_W). The weights are tuned
by inspection, not fit; tune further against diag_placement_topology output.

`precedence` is accepted for signature parity with place_cores (same call
site in generate.py) but is NOT YET used by the objective — unlike
place_cores, which uses it only to order a greedy pass, a joint CP-SAT solve
has no equivalent "place order" to seed; a precedence-aware term (e.g.
rewarding the winner's area over the loser's) is future work.

Public API:
    solve_layout(nodes, demands, mm, boundary, boundary_demands, edge_types,
                 adjacencies, precedence=None, pair_weights=None,
                 cpsat_seed=2026, cpsat_workers=8, cpsat_time_s=5.0)
        -> {cores, nodes, report}
"""
import math
import time
from fractions import Fraction

from ortools.sat.python import cp_model
from shapely.geometry import Polygon as SPoly, Point as SPoint

GRID_MM = 100.0                 # CP-SAT grid cell, in mm
DOOR_MIN_M = 0.9                # hard door/open shared-wall floor
SOFT_WALL_MIN_M = 0.3           # soft wall-adjacency shared-wall floor
EDGE_FLOOR_M = {"facing_corridor": 0.9, "external": 1.2, "shared_wall": 0.3}
ASPECT_MAX = 2.5
EDGE_EPS_UNITS_MM = 50.0        # half a grid cell, used to probe interior side

# Objective weights (module constants, all applied as integer coefficients on
# integer grid-cell variables — see module docstring).
OBJ_AREA_W = 20          # reward per grid-cell^2 of covered area (dominant)
OBJ_NODE_W = 1           # penalty per grid-cell of 2x|centre - node| (L1)
OBJ_WALL_W = 60          # scale on each pair_weight unit, soft wall adjacency
OBJ_FRONT_SOFT_W = 150   # penalty per grid-cell of front shortfall (soft mode)

TIME_LIMIT_S = 5.0
NUM_WORKERS = 8
RANDOM_SEED = 2026

_BIG = 10 ** 7


def _aspect_ratio(a):
    """aspect_max (float) as the (num, den) pair the model uses: 2.5 -> (5, 2)."""
    fr = Fraction(float(a)).limit_denominator(10)
    return fr.numerator, fr.denominator


def _is_door_or_open(e):
    """Mirrors core_placement._is_door_or_open / topo_sensors._passable:
    `separation` is authoritative when present, else `connection` 1 or 2."""
    sep = e.get("separation")
    if sep in ("door", "open"):
        return True
    if sep == "wall":
        return False
    return e.get("connection") in (1, 2)


def _orthogonal(boundary, tol=1e-6):
    nb = len(boundary)
    for k in range(nb):
        (ax, ay), (bx, by) = boundary[k], boundary[(k + 1) % nb]
        if abs(ax - bx) > tol and abs(ay - by) > tol:
            return False
    return True


def _to_cells(v, mm):
    return int(round(v * mm / GRID_MM))


def _forbidden_rects(boundary, bpoly, mm):
    """Decompose bbox-minus-boundary into fixed forbidden rectangles, on the
    grid formed by the boundary's own vertex coordinates (valid for an
    orthogonal polygon: every such cell is wholly inside or outside it)."""
    xs_u = sorted({p[0] for p in boundary})
    ys_u = sorted({p[1] for p in boundary})
    # Conservative snapping (2026-11-03): the allowed bbox rounds INWARD and each
    # forbidden rectangle rounds OUTWARD, so a placed room never crosses the true
    # boundary. Nearest-cell rounding let rooms overhang by up to half a cell; the
    # realiser then clipped them to the boundary and they missed min_edge by a few
    # tens of mm (D24 near misses, e.g. 2373 mm against 2400).
    # The cut grid holds both roundings of every vertex coordinate, so its cells
    # never overlap; a cell is forbidden unless it lies wholly inside the
    # boundary (they share one NoOverlap2D with the rooms, so overlap = infeasible).
    fl = lambda v: int(math.floor(v * mm / GRID_MM + 1e-9))
    ce = lambda v: int(math.ceil(v * mm / GRID_MM - 1e-9))
    xs_c = sorted({fl(x) for x in xs_u} | {ce(x) for x in xs_u})
    ys_c = sorted({fl(y) for y in ys_u} | {ce(y) for y in ys_u})
    inner = bpoly.buffer(1e-9)
    u = GRID_MM / mm
    bbox_cells = (ce(xs_u[0]), ce(ys_u[0]), fl(xs_u[-1]), fl(ys_u[-1]))
    forbidden = []
    for i in range(len(xs_c) - 1):
        for j in range(len(ys_c) - 1):
            x0, x1, y0, y1 = xs_c[i], xs_c[i + 1], ys_c[j], ys_c[j + 1]
            if x1 <= bbox_cells[0] or x0 >= bbox_cells[2] or y1 <= bbox_cells[1] or y0 >= bbox_cells[3]:
                continue   # outside the allowed bbox: the variable domains already exclude it
            cell = SPoly([(x0 * u, y0 * u), (x1 * u, y0 * u), (x1 * u, y1 * u), (x0 * u, y1 * u)])
            if not inner.contains(cell):
                forbidden.append((max(x0, bbox_cells[0]), max(y0, bbox_cells[1]),
                                  min(x1, bbox_cells[2]), min(y1, bbox_cells[3])))
    return forbidden, bbox_cells


def _edge_geometry(boundary, bpoly, edge_types, mm):
    """Per boundary edge: which room coordinate must equal it to "front" it
    (the interior side), the perpendicular overlap range, and the edge-type
    floor — all in grid cells. Orthogonal boundary only."""
    nb = len(boundary)
    eps = EDGE_EPS_UNITS_MM / mm
    etype = lambda E: (edge_types[E] if (edge_types and 0 <= E < len(edge_types))  # noqa: E731
                       else "external")
    # Edges round INWARD (2026-11-03), matching the conservative snapping in
    # _forbidden_rects: the fronting coordinate moves towards the interior and
    # the span shrinks, so a fronting room is reachable inside the allowed cells.
    fl = lambda v: int(math.floor(v * mm / GRID_MM + 1e-9))  # noqa: E731
    ce = lambda v: int(math.ceil(v * mm / GRID_MM - 1e-9))   # noqa: E731
    geoms = {}
    for E in range(nb):
        (ax, ay), (bx, by) = boundary[E], boundary[(E + 1) % nb]
        floor_cells = int(round(EDGE_FLOOR_M.get(etype(E), 0.3) * 10))
        if abs(ax - bx) < 1e-9:                      # vertical edge, x == ax
            ex_u = ax
            y_mid = (ay + by) / 2.0
            left_in = bpoly.contains(SPoint(ex_u - eps, y_mid))
            attr = "x1" if left_in else "x0"
            ylo, yhi = sorted((ay, by))
            lo, hi = ce(ylo), fl(yhi)
            geoms[E] = {"axis": "x", "coord": fl(ex_u) if left_in else ce(ex_u), "attr": attr,
                       "lo": lo, "hi": hi, "floor": floor_cells, "span_attrs": ("y0", "y1")}
        elif abs(ay - by) < 1e-9:                     # horizontal edge, y == ay
            ey_u = ay
            x_mid = (ax + bx) / 2.0
            below_in = bpoly.contains(SPoint(x_mid, ey_u - eps))
            attr = "y1" if below_in else "y0"
            xlo, xhi = sorted((ax, bx))
            lo, hi = ce(xlo), fl(xhi)
            geoms[E] = {"axis": "y", "coord": fl(ey_u) if below_in else ce(ey_u), "attr": attr,
                       "lo": lo, "hi": hi, "floor": floor_cells, "span_attrs": ("x0", "x1")}
        # a non-orthogonal edge never reaches here (caller checks _orthogonal first)
    return geoms


def _add_touch_or(model, A, B, min_overlap, tag):
    """Four touch options (A left-of B, B left-of A, A below B, B below A),
    each requiring the shared-edge coincidence AND >= min_overlap of overlap
    on the perpendicular axis. Returns (options, overlap_x, overlap_y) — the
    caller either forces AddBoolOr(options) (hard) or leaves them free and
    reads a derived "touches" bool (soft)."""
    ylo = model.NewIntVar(-_BIG, _BIG, f"ylo_{tag}")
    yhi = model.NewIntVar(-_BIG, _BIG, f"yhi_{tag}")
    model.AddMaxEquality(ylo, [A["y0"], B["y0"]])
    model.AddMinEquality(yhi, [A["y1"], B["y1"]])
    ov_y = model.NewIntVar(-_BIG, _BIG, f"ovy_{tag}")
    model.Add(ov_y == yhi - ylo)

    xlo = model.NewIntVar(-_BIG, _BIG, f"xlo_{tag}")
    xhi = model.NewIntVar(-_BIG, _BIG, f"xhi_{tag}")
    model.AddMaxEquality(xlo, [A["x0"], B["x0"]])
    model.AddMinEquality(xhi, [A["x1"], B["x1"]])
    ov_x = model.NewIntVar(-_BIG, _BIG, f"ovx_{tag}")
    model.Add(ov_x == xhi - xlo)

    b_l = model.NewBoolVar(f"left_{tag}")    # A directly left of B
    b_r = model.NewBoolVar(f"right_{tag}")   # B directly left of A
    b_d = model.NewBoolVar(f"below_{tag}")   # A directly below B
    b_u = model.NewBoolVar(f"above_{tag}")   # B directly below A
    model.Add(A["x1"] == B["x0"]).OnlyEnforceIf(b_l)
    model.Add(ov_y >= min_overlap).OnlyEnforceIf(b_l)
    model.Add(B["x1"] == A["x0"]).OnlyEnforceIf(b_r)
    model.Add(ov_y >= min_overlap).OnlyEnforceIf(b_r)
    model.Add(A["y1"] == B["y0"]).OnlyEnforceIf(b_d)
    model.Add(ov_x >= min_overlap).OnlyEnforceIf(b_d)
    model.Add(B["y1"] == A["y0"]).OnlyEnforceIf(b_u)
    model.Add(ov_x >= min_overlap).OnlyEnforceIf(b_u)
    return [b_l, b_r, b_d, b_u], ov_x, ov_y


def _shared_wall_m(rect_a, rect_b, mm):
    """Realised shared-wall length (metres) between two solved rectangles,
    (x0, y0, x1, y1) in boundary units — mirrors core_placement._shared_wall."""
    ax0, ay0, ax1, ay1 = rect_a
    bx0, by0, bx1, by1 = rect_b
    m = mm / 1000.0
    tol = 1e-6
    if abs(ax1 - bx0) < tol or abs(bx1 - ax0) < tol:
        lo, hi = max(ay0, by0), min(ay1, by1)
        return max(hi - lo, 0.0) * m
    if abs(ay1 - by0) < tol or abs(by1 - ay0) < tol:
        lo, hi = max(ax0, bx0), min(ax1, bx1)
        return max(hi - lo, 0.0) * m
    return 0.0


def _default_pair_weights(adjacencies, modeled_ids):
    """weight 1 for every source room<->room edge with connection 0 (a plain
    wall) between two modelled rooms."""
    out = {}
    for e in adjacencies or []:
        if _is_door_or_open(e):
            continue
        if e.get("connection") not in (0, None):
            continue
        a, b = str(e.get("u")), str(e.get("v"))
        if a in modeled_ids and b in modeled_ids and a != b:
            out[(a, b)] = 1.0
    return out


def _add_entrance_front(model, r, g, entrance, mm, m):
    """Hard: rectangle r fronts boundary edge g and covers the entrance on it."""
    model.Add(r[g["attr"]] == g["coord"])
    lo_a, hi_a = r[g["span_attrs"][0]], r[g["span_attrs"][1]]
    ex0, ey0, ex1, ey1 = entrance
    if math.hypot(ex1 - ex0, ey1 - ey0) * m < 1e-3:   # a point
        # A door's width around the point, clipped to the edge. Reaching the point
        # alone lets a room end at the door jamb and leaves the door to its neighbour.
        pt_c = _to_cells(ey0 if g["axis"] == "x" else ex0, mm)
        half = int(math.ceil(DOOR_MIN_M * 10 / 2.0))
        model.Add(lo_a <= max(pt_c - half, g["lo"]))
        model.Add(hi_a >= min(pt_c + half, g["hi"]))
    else:
        lo_u = ey0 if g["axis"] == "x" else ex0
        hi_u = ey1 if g["axis"] == "x" else ex1
        lo_u, hi_u = min(lo_u, hi_u), max(lo_u, hi_u)
        e_lo, e_hi = _to_cells(lo_u, mm), _to_cells(hi_u, mm)
        elo = model.NewIntVar(-_BIG, _BIG, "ent_lo")
        ehi = model.NewIntVar(-_BIG, _BIG, "ent_hi")
        model.AddMaxEquality(elo, [lo_a, e_lo])
        model.AddMinEquality(ehi, [hi_a, e_hi])
        model.Add(ehi - elo >= int(round(DOOR_MIN_M * 10)))


def _set_limits(solver, cpsat_time_s, cpsat_workers, cpsat_seed, cpsat_det_time=None):
    """Solver limits. cpsat_det_time (gate runs) replaces the wall-clock limit
    with a deterministic one, so the same model gives the same plan."""
    if cpsat_det_time is not None:
        solver.parameters.max_deterministic_time = float(cpsat_det_time)
    else:
        solver.parameters.max_time_in_seconds = cpsat_time_s
    solver.parameters.num_search_workers = cpsat_workers
    solver.parameters.random_seed = cpsat_seed


def solve_layout(nodes, demands, mm, boundary, boundary_demands, edge_types,
                 adjacencies, precedence=None, pair_weights=None, entry_room=None, entrance=None,
                 cpsat_seed=RANDOM_SEED, cpsat_workers=NUM_WORKERS, cpsat_time_s=TIME_LIMIT_S,
                 cpsat_det_time=None):
    if not mm or mm <= 0:
        return {"cores": {}, "nodes": nodes, "report": {"skipped": "no mm scale"}}
    if not _orthogonal(boundary):
        return {"cores": {}, "nodes": nodes,
                "report": {"unsupported": "non-orthogonal boundary"}}

    bpoly = SPoly(boundary)
    if not bpoly.is_valid:
        bpoly = bpoly.buffer(0)

    dims = {}                      # str(id) -> {"id", min_edge_cells, min_area_cells, pos}
    for n in nodes:
        rid = n["id"]
        sid = str(rid)
        d = demands.get(rid) or {}
        me_mm = d.get("min_edge", 0.0)
        if me_mm and me_mm > 0:
            dims[sid] = {
                "id": rid,
                "min_edge_cells": int(math.ceil(me_mm / GRID_MM)),
                "min_area_cells": int(math.ceil(d.get("min_area", 0.0) / (GRID_MM * GRID_MM))),
                "pos": tuple(n["pos"]),
                # aspect limit from demands[id].aspect_max (hallway 6.0, others 2.5)
                "aspect": _aspect_ratio(d.get("aspect_max", ASPECT_MAX)),
            }
    report = {"rooms": {}, "modeled": sorted(dims)}
    if not dims:
        return {"cores": {}, "nodes": nodes, "report": {**report, "reason": "no room has a min_edge demand"}}

    forbidden, (bx0c, by0c, bx1c, by1c) = _forbidden_rects(boundary, bpoly, mm)
    edge_geoms = _edge_geometry(boundary, bpoly, edge_types, mm)
    bd = {str(k): [E for E in v if E in edge_geoms] for k, v in (boundary_demands or {}).items()}

    # entrance (2026-11-04): the entry room fronts the drawn entrance, as in
    # solve_layout_dim. Hard; released only when no layout exists with it.
    entry_sid = str(entry_room) if entry_room is not None else None
    ent_mid = None
    if entrance and len(entrance) >= 4:
        ent_mid = ((entrance[0] + entrance[2]) / 2.0, (entrance[1] + entrance[3]) / 2.0)
    ent_edge = _nearest_boundary_edge(boundary, ent_mid) if ent_mid is not None else None
    entrance_status = {"supplied": ent_mid is not None, "entry_room": entry_sid,
                       "entry_modeled": entry_sid in dims, "applied": False, "edge": ent_edge}

    door_pairs = []   # (a, b) both modeled, door/open
    for e in adjacencies or []:
        a, b = str(e.get("u")), str(e.get("v"))
        if a in dims and b in dims and a != b and _is_door_or_open(e):
            door_pairs.append(tuple(sorted((a, b))))
    door_pairs = sorted(set(door_pairs))

    pw = pair_weights if pair_weights is not None else _default_pair_weights(adjacencies, set(dims))
    wall_pairs = [(a, b, w) for (a, b), w in pw.items()
                 if a in dims and b in dims and a != b and w]

    m = mm / 1000.0

    def _build(front_hard, entry_hard=True):
        model = cp_model.CpModel()
        rooms = {}
        for sid, d in dims.items():
            w = model.NewIntVar(d["min_edge_cells"], bx1c - bx0c, f"w_{sid}")
            h = model.NewIntVar(d["min_edge_cells"], by1c - by0c, f"h_{sid}")
            x0 = model.NewIntVar(bx0c, bx1c, f"x0_{sid}")
            y0 = model.NewIntVar(by0c, by1c, f"y0_{sid}")
            x1 = model.NewIntVar(bx0c, bx1c, f"x1_{sid}")
            y1 = model.NewIntVar(by0c, by1c, f"y1_{sid}")
            x_itv = model.NewIntervalVar(x0, w, x1, f"xitv_{sid}")
            y_itv = model.NewIntervalVar(y0, h, y1, f"yitv_{sid}")
            area_max = (bx1c - bx0c) * (by1c - by0c)
            area = model.NewIntVar(max(d["min_area_cells"], 0), max(area_max, 1), f"area_{sid}")
            model.AddMultiplicationEquality(area, [w, h])
            an, ad = d["aspect"]          # aspect <= an/ad, both ways (2.5, hallways 6)
            model.Add(ad * w <= an * h)
            model.Add(ad * h <= an * w)
            rooms[sid] = {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "w": w, "h": h,
                         "area": area, "x_itv": x_itv, "y_itv": y_itv}

        # keep inside the boundary: every room + every forbidden cell, one no-overlap
        boxes = [(r["x_itv"], r["y_itv"]) for r in rooms.values()]
        for i, (fx0, fy0, fx1, fy1) in enumerate(forbidden):
            fxi = model.NewFixedSizeIntervalVar(fx0, fx1 - fx0, f"fx_{i}")
            fyi = model.NewFixedSizeIntervalVar(fy0, fy1 - fy0, f"fy_{i}")
            boxes.append((fxi, fyi))
        model.AddNoOverlap2D([b[0] for b in boxes], [b[1] for b in boxes])

        # hard door/open adjacency: >= DOOR_MIN_M shared wall
        door_min_cells = int(round(DOOR_MIN_M * 10))
        for a, b in door_pairs:
            opts, _, _ = _add_touch_or(model, rooms[a], rooms[b], door_min_cells, f"door_{a}_{b}")
            model.AddBoolOr(opts)

        # boundary-front demands
        front_terms = []
        fronted_bool = {}   # (sid, E) -> BoolVar (soft) — used for reporting
        for sid, edges in bd.items():
            if sid not in rooms:
                continue
            for E in edges:
                g = edge_geoms[E]
                r = rooms[sid]
                lo_a, hi_a = r[g["span_attrs"][0]], r[g["span_attrs"][1]]
                lo = model.NewIntVar(-_BIG, _BIG, f"flo_{sid}_{E}")
                hi = model.NewIntVar(-_BIG, _BIG, f"fhi_{sid}_{E}")
                model.AddMaxEquality(lo, [lo_a, g["lo"]])
                model.AddMinEquality(hi, [hi_a, g["hi"]])
                overlap = model.NewIntVar(-_BIG, _BIG, f"fov_{sid}_{E}")
                model.Add(overlap == hi - lo)
                if front_hard:
                    model.Add(r[g["attr"]] == g["coord"])
                    model.Add(overlap >= g["floor"])
                else:
                    b_front = model.NewBoolVar(f"bfront_{sid}_{E}")
                    model.Add(r[g["attr"]] == g["coord"]).OnlyEnforceIf(b_front)
                    model.Add(r[g["attr"]] != g["coord"]).OnlyEnforceIf(b_front.Not())
                    shortfall = model.NewIntVar(0, g["floor"], f"short_{sid}_{E}")
                    model.Add(shortfall >= g["floor"] - overlap).OnlyEnforceIf(b_front)
                    model.Add(shortfall >= g["floor"]).OnlyEnforceIf(b_front.Not())
                    front_terms.append(shortfall)
                    fronted_bool[(sid, E)] = b_front

        # soft wall adjacency
        wall_terms = []
        for a, b, w in wall_pairs:
            opts, _, _ = _add_touch_or(model, rooms[a], rooms[b], int(round(SOFT_WALL_MIN_M * 10)),
                                       f"wall_{a}_{b}")
            touch = model.NewBoolVar(f"touch_{a}_{b}")
            for o in opts:
                model.AddImplication(o, touch)
            model.AddBoolOr(opts).OnlyEnforceIf(touch)
            wall_terms.append((touch, w))

        # objective
        obj = []
        for sid, r in rooms.items():
            obj.append(OBJ_AREA_W * r["area"])
            px, py = dims[sid]["pos"]
            node_x = _to_cells(px, mm)
            node_y = _to_cells(py, mm)
            dx = model.NewIntVar(-_BIG, _BIG, f"dx_{sid}")
            dy = model.NewIntVar(-_BIG, _BIG, f"dy_{sid}")
            model.Add(dx == (r["x0"] + r["x1"]) - 2 * node_x)
            model.Add(dy == (r["y0"] + r["y1"]) - 2 * node_y)
            adx = model.NewIntVar(0, _BIG, f"adx_{sid}")
            ady = model.NewIntVar(0, _BIG, f"ady_{sid}")
            model.AddAbsEquality(adx, dx)
            model.AddAbsEquality(ady, dy)
            obj.append(-OBJ_NODE_W * (adx + ady))
        for touch, w in wall_terms:
            obj.append(int(round(OBJ_WALL_W * w)) * touch)
        for shortfall in front_terms:
            obj.append(-OBJ_FRONT_SOFT_W * shortfall)
        if entry_hard and entry_sid in rooms and edge_geoms.get(ent_edge) is not None:
            _add_entrance_front(model, rooms[entry_sid], edge_geoms[ent_edge], entrance, mm, m)
            entrance_status["applied"] = True
        model.Maximize(sum(obj))
        return model, rooms, fronted_bool

    t0 = time.time()
    model, rooms, fronted_bool = _build(front_hard=True)
    solver = cp_model.CpSolver()
    _set_limits(solver, cpsat_time_s, cpsat_workers, cpsat_seed, cpsat_det_time)
    status = solver.Solve(model)
    relaxation = "hard"
    if status in (cp_model.INFEASIBLE, cp_model.MODEL_INVALID):
        model, rooms, fronted_bool = _build(front_hard=False)
        solver = cp_model.CpSolver()
        _set_limits(solver, cpsat_time_s, cpsat_workers, cpsat_seed, cpsat_det_time)
        status = solver.Solve(model)
        relaxation = "soft" if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else "infeasible"
    if status in (cp_model.INFEASIBLE, cp_model.MODEL_INVALID) and entrance_status["applied"]:
        entrance_status["applied"] = False
        model, rooms, fronted_bool = _build(front_hard=False, entry_hard=False)
        solver = cp_model.CpSolver()
        _set_limits(solver, cpsat_time_s, cpsat_workers, cpsat_seed, cpsat_det_time)
        status = solver.Solve(model)
        relaxation = "no_entrance" if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else "infeasible"
    solve_time = time.time() - t0

    report.update({
        "status": solver.StatusName(status),
        "relaxation": relaxation,
        "solve_time_s": round(solve_time, 3),
        "objective": (solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None),
        "door_pairs": [[a, b] for a, b in door_pairs],
        "entrance": entrance_status,
    })

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for sid in dims:
            report["rooms"][sid] = {"placed": False}
        return {"cores": {}, "nodes": nodes, "report": report}

    placed = {}
    m = mm / 1000.0
    for sid, r in rooms.items():
        x0 = solver.Value(r["x0"]) * GRID_MM / mm
        y0 = solver.Value(r["y0"]) * GRID_MM / mm
        x1 = solver.Value(r["x1"]) * GRID_MM / mm
        y1 = solver.Value(r["y1"]) * GRID_MM / mm
        placed[sid] = (x0, y0, x1, y1)

    rect_attr_idx = {"x0": 0, "y0": 1, "x1": 2, "y1": 3}
    for sid, r in rooms.items():
        rect = placed[sid]
        fronted, unfronted = [], []
        for E in bd.get(sid, []):
            g = edge_geoms[E]
            room_coord = rect[rect_attr_idx[g["attr"]]]
            edge_coord = g["coord"] * GRID_MM / mm
            on_edge = abs(room_coord - edge_coord) < 1e-6
            if g["axis"] == "x":
                lo_r, hi_r = rect[1], rect[3]
            else:
                lo_r, hi_r = rect[0], rect[2]
            lo_edge, hi_edge = g["lo"] * GRID_MM / mm, g["hi"] * GRID_MM / mm
            overlap_m = max(0.0, min(hi_r, hi_edge) - max(lo_r, lo_edge)) * m
            floor_m = g["floor"] * GRID_MM / 1000.0
            if on_edge and overlap_m + 1e-6 >= floor_m:
                fronted.append(E)
            else:
                unfronted.append(E)
        door_met = []
        for a, b in door_pairs:
            if sid not in (a, b):
                continue
            other = b if sid == a else a
            shared = _shared_wall_m(rect, placed[other], mm)
            door_met.append({"room": other, "shared_m": round(shared, 3), "ok": shared + 1e-6 >= DOOR_MIN_M})
        report["rooms"][sid] = {
            "placed": True,
            "core": [round(v, 6) for v in rect],
            "area_m2": round((rect[2] - rect[0]) * (rect[3] - rect[1]) * m * m, 3),
            "fronted": fronted,
            "unfronted": unfronted,
            "door_pairs_met": door_met,
        }

    new_nodes = []
    for n in nodes:
        sid = str(n["id"])
        rect = placed.get(sid)
        if rect is None:
            new_nodes.append(n)
            continue
        p = tuple(n["pos"])
        inside = rect[0] <= p[0] <= rect[2] and rect[1] <= p[1] <= rect[3]
        pos = list(p) if inside else [(rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0]
        report["rooms"][sid]["moved_m"] = round(math.hypot(pos[0] - p[0], pos[1] - p[1]) * m, 3)
        report["rooms"][sid]["from"] = [round(v, 6) for v in p]
        report["rooms"][sid]["to"] = [round(v, 6) for v in pos]
        new_nodes.append(dict(n, pos=pos))

    return {"cores": placed, "nodes": new_nodes, "report": report}


# ── cpsat_dim: size the retrieved arrangement, don't re-arrange it ───────────
#
# solve_layout re-places every modelled room freely (only a soft L1 pull to its
# source node) — so it can, and does, put a bedroom at the entrance
# (test_boundary_005b rank 1/3: entrance_valid fails) and drop more source
# edges than no placement at all. solve_layout_dim keeps the same objective
# and NoOverlap2D/min-size/aspect/boundary-front machinery, but adds the
# retrieved plan's own arrangement as hard constraints: every source edge
# (door/open AND plain wall) must touch on the SAME SIDE it touches on in the
# source, every far-apart non-adjacent pair keeps its source centre order, and
# the source entry room must front the target entrance. CP-SAT then only
# trades room SIZE against the boundary and min-size/aspect demands — the
# retrieved plan supplies the layout.

ORDER_MIN_M = 0.5        # centre-order constraint threshold (task-specified)
_SIDE_IDX = {"a_left_b": 0, "b_left_a": 1, "a_below_b": 2, "b_below_a": 3}
W_SRC_WALL = 3           # default objective weight, a source wall (connection
                         # 0) touching its derived side — pair_weights overrides
                         # a given pair (2026-11-02: walls demoted hard->soft,
                         # see solve_layout_dim docstring)


def _poly_bbox_units(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _side_from_polys(poly_a, poly_b, tol):
    """Which of _add_touch_or's 4 options holds between two SOURCE room
    polygons (already mapped to the target frame) — from the actual shared-
    wall geometry (shapely boundary intersection), not a bbox-corner guess.
    A bbox-corner test is wrong whenever a room is non-rectangular (an L or a
    notch): the true touching wall can sit short of the bbox edge (2026-11-02
    finding, test_boundary_005b rank 1 r_2/r_4 — r_2 is a 10-vertex L, its
    bbox's right edge is 0.055 units short of the wall it actually shares with
    r_4). Returns None if the two don't share an axis-aligned segment longer
    than `tol` on one axis and within `tol` on the other (a corner-only touch,
    or a non-orthogonal / multi-segment contact)."""
    try:
        pa = SPoly(poly_a)
        pb = SPoly(poly_b)
        if not pa.is_valid:
            pa = pa.buffer(0)
        if not pb.is_valid:
            pb = pb.buffer(0)
        shared = pa.boundary.intersection(pb.boundary)
    except Exception:
        return None
    if shared.is_empty:
        return None
    geoms = list(getattr(shared, "geoms", [shared]))
    pts = []
    for g in geoms:
        coords = getattr(g, "coords", None)
        if coords:
            pts.extend(list(coords))
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    dx, dy = max(xs) - min(xs), max(ys) - min(ys)
    cax, cbx = pa.centroid.x, pb.centroid.x
    cay, cby = pa.centroid.y, pb.centroid.y
    if dx <= tol and dy > tol:
        return "a_left_b" if cax < cbx else "b_left_a"
    if dy <= tol and dx > tol:
        return "a_below_b" if cay < cby else "b_below_a"
    return None


def _side_from_nodes(pos_a, pos_b):
    """Fallback side when source polygons aren't available for one of the two
    rooms: the dominant axis of the (target-frame) node-to-node offset."""
    dx, dy = pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]
    if abs(dx) >= abs(dy):
        return "a_left_b" if dx > 0 else "b_left_a"
    return "a_below_b" if dy > 0 else "b_below_a"


def _nearest_boundary_edge(boundary, pt):
    """Index of the raw boundary edge (boundary[i] -> boundary[i+1]) nearest
    `pt` — mirrors topo_signature.default_edge_types' entrance-edge lookup, so
    `edge_geoms` (keyed the same way) can be read directly."""
    px, py = pt
    nb = len(boundary)
    best_i, best_d = 0, float("inf")
    for i in range(nb):
        ax, ay = boundary[i]
        bx, by = boundary[(i + 1) % nb]
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
        d = (mx - px) ** 2 + (my - py) ** 2
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def solve_layout_dim(nodes, demands, mm, boundary, boundary_demands, edge_types,
                     adjacencies, precedence=None, pair_weights=None,
                     source_rooms=None, entry_room=None, entrance=None,
                     cpsat_seed=RANDOM_SEED, cpsat_workers=NUM_WORKERS,
                     cpsat_time_s=TIME_LIMIT_S, cpsat_det_time=None):
    """Same job/seam as solve_layout (see module docstring), but the retrieved
    plan's own arrangement drives placement: door/open edges are hard (touch
    on their source side), far-apart non-adjacent pairs keep their source
    centre order (hard), and the source entry room fronts the target entrance
    (hard). Only room SIZE is free, plus — 2026-11-02 — plain-wall (connection
    0) touch, which is now SOFT: a same-side touch at the source (>=
    SOFT_WALL_MIN_M) is rewarded in the objective, weight W_SRC_WALL (3) by
    default, overridden per pair by `pair_weights`. Walls were hard until
    2026-11-02; that made every one of the 18 gate plans INFEASIBLE (verified:
    not the forced side, not a space shortfall — the forced wall-touch GRAPH
    itself cannot be realised by one rectangle per room when a source room is
    non-rectangular, which RPLAN rooms often are specifically to satisfy an
    adjacency degree a rectangle can't).

    source_rooms: {str(room_id): [[x, y], ...]} — source room polygons already
    mapped into the TARGET frame (topo_signature.map_rooms_to_target, the same
    transform that produces node_pos). Preferred arrangement source. A room
    missing from it falls back to `nodes`' (target-frame) position for both
    side-derivation and centre-order — report["arrangement_source"] says which
    was used, overall and per-room.

    entry_room: the SOURCE entry room id (topo_signature's nearest-room-to-
    entrance test, on the SOURCE plan). entrance: the Spec's entrance
    [x0,y0,x1,y1], already in the TARGET frame (same one boundary_demands and
    the sensors read).

    Relaxation order on infeasible (never relaxes door/open TOUCH or the
    entrance): 1) centre-order soft (dropped), 2) additionally free door/open
    edges' forced side (touch stays hard, AddBoolOr over all 4 sides instead of
    the one derived side) — each tried with boundary fronts hard then soft.
    report["relaxations"] names what was dropped to reach a feasible solve. If
    no rung solves, report["unsolved"]=True and cores stay empty — a silent
    fallback to "off" would hide the failure, so the diagnostic gate counts
    this explicitly instead of inferring it from an empty cores dict.
    """
    if not mm or mm <= 0:
        return {"cores": {}, "nodes": nodes, "report": {"skipped": "no mm scale"}}
    if not _orthogonal(boundary):
        return {"cores": {}, "nodes": nodes,
                "report": {"unsupported": "non-orthogonal boundary"}}

    bpoly = SPoly(boundary)
    if not bpoly.is_valid:
        bpoly = bpoly.buffer(0)

    dims = {}
    for n in nodes:
        rid = n["id"]
        sid = str(rid)
        d = demands.get(rid) or {}
        me_mm = d.get("min_edge", 0.0)
        if me_mm and me_mm > 0:
            dims[sid] = {
                "id": rid,
                "min_edge_cells": int(math.ceil(me_mm / GRID_MM)),
                "min_area_cells": int(math.ceil(d.get("min_area", 0.0) / (GRID_MM * GRID_MM))),
                "pos": tuple(n["pos"]),
                # aspect limit from demands[id].aspect_max (hallway 6.0, others 2.5)
                "aspect": _aspect_ratio(d.get("aspect_max", ASPECT_MAX)),
            }
    report = {"rooms": {}, "modeled": sorted(dims)}
    if not dims:
        return {"cores": {}, "nodes": nodes, "report": {**report, "reason": "no room has a min_edge demand"}}

    # arrangement source: prefer the mapped source polygons; a room missing
    # from source_rooms (not reachable) falls back to its node position.
    tol_units = 2.0 * GRID_MM / mm
    room_poly, room_bbox, poly_src = {}, {}, []
    for sid in dims:
        poly = (source_rooms or {}).get(sid)
        if poly and len(poly) >= 3:
            room_poly[sid] = poly
            room_bbox[sid] = _poly_bbox_units(poly)
            poly_src.append(sid)
    arrangement_source = ("polygons" if len(poly_src) == len(dims) else
                          "polygons+node_fallback" if poly_src else "nodes")
    report["arrangement_source"] = arrangement_source
    report["arrangement_source_rooms"] = {"polygons": poly_src,
                                          "nodes": sorted(set(dims) - set(poly_src))}

    def _center(sid):
        if sid in room_bbox:
            x0, y0, x1, y1 = room_bbox[sid]
            return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        return dims[sid]["pos"]

    def _side(a, b):
        if a in room_poly and b in room_poly:
            s = _side_from_polys(room_poly[a], room_poly[b], tol_units)
            if s is not None:
                return s
        return _side_from_nodes(dims[a]["pos"], dims[b]["pos"])

    # every source edge between two modelled rooms: door/open hard touch,
    # derived side; plain wall (connection 0) a SOFT same-side reward instead
    # (2026-11-02 — see docstring). pair_weights overrides W_SRC_WALL per pair.
    def _wall_weight(a, b):
        if pair_weights:
            for k in ((a, b), (b, a)):
                if k in pair_weights and pair_weights[k]:
                    return pair_weights[k]
        return W_SRC_WALL

    adjacent_pairs = set()
    door_specs = []      # (a, b, floor_cells, side) — hard
    wall_src_specs = []  # (a, b, side, weight) — soft
    for e in adjacencies or []:
        a, b = str(e.get("u")), str(e.get("v"))
        if a not in dims or b not in dims or a == b:
            continue
        adjacent_pairs.add(tuple(sorted((a, b))))
        side = _side(a, b)
        if _is_door_or_open(e):
            door_specs.append((a, b, int(round(DOOR_MIN_M * 10)), side))
        else:
            wall_src_specs.append((a, b, side, _wall_weight(a, b)))

    # non-adjacent pairs whose source centres differ > ORDER_MIN_M on an axis:
    # keep that axis' centre order, hard (linear on x0+x1 / y0+y1).
    m = mm / 1000.0
    order_specs = []   # (a, b, axis, sign)  sign=+1: a <= b, -1: a >= b
    sids = sorted(dims)
    for i, a in enumerate(sids):
        for b in sids[i + 1:]:
            if tuple(sorted((a, b))) in adjacent_pairs:
                continue
            ca, cb = _center(a), _center(b)
            for axis_i, axis_name in ((0, "x"), (1, "y")):
                if abs(ca[axis_i] - cb[axis_i]) * m > ORDER_MIN_M:
                    order_specs.append((a, b, axis_name, 1 if ca[axis_i] < cb[axis_i] else -1))

    door_pairs = sorted({(a, b) for a, b, _, _s in door_specs})

    forbidden, (bx0c, by0c, bx1c, by1c) = _forbidden_rects(boundary, bpoly, mm)
    edge_geoms = _edge_geometry(boundary, bpoly, edge_types, mm)
    bd = {str(k): [E for E in v if E in edge_geoms] for k, v in (boundary_demands or {}).items()}

    entry_sid = str(entry_room) if entry_room is not None else None
    ent_mid = None
    if entrance and len(entrance) >= 4:
        ent_mid = ((entrance[0] + entrance[2]) / 2.0, (entrance[1] + entrance[3]) / 2.0)
    entrance_status = {"supplied": ent_mid is not None, "entry_room": entry_sid,
                       "entry_modeled": entry_sid in dims, "applied": False}

    def _build(front_hard, order_hard, door_side_free):
        model = cp_model.CpModel()
        rooms = {}
        for sid, d in dims.items():
            w = model.NewIntVar(d["min_edge_cells"], bx1c - bx0c, f"w_{sid}")
            h = model.NewIntVar(d["min_edge_cells"], by1c - by0c, f"h_{sid}")
            x0 = model.NewIntVar(bx0c, bx1c, f"x0_{sid}")
            y0 = model.NewIntVar(by0c, by1c, f"y0_{sid}")
            x1 = model.NewIntVar(bx0c, bx1c, f"x1_{sid}")
            y1 = model.NewIntVar(by0c, by1c, f"y1_{sid}")
            x_itv = model.NewIntervalVar(x0, w, x1, f"xitv_{sid}")
            y_itv = model.NewIntervalVar(y0, h, y1, f"yitv_{sid}")
            area_max = (bx1c - bx0c) * (by1c - by0c)
            area = model.NewIntVar(max(d["min_area_cells"], 0), max(area_max, 1), f"area_{sid}")
            model.AddMultiplicationEquality(area, [w, h])
            an, ad = d["aspect"]
            model.Add(ad * w <= an * h)
            model.Add(ad * h <= an * w)
            rooms[sid] = {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "w": w, "h": h,
                         "area": area, "x_itv": x_itv, "y_itv": y_itv}

        boxes = [(r["x_itv"], r["y_itv"]) for r in rooms.values()]
        for i, (fx0, fy0, fx1, fy1) in enumerate(forbidden):
            fxi = model.NewFixedSizeIntervalVar(fx0, fx1 - fx0, f"fx_{i}")
            fyi = model.NewFixedSizeIntervalVar(fy0, fy1 - fy0, f"fy_{i}")
            boxes.append((fxi, fyi))
        model.AddNoOverlap2D([b[0] for b in boxes], [b[1] for b in boxes])

        # door/open: hard touch, forced onto its derived side unless
        # door_side_free=True (second relaxation rung — touch stays hard,
        # AddBoolOr over all 4 sides instead of the one pinned side).
        edge_dbg = []
        for a, b, floor_cells, side in door_specs:
            opts, _, _ = _add_touch_or(model, rooms[a], rooms[b], floor_cells, f"adj_{a}_{b}")
            if door_side_free:
                model.AddBoolOr(opts)
                edge_dbg.append({"a": a, "b": b, "door": True, "side": side, "forced": False})
            else:
                model.Add(opts[_SIDE_IDX[side]] == 1)
                edge_dbg.append({"a": a, "b": b, "door": True, "side": side, "forced": True})

        # centre order (first relaxation rung)
        order_dbg = []
        if order_hard:
            for a, b, axis, sign in order_specs:
                if axis == "x":
                    ea = rooms[a]["x0"] + rooms[a]["x1"]
                    eb = rooms[b]["x0"] + rooms[b]["x1"]
                else:
                    ea = rooms[a]["y0"] + rooms[a]["y1"]
                    eb = rooms[b]["y0"] + rooms[b]["y1"]
                if sign > 0:
                    model.Add(ea <= eb)
                else:
                    model.Add(ea >= eb)
                order_dbg.append({"a": a, "b": b, "axis": axis, "sign": sign})

        # boundary-front demands — identical to solve_layout
        front_terms = []
        fronted_bool = {}
        for sid, edges in bd.items():
            if sid not in rooms:
                continue
            for E in edges:
                g = edge_geoms[E]
                r = rooms[sid]
                lo_a, hi_a = r[g["span_attrs"][0]], r[g["span_attrs"][1]]
                lo = model.NewIntVar(-_BIG, _BIG, f"flo_{sid}_{E}")
                hi = model.NewIntVar(-_BIG, _BIG, f"fhi_{sid}_{E}")
                model.AddMaxEquality(lo, [lo_a, g["lo"]])
                model.AddMinEquality(hi, [hi_a, g["hi"]])
                overlap = model.NewIntVar(-_BIG, _BIG, f"fov_{sid}_{E}")
                model.Add(overlap == hi - lo)
                if front_hard:
                    model.Add(r[g["attr"]] == g["coord"])
                    model.Add(overlap >= g["floor"])
                else:
                    b_front = model.NewBoolVar(f"bfront_{sid}_{E}")
                    model.Add(r[g["attr"]] == g["coord"]).OnlyEnforceIf(b_front)
                    model.Add(r[g["attr"]] != g["coord"]).OnlyEnforceIf(b_front.Not())
                    shortfall = model.NewIntVar(0, g["floor"], f"short_{sid}_{E}")
                    model.Add(shortfall >= g["floor"] - overlap).OnlyEnforceIf(b_front)
                    model.Add(shortfall >= g["floor"]).OnlyEnforceIf(b_front.Not())
                    front_terms.append(shortfall)
                    fronted_bool[(sid, E)] = b_front

        # entrance: the source entry room fronts the target entrance — always
        # hard, never relaxed.
        if ent_mid is not None and entry_sid in rooms:
            E_ent = _nearest_boundary_edge(boundary, ent_mid)
            g = edge_geoms.get(E_ent)
            if g is not None:
                _add_entrance_front(model, rooms[entry_sid], g, entrance, mm, m)
                entrance_status["applied"] = True
                entrance_status["edge"] = E_ent

        # plain-wall (connection 0) source edges: SOFT same-side reward — the
        # specific derived-side bool, not an any-side OR, since the reward is
        # for reproducing the SOURCE touch, not merely some touch (pair_weights
        # hook: W_SRC_WALL default, overridden per pair by _wall_weight above).
        wall_terms = []
        for a, b, side, weight in wall_src_specs:
            opts, _, _ = _add_touch_or(model, rooms[a], rooms[b], int(round(SOFT_WALL_MIN_M * 10)),
                                       f"wsrc_{a}_{b}")
            wall_terms.append((opts[_SIDE_IDX[side]], weight))

        obj = []
        for sid, r in rooms.items():
            obj.append(OBJ_AREA_W * r["area"])
            px, py = dims[sid]["pos"]
            node_x = _to_cells(px, mm)
            node_y = _to_cells(py, mm)
            dx = model.NewIntVar(-_BIG, _BIG, f"dx_{sid}")
            dy = model.NewIntVar(-_BIG, _BIG, f"dy_{sid}")
            model.Add(dx == (r["x0"] + r["x1"]) - 2 * node_x)
            model.Add(dy == (r["y0"] + r["y1"]) - 2 * node_y)
            adx = model.NewIntVar(0, _BIG, f"adx_{sid}")
            ady = model.NewIntVar(0, _BIG, f"ady_{sid}")
            model.AddAbsEquality(adx, dx)
            model.AddAbsEquality(ady, dy)
            obj.append(-OBJ_NODE_W * (adx + ady))
        for touch, w in wall_terms:
            obj.append(int(round(OBJ_WALL_W * w)) * touch)
        for shortfall in front_terms:
            obj.append(-OBJ_FRONT_SOFT_W * shortfall)
        model.Maximize(sum(obj))
        return model, rooms, fronted_bool, {"edges": edge_dbg, "order": order_dbg}

    # relaxation ladder (never relaxes door/open TOUCH or the entrance):
    # rung 0: order hard, door side forced. rung a: order soft. rung b:
    # order soft + door side free (touch still hard). Each rung tried with
    # boundary fronts hard then soft (kept from solve_layout).
    combos = [
        {"order": True, "door_free": False},
        {"order": False, "door_free": False},
        {"order": False, "door_free": True},
    ]
    t0 = time.time()
    status, relaxation, model, rooms, fronted_bool, build_dbg = (None,) * 6
    relaxations = []
    for combo in combos:
        for front_hard in (True, False):
            model, rooms, fronted_bool, build_dbg = _build(front_hard, combo["order"], combo["door_free"])
            solver = cp_model.CpSolver()
            _set_limits(solver, cpsat_time_s, cpsat_workers, cpsat_seed, cpsat_det_time)
            status = solver.Solve(model)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                relaxations = (["centre_order"] if not combo["order"] else []) + \
                             (["door_open_side"] if combo["door_free"] else []) + \
                             (["boundary_fronts"] if not front_hard else [])
                relaxation = "hard" if not relaxations else "+".join(relaxations)
                break
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
    else:
        relaxation = "infeasible"
    solve_time = time.time() - t0
    unsolved = status not in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    report.update({
        "status": ("cpsat_dim_unsolved" if unsolved else solver.StatusName(status)),
        "solver_status": solver.StatusName(status),
        "unsolved": unsolved,
        "relaxation": relaxation,
        "relaxations": relaxations,
        "solve_time_s": round(solve_time, 3),
        "objective": (solver.ObjectiveValue() if not unsolved else None),
        "door_pairs": [[a, b] for a, b in door_pairs],
        "wall_src_pairs": [[a, b] for a, b, _s, _w in wall_src_specs],
        "order_constraints": len(order_specs),
        "entrance": entrance_status,
    })

    if unsolved:
        # No silent fallback: report.unsolved=True flags this plan explicitly
        # (the diag gate counts it) rather than letting an empty cores dict
        # look indistinguishable from mode "off".
        for sid in dims:
            report["rooms"][sid] = {"placed": False}
        return {"cores": {}, "nodes": nodes, "report": report}

    placed = {}
    for sid, r in rooms.items():
        x0 = solver.Value(r["x0"]) * GRID_MM / mm
        y0 = solver.Value(r["y0"]) * GRID_MM / mm
        x1 = solver.Value(r["x1"]) * GRID_MM / mm
        y1 = solver.Value(r["y1"]) * GRID_MM / mm
        placed[sid] = (x0, y0, x1, y1)

    rect_attr_idx = {"x0": 0, "y0": 1, "x1": 2, "y1": 3}
    for sid, r in rooms.items():
        rect = placed[sid]
        fronted, unfronted = [], []
        for E in bd.get(sid, []):
            g = edge_geoms[E]
            room_coord = rect[rect_attr_idx[g["attr"]]]
            edge_coord = g["coord"] * GRID_MM / mm
            on_edge = abs(room_coord - edge_coord) < 1e-6
            if g["axis"] == "x":
                lo_r, hi_r = rect[1], rect[3]
            else:
                lo_r, hi_r = rect[0], rect[2]
            lo_edge, hi_edge = g["lo"] * GRID_MM / mm, g["hi"] * GRID_MM / mm
            overlap_m = max(0.0, min(hi_r, hi_edge) - max(lo_r, lo_edge)) * m
            floor_m = g["floor"] * GRID_MM / 1000.0
            if on_edge and overlap_m + 1e-6 >= floor_m:
                fronted.append(E)
            else:
                unfronted.append(E)
        door_met = []
        for a, b in door_pairs:
            if sid not in (a, b):
                continue
            other = b if sid == a else a
            shared = _shared_wall_m(rect, placed[other], mm)
            door_met.append({"room": other, "shared_m": round(shared, 3), "ok": shared + 1e-6 >= DOOR_MIN_M})
        report["rooms"][sid] = {
            "placed": True,
            "core": [round(v, 6) for v in rect],
            "area_m2": round((rect[2] - rect[0]) * (rect[3] - rect[1]) * m * m, 3),
            "fronted": fronted,
            "unfronted": unfronted,
            "door_pairs_met": door_met,
            "arrangement_source": "polygons" if sid in room_bbox else "nodes",
        }

    new_nodes = []
    for n in nodes:
        sid = str(n["id"])
        rect = placed.get(sid)
        if rect is None:
            new_nodes.append(n)
            continue
        p = tuple(n["pos"])
        inside = rect[0] <= p[0] <= rect[2] and rect[1] <= p[1] <= rect[3]
        pos = list(p) if inside else [(rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0]
        report["rooms"][sid]["moved_m"] = round(math.hypot(pos[0] - p[0], pos[1] - p[1]) * m, 3)
        report["rooms"][sid]["from"] = [round(v, 6) for v in p]
        report["rooms"][sid]["to"] = [round(v, 6) for v in pos]
        new_nodes.append(dict(n, pos=pos))

    return {"cores": placed, "nodes": new_nodes, "report": report}
