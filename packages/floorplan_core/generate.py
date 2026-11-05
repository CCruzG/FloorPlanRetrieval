"""
generate.py — Stage 0 of the agent harness (docs/agent-harness-build-plan.md §5).

The node-driven geometric solver and the Spec/Plan seam.

Where rectangular_arrange.arrange() builds its grid from the SOURCE plan's room
bounding boxes (cell-assignment-from-geometry), this derives room polygons from
the GRAPH: node positions + the target boundary. It is the locked Generator the
harness wraps — the agent edits a Spec (which room sits at which node, plus
adjacency and dimensional demands) and never touches geometry; generate(spec)
realises the geometry.

Method (node-seeded grid, the arrange-continuous route the build plan calls for):
  1. grid separators from the node coordinates (cluster node x's / y's, bracket
     with the boundary bbox, cut at the midpoints between bands);
  2. assign each node's CONTAINING cell to it, then fill the rest by nearest node;
     step 3 warps node positions with the grid. On a convex boundary this keeps
     each node inside its room; on a NON-convex boundary a node-grid column can
     cross a notch, so a node's clipped room may not contain it. Those cases are
     flagged (report.node_relocated) and the display marker is moved to a
     guaranteed-interior point. This is the known weakness of the grid realiser —
     a true Voronoi partition would guarantee node-in-cell, at the cost of
     non-rectangular rooms;
  3. re-solve the grid-line positions to fill the target boundary, enforcing each
     room's minimum edge via the longest-path solve and handing slack to flexible
     rooms (reused verbatim from rectangular_arrange); once that solve is feasible,
     an area-lift loop raises a room's per-axis minimum (one axis, or both, picking
     the worst-off room first) whenever its solved cell area misses min_area, and
     re-solves — a trial is kept only if both axes stay feasible and the room's
     bbox long/short ratio does not exceed ASPECT_MAX (or its pre-trial ratio),
     so a lift cannot turn a room into a sliver, and no room that met min_area
     before the trial falls below it;
  4. build each room's polygon from its cells and clip to the boundary polygon, so
     a non-rectangular boundary is followed rather than overflowed.

Stage 0 is geometry only: the report carries coverage / min-size / dropped /
clipped signals, NOT daylight or access — those are Stage-1 sensors
(topo_sensors) that read this Plan.

Public API:
    generate(spec, options=None) -> Plan          # validates the Spec first
    reassign(spec, room_a, room_b) -> Spec        # swap two rooms' nodes (seam demo)
The Spec itself (builder, options, validator) lives in spec.py.
"""

import math
from collections import deque

from shapely.geometry import Polygon as SPoly, Point as SPoint, box as sbox, LineString
from shapely.ops import unary_union, nearest_points

from packages.floorplan_core.rectangular_arrange import (
    _bbox, _cluster, _flex, _largest_ring,
    _realised_adjacency, reconcile_edges, FLEX_DEFAULT,
)
# Canonical labels (int type / "Kitchen" / "living_room" -> category vocabulary).
# rectangular_arrange._label passes the raw spelling through, which missed every
# MINIMUM_SIZES / FLEX key, so the solver ran with zero minimums.
from packages.floorplan_core.topo_signature import _label
from packages.floorplan_core.spec import (
    build_options, validate_spec, CPSAT_SEED, CPSAT_WORKERS, CPSAT_TIME_S,
)

# Minimum room edges (mm) live in the fitting-prototype schema; rectangular_arrange
# puts that dir on sys.path at import time, so this import resolves after it.
from schema import MINIMUM_SIZES, SCALE  # noqa: E402

SCALE_M = SCALE / 1000.0   # normalized boundary unit -> metres (same as the raycast)
# BADS minimum opening widths the raycast tests a contact against, by edge type.
_EDGE_FLOOR_M = {"facing_corridor": 0.9, "external": 1.2}


# Pin-guard scale: 0.0 pins every boundary vertex (default); 1.0 skips boundary
# jogs shorter than the smallest room min_edge, letting them fall back to the
# clip. Measured 2026-10-30 on test_boundary_001b/005b (top 3): all pins -> 7
# failing rooms, 0 clip losses; guard -> 13 (clip losses return); no pins -> 9.
PIN_MIN_SEG_FACTOR = 0.0

# Area-lift aspect guard (step 3): a lift trial is rejected if it would push a
# room's bbox long/short ratio above this — a room may already be worse than
# ASPECT_MAX going in, in which case the cap is that prior ratio (never tighten).
ASPECT_MAX = 2.5


def _core_block(cells, node_cell, xs, ys, min_edge):
    """The room's core block: the rectangle of its own grid cells that step 3
    holds to min_edge on both axes, in place of the cell bbox. A bbox span lets an
    L pass with a one-column arm (D24 core fails: no min_edge square fits); a core
    span guarantees a min_edge rectangle somewhere in the room.

    Picks, over every all-owned rectangle [c0, c1) x [r0, r1): contains the node
    cell first, then pre-solve min side capped at min_edge (the block needing the
    least stretch), then pre-solve area. Returns (c0, c1, r0, r1)."""
    cs = set(cells)
    cmin = min(c[0] for c in cs); cmax = max(c[0] for c in cs) + 1
    rmin = min(c[1] for c in cs); rmax = max(c[1] for c in cs) + 1
    best, best_key = None, None
    for c0 in range(cmin, cmax):
        for r0 in range(rmin, rmax):
            if (c0, r0) not in cs:
                continue
            c1_lim = cmax
            for r1 in range(r0 + 1, rmax + 1):
                # widest run of owned cells along row r1-1 from c0, capped by rows above
                c1 = c0
                while c1 < c1_lim and (c1, r1 - 1) in cs:
                    c1 += 1
                c1_lim = c1
                if c1_lim == c0:
                    break
                w = xs[c1_lim] - xs[c0]
                h = ys[r1] - ys[r0]
                has_node = (node_cell is not None and c0 <= node_cell[0] < c1_lim
                            and r0 <= node_cell[1] < r1)
                cap = min_edge if min_edge > 0 else float("inf")
                key = (has_node, min(w, h, cap), w * h)
                if best_key is None or key > best_key:
                    best, best_key = (c0, c1_lim, r0, r1), key
    return best


# ── Spec / Plan ──────────────────────────────────────────────────────────────
# The Spec (spec.py: build_spec, build_options, validate_spec; field table in
# ARCHITECTURE.md "The Spec") is a plain dict. It is the ONLY thing a move writes.
# Switches that are not part of the description (core_placement, core_block,
# area_lift, source_rooms, pair_weights, CP-SAT seed/workers/time) travel in
# `options`. Plan = {"rooms":[{id,type,polygon}], "edges", "entrance", "node_pos",
#                    "feasible", "report"}


def _poly_area(poly):
    """Shoelace area of a simple polygon [[x, y], ...] (sign-agnostic)."""
    n = len(poly)
    return abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
                   for i in range(n))) / 2.0


def reassign(spec, room_a, room_b):
    """Return a new Spec with rooms a and b swapped onto each other's node
    positions — the minimal spec edit that re-derives geometry through the seam
    (the Stage-0 preview of Stage-2's SwapNodes move). Pure; does not mutate."""
    new_nodes = []
    pos_by_id = {n["id"]: n["pos"] for n in spec["nodes"]}
    for n in spec["nodes"]:
        m = dict(n)
        if n["id"] == room_a:
            m["pos"] = list(pos_by_id[room_b])
        elif n["id"] == room_b:
            m["pos"] = list(pos_by_id[room_a])
        new_nodes.append(m)
    out = dict(spec)
    out["nodes"] = new_nodes
    return out


# ── the solver ───────────────────────────────────────────────────────────────

def _grid_lines(coords, lo, hi, tol, fixed=(), min_seg=0.0, walls=None):
    """Separator lines along one axis from a bag of node coordinates: cluster the
    coords into bands, cut at the midpoints between consecutive bands, and bracket
    with the boundary extent [lo, hi]. n bands -> n columns.

    `fixed` are boundary-vertex coordinates on this axis (notch lines). They are
    added as lines so every cell of an orthogonal boundary lies wholly inside or
    wholly outside it — the solver then never credits a room with area the clip
    removes. A node cut within `tol` of a fixed line is dropped (the fixed line
    does its job). A fixed line closer than `min_seg` (the smallest room min_edge)
    to the extent or to another kept pin is skipped: pinning a short boundary jog
    would make a strip no room can occupy, so that jog falls back to the clip.
    Returns (lines, fixed_idx): fixed_idx are the indices of lines pinned to the
    boundary (always including both ends)."""
    if walls is not None:
        # options["source_grid"], experiment 2026-11-05: cut at the source plan's
        # walls (mapped to the target frame), not midway between nodes.
        cuts = _cluster([c for c in walls if lo + tol < c < hi - tol], tol * 0.5)
    else:
        bands = _cluster([c for c in coords if lo - tol <= c <= hi + tol], tol)
        cuts = [(bands[k] + bands[k + 1]) / 2.0 for k in range(len(bands) - 1)]
    gap = max(tol, min_seg)
    pins = []
    for f in sorted(set(fixed)):
        if lo + gap <= f <= hi - gap and (not pins or f - pins[-1] >= gap):
            pins.append(f)
    cuts = [c for c in cuts if all(abs(c - f) > tol for f in pins)]
    tagged = sorted([(lo, True), (hi, True)] + [(f, True) for f in pins]
                    + [(c, False) for c in cuts])
    lines = [v for v, _ in tagged]
    fixed_idx = {k for k, (_, is_pin) in enumerate(tagged) if is_pin}
    return lines, fixed_idx


def _solve_to_source(n_sep, spans, total, widths, eps, fixed=None):
    """Separator positions that keep each gap close to its source width (experiment,
    2026-11-05, options["source_grid"] True). _solve_positions puts a span's whole
    minimum into its last gap, so a narrow gap can inherit a neighbour's minimum.
    Here the deviation from the source is minimised, weighted so a narrow gap is the
    most protected: slack falls to the wide gaps. Same return as _solve_positions;
    falls back to it when no layout meets the minimums."""
    from ortools.sat.python import cp_model
    unit = total / 100000.0
    to_int = lambda v: int(round(v / unit))  # noqa: E731
    pins = dict(fixed or {})
    pins[0], pins[n_sep] = 0.0, total
    keys = sorted(pins)
    for a, b in zip(keys, keys[1:]):
        eps = min(eps, (pins[b] - pins[a]) / (2.0 * (b - a)))
    T, E = 100000, max(to_int(eps), 1)
    model = cp_model.CpModel()
    pos = [model.NewIntVar(0, T, f"p{k}") for k in range(n_sep + 1)]
    for k, v in pins.items():
        model.Add(pos[k] == to_int(v))
    for k in range(n_sep):
        model.Add(pos[k + 1] - pos[k] >= E)
    for lo, hi, mn in spans:
        model.Add(pos[hi] - pos[lo] >= int(math.ceil(mn / unit)))
    narrowest = max(min(widths), eps)
    cost = []
    for k in range(n_sep):
        dev = model.NewIntVar(0, T, f"d{k}")
        w = to_int(widths[k])
        model.Add(dev >= pos[k + 1] - pos[k] - w)
        model.Add(dev >= w - pos[k + 1] + pos[k])
        cost.append(dev * max(int(round(100 * narrowest / max(widths[k], narrowest))), 1))
    model.Minimize(sum(cost))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 1
    solver.parameters.max_deterministic_time = 2.0
    if solver.Solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _solve_positions(n_sep, spans, total, widths, eps, fixed=fixed)
    out = [solver.Value(p) * unit for p in pos]
    out[-1] = total
    return out, True


def _solve_positions(n_sep, spans, total, gap_flex, eps, fixed=None):
    """Separator positions along one axis by longest path (the arrange algorithm,
    made unit-relative via `eps` so it works in normalized OR mm space).

    `spans` are (lo, hi, min_size): position[hi] - position[lo] >= min_size for a
    room occupying grid columns [lo, hi). Returns (positions in [0, total],
    all_min_met); every gap is at least `eps`. Slack over the minimal feasible
    layout is handed to the gaps a flexible room occupies.

    `fixed` {k: position} pins separators to the boundary (notch lines). Without
    interior pins this is the original single-segment solve. With them, each
    segment between consecutive pins gets the min layout + its own slack by flex,
    then alternating forward/backward passes repair spans that cross a pin."""
    m = n_sep
    spans_by_hi, spans_by_lo = {}, {}
    for lo, hi, mn in spans:
        spans_by_hi.setdefault(hi, []).append((lo, mn))
        spans_by_lo.setdefault(lo, []).append((hi, mn))
    pins = dict(fixed or {})
    pins[0], pins[m] = 0.0, total
    interior = [k for k in pins if 0 < k < m]
    if not interior:
        pos = [0.0] * (m + 1)
        for k in range(1, m + 1):
            pos[k] = pos[k - 1] + eps
            for lo, mn in spans_by_hi.get(k, []):
                pos[k] = max(pos[k], pos[lo] + mn)
        min_total = pos[m]
        if min_total <= total:
            slack = total - min_total
            wsum = sum(gap_flex) or 1.0
            widths = [pos[k + 1] - pos[k] for k in range(m)]
            widths = [w + slack * (gf / wsum) for w, gf in zip(widths, gap_flex)]
            out = [0.0]
            for w in widths:
                out.append(out[-1] + w)
            out[-1] = total
            return out, True
        scale = total / min_total if min_total > 0 else 1.0
        return [p * scale for p in pos], False

    # pinned: a gap can be no wider than its segment allows, so cap eps per segment
    keys = sorted(pins)
    for a, b in zip(keys, keys[1:]):
        eps = min(eps, (pins[b] - pins[a]) / (2.0 * (b - a)))
    # 1) per-segment min layout + slack by flex
    pos = [0.0] * (m + 1)
    for a, b in zip(keys, keys[1:]):
        pos[a] = pins[a]
        for k in range(a + 1, b + 1):
            pos[k] = pos[k - 1] + eps
            for lo, mn in spans_by_hi.get(k, []):
                if lo >= a:
                    pos[k] = max(pos[k], pos[lo] + mn)
        slack = max(pins[b] - pos[b], 0.0)
        widths = [pos[k + 1] - pos[k] for k in range(a, b)]
        fl = gap_flex[a:b]
        wsum = sum(fl) or 1.0
        acc = pins[a]
        for i, k in enumerate(range(a, b)):
            acc += widths[i] + slack * (fl[i] / wsum)
            pos[k + 1] = acc
        pos[b] = pins[b]
    # 2) repair spans crossing a pin (pins never move)
    for _ in range(4 * (m + 1)):
        changed = False
        for k in range(1, m + 1):
            if k in pins:
                continue
            lb = max([pos[k - 1] + eps] + [pos[lo] + mn for lo, mn in spans_by_hi.get(k, [])])
            if pos[k] < lb - 1e-12:
                pos[k], changed = lb, True
        for k in range(m - 1, 0, -1):
            if k in pins:
                continue
            ub = min([pos[k + 1] - eps] + [pos[hi] - mn for hi, mn in spans_by_lo.get(k, [])])
            if pos[k] > ub + 1e-12:
                pos[k], changed = ub, True
        if not changed:
            break
    tol = 1e-9 * max(total, 1.0)
    ok = (all(pos[k + 1] - pos[k] >= eps - tol for k in range(m))
          and all(pos[hi] - pos[lo] >= mn - tol for lo, hi, mn in spans))
    return pos, ok


def _edge_border_cells(boundary, xs, ys, ncol, nrow):
    """Map each TARGET boundary edge index -> the set of grid cells (i, j) that
    front it. Geometric, so it is correct on non-convex (L-shaped) boundaries: each
    grid cell is clipped to the boundary, and a cell fronts edge e when the clipped
    cell shares a wall segment with e (positive overlap length). Grid lines xs/ys
    bracket the bbox; re-dimensioning slides them but preserves which cell borders
    which edge, so this pre-resolve mapping predicts the realised frontage."""
    bpoly = SPoly(boundary)
    if not bpoly.is_valid:
        bpoly = bpoly.buffer(0)
    _xsb = [p[0] for p in boundary]
    _ysb = [p[1] for p in boundary]
    diag = math.hypot(max(_xsb) - min(_xsb), max(_ysb) - min(_ysb)) or 1.0
    tol = 1e-6 * diag
    nb = len(boundary)
    elines = [LineString([boundary[k], boundary[(k + 1) % nb]]) for k in range(nb)]
    out = {k: set() for k in range(nb)}
    for i in range(ncol):
        for j in range(nrow):
            g = sbox(xs[i], ys[j], xs[i + 1], ys[j + 1]).intersection(bpoly)
            if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            if g.area <= tol * tol:
                continue
            gb = g.boundary
            if gb is None or gb.is_empty:
                continue
            for k, el in enumerate(elines):
                shared = gb.intersection(el)
                if not shared.is_empty and shared.length > tol:
                    out[k].add((i, j))
    return out


def _num_components(cells):
    """Count of 4-connected components over a set of (i, j) grid cells."""
    cset = set(cells)
    seen = set()
    comps = 0
    for c in cset:
        if c in seen:
            continue
        comps += 1
        q = deque([c])
        seen.add(c)
        while q:
            i, j = q.popleft()
            for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if nb in cset and nb not in seen:
                    seen.add(nb)
                    q.append(nb)
    return comps


def _bfs_cell_path(starts, goals, ncol, nrow, allowed=None):
    """Shortest 4-connected grid path from any start cell to any goal cell
    (fewest cells = fewest transfers). Returns the path incl. endpoints, or None.
    `allowed` (a set of cells) restricts the walk — cells outside the boundary
    have no owner and must not be crossed or transferred."""
    starts = set(starts)
    goals = set(goals)
    prev = {}
    q = deque()
    for s in starts:
        prev[s] = None
        q.append(s)
    while q:
        c = q.popleft()
        if c in goals and c not in starts:
            path = [c]
            while prev[c] is not None:
                c = prev[c]
                path.append(c)
            return list(reversed(path))
        i, j = c
        for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if (0 <= nb[0] < ncol and 0 <= nb[1] < nrow and nb not in prev
                    and (allowed is None or nb in allowed)):
                prev[nb] = c
                q.append(nb)
    return None


def _entrance_cell(boundary, entrance, xs, ys, ncol, nrow, edge_cell_sets):
    """Grid cell in front of the entrance: of the cells that front the boundary edge
    nearest the entrance's midpoint, the one whose span contains it (else the nearest)."""
    if not entrance or len(entrance) < 4:
        return None
    pt = SPoint((entrance[0] + entrance[2]) / 2.0, (entrance[1] + entrance[3]) / 2.0)
    nb = len(boundary)
    E = min(range(nb), key=lambda k: LineString([boundary[k], boundary[(k + 1) % nb]]).distance(pt))
    cells = edge_cell_sets.get(E) or set()
    if not cells:
        return None
    return min(cells, key=lambda c: (sbox(xs[c[0]], ys[c[1]], xs[c[0] + 1], ys[c[1] + 1]).distance(pt), c))


def _repair_boundary_contacts(owner, boundary_demands, edge_cell_sets, ncol, nrow,
                              core_cells=None, first=None):
    """Make each room front its demanded target edges by transferring border cells
    (topology -> geometry). A transfer is committed only if it keeps every donor
    non-empty and 4-connected AND keeps each donor's own fronts; rooms process
    most-constrained-first for determinism. Returns (owner, report); report lists
    satisfied / unmet demands and the cells moved. Demands it cannot satisfy are
    reported, not forced — the guarantee is conditional on a feasible transfer.

    core_cells: {cell: rid} — step 0's placed-core cells. A cell step 0 placed
    inside a room's core can never be moved OUT of it here: that promise (a
    min_edge rectangle somewhere in the room) is step 0's contract with step 3,
    and 2b stealing from it silently breaks a DIFFERENT room's core without
    the repair's own `valid()` check (donor connected + keeps its own fronts)
    ever seeing it (2026-11-02 finding — see generate()'s 2b call site).

    first: a room to process before all others (the entry room, 2026-11-04)."""
    report = {"satisfied": [], "unmet": [], "transfers": []}
    if not boundary_demands:
        return owner, report
    core_cells = core_cells or {}

    def fronts(rid, E, own):
        return any(own.get(c) == rid for c in (edge_cell_sets.get(E) or set()))

    def valid(old_owner, new_owner, moved):
        for d in {old_owner[c] for c in moved}:
            new_cells = [c for c, r in new_owner.items() if r == d]
            if not new_cells or _num_components(new_cells) != 1:
                return False
            if any(not fronts(d, E, new_owner) for E in boundary_demands.get(d, [])):
                return False
        return True

    order = sorted(boundary_demands, key=lambda r: (r != first, -len(boundary_demands[r]), str(r)))
    for rid in order:
        for E in boundary_demands[rid]:
            if fronts(rid, E, owner):
                report["satisfied"].append([str(rid), E])
                continue
            starts = [c for c, r in owner.items() if r == rid]
            goals = [c for c in (edge_cell_sets.get(E) or set()) if owner.get(c) != rid]
            # Shortest path to EACH candidate edge cell; try them shortest-first and
            # commit the first transfer that validates. Trying only the single
            # nearest cell would give up when its donor can't spare it, even though
            # a slightly farther cell (a donor with room to spare) would work.
            cands = []
            for g in goals:
                p = (_bfs_cell_path(starts, {g}, ncol, nrow, allowed=set(owner))
                     if starts else None)
                if p:
                    cands.append(p)
            cands.sort(key=len)
            committed = False
            for path in cands:
                moved = [c for c in path if owner.get(c) != rid]
                if any(core_cells.get(c) not in (None, rid) for c in moved):
                    continue   # would steal a cell from another room's placed core
                trial = dict(owner)
                for c in moved:
                    trial[c] = rid
                if valid(owner, trial, moved):
                    owner = trial
                    report["satisfied"].append([str(rid), E])
                    report["transfers"].append(
                        {"room": str(rid), "edge": E, "cells": [list(c) for c in moved]})
                    committed = True
                    break
            if not committed:
                report["unmet"].append([str(rid), E])
    return owner, report


def _precedence_claims(owner, precedence, contain, xs, ys, min_edge_of,
                       boundary_demands, edge_cell_sets, choice=None, edge_types=None):
    """Step 2c — precedence core claim (the SetPrecedence move's realisation).
    For each [winner, loser] pair, the winner is re-cut as one rectangle of grid
    cells drawn from its own cells and the loser's: the rectangle may sit anywhere
    over the two rooms (never on the loser's node cell; when it leaves out the
    winner's node cell, the winner's node moves to the rectangle — the move
    relocates the room — and the entry reports node_moved),
    and the winner's cells left outside it go back to the loser (or, when a
    leftover piece does not touch the loser, to the neighbour it shares most sides
    with). An S- or L-shaped room becomes a rectangle at its neighbour's expense,
    and returns the area it does not use.

    A rectangle is valid only if every room it touches stays non-empty and
    4-connected and keeps its demanded fronts — except that the winner may drop a
    facing_corridor / shared_wall front (the precedence trades its corridor
    contact for a usable rectangle; reachability / access re-judge it). The
    winner keeps its external (window) fronts. Dropped fronts are reported. Pre-solve widths cannot rank the
    valid ones — the step-3 solve re-sizes whichever rectangle becomes the core —
    so this function lists them (report[k]["candidates"], fewest cells moved
    first) and applies the one `choice[k]` names; generate() picks it by trial
    solve. A pair absent from `choice` (or mapped to None) is not applied.
    Returns (owner, report)."""
    report = []
    if not precedence:
        return owner, report
    choice = choice or {}
    id_by_str = {str(r): r for r in set(owner.values())}

    def holds(rid, E, own):
        return any(own.get(c) == rid for c in (edge_cell_sets.get(E) or set()))

    def etype(E):
        return edge_types[E] if (edge_types and 0 <= E < len(edge_types)) else "external"

    def fronts(rid, own, winner=False):
        return all(holds(rid, E, own) for E in boundary_demands.get(rid, [])
                   if not (winner and etype(E) in ("facing_corridor", "shared_wall")))

    def claim(own, w, l, rect):
        """Apply one rectangle; returns (trial owner, leftover) or None if invalid."""
        cw = {c for c, r in own.items() if r == w}
        trial = dict(own)
        for c in rect:
            trial[c] = w
        leftover = cw - rect
        pieces, seen = [], set()
        for c in leftover:
            if c in seen:
                continue
            comp, q = set(), deque([c])
            seen.add(c)
            while q:
                i, j = q.popleft()
                comp.add((i, j))
                for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if nb in leftover and nb not in seen:
                        seen.add(nb)
                        q.append(nb)
            pieces.append(comp)
        touched = {l}
        for comp in pieces:
            votes = {}
            for i, j in comp:
                for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    r = trial.get(nb)
                    if nb not in comp and r is not None and r != w:
                        votes[r] = votes.get(r, 0) + 1
            if not votes:
                return None
            to = l if l in votes else max(votes, key=lambda r: (votes[r], str(r)))
            for c in comp:
                trial[c] = to
            touched.add(to)
        for r in touched | {w}:
            rc = [c for c, o in trial.items() if o == r]
            if not rc or _num_components(rc) != 1 or not fronts(r, trial, winner=(r == w)):
                return None
        return trial, leftover

    for k, pair in enumerate(precedence):
        w_s, l_s = str(pair[0]), str(pair[1])
        w, l = id_by_str.get(w_s), id_by_str.get(l_s)
        entry = {"winner": w_s, "loser": l_s, "applied": False, "candidates": []}
        report.append(entry)
        if w is None or l is None or w == l:
            entry["reason"] = "unknown room"
            continue
        cw = {c for c, r in owner.items() if r == w}
        cl = {c for c, r in owner.items() if r == l}
        pool = cw | cl
        nw, nl = contain.get(w), contain.get(l)
        cmin = min(c[0] for c in pool); cmax = max(c[0] for c in pool) + 1
        rmin = min(c[1] for c in pool); rmax = max(c[1] for c in pool) + 1
        cands = []
        for c0 in range(cmin, cmax):
            for c1 in range(c0 + 1, cmax + 1):
                for r0 in range(rmin, rmax):
                    for r1 in range(r0 + 1, rmax + 1):
                        rect = {(i, j) for i in range(c0, c1) for j in range(r0, r1)}
                        if rect == cw or not rect <= pool:
                            continue
                        if nl is not None and nl in rect:
                            continue
                        if claim(owner, w, l, rect) is None:
                            continue
                        moved = len(rect - cw) + len(cw - rect)
                        cands.append((moved, -len(rect), (c0, c1, r0, r1)))
        cands.sort()
        entry["candidates"] = [list(t[2]) for t in cands]
        pick = choice.get(k)
        if pick is None:
            entry["reason"] = ("no valid rectangle" if not cands
                               else "not chosen" if k in choice else "pending trial")
            continue
        c0, c1, r0, r1 = pick
        rect = {(i, j) for i in range(c0, c1) for j in range(r0, r1)}
        res = claim(owner, w, l, rect)
        if res is None:
            entry["reason"] = "chosen rectangle no longer valid"
            continue
        before = owner
        owner, leftover = res
        entry.update({"applied": True, "rect": [c0, c1, r0, r1],
                      "node_moved": nw not in rect,
                      "fronts_dropped": [E for E in boundary_demands.get(w, [])
                                         if holds(w, E, before) and not holds(w, E, owner)],
                      "taken": [list(c) for c in sorted(rect - cw)],
                      "returned": [list(c) for c in sorted(leftover)]})
    return owner, report


def _edge_floor_units(edge_types, E):
    """The target-unit width floor a contact on edge E must clear to count as a
    window/door (BADS). shared_wall / notch edges have no opening floor."""
    et = edge_types[E] if (edge_types and 0 <= E < len(edge_types)) else "external"
    return _EDGE_FLOOR_M.get(et, 0.0) / SCALE_M


def _fronts_edge_geom(poly, eline, floor_units):
    """True if a room polygon shares at least `floor_units` of wall with edge line."""
    try:
        shared = poly.boundary.intersection(eline)
    except Exception:
        return False
    return (not shared.is_empty) and shared.length >= floor_units


def _real_parts(geom, eps=1e-12):
    """Count of non-sliver polygon parts (1 for a clean single room)."""
    if geom.is_empty:
        return 0
    if geom.geom_type == "Polygon":
        return 1
    return sum(1 for g in geom.geoms
               if g.geom_type == "Polygon" and g.area > eps)


def _largest_poly(geom):
    """The largest polygon part of a (possibly multi) geometry."""
    if geom.geom_type == "Polygon":
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    return max(parts, key=lambda p: p.area) if parts else geom


def _frange(lo, hi, step):
    out, v = [], lo
    while v <= hi + 1e-12:
        out.append(v)
        v += step
    if out and out[-1] < hi:
        out.append(hi)
    return out


def _carve_arm(poly_by_id, bpoly, R_id, eline, bd, elines, edge_types, arm_w):
    """Give room R_id a thin L-shaped arm reaching edge `eline`, carved from the
    rooms it crosses (the source's thin reach that a rectangular partition can't
    make). A carve commits only if every donor stays non-empty, single-piece and
    keeps its own demanded fronts, and R stays single-piece. Mutates poly_by_id;
    returns the list of donor ids on success (possibly empty), None on failure."""
    Rp = poly_by_id.get(R_id)
    if Rp is None or Rp.is_empty:
        return None
    pr, _pe = nearest_points(Rp, eline)
    (ex0, ey0) = eline.coords[0]
    (ex1, ey1) = eline.coords[-1]
    vertical = abs(ex1 - ex0) < abs(ey1 - ey0)
    rminx, rminy, rmaxx, rmaxy = Rp.bounds
    if vertical:
        lo = max(min(ey0, ey1), rminy)
        hi = min(max(ey0, ey1), rmaxy)
        x_lo, x_hi = sorted((ex0, pr.x))
        band = lambda s: sbox(x_lo, s, x_hi, s + arm_w)      # noqa: E731
        anchor = pr.y
    else:
        lo = max(min(ex0, ex1), rminx)
        hi = min(max(ex0, ex1), rmaxx)
        y_lo, y_hi = sorted((ey0, pr.y))
        band = lambda s: sbox(s, y_lo, s + arm_w, y_hi)      # noqa: E731
        anchor = pr.x
    if hi - lo < arm_w:                   # not enough shared span to seat the arm
        return None
    cands = sorted(_frange(lo, hi - arm_w, arm_w / 2.0),
                   key=lambda s: abs((s + arm_w / 2.0) - anchor))
    for s in cands:
        arm = band(s).intersection(bpoly)
        if arm.is_empty or arm.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        if not _fronts_edge_geom(arm, eline, arm_w * 0.9):   # must reach the edge
            continue
        donors = [rid for rid, p in poly_by_id.items()
                  if rid != R_id and p is not None and not p.is_empty
                  and p.intersects(arm) and p.intersection(arm).area > 1e-12]
        trial, ok = {}, True
        for d in donors:
            d2 = poly_by_id[d].difference(arm)
            if d2.is_empty or _real_parts(d2) != 1:
                ok = False
                break
            if any(not _fronts_edge_geom(d2, elines[E], _edge_floor_units(edge_types, E))
                   for E in bd.get(d, [])):
                ok = False
                break
            trial[d] = _largest_poly(d2)
        if not ok:
            continue
        r2 = unary_union([Rp, arm])
        if _real_parts(r2) != 1:
            continue
        for d, g in trial.items():
            poly_by_id[d] = g
        poly_by_id[R_id] = _largest_poly(r2)
        return list(trial)
    return None


def _carve_unmet_arms(poly_by_id, boundary, bpoly, bd, unmet, edge_types):
    """Try a thin-arm carve for each unmet boundary demand. Returns the demands
    satisfied by carving (same [str_id, edge] shape as the repair report) and
    {donor_id: [recipient_id, ...]} — which rooms gave area to an arm."""
    nb = len(boundary)
    elines = [LineString([boundary[k], boundary[(k + 1) % nb]]) for k in range(nb)]
    id_by_str = {str(k): k for k in poly_by_id}
    satisfied, donors_of = [], {}
    for item in list(unmet):
        rid_s, E = item
        rid = id_by_str.get(str(rid_s))
        if rid is None or not (0 <= E < nb):
            continue
        arm_w = _edge_floor_units(edge_types, E) * 1.05
        if arm_w <= 0:
            continue
        donors = _carve_arm(poly_by_id, bpoly, rid, elines[E], bd, elines, edge_types, arm_w)
        if donors is not None:
            satisfied.append(item)
            for d in donors:
                donors_of.setdefault(d, []).append(rid)
    return satisfied, donors_of


# Step-2c trial solves: at most this many candidate rectangles are re-solved per
# precedence pair (candidates are listed fewest-cells-moved first).
CLAIM_TRIALS = 16


def _claim_score(plan, w, l):
    """Rank a trial generate for a precedence pair: winner core ok, loser core ok,
    rooms with a core ok, step-3 min sizes met (ok = meets min_edge and intact)."""
    rep = plan.get("report") or {}
    cores = (rep.get("grid_lines") or {}).get("cores") or {}
    ok = {k: bool(v.get("meets") and v.get("intact")) for k, v in cores.items()}
    return (ok.get(str(w), False), ok.get(str(l), False), sum(ok.values()),
            bool(rep.get("min_sizes_met")))


def _generate_with_claims(spec, options):
    """Resolve spec["precedence"] pair by pair: list the valid rectangles (a probe
    generate), re-solve each (up to CLAIM_TRIALS), keep the best by _claim_score
    if it is no worse than not claiming, then generate once with every choice."""
    prec = spec.get("precedence") or []
    choice, trials_log = {}, []
    for k, pair in enumerate(prec):
        probe = _realise(spec, options, dict(choice))
        entries = (probe.get("report") or {}).get("precedence") or []
        cands = entries[k]["candidates"] if k < len(entries) else []
        base = _claim_score(probe, pair[0], pair[1])
        best, best_score = None, None
        for rect in cands[:CLAIM_TRIALS]:
            trial = _realise(spec, options, {**choice, k: tuple(rect)})
            sc = _claim_score(trial, pair[0], pair[1])
            if best_score is None or sc > best_score:
                best, best_score = tuple(rect), sc
        choice[k] = best if (best is not None and best_score >= base) else None
        trials_log.append({"pair": [str(pair[0]), str(pair[1])], "tried": min(len(cands), CLAIM_TRIALS),
                           "baseline": list(base), "best": list(best_score) if best_score else None})
    final = _realise(spec, options, choice)
    for e, t in zip((final.get("report") or {}).get("precedence") or [], trials_log):
        e["trial"] = t
        if not e["applied"] and e.get("reason") == "not chosen":
            e["reason"] = "no rectangle scored at least as well as not claiming"
    return final


def generate(spec, options=None):
    """Realise a Plan from a Spec: room polygons from node positions + boundary.

    Validates the Spec first (spec.validate_spec) and raises ValueError on a
    malformed one. Does not change `spec` or `options`. See the module docstring
    for the four steps. Returns the Plan dict.
    """
    validate_spec(spec)
    options = build_options() if options is None else options
    if spec.get("precedence"):
        return _generate_with_claims(spec, options)
    return _realise(spec, options, None)


def _realise(spec, options, claim_choice):
    """The Generator proper. `claim_choice` is internal: the step-2c rectangle
    chosen per precedence pair (None = not resolved yet)."""
    boundary = spec["boundary"]
    nodes = [n for n in spec.get("nodes", []) if n.get("pos")]
    if not nodes or len(boundary) < 3:
        return {"rooms": [], "edges": spec.get("adjacencies", []),
                "entrance": spec.get("entrance"), "node_pos": {},
                "feasible": False,
                "report": {"reason": "no nodes or degenerate boundary",
                           "debug_steps": []}}

    bx0, by0, bx1, by1 = _bbox(boundary)
    BW, BH = bx1 - bx0, by1 - by0
    mm = spec.get("mm_per_unit")           # boundary-units -> mm, for min-size
    debug_steps = []

    # ── 0. core placement (SPIKE, options["core_placement"], default off): a core
    # block per room, sized by use and placed inside the boundary; nodes move into
    # their blocks. The block edges become pinned grid lines and its cells are
    # seeded to the room, so every placed room starts from a min_edge rectangle.
    placed_cores, placement_report = {}, None
    _cp_mode = options.get("core_placement")
    if _cp_mode in (False, None, "", "off"):
        _cp_mode = False
    if _cp_mode and mm and mm > 0:
        if _cp_mode == "cpsat":
            from packages.floorplan_core.layout_cpsat import solve_layout as _place
            _pl = _place(nodes, spec.get("demands", {}) or {}, mm, boundary,
                        spec.get("boundary_demands") or {}, spec.get("edge_types"),
                        spec.get("adjacencies") or [], spec.get("precedence") or [],
                        pair_weights=options.get("pair_weights"),
                        entry_room=spec.get("entry_room"),
                        entrance=spec.get("entrance"),
                        cpsat_seed=options.get("cpsat_seed", CPSAT_SEED),
                        cpsat_workers=options.get("cpsat_workers", CPSAT_WORKERS),
                        cpsat_time_s=options.get("cpsat_time_s", CPSAT_TIME_S),
                        cpsat_det_time=options.get("cpsat_det_time"))
        elif _cp_mode == "cpsat_dim":
            # Sizes rooms on the RETRIEVED arrangement — the source plan supplies
            # the layout, CP-SAT only sizes/places each room's rectangle around it
            # (see layout_cpsat.solve_layout_dim docstring for the constraint set).
            from packages.floorplan_core.layout_cpsat import solve_layout_dim as _place
            _pl = _place(nodes, spec.get("demands", {}) or {}, mm, boundary,
                        spec.get("boundary_demands") or {}, spec.get("edge_types"),
                        spec.get("adjacencies") or [], spec.get("precedence") or [],
                        pair_weights=options.get("pair_weights"),
                        source_rooms=options.get("source_rooms"),
                        entry_room=spec.get("entry_room"),
                        entrance=spec.get("entrance"),
                        cpsat_seed=options.get("cpsat_seed", CPSAT_SEED),
                        cpsat_workers=options.get("cpsat_workers", CPSAT_WORKERS),
                        cpsat_time_s=options.get("cpsat_time_s", CPSAT_TIME_S),
                        cpsat_det_time=options.get("cpsat_det_time"))
        else:
            from packages.floorplan_core.core_placement import place_cores as _place
            _pl = _place(nodes, spec.get("demands", {}) or {}, mm, boundary,
                        spec.get("boundary_demands") or {}, spec.get("edge_types"),
                        spec.get("adjacencies") or [], spec.get("precedence") or [])
        nodes = _pl["nodes"]
        id_of = {str(n["id"]): n["id"] for n in nodes}
        placed_cores = {id_of[k]: v for k, v in _pl["cores"].items() if k in id_of}
        placement_report = _pl["report"]
        debug_steps.append({"step": "0_core_placement", "mode": _cp_mode,
                            "placed": len(placed_cores),
                            "unplaced": [k for k, v in placement_report.get("rooms", {}).items()
                                         if not v.get("placed")]})

    # ── 1. node-seeded grid ──────────────────────────────────────────────────
    tol_x = 0.04 * max(BW, 1e-9)
    tol_y = 0.04 * max(BH, 1e-9)
    # Boundary-vertex coordinates become pinned grid lines (notch-aligned), so on
    # an orthogonal boundary every cell is wholly inside or wholly outside it.
    _dem = spec.get("demands", {}) or {}
    _edges = [d.get("min_edge", 0.0) for d in _dem.values() if d.get("min_edge")]
    min_seg = (min(_edges) / mm) if (_edges and mm and mm > 0) else 0.0
    min_seg *= PIN_MIN_SEG_FACTOR
    # placed core edges are pinned like boundary vertices: the solve keeps them
    _cx = [v for c in placed_cores.values() for v in (c[0], c[2])]
    _cy = [v for c in placed_cores.values() for v in (c[1], c[3])]
    # options["source_grid"] (experiment 2026-11-05): the source's walls set the grid
    # and its rooms own the cells. Off by default; needs options["source_rooms"].
    src_polys = {}
    if options.get("source_grid"):
        for n in nodes:
            pts = (options.get("source_rooms") or {}).get(str(n["id"]))
            if pts and len(pts) >= 3:
                g_ = SPoly(pts)
                src_polys[n["id"]] = g_ if g_.is_valid else g_.buffer(0)
    _wx = [v for g_ in src_polys.values() for v in (g_.bounds[0], g_.bounds[2])] if src_polys else None
    _wy = [v for g_ in src_polys.values() for v in (g_.bounds[1], g_.bounds[3])] if src_polys else None
    xs, x_pins = _grid_lines([n["pos"][0] for n in nodes], bx0, bx1, tol_x,
                             fixed=[p[0] for p in boundary] + _cx, min_seg=min_seg,
                             walls=_wx)
    ys, y_pins = _grid_lines([n["pos"][1] for n in nodes], by0, by1, tol_y,
                             fixed=[p[1] for p in boundary] + _cy, min_seg=min_seg,
                             walls=_wy)
    ncol, nrow = len(xs) - 1, len(ys) - 1
    debug_steps.append({"step": "1_grid", "ncol": ncol, "nrow": nrow,
                        "nodes": len(nodes),
                        "pinned_x": len(x_pins) - 2, "pinned_y": len(y_pins) - 2})
    bpoly_g = SPoly(boundary)
    if not bpoly_g.is_valid:
        bpoly_g = bpoly_g.buffer(0)
    # a cell is usable when its centre lies in the boundary; outside cells get no
    # owner, so no room is sized with area the clip would remove
    inside = {(i, j) for i in range(ncol) for j in range(nrow)
              if bpoly_g.contains(SPoint((xs[i] + xs[i + 1]) / 2.0,
                                         (ys[j] + ys[j + 1]) / 2.0))}

    # ── 2. assign cells: seed each node's CONTAINING cell to it, then fill the
    # rest by nearest node. Seeding guarantees a node owns the cell it sits in,
    # so (with the node remap in step 3b) every node lands inside its own room.
    # Pure nearest-centre assignment does NOT guarantee this — a neighbour can be
    # closer to a cell's centre than the node sitting inside that cell.
    def _cell_index(v, lines):
        for k in range(len(lines) - 1):
            if lines[k] <= v <= lines[k + 1]:
                return k
        return 0 if v <= lines[0] else len(lines) - 2

    contain = {n["id"]: (_cell_index(n["pos"][0], xs), _cell_index(n["pos"][1], ys))
               for n in nodes}
    owner = {}
    # step 0's cores: every cell whose centre lies in a placed core is its room's.
    # core_cells records WHICH room's core each such cell belongs to, so 2b's
    # repair (below) can never transfer a cell step 0 already promised to a
    # room away from it — see core_cells' docstring note at the 2b call.
    core_cells = {}
    for rid, (cx0, cy0, cx1, cy1) in placed_cores.items():
        for i in range(ncol):
            if not cx0 < (xs[i] + xs[i + 1]) / 2.0 < cx1:
                continue
            for j in range(nrow):
                if cy0 < (ys[j] + ys[j + 1]) / 2.0 < cy1 and (i, j) in inside:
                    owner[(i, j)] = rid
                    core_cells[(i, j)] = rid
    if src_polys:
        for (i, j) in inside:
            if (i, j) in owner:
                continue
            pt_ = SPoint((xs[i] + xs[i + 1]) / 2.0, (ys[j] + ys[j + 1]) / 2.0)
            for nid_, g_ in src_polys.items():
                if g_.contains(pt_):
                    owner[(i, j)] = nid_
                    break
    node_conflicts = []          # nodes sharing one grid cell — only one can own it
    for n in nodes:
        cell = contain[n["id"]]
        if owner.get(cell) == n["id"]:
            continue
        if cell in owner:
            node_conflicts.append(n["id"])
        else:
            owner[cell] = n["id"]
    for i in range(ncol):
        ccx = (xs[i] + xs[i + 1]) / 2.0
        for j in range(nrow):
            if (i, j) in owner or (i, j) not in inside:
                continue
            ccy = (ys[j] + ys[j + 1]) / 2.0
            best, bd = None, float("inf")
            for n in nodes:
                # rooms with a placed core grow from the core, not the node point
                c = placed_cores.get(n["id"])
                if c is not None:
                    dx = max(c[0] - ccx, 0.0, ccx - c[2])
                    dy = max(c[1] - ccy, 0.0, ccy - c[3])
                    d = dx * dx + dy * dy
                elif n["id"] in src_polys:
                    d = src_polys[n["id"]].distance(SPoint(ccx, ccy)) ** 2
                else:
                    px, py = n["pos"]
                    d = (px - ccx) ** 2 + (py - ccy) ** 2
                if d < bd:
                    best, bd = n["id"], d
            owner[(i, j)] = best

    # ── 2b. boundary-contact repair (topology -> geometry): honour each room's
    # demanded room↔boundary fronts by transferring border cells, preserving
    # contiguity and every donor's own fronts. Demand keys are normalized to the
    # owner's room-id type. Unmet demands are reported, not forced. core_cells
    # (step 0's placed-core cells, keyed by the room they were placed for) is
    # threaded in so a transfer can never take a cell OUT of another room's
    # placed core — the step-0/2b seam gap found 2026-11-02 (test_boundary_003
    # rank 1, test_boundary_004 rank 2, cpsat mode): a cell inside r_5's placed
    # core was reassigned to r_4 to satisfy r_4's boundary front, because
    # `valid()` only checked the donor stayed connected and kept ITS OWN
    # fronts — not that step 0's min_edge promise to it survived. The grid
    # solve (step 3) then re-pinned r_5's core edges but the *interior* cell
    # was gone, so the clipped room no longer contained its node
    # (report.all_nodes_inside False, report.node_relocated=["r_5"]).
    #
    # Entrance (2026-11-04): the entry room claims the cell in front of the drawn
    # entrance, first, as a demand on a pseudo-edge. Nearest-node fill alone gives
    # that cell to whichever node is closest (005b: a hallway arm, not the primary
    # the entrance opens onto). The pseudo-edge stays local to this call.
    raw_bd = spec.get("boundary_demands") or {}
    bd = {}
    id_by_str = {str(r): r for r in set(owner.values())}
    entry_id = id_by_str.get(str(spec.get("entry_room")))
    edge_cell_sets = _edge_border_cells(boundary, xs, ys, ncol, nrow) if (raw_bd or entry_id is not None) else {}
    ent_cell = (_entrance_cell(boundary, spec.get("entrance"), xs, ys, ncol, nrow, edge_cell_sets)
                if entry_id is not None else None)
    bd = {id_by_str[k]: list(v) for k, v in raw_bd.items() if k in id_by_str}
    if bd or ent_cell is not None:
        repair_bd, repair_cells = {k: list(v) for k, v in bd.items()}, edge_cell_sets
        if ent_cell is not None:
            repair_bd[entry_id] = ["entrance"] + repair_bd.get(entry_id, [])
            repair_cells = {**edge_cell_sets, "entrance": {ent_cell}}
        owner, contacts_report = _repair_boundary_contacts(
            owner, repair_bd, repair_cells, ncol, nrow, core_cells=core_cells, first=entry_id)
    else:
        contacts_report = {"satisfied": [], "unmet": [], "transfers": []}
    entrance_met = ent_cell is not None and owner.get(ent_cell) == entry_id
    contacts_report["unmet"] = [u for u in contacts_report["unmet"] if u[1] != "entrance"]
    contacts_report["satisfied"] = [u for u in contacts_report["satisfied"] if u[1] != "entrance"]
    contacts_report["entrance"] = {"entry_room": str(spec.get("entry_room")), "met": entrance_met,
                                   "applied": ent_cell is not None}
    debug_steps.append({"step": "2b_contacts",
                        "satisfied": len(contacts_report["satisfied"]),
                        "unmet": len(contacts_report["unmet"]),
                        "transfers": len(contacts_report["transfers"])})

    # ── 2c. precedence core claim (SetPrecedence): spec["precedence"] is a list
    # of [winner, loser]; each winner is re-cut as a rectangle over its own and the
    # loser's cells, returning what it does not use. Empty by default (no claims).
    _dem2 = spec.get("demands", {}) or {}
    owner, precedence_report = _precedence_claims(
        owner, spec.get("precedence") or [], contain, xs, ys,
        lambda r: (((_dem2.get(r) or {}).get("min_edge", 0.0) / mm) if (mm and mm > 0) else 0.0),
        bd, edge_cell_sets if raw_bd else {}, choice=claim_choice,
        edge_types=spec.get("edge_types"))
    # a claimed rectangle that left out the winner's node cell takes the node with
    # it: centre of the rectangle (pre-solve), on a copy — never the caller's spec
    for e in precedence_report:
        if e.get("applied") and e.get("node_moved"):
            c0, c1, r0, r1 = e["rect"]
            nid = next((n["id"] for n in nodes if str(n["id"]) == e["winner"]), None)
            if nid is None:
                continue
            pos = [(xs[c0] + xs[c1]) / 2.0, (ys[r0] + ys[r1]) / 2.0]
            nodes = [dict(n, pos=pos) if n["id"] == nid else n for n in nodes]
            contain[nid] = (min(max(_cell_index(pos[0], xs), c0), c1 - 1),
                            min(max(_cell_index(pos[1], ys), r0), r1 - 1))
    # a claim moves cells across edges, so re-read every demand against the new
    # ownership (a returned corridor cell can satisfy the loser's unmet demand)
    if raw_bd and any(e.get("applied") for e in precedence_report):
        # a front a winner traded away is "dropped", not unmet — 4b must not
        # carve an arm back to it and undo the claim
        dropped_fronts = {(e["winner"], E) for e in precedence_report
                          if e.get("applied") for E in e.get("fronts_dropped") or []}
        contacts_report["satisfied"], contacts_report["unmet"] = [], []
        contacts_report["dropped_by_precedence"] = []
        for rid in sorted(bd, key=lambda r: (-len(bd[r]), str(r))):
            for E in bd[rid]:
                held = any(owner.get(c) == rid for c in (edge_cell_sets.get(E) or set()))
                key = ("satisfied" if held else
                       "dropped_by_precedence" if (str(rid), E) in dropped_fronts else "unmet")
                contacts_report[key].append([str(rid), E])
    debug_steps.append({"step": "2c_precedence",
                        "pairs": len(precedence_report),
                        "applied": sum(1 for e in precedence_report if e["applied"])})

    cells_by_room = {}
    for cell, rid in owner.items():
        cells_by_room.setdefault(rid, []).append(cell)
    dropped = [n["id"] for n in nodes if n["id"] not in cells_by_room]
    debug_steps.append({"step": "2_assign", "rooms_with_cells": len(cells_by_room),
                        "dropped": dropped})

    # ── 3. re-solve grid-line positions to fill the boundary (min-size solve) ─
    label_by_id = {n["id"]: n["label"] for n in nodes}
    demands = spec.get("demands", {})

    def _min_edge(rid):
        me = (demands.get(rid) or {}).get("min_edge", 0.0)
        if mm and mm > 0:                  # mm -> boundary units
            return me / mm
        return 0.0                         # no mm scale -> pure partition

    room_bbox = {}      # rid -> (cmin, cmax, rmin, rmax) grid-cell bbox, fixed
    min_x, min_y = {}, {}   # rid -> per-axis minimum (boundary units), lifted below
    for rid, cells in cells_by_room.items():
        cmin = min(c[0] for c in cells); cmax = max(c[0] for c in cells) + 1
        rmin = min(c[1] for c in cells); rmax = max(c[1] for c in cells) + 1
        room_bbox[rid] = (cmin, cmax, rmin, rmax)
        me = _min_edge(rid)
        min_x[rid] = me
        min_y[rid] = me
    # core block per room (options["core_block"], default on; the app's toggle): the
    # min_edge spans bind on this rectangle of the room's own cells, not its bbox,
    # so every room keeps a min_edge rectangle (D24 core). Off = bbox spans.
    use_core = options.get("core_block", True)
    core_box = {rid: _core_block(cells, contain.get(rid), xs, ys, _min_edge(rid))
                for rid, cells in cells_by_room.items()}
    span_box = {rid: (core_box[rid] if use_core and core_box[rid] else room_bbox[rid])
                for rid in cells_by_room}
    # Slack weights per column/row. When the spec carries area_frac (D1), weight
    # each column/row by the target AREA sitting in it: spread each room's target
    # fraction over its cells, then sum per column and per row. With min_edge ~ 0
    # the solve makes widths/heights ~proportional to this demand, so a room that
    # covers a large share of the source plan gets a proportionally large share of
    # the boundary. Exact per-room areas can't be hit on a shared grid (ncol+nrow
    # DOF < n rooms); this gets the proportions directionally right. Falls back to
    # the flexibility tiers when no area_frac is present.
    have_area = any((demands.get(rid) or {}).get("area_frac") for rid in cells_by_room)
    if have_area:
        cell_target = {}
        for rid, cells in cells_by_room.items():
            per = ((demands.get(rid) or {}).get("area_frac") or 0.0) / max(len(cells), 1)
            for c in cells:
                cell_target[c] = per
        col_flex = [max(sum(cell_target.get((i, j), 0.0) for j in range(nrow)), 1e-6)
                    for i in range(ncol)]
        row_flex = [max(sum(cell_target.get((i, j), 0.0) for i in range(ncol)), 1e-6)
                    for j in range(nrow)]
    else:
        col_flex = [max([_flex(label_by_id[owner[(i, j)]])
                         for j in range(nrow) if (i, j) in owner] or [FLEX_DEFAULT])
                    for i in range(ncol)]
        row_flex = [max([_flex(label_by_id[owner[(i, j)]])
                         for i in range(ncol) if (i, j) in owner] or [FLEX_DEFAULT])
                    for j in range(nrow)]
    if src_polys:   # slack follows the source's own column and row sizes
        col_flex = [max(xs[i + 1] - xs[i], 1e-6) for i in range(ncol)]
        row_flex = [max(ys[j + 1] - ys[j], 1e-6) for j in range(nrow)]
    # _longest_path_positions needs positive EPS gaps; it works in whatever units
    # the spans are in. We solve in boundary units with a [0, BW]/[0, BH] extent.
    eps_x = 0.02 * max(BW, 1e-9)
    eps_y = 0.02 * max(BH, 1e-9)

    # notch lines stay on the boundary; only node-derived cuts move. Factored so
    # the area-lift loop below can re-solve both axes from new per-room minimums.
    # per-axis span ranges the min constraints bind on (core block, or bbox)
    span_x = {r: (span_box[r][0], span_box[r][1]) for r in cells_by_room}
    span_y = {r: (span_box[r][2], span_box[r][3]) for r in cells_by_room}

    # source_grid "flex" keeps the slack-by-flex solve; True keeps source widths
    _solver = _solve_to_source if (src_polys and options.get("source_grid") is True) else _solve_positions

    def _solve_x(mx, sx):
        xt_, ok_ = _solver(ncol, [(sx[r][0], sx[r][1], mx[r]) for r in cells_by_room],
                           BW, col_flex, eps_x,
                           fixed={k: xs[k] - bx0 for k in x_pins})
        return [bx0 + v for v in xt_], ok_

    def _solve_y(my, sy):
        yt_, ok_ = _solver(nrow, [(sy[r][0], sy[r][1], my[r]) for r in cells_by_room],
                           BH, row_flex, eps_y,
                           fixed={k: ys[k] - by0 for k in y_pins})
        return [by0 + v for v in yt_], ok_

    def _solve_axes(mx, my):
        xt_, x_ok_ = _solve_x(mx, span_x)
        yt_, y_ok_ = _solve_y(my, span_y)
        return xt_, yt_, x_ok_, y_ok_

    # core relaxation: side-by-side cores can need more than the boundary's width.
    # Per axis, while infeasible, revert one room's core span to its bbox span —
    # the one whose revert alone makes the axis feasible, else the one with the
    # narrowest pre-solve core (furthest from min_edge) — so cores bind wherever
    # they fit and an infeasible solve never shrinks the whole grid for them.
    core_relaxed = {"x": [], "y": []}
    if use_core:
        for axis, solve, spans, lines, k0, k1 in (
                ("x", _solve_x, span_x, xs, 0, 1), ("y", _solve_y, span_y, ys, 2, 3)):
            mins = min_x if axis == "x" else min_y
            _, ok = solve(mins, spans)
            while not ok:
                cands = [r for r in cells_by_room
                         if spans[r] != (room_bbox[r][k0], room_bbox[r][k1])]
                if not cands:
                    break
                pick = None
                for r in sorted(cands, key=lambda r: lines[spans[r][1]] - lines[spans[r][0]]):
                    trial = dict(spans); trial[r] = (room_bbox[r][k0], room_bbox[r][k1])
                    if solve(mins, trial)[1]:
                        pick = r
                        break
                if pick is None:
                    pick = min(cands, key=lambda r: lines[spans[r][1]] - lines[spans[r][0]])
                spans[pick] = (room_bbox[pick][k0], room_bbox[pick][k1])
                core_relaxed[axis].append(pick)
                _, ok = solve(mins, spans)

    xt, yt, x_ok, y_ok = _solve_axes(min_x, min_y)

    # ── area lift: min_edge alone can satisfy both bbox edges and still miss
    # min_area (e.g. a room with width*height < min_area, each edge individually
    # fine). Only run when the baseline solve is already feasible — a lift must
    # never be asked to rescue an infeasible layout — and only when mm_per_unit is
    # known (min_area is in mm², demands[rid]["min_area"] otherwise meaningless).
    area_rounds = 0
    area_lifted, area_unmet = [], []
    lift_factor = {}
    # options["area_lift"] = False turns the lift off (the app's before/after toggle)
    if options.get("area_lift", True) and mm and mm > 0 and x_ok and y_ok:
        a_min = {rid: (((demands.get(rid) or {}).get("min_area", 0.0)) / (mm ** 2))
                for rid in cells_by_room}

        def _area(rid, xt_, yt_):
            return sum((xt_[i + 1] - xt_[i]) * (yt_[j + 1] - yt_[j])
                       for (i, j) in cells_by_room[rid])

        area_first = {rid: _area(rid, xt, yt) for rid in cells_by_room}
        unmet_set = set()
        for _ in range(3 * len(cells_by_room)):
            areas = {rid: _area(rid, xt, yt) for rid in cells_by_room}
            deficient = [rid for rid in cells_by_room
                        if a_min[rid] > 0 and rid not in unmet_set
                        and areas[rid] < a_min[rid] * (1 - 1e-6)]
            if not deficient:
                break
            area_rounds += 1
            # rooms that meet min_area now hold only min_edge as a hard constraint;
            # a lift must not take area they need (no starving a neighbour)
            met = [r for r in cells_by_room
                   if a_min[r] > 0 and areas[r] >= a_min[r] * (1 - 1e-6)]
            rid = max(deficient, key=lambda r: a_min[r] / areas[r])
            f = a_min[rid] / areas[rid]
            # the spans the solve binds (core block, or bbox with core_block off)
            (cmin, cmax), (rmin, rmax) = span_x[rid], span_y[rid]
            w, h = xt[cmax] - xt[cmin], yt[rmax] - yt[rmin]
            short, long_ = min(w, h), max(w, h)
            cap = max(ASPECT_MAX, (long_ / short) if short > 1e-12 else float("inf"))
            sq = math.sqrt(f)
            trials = [(max(min_x[rid], w * sq), max(min_y[rid], h * sq)),
                      (max(min_x[rid], w * f), min_y[rid]),
                      (min_x[rid], max(min_y[rid], h * f))]
            accepted = False
            for tx, ty in trials:
                trial_x = dict(min_x); trial_x[rid] = tx
                trial_y = dict(min_y); trial_y[rid] = ty
                txt, tyt, tok, tok_y = _solve_axes(trial_x, trial_y)
                if not (tok and tok_y):
                    continue
                nw, nh = txt[cmax] - txt[cmin], tyt[rmax] - tyt[rmin]
                nshort, nlong = min(nw, nh), max(nw, nh)
                if nshort > 1e-12 and (nlong / nshort) > cap + 1e-9:
                    continue
                if any(_area(r, txt, tyt) < a_min[r] * (1 - 1e-6) for r in met):
                    continue
                min_x, min_y = trial_x, trial_y
                xt, yt, x_ok, y_ok = txt, tyt, tok, tok_y
                accepted = True
                break
            if accepted:
                if rid not in area_lifted:
                    area_lifted.append(rid)
                base = area_first[rid]
                lift_factor[rid] = (_area(rid, xt, yt) / base) if base > 0 else 1.0
            else:
                unmet_set.add(rid)
                area_unmet.append(rid)

    debug_steps.append({"step": "3_resolve", "min_sizes_met": bool(x_ok and y_ok),
                        "area_rounds": area_rounds,
                        "area_lifted": [str(r) for r in area_lifted],
                        "area_unmet": [str(r) for r in area_unmet],
                        "core_relaxed": {a: [str(r) for r in v] for a, v in core_relaxed.items()}})
    # solved grid, for the canvas (drawn under the generated rooms): lines in
    # target space, which are pinned to the boundary, and each cell's owner after
    # the 2b repair (None = outside the boundary)
    grid_lines = {
        "xs": [round(v, 6) for v in xt], "ys": [round(v, 6) for v in yt],
        "pinned_x": sorted(x_pins), "pinned_y": sorted(y_pins),
        "cells": [[i, j, (str(owner[(i, j)]) if (i, j) in owner else None)]
                  for i in range(ncol) for j in range(nrow)],
        # per-room core block on the solved grid, for the canvas: rect in target
        # space, size in mm against min_edge; "intact" is added after step 4b
        "core_block_on": bool(use_core),
        "cores": {},
    }
    for rid, cb in core_box.items():
        if not cb:
            continue
        c0, c1, r0, r1 = cb
        w_u, h_u = xt[c1] - xt[c0], yt[r1] - yt[r0]
        me_mm = (demands.get(rid) or {}).get("min_edge", 0.0)
        grid_lines["cores"][str(rid)] = {
            "cells": [c0, c1, r0, r1],
            "rect": [round(xt[c0], 6), round(yt[r0], 6), round(xt[c1], 6), round(yt[r1], 6)],
            "w_mm": round(w_u * mm, 1) if mm else None,
            "h_mm": round(h_u * mm, 1) if mm else None,
            "min_edge_mm": me_mm,
            "meets": bool(mm and me_mm and min(w_u, h_u) * mm >= me_mm - 1.0),
            "contains_node": bool(contain.get(rid) and c0 <= contain[rid][0] < c1
                                  and r0 <= contain[rid][1] < r1),
            # axes on which the core span was reverted to the bbox to stay feasible
            "relaxed": [a for a in ("x", "y") if rid in core_relaxed[a]],
        }

    # ── 3b. remap node positions through the same warp the grid underwent, so a
    # node keeps its fractional spot inside its (re-dimensioned) cell — i.e. stays
    # inside its own room. Without this, re-solving the separators to fill the
    # boundary would leave the fixed node outside its moved footprint.
    def _warp(v, old, new):
        k = _cell_index(v, old)
        lo, hi = old[k], old[k + 1]
        f = (v - lo) / (hi - lo) if hi > lo else 0.0
        return new[k] + f * (new[k + 1] - new[k])

    node_xy = {n["id"]: [_warp(n["pos"][0], xs, xt), _warp(n["pos"][1], ys, yt)]
               for n in nodes}

    # ── 4. build room polygons, clipped to the boundary ──────────────────────
    bpoly = SPoly(boundary)
    if not bpoly.is_valid:
        bpoly = bpoly.buffer(0)
    clip_needed = bpoly.area < BW * BH - (1e-6 * BW * BH)

    out_rooms, room_report, clipped_out, splits, node_relocated = [], [], [], 0, []
    covered = 0.0
    for n in nodes:
        rid = n["id"]
        cells = cells_by_room.get(rid)
        if not cells:
            continue
        rects = [sbox(xt[i], yt[j], xt[i + 1], yt[j + 1]) for (i, j) in cells]
        merged = unary_union(rects)
        geom = merged.intersection(bpoly) if clip_needed else merged
        if geom.is_empty or geom.area < (1e-9 * BW * BH):
            clipped_out.append(rid)
            continue
        if geom.geom_type == "MultiPolygon":
            splits += 1
        poly = _largest_ring(geom)
        covered += geom.area
        label = label_by_id[rid]
        area_units = geom.area
        # node-in-room is the room's LABEL and must hold for the display; room
        # SIZE/placement is the quality signal (measured here + by Stage-1 sensors).
        # If the warped node falls outside its clipped room (happens when a node-
        # grid column crosses a boundary notch), relocate the MARKER to a
        # guaranteed-interior point and flag it — node_relocated is the honest
        # count of rooms that did not naturally contain their node.
        rpoly = SPoly(poly)
        natural_inside = (rpoly.buffer(1e-9).contains(SPoint(*node_xy[rid]))
                          if len(poly) >= 3 else False)
        if not natural_inside and len(poly) >= 3:
            rep = rpoly.representative_point()
            node_xy[rid] = [rep.x, rep.y]
            node_relocated.append(rid)
        node_inside = natural_inside
        out_rooms.append({"id": rid, "type": label, "polygon": poly,
                          "_area": geom.area})
        target_frac = (demands.get(rid) or {}).get("area_frac")
        actual_frac = area_units / (bpoly.area or 1.0)
        # bounding-box dimensions (short = the min_room_size edge metric) so the UI
        # can show width × length and tell an edge violation from an area one.
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        bw = (max(xs) - min(xs)) if xs else 0.0
        bh = (max(ys) - min(ys)) if ys else 0.0
        dem = demands.get(rid) or {}
        room_report.append({
            "id": rid, "label": label,
            "area": round(area_units, 6),
            "area_frac": round(actual_frac, 3),
            "target_frac": (round(target_frac, 3) if target_frac is not None else None),
            "node_inside": bool(node_inside),
            "short": round(min(bw, bh), 6),
            "long": round(max(bw, bh), 6),
            "min_edge": dem.get("min_edge", 0.0),   # mm
            "min_area": dem.get("min_area", 0.0),    # mm²
            # area provenance (boundary units²): after the step-3 solve, before the
            # boundary clip; after the clip. With the final `area` this tells a
            # grid-solve shortfall from a clip loss from an arm-carve loss.
            "area_rect": round(merged.area, 6),
            "area_clip": round(area_units, 6),
            "arm_donor_to": [],
            "area_lift": round(lift_factor.get(rid, 1.0), 4),
        })
    # ── 4b. thin-arm carve: for a boundary demand the rectangular partition could
    # not satisfy, let the room annex a thin corridor to reach its demanded edge,
    # becoming L-shaped (the source's thin reach). Donors stay whole and keep their
    # own fronts; demands still unsatisfiable remain in report.boundary_contacts.
    if contacts_report.get("unmet") and raw_bd:
        poly_by_id = {r["id"]: SPoly(r["polygon"]) for r in out_rooms
                      if len(r["polygon"]) >= 3}
        arm_sat, arm_donors = _carve_unmet_arms(poly_by_id, boundary, bpoly, bd,
                                                contacts_report["unmet"],
                                                spec.get("edge_types"))
        if arm_sat:
            rep_by_id = {rr["id"]: rr for rr in room_report}
            for d, recips in arm_donors.items():
                if d in rep_by_id:
                    rep_by_id[d]["arm_donor_to"] = recips
            for r in out_rooms:
                g = poly_by_id.get(r["id"])
                if g is None or g.is_empty:
                    continue
                ring = _largest_ring(g)
                r["polygon"] = ring
                r["_area"] = g.area
                rr = rep_by_id.get(r["id"])
                if rr:
                    rxs = [p[0] for p in ring]
                    rys = [p[1] for p in ring]
                    rr["area"] = round(g.area, 6)
                    rr["area_frac"] = round(g.area / (bpoly.area or 1.0), 3)
                    rr["short"] = round(min(max(rxs) - min(rxs), max(rys) - min(rys)), 6)
                    rr["long"] = round(max(max(rxs) - min(rxs), max(rys) - min(rys)), 6)
            for it in arm_sat:
                contacts_report["unmet"].remove(it)
                contacts_report["satisfied"].append(it)
            contacts_report.setdefault("arms", []).extend(
                [[str(a[0]), a[1]] for a in arm_sat])

    out_rooms.sort(key=lambda r: r.pop("_area"), reverse=True)

    # area-ratio fidelity: mean |actual - target| over rooms that carry a target.
    _af = [(rr["area_frac"], rr["target_frac"]) for rr in room_report
           if rr["target_frac"] is not None]
    area_mae = round(sum(abs(a - t) for a, t in _af) / len(_af), 4) if _af else None

    # node markers for display (after any relocation) — always inside their room.
    node_pos_out = {str(n["id"]): [round(node_xy[n["id"]][0], 6), round(node_xy[n["id"]][1], 6)]
                    for n in nodes}

    realised_adj = _realised_adjacency(owner, ncol, nrow)
    # Reconcile the carried room↔room edges against the geometry we just realised:
    # connections (door/open) follow room identity where still adjacent, walls are
    # re-derived, stranded connections are reported. Without this the solver echoes
    # the source plan's edges unchanged, so a moved room keeps stale adjacencies the
    # new layout does not have (and the sensors below, which read plan["edges"],
    # would score against them).
    recon_edges, recon_report = reconcile_edges(realised_adj, spec.get("adjacencies"))
    coverage = covered / (bpoly.area or 1.0)
    all_nodes_inside = all(rr["node_inside"] for rr in room_report)
    # Stage-0 feasibility = a COHERENT partition: every room placed, every node
    # inside its room, boundary followed. Min room size is a Stage-1 sensor, not
    # a Stage-0 gate, so it is reported (min_sizes_met) but not gated here.
    feasible = (not dropped and not clipped_out and splits == 0
                and all_nodes_inside and not node_relocated)
    # core block intact = the final room (after clip + 4b carve) still covers it
    final_by_id = {str(r["id"]): r["polygon"] for r in out_rooms
                   if len(r.get("polygon") or []) >= 3}
    for rid, cb in grid_lines["cores"].items():
        poly = final_by_id.get(rid)
        if poly is None:
            cb["intact"] = False
            continue
        g = SPoly(poly)
        if not g.is_valid:
            g = g.buffer(0)
        x0c, y0c, x1c, y1c = cb["rect"]
        rect = SPoly([(x0c, y0c), (x1c, y0c), (x1c, y1c), (x0c, y1c)])
        cb["intact"] = bool(rect.area > 0
                            and g.intersection(rect).area >= rect.area * (1 - 1e-3))
    debug_steps.append({"step": "4_polygons", "rooms": len(out_rooms),
                        "coverage": round(coverage, 4), "splits": splits,
                        "clipped_out": clipped_out,
                        "all_nodes_inside": all_nodes_inside})

    return {
        "rooms": out_rooms,
        "edges": recon_edges,
        "entrance": spec.get("entrance"),
        "node_pos": node_pos_out,
        "feasible": feasible,
        "report": {
            "grid": [ncol, nrow],
            "grid_lines": grid_lines,
            "precedence": precedence_report,
            "placement": placement_report,
            "coverage": round(coverage, 4),
            "min_sizes_met": bool(x_ok and y_ok),
            "dropped_rooms": dropped,
            "clipped_out_rooms": clipped_out,
            "clip_splits": splits,
            "all_nodes_inside": all_nodes_inside,
            "area_mae": area_mae,
            "node_relocated": node_relocated,
            "node_conflicts": node_conflicts,
            "realised_adjacencies": len(realised_adj),
            "edge_reconciliation": recon_report,
            "boundary_contacts": contacts_report,
            "rooms": room_report,
            "debug_steps": debug_steps,
        },
    }


if __name__ == "__main__":
    # Self-check: 4 nodes in the quadrants of a unit square must tile it into 4
    # coherent rooms, each containing its node and covering the whole boundary;
    # a reassignment (swap two nodes) must re-derive the geometry through the seam.
    from shapely.geometry import Point as _P

    _boundary = [[0, 0], [1, 0], [1, 1], [0, 1]]
    _spec = {
        "boundary": _boundary,
        "nodes": [
            {"id": "bed", "room_id": "bed", "kind": "whole", "label": "bedroom", "pos": [0.25, 0.25]},
            {"id": "bath", "room_id": "bath", "kind": "whole", "label": "bathroom", "pos": [0.75, 0.25]},
            {"id": "kit", "room_id": "kit", "kind": "whole", "label": "kitchen", "pos": [0.25, 0.75]},
            {"id": "liv", "room_id": "liv", "kind": "whole", "label": "living", "pos": [0.75, 0.75]},
        ],
        "adjacencies": [], "demands": {}, "edge_types": None,
        "entrance": None, "mm_per_unit": None,
    }
    plan = generate(_spec)
    assert plan["feasible"], plan["report"]
    assert plan["report"]["coverage"] > 0.999, plan["report"]["coverage"]
    assert plan["report"]["all_nodes_inside"], plan["report"]
    assert len(plan["rooms"]) == 4, plan["rooms"]
    # every node sits inside its own room
    poly_by_id = {r["id"]: SPoly(r["polygon"]) for r in plan["rooms"]}
    for n in _spec["nodes"]:
        assert poly_by_id[n["id"]].buffer(1e-9).contains(_P(*n["pos"])), n

    # reassignment re-derives: swap bed <-> liv, the bedroom's polygon must move
    # to where living's was (top-right) — geometry follows the spec edit.
    bed_before = SPoly(poly_by_id["bed"]).centroid
    spec2 = reassign(_spec, "bed", "liv")
    plan2 = generate(spec2)
    poly2 = {r["id"]: SPoly(r["polygon"]) for r in plan2["rooms"]}
    bed_after = poly2["bed"].centroid
    moved = (bed_after.x - bed_before.x) ** 2 + (bed_after.y - bed_before.y) ** 2
    assert moved > 0.1, f"reassignment did not move the room: {moved}"
    assert plan2["report"]["all_nodes_inside"], plan2["report"]

    # non-rectangular boundary (L-shape): still a valid clipped partition
    _L = [[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]]
    specL = dict(_spec); specL["boundary"] = _L
    planL = generate(specL)
    assert planL["rooms"], planL["report"]
    assert planL["report"]["coverage"] > 0.98, planL["report"]["coverage"]

    print("generate.py Stage-0 self-check OK")
    print("  square:", plan["report"]["grid"], "coverage", plan["report"]["coverage"],
          "nodes_inside", plan["report"]["all_nodes_inside"])
    print("  reassign moved bed by", round(moved, 3))
    print("  L-shape coverage", planL["report"]["coverage"])
