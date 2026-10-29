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
     rooms (reused verbatim from rectangular_arrange);
  4. build each room's polygon from its cells and clip to the boundary polygon, so
     a non-rectangular boundary is followed rather than overflowed.

Stage 0 is geometry only: the report carries coverage / min-size / dropped /
clipped signals, NOT daylight or access — those are Stage-1 sensors
(topo_sensors) that read this Plan.

Public API:
    spec_from_plan(plan, boundary, edge_types=None, entrance=None,
                   mm_per_unit=None) -> Spec
    generate(spec) -> Plan
    reassign(spec, room_a, room_b) -> Spec        # swap two rooms' nodes (seam demo)
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

# Minimum room edges (mm) live in the fitting-prototype schema; rectangular_arrange
# puts that dir on sys.path at import time, so this import resolves after it.
from schema import MINIMUM_SIZES, SCALE  # noqa: E402

SCALE_M = SCALE / 1000.0   # normalized boundary unit -> metres (same as the raycast)
# BADS minimum opening widths the raycast tests a contact against, by edge type.
_EDGE_FLOOR_M = {"facing_corridor": 0.9, "external": 1.2}


# ── Spec / Plan ──────────────────────────────────────────────────────────────
# Both are plain dicts to stay continuous with the app's dict-centric plan format
# (apps/web <-> apps/api send dicts). The Spec is the ONLY thing the agent writes.
#
# Spec = {
#   "boundary":        [[x, y], ...],              target boundary polygon
#   "node_assignment": [{"id", "label", "pos":[x,y]}],  room -> node (editable)
#   "adjacencies":     [{"u", "v", ...}],          room<->room graph (fixed)
#   "demands":         {room_id: {"min_edge": mm, "min_area": mm2}},  per-room
#   "edge_types":      [str] | None,               boundary typed edges (context)
#   "entrance":        [x0,y0,x1,y1] | None,
#   "mm_per_unit":     float | None,               boundary-units -> mm (min-size scale)
#   "boundary_demands":{room_id: [target_edge_index]} | None,  room↔boundary fronts
#                       the realiser must honour (topology -> geometry, step 2b)
# }
# Plan = {"rooms":[{id,type,polygon}], "edges", "entrance", "node_pos",
#         "feasible", "report"}


def _poly_area(poly):
    """Shoelace area of a simple polygon [[x, y], ...] (sign-agnostic)."""
    n = len(poly)
    return abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
                   for i in range(n))) / 2.0


def spec_from_plan(plan, boundary, edge_types=None, entrance=None,
                   mm_per_unit=None):
    """Extract an editable Spec from a retrieved + topo-ranked plan.

    node positions come from the topo layer (`node_pos`, placed in the target
    frame by topo_signature); when absent, each room's node falls back to its
    polygon centroid. Demands come from MINIMUM_SIZES by label, plus `area_frac`
    — the room's share of the source plan's total room area (a dimensional demand,
    D1): the solver sizes the generated room toward the same proportion.
    """
    node_pos = plan.get("node_pos") or (plan.get("retrieval") or {}).get("node_pos") or {}
    nodes, demands = [], {}
    for r in plan.get("rooms") or []:
        rid = r.get("id")
        poly = r.get("polygon")
        if not poly or len(poly) < 3:
            continue
        label = _label(r)
        pos = node_pos.get(str(rid))
        if pos is None:
            pos = [sum(p[0] for p in poly) / len(poly),
                   sum(p[1] for p in poly) / len(poly)]
        nodes.append({"id": rid, "label": label, "pos": [float(pos[0]), float(pos[1])]})
        ms = MINIMUM_SIZES.get(label, {})
        demands[rid] = {"min_edge": ms.get("min_edge", 0.0),
                        "min_area": ms.get("area", 0.0),
                        "_src_area": _poly_area(poly)}
    # area_frac: each room's share of the source plan's total room area.
    total_area = sum(d["_src_area"] for d in demands.values()) or 1.0
    for d in demands.values():
        d["area_frac"] = d.pop("_src_area") / total_area
    return {
        "boundary": [[float(x), float(y)] for x, y in boundary],
        "node_assignment": nodes,
        "adjacencies": plan.get("edges") or [],
        "demands": demands,
        "edge_types": edge_types,
        "entrance": entrance if entrance is not None else plan.get("entrance"),
        "mm_per_unit": mm_per_unit,
    }


def reassign(spec, room_a, room_b):
    """Return a new Spec with rooms a and b swapped onto each other's node
    positions — the minimal spec edit that re-derives geometry through the seam
    (the Stage-0 preview of Stage-2's SwapNodes move). Pure; does not mutate."""
    new_nodes = []
    pos_by_id = {n["id"]: n["pos"] for n in spec["node_assignment"]}
    for n in spec["node_assignment"]:
        m = dict(n)
        if n["id"] == room_a:
            m["pos"] = list(pos_by_id[room_b])
        elif n["id"] == room_b:
            m["pos"] = list(pos_by_id[room_a])
        new_nodes.append(m)
    out = dict(spec)
    out["node_assignment"] = new_nodes
    return out


# ── the solver ───────────────────────────────────────────────────────────────

def _grid_lines(coords, lo, hi, tol):
    """Separator lines along one axis from a bag of node coordinates: cluster the
    coords into bands, cut at the midpoints between consecutive bands, and bracket
    with the boundary extent [lo, hi]. n bands -> n columns."""
    bands = _cluster([c for c in coords if lo - tol <= c <= hi + tol], tol)
    if not bands:
        return [lo, hi]
    cuts = [(bands[k] + bands[k + 1]) / 2.0 for k in range(len(bands) - 1)]
    return [lo] + cuts + [hi]


def _solve_positions(n_sep, spans, total, gap_flex, eps):
    """Separator positions along one axis by longest path (the arrange algorithm,
    made unit-relative via `eps` so it works in normalized OR mm space).

    `spans` are (lo, hi, min_size): position[hi] - position[lo] >= min_size for a
    room occupying grid columns [lo, hi). Returns (positions in [0, total],
    all_min_met); every gap is at least `eps`. Slack over the minimal feasible
    layout is handed to the gaps a flexible room occupies."""
    m = n_sep
    pos = [0.0] * (m + 1)
    spans_by_hi = {}
    for lo, hi, mn in spans:
        spans_by_hi.setdefault(hi, []).append((lo, mn))
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


def _bfs_cell_path(starts, goals, ncol, nrow):
    """Shortest 4-connected grid path from any start cell to any goal cell
    (fewest cells = fewest transfers). Returns the path incl. endpoints, or None."""
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
            if 0 <= nb[0] < ncol and 0 <= nb[1] < nrow and nb not in prev:
                prev[nb] = c
                q.append(nb)
    return None


def _repair_boundary_contacts(owner, boundary_demands, edge_cell_sets, ncol, nrow):
    """Make each room front its demanded target edges by transferring border cells
    (topology -> geometry). A transfer is committed only if it keeps every donor
    non-empty and 4-connected AND keeps each donor's own fronts; rooms process
    most-constrained-first for determinism. Returns (owner, report); report lists
    satisfied / unmet demands and the cells moved. Demands it cannot satisfy are
    reported, not forced — the guarantee is conditional on a feasible transfer."""
    report = {"satisfied": [], "unmet": [], "transfers": []}
    if not boundary_demands:
        return owner, report

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

    order = sorted(boundary_demands, key=lambda r: (-len(boundary_demands[r]), str(r)))
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
                p = _bfs_cell_path(starts, {g}, ncol, nrow) if starts else None
                if p:
                    cands.append(p)
            cands.sort(key=len)
            committed = False
            for path in cands:
                moved = [c for c in path if owner.get(c) != rid]
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


def generate(spec):
    """Realise a Plan from a Spec: room polygons from node positions + boundary.

    See the module docstring for the four steps. Returns the Plan dict.
    """
    boundary = spec["boundary"]
    nodes = [n for n in spec.get("node_assignment", []) if n.get("pos")]
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

    # ── 1. node-seeded grid ──────────────────────────────────────────────────
    tol_x = 0.04 * max(BW, 1e-9)
    tol_y = 0.04 * max(BH, 1e-9)
    xs = _grid_lines([n["pos"][0] for n in nodes], bx0, bx1, tol_x)
    ys = _grid_lines([n["pos"][1] for n in nodes], by0, by1, tol_y)
    ncol, nrow = len(xs) - 1, len(ys) - 1
    debug_steps.append({"step": "1_grid", "ncol": ncol, "nrow": nrow,
                        "nodes": len(nodes)})

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
    node_conflicts = []          # nodes sharing one grid cell — only one can own it
    for n in nodes:
        cell = contain[n["id"]]
        if cell in owner:
            node_conflicts.append(n["id"])
        else:
            owner[cell] = n["id"]
    for i in range(ncol):
        ccx = (xs[i] + xs[i + 1]) / 2.0
        for j in range(nrow):
            if (i, j) in owner:
                continue
            ccy = (ys[j] + ys[j + 1]) / 2.0
            best, bd = None, float("inf")
            for n in nodes:
                px, py = n["pos"]
                d = (px - ccx) ** 2 + (py - ccy) ** 2
                if d < bd:
                    best, bd = n["id"], d
            owner[(i, j)] = best

    # ── 2b. boundary-contact repair (topology -> geometry): honour each room's
    # demanded room↔boundary fronts by transferring border cells, preserving
    # contiguity and every donor's own fronts. Demand keys are normalized to the
    # owner's room-id type. Unmet demands are reported, not forced.
    raw_bd = spec.get("boundary_demands") or {}
    bd = {}
    if raw_bd:
        id_by_str = {str(r): r for r in set(owner.values())}
        bd = {id_by_str[k]: list(v) for k, v in raw_bd.items() if k in id_by_str}
        edge_cell_sets = _edge_border_cells(boundary, xs, ys, ncol, nrow)
        owner, contacts_report = _repair_boundary_contacts(
            owner, bd, edge_cell_sets, ncol, nrow)
    else:
        contacts_report = {"satisfied": [], "unmet": [], "transfers": []}
    debug_steps.append({"step": "2b_contacts",
                        "satisfied": len(contacts_report["satisfied"]),
                        "unmet": len(contacts_report["unmet"]),
                        "transfers": len(contacts_report["transfers"])})

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

    x_spans, y_spans = [], []
    for rid, cells in cells_by_room.items():
        cmin = min(c[0] for c in cells); cmax = max(c[0] for c in cells) + 1
        rmin = min(c[1] for c in cells); rmax = max(c[1] for c in cells) + 1
        me = _min_edge(rid)
        x_spans.append((cmin, cmax, me))
        y_spans.append((rmin, rmax, me))
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
    # _longest_path_positions needs positive EPS gaps; it works in whatever units
    # the spans are in. We solve in boundary units with a [0, BW]/[0, BH] extent.
    eps_x = 0.02 * max(BW, 1e-9)
    eps_y = 0.02 * max(BH, 1e-9)
    xt, x_ok = _solve_positions(ncol, x_spans, BW, col_flex, eps_x)
    yt, y_ok = _solve_positions(nrow, y_spans, BH, row_flex, eps_y)
    xt = [bx0 + v for v in xt]
    yt = [by0 + v for v in yt]
    debug_steps.append({"step": "3_resolve", "min_sizes_met": bool(x_ok and y_ok)})

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
        "node_assignment": [
            {"id": "bed",  "label": "bedroom", "pos": [0.25, 0.25]},
            {"id": "bath", "label": "bathroom", "pos": [0.75, 0.25]},
            {"id": "kit",  "label": "kitchen", "pos": [0.25, 0.75]},
            {"id": "liv",  "label": "living", "pos": [0.75, 0.75]},
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
    for n in _spec["node_assignment"]:
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
