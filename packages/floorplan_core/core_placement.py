"""
core_placement.py — Stage-0 step 0 (SPIKE, 2026-11-02): place a core block per
room before the grid is cut, and move each node into its block.

The node-driven solver treats node positions as fixed: every room inherits
wherever its node landed, and D24 core failures (no min_edge square fits) are
repaired after the fact. Here each room first gets a core block sized by use (the
min_edge square of its type — D24's core test), placed inside the boundary one
room at a time; the node moves into the placed core, and generate() cuts the grid
on the core edges and seeds the core cells to their room. Rooms grow from there
(nearest-core fill, the step-3 solve). Off by default: spec["core_placement"].

Room<->room topology (2026-11-02) is a constraint on both the placement ORDER
and each room's candidate SCORE:
  order     a graph walk, not a flat size sort: starts from the current
            precedence/boundary-demand/size key's first room, then repeatedly
            adds the unplaced room with the most door/open edges to rooms
            already in order (ties broken by that same key); rooms with no
            door/open edge to anything go last. A bathroom or dining room with
            a door to an already-placed room is seated while that wall is
            still free, instead of last by size.
  hard      a candidate is rejected when an already-placed door/open neighbour
            (`connection` 1/2 or `separation` "door"/"open") would share less
            than DOOR_MIN_M of core wall with it. When every candidate at a
            room violates at least one such neighbour, the candidate with the
            fewest violations wins (then lowest cost), and the still-violated
            pairs are recorded in report["rooms"][r]["relaxed"] rather than
            silently dropped.
  soft      the W_ADJ adjacency term is the SHORTFALL of shared wall below a
            required floor (DOOR_MIN_M for a door/open neighbour, else
            EDGE_FLOOR_M["shared_wall"]) — not gap distance. A corner touch
            (shared wall 0) scores the full shortfall, the same as no contact;
            it is not adjacency.

Each candidate position is also scored on:
  node      distance from the source node to the block (keep the retrieved layout)
  fronts    each demanded boundary edge the block does not front (+ its distance)
  order     left/right, above/below flips against the source node arrangement
  sliver    a gap narrower than SLIVER_M to a boundary edge or placed core — the
            thin strip no room can use (the narrow passages the cell fill makes)
Lowest cost wins, after violations. A room with no feasible position keeps its
node and has no core (reported as unplaced). Pure: returns new node dicts, never
mutates the input.

Public API:
    place_cores(nodes, demands, mm, boundary, boundary_demands, edge_types,
                adjacencies, precedence=None) -> {cores, nodes, report}
"""
import math

from shapely.geometry import Polygon as SPoly, LineString, box as sbox
from shapely.prepared import prep

SLIVER_M = 0.9          # a gap narrower than this between walls is unusable floor
LATTICE_M = 0.3         # candidate-position lattice step
EDGE_FLOOR_M = {"facing_corridor": 0.9, "external": 1.2, "shared_wall": 0.3}
ORDER_TOL_M = 0.5       # source offsets below this do not fix an order
DOOR_MIN_M = 0.9        # a shared core wall narrower than this cannot carry the
                        # door/open connection the source plan demands — same
                        # value as topo_signature's DOOR_MIN_M (defined locally
                        # there too; it carries no module-level export)

W_NODE, W_FRONT, W_FRONT_DIST, W_ADJ, W_ORDER, W_SLIVER = 1.0, 10.0, 2.0, 0.5, 3.0, 5.0


def _rect_point_dist(r, p):
    x0, y0, x1, y1 = r
    dx = max(x0 - p[0], 0.0, p[0] - x1)
    dy = max(y0 - p[1], 0.0, p[1] - y1)
    return math.hypot(dx, dy)


def _overlap(a, b, tol):
    return a[0] < b[2] - tol and b[0] < a[2] - tol and a[1] < b[3] - tol and b[1] < a[3] - tol


def _is_door_or_open(e):
    """Is this room<->room edge a door or open connection, not a wall? Mirrors
    topo_sensors._passable: `separation` is authoritative when present, else
    `connection` 1 (door) or 2 (open)."""
    sep = e.get("separation")
    if sep in ("door", "open"):
        return True
    if sep == "wall":
        return False
    return e.get("connection") in (1, 2)


def place_cores(nodes, demands, mm, boundary, boundary_demands, edge_types,
                adjacencies, precedence=None):
    if not mm or mm <= 0:
        return {"cores": {}, "nodes": nodes, "report": {"skipped": "no mm scale"}}
    m = mm / 1000.0                      # metres per boundary unit
    u = lambda metres: metres / m        # noqa: E731  metres -> boundary units
    bpoly = SPoly(boundary)
    if not bpoly.is_valid:
        bpoly = bpoly.buffer(0)
    bprep = prep(bpoly.buffer(u(0.001)))
    bx0, by0, bx1, by1 = bpoly.bounds
    nb = len(boundary)
    elines = [LineString([boundary[k], boundary[(k + 1) % nb]]) for k in range(nb)]
    etype = lambda E: (edge_types[E] if (edge_types and 0 <= E < len(edge_types))  # noqa: E731
                       else "external")
    # axis-aligned boundary edges as sliver obstacles: (x, ylo, yhi) / (y, xlo, xhi)
    vert, horz = [], []
    for k in range(nb):
        (ax, ay), (bx, by) = boundary[k], boundary[(k + 1) % nb]
        if abs(ax - bx) < 1e-9:
            vert.append((ax, min(ay, by), max(ay, by)))
        elif abs(ay - by) < 1e-9:
            horz.append((ay, min(ax, bx), max(ax, bx)))
    vx = sorted({v[0] for v in vert}); hy = sorted({h[0] for h in horz})

    bd = {str(k): list(v) for k, v in (boundary_demands or {}).items()}
    adj = {}
    for e in adjacencies or []:
        a, b = str(e.get("u")), str(e.get("v"))
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    src = {str(n["id"]): tuple(n["pos"]) for n in nodes}

    # core per room: the min_edge square D24's core test asks for (boundary units).
    # Area is left to growth and the step-3 lift — a min_area-sized core
    # (min_edge x min_area/min_edge) packed ~80% of 001b and starved later rooms.
    # Rooms without a minimum get none.
    dims = {}
    for n in nodes:
        me = (demands.get(n["id"]) or {}).get("min_edge", 0.0)
        if me > 0:
            dims[str(n["id"])] = (me / mm, me / mm)

    winners = [str(p[0]) for p in precedence or []]
    rank_w = {r: i for i, r in enumerate(winners)}
    key = lambda r: (rank_w.get(r, len(winners)), -len(bd.get(r, [])),  # noqa: E731
                     -dims[r][0] * dims[r][1], r)

    door_adj = {}
    for e in adjacencies or []:
        if not _is_door_or_open(e):
            continue
        a, b = str(e.get("u")), str(e.get("v"))
        if a in dims and b in dims:
            door_adj.setdefault(a, set()).add(b)
            door_adj.setdefault(b, set()).add(a)

    # Order is a graph walk, not a flat sort: start from the current key's
    # first room, then repeatedly add the unplaced room with the most door/
    # open edges to rooms already in order (ties broken by the key). Rooms
    # with no door/open edge to anything go last, in key order.
    base = sorted(dims, key=key)
    isolated = [r for r in base if not door_adj.get(r)]
    connected = set(r for r in dims if door_adj.get(r))
    order = []
    if connected:
        start = next(r for r in base if r in connected)
        order.append(start)
        connected.discard(start)
    while connected:
        nxt, nxt_n = None, -1
        for r in sorted(connected, key=key):
            n = len(door_adj.get(r, ()) & set(order))
            if n > nxt_n:
                nxt, nxt_n = r, n
        order.append(nxt)
        connected.discard(nxt)
    order += isolated

    placed, report = {}, {"order": order, "rooms": {}}
    step = u(LATTICE_M)
    tol = u(0.001)

    def _shared_wall(a, b):
        """Length of wall two placed core rects share, edge to edge — zero
        unless one rect's face lies flush on the other's (a corner touch, where
        only a point coincides, is not a shared wall)."""
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        if abs(ax1 - bx0) < tol or abs(bx1 - ax0) < tol:
            lo, hi = max(ay0, by0), min(ay1, by1)
            return max(hi - lo, 0.0) * m
        if abs(ay1 - by0) < tol or abs(by1 - ay0) < tol:
            lo, hi = max(ax0, bx0), min(ax1, bx1)
            return max(hi - lo, 0.0) * m
        return 0.0

    def sliver_cost(r):
        x0, y0, x1, y1 = r
        cost = 0.0
        sides = []
        # nearest parallel obstacle on each side, within the side's span
        lefts = [x0 - x for x, lo, hi in vert if x <= x0 + tol and lo < y1 - tol and hi > y0 + tol]
        lefts += [x0 - c[2] for c in placed.values() if c[2] <= x0 + tol and c[1] < y1 - tol and c[3] > y0 + tol]
        rights = [x - x1 for x, lo, hi in vert if x >= x1 - tol and lo < y1 - tol and hi > y0 + tol]
        rights += [c[0] - x1 for c in placed.values() if c[0] >= x1 - tol and c[1] < y1 - tol and c[3] > y0 + tol]
        downs = [y0 - y for y, lo, hi in horz if y <= y0 + tol and lo < x1 - tol and hi > x0 + tol]
        downs += [y0 - c[3] for c in placed.values() if c[3] <= y0 + tol and c[0] < x1 - tol and c[2] > x0 + tol]
        ups = [y - y1 for y, lo, hi in horz if y >= y1 - tol and lo < x1 - tol and hi > x0 + tol]
        ups += [c[1] - y1 for c in placed.values() if c[1] >= y1 - tol and c[0] < x1 - tol and c[2] > x0 + tol]
        for gaps in (lefts, rights, downs, ups):
            if gaps:
                sides.append(max(min(gaps), 0.0) * m)
        for g in sides:
            if 1e-3 < g < SLIVER_M:
                cost += W_SLIVER * (1.0 + (SLIVER_M - g) / SLIVER_M)
        return cost

    for r in order:
        w0, h0 = dims[r]
        p = src[r]
        door_nbrs = [v for v in door_adj.get(r, ()) if v in placed]
        best, best_viol, best_cost, best_terms, best_shared = None, None, None, None, None
        for (w, h) in {(w0, h0), (h0, w0)}:
            xs_c = set()
            for x in vx + [c[0] for c in placed.values()] + [c[2] for c in placed.values()]:
                xs_c.update((x, x - w))
            xs_c.add(p[0] - w / 2.0)
            k = 0
            while bx0 + k * step <= bx1 - w + 1e-12:
                xs_c.add(bx0 + k * step)
                k += 1
            ys_c = set()
            for y in hy + [c[1] for c in placed.values()] + [c[3] for c in placed.values()]:
                ys_c.update((y, y - h))
            ys_c.add(p[1] - h / 2.0)
            k = 0
            while by0 + k * step <= by1 - h + 1e-12:
                ys_c.add(by0 + k * step)
                k += 1
            xs_c = [x for x in xs_c if bx0 - tol <= x <= bx1 - w + tol]
            ys_c = [y for y in ys_c if by0 - tol <= y <= by1 - h + tol]
            for x0 in xs_c:
                for y0 in ys_c:
                    rect = (x0, y0, x0 + w, y0 + h)
                    if any(_overlap(rect, c, tol) for c in placed.values()):
                        continue
                    if not bprep.contains(sbox(*rect)):
                        continue
                    # door/open hard constraint: a shared wall below DOOR_MIN_M
                    # on an already-placed door/open neighbour is a violation,
                    # not an instant reject — when every candidate violates
                    # something, the least-violating one still wins (below).
                    shared_door = {v: _shared_wall(rect, placed[v]) for v in door_nbrs}
                    viol = sum(1 for s in shared_door.values() if s + 1e-6 < DOOR_MIN_M)
                    t_node = W_NODE * _rect_point_dist(rect, p) * m
                    t_front, fronted = 0.0, []
                    rb = sbox(*rect).boundary
                    for E in bd.get(r, []):
                        shared = rb.intersection(elines[E]).length * m
                        if shared + 1e-6 >= EDGE_FLOOR_M.get(etype(E), 0.3):
                            fronted.append(E)
                        else:
                            t_front += W_FRONT + W_FRONT_DIST * sbox(*rect).distance(elines[E]) * m
                    # adjacency cost = shortfall of shared wall below a required
                    # floor (DOOR_MIN_M for a door/open neighbour, else the
                    # generic shared_wall floor) — not gap distance. A corner
                    # touch (shared wall 0) scores the full floor, same as no
                    # contact: it is not adjacency.
                    t_adj = 0.0
                    for v in adj.get(r, ()):
                        if v not in placed:
                            continue
                        need = DOOR_MIN_M if v in door_adj.get(r, ()) else EDGE_FLOOR_M.get("shared_wall", 0.3)
                        t_adj += W_ADJ * max(0.0, need - _shared_wall(rect, placed[v]))
                    t_ord = 0.0
                    cx, cy = (rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0
                    for v, c in placed.items():
                        sx, sy = src[r][0] - src[v][0], src[r][1] - src[v][1]
                        nx, ny = cx - (c[0] + c[2]) / 2.0, cy - (c[1] + c[3]) / 2.0
                        if abs(sx) * m > ORDER_TOL_M and sx * nx < 0:
                            t_ord += W_ORDER
                        if abs(sy) * m > ORDER_TOL_M and sy * ny < 0:
                            t_ord += W_ORDER
                    t_sl = sliver_cost(rect)
                    cost = t_node + t_front + t_adj + t_ord + t_sl
                    if (best_viol is None or viol < best_viol or
                            (viol == best_viol and cost < best_cost)):
                        best, best_viol, best_cost = rect, viol, cost
                        best_terms = {"node": t_node, "fronts": t_front, "adjacency": t_adj,
                                      "order": t_ord, "sliver": t_sl, "fronted": fronted}
                        best_shared = shared_door
        if best is None:
            report["rooms"][r] = {"placed": False}
            continue
        placed[r] = best
        relaxed = [{"room": v, "shared_m": round(best_shared.get(v, 0.0), 3)}
                   for v in door_nbrs if best_shared.get(v, 0.0) + 1e-6 < DOOR_MIN_M]
        report["rooms"][r] = {"placed": True,
                              "core": [round(v, 6) for v in best],
                              "cost": round(best_cost, 3),
                              "terms": {k: (round(v, 3) if isinstance(v, float) else v)
                                        for k, v in best_terms.items()},
                              "unfronted": [E for E in bd.get(r, [])
                                            if E not in best_terms["fronted"]],
                              "relaxed": relaxed}

    new_nodes = []
    for n in nodes:
        c = placed.get(str(n["id"]))
        if c is None:
            new_nodes.append(n)
            continue
        p = tuple(n["pos"])
        inside = c[0] <= p[0] <= c[2] and c[1] <= p[1] <= c[3]
        pos = list(p) if inside else [(c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0]
        report["rooms"][str(n["id"])]["moved_m"] = round(math.hypot(pos[0] - p[0], pos[1] - p[1]) * m, 3)
        report["rooms"][str(n["id"])]["from"] = [round(v, 6) for v in p]
        report["rooms"][str(n["id"])]["to"] = [round(v, 6) for v in pos]
        new_nodes.append(dict(n, pos=pos))
    return {"cores": placed, "nodes": new_nodes, "report": report}
