"""Plate assembler (plan 2026-11-06). Contract: ARCHITECTURE.md, "Plate assembler".

Pure functions, no Flask, deterministic: same input gives byte-identical output.
All lengths in metres, areas in m2. Axis-aligned geometry only (orthogonal outlines).

    assemble_plate(boundary_m, mix, street_edges=()) -> dict

Method: generate candidates, reject by hard rules, select by the style sheet order.
  1. Cut the outline into rectangles at reflex vertices (every H/V choice per vertex).
  2. Per wing: access by depth (double / single / through).
  3. One core and one hall network in one "host" wing (a hall needs depth >= d + h).
  4. The rest of the plate is cut into pieces; each piece is cut across into n equal
     rectangular units (n = 0 only when no n >= 1 is valid: the area is then leftover).
  5. Hard rules are checked on every candidate; survivors are ranked (rule 8).

Every threshold is in PARAMS.
"""
import itertools
import math
import os
import statistics

from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

TYPES = ("1B1b", "2B1b", "2B2b", "3B2b")

PARAMS = {
    # rule 2
    "hall_w": 1.5,
    "min_depth": 5.0,
    # rule 3
    # rule 3a (second pass): north is the sun side, relative to true north
    "sun_bearing_deg": 0.0,
    "sun_half_angle_deg": 60.0,
    # rule 3b (second pass): block = service zone + lobby, about 20 m2, up to 8 m deep
    "core_area": 20.0,
    "core_min_w": 2.8,
    "core_max_depth": 8.0,
    "lobby_area": 8.0,
    "service_min_depth": 3.0,
    # rule 5: provisional limits per type
    "types": {
        "1B1b": {"min_area": 36.0, "target_area": 50.0, "min_frontage": 4.5},
        "2B1b": {"min_area": 50.0, "target_area": 65.0, "min_frontage": 5.5},
        "2B2b": {"min_area": 55.0, "target_area": 72.0, "min_frontage": 5.5},
        "3B2b": {"min_area": 65.0, "target_area": 90.0, "min_frontage": 6.5},
    },
    "max_area_factor": 1.4,
    # rule 7
    "door_contact": 0.9,
    "facade_min": 2.4,
    # rule 8 (A6)
    "mix_tolerance": 1,
    # candidate generation (not rules): grid for core and hall-end positions
    "grid": 1.0,       # hall-end positions
    "grid_core": 2.0,  # core positions
    "grid_long": 2.0,  # hall-end step for wings longer than "long_wing" (keeps the search near 5 s)
    "long_wing": 24.0,
    "max_units_per_piece": 6,
    "max_reflex": 10,
    # rule 8a: selection order. Names: mix, sun, area_dev, commons, walk, even, residual, street.
    "selection": ("mix", "sun", "commons", "walk", "area_dev", "even", "residual", "street"),
    # rule 6 / second pass point 4: False = area too small for a unit makes the candidate a reject
    # (no dead strips, no residual); True keeps it as residual (drawn yellow)
    "allow_leftover": False,
    "keep_best": 12,
    # rule 5a: an L's arm is at least as long as the hall is wide
    "min_band": 1.5,
    # numerics
    "eps": 1e-6,
    "snap_decimals": 3,
    "out_decimals": 4,
}

# Diagnostics only: ASSEMBLER_SELECTION="mix,residual,sun,..." overrides the rule 8a order.
if os.environ.get("ASSEMBLER_SELECTION"):
    PARAMS["selection"] = tuple(k.strip() for k in os.environ["ASSEMBLER_SELECTION"].split(",") if k.strip())

REJECT_KEYS = (
    "outside_outline", "overlap", "unit_below_min_area", "unit_below_min_depth",
    "unit_outside_type_limits", "unit_no_door", "unit_no_facade", "unit_not_rect_or_L", "hall_disconnected", "hall_no_door",
    "hall_to_facade_unneeded", "no_units",
)


# ── small rect helpers: rect = (x0, y0, x1, y1) ──────────────────────────────

def _ov(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _contact(r, s, eps):
    """Length of boundary two non-overlapping rects share."""
    n = 0.0
    if abs(r[2] - s[0]) < eps or abs(r[0] - s[2]) < eps:
        n += _ov(r[1], r[3], s[1], s[3])
    if abs(r[3] - s[1]) < eps or abs(r[1] - s[3]) < eps:
        n += _ov(r[0], r[2], s[0], s[2])
    return n


def _area(r):
    return (r[2] - r[0]) * (r[3] - r[1])


def _rect_poly(r):
    return [(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])]


# ── outline ───────────────────────────────────────────────────────────────────

def _prepare(boundary_m):
    """Round to 1 mm, drop repeated and collinear vertices. Return (pts, error)."""
    dec = PARAMS["snap_decimals"]
    pts = []
    for p in boundary_m:
        q = (round(float(p[0]), dec), round(float(p[1]), dec))
        if not pts or q != pts[-1]:
            pts.append(q)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        n = len(pts)
        for i in range(n):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
            if (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1]):
                pts.pop(i)
                changed = True
                break
    if len(pts) < 4:
        return None, "outline needs at least 4 vertices after cleaning"
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        if a[0] != b[0] and a[1] != b[1]:
            return None, f"edge {i} is not axis-aligned"
    poly = Polygon(pts)
    if not poly.is_valid or poly.area <= 0:
        return None, "outline is not a valid polygon"
    return pts, None


def _signed_area(pts):
    return 0.5 * sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                     for i in range(len(pts)))


def _edge_rects(pts):
    out = []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        out.append((min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])))
    return out


# ── rule 1: decompositions ────────────────────────────────────────────────────

def _decompositions(pts):
    """Every way to cut the outline into rectangles by extending each reflex vertex
    horizontally or vertically. Returns a list of rect lists (deduplicated, in order)."""
    xs = sorted({p[0] for p in pts})
    ys = sorted({p[1] for p in pts})
    xi = {x: i for i, x in enumerate(xs)}
    yi = {y: i for i, y in enumerate(ys)}
    nx, ny = len(xs) - 1, len(ys) - 1
    poly = Polygon(pts)
    inside = [[poly.contains(Point((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2)) for j in range(ny)]
              for i in range(nx)]
    sgn = 1 if _signed_area(pts) > 0 else -1
    n = len(pts)
    reflex = []  # (vertex, hdir, vdir): directions of the extensions into the interior
    for k in range(n):
        p, c, q = pts[k - 1], pts[k], pts[(k + 1) % n]
        cross = (c[0] - p[0]) * (q[1] - c[1]) - (c[1] - p[1]) * (q[0] - c[0])
        if cross * sgn < 0:
            hn = p if p[1] == c[1] else q   # neighbour on the horizontal edge
            vn = p if p[0] == c[0] else q
            hdir = 1 if c[0] > hn[0] else -1
            vdir = 1 if c[1] > vn[1] else -1
            reflex.append((c, hdir, vdir))
    if len(reflex) > PARAMS["max_reflex"]:
        return None
    results, seen = [], set()
    for choice in itertools.product("HV", repeat=len(reflex)):
        walls = set()
        for (c, hdir, vdir), ch in zip(reflex, choice):
            if ch == "H":
                j = yi[c[1]]
                rng = range(xi[c[0]], nx) if hdir > 0 else range(xi[c[0]] - 1, -1, -1)
                for i in rng:
                    if 0 < j < ny and inside[i][j - 1] and inside[i][j]:
                        walls.add(("H", i, j))
                    else:
                        break
            else:
                i = xi[c[0]]
                rng = range(yi[c[1]], ny) if vdir > 0 else range(yi[c[1]] - 1, -1, -1)
                for j in rng:
                    if 0 < i < nx and inside[i - 1][j] and inside[i][j]:
                        walls.add(("V", i, j))
                    else:
                        break
        seen_cell, rects, ok = set(), [], True
        for i in range(nx):
            for j in range(ny):
                if not inside[i][j] or (i, j) in seen_cell:
                    continue
                stack, comp = [(i, j)], []
                seen_cell.add((i, j))
                while stack:
                    a, b = stack.pop()
                    comp.append((a, b))
                    for da, db, wall in ((1, 0, ("V", a + 1, b)), (-1, 0, ("V", a, b)),
                                         (0, 1, ("H", a, b + 1)), (0, -1, ("H", a, b))):
                        na, nb = a + da, b + db
                        if 0 <= na < nx and 0 <= nb < ny and inside[na][nb] and (na, nb) not in seen_cell \
                                and wall not in walls:
                            seen_cell.add((na, nb))
                            stack.append((na, nb))
                ia = [c[0] for c in comp]
                ib = [c[1] for c in comp]
                r = (xs[min(ia)], ys[min(ib)], xs[max(ia) + 1], ys[max(ib) + 1])
                cells = sum((xs[a + 1] - xs[a]) * (ys[b + 1] - ys[b]) for a, b in comp)
                if abs(cells - _area(r)) > 1e-9:
                    ok = False
                    break
                rects.append(r)
            if not ok:
                break
        if not ok:
            continue
        key = tuple(sorted(rects))
        if key not in seen:
            seen.add(key)
            results.append(list(key))
    return results


# ── mix ───────────────────────────────────────────────────────────────────────

def _proportions(mix):
    if not isinstance(mix, dict):
        raise ValueError("mix must be an object keyed by unit type")
    w = {}
    for t in TYPES:
        v = mix.get(t, 0)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"mix['{t}'] must be a number >= 0")
        w[t] = float(v)
    for k in mix:
        if k not in TYPES:
            raise ValueError(f"unknown unit type '{k}'")
    s = sum(w.values())
    if s <= 0:
        raise ValueError("mix needs at least one weight above zero")
    return {t: w[t] / s for t in TYPES}


def _largest_remainder(props, n):
    raw = [props[t] * n for t in TYPES]
    base = [int(math.floor(r + 1e-9)) for r in raw]
    left = n - sum(base)
    order = sorted(range(len(TYPES)), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[:left]:
        base[i] += 1
    return tuple(base)


_ASSIGN_CACHE = {}


_ASSIGN_CACHE = {}


def _best_counts(masks, req):
    """Rule 5b: types follow size. `masks` are eligibility masks of the units in ascending area
    order; the smallest units take the smallest types, so a count vector fixes the assignment.
    Returns (error, [count vectors with that error])."""
    key = (masks, req)
    hit = _ASSIGN_CACHE.get(key)
    if hit is not None:
        return hit
    n = len(masks)
    best, vecs = None, []

    def rec(t, i, counts):
        nonlocal best, vecs
        if t == 4:
            if i == n:
                err = sum(abs(counts[k] - req[k]) for k in range(4))
                if best is None or err < best:
                    best, vecs = err, [counts]
                elif err == best:
                    vecs.append(counts)
            return
        k = 0
        while True:
            rec(t + 1, i + k, counts + (k,))
            if i + k < n and masks[i + k] >> t & 1:
                k += 1
            else:
                break

    rec(0, 0, ())
    out = (best, vecs)
    if len(_ASSIGN_CACHE) > 200000:
        _ASSIGN_CACHE.clear()
    _ASSIGN_CACHE[key] = out
    return out


def _types_for(counts):
    seq = []
    for t, c in enumerate(counts):
        seq.extend([t] * c)
    return seq


# ── unit evaluation (rules 5, 5a and 7) ──────────────────────────────────────

def _type_mask(area, frontage):
    P = PARAMS
    m = 0
    for i, t in enumerate(TYPES):
        spec = P["types"][t]
        if spec["min_area"] - 1e-9 <= area <= P["max_area_factor"] * spec["target_area"] + 1e-9 \
                and frontage >= spec["min_frontage"] - 1e-9:
            m |= 1 << i
    return m


def _allowed_mask(props):
    """Bit mask of the types with a mix weight above 0. A weight of 0 excludes the type."""
    m = 0
    for i, t in enumerate(TYPES):
        if props[t] > 0:
            m |= 1 << i
    return m


class _Ctx:
    def __init__(self, pts, street_edges, north_deg, allowed=None):
        self.allowed = (1 << len(TYPES)) - 1 if allowed is None else allowed
        # units rejected only because their area/frontage fit excluded types: parts -> (area, raw mask)
        self.blocked = {}
        self.pts = pts
        self.north_deg = north_deg
        self.edges = _edge_rects(pts)
        self.plate_area = Polygon(pts).area
        n = len(pts)
        self.street = [self.edges[e] for e in street_edges if 0 <= e < n]
        self.xs = sorted({p[0] for p in pts})
        self.ys = sorted({p[1] for p in pts})
        # rule 3a: facade bearing from the outward normal and north_deg (clockwise from canvas up)
        sgn = 1 if _signed_area(pts) > 0 else -1
        th = math.radians(north_deg)
        north = (math.sin(th), -math.cos(th))
        east = (math.cos(th), math.sin(th))
        self.bearing = []   # compass bearing of each edge's outward normal, degrees [0, 360)
        self.sun_w = []     # weight of each edge for the sun-side test
        self.sun_side = []  # True when the facade is within the sun half-angle
        sun = math.radians(PARAMS["sun_bearing_deg"])
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            dx, dy = b[0] - a[0], b[1] - a[1]
            ln = math.hypot(dx, dy)
            nx, ny = (dy / ln * sgn, -dx / ln * sgn)
            brg = math.degrees(math.atan2(nx * east[0] + ny * east[1], nx * north[0] + ny * north[1])) % 360.0
            self.bearing.append(round(brg, 6))
            rel = math.radians(brg) - sun
            self.sun_w.append(max(0.0, math.cos(rel)))
            self.sun_side.append(abs(((brg - PARAMS["sun_bearing_deg"] + 180) % 360) - 180) < PARAMS["sun_half_angle_deg"] - 1e-9)

    def _edge_overlap(self, r, i):
        e, eps = self.edges[i], PARAMS["eps"]
        if e[0] == e[2]:
            if abs(e[0] - r[0]) < eps or abs(e[0] - r[2]) < eps:
                return _ov(r[1], r[3], e[1], e[3])
        elif abs(e[1] - r[1]) < eps or abs(e[1] - r[3]) < eps:
            return _ov(r[0], r[2], e[0], e[2])
        return 0.0

    def facade(self, r):
        return sum(self._edge_overlap(r, i) for i in range(len(self.edges)))

    def sun_front(self, r):
        """(weighted frontage on sun-facing facades, length on sun-side facades, facade list)."""
        wt, hard, facs = 0.0, 0.0, []
        for i in range(len(self.edges)):
            L = self._edge_overlap(r, i)
            if L > 1e-9:
                wt += L * self.sun_w[i]
                if self.sun_side[i]:
                    hard += L
                facs.append({"edge": i, "bearing_deg": round(self.bearing[i], 1), "length_m": round(L, 4),
                             "sun_side": self.sun_side[i]})
        return wt, hard, facs

    def street_len(self, r):
        eps = PARAMS["eps"]
        tot = 0.0
        for e in self.street:
            if e[0] == e[2]:
                if abs(e[0] - r[0]) < eps or abs(e[0] - r[2]) < eps:
                    tot += _ov(r[1], r[3], e[1], e[3])
            else:
                if abs(e[1] - r[1]) < eps or abs(e[1] - r[3]) < eps:
                    tot += _ov(r[0], r[2], e[0], e[2])
        return tot


def _clean_ring(coords):
    pts = [(round(x, 6), round(y, 6)) for x, y in coords[:-1]]
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            if (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1]):
                pts.pop(i)
                changed = True
                break
    return pts


def _union_ring(parts):
    """Counter-clockwise ring of the union of rects, or None if it is not one simple polygon."""
    if len(parts) == 1:
        r = parts[0]
        return [(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3])]
    u = unary_union([box(*p) for p in parts])
    if u.geom_type != "Polygon" or u.interiors:
        return None
    ring = _clean_ring(list(orient(u, 1.0).exterior.coords))
    k = min(range(len(ring)), key=lambda i: (ring[i][1], ring[i][0]))
    return ring[k:] + ring[:k]


def _l_ok(parts):
    """Rule 5a: base rect plus one extra forms a rectangle or an L with one step. The extra must lie
    within the base's side span and be flush with one end of it."""
    if len(parts) != 2:
        return False
    r, e, eps = parts[0], parts[1], PARAMS["eps"]
    if abs(r[2] - e[0]) < eps or abs(r[0] - e[2]) < eps:      # extra beside the base (x sides touch)
        lo, hi, elo, ehi = r[1], r[3], e[1], e[3]
    elif abs(r[3] - e[1]) < eps or abs(r[1] - e[3]) < eps:    # extra above or below the base
        lo, hi, elo, ehi = r[0], r[2], e[0], e[2]
    else:
        return False
    if elo < lo - eps or ehi > hi + eps:
        return False
    return abs(elo - lo) < eps or abs(ehi - hi) < eps


def _unit_eval(parts, doors, ctx, walk_c):
    """parts: [base rect, extras...]. Returns (record, violations).
    record = (area, cut_w, depth, facade, door, entrance, mask, walk)."""
    P, eps = PARAMS, PARAMS["eps"]
    viol = set()
    if len(parts) > 1 and not _l_ok(parts):
        viol.add("unit_not_rect_or_L")
    area = sum(_area(r) for r in parts)
    facade = sum(ctx.facade(r) for r in parts)
    door, best, ent, vertical, dpart = 0.0, 0.0, None, False, parts[0]
    for r in parts:
        for c in doors:
            L = _contact(r, c, eps)
            if L <= eps:
                continue
            door += L
            if L > best + 1e-9:
                best, dpart = L, r
                if abs(r[2] - c[0]) < eps or abs(r[0] - c[2]) < eps:
                    vertical = True
                    x = r[2] if abs(r[2] - c[0]) < eps else r[0]
                    lo, hi = max(r[1], c[1]), min(r[3], c[3])
                    ent = (x, (lo + hi) / 2)
                else:
                    vertical = False
                    y = r[3] if abs(r[3] - c[1]) < eps else r[1]
                    lo, hi = max(r[0], c[0]), min(r[2], c[2])
                    ent = ((lo + hi) / 2, y)
    w, h = dpart[2] - dpart[0], dpart[3] - dpart[1]
    if ent is None:
        depth = min(w, h)
        ent = ((parts[0][0] + parts[0][2]) / 2, (parts[0][1] + parts[0][3]) / 2)
    else:
        depth = w if vertical else h
    if area < min(s["min_area"] for s in P["types"].values()) - 1e-9:
        viol.add("unit_below_min_area")
    if depth < P["min_depth"] - 1e-9:
        viol.add("unit_below_min_depth")
    if door < P["door_contact"] - 1e-9:
        viol.add("unit_no_door")
    if facade < P["facade_min"] - 1e-9:
        viol.add("unit_no_facade")
    raw = _type_mask(area, facade)
    mask = raw & ctx.allowed
    if mask == 0 and "unit_below_min_area" not in viol and "unit_not_rect_or_L" not in viol:
        viol.add("unit_outside_type_limits")
        if raw != 0 and len(viol) == 1:
            ctx.blocked[tuple(parts)] = (area, raw)
    walk = abs(ent[0] - walk_c[0]) + abs(ent[1] - walk_c[1])
    return (area, None, depth, facade, door, ent, mask, walk), viol


# ── skeleton: core block + hall in one host wing, then pieces ────────────────

def _wing_access(rect):
    """Rule 2 for a wing. Returns (axis, L, D, mode)."""
    w, h = rect[2] - rect[0], rect[3] - rect[1]
    axis = "x" if w >= h else "y"
    L, D = (w, h) if axis == "x" else (h, w)
    P = PARAMS
    if D >= 2 * P["min_depth"] + P["hall_w"] - 1e-9:
        mode = "double"
    elif D >= P["min_depth"] + P["hall_w"] - 1e-9:
        mode = "single"
    else:
        mode = "through"
    return axis, L, D, mode


def _skeletons(wing, ctx):
    """Yield skeleton dicts for one host wing (rules 2, 3, 3b, 4a)."""
    P = PARAMS
    axis, L, D, mode = _wing_access(wing)
    if mode == "through":
        return
    h, d = P["hall_w"], P["min_depth"]
    x0, y0 = wing[0], wing[1]
    tmin = min(s["min_area"] for s in P["types"].values())
    tmax = P["max_area_factor"] * max(s["target_area"] for s in P["types"].values())
    arm_min = min(s["min_frontage"] for s in P["types"].values())
    we_lo, we_hi = max(d, tmin / D), tmax / D
    grid = P["grid"] if L <= P["long_wing"] else P["grid_long"]
    off = x0 if axis == "x" else y0
    snaps = sorted({round(c - off, 6) for c in (ctx.xs if axis == "x" else ctx.ys) if 0 < c - off < L})
    base = sorted({round(k * grid, 6) for k in range(0, int(L / grid + 1e-9) + 1)} | set(snaps) | {round(L, 6)})
    eps = P["eps"]

    def uv(u0, u1, v0, v1, flip):
        if flip:
            v0, v1 = D - v1, D - v0
        if axis == "x":
            return (x0 + u0, y0 + v0, x0 + u1, y0 + v1)
        return (x0 + v0, y0 + u0, x0 + v1, y0 + u1)

    a_opts = [None]
    if mode == "double":
        even = (D - h) / 2
        a_opts = sorted({a for a in (round(d, 6), round(even, 6), round(D - h - d, 6))
                         if a + h <= P["core_max_depth"] + 1e-9 and D - h - a >= d - 1e-9})
    for a in a_opts:
        for flip in (False, True):
            if mode == "double":
                cd = a + h
                hv = (a, a + h)
                sliver_v = None
            else:
                cd = min(P["core_max_depth"], D)
                hv = (0.0, h)
                sliver_v = (cd, D) if D - cd > eps else None
            cw = math.ceil(max(P["core_min_w"], P["core_area"] / cd) * 10 - 1e-9) / 10
            if cw > L:
                continue
            ld = max(h, min(P["lobby_area"] / cw, cd - P["service_min_depth"]))
            if mode == "double":
                lobby_v, service_v = (cd - ld, cd), (0.0, cd - ld)
            else:
                lobby_v, service_v = (0.0, ld), (ld, cd)
            gc = P["grid_core"]
            cbase = {round(k * gc, 6) for k in range(0, int(L / gc + 1e-9) + 1)} | set(snaps) | {0.0, round(L, 6)}
            u0s = sorted({g for g in cbase if g <= L - cw + 1e-9}
                         | {round(s - cw, 6) for s in snaps + [round(L, 6)] if s - cw >= 0})
            for u0 in u0s:
                ur0 = round(u0 + cw, 6)
                mb = P["min_band"]
                lefts = [0.0, u0] + [g for g in base if mb - 1e-9 <= g < u0 - arm_min + 1e-9]
                rights = [round(L, 6), ur0] + [g for g in base if ur0 + arm_min - 1e-9 < g <= L - mb + 1e-9]
                lefts = [g for g in lefts if g < eps or g >= mb - 1e-9]
                rights = [g for g in rights if L - g < eps or L - g >= mb - 1e-9]
                block = uv(u0, ur0, 0.0, cd, flip)
                lobby = uv(u0, ur0, lobby_v[0], lobby_v[1], flip)
                service = uv(u0, ur0, service_v[0], service_v[1], flip)
                for uL in sorted(set(lefts)):
                    mL = ["none"] if uL < eps else (["T"] if we_lo - 1e-9 <= uL <= we_hi + 1e-9 else []) \
                        + ["A"] + (["B"] if mode == "double" else [])
                    for uR in sorted(set(rights)):
                        bl = L - uR
                        mR = ["none"] if bl < eps else (["T"] if we_lo - 1e-9 <= bl <= we_hi + 1e-9 else []) \
                            + ["A"] + (["B"] if mode == "double" else [])
                        halls = []
                        if u0 - uL > eps:
                            halls.append(uv(uL, u0, hv[0], hv[1], flip))
                        if uR - ur0 > eps:
                            halls.append(uv(ur0, uR, hv[0], hv[1], flip))
                        for ml in mL:
                            for mr in mR:
                                sl_sides = ("L", "R") if sliver_v else (None,)
                                for sl in sl_sides:
                                    startL = uL if ml == "T" else 0.0
                                    endR = uR if mr == "T" else L
                                    bandL = uv(0, uL, hv[0], hv[1], flip) if uL > eps else None
                                    bandR = uv(uR, L, hv[0], hv[1], flip) if bl > eps else None
                                    exL, exR, exB_f, exB_l, exS_L, exS_R = [], [], [], [], [], []
                                    if ml in ("A", "B") and bandL:
                                        (exL if ml == "A" else exB_f).append((bandL, "first"))
                                    if mr in ("A", "B") and bandR:
                                        (exR if mr == "A" else exB_l).append((bandR, "last"))
                                    if sl and sliver_v:
                                        sv = uv(u0, ur0, sliver_v[0], sliver_v[1], flip)
                                        (exL if sl == "L" else exR).append((sv, "last" if sl == "L" else "first"))
                                    pieces = []
                                    if ml == "T":
                                        pieces.append((uv(0, uL, 0, D, flip), "through", None, ()))
                                    if mr == "T":
                                        pieces.append((uv(uR, L, 0, D, flip), "through", None, ()))
                                    if mode == "double":
                                        if u0 - startL > eps:
                                            pieces.append((uv(startL, u0, 0, hv[0], flip), "row", axis, tuple(exL)))
                                        if endR - ur0 > eps:
                                            pieces.append((uv(ur0, endR, 0, hv[0], flip), "row", axis, tuple(exR)))
                                        if endR - startL > eps:
                                            pieces.append((uv(startL, endR, hv[1], D, flip), "row", axis,
                                                           tuple(exB_f + exB_l)))
                                    else:
                                        if u0 - startL > eps:
                                            pieces.append((uv(startL, u0, h, D, flip), "row", axis, tuple(exL)))
                                        if endR - ur0 > eps:
                                            pieces.append((uv(ur0, endR, h, D, flip), "row", axis, tuple(exR)))
                                    yield {
                                        "wing": wing, "axis": axis, "L": L, "D": D, "mode": mode, "a": a, "flip": flip,
                                        "block": block, "lobby": lobby, "service": service, "core_dims": (cw, cd),
                                        "halls": halls, "pieces": pieces, "u0": u0, "uL": uL, "uR": uR,
                                        "ends": (ml, mr),
                                    }


def _merge_variants(pieces):
    """Variants of the non-row pieces where adjacent pieces that form a rectangle are merged."""
    fixed = [p for p in pieces if p[1] == "row"]
    free = [p for p in pieces if p[1] != "row"]
    eps = PARAMS["eps"]

    def union(r, s):
        if abs(r[0] - s[0]) < eps and abs(r[2] - s[2]) < eps and (abs(r[3] - s[1]) < eps or abs(s[3] - r[1]) < eps):
            return (r[0], min(r[1], s[1]), r[2], max(r[3], s[3]))
        if abs(r[1] - s[1]) < eps and abs(r[3] - s[3]) < eps and (abs(r[2] - s[0]) < eps or abs(s[2] - r[0]) < eps):
            return (min(r[0], s[0]), r[1], max(r[2], s[2]), r[3])
        return None

    def fin(rs):
        out = []
        for r in rs:
            w, h = r[2] - r[0], r[3] - r[1]
            out.append((r, "wing", "x" if w >= h else "y", ()))
        return out

    rects = [p[0] for p in free]
    variants = [free]
    seen = {tuple(sorted(rects))}
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            u = union(rects[i], rects[j])
            if u is None:
                continue
            rest = [r for k, r in enumerate(rects) if k not in (i, j)] + [u]
            key = tuple(sorted(rest))
            if key not in seen:
                seen.add(key)
                variants.append(fin(sorted(rest)))
    cur = list(rects)
    changed = True
    while changed:
        changed = False
        for i in range(len(cur)):
            for j in range(i + 1, len(cur)):
                u = union(cur[i], cur[j])
                if u is not None:
                    cur = [r for k, r in enumerate(cur) if k not in (i, j)] + [u]
                    changed = True
                    break
            if changed:
                break
    key = tuple(sorted(cur))
    if key not in seen:
        variants.append(fin(sorted(cur)))
    return [fixed + v for v in variants]


def _split(piece, n):
    r, kind, cut, extras = piece
    if n == 1:
        base = [r]
    else:
        base = []
        for k in range(n):
            if cut == "x":
                base.append((r[0] + (r[2] - r[0]) * k / n, r[1], r[0] + (r[2] - r[0]) * (k + 1) / n, r[3]))
            else:
                base.append((r[0], r[1] + (r[3] - r[1]) * k / n, r[2], r[1] + (r[3] - r[1]) * (k + 1) / n))
    units = [[b] for b in base]
    for rect, where in extras:
        units[0 if where == "first" else n - 1].append(rect)
    return units


def _piece_options(piece, doors, ctx, walk_c, cache):
    """Options for one piece: list of (n, unit part lists, records, viol). The leftover option
    (n = 0) is added only when no n >= 1 is valid. Returns (considered, valid)."""
    key = (piece, tuple(doors))
    hit = cache.get(key)
    if hit is not None:
        return hit
    P = PARAMS
    r, kind, cut, extras = piece
    tmin = min(s["min_area"] for s in P["types"].values())
    if _area(r) + sum(_area(e[0]) for e in extras) < tmin - 1e-9:
        nmax = 0
        if not P["allow_leftover"]:
            out = ([(0, [], [], {"unit_below_min_area"})], [])  # a piece no unit can take rejects the candidate
            cache[key] = out
            return out
    elif kind == "through":
        nmax = 1
    else:
        length = (r[2] - r[0]) if cut == "x" else (r[3] - r[1])
        nmax = max(1, min(P["max_units_per_piece"], int(length / min(s["min_frontage"] for s in P["types"].values()) + 1e-9)))
    considered = []
    for n in range(1, nmax + 1):
        units = _split(piece, n)
        recs, viol = [], set()
        for parts in units:
            rec, v = _unit_eval(parts, doors, ctx, walk_c)
            u = parts[0]
            w = (u[2] - u[0]) if cut == "x" else (u[3] - u[1]) if cut else min(u[2] - u[0], u[3] - u[1])
            recs.append(rec[:1] + (w,) + rec[2:])
            viol |= v
        considered.append((n, units, recs, viol))
    valid = [o for o in considered if not o[3]]
    if not valid and P["allow_leftover"] and _area(r) + sum(_area(e[0]) for e in extras) < tmin - 1e-9:
        # rule 6: only area too small for any unit is leftover; a larger piece that cannot hold a
        # valid unit makes the candidate a reject (no dead strips, no silent residual)
        left = (0, [], [], set())
        valid = [left]
        considered = considered + [left]
    out = (considered, valid)
    cache[key] = out
    return out


# ── candidate search ─────────────────────────────────────────────────────────

def _wall_ends(sk):
    """Rule 4a: halls whose end lies on the wing end (a facade). Returns [(hall, end_x_or_y, axis_index)]."""
    out = []
    eps = PARAMS["eps"]
    wing = sk["wing"]
    ai = 0 if sk["axis"] == "x" else 1
    lo, hi = wing[ai], wing[ai + 2]
    for hl in sk["halls"]:
        if abs(hl[ai] - lo) < eps:
            out.append((hl, lo, ai))
        if abs(hl[ai + 2] - hi) < eps:
            out.append((hl, hi, ai))
    return out


def _hall_ends_needed(wall_ends, combo):
    """A hall may reach a facade only if shortening it by one hall width would strand a unit:
    some unit ending there keeps less than the door contact once 1.5 m of hall is cut."""
    P, eps = PARAMS, PARAMS["eps"]
    for hall, end, ai in wall_ends:
        for o in combo:
            for parts, rec in zip(o[1], o[2]):
                reach = False
                for r in parts:
                    if _contact(r, hall, eps) > eps:
                        lo = max(r[ai], hall[ai])
                        hi = min(r[ai + 2], hall[ai + 2])
                        if abs(lo - end) < eps or abs(hi - end) < eps:
                            reach = True
                if reach and rec[4] - P["hall_w"] < P["door_contact"] - eps:
                    return True
    return False


def _search(pts, props, street_edges, north_deg):
    P = PARAMS
    ctx = _Ctx(pts, street_edges, north_deg, _allowed_mask(props))
    decs = _decompositions(pts)
    if decs is None:
        return None, {"reason": f"more than {P['max_reflex']} reflex vertices"}
    tried = survived = 0
    rejects = {k: 0 for k in REJECT_KEYS}
    best = []
    gen_idx = 0
    eps = P["eps"]
    seen_sk = set()
    targets = [P["types"][t]["target_area"] for t in TYPES]
    for dec in decs:
        for hi, wing in enumerate(dec):
            for sk in _skeletons(wing, ctx):
                others = [(r, "wing", "x" if (r[2] - r[0]) >= (r[3] - r[1]) else "y", ())
                          for k, r in enumerate(dec) if k != hi]
                doors = sk["halls"] + [sk["lobby"]]
                block = sk["block"]
                walk_c = ((sk["lobby"][0] + sk["lobby"][2]) / 2, (sk["lobby"][1] + sk["lobby"][3]) / 2)
                sun_w, sun_hard, _ = ctx.sun_front(block)
                commons_area = _area(block) + sum(_area(r) for r in sk["halls"])
                street_c = ctx.street_len(block) + sum(ctx.street_len(r) for r in sk["halls"])
                skkey = (block, sk["lobby"], tuple(sk["halls"]), tuple(sorted(p[0] for p in sk["pieces"])),
                         tuple(p[3] for p in sk["pieces"]), tuple(dec), wing)
                wall_ends = _wall_ends(sk)
                cache = {}
                for var in _merge_variants(sk["pieces"] + others):
                    vkey = (skkey, tuple(p[0] for p in var), tuple(p[3] for p in var))
                    if vkey in seen_sk:
                        continue
                    seen_sk.add(vkey)
                    opts = [_piece_options(p, doors, ctx, walk_c, cache) for p in var]
                    total = 1
                    for cons, _ in opts:
                        total *= len(cons)
                    tried += total
                    for rule in REJECT_KEYS:
                        if rule in ("outside_outline", "overlap", "hall_disconnected", "hall_no_door",
                                    "hall_to_facade_unneeded", "no_units"):
                            continue
                        ok = 1
                        for cons, _ in opts:
                            ok *= sum(1 for o in cons if rule not in o[3])
                        rejects[rule] += total - ok
                    for combo in itertools.product(*[v for _, v in opts]):
                        n_units = sum(o[0] for o in combo)
                        if n_units == 0:
                            rejects["no_units"] += 1
                            continue
                        if wall_ends and not _hall_ends_needed(wall_ends, combo):
                            rejects["hall_to_facade_unneeded"] += 1
                            continue
                        survived += 1
                        recs = [rec for o in combo for rec in o[2]]
                        order = sorted(recs, key=lambda rc: (round(rc[0], 6), rc[6]))
                        masks = tuple(rc[6] for rc in order)
                        req = _largest_remainder(props, n_units)
                        err, vecs = _best_counts(masks, req)
                        uarea = sum(rc[0] for rc in recs)
                        dev = None
                        for vec in vecs:
                            dv = sum(abs(order[i][0] - targets[t]) for i, t in enumerate(_types_for(vec)))
                            if dev is None or dv < dev - 1e-9:
                                dev = dv
                        resid = sum(_area(var[i][0]) + sum(_area(e[0]) for e in var[i][3])
                                    for i, o in enumerate(combo) if o[0] == 0)
                        walk = max(rc[7] for rc in recs)
                        widths = [rc[1] for rc in recs]
                        even = statistics.pstdev(widths) if len(widths) > 1 else 0.0
                        # rule 8a
                        vals = {"mix": err, "sun": round(sun_w, 6), "area_dev": round(dev, 6),
                                "commons": round(commons_area, 6), "walk": round(walk, 6),
                                "even": round(even, 6), "residual": round(resid, 6), "street": round(street_c, 6)}
                        key = tuple(vals[k] for k in P["selection"]) + (gen_idx,)
                        gen_idx += 1
                        if len(best) < P["keep_best"] or key < best[-1][0]:
                            best.append((key, (sk, var, combo, req, others)))
                            best.sort(key=lambda t: t[0])
                            del best[P["keep_best"]:]
    return {"best": best, "tried": tried, "survived": survived, "rejects": rejects, "ctx": ctx}, None


# ── output ───────────────────────────────────────────────────────────────────

def _r(v):
    return round(v + 0.0, PARAMS["out_decimals"])


def _poly_out(poly):
    return [[_r(x), _r(y)] for x, y in poly]


def _edge_intervals(a, b, rects, eps):
    """Intervals (t0, t1) along the axis-aligned segment a->b covered by the sides of `rects`."""
    out = []
    if a[0] == b[0]:
        lo, hi = sorted((a[1], b[1]))
        for e in rects:
            if abs(e[0] - a[0]) < eps or abs(e[2] - a[0]) < eps:
                o0, o1 = max(lo, e[1]), min(hi, e[3])
                if o1 - o0 > eps:
                    out.append((o0, o1))
    else:
        lo, hi = sorted((a[0], b[0]))
        for e in rects:
            if abs(e[1] - a[1]) < eps or abs(e[3] - a[1]) < eps:
                o0, o1 = max(lo, e[0]), min(hi, e[2])
                if o1 - o0 > eps:
                    out.append((o0, o1))
    return out


def _typed_polygon(ring, ctx, doors):
    """Ring with extra vertices where contact changes, and one type per edge (rule 9)."""
    eps = PARAMS["eps"]
    poly, types = [], []
    n = len(ring)
    for k in range(n):
        a, b = ring[k], ring[(k + 1) % n]
        horizontal = a[1] == b[1]
        ai = 0 if horizontal else 1
        lo, hi = sorted((a[ai], b[ai]))
        ext = _edge_intervals(a, b, ctx.edges, eps)
        com = _edge_intervals(a, b, doors, eps)
        cuts = {lo, hi}
        for s, e in ext + com:
            cuts.add(max(lo, min(hi, s)))
            cuts.add(max(lo, min(hi, e)))
        cs = sorted(cuts)
        if a[ai] > b[ai]:
            cs.reverse()
        for i in range(len(cs) - 1):
            p0 = (cs[i], a[1]) if horizontal else (a[0], cs[i])
            mid = (cs[i] + cs[i + 1]) / 2
            kind = "shared_wall"
            if any(s - eps <= mid <= e + eps for s, e in ext):
                kind = "external"
            elif any(s - eps <= mid <= e + eps for s, e in com):
                kind = "facing_corridor"
            poly.append(p0)
            types.append(kind)
    return poly, types


def _build(sk, var, combo, req, ctx, props):
    P = PARAMS
    doors = sk["halls"] + [sk["lobby"]]
    unit_parts, recs = [], []
    for o in combo:
        unit_parts.extend(o[1])
        recs.extend(o[2])
    order = sorted(range(len(unit_parts)), key=lambda i: (unit_parts[i][0][1], unit_parts[i][0][0]))
    unit_parts = [unit_parts[i] for i in order]
    recs = [recs[i] for i in order]
    # rule 5b: types follow size
    by_area = sorted(range(len(recs)), key=lambda i: (round(recs[i][0], 6), recs[i][6], i))
    masks = tuple(recs[i][6] for i in by_area)
    err, vecs = _best_counts(masks, req)
    targets = [P["types"][t]["target_area"] for t in TYPES]
    vec = min(vecs, key=lambda v: (sum(abs(recs[by_area[i]][0] - targets[t]) for i, t in enumerate(_types_for(v))), v))
    types = [None] * len(recs)
    for pos, t in enumerate(_types_for(vec)):
        types[by_area[pos]] = t
    pend = []
    for i, o in enumerate(combo):
        if o[0] == 0:
            pend.append(var[i][0])
            pend.extend(e[0] for e in var[i][3])
    c = sk["block"]
    cw, cd = sk["core_dims"]
    wt, hard, facs = ctx.sun_front(c)
    cores = [{
        "id": "core-1", "type": "stair_lift", "polygon": _poly_out(_rect_poly(c)),
        "service_polygon": _poly_out(_rect_poly(sk["service"])),
        "lobby_polygon": _poly_out(_rect_poly(sk["lobby"])),
        "centroid": [_r((c[0] + c[2]) / 2), _r((c[1] + c[3]) / 2)],
        "width": _r(cw), "depth": _r(cd),
        "facades": facs, "sun_frontage_m": _r(hard),
    }]
    corridors = []
    for k, r in enumerate(sk["halls"]):
        horiz = (r[2] - r[0]) >= (r[3] - r[1])
        mid = ((r[1] + r[3]) / 2) if horiz else ((r[0] + r[2]) / 2)
        cl = [[_r(r[0]), _r(mid)], [_r(r[2]), _r(mid)]] if horiz else [[_r(mid), _r(r[1])], [_r(mid), _r(r[3])]]
        corridors.append({"id": f"hall-{k + 1}", "kind": "hall", "polygon": _poly_out(_rect_poly(r)),
                          "width": _r(min(r[2] - r[0], r[3] - r[1])), "centreline": cl})
    units = []
    for i, (parts, rec) in enumerate(zip(unit_parts, recs)):
        ring = _union_ring(parts)
        poly, edge_types = _typed_polygon(ring, ctx, doors)
        units.append({
            "id": f"u{i + 1}", "type": TYPES[types[i]], "polygon": _poly_out(poly),
            "typed_edges": [{"edge_index": k, "type": t} for k, t in enumerate(edge_types)],
            "entrance_point": [_r(rec[5][0]), _r(rec[5][1])],
            "depth": _r(rec[2]), "area": _r(rec[0]), "shape": "L" if len(ring) == 6 else "rect",
        })
    residual = [_poly_out(_rect_poly(r)) for r in pend]
    n = len(units)
    achieved = {t: sum(1 for u in units if u["type"] == t) for t in TYPES}
    requested_counts = {t: req[k] for k, t in enumerate(TYPES)}
    err_t = {t: achieved[t] - requested_counts[t] for t in TYPES}
    lob = sk["lobby"]
    walk_c = ((lob[0] + lob[2]) / 2, (lob[1] + lob[3]) / 2)
    walk = max(abs(rec[5][0] - walk_c[0]) + abs(rec[5][1] - walk_c[1]) for rec in recs)
    commons_area = _area(c) + sum(_area(r) for r in sk["halls"])
    res_area = sum(_area(r) for r in pend)
    typology = []
    for k, w in enumerate(sorted([sk["wing"]] + list(sk["others_wings"]))):
        axis, L, D, mode = _wing_access(w)
        typology.append({"id": f"w{k + 1}", "rect": [_r(v) for v in w], "length": _r(L), "depth": _r(D),
                         "access": {"double": "double_loaded", "single": "single_loaded", "through": "through"}[mode],
                         "host": w == sk["wing"]})
    metrics = {
        "units": n,
        "requested_mix": {t: _r(props[t]) for t in TYPES},
        "requested_counts": requested_counts,
        "achieved_counts": achieved,
        "mix_error_per_type": err_t,
        "mix_error_total": sum(abs(v) for v in err_t.values()),
        "mix_within_one": all(abs(v) <= P["mix_tolerance"] for v in err_t.values()),
        "commons_pct": round(100 * commons_area / ctx.plate_area, 2),
        "max_walk_m": _r(walk),
        "residual_m2": _r(res_area),
        "unit_areas_m2": [_r(rec[0]) for rec in recs],
        "l_units": sum(1 for u in units if u["shape"] == "L"),
        "plate_area_m2": _r(ctx.plate_area),
        "unit_area_total_m2": _r(sum(rec[0] for rec in recs)),
        "commons_m2": _r(commons_area),
        "core_m2": _r(_area(c)),
        "lobby_m2": _r(_area(sk["lobby"])),
        "service_m2": _r(_area(sk["service"])),
        "core_facades": facs,
        "core_sun_frontage_m": _r(hard),
        "north_deg": ctx.north_deg,
    }
    return {
        "boundary": [[_r(x), _r(y)] for x, y in ctx.pts],
        "typology": typology, "cores": cores, "corridors": corridors, "units": units,
        "residual": residual, "metrics": metrics,
    }


# ── verification of an output dict (shapely; independent of the construction) ─

def verify_result(result, allowed_types=None):
    """Check an assembler output against the hard rules. Returns counts per rule (all 0 = pass).
    allowed_types: types a unit may take (default all); any other type counts as bad_type."""
    P = PARAMS
    out = {k: 0 for k in REJECT_KEYS}
    out["bad_type"] = 0
    B = Polygon(result["boundary"])
    Bb = B.buffer(1e-6)
    cores = [Polygon(c["polygon"]) for c in result["cores"]]
    lobbies = [Polygon(c["lobby_polygon"]) for c in result["cores"]]
    halls = [Polygon(c["polygon"]) for c in result["corridors"]]
    units = [Polygon(u["polygon"]) for u in result["units"]]
    for g in cores + halls + units:
        if not Bb.contains(g):
            out["outside_outline"] += 1
    items = units + cores + halls
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i].intersection(items[j]).area > 1e-6:
                out["overlap"] += 1
    doors = unary_union(lobbies + halls)
    for u, g in zip(result["units"], units):
        spec = P["types"].get(u["type"])
        if spec is None or (allowed_types is not None and u["type"] not in allowed_types):
            out["bad_type"] += 1
            continue
        if g.area < spec["min_area"] - 1e-6:
            out["unit_below_min_area"] += 1
        if g.area > P["max_area_factor"] * spec["target_area"] + 1e-6:
            out["unit_outside_type_limits"] += 1
        ring = _clean_ring(list(g.exterior.coords))
        if len(ring) > 6:
            out["unit_not_rect_or_L"] += 1
        door = g.intersection(doors).length
        if door < P["door_contact"] - 1e-6:
            out["unit_no_door"] += 1
        if g.intersection(B.boundary).length < P["facade_min"] - 1e-6:
            out["unit_no_facade"] += 1
        # depth: extent of the unit across its longest door contact
        best, depth = -1.0, None
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            seg = LineString([a, b])
            L = seg.intersection(doors).length
            if L > best + 1e-9:
                mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                big = 1000.0
                line = LineString([(mx, my - big), (mx, my + big)]) if a[1] == b[1] else LineString([(mx - big, my), (mx + big, my)])
                inner = g.intersection(line)
                depth = inner.length if not inner.is_empty else 0.0
                best = L
        if depth is not None and depth < P["min_depth"] - 1e-6:
            out["unit_below_min_depth"] += 1
    for h in halls:
        if not any(g.intersection(h).length >= P["door_contact"] - 1e-6 for g in units):
            out["hall_no_door"] += 1
    net = cores + halls
    if net:
        comp = list(range(len(net)))

        def find(i):
            while comp[i] != i:
                comp[i] = comp[comp[i]]
                i = comp[i]
            return i
        for i in range(len(net)):
            for j in range(i + 1, len(net)):
                if net[i].intersection(net[j]).length > 1e-6:
                    comp[find(i)] = find(j)
        if len({find(i) for i in range(len(net))}) > 1:
            out["hall_disconnected"] += 1
    return out


# ── public entry ─────────────────────────────────────────────────────────────

def _empty(status, reason, boundary, props, extra_metrics=None, rejects=None):
    m = {"requested_mix": {t: _r(props[t]) for t in TYPES} if props else {},
         "candidates_tried": 0, "candidates_survived": 0, "units": 0}
    if extra_metrics:
        m.update(extra_metrics)
    return {"status": status, "reason": reason, "boundary": boundary, "typology": [], "cores": [],
            "corridors": [], "units": [], "residual": [], "metrics": m,
            "rejects": rejects if rejects is not None else {k: 0 for k in REJECT_KEYS}}


def assemble_plate(boundary_m, mix, street_edges=(), north_deg=0.0):
    """Assemble core block, halls and units on an orthogonal outline (metres).

    north_deg: bearing of north, degrees clockwise from canvas up (plate library convention).
    Returns a dict with status "ok", "infeasible" or "unsupported". Raises ValueError for a
    malformed mix. Same input gives the same output."""
    props = _proportions(mix)
    streets = sorted({int(e) for e in (street_edges or ())})
    north = 0.0 if north_deg is None else float(north_deg)
    try:
        pts, err = _prepare(boundary_m)
    except (TypeError, ValueError, IndexError):
        pts, err = None, "boundary_m must be a list of [x, y]"
    raw = [[_r(float(p[0])), _r(float(p[1]))] for p in boundary_m] if boundary_m else []
    if pts is None:
        return _empty("unsupported", err, raw, props)
    boundary = [[_r(x), _r(y)] for x, y in pts]
    _ASSIGN_CACHE.clear()
    found, fail = _search(pts, props, streets, north)
    if found is None:
        return _empty("unsupported", fail["reason"], boundary, props)
    ctx = found["ctx"]
    allowed = tuple(t for t in TYPES if props[t] > 0)
    base_metrics = {"candidates_tried": found["tried"], "candidates_survived": found["survived"]}
    for key, (sk, var, combo, req, others) in found["best"]:
        sk = dict(sk)
        sk["others_wings"] = sorted(o[0] for o in others)
        res = _build(sk, var, combo, req, ctx, props)
        bad = verify_result(res, allowed)
        if sum(bad.values()) == 0:
            res["status"] = "ok"
            res["reason"] = ""
            res["metrics"].update(base_metrics)
            res["rejects"] = found["rejects"]
            return res
        for k, v in bad.items():
            if k in found["rejects"]:
                found["rejects"][k] += v
    reason = "no candidate satisfies the hard rules" if found["survived"] == 0 else "no verified candidate"
    res = _empty("infeasible", reason, boundary, props, base_metrics, found["rejects"])
    excluded = [t for t in TYPES if t not in allowed]
    if excluded:
        _explain_exclusion(res, ctx, excluded, allowed)
    return res


def _explain_exclusion(res, ctx, excluded, allowed):
    """Task B: say what the excluded types cost. Adds excluded_types and blocked_by_exclusion
    to an infeasible result, and one sentence to its reason when candidate units were blocked."""
    P = PARAMS
    areas = [a for a, _ in ctx.blocked.values()]
    fit = sorted({i for _, raw in ctx.blocked.values() for i, t in enumerate(TYPES) if raw >> i & 1 and t in excluded})
    info = {"count": len(areas),
            "min_area_m2": round(min(areas), 1) if areas else None,
            "max_area_m2": round(max(areas), 1) if areas else None}
    res["excluded_types"] = list(excluded)
    res["blocked_by_exclusion"] = info
    if not areas:
        return
    caps = {t: P["max_area_factor"] * P["types"][t]["target_area"] for t in allowed}
    top = max(allowed, key=lambda t: (caps[t], t))
    fit_names = ", ".join(TYPES[i] for i in fit)
    res["reason"] += (f"; no split without {', '.join(excluded)}: {info['count']} candidate units of "
                      f"{info['min_area_m2']:.1f}\u2013{info['max_area_m2']:.1f} m\u00b2 fit only {fit_names}; "
                      f"{top} allows up to {caps[top]:.1f} m\u00b2.")
