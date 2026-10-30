"""
topo_sensors.py — Stage-1 sensor registry for a retrieved / generated plan.

Sensors are the central development object of the harness (build plan §2). Each
sensor has a fixed contract:

    kind:  "hard" | "soft"          hard = a compliance gate; soft = amenity score
    evaluate(ctx) -> {ok, failing_rooms, margin, detail}
        ok:            bool
        failing_rooms: [room_id, ...]
        margin:        float — signed distance to the pass boundary (0 = at it,
                       negative = failing by this much). The continuous signal the
                       Objective needs to tell whether a move got closer.

The registry is open: the interface never changes, the set of sensors grows. The
regulatory basis and thresholds live in `projects/project-44/docs/bads-sensor-model.md`
(Better Apartment Design Standards, Victoria). Topology checks read the room graph
and the derived room↔boundary layer; geometric checks (min size, room depth) read
the Stage-0 polygons. Category sets are reused from topo_signature so retrieval,
these sensors, and the generator judge by one rule set.

Public API:
    run_sensors(plan, room_edges, requirements=None, target_boundary=None,
                spec=None) -> {summary, checks}
"""

import math

from shapely.geometry import Polygon as SPoly, Point as SPoint
from shapely.ops import unary_union

from packages.floorplan_core.topo_signature import (
    _label,
    BEDROOMS,
    COMMON,
    TRANSIT,
    DAYLIGHT_NEEDING,
    WINDOW_MIN_WIDTH_M,   # shared with the raycast that drops narrower contacts
)
# topo_signature's import (above) already put the fitting-prototype schema
# module on sys.path — reuse it rather than re-deriving mm sizing rules here.
from schema import MINIMUM_SIZES, SCALE  # noqa: E402

# Habitable rooms (BADS): bedrooms + living/dining/kitchen. These need daylight
# (D26) and are subject to the depth limit (D25). Distinct from topo_signature's
# DAYLIGHT_NEEDING (which excludes kitchen/dining and drives boundary_capacity's
# frontage demand): a kitchen needs light, but via its open-plan component, not by
# demanding its own boundary frontage. See bads-sensor-model.md §3.
HABITABLE = BEDROOMS | COMMON

# ── thresholds (data; see bads-sensor-model.md) ───────────────────────────────
CEILING_MM = 2700               # assumed ceiling height (BADS depth is ×ceiling)
DEPTH_SINGLE_MM = 2.5 * CEILING_MM      # D25 single-aspect habitable room: 6750 mm
DEPTH_OPENPLAN_LDK_MM = 9000            # D25 open-plan living+dining+kitchen: 9 m
USABLE_WIDTH_MM = 900           # D24 usable floor: narrower parts are not usable area (door / corridor floor; working value, no BADS source checked)

# ── access rule table (data, not hardcoded) ───────────────────────────────────
ACCESS_RULES = {
    "bathroom": {"category": {"bathroom"}, "providers": TRANSIT, "quantifier": "at_least_one"},
    "bedroom": {"category": BEDROOMS, "providers": TRANSIT, "quantifier": "each"},
}


# ── geometry helpers ──────────────────────────────────────────────────────────

def _poly_area(poly):
    """Shoelace area of a simple polygon [[x, y], ...] (sign-agnostic)."""
    n = len(poly)
    return abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
                   for i in range(n))) / 2.0


def _bbox_min_side(poly):
    """Shorter side of the axis-aligned bounding box — a cheap proxy for a room's
    minimum internal dimension (D24 min_edge). Rooms here are grid-derived, so
    axis-aligned bbox is faithful; a rotated-rect would be the refinement."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(max(xs) - min(xs), max(ys) - min(ys))


def _usable_parts(poly, width):
    """The parts of a room a width×width square can reach: a morphological opening
    (erode by width/2, dilate back) with mitred joins, exact for the axis-aligned
    rooms the grid realiser makes. Arms and necks narrower than `width` drop out,
    which the bbox proxy cannot see (an L with a 0.6 m arm has a 1.5 m bbox).
    Returns the opened pieces, largest first ([] = no square fits anywhere)."""
    g = SPoly(poly)
    if not g.is_valid:
        g = g.buffer(0)
    if width <= 0:
        return [g]
    r = width / 2.0
    opened = g.buffer(-r, join_style=2).buffer(r, join_style=2).intersection(g)
    tiny = 1e-9 * max(g.area, 1e-12)
    parts = [p for p in getattr(opened, "geoms", [opened])
             if p.geom_type == "Polygon" and p.area > tiny]
    return sorted(parts, key=lambda p: p.area, reverse=True)


def _max_square_side(poly, hi):
    """Largest square side that fits in the room, by bisection on the opening
    (only called when the room's min_edge does not fit, for a continuous margin)."""
    lo = 0.0
    for _ in range(20):
        mid = (lo + hi) / 2.0
        if _usable_parts(poly, mid):
            lo = mid
        else:
            hi = mid
    return lo


def _inward_normal(geom, a, b):
    """Unit normal of segment a→b pointing into `geom` (tested a hair off the
    midpoint), or None for a degenerate segment / no side inside."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    if L < 1e-12:
        return None
    nx, ny = -dy / L, dx / L
    mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
    h = 1e-4 * L
    if geom.contains(SPoint(mx + nx * h, my + ny * h)):
        return (nx, ny)
    if geom.contains(SPoint(mx - nx * h, my - ny * h)):
        return (-nx, -ny)
    return None


def _perp_dist(p, a, b):
    """Perpendicular distance from point p to the infinite line through a, b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    if L == 0:
        return 0.0
    return abs(dx * (a[1] - p[1]) - dy * (a[0] - p[0])) / L


def _passable(e, has_connection_field):
    """Is this edge open for movement — door or open, not a wall? `separation`/
    `wall_type` is authoritative when present (SetConnection writes it); otherwise
    fall back to `connection` (1=door, 2=open). has_connection_field is a PLAN-WIDE
    flag: when no edge anywhere in the plan carries `connection` (older fixtures),
    every edge is treated as passable rather than stranding every room — the
    ponytail the old single-field check also had."""
    t = e.get("separation") or e.get("wall_type")
    if t in ("door", "open"):
        return True
    if t == "wall":
        return False
    if has_connection_field:
        return e.get("connection") in (1, 2)
    return True


# ── context ───────────────────────────────────────────────────────────────────

def _build_ctx(plan, room_edges, requirements, target_boundary=None, spec=None):
    rooms = plan.get("rooms") or []
    label_by_id = {r.get("id"): _label(r) for r in rooms}
    poly_by_id = {r.get("id"): r.get("polygon") for r in rooms
                  if r.get("polygon") and len(r.get("polygon")) >= 3}

    edges = plan.get("edges") or []
    has_connection_field = any("connection" in e for e in edges)

    door_adj = {}
    for e in edges:
        u, v = e.get("u"), e.get("v")
        if u is None or v is None:
            continue
        if not _passable(e, has_connection_field):
            continue
        door_adj.setdefault(u, set()).add(v)
        door_adj.setdefault(v, set()).add(u)

    # entry room = room nearest the entrance midpoint (topo_signature's method)
    entry = None
    ent = plan.get("entrance")
    if ent and len(ent) >= 4:
        ent_pt = SPoint((ent[0] + ent[2]) / 2.0, (ent[1] + ent[3]) / 2.0)
        best_d = float("inf")
        for r in rooms:
            poly = r.get("polygon")
            if not poly or len(poly) < 3:
                continue
            try:
                d = ent_pt.distance(SPoly(poly))
            except Exception:
                continue
            if d < best_d:
                entry, best_d = r.get("id"), d

    # boundary-units -> mm, for the geometric sensors. Follows boundary_capacity's
    # convention: coords are in normalised units, ×SCALE = mm (schema.SCALE).
    mm = (spec or {}).get("mm_per_unit")
    if not mm or mm <= 0:
        mm = SCALE

    ctx = {
        "plan": plan,
        "rooms": rooms,
        "label_by_id": label_by_id,
        "poly_by_id": poly_by_id,
        "door_adj": door_adj,
        "has_connection_field": has_connection_field,
        "entry": entry,
        "room_edges": room_edges or {},
        "requirements": requirements,
        "target_boundary": target_boundary,
        "spec": spec,
        "mm": mm,
    }
    ctx["components"] = _open_plan_components(ctx)
    return ctx


def _nm(ctx, rid):
    """One display convention for a room in every sensor detail: `r_2 (kitchen)`.
    failing_rooms stays a list of bare ids (data for moves / the overlay)."""
    return f"{rid} ({ctx['label_by_id'].get(rid, '?')})"


def _nms(ctx, rids):
    return ", ".join(_nm(ctx, r) for r in sorted(rids, key=str)) or "none"


def _ids_by_label(ctx, labels):
    return [rid for rid, lab in ctx["label_by_id"].items() if lab in labels]


def _external_edges(ctx, rid):
    return [e for e in ctx["room_edges"].get(str(rid), []) if e.get("edge_type") == "external"]


def _is_open_edge(ctx, e, u, v):
    """Is this a wall-less (open-plan) edge? Only when it says so: `separation`/
    `wall_type` "open", `open: True` or connection 2. A door is a wall with a hole
    in it, not open plan — even between living, dining and kitchen. (An earlier
    COMMON↔COMMON inference merged door-linked common rooms, which let a windowless
    kitchen borrow the living room's daylight and measured depth across the doors;
    dropped 2026-10-30 now that SetConnection writes explicit open edges.) See
    bads-sensor-model.md §3."""
    t = e.get("separation") or e.get("wall_type")
    if t in ("wall", "door", "open"):
        return t == "open"
    return e.get("open") is True or e.get("connection") == 2


def _open_plan_components(ctx):
    """Connected components of rooms joined by `open` (wall-less) edges. The unit
    the daylight (D26) and depth (D25) sensors judge: an open-plan living+kitchen
    is one habitable room sharing one window. Returns a list of sets of room ids."""
    rooms = list(ctx["label_by_id"].keys())
    parent = {r: r for r in rooms}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in ctx["plan"].get("edges") or []:
        u, v = e.get("u"), e.get("v")
        if u in parent and v in parent and _is_open_edge(ctx, e, u, v):
            union(u, v)

    comps = {}
    for r in rooms:
        comps.setdefault(find(r), set()).add(r)
    return list(comps.values())


# ── topology sensors (hard) ─────────────────────────────────────────────────────

def check_reachability(ctx):
    entry = ctx["entry"]
    all_ids = list(ctx["label_by_id"].keys())
    if entry is None or not all_ids:
        return {"ok": False, "failing_rooms": all_ids, "margin": -len(all_ids),
                "detail": "no entry room found"}
    seen = {entry}
    queue = [entry]
    while queue:
        cur = queue.pop()
        for nb in ctx["door_adj"].get(cur, ()):
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    unreachable = [rid for rid in all_ids if rid not in seen]
    ok = not unreachable
    detail = "all rooms reachable from entry" if ok else \
        f"unreachable from entry {_nm(ctx, entry)}: {_nms(ctx, unreachable)}"
    return {"ok": ok, "failing_rooms": unreachable, "margin": -float(len(unreachable)),
            "detail": detail}


def _evaluate_access_rule(ctx, rule):
    subjects = _ids_by_label(ctx, rule["category"])
    providers = rule["providers"]
    has_provider = {
        rid: any(ctx["label_by_id"].get(nb) in providers for nb in ctx["door_adj"].get(rid, ()))
        for rid in subjects
    }
    if rule["quantifier"] == "each":
        failing = [rid for rid in subjects if not has_provider.get(rid)]
    elif rule["quantifier"] == "at_least_one":
        failing = subjects if subjects and not any(has_provider.values()) else []
    else:
        raise ValueError(f"unknown quantifier {rule['quantifier']!r}")
    return failing, not failing


DOOR_MIN_MM = 900   # a shared wall narrower than a door cannot take one


def _door_candidates(ctx, rid, providers):
    """Provider rooms that share a WALL with `rid` wide enough for a door — where
    SetConnection(rid, nb, door) could restore access. Reads the realised (wall)
    edges — excluding any edge already door or open, not just connection==1, so an
    `open` edge (separation/connection==2) is not re-offered as a door candidate —
    then checks the shared wall length on the polygons."""
    out = []
    pa = ctx["poly_by_id"].get(rid)
    for e in ctx["plan"].get("edges") or []:
        u, v = e.get("u"), e.get("v")
        if rid not in (u, v) or _passable(e, ctx["has_connection_field"]):
            continue
        nb = v if u == rid else u
        if ctx["label_by_id"].get(nb) not in providers:
            continue
        pb = ctx["poly_by_id"].get(nb)
        if pa and pb:
            shared_mm = SPoly(pa).intersection(SPoly(pb)).length * ctx["mm"]
            if shared_mm < DOOR_MIN_MM:
                continue
        out.append(nb)
    return sorted(set(out), key=str)


def check_access_typed(ctx):
    """Typed access: each subject room needs a DOOR to a provider (transit) room.
    Strict — it reads doors, not adjacency. For each failing room it also names
    the wall-adjacent providers a door could go through (door_candidates), the
    exact-effect repair for the SetConnection move."""
    all_failing, all_detail, candidates = [], [], []
    ok = True
    for name, rule in ACCESS_RULES.items():
        failing, rule_ok = _evaluate_access_rule(ctx, rule)
        if not rule_ok:
            ok = False
            all_failing.extend(failing)
            parts = []
            for rid in failing:
                cands = _door_candidates(ctx, rid, rule["providers"])
                candidates.extend([[rid, nb] for nb in cands])
                parts.append(f"{_nm(ctx, rid)} — no door to a transit room"
                             + (f"; door possible via {_nms(ctx, cands)}" if cands
                                else "; no wall-adjacent transit room"))
            all_detail.append(f"{name}: " + "; ".join(parts))
    detail = "all access rules satisfied" if ok else "; ".join(all_detail)
    return {"ok": ok, "failing_rooms": all_failing, "margin": -float(len(all_failing)),
            "detail": detail, "door_candidates": candidates}


def check_entrance_valid(ctx):
    entry = ctx["entry"]
    if entry is None:
        return {"ok": False, "failing_rooms": [], "margin": -1.0,
                "detail": "no entry room found"}
    entry_label = ctx["label_by_id"].get(entry)
    ok = entry_label in TRANSIT
    detail = f"entry room {_nm(ctx, entry)} is a transit room" if ok else \
        f"entrance opens into {_nm(ctx, entry)}, not a transit room"
    return {"ok": ok, "failing_rooms": [] if ok else [entry],
            "margin": 0.0 if ok else -1.0, "detail": detail}


# ── daylight (hard, D26) — per open-plan component ──────────────────────────────

def check_daylight(ctx):
    """D26: every habitable room has a window in an external wall. Judged per
    open-plan component — a lit living room lights an open kitchen in the same
    component. Borrowed light through a *door* (a non-open edge) does not count:
    such rooms are separate components and need their own window. margin = the
    worst component's widest external contact minus the min window width (m)."""
    failing, details, worst = [], [], None
    for comp in ctx["components"]:
        habitable = [r for r in comp if ctx["label_by_id"].get(r) in HABITABLE]
        if not habitable:
            continue
        best_w = max(
            (ed.get("width_m", 0.0) for r in comp for ed in _external_edges(ctx, r)),
            default=0.0)
        m = best_w - WINDOW_MIN_WIDTH_M
        worst = m if worst is None else min(worst, m)
        if best_w < WINDOW_MIN_WIDTH_M:
            failing.extend(habitable)
            details.append(f"{_nms(ctx, comp)} — no external window ≥ {WINDOW_MIN_WIDTH_M} m")
    ok = not failing
    detail = "every habitable room (or its open-plan component) has an external window" \
        if ok else "; ".join(details)
    return {"ok": ok, "failing_rooms": failing,
            "margin": 0.0 if worst is None else worst, "detail": detail}


# ── room depth (hard, D25) — per open-plan component ────────────────────────────

def check_room_depth(ctx):
    """D25: single-aspect habitable room depth ≤ 2.5 × ceiling height (6.75 m at a
    2.7 m ceiling); an open-plan living+dining+kitchen component may reach 9 m.
    Judged WALL BY WALL on the merged component outline: a wall OPPOSES a window
    when the two are parallel and face each other (inward normals antiparallel,
    the wall on the room side of the window). A wall's depth = the perpendicular
    distance to its nearest opposing window; the component passes when every wall
    that opposes a window is within the limit (one near window is enough — several
    windows are several checks, not one). Walls opposing no window carry no depth
    claim. Windows = external contacts ≥ WINDOW_MIN_WIDTH_M. One-window rules
    failed both ways: the largest arc_frac picked a 1.0 m notch, the nearest
    window measured sideways from a step in the facade. margin = limit − the
    worst wall's depth (mm)."""
    boundary = ctx["target_boundary"]
    mm = ctx["mm"]
    if not boundary or len(boundary) < 3:
        return {"ok": True, "failing_rooms": [], "margin": 0.0,
                "detail": "no target_boundary supplied — skipped", "skipped": True}
    n = len(boundary)
    bpoly = SPoly(boundary)
    if not bpoly.is_valid:
        bpoly = bpoly.buffer(0)
    failing, details, worst = [], [], None
    for comp in ctx["components"]:
        habitable = [r for r in comp if ctx["label_by_id"].get(r) in HABITABLE]
        if not habitable:
            continue
        win_idx = {ed["target_edge_index"] % n for r in comp for ed in _external_edges(ctx, r)
                   if ed.get("target_edge_index") is not None
                   and ed.get("width_m", 0.0) >= WINDOW_MIN_WIDTH_M}
        if not win_idx:
            continue  # unlit — the daylight sensor owns this failure
        polys = [SPoly(ctx["poly_by_id"][r]) for r in comp if r in ctx["poly_by_id"]]
        if not polys:
            continue
        merged = unary_union(polys)
        windows = []    # (point on window line, inward unit normal)
        for t in win_idx:
            a, b = boundary[t], boundary[(t + 1) % n]
            nrm = _inward_normal(bpoly, a, b)
            if nrm:
                windows.append((a, nrm))
        depth_mm = 0.0
        for g in getattr(merged, "geoms", [merged]):
            ring = list(g.exterior.coords)
            for k in range(len(ring) - 1):
                a, b = ring[k], ring[k + 1]
                wn = _inward_normal(g, a, b)
                if not wn:
                    continue
                mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
                # a wall on a window line IS a window wall: it is lit, so it is not
                # held to a window across the room (dual aspect)
                if any(wn[0] * nx + wn[1] * ny > 0.99
                       and abs((mx - p[0]) * nx + (my - p[1]) * ny) < 1e-6
                       for p, (nx, ny) in windows):
                    continue
                ds = [(mx - p[0]) * nx + (my - p[1]) * ny
                      for p, (nx, ny) in windows
                      if wn[0] * nx + wn[1] * ny < -0.99]       # facing each other
                ds = [d for d in ds if d > 1e-9]                # wall on the room side
                if ds:
                    depth_mm = max(depth_mm, min(ds) * mm)
        labels = {ctx["label_by_id"].get(r) for r in comp}
        is_ldk = ("kitchen" in labels) and bool(labels & {"living", "living_room", "dining"})
        limit = DEPTH_OPENPLAN_LDK_MM if is_ldk else DEPTH_SINGLE_MM
        m = limit - depth_mm
        worst = m if worst is None else min(worst, m)
        if depth_mm > limit:
            failing.extend(habitable)
            details.append(f"{_nms(ctx, comp)} — a wall is {round(depth_mm)}mm from its "
                           f"nearest opposing window > limit {round(limit)}mm")
    ok = not failing
    detail = "all habitable rooms within the depth limit" if ok else "; ".join(details)
    return {"ok": ok, "failing_rooms": failing,
            "margin": 0.0 if worst is None else worst, "detail": detail}


# ── boundary frontage capacity (hard) ───────────────────────────────────────────

def check_boundary_capacity(ctx):
    """Per-TARGET-edge min-frontage capacity: each daylight room's minimum edge
    (schema.MINIMUM_SIZES) is demanded of its primary external edge. Fails if any
    target edge's summed demand exceeds its own length. margin = the tightest
    (length − demand) across demanded edges, in mm."""
    boundary = ctx["target_boundary"]
    if not boundary or len(boundary) < 3:
        return {"ok": True, "failing_rooms": [], "margin": 0.0,
                "detail": "no target_boundary supplied — skipped", "skipped": True}
    demand, demand_rooms, no_min_edge = {}, {}, []
    for rid in _ids_by_label(ctx, DAYLIGHT_NEEDING):
        ext = _external_edges(ctx, rid)
        if not ext:
            continue
        primary = max(ext, key=lambda e: e.get("arc_frac", 0))
        tidx = primary.get("target_edge_index")
        if tidx is None:
            continue
        sizes = MINIMUM_SIZES.get(ctx["label_by_id"].get(rid))
        if sizes is None:
            no_min_edge.append(rid)
            continue
        demand[tidx] = demand.get(tidx, 0) + sizes["min_edge"]
        demand_rooms.setdefault(tidx, []).append(rid)

    if not demand:
        detail = "no external daylight demand against the target boundary"
        if no_min_edge:
            detail += f" (skipped, no MINIMUM_SIZES entry: {_nms(ctx, no_min_edge)})"
        return {"ok": True, "failing_rooms": [], "margin": 0.0, "detail": detail}

    n = len(boundary)
    over_detail, rooms, worst = [], [], None
    for tidx, mm_demand in demand.items():
        a, b = boundary[tidx], boundary[(tidx + 1) % n]
        length_mm = math.hypot(b[0] - a[0], b[1] - a[1]) * SCALE
        slack = length_mm - mm_demand
        worst = slack if worst is None else min(worst, slack)
        if mm_demand > length_mm:
            over_detail.append(f"edge {tidx}: demand {mm_demand}mm > length "
                               f"{round(length_mm)}mm ({_nms(ctx, demand_rooms[tidx])})")
            rooms.extend(demand_rooms[tidx])
    ok = not over_detail
    detail = "all target edges have sufficient frontage capacity" if ok else \
        "over-subscribed: " + "; ".join(over_detail)
    if no_min_edge:
        detail += f" (skipped, no MINIMUM_SIZES entry: {_nms(ctx, no_min_edge)})"
    return {"ok": ok, "failing_rooms": rooms,
            "margin": 0.0 if worst is None else worst, "detail": detail}


# ── min room size (hard, D24) — geometric ───────────────────────────────────────

def check_min_room_size(ctx):
    """D24: each room meets its minimum internal dimension and area (MINIMUM_SIZES).
    Two tests, two thresholds, neither on the bbox:
      core — a min_edge square fits somewhere in the room (BADS sets a minimum
             rectangle; its area is that rectangle's area);
      area — the room's USABLE area meets min_area. Usable = reachable by a
             USABLE_WIDTH_MM square (furniture / circulation), one threshold for
             every room — a 1.2 m arm off a bedroom is usable, a 0.6 m one is not.
             The usable piece that holds the core counts (the largest when no core
             fits). Parts narrower than USABLE_WIDTH_MM are reported as below min
             width; a usable footprint that splits into pieces (a neck narrower
             than USABLE_WIDTH_MM) is reported as pinched.
    margin = the worst room's shortfall, min(edge − min_edge, area − min_area), in
    mm / mm²."""
    if not ctx["poly_by_id"]:
        return {"ok": True, "failing_rooms": [], "margin": 0.0,
                "detail": "no room polygons — skipped", "skipped": True}
    mm = ctx["mm"]
    failing, details, skipped, worst = [], [], [], None
    pinched, slivers = [], []
    # 1 mm of slack on each square: the solver places spans at exactly min_edge,
    # and float error must not read as a violation
    use_units = (USABLE_WIDTH_MM - 1.0) / mm
    for rid, poly in ctx["poly_by_id"].items():
        req = MINIMUM_SIZES.get(ctx["label_by_id"].get(rid))
        if not req:
            skipped.append(rid)
            continue
        area_mm2 = _poly_area(poly) * mm * mm
        core = _usable_parts(poly, max(req["min_edge"] - 1.0, 0.0) / mm)
        if core:
            edge_mm = max(_bbox_min_side(list(core[0].exterior.coords)) * mm,
                          req["min_edge"])
        else:
            edge_mm = _max_square_side(poly, _bbox_min_side(poly)) * mm
        use = _usable_parts(poly, use_units)
        home = next((p for p in use if core and p.intersects(core[0])),
                    use[0] if use else None)
        usable_mm2 = home.area * mm * mm if home is not None else 0.0
        dropped_mm2 = area_mm2 - sum(p.area for p in use) * mm * mm
        if len(use) > 1:
            pinched.append(rid)
        if dropped_mm2 > 1e4:
            slivers.append(f"{_nm(ctx, rid)} {dropped_mm2 / 1e6:.2f}m²")
        m_edge = edge_mm - req["min_edge"]
        m_area = usable_mm2 - req["area"]
        # normalise the two margins to mm so one scalar is comparable: area margin
        # expressed as the edge it would add at the room's min width.
        m = min(m_edge, m_area / max(req["min_edge"], 1))
        worst = m if worst is None else min(worst, m)
        # 0.01 m² area tolerance (float error)
        if m_edge < -1.0 or m_area < -1e4:
            failing.append(rid)
            details.append(f"{_nm(ctx, rid)}: edge {round(edge_mm)}mm"
                           f"/{req['min_edge']}, usable area {usable_mm2 / 1e6:.2f}m²"
                           f"/{req['area'] / 1e6:.2f} (of {area_mm2 / 1e6:.2f})")
    ok = not failing
    detail = "all rooms meet minimum size" if ok else "undersized: " + "; ".join(details)
    if pinched:
        detail += f" · pinched: {_nms(ctx, pinched)}"
    if slivers:
        detail += f" · below min width: {', '.join(slivers)}"
    if skipped:
        detail += f" (no MINIMUM_SIZES entry, skipped: {_nms(ctx, skipped)})"
    return {"ok": ok, "failing_rooms": failing,
            "margin": 0.0 if worst is None else worst, "detail": detail,
            "pinched_rooms": pinched}


# ── node-in-room (soft, diagnostic) ─────────────────────────────────────────────

def check_node_in_room(ctx):
    """Soft diagnostic: how many rooms did not naturally contain their node after
    the Stage-0 solve (report.node_relocated). Ranks below placement and geometric
    compliance by design — the marker is relocated, the plan stands. margin =
    −(count). Skipped when the plan carries no Stage-0 report."""
    report = ctx["plan"].get("report")
    if not report:
        return {"ok": True, "failing_rooms": [], "margin": 0.0,
                "detail": "no Stage-0 report — skipped", "skipped": True}
    relocated = report.get("node_relocated") or []
    ok = not relocated
    detail = "every node sits inside its own room" if ok else \
        f"{len(relocated)} room(s) did not contain their node: {_nms(ctx, relocated)}"
    return {"ok": ok, "failing_rooms": list(relocated), "margin": -float(len(relocated)),
            "detail": detail}


# ── registry: (name, fn, kind). Order = report order. ───────────────────────────
# Order = hierarchy of hardness (most binding first). boundary_capacity leads: it
# is the one constraint a move inside the unit cannot relax — the only release is to
# change a boundary edge's TYPE (e.g. inter-tenancy → exterior), which is a decision
# at the plate level, above the unit harness. Everything below it is repairable by a
# move that rearranges or re-dimensions rooms.
REGISTRY = [
    ("boundary_capacity", check_boundary_capacity, "hard"),
    ("reachability",     check_reachability,     "hard"),
    ("access_typed",     check_access_typed,     "hard"),
    ("entrance_valid",   check_entrance_valid,   "hard"),
    ("daylight",         check_daylight,         "hard"),
    ("room_depth",       check_room_depth,       "hard"),
    ("min_room_size",    check_min_room_size,    "hard"),
    ("node_in_room",     check_node_in_room,     "soft"),
]


# Sensors that read realised room polygons. At the topology stage (a retrieved
# plan, before /generate) the polygons are the SOURCE plan's, not geometry in the
# target boundary, so these say nothing about the design yet — layer="topology"
# skips them.
GEOMETRIC_SENSORS = {"room_depth", "min_room_size", "node_in_room"}
# Pre-geometry feasibility questions, asked of the retrieved plan only. Once
# geometry exists the realised result is judged directly (D26 daylight, D24 size):
# boundary_capacity pins each daylight room's min width to ONE edge, so a room
# spread along a stepped facade read as over-subscribing it — layer="geometry"
# skips these.
TOPOLOGY_ONLY_SENSORS = {"boundary_capacity"}


def run_sensors(plan, room_edges, requirements=None, target_boundary=None, spec=None,
                layer=None):
    """Run the scored sensor registry. Returns {summary, checks}. Each check:
    {sensor, kind, ok, failing_rooms, rooms (legacy alias), margin, detail}.
    summary: {passed, total, gates_pass (all hard ok), soft_score (Σ soft margins)}.
    layer="topology" (a retrieved plan) skips GEOMETRIC_SENSORS; layer="geometry"
    (a generated plan) skips TOPOLOGY_ONLY_SENSORS; None runs everything."""
    ctx = _build_ctx(plan, room_edges, requirements, target_boundary, spec)
    checks = []
    for name, fn, kind in REGISTRY:
        if layer == "topology" and name in GEOMETRIC_SENSORS:
            continue
        if layer == "geometry" and name in TOPOLOGY_ONLY_SENSORS:
            continue
        res = fn(ctx)
        res["sensor"] = name
        res["kind"] = kind
        res.setdefault("failing_rooms", [])
        res["rooms"] = res["failing_rooms"]   # legacy alias for the frontend
        checks.append(res)
    passed = sum(1 for c in checks if c["ok"])
    gates_pass = all(c["ok"] for c in checks if c["kind"] == "hard")
    soft_score = sum(c["margin"] for c in checks if c["kind"] == "soft")
    return {"summary": {"passed": passed, "total": len(checks),
                        "gates_pass": gates_pass, "soft_score": round(soft_score, 4)},
            "checks": checks}
