"""
topo_sensors.py — topology-only sensor registry for a retrieved/adapted plan.

Pure topology checks (reachability, adjacency-by-type, perimeter/daylight
presence) over a plan's room graph and room_edges. No geometry-stage concerns
(room size, overlap, clearance) — those belong to a later, placement-aware
sensor set. Reuses topo_signature's room labelling and category sets so
retrieval, this module, and the placed-arrangement sensors judge by one rule
set (see topo_signature.py's module docstring).

Public API:
    run_sensors(plan, room_edges, requirements=None, target_boundary=None) -> dict
"""

from packages.floorplan_core.topo_signature import (
    _label,
    BEDROOMS,
    COMMON,
    TRANSIT,
    DAYLIGHT_NEEDING,
)
# topo_signature's import (above) already put the fitting-prototype schema
# module on sys.path — reuse it rather than re-deriving mm sizing rules here.
from schema import MINIMUM_SIZES, SCALE  # noqa: E402

# ── rule table (data, not hardcoded) ─────────────────────────────────────────
# Each rule: a room category must have a door-adjacent neighbour in
# `providers`, either for "each" member of the category or "at_least_one".
ACCESS_RULES = {
    "bathroom": {"category": {"bathroom"}, "providers": TRANSIT, "quantifier": "at_least_one"},
    "bedroom": {"category": BEDROOMS, "providers": TRANSIT, "quantifier": "each"},
}


# ── context ───────────────────────────────────────────────────────────────────

def _build_ctx(plan, room_edges, requirements, target_boundary=None):
    rooms = plan.get("rooms") or []
    label_by_id = {r.get("id"): _label(r) for r in rooms}

    edges = plan.get("edges") or []
    has_connection_field = any("connection" in e for e in edges)

    door_adj = {}
    for e in edges:
        u, v = e.get("u"), e.get("v")
        if u is None or v is None:
            continue
        # ponytail: some plans omit `connection` entirely (older fixtures) —
        # fall back to treating every edge as passable rather than stranding
        # every room. Flagged in ctx so callers/tests can see it happened.
        passable = (e.get("connection") == 1) if has_connection_field else True
        if not passable:
            continue
        door_adj.setdefault(u, set()).add(v)
        door_adj.setdefault(v, set()).add(u)

    # entry room = room nearest the entrance midpoint (topo_signature's method)
    entry = None
    ent = plan.get("entrance")
    if ent and len(ent) >= 4:
        from shapely.geometry import Polygon as SPoly, Point as SPoint
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

    return {
        "plan": plan,
        "rooms": rooms,
        "label_by_id": label_by_id,
        "door_adj": door_adj,
        "has_connection_field": has_connection_field,
        "entry": entry,
        "room_edges": room_edges or {},
        "requirements": requirements,
        "target_boundary": target_boundary,
    }


def _ids_by_label(ctx, labels):
    return [rid for rid, lab in ctx["label_by_id"].items() if lab in labels]


def _external_edges(ctx, rid):
    return [e for e in ctx["room_edges"].get(str(rid), []) if e.get("edge_type") == "external"]


# ── checks ────────────────────────────────────────────────────────────────────

def check_reachability(ctx):
    entry = ctx["entry"]
    all_ids = list(ctx["label_by_id"].keys())
    if entry is None or not all_ids:
        return {"sensor": "reachability", "ok": False, "rooms": all_ids,
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
        f"unreachable from entry room {entry}: {unreachable}"
    return {"sensor": "reachability", "ok": ok, "rooms": unreachable, "detail": detail}


def _evaluate_access_rule(ctx, name, rule):
    subjects = _ids_by_label(ctx, rule["category"])
    providers = rule["providers"]
    has_provider = {
        rid: any(ctx["label_by_id"].get(nb) in providers for nb in ctx["door_adj"].get(rid, ()))
        for rid in subjects
    }

    if rule["quantifier"] == "each":
        failing = [rid for rid in subjects if not has_provider.get(rid)]
        ok = not failing
    elif rule["quantifier"] == "at_least_one":
        failing = subjects if subjects and not any(has_provider.values()) else []
        ok = not failing
    else:
        raise ValueError(f"unknown quantifier {rule['quantifier']!r}")

    return failing, ok


def check_access_typed(ctx):
    all_failing, all_detail = [], []
    ok = True
    for name, rule in ACCESS_RULES.items():
        failing, rule_ok = _evaluate_access_rule(ctx, name, rule)
        if not rule_ok:
            ok = False
            all_failing.extend(failing)
            all_detail.append(f"{name}: {failing} have no door-adjacent transit room")
    detail = "all access rules satisfied" if ok else "; ".join(all_detail)
    return {"sensor": "access_typed", "ok": ok, "rooms": all_failing, "detail": detail}


def check_bedroom_daylight(ctx):
    bedrooms = _ids_by_label(ctx, BEDROOMS)
    failing = [rid for rid in bedrooms if not _external_edges(ctx, rid)]
    ok = not failing
    detail = "every bedroom has an external room_edge" if ok else \
        f"bedrooms with no external room_edge: {failing}"
    return {"sensor": "bedroom_daylight", "ok": ok, "rooms": failing, "detail": detail}


def check_common_daylight(ctx):
    common = _ids_by_label(ctx, COMMON)
    served = any(_external_edges(ctx, rid) for rid in common)
    ok = served or not common
    rooms = [] if ok else common
    detail = "at least one common room has an external room_edge" if ok else \
        f"no common room has an external room_edge: {common}"
    return {"sensor": "common_daylight", "ok": ok, "rooms": rooms, "detail": detail}


def check_entrance_valid(ctx):
    entry = ctx["entry"]
    if entry is None:
        return {"sensor": "entrance_valid", "ok": False, "rooms": [],
                "detail": "no entry room found"}
    entry_label = ctx["label_by_id"].get(entry)
    ok = entry_label in TRANSIT
    rooms = [] if ok else [entry]
    detail = f"entry room {entry} ({entry_label}) is a transit room" if ok else \
        f"entrance opens into {entry_label} (room {entry}), not a transit room"
    return {"sensor": "entrance_valid", "ok": ok, "rooms": rooms, "detail": detail}


def check_boundary_capacity(ctx):
    """Per-TARGET-edge min-frontage capacity: each daylight room's minimum edge
    (schema.MINIMUM_SIZES) is demanded of its PRIMARY external edge (the
    target_edge_index of its largest-arc_frac external room_edges entry).
    Fails if any target edge's summed demand exceeds its own length."""
    boundary = ctx["target_boundary"]
    if not boundary or len(boundary) < 3:
        return {"sensor": "boundary_capacity", "ok": True, "rooms": [],
                "detail": "no target_boundary supplied — skipped"}

    demand = {}        # target_edge_index -> demand mm
    demand_rooms = {}  # target_edge_index -> [room_id, ...]
    no_min_edge = []
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
            no_min_edge.append(rid)  # no sizing rule for this room type — skip, don't crash
            continue
        demand[tidx] = demand.get(tidx, 0) + sizes["min_edge"]
        demand_rooms.setdefault(tidx, []).append(rid)

    if not demand:
        detail = "no external daylight demand against the target boundary"
        if no_min_edge:
            detail += f" (skipped, no MINIMUM_SIZES entry: {no_min_edge})"
        return {"sensor": "boundary_capacity", "ok": True, "rooms": [], "detail": detail}

    n = len(boundary)
    over_detail, rooms = [], []
    for tidx, mm in demand.items():
        a, b = boundary[tidx], boundary[(tidx + 1) % n]
        length_mm = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 * SCALE
        if mm > length_mm:
            over_detail.append(
                f"edge {tidx}: demand {mm}mm > length {round(length_mm)}mm "
                f"(rooms {demand_rooms[tidx]})"
            )
            rooms.extend(demand_rooms[tidx])

    ok = not over_detail
    detail = "all target edges have sufficient frontage capacity" if ok else \
        "over-subscribed: " + "; ".join(over_detail)
    if no_min_edge:
        detail += f" (skipped, no MINIMUM_SIZES entry: {no_min_edge})"
    return {"sensor": "boundary_capacity", "ok": ok, "rooms": rooms, "detail": detail}


# Registry order = report order.
REGISTRY = [
    check_reachability,
    check_access_typed,
    check_bedroom_daylight,
    check_common_daylight,
    check_entrance_valid,
    check_boundary_capacity,
]


def run_sensors(plan, room_edges, requirements=None, target_boundary=None):
    ctx = _build_ctx(plan, room_edges, requirements, target_boundary)
    checks = [check(ctx) for check in REGISTRY]
    passed = sum(1 for c in checks if c["ok"])
    return {"summary": {"passed": passed, "total": len(checks)}, "checks": checks}
