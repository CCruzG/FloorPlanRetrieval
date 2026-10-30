"""
moves.py — Stage-2 move registry for the agent harness (the Act layer).

Companion to topo_sensors.py (Observe) and generate.py (the locked Generator). A
move edits only the Spec — node assignment, edge connections, entrance — never
geometry; the Generator realises the edit and the sensors re-score. Design:
workspace `docs/moves-catalogue.md` (Move contract + SetConnection row).

Contract (by analogy with topo_sensors' sensor `evaluate` contract):
    name
    applicable(plan, sensors) -> bool              cheap guard: is there a
                                                    flagged condition this move
                                                    could address?
    propose(plan, sensors, ctx_inputs) -> [edit]   enumerate candidate Spec edits
    predict_effect(plan, edit, ctx_inputs) -> {summary, checks}
                                                    a near-symbolic estimate of
                                                    the resulting sensor change,
                                                    used to rank candidates before
                                                    a full rollout
    apply(spec, edit) -> spec                      write the edit into the Spec
                                                    (pure — does not mutate)

ctx_inputs = {room_edges, requirements, target_boundary, spec} — exactly what
run_sensors needs, threaded through so predict_effect can re-score without a
caller re-deriving them.

Public API:
    MOVES -> [Move, ...]           the open registry, like topo_sensors.REGISTRY
    SetConnectionMove
"""

import copy

from shapely.geometry import Polygon as SPoly

from packages.floorplan_core.topo_sensors import (
    _build_ctx, _passable, run_sensors, DOOR_MIN_MM,
)
from packages.floorplan_core.topo_signature import BEDROOMS

# "private" for the open-plan guard: bedrooms + bathroom (moves-catalogue.md —
# narrower than topo_signature.PRIVATE, which also includes storage; storage has
# no daylight/depth claim so SetConnection never needs to gate it here).
PRIVATE_FOR_OPEN = BEDROOMS | {"bathroom"}

_TYPE_TO_CONNECTION = {"wall": 0, "door": 1, "open": 2}

# Sensors SetConnection can address (moves-catalogue.md registry row).
_TARGET_SENSORS = ("access_typed", "reachability", "daylight")

_RANK_CAP = 10


# ── edge helpers (shared by propose / predict_effect / apply) ────────────────

def _edge_type(edge):
    """Canonical type ("wall"/"door"/"open") of an already-realised edge dict.
    separation/wall_type wins (SetConnection always writes it); else connection."""
    t = edge.get("separation") or edge.get("wall_type")
    if t in ("wall", "door", "open"):
        return t
    c = edge.get("connection")
    if c == 1:
        return "door"
    if c == 2:
        return "open"
    return "wall"


def _find_edge(plan, u, v):
    """The plan['edges'] entry between u and v, unordered — None if the pair is
    not realised-adjacent."""
    pair = {u, v}
    for e in plan.get("edges") or []:
        if {e.get("u"), e.get("v")} == pair:
            return e
    return None


def _set_edge_list(edges, u, v, new_type):
    """Mutate `edges` in place: set (or append) the u–v entry's connection +
    separation. Shared by predict_effect (on a deep-copied plan) and apply (on a
    deep-copied spec's adjacencies) — same list-of-edge-dict shape in both."""
    conn = _TYPE_TO_CONNECTION[new_type]
    pair = {u, v}
    for e in edges:
        if {e.get("u"), e.get("v")} == pair:
            e["connection"] = conn
            e["separation"] = new_type
            return
    edges.append({"u": u, "v": v, "connection": conn, "separation": new_type})


def _guard(ctx, u, v, new_type):
    """The four SetConnection guards (moves-catalogue.md), shared by propose
    (reject before offering) and apply-validation (reject before writing):
      - u, v realised-adjacent (an edge exists in plan['edges'])
      - no-op (edge already carries that type) rejected
      - door: shared wall length >= DOOR_MIN_MM, same mm scale as the sensors
      - open: neither room is private (bedroom/bathroom)
    Returns (ok, reason)."""
    if u is None or v is None or u == v:
        return False, "degenerate pair"
    edge = _find_edge(ctx["plan"], u, v)
    if edge is None:
        return False, "not realised-adjacent"
    if _edge_type(edge) == new_type:
        return False, "no-op"
    if new_type == "door":
        pa, pb = ctx["poly_by_id"].get(u), ctx["poly_by_id"].get(v)
        if not pa or not pb:
            return False, "missing room polygon"
        shared_mm = SPoly(pa).intersection(SPoly(pb)).length * ctx["mm"]
        if shared_mm < DOOR_MIN_MM:
            return False, "shared wall too narrow for a door"
    elif new_type == "open":
        lu, lv = ctx["label_by_id"].get(u), ctx["label_by_id"].get(v)
        if lu in PRIVATE_FOR_OPEN or lv in PRIVATE_FOR_OPEN:
            return False, "a private room cannot be opened"
    elif new_type != "wall":
        return False, f"unknown type {new_type!r}"
    return True, "ok"


def _make_edit(ctx, u, v, new_type, reason):
    ok, _why = _guard(ctx, u, v, new_type)
    if not ok:
        return None
    return {"move": "SetConnection", "u": u, "v": v, "type": new_type, "reason": reason}


# ── SetConnection ─────────────────────────────────────────────────────────────

def _applicable(sensors):
    checks = {c.get("sensor"): c for c in (sensors or {}).get("checks", [])}
    return any(not checks[n]["ok"] for n in _TARGET_SENSORS if n in checks)


def _propose(plan, sensors, ctx_inputs):
    ctx_inputs = ctx_inputs or {}
    ctx = _build_ctx(plan, ctx_inputs.get("room_edges"), ctx_inputs.get("requirements"),
                     ctx_inputs.get("target_boundary"), ctx_inputs.get("spec"))
    checks = {c.get("sensor"): c for c in (sensors or {}).get("checks", [])}
    candidates = []

    # door — access_typed's own door_candidates (the sensor's exact-effect input),
    # plus, for each room unreachable from entry, its wall-adjacent reachable
    # neighbours (a door there restores the missing path).
    access = checks.get("access_typed")
    if access and not access.get("ok", True):
        for pair in access.get("door_candidates") or []:
            rid, nb = pair[0], pair[1]
            edit = _make_edit(ctx, rid, nb, "door", "access_typed")
            if edit:
                candidates.append(edit)

    reach = checks.get("reachability")
    if reach and not reach.get("ok", True):
        all_ids = set(ctx["label_by_id"].keys())
        unreachable = set(reach.get("failing_rooms") or [])
        reachable_ids = all_ids - unreachable
        for rid in unreachable:
            for e in ctx["plan"].get("edges") or []:
                u, v = e.get("u"), e.get("v")
                if rid not in (u, v) or _passable(e, ctx["has_connection_field"]):
                    continue    # not this room's edge, or already passable (not a wall)
                nb = v if u == rid else u
                if nb not in reachable_ids:
                    continue
                edit = _make_edit(ctx, rid, nb, "door", "reachability")
                if edit:
                    candidates.append(edit)

    # open — daylight failing rooms, merged with an adjacent non-private room,
    # when the failing room itself is non-private. Not room_depth: depth is
    # geometric, and merging only adds floor behind the same window.
    daylight = checks.get("daylight")
    daylight_failing = set((daylight or {}).get("failing_rooms") or []) \
        if daylight and not daylight.get("ok", True) else set()
    for rid in daylight_failing:
        if ctx["label_by_id"].get(rid) in PRIVATE_FOR_OPEN:
            continue
        for e in ctx["plan"].get("edges") or []:
            u, v = e.get("u"), e.get("v")
            if rid not in (u, v):
                continue
            nb = v if u == rid else u
            if ctx["label_by_id"].get(nb) in PRIVATE_FOR_OPEN:
                continue
            edit = _make_edit(ctx, rid, nb, "open", "daylight")
            if edit:
                candidates.append(edit)

    # de-dup: the same (u, v, type) can surface from two reasons (e.g. a door
    # proposed for both access_typed and reachability) — keep the first seen.
    seen, uniq = set(), []
    for edit in candidates:
        key = (frozenset((edit["u"], edit["v"])), edit["type"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(edit)

    return _rank(plan, uniq, ctx_inputs)


def _rank(plan, candidates, ctx_inputs):
    """Rank by predicted improvement: fewer failing hard sensors first, then the
    larger summed margin. predict_effect is exact (geometry is unchanged), so this
    ranking costs no rollout — only a sensor re-score per candidate. Capped at 10."""
    scored = []
    for edit in candidates:
        pred = _predict_effect(plan, edit, ctx_inputs)
        pred_checks = pred.get("checks", [])
        n_failing_hard = sum(1 for c in pred_checks if c["kind"] == "hard" and not c["ok"])
        margin_sum = sum(c["margin"] for c in pred_checks)
        scored.append((n_failing_hard, -margin_sum, edit))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [edit for _, _, edit in scored[:_RANK_CAP]]


def _predict_effect(plan, edit, ctx_inputs):
    """Exact: a SetConnection edit changes only an edge's connection/separation,
    never geometry, so re-scoring a deep copy of `plan` with the edit applied is
    the real post-move sensor result, not an estimate — apply + generate() lands
    on the identical numbers (checked by tools/gate_set_connection.py)."""
    new_plan = copy.deepcopy(plan)
    _set_edge_list(new_plan.setdefault("edges", []), edit["u"], edit["v"], edit["type"])
    ctx_inputs = ctx_inputs or {}
    return run_sensors(new_plan, ctx_inputs.get("room_edges"),
                       requirements=ctx_inputs.get("requirements"),
                       target_boundary=ctx_inputs.get("target_boundary"),
                       spec=ctx_inputs.get("spec"), layer="geometry")


def _apply(spec, edit):
    """Pure: deep-copy the Spec and set the u–v edge's connection + separation in
    spec['adjacencies'] (the room<->room graph the Generator reads), adding the
    edge if the pair was not already one of the plan's carried entries."""
    new_spec = copy.deepcopy(spec)
    _set_edge_list(new_spec.setdefault("adjacencies", []), edit["u"], edit["v"], edit["type"])
    return new_spec


# ── move contract / registry ──────────────────────────────────────────────────

class Move:
    """Base contract every move satisfies (moves-catalogue.md 'Move contract')."""
    name = "Move"

    def applicable(self, plan, sensors):
        raise NotImplementedError

    def propose(self, plan, sensors, ctx_inputs):
        raise NotImplementedError

    def predict_effect(self, plan, edit, ctx_inputs):
        raise NotImplementedError

    def apply(self, spec, edit):
        raise NotImplementedError


class SetConnectionMove(Move):
    """Flip an edge's connection between wall / door / open. The three-valued-edge
    move declared in moves-catalogue.md, replacing the earlier OpenPartition /
    CreateConnection sketches now that edges carry `separation`."""
    name = "SetConnection"

    def applicable(self, plan, sensors):
        return _applicable(sensors)

    def propose(self, plan, sensors, ctx_inputs):
        return _propose(plan, sensors, ctx_inputs)

    def predict_effect(self, plan, edit, ctx_inputs):
        return _predict_effect(plan, edit, ctx_inputs)

    def apply(self, spec, edit):
        return _apply(spec, edit)


# Open registry, like topo_sensors.REGISTRY — the set of moves grows, the
# contract above never changes.
MOVES = [SetConnectionMove()]
