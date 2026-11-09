"""Plate composition record (ARCHITECTURE.md, "Plate composition"). Pure functions, no Flask.

A ComposedPlate holds the assembler output plus one Slot per unit. A slot records what
was asked of retrieval (requirements), what came back (candidates), which candidate was
used (chosen), the generated plan, the generator options, the sensor result and a status.

Units: assembler output is in plate metres. Slot plans are kept in app units (normalised,
1 unit = METRES_PER_UNIT metres), the units /retrieve and /generate use. to_export()
converts plan geometry to plate metres.

Status transitions: pending -> generated | failed; generated | failed | edited -> edited.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

METRES_PER_UNIT = 18.0

PENDING, GENERATED, FAILED, EDITED = "pending", "generated", "failed", "edited"
STATUSES = (PENDING, GENERATED, FAILED, EDITED)
_ALLOWED = {
    PENDING: {GENERATED, FAILED},
    GENERATED: {EDITED},
    FAILED: {EDITED},
    EDITED: {EDITED},
}

# Generator options used for every unit in a first pass (the app defaults, with the
# source grid and node expansion on; step-0 core placement off).
DEFAULT_OPTIONS = {
    "area_lift": True,
    "core_block": True,
    "core_placement": False,
    "expand_nodes": True,
    "source_grid": True,
}
TOPN = 5


@dataclass
class Slot:
    unit_id: str
    requirements: Dict[str, Any]
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    chosen: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None
    options: Dict[str, Any] = field(default_factory=dict)
    sensors: Optional[Dict[str, Any]] = None
    status: str = PENDING
    error: Optional[str] = None


@dataclass
class ComposedPlate:
    assembler: Dict[str, Any]
    slots: Dict[str, Slot]
    options: Dict[str, Any]
    created: str


# ── requirements ─────────────────────────────────────────────────────────────

def requirements_for(unit: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieval requirements for an assembled unit: bedrooms from the first digit of the
    type ("2B1b" -> 2), one bathroom, no balcony. The balcony field exists so retrieval
    receives it; nothing in the UI sets it yet."""
    t = str(unit.get("type") or "")
    bedrooms = int(t[0]) if t[:1].isdigit() else 1
    return {"bedrooms": bedrooms, "bathrooms": 1, "balcony": False}


# ── slots ────────────────────────────────────────────────────────────────────

def new_slot(unit: Dict[str, Any]) -> Slot:
    return Slot(unit_id=str(unit["id"]), requirements=requirements_for(unit))


def _move(slot: Slot, status: str) -> None:
    if status not in _ALLOWED.get(slot.status, set()):
        raise ValueError(f"slot {slot.unit_id}: cannot go from {slot.status} to {status}")
    slot.status = status


def sensors_record(sensors: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Summary and failing checks of a run_sensors result."""
    if not isinstance(sensors, dict):
        return None
    checks = sensors.get("checks")
    failing = [c for c in checks if isinstance(c, dict) and not c.get("ok") and not c.get("skipped")] \
        if isinstance(checks, list) else []
    return {"summary": sensors.get("summary") or {}, "failing": failing}


def hard_fail(sensors: Optional[Dict[str, Any]]) -> bool:
    """True when a hard sensor fails (summary.gates_pass false)."""
    s = (sensors or {}).get("summary") or {}
    return s.get("gates_pass") is False


def candidate_record(plan: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """One retrieved plan as a candidate. `plan` is the server plan (kept for the result cards)."""
    ret = plan.get("retrieval") or {}
    pid = plan.get("id")
    if pid is None:
        pid = ret.get("source_id")
    return {"plan_id": pid, "rank": rank, "dist": ret.get("dist"), "plan": plan}


def tag_source_types(rooms: List[Dict[str, Any]], source_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Copy `source_type` (the numeric colour key) from the retrieved plan onto generated
    rooms by id; node ids ("r_2/primary") take their source room's type."""
    by_id = {str(r.get("id")): r.get("source_type", r.get("type")) for r in source_plan.get("rooms", [])}
    out = []
    for rm in rooms:
        rid = str(rm.get("id"))
        st = by_id.get(rid, by_id.get(rid.split("/")[0], rm.get("type")))
        out.append({**rm, "source_type": st})
    return out


def plan_record(generated: Dict[str, Any], source_plan: Dict[str, Any]) -> Dict[str, Any]:
    """The part of a generate result a slot keeps (app units)."""
    return {
        "rooms": tag_source_types(generated.get("rooms", []), source_plan),
        "edges": generated.get("edges", []),
        "node_pos": generated.get("node_pos", {}),
        "doors": generated.get("doors", []),
        "frame": generated.get("frame"),
        "room_edges": generated.get("room_edges", {}),
        "feasible": generated.get("feasible"),
        "report": generated.get("report", {}),
    }


def mark_generated(slot: Slot, candidates, chosen, plan, options, sensors) -> Slot:
    _move(slot, GENERATED)
    slot.candidates, slot.chosen, slot.plan = list(candidates), dict(chosen), plan
    slot.options, slot.sensors, slot.error = dict(options), sensors_record(sensors), None
    return slot


def mark_failed(slot: Slot, error: str, candidates=None) -> Slot:
    _move(slot, FAILED)
    slot.error = str(error)
    if candidates is not None:
        slot.candidates = list(candidates)
    return slot


def mark_edited(slot: Slot, chosen, plan, options, sensors, candidates=None) -> Slot:
    """Drill-down result written back to the slot."""
    _move(slot, EDITED)
    slot.chosen, slot.plan = dict(chosen), plan
    slot.options, slot.sensors, slot.error = dict(options), sensors_record(sensors), None
    if candidates is not None:
        slot.candidates = list(candidates)
    return slot


# ── composition ──────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_composed(assembler: Dict[str, Any], options: Optional[Dict[str, Any]] = None,
                 created: Optional[str] = None) -> ComposedPlate:
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    slots = {str(u["id"]): new_slot(u) for u in assembler.get("units", [])}
    return ComposedPlate(assembler=assembler, slots=slots, options=opts, created=created or now_iso())


def compose_unit(unit: Dict[str, Any], slot: Slot, options: Dict[str, Any],
                 retrieve: Callable[..., List[Dict[str, Any]]],
                 generate: Callable[..., Dict[str, Any]]) -> Slot:
    """Fill one slot. retrieve(boundary, entrance, typed_edges, requirements, topn) -> plans;
    generate(fp, boundary, entrance, typed_edges, requirements, options) -> dict with
    rooms, edges, node_pos, room_edges, feasible, report, sensors. Both work in app units.
    Any exception marks the slot failed; it is not re-raised."""
    k = METRES_PER_UNIT
    boundary = [[x / k, y / k] for x, y in unit["polygon"]]
    ent = unit.get("entrance_point")
    entrance = [ent[0] / k, ent[1] / k] if ent else None
    typed = unit.get("typed_edges") or []
    cands: List[Dict[str, Any]] = []
    try:
        plans = retrieve(boundary, entrance, typed, slot.requirements, TOPN)
        if not plans:
            raise ValueError("retrieval returned no plans")
        cands = [candidate_record(p, i + 1) for i, p in enumerate(plans)]
        chosen = {"rank": 1, "mirror": False}
        source = plans[0]
        gen = generate(source, boundary, entrance, typed, slot.requirements, options)
        mark_generated(slot, cands, chosen, plan_record(gen, source), options, gen.get("sensors"))
    except Exception as exc:  # one unit failing must not stop the others
        mark_failed(slot, str(exc) or type(exc).__name__, cands or None)
    return slot


def compose_plate(assembler: Dict[str, Any], retrieve, generate,
                  options: Optional[Dict[str, Any]] = None) -> ComposedPlate:
    composed = new_composed(assembler, options)
    for u in assembler.get("units", []):
        compose_unit(u, composed.slots[str(u["id"])], composed.options, retrieve, generate)
    return composed


# ── serialisation ────────────────────────────────────────────────────────────

def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, bool) or v is None or isinstance(v, (str, int)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if hasattr(v, "item"):  # numpy scalar
        return _jsonable(v.item())
    if hasattr(v, "tolist"):
        return _jsonable(v.tolist())
    return str(v)


def _m(pt) -> List[float]:
    return [pt[0] * METRES_PER_UNIT, pt[1] * METRES_PER_UNIT]


def slot_to_dict(slot: Slot, full: bool = True) -> Dict[str, Any]:
    """Slot as a JSON-safe dict in app units. full=True keeps the candidates' plans."""
    d = {
        "unit_id": slot.unit_id, "requirements": slot.requirements,
        "candidates": [c if full else {k: v for k, v in c.items() if k != "plan"} for c in slot.candidates],
        "chosen": slot.chosen, "plan": slot.plan, "options": slot.options,
        "sensors": slot.sensors, "status": slot.status, "error": slot.error,
    }
    return _jsonable(d)


def to_record(composed: ComposedPlate) -> Dict[str, Any]:
    """The record sent to the browser (app units, candidates carry their plans)."""
    return _jsonable({
        "assembler": composed.assembler, "options": composed.options, "created": composed.created,
        "slots": {uid: slot_to_dict(s, True) for uid, s in composed.slots.items()},
    })


def from_record(rec: Dict[str, Any]) -> ComposedPlate:
    """Rebuild a ComposedPlate from to_record() output (the browser sends it back, edited)."""
    slots = {}
    for uid, d in (rec.get("slots") or {}).items():
        status = d.get("status", PENDING)
        if status not in STATUSES:
            raise ValueError(f"slot {uid}: unknown status {status!r}")
        slots[str(uid)] = Slot(
            unit_id=str(d.get("unit_id", uid)), requirements=d.get("requirements") or {},
            candidates=list(d.get("candidates") or []), chosen=d.get("chosen"), plan=d.get("plan"),
            options=d.get("options") or {}, sensors=d.get("sensors"), status=status, error=d.get("error"))
    return ComposedPlate(assembler=rec.get("assembler") or {}, slots=slots,
                         options=rec.get("options") or {}, created=rec.get("created") or now_iso())


def _plan_export(plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not plan:
        return None
    return {
        "rooms": [{
            "id": r.get("id"), "label": r.get("label"), "type": r.get("type"),
            "source_type": r.get("source_type"),
            "polygon_m": [_m(p) for p in r.get("polygon", [])],
        } for r in plan.get("rooms", [])],
        "edges": plan.get("edges", []),
        "node_pos_m": {k: _m(v) for k, v in (plan.get("node_pos") or {}).items()
                       if isinstance(v, (list, tuple)) and len(v) >= 2},
        "doors_m": [{**{k: d.get(k) for k in ("u", "v", "kind", "width_mm", "segment_mm")},
                     "centre": _m(d["centre"]), "wall": [_m(q) for q in d.get("wall", [])]}
                    for d in plan.get("doors", []) if d.get("centre")],
        "feasible": plan.get("feasible"),
        "report": plan.get("report", {}),
    }


def to_export(composed: ComposedPlate) -> Dict[str, Any]:
    """JSON-safe export. Room polygons and node positions are in plate metres
    (normalised x METRES_PER_UNIT). Candidates keep plan id, rank and distance only."""
    slots = {}
    for uid, s in composed.slots.items():
        slots[uid] = {
            "unit_id": s.unit_id, "requirements": s.requirements,
            "candidates": [{k: c.get(k) for k in ("plan_id", "rank", "dist")} for c in s.candidates],
            "chosen": s.chosen, "plan": _plan_export(s.plan), "options": s.options,
            "sensors": s.sensors, "status": s.status, "error": s.error,
        }
    return _jsonable({
        "created": composed.created, "options": composed.options,
        "plate": composed.assembler, "slots": slots,
        "units": "metres (plate frame, y down)",
    })
