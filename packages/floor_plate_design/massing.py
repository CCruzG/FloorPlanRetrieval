"""Building massing from a composed plate (ARCHITECTURE.md, "Plate composition"). Pure, no Flask.

building_from_composed() stacks one composed plate into storeys: a slab, room prisms, hall
prisms and lobby prisms per storey, windows per storey and slot, one core prism through the
full height, a roof slab. The ground floor differs at the main entrance: the unit nearest the
entrance becomes one `entrance_lobby` prism with no windows.

Frame: plate metres, y down, z up. Layout of one storey: the floor slab fills the first
SLAB_M of the storey; rooms, halls and lobbies fill the rest. The roof slab sits on top of the
last storey. total_height_m is the top of the roof slab. The core runs from z = 0 to
total_height_m + CORE_OVERRUN_M.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from shapely.geometry import Point, Polygon

from .compose import ComposedPlate, _plan_export

SLAB_M = 0.25
GROUND_EXTRA_M = 0.5
CORE_OVERRUN_M = 1.2
_R = 4


def _pts(poly) -> List[List[float]]:
    return [[round(float(p[0]), _R), round(float(p[1]), _R)] for p in poly]


def _storeys_allowed(storeys: int, f2f: float, gf2f: float, max_height_m: Optional[float]) -> int:
    """Storeys that fit under max_height_m (ground floor counted first, roof slab included). At least 1."""
    n = storeys
    if max_height_m is not None:
        room = max_height_m - SLAB_M - gf2f
        fit = 1 + int(math.floor(room / f2f + 1e-9)) if room >= 0 else 1
        n = min(n, fit)
    return max(1, n)


def _entrance_unit(units: List[Dict[str, Any]], pt) -> Optional[str]:
    """Id of the unit whose outline contains `pt`, else the one nearest to it."""
    if pt is None or not units:
        return None
    p = Point(float(pt[0]), float(pt[1]))
    best, best_d = None, None
    for u in units:
        poly = Polygon(u["polygon"])
        d = 0.0 if poly.buffer(1e-6).contains(p) else poly.distance(p)
        if best_d is None or d < best_d:
            best, best_d = str(u["id"]), d
    return best


def building_from_composed(composed: ComposedPlate, storeys: int, floor_to_floor_m: float,
                           ground_floor_to_floor_m: Optional[float] = None,
                           max_height_m: Optional[float] = None,
                           main_entrance_m: Optional[List[float]] = None) -> Dict[str, Any]:
    if isinstance(storeys, bool) or not isinstance(storeys, int) or storeys < 1:
        raise ValueError("storeys must be an integer of at least 1")
    if not isinstance(floor_to_floor_m, (int, float)) or floor_to_floor_m <= SLAB_M:
        raise ValueError(f"floor_to_floor_m must be a number above {SLAB_M}")
    f2f = float(floor_to_floor_m)
    gf2f = float(ground_floor_to_floor_m) if ground_floor_to_floor_m is not None else f2f + GROUND_EXTRA_M
    if gf2f <= SLAB_M:
        raise ValueError(f"ground_floor_to_floor_m must be above {SLAB_M}")
    if max_height_m is not None and (not isinstance(max_height_m, (int, float)) or max_height_m <= 0):
        raise ValueError("max_height_m must be a positive number")
    if main_entrance_m is None:
        main_entrance_m = (composed.options or {}).get("main_entrance_m")

    asm = composed.assembler or {}
    boundary = asm.get("boundary") or []
    if len(boundary) < 3:
        raise ValueError("composed plate has no boundary")
    n = _storeys_allowed(storeys, f2f, gf2f, max_height_m)

    heights, z = [], 0.0
    for i in range(n):
        top = z + (gf2f if i == 0 else f2f)
        heights.append({"storey": i, "z0": round(z, 4), "z1": round(top, 4)})
        z = top
    total = round(z + SLAB_M, 4)

    units = asm.get("units") or []
    entrance_uid = _entrance_unit(units, main_entrance_m)

    prisms: List[Dict[str, Any]] = []
    windows: List[Dict[str, Any]] = []
    fallback_units: List[str] = []

    def add(stem, storey, role, ptype, polygon, z0, z1):
        prisms.append({"id": f"{stem}@{storey}", "stem": stem, "storey": storey, "role": role,
                       "type": ptype, "polygon": _pts(polygon), "z0": round(z0, 4), "z1": round(z1, 4)})

    for h in heights:
        i, base = h["storey"], h["z0"]
        floor_top = base + SLAB_M
        room_top = h["z1"]
        add("slab", i, "slab", None, boundary, base, floor_top)
        for c in asm.get("cores") or []:
            lob = c.get("lobby_polygon")
            if lob:
                add(f"lobby-{c.get('id')}", i, "lobby", None, lob, floor_top, room_top)
        for c in asm.get("corridors") or []:
            if c.get("polygon"):
                add(f"hall-{c.get('id')}", i, "hall", None, c["polygon"], floor_top, room_top)
        for u in units:
            uid = str(u["id"])
            if i == 0 and uid == entrance_uid:
                add(f"{uid}/entrance_lobby", i, "entrance_lobby", None, u["polygon"], floor_top, room_top)
                continue
            slot = composed.slots.get(uid)
            plan = _plan_export(slot.plan) if slot is not None and slot.plan else None
            rooms = [r for r in (plan or {}).get("rooms", []) if len(r.get("polygon_m") or []) >= 3]
            if not rooms:
                if i == 0:
                    fallback_units.append(uid)
                add(f"{uid}/unit_fallback", i, "unit_fallback", None, u["polygon"], floor_top, room_top)
                continue
            for r in rooms:
                st = r.get("source_type")
                add(f"{uid}/{r.get('id')}", i, "room", st if st is not None else r.get("type"),
                    r["polygon_m"], floor_top, room_top)
            for k, w in enumerate(plan.get("windows_m") or []):
                if not w.get("centre") or len(w.get("wall") or []) < 2:
                    continue
                sill = (w.get("sill_mm") or 0) / 1000.0
                head = (w.get("head_mm") or 2400) / 1000.0
                z0 = min(floor_top + sill, room_top)
                z1 = min(floor_top + head, room_top)
                if z1 - z0 < 0.05:
                    continue
                windows.append({
                    "id": f"{uid}/w{k}@{i}", "role": "window", "storey": i, "unit_id": uid,
                    "room": w.get("room"), "wall": _pts(w["wall"]), "centre": _pts([w["centre"]])[0],
                    "width_m": round((w.get("width_mm") or 0) / 1000.0, 4),
                    "z0": round(z0, 4), "z1": round(z1, 4)})
    add_roof = {"id": f"roof@{n}", "stem": "roof", "storey": n, "role": "roof", "type": None,
                "polygon": _pts(boundary), "z0": round(z, 4), "z1": total}
    prisms.append(add_roof)
    for c in asm.get("cores") or []:
        poly = c.get("service_polygon") or c.get("polygon")
        if poly:
            prisms.append({"id": f"core-{c.get('id')}", "stem": f"core-{c.get('id')}", "storey": None,
                           "role": "core", "type": None, "polygon": _pts(poly),
                           "z0": 0.0, "z1": round(total + CORE_OVERRUN_M, 4)})

    if main_entrance_m is None:
        ent_note = "no main entrance given: the ground floor equals a typical storey"
    else:
        ent_note = f"unit {entrance_uid} is the entrance lobby on the ground floor"
    capped = n < storeys
    report = {
        "storeys_requested": storeys, "storeys_achieved": n, "capped_by_max_height": capped,
        "max_height_m": max_height_m, "floor_to_floor_m": f2f, "ground_floor_to_floor_m": gf2f,
        "slab_m": SLAB_M, "core_overrun_m": CORE_OVERRUN_M,
        "entrance_unit": entrance_uid if main_entrance_m is not None else None,
        "entrance": ent_note, "unit_fallbacks": fallback_units,
        "prism_count": len(prisms), "window_count": len(windows),
    }
    return {"storeys_requested": storeys, "storeys": n, "heights": heights, "total_height_m": total,
            "prisms": prisms, "windows": windows, "report": report}
