"""
swap.py — graph-preserving room swap.

swap_nodes(plan, room_a, room_b, ...) exchanges two rooms' footprints and lets
rectangular_arrange.arrange() re-realise the topology, then reports what the
swap actually did to the room-adjacency graph rather than trying to forbid
anything but the obviously pointless case (no shared neighbour at all — a
swap between two rooms in different parts of the plan can't preserve
adjacency by definition, so it's rejected before arrange() ever runs).
"""

from packages.floorplan_core import rectangular_arrange as ra


def _plan_adjacency(edges: list) -> dict:
    """plan['edges'] ([{u, v}, ...]) -> {room_id: {neighbour_id, ...}}."""
    adj = {}
    for e in edges or []:
        u, v = e.get("u"), e.get("v")
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def _arrange_capturing_adjacency(rooms, boundary_poly_mm, source_edges, entrance_mid):
    """Call the real arrange() unmodified, but capture the realised-adjacency
    set it computes internally (arrange() computes it but only ever returns it
    reduced to label-string 'dropped' text, not the {id, id} pairs we need).
    Spies on the module-level _realised_adjacency by swapping it out for the
    duration of the call, then restores it — arrange.py itself is untouched."""
    captured = {}
    real_fn = ra._realised_adjacency

    def _spy(owner, ncol, nrow):
        result = real_fn(owner, ncol, nrow)
        captured["realised_adj"] = result
        return result

    ra._realised_adjacency = _spy
    try:
        out = ra.arrange(rooms, boundary_poly_mm, source_edges, entrance_mid)
    finally:
        ra._realised_adjacency = real_fn
    return out, captured.get("realised_adj", set())


def swap_nodes(plan: dict, room_a: str, room_b: str,
               boundary_poly_mm: list, entrance_mid=None) -> dict:
    """Swap room_a and room_b's footprints and re-arrange into the boundary.

    Guard is LOOSE (share >= 1 common neighbour), not strict identical
    neighbour sets — a swap is allowed and its real effect on the adjacency
    graph is reported via adjacency_delta rather than being predicted upfront.
    """
    edges = plan.get("edges", [])
    adj = _plan_adjacency(edges)
    if not (adj.get(room_a, set()) & adj.get(room_b, set())):
        return {"status": "rejected", "reason": "no shared neighbour",
                "adjacency_delta": None}

    rooms = plan["rooms"]
    rooms_by_id = {r["id"]: r for r in rooms}
    poly_a, poly_b = rooms_by_id[room_a]["polygon"], rooms_by_id[room_b]["polygon"]
    swapped_rooms = []
    for r in rooms:
        nr = dict(r)
        if r["id"] == room_a:
            nr["polygon"] = poly_b
        elif r["id"] == room_b:
            nr["polygon"] = poly_a
        swapped_rooms.append(nr)

    arrangement, realised_adj = _arrange_capturing_adjacency(
        swapped_rooms, boundary_poly_mm, edges, entrance_mid)

    original_edges = {frozenset((e["u"], e["v"])) for e in edges}
    gained = [sorted(pair) for pair in realised_adj - original_edges]
    dropped = [sorted(pair) for pair in original_edges - realised_adj]

    return {
        "status": "ok",
        "arrangement": arrangement,
        "adjacency_delta": {"gained": gained, "dropped": dropped},
        "guard": "shared-neighbour",
    }
