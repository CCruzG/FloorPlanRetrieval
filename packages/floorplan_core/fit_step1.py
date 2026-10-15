"""
packages/floorplan_core/fit_step1.py — Spatial reasoning agent for Fit Step 1.

Calls Claude with the fit-step1 brief and the set_node_position tool.
Accepts the raw retrieved floor plan and the target boundary; returns
proposed centroid positions for each room node.
"""

import json
import os

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"

_SET_NODE_POSITION_TOOL = {
    "name": "set_node_position",
    "description": (
        "Set the centroid position of a room inside the target boundary. "
        "Call this once for every room in the plan. "
        "The id must be the exact string from the room's `id` field "
        "(e.g. 'r_0', 'r_1') — never a room label or an invented name."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Exact room id from the plan (e.g. 'r_0').",
            },
            "x": {
                "type": "number",
                "description": "Normalised x coordinate in [0, 1].",
            },
            "y": {
                "type": "number",
                "description": "Normalised y coordinate in [0, 1].",
            },
        },
        "required": ["id", "x", "y"],
    },
}


def _nearest_pt_on_seg(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return ax, ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    return ax + t * dx, ay + t * dy


def _nearest_pt_on_boundary(px, py, boundary):
    best_dist_sq = float("inf")
    best = (px, py)
    n = len(boundary)
    for i in range(n):
        ax, ay = boundary[i][0], boundary[i][1]
        bx, by = boundary[(i + 1) % n][0], boundary[(i + 1) % n][1]
        cx, cy = _nearest_pt_on_seg(px, py, ax, ay, bx, by)
        d2 = (cx - px) ** 2 + (cy - py) ** 2
        if d2 < best_dist_sq:
            best_dist_sq = d2
            best = (cx, cy)
    return best


def _centroid(polygon):
    if not polygon:
        return [0.0, 0.0]
    return [
        round(sum(v[0] for v in polygon) / len(polygon), 5),
        round(sum(v[1] for v in polygon) / len(polygon), 5),
    ]


def _bbox(vertices):
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return min(xs), min(ys), max(xs), max(ys)


def _affine_map(cx, cy, orig_bbox, tgt_bbox):
    """Map a point from original boundary bbox space to target boundary bbox space."""
    ox0, oy0, ox1, oy1 = orig_bbox
    tx0, ty0, tx1, ty1 = tgt_bbox
    ow, oh = (ox1 - ox0) or 1e-6, (oy1 - oy0) or 1e-6
    x = tx0 + (cx - ox0) / ow * (tx1 - tx0)
    y = ty0 + (cy - oy0) / oh * (ty1 - ty0)
    return round(x, 5), round(y, 5)


def _slim_plan(plan: dict, target_boundary: list, suggested_positions: dict = None) -> dict:
    """Return the fields the agent needs with suggested positions.

    Uses pre-computed suggested_positions (from JS, rotation/mirror-corrected)
    when provided; falls back to a simple Python affine from plan bbox → target bbox.
    """
    orig_bdry = [[v[0], v[1]] for v in plan.get("boundary", [])]

    # Compute affine bbox only if we need the Python fallback
    if not suggested_positions:
        orig_bbox = _bbox(orig_bdry) if orig_bdry else (0, 0, 1, 1)
        tgt_bbox = _bbox(target_boundary) if target_boundary else (0, 0, 1, 1)

    rooms = []
    for r in plan.get("rooms", []):
        rid = r["id"]
        if suggested_positions and rid in suggested_positions:
            sp = suggested_positions[rid]
            sx, sy = sp["x"], sp["y"]
        else:
            cx, cy = _centroid(r.get("polygon", []))
            sx, sy = _affine_map(cx, cy, orig_bbox, tgt_bbox)
        rooms.append({
            "id": rid,
            "source_type": r.get("source_type", r.get("type", -1)),
            "label": r.get("label", ""),
            "suggested_position": [sx, sy],
        })

    edges = [
        {"u": e["u"], "v": e["v"], "connection": e["connection"]}
        for e in plan.get("edges", [])
    ]
    return {
        "boundary": orig_bdry,
        "entrance": plan.get("entrance"),
        "rooms": rooms,
        "edges": edges,
    }


class FitStep1Error(Exception):
    pass


def run_fit_step1_agent(
    plan: dict,
    target_boundary: list,
    brief_text: str,
    entrance_point: list = None,
    suggested_positions: dict = None,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
) -> dict:
    """
    Call Claude to reposition room centroids inside the target boundary.

    Args:
        plan:            Retrieved floor plan dict (v2.1.0).
        target_boundary: [[x, y], ...] — the new boundary to fit into.
        brief_text:      Contents of fit-step1-brief.md (system prompt).
        model:           Claude model ID.

    Returns:
        {"positions": {"r_0": {"x": float, "y": float}, ...}}

    Raises:
        FitStep1Error on missing API key or configuration problem.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not dry_run and not api_key:
        raise FitStep1Error("ANTHROPIC_API_KEY environment variable is not set")

    # Entrance position on the new boundary
    if entrance_point and len(entrance_point) >= 2:
        ex, ey = entrance_point[0], entrance_point[1]
    else:
        # Fallback: snap plan's original entrance centroid to the new boundary
        ent = plan.get("entrance", [])
        if len(ent) >= 4:
            ent_cx = (ent[0] + ent[2]) / 2
            ent_cy = (ent[1] + ent[3]) / 2
            ex, ey = _nearest_pt_on_boundary(ent_cx, ent_cy, target_boundary)
        else:
            ex, ey = target_boundary[0][0], target_boundary[0][1]

    rooms = plan.get("rooms", [])
    room_inventory = ", ".join(
        f'"{r["id"]}" ({r.get("label", r.get("id"))})'
        for r in rooms
    )

    user_message = (
        "Target boundary (normalised [0,1] coordinates — all rooms must be inside this):\n"
        "```json\n"
        + json.dumps(target_boundary)
        + "\n```\n\n"
        f"Entrance position on the new boundary (fixed — do NOT call set_node_position for it): "
        f"[{round(ex, 5)}, {round(ey, 5)}]\n\n"
        "Floor plan with suggested positions (affine-mapped from the original layout):\n"
        "```json\n"
        + json.dumps(_slim_plan(plan, target_boundary, suggested_positions), indent=2)
        + "\n```\n\n"
        f"Call set_node_position for each of these {len(rooms)} rooms "
        f"(do NOT call it for the entrance): {room_inventory}."
    )

    if dry_run:
        print("=" * 72)
        print("SYSTEM PROMPT")
        print("=" * 72)
        print(brief_text)
        print()
        print("=" * 72)
        print("USER MESSAGE")
        print("=" * 72)
        print(user_message)
        return {"positions": {}}

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": user_message}]
    positions = {}

    while True:
        with client.messages.stream(
            model=model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=brief_text,
            tools=[_SET_NODE_POSITION_TOOL],
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        tool_calls_this_turn = []
        for block in response.content:
            if block.type == "tool_use":
                inp = block.input
                positions[inp["id"]] = {"x": inp["x"], "y": inp["y"]}
                tool_calls_this_turn.append({"id": block.id, "input": inp})

        if response.stop_reason == "end_turn" or not tool_calls_this_turn:
            break

        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": json.dumps({"ok": True, "id": tc["input"]["id"]}),
                }
                for tc in tool_calls_this_turn
            ],
        })

    # Follow-up for any rooms the agent skipped
    missing = [r for r in rooms if r["id"] not in positions]
    if missing:
        missing_list = ", ".join(
            f'"{r["id"]}" ({r.get("label", r["id"])})' for r in missing
        )
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": (
                f"You have not yet positioned: {missing_list}. "
                "Please call set_node_position for each one now."
            ),
        })
        with client.messages.stream(
            model=model,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            system=brief_text,
            tools=[_SET_NODE_POSITION_TOOL],
            messages=messages,
        ) as stream:
            followup = stream.get_final_message()
        for block in followup.content:
            if block.type == "tool_use":
                inp = block.input
                positions[inp["id"]] = {"x": inp["x"], "y": inp["y"]}

    return {"positions": positions}
