"""Node graph: the block-level graph of a plan (brain, no Flask, no DOM).

The node graph is the primary graph from rules v1.2 on (decision 2026-11-03);
the plan's room graph (``rooms`` + ``edges``) stays as its source and as legacy
data. It is stored as a sidecar (one gzipped JSON line per plan, keyed by plan
id, header line = rules version + parameters) until the rules freeze, then
baked into an enriched dataset copy (v2.2.0).

Nodes come from ``part_labels`` (rect_decompose -> rules): a whole room is one
node (``r_3``); a split room gives ``r_3/primary``, ``r_3/circ<k>``,
``r_3/stor<i>``, ``r_3/nook<i>`` and, rarely, ``r_3/p<i>`` (residual).
Node fields: id, room_id, room_label, kind, polygon (rings, plan units),
area_m2, ext_m (contact with the plan boundary), parts (part indices), flags.
Edge kinds:
  door  -- a door (or the entrance, b = "exterior") between two nodes
  open  -- an RPlan open edge (connection 2) between rooms, placed on nodes
  block -- an open edge between two nodes of one room (shared >= connect_min_m)
  wall  -- nodes of different rooms sharing >= 0.05 m with no door/open edge
Per-plan metrics:
  distribution_ratio -- share of the rooms the living room serves (door/open
      edges, exterior excluded) that are reached only through its circulation
      nodes, not its primary. 0 = rooms open straight off the living area (or
      no circulation); 1 = the living area distributes everything through a
      circulation band. None when the plan has no living room or it serves no room.
"""
from __future__ import annotations

import gzip
import json
from functools import lru_cache

from shapely.geometry import Polygon
from shapely.ops import unary_union

from .part_labels import _EXT_EPS, _MIN_M, _plan_boundary, _rings, _shared_len, label_parts
from .rect_decompose import decompose_plan

RULES_VERSION = "1.2"
SCHEMA_VERSION = "node_graph/2"  # /2 (2026-11-03): adds per-plan metrics.distribution_ratio


def _kind(node_id: str) -> str:
    if "/" not in node_id:
        return "whole"
    tail = node_id.split("/", 1)[1]
    for k in ("primary", "circ", "stor", "nook"):
        if tail.startswith(k):
            return k
    return "residual"


def build_node_graph(plan: dict, *, decompose_kw=None, label_kw=None) -> dict:
    """Return the node graph of one RPlan v2.1.0 plan."""
    dec = decompose_plan(plan, **(decompose_kw or {}))
    lab = label_parts(plan, dec, **(label_kw or {}))
    S = lab["params"]["scale_m"]
    bline = _plan_boundary(plan)
    bzone = bline.buffer(_EXT_EPS) if bline is not None else None

    polys = {(r["id"], i): Polygon(pt["polygon"]) for r in dec["rooms"] for i, pt in enumerate(r["parts"])
             if len(pt["polygon"]) >= 3}
    nodes, geom = [], {}
    for ro in lab["rooms"]:
        by_node = {}
        for p in ro["parts"]:
            by_node.setdefault(p["node"], []).append(p)
        for nid, parts in by_node.items():
            g = unary_union([polys[(ro["id"], p["index"])] for p in parts if (ro["id"], p["index"]) in polys])
            if g.is_empty:
                continue
            geom[nid] = g
            kind = _kind(nid)
            flags = {}
            if kind == "primary":
                flags["pass_through"] = bool(ro["pass_through"])
                if ro["kitchen_openable"] is not None:
                    flags["kitchen_openable"] = bool(ro["kitchen_openable"])
                for role in ("zone", "jog", "alcove"):
                    c = sum(p["role"] == role for p in parts)
                    if c:
                        flags[role] = c
                if any(p.get("dining_candidate") for p in parts):
                    flags["dining_candidate"] = True
            nodes.append({"id": nid, "room_id": ro["id"], "room_label": ro["label"], "kind": kind,
                          "polygon": _rings(g), "area_m2": round(g.area * S * S, 3),
                          "ext_m": round(g.boundary.intersection(bzone).length * S, 3) if bzone is not None else 0.0,
                          "parts": sorted(p["index"] for p in parts), "flags": flags})

    edges, linked = [], set()
    for c in lab["connections"]:
        e = {"a": c["a"], "b": c["b"], "kind": c["kind"]}
        if c["kind"] == "door":
            e["door_id"] = c["door_id"]
        edges.append(e)
        linked.add(frozenset((c["a"], c["b"])))
    for ro in lab["rooms"]:
        for be in ro["block_edges"]:
            if be["kind"] == "open":
                edges.append({"a": be["a"], "b": be["b"], "kind": "block", "shared_m": round(be["shared_m"], 3)})
    ids = list(geom)
    room_of = {n["id"]: n["room_id"] for n in nodes}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if room_of[a] == room_of[b] or frozenset((a, b)) in linked:
                continue
            ga, gb = geom[a], geom[b]
            if not ga.envelope.buffer(1e-4).intersects(gb.envelope):
                continue
            ln = _shared_len(ga, gb) * S
            if ln >= _MIN_M:
                edges.append({"a": a, "b": b, "kind": "wall", "shared_m": round(ln, 3)})
    return {"id": plan.get("id"), "nodes": nodes, "edges": edges, "metrics": distribution_metrics(nodes, edges),
            "summary": {"n_nodes": len(nodes), "n_split_rooms": sum(len(r["parts"]) > 1 for r in lab["rooms"]),
                        "n_door_unresolved": lab["summary"]["n_door_unresolved"],
                        "n_open_unresolved": lab["summary"]["n_open_unresolved"]}}


def distribution_metrics(nodes: list, edges: list) -> dict:
    """Living-room distribution ratio (see module docstring)."""
    living = [n for n in nodes if n["room_label"] == "LivingRoom"]
    if not living:
        return {"distribution_ratio": None, "living_reach": 0, "living_reach_via_circ": 0}
    room_of = {n["id"]: n["room_id"] for n in nodes}
    lr_rooms = {n["room_id"] for n in living}
    via = {"circ": set(), "other": set()}
    for e in edges:
        if e["kind"] not in ("door", "open"):
            continue
        for x, y in ((e["a"], e["b"]), (e["b"], e["a"])):
            if room_of.get(x) in lr_rooms and y != "exterior" and room_of.get(y) not in lr_rooms:
                via["circ" if _kind(x) == "circ" else "other"].add(room_of[y])
    reach = via["circ"] | via["other"]
    only = via["circ"] - via["other"]
    return {"distribution_ratio": round(len(only) / len(reach), 4) if reach else None,
            "living_reach": len(reach), "living_reach_via_circ": len(only)}


def sidecar_meta(*, decompose_kw=None, label_kw=None, source=None) -> dict:
    """Header line: rules version + effective parameters, so a stale sidecar is detectable."""
    import inspect
    dsig = {k: v.default for k, v in inspect.signature(decompose_plan).parameters.items()
            if v.default is not inspect.Parameter.empty}
    lsig = {k: v.default for k, v in inspect.signature(label_parts).parameters.items()
            if v.default is not inspect.Parameter.empty}
    dsig.update(decompose_kw or {}); lsig.update(label_kw or {})
    dsig = {k: list(v) if isinstance(v, tuple) else v for k, v in dsig.items()}
    return {"_meta": {"schema": SCHEMA_VERSION, "rules_version": RULES_VERSION,
                      "decompose_params": dsig, "label_params": lsig, "source": source}}


def current_meta() -> dict:
    return sidecar_meta()["_meta"]


@lru_cache(maxsize=2)
def load_sidecar(path: str) -> tuple[dict, dict]:
    """Load a sidecar into (meta, {plan_id: graph}). Cached per path."""
    meta, graphs = {}, {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            if "_meta" in d:
                meta = d["_meta"]
            else:
                graphs[str(d["id"])] = d
    return meta, graphs


def is_stale(meta: dict) -> bool:
    """True when the sidecar was built with other rules or parameters than the code now uses."""
    cur = current_meta()
    return any(meta.get(k) != cur[k] for k in ("schema", "rules_version", "decompose_params", "label_params"))


_KIND_LABEL = {"circ": ("Hallway", "hallway"), "stor": ("Storage", "storage")}
_EDGE_CONNECTION = {"wall": 0, "door": 1, "open": 2, "block": 2}


def _split_circulation(plan, node_graph):
    """Replace each multi-part circulation node by one node per rectangle part.

    Arms are joined by ``block`` edges where they share >= 0.05 m; every edge of
    the region is re-attached to the arm that carries it (doors by door id,
    other edges to the arm with the longest shared boundary)."""
    dec = decompose_plan(plan)
    lab = label_parts(plan, dec)
    S = lab["params"]["scale_m"]
    part_poly = {(r["id"], i): Polygon(pt["polygon"]) for r in dec["rooms"] for i, pt in enumerate(r["parts"])
                 if len(pt["polygon"]) >= 3}
    door_part = {}
    arms = {}  # circ node id -> [(arm id, polygon, door ids)]
    for ro in lab["rooms"]:
        for reg in ro["regions"]:
            idxs = reg["part_indices"]
            if len(idxs) < 2:
                continue
            lst = []
            for k, i in enumerate(idxs):
                members = [i] + [q["index"] for q in ro["parts"] if q["absorbed_into"] == i]
                poly = unary_union([part_poly[(ro["id"], m)] for m in members if (ro["id"], m) in part_poly])
                doors = set(ro["parts"][i]["door_ids"])
                lst.append((f"{reg['node']}{chr(97 + k)}", poly, doors))
            arms[reg["node"]] = lst
    if not arms:
        return node_graph
    geom = {n["id"]: unary_union([Polygon(r) for r in n["polygon"] if len(r) >= 3]) for n in node_graph["nodes"]}
    nodes, edges = [], []
    for n in node_graph["nodes"]:
        if n["id"] not in arms:
            nodes.append(n)
            continue
        for aid, poly, _ in arms[n["id"]]:
            nodes.append({**n, "id": aid, "polygon": _rings(poly), "area_m2": round(poly.area * S * S, 3)})
        lst = arms[n["id"]]
        for x in range(len(lst)):
            for y in range(x + 1, len(lst)):
                ln = _shared_len(lst[x][1], lst[y][1]) * S
                if ln >= _MIN_M:
                    edges.append({"a": lst[x][0], "b": lst[y][0], "kind": "block", "shared_m": round(ln, 3)})

    def arm_for(cid, other, door_id):
        lst = arms[cid]
        if door_id is not None:
            for aid, _, doors in lst:
                if door_id in doors:
                    return aid
        og = geom.get(other)
        if og is None:
            return lst[0][0]
        return max(lst, key=lambda t: _shared_len(t[1], og))[0]
    for e in node_graph["edges"]:
        a, b = e["a"], e["b"]
        if a in arms:
            a = arm_for(a, e["b"], e.get("door_id"))
        if b in arms:
            b = arm_for(b, e["a"], e.get("door_id"))
        edges.append({**e, "a": a, "b": b})
    return {**node_graph, "nodes": nodes, "edges": edges}


def entry_node(node_graph: dict) -> str | None:
    """Node the entrance opens onto: the node graph's edge to "exterior"."""
    for e in node_graph.get("edges", []):
        if e.get("b") == "exterior":
            return e["a"]
        if e.get("a") == "exterior":
            return e["b"]
    return None


def entry_room_of(plan: dict) -> str | None:
    """Id of what the entrance opens onto, for generation: the entry node of an
    expanded plan, else the source room of the plan's entry node."""
    if plan.get("node_graph_expanded"):
        return plan.get("entry_node")
    g = build_node_graph(plan)
    nid = entry_node(g)
    return next((n["room_id"] for n in g["nodes"] if n["id"] == nid), None)


def display_graph(plan: dict, node_graph: dict) -> dict:
    """Node graph for drawing on a retrieved plan (2026-11-04).

    Circulation is split into its rectangular arms, the units generation places.
    Each node gets ``pos``, a label point inside its polygon (the centroid of an
    L/T node can fall outside it). Polygons stay in plan units, the frame of the
    plan's rooms. Edges to "exterior" leave the edge list; ``entry_node`` keeps the
    entrance's node.
    """
    g = _split_circulation(plan, node_graph)
    nodes = []
    for n in g["nodes"]:
        geom = unary_union([Polygon(r) for r in n.get("polygon") or [] if len(r) >= 3])
        if geom.is_empty:
            continue
        pt = geom.centroid
        if not geom.contains(pt):
            pt = geom.representative_point()
        nodes.append({"id": n["id"], "room_id": n["room_id"], "kind": n["kind"], "polygon": n["polygon"],
                      "area_m2": n["area_m2"], "pos": [round(pt.x, 5), round(pt.y, 5)]})
    ids = {n["id"] for n in nodes}
    edges = [{"a": e["a"], "b": e["b"], "kind": e["kind"]} for e in g["edges"]
             if e["a"] in ids and e["b"] in ids]
    return {"nodes": nodes, "edges": edges, "entry_node": entry_node(g), "metrics": g.get("metrics", {})}


def anchor_node(expanded_plan: dict, room_id: str) -> str:
    """Node that stands for a source room in an expanded plan: its whole or primary
    node, else its first node. An id that is already a node, or unknown, is returned as is."""
    own = [r for r in expanded_plan.get("rooms", []) if r.get("room_id") == room_id]
    for r in own:
        if r.get("node_kind") in ("whole", "primary"):
            return r["id"]
    return own[0]["id"] if own else room_id


def expand_plan(plan: dict, node_graph: dict, split_circulation: bool = True) -> dict:
    """Room-level plan whose rooms are the node graph's nodes (2026-11-03).

    Circulation nodes become rooms labelled Hallway (transit, MINIMUM_SIZES
    "hallway"); storage blocks become Storage; primary / whole / nook nodes keep
    the room's type and label (a nook becomes its own room, open to the
    primary). Edges: door -> connection 1, block (same room) and open -> 2 with
    separation "open", wall -> 0. Room ids are node ids ("r_2/circ0"); each room
    carries ``room_id`` (its source room). Boundary, entrance, doors and windows
    are copied unchanged. The generator and the sensors then see circulation as
    a room open to the living area, the arrangement designers actually drew.
    """
    if split_circulation:
        node_graph = _split_circulation(plan, node_graph)
    src = {r["id"]: r for r in plan.get("rooms", [])}
    rooms = []
    for n in node_graph.get("nodes", []):
        rings = [r for r in (n.get("polygon") or []) if len(r) >= 3]
        if not rings:
            continue
        ring = max(rings, key=lambda r: abs(Polygon(r).area))
        base = src.get(n["room_id"], {})
        if n["kind"] in _KIND_LABEL:
            label, typ = _KIND_LABEL[n["kind"]]
        else:
            label, typ = base.get("label"), base.get("type")
        rooms.append({"id": n["id"], "type": typ, "label": label, "polygon": ring,
                      "room_id": n["room_id"], "node_kind": n["kind"]})
    ids = {r["id"] for r in rooms}
    edges, seen = [], set()
    for e in node_graph.get("edges", []):
        a, b = e["a"], e["b"]
        if a not in ids or b not in ids:
            continue
        key = frozenset((a, b))
        conn = _EDGE_CONNECTION.get(e["kind"], 0)
        if key in seen:
            # keep the strongest connection between a pair (open > door > wall)
            for ex in edges:
                if frozenset((ex["u"], ex["v"])) == key and conn > ex["connection"]:
                    ex["connection"] = conn
                    if conn == 2:
                        ex["separation"] = "open"
            continue
        seen.add(key)
        ed = {"u": a, "v": b, "connection": conn}
        if conn == 2:
            ed["separation"] = "open"
        edges.append(ed)
    out = {k: v for k, v in plan.items() if k not in ("rooms", "edges")}
    out["rooms"], out["edges"] = rooms, edges
    out["node_graph_expanded"] = True
    # The entrance is not an edge between rooms, so it is carried by name. Generation
    # reads it: the entry node, not the nearest node, fronts the drawn entrance.
    out["entry_node"] = entry_node(node_graph)
    return out
