"""Rules v1.2: label the parts of decomposed rooms (brain, no Flask, no DOM).

Input: an RPlan v2.1.0 plan (coords normalised, x ``scale_m`` = metres) and the
``decompose_plan`` result for the same plan. Output: per-part roles, nodes,
circulation regions, intra-room block edges, wall contacts and a node-level
connection list (doors and open edges).

Definitions: s = part short side (m). A door (or the entrance, id
``"entrance"``) *faces* a part when the length of the part boundary inside the
door box (buffered by 0.25 x door thickness) is >= ``door_contact_frac`` x the
door box's long side. ``ext`` = part boundary on the plan boundary, length
>= ``ext_min_m``.

Per room with more than one part:
1. Primary = largest part by area (role ``primary``). Single-part room: ``whole``.
2. Thin parts (non-primary): no door and s < ``absorb_nodoor_max_m``, or a door
   and s < ``absorb_door_max_m``. Exception (v1.2, 2026-11-03): a no-door part
   with s < ``absorb_nodoor_max_m`` is a ``jog`` (a projection such as a bay
   window), not thin, when it projects (ext >= long + ``jog_project_frac`` x s),
   s >= ``jog_min_m`` (raster floor) and cover < ``jog_max_cover``. Cover = the
   part's long side / the room's wall it sits on (the straight, continuous run of
   the rest of the room's outline on the attachment line); a full-length strip
   (cover ~1) is a wall offset and is absorbed. ``jog_min_m=None`` disables the exception.
3. Every other non-primary part, on its own rectangle:
   door & s <= ``circ_max_short_m`` -> circulation; door & s > that -> zone;
   no door, s <= ``storage_max_short_m``: not ext -> storage, ext -> jog;
   no door, s >= ``nook_min_short_m`` and ext -> nook; else alcove.
4. Thin parts are absorbed smallest area first. Candidates share boundary along
   the thin part's long side, >= ``absorb_contact_frac`` x its long side;
   prefer a circulation absorber, else the longest shared length; with no
   candidate, the neighbour with the longest shared edge on any side. A thin
   part goes only into a non-thin part or into the absorber of an already
   absorbed thin part. The absorber keeps its role; doors transfer to it.
5. Nodes: primary/zone/jog/alcove -> ``<room>/primary``; circulation regions ->
   ``<room>/circ<k>``; storage -> ``<room>/stor<i>``; nook -> ``<room>/nook<i>``;
   absorbed parts -> their absorber's node; single-part room -> ``<room>``.
6. Circulation parts (with their absorbed parts) joined by shared edges
   >= ``connect_min_m`` form one region.
7. ``block_edges``: two nodes of one room sharing >= ``connect_min_m`` -> kind
   ``open``; sharing 0.05 m up to that -> kind ``touch`` (not a connection).
8. ``pass_through``: >= 2 circulation regions and the primary has an ``open``
   edge to at least two. ``isolated_circulation``: region with no ``open``
   edge and no facing door.
9. ``connections``: doors (kind ``door``) and RPlan edges with connection 2
   (kind ``open``; side = node with the longest shared wall with the other room).
10. ``wall_contacts`` per node; ``kitchen_openable`` per split room.
"""
from __future__ import annotations

from shapely.geometry import LineString, Polygon, box
from shapely.ops import linemerge, unary_union

_EPS = 1e-5          # normalised-unit tolerance for shared-boundary tests
_EXT_EPS = 5e-4      # normalised tolerance for "on the plan boundary"
_MIN_M = 0.05        # shared length below this (m) is ignored


def _door_box(bbox):
    x0, x1 = sorted((bbox[0], bbox[2]))
    y0, y1 = sorted((bbox[1], bbox[3]))
    return x0, y0, x1, y1


def _contact(part_poly: Polygon, bbox, frac: float) -> float:
    """Return boundary length inside the buffered door box, or 0 if below threshold."""
    x0, y0, x1, y1 = _door_box(bbox)
    dx, dy = x1 - x0, y1 - y0
    long_side, thick = max(dx, dy), min(dx, dy)
    if long_side <= 0:
        return 0.0
    zone = box(x0, y0, x1, y1).buffer(0.25 * thick, join_style=2)
    length = part_poly.boundary.intersection(zone).length
    return length if length >= frac * long_side else 0.0


def _rings(geom):
    return [[list(c) for c in g.exterior.coords[:-1]] for g in getattr(geom, "geoms", [geom])
            if isinstance(g, Polygon)]


def _lines(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    return [l for g in getattr(geom, "geoms", []) for l in _lines(g)]


def _shared(a, b):
    """Shared boundary of two geometries: (length in plan units, [LineString])."""
    ls = _lines(a.boundary.intersection(b.boundary.buffer(_EPS)))
    return sum(l.length for l in ls), ls


def _shared_len(a, b):
    return _shared(a, b)[0]


def _axis_len(lines, axis):
    """Length of segments of ``lines`` running along ``axis`` (0 = x, 1 = y)."""
    tot = 0.0
    for l in lines:
        c = list(l.coords)
        for p, q in zip(c, c[1:]):
            d = (abs(q[0] - p[0]), abs(q[1] - p[1]))
            if d[axis] >= d[1 - axis]:
                tot += (d[0] ** 2 + d[1] ** 2) ** 0.5
    return tot


def _plan_boundary(plan):
    pts = [(p[0], p[1]) for p in (plan.get("boundary") or []) if len(p) >= 2]
    if len(pts) >= 3:
        return LineString(pts + [pts[0]])
    polys = [Polygon(r["polygon"]) for r in plan.get("rooms", []) if r.get("polygon") and len(r["polygon"]) >= 3]
    if not polys:
        return None
    u = unary_union([p.buffer(0) for p in polys])
    return u.boundary


def label_parts(plan: dict, decomposition: dict, *, scale_m=18.0,
                door_contact_frac=0.5, absorb_nodoor_max_m=0.5, absorb_door_max_m=0.85,
                circ_max_short_m=2.0, storage_max_short_m=1.0, nook_min_short_m=1.5,
                ext_min_m=0.9, connect_min_m=0.85, absorb_contact_frac=0.5,
                dining_min_wall_m=0.85, jog_min_m=0.25, jog_max_cover=0.9,
                jog_project_frac=0.5) -> dict:
    params = {"scale_m": scale_m, "door_contact_frac": door_contact_frac,
              "dining_min_wall_m": dining_min_wall_m,
              "absorb_nodoor_max_m": absorb_nodoor_max_m, "absorb_door_max_m": absorb_door_max_m,
              "circ_max_short_m": circ_max_short_m, "storage_max_short_m": storage_max_short_m,
              "nook_min_short_m": nook_min_short_m, "ext_min_m": ext_min_m,
              "connect_min_m": connect_min_m, "absorb_contact_frac": absorb_contact_frac,
              "jog_min_m": jog_min_m, "jog_max_cover": jog_max_cover, "jog_project_frac": jog_project_frac}
    S = scale_m
    openings = [(d.get("id"), d.get("bbox"), d.get("room_refs") or []) for d in plan.get("doors", [])
                if d.get("bbox") and len(d["bbox"]) >= 4]
    ent = plan.get("entrance")
    if ent and len(ent) >= 4:
        openings.append(("entrance", ent, []))
    bline = _plan_boundary(plan)
    bzone = bline.buffer(_EXT_EPS) if bline is not None else None

    polys = {}  # (room_id, part_index) -> Polygon
    for room in decomposition.get("rooms", []):
        for i, pt in enumerate(room.get("parts", [])):
            polys[(room["id"], i)] = Polygon(pt["polygon"])

    facing = {k: {} for k in polys}  # (room_id, part_index) -> {door_id: contact}
    for did, bbox, _ in openings:
        for k, poly in polys.items():
            c = _contact(poly, bbox, door_contact_frac)
            if c > 0:
                facing[k][did] = c

    n_jog_proj = 0
    counts = dict(circulation=0, zone=0, jog=0, alcove=0, storage=0, nook=0, absorbed=0, residual=0)
    n_reg = n_pass = n_iso = 0
    node_of = {}
    primary_node = {}
    node_geom = {}   # node id -> geometry
    room_geom = {}   # room id -> geometry
    rooms_out = []

    for room in decomposition.get("rooms", []):
        rid, rlabel = room["id"], room.get("label")
        parts = room.get("parts", [])
        n = len(parts)
        P = [polys[(rid, i)] for i in range(n)]
        areas = [p.area for p in P]
        dims = []
        for i, pt in enumerate(parts):
            x0, y0, x1, y1 = P[i].bounds
            w, h = (x1 - x0) * S, (y1 - y0) * S
            sm = pt.get("short_m", min(w, h))
            lm = pt.get("long_m", max(w, h))
            dims.append((float(sm), float(lm), 0 if w >= h else 1))
        ext_m = [(P[i].boundary.intersection(bzone).length * S if bzone is not None else 0.0) for i in range(n)]
        has_door = [bool(facing[(rid, i)]) for i in range(n)]
        multi = n > 1
        main = max(range(n), key=lambda i: areas[i]) if n else None

        role = {}
        cover = {}
        absorbed_into = {i: None for i in range(n)}

        def projecting_jog(i):
            """v1.2: a thin no-door part that projects and covers only part of the room's wall.

            Cover = strip long side / the room's wall it sits on: the straight,
            continuous run of the rest of the room's outline on the strip's
            attachment line (whole room, not one neighbouring part).
            """
            s, l, axis = dims[i]
            if jog_min_m is None or s < jog_min_m or ext_m[i] < l + jog_project_frac * s:
                return False
            rest = unary_union([P[c] for c in range(n) if c != i])
            _, ls = _shared(P[i], rest)

            def along(g):
                (ax, ay), (bx, by) = g.coords[0], g.coords[-1]
                return abs(bx - ax) > abs(by - ay) if axis == 0 else abs(by - ay) > abs(bx - ax)
            segs = [LineString([a, b]) for g in ls for a, b in zip(g.coords[:-1], g.coords[1:])]
            att = [g for g in segs if along(g) and g.length * S >= _MIN_M]
            if not att:
                return False
            seg = max(att, key=lambda g: g.length)
            k = seg.coords[0][1] if axis == 0 else seg.coords[0][0]
            x0, y0, x1, y1 = rest.bounds
            line = (LineString([(x0 - 1, k), (x1 + 1, k)]) if axis == 0
                    else LineString([(k, y0 - 1), (k, y1 + 1)]))
            pieces = [g for g in _lines(rest.boundary.intersection(line.buffer(_EPS)))
                      if along(g) and g.length > 3 * _EPS]
            merged = linemerge(pieces) if pieces else None
            runs = _lines(merged) if merged is not None else []
            wall = sum(r.length for r in runs if r.distance(seg.centroid) <= 2 * _EPS) * S
            cover[i] = l / wall if wall > 0 else float("inf")
            return cover[i] < jog_max_cover

        if not multi:
            if n:
                role[0] = "whole"
        else:
            thin = []
            for i in range(n):
                s = dims[i][0]
                if i == main:
                    role[i] = "primary"
                elif not has_door[i] and s < absorb_nodoor_max_m and projecting_jog(i):
                    role[i] = "jog"
                    n_jog_proj += 1
                elif (not has_door[i] and s < absorb_nodoor_max_m) or (has_door[i] and s < absorb_door_max_m):
                    thin.append(i)
                elif has_door[i]:
                    role[i] = "circulation" if s <= circ_max_short_m else "zone"
                elif s <= storage_max_short_m:
                    role[i] = "jog" if ext_m[i] >= ext_min_m else "storage"
                elif s >= nook_min_short_m and ext_m[i] >= ext_min_m:
                    role[i] = "nook"
                else:
                    role[i] = "alcove"
            # absorption, smallest area first, restart after each success
            remaining = set(thin)
            root = {}  # absorbed thin idx -> final absorber idx

            def resolve(c):
                return root.get(c, c)

            def pick(t):
                long_len = dims[t][1] / S
                axis = dims[t][2]
                cands, anyc = [], []
                for c in range(n):
                    if c == t or c in remaining:
                        continue
                    tot, ls = _shared(P[t], P[c])
                    if tot * S < _MIN_M:
                        continue
                    r = resolve(c)
                    anyc.append((tot, r))
                    if _axis_len(ls, axis) >= absorb_contact_frac * long_len:
                        cands.append((role.get(r) == "circulation", tot, r))
                if cands:
                    return max(cands, key=lambda x: (x[0], x[1]))[2]
                if anyc:
                    return max(anyc, key=lambda x: x[0])[1]
                return None

            while remaining:
                for t in sorted(remaining, key=lambda i: areas[i]):
                    r = pick(t)
                    if r is not None:
                        root[t] = r
                        absorbed_into[t] = r
                        role[t] = "absorbed"
                        remaining.discard(t)
                        break
                else:
                    for t in remaining:  # disconnected leftovers stay visible
                        role[t] = "residual"
                    break
            # absorbed_into points at the final absorber
            for t, r in root.items():
                absorbed_into[t] = resolve(r)

        # door transfer
        door_ids = {i: list(facing[(rid, i)]) for i in range(n)}
        transferred = {i: [] for i in range(n)}
        for t in range(n):
            a = absorbed_into[t]
            if a is not None:
                for d in door_ids[t]:
                    if d not in door_ids[a]:
                        door_ids[a].append(d)
                transferred[t] = door_ids[t]
                door_ids[t] = []

        # circulation regions
        def members(i):  # a non-absorbed part plus the parts absorbed into it
            return [i] + [t for t in range(n) if absorbed_into[t] == i]
        geo = {i: unary_union([P[m] for m in members(i)]) for i in range(n) if absorbed_into[i] is None}
        circ = [i for i in range(n) if role.get(i) == "circulation"]
        comp = {i: i for i in circ}

        def find(a):
            while comp[a] != a:
                comp[a] = comp[comp[a]]
                a = comp[a]
            return a
        for a in circ:
            for b in circ:
                if a < b and _shared_len(geo[a], geo[b]) * S >= connect_min_m:
                    comp[find(a)] = find(b)
        groups = {}
        for i in circ:
            groups.setdefault(find(i), []).append(i)
        regions, region_of = [], {}
        for k, idxs in enumerate(sorted(groups.values(), key=min)):
            u = unary_union([geo[i] for i in idxs])
            for i in idxs:
                region_of[i] = k
            regions.append({"region": k, "node": f"{rid}/circ{k}", "role": "circulation", "part_indices": idxs,
                            "absorbed_indices": [t for i in idxs for t in range(n) if absorbed_into[t] == i],
                            "polygon": (_rings(u) or [[]])[0], "area_m2": u.area * S ** 2})

        # nodes
        node = {}
        n_stor = n_nook = 0
        for i in range(n):
            r = role.get(i)
            if r == "whole":
                node[i] = rid
            elif r in ("primary", "zone", "jog", "alcove"):
                node[i] = f"{rid}/primary"
            elif r == "circulation":
                node[i] = f"{rid}/circ{region_of[i]}"
            elif r == "storage":
                node[i] = f"{rid}/stor{n_stor}"; n_stor += 1
            elif r == "nook":
                node[i] = f"{rid}/nook{n_nook}"; n_nook += 1
            elif r == "residual":
                node[i] = f"{rid}/p{i}"
        for i in range(n):
            if absorbed_into[i] is not None:
                node[i] = node[absorbed_into[i]]
        for i in range(n):
            node_of[(rid, i)] = node[i]
        if n:
            primary_node[rid] = node[main]
        ngeo = {}
        for i in range(n):
            ngeo.setdefault(node[i], []).append(P[i])
        ngeo = {k: unary_union(v) for k, v in ngeo.items()}
        node_geom.update(ngeo)
        room_geom[rid] = unary_union(P) if n else None

        # block edges
        block_edges = []
        names = list(ngeo)
        for ai in range(len(names)):
            for bi in range(ai + 1, len(names)):
                tot, ls = _shared(ngeo[names[ai]], ngeo[names[bi]])
                if tot * S < _MIN_M or not ls:
                    continue
                best = max(ls, key=lambda l: l.length)
                c = list(best.coords)
                seg = [list(c[0]), list(c[-1])]
                block_edges.append({"a": names[ai], "b": names[bi], "shared_m": tot * S,
                                    "kind": "open" if tot * S >= connect_min_m else "touch",
                                    "mid": [(seg[0][0] + seg[1][0]) / 2, (seg[0][1] + seg[1][1]) / 2],
                                    "seg": seg})
        prim = primary_node.get(rid)
        reg_nodes = [r["node"] for r in regions]
        open_pairs = {frozenset((e["a"], e["b"])) for e in block_edges if e["kind"] == "open"}
        pass_through = bool(multi and len(regions) >= 2 and
                            sum(frozenset((prim, rn)) in open_pairs for rn in reg_nodes) >= 2)
        isolated = []
        for reg in regions:
            has_open = any(reg["node"] in p for p in open_pairs)
            reg_doors = [d for i in reg["part_indices"] for d in door_ids[i]]
            if not has_open and not reg_doors:
                isolated.append(reg["node"])

        out_parts = []
        for i, pt in enumerate(parts):
            r = role.get(i)
            if r in ("whole", "primary", "zone", "jog", "alcove"):
                lab = rlabel
            elif r == "residual":
                lab = "residual"
            else:
                lab = r
            op = {"index": i, "role": r, "label": lab, "door_ids": door_ids[i],
                  "region": region_of.get(i, region_of.get(absorbed_into[i]) if absorbed_into[i] is not None else None),
                  "node": node[i], "absorbed_into": absorbed_into[i],
                  "short_m": dims[i][0], "long_m": dims[i][1], "ext_m": ext_m[i]}
            if transferred[i]:
                op["door_ids_transferred"] = transferred[i]
            if i in cover:
                op["proj_cover"] = cover[i]
            if i == main and multi:
                op["pass_through"] = pass_through
            out_parts.append(op)
            if r in counts:
                counts[r] += 1
        n_reg += len(regions)
        n_pass += pass_through
        n_iso += len(isolated)
        rooms_out.append({"id": rid, "label": rlabel, "parts": out_parts, "regions": regions,
                          "pass_through": pass_through, "isolated_circulation": isolated,
                          "block_edges": block_edges, "wall_contacts": [], "kitchen_openable": None})

    # wall contacts + kitchen_openable
    labels = {r["id"]: r["label"] for r in rooms_out}
    for ro in rooms_out:
        rid = ro["id"]
        for nd in sorted({p["node"] for p in ro["parts"]}):
            ng = node_geom.get(nd)
            if ng is None or ng.is_empty:   # empty / degenerate room polygon (e.g. RPlan 67697 r_1)
                continue
            for oid, og in room_geom.items():
                if oid == rid or og is None or og.is_empty:
                    continue
                ln = _shared_len(node_geom[nd], og) * S
                if ln >= _MIN_M:
                    ro["wall_contacts"].append({"node": nd, "room_id": oid, "room_label": labels.get(oid),
                                                "length_m": ln})
        if len(ro["parts"]) > 1:
            kit = [c for c in ro["wall_contacts"] if "kitchen" in str(c["room_label"]).lower()]
            if kit:
                prim = primary_node.get(rid)
                tot = {}
                for c in kit:
                    if c["node"] == prim:
                        tot[c["room_id"]] = tot.get(c["room_id"], 0.0) + c["length_m"]
                ro["kitchen_openable"] = any(v >= 1.2 for v in tot.values())

    # dining candidates (v1.1): a zone the kitchen door faces, or that shares
    # >= dining_min_wall_m (0.85 m, door width) of wall with a kitchen, is labelled "dining?" -- only
    # when the plan has no DiningRoom. Label only: the zone stays on the primary
    # node. Contacts are measured per part (part + the parts absorbed into it),
    # because zones are folded into the primary node's geometry.
    # Not yet handled (topology pass, later): a zone that does not touch the
    # kitchen but sits between a dining candidate and the primary (RPlan 45549 p0).
    room_label_of = {r["id"]: str(r.get("label") or "") for r in plan.get("rooms", [])}
    has_dining = any("dining" in l.lower() for l in room_label_of.values())
    kitchen_ids = [k for k, l in room_label_of.items() if "kitchen" in l.lower()]
    kitchen_doors = {did for did, _, refs in openings if refs and refs[0] in kitchen_ids}
    n_dining = 0
    for ro in rooms_out:
        if len(ro["parts"]) < 2:
            continue
        rid = ro["id"]
        for p in ro["parts"]:
            if p["role"] in ("absorbed", "residual"):
                continue
            members = [polys[(rid, p["index"])]] + [polys[(rid, q["index"])] for q in ro["parts"]
                                                     if q["absorbed_into"] == p["index"]]
            geom = unary_union(members)
            kw = 0.0
            for k in kitchen_ids:
                kg = room_geom.get(k)
                if kg is not None and not kg.is_empty and k != rid:
                    kw += _shared_len(geom, kg) * S
            p["kitchen_wall_m"] = kw
            if p["role"] == "zone":
                kdoor = bool(set(p["door_ids"]) & kitchen_doors)
                p["dining_candidate"] = (not has_dining) and (kdoor or kw >= dining_min_wall_m)
                if p["dining_candidate"]:
                    p["label"] = "dining?"
                    n_dining += 1

    # connections
    connections, unresolved, open_unresolved = [], 0, 0
    for did, _, refs in openings:
        hits = [(k, facing[k][did]) for k in polys if did in facing[k]]
        if did == "entrance":
            if not hits:
                unresolved += 1
                continue
            k = max(hits, key=lambda h: h[1])[0]
            connections.append({"door_id": did, "kind": "door", "a": node_of[k], "b": "exterior"})
            continue
        if not refs:
            unresolved += 1
            continue
        ref = refs[0]
        own = [h for h in hits if h[0][0] == ref]
        a = node_of[max(own, key=lambda h: h[1])[0]] if own else primary_node.get(ref)
        other = [h for h in hits if h[0][0] != ref]
        if a is None or not other:
            unresolved += 1
            continue
        connections.append({"door_id": did, "kind": "door", "a": a,
                            "b": node_of[max(other, key=lambda h: h[1])[0]]})

    def side(rid, other_geom):
        best, bl = None, 0.0
        for nd, g in node_geom.items():
            if nd == rid or nd.startswith(rid + "/"):
                ln = _shared_len(g, other_geom) * S
                if ln > bl:
                    best, bl = nd, ln
        return best if bl >= _MIN_M else None

    for e in plan.get("edges", []):
        if e.get("connection") != 2:
            continue
        u, v = e.get("u"), e.get("v")
        if room_geom.get(u) is None or room_geom.get(v) is None:
            open_unresolved += 1
            continue
        a, b = side(u, room_geom[v]), side(v, room_geom[u])
        if a is None or b is None:
            open_unresolved += 1
            continue
        connections.append({"door_id": None, "kind": "open", "edge": [u, v], "a": a, "b": b})

    summary = {f"n_{k}": v for k, v in counts.items()}
    summary.update({"n_regions": n_reg, "n_pass_through": n_pass, "n_isolated_circulation": n_iso,
                    "n_dining_candidate": n_dining, "n_jog_projecting": n_jog_proj,
                    "n_door_unresolved": unresolved, "n_open_unresolved": open_unresolved})
    return {"params": params, "rooms": rooms_out, "connections": connections, "summary": summary}
