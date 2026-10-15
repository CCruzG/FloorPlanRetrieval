"""
rectangular_arrange.py — Step 5 of the arrangement refactor
(docs/arrangement-refactor-build-plan.md).

The placement engine that REPLACES ARAP's shape-preserving warp. Where the
align -> solve -> rectify -> fill chain deforms the retrieved plan's mesh to
the target boundary, this re-DIMENSIONS the retrieved plan's *topology*: the
arrangement chosen by retrieval is realised as a fresh axis-aligned partition
of the target boundary, not a warp of the source geometry.

The route (approach C, chosen 2026-08-21): retrieval already handed us a
realised rectangular arrangement — its rooms tile a region and its relation
codes / bboxes give the combinatorial structure. So we do NOT generate a
rectangular dual from the abstract adjacency graph. We:

  1. extract the shared grid implied by the source room bboxes (the maximal
     vertical / horizontal separators);
  2. assign each grid cell to the room that actually covers it (by max polygon
     overlap, so an L-shaped source room keeps its notch instead of claiming
     it via its bbox);
  3. absorb any unclaimed cell into a flexible neighbour (living, then dining)
     — "living takes the irregular residual";
  4. re-solve the grid-line positions to fill the TARGET boundary's bounding
     box, enforcing each room's minimum edge via a longest-path solve and
     handing the leftover slack to the flexible rooms;
  5. clip each room to the actual boundary polygon, so a non-rectangular
     boundary (an L, a notch) is followed rather than overflowed. The bbox grid
     assigns every point to exactly one room and the boundary is a subset of
     the bbox, so the clip is an exact partition of the boundary — rooms lying
     in the notch shrink or drop out (reported), never overlap.

By construction the output is always a valid partition (rooms are unions of
cells of one shared grid — overlaps and holes are impossible once cells are
assigned), so the only geometric failure a room can have is being below its
minimum size, which is reported rather than prevented. The one thing that can
defeat the method is a source arrangement that is not grid-alignable (a cell
claimed by two rooms at >50% each); that returns mode="fallback" with the
source boxes scaled into the boundary and marked, per the build plan's
"start with a fallback path".

Public API:
    arrange(rooms_mm, boundary_poly_mm, source_edges, entrance_mid) -> dict
"""

import sys
from pathlib import Path

from shapely.geometry import Polygon as SPoly, box as sbox, LineString
from shapely.ops import unary_union

# schema lives in the fitting-prototype dir, which arap_bridge puts on sys.path
# at import time; guard the import so this module is also importable stand-alone.
_FP_DIR = Path(
    "/Users/camilo/Documents/Work_FBI/Artefacts/project-44/code/fitting-prototype"
)
if str(_FP_DIR) not in sys.path:
    sys.path.insert(0, str(_FP_DIR))
from schema import MINIMUM_SIZES  # noqa: E402

# Flexibility tiers (build plan Step 5): living/dining flexible, kitchen semi,
# circulation absorbs, everything else rigid. Used two ways — a flexible room
# absorbs unclaimed cells, and slack in the min-size solve is handed to the
# columns/rows a flexible room occupies so it expands rather than the rigid rooms.
FLEX = {"living": 3.0, "dining": 3.0, "hallway": 2.0, "kitchen": 1.5}
FLEX_DEFAULT = 1.0
# Absorption preference for an unclaimed cell — the residual should land in the
# living room first, so it carries the irregularity.
ABSORB_PRIORITY = ["living", "dining", "hallway", "kitchen"]

EPS_MM = 500.0  # minimum width/height of any grid gap, so no sliver columns

# Daylight tiers (Stage 2 of the re-solve loop). Bedrooms/study need DIRECT
# daylight — their own window on an EXTERNAL boundary edge. Common areas may
# borrow SECONDARY daylight: a living/dining/kitchen is lit if it, OR an adjacent
# common-area room, touches an external edge (open-plan light sharing). Only one
# hop — light does not leak down an arbitrarily long chain of internal rooms.
# Bathrooms, circulation and storage are not daylight-assessed.
COMMON_AREA = {"living", "dining", "kitchen", "living_room"}
STRICT_DAYLIGHT = {"bedroom", "master_bedroom", "study", "family"}

# Access tiers (Stage 2b of the re-solve loop). A PRIVATE room must reach the
# unit's circulation: either it fronts the boundary's facing-corridor edge (the
# entrance side) directly, or it is adjacent to an open-plan / circulation
# PROVIDER (living/dining/kitchen/hallway). One-hop adjacency to an open-plan
# space counting as access (rule confirmed 2026-08-18) is what stops a bedroom
# that opens onto the living room being false-flagged — the false positive a
# naive door-to-corridor test makes. This mirrors vlm_refine.surface_defects on
# the arrange grid, reading realised adjacency instead of shared-wall geometry.
ACCESS_PRIVATE = {"bedroom", "master_bedroom", "bathroom", "storage"}
ACCESS_PROVIDERS = {"living", "dining", "kitchen", "hallway", "living_room"}
DOOR_MM = 800.0  # a wall shorter than this cannot carry a door


def _external_edge_line(boundary_poly_mm, edge_types):
    """Union of the boundary's EXTERNAL edges as a shapely line, or None when no
    edge types are supplied (daylight then goes unchecked). edge_types is a
    per-edge-index list aligned to boundary_poly_mm."""
    if not edge_types:
        return None
    n = len(boundary_poly_mm)
    segs = [LineString([boundary_poly_mm[i], boundary_poly_mm[(i + 1) % n]])
            for i, t in enumerate(edge_types) if t == "external" and i < n]
    return unary_union(segs) if segs else None


def _corridor_edge_line(boundary_poly_mm, edge_types):
    """Union of the boundary's FACING_CORRIDOR edges (the unit's front, off the
    shared corridor) as a shapely line, or None when no edge types are supplied.
    A private room fronting one of these has direct access. Aligned to
    boundary_poly_mm by edge index, like _external_edge_line."""
    if not edge_types:
        return None
    n = len(boundary_poly_mm)
    segs = [LineString([boundary_poly_mm[i], boundary_poly_mm[(i + 1) % n]])
            for i, t in enumerate(edge_types) if t == "facing_corridor" and i < n]
    return unary_union(segs) if segs else None


def _touches_external(geom, ext_line, tol_mm=50.0):
    """True if the room shares a boundary segment (not just a point) with an
    external edge — i.e. it can carry a window."""
    if ext_line is None or ext_line.is_empty:
        return False
    shared = geom.boundary.intersection(ext_line)
    return getattr(shared, "length", 0.0) > tol_mm


def _label(room: dict) -> str:
    return room.get("label", room.get("type", "unknown"))


def _flex(label: str) -> float:
    return FLEX.get(label, FLEX_DEFAULT)


def _bbox(poly) -> tuple:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _cluster(values: list, tol: float) -> list:
    """Sorted representative lines from a bag of coordinates: any run within
    `tol` collapses to its mean. This is what turns the source rooms' many
    near-coincident edges into one clean set of grid separators."""
    if not values:
        return []
    vs = sorted(values)
    groups = [[vs[0]]]
    for v in vs[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def _snap_index(value: float, lines: list) -> int:
    return min(range(len(lines)), key=lambda i: abs(lines[i] - value))


def _longest_path_positions(n_sep: int, spans: list, total: float,
                            gap_flex: list) -> tuple:
    """Solve separator positions along one axis.

    n_sep separators (0..n_sep-1 gaps). `spans` is a list of
    (lo, hi, min_size) constraints meaning position[hi] - position[lo] >=
    min_size (a room occupying grid columns [lo, hi)). Returns
    (positions, all_min_met): positions in [0, total], every gap >= EPS_MM,
    every span >= its min_size when the boundary is large enough; if it is
    not, positions are the minimal feasible layout scaled down to `total` and
    all_min_met is False."""
    m = n_sep  # number of separators including both ends is m+1
    # Minimal feasible positions by longest path in topological (left-to-right)
    # order: every span (lo, hi) constrains a separator to the right of an
    # earlier one, so processing separators in index order finalises each.
    pos = [0.0] * (m + 1)
    spans_by_hi = {}
    for lo, hi, mn in spans:
        spans_by_hi.setdefault(hi, []).append((lo, mn))
    for k in range(1, m + 1):
        pos[k] = pos[k - 1] + EPS_MM
        for lo, mn in spans_by_hi.get(k, []):
            pos[k] = max(pos[k], pos[lo] + mn)
    min_total = pos[m]

    if min_total <= total:
        # Room minimums all fit — distribute the slack to the gaps a flexible
        # room occupies, so living/dining expand and the rigid rooms stay near
        # their minimum. Result is a layout, not a uniform grid.
        slack = total - min_total
        wsum = sum(gap_flex) or 1.0
        widths = [pos[k + 1] - pos[k] for k in range(m)]
        widths = [w + slack * (gf / wsum) for w, gf in zip(widths, gap_flex)]
        out = [0.0]
        for w in widths:
            out.append(out[-1] + w)
        # Guard against float drift so the last separator sits exactly on the
        # boundary.
        out[-1] = total
        return out, True

    # Boundary too small for the minimums — scale the minimal layout down and
    # report that some rooms are under-size (they will be flagged per-room).
    scale = total / min_total if min_total > 0 else 1.0
    return [p * scale for p in pos], False


def _rotate_about(x, y, cx, cy, k):
    """Point (x, y) rotated k×90° about centre (cx, cy). k is taken mod 4."""
    dx, dy = x - cx, y - cy
    k &= 3
    if k == 1:
        dx, dy = -dy, dx
    elif k == 2:
        dx, dy = -dx, -dy
    elif k == 3:
        dx, dy = dy, -dx
    return cx + dx, cy + dy


def _orient_rooms_to_entrance(rooms, entrance_mid, target_entrance, target_center):
    """Rotate the source rooms by one of the four 90° rotations so the source
    plan's entrance points the same way, relative to the plan's centre, as the
    target boundary's entrance does relative to the boundary's centre.

    This is the arrangement-level analogue of similarity_align on the ARAP path,
    which arrange() otherwise skips: without it the retrieved plan is placed in
    its raw RPlan orientation and the entrance lands on whatever side the source
    happened to use. arrange() re-dimensions the result into the target bbox
    afterwards, so ONLY orientation matters here — translation and scale are
    handled downstream. Reflection is deliberately excluded: it is applied
    upstream by the retrieval mirror flag, and a single entrance direction cannot
    disambiguate a reflected copy from an unreflected one. Returns (rooms, k)
    with k the applied rotation in 90° steps (0 = identity, used whenever either
    entrance is missing or sits on the centre)."""
    if not entrance_mid or not target_entrance:
        return rooms, 0
    pts = [p for r in rooms for p in r["polygon"]]
    scx = (min(p[0] for p in pts) + max(p[0] for p in pts)) / 2
    scy = (min(p[1] for p in pts) + max(p[1] for p in pts)) / 2
    sdx, sdy = entrance_mid[0] - scx, entrance_mid[1] - scy
    tdx, tdy = target_entrance[0] - target_center[0], target_entrance[1] - target_center[1]
    if sdx * sdx + sdy * sdy < 1.0 or tdx * tdx + tdy * tdy < 1.0:
        return rooms, 0
    # Pick the rotation whose rotated source-entrance direction best agrees with
    # the target-entrance direction (max dot product; |source dir| is constant
    # across rotations, so no normalisation is needed for the argmax).
    best_k, best_dot = 0, None
    for k in range(4):
        rx, ry = _rotate_about(sdx, sdy, 0.0, 0.0, k)
        dot = rx * tdx + ry * tdy
        if best_dot is None or dot > best_dot:
            best_k, best_dot = k, dot
    if best_k == 0:
        return rooms, 0
    oriented = []
    for r in rooms:
        nr = dict(r)
        nr["polygon"] = [_rotate_about(px, py, scx, scy, best_k)
                         for (px, py) in r["polygon"]]
        oriented.append(nr)
    return oriented, best_k


def arrange(rooms_mm: list, boundary_poly_mm: list,
            source_edges: list | None = None, entrance_mid=None,
            target_entrance=None, edge_types=None) -> dict:
    """Place the retrieved arrangement into the target boundary as a rectangular
    partition. rooms_mm: [{id, label/type, polygon(mm)}] from the raw retrieved
    plan (NOT ARAP output). boundary_poly_mm: target boundary polygon in mm.
    source_edges: the plan's relation-coded adjacency edges, used only to report
    which adjacencies survived. Returns rooms in the same {id, type, polygon(mm)}
    shape the canvas draws, plus a report."""
    rooms = [r for r in rooms_mm if len(r.get("polygon", [])) >= 3]
    if not rooms:
        return {"rooms": [], "feasible": False, "mode": "fallback",
                "report": {"reason": "no source rooms"}}

    bx0, by0, bx1, by1 = _bbox(boundary_poly_mm)
    BW, BH = bx1 - bx0, by1 - by0

    # ── 0. orient the source to the target entrance ──────────────────────────
    # arrange() skips the ARAP chain, and with it similarity_align — so unless we
    # orient here, the plan is placed in its raw RPlan orientation. Rotate it so
    # its entrance faces the way the target entrance does before extracting the
    # grid; everything below then works in the oriented frame.
    rooms, rotation_k = _orient_rooms_to_entrance(
        rooms, entrance_mid, target_entrance, ((bx0 + bx1) / 2, (by0 + by1) / 2))

    # ── 1. shared grid from source bboxes ────────────────────────────────────
    src_boxes = {r["id"]: _bbox(r["polygon"]) for r in rooms}
    sx0 = min(b[0] for b in src_boxes.values())
    sy0 = min(b[1] for b in src_boxes.values())
    sx1 = max(b[2] for b in src_boxes.values())
    sy1 = max(b[3] for b in src_boxes.values())
    tol_x = 0.04 * max(sx1 - sx0, 1.0)
    tol_y = 0.04 * max(sy1 - sy0, 1.0)
    xs = _cluster([b[0] for b in src_boxes.values()] + [b[2] for b in src_boxes.values()], tol_x)
    ys = _cluster([b[1] for b in src_boxes.values()] + [b[3] for b in src_boxes.values()], tol_y)
    # Force the extremes to the true source extent so the grid spans everything.
    xs[0], xs[-1] = sx0, sx1
    ys[0], ys[-1] = sy0, sy1
    ncol, nrow = len(xs) - 1, len(ys) - 1
    if ncol < 1 or nrow < 1:
        return _fallback(rooms, src_boxes, sx0, sy0, sx1, sy1, bx0, by0, BW, BH,
                         "degenerate grid", rotation_k)

    # ── 2. assign each cell to the room covering it most ─────────────────────
    src_polys = {r["id"]: SPoly(r["polygon"]) for r in rooms
                 if SPoly(r["polygon"]).is_valid}
    owner = {}           # (i, j) -> room id
    holes = []           # unclaimed cells
    overlaps = 0
    for i in range(ncol):
        for j in range(nrow):
            cell = sbox(xs[i], ys[j], xs[i + 1], ys[j + 1])
            carea = cell.area or 1.0
            best_id, best_frac = None, 0.0
            claimants = 0
            for rid, poly in src_polys.items():
                try:
                    frac = cell.intersection(poly).area / carea
                except Exception:
                    frac = 0.0
                if frac > 0.5:
                    claimants += 1
                if frac > best_frac:
                    best_id, best_frac = rid, frac
            if claimants > 1:
                overlaps += 1
            if best_frac > 0.5:
                owner[(i, j)] = best_id
            else:
                holes.append((i, j))

    # A source arrangement that is not grid-alignable shows up as cells two
    # rooms both fill — the one case approach C cannot realise. Fall back.
    if overlaps > 0:
        return _fallback(rooms, src_boxes, sx0, sy0, sx1, sy1, bx0, by0, BW, BH,
                         f"{overlaps} non-grid-alignable cell(s)", rotation_k)

    # ── 3. absorb unclaimed cells into a flexible neighbour ──────────────────
    absorbed = 0
    for (i, j) in holes:
        nbr = _flexible_neighbour(i, j, owner, ncol, nrow, rooms)
        if nbr is not None:
            owner[(i, j)] = nbr
            absorbed += 1

    # Rooms that ended up with no cell at all (fully squeezed out) — keep them
    # out of the partition but report them.
    cells_by_room = {}
    for (i, j), rid in owner.items():
        cells_by_room.setdefault(rid, []).append((i, j))
    dropped_rooms = [r["id"] for r in rooms if r["id"] not in cells_by_room]

    # ── 4. re-solve grid-line positions to fill the target boundary ──────────
    label_by_id = {r["id"]: _label(r) for r in rooms}
    x_spans, y_spans = [], []
    for rid, cells in cells_by_room.items():
        cmin = min(c[0] for c in cells)
        cmax = max(c[0] for c in cells) + 1
        rmin = min(c[1] for c in cells)
        rmax = max(c[1] for c in cells) + 1
        me = MINIMUM_SIZES.get(label_by_id[rid], {}).get("min_edge", EPS_MM)
        x_spans.append((cmin, cmax, me))
        y_spans.append((rmin, rmax, me))

    col_flex = [max([_flex(label_by_id[owner[(i, j)]])
                     for j in range(nrow) if (i, j) in owner] or [FLEX_DEFAULT])
                for i in range(ncol)]
    row_flex = [max([_flex(label_by_id[owner[(i, j)]])
                     for i in range(ncol) if (i, j) in owner] or [FLEX_DEFAULT])
                for j in range(nrow)]

    xt, x_ok = _longest_path_positions(ncol, x_spans, BW, col_flex)
    yt, y_ok = _longest_path_positions(nrow, y_spans, BH, row_flex)
    xt = [bx0 + v for v in xt]
    yt = [by0 + v for v in yt]

    # ── 5. build room polygons from their cells, clipped to the boundary ─────
    # The solve fills the boundary's bounding box; a non-rectangular boundary
    # (an L, a notch) leaves cells lying outside the real polygon. Clipping each
    # room to the boundary makes the partition follow the actual outline instead
    # of overflowing the bbox. Because the bbox grid assigns every point to
    # exactly one room and the boundary is a subset of the bbox, the clip is an
    # EXACT partition of the boundary — no interior gaps, no overlaps; the notch
    # is simply uncovered (it is outside the unit). A room sitting mostly in the
    # notch shrinks or vanishes, which is honest signal that the retrieved
    # arrangement does not fit this boundary shape.
    #
    # Only clip when the boundary is not its own bbox — a rectangular unit needs
    # no clip, so the common case is untouched and cannot regress.
    bpoly = SPoly(boundary_poly_mm)
    if not bpoly.is_valid:
        bpoly = bpoly.buffer(0)
    clip_needed = bpoly.area < BW * BH - 1.0

    # Daylight check (Stage 2): a habitable room must touch an external edge.
    # Only evaluated when edge types are supplied; unset → not checked.
    do_daylight = edge_types is not None
    do_access = edge_types is not None
    ext_line = _external_edge_line(boundary_poly_mm, edge_types)
    corr_line = _corridor_edge_line(boundary_poly_mm, edge_types)

    # A room whose cells frame another room unions to a polygon with a hole.
    # The canvas draws single rings, so a holed ("donut") room can't be shown
    # faithfully — _largest_ring keeps the exterior, which fills the hole and
    # would overlap the enclosed room. These enclosures are counted and
    # reported (and drop feasibility for Step 6's guard); drawing the enclosed,
    # smaller rooms last keeps the picture correct for the visual gate.
    out_rooms, room_report, enclosures, clip_splits, clipped_out = [], [], 0, 0, []
    direct_by_id = {}   # room id -> touches an external edge (direct daylight)
    geom_by_id = {}     # room id -> clipped shapely geom (for the access sensor)
    for r in rooms:
        rid = r["id"]
        cells = cells_by_room.get(rid)
        if not cells:
            continue
        rects = [sbox(xt[i], yt[j], xt[i + 1], yt[j + 1]) for (i, j) in cells]
        merged = unary_union(rects)
        # Enclosure is a property of the pre-clip topology (a room framing
        # another → hole or split); detect it before clipping.
        is_enclosure = (
            (getattr(merged, "interiors", None) and len(merged.interiors) > 0)
            or merged.geom_type == "MultiPolygon")
        geom = merged.intersection(bpoly) if clip_needed else merged
        if geom.is_empty or geom.area < 1.0:
            # The boundary shape squeezed this room out entirely.
            clipped_out.append(rid)
            continue
        if is_enclosure:
            enclosures += 1
        elif geom.geom_type == "MultiPolygon":
            # Clipping split the room across the notch; the canvas draws one
            # ring, so _largest_ring keeps the biggest part and we flag it.
            clip_splits += 1
        poly = _largest_ring(geom)
        label = label_by_id[rid]
        area_m2 = geom.area / 1e6
        min_m2 = MINIMUM_SIZES.get(label, {}).get("area", 0) / 1e6
        # Direct daylight only here; secondary sharing needs adjacency, resolved
        # after the loop once realised_adj is known.
        direct_by_id[rid] = _touches_external(geom, ext_line) if do_daylight else None
        geom_by_id[rid] = geom
        out_rooms.append({"id": rid, "type": label, "polygon": poly,
                          "_area": geom.area})
        room_report.append({
            "id": rid, "label": label,
            "area_m2": round(area_m2, 1), "min_area_m2": round(min_m2, 1),
            "ok": area_m2 + 1e-6 >= min_m2,
            "daylight_ok": None,   # resolved below
            "access_ok": None,     # resolved below
        })

    # Largest first so any enclosed room draws on top of the room framing it.
    out_rooms.sort(key=lambda r: r.pop("_area"), reverse=True)

    realised_adj = _realised_adjacency(owner, ncol, nrow)
    dropped_adj = _dropped_adjacency(source_edges, rooms, realised_adj)

    # Realised adjacency as id -> neighbour set, shared by the daylight and
    # access sensors below.
    adj = {}
    for pair in realised_adj:
        a, b = tuple(pair)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    # ── daylight resolution (needs adjacency) ────────────────────────────────
    # Strict rooms (bedrooms/study) need their own direct light; common areas
    # may borrow secondary light from an adjacent common area that has direct.
    no_daylight = []
    if do_daylight:
        common_direct = {rid for rid, d in direct_by_id.items()
                         if d and label_by_id[rid] in COMMON_AREA}
        for rr in room_report:
            rid, label = rr["id"], rr["label"]
            direct = direct_by_id.get(rid)
            if label in STRICT_DAYLIGHT:
                ok = bool(direct)
            elif label in COMMON_AREA:
                ok = bool(direct) or any(nb in common_direct for nb in adj.get(rid, ()))
            else:
                continue   # not daylight-assessed; leave daylight_ok = None
            rr["daylight_ok"] = ok
            if not ok:
                no_daylight.append(label)

    # ── access resolution (needs adjacency) ──────────────────────────────────
    # A private room (bedroom/bath/storage) reaches circulation if it fronts the
    # facing-corridor edge directly, or is adjacent to an open-plan / circulation
    # provider. One-hop open-plan adjacency counting as access (2026-08-18) is
    # what keeps a bedroom off the living room from being false-flagged.
    no_access = []
    if do_access:
        for rr in room_report:
            rid, label = rr["id"], rr["label"]
            if label not in ACCESS_PRIVATE:
                continue   # providers self-serve; not access-assessed
            served = _touches_external(geom_by_id[rid], corr_line, tol_mm=DOOR_MM)
            if not served:
                served = any(label_by_id[nb] in ACCESS_PROVIDERS
                             for nb in adj.get(rid, ()))
            rr["access_ok"] = served
            if not served:
                no_access.append(label)

    mode = "clean" if absorbed == 0 else "absorbed"
    if clipped_out or clip_splits:
        mode = "clipped"
    feasible = (x_ok and y_ok and not dropped_rooms and not clipped_out
                and enclosures == 0 and clip_splits == 0 and not no_daylight
                and not no_access)
    return {
        "rooms": out_rooms,
        "feasible": feasible,
        "mode": mode,
        "report": {
            "grid": [ncol, nrow],
            "rotation_deg": rotation_k * 90,
            "cells_absorbed": absorbed,
            "min_sizes_met": x_ok and y_ok,
            "enclosures": enclosures,
            "clipped_to_boundary": clip_needed,
            "clip_splits": clip_splits,
            "clipped_out_rooms": clipped_out,
            "dropped_rooms": dropped_rooms,
            "dropped_adjacencies": dropped_adj,
            "no_daylight": no_daylight,
            "no_access": no_access,
            "rooms": room_report,
        },
    }


def _flexible_neighbour(i, j, owner, ncol, nrow, rooms):
    """The room to absorb an unclaimed cell into: an edge-adjacent owned cell,
    preferring the most flexible occupant (living first)."""
    label_by_id = {r["id"]: _label(r) for r in rooms}
    cand = []
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = i + di, j + dj
        if 0 <= ni < ncol and 0 <= nj < nrow and (ni, nj) in owner:
            cand.append(owner[(ni, nj)])
    if not cand:
        return None
    def rank(rid):
        lab = label_by_id.get(rid, "")
        return ABSORB_PRIORITY.index(lab) if lab in ABSORB_PRIORITY else len(ABSORB_PRIORITY)
    return min(cand, key=rank)


def _realised_adjacency(owner, ncol, nrow) -> set:
    adj = set()
    for i in range(ncol):
        for j in range(nrow):
            a = owner.get((i, j))
            if a is None:
                continue
            for di, dj in ((1, 0), (0, 1)):
                b = owner.get((i + di, j + dj))
                if b is not None and b != a:
                    adj.add(frozenset((a, b)))
    return adj


def _dropped_adjacency(source_edges, rooms, realised_adj) -> list:
    if not source_edges:
        return []
    label_by_id = {r["id"]: _label(r) for r in rooms}
    ids = set(label_by_id)
    dropped = []
    for e in source_edges:
        u, v = e.get("u"), e.get("v")
        if u in ids and v in ids and frozenset((u, v)) not in realised_adj:
            dropped.append(f"{label_by_id[u]}–{label_by_id[v]}")
    return dropped


def _largest_ring(geom) -> list:
    """Exterior coords of a (possibly multi-part) shapely geometry as an
    (x, y) list — flexible rooms can be rectilinear, so return the outer ring
    of the largest part. Boundary clipping can also yield a MultiPolygon or a
    GeometryCollection (with stray edges/points from shared boundaries), so keep
    only the polygonal parts and take the largest."""
    if geom.geom_type == "Polygon":
        return [(x, y) for x, y in geom.exterior.coords[:-1]]
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    if not parts:
        return []
    parts.sort(key=lambda p: p.area, reverse=True)
    return [(x, y) for x, y in parts[0].exterior.coords[:-1]]


def _fallback(rooms, src_boxes, sx0, sy0, sx1, sy1, bx0, by0, BW, BH, reason,
              rotation_k=0) -> dict:
    """Non-grid-alignable source — scale the source boxes proportionally into
    the target boundary bbox so the canvas shows the arrangement and the
    failure is visible, marked mode='fallback'."""
    sw = max(sx1 - sx0, 1.0)
    sh = max(sy1 - sy0, 1.0)
    out_rooms = []
    for r in rooms:
        x0, y0, x1, y1 = src_boxes[r["id"]]
        nx0 = bx0 + (x0 - sx0) / sw * BW
        nx1 = bx0 + (x1 - sx0) / sw * BW
        ny0 = by0 + (y0 - sy0) / sh * BH
        ny1 = by0 + (y1 - sy0) / sh * BH
        out_rooms.append({
            "id": r["id"], "type": _label(r),
            "polygon": [(nx0, ny0), (nx1, ny0), (nx1, ny1), (nx0, ny1)],
        })
    return {
        "rooms": out_rooms, "feasible": False, "mode": "fallback",
        "report": {"reason": reason, "rotation_deg": rotation_k * 90},
    }
