"""
topo_signature.py — Step 1 of the topology → generate → select build
(docs/topology-generate-select-build-plan.md).

Scores a retrieved plan by the fitness of its TOPOLOGY to the *target boundary*,
not by shape similarity. The discriminator is a **perimeter-ring alignment**
(approach B, 2026-09-03): each plan is augmented with its room↔boundary-edge
adjacency computed GEOMETRICALLY (which rooms sit on which boundary arcs) — never
from the dataset's `windows`, which are sprinkled on exterior edges by the
generation tool and do not carry design intent. Both the source plan and the
target boundary are reduced to a cyclic ring anchored at the entrance; the
source's daylight demand (where bedrooms/living sit on the perimeter) is
arc-aligned against the target's typed edges. A topology is suitable when the
rooms that need a window land on `external` arcs of the target.

This is the boundary-coupled signal retrieval was missing: intrinsic quality
saturates on RPlan (real apartments are uniformly well-formed), so the question
worth asking is not "is this a good plan" but "does this plan's perimeter demand
match THIS boundary's external supply."

Public API:
    topological_fitness(plan, target_boundary, target_edge_types=None,
                        entrance_point=None, requirements=None) -> dict
    default_edge_types(boundary, entrance_point) -> list[str]
"""

import math
import sys
from pathlib import Path

from shapely.geometry import Polygon as SPoly, Point as SPoint
from shapely.ops import nearest_points

_FP_DIR = Path(
    "/Users/camilo/Documents/Work_FBI/Artefacts/project-44/code/fitting-prototype"
)
if str(_FP_DIR) not in sys.path:
    sys.path.insert(0, str(_FP_DIR))
from schema import ROOM_TYPE_MAP  # noqa: E402

# Room categories, aligned to rectangular_arrange's daylight/access tiers so
# retrieval and the placed-arrangement sensors judge by one rule set.
DAYLIGHT_NEEDING = {"living", "living_room", "master_bedroom", "bedroom", "study"}
BEDROOMS = {"master_bedroom", "bedroom", "study"}
COMMON = {"living", "dining", "kitchen", "living_room"}
CIRCULATION = {"hallway", "corridor", "entrance", "foyer"}
TRANSIT = COMMON | CIRCULATION
PRIVATE = {"bedroom", "master_bedroom", "study", "bathroom", "storage"}

EXTERNAL = "external"
CORRIDOR = "facing_corridor"
SHARED = "shared_wall"

N_SAMPLES = 96          # perimeter sampling resolution
# A daylight room counts as served when at least this fraction of its perimeter
# presence lands on external target arcs — enough for a real window, not a sliver.
DAYLIGHT_MIN_EXTERNAL_FRAC = 0.34

WEIGHTS = {
    "program": 0.25,
    "daylight_fit": 0.50,   # the boundary-coupled discriminator
    "entrance_access": 0.10,
    "private_access": 0.15,
}


# ── labels / graph (intrinsic terms) ─────────────────────────────────────────

def _label(room: dict) -> str:
    t = room.get("type")
    if isinstance(t, int) and t in ROOM_TYPE_MAP:
        return ROOM_TYPE_MAP[t]
    return str(room.get("label", "")).strip().lower().replace(" ", "_")


def _adjacency(plan: dict) -> dict:
    adj = {}
    for e in plan.get("edges") or []:
        u, v = e.get("u"), e.get("v")
        if u is None or v is None:
            continue
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def _reachable_through_transit(adj, label_by_id, start):
    if start is None:
        return set()
    reachable = {start}
    queue = [start]
    while queue:
        cur = queue.pop()
        if cur != start and label_by_id.get(cur) not in TRANSIT:
            continue
        for nb in adj.get(cur, ()):
            if nb not in reachable:
                reachable.add(nb)
                queue.append(nb)
    return reachable


def _program_match(counts, requirements):
    if not requirements:
        return 1.0
    req_bed = int(requirements.get("bedrooms", 2) or 0)
    req_bath = int(requirements.get("bathrooms", 1) or 0)
    plan_bed = counts.get("master_bedroom", 0) + counts.get("bedroom", 0)
    plan_bath = counts.get("bathroom", 0)
    bed_err = abs(plan_bed - req_bed) / max(req_bed, 1)
    bath_err = abs(plan_bath - req_bath) / max(req_bath, 1)
    return max(0.0, 1.0 - 0.5 * (bed_err + bath_err))


# ── perimeter ring (approach B) ──────────────────────────────────────────────

def _xy(pts):
    return [(float(p[0]), float(p[1])) for p in pts]


def _signed_area(pts):
    n = len(pts)
    return 0.5 * sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    )


def _oriented_edges(pts, edge_types=None):
    """Boundary edges as [p0, p1, length, type] in CCW order. Reversing to CCW
    keeps each type with its geometric edge, so a per-edge-index type list stays
    valid after normalisation."""
    n = len(pts)
    edges = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        length = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        t = edge_types[i] if (edge_types and i < len(edge_types)) else None
        # e[4] carries the RAW (as-given) polygon edge index, so it survives the
        # CW→CCW reversal below and any exposed edge index refers to the boundary
        # as drawn, not the re-wound copy (a CW boundary would otherwise map to
        # the wrong edge — both in room_edges connectors and boundary_capacity).
        edges.append([a, b, length, t, i])
    if _signed_area(pts) < 0:   # CW → reverse to CCW, swapping endpoints
        edges = [[e[1], e[0], e[2], e[3], e[4]] for e in reversed(edges)]
    return edges


def _nearest_edge_index(edges, point):
    """Index of the CCW edge whose midpoint is nearest `point` — the ring's
    anchor (entrance on the source, corridor on the target)."""
    px, py = point
    best_i, best_d = 0, float("inf")
    for i, e in enumerate(edges):
        mx, my = (e[0][0] + e[1][0]) / 2.0, (e[0][1] + e[1][1]) / 2.0
        d = (mx - px) ** 2 + (my - py) ** 2
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def _edge_outward_normal_angle(a, b):
    """Outward-normal angle of a CCW polygon edge a->b: edge direction rotated
    -90 degrees."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    return math.atan2(-dx, dy)


def _norm(angle, ref):
    """`angle` relative to `ref`, folded into [0, 2*pi)."""
    return (angle - ref) % (2 * math.pi)


def _circ_dist(x, y):
    d = (x - y) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def _ring_samples(edges, anchor_idx, n=N_SAMPLES):
    """n points sampled at equal arc-length around the perimeter, starting at the
    anchor edge, CCW. Returns list of (x, y, edge_type, orig_edge_index) — the
    last is the sample's edge index into `edges` before the anchor rotation,
    i.e. the plan's own boundary-polygon edge index."""
    n_edges = len(edges)
    ordered = edges[anchor_idx:] + edges[:anchor_idx]
    total = sum(e[2] for e in ordered) or 1.0
    starts, s = [], 0.0
    for e in ordered:
        starts.append(s)
        s += e[2]
    step = total / n
    out = []
    for k in range(n):
        d = k * step
        for j in range(len(ordered)):
            e = ordered[j]
            if d < starts[j] + e[2] or j == len(ordered) - 1:
                frac = (d - starts[j]) / (e[2] if e[2] > 1e-9 else 1.0)
                px = e[0][0] + frac * (e[1][0] - e[0][0])
                py = e[0][1] + frac * (e[1][1] - e[0][1])
                orig_idx = e[4]   # raw as-drawn boundary edge index (winding-invariant)
                out.append((px, py, e[3], orig_idx))
                break
    return out


def _entrance_point_of_plan(plan):
    ent = plan.get("entrance")
    if not ent or len(ent) < 4:
        return None
    return ((ent[0] + ent[2]) / 2.0, (ent[1] + ent[3]) / 2.0)


def _source_ring_rooms(plan, n=N_SAMPLES):
    """The source plan's perimeter as a ring of room ids, anchored at the
    entrance, CCW. Each sample is the room whose polygon sits on that boundary
    arc (nearest room) — the geometric room↔boundary augmentation, computed
    without touching the window annotations. Returns
    (room_ids, source_edge_idxs, label_by_id) — source_edge_idxs is, per
    sample, the edge index into the plan's own boundary polygon (pre-anchor-
    rotation) that sample sits on — or (None, None, {}) if the plan has no
    usable entrance/boundary."""
    bpts = _xy(plan.get("boundary") or [])
    if len(bpts) < 3:
        return None, None, {}
    ent_pt = _entrance_point_of_plan(plan)
    if ent_pt is None:
        return None, None, {}
    edges = _oriented_edges(bpts)
    anchor = _nearest_edge_index(edges, ent_pt)
    samples = _ring_samples(edges, anchor, n)

    rooms = [r for r in (plan.get("rooms") or []) if len(r.get("polygon", [])) >= 3]
    polys, ids, label_by_id = [], [], {}
    for r in rooms:
        try:
            polys.append(SPoly(r["polygon"]))
        except Exception:
            continue
        ids.append(r["id"])
        label_by_id[r["id"]] = _label(r)

    ring_ids, ring_edge_idxs = [], []
    for (px, py, _t, orig_idx) in samples:
        p = SPoint(px, py)
        best_id, best_d = None, float("inf")
        for rid, poly in zip(ids, polys):
            d = p.distance(poly)
            if d < best_d:
                best_id, best_d = rid, d
        ring_ids.append(best_id)
        ring_edge_idxs.append(orig_idx)
    return ring_ids, ring_edge_idxs, label_by_id


def default_edge_types(boundary, entrance_point):
    """Fallback typing when the target boundary carries none: every edge
    external except the one nearest the entrance, which faces the corridor —
    matching the arrange-side default so retrieval and generation agree."""
    bpts = _xy(boundary)
    n = len(bpts)
    types = [EXTERNAL] * n
    if entrance_point is not None:
        edges = _oriented_edges(bpts)  # CCW; index space differs, so type by geometry
        # find the raw edge nearest the entrance and mark corridor
        px, py = entrance_point[0], entrance_point[1]
        best_i, best_d = 0, float("inf")
        for i in range(n):
            a, b = bpts[i], bpts[(i + 1) % n]
            mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
            d = (mx - px) ** 2 + (my - py) ** 2
            if d < best_d:
                best_i, best_d = i, d
        types[best_i] = CORRIDOR
    return types


def _room_src_edge_fracs(ring_ids, ring_edge_idxs):
    """Per room: {source_raw_edge_idx: sample_count}, and per-room sample
    totals — the source edges each room genuinely fronts (from the correct
    source-ring augmentation), independent of any target."""
    counts, totals = {}, {}
    for rid, eidx in zip(ring_ids, ring_edge_idxs):
        if rid is None:
            continue
        counts.setdefault(rid, {})
        counts[rid][eidx] = counts[rid].get(eidx, 0) + 1
        totals[rid] = totals.get(rid, 0) + 1
    return counts, totals


def _build_room_edges(room_edge_counts, room_totals, all_room_ids, edge_map):
    """room_edges by-product: for each room, one entry per fronted SOURCE
    boundary edge, mapped to a TARGET edge by direction (edge_map). Rooms
    with no perimeter presence get an empty list."""
    room_edges = {}
    for rid in all_room_ids:
        total = room_totals.get(rid, 0)
        entries = []
        if total:
            for eidx, cnt in sorted(room_edge_counts.get(rid, {}).items()):
                tidx, ttype = edge_map[eidx]
                entries.append({
                    "edge_index": eidx,
                    "target_edge_index": tidx,
                    "edge_type": ttype,
                    "arc_frac": round(cnt / total, 3),
                })
        room_edges[str(rid)] = entries
    return room_edges


# ── node placement in target frame (pre-processing for raycast connections) ──

def _polygon_centroid(pts):
    """Area-weighted polygon centroid, matching the frontend `bdryCenter`
    (falls back to the vertex mean for a degenerate ring)."""
    n = len(pts)
    if n == 0:
        return (0.5, 0.5)
    area = cx = cy = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area /= 2.0
    if abs(area) < 1e-10:
        return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)
    return (cx / (6 * area), cy / (6 * area))


def _bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _pt_seg_dist2(p, a, b):
    """Squared distance from point p to segment a-b."""
    px, py = p; ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    if ll == 0:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ll))
    return (px - (ax + t * dx)) ** 2 + (py - (ay + t * dy)) ** 2


def _display_transform(bpts, tpts, ent_pt, entrance_point, mirror):
    """The frontend display transform, in normalized coords, as two closures.

    mirror (y-flip about the source centroid) + a 90°-snapped entrance-alignment
    rotation, then a UNIFORM contain-scale (min axis ratio) about the source→
    target centroids. Identical to the frontend's drawFP/contain-scale math (a
    direction ignores the uniform scale + translation), so nodes, connectors and
    the drawn plan all agree. Returns (to_target(x,y)->[x,y], to_dir(dx,dy)->(dx,dy)).
    """
    c_s = _polygon_centroid(bpts)          # source centroid (mirror/rot pivot)
    c_t = _polygon_centroid(tpts)          # target centroid (contain pivot)
    rot = 0.0
    if ent_pt is not None and entrance_point is not None:
        sem_x, sem_y = ent_pt
        if mirror:
            sem_y = 2 * c_s[1] - sem_y     # mirror is applied before rotation
        p_dir = math.atan2(sem_y - c_s[1], sem_x - c_s[0])
        q_dir = math.atan2(entrance_point[1] - c_t[1], entrance_point[0] - c_t[0])
        half = math.pi / 2
        rot = round((q_dir - p_dir) / half) * half
    cos_r, sin_r = math.cos(rot), math.sin(rot)

    def orient(x, y):
        if mirror:
            y = 2 * c_s[1] - y
        dx, dy = x - c_s[0], y - c_s[1]
        return (c_s[0] + dx * cos_r - dy * sin_r,
                c_s[1] + dx * sin_r + dy * cos_r)

    def to_dir(dx, dy):
        if mirror:
            dy = -dy
        return (dx * cos_r - dy * sin_r, dx * sin_r + dy * cos_r)

    rot_bdry = [orient(x, y) for x, y in bpts]
    smin_x, smin_y, smax_x, smax_y = _bbox(rot_bdry)
    tmin_x, tmin_y, tmax_x, tmax_y = _bbox(tpts)
    s_w = max(smax_x - smin_x, 1e-9)
    s_h = max(smax_y - smin_y, 1e-9)
    scale = min((tmax_x - tmin_x) / s_w, (tmax_y - tmin_y) / s_h)

    def to_target(x, y):
        ox, oy = orient(x, y)
        return [c_t[0] + scale * (ox - c_s[0]),
                c_t[1] + scale * (oy - c_s[1])]

    return to_target, to_dir


def _place_nodes_in_target(plan, target_boundary, entrance_point, mirror):
    """Room centroids mapped into the TARGET boundary's frame (via the display
    transform), so every node lands INSIDE the boundary — the raycast connectors
    need a valid interior ray origin. Any centroid still outside the actual
    polygon (a non-convex notch) is pulled to the nearest interior point — the
    stray guard. Returns {str(room_id): [x, y]} in target space.
    """
    bpts = _xy(plan.get("boundary") or [])
    tpts = _xy(target_boundary or [])
    if len(bpts) < 3 or len(tpts) < 3:
        return {}
    to_target, _ = _display_transform(
        bpts, tpts, _entrance_point_of_plan(plan), entrance_point, mirror)

    tpoly = SPoly(tpts)
    if not tpoly.is_valid:
        tpoly = tpoly.buffer(0)
    rp = tpoly.representative_point()      # guaranteed strictly inside

    node_pos = {}
    for r in plan.get("rooms") or []:
        poly = r.get("polygon")
        if not poly or len(poly) < 3:
            continue
        mx = sum(p[0] for p in poly) / len(poly)
        my = sum(p[1] for p in poly) / len(poly)
        pt = to_target(mx, my)
        if not tpoly.contains(SPoint(pt)):
            near = nearest_points(tpoly, SPoint(pt))[0]   # nearest point of polygon
            pt = [near.x + 0.02 * (rp.x - near.x),        # nudge inward to interior
                  near.y + 0.02 * (rp.y - near.y)]
        node_pos[str(r.get("id"))] = [round(pt[0], 5), round(pt[1], 5)]
    return node_pos


def _raycast_room_edges(plan, target_boundary, entrance_point, mirror, target_edge_types):
    """Room↔target-edge adjacency by RAYCAST (2026-09-16), the single source both
    the drawn connectors and the daylight/capacity sensors consume.

    A room fronts the boundary along the outward normals of its EXTERIOR walls —
    room-polygon edges whose midpoint lies on the plan perimeter. A ray from the
    node (display-frame, inside the boundary) along each normal connects to the
    first target edge it crosses. Interior rooms (no exterior wall) get no entry.
    Returns {str(room_id): [{target_edge_index, edge_index, edge_type, arc_frac,
    target_point}]}. Replaces the direction-mapped ring connectors, which were
    computed in a different orientation and disagreed with the placed nodes.
    """
    bpts = _xy(plan.get("boundary") or [])
    tpts = _xy(target_boundary or [])
    if len(bpts) < 3 or len(tpts) < 3:
        return {}
    to_target, to_dir = _display_transform(
        bpts, tpts, _entrance_point_of_plan(plan), entrance_point, mirror)
    n = len(tpts)
    nb = len(bpts)
    types = target_edge_types or []
    EPS2 = 0.02 ** 2   # ponytail: plan-unit tolerance for "wall on plan perimeter"

    def ray_first_edge(px, py, dx, dy):
        best_u, best_i, best_pt = float("inf"), -1, None
        for i in range(n):
            ax, ay = tpts[i]
            bx, by = tpts[(i + 1) % n]
            v1x, v1y = px - ax, py - ay
            v2x, v2y = bx - ax, by - ay
            v3x, v3y = -dy, dx
            den = v2x * v3x + v2y * v3y
            if abs(den) < 1e-9:
                continue
            u = (v2x * v1y - v2y * v1x) / den       # distance along ray
            s2 = (v1x * v3x + v1y * v3y) / den       # param along segment
            if u >= 1e-6 and -1e-6 <= s2 <= 1 + 1e-6 and u < best_u:
                best_u, best_i, best_pt = u, i, [px + u * dx, py + u * dy]
        return best_i, best_pt

    room_edges = {}
    for r in plan.get("rooms") or []:
        poly = r.get("polygon")
        if not poly or len(poly) < 3:
            continue
        cxp = sum(p[0] for p in poly) / len(poly)
        cyp = sum(p[1] for p in poly) / len(poly)
        nx0, ny0 = to_target(cxp, cyp)               # ray origin = node
        fronts = {}   # tidx -> {"w": summed wall length, "maxw":.., "pt":[..]}
        m = len(poly)
        for k in range(m):
            ax, ay = poly[k]
            bx, by = poly[(k + 1) % m]
            midx, midy = (ax + bx) / 2.0, (ay + by) / 2.0
            on_perim = any(
                _pt_seg_dist2((midx, midy), bpts[j], bpts[(j + 1) % nb]) < EPS2
                for j in range(nb)
            )
            if not on_perim:
                continue
            nx, ny = -(by - ay), (bx - ax)           # outward normal (plan coords)
            L = math.hypot(nx, ny) or 1.0
            nx, ny = nx / L, ny / L
            if nx * (midx - cxp) + ny * (midy - cyp) < 0:
                nx, ny = -nx, -ny
            ddx, ddy = to_dir(nx, ny)
            tidx, hit = ray_first_edge(nx0, ny0, ddx, ddy)
            if tidx < 0:
                continue
            wlen = math.hypot(bx - ax, by - ay)
            cur = fronts.get(tidx) or {"w": 0.0, "maxw": -1.0, "pt": None}
            cur["w"] += wlen
            if wlen > cur["maxw"]:
                cur["maxw"] = wlen
                cur["pt"] = [round(hit[0], 5), round(hit[1], 5)]
            fronts[tidx] = cur
        if fronts:
            total = sum(v["w"] for v in fronts.values()) or 1.0
            room_edges[str(r.get("id"))] = [
                {
                    "target_edge_index": tidx,
                    "edge_index": tidx,
                    "edge_type": types[tidx] if 0 <= tidx < len(types) else EXTERNAL,
                    "arc_frac": round(v["w"] / total, 4),
                    "target_point": v["pt"],
                }
                for tidx, v in sorted(fronts.items())
            ]
    return room_edges


def _daylight_fit(plan, target_boundary, target_edge_types, entrance_point):
    """Fraction of the source's daylight-needing rooms that land on `external`
    edges of the target boundary, matched by OUTWARD-NORMAL DIRECTION rather
    than arc-length position — arc-length stretching doesn't preserve which
    way a room actually faces once the target shape differs from the source.

    Each source boundary edge is given an angle relative to the source's own
    entrance-anchor edge normal; each target edge likewise relative to the
    target's entrance-anchor edge normal. A source edge maps to whichever
    target edge has the closest normal angle. Tries this direct mapping and
    its mirror (reflection about the entrance anchor); returns
    (score, detail, mirror, room_edges)."""
    ring_ids, ring_edge_idxs, label_by_id = _source_ring_rooms(plan)
    if ring_ids is None:
        return 0.5, {"reason": "no source entrance"}, False, {}

    tpts = _xy(target_boundary)
    if len(tpts) < 3 or entrance_point is None:
        return 0.5, {"reason": "no target boundary/entrance"}, False, {}

    room_edge_counts, room_totals = _room_src_edge_fracs(ring_ids, ring_edge_idxs)

    # source edges: angle of each edge's outward normal relative to the
    # entrance-anchor edge's own outward normal
    bpts = _xy(plan.get("boundary") or [])
    s_edges = _oriented_edges(bpts)
    s_anchor = _nearest_edge_index(s_edges, _entrance_point_of_plan(plan))
    s_ref = _edge_outward_normal_angle(*s_edges[s_anchor][:2])
    s_theta = {
        e[4]: _norm(_edge_outward_normal_angle(e[0], e[1]), s_ref) for e in s_edges
    }

    # target edges: same, relative to the target's own entrance-anchor edge
    t_edges = _oriented_edges(tpts, target_edge_types)
    t_anchor = _nearest_edge_index(t_edges, entrance_point)
    t_ref = _edge_outward_normal_angle(*t_edges[t_anchor][:2])
    t_edge_info = [
        (e[4], e[3], _norm(_edge_outward_normal_angle(e[0], e[1]), t_ref))
        for e in t_edges
    ]

    def build_map(mirror):
        # mirror reflects every source normal about the entrance anchor —
        # reversing which physical side (e.g. left vs right of the door) each
        # source edge faces, the two hypotheses a source plan can be laid
        # into a target under.
        edge_map = {}
        for eidx, ang in s_theta.items():
            if mirror:
                ang = (2 * math.pi - ang) % (2 * math.pi)
            tidx, ttype, _ = min(t_edge_info, key=lambda t: _circ_dist(ang, t[2]))
            edge_map[eidx] = (tidx, ttype)
        return edge_map

    all_room_ids = list(label_by_id.keys())
    daylight_ids = [rid for rid, lab in label_by_id.items() if lab in DAYLIGHT_NEEDING]

    direct_map = build_map(False)
    if not daylight_ids:
        room_edges = _build_room_edges(room_edge_counts, room_totals, all_room_ids, direct_map)
        return 1.0, {"daylight_rooms": 0}, False, room_edges

    def score_map(edge_map):
        served, per_room = 0, {}
        for rid in daylight_ids:
            counts = room_edge_counts.get(rid)
            total = room_totals.get(rid, 0)
            if not counts:               # interior room — cannot get a window anywhere
                per_room[label_by_id[rid]] = 0.0
                continue
            ext = sum(cnt for eidx, cnt in counts.items() if edge_map[eidx][1] == EXTERNAL)
            frac = ext / total
            per_room[label_by_id[rid]] = round(frac, 2)
            if frac >= DAYLIGHT_MIN_EXTERNAL_FRAC:
                served += 1
        return served, per_room

    mirror_map = build_map(True)
    direct_served, direct_detail = score_map(direct_map)
    mirror_served, mirror_detail = score_map(mirror_map)

    if mirror_served > direct_served:
        edge_map, detail, mirror, served = mirror_map, mirror_detail, True, mirror_served
    else:
        edge_map, detail, mirror, served = direct_map, direct_detail, False, direct_served

    room_edges = _build_room_edges(room_edge_counts, room_totals, all_room_ids, edge_map)
    return served / len(daylight_ids), {"per_room": detail, "mirror": mirror}, mirror, room_edges


# ── public entry ─────────────────────────────────────────────────────────────

def topological_fitness(plan, target_boundary=None, target_edge_types=None,
                        entrance_point=None, requirements=None):
    """Topological fitness of a retrieved plan to the target boundary, in [0, 1].

    Components (combined by WEIGHTS):
      • program         — room multiset vs requested bedrooms/bathrooms
      • daylight_fit    — perimeter-ring alignment: fraction of daylight-needing
                          rooms landing on external arcs of the target (approach B)
      • entrance_access — common areas reachable from the entrance through transit
      • private_access  — private rooms adjacent to a transit provider
    """
    rooms = plan.get("rooms") or []
    label_by_id = {r.get("id"): _label(r) for r in rooms}
    counts = {}
    for lab in label_by_id.values():
        counts[lab] = counts.get(lab, 0) + 1
    adj = _adjacency(plan)

    # daylight_fit (approach B) — the boundary-coupled discriminator
    if target_boundary is not None:
        # _daylight_fit still supplies the ranking SCORE (ring alignment). The
        # exposed room_edges + node_pos use the DISPLAY mirror (shape-match, the
        # one the frontend draws with) so nodes, raycast connectors and sensors
        # all agree — the ring mirror is a ranking-internal choice and diverged.
        dfit, dfit_detail, _ring_mirror, _ring_edges = _daylight_fit(
            plan, target_boundary, target_edge_types, entrance_point)
        mirror = bool((plan.get("retrieval") or {}).get("mirror", False))
        node_pos = _place_nodes_in_target(plan, target_boundary, entrance_point, mirror)
        room_edges = _raycast_room_edges(
            plan, target_boundary, entrance_point, mirror, target_edge_types)
    else:
        dfit, dfit_detail, mirror = 0.5, {"reason": "no target boundary"}, False
        room_edges = {}
        node_pos = {}

    # entrance access (intrinsic graph term)
    commons = [rid for rid, lab in label_by_id.items() if lab in COMMON]
    entry = None
    ent_pt = _entrance_point_of_plan(plan)
    if ent_pt is not None:
        best_d = float("inf")
        for r in rooms:
            poly = r.get("polygon")
            if not poly or len(poly) < 3:
                continue
            try:
                d = SPoint(ent_pt).distance(SPoly(poly))
            except Exception:
                continue
            if d < best_d:
                entry, best_d = r.get("id"), d
    reachable = _reachable_through_transit(adj, label_by_id, entry)
    if commons and entry is not None:
        entrance_access = sum(1 for rid in commons if rid in reachable) / len(commons)
    elif entry is None:
        entrance_access = 0.5
    else:
        entrance_access = 1.0

    # private access (intrinsic graph term)
    privates = [rid for rid, lab in label_by_id.items() if lab in PRIVATE]
    if privates:
        served = sum(
            1 for rid in privates
            if any(label_by_id.get(nb) in TRANSIT for nb in adj.get(rid, ()))
        )
        private_access = served / len(privates)
    else:
        private_access = 1.0

    components = {
        "program": round(_program_match(counts, requirements), 3),
        "daylight_fit": round(dfit, 3),
        "entrance_access": round(entrance_access, 3),
        "private_access": round(private_access, 3),
    }
    score = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    return {
        "score": round(score, 4),
        "components": components,
        "counts": counts,
        "daylight_detail": dfit_detail,
        "mirror": mirror,
        "room_edges": room_edges,
        "node_pos": node_pos,
    }


if __name__ == "__main__":
    # Self-check for the node-placement pre-processing: every node must land
    # inside the target boundary, incl. a non-convex (L-shaped) target where a
    # naive map would leave a node in the notch.
    _plan = {
        "boundary": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "entrance": [0.45, 0.0, 0.55, 0.02],
        "rooms": [
            {"id": 1, "polygon": [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]},
            {"id": 2, "polygon": [[0.5, 0], [1, 0], [1, 0.5], [0.5, 0.5]]},
            {"id": 3, "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]},   # centre ~ (0.5,0.75)
        ],
    }
    # smaller square target — contain must shrink, all nodes inside
    _sq = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    pos = _place_nodes_in_target(_plan, _sq, [0.5, 0.2], mirror=False)
    poly = SPoly(_sq)
    assert len(pos) == 3, pos
    for rid, p in pos.items():
        assert poly.contains(SPoint(p)), f"square: node {rid} at {p} outside target"

    # L-shaped target: notch at top-right. Room 3's centre maps high-centre and
    # would fall in/near the notch; the stray guard must pull it inside.
    _L = [[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]]
    posL = _place_nodes_in_target(_plan, _L, [0.5, 0.0], mirror=False)
    Lpoly = SPoly(_L)
    for rid, p in posL.items():
        assert Lpoly.contains(SPoint(p)), f"L: node {rid} at {p} outside target"

    # Raycast room_edges: every room with an exterior wall fronts a valid target
    # edge; the bottom-left room (walls on the plan's left + bottom) must reach
    # the target's left/bottom edges, not the top.
    _types = ["external", "external", "external", "facing_corridor"]  # edges 0..3 of _sq
    re = _raycast_room_edges(_plan, _sq, [0.5, 0.2], mirror=False, target_edge_types=_types)
    assert re, "raycast produced no room_edges"
    for rid, conns in re.items():
        for cn in conns:
            assert 0 <= cn["target_edge_index"] < len(_sq), cn
            assert cn["edge_type"] in _types, cn
    # room 1 sits bottom-left; its fronted edges must be the bottom (0) or left (3),
    # never the top (2).
    assert all(cn["target_edge_index"] in (0, 3) for cn in re.get("1", [])), re.get("1")

    print("topo_signature node-placement self-check OK", pos)
    print("topo_signature raycast self-check OK", re)
