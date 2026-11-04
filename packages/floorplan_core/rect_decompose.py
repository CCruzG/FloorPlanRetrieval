"""Rectilinear polygon decomposition (geometry only, no labels).

Cuts an axis-aligned polygon (L, T, U, Z, plus, staircase ...) into rectangles
along the shortest valid cut from a reflex vertex, iteratively. An optional
sliver pass merges thin rectangles back into a neighbour when they share a long
enough boundary, and reports the undone cuts as ``reversed_cuts``.

Ported from the idea in ``ref-docs/test10_split_polygon.ipynb`` (``splitPolygon``).

Notes
- ``short`` / ``long`` are the side lengths of the axis-aligned bounding box.
  For non-rectangular parts they are bbox sides, not true widths.
- ``is_rect`` is computed from the final geometry (4 vertices). A merged part
  that happens to be a rectangle is reported as ``is_rect=True``.
- Slivers are only evaluated on rectangular parts.
"""
from __future__ import annotations

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import split as _shp_split, unary_union

__all__ = ["decompose", "decompose_plan"]


# ----------------------------------------------------------------- cleaning
def _clean_coords(pts, tol):
    pts = [(float(p[0]), float(p[1])) for p in pts]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]

    def dedupe(ps):
        out = []
        for p in ps:
            if not out or abs(p[0] - out[-1][0]) > tol or abs(p[1] - out[-1][1]) > tol:
                out.append(p)
        while len(out) > 1 and abs(out[0][0] - out[-1][0]) <= tol and abs(out[0][1] - out[-1][1]) <= tol:
            out.pop()
        return out

    pts = dedupe(pts)
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        n = len(pts)
        for i in range(n):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
            ux, uy, vx, vy = b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1]
            if abs(ux * vy - uy * vx) <= tol * max(1.0, abs(ux) + abs(uy) + abs(vx) + abs(vy)):
                pts.pop(i)
                pts = dedupe(pts)
                changed = True
                break
    return pts


def _axis_aligned(pts, tol):
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        if abs(a[0] - b[0]) > tol and abs(a[1] - b[1]) > tol:
            return False
    return True


def _to_poly(poly_or_pts):
    """Return (coords_list_or_None, holes_flag)."""
    if isinstance(poly_or_pts, Polygon):
        if poly_or_pts.is_empty:
            return [], False
        return list(poly_or_pts.exterior.coords), len(poly_or_pts.interiors) > 0
    return [tuple(p[:2]) for p in poly_or_pts], False


def _poly_from_clean(pts):
    p = Polygon(pts)
    return orient(p, 1.0)


def _ccw_coords(poly):
    pts = list(orient(poly, 1.0).exterior.coords)[:-1]
    i = min(range(len(pts)), key=lambda k: (pts[k][0], pts[k][1]))
    pts = pts[i:] + pts[:i]
    return [[x, y] for x, y in pts]


def _bbox_sides(poly):
    x0, y0, x1, y1 = poly.bounds
    w, h = x1 - x0, y1 - y0
    return min(w, h), max(w, h)


def _part(poly, tol, sliver_max=None):
    short, long_ = _bbox_sides(poly)
    is_rect = len(poly.exterior.coords) - 1 == 4
    sliver = bool(sliver_max is not None and is_rect and short <= sliver_max + tol)
    return {"polygon": _ccw_coords(poly), "is_rect": bool(is_rect), "short": short, "long": long_, "sliver": sliver}


def _fallback(status, coords, tol):
    pts = [[float(p[0]), float(p[1])] for p in coords]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    xs = [p[0] for p in pts] or [0.0]
    ys = [p[1] for p in pts] or [0.0]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    return {
        "status": status,
        "parts": [{"polygon": pts, "is_rect": False, "short": min(w, h), "long": max(w, h), "sliver": False}],
        "cuts": [],
        "reversed_cuts": [],
    }


# ------------------------------------------------------------------- cutting
def _reflex_vertices(pts):
    """pts CCW. Returns list of (index, d_in, d_out) for reflex vertices."""
    n = len(pts)
    out = []
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        ux, uy, vx, vy = b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1]
        if ux * vy - uy * vx < 0:
            out.append((i, (ux, uy), (vx, vy)))
    return out


def _unit(d):
    return (1 if d[0] > 0 else -1 if d[0] < 0 else 0, 1 if d[1] > 0 else -1 if d[1] < 0 else 0)


def _candidate_cuts(poly, tol):
    pts = list(poly.exterior.coords)[:-1]
    x0, y0, x1, y1 = poly.bounds
    reach = (x1 - x0) + (y1 - y0) + 1.0
    boundary = poly.exterior
    cands = []
    for i, d_in, d_out in _reflex_vertices(pts):
        v = pts[i]
        din = _unit(d_in)
        dout = _unit(d_out)
        for d in (din, (-dout[0], -dout[1])):
            far = (v[0] + d[0] * reach, v[1] + d[1] * reach)
            inter = LineString([v, far]).intersection(boundary)
            hits = []
            for g in getattr(inter, "geoms", [inter]):
                if g.is_empty:
                    continue
                cs = [(g.x, g.y)] if isinstance(g, Point) else list(g.coords)
                for c in cs:
                    dist = (c[0] - v[0]) * d[0] + (c[1] - v[1]) * d[1]
                    if dist > tol:
                        hits.append((dist, c))
            if not hits:
                continue
            dist, c = min(hits)
            hit = (v[0] + d[0] * dist, v[1] + d[1] * dist)
            shrink = dist * 0.01
            inner = LineString([(v[0] + d[0] * shrink, v[1] + d[1] * shrink),
                                (hit[0] - d[0] * shrink, hit[1] - d[1] * shrink)])
            if not poly.contains(inner):
                continue
            cands.append((round(dist, 9), v[0], v[1], d, v, hit))
    cands.sort(key=lambda t: t[:4])
    return cands


def _pieces(geom):
    return [g for g in getattr(geom, "geoms", [geom]) if isinstance(g, Polygon) and not g.is_empty]


def _clean_piece(g, tol):
    pts = _clean_coords(list(g.exterior.coords), tol)
    if len(pts) < 3:
        return None
    p = _poly_from_clean(pts)
    return p if p.area > tol else None


def _split_all(poly, tol):
    """Return (rect_polys, cuts) or None when stuck."""
    work = [poly]
    done, cuts = [], []
    guard = 0
    while work:
        guard += 1
        if guard > 10000:
            return None
        p = work.pop()
        if not _reflex_vertices(list(p.exterior.coords)[:-1]):
            done.append(p)
            continue
        applied = False
        for _, _, _, _, v, hit in _candidate_cuts(p, tol):
            try:
                res = _shp_split(p, LineString([v, hit]))
            except Exception:
                continue
            pcs = _pieces(res)
            if len(pcs) < 2 or abs(sum(g.area for g in pcs) - p.area) > 1e-9 * p.area:
                continue
            clean = [_clean_piece(g, tol) for g in pcs]
            if any(c is None for c in clean):
                continue
            cuts.append([[v[0], v[1]], [hit[0], hit[1]]])
            work.extend(clean)
            applied = True
            break
        if not applied:
            return None
    return done, cuts


# -------------------------------------------------------------- sliver pass
def _segments(geom):
    segs = []
    for g in getattr(geom, "geoms", [geom]):
        if isinstance(g, LineString) and not g.is_empty:
            cs = list(g.coords)
            segs += [(cs[i], cs[i + 1]) for i in range(len(cs) - 1)]
        elif hasattr(g, "geoms"):
            segs += _segments(g)
    return segs


def _shared_parallel(a, b, axes, tol):
    inter = a.boundary.intersection(b.boundary)
    keep = []
    for p, q in _segments(inter):
        horiz = abs(p[1] - q[1]) <= tol
        vert = abs(p[0] - q[0]) <= tol
        if (horiz and "h" in axes) or (vert and "v" in axes):
            keep.append(LineString([p, q]))
    return keep, sum(s.length for s in keep)


def _lines_to_cuts(geom, tol):
    out = []
    for g in getattr(geom, "geoms", [geom]):
        if isinstance(g, LineString) and not g.is_empty and g.length > tol:
            cs = list(g.coords)
            out.append([[cs[0][0], cs[0][1]], [cs[-1][0], cs[-1][1]]])
    return out


def _sliver_pass(polys, cuts, sliver_max, merge_min_contact, tol):
    polys = list(polys)
    cuts = [list(c) for c in cuts]
    reversed_cuts = []
    changed = True
    while changed:
        changed = False
        order = sorted(range(len(polys)), key=lambda i: (_bbox_sides(polys[i])[0], polys[i].bounds))
        for i in order:
            p = polys[i]
            if len(p.exterior.coords) - 1 != 4:
                continue
            short, long_ = _bbox_sides(p)
            if short > sliver_max + tol:
                continue
            axes = set()
            if long_ - short <= tol:
                axes = {"h", "v"}
            else:
                x0, y0, x1, y1 = p.bounds
                axes = {"h"} if (x1 - x0) > (y1 - y0) else {"v"}
            best = None
            for j, q in enumerate(polys):
                if j == i:
                    continue
                segs, ln = _shared_parallel(p, q, axes, tol)
                if ln >= merge_min_contact - tol and ln > tol and (best is None or ln > best[0] + tol):
                    best = (ln, j, segs)
            if best is None:
                continue
            _, j, segs = best
            merged = polys[i].union(polys[j])
            if not isinstance(merged, Polygon):
                continue
            pts = _clean_coords(list(merged.exterior.coords), tol)
            if len(pts) < 3 or len(merged.interiors) > 0:
                continue
            su = unary_union(segs)
            new_cuts = []
            for c in cuts:
                cl = LineString(c)
                ov = cl.intersection(su)
                if ov.is_empty or ov.length <= tol:
                    new_cuts.append(c)
                    continue
                reversed_cuts += _lines_to_cuts(ov, tol)
                new_cuts += _lines_to_cuts(cl.difference(ov), tol)
            cuts = new_cuts
            newp = _poly_from_clean(pts)
            polys = [q for k, q in enumerate(polys) if k not in (i, j)] + [newp]
            changed = True
            break
    return polys, cuts, reversed_cuts


# ------------------------------------------------------------------- public
def decompose(poly, *, sliver_max=None, merge_min_contact=None, tol=1e-9) -> dict:
    """Decompose a rectilinear polygon into rectangles. See module docstring.

    Works in the input's own units. ``sliver_max=None`` disables the sliver pass.
    Never raises for bad geometry; returns status ``invalid``, ``not_rectilinear``
    or ``no_cut`` with the input as a single non-rect part.
    """
    try:
        coords, has_holes = _to_poly(poly)
    except Exception:
        return _fallback("invalid", [], tol)
    try:
        raw = [(float(c[0]), float(c[1])) for c in coords]
    except Exception:
        return _fallback("invalid", [], tol)
    if has_holes or len(raw) < 3:
        return _fallback("invalid", raw, tol)
    pts = _clean_coords(raw, tol)
    if len(pts) < 3:
        return _fallback("invalid", raw, tol)
    sp = Polygon(pts)
    if not sp.is_valid or sp.area <= tol:
        return _fallback("invalid", pts, tol)
    if not _axis_aligned(pts, tol):
        return _fallback("not_rectilinear", pts, tol)
    p = _poly_from_clean(pts)
    if not _reflex_vertices(list(p.exterior.coords)[:-1]):
        part = _part(p, tol)
        return {"status": "rectangle", "parts": [part], "cuts": [], "reversed_cuts": []}
    res = _split_all(p, tol)
    if res is None:
        return _fallback("no_cut", pts, tol)
    polys, cuts = res
    reversed_cuts = []
    if sliver_max is not None:
        mc = merge_min_contact if merge_min_contact is not None else 0.0
        polys, cuts, reversed_cuts = _sliver_pass(polys, cuts, sliver_max, mc, tol)
    polys.sort(key=lambda g: (g.bounds[1], g.bounds[0], g.bounds[3], g.bounds[2]))
    return {
        "status": "ok",
        "parts": [_part(g, tol, sliver_max) for g in polys],
        "cuts": cuts,
        "reversed_cuts": reversed_cuts,
    }


def _seg_len(c):
    return ((c[1][0] - c[0][0]) ** 2 + (c[1][1] - c[0][1]) ** 2) ** 0.5


def decompose_plan(plan: dict, *, scale_m=18.0, sliver_max_m=0.2, merge_min_contact_m=1.2,
                   skip_labels=("Balcony", "Kitchen", "Bathroom")) -> dict:
    """Decompose every room of an RPlan v2.1.0 plan (coords normalised 0-1).

    Coordinates stay in plan units; lengths are reported in metres.
    Rooms whose label is in ``skip_labels`` are not decomposed: they return as
    one part with status ``skipped`` (excluded from the block analysis,
    2026-11-03: balconies, kitchens, bathrooms; they stay in the plan as whole rooms for doors and contacts).
    """
    sm = sliver_max_m / scale_m if sliver_max_m is not None else None
    mc = merge_min_contact_m / scale_m if merge_min_contact_m is not None else None
    skip = set(skip_labels or ())
    rooms, counts = [], {}
    n_split = n_parts = n_sl = n_rev = 0
    for r in plan.get("rooms", []):
        if r.get("label") in skip:
            d = _fallback("skipped", r.get("polygon") or [], 1e-9)
        else:
            d = decompose(r["polygon"], sliver_max=sm, merge_min_contact=mc)
        for pt in d["parts"]:
            pt["short_m"] = pt["short"] * scale_m
            pt["long_m"] = pt["long"] * scale_m
        d["cut_lengths_m"] = [_seg_len(c) * scale_m for c in d["cuts"]]
        d["reversed_cut_lengths_m"] = [_seg_len(c) * scale_m for c in d["reversed_cuts"]]
        rooms.append({"id": r.get("id"), "label": r.get("label"), "type": r.get("type"), **d})
        counts[d["status"]] = counts.get(d["status"], 0) + 1
        n_split += len(d["parts"]) > 1
        n_parts += len(d["parts"])
        n_sl += sum(pt["sliver"] for pt in d["parts"])
        n_rev += len(d["reversed_cuts"])
    return {
        "scale_m": scale_m,
        "sliver_max_m": sliver_max_m,
        "merge_min_contact_m": merge_min_contact_m,
        "rooms": rooms,
        "summary": {"n_rooms": len(rooms), "n_split": int(n_split), "n_parts": n_parts,
                    "n_slivers": int(n_sl), "n_reversed": n_rev, "status_counts": counts},
    }
