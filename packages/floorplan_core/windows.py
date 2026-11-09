"""Windows derived from a generated plan (G1 part 2a, 2026-11-09). Brain, pure.

Windows are not stated in the Spec. This function reads the generated rooms, the
boundary and its edge types, and places them, as doors.py does for doors:

  * Contact = a room wall lying on a boundary edge. Runs come from
    `topo_signature.contact_runs` (the logic of `exact_room_edges`, merged).
  * Only `external` edges carry windows.
  * Habitable rooms (living, dining, bedroom, master bedroom, study) are windowed.
    Kitchen and bathroom are windowed when they have an external run; no failure
    otherwise. Hallway, storage, nook, circulation and other types get none.
  * A run shorter than `window_min_mm` (600) gets no window. Living and dining get
    one window per qualifying run; other types get one, on their longest run.
  * Width = (run - 2 jamb) x `window_fill` (0.6), at least `window_min_mm` and at
    most run - 2 jamb. When run - 2 jamb is below `window_min_mm`, the window takes
    the whole free run, unless that is below `window_floor_mm` (300): then unplaced.
  * Centred on the run; moved to keep `window_jamb_mm` (300) clear of the run ends
    and of any door on the same wall (the entrance door).
  * Heights by type (`window_heights`, mm above floor).
  * Report only: glazed area / floor area per habitable room, flagged below 0.10.

Parameters live in `options` (spec.build_options); same inputs give the same
output. Coordinates are plan units; one unit = `mm_per_unit` mm.
"""
import math

from shapely.geometry import Polygon as SPoly

from packages.floorplan_core.topo_signature import contact_runs, boundary_eps

DEFAULTS = {
    "window_min_mm": 600.0,
    "window_jamb_mm": 300.0,
    "window_fill": 0.6,
    "window_floor_mm": 300.0,
    "glazing_min_ratio": 0.10,
}

HABITABLE = {"living", "dining", "bedroom", "master_bedroom", "study"}
OPTIONAL = {"kitchen", "bathroom"}
ONE_WINDOW_TYPES = {"bedroom", "master_bedroom", "study", "kitchen", "bathroom"}
# type -> (sill, head) in mm above finished floor
HEIGHTS = {
    "living": (0.0, 2400.0), "dining": (0.0, 2400.0),
    "bedroom": (900.0, 2400.0), "master_bedroom": (900.0, 2400.0), "study": (900.0, 2400.0),
    "kitchen": (1050.0, 2400.0), "bathroom": (1500.0, 2400.0),
}
_ALIAS = {"living_room": "living", "livingroom": "living", "master": "master_bedroom",
          "masterbedroom": "master_bedroom", "bed_room": "bedroom"}


def _norm(t):
    s = str(t or "").strip().lower().replace(" ", "_").replace("-", "_")
    return _ALIAS.get(s, s)


def _door_intervals(doors, a, b, eps):
    """Door spans [lo, hi] (edge units) lying on the boundary edge a-b."""
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    out = []
    for d in doors or []:
        c = d.get("centre")
        if not c:
            continue
        off = abs((c[0] - a[0]) * uy - (c[1] - a[1]) * ux)
        t = (c[0] - a[0]) * ux + (c[1] - a[1]) * uy
        if off <= max(eps, 1e-6) * 10 and -eps <= t <= L + eps:
            half = float(d.get("width") or 0.0) / 2.0
            out.append((t - half, t + half))
    return sorted(out)


def _free_segments(lo, hi, doors, jamb):
    """[lo, hi] minus each door span widened by `jamb`."""
    segs = [(lo, hi)]
    for dl, dh in doors:
        dl, dh = dl - jamb, dh + jamb
        nxt = []
        for s0, s1 in segs:
            if dh <= s0 or dl >= s1:
                nxt.append((s0, s1))
                continue
            if dl > s0:
                nxt.append((s0, dl))
            if dh < s1:
                nxt.append((dh, s1))
        segs = nxt
    return [s for s in segs if s[1] - s[0] > 1e-9]


def derive_windows(rooms, boundary, edge_types, mm_per_unit, doors=None, options=None):
    """Return (windows, unplaced, glazing). See the module docstring.

    windows: [{room, edge_index, centre, wall [a, b], width, width_mm, sill_mm,
               head_mm, run_mm, kind "window"}]; `wall` is the contact run's ends and
               `centre` lies on it. Plan units except the *_mm fields.
    unplaced: [{room, reason}].
    glazing: [{room, type, glazed_m2, floor_m2, ratio, flag}] for habitable rooms.
    """
    opt = {**DEFAULTS, **{k: v for k, v in (options or {}).items() if k in DEFAULTS and v is not None}}
    heights = {**HEIGHTS, **{_norm(k): tuple(v) for k, v in ((options or {}).get("window_heights") or {}).items()}}
    one_window = {_norm(t) for t in ((options or {}).get("one_window_types") or ONE_WINDOW_TYPES)}
    windows, unplaced, glazing = [], [], []
    if not mm_per_unit or mm_per_unit <= 0 or not boundary or len(boundary) < 3:
        return windows, unplaced, glazing
    mm = float(mm_per_unit)
    tpts = [(float(p[0]), float(p[1])) for p in boundary]
    n = len(tpts)
    types = list(edge_types or [])
    eps = boundary_eps(tpts)
    min_u = opt["window_min_mm"] / mm
    jamb_u = opt["window_jamb_mm"] / mm
    floor_u = opt["window_floor_mm"] / mm
    door_spans = {}

    def doors_on(i):
        if i not in door_spans:
            door_spans[i] = _door_intervals(doors, tpts[i], tpts[(i + 1) % n], eps)
        return door_spans[i]

    for r in sorted(rooms or [], key=lambda r: str(r.get("id"))):
        rid, rtype = str(r.get("id")), _norm(r.get("type"))
        poly = r.get("polygon") or []
        if len(poly) < 3:
            continue
        habitable = rtype in HABITABLE
        if not habitable and rtype not in OPTIONAL:
            continue
        runs = []   # (edge index, lo, hi) on external edges
        for i, merged in sorted(contact_runs(poly, tpts, eps).items()):
            et = types[i] if 0 <= i < len(types) else "external"
            if et != "external":
                continue
            for lo, hi in merged:
                if hi - lo >= min_u - 1e-9:
                    runs.append((i, lo, hi))
        if rtype in one_window and runs:
            runs = [max(runs, key=lambda q: (round(q[2] - q[1], 9), -q[0], -q[1]))]
        placed = 0
        sill, head = heights[rtype]
        for i, lo, hi in runs:
            a, b = tpts[i], tpts[(i + 1) % n]
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
            avail = (hi - lo) - 2 * jamb_u
            if avail >= min_u - 1e-9:
                w = min(max(avail * opt["window_fill"], min_u), avail)
            else:
                w = avail
            if w < floor_u - 1e-9:
                if habitable:
                    unplaced.append({"room": rid, "reason": f"run {round((hi - lo) * mm)} mm too short for a window"})
                continue
            segs = _free_segments(lo + jamb_u, hi - jamb_u, doors_on(i), jamb_u)
            mid = (lo + hi) / 2.0
            best = None
            for s0, s1 in segs:
                if s1 - s0 >= w - 1e-9:
                    c = min(max(mid, s0 + w / 2.0), s1 - w / 2.0)
                    cand = (abs(c - mid), c, w)
                else:
                    cand = None
                if cand and (best is None or cand[0] < best[0] - 1e-12):
                    best = cand
            if best is None:
                # no free span takes the full width: use the widest span if it is wide enough
                wide = [s for s in segs if s[1] - s[0] >= floor_u - 1e-9]
                if wide:
                    s0, s1 = max(wide, key=lambda s: s[1] - s[0])
                    w2 = s1 - s0
                    best = (0.0, (s0 + s1) / 2.0, w2)
            if best is None:
                if habitable:
                    unplaced.append({"room": rid, "reason": "door leaves no clear span on the run"})
                continue
            _, c, w = best
            windows.append({"room": rid, "edge_index": i,
                            "centre": [a[0] + ux * c, a[1] + uy * c],
                            "wall": [[a[0] + ux * lo, a[1] + uy * lo], [a[0] + ux * hi, a[1] + uy * hi]],
                            "width": w, "width_mm": round(w * mm, 3), "run_mm": round((hi - lo) * mm, 1),
                            "sill_mm": sill, "head_mm": head, "kind": "window"})
            placed += 1
        if habitable:
            g = SPoly(poly)
            floor_m2 = (g.area if g.is_valid else g.buffer(0).area) * (mm / 1000.0) ** 2
            glazed = sum(w["width_mm"] * (w["head_mm"] - w["sill_mm"]) / 1e6
                         for w in windows if w["room"] == rid)
            ratio = glazed / floor_m2 if floor_m2 > 0 else 0.0
            glazing.append({"room": rid, "type": rtype, "glazed_m2": round(glazed, 3),
                            "floor_m2": round(floor_m2, 3), "ratio": round(ratio, 4),
                            "flag": bool(ratio < opt["glazing_min_ratio"])})
    return windows, unplaced, glazing
