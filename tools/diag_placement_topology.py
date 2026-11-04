"""
diag_placement_topology.py — gate for Stage-0 step-0 core placement, modes
off / cpsat / cpsat_dim (packages/floorplan_core/core_placement.py,
layout_cpsat.py, spec["core_placement"]).

Replays the live app's pipeline (retrieve -> top 3 -> generate) through
apps/api/server.py's Flask app via test_client() — no server process started,
nothing under packages/ or apps/ is modified. Runs placement off / cpsat /
cpsat_dim for six fixed test boundaries, top 3 retrieved plans each. A
non-orthogonal boundary is skipped (CP-SAT is unsupported on one) and reported.

cpsat_dim (2026-11-02) sizes the retrieved arrangement instead of re-arranging
it (layout_cpsat.solve_layout_dim); this script is its gate. For each
(boundary, rank, mode) records, over all seven geometry-layer sensors:
  sensors_ok        {sensor: ok} for all seven (reachability, access_typed,
                    entrance_valid, daylight, room_depth, min_room_size,
                    node_in_room)
  passed / total    sensors.summary counts
  entrance_detail   the entrance_valid check's own detail string
  d24_failing       count of min_room_size failing_rooms
  feasible          the generated plan's top-level feasible flag
  lost_source_edges source room<->room edges (src fp["edges"]) with no realised
                    contact at all (shared wall ~0) in the generated geometry,
                    split into door/open (connection 1/2 or separation door/
                    open) vs all
  door_open_under_0.9m  door/open source edges that ARE realised-adjacent but
                    below the 0.9 m door/window floor
  order_violated    non-adjacent room pairs whose SOURCE centres (mapped to
                    the target frame, topo_signature.map_rooms_to_target)
                    differ > 0.5 m on an axis, but whose GENERATED centres
                    reverse that order on the same axis — the retrieved
                    plan's room arrangement, not honoured
  placement         (cpsat/cpsat_dim) {status, relaxation, relaxations,
                    solve_time_s, objective, arrangement_source} from
                    report["placement"]
  relaxed_pairs     (greedy, if ever re-added) report["placement"]["rooms"]
                    [r]["relaxed"] entries

Totals per mode sum every measure above; the gate (cpsat_dim vs off/cpsat) is
printed and saved alongside.

Read-only diagnostic. Output: data/outputs/diagnostics/placement_cpsat_dim_<date>.json
(pass a suffix, e.g. `diag_placement_topology.py baseline`, to keep runs apart).
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]          # 02_Retrieval_v02/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shapely.geometry import Polygon as SPoly  # noqa: E402

RUN_DATE = __import__("datetime").date.today().isoformat()

BOUNDARY_LIB_PATH = REPO_ROOT.parent / "test-boundaries" / "boundary_library.json"
TARGET_NAMES = ["test_boundary_001", "test_boundary_002", "test_boundary_003",
                "test_boundary_004", "test_boundary_001b", "test_boundary_005b"]
TOPN = 3
MM_PER_UNIT = 18000.0   # schema.SCALE; see diag_min_size.py's note on this value
MODES = [False, "cpsat", "cpsat_dim"]
# 2026-11-03: every mode runs on the room plan and on the node-expanded plan
# (/generate expand_nodes); node runs are collapsed back to source rooms
# (id prefix before "/") before edges and order are measured.
RUNS = [(m, ex) for ex in (False, True) for m in MODES]
SENSOR_NAMES = ["reachability", "access_typed", "entrance_valid", "daylight",
                "room_depth", "min_room_size", "node_in_room"]

OUT_DIR = REPO_ROOT / "data" / "outputs" / "diagnostics"
OUT_PATH = OUT_DIR / (f"placement_cpsat_dim_{RUN_DATE}"
                      + (f"_{sys.argv[1]}" if len(sys.argv) > 1 else "") + ".json")

DOOR_MIN_M = 0.9
ORDER_MIN_M = 0.5
M_PER_UNIT = MM_PER_UNIT / 1000.0  # boundary units -> metres, matching 18.0


def _orthogonal(boundary, tol=1e-6):
    nb = len(boundary)
    for k in range(nb):
        (ax, ay), (bx, by) = boundary[k], boundary[(k + 1) % nb]
        if abs(ax - bx) > tol and abs(ay - by) > tol:
            return False
    return True


def load_targets():
    lib = json.loads(BOUNDARY_LIB_PATH.read_text())
    by_name = {e["name"]: e for e in lib}
    missing = [n for n in TARGET_NAMES if n not in by_name]
    if missing:
        raise SystemExit(f"boundary(ies) not found in {BOUNDARY_LIB_PATH}: {missing}")
    return [by_name[n] for n in TARGET_NAMES]


def entry_to_request_parts(entry):
    boundary = [[v["x"], v["y"]] for v in entry["vertices"]]
    entrance_point = [entry["entrance"]["x"], entry["entrance"]["y"]]
    typed_edges = [{"edge_index": i, "type": t}
                   for i, t in enumerate(entry.get("edgeTypes") or []) if t]
    requirements = entry.get("requirements") or {}
    return boundary, entrance_point, typed_edges, requirements


def _is_door_or_open(e):
    sep = e.get("separation")
    if sep in ("door", "open"):
        return True
    if sep == "wall":
        return False
    return e.get("connection") in (1, 2)


def realised_pairs(rooms, tol_m=0.05):
    """room<->room shared-wall length (m) for every pair that actually touches
    along a wall, keyed by frozenset(u, v). Mirrors the reference script's
    `realised()` (adj3.py / adj_check.py)."""
    g = {str(r["id"]): SPoly(r["polygon"]).buffer(0)
         for r in rooms if len(r.get("polygon") or []) >= 3}
    ids = sorted(g)
    out = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            length = g[a].boundary.intersection(g[b].boundary).length * M_PER_UNIT
            if length > tol_m:
                out[frozenset((a, b))] = round(length, 3)
    return out


def _centers(rooms):
    """{str(id): (x, y)} polygon centroid for every room with a usable polygon."""
    out = {}
    for r in rooms:
        poly = r.get("polygon")
        if not poly or len(poly) < 3:
            continue
        try:
            c = SPoly(poly).buffer(0).centroid
            out[str(r["id"])] = (c.x, c.y)
        except Exception:
            continue
    return out


def order_specs_for(fp, boundary, entrance_point):
    """Non-adjacent source room pairs whose SOURCE centres (mapped to the
    target frame) differ > ORDER_MIN_M on an axis — the same rule
    layout_cpsat.solve_layout_dim uses to pin centre order, used here as a
    mode-independent baseline every generated plan is checked against."""
    from packages.floorplan_core.topo_signature import map_rooms_to_target
    mirror = bool((fp.get("retrieval") or {}).get("mirror", False))
    mapped = map_rooms_to_target(fp, boundary, entrance_point, mirror)
    centers = {}
    for rid, poly in mapped.items():
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        centers[rid] = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
    adjacent = set()
    for e in fp.get("edges") or []:
        a, b = str(e.get("u")), str(e.get("v"))
        if a != b:
            adjacent.add(tuple(sorted((a, b))))
    specs = []
    ids = sorted(centers)
    thresh = ORDER_MIN_M / M_PER_UNIT
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if tuple(sorted((a, b))) in adjacent:
                continue
            for axis in (0, 1):
                d = centers[a][axis] - centers[b][axis]
                if abs(d) * M_PER_UNIT > ORDER_MIN_M:
                    specs.append((a, b, axis, 1 if d < 0 else -1))
    return specs


def order_violations(specs, gen_centers):
    """Count of specs whose sign the GENERATED centres reverse (both rooms
    must be present in the generated plan to judge it)."""
    n = 0
    for a, b, axis, sign in specs:
        ca, cb = gen_centers.get(a), gen_centers.get(b)
        if ca is None or cb is None:
            continue
        d = ca[axis] - cb[axis]
        if sign > 0 and d > 1e-9:
            n += 1
        elif sign < 0 and d < -1e-9:
            n += 1
    return n


def mode_label(m, expand=False):
    base = "off" if m is False else m
    return f"nodes+{base}" if expand else base


def collapse_rooms(rooms):
    """Node-expanded generated rooms -> one polygon per source room (id prefix)."""
    from shapely.ops import unary_union
    groups = {}
    for r in rooms:
        if len(r.get("polygon") or []) < 3:
            continue
        groups.setdefault(str(r["id"]).split("/", 1)[0], []).append(SPoly(r["polygon"]).buffer(0))
    out = []
    for rid, ps in groups.items():
        g = unary_union(ps)
        if g.geom_type != "Polygon":
            g = max(g.geoms, key=lambda x: x.area)
        out.append({"id": rid, "polygon": [list(c) for c in g.exterior.coords[:-1]]})
    return out


def main():
    from apps.api import server as server_mod
    from packages.floorplan_core.index_store import build_index_store_from_npz

    npz_path = REPO_ROOT / "data" / "index" / "rplan_index_v5.npz"
    print(f"loading index: {npz_path} ...", file=sys.stderr)
    server_mod.INDEX_STORE = build_index_store_from_npz(
        npz_path=str(npz_path),
        data_dir=str(server_mod.RPLAN_DATASET_DIR),
        logger=server_mod.logger,
    )
    client = server_mod.app.test_client()
    targets = load_targets()

    out = {"generated_at": RUN_DATE, "mm_per_unit_used": MM_PER_UNIT,
          "modes": [mode_label(m, ex) for m, ex in RUNS], "boundaries": []}
    totals = {mode_label(m, ex): {"d24_failing": 0, "lost_source_edges": 0,
                              "lost_source_wall_edges": 0,
                              "lost_source_door_open_edges": 0,
                              "door_open_under_0.9m": 0, "infeasible": 0,
                              "unsolved": 0,
                              "reachability_fail": 0, "entrance_fail": 0,
                              "order_violated": 0, "sensors_passed": 0,
                              "sensors_total": 0, "plans": 0}
             for m, ex in RUNS}

    for entry in targets:
        name = entry["name"]
        boundary, entrance_point, typed_edges, requirements = entry_to_request_parts(entry)
        orthogonal = _orthogonal(boundary)
        b_out = {"name": name, "orthogonal": orthogonal, "ranks": []}
        if not orthogonal:
            print(f"skipping {name}: non-orthogonal boundary (cpsat unsupported)", file=sys.stderr)
            out["boundaries"].append(b_out)
            continue

        plans = client.post("/retrieve", json={
            "boundary": boundary, "entrance_point": entrance_point,
            "requirements": requirements, "topn": TOPN, "typed_edges": typed_edges,
        }).get_json()["plans"][:TOPN]

        for rank, fp in enumerate(plans, 1):
            src_edges = fp.get("edges", [])
            specs = order_specs_for(fp, boundary, entrance_point)
            rank_out = {"rank": rank, "src_edge_count": len(src_edges),
                       "order_constraints": len(specs)}
            for mode, expand in RUNS:
                label = mode_label(mode, expand)
                g = client.post("/generate", json={
                    "fp": fp, "boundary": boundary, "entrance_point": entrance_point,
                    "typed_edges": typed_edges, "requirements": requirements,
                    "mm_per_unit": MM_PER_UNIT, "core_placement": mode,
                    "expand_nodes": expand,
                }).get_json()
                checks = (g.get("sensors") or {}).get("checks") or []
                summary = (g.get("sensors") or {}).get("summary") or {}
                by_name = {c["sensor"]: c for c in checks}
                d24 = by_name.get("min_room_size", {})
                reach = by_name.get("reachability", {})
                ent = by_name.get("entrance_valid", {})
                gen_rooms = collapse_rooms(g.get("rooms") or []) if expand else (g.get("rooms") or [])
                ra = realised_pairs(gen_rooms)
                gen_centers = _centers(gen_rooms)

                lost, lost_door, lost_wall = [], [], []
                degraded_door = []
                for e in src_edges:
                    u, v = str(e.get("u")), str(e.get("v"))
                    key = frozenset((u, v))
                    door = _is_door_or_open(e)
                    if key not in ra:
                        lost.append([u, v])
                        if door:
                            lost_door.append([u, v])
                        else:
                            lost_wall.append([u, v])
                    elif door and ra[key] < DOOR_MIN_M:
                        degraded_door.append([u, v, ra[key]])

                ov = order_violations(specs, gen_centers)

                entry_out = {
                    "feasible": g.get("feasible"),
                    "sensors_ok": {s: by_name.get(s, {}).get("ok") for s in SENSOR_NAMES},
                    "passed": summary.get("passed"),
                    "total": summary.get("total"),
                    "entrance_valid_ok": ent.get("ok"),
                    "entrance_valid_detail": ent.get("detail"),
                    "d24_failing": len(d24.get("failing_rooms") or []),
                    "reachability_ok": reach.get("ok"),
                    "lost_source_edges": len(lost),
                    "lost_source_edges_door_open": lost_door,
                    "lost_source_wall_edges": len(lost_wall),
                    "lost_source_door_open_edges": len(lost_door),
                    "door_open_under_0.9m": degraded_door,
                    "order_violated": ov,
                }
                placement = ((g.get("report") or {}).get("placement") or {})
                if mode == "greedy":
                    relaxed = []
                    for rid, rv in (placement.get("rooms") or {}).items():
                        for rel in rv.get("relaxed") or []:
                            relaxed.append({"room": rid, **rel})
                    entry_out["relaxed_pairs"] = relaxed
                unsolved = False
                if mode in ("cpsat", "cpsat_dim"):
                    unsolved = bool(placement.get("unsolved"))
                    entry_out["cpsat"] = {
                        "status": placement.get("status"),
                        "unsolved": unsolved,
                        "relaxation": placement.get("relaxation"),
                        "relaxations": placement.get("relaxations"),
                        "solve_time_s": placement.get("solve_time_s"),
                        "objective": placement.get("objective"),
                        "arrangement_source": placement.get("arrangement_source"),
                    }
                rank_out[label] = entry_out

                t = totals[label]
                t["plans"] += 1
                t["d24_failing"] += entry_out["d24_failing"]
                t["lost_source_edges"] += entry_out["lost_source_edges"]
                t["lost_source_wall_edges"] += entry_out["lost_source_wall_edges"]
                t["lost_source_door_open_edges"] += entry_out["lost_source_door_open_edges"]
                t["door_open_under_0.9m"] += len(degraded_door)
                t["order_violated"] += ov
                t["sensors_passed"] += (summary.get("passed") or 0)
                t["sensors_total"] += (summary.get("total") or 0)
                if not entry_out["feasible"]:
                    t["infeasible"] += 1
                if unsolved:
                    t["unsolved"] += 1
                if entry_out["reachability_ok"] is False:
                    t["reachability_fail"] += 1
                if ent.get("ok") is False:
                    t["entrance_fail"] += 1
            b_out["ranks"].append(rank_out)
        out["boundaries"].append(b_out)

    out["totals"] = totals

    # ── gate: cpsat_dim vs off / cpsat ────────────────────────────────────────
    off, cpsat, dim = totals["off"], totals["cpsat"], totals["cpsat_dim"]
    criteria = [
        ("entrance_fail <= off",  dim["entrance_fail"] <= off["entrance_fail"],
         dim["entrance_fail"], off["entrance_fail"]),
        ("lost_source_edges <= off", dim["lost_source_edges"] <= off["lost_source_edges"],
         dim["lost_source_edges"], off["lost_source_edges"]),
        ("order_violated == 0", dim["order_violated"] == 0, dim["order_violated"], 0),
        ("sensors_passed >= max(off, cpsat)",
         dim["sensors_passed"] >= max(off["sensors_passed"], cpsat["sensors_passed"]),
         dim["sensors_passed"], max(off["sensors_passed"], cpsat["sensors_passed"])),
        ("d24_failing <= off", dim["d24_failing"] <= off["d24_failing"],
         dim["d24_failing"], off["d24_failing"]),
        ("infeasible <= off", dim["infeasible"] <= off["infeasible"],
         dim["infeasible"], off["infeasible"]),
    ]
    gate = {"criteria": [{"name": n, "pass": bool(p), "cpsat_dim": a, "bound": b}
                         for n, p, a, b in criteria],
           "pass": all(p for _, p, _, _ in criteria)}
    out["gate"] = gate

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH}", file=sys.stderr)
    print(json.dumps(totals, indent=2))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
