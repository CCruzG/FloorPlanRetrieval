"""
diag_min_size.py — diagnose min_room_size (D24) sensor failures on generated
plans, for two fixed test boundaries.

Replays the live app's pipeline (retrieve -> top 3 -> generate) through
apps/api/server.py's Flask app via test_client() — no server process started,
nothing under packages/ or apps/ is modified. For every room that fails
min_edge or min_area, records the numbers and assigns exactly one cause
(A/D/C/B/E, in that priority order — see module docstring in the task brief).

Read-only diagnostic. Output: data/outputs/diagnostics/min_size_<date>.json.

2026-11-02: classifies on the usable-footprint D24 (topo_sensors.check_min_room_size),
not the bbox. Two tests: core (a min_edge square fits) and usable area (area a
USABLE_WIDTH_MM square reaches >= min_area). Causes, in priority order:
  A        budget infeasible (sum of min areas > boundary, or step-3 solve infeasible)
  D        target_frac x boundary area < min_area (the area demand is below the floor)
  S-*      core fails — no min_edge square fits. Split by the stage the core is lost,
           rebuilding the room from report.grid_lines.cells (step 3, after 2b):
    S-fill   no core in the step-3 cell union, which is not a rectangle (L/T fill)
    S-narrow no core in the step-3 cell union, which is a rectangle (solve too narrow)
    S-clip   core fits the cell union, lost when clipped to the boundary
    S-arm    core survives the clip, lost to a 4b arm carve (room is an arm donor)
  area-only failures (core fits, usable area < min_area):
    W        raw area >= min_area but usable area below — lost to parts < 0.9 m wide
    C-arm / C-clip / B   raw area short; lost to the arm, the clip, or the solve
"""
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]          # 02_Retrieval_v02/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shapely.geometry import Polygon as SPoly, box as sbox  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

RUN_DATE = date.today().isoformat()

# ── fixed inputs (per coordinator instruction, overriding the original brief's
# data/outputs/boundaries/boundary_library.json) ────────────────────────────
BOUNDARY_LIB_PATH = REPO_ROOT.parent / "test-boundaries" / "boundary_library.json"
TARGET_NAMES = ["test_boundary_001b", "test_boundary_005b"]
TOPN = 3

# mm_per_unit: NOT actually sent by the live frontend's /generate call (see
# gap note in the report) — apps/web/src/main.js:3091 posts fp/boundary/
# entrance_point/typed_edges/requirements/swap only. Using the value the
# frontend WOULD send were it to follow /generate's documented contract:
# apps/web/src/constants.js METRES_PER_UNIT=18 -> 18*1000 = 18000 mm/unit,
# which equals schema.SCALE (18_000) — the same fallback topo_sensors uses
# when mm_per_unit is absent. Passing it explicitly also turns on min_edge
# enforcement in generate()'s step-3 solve, which is silently OFF otherwise.
MM_PER_UNIT = 18000.0

OUT_DIR = REPO_ROOT / "data" / "outputs" / "diagnostics"
# Pass a suffix to keep runs apart, e.g. `diag_min_size.py after-fix`.
OUT_PATH = OUT_DIR / (f"min_size_{RUN_DATE}"
                      + (f"_{sys.argv[1]}" if len(sys.argv) > 1 else "") + ".json")


def _poly_area_units(verts_xy):
    n = len(verts_xy)
    return abs(sum(verts_xy[i][0] * verts_xy[(i + 1) % n][1]
                   - verts_xy[(i + 1) % n][0] * verts_xy[i][1]
                   for i in range(n))) / 2.0


def _is_non_convex(boundary_xy, tol=1e-9):
    poly = SPoly(boundary_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    hull = poly.convex_hull
    return (hull.area - poly.area) > tol * max(hull.area, 1.0)


def _clean(g):
    return g if g.is_valid else g.buffer(0)


def _cell_union(grid_lines, rid):
    """The room's step-3 footprint: the union of the grid cells it owns (after the
    2b repair, before the clip and the 4b carve), in boundary units."""
    xs, ys = grid_lines.get("xs") or [], grid_lines.get("ys") or []
    boxes = [sbox(xs[i], ys[j], xs[i + 1], ys[j + 1])
             for i, j, owner in grid_lines.get("cells") or []
             if owner == rid and i + 1 < len(xs) and j + 1 < len(ys)]
    return unary_union(boxes) if boxes else None


def _rectangularity(g):
    """area / bbox area: 1.0 for a rectangle, below it for an L or T."""
    x0, y0, x1, y1 = g.bounds
    bb = (x1 - x0) * (y1 - y0)
    return g.area / bb if bb > 0 else 0.0


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


def main():
    # ── bring up the app in-process (no server) ──────────────────────────────
    from apps.api import server as server_mod
    from packages.floorplan_core.index_store import build_index_store_from_npz
    from packages.floorplan_core.topo_signature import _label as canonical_label
    from packages.floorplan_core.rectangular_arrange import _label as raw_label
    from packages.floorplan_core.topo_sensors import (
        _usable_parts, _max_square_side, _bbox_min_side, USABLE_WIDTH_MM)
    from schema import MINIMUM_SIZES  # noqa: E402 (fitting-prototype dir on

    npz_path = REPO_ROOT / "data" / "index" / "rplan_index_v5.npz"
    print(f"loading index: {npz_path} ...", file=sys.stderr)
    server_mod.INDEX_STORE = build_index_store_from_npz(
        npz_path=str(npz_path),
        data_dir=str(server_mod.RPLAN_DATASET_DIR),
        logger=server_mod.logger,
    )
    print(f"  {len(server_mod.INDEX_STORE.names)} plans in index, "
          f"{len(server_mod.INDEX_STORE.plan_cache) // 2 if server_mod.INDEX_STORE.plan_cache else 0} "
          f"JSONs cached", file=sys.stderr)

    client = server_mod.app.test_client()

    targets = load_targets()

    out = {
        "generated_at": RUN_DATE,
        "d24_basis": "usable footprint (core square + USABLE_WIDTH_MM opening)",
        "mm_per_unit_used": MM_PER_UNIT,
        "mm_per_unit_gap": (
            "apps/web/src/main.js's actual /generate POST body (runGenerate, "
            "~line 3091) does NOT include mm_per_unit — only fp, boundary, "
            "entrance_point, typed_edges, requirements, swap. This script "
            "passes mm_per_unit=18000 (constants.js METRES_PER_UNIT*1000, == "
            "schema.SCALE) because that is the value the frontend's own scale "
            "convention implies, and generate()'s step-3 min_edge enforcement "
            "is OFF when mm_per_unit is absent (generate.py _min_edge() "
            "returns 0.0). The live app therefore never enforces min_edge "
            "during the solve; topo_sensors.check_min_room_size still judges "
            "with the correct scale only because its fallback (schema.SCALE) "
            "happens to equal the frontend's real scale."
        ),
        "label_canonicalization_bug": (
            "ROOT CAUSE, confirmed empirically (see per-room "
            "generate_report_min_edge_mm/generate_report_min_area_mm2, all "
            "0.0 below): generate.py imports _label from "
            "rectangular_arrange (packages/floorplan_core/rectangular_arrange.py "
            "line 128: `room.get('label', room.get('type','unknown'))`, a raw "
            "passthrough, no normalisation). Source RPlan plan JSON carries "
            "BOTH type=<int 0-9> and label=<PascalCase string, e.g. 'Kitchen', "
            "'DiningRoom', 'MasterBedroom'>. rectangular_arrange._label returns "
            "the raw label string, which never matches schema.MINIMUM_SIZES' "
            "lowercase/snake_case keys ('kitchen','dining','master_bedroom',...), "
            "so describe()'s MINIMUM_SIZES.get(label,{}) is {} for every "
            "room, and demands[rid]['min_edge']/['min_area'] default to 0.0 "
            "universally. generate()'s step-3 solver (_min_edge(rid)) therefore "
            "enforces NO minimum room dimension for ANY room, independent of "
            "mm_per_unit, and report.rooms' min_edge/min_area fields (meant for "
            "UI/QA use) are always 0 too. topo_sensors.check_min_room_size uses "
            "topo_signature._label instead (packages/floorplan_core/"
            "topo_signature.py line 90), which DOES canonicalise (int type -> "
            "ROOM_TYPE_MAP, or string label lowercased + _LABEL_ALIAS), so it "
            "alone computes the true requirement and correctly flags "
            "violations the generator never tried to avoid. This is a data "
            "mismatch between two _label() implementations with the same name "
            "in different modules, not a geometry problem."
        ),
        "boundaries": [],
    }

    counts = {"boundaries_run": 0, "plans_generated": 0, "plans_with_failure": 0,
              "failing_rooms": 0, "fails_core": 0, "fails_usable_area": 0,
              "fails_both": 0,
              "failing_rooms_by_cause": {"A": 0, "D": 0, "S-fill": 0, "S-narrow": 0,
                                         "S-clip": 0, "S-arm": 0, "W": 0,
                                         "C-clip": 0, "C-arm": 0, "B": 0, "E": 0},
              "failing_rooms_by_label": {},
              "core_lost_in_2b_recipient": 0,
              "lift_met_raw_area_not_usable": 0}

    for entry in targets:
        name = entry["name"]
        boundary, entrance_point, typed_edges, requirements = entry_to_request_parts(entry)
        boundary_area_units = _poly_area_units(boundary)
        boundary_area_mm2 = boundary_area_units * MM_PER_UNIT ** 2
        non_convex = _is_non_convex(boundary)
        counts["boundaries_run"] += 1

        retrieve_body = {
            "boundary": boundary,
            "entrance_point": entrance_point,
            "requirements": requirements,
            "topn": TOPN,
            "typed_edges": typed_edges,
        }
        rv = client.post("/retrieve", json=retrieve_body)
        rdata = rv.get_json()
        if rv.status_code != 200 or rdata.get("error"):
            out["boundaries"].append({
                "name": name, "error": f"/retrieve failed: "
                f"status={rv.status_code} body={rdata}"})
            continue

        plans = rdata.get("plans") or []
        b_record = {
            "name": name,
            "non_convex": non_convex,
            "boundary_area_units": boundary_area_units,
            "boundary_area_mm2": boundary_area_mm2,
            "retrieve_count": len(plans),
            "results": [],
        }

        for rank, plan in enumerate(plans[:TOPN], start=1):
            gen_body = {
                "fp": plan,
                "boundary": boundary,
                "entrance_point": entrance_point,
                "typed_edges": typed_edges,
                "requirements": requirements,
                "swap": None,
                "mm_per_unit": MM_PER_UNIT,
            }
            gv = client.post("/generate", json=gen_body)
            gdata = gv.get_json()
            if gv.status_code != 200 or gdata.get("error"):
                b_record["results"].append({
                    "rank": rank, "source_id": plan.get("retrieval", {}).get("source_id"),
                    "error": f"/generate failed: status={gv.status_code} body={gdata}"})
                continue

            counts["plans_generated"] += 1
            report = gdata.get("report") or {}
            sensors = gdata.get("sensors") or {}
            checks = sensors.get("checks") or []
            min_sz = next((c for c in checks if c.get("sensor") == "min_room_size"), None)
            min_sizes_met_plan = bool(report.get("min_sizes_met"))
            room_reports = {str(r["id"]): r for r in (report.get("rooms") or [])}
            rooms_by_id = {str(r["id"]): r for r in (gdata.get("rooms") or [])}
            boundary_contacts = report.get("boundary_contacts") or {}
            arms = boundary_contacts.get("arms") or []
            arm_recipient_ids = {str(a[0]) for a in arms}
            transfer_recipient_ids = {str(t["room"]) for t in
                                      (boundary_contacts.get("transfers") or [])}
            grid_lines = report.get("grid_lines") or {}
            bpoly = _clean(SPoly(boundary))

            # ROOT-CAUSE FINDING: generate.py's own demand bookkeeping
            # (build_spec -> report.rooms min_edge/min_area) looks up
            # MINIMUM_SIZES by rectangular_arrange._label(room), which is a
            # raw `room.get("label", room.get("type"))` passthrough — NOT
            # canonicalized. Source plans carry label="Kitchen"/"DiningRoom"/
            # "MasterBedroom" (PascalCase) alongside type=<int>; that raw
            # string never matches MINIMUM_SIZES' lowercase/snake_case keys,
            # so MINIMUM_SIZES.get(label) is {} and min_edge/min_area silently
            # default to 0.0 for every room, every plan (confirmed empirically
            # below: every report.rooms entry carries min_edge=min_area=0).
            # topo_sensors.check_min_room_size uses topo_signature._label
            # instead, which DOES canonicalize (int type -> ROOM_TYPE_MAP, or
            # string label lowercased + _LABEL_ALIAS), so it computes the real
            # requirement and is the only place in the pipeline that correctly
            # judges min size. This script recomputes the TRUE requirement the
            # same way the sensor does, rather than trusting report.rooms.
            true_req = {}
            for rid, room in rooms_by_id.items():
                raw_lbl = raw_label(room)
                canon_lbl = canonical_label(room)
                sizes = MINIMUM_SIZES.get(canon_lbl) or {}
                true_req[rid] = {
                    "raw_label": raw_lbl,
                    "canonical_label": canon_lbl,
                    "min_edge_mm": sizes.get("min_edge", 0.0),
                    "min_area_mm2": sizes.get("area", 0.0),
                    "generate_report_min_edge_mm": (room_reports.get(rid) or {}).get("min_edge"),
                    "generate_report_min_area_mm2": (room_reports.get(rid) or {}).get("min_area"),
                }

            sum_min_area_mm2 = sum(v["min_area_mm2"] for v in true_req.values())
            budget_infeasible = (sum_min_area_mm2 > boundary_area_mm2) or (not min_sizes_met_plan)

            result_record = {
                "rank": rank,
                "source_id": plan.get("retrieval", {}).get("source_id"),
                "feasible_stage0": gdata.get("feasible"),
                "min_sizes_met_plan": min_sizes_met_plan,
                "sum_min_area_mm2_true": sum_min_area_mm2,
                "budget_infeasible": budget_infeasible,
                "min_room_size_sensor_ok": min_sz.get("ok") if min_sz else None,
                "min_room_size_sensor_detail": min_sz.get("detail") if min_sz else None,
                "arm_recipients_this_plan": sorted(arm_recipient_ids),
                "generate_report_demands_all_zero": all(
                    (v["generate_report_min_edge_mm"] in (0, 0.0, None))
                    and (v["generate_report_min_area_mm2"] in (0, 0.0, None))
                    for v in true_req.values()) if true_req else None,
                "failing_rooms": [],
            }

            failing_ids = set(str(x) for x in (min_sz.get("failing_rooms") if min_sz else []) or [])
            if failing_ids:
                counts["plans_with_failure"] += 1

            for rid in sorted(failing_ids):
                rr = room_reports.get(rid)
                tr = true_req.get(rid)
                if rr is None or tr is None:
                    result_record["failing_rooms"].append({
                        "room_id": rid, "cause": "E",
                        "cause_detail": "room id in sensor.failing_rooms but not found "
                                        "in report.rooms / gdata.rooms — data gap"})
                    counts["failing_rooms_by_cause"]["E"] += 1
                    continue

                label = tr["canonical_label"]
                min_edge_req_mm = tr["min_edge_mm"]
                min_area_req_mm2 = tr["min_area_mm2"]
                actual_edge_mm = (rr.get("short") or 0.0) * MM_PER_UNIT
                actual_area_mm2 = (rr.get("area") or 0.0) * MM_PER_UNIT ** 2
                target_frac = rr.get("target_frac")
                target_area_mm2 = (target_frac * boundary_area_mm2
                                   if target_frac is not None else None)
                # usable-footprint D24, same functions and slack as the sensor
                mm = MM_PER_UNIT
                core_w = max(min_edge_req_mm - 1.0, 0.0) / mm
                use_w = (USABLE_WIDTH_MM - 1.0) / mm
                final_poly = rooms_by_id[rid].get("polygon") or []
                final_g = _clean(SPoly(final_poly)) if len(final_poly) >= 3 else None
                core_final = bool(final_g is not None and _usable_parts(final_poly, core_w))
                if final_g is not None and not core_final:
                    max_sq_mm = _max_square_side(final_poly, _bbox_min_side(final_poly)) * mm
                else:
                    max_sq_mm = None
                use_parts = _usable_parts(final_poly, use_w) if final_g is not None else []
                core_parts = _usable_parts(final_poly, core_w) if core_final else []
                home = next((p for p in use_parts if core_parts and p.intersects(core_parts[0])),
                            use_parts[0] if use_parts else None)
                usable_mm2 = home.area * mm * mm if home is not None else 0.0
                fails_core = not core_final
                fails_usable = usable_mm2 < min_area_req_mm2 - 1e4

                # the same room at each stage: step-3 cells -> clip -> final
                cell_g = _cell_union(grid_lines, rid)
                clip_g = _clean(cell_g).intersection(bpoly) if cell_g is not None else None

                def _core_fits(g):
                    if g is None or g.is_empty:
                        return None
                    if g.geom_type != "Polygon":
                        g = max(g.geoms, key=lambda p: p.area)
                    return bool(_usable_parts(list(g.exterior.coords), core_w))

                core_cells = _core_fits(cell_g)
                core_clip = _core_fits(clip_g)
                cell_rect = (round(_rectangularity(cell_g), 3)
                             if cell_g is not None and not cell_g.is_empty else None)

                fails_edge = fails_core
                fails_area = fails_usable
                # area provenance exposed by generate() (boundary units²)
                rect_area_mm2 = (rr.get("area_rect") or 0.0) * MM_PER_UNIT ** 2
                clip_area_mm2 = (rr.get("area_clip") or 0.0) * MM_PER_UNIT ** 2

                room_rec = {
                    "room_id": rid,
                    "label": label,
                    "raw_label_generate_used": tr["raw_label"],
                    "generate_report_min_edge_mm": tr["generate_report_min_edge_mm"],
                    "generate_report_min_area_mm2": tr["generate_report_min_area_mm2"],
                    "min_edge_required_mm": min_edge_req_mm,
                    "min_edge_actual_mm": round(actual_edge_mm, 1),
                    "min_area_required_mm2": min_area_req_mm2,
                    "min_area_actual_mm2": round(actual_area_mm2, 1),
                    "target_area_mm2": (round(target_area_mm2, 1)
                                        if target_area_mm2 is not None else None),
                    "area_after_step3_before_clip_mm2": round(rect_area_mm2, 1),
                    "area_after_clip_mm2": round(clip_area_mm2, 1),
                    "final_area_mm2": round(actual_area_mm2, 1),
                    "is_4b_arm_recipient": rid in arm_recipient_ids,
                    "is_4b_arm_donor": bool(rr.get("arm_donor_to")),
                    "boundary_non_convex": non_convex,
                    "fails_core": fails_core,
                    "fails_usable_area": fails_usable,
                    "usable_area_mm2": round(usable_mm2, 1),
                    "max_square_side_mm": (round(max_sq_mm, 1)
                                           if max_sq_mm is not None else None),
                    "bbox_min_side_mm": round(actual_edge_mm, 1),
                    "core_fits_cells": core_cells,
                    "core_fits_clip": core_clip,
                    "core_fits_final": core_final,
                    "cells_rectangularity": cell_rect,
                    "final_rectangularity": (round(_rectangularity(final_g), 3)
                                             if final_g is not None else None),
                    "final_vertices": len(final_poly),
                    "is_2b_transfer_recipient": rid in transfer_recipient_ids,
                    "area_lift": rr.get("area_lift"),
                }

                if budget_infeasible:
                    cause = "A"
                    detail = ("sum(true min_area) > boundary area" if sum_min_area_mm2 > boundary_area_mm2
                              else "plan-level min_sizes_met == False (step-3 solve infeasible)")
                elif target_area_mm2 is not None and target_area_mm2 < min_area_req_mm2:
                    cause = "D"
                    detail = (f"target_frac×boundary_area={target_area_mm2:.0f}mm² "
                              f"< true min_area {min_area_req_mm2:.0f}mm²")
                elif fails_core:
                    # core shape: find the stage where the min_edge square stops fitting
                    if core_cells is False:
                        cause = "S-narrow" if (cell_rect or 0) >= 0.999 else "S-fill"
                    elif core_clip is False:
                        cause = "S-clip"
                    elif rr.get("arm_donor_to"):
                        cause = "S-arm"
                    else:
                        # cells and clip both hold a core but the final room does not,
                        # and the room gave no arm: a reshaping this script cannot see
                        cause = "E"
                    if rid in transfer_recipient_ids:
                        counts["core_lost_in_2b_recipient"] += 1
                    detail = (f"core {min_edge_req_mm:.0f} mm: cells {core_cells} "
                              f"(rect {cell_rect}) -> clip {core_clip} -> final {core_final}; "
                              f"max square {max_sq_mm or 0:.0f} mm, bbox min side "
                              f"{actual_edge_mm:.0f} mm; usable {usable_mm2:.0f}/"
                              f"{min_area_req_mm2:.0f} mm²; arm donor to "
                              f"{rr.get('arm_donor_to') or []}; 2b recipient "
                              f"{rid in transfer_recipient_ids}")
                else:
                    # area only. Usable area short while raw area is enough = the
                    # room's area sits in parts narrower than USABLE_WIDTH_MM (W).
                    # Otherwise area provenance: step-3 rect -> clip -> 4b arm carve;
                    # a loss at a stage = met the floor before it, not after.
                    lost_to_clip = rect_area_mm2 >= min_area_req_mm2 > clip_area_mm2
                    lost_to_arm = bool(rr.get("arm_donor_to")) and (
                        clip_area_mm2 >= min_area_req_mm2 > actual_area_mm2)
                    if actual_area_mm2 >= min_area_req_mm2:
                        cause = "W"
                        if (rr.get("area_lift") or 1.0) > 1.0:
                            counts["lift_met_raw_area_not_usable"] += 1
                    elif lost_to_arm:
                        cause = "C-arm"
                    elif lost_to_clip:
                        cause = "C-clip"
                    else:
                        cause = "B"
                    detail = (f"usable {usable_mm2:.0f} of final {actual_area_mm2:.0f} mm² "
                              f"(min {min_area_req_mm2:.0f}); rect {rect_area_mm2:.0f} -> "
                              f"clip {clip_area_mm2:.0f}; lift {rr.get('area_lift')}; "
                              f"arm donor to {rr.get('arm_donor_to') or []}")

                # where the core is lost, for every core failure — A and D are
                # plan/demand conditions and must not hide the shape signal
                if fails_core:
                    if core_cells is False:
                        stage = "cells-rect" if (cell_rect or 0) >= 0.999 else "cells-LT"
                    elif core_clip is False:
                        stage = "clip"
                    else:
                        stage = "arm" if rr.get("arm_donor_to") else "unknown"
                    room_rec["core_lost_at"] = stage
                    cs = counts.setdefault("core_lost_at_all", {})
                    cs[stage] = cs.get(stage, 0) + 1

                counts["failing_rooms"] += 1
                counts["fails_core"] += int(fails_core)
                counts["fails_usable_area"] += int(fails_usable)
                counts["fails_both"] += int(fails_core and fails_usable)

                room_rec["cause"] = cause
                room_rec["cause_detail"] = detail
                result_record["failing_rooms"].append(room_rec)
                counts["failing_rooms_by_cause"][cause] = counts["failing_rooms_by_cause"].get(cause, 0) + 1
                counts["failing_rooms_by_label"][label] = counts["failing_rooms_by_label"].get(label, 0) + 1

            b_record["results"].append(result_record)

        out["boundaries"].append(b_record)

    # apply_best_expansion / check_plan_compliance — inspected by reading
    # fit_planner.py and grepping callers; recorded here for the report, not
    # computed live (no route exercises them in the current app).
    out["apply_best_expansion_finding"] = (
        "apply_best_expansion and check_plan_compliance live in "
        "packages/floorplan_core/fit_planner.py and are imported only by "
        "apps/api/server.py's /fit_harness route (around line 907), where "
        "check_plan_compliance gates a per-room compliance loop and "
        "apply_best_expansion applies the best interior-edge push move for a "
        "non-compliant room. /fit_harness belongs to the retired ARAP realiser "
        "lineage (ARCHITECTURE.md 'Retired / legacy'): it is absent from "
        "apps/web/vite.config.js's API_ROUTES proxy list and apps/web/src has "
        "no call to it, so no current route or frontend path uses these two "
        "functions — the live /generate path (Stage-0 node-driven solver) does "
        "not call them."
    )
    out["counts"] = counts

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT_PATH}", file=sys.stderr)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
