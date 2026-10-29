"""
diag_min_size.py — diagnose min_room_size (D24) sensor failures on generated
plans, for two fixed test boundaries.

Replays the live app's pipeline (retrieve -> top 3 -> generate) through
apps/api/server.py's Flask app via test_client() — no server process started,
nothing under packages/ or apps/ is modified. For every room that fails
min_edge or min_area, records the numbers and assigns exactly one cause
(A/D/C/B/E, in that priority order — see module docstring in the task brief).

Read-only diagnostic. Output: data/outputs/diagnostics/min_size_<date>.json.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]          # 02_Retrieval_v02/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shapely.geometry import Polygon as SPoly  # noqa: E402

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
OUT_PATH = OUT_DIR / ("min_size_2026-10-29"
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
        "generated_at": "2026-10-29",
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
            "so spec_from_plan()'s MINIMUM_SIZES.get(label,{}) is {} for every "
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
              "failing_rooms_by_cause": {"A": 0, "B": 0, "C-clip": 0, "C-arm": 0,
                                         "D": 0, "E": 0, "B-or-C": 0},
              "failing_rooms_by_label": {}}

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

            # ROOT-CAUSE FINDING: generate.py's own demand bookkeeping
            # (spec_from_plan -> report.rooms min_edge/min_area) looks up
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
                fails_edge = actual_edge_mm < min_edge_req_mm
                fails_area = actual_area_mm2 < min_area_req_mm2
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
                    "fails_edge": fails_edge,
                    "fails_area": fails_area,
                }

                if budget_infeasible:
                    cause = "A"
                    detail = ("sum(true min_area) > boundary area" if sum_min_area_mm2 > boundary_area_mm2
                              else "plan-level min_sizes_met == False (step-3 solve infeasible)")
                elif target_area_mm2 is not None and target_area_mm2 < min_area_req_mm2:
                    cause = "D"
                    detail = (f"target_frac×boundary_area={target_area_mm2:.0f}mm² "
                              f"< true min_area {min_area_req_mm2:.0f}mm²")
                else:
                    # Area provenance: step-3 rect -> clip -> 4b arm carve.
                    # A loss at a stage = the room met the floor before it and
                    # not after. Edge-only failures use a >2% area drop as the
                    # signal that the stage reshaped the room.
                    lost_to_clip = ((rect_area_mm2 >= min_area_req_mm2 > clip_area_mm2)
                                    if fails_area else clip_area_mm2 < 0.98 * rect_area_mm2)
                    lost_to_arm = bool(rr.get("arm_donor_to")) and (
                        (clip_area_mm2 >= min_area_req_mm2 > actual_area_mm2)
                        if fails_area else actual_area_mm2 < 0.98 * clip_area_mm2)
                    if lost_to_arm:
                        cause = "C-arm"
                    elif lost_to_clip:
                        cause = "C-clip"
                    else:
                        cause = "B"
                    detail = (f"rect {rect_area_mm2:.0f} -> clip {clip_area_mm2:.0f} -> "
                              f"final {actual_area_mm2:.0f} mm² (min {min_area_req_mm2:.0f}); "
                              f"edge {actual_edge_mm:.0f}/{min_edge_req_mm:.0f} mm; "
                              f"arm donor to {rr.get('arm_donor_to') or []}")

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
