#!/usr/bin/env python3
"""Run the existing floor plate generators on the saved library plates.

Baseline only: no generator code changes. For each plate and unit mix it runs
generate_initial_layout and generate_subdivision_response (harness mode), then
measures the output and draws it with the style sheet's conventions.

Run: .venv/bin/python tools/diag_plate_generator.py [--tag NAME]
Output: data/outputs/diagnostics/plate_generator_<tag>.json and .png
"""
import argparse
import contextlib
import datetime
import io
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402
from shapely.geometry import Polygon as SPoly  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from packages.floor_plate_design import library  # noqa: E402
from packages.floor_plate_design.assembler import assemble_plate  # noqa: E402
from packages.floor_plate_design.generator import (  # noqa: E402
    generate_initial_layout,
    generate_subdivision_response,
)

# Unit split is not stored with a plate, so these are test inputs.
MIXES = {"2x1B+1x2B": {"1bed": 2, "2bed": 1}, "2x1B+2x2B": {"1bed": 2, "2bed": 2}}
TARGET = {"1bed": 55, "2bed": 70}
MIN_AREA = {"1bed": 36, "2bed": 50}
CONSTRAINTS = {"corridor_width": 1500, "max_unit_depth": 8000, "min_entrance_overlap": 1200,
               "min_common_area": 20, "max_travel_distance": 30000, "core_type": "stair_lift"}
SEED = 0
# The request the app's "generate from drawn boundary" sends (main.js).
APP_SEED = 7
SUBDIVISION_CONSTRAINTS = {
    "min_depth_m": 5.5, "min_area_m2": 35, "area_tolerance_m2": 0.01, "max_candidates": 8,
    "corridor_width": 1200, "core_type": "stair_lift",
    "unit_specs": {"1bed": {"target_area_m2": 55, "min_area_m2": 35, "min_depth_m": 5.5},
                   "2bed": {"target_area_m2": 70, "min_area_m2": 50, "min_depth_m": 5.5},
                   "3bed": {"target_area_m2": 90, "min_area_m2": 65, "min_depth_m": 5.5}},
}
SUBDIVISION_SCORING = {"placed_unit_reward": 10, "type_priority": {"1bed": 1, "2bed": 2, "3bed": 3},
                       "shortfall_penalty": 8, "residual_area_penalty": 2, "residual_fragmentation_penalty": 3,
                       "area_deviation_penalty": 1, "compactness_penalty": 0.5}


def to_mm(boundary_m):
    return [(x * 1000.0, y * 1000.0) for x, y in boundary_m]


def spoly(poly):
    p = SPoly(poly)
    return p if p.is_valid else p.buffer(0)


def is_rect(poly, tol=1.0):
    p = spoly(poly)
    return p.area > 0 and abs(p.minimum_rotated_rectangle.area - p.area) / p.area * 100 < tol


def measure(boundary_mm, units, cores, corridors, residual, shortfall=None):
    """All geometry in mm. Share figures are of the input outline."""
    outline = spoly(boundary_mm)
    a = outline.area
    area_m2 = lambda items: sum(spoly(i).area for i in items if len(i) >= 3) / 1e6
    all_parts = [spoly(i) for i in units + cores + corridors + residual if len(i) >= 3]
    outside = sum(g.difference(outline.buffer(1)).area for g in all_parts) / 1e6
    u_shapes = [spoly(u) for u in units]
    overlap = 0.0
    for i, s in enumerate(u_shapes):
        for t in u_shapes[i + 1:]:
            overlap += s.intersection(t).area
    covered = unary_union(all_parts).intersection(outline).area if all_parts else 0.0
    commons = area_m2(cores) + area_m2(corridors)
    exterior = outline.exterior
    no_front = 0
    for u in u_shapes:
        if u.exterior.intersection(exterior.buffer(50)).length < 900:
            no_front += 1
    return {
        "units": len(units),
        "unit_areas_m2": [round(s.area / 1e6, 1) for s in u_shapes],
        "non_rectangular_units": sum(not is_rect(u) for u in units),
        "units_with_no_facade": no_front,
        "core_count": len(cores),
        "core_m2": round(area_m2(cores), 1),
        "corridor_m2": round(area_m2(corridors), 1),
        "commons_pct_of_plate": round(100 * commons * 1e6 / a, 1),
        "residual_m2": round(area_m2(residual), 1),
        "uncovered_m2": round((a - covered) / 1e6, 1),
        "outside_outline_m2": round(outside, 2),
        "unit_overlap_m2": round(overlap / 1e6, 2),
        "shortfall": shortfall or {},
    }


def run_initial(boundary_mm, mix):
    unit_mix = [{"type": t, "count": n, "min_area": MIN_AREA[t], "target_area": TARGET[t]} for t, n in mix.items()]
    with contextlib.redirect_stdout(io.StringIO()):
        fp = generate_initial_layout(list(boundary_mm), unit_mix, dict(CONSTRAINTS))
    units = [u.polygon for u in fp.units]
    cores = [c.polygon for c in fp.cores]
    corridors = [c.polygon for c in fp.corridors]
    residual = [list(r) for r in fp.unresolved_regions]
    snap = fp.compliance_snapshot
    # This route translates the boundary to its origin, so outline and parts share fp.boundary.
    return {
        "boundary": list(fp.boundary), "units": units, "cores": cores, "corridors": corridors,
        "residual": residual,
        "metrics": measure(fp.boundary, units, cores, corridors, residual),
        "typology": f"{fp.typology.type.value}/{getattr(fp.typology.sub_type, 'value', fp.typology.sub_type)}",
        "typology_reason": fp.typology.reasoning,
        "violations": list(snap.violations) if snap else [],
        "warnings": list(snap.warnings) if snap else [],
    }


def run_subdivision(boundary_mm, mix):
    with contextlib.redirect_stdout(io.StringIO()):
        res = generate_subdivision_response(list(boundary_mm), dict(mix), constraints=dict(SUBDIVISION_CONSTRAINTS),
                                            scoring=dict(SUBDIVISION_SCORING), mode="single", seed=APP_SEED)
    units = [u["polygon"] for u in res.get("placed_units", [])]
    cores = [c["polygon"] for c in res.get("cores", [])]
    corridors = [c["polygon"] for c in res.get("corridors", [])]
    residual = [z["polygon"] for z in res.get("residual_zones", [])]
    return {
        "boundary": list(boundary_mm), "units": units, "cores": cores, "corridors": corridors,
        "residual": residual,
        "metrics": measure(boundary_mm, units, cores, corridors, residual, res.get("shortfall")),
        "typology": "subdivide_floor_plate (app request)",
        "typology_reason": "",
        "violations": [],
        "warnings": list(res.get("warnings", [])),
        "mix_adjustment": res.get("mix_adjustment"),
        "unit_types": [u.get("type") for u in res.get("placed_units", [])],
        "typed_edges": [u.get("typed_edges") for u in res.get("placed_units", [])],
    }


def run_assembler(boundary_mm, mix_w, street_edges=(), north_deg=0.0):
    """Third route (2026-11-06): the plate assembler. Weights over 1B1b, 2B1b, 2B2b, 3B2b."""
    res = assemble_plate([(x / 1000.0, y / 1000.0) for x, y in boundary_mm], mix_w, street_edges, north_deg)
    mm = lambda poly: [(x * 1000.0, y * 1000.0) for x, y in poly]
    units = [mm(u["polygon"]) for u in res["units"]]
    cores = [mm(c["polygon"]) for c in res["cores"]]
    corridors = [mm(c["polygon"]) for c in res["corridors"]]
    residual = [mm(r) for r in res["residual"]]
    m = res["metrics"]
    met = measure([(x * 1000.0, y * 1000.0) for x, y in res["boundary"]] if res["boundary"] else boundary_mm,
                  units, cores, corridors, residual)
    if res["status"] != "ok":
        met["shortfall"] = {"status": res["status"], "reason": res["reason"], "rejects": res["rejects"]}
    return {
        "boundary": mm(res["boundary"]) if res["boundary"] else list(boundary_mm), "units": units, "cores": cores,
        "corridors": corridors, "residual": residual, "metrics": met,
        "typology": f"assembler/{res['status']}", "typology_reason": res["reason"],
        "violations": [], "warnings": [], "unit_types": [u["type"] for u in res["units"]],
        "assembler": {k: m.get(k) for k in ("achieved_counts", "requested_counts", "mix_error_per_type", "commons_pct",
                                            "max_walk_m", "residual_m2", "candidates_tried", "candidates_survived")},
        "rejects": res["rejects"],
    }


ROUTES = {"initial_layout": run_initial, "subdivision": run_subdivision}
# Assembler mixes are weights over (1B1b, 2B1b, 2B2b, 3B2b).
ASSEMBLER_MIXES = {"w1100": {"1B1b": 1, "2B1b": 1}, "w1010": {"1B1b": 1, "2B2b": 1},
                   "w1101": {"1B1b": 1, "2B1b": 1, "3B2b": 1}}
INK, CYAN, YELLOW, MAGENTA, GROUND = "#161616", "#00AEEF", "#FFF200", "#EC008C", "#F4F4F2"


def draw(ax, r):
    def add(poly, face, lw, edge=INK, z=2, hatch=None):
        if len(poly) >= 3:
            ax.add_patch(MplPolygon([(x / 1000, y / 1000) for x, y in poly], closed=True, facecolor=face,
                                    edgecolor=edge, linewidth=lw, zorder=z, hatch=hatch))
    for z_poly in r["residual"]:
        add(z_poly, YELLOW, .5, z=2)
    for u in r["units"]:
        add(u, GROUND, 1.3, z=3)
    for c in r["corridors"]:
        add(c, CYAN, .5, z=4)
    for c in r["cores"]:
        add(c, INK, 1.2, z=5)
    add(r["boundary"], "none", 2.0, z=6)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.autoscale_view()


def caption(r):
    m = r["metrics"]
    extra = f" | shortfall {m['shortfall']}" if m["shortfall"] else ""
    return (f"{r['typology']}\n{m['units']} units {m['unit_areas_m2']} m²{extra}\n"
            f"cores {m['core_count']}, commons {m['commons_pct_of_plate']}%, residual {m['residual_m2']} m², "
            f"uncovered {m['uncovered_m2']} m²\nnon-rect {m['non_rectangular_units']}, no facade "
            f"{m['units_with_no_facade']}, outside {m['outside_outline_m2']} m²")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=datetime.date.today().isoformat())
    args = ap.parse_args()
    plates = library.list_plates()["plates"]
    out = {"seed": SEED, "constraints": CONSTRAINTS, "mixes": MIXES, "assembler_mixes": ASSEMBLER_MIXES, "plates": []}
    rows = []
    for p in plates:
        rows += [(p, mn, rn) for mn in MIXES for rn in ROUTES]
        rows += [(p, mn, "assembler") for mn in ASSEMBLER_MIXES]
    ncol = len(MIXES) * len(ROUTES) + len(ASSEMBLER_MIXES)
    fig, axes = plt.subplots(len(plates), ncol, figsize=(4.2 * ncol, 5.0 * len(plates)), squeeze=False)
    fig.patch.set_facecolor(GROUND)
    for i, (p, mix_name, route) in enumerate(rows):
        ax = axes[i // ncol][i % ncol]
        title = f"{p['id']} · {mix_name} · {route}"
        boundary_mm = to_mm(p["boundary_m"])
        rec = {"plate": p["id"], "name": p["name"], "area_m2": p["area_m2"], "mix": mix_name, "route": route}
        try:
            if route == "assembler":
                r = run_assembler(boundary_mm, ASSEMBLER_MIXES[mix_name], p.get("street_edges", []), p.get("north_deg", 0.0))
            else:
                r = ROUTES[route](boundary_mm, MIXES[mix_name])
            rec.update({k: v for k, v in r.items() if k != "boundary"})
            rec["ok"] = True
            draw(ax, r)
            ax.set_title(title + "\n" + caption(r), fontsize=7, loc="left", family="monospace")
        except Exception as exc:  # a failure on a plate is a result, not a crash
            rec.update({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                        "trace": traceback.format_exc(limit=3)})
            ax.axis("off")
            ax.set_title(title + f"\nERROR {type(exc).__name__}: {str(exc)[:80]}", fontsize=7, loc="left",
                         family="monospace", color=MAGENTA)
        out["plates"].append(rec)
    d = REPO / "data" / "outputs" / "diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    jp, pp = d / f"plate_generator_{args.tag}.json", d / f"plate_generator_{args.tag}.png"
    jp.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    fig.tight_layout()
    fig.savefig(pp, dpi=130, facecolor=GROUND)
    print(jp)
    print(pp)
    for rec in out["plates"]:
        if rec["ok"]:
            m = rec["metrics"]
            print(f"{rec['plate']:18s} {rec['mix']:11s} {rec['route']:15s} {rec['typology'][:28]:28s} units {m['units']} "
                  f"{m['unit_areas_m2']} commons {m['commons_pct_of_plate']}% resid {m['residual_m2']} uncov {m['uncovered_m2']} "
                  f"nonrect {m['non_rectangular_units']} nofac {m['units_with_no_facade']} out {m['outside_outline_m2']} "
                  f"overlap {m['unit_overlap_m2']} short {m['shortfall']} viol {len(rec['violations'])}")
        else:
            print(f"{rec['plate']:18s} {rec['mix']:11s} {rec['route']:15s} ERROR {rec['error'][:110]}")


if __name__ == "__main__":
    main()
