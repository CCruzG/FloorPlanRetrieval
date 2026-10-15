#!/usr/bin/env python3
"""
tools/test_s1_baseline.py — Stage 1: baseline affine-mapping diagnostic.

Shows the original floor plan (boundary + room polygons + centroids) alongside
the affine-mapped centroids on a target boundary.  Two panels, side by side.

Usage:
    python tools/test_s1_baseline.py                          # self-fit, plan 987
    python tools/test_s1_baseline.py --target rect            # map to rectangle
    python tools/test_s1_baseline.py --plan 290 --target rect
    python tools/test_s1_baseline.py --target rect --rot 90   # with 90° rotation

Output: test_s1_baseline.png  (or --out path)
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "fixtures" / "plans" / "test_data"

# ── room styling ──────────────────────────────────────────────────────────────

ROOM_COLOR = {
    0: "#F4A460",  # Living
    1: "#6495ED",  # MasterBed
    2: "#90EE90",  # Kitchen
    3: "#FFFACD",  # Bathroom
    4: "#FFA07A",  # Dining
    5: "#87CEEB",  # Bedroom
    7: "#87CEEB",  # Bedroom
    8: "#87CEEB",  # Bedroom
    9: "#98FB98",  # Balcony
    11: "#D3D3D3", # Storage
}

SHORT_LABEL = {
    0: "Living", 1: "MasterBed", 2: "Kitchen", 3: "Bath",
    4: "Dining",  5: "Bed",  7: "Bed",  8: "Bed",
    9: "Balcony", 11: "Storage",
}

# ── geometry helpers ──────────────────────────────────────────────────────────

def poly_centroid(pts):
    n = len(pts)
    if n == 0:
        return 0.0, 0.0
    return sum(v[0] for v in pts) / n, sum(v[1] for v in pts) / n


def bdry_centroid(pts):
    """Shoelace polygon centroid — matches JS bdryCenter()."""
    n = len(pts)
    area = cx = cy = 0.0
    for i in range(n):
        x0, y0 = pts[i][0], pts[i][1]
        x1, y1 = pts[(i + 1) % n][0], pts[(i + 1) % n][1]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area /= 2.0
    if abs(area) < 1e-12:
        return poly_centroid(pts)
    return cx / (6 * area), cy / (6 * area)


def rotate_pt(x, y, deg, cx, cy):
    """Rotate around (cx, cy). Positive = clockwise in Y-down screen space."""
    a = math.radians(deg)
    dx, dy = x - cx, y - cy
    return (cx + dx * math.cos(a) - dy * math.sin(a),
            cy + dx * math.sin(a) + dy * math.cos(a))


def apply_transform(x, y, rot, mirror, rx, ry):
    if mirror:
        x = 2 * rx - x
    if rot:
        x, y = rotate_pt(x, y, rot, rx, ry)
    return x, y


def bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def affine(x, y, src_bb, tgt_bb):
    sx0, sy0, sx1, sy1 = src_bb
    tx0, ty0, tx1, ty1 = tgt_bb
    sw = (sx1 - sx0) or 1e-9
    sh = (sy1 - sy0) or 1e-9
    return (tx0 + (x - sx0) / sw * (tx1 - tx0),
            ty0 + (y - sy0) / sh * (ty1 - ty0))

# ── target boundary presets ───────────────────────────────────────────────────

TARGET_PRESETS = {
    # entrance will be on the TOP edge (same as plan 987)
    "rect": [[0.10, 0.10], [0.72, 0.10], [0.72, 0.82], [0.10, 0.82]],

    # entrance on the LEFT edge — triggers the rotation problem
    "rect_left": [[0.10, 0.10], [0.72, 0.10], [0.72, 0.82], [0.10, 0.82]],

    # a different L-shape
    "lshape": [
        [0.10, 0.10], [0.72, 0.10], [0.72, 0.45],
        [0.50, 0.45], [0.50, 0.82], [0.10, 0.82],
    ],
}

# ── drawing helpers ───────────────────────────────────────────────────────────

def setup_ax(ax, title):
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05, -0.05)   # Y-down: 0 at top, 1 at bottom
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("x  →")
    ax.set_ylabel("y  ↓")
    ax.grid(True, alpha=0.25, linewidth=0.5)


def draw_boundary(ax, bdry_xy, color="#333333", lw=2.0, fill_color="#F5F5F5"):
    poly = MplPolygon(bdry_xy, closed=True,
                      facecolor=fill_color, edgecolor=color, linewidth=lw, zorder=1)
    ax.add_patch(poly)


def draw_room_polygon(ax, poly_xy, stype, alpha=0.35):
    color = ROOM_COLOR.get(stype, "#CCCCCC")
    patch = MplPolygon(poly_xy, closed=True,
                       facecolor=color, edgecolor="#999999",
                       linewidth=0.8, alpha=alpha, zorder=2)
    ax.add_patch(patch)


def draw_centroid(ax, x, y, stype, label, show_label=True):
    color = ROOM_COLOR.get(stype, "#CCCCCC")
    ax.plot(x, y, "o", color=color, markersize=13,
            markeredgecolor="#222222", markeredgewidth=1.2, zorder=4)
    if show_label:
        ax.annotate(label, (x, y),
                    textcoords="offset points", xytext=(0, 9),
                    fontsize=7.5, ha="center", va="bottom", zorder=5)


def draw_entrance(ax, x, y, label="Entry"):
    ax.plot(x, y, "D", color="#FF6200", markersize=11,
            markeredgecolor="#222222", markeredgewidth=1.2, zorder=5)
    ax.annotate(label, (x, y),
                textcoords="offset points", xytext=(0, 9),
                fontsize=7.5, ha="center", va="bottom",
                color="#FF6200", fontweight="bold", zorder=6)

# ── main logic ────────────────────────────────────────────────────────────────

def load_plan(plan_id):
    path = DATA_DIR / f"{plan_id}.json"
    return json.loads(path.read_text())


def run(plan_id, target_key, rot, mirror, out_path):
    plan = load_plan(plan_id)

    # --- extract plan data ---
    fp_bdry_raw = plan.get("boundary", [])                    # [[x,y,dir,door],...]
    fp_bdry = [[v[0], v[1]] for v in fp_bdry_raw]
    rooms = plan.get("rooms", [])
    ent_raw = plan.get("entrance", [])
    ent_orig = (
        ((ent_raw[0] + ent_raw[2]) / 2, (ent_raw[1] + ent_raw[3]) / 2)
        if len(ent_raw) >= 4 else None
    )

    # --- target boundary ---
    if target_key == "self":
        tgt_bdry = fp_bdry
    elif target_key in TARGET_PRESETS:
        tgt_bdry = TARGET_PRESETS[target_key]
    else:
        raise ValueError(f"Unknown target: {target_key!r}. Options: self, {', '.join(TARGET_PRESETS)}")

    # --- transform: apply rot/mirror to plan space, then affine to target ---
    if rot or mirror:
        rc_x, rc_y = bdry_centroid(fp_bdry)
    else:
        rc_x, rc_y = 0.0, 0.0

    # Rotate the plan boundary to get the reference bbox for the affine
    t_bdry = [apply_transform(v[0], v[1], rot, mirror, rc_x, rc_y) for v in fp_bdry]
    src_bb = bbox(t_bdry)
    tgt_bb = bbox(tgt_bdry)

    # Map entrance
    ent_mapped = None
    if ent_orig:
        ex, ey = apply_transform(ent_orig[0], ent_orig[1], rot, mirror, rc_x, rc_y)
        ent_mapped = affine(ex, ey, src_bb, tgt_bb)

    # Map room centroids
    room_info = []  # (id, label, stype, orig_cx, orig_cy, mapped_cx, mapped_cy)
    for r in rooms:
        poly = r.get("polygon", [])
        if not poly:
            continue
        cx, cy = poly_centroid(poly)
        tx, ty = apply_transform(cx, cy, rot, mirror, rc_x, rc_y)
        mx, my = affine(tx, ty, src_bb, tgt_bb)
        room_info.append((
            r["id"],
            SHORT_LABEL.get(r.get("source_type", -1), r.get("label", r["id"])),
            r.get("source_type", -1),
            cx, cy,
            mx, my,
        ))

    # ── figure ────────────────────────────────────────────────────────────────
    fig, (ax_orig, ax_mapped) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f"Stage 1 — Baseline  |  plan {plan_id}  |  target: {target_key}"
        f"  |  rot: {rot}°  mirror: {mirror}",
        fontsize=12,
    )

    # ── LEFT panel: original plan ─────────────────────────────────────────────
    setup_ax(ax_orig, f"Original plan {plan_id}")
    draw_boundary(ax_orig, fp_bdry)

    for r in rooms:
        poly = r.get("polygon", [])
        if poly:
            draw_room_polygon(ax_orig, poly, r.get("source_type", -1))

    for rid, label, stype, cx, cy, mx, my in room_info:
        draw_centroid(ax_orig, cx, cy, stype, label)

    if ent_orig:
        draw_entrance(ax_orig, ent_orig[0], ent_orig[1])

    ax_orig.annotate(
        f"{len(rooms)} rooms  |  entrance: ({ent_orig[0]:.3f}, {ent_orig[1]:.3f})" if ent_orig else f"{len(rooms)} rooms",
        xy=(0.01, 0.01), xycoords="axes fraction",
        fontsize=8, color="#555555",
    )

    # ── RIGHT panel: affine-mapped centroids ──────────────────────────────────
    setup_ax(ax_mapped, f"Affine mapped → {target_key}  (rot={rot}°)")
    draw_boundary(ax_mapped, tgt_bdry)

    for rid, label, stype, cx, cy, mx, my in room_info:
        draw_centroid(ax_mapped, mx, my, stype, label)

    if ent_mapped:
        draw_entrance(ax_mapped, ent_mapped[0], ent_mapped[1])

    # Mark each mapped centroid as inside/outside the target boundary bbox
    # (visual quick-check — proper point-in-polygon comes in Stage 2)
    bb = tgt_bb
    for rid, label, stype, cx, cy, mx, my in room_info:
        inside = bb[0] <= mx <= bb[2] and bb[1] <= my <= bb[3]
        ax_mapped.plot(mx, my, "x" if not inside else "",
                       color="red", markersize=8, markeredgewidth=2, zorder=6)

    ax_mapped.annotate(
        f"src bbox: x[{src_bb[0]:.2f},{src_bb[2]:.2f}]  y[{src_bb[1]:.2f},{src_bb[3]:.2f}]\n"
        f"tgt bbox: x[{tgt_bb[0]:.2f},{tgt_bb[2]:.2f}]  y[{tgt_bb[1]:.2f},{tgt_bb[3]:.2f}]",
        xy=(0.01, 0.01), xycoords="axes fraction",
        fontsize=7.5, color="#555555",
    )

    # ── coordinate table below ────────────────────────────────────────────────
    col_labels = ["id", "label", "orig x", "orig y", "mapped x", "mapped y"]
    table_data = [
        [rid, label, f"{cx:.3f}", f"{cy:.3f}", f"{mx:.3f}", f"{my:.3f}"]
        for rid, label, stype, cx, cy, mx, my in room_info
    ]
    tbl = fig.add_axes([0.05, -0.18, 0.90, 0.15])
    tbl.axis("off")
    t = tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 — baseline affine mapping")
    parser.add_argument("--plan",   default="987",  help="plan id (filename without .json)")
    parser.add_argument("--target", default="self",
                        help="target boundary: self | rect | lshape (default: self)")
    parser.add_argument("--rot",    type=float, default=0,   help="rotation degrees")
    parser.add_argument("--mirror", action="store_true",      help="mirror horizontally")
    parser.add_argument("--out",    default="test_s1_baseline.png")
    args = parser.parse_args()
    run(args.plan, args.target, args.rot, args.mirror, args.out)


if __name__ == "__main__":
    main()
