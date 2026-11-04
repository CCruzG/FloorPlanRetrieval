"""Render synthetic rect_decompose cases and a grid of RPlan plans to PNG.

Output: data/outputs/diagnostics/rect_decompose_synthetic/
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MPoly  # noqa: E402

from packages.floorplan_core.rect_decompose import decompose, decompose_plan  # noqa: E402
from packages.floorplan_core.tests.test_rect_decompose import CASES  # noqa: E402

OUT = REPO_ROOT / "data/outputs/diagnostics/rect_decompose_synthetic"
RPLAN = Path("/Users/camilo/Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json")
COLORS = ["#cfe8ff", "#ffe3b8", "#d4f1c5", "#f3d1f0", "#fff3a8", "#cdeff0", "#f6cccc", "#dcdcf5"]


def draw(ax, res, outline=None, lw=1.2):
    for i, p in enumerate(res["parts"]):
        ax.add_patch(MPoly(p["polygon"], closed=True, facecolor=COLORS[i % len(COLORS)],
                           edgecolor="none", hatch="///" if p["sliver"] else None))
    if outline is not None:
        ax.add_patch(MPoly(outline, closed=True, fill=False, edgecolor="black", lw=lw))
    for c in res["cuts"]:
        ax.plot([c[0][0], c[1][0]], [c[0][1], c[1][1]], "r--", lw=lw)
    for c in res["reversed_cuts"]:
        ax.plot([c[0][0], c[1][0]], [c[0][1], c[1][1]], "b:", lw=lw + 0.6)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = dict(CASES)
    cases["sloped"] = ([(0, 0), (4, 0), (4, 2), (2, 3), (0, 2)], 1, "not_rectilinear", {})
    cases["self_intersecting"] = ([(0, 0), (4, 4), (4, 0), (0, 4)], 1, "invalid", {})
    files = []
    for name, (coords, _, _, kw) in cases.items():
        res = decompose(coords, **kw)
        fig, ax = plt.subplots(figsize=(4, 4))
        draw(ax, res, outline=coords)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_title(f"{name}: {res['status']}, {len(res['parts'])} part(s)", fontsize=9)
        fp = OUT / f"{name}.png"
        fig.savefig(fp, dpi=110, bbox_inches="tight")
        plt.close(fig)
        files.append(fp.name)
    ids = [0, 1, 10, 100, 1000, 59631]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for ax, pid in zip(axs.ravel(), ids):
        plan = json.load(open(RPLAN / f"{pid}.json"))
        out = decompose_plan(plan)
        for room, src in zip(out["rooms"], plan["rooms"]):
            draw(ax, room, outline=src["polygon"], lw=0.9)
        s = out["summary"]
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.autoscale_view()
        ax.set_title(f"RPlan {pid}: {s['n_rooms']} rooms, {s['n_parts']} parts, {s['n_slivers']} slivers, "
                     f"{s['n_reversed']} reversed, {s['status_counts']}", fontsize=8)
    fp = OUT / "rplan_grid.png"
    fig.savefig(fp, dpi=110, bbox_inches="tight")
    plt.close(fig)
    files.append(fp.name)
    print("\n".join(files))


if __name__ == "__main__":
    main()
