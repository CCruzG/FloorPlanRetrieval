"""
explore_index.py — Visualise the descriptor space of the floor plan index.

Produces an interactive HTML file with three scatter plots (PCA, t-SNE, UMAP),
each point representing one floor plan. Points are coloured by a chosen
metadata attribute (bedrooms, rooms, boundary area, aspect ratio).

Usage:
    python explore_index.py
    python explore_index.py --index_bin browser_files/index.bin \
                             --index_names browser_files/index_names.json \
                             --data_dir test_data \
                             --out index_explorer.html
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

# ── args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--index_bin",   default="browser_files/index.bin")
parser.add_argument("--index_names", default="browser_files/index_names.json")
parser.add_argument("--data_dir",    default="test_data")
parser.add_argument("--out",         default="index_explorer.html")
parser.add_argument("--tsne_perp",   type=int, default=30,
                    help="t-SNE perplexity (default 30)")
parser.add_argument("--umap_nn",     type=int, default=15,
                    help="UMAP n_neighbors (default 15)")
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--library",     default="boundary_library.json",
                    help="Boundary library JSON exported from the browser tool")
args = parser.parse_args()

# ── load index ────────────────────────────────────────────────────────────────

print("Loading index…")
raw = np.frombuffer(open(args.index_bin, "rb").read(), dtype=np.float32)
names = json.load(open(args.index_names))
dim = raw.size // len(names)
descs = raw.reshape(len(names), dim).astype(np.float64)
print(f"  {len(names)} plans  ·  {dim}-dim descriptors")

# ── load metadata ─────────────────────────────────────────────────────────────

BEDROOM_TYPES  = {1, 5, 7}   # master, child, second bedroom (source_type)
BATHROOM_TYPES = {3}

print("Loading plan metadata…")
meta = []   # list of dicts, one per plan
for name in names:
    stem = Path(name).stem
    path = os.path.join(args.data_dir, stem + ".json")
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        meta.append(None)
        continue

    rooms = d.get("rooms", [])
    n_rooms = len(rooms)
    n_bed = sum(1 for r in rooms if r.get("source_type", r.get("type", -1)) in BEDROOM_TYPES)
    n_bath = sum(1 for r in rooms if r.get("source_type", r.get("type", -1)) in BATHROOM_TYPES)

    bdry = d.get("boundary", [])
    xs = [v[0] for v in bdry]
    ys = [v[1] for v in bdry]
    bw = max(xs) - min(xs) if xs else 0
    bh = max(ys) - min(ys) if ys else 0
    area = bw * bh
    aspect = (bw / bh) if bh > 1e-6 else 1.0
    n_verts = len(bdry)

    meta.append({
        "id":      d.get("id", stem),
        "n_rooms": n_rooms,
        "n_bed":   n_bed,
        "n_bath":  n_bath,
        "area":    round(area, 4),
        "aspect":  round(aspect, 3),
        "n_verts": n_verts,
    })

# fill missing
fallback = {"id": "?", "n_rooms": 0, "n_bed": 0, "n_bath": 0,
            "area": 0, "aspect": 1, "n_verts": 0}
meta = [m if m is not None else dict(fallback) for m in meta]
print(f"  metadata loaded for {sum(1 for m in meta if m['id'] != '?')} / {len(meta)} plans")

# ── dimensionality reduction ──────────────────────────────────────────────────

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

scaler = StandardScaler()
X = scaler.fit_transform(descs)

print("PCA…")
pca = PCA(n_components=2, random_state=args.seed)
xy_pca = pca.fit_transform(X)
explained = pca.explained_variance_ratio_
print(f"  explained variance: PC1={explained[0]:.1%}  PC2={explained[1]:.1%}")

print("t-SNE…")
tsne = TSNE(n_components=2, perplexity=args.tsne_perp,
            random_state=args.seed, max_iter=1000, verbose=0)
xy_tsne = tsne.fit_transform(X)

print("UMAP…")
try:
    import umap
    reducer = umap.UMAP(n_neighbors=args.umap_nn, min_dist=0.1,
                        random_state=args.seed, verbose=False)
    xy_umap = reducer.fit_transform(X)
except ImportError:
    print("  umap-learn not available — skipping UMAP")
    xy_umap = None

# ── load + project library boundaries ───────────────────────────────────────

from sem_geo_retrieval import combined_desc, requirements_sem_hint

def _desc_from_vertices(vertices, requirements=None, shape_w=1.0, sem_w=0.5):
    """Compute a combined descriptor from raw {x,y} boundary vertices."""
    legacy = {
        "name": "query",
        "boundary": [[v["x"] * 256, v["y"] * 256] for v in vertices],
        "coord":  [],
        "labels": [],
    }
    return combined_desc(legacy, shape_w, sem_w, requirements=requirements)

lib_entries = []
lib_xy_pca  = []
lib_xy_tsne = []
lib_xy_umap = []

if os.path.exists(args.library):
    print(f"Loading boundary library: {args.library}")
    lib = json.load(open(args.library))
    for entry in lib:
        try:
            req = entry.get("requirements")
            d = _desc_from_vertices(entry["vertices"], requirements=req).astype(np.float64)
        except Exception as e:
            print(f"  skipping '{entry['name']}': {e}")
            continue
        d_scaled = scaler.transform(d.reshape(1, -1))
        lib_entries.append(entry["name"])
        lib_xy_pca.append(pca.transform(d_scaled)[0])
        # t-SNE: approximate via weighted average of k nearest neighbours
        dists = np.linalg.norm(X - d_scaled, axis=1)
        knn_idx = np.argsort(dists)[:5]
        weights = 1.0 / (dists[knn_idx] + 1e-9)
        weights /= weights.sum()
        lib_xy_tsne.append((xy_tsne[knn_idx] * weights[:, None]).sum(axis=0))
        if xy_umap is not None:
            lib_xy_umap.append(reducer.transform(d_scaled)[0])
    if lib_entries:
        lib_xy_pca  = np.array(lib_xy_pca)
        lib_xy_tsne = np.array(lib_xy_tsne)
        lib_xy_umap = np.array(lib_xy_umap) if xy_umap is not None else None
        print(f"  projected {len(lib_entries)} library boundaries")
else:
    print(f"No library file found at {args.library} — skipping overlay")


def make_lib_scatter(xy, names):
    """Star markers for library boundary entries."""
    return go.Scatter(
        x=xy[:, 0], y=xy[:, 1],
        mode="markers+text",
        marker=dict(
            symbol="star",
            size=14,
            color="#f97316",
            line=dict(color="#7c2d12", width=1),
        ),
        text=names,
        textposition="top center",
        textfont=dict(size=9, color="#1e293b"),
        hovertext=[f"<b>Query: {n}</b>" for n in names],
        hoverinfo="text",
        name="saved boundaries",
        showlegend=True,
    )


# ── build plotly figure ───────────────────────────────────────────────────────

import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_FIELDS = [
    ("n_bed",   "bedrooms",     "RdYlGn"),
    ("n_bath",  "bathrooms",    "Blues"),
    ("n_rooms", "total rooms",  "Viridis"),
    ("area",    "boundary area","Plasma"),
    ("aspect",  "aspect ratio", "Cividis"),
]

def make_scatter(xy, meta, color_field, colorscale, title, subplot_title):
    vals  = [m[color_field] for m in meta]
    texts = [
        f"<b>{m['id']}</b><br>"
        f"bedrooms: {m['n_bed']}  bathrooms: {m['n_bath']}<br>"
        f"total rooms: {m['n_rooms']}<br>"
        f"area: {m['area']}  aspect: {m['aspect']}<br>"
        f"boundary verts: {m['n_verts']}"
        for m in meta
    ]
    return go.Scatter(
        x=xy[:, 0], y=xy[:, 1],
        mode="markers",
        marker=dict(
            size=4,
            color=vals,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=title, thickness=12, len=0.8),
            opacity=0.75,
        ),
        text=texts,
        hoverinfo="text",
        name=subplot_title,
    )

# one tab per colour attribute, each tab has PCA / t-SNE / UMAP side by side
n_cols = 3 if xy_umap is not None else 2
subplot_titles = [
    f"PCA  (PC1 {explained[0]:.1%} + PC2 {explained[1]:.1%} variance)",
    f"t-SNE  (perplexity={args.tsne_perp})",
] + ([f"UMAP  (n_neighbors={args.umap_nn})"] if xy_umap is not None else [])

# build a figure per colour field, then expose via buttons
print("Building figure…")

figs = []
for cf, label, cs in COLOR_FIELDS:
    traces = [make_scatter(xy_pca, meta, cf, cs,
                           label, f"PCA · colour={label}")]
    traces.append(make_scatter(xy_tsne, meta, cf, cs,
                               label, f"t-SNE · colour={label}"))
    if xy_umap is not None:
        traces.append(make_scatter(xy_umap, meta, cf, cs,
                                   label, f"UMAP · colour={label}"))
    # library overlays — one per projection, in same order
    if lib_entries:
        traces.append(make_lib_scatter(lib_xy_pca,  lib_entries))   # col 1
        traces.append(make_lib_scatter(lib_xy_tsne, lib_entries))   # col 2
        if xy_umap is not None and lib_xy_umap is not None:
            traces.append(make_lib_scatter(lib_xy_umap, lib_entries))  # col 3
    figs.append((label, traces))

# trace layout: [pca, tsne, umap, lib_pca, lib_tsne, lib_umap] per tab
# col mapping: trace index within tab → subplot column
# proj traces  0..n_proj-1  → col 1..n_proj
# lib  traces  n_proj..2*n_proj-1 → same cols
all_traces = []
n_proj = n_cols
n_lib  = n_cols if lib_entries else 0
traces_per_tab = n_proj + n_lib
_col_map = list(range(1, n_proj+1)) + list(range(1, n_proj+1))  # proj then lib, same cols

for i, (label, traces) in enumerate(figs):
    for t in traces:
        t.visible = (i == 0)
        all_traces.append(t)

buttons = []
for i, (label, _) in enumerate(figs):
    vis = [False] * len(all_traces)
    for j in range(traces_per_tab):
        vis[i * traces_per_tab + j] = True
    buttons.append(dict(label=label, method="update",
                        args=[{"visible": vis},
                              {"title": f"Floor Plan Descriptor Space — colour: {label}"}]))

fig = make_subplots(rows=1, cols=n_cols,
                    subplot_titles=subplot_titles,
                    horizontal_spacing=0.06)

for i, t in enumerate(all_traces):
    col = _col_map[i % traces_per_tab]
    fig.add_trace(t, row=1, col=col)

fig.update_layout(
    title=f"Floor Plan Descriptor Space — colour: {figs[0][0]}",
    height=650,
    template="plotly_white",
    showlegend=True,
    updatemenus=[dict(
        buttons=buttons,
        direction="down",
        showactive=True,
        x=0.01, xanchor="left",
        y=1.12, yanchor="top",
        bgcolor="#f1f5f9",
        bordercolor="#cbd5e1",
        font=dict(size=11),
    )],
    annotations=[
        dict(text="<b>colour by:</b>", showarrow=False,
             x=0.01, y=1.155, xref="paper", yref="paper",
             font=dict(size=11)),
    ],
    margin=dict(t=110, b=40, l=40, r=40),
)

# make subplot titles larger and bold (they are the first n_cols annotations)
for i in range(n_cols):
    fig.layout.annotations[i].update(font=dict(size=13, color="#1e293b"),
                                      y=fig.layout.annotations[i].y + 0.01)

# tidy axes
for col in range(1, n_cols + 1):
    fig.update_xaxes(showticklabels=False, showgrid=True,
                     gridcolor="#e2e8f0", zeroline=False, row=1, col=col)
    fig.update_yaxes(showticklabels=False, showgrid=True,
                     gridcolor="#e2e8f0", zeroline=False, row=1, col=col)

out_path = args.out
fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
print(f"\nSaved → {out_path}")
print("Open in your browser to explore.")
