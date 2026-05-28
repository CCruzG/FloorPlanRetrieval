# Floor Plan Retrieval Tool

A browser-based tool for retrieving similar floor plans by drawing a boundary polygon, optionally specifying an entrance, and setting room composition requirements. Results are ranked from a pre-built index of 2000 plans and rendered with room colours, door/window markers, and an optional room adjacency graph.

Retrieval is **rotation- and reflection-aware**: each candidate plan is evaluated in all 8 orientations (4 rotations × 2 mirror states) and displayed at the orientation that best matches the query entrance direction.

---

## Requirements

- Python 3.10+
- A Unix-like environment (Linux / macOS) or WSL on Windows

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/CCruzG/FloorPlanRetrieval.git
cd FloorPlanRetrieval

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install flask numpy matplotlib shapely
```

## Repository Architecture

The repository uses a monorepo-style layout with clear ownership boundaries:

- `apps/api/` for the HTTP server surface
- `apps/web/` for the browser application
- `packages/floorplan_core/` for reusable Python domain logic
- `tools/` for offline/dev scripts
- `data/` for fixtures, indexes, and generated outputs
- `configs/` for workflow/config assets
- `docs/` for architecture and developer documentation

The canonical migration plan is documented in [docs/architecture/repository-architecture.md](/home/camilo/Projects/02_Retrieval_v02/docs/architecture/repository-architecture.md).

The canonical API entrypoint is `apps/api/server.py`.

---

## Running the server

```bash
source .venv/bin/activate
python apps/api/server.py
```

Then open **http://127.0.0.1:5000** in your browser.

For frontend-only development with Vite's dev server:

```bash
npm --prefix apps/web install
npm --prefix apps/web run dev
```

Auxiliary utility pages are kept under `apps/web/legacy` while `apps/web` remains the canonical frontend workspace.

The Flask server still provides the retrieval, fit, render, and static asset endpoints used by the app.

If you want to use the OpenAI render backend, set the key in the server environment before starting Flask:

```bash
export OPENAI_API_KEY="sk-..."
python apps/api/server.py
```

To expose the server on the local network (e.g. for a colleague on the same Wi-Fi):

```bash
python apps/api/server.py --host 0.0.0.0
```

Then open **http://&lt;your-machine-ip&gt;:5000** from another device.

### Server options

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address (`0.0.0.0` for network access) |
| `--port` | `5000` | Port number |
| `--data_dir` | `data/fixtures/plans/test_data` | Directory containing floor plan JSON files |
| `--index_bin` | `data/index/browser/index.bin` | Pre-built descriptor index (float32 binary) |
| `--index_names` | `data/index/browser/index_names.json` | Index name list |
| `--allow-remote-comfy` | off | Allow `comfy_url` values outside `localhost` / `127.0.0.1` |
| `--debug` | off | Enable Flask debug mode |

### Render backend notes

- OpenAI rendering now reads `OPENAI_API_KEY` from the server environment only. The browser does not send the key to `/render`.
- ComfyUI URLs are restricted to `localhost`, `127.0.0.1`, or `::1` by default. Use `--allow-remote-comfy` if you explicitly want to allow a remote ComfyUI host.
- The browser loads non-secret runtime defaults from `GET /config` (render backend, default ComfyUI URL, default workflow filename, OpenAI model/quality, and descriptor metadata), so these values are centralized server-side.

---

## Using the browser tool

### Drawing a boundary

1. **Click** on the canvas to place boundary vertices.
2. Hold **Shift** while clicking to lock the edge to horizontal or vertical.
3. Hover near a placed vertex to see **projection lines** — the cursor snaps to them automatically.
4. **Double-click**, press **Enter**, or click the first vertex (shown in red) to close the boundary.
5. Press **Esc** to clear and start over.

### Placing the entrance

After closing the boundary, click **Set Entrance** to enter entrance-placement mode. The cursor snaps to the nearest boundary edge; a yellow arrow indicates the outward direction (approach side). Click to confirm the entrance position.

The entrance is shown as a grey notch on the boundary. Click **Clear Entrance** to remove it.

### Scale and orientation

- A **5 m scale bar** and **3 m grid** are shown at the bottom of the canvas. The canvas assumes 1 normalised unit = 18 m.
- Dimension labels show the boundary extents in metres.

### Requirements & parameters

- **Bedrooms / bathrooms** — set the desired room counts using the sliders.
- **Results** — number of alternatives to retrieve (1–20).
- **Shape weight** — how much the boundary shape influences retrieval (0 = entrance/semantic only).
- **Entrance weight** — how much the entrance position and approach direction influence retrieval.
- **Semantic weight** — how much room composition influences retrieval (0 = shape/entrance only).

### Retrieving

Click **retrieve** once the boundary is closed. Results appear in the right panel as thumbnails. Each thumbnail is rendered at the best-matching orientation (rotation + optional mirror). Click a thumbnail to display the full plan on the canvas, aligned to the query boundary by centroid and entrance direction.

### Room graph

Check **show room graph** to overlay a graph of room adjacencies on the displayed plan. Solid edges indicate door connections; dashed edges indicate wall adjacency.

---

## Plan caching

Retrieved plans are cached in the browser using the **Cache API** (`fp-plans-v1`). Plans load once per session and are served from cache on subsequent queries, making repeated retrievals fast. Click **Clear plan cache** in the controls panel to force a fresh fetch from the server.

---

## Boundary library

Boundaries can be saved, reused, and shared for batch studies.

- **Save boundary…** — saves the current closed boundary (vertices + entrance if set) with a name into browser `localStorage`.
- Click a name in the library list to **load** it back onto the canvas.
- **✎** to rename, **✕** to delete.
- **Export library JSON** — downloads all saved boundaries as `boundary_library.json`.
- **Import library JSON** — merges boundaries from a JSON file into the local library.

The exported format is:
```json
[
  {
    "name": "rectangle-large",
    "vertices": [{"x": 0.2, "y": 0.2}, ...],
    "saved": "2026-04-23T10:00:00.000Z"
  }
]
```

---

## Rebuilding the index

The pre-built index uses a **63-dimensional descriptor**:

| Dimensions | Description |
|---|---|
| 0–47 | Shape: 48-bin angle histogram of boundary edges |
| 48–51 | Entrance: `[rel_x, rel_y, cos_θ, sin_θ]` in normalised boundary frame |
| 52–62 | Rooms: 11-bin histogram of room type counts |

To rebuild the index from a different dataset:

```bash
source .venv/bin/activate
python tools/build_index_ent.py --data_dir data/fixtures/plans/test_data --out_dir data/index/browser
```

This writes `data/index/browser/index.bin` (float32 binary, plans × 63) and `data/index/browser/index_names.json`.

### Retrieval weights

| Flag | Default | Description |
|---|---|---|
| `--shape_w` | `1.0` | Weight for boundary shape descriptor |
| `--ent_w` | `1.0` | Weight for entrance descriptor |
| `--room_w` | `0.5` | Weight for room composition descriptor |
| `--topn` | `5` | Number of results to return |

---

## Floor plan JSON schema

Plans use schema version **2.1.0** with normalised `[0, 1]` coordinates (1 unit = 18 m).

> **Query boundary vs. stored plans** — The browser sends the user-drawn boundary to the server in the same `[x, y, direction, is_door_vertex]` vertex format, but `direction` and `is_door_vertex` are always `0` (the server strips them immediately and uses only the `[x, y]` pairs). The query entrance is sent as a single point `[x, y]` rather than a bounding box. Neither the query boundary nor the entrance are persisted as a JSON file; only retrieved/adapted plans use the full schema below.

### Top-level fields

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | Always `"2.1.0"` |
| `id` | string | Plan identifier |
| `name` | string | Human-readable name |
| `source` | string | Dataset origin (e.g. `"rplan"`) |
| `units` | string | Always `"normalised"` |
| `normalisation_multiplier` | number | Scale factor applied (typically `1.0`) |
| `boundary` | array | Outer footprint vertices (see below) |
| `entrance` | array | Front-door bounding box `[x1, y1, x2, y2]` |
| `rooms` | array | Room objects (see below) |
| `edges` | array | Room adjacency graph edges |
| `doors` | array | Interior door objects |
| `windows` | array | Window objects |
| `walls` | null | Reserved, not currently used |
| `requirements` | null | Reserved, not currently used |
| `retrieval` | null / object | Populated by the server after retrieval |

### `boundary`

Each vertex is `[x, y, direction, is_door_vertex]`:

| Field | Description |
|---|---|
| `x`, `y` | Normalised `[0, 1]` coordinates |
| `direction` | Outward-facing cardinal direction of the following edge: `0`=left, `1`=right, `2`=down, `3`=up |
| `is_door_vertex` | `1` if this vertex is part of the main entrance opening, `0` otherwise |

### `rooms`

Each room object has:

| Field | Description |
|---|---|
| `id` | Unique string identifier (e.g. `"r_0"`) |
| `type` | Integer room-type label (raw dataset value) |
| `source_type` | Normalised room type used by the retrieval pipeline (see table below) |
| `label` | Human-readable label string |
| `polygon` | List of `[x, y]` vertices defining the room footprint (normalised coords) |
| `bbox` | Axis-aligned bounding box `[x_min, y_min, x_max, y_max]` |
| `bbox_interior` | Slightly inset bounding box (excludes shared walls) |

**`source_type` values:**

| Value | Room type |
|---|---|
| 0 | Living room |
| 1 | Master bedroom |
| 2 | Kitchen |
| 3 | Bathroom |
| 4 | Dining room |
| 5 | Child bedroom |
| 6 | Study |
| 7 | Second bedroom |
| 8 | Guest room |
| 9 | Balcony |
| 10 | Entrance / hall |
| 11 | Storage |

### `edges`

Each edge object has:

| Field | Description |
|---|---|
| `u`, `v` | Room IDs of the two connected rooms |
| `relation` | Spatial relation code (dataset-specific integer) |
| `connection` | `0` = shared wall, `1` = door connection |

### `doors` and `windows`

| Field | Description |
|---|---|
| `id` | Unique string identifier |
| `type` | `"door"` or `"window"` |
| `bbox` | Bounding box `[x1, y1, x2, y2]` in normalised coords |
| `orientation` | `"horizontal"` or `"vertical"` |
| `room_refs` | List of room IDs the opening belongs to |

---

## Project structure

```
apps/web/                Canonical frontend app (Vite HTML and src modules)
apps/web/legacy/         Auxiliary utility pages (retrieval tool and index explorer)
apps/api/                Canonical Flask API entrypoint(s)
packages/floorplan_core/ Canonical Python domain logic (retrieval, fitting, rendering)
tools/                   Canonical offline CLI tools (build/export/explore/test harnesses)
data/index/browser/      Pre-built browser index assets
data/fixtures/plans/test_data/  Sample floor plan JSON files (schema v2.1.0)
data/outputs/adapted/    Example adapted plans
data/outputs/boundaries/ Exported boundary libraries
data/outputs/images/     Generated example images
docs/architecture/       Architecture and migration docs
docs/dev/                Dev design notes, prompts, and reports
```

