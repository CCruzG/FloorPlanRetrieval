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
| `relation` | Spatial relation of v relative to u (integer, see table below) |
| `connection` | `0` = shared wall, `1` = door connection |

`relation` values (direction of v from u):

| Value | Name |
|---|---|
| 0 | `left-above` |
| 1 | `left-below` |
| 2 | `left-of` |
| 3 | `above` |
| 4 | `inside` |
| 5 | `surrounding` |
| 6 | `below` |
| 7 | `right-of` |
| 8 | `right-above` |
| 9 | `right-below` |

### `doors` and `windows`

| Field | Description |
|---|---|
| `id` | Unique string identifier |
| `type` | `"door"` or `"window"` |
| `bbox` | Bounding box `[x1, y1, x2, y2]` in normalised coords |
| `orientation` | `"horizontal"` or `"vertical"` |
| `room_refs` | List of room IDs the opening belongs to |

---

## Dev log

### 2026-06-18 — Fit Step 1 affine baseline and staged testing protocol

**Goal:** establish a geometrically correct affine initialisation for room nodes
before the agent call, and define computable success criteria for each stage of
the repositioning pipeline.

#### Context

The agent was producing wrong results — rooms on opposite sides of the target
boundary, entrance on the wrong edge — because the affine transform used centroid
spans with a 0.65 compression factor and ignored the `r.rot`/`r.mirror` values
returned by retrieval. The agent also had no observation capability and no
computable success criteria, so it could not self-correct.

#### What was built

**Corrected affine transform in `computeFitGraph` (`apps/web/src/main.js`)**

Replaced the centroid-span/0.65 approach with a plan-boundary-bbox → target-
boundary-bbox mapping:
- Apply `applyTransform` (rotation + mirror from retrieval) to the plan boundary
  vertices first, compute the bbox of those transformed vertices.
- Map each room centroid through the same transform, then affine from the
  transformed bbox to the target boundary bbox (1:1, no compression).
- This preserves relative room layout and correctly accounts for entrance
  orientation changes.

**`suggested_positions` passed to server**

`computeFitGraph` now pre-computes rotation/mirror-corrected centroid positions
and includes them in the `/fit_step1` payload as `suggested_positions: {id: {x, y}}`.
`fit_step1.py` uses these directly; the Python affine is only a fallback when
they are absent.

**Stage 1 diagnostic script (`tools/test_s1_baseline.py`)**

Visual test harness for the affine baseline. Produces a two-panel PNG:
- Left: original plan with boundary polygon, room polygons, centroids, entrance.
- Right: affine-mapped centroids on the target boundary, with bbox annotations.

Three test cases confirmed:
- `test_s1_self.png` — self-fit is identity (correct).
- `test_s1_rect.png` — rect target, no rotation: relative layout preserved.
- `test_s1_rect_rot90.png` — 90° clockwise: entrance moves as expected.

**Relation codes documented (`README.md`)**

Added the `relation` integer-to-name mapping table (0–9) to the `### edges`
schema section.

#### Current state

Stage 1 (affine baseline) is complete and visually verified. The staged testing
protocol is defined but not yet executed past Stage 1:

| Stage | Description | Status |
|---|---|---|
| 1 | Affine baseline visual check | **done** |
| 2 | Containment — `point_in_polygon` + `snap_to_interior` | not started |
| 3 | Radial depth rank (C2, Kendall tau ≥ 0.8) | not started |
| 4 | Angular order (C3) | not started |
| 5 | Correction loop (C2/C3 fixes) | not started |
| 6 | Graph proximity (C4) | not started |
| 7 | Type affinity (C5) | not started |

The LLM agent is suspended until the algorithmic stages are in place to give it
correct starting positions and computable success criteria to check against.

---

### 2026-06-18 — Fit Step 1 agent test harness

**Goal:** wire the fit-step1 brief to a real Claude call so the spatial reasoning
loop can be exercised end-to-end from the command line before the browser endpoint
is built.

#### What was built

**`tools/test_fit_step1.py`**

Standalone Python script that:
1. Loads a plan JSON from `data/fixtures/plans/test_data/` (default: plan 987).
2. Builds the layout graph — boundary polygon, room nodes (centroid of each room
   polygon), and an entrance node whose centroid is snapped to the nearest point
   on the boundary perimeter via `_nearest_pt_on_boundary`.  A synthetic
   `connection: 1` edge is inserted from the entrance to the nearest common-zone
   room (sourceType 0 or 4), matching the JS behaviour in `computeFitGraph`.
3. Reads `docs/dev/design/fit-step1-brief.md` and passes it as the system prompt.
4. Calls Claude Opus 4.8 with adaptive thinking and `set_node_position` as the
   single tool.  Uses streaming (`stream.get_final_message()`) and an agentic
   loop that feeds tool results back until `stop_reason == "end_turn"`.
5. Prints each `set_node_position` call as it arrives, then a summary table of
   final positions with room labels.

```bash
source .venv/bin/activate
python tools/test_fit_step1.py
python tools/test_fit_step1.py --plan data/fixtures/plans/test_data/290.json
```

---

### 2026-06-17 — Fit tab and spatial reasoning brief

**Goal:** replace the heuristic fitting pipeline with a step-by-step harness
driven by spatial reasoning. The heuristic pipeline (edge-push → clip →
fill_residual) was producing topologically incorrect results — rooms clustering
at concave boundary vertices, rooms outside the boundary — because it encoded no
understanding of what a residential layout should look like.

#### What was built

**Fit tab (`apps/web/index.html`, `apps/web/src/main.js`, `apps/web/src/canvas-renderer.js`)**

New tab in the main view bar (next to Floor plate, Unit, 3D View). When active:
- The retrieved plan is rendered faded (45% alpha) behind the target boundary.
- The target boundary is rendered prominently (solid blue fill + thick stroke).
- A dedicated right-panel palette replaces the standard controls, with a
  three-step workflow (Map layout graph → Grow rooms → Verify), view toggles,
  a status log, and a "▶ run step N" button.
- Auto-fit on retrieve has been removed. The Fit tab is the only entry point
  for fitting work.

**Layout graph mapping (`computeFitGraph` in `main.js`)**

"▶ run step 1" computes a graph of room nodes mapped into the target boundary:
- Room centroids are extracted from `fp.rooms`, with rotation/mirror applied to
  match the displayed orientation.
- An affine transform (plan centroid → boundary centroid, scaled to fit) maps
  all room nodes into the boundary.
- The **entrance node** is a special case: its position is snapped to the
  nearest point on the target boundary perimeter using `_nearestPtOnBoundary`,
  and a synthetic edge connects it to the nearest common-area room (living or
  dining). The entrance is rendered as a **square** to signal it is a fixed
  boundary anchor; all other room nodes are rendered as circles.
- Adjacency edges are drawn with solid lines (door connections, `connection: 1`)
  and dashed lines (shared wall, no door, `connection: 0`).

**Spatial reasoning brief (`docs/dev/design/fit-step1-brief.md`)**

Behaviour specification for the agent that will reposition nodes in Step 1.
Key principles:
- All edges in `fp.edges` represent shared walls. `connection: 1` means the
  wall has a door (circulation); `connection: 0` means wall adjacency only.
- Entrance is a fixed anchor on the boundary perimeter.
- Bedrooms must be placed near the boundary (exterior wall for windows).
- At least one bathroom must be reachable from the common zone.
- Balconies must be on the perimeter.
- The agent traces the solid-edge circulation skeleton outward from the
  entrance to verify every room is reachable without passing through a
  private room.

#### Current state

The graph visualisation is working. Node positions from the current affine
transform are compressed toward the boundary centre (a known limitation of
treating centroid span as plan span). The next step is to replace the affine
initialisation with the agent call: a backend endpoint that sends the graph
and boundary to Claude with the brief, receives `set_node_position` calls,
and returns updated node positions to the browser.

Two features noted for future implementation:
- **Boundary segment flags** (exterior vs. corridor-facing) — lets the agent
  distinguish walls suitable for bedrooms from interior-facing walls.
- **`hintAngle` per node** — the angle from the original plan's entrance to
  each room centroid, giving the agent a directional starting hypothesis
  without requiring the new boundary shape to match the original.

---

### 2026-06-16 — Fitting harness wired to the browser

**Goal:** end-to-end test path — draw a boundary → retrieve a plan → fit it using the agent harness tools → inspect invariant results — all from the browser without touching Python directly.

#### What was built

**`/fit_harness` endpoint (`apps/api/server.py`)**

New POST endpoint that runs a three-step automated fit:

1. **Edge-push** (`fit_rooms_in_boundary`) — translates each room until it is fully inside the query boundary. Bedrooms that straddle the boundary edge end up touching it, satisfying G3.
2. **Clip** (`clip_room_to_boundary`) — trims any polygon protrusion remaining after translation.
3. **Fill residual** (`fill_residual`) — sets the common-area room (living/dining) to `boundary − union(all other rooms)`, guaranteeing 100% coverage (G4).

The endpoint then runs `run_invariants` and returns the full G1–G6 report alongside the fitted plan JSON.

**G6 proportion invariant (`packages/floorplan_core/fit_invariants.py`)**

New `check_room_proportions(plan, original_plan, tolerance=0.10)` function. For each non-common room it compares:
- **Area**: Shapely polygon area — must stay within ±10% of the pre-fit area.
- **Aspect ratio**: bounding-box width/height — must stay within ±10% of the pre-fit ratio.

`run_invariants` now accepts an optional `original_plan` argument; when provided it runs G6 and appends failures like `"G6: r_4 exceeds ±10% proportion limit (area ×0.875, aspect ×0.875)"`.

**Frontend (`apps/web/src/main.js`)**

- `runRetrieveFitHarness()` — one-click function: retrieves top plan for the drawn boundary, transforms the query boundary into the plan's coordinate frame (`boundaryInPlanSpace`), calls `/fit_harness`, stores the result in `fittedCache` with `fitMode=true` so the fitted plan renders on the canvas.
- "**Retrieve + Fit Harness**" section in the dev panel (`?dev=true`) tools tab — button + log area that shows the full invariant report (G1–G5 pass/fail, coverage %, G6 proportion summary, per-room violation details).

#### Status

| Invariant | Status |
|---|---|
| G1 containment | passing |
| G2 overlaps | passing |
| G3 bedrooms touch boundary | passing after adding edge-push step |
| G4 100% coverage | passing after adding edge-push step (fill_residual now works on correctly-positioned rooms) |
| G5 common area ≥ 0 | passing |
| G6 ±10% proportions | **still failing** — rooms that get clipped lose area/aspect ratio; next step is to investigate whether the boundary mismatch is too large for the retrieved plan, or whether a scale/transform step before fitting would reduce clipping |

#### Next steps

- Investigate G6 failures: are they caused by poor retrieval matches (plan shape too different from query boundary) or by the fitting strategy clipping too aggressively?
- Consider a pre-fit scale step that normalises the plan's room cluster to the query boundary's bounding box before edge-push, reducing the clipping needed.
- Alternatively, surface a "fit quality score" based on how much each room was clipped so the user can choose a better-matched plan from the retrieval results.

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

