# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the project

**Flask API server (primary entry point):**
```bash
source .venv/bin/activate
python apps/api/server.py
# Opens at http://127.0.0.1:5000
```

**Frontend dev server (Vite, for UI-only iteration):**
```bash
npm --prefix apps/web run dev
# The Flask server must still run separately for API calls
```

**Root-level npm scripts** delegate to `apps/web`:
```bash
npm run dev      # equivalent to npm --prefix apps/web run dev
npm run build    # equivalent to npm --prefix apps/web run build
```

**Rebuild the retrieval index from the dataset:**
```bash
source .venv/bin/activate
python tools/build_index_ent.py --data_dir data/fixtures/plans/test_data --out_dir data/index/browser
```

**Run tests:**
```bash
source .venv/bin/activate
pytest
# Single test file:
python tools/test_fit_plan.py
```

**Server flags worth knowing:**
- `--debug` — Flask debug/reload mode
- `--allow-remote-comfy` — allow non-localhost ComfyUI URLs
- `--host 0.0.0.0` — expose on LAN

## Architecture

This is a monorepo with a clear four-layer ownership model:

```
apps/api/server.py            HTTP routes only — no business logic inline
packages/floorplan_core/      Retrieval, fitting, rendering domain logic
packages/floor_plate_design/  Floor plate generation and subdivision
apps/web/src/                 Browser JS modules (Vite, vanilla ES modules)
tools/                        Offline CLI scripts (index building, exploration)
data/                         Non-code artifacts: fixtures, built indexes, outputs
configs/                      ComfyUI workflow JSON, room type config
```

### Request pipeline

```
Browser → POST /retrieve   → packages/floorplan_core/retrieval.py
        → POST /fit        → packages/floorplan_core/fitting.py
        → POST /render     → packages/floorplan_core/render_backend.py
        → POST /generate_floor_plate    → packages/floor_plate_design/generator.py
        → POST /subdivide_floor_plate   → packages/floor_plate_design/generator.py
```

Route handlers in `server.py` follow a strict pattern: parse request via `request_parsing.py`, call the domain function, serialize the result. No business logic lives in the route handler itself.

### Python packages

**`packages/floorplan_core/`** — the core retrieval system:
- `geo_ent_num_retrieval.py` — 63-dim descriptor computation (48 shape + 4 entrance + 11 room-count)
- `retrieval.py` — distance search over the preloaded index matrix
- `index_store.py` — index loading at startup (`INDEX_STORE` global in server)
- `fitting.py` / `fit_plan.py` / `fit_rooms.py` — room placement within a target boundary
- `render_backend.py` / `comfy_client.py` — ComfyUI and OpenAI rendering backends
- `request_parsing.py` — shared request validation/parsing (raises `RequestValidationError`)
- `adapt.py` — boundary adaptation helpers

**`packages/floor_plate_design/`** — floor plate generation (newer package):
- `types.py` — all dataclasses and enums; **all dimensions in mm, areas in m²**
- `generator.py` — 10-stage pipeline: boundary analysis → typology → cores → corridors → units → compliance
- `geometry.py` — polygon helpers (Shapely-based)
- `compliance.py` — compliance snapshot computation

### Browser JS modules (`apps/web/src/`)

The frontend is vanilla ES modules bundled with Vite. Key modules:
- `main.js` — top-level state wiring and event handlers
- `state.js` — `createInitialState()`, all mutable state lives here
- `canvas-renderer.js` — plan drawing on `<canvas>`
- `retrieval.js` — API calls, cache logic, result rendering
- `floor-plate-tools.js` — browser-side floor plate analysis and typology checks (works against the JS offset-subdivision engine)
- `offset-subdivision.js` — JS port of the subdivision solver (runs client-side; server-side counterpart is `packages/floor_plate_design/`)
- `viewer-3d.js` — 3D extrusion view
- `render-pipeline.js` — depth map → AI render flow
- `geometry.js` / `descriptors.js` — coordinate math and descriptor computation mirrored from Python

### Coordinate systems

- **Retrieval plans:** normalised `[0, 1]` float coordinates (1 unit = 18 m)
- **Floor plate design:** millimetres (integers or floats). The `/subdivide_floor_plate` endpoint auto-detects mm vs m and normalises; responses are returned in metres.
- **Canvas:** pixel coordinates transformed via `n2c` / `c2n` helpers in `geometry.js`

### Index

At startup, `server.py` loads `data/index/browser/index.bin` (float32, N × 63 matrix) and `data/index/browser/index_names.json` into memory. All floor plan JSONs from `data/fixtures/plans/test_data/` are also loaded into `plan_cache`. The index is built offline by `tools/build_index_ent.py`.

### Render backends

- **ComfyUI** (`backend: "comfy"`) — local ComfyUI instance; workflow JSON served from `configs/workflows/`; URL restricted to localhost by default
- **OpenAI** (`backend: "openai"`) — reads `OPENAI_API_KEY` from server environment; key is never sent from browser to server

### Floor plan JSON schema

Schema version `2.1.0`. Boundary vertices are `[x, y, direction, is_door_vertex]`. Room `source_type` values 0–11 map to living room through storage (see `configs/room_types.json`). The `retrieval` field on a plan is populated server-side after retrieval.
