---
title: FloorPlanRetrieval — Refactoring Plan
project: project-44
created: 2026-05-27
status: completed
branch: feature/refactor (branch off dev_cc)
---

Completion note (2026-05-27): all phases in this plan were implemented and validated in the working refactor branch.

# FloorPlanRetrieval — Refactoring Plan

## Context

`retrieval_tool.html` is 3,945 lines and `server.py` is 1,907 lines. Both are monolithic. The goal is modularity — splitting both files into well-bounded modules with clean interfaces. The refactor must not change any observable behaviour.

---

## Phase 0 — Pre-Refactor Fixes

Do these on the branch **before any structural changes**. They are bug fixes that establish a clean baseline. No file moves, no new modules.

### 0a. Fix `N_SAMP` mismatch (data integrity bug)

**File:** `retrieval_tool.html`, line ~601  
**Problem:** Python `geo_ent_num_retrieval.py` uses `N_SAMP = 256`. The JS uses `const N_SAMP = 128`. The shape descriptor the client computes in `runRetrievalLocal()` does not match the descriptor stored in `index.bin`, which was built with the Python value. Local retrieval and server retrieval return different rankings for the same query.  
**Fix:** Change JS constant to `const N_SAMP = 256`.

### 0b. Add `try/finally` to `captureDepthAndPreview()`

**File:** `retrieval_tool.html`  
**Problem:** The function mutates `camera.near`, `camera.far`, and `scene.overrideMaterial` sequentially. If any exception is thrown mid-capture, the renderer is left in depth-material override mode — the 3D scene renders black until the user rebuilds the scene.  
**Fix:** Wrap the entire capture body in `try/finally`. Restore `camera.near`, `camera.far`, and `scene.overrideMaterial = null` in the `finally` block unconditionally.

### 0c. Scope `document`-level `keydown` listener

**File:** `retrieval_tool.html`  
**Problem:** The `keydown` handler fires on `document`. Pressing Enter while typing in the render prompt textarea calls `closeBoundary()` if 3+ vertices exist.  
**Fix:** Add at the top of the keydown handler:
```js
if (e.target.matches('input, textarea, select')) return;
```

### 0d. Remove dead code

The following code is unreachable or unused and should be deleted before the refactor to keep diffs clean:

| Location | What to remove | Reason |
|---|---|---|
| `server.py` | `_render_with_python_controlnet()` and `_get_python_controlnet_pipeline()` and `py_pipe_cache` | No route dispatches to `backend === "python"`. Hundreds of lines. |
| `server.py` | `/adapt` route and `adapt()` handler | Never called from the client. |
| `retrieval_tool.html` | `entDistMinRotation()` function | Defined but never called. `runRetrievalLocal()` inlines its own version. |
| `retrieval_tool.html` | `gridM` step-selection loop in `drawScaleGrid()` (lines ~2,630–2,634) | `gridM` is unconditionally overwritten to `3` three lines later. The loop is dead. |

---

## Canonical Room Type Definition

**Decision:** establish a single source of truth for room types shared by both Python and JS.

Create `room_types.json` at the repo root:

```json
{
  "version": "1.0",
  "types": [
    { "id": 0,  "name": "LivingRoom",  "abbr": "LR",  "group": "common"   },
    { "id": 1,  "name": "MasterRoom",  "abbr": "MR",  "group": "bedroom"  },
    { "id": 2,  "name": "Kitchen",     "abbr": "K",   "group": "kitchen"  },
    { "id": 3,  "name": "Bathroom",    "abbr": "Ba",  "group": "bathroom" },
    { "id": 4,  "name": "DiningRoom",  "abbr": "DR",  "group": "common"   },
    { "id": 5,  "name": "ChildRoom",   "abbr": "CR",  "group": "bedroom"  },
    { "id": 6,  "name": "StudyRoom",   "abbr": "SR",  "group": "misc"     },
    { "id": 7,  "name": "SecondRoom",  "abbr": "S2",  "group": "bedroom"  },
    { "id": 8,  "name": "GuestRoom",   "abbr": "GR",  "group": "bedroom"  },
    { "id": 9,  "name": "Balcony",     "abbr": "Bl",  "group": "balcony"  },
    { "id": 10, "name": "Entrance",    "abbr": "En",  "group": "misc"     },
    { "id": 11, "name": "Storage",     "abbr": "St",  "group": "misc"     },
    { "id": 12, "name": "WalkIn",      "abbr": "Wi",  "group": "misc"     }
  ],
  "groups": {
    "bedroom":  [1, 5, 7, 8],
    "bathroom": [3],
    "kitchen":  [2],
    "common":   [0, 4],
    "misc":     [6, 10, 11, 12],
    "balcony":  [9]
  },
  "fit_order": ["bedroom", "bathroom", "kitchen", "misc", "common"],
  "common_area_ids": [0, 4],
  "balcony_id": 9
}
```

**Python usage:** Load once at server startup:
```python
import json, pathlib
ROOM_TYPES = json.loads((pathlib.Path(__file__).parent / "room_types.json").read_text())
ROOM_GROUPS = {g: set(ids) for g, ids in ROOM_TYPES["groups"].items()}
BALCONY_TYPE_ID = ROOM_TYPES["balcony_id"]
COMMON_AREA_TYPES = set(ROOM_TYPES["common_area_ids"])
FIT_ORDER = [ROOM_GROUPS[g] for g in ROOM_TYPES["fit_order"]]
```

**JS usage:** Served statically, fetched once in `constants.js`:
```js
export const ROOM_TYPES = await fetch('/room_types.json').then(r => r.json());
export const ROOM_GROUPS = Object.fromEntries(
  Object.entries(ROOM_TYPES.groups).map(([g, ids]) => [g, new Set(ids)])
);
export const FIT_ORDER = ROOM_TYPES.fit_order.map(g => ROOM_GROUPS[g]);
export const COMMON_AREA_TYPES = new Set(ROOM_TYPES.common_area_ids);
export const BALCONY_TYPE_ID = ROOM_TYPES.balcony_id;
```

Replace all hardcoded `new Set([1,5,7])` / `new Set([3])` etc. references in both `retrieval_tool.html` and `fit_rooms.py` with the imported values.

---

## Build System

Introduce **Vite**. No webpack, no custom bundler.

```bash
npm create vite@latest . -- --template vanilla
```

Move `retrieval_tool.html` → `index.html`. The existing HTML structure stays intact. Add `<script type="module" src="/src/main.js"></script>` before `</body>`. Remove the single `<script>` block — all JS moves to `src/`.

Dev server: `npm run dev` (replaces opening the HTML file directly, and solves the `file://` CORS problem with `fetchWithCache`).  
Production build: `npm run build` → `dist/` folder, deployable as-is.

---

## Target Module Structure

### Server — `server.py` → 6 files

```
server.py           — Flask app, all route handlers, argparse, startup (~150 lines)
index_store.py      — load_index(), load_plans(), ensure_manifest(), IndexStore dataclass
retrieval.py        — run_retrieval(), _req_to_room_desc(), result ranking, plan lookup
fitting.py          — adapt_plan(), fit_rooms() — thin wrappers around fit_plan.py / fit_rooms.py
render_backend.py   — _render_with_openai(), _render_with_comfy_local(),
                      _load_comfy_workflow(), _inject_prompt_and_depth()
comfy_client.py     — _http_post_json(), _http_get_json(), _http_get_bytes(),
                      _http_post_multipart() — all urllib usage isolated here
```

**`server.py` route handlers** must contain no business logic — each handler validates input, calls a function from one of the above modules, and returns a JSON response.

**`index_store.py`** exposes a typed `IndexStore` dataclass:
```python
@dataclass
class IndexStore:
    descs:   np.ndarray        # (N, TOTAL_DIM) float32
    shapes:  np.ndarray        # (N, SHAPE_DIM) float32
    ents:    np.ndarray        # (N, ENT_DIM)   float32
    rooms:   np.ndarray        # (N, ROOM_DIM)  float32
    names:   list[str]
    dim:     int
    plan_cache: dict[str, dict]
```
Eliminates all module-level globals (`db_descs`, `db_names`, `db_shapes`, etc.) from `server.py`.

**`render_backend.py`** public interface:
```python
def render(backend: str, params: dict) -> tuple[str, dict]:
    """Returns (image_data_url, meta_dict)."""
```

### Front-end — `retrieval_tool.html` JS → 10 ES modules

```
src/
  main.js             — DOMContentLoaded wiring, addEventListener registration,
                        resize(), buildLegend(), autoLoad() call
  constants.js        — all const declarations + room_types.json import
  state.js            — all let mutable state, exported as plain object
  geometry.js         — n2c, c2n, pdist, planToCanvas, canvasToPlan, ptInPoly,
                        ptToSegDistSq, rotatePtAround, bdryCenter, samplePerimeter,
                        polyArea, projSnap, snapToEdge, ghost
  descriptors.js      — shapeDesc, semDesc, combinedDesc, l2dist,
                        entranceDescJS, roomCountDescFromReqs
  canvas-renderer.js  — render(), drawFP(), drawDoorsWindows(), drawGraph(),
                        drawBoundaryOverlay(), drawScaleGrid(), drawScaleBar(),
                        drawDisplacementArrows(), drawFitHighlights(), updateCursor()
  boundary-editor.js  — closeBoundary(), clearBoundary(), toggleEntranceMode(),
                        clearEntrance(), updateEntranceInfo(), hitTestRooms(),
                        hitTestEntrance(), saveBoundary(), loadBoundary(),
                        renderLib(), exportLibrary(), importLibrary()
  retrieval.js        — runRetrieval(), runRetrievalLocal(), renderAdaptedResults(),
                        renderList(), selectResult(), checkReady(), setStatus(),
                        tryFinalise(), setupDrop(), fetchWithCache(), clearPlanCache(),
                        toggleFitMode(), fetchFitted(), fitPlanLocal()
  viewer-3d.js        — init3DViewer(), build3DSceneFromActivePlan(),
                        sync3DSceneIfNeeded(), captureDepthAndPreview(),
                        and all Three.js geometry builders, ensureThreeDeps()
  render-pipeline.js  — captureDepthAndPreview(), composeRenderPrompt(),
                        refreshStructuredPromptPreview(), generateRenderMock(),
                        checkComfyConnection(), loadWorkflowOverride(),
                        updateRenderResultUI(), updateCapturePanel(), setCaptureView(),
                        downloadLatestCaptureBundle()
```

**Dependency rules (strict):**  
`constants` and `state` have no imports.  
`geometry` imports `constants`, reads `state`.  
`descriptors` imports `constants`, `geometry`.  
`canvas-renderer` imports `constants`, reads `state`, imports `geometry`. **Never writes state.**  
`boundary-editor` imports `constants`, `state` (reads + writes), `geometry`, `canvas-renderer`.  
`retrieval` imports `constants`, `state`, `geometry`, `descriptors`, `canvas-renderer`.  
`viewer-3d` imports `constants`, `state`, `geometry`. External: Three.js via CDN.  
`render-pipeline` imports `constants`, `state`, `viewer-3d`.  
`main` imports everything; is the only file that registers event listeners on `document` and `window`.

---

## Fixes to Bake In Per Phase

These are not separate tickets — implement them during the phase that touches the relevant code.

| Phase | Fix |
|---|---|
| 1 | Replace all inline `oninput`/`onchange`/`onclick` HTML attributes with `addEventListener` in `main.js` |
| 2 | Memoise `bdryCenter()` result per frame in `canvas-renderer.js` (dirty flag on boundary change) |
| 3 | Add `!resp.ok` guard in `fetchWithCache()` with descriptive error; add `AbortController` + 15 s timeout to `/retrieve` fetch |
| 4 | `composeRenderPrompt()` reads `state`, not DOM; `#btn-generate-render` disabled reactively via `updateRenderReadiness()`; add `AbortController` + 200 s timeout to `/render` fetch; add `reader.onerror` to `loadWorkflowOverride()` |
| 5 | Fix `threeDepsPromise` — reset to `null` on rejection so retry is possible without page reload |
| 7 | Move OpenAI API key to server env var (`OPENAI_API_KEY`), remove from request body; add `static_files()` allowlist (only `.html`, `.js`, `.json`, `.bin`, `.png`, `.css`); validate `comfy_url` against `localhost` / `127.0.0.1` only unless a server flag enables remote |

---

## Phasing

### Phase 0 — Pre-refactor fixes
**Deliverable:** clean baseline commit on `feature/refactor`.  
N_SAMP fix, try/finally in captureDepthAndPreview, keydown guard, dead code removal, `room_types.json` created and imported by both Python and JS (replacing all hardcoded type sets).

### Phase 1 — Foundation
**Deliverable:** Vite running, `constants.js` + `state.js` extracted, all inline HTML event handlers replaced.  
The app must behave identically. No functions have moved — only their wiring.

### Phase 2 — Pure functions
**Deliverable:** `geometry.js` + `descriptors.js` as separate files, fully tested by running retrieval and confirming result rankings are unchanged.  
These modules have zero DOM access and zero state writes — easiest to verify.

### Phase 3 — Data and retrieval layer
**Deliverable:** `retrieval.js` (client) + `index_store.py` + `retrieval.py` (server).  
Add `!resp.ok` guard and `AbortController` timeout. Verify local and server retrieval now return the same rankings (N_SAMP fix from Phase 0 should already have closed this gap).

### Phase 4 — Render pipeline
**Deliverable:** `render-pipeline.js` (client) + `render_backend.py` + `comfy_client.py` (server).  
`composeRenderPrompt()` reads state, not DOM. Generate button disabled reactively via `updateRenderReadiness()`. All fetch timeouts and error handling in place.

### Phase 5 — 3D viewer
**Deliverable:** `viewer-3d.js`.  
Fix `threeDepsPromise` rejection caching.

### Phase 6 — Canvas and boundary
**Deliverable:** `canvas-renderer.js` + `boundary-editor.js`.  
Most coupled layer — do last. `render()` must never write state.

### Phase 7 — Hardening
**Deliverable:** `main.js` final wiring, security fixes, `fitting.py` extracted from server, `/adapt` route removed, `static_files` allowlist, OpenAI key to env.

---

## Hardcoded Values to Centralise

The following values appear in multiple files and must be changed in all places together. During refactor, move each to its canonical location:

| Value | Current locations | Canonical location after refactor |
|---|---|---|
| `comfyui-sdxl-depth-multi-workflow.json` | `server.py` (×1), `retrieval_tool.html` (×2) | `server.py` config constant; exposed via a `/config` GET endpoint so the client can display the active filename dynamically |
| `TOTAL_DIM=63`, `SHAPE_DIM=48`, `ENT_DIM=4`, `ROOM_DIM=11` | `server.py`, `retrieval_tool.html`, `geo_ent_num_retrieval.py` | `geo_ent_num_retrieval.py` (Python source of truth); JS reads from a `/config` endpoint |
| `N_SAMP=256`, `N_FREQS=48`, `SCALE=256` | `geo_ent_num_retrieval.py`, `retrieval_tool.html` | Same — JS syncs from `/config` |
| `ROOM_HEIGHT=2.8`, `METRES_PER_UNIT=18` | `retrieval_tool.html` | `constants.js` |
| `http://127.0.0.1:8188` | `retrieval_tool.html` (×4) | `constants.js` `DEFAULT_COMFY_URL` |
| `fp-plans-v1` Cache API key | `retrieval_tool.html` | `constants.js` — append schema version from `/config` |

---

## Notes for the Coding Agent

- Work phase by phase. Do not begin Phase 2 until Phase 1 passes a manual smoke test (retrieval runs, 3D view loads, render submits).
- Every phase should end with a commit. The commit message should name the phase.
- If time is tight, Phases 5–7 can follow in a subsequent sprint.
- `getActiveFPState()` returns a live reference into `results[]`. Callers that store the `fp` field and later mutate it are directly mutating the results array. This is intentional behaviour for the drag editor — do not change it, but document it in a JSDoc comment.
- `render()` (2D canvas) is called on every mousemove without frame-rate limiting. This is a known performance issue (R3) — do not fix it during this refactor to avoid scope creep. Log it as a follow-up.
- The `/config` endpoint proposed above does not exist yet. Until it is implemented in Phase 3, JS constants can remain duplicated — the key fix is Phase 0's N_SAMP correction.
