"""
apps/api/server.py — Flask server connecting the browser floor plan tool to the
Python retrieval and adaptation pipeline.

Pipeline stage context:  retrieve → fit → render → present

Routes
------
GET  /                  Serve index.html
GET  /config            Serve browser runtime config (non-secret)
GET  /<path:filename>   Serve allowlisted static files (.html/.js/.json/.bin/.png/.css)
POST /retrieve          Run retrieval and return ranked plans as JSON
POST /fit               Fit rooms to the query boundary, return updated plan JSON
POST /fit_schema        Run the schema.py-native adaptation chain (fitting-prototype),
                         return a plan-drawing-panel-ready fixture
POST /render            Run the selected rendering backend and return an image
POST /comfy_health      Check ComfyUI connectivity
POST /fit_step1         Spatial reasoning agent — reposition layout graph nodes
POST /topo_check        Topology-only sensor registry (reachability, typed access,
                         daylight, entrance) over a plan, recomputed fresh
"""

import argparse
import logging
import sys
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.floorplan_core.geo_ent_num_retrieval import (
    N_FREQS, N_SAMP, ENT_DIM, ROOM_DIM,
)
from packages.floorplan_core.fit_step1 import FitStep1Error, run_fit_step1_agent
from packages.floorplan_core.fitting import FittingError, fit_rooms_in_boundary
from packages.floorplan_core.index_store import build_index_store, build_index_store_from_npz, ensure_manifest
from packages.floorplan_core.request_parsing import (
    RequestValidationError,
    parse_comfy_health_request,
    parse_fit_request,
    parse_fit_step1_request,
    parse_render_request,
    parse_retrieve_request,
)
from packages.floorplan_core.render_backend import DEFAULT_COMFY_WORKFLOW_FILENAME, check_comfy_health, render
from packages.floorplan_core.retrieval import RetrievalError, run_retrieval
from packages.floorplan_core.schema_bridge import run_fit_pipeline
from packages.floorplan_core.topo_signature import topological_fitness, default_edge_types
from packages.floorplan_core.topo_sensors import run_sensors
from packages.floorplan_core.arap_bridge import (
    STEPS as ARAP_STEPS,
    CANVAS_STEPS as ARAP_CANVAS_STEPS,
    feasibility_check as arap_feasibility_check,
    rerank_unit as arap_rerank_unit,
)

SHAPE_DIM = N_FREQS                              # 48
TOTAL_DIM = SHAPE_DIM + ENT_DIM + ROOM_DIM      # 64 (canonical: 48 + 5 + 11)

# Canonical retrieval corpus (2026-08-10 decision — see docs/pipeline-architecture.md
# "Cross-cutting concerns" / artefacts.md's Retrieval prototype entry). Lives in
# Artefacts, shared with fitting-prototype and retrieval-prototype, not duplicated
# into this repo's own data/ — 80,791 plans.
RPLAN_DATASET_DIR = Path(
    "/Users/camilo/Documents/Work_FBI/Artefacts/project-44/data/RPlan-dataset-json"
)

# ── globals (populated at startup) ───────────────────────────────────────────
WEB_DIR = REPO_ROOT / "apps" / "web"
WEB_UTILITY_DIR = WEB_DIR / "legacy"
DATA_INDEX_DIR = REPO_ROOT / "data" / "index" / "browser"
DATA_MANIFEST_FILE = REPO_ROOT / "data" / "index" / "manifest.json"
CONFIG_WORKFLOWS_DIR = REPO_ROOT / "configs" / "workflows"
ROOM_TYPES_FILE = REPO_ROOT / "configs" / "room_types.json"
DATA_FIXTURES_TEST_DIR = REPO_ROOT / "data" / "fixtures" / "plans" / "test_data"
FIT_STEP1_BRIEF_PATH = REPO_ROOT / "docs" / "dev" / "design" / "fit-step1-brief.md"
INDEX_STORE = None
ALLOWED_STATIC_EXTS = {".html", ".js", ".json", ".bin", ".png", ".css"}
WORKFLOW_STATIC_FILES = {
    "comfyui-flux-depth-workflow.json",
    "comfyui-sdxl-depth-multi-workflow.json",
}
UTILITY_WEB_FILES = {"retrieval_tool.html", "index_explorer.html"}
ALLOW_REMOTE_COMFY = False
DEFAULT_RENDER_BACKEND = "comfy"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_OPENAI_MODEL = "gpt-image-1"
DEFAULT_OPENAI_QUALITY = "medium"
DEFAULT_PLAN_CACHE_KEY = "fp-plans-v1"

app = Flask(__name__, static_folder=None)
logger = logging.getLogger(__name__)


# ── static routes ─────────────────────────────────────────────────────────────

@app.route("/config")
def client_config():
    """Return non-secret runtime config consumed by the browser app."""
    return jsonify({
        "status": "ok",
        "config": {
            "render": {
                "default_backend": DEFAULT_RENDER_BACKEND,
                "comfy_url": DEFAULT_COMFY_URL,
                "workflow_filename": DEFAULT_COMFY_WORKFLOW_FILENAME,
                "openai_model": DEFAULT_OPENAI_MODEL,
                "openai_quality": DEFAULT_OPENAI_QUALITY,
            },
            "descriptors": {
                "shape_dim": SHAPE_DIM,
                "ent_dim": ENT_DIM,
                "room_dim": ROOM_DIM,
                "total_dim": TOTAL_DIM,
                "n_freqs": N_FREQS,
                "n_samp": N_SAMP,
                "scale": 256,
            },
            "cache": {
                "plan_cache_key": DEFAULT_PLAN_CACHE_KEY,
            },
            "security": {
                "allow_remote_comfy": ALLOW_REMOTE_COMFY,
            },
        },
    })

@app.route("/")
def index():
    if (WEB_DIR / "index.html").exists():
        return send_from_directory(WEB_DIR, "index.html")
    return send_from_directory(REPO_ROOT, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve static files from explicit safe locations.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_STATIC_EXTS:
        return jsonify({"error": f"disallowed file type: {suffix or '(none)'}"}), 404

    web_full = WEB_DIR / filename
    if web_full.exists():
        return send_from_directory(WEB_DIR, filename)

    # Utility pages are served from apps/web/legacy while keeping stable URLs.
    if filename in UTILITY_WEB_FILES:
        utility_file = WEB_UTILITY_DIR / filename
        if utility_file.exists():
            return send_from_directory(WEB_UTILITY_DIR, filename)

    # Index alias: prefer data-owned index path.
    if filename in {"index.bin", "index_names.json"}:
        idx_path = DATA_INDEX_DIR / filename
        if idx_path.exists():
            return send_from_directory(DATA_INDEX_DIR, filename)

    # Manifest is served from its canonical data-owned path.
    if filename == "manifest.json":
        if DATA_MANIFEST_FILE.exists():
            return send_from_directory(DATA_MANIFEST_FILE.parent, DATA_MANIFEST_FILE.name)

    # Dataset alias: /test_data/<file>.json -> data/fixtures/plans/test_data/<file>.json
    if filename.startswith("test_data/"):
        rel = filename[len("test_data/"):]
        preferred = DATA_FIXTURES_TEST_DIR / rel
        if preferred.exists():
            return send_from_directory(DATA_FIXTURES_TEST_DIR, rel)

    # Workflow config alias: keep URL compatibility while files live in configs/workflows.
    if filename in WORKFLOW_STATIC_FILES:
        wf_path = CONFIG_WORKFLOWS_DIR / filename
        if wf_path.exists():
            return send_from_directory(CONFIG_WORKFLOWS_DIR, filename)

    # Compatibility alias: keep URL stable while room type config lives in configs/.
    if filename == "room_types.json" and ROOM_TYPES_FILE.exists():
        return send_from_directory(ROOM_TYPES_FILE.parent, ROOM_TYPES_FILE.name)

    return jsonify({"error": f"file not found: {filename}"}), 404


# ── /retrieve endpoint ────────────────────────────────────────────────────────

@app.route("/retrieve", methods=["POST"])
def retrieve():
    global INDEX_STORE

    # ── index guard ───────────────────────────────────────────────────────────
    if INDEX_STORE is None:
        return jsonify({"error": "index not loaded"}), 503

    # ── parse request ──────────────────────────────────────────────────────────
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    try:
        parsed = parse_retrieve_request(body)
    except RequestValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        payload = run_retrieval(
            index_store=INDEX_STORE,
            boundary=parsed["boundary"],
            entrance_point=parsed["entrance_point"],
            requirements=parsed["requirements"],
            topn=parsed["topn"],
            shape_w=parsed["shape_w"],
            ent_w=parsed["ent_w"],
            room_w=parsed["room_w"],
            logger=logger,
            typed_edges=parsed.get("typed_edges"),
        )
    except RetrievalError as exc:
        logger.exception("Retrieval failed")
        return jsonify({"error": str(exc)}), 500

    return jsonify(payload)


# ── /fit endpoint ─────────────────────────────────────────────────────────────

@app.route("/fit", methods=["POST"])
def fit():
    """
    Move regular rooms of a floor plan to the nearest position fully inside
    the query boundary.

    Request body (JSON):
        fp       : floor plan dict (v2.1.0)
        boundary : [[x, y], ...] — query boundary in [0, 1] coords
                   (v2.1.0 [x,y,dir,is_door] format also accepted)

    Response (JSON):
        {status, fp, stats: {moved, skipped, no_fit}}
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    try:
        parsed = parse_fit_request(body)
    except RequestValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        fitted_fp, stats = fit_rooms_in_boundary(parsed["fp"], parsed["boundary"])
    except FittingError as exc:
        logger.exception("fit_plan failed")
        return jsonify({"error": str(exc)}), 500

    return jsonify({"status": "ok", "fp": fitted_fp, "stats": stats})


# ── /fit_schema endpoint ──────────────────────────────────────────────────────

@app.route("/fit_schema", methods=["POST"])
def fit_schema():
    """
    Run the schema.py-native adaptation chain (place_initial_layout ->
    fit_rooms_to_boundary -> preprocess_layout -> phase25 diagnose/agent/execute
    repair loop, from fitting-prototype) on a retrieved plan against the
    boundary the user drew, and return the same {schema_version, label,
    boundary, plan, violations} fixture shape the architectural-drawing panel
    already renders (docs/pipeline-architecture.md, task 1). The repair loop
    calls a local Ollama model (qwen3.5:9b) for up to 5 passes — can take
    a while; the frontend gives this a generous timeout.

    Request body (JSON):
        fp             : retrieved plan dict (v2.1.0), in the plan's own space
        boundary       : [[x, y], ...] target boundary, same space as fp
                          (i.e. already passed through boundaryInPlanSpace())
        entrance_point : [x, y] target entrance, same space as boundary
        label          : optional string for the returned fixture's label

    Response (JSON):
        {status, fixture: {schema_version, label, boundary, plan, violations}}
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    fp = body.get("fp")
    boundary = body.get("boundary")
    entrance_point = body.get("entrance_point")
    if not fp or not boundary or len(boundary) < 3:
        return jsonify({"error": "missing or invalid fp/boundary"}), 400
    if not entrance_point:
        return jsonify({"error": "missing entrance_point"}), 400

    try:
        fixture = run_fit_pipeline(fp, boundary, entrance_point, label=body.get("label", "live"))
    except Exception as exc:
        logger.exception("fit_schema pipeline failed")
        return jsonify({"error": str(exc)}), 500

    return jsonify({"status": "ok", "fixture": fixture})


# ── /arap_step endpoint ───────────────────────────────────────────────────────

@app.route("/arap_step", methods=["POST"])
def arap_step():
    """
    Run the actual ARAP mesh-deformation pipeline (fitting-prototype's
    arap_adapt.py — distinct from /fit_schema's phase1/phase2/phase25 chain,
    which never calls arap_adapt.py at all) up to one named step, and return
    a plan-drawing-panel-ready fixture for that step's geometry. Stateless —
    each step recomputes from scratch through that point (cheap: pure
    numpy/scipy, no LLM), rather than threading state between requests.

    Request body (JSON):
        step           : one of "align" | "rectify" | "reconstruct" | "refine"
        fp             : retrieved plan dict (v2.1.0), in the plan's own space
        boundary       : [[x, y], ...] target boundary, as drawn (query space)
        entrance_point : [x, y] target entrance, same space as boundary

    Response (JSON):
        {status, fixture: {schema_version, label, boundary, plan, violations,
                            score?}}
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    step = body.get("step")
    if step not in ARAP_STEPS:
        return jsonify({"error": f"unknown step {step!r} — expected one of {list(ARAP_STEPS)}"}), 400

    fp = body.get("fp")
    boundary = body.get("boundary")
    entrance_point = body.get("entrance_point")
    if not fp or not boundary or len(boundary) < 3:
        return jsonify({"error": "missing or invalid fp/boundary"}), 400
    if not entrance_point:
        return jsonify({"error": "missing entrance_point"}), 400

    try:
        fixture = ARAP_STEPS[step](fp, boundary, entrance_point)
    except Exception as exc:
        logger.exception("arap_step (%s) failed", step)
        return jsonify({"error": str(exc)}), 500

    return jsonify({"status": "ok", "fixture": fixture})


# ── /arap_canvas endpoint ─────────────────────────────────────────────────────

@app.route("/arap_canvas", methods=["POST"])
def arap_canvas():
    """
    Run one ARAP pipeline step and return room polygons in the app's own
    v2.1.0 canvas shape (int type, [0,1]-normalised polygon) — for drawing
    directly via the app's existing drawFP canvas renderer, not a separate
    schema.py-fixture modal. Steps: "align" (build_mesh + similarity_align),
    "solve" (+ compute_handles + arap_solve), "rectify" (+ rectify_mesh +
    rectify_rooms), "fill" (+ fill_boundary_with_living +
    consolidate_living), "oracle" (+ vlm_refine.surface_defects — measures,
    doesn't change), "agent" (+ vlm_refine.orchestrate_refinement — oracle
    + deterministic guard + VLM paired-preference gate; needs Ollama
    serving qwen3-vl:8b) — see arap_bridge.CANVAS_STEPS.

    Request body (JSON): step, fp, boundary, entrance_point — same as /arap_step.
    Response (JSON): {status, rooms: [{id, type, polygon}], ...step-specific
    extras (e.g. "defects" for oracle, "verified"/"log"/"query_id" for agent)}
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    step = body.get("step")
    if step not in ARAP_CANVAS_STEPS:
        return jsonify({"error": f"unknown step {step!r} — expected one of {list(ARAP_CANVAS_STEPS)}"}), 400

    fp = body.get("fp")
    boundary = body.get("boundary")
    entrance_point = body.get("entrance_point")
    # Optional: the boundary's own edge classification, as [{edge_index, type}].
    # A floor-plate unit carries it; a freehand boundary omits it and the
    # bridge synthesises the default. Passed through so a unit's shared_wall
    # edges survive the handoff instead of being relabelled external.
    typed_edges = body.get("boundary_typed_edges")
    if not fp or not boundary or len(boundary) < 3:
        return jsonify({"error": "missing or invalid fp/boundary"}), 400
    if not entrance_point:
        return jsonify({"error": "missing entrance_point"}), 400

    try:
        result = ARAP_CANVAS_STEPS[step](fp, boundary, entrance_point, typed_edges=typed_edges)
    except Exception as exc:
        logger.exception("arap_canvas (%s) failed", step)
        return jsonify({"error": str(exc)}), 500

    # Pass through whatever the step returned (rooms always; door when the
    # step tracks it) rather than hardcoding which keys survive — arap_bridge
    # owns what a given step produces, this route shouldn't silently drop
    # fields a step adds (this is exactly how "door" went missing before).
    return jsonify({"status": "ok", **result})


# ── /feasibility_check endpoint ────────────────────────────────────────────────

@app.route("/feasibility_check", methods=["POST"])
def feasibility_check_route():
    """Stage 0 pre-analysis — advisory check that the requested program can fit
    the drawn boundary (area + external frontage), before retrieval. See
    arap_bridge.feasibility_check and docs/re-solve-loop-build-plan.md."""
    body = request.get_json(force=True)
    boundary = body.get("boundary")
    entrance_point = body.get("entrance_point")
    req = body.get("req") or {}
    typed_edges = body.get("boundary_typed_edges")
    if not boundary or len(boundary) < 3:
        return jsonify({"error": "missing or invalid boundary"}), 400
    if not entrance_point:
        return jsonify({"error": "missing entrance_point"}), 400
    try:
        result = arap_feasibility_check(boundary, entrance_point, req, typed_edges)
    except Exception as exc:
        logger.exception("feasibility_check failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify({"status": "ok", **result})


# ── /rerank_unit endpoint ─────────────────────────────────────────────────────

@app.route("/rerank_unit", methods=["POST"])
def rerank_unit_route():
    """Step 4 — viability rerank over a unit's top candidates × mirror state
    (arrangement-refactor-build-plan.md Step 4). Replaces the blind rank-1 pick:
    every variant is adapted and scored by the arrangement viability score, and
    the most viable wins.

    Request body (JSON): variants [{fp, tag, candidate, mirror}], boundary,
    entrance_point, boundary_typed_edges.
    Response (JSON): the winning arrangement (rooms/door/defects, same shape as
    /arap_canvas oracle) plus winner + the scored table.
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    variants = body.get("variants")
    boundary = body.get("boundary")
    entrance_point = body.get("entrance_point")
    typed_edges = body.get("boundary_typed_edges")
    if not variants or not boundary or len(boundary) < 3:
        return jsonify({"error": "missing or invalid variants/boundary"}), 400
    if not entrance_point:
        return jsonify({"error": "missing entrance_point"}), 400

    try:
        result = arap_rerank_unit(variants, boundary, entrance_point, typed_edges=typed_edges)
    except Exception as exc:
        logger.exception("rerank_unit failed")
        return jsonify({"error": str(exc)}), 500

    return jsonify({"status": "ok", **result})


# ── /fit_step1 endpoint ───────────────────────────────────────────────────────

@app.route("/fit_step1", methods=["POST"])
def fit_step1():
    """
    Spatial reasoning agent — reposition layout graph nodes.

    Request body (JSON):
        fp       : floor plan dict (v2.1.0) — the retrieved plan to refit
        boundary : [[x, y], ...] — the target boundary in normalised [0,1] coords

    Response (JSON):
        {status, positions: {id: {x, y}, ...}}
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    try:
        parsed = parse_fit_step1_request(body)
    except RequestValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    if not FIT_STEP1_BRIEF_PATH.exists():
        return jsonify({"error": "fit-step1 brief not found on server"}), 503

    try:
        result = run_fit_step1_agent(
            plan=parsed["fp"],
            target_boundary=parsed["boundary"],
            entrance_point=parsed["entrance_point"],
            suggested_positions=parsed["suggested_positions"],
            brief_text=FIT_STEP1_BRIEF_PATH.read_text(),
        )
    except FitStep1Error as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        logger.exception("fit_step1 agent failed")
        return jsonify({"error": "agent error — check server logs"}), 500

    return jsonify({"status": "ok", "positions": result["positions"]})


# ── /render endpoint (Phase 6) ───────────────────────────────────────────────

@app.route("/render", methods=["POST"])
def render_image():
    """
    Generate an architectural render from depth map + prompt.

            Request body:
                backend          : "openai" (default) or "comfy"
      prompt           : str
      depth_data_url   : data:image/png;base64,...
        building_mask_data_url : optional data:image/png;base64,...
        depth_invert               : optional bool (reserved)
            comfy_url                  : optional str (comfy backend; default http://127.0.0.1:8188)
            workflow_override          : optional dict (comfy backend; workflow JSON)
            segmentation_data_url      : optional data:image/png;base64,... (comfy backend)
            openai_api_key             : optional str (openai backend; falls back to OPENAI_API_KEY env)
            openai_model               : optional str (openai backend; default gpt-image-1)
            openai_quality             : optional str (openai backend; default medium)
            openai_width               : optional int (openai backend target width)
            openai_height              : optional int (openai backend target height)

    Response body:
      {status, backend, image_data_url, meta}
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    try:
        parsed = parse_render_request(body)
    except RequestValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    backend = parsed["backend"]

    try:
        image_data_url, meta = render(
            backend=backend,
            params=parsed["params"],
            base_dir=REPO_ROOT,
            logger=logger,
            allow_remote_comfy=ALLOW_REMOTE_COMFY,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Render failed")
        if backend in {"comfy", "comfyui", "local"}:
            return jsonify({"error": f"comfy render failed: {exc}"}), 500
        return jsonify({"error": f"openai render failed: {exc}"}), 500

    resolved_backend = "comfy" if backend in {"comfy", "comfyui", "local"} else "openai"
    return jsonify({
        "status": "ok",
        "backend": resolved_backend,
        "image_data_url": image_data_url,
        "meta": meta,
    })


@app.route("/comfy_health", methods=["POST"])
def comfy_health():
    """Check if ComfyUI is reachable and responding."""
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    comfy_url = parse_comfy_health_request(body)
    try:
        summary = check_comfy_health(comfy_url, allow_remote_comfy=ALLOW_REMOTE_COMFY)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"could not reach ComfyUI at {comfy_url}: {exc}"}), 502

    return jsonify({"status": "ok", "summary": summary})


# ── /generate_floor_plate endpoint ────────────────────────────────────────────

@app.route("/generate_floor_plate", methods=["POST"])
def generate_floor_plate():
    """
    Generate an initial floor plate layout from boundary and unit mix.

    Request body (JSON):
        boundary    : [[x, y], ...] — site boundary in mm coordinates
        unit_mix    : [{"type": "1bed", "count": 3, "target_area": 55}, ...]
        constraints : {
                        "corridor_width": 1500,
                        "max_unit_depth": 8000,
                        "min_entrance_overlap": 1200,
                        "min_common_area": 20,
                        "max_travel_distance": 30000,
                        "core_type": "stair_lift"
                      }

    Response (JSON):
        {
            status: "ok",
            floor_plate: {
                boundary, typology, cores, corridors, units, 
                unresolved_regions, compliance_snapshot, generation_log
            }
        }
    """
    try:
        from packages.floor_plate_design.generator import generate_initial_layout
        from packages.floor_plate_design.types import (
            EdgeType,
            TypologyType,
            SlabSubType,
            ConfidenceLevel,
        )
    except ImportError as exc:
        return jsonify({"error": f"floor_plate_design module not found: {exc}"}), 500

    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    # Validate request
    if "boundary" not in body:
        return jsonify({"error": "missing 'boundary'"}), 400
    if "unit_mix" not in body:
        return jsonify({"error": "missing 'unit_mix'"}), 400

    boundary = body["boundary"]
    unit_mix = body["unit_mix"]
    constraints = body.get("constraints", {})

    # Set defaults for missing constraints
    if "corridor_width" not in constraints:
        constraints["corridor_width"] = 1500
    if "max_unit_depth" not in constraints:
        constraints["max_unit_depth"] = 8000
    if "min_entrance_overlap" not in constraints:
        constraints["min_entrance_overlap"] = 1200
    if "min_common_area" not in constraints:
        constraints["min_common_area"] = 20
    if "max_travel_distance" not in constraints:
        constraints["max_travel_distance"] = 30000
    if "core_type" not in constraints:
        constraints["core_type"] = "stair_lift"

    try:
        # Validate input
        if not isinstance(boundary, list) or len(boundary) < 3:
            return jsonify({"error": "boundary must be a list of at least 3 [x, y] points"}), 400
        if not isinstance(unit_mix, list) or len(unit_mix) == 0:
            return jsonify({"error": "unit_mix must be a non-empty list"}), 400

        # Convert boundary to tuples
        boundary_tuples = [tuple(p) for p in boundary]

        # Generate layout
        floor_plate = generate_initial_layout(boundary_tuples, unit_mix, constraints)

        # Serialize to JSON-compatible format
        result = {
            "boundary": floor_plate.boundary,
            "analysis": floor_plate.analysis,
            "typology": {
                "type": floor_plate.typology.type.value,
                "sub_type": floor_plate.typology.sub_type.value if hasattr(floor_plate.typology.sub_type, 'value') else str(floor_plate.typology.sub_type),
                "confidence": floor_plate.typology.confidence.value,
                "reasoning": floor_plate.typology.reasoning,
            },
            "cores": [
                {
                    "id": core.id,
                    "polygon": core.polygon,
                    "type": core.type.value,
                    "centroid": list(core.centroid),
                    "width": core.width,
                    "depth": core.depth,
                }
                for core in floor_plate.cores
            ],
            "corridors": [
                {
                    "id": corr.id,
                    "polygon": corr.polygon,
                    "width": corr.width,
                    "centreline": corr.centreline,
                }
                for corr in floor_plate.corridors
            ],
            "units": [
                {
                    "id": unit.id,
                    "type": unit.type,
                    "polygon": unit.polygon,
                    "typed_edges": [
                        {"edge_index": e.edge_index, "type": e.type.value}
                        for e in unit.typed_edges
                    ],
                    "entrance_point": list(unit.entrance_point),
                    "depth": unit.depth,
                    "area": unit.area,
                }
                for unit in floor_plate.units
            ],
            "unresolved_regions": floor_plate.unresolved_regions,
            "compliance_snapshot": {
                "common_area_size": {
                    "check_name": floor_plate.compliance_snapshot.common_area_size.check_name,
                    "pass": floor_plate.compliance_snapshot.common_area_size.pass_,
                    "value": floor_plate.compliance_snapshot.common_area_size.value,
                    "limit": floor_plate.compliance_snapshot.common_area_size.limit,
                    "message": floor_plate.compliance_snapshot.common_area_size.message,
                },
                "violations": floor_plate.compliance_snapshot.violations,
                "warnings": floor_plate.compliance_snapshot.warnings,
            },
            "generation_log": floor_plate.generation_log,
        }

        return jsonify({"status": "ok", "floor_plate": result})

    except Exception as exc:
        logger.exception("Floor plate generation failed")
        return jsonify({"error": str(exc)}), 500


# ── /subdivide_floor_plate endpoint ──────────────────────────────────────────

@app.route("/subdivide_floor_plate", methods=["POST"])
def subdivide_floor_plate():
    """Run the subdivision-only contract solver in single or harness mode."""
    try:
        from packages.floor_plate_design.generator import generate_subdivision_response
    except ImportError as exc:
        return jsonify({"error": f"floor_plate_design module not found: {exc}"}), 500

    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    boundary = body.get("boundary")
    unit_mix = body.get("unit_mix")
    constraints = body.get("constraints", {}) or {}
    scoring = body.get("scoring", {}) or {}
    mode = str(body.get("mode", "single") or "single").strip().lower()
    seed = body.get("seed")

    if mode not in {"single", "harness"}:
        return jsonify({"error": "mode must be 'single' or 'harness'"}), 400
    if not isinstance(boundary, list) or len(boundary) < 3:
        return jsonify({"error": "boundary must be a list of at least 3 [x, y] points"}), 400
    if not isinstance(unit_mix, dict):
        return jsonify({"error": "unit_mix must be an object mapping unit type to count"}), 400
    if not isinstance(constraints, dict):
        return jsonify({"error": "constraints must be an object"}), 400
    if not isinstance(scoring, dict):
        return jsonify({"error": "scoring must be an object"}), 400

    def _to_mm(point):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("boundary points must be [x, y]")
        x, y = float(point[0]), float(point[1])
        scale = 1000.0 if max(abs(x), abs(y)) <= 1000.0 else 1.0
        return (x * scale, y * scale)

    try:
        boundary_mm = [_to_mm(point) for point in boundary]
        result = generate_subdivision_response(
            boundary_mm,
            unit_mix,
            constraints=constraints,
            scoring=scoring,
            mode=mode,
            seed=seed if isinstance(seed, int) else None,
        )

        def _to_m_polygon(poly):
            return [[float(x) / 1000.0, float(y) / 1000.0] for x, y in poly]

        def _convert_single_response(payload):
            if not payload:
                return payload
            converted = dict(payload)
            converted["placed_units"] = [
                {
                    **unit,
                    "polygon": _to_m_polygon(unit["polygon"]),
                    **({"entrance_point": [unit["entrance_point"][0] / 1000.0,
                                           unit["entrance_point"][1] / 1000.0]}
                       if unit.get("entrance_point") else {}),
                }
                for unit in payload.get("placed_units", [])
            ]
            converted["residual_zones"] = [
                {
                    **zone,
                    "polygon": _to_m_polygon(zone["polygon"]),
                }
                for zone in payload.get("residual_zones", [])
            ]
            converted["cores"] = [
                {
                    **core,
                    "polygon": _to_m_polygon(core["polygon"]),
                    "centroid": [core["centroid"][0] / 1000.0, core["centroid"][1] / 1000.0],
                    "width": core.get("width", 0) / 1000.0,
                    "depth": core.get("depth", 0) / 1000.0,
                }
                for core in payload.get("cores", [])
            ]
            converted["corridors"] = [
                {
                    **corr,
                    "polygon": _to_m_polygon(corr["polygon"]),
                    "width": corr.get("width", 0) / 1000.0,
                    "centreline": [[p[0] / 1000.0, p[1] / 1000.0] for p in corr.get("centreline", [])],
                }
                for corr in payload.get("corridors", [])
            ]
            if "best" in converted and isinstance(converted["best"], dict):
                converted["best"] = _convert_single_response(converted["best"])
            if isinstance(converted.get("candidates"), list):
                converted["candidates"] = [_convert_single_response(candidate) for candidate in converted["candidates"]]
            return converted

        result = _convert_single_response(result)
        return jsonify({"status": "ok", "response": result})

    except Exception as exc:
        logger.exception("Subdivision solver failed")
        return jsonify({"error": str(exc)}), 500


# ── /fit_harness endpoint ────────────────────────────────────────────────────

@app.route("/fit_harness", methods=["POST"])
def fit_harness():
    """
    Fit a retrieved plan to a boundary using the agent harness tools.

    Pipeline:
    0. scale_plan_to_boundary — uniformly scale all rooms so the plan's bounding
       box approximately matches the query boundary's bounding box.  This replaces
       the edge-push approach: pre-scaling preserves relative room positions and
       avoids the clustering / inter-room overlap that the edge-push creates.
    1. clip_room_to_boundary — clip each non-common room to the boundary polygon
       to handle any rooms that land outside due to non-rectangular boundaries.
    2. resolve overlaps by priority — bedroom > bath/kitchen > misc.  Higher-
       priority rooms cut from lower-priority rooms via subtract_room.
    3. compliance adjustment loop — for each room failing architectural standards
       (G6), runs plan_room_expansion to find the best interior edge push, applies
       it (with counter-move to the neighbor), then re-clips.  Up to MAX_ADJUST_ITERS.
    4. fill_residual — sets the common-area room to boundary minus all others,
       guaranteeing 100% coverage (G4).
    Finally runs run_invariants to check G1–G6.

    Both fp and boundary must be in the same normalized [0,1] coordinate space.
    The caller is responsible for transforming the query boundary into the plan's
    coordinate frame before sending (see boundaryInPlanSpace in main.js).

    Request body (JSON):
        fp       : floor plan dict (v2.1.0) with rooms in normalized [0,1] space
        boundary : [[x, y], ...] — target boundary in the same space

    Response (JSON):
        {status, fp, invariants}
        invariants includes debug_steps showing overlap count at each stage.
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    fp = body.get("fp")
    boundary = body.get("boundary")

    if not isinstance(fp, dict) or not fp.get("rooms"):
        return jsonify({"error": "missing or invalid 'fp' (must have rooms)"}), 400
    if not isinstance(boundary, list) or len(boundary) < 3:
        return jsonify({"error": "boundary must be a list of at least 3 [x, y] points"}), 400

    try:
        from packages.floorplan_core.fit_boolean import (
            clip_room_to_boundary, fill_residual, subtract_room,
        )
        from packages.floorplan_core.fit_invariants import run_invariants, check_overlaps
        from packages.floorplan_core.fit_planner import (
            PLAN_MM_PER_UNIT, check_plan_compliance, apply_best_expansion,
        )
        from packages.floorplan_core.fit_transform import scale_plan_to_boundary
    except ImportError as exc:
        return jsonify({"error": f"fit harness modules not found: {exc}"}), 500

    COMMON_TYPES     = {0, 4}
    MAX_ADJUST_ITERS = 10
    OVERLAP_TOL      = 1e-4
    # Lower index = higher priority (keeps its shape when overlapping).
    # Common-area rooms (0, 4) always yield — replaced by fill_residual.
    OVERLAP_PRIORITY = [1, 5, 7, 8, 3, 2, 6, 9, 10, 11, 12]

    def _room_priority(room):
        st = room.get("source_type", 99)
        if st in COMMON_TYPES:
            return 99
        try:
            return OVERLAP_PRIORITY.index(st)
        except ValueError:
            return 50

    def _step_snapshot(label, p):
        n = len([o for o in check_overlaps(p) if o["overlap_area"] > OVERLAP_TOL])
        return {"step": label, "overlap_count": n}

    debug_steps = []

    try:
        # Step 0: pre-scale — uniformly scale rooms so the plan's combined bbox
        # approximately fills the query boundary.  This preserves relative room
        # positions and avoids the heavy inter-room overlaps the edge-push creates.
        plan = scale_plan_to_boundary(fp, boundary)
        debug_steps.append(_step_snapshot("0_prescale", plan))

        # Step 1: clip each non-common room to the boundary polygon.
        # Handles rooms that land slightly outside after scaling a non-rectangular
        # boundary.  No translation applied — rooms keep their relative positions.
        for room in list(plan.get("rooms", [])):
            if room.get("source_type", -1) in COMMON_TYPES:
                continue
            try:
                plan = clip_room_to_boundary(plan, room["id"], boundary)
            except Exception as exc:
                logger.warning("clip_room_to_boundary skipped %s: %s", room["id"], exc)
        debug_steps.append(_step_snapshot("1_clip", plan))

        # Step 2: resolve any residual room-room overlaps by priority.
        # Higher-priority rooms cut from lower-priority rooms.
        non_common = sorted(
            [r for r in plan.get("rooms", []) if r.get("source_type", -1) not in COMMON_TYPES],
            key=_room_priority,
        )
        for i, donor in enumerate(non_common):
            for cutter in non_common[:i]:
                try:
                    plan = subtract_room(plan, donor["id"], cutter["id"])
                except Exception as exc:
                    logger.warning("subtract_room %s←%s: %s", donor["id"], cutter["id"], exc)
        debug_steps.append(_step_snapshot("2_overlap_resolve", plan))

        # Step 3: compliance adjustment loop
        adjust_log = []
        for iteration in range(MAX_ADJUST_ITERS):
            compliance = check_plan_compliance(plan, PLAN_MM_PER_UNIT)
            if compliance["all_compliant"]:
                break
            non_compliant = [
                rid for rid, info in compliance["rooms"].items()
                if info.get("has_standards") and not info.get("compliant")
            ]
            if not non_compliant:
                break
            room_id = non_compliant[0]
            plan, applied = apply_best_expansion(plan, room_id, boundary, PLAN_MM_PER_UNIT)
            if applied is None:
                logger.info("fit_harness adjust: no valid move for %s", room_id)
                break
            adjust_log.append({
                "iteration": iteration + 1,
                "room_id":   room_id,
                "edge":      applied["edge_index"],
                "delta_mm":  applied["delta_mm"],
                "objective": applied["objective"],
                "score":     applied["score"],
            })
            try:
                plan = clip_room_to_boundary(plan, room_id, boundary)
            except Exception as exc:
                logger.warning("clip after expansion failed for %s: %s", room_id, exc)
        debug_steps.append(_step_snapshot("3_compliance_adjust", plan))

        # Step 4: fill residual
        filled = False
        for room in plan.get("rooms", []):
            if room.get("source_type", -1) in COMMON_TYPES:
                try:
                    plan = fill_residual(plan, room["id"], boundary)
                    filled = True
                except Exception as exc:
                    logger.warning("fill_residual failed for %s: %s", room["id"], exc)
                break
        if not filled:
            logger.warning("fit_harness: no common-area room found; G4 may fail")
        debug_steps.append(_step_snapshot("4_fill_residual", plan))

        invariants = run_invariants(plan, boundary, scale_mm_per_unit=PLAN_MM_PER_UNIT)
        invariants["adjust_log"]  = adjust_log
        invariants["debug_steps"] = debug_steps
        return jsonify({"status": "ok", "fp": plan, "invariants": invariants})

    except Exception as exc:
        logger.exception("fit_harness failed")
        return jsonify({"error": str(exc)}), 500


# ── /swap_nodes endpoint ─────────────────────────────────────────────────────

@app.route("/swap_nodes", methods=["POST"])
def swap_nodes_route():
    """
    Swap two rooms' arrangement positions, graph-preserving where possible.

    Request body (JSON):
        fp           : plan dict with 'rooms' ([{id, type/label, polygon}]) and
                       'edges' ([{u, v}] room-id adjacency pairs)
        boundary     : [[x, y], ...] target boundary polygon (mm)
        room_a       : id of the first room to swap
        room_b       : id of the second room to swap
        entrance_mid : optional [x, y] entrance midpoint (mm)

    Response (JSON): the swap_nodes() result — {status, arrangement,
    adjacency_delta, guard} on success, or {status: "rejected", reason,
    adjacency_delta: null} when the rooms share no neighbour.
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    fp = body.get("fp")
    boundary = body.get("boundary")
    room_a = body.get("room_a")
    room_b = body.get("room_b")
    entrance_mid = body.get("entrance_mid")

    if not isinstance(fp, dict) or not fp.get("rooms"):
        return jsonify({"error": "missing or invalid 'fp' (must have rooms)"}), 400
    if not isinstance(boundary, list) or len(boundary) < 3:
        return jsonify({"error": "boundary must be a list of at least 3 [x, y] points"}), 400
    if not room_a or not room_b:
        return jsonify({"error": "room_a and room_b are required"}), 400

    room_ids = {r.get("id") for r in fp["rooms"]}
    if room_a not in room_ids or room_b not in room_ids:
        return jsonify({"error": "room_a and room_b must be ids in fp['rooms']"}), 400

    try:
        from packages.floorplan_core.swap import swap_nodes
        result = swap_nodes(fp, room_a, room_b, boundary, entrance_mid)
        return jsonify(result)
    except Exception as exc:
        logger.exception("swap_nodes failed")
        return jsonify({"error": str(exc)}), 500


# ── /topo_check endpoint ──────────────────────────────────────────────────────

@app.route("/topo_check", methods=["POST"])
def topo_check_route():
    """
    Run the topology-only sensor registry (topo_sensors.run_sensors) against a
    plan, re-checkable after edits (swap_nodes, ARAP steps, ...) since it
    recomputes room_edges fresh rather than trusting a stale retrieval-time copy.

    Request body (JSON):
        fp             : plan dict ('rooms', 'edges', 'entrance')
        boundary       : [[x, y], ...] target boundary polygon
        entrance_point : [x, y] target entrance point, same space as boundary
        requirements   : optional {bedrooms, bathrooms}
        typed_edges    : optional target boundary [{edge_index, type}] list
                         (defaulted the same way /retrieve does when absent)

    Response (JSON): run_sensors() result — {summary: {passed, total},
    checks: [{sensor, ok, rooms, detail}, ...]}
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    fp = body.get("fp")
    boundary = body.get("boundary")
    entrance_point = body.get("entrance_point")
    requirements = body.get("requirements")
    typed_edges = body.get("typed_edges")

    if not isinstance(fp, dict) or not fp.get("rooms"):
        return jsonify({"error": "missing or invalid 'fp' (must have rooms)"}), 400
    if not isinstance(boundary, list) or len(boundary) < 3:
        return jsonify({"error": "boundary must be a list of at least 3 [x, y] points"}), 400
    if not entrance_point:
        return jsonify({"error": "missing entrance_point"}), 400

    try:
        resolved_types = default_edge_types(boundary, entrance_point)
        for e in typed_edges or []:
            i = e.get("edge_index")
            if i is not None and 0 <= i < len(boundary) and e.get("type"):
                resolved_types[i] = e["type"]
        topo = topological_fitness(
            fp,
            target_boundary=boundary,
            target_edge_types=resolved_types,
            entrance_point=entrance_point,
            requirements=requirements,
        )
        result = run_sensors(fp, topo["room_edges"], requirements, target_boundary=boundary)
    except Exception as exc:
        logger.exception("topo_check failed")
        return jsonify({"error": str(exc)}), 500

    # room_edges + node_pos are returned too so a caller re-checking after an
    # edit can redraw the topology graph (nodes in-boundary + boundary connectors)
    # against the recomputed alignment.
    return jsonify({"status": "ok", "room_edges": topo["room_edges"],
                    "node_pos": topo.get("node_pos", {}), **result})


# ── /generate endpoint (Stage 0 — node-driven geometric solver) ───────────────

@app.route("/generate", methods=["POST"])
def generate_route():
    """
    Stage 0 of the agent harness (docs/agent-harness-build-plan.md): realise room
    geometry from the GRAPH — node positions + boundary — via the node-driven
    solver (packages.floorplan_core.generate), not from the source plan's bboxes.

    Node positions come from the topology layer (topological_fitness, the same
    placement the topology graph is drawn from), so the generated rooms sit on the
    same nodes the Observe/Act loop manipulates. Pass `swap: [id_a, id_b]` to edit
    the Spec's node assignment and re-derive geometry through the seam — the
    Stage-0 proof that the agent writes a Spec and the Generator realises it.

    Request body (JSON):
        fp             : plan dict ('rooms', 'edges', 'entrance')
        boundary       : [[x, y], ...] target boundary polygon
        entrance_point : [x, y] target entrance point, same space as boundary
        typed_edges    : optional target [{edge_index, type}] (defaulted like /retrieve)
        requirements   : optional {bedrooms, bathrooms}
        swap           : optional [id_a, id_b] — reassign two rooms' nodes
        mm_per_unit    : optional float — boundary-units -> mm (enables min-size);
                         defaults to the plan SCALE when omitted

    Response (JSON): {status, rooms, node_pos, room_edges, spec_nodes, feasible,
    report} — report.debug_steps carries the per-stage snapshot for the visual test.
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    fp = body.get("fp")
    boundary = body.get("boundary")
    entrance_point = body.get("entrance_point")
    typed_edges = body.get("typed_edges")
    requirements = body.get("requirements")
    swap = body.get("swap")
    mm_per_unit = body.get("mm_per_unit")

    if not isinstance(fp, dict) or not fp.get("rooms"):
        return jsonify({"error": "missing or invalid 'fp' (must have rooms)"}), 400
    if not isinstance(boundary, list) or len(boundary) < 3:
        return jsonify({"error": "boundary must be a list of at least 3 [x, y] points"}), 400
    if not entrance_point:
        return jsonify({"error": "missing entrance_point"}), 400

    try:
        from packages.floorplan_core.generate import (
            spec_from_plan, generate, reassign, SCALE as _PLAN_SCALE,
        )
        # The frontend does not send mm_per_unit; without it the solver drops every
        # min_edge to 0. Default to the plan scale (boundary units -> mm).
        if not mm_per_unit:
            mm_per_unit = float(_PLAN_SCALE)
        # Resolve boundary edge types the same way /retrieve and /topo_check do,
        # then compute the topology layer so node_pos (and room_edges for the
        # drawn connectors) match the placement the graph is drawn from.
        resolved_types = default_edge_types(boundary, entrance_point)
        for e in typed_edges or []:
            i = e.get("edge_index")
            if i is not None and 0 <= i < len(boundary) and e.get("type"):
                resolved_types[i] = e["type"]
        topo = topological_fitness(
            fp, target_boundary=boundary, target_edge_types=resolved_types,
            entrance_point=entrance_point, requirements=requirements,
        )
        fp_with_nodes = dict(fp)
        fp_with_nodes["node_pos"] = body.get("node_pos") or topo.get("node_pos", {})

        # Score against the DRAWN entrance (target frame), not the source plan's
        # entrance (source frame): the generated geometry sits in the drawn
        # boundary, so entry-room detection must use where the user put the door.
        drawn_entrance = ([entrance_point[0], entrance_point[1],
                           entrance_point[0], entrance_point[1]]
                          if entrance_point else fp.get("entrance"))
        spec = spec_from_plan(
            fp_with_nodes, boundary, edge_types=resolved_types,
            entrance=drawn_entrance, mm_per_unit=mm_per_unit,
        )
        # Thread the room↔boundary target (which target edge each room should front,
        # from the topo-rank ring alignment) into the Spec so the realiser honours it
        # (generate step 2b). Keyed by room id; survives a swap (demand follows
        # identity, which is what the swap edits the node of, not the id).
        boundary_demands = {}
        for rid, contacts in (topo.get("room_edges") or {}).items():
            idxs = sorted({c.get("target_edge_index") for c in contacts
                           if c.get("target_edge_index") is not None})
            if idxs:
                boundary_demands[str(rid)] = idxs
        spec["boundary_demands"] = boundary_demands
        if swap and len(swap) == 2:
            spec = reassign(spec, swap[0], swap[1])
        plan = generate(spec)
        # Re-derive room↔boundary from the GENERATED geometry. The solver re-tiles
        # the boundary, so the source's room_edges are stale (a room drawn off an
        # edge still carried the source contact — the LR→west phantom). Polygons are
        # in target space, so raycast directly, skipping the display transform.
        from packages.floorplan_core.topo_signature import _raycast_room_edges
        gen_room_edges = _raycast_room_edges(
            plan, boundary, entrance_point, mirror=False,
            target_edge_types=resolved_types, node_pos=plan.get("node_pos"),
            in_target_space=True,
        )
        # Stage-1: score the generated geometry with the sensor registry, using the
        # freshly re-derived room_edges so daylight/capacity judge the realised frontage.
        from packages.floorplan_core.topo_sensors import run_sensors
        sensors = run_sensors(plan, gen_room_edges,
                              requirements=requirements, target_boundary=boundary,
                              spec=spec)
    except Exception as exc:
        logger.exception("generate failed")
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "status": "ok",
        "rooms": plan["rooms"],
        "edges": plan.get("edges", []),
        "node_pos": plan.get("node_pos", {}),
        "room_edges": gen_room_edges,
        "spec_nodes": spec["node_assignment"],
        "feasible": plan["feasible"],
        "report": plan["report"],
        "sensors": sensors,
    })


# ── CLI / entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Floor plan retrieval + adaptation server"
    )
    parser.add_argument("--host",          default="127.0.0.1")
    parser.add_argument("--port",          type=int, default=5000)
    parser.add_argument("--data_dir",      default=str(RPLAN_DATASET_DIR),
                        help="Directory containing plan JSON files (default: full RPlan corpus, matches the canonical index)")
    parser.add_argument("--index_npz",     default=str(REPO_ROOT / "data" / "index" / "rplan_index_v5.npz"),
                        help="Path to the canonical .npz descriptor index (shape/entrance/room arrays, complex-valued shape)")
    parser.add_argument("--index_bin",     default=None,
                        help="Legacy flat-.bin descriptor matrix — overrides --index_npz if set (old 4-dim-entrance/real-shape format, not canonical)")
    parser.add_argument("--index_names",   default=str(REPO_ROOT / "data" / "index" / "browser" / "index_names.json"),
                        help="Path to index names JSON (only used with --index_bin)")
    parser.add_argument("--manifest",      default=str(REPO_ROOT / "data" / "index" / "manifest.json"),
                        help="Path to manifest JSON (auto-generated if missing)")
    parser.add_argument("--allow-remote-comfy", action="store_true",
                        help="Allow comfy_url values that are not localhost/127.0.0.1")
    parser.add_argument("--debug",         action="store_true")
    args = parser.parse_args()

    ALLOW_REMOTE_COMFY = bool(args.allow_remote_comfy)

    print("Floor plan server starting…")

    # ── load index ────────────────────────────────────────────────────────────
    try:
        if args.index_bin:
            INDEX_STORE = build_index_store(
                bin_path=args.index_bin,
                names_path=args.index_names,
                data_dir=args.data_dir,
                total_dim=TOTAL_DIM,
                shape_dim=SHAPE_DIM,
                ent_dim=ENT_DIM,
                logger=logger,
            )
        else:
            INDEX_STORE = build_index_store_from_npz(
                npz_path=args.index_npz,
                data_dir=args.data_dir,
                logger=logger,
            )
        n = len(INDEX_STORE.names)
        if INDEX_STORE.shapes is not None:
            print(f"  index:  {n} plans, dim {INDEX_STORE.dim}  "
                  f"[entrance-aware: shape:{SHAPE_DIM} + ent:{ENT_DIM} + rooms:{ROOM_DIM}]")
        else:
            print(f"  index:  {n} plans, dim {INDEX_STORE.dim}  [entrance component unavailable]")
    except FileNotFoundError as exc:
        print(f"  ERROR loading index: {exc}", file=sys.stderr)
        print("  Hint: use --index_bin and --index_names to specify paths.",
              file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"  ERROR loading index: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  plans:  {len(INDEX_STORE.plan_cache)//2 if INDEX_STORE.plan_cache else 0} JSONs loaded from {args.data_dir}/")

    # ── ensure manifest ───────────────────────────────────────────────────────
    ensure_manifest(args.manifest, args.data_dir, logger=logger)
    print(f"  manifest: {args.manifest}")

    print(f"  url:    http://{args.host}:{args.port}/")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)
