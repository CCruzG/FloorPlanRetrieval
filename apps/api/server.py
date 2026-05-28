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
POST /render            Run the selected rendering backend and return an image
POST /comfy_health      Check ComfyUI connectivity
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
from packages.floorplan_core.fitting import FittingError, fit_rooms_in_boundary
from packages.floorplan_core.index_store import build_index_store, ensure_manifest
from packages.floorplan_core.request_parsing import (
    RequestValidationError,
    parse_comfy_health_request,
    parse_fit_request,
    parse_render_request,
    parse_retrieve_request,
)
from packages.floorplan_core.render_backend import DEFAULT_COMFY_WORKFLOW_FILENAME, check_comfy_health, render
from packages.floorplan_core.retrieval import RetrievalError, run_retrieval

SHAPE_DIM = N_FREQS                              # 48
TOTAL_DIM = SHAPE_DIM + ENT_DIM + ROOM_DIM      # 63

# ── globals (populated at startup) ───────────────────────────────────────────
WEB_DIR = REPO_ROOT / "apps" / "web"
WEB_UTILITY_DIR = WEB_DIR / "legacy"
DATA_INDEX_DIR = REPO_ROOT / "data" / "index" / "browser"
DATA_MANIFEST_FILE = REPO_ROOT / "data" / "index" / "manifest.json"
CONFIG_WORKFLOWS_DIR = REPO_ROOT / "configs" / "workflows"
ROOM_TYPES_FILE = REPO_ROOT / "configs" / "room_types.json"
DATA_FIXTURES_TEST_DIR = REPO_ROOT / "data" / "fixtures" / "plans" / "test_data"
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
    parser.add_argument("--data_dir",      default="data/fixtures/plans/test_data",
                        help="Directory containing plan JSON files")
    parser.add_argument("--index_bin",     default="data/index/browser/index.bin",
                        help="Path to precomputed descriptor matrix (.bin)")
    parser.add_argument("--index_names",   default="data/index/browser/index_names.json",
                        help="Path to index names JSON")
    parser.add_argument("--manifest",      default="data/index/manifest.json",
                        help="Path to manifest JSON (auto-generated if missing)")
    parser.add_argument("--allow-remote-comfy", action="store_true",
                        help="Allow comfy_url values that are not localhost/127.0.0.1")
    parser.add_argument("--debug",         action="store_true")
    args = parser.parse_args()

    ALLOW_REMOTE_COMFY = bool(args.allow_remote_comfy)

    print("Floor plan server starting…")

    # ── load index ────────────────────────────────────────────────────────────
    try:
        INDEX_STORE = build_index_store(
            bin_path=args.index_bin,
            names_path=args.index_names,
            data_dir=args.data_dir,
            total_dim=TOTAL_DIM,
            shape_dim=SHAPE_DIM,
            ent_dim=ENT_DIM,
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
