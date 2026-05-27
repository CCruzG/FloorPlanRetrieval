"""
server.py — Flask server connecting the browser floor plan tool to the
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

from geo_ent_num_retrieval import (
    N_FREQS, N_SAMP, ENT_DIM, ROOM_DIM,
)
from fitting import FittingError, fit_rooms_in_boundary
from index_store import build_index_store, ensure_manifest
from request_parsing import (
    RequestValidationError,
    parse_comfy_health_request,
    parse_fit_request,
    parse_render_request,
    parse_retrieve_request,
)
from render_backend import DEFAULT_COMFY_WORKFLOW_FILENAME, check_comfy_health, render
from retrieval import RetrievalError, run_retrieval

SHAPE_DIM = N_FREQS                              # 48
TOTAL_DIM = SHAPE_DIM + ENT_DIM + ROOM_DIM      # 63

# ── globals (populated at startup) ───────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
INDEX_STORE = None
ALLOWED_STATIC_EXTS = {".html", ".js", ".json", ".bin", ".png", ".css"}
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
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve any file from the project directory.

    Falls back to browser_files/ for index.bin and index_names.json so that
    the browser auto-load works when those files live in that subdirectory.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_STATIC_EXTS:
        return jsonify({"error": f"disallowed file type: {suffix or '(none)'}"}), 404

    full = BASE_DIR / filename
    if full.exists():
        return send_from_directory(BASE_DIR, filename)

    # Fallback: try browser_files/
    alt = BASE_DIR / "browser_files" / Path(filename).name
    if alt.exists():
        return send_from_directory(BASE_DIR / "browser_files", Path(filename).name)

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
            base_dir=BASE_DIR,
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
    parser.add_argument("--data_dir",      default="test_data",
                        help="Directory containing plan JSON files")
    parser.add_argument("--index_bin",     default="browser_files/index.bin",
                        help="Path to precomputed descriptor matrix (.bin)")
    parser.add_argument("--index_names",   default="browser_files/index_names.json",
                        help="Path to index names JSON")
    parser.add_argument("--manifest",      default="manifest.json",
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
            print(f"  index:  {n} plans, dim {INDEX_STORE.dim}  [legacy format - entrance ignored]")
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
