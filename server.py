"""
server.py — Flask server connecting the browser floor plan tool to the
Python retrieval and adaptation pipeline.

Pipeline stage context:  retrieve → adapt → evaluate → present

Routes
------
GET  /                  Serve retrieval_tool.html
GET  /<path:filename>   Serve static files (index.bin, manifest.json, test_data/*.json, …)
POST /retrieve          Run retrieval + adaptation, return adapted plans as JSON
"""

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from geo_ent_num_retrieval import (
    shape_desc, entrance_desc_from_point, room_count_desc,
    retrieve as geo_retrieve,
    N_FREQS, ENT_DIM, ROOM_DIM,
)

SHAPE_DIM = N_FREQS                              # 48
TOTAL_DIM = SHAPE_DIM + ENT_DIM + ROOM_DIM      # 63

# ── globals (populated at startup) ───────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.resolve()
db_descs    = None   # (N, TOTAL_DIM) float32 — kept for legacy / dim detection
db_shapes   = None   # (N, SHAPE_DIM) float32
db_ents     = None   # (N, ENT_DIM)   float32
db_rooms    = None   # (N, ROOM_DIM)  float32
db_names    = None   # list[str]  e.g. ["0.png", "1.png", …]
plan_cache  = {}     # stem → v2.1.0 dict

app = Flask(__name__, static_folder=None)
logger = logging.getLogger(__name__)


# ── static routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "retrieval_tool.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve any file from the project directory.

    Falls back to browser_files/ for index.bin and index_names.json so that
    the browser auto-load works when those files live in that subdirectory.
    """
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
    global db_descs, db_names, plan_cache

    # ── index guard ───────────────────────────────────────────────────────────
    if db_descs is None or db_names is None:
        return jsonify({"error": "index not loaded"}), 503

    # ── parse request ──────────────────────────────────────────────────────────
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    boundary        = body.get("boundary")
    entrance_point  = body.get("entrance_point")   # [x, y] or None
    requirements    = body.get("requirements")
    topn            = int(body.get("topn",    5))
    shape_w         = float(body.get("shape_w",  1.0))
    ent_w           = float(body.get("ent_w",    2.0))
    room_w          = float(body.get("room_w",   1.0))
    # ── validate boundary ─────────────────────────────────────────────────────
    if not boundary or len(boundary) < 3:
        return jsonify({"error": "boundary must have at least 3 vertices"}), 400

    # boundary from browser: [[x, y, dir, is_door], …] — first two values are coords
    bdry_segs = [[v[0], v[1]] for v in boundary]

    # normalise to CW winding so descriptors are consistent regardless of draw order
    signed_area = sum(
        bdry_segs[i][0] * bdry_segs[(i+1) % len(bdry_segs)][1] -
        bdry_segs[(i+1) % len(bdry_segs)][0] * bdry_segs[i][1]
        for i in range(len(bdry_segs))
    )
    if signed_area > 0:   # CCW → reverse to CW
        bdry_segs = bdry_segs[::-1]

    # ── Step 1: compute query descriptors ────────────────────────────────────
    try:
        q_shape = shape_desc(bdry_segs)

        if entrance_point:
            q_ent = entrance_desc_from_point(entrance_point, bdry_segs)
        else:
            q_ent = np.zeros(ENT_DIM, dtype=np.float32)
            ent_w = 0.0   # ignore entrance component when none provided

        q_room = _req_to_room_desc(requirements)
    except Exception as exc:
        logger.exception("Descriptor computation failed")
        return jsonify({"error": f"descriptor error: {exc}"}), 500

    # ── Step 2: retrieve top-N ────────────────────────────────────────────────
    try:
        if db_shapes is not None:
            # New 63-dim index: rotation-aware entrance retrieval
            idx, dists, _, _, _, best_angles, best_mirror = geo_retrieve(
                q_shape, q_ent, q_room,
                db_shapes, db_ents, db_rooms,
                shape_w, ent_w, room_w, topn,
            )
        else:
            # Legacy flat index fallback
            q_desc = np.concatenate([q_shape, q_ent, q_room]).astype(np.float32)
            flat_dists = np.linalg.norm(db_descs - q_desc[np.newaxis], axis=1)
            idx   = np.argsort(flat_dists)[:topn]
            dists = flat_dists[idx]
            best_angles = np.zeros(len(idx))
            best_mirror = np.zeros(len(idx), dtype=bool)
    except Exception as exc:
        logger.exception("Distance computation failed")
        return jsonify({"error": f"retrieval error: {exc}"}), 500

    # ── Step 3: collect candidate plans ───────────────────────────────────────
    candidates = []
    cand_idx   = []   # parallel — which db index each candidate came from
    for i in idx:
        raw_name = db_names[i]
        stem = Path(raw_name).stem
        plan = plan_cache.get(stem) or plan_cache.get(raw_name)
        if plan is None:
            logger.warning("Plan not found in cache for db entry '%s'", raw_name)
            continue
        candidates.append(plan)
        cand_idx.append(int(i))

    if not candidates:
        return jsonify({
            "status":   "ok",
            "query_id": "query",
            "count":    0,
            "plans":    [],
        })

    # ── Step 4: attach retrieval distances and sort ───────────────────────────
    result_list = []
    for rank, (plan, db_i) in enumerate(zip(candidates, cand_idx)):
        p = copy.deepcopy(plan)
        if not isinstance(p.get("retrieval"), dict):
            p["retrieval"] = {}
        p["retrieval"]["dist"] = float(dists[rank])
        p["retrieval"]["source_id"] = db_names[db_i]
        p["retrieval"]["rot"]    = float(best_angles[rank])
        p["retrieval"]["mirror"] = bool(best_mirror[rank])
        result_list.append(p)

    result_list.sort(
        key=lambda p: (p.get("retrieval") or {}).get("dist", float("inf"))
    )

    # ── Step 5: return response ───────────────────────────────────────────────
    return jsonify({
        "status":   "ok",
        "query_id": "query",
        "count":    len(result_list),
        "plans":    result_list,
    })


# ── /adapt endpoint ───────────────────────────────────────────────────────────

@app.route("/adapt", methods=["POST"])
def adapt():
    """
    Fit a retrieved floor plan to a user-drawn target boundary using the
    full geometric fitting algorithm in fit_plan.py.

    Request body (JSON):
        source_plan : floor plan dict (v2.1.0)
        target      : {"boundary": [[x,y,dir,is_door],...], "entrance": [x, y]}

    Response (JSON):
        Fitted floor plan in schema v2.1.0 with a "fit_result" key.
    """
    from fit_plan import fit_plan as _fit_plan

    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    source_plan = body.get("source_plan")
    target      = body.get("target")

    if not source_plan:
        return jsonify({"error": "missing source_plan"}), 400
    if not target or not target.get("boundary"):
        return jsonify({"error": "missing target.boundary"}), 400

    try:
        result = _fit_plan(source_plan, target)
    except Exception as exc:
        logger.exception("fit_plan failed")
        return jsonify({"error": f"fit error: {exc}"}), 500

    return jsonify(result)


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
    from fit_rooms import fit_plan

    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON body"}), 400

    fp       = body.get("fp")
    boundary = body.get("boundary")

    if not fp or not boundary or len(boundary) < 3:
        return jsonify({"error": "missing or invalid fp/boundary"}), 400

    # Accept [x,y,dir,is_door] format — only first two values matter
    bdry_xy = [[v[0], v[1]] for v in boundary]

    try:
        fitted_fp = fit_plan(fp, bdry_xy)
    except Exception as exc:
        logger.exception("fit_plan failed")
        return jsonify({"error": f"fit error: {exc}"}), 500

    stats = fitted_fp.pop("_fit_stats", {})
    return jsonify({"status": "ok", "fp": fitted_fp, "stats": stats})


# ── query helpers ─────────────────────────────────────────────────────────────

def _req_to_room_desc(requirements):
    """
    Synthesise an 11-dim room-count descriptor from user requirements dict.

    Uses the same ROOM_TYPES order as geo_ent_num_retrieval:
      [0]Living [1]Master [2]Kitchen [3]Bathroom [4]Dining
      [5]Child  [6]Study  [7]Second  [8]Balcony  [9]Storage [10]total
    Maximums: 2, 1, 1, 3, 1, 2, 1, 2, 2, 2, 15
    """
    if not requirements:
        requirements = {}

    bedrooms  = int(requirements.get('bedrooms',  2))
    bathrooms = int(requirements.get('bathrooms', 1))

    living   = 1
    master   = 1 if bedrooms > 0 else 0
    kitchen  = 1
    dining   = 1
    child    = max(0, bedrooms - 1)   # rooms beyond master
    total    = living + master + kitchen + dining + child + bathrooms

    feat = np.array([
        min(living  / 2,  1.0),   # 0 Living
        min(master  / 1,  1.0),   # 1 Master
        min(kitchen / 1,  1.0),   # 2 Kitchen
        min(bathrooms/3,  1.0),   # 3 Bathroom
        min(dining  / 1,  1.0),   # 4 Dining
        min(child   / 2,  1.0),   # 5 Child
        0.0,                      # 6 Study
        0.0,                      # 7 Second
        0.0,                      # 8 Balcony
        0.0,                      # 9 Storage
        min(total   / 15, 1.0),   # 10 total
    ], dtype=np.float32)
    return feat


# ── startup helpers ───────────────────────────────────────────────────────────

def load_index(bin_path, names_path):
    """
    Load descriptor matrix and names list.
    Returns (descs, names, dim, shapes, ents, rooms).
    shapes/ents/rooms are None when the index is not the new 63-dim format.
    """
    with open(names_path) as fh:
        names = json.load(fh)
    n = len(names)
    raw = np.fromfile(bin_path, dtype=np.float32)
    if n == 0 or raw.size % n != 0:
        raise ValueError(
            f"index size mismatch: {raw.size} floats, {n} names"
            f" (remainder {raw.size % n})"
        )
    dim = raw.size // n
    descs = raw.reshape(n, dim)

    # Split into components only when the new 63-dim format is detected
    if dim == TOTAL_DIM:
        shapes = descs[:, :SHAPE_DIM]
        ents   = descs[:, SHAPE_DIM:SHAPE_DIM + ENT_DIM]
        rooms  = descs[:, SHAPE_DIM + ENT_DIM:]
    else:
        shapes = ents = rooms = None

    return descs, names, dim, shapes, ents, rooms


def load_plans(data_dir):
    """Load all *.json plan files from data_dir into a dict keyed by stem."""
    cache = {}
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob("*.json"))
    for p in paths:
        try:
            with open(p) as fh:
                data = json.load(fh)
            stem = p.stem
            cache[stem] = data
            if data.get("id"):
                cache[data["id"]] = data
        except Exception as exc:
            logger.warning("Skipping %s: %s", p, exc)
    return cache


def ensure_manifest(manifest_path, data_dir):
    """Return manifest list; generate and save if file is absent."""
    mp = Path(manifest_path)
    if mp.exists():
        with open(mp) as fh:
            return json.load(fh)
    logger.warning(
        "manifest.json not found at '%s' — generating from %s/", manifest_path, data_dir
    )
    names = sorted(p.name for p in Path(data_dir).glob("*.json"))
    with open(mp, "w") as fh:
        json.dump(names, fh, indent=2)
    return names


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
    parser.add_argument("--debug",         action="store_true")
    args = parser.parse_args()

    print("Floor plan server starting…")

    # ── load index ────────────────────────────────────────────────────────────
    try:
        db_descs, db_names, dim, db_shapes, db_ents, db_rooms = load_index(
            args.index_bin, args.index_names)
        n = len(db_names)
        if db_shapes is not None:
            print(f"  index:  {n} plans, dim {dim}  "
                  f"[entrance-aware: shape:{SHAPE_DIM} + ent:{ENT_DIM} + rooms:{ROOM_DIM}]")
        else:
            print(f"  index:  {n} plans, dim {dim}  [legacy format — entrance ignored]")
    except FileNotFoundError as exc:
        print(f"  ERROR loading index: {exc}", file=sys.stderr)
        print("  Hint: use --index_bin and --index_names to specify paths.",
              file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"  ERROR loading index: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── load plan cache ───────────────────────────────────────────────────────
    plan_cache = load_plans(args.data_dir)
    m = len({k for k in plan_cache if not k.startswith("r_")
             and "/" not in k})
    print(f"  plans:  {len(plan_cache)//2 if plan_cache else 0} JSONs loaded from {args.data_dir}/")

    # ── ensure manifest ───────────────────────────────────────────────────────
    ensure_manifest(args.manifest, args.data_dir)
    print(f"  manifest: {args.manifest}")

    print(f"  url:    http://{args.host}:{args.port}/")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)
