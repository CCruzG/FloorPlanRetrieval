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

from sem_geo_retrieval import combined_desc, boundary_v2_to_segs

# ── globals (populated at startup) ───────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.resolve()
db_descs    = None   # (N, dim) float32 numpy array
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

    boundary    = body.get("boundary")
    entrance    = body.get("entrance")
    requirements = body.get("requirements")
    topn        = int(body.get("topn",   5))
    shape_w     = float(body.get("shape_w", 1.0))
    sem_w       = float(body.get("sem_w",   0.5))
    # ── validate boundary ─────────────────────────────────────────────────────
    if not boundary or len(boundary) < 3:
        return jsonify({"error": "boundary must have at least 3 vertices"}), 400

    # ── Step 1: build query_fp (v2.1.0 format) ────────────────────────────────
    query_fp = {
        "schema_version": "2.1.0",
        "id":             "query",
        "source":         "user",
        "boundary":       boundary,
        "entrance":       entrance,
        "rooms":          [],
        "edges":          [],
        "walls":          None,
        "requirements":   requirements,
        "retrieval":      None,
    }

    # ── Step 2: build legacy format for descriptor computation ────────────────
    # boundary_v2_to_segs expects [[x,y,dir,is_door],…] and returns [[x*256,y*256],…]
    legacy_bdry = boundary_v2_to_segs(boundary)
    query_legacy = {
        "name":     "query",
        "boundary": legacy_bdry,
        "coord":    [],
        "labels":   [],
    }
    # ── Step 3: compute query descriptor ─────────────────────────────────────
    try:
        q_desc = combined_desc(query_legacy, shape_w, sem_w)
    except Exception as exc:
        logger.exception("Descriptor computation failed")
        return jsonify({"error": f"descriptor error: {exc}"}), 500

    # ── Step 4: retrieve top-N ────────────────────────────────────────────────
    try:
        dists = np.linalg.norm(db_descs - q_desc[np.newaxis], axis=1)
        idx   = np.argsort(dists)[:topn]
    except Exception as exc:
        logger.exception("Distance computation failed")
        return jsonify({"error": f"retrieval error: {exc}"}), 500

    # ── Step 5: collect candidate plans ───────────────────────────────────────
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

    # ── Step 6: attach retrieval distances and sort ───────────────────────────
    result_list = []
    for plan, db_i in zip(candidates, cand_idx):
        import copy
        p = copy.deepcopy(plan)
        p.setdefault("retrieval", {})["dist"] = float(dists[db_i])
        result_list.append(p)

    result_list.sort(
        key=lambda p: (p.get("retrieval") or {}).get("dist", float("inf"))
    )

    # ── Step 7: return response ───────────────────────────────────────────────
    return jsonify({
        "status":   "ok",
        "query_id": "query",
        "count":    len(result_list),
        "plans":    result_list,
    })


# ── startup helpers ───────────────────────────────────────────────────────────

def load_index(bin_path, names_path):
    """Load descriptor matrix and names list. Returns (descs, names) or raises."""
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
    return descs, names, dim


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
        db_descs, db_names, dim = load_index(args.index_bin, args.index_names)
        n = len(db_names)
        print(f"  index:  {n} plans, dim {dim}")
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
