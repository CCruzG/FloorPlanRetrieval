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
import base64
import copy
import gc
import importlib
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from io import BytesIO
from urllib import parse as urlparse
from urllib import request as urlrequest
from urllib import error as urlerror

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
py_pipe_cache = {}   # (base_model, controlnet_model, dtype) -> (pipe, device)

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

    backend = str(body.get("backend", "openai")).strip().lower()
    prompt = str(body.get("prompt", "")).strip()
    depth_data_url = body.get("depth_data_url")
    building_mask_data_url = body.get("building_mask_data_url")
    normal_data_url = body.get("normal_data_url")
    silhouette_data_url = body.get("silhouette_data_url")
    opacity_mask_data_url = body.get("opacity_mask_data_url")
    depth_invert_raw = body.get("depth_invert", False)
    if isinstance(depth_invert_raw, str):
        depth_invert = depth_invert_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        depth_invert = bool(depth_invert_raw)

    if not prompt:
        return jsonify({"error": "missing prompt"}), 400
    if not depth_data_url or not isinstance(depth_data_url, str):
        return jsonify({"error": "missing depth_data_url"}), 400

    if backend in {"comfy", "comfyui", "local"}:
        comfy_url = str(body.get("comfy_url") or "http://127.0.0.1:8188").rstrip("/")
        workflow_override = body.get("workflow_override")
        try:
            image_data_url, meta = _render_with_comfy_local(
                comfy_url=comfy_url,
                prompt=prompt,
                depth_data_url=depth_data_url,
                building_mask_data_url=building_mask_data_url,
                normal_data_url=normal_data_url,
                silhouette_data_url=silhouette_data_url,
                opacity_mask_data_url=opacity_mask_data_url,
                depth_invert=depth_invert,
                workflow_override=workflow_override,
            )
        except Exception as exc:
            logger.exception("ComfyUI render failed")
            return jsonify({"error": f"comfy render failed: {exc}"}), 500

        return jsonify({
            "status": "ok",
            "backend": "comfy",
            "image_data_url": image_data_url,
            "meta": meta,
        })

    if backend != "openai":
        return jsonify({"error": f"unsupported backend: {backend}. supported: 'openai', 'comfy'"}), 400

    try:
        image_data_url, meta = _render_with_openai(
            prompt=prompt,
            depth_data_url=depth_data_url,
            openai_api_key=body.get("openai_api_key"),
            model=str(body.get("openai_model") or "gpt-image-1"),
            quality=str(body.get("openai_quality") or "medium"),
            width=int(body.get("openai_width") or body.get("width") or 1024),
            height=int(body.get("openai_height") or body.get("height") or 1024),
        )
    except Exception as exc:
        logger.exception("OpenAI render failed")
        return jsonify({"error": f"openai render failed: {exc}"}), 500

    return jsonify({
        "status": "ok",
        "backend": "openai",
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

    comfy_url = str(body.get("comfy_url") or "http://127.0.0.1:8188").rstrip("/")
    try:
        _, payload = _http_get_json(f"{comfy_url}/system_stats", timeout=10)
    except Exception as exc:
        return jsonify({"error": f"could not reach ComfyUI at {comfy_url}: {exc}"}), 502

    devices = payload.get("devices") if isinstance(payload, dict) else None
    n_devices = len(devices) if isinstance(devices, list) else 0
    summary = f"ComfyUI reachable at {comfy_url} ({n_devices} device(s))"
    return jsonify({"status": "ok", "summary": summary})


def _decode_data_url_png(data_url):
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise ValueError("depth_data_url must be a PNG data URL")
    raw = data_url[len(prefix):]
    return base64.b64decode(raw)


def _build_cuda_candidate_ids(torch, requested_cuda_device=None):
    if not torch.cuda.is_available():
        return []

    n = torch.cuda.device_count()
    ids = list(range(n))

    if requested_cuda_device is not None:
        try:
            rid = int(requested_cuda_device)
            if 0 <= rid < n:
                return [rid]
        except Exception:
            pass

    # Auto mode: prefer device with most currently free VRAM.
    scored = []
    for idx in ids:
        try:
            free_b, _ = torch.cuda.mem_get_info(idx)
        except Exception:
            free_b = 0
        scored.append((free_b, idx))
    scored.sort(reverse=True)
    return [idx for _, idx in scored]


def _normalize_requested_cuda_device(requested_cuda_device):
    if requested_cuda_device is None:
        return None
    if isinstance(requested_cuda_device, str):
        s = requested_cuda_device.strip().lower()
        if s in {"", "none", "null", "auto"}:
            return None
    try:
        return int(requested_cuda_device)
    except Exception:
        return None


def _build_balanced_max_memory_map(torch, reserve_gib=1.5):
    max_memory = {}
    try:
        n = int(torch.cuda.device_count())
    except Exception:
        n = 0
    for idx in range(n):
        try:
            free_b, _ = torch.cuda.mem_get_info(idx)
            free_gib = max(1, int((free_b / float(1024 ** 3)) - float(reserve_gib)))
        except Exception:
            free_gib = 10
        max_memory[idx] = f"{free_gib}GiB"
    return max_memory


def _detect_controlnet_architecture(base_model_id, controlnet_model_id):
    joined = f"{base_model_id} {controlnet_model_id}".lower()
    return "flux" if "flux" in joined else "sdxl"


def _exception_chain_messages(exc):
    msgs = []
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msgs.append(str(cur))
        cur = cur.__cause__ or cur.__context__
    return "\n".join(msgs).lower()


def _is_cuda_oom_error(exc):
    msg = _exception_chain_messages(exc)
    return ("cuda" in msg) and ("out of memory" in msg or "oom" in msg)


def _get_python_controlnet_pipeline(
    base_model_id,
    controlnet_model_id,
    dtype_name="float16",
    requested_cuda_device=None,
):
    requested_cuda_device = _normalize_requested_cuda_device(requested_cuda_device)
    key = (base_model_id, controlnet_model_id, dtype_name, requested_cuda_device)
    if key in py_pipe_cache:
        return py_pipe_cache[key]

    try:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
    except Exception as exc:
        raise RuntimeError(
            "missing python backend deps. install: pip install torch diffusers transformers accelerate safetensors pillow"
        ) from exc

    arch = _detect_controlnet_architecture(base_model_id, controlnet_model_id)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_name, torch.float16)

    if arch == "flux":
        FluxControlNetModel = getattr(diffusers, "FluxControlNetModel", None)
        FluxControlNetPipeline = getattr(diffusers, "FluxControlNetPipeline", None)
        if FluxControlNetModel is None or FluxControlNetPipeline is None:
            raise RuntimeError(
                "installed diffusers does not provide FluxControlNetPipeline; please upgrade diffusers"
            )

        # Be tolerant to stale or invalid FLUX ControlNet repo IDs.
        fallback_ids = [
            controlnet_model_id,
            "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
            "jasperai/Flux.1-dev-Controlnet-Depth",
            "XLabs-AI/flux-controlnet-depth-diffusers",
        ]
        unique_ids = []
        for mid in fallback_ids:
            if mid and mid not in unique_ids:
                unique_ids.append(mid)

        controlnet = None
        last_exc = None
        resolved_controlnet_id = controlnet_model_id
        for mid in unique_ids:
            try:
                controlnet = FluxControlNetModel.from_pretrained(mid, torch_dtype=torch_dtype)
                resolved_controlnet_id = mid
                break
            except Exception as exc:
                last_exc = exc

        if controlnet is None:
            raise RuntimeError(
                "could not load FLUX depth ControlNet. Tried: " + ", ".join(unique_ids)
            ) from last_exc

        base_fallback_ids = [
            base_model_id,
            "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-schnell",
            "camenduru/FLUX.1-dev-diffusers",
            "ChuckMcSneed/FLUX.1-dev",
        ]
        unique_base_ids = []
        for mid in base_fallback_ids:
            if mid and mid not in unique_base_ids:
                unique_base_ids.append(mid)

        pipe = None
        last_base_exc = None
        resolved_base_model_id = base_model_id
        loaded_with_balanced_map = False
        for mid in unique_base_ids:
            try:
                use_balanced_map = bool(
                    torch.cuda.is_available()
                    and requested_cuda_device is None
                    and int(torch.cuda.device_count()) > 1
                )
                load_kwargs = {
                    "controlnet": controlnet,
                    "torch_dtype": torch_dtype,
                }
                if use_balanced_map:
                    load_kwargs["device_map"] = "balanced"
                    load_kwargs["max_memory"] = _build_balanced_max_memory_map(torch)
                pipe = FluxControlNetPipeline.from_pretrained(
                    mid,
                    **load_kwargs,
                )
                resolved_base_model_id = mid
                loaded_with_balanced_map = use_balanced_map
                break
            except Exception as exc:
                last_base_exc = exc

        if pipe is None:
            raise RuntimeError(
                "could not load FLUX base pipeline. Tried: "
                + ", ".join(unique_base_ids)
                + ". If you want BFL official models, run `hf auth login` and request access on Hugging Face."
            ) from last_base_exc

        base_model_id = resolved_base_model_id
        controlnet_model_id = resolved_controlnet_id
        if loaded_with_balanced_map:
            logger.info("Loaded FLUX pipeline with balanced multi-GPU dispatch")
    else:
        ControlNetModel = getattr(diffusers, "ControlNetModel")
        StableDiffusionXLControlNetPipeline = getattr(diffusers, "StableDiffusionXLControlNetPipeline")
        controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch_dtype)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )

    if torch.cuda.is_available():
        cuda_candidates = _build_cuda_candidate_ids(torch, requested_cuda_device=requested_cuda_device)
        device = None
        last_cuda_exc = None

        # If already dispatched by Accelerate device_map, don't move modules again.
        hf_map = getattr(pipe, "hf_device_map", None)
        if isinstance(hf_map, dict) and hf_map:
            gpu_ids = set()
            for v in hf_map.values():
                if isinstance(v, int):
                    gpu_ids.add(v)
                elif isinstance(v, str) and v.startswith("cuda"):
                    try:
                        gpu_ids.add(int(v.split(":", 1)[1]))
                    except Exception:
                        gpu_ids.add(0)
            if gpu_ids:
                if len(gpu_ids) > 1:
                    device = "cuda:balanced:" + ",".join(str(i) for i in sorted(gpu_ids))
                else:
                    device = f"cuda:{sorted(gpu_ids)[0]}"

                # In some FLUX+ControlNet setups, balanced dispatch can leave
                # controlnet on CPU while hidden states are on CUDA, causing
                # matmul device mismatch. Pin controlnet to transformer device.
                if arch == "flux" and hasattr(pipe, "controlnet") and hasattr(pipe, "transformer"):
                    try:
                        exec_dev = getattr(pipe, "_execution_device", None)
                        exec_dev = str(exec_dev) if exec_dev is not None else ""
                        tr_dev = str(next(pipe.transformer.parameters()).device)
                        target_dev = None

                        # Prefer the pipeline execution device when available.
                        if exec_dev.startswith("cuda"):
                            target_dev = exec_dev

                        # If transformer is already concretely on CUDA, use it.
                        if target_dev is None and tr_dev.startswith("cuda"):
                            target_dev = tr_dev

                        # As a stable fallback in balanced mode, prefer the
                        # first CUDA device rather than the highest index.
                        if target_dev is None and gpu_ids:
                            target_dev = f"cuda:{sorted(gpu_ids)[0]}"

                        if target_dev is not None:
                            pipe.controlnet = pipe.controlnet.to(target_dev)
                            logger.info("Pinned FLUX controlnet device to %s", target_dev)
                    except Exception:
                        pass

        # FLUX pipelines usually sit right at VRAM limits on 16GB cards when fully
        # moved to CUDA. Prefer CPU offload first to preserve runtime headroom.
        if device is None and arch == "flux" and hasattr(pipe, "enable_model_cpu_offload") and cuda_candidates:
            offload_idx = cuda_candidates[0]
            try:
                pipe.enable_model_cpu_offload(gpu_id=offload_idx)
                device = f"cpu_offload:cuda:{offload_idx}"
            except Exception as exc:
                last_cuda_exc = exc
                device = None

        if device is None:
            for idx in cuda_candidates:
                try:
                    device = f"cuda:{idx}"
                    pipe = pipe.to(device)
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                    except Exception:
                        pass
                    break
                except torch.OutOfMemoryError as exc:
                    last_cuda_exc = exc
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    device = None

        if device is None:
            # If full placement fails, try CPU offload for large pipelines (e.g. FLUX).
            if hasattr(pipe, "enable_model_cpu_offload") and cuda_candidates:
                offload_idx = cuda_candidates[0]
                try:
                    pipe.enable_model_cpu_offload(gpu_id=offload_idx)
                    device = f"cpu_offload:cuda:{offload_idx}"
                except Exception:
                    device = None

        # Extra memory guards; noop if unsupported by the selected pipeline.
        try:
            pipe.enable_attention_slicing("max")
        except Exception:
            pass
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass

        if device is None:
            if last_cuda_exc is not None:
                raise RuntimeError(
                    "insufficient VRAM on available GPUs; tried GPUs "
                    + ", ".join(str(i) for i in cuda_candidates)
                ) from last_cuda_exc
            device = "cpu"
            pipe = pipe.to(device)
    else:
        device = "cpu"
        pipe = pipe.to(device)

    py_pipe_cache[key] = (pipe, device, arch)
    return pipe, device, arch


def _render_with_python_controlnet(
    prompt,
    depth_data_url,
    building_mask_data_url=None,
    depth_invert=False,
    negative_prompt="",
    width=1024,
    height=1024,
    num_inference_steps=28,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,
    base_model_id="black-forest-labs/FLUX.1-dev",
    controlnet_model_id="Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
    seed=None,
    requested_cuda_device=None,
    flux_max_sequence_length=128,
):
    try:
        torch = importlib.import_module("torch")
        Image = importlib.import_module("PIL.Image")
    except Exception as exc:
        raise RuntimeError(
            "missing python backend deps. install: pip install torch diffusers transformers accelerate safetensors pillow"
        ) from exc

    tgt_w = max(64, int(width))
    tgt_h = max(64, int(height))
    depth_png = _decode_data_url_png(depth_data_url)
    depth_img_l = Image.open(BytesIO(depth_png)).convert("L").resize((tgt_w, tgt_h))
    depth_np = np.array(depth_img_l, dtype=np.float32)

    mask_applied = False
    depth_background_mode = "none"
    depth_background_value = None
    if building_mask_data_url and isinstance(building_mask_data_url, str):
        try:
            mask_png = _decode_data_url_png(building_mask_data_url)
            mask_img_l = Image.open(BytesIO(mask_png)).convert("L").resize((tgt_w, tgt_h))
            mask_np = np.array(mask_img_l, dtype=np.float32) > 20.0

            if np.any(mask_np):
                vals = depth_np[mask_np]
                lo = float(np.percentile(vals, 2.0))
                hi = float(np.percentile(vals, 98.0))
                if hi <= lo:
                    lo = float(np.min(vals))
                    hi = float(np.max(vals) + 1.0)

                norm = np.clip((depth_np - lo) / max(1e-6, hi - lo), 0.0, 1.0)
                gamma = 0.75  # expands depth contrast for near/far separation
                norm = np.power(norm, gamma)

                # Keep out-of-mask area neutral so ControlNet focuses on volume geometry
                # instead of interpreting the whole background as a very near/far plane.
                neutral = float(np.median(norm[mask_np]))
                neutral = float(np.clip(neutral, 0.35, 0.65))
                depth_np = np.where(mask_np, norm * 255.0, neutral * 255.0)

                depth_background_mode = "neutral_median"
                depth_background_value = neutral
                mask_applied = True
            else:
                depth_np[:] = 127.5
                depth_background_mode = "neutral_constant"
                depth_background_value = 0.5
        except Exception:
            # Fallback safely if mask decode fails.
            pass

    if not mask_applied:
        lo = float(np.percentile(depth_np, 2.0))
        hi = float(np.percentile(depth_np, 98.0))
        if hi <= lo:
            lo = float(np.min(depth_np))
            hi = float(np.max(depth_np) + 1.0)
        norm = np.clip((depth_np - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        gamma = 0.75
        depth_np = np.power(norm, gamma) * 255.0

    if depth_invert:
        depth_np = 255.0 - depth_np

    depth_img = Image.fromarray(depth_np.astype(np.uint8), mode="L").convert("RGB")

    pipe, device, arch = _get_python_controlnet_pipeline(
        base_model_id=base_model_id,
        controlnet_model_id=controlnet_model_id,
        dtype_name="float16" if torch.cuda.is_available() else "float32",
        requested_cuda_device=requested_cuda_device,
    )

    def _snap64(v):
        return max(64, int(v) // 64 * 64)

    def _run_pipe(run_w, run_h, run_steps, run_max_seq_len):
        img_cond = depth_img.resize((run_w, run_h))
        common_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "num_inference_steps": max(1, int(run_steps)),
            "guidance_scale": float(guidance_scale),
            "controlnet_conditioning_scale": float(controlnet_conditioning_scale),
            "generator": generator,
            "width": run_w,
            "height": run_h,
        }
        if arch == "flux":
            common_kwargs["max_sequence_length"] = int(max(64, run_max_seq_len))
            return pipe(control_image=img_cond, **common_kwargs)
        return pipe(image=img_cond, **common_kwargs)

    generator = None
    use_seed = None
    if seed is not None:
        try:
            use_seed = int(seed)
            gen_device = device if isinstance(device, str) and device.startswith("cuda") else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(use_seed)
        except Exception:
            use_seed = None
            generator = None

    run_w = tgt_w
    run_h = tgt_h
    run_steps = max(1, int(num_inference_steps))
    run_max_seq_len = max(64, int(flux_max_sequence_length))
    oom_fallback_used = False

    try:
        out = _run_pipe(run_w, run_h, run_steps, run_max_seq_len)
    except torch.OutOfMemoryError:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

        run_w = _snap64(max(256, int(tgt_w * 0.625)))
        run_h = _snap64(max(256, int(tgt_h * 0.625)))
        run_steps = max(8, int(run_steps * 0.60))
        run_max_seq_len = max(96, min(run_max_seq_len, 128))
        oom_fallback_used = True

        try:
            out = _run_pipe(run_w, run_h, run_steps, run_max_seq_len)
        except torch.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA OOM during python_controlnet render. "
                f"Tried requested {tgt_w}x{tgt_h}/{num_inference_steps}/max_seq={flux_max_sequence_length} "
                f"and fallback {run_w}x{run_h}/{run_steps}/max_seq={run_max_seq_len}."
            ) from exc
        logger.warning(
            "python_controlnet OOM recovered via fallback from %sx%s/%s to %sx%s/%s",
            tgt_w,
            tgt_h,
            int(num_inference_steps),
            run_w,
            run_h,
            run_steps,
        )
    img = out.images[0]

    buf = BytesIO()
    img.save(buf, format="PNG")
    out_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    out_data_url = f"data:image/png;base64,{out_b64}"

    meta = {
        "device": device,
        "base_model_id": base_model_id,
        "controlnet_model_id": controlnet_model_id,
        "architecture": arch,
        "width": int(width),
        "height": int(height),
        "steps": int(num_inference_steps),
        "effective_width": int(run_w),
        "effective_height": int(run_h),
        "effective_steps": int(run_steps),
        "flux_max_sequence_length": int(run_max_seq_len),
        "oom_fallback_used": bool(oom_fallback_used),
        "guidance_scale": float(guidance_scale),
        "controlnet_conditioning_scale": float(controlnet_conditioning_scale),
        "seed": use_seed,
        "mask_applied": bool(mask_applied),
        "depth_invert": bool(depth_invert),
        "depth_background_mode": depth_background_mode,
        "depth_background_value": depth_background_value,
    }
    return out_data_url, meta


def _load_comfy_workflow(workflow_override):
    if isinstance(workflow_override, dict) and workflow_override:
        return copy.deepcopy(workflow_override)

    wf_path = BASE_DIR / "comfyui-sdxl-depth-multi-workflow.json"
    if not wf_path.exists():
        raise FileNotFoundError(
            f"workflow file not found: {wf_path.name}. "
            "Add it to the repo root or send workflow JSON in request body."
        )
    with open(wf_path) as fh:
        return json.load(fh)


def _inject_prompt_and_depth(
    workflow,
    prompt,
    uploaded_filename,
    uploaded_mask_filename=None,
    uploaded_normal_filename=None,
    uploaded_silhouette_filename=None,
    uploaded_opacity_filename=None,
):
    """Best-effort workflow injection for ComfyUI workflow payloads."""
    if not isinstance(workflow, dict):
        raise ValueError("workflow must be a dict")

    # Accept wrapped payloads that place API nodes under common wrapper keys.
    if isinstance(workflow.get("prompt"), dict):
        workflow_nodes = workflow["prompt"]
    elif isinstance(workflow.get("workflow"), dict):
        workflow_nodes = workflow["workflow"]
    else:
        workflow_nodes = workflow

    if not isinstance(workflow_nodes, dict):
        raise ValueError("workflow prompt must be a dict keyed by node id")

    # UI-export workflows are not directly executable by /prompt.
    if isinstance(workflow_nodes.get("nodes"), list):
        raise ValueError(
            "workflow override appears to be ComfyUI UI export format. "
            "Please export API format (Save as API format) and upload that JSON."
        )

    node_types = []
    for _, node in workflow_nodes.items():
        if isinstance(node, dict):
            t = node.get("class_type") or node.get("type")
            if t:
                node_types.append(str(t).strip().lower())

    has_sampler = any(("ksampler" in t) or ("sampler" in t) for t in node_types)
    has_modelish = any(
        ("checkpointloader" in t)
        or ("unetloader" in t)
        or ("cliploader" in t)
        or ("dualcliploader" in t)
        or ("diffusion" in t)
        or ("model" in t)
        for t in node_types
    )
    has_decode = any(("decode" in t) or ("vaedecode" in t) for t in node_types)
    has_controlnet_apply = any(("controlnetapply" in t) or ("advanced" in t and "controlnet" in t) for t in node_types)
    has_prompt_encode = any(("cliptextencode" in t) or ("textencode" in t) for t in node_types)
    has_latent = any(("latent" in t) for t in node_types)
    has_output_save = any(("saveimage" in t) for t in node_types)
    has_image_io_only = all(
        (
            ("loadimage" in t)
            or ("saveimage" in t)
            or ("previewimage" in t)
            or ("imagescale" in t)
        )
        for t in node_types
    ) if node_types else False

    generative_signals = (
        has_sampler
        or (has_modelish and has_decode)
        or has_controlnet_apply
        or (has_prompt_encode and has_modelish)
        or (has_latent and has_output_save)
    )
    likely_passthrough = (not generative_signals) and has_image_io_only
    if likely_passthrough:
        preview = ", ".join(sorted(set(node_types))[:12]) if node_types else "none"
        raise ValueError(
            "workflow appears to be non-generative (likely depth passthrough). "
            "Expected a full generation graph with sampler/model nodes, not only image load/save. "
            f"Detected node types: {preview}"
        )

    prompt_set = False
    image_set = False
    mask_set = False

    # 1) Put prompt text into CLIPTextEncode-like nodes (prefer "positive" titled node if present)
    ordered_nodes = list(workflow_nodes.items())
    for _, node in ordered_nodes:
        if not isinstance(node, dict):
            continue
        cls = node.get("class_type") or node.get("type")
        if cls == "CLIPTextEncode":
            inputs = node.setdefault("inputs", {})
            if "text" in inputs and isinstance(inputs["text"], str):
                if not prompt_set:
                    inputs["text"] = prompt
                    prompt_set = True

    # fallback: any node with literal text input
    if not prompt_set:
        for _, node in ordered_nodes:
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs") or {}
            if "text" in inputs and isinstance(inputs["text"], str):
                inputs["text"] = prompt
                prompt_set = True
                break

    # 2) Put uploaded filenames into targeted image-loader-like nodes.
    image_loader_types = {
        "loadimage",
        "loadimagemask",
        "vhs_loadimagepath",
        "image load",
        "loadimagefrompath",
    }
    loader_nodes = []
    for node_id, node in ordered_nodes:
        if not isinstance(node, dict):
            continue
        cls = str(node.get("class_type") or node.get("type") or "").strip().lower()
        inputs = node.setdefault("inputs", {})
        if cls in image_loader_types or "loadimage" in cls:
            title = str(((node.get("_meta") or {}).get("title") or "")).lower()
            key_blob = " ".join(str(k).lower() for k in (inputs or {}).keys())
            value_blob = " ".join(
                str(v).lower()
                for v in (
                    inputs.get("image"),
                    inputs.get("path"),
                    inputs.get("filename"),
                )
                if isinstance(v, str)
            )
            loader_nodes.append((str(node_id), node, inputs, cls, title, key_blob, value_blob))

    used_loader_ids = set()

    def _set_loader_image(inputs, filename):
        if "image" in inputs:
            inputs["image"] = filename
        elif "path" in inputs:
            inputs["path"] = filename
        elif "filename" in inputs:
            inputs["filename"] = filename
        else:
            inputs["image"] = filename

    def _assign_to_loaders(filename, predicate, max_assign=1):
        if not filename:
            return False
        assigned = False
        assigned_count = 0
        for node_id, _node, inputs, cls, title, key_blob, value_blob in loader_nodes:
            if node_id in used_loader_ids:
                continue
            if not predicate(cls, title, key_blob, value_blob, inputs):
                continue
            _set_loader_image(inputs, filename)
            used_loader_ids.add(node_id)
            assigned = True
            assigned_count += 1
            if max_assign and assigned_count >= max_assign:
                break
        return assigned

    def _assign_by_keys(filename, keys):
        if not filename:
            return False
        assigned = False
        for _node_id, node in ordered_nodes:
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs") or {}
            for key in keys:
                if key in inputs and isinstance(inputs.get(key), str):
                    inputs[key] = filename
                    assigned = True
        return assigned

    # Depth map: prefer loaders explicitly labelled for depth.
    image_set = _assign_to_loaders(
        uploaded_filename,
        lambda _cls, title, key_blob, value_blob, _inputs: (
            "depth" in title
            or "depth" in key_blob
            or "depth" in value_blob
        ),
        max_assign=1,
    )
    if not image_set:
        image_set = _assign_to_loaders(uploaded_filename, lambda *_args: True, max_assign=1)

    # Normal map loaders.
    _assign_to_loaders(
        uploaded_normal_filename,
        lambda _cls, title, key_blob, value_blob, _inputs: (
            "normal" in title
            or "normal" in key_blob
            or "normal" in value_blob
        ),
        max_assign=1,
    )
    _assign_by_keys(uploaded_normal_filename, ("normal", "normal_map", "input_normal", "image_normal"))

    # Silhouette map loaders.
    _assign_to_loaders(
        uploaded_silhouette_filename,
        lambda _cls, title, key_blob, value_blob, _inputs: (
            "silhouette" in title
            or "silhouette" in key_blob
            or "silhouette" in value_blob
            or "outline" in title
            or "outline" in value_blob
            or "building mask" in title
            or "building mask" in value_blob
        ),
        max_assign=1,
    )
    _assign_by_keys(uploaded_silhouette_filename, ("silhouette", "outline", "building_mask"))

    # Opacity/material mask loaders.
    _assign_to_loaders(
        uploaded_opacity_filename,
        lambda _cls, title, key_blob, value_blob, _inputs: (
            "opacity" in title
            or "alpha" in title
            or "semantic" in title
            or "material" in title
            or "id mask" in title
            or "opacity" in key_blob
            or "alpha" in key_blob
            or "semantic" in key_blob
            or "material" in key_blob
            or "opacity" in value_blob
            or "alpha" in value_blob
            or "semantic" in value_blob
            or "material" in value_blob
        ),
        max_assign=1,
    )
    _assign_by_keys(uploaded_opacity_filename, ("opacity", "alpha", "semantic", "material_mask", "id_mask"))

    # Binary mask map loaders (legacy building mask).
    mask_set = _assign_to_loaders(
        uploaded_mask_filename,
        lambda cls, title, key_blob, value_blob, _inputs: (
            cls == "loadimagemask"
            or (
                "mask" in title
                and "normal" not in title
                and "opacity" not in title
                and "alpha" not in title
            )
            or (
                "mask" in key_blob
                and "normal" not in key_blob
                and "opacity" not in key_blob
                and "alpha" not in key_blob
            )
            or (
                "mask" in value_blob
                and "normal" not in value_blob
                and "opacity" not in value_blob
                and "alpha" not in value_blob
            )
        ),
        max_assign=1,
    )

    if uploaded_mask_filename:
        for _node_id, node in ordered_nodes:
            if not isinstance(node, dict):
                continue
            cls = str(node.get("class_type") or node.get("type") or "").strip().lower()
            inputs = node.setdefault("inputs", {})
            if cls == "loadimagemask":
                if "image" in inputs and isinstance(inputs.get("image"), str):
                    inputs["image"] = uploaded_mask_filename
                    mask_set = True
                if "mask" in inputs and isinstance(inputs.get("mask"), str):
                    inputs["mask"] = uploaded_mask_filename
                    mask_set = True

    if not mask_set:
        _assign_by_keys(uploaded_mask_filename, ("mask", "image_mask"))

    # fallback: any node input field literally named image with string value
    if not image_set:
        for _node_id, node in ordered_nodes:
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs") or {}
            for key in ("image", "depth_image", "image_path", "path", "filename"):
                if key in inputs and isinstance(inputs[key], str):
                    inputs[key] = uploaded_filename
                    image_set = True
                    break
            if image_set:
                break

    # last-resort fallback: replace png/jpg-like string input on image-related nodes
    if not image_set:
        for _node_id, node in ordered_nodes:
            if not isinstance(node, dict):
                continue
            cls = str((node.get("class_type") or node.get("type") or "")).lower()
            if "image" not in cls and "depth" not in cls and "load" not in cls:
                continue
            inputs = node.get("inputs") or {}
            for key, value in inputs.items():
                if isinstance(value, str) and value.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    inputs[key] = uploaded_filename
                    image_set = True
                    break
            if image_set:
                break

    if not image_set:
        seen_types = []
        for _, node in ordered_nodes:
            if isinstance(node, dict):
                t = node.get("class_type") or node.get("type")
                if t:
                    seen_types.append(str(t))
        seen_types = sorted(set(seen_types))
        if seen_types:
            preview = ", ".join(seen_types[:12])
            raise ValueError(
                "could not inject depth image into workflow (no image loader found). "
                f"Detected node types: {preview}"
            )
        raise ValueError("could not inject depth image into workflow (no image loader found)")

    return workflow


def _http_post_json(url, payload, timeout=30, extra_headers=None):
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if isinstance(extra_headers, dict):
        headers.update(extra_headers)
    req = urlrequest.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            return resp.status, (json.loads(text) if text else {})
    except urlerror.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} POST {url}: {text[:300]}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc.reason}") from exc


def _http_get_json(url, timeout=30):
    req = urlrequest.Request(url, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            return resp.status, (json.loads(text) if text else {})
    except urlerror.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} GET {url}: {text[:300]}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"GET {url} failed: {exc.reason}") from exc


def _http_get_bytes(url, params=None, timeout=30):
    full_url = url
    if params:
        full_url = f"{url}?{urlparse.urlencode(params)}"
    req = urlrequest.Request(full_url, method="GET")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urlerror.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} GET {full_url}: {text[:300]}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"GET {full_url} failed: {exc.reason}") from exc


def _http_post_multipart(url, fields, files, timeout=30, extra_headers=None):
    boundary = f"----fp-boundary-{uuid.uuid4().hex}"
    body = bytearray()

    for key, val in (fields or {}).items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(val).encode("utf-8"))
        body.extend(b"\r\n")

    for key, file_info in (files or {}).items():
        filename = file_info.get("filename", "upload.bin")
        content_type = file_info.get("content_type", "application/octet-stream")
        content = file_info.get("content", b"")
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(content)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if isinstance(extra_headers, dict):
        headers.update(extra_headers)
    req = urlrequest.Request(
        url,
        data=bytes(body),
        headers=headers,
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            return resp.status, (json.loads(text) if text else {})
    except urlerror.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} POST {url}: {text[:300]}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc.reason}") from exc


def _pick_openai_image_size(width, height):
    # Keep to common currently-supported sizes for cloud image generation.
    choices = [(1024, 1024), (1024, 1536), (1536, 1024)]
    tw = max(256, int(width))
    th = max(256, int(height))
    best = min(choices, key=lambda wh: abs(wh[0] - tw) + abs(wh[1] - th))
    return f"{best[0]}x{best[1]}"


def _extract_openai_image_data_url(resp_json):
    data = (resp_json or {}).get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError("OpenAI returned no image data")
    first = data[0] or {}
    b64 = first.get("b64_json")
    if b64 and isinstance(b64, str):
        return f"data:image/png;base64,{b64}"
    url = first.get("url")
    if url and isinstance(url, str):
        _, img_bytes = _http_get_bytes(url, timeout=90)
        out_b64 = base64.b64encode(img_bytes).decode("ascii")
        return f"data:image/png;base64,{out_b64}"
    raise RuntimeError("OpenAI image response missing b64_json/url")


def _render_with_openai(
    prompt,
    depth_data_url,
    openai_api_key=None,
    model="gpt-image-1",
    quality="medium",
    width=1024,
    height=1024,
):
    api_key = str(openai_api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("missing OpenAI API key (provide openai_api_key or set OPENAI_API_KEY)")

    base_url = "https://api.openai.com/v1/images"
    size = _pick_openai_image_size(width, height)
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    clean_quality = str(quality or "medium").strip().lower()
    if clean_quality not in {"low", "medium", "high", "auto"}:
        clean_quality = "medium"

    used_endpoint = "generations"
    used_depth_reference = False
    first_error = None

    # First attempt: edits endpoint with the depth map as reference image.
    try:
        depth_png = _decode_data_url_png(depth_data_url)
        fields = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": clean_quality,
        }
        files = {
            "image": {
                "filename": "depth.png",
                "content_type": "image/png",
                "content": depth_png,
            }
        }
        _, payload = _http_post_multipart(
            f"{base_url}/edits",
            fields=fields,
            files=files,
            timeout=180,
            extra_headers=headers,
        )
        out_data_url = _extract_openai_image_data_url(payload)
        used_endpoint = "edits"
        used_depth_reference = True
    except Exception as exc:
        first_error = str(exc)
        # Fallback: plain prompt generation.
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": clean_quality,
        }
        _, resp = _http_post_json(
            f"{base_url}/generations",
            payload,
            timeout=180,
            extra_headers=headers,
        )
        out_data_url = _extract_openai_image_data_url(resp)

    meta = {
        "provider": "openai",
        "model": model,
        "quality": clean_quality,
        "size": size,
        "requested_width": int(width),
        "requested_height": int(height),
        "endpoint": used_endpoint,
        "used_depth_reference": bool(used_depth_reference),
        "edits_error": first_error,
    }
    return out_data_url, meta


def _render_with_comfy_local(
    comfy_url,
    prompt,
    depth_data_url,
    building_mask_data_url=None,
    normal_data_url=None,
    silhouette_data_url=None,
    opacity_mask_data_url=None,
    depth_invert=False,
    workflow_override=None,
):
    png_bytes = _decode_data_url_png(depth_data_url)
    if depth_invert:
        try:
            Image = importlib.import_module("PIL.Image")
            depth_img = Image.open(BytesIO(png_bytes)).convert("L")
            depth_np = np.array(depth_img, dtype=np.uint8)
            depth_np = (255 - depth_np).astype(np.uint8)
            out_buf = BytesIO()
            Image.fromarray(depth_np, mode="L").save(out_buf, format="PNG")
            png_bytes = out_buf.getvalue()
        except Exception:
            # Keep original depth if inversion preprocessing fails.
            pass
    mask_png_bytes = None
    if building_mask_data_url and isinstance(building_mask_data_url, str):
        try:
            mask_png_bytes = _decode_data_url_png(building_mask_data_url)
        except Exception:
            mask_png_bytes = None
    normal_png_bytes = None
    if normal_data_url and isinstance(normal_data_url, str):
        try:
            normal_png_bytes = _decode_data_url_png(normal_data_url)
        except Exception:
            normal_png_bytes = None
    silhouette_png_bytes = None
    if silhouette_data_url and isinstance(silhouette_data_url, str):
        try:
            silhouette_png_bytes = _decode_data_url_png(silhouette_data_url)
        except Exception:
            silhouette_png_bytes = None
    opacity_png_bytes = None
    if opacity_mask_data_url and isinstance(opacity_mask_data_url, str):
        try:
            opacity_png_bytes = _decode_data_url_png(opacity_mask_data_url)
        except Exception:
            opacity_png_bytes = None

    upload_url = f"{comfy_url}/upload/image"

    def _upload_png(prefix, content):
        file_name = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
        _, upj = _http_post_multipart(
            upload_url,
            fields={"type": "input", "overwrite": "true"},
            files={
                "image": {
                    "filename": file_name,
                    "content_type": "image/png",
                    "content": content,
                }
            },
            timeout=30,
        )
        return upj.get("name") or file_name

    uploaded_name = _upload_png("depth", png_bytes)
    uploaded_mask_name = _upload_png("mask", mask_png_bytes) if mask_png_bytes else None
    uploaded_normal_name = _upload_png("normal", normal_png_bytes) if normal_png_bytes else None
    uploaded_silhouette_name = _upload_png("silhouette", silhouette_png_bytes) if silhouette_png_bytes else None
    uploaded_opacity_name = _upload_png("opacity", opacity_png_bytes) if opacity_png_bytes else None

    # Prepare prompt payload
    workflow_fallback_used = False
    wf = _load_comfy_workflow(workflow_override)
    try:
        wf = _inject_prompt_and_depth(
            wf,
            prompt,
            uploaded_name,
            uploaded_mask_filename=uploaded_mask_name,
            uploaded_normal_filename=uploaded_normal_name,
            uploaded_silhouette_filename=uploaded_silhouette_name,
            uploaded_opacity_filename=uploaded_opacity_name,
        )
    except ValueError as exc:
        msg = str(exc).lower()
        if workflow_override and ("non-generative" in msg or "depth passthrough" in msg):
            logger.warning(
                "workflow_override rejected as non-generative; falling back to repo default workflow"
            )
            workflow_fallback_used = True
            wf = _load_comfy_workflow(None)
            wf = _inject_prompt_and_depth(
                wf,
                prompt,
                uploaded_name,
                uploaded_mask_filename=uploaded_mask_name,
                uploaded_normal_filename=uploaded_normal_name,
                uploaded_silhouette_filename=uploaded_silhouette_name,
                uploaded_opacity_filename=uploaded_opacity_name,
            )
        else:
            raise

    # Keep a node map for selecting the correct final output image later.
    if isinstance(wf.get("prompt"), dict):
        wf_nodes = wf.get("prompt") or {}
    elif isinstance(wf.get("workflow"), dict):
        wf_nodes = wf.get("workflow") or {}
    else:
        wf_nodes = wf

    submit_url = f"{comfy_url}/prompt"
    client_id = uuid.uuid4().hex
    _, sbj = _http_post_json(
        submit_url,
        payload={"prompt": wf, "client_id": client_id},
        timeout=30,
    )
    prompt_id = sbj.get("prompt_id")
    if not prompt_id:
        raise RuntimeError("ComfyUI did not return prompt_id")

    # Poll history for completion
    history_url = f"{comfy_url}/history/{prompt_id}"
    deadline = time.time() + 180
    history_obj = None
    while time.time() < deadline:
        _, hj = _http_get_json(history_url, timeout=20)
        if isinstance(hj, dict) and hj.get(prompt_id):
            history_obj = hj[prompt_id]
            break
        time.sleep(1.0)

    if history_obj is None:
        raise TimeoutError("ComfyUI generation timed out waiting for history")

    outputs = history_obj.get("outputs") or {}
    image_desc = None

    # Prefer SaveImage outputs; some workflows also emit preview/control images.
    save_node_ids = []
    if isinstance(wf_nodes, dict):
        for node_id, node in wf_nodes.items():
            if not isinstance(node, dict):
                continue
            cls = str(node.get("class_type") or node.get("type") or "").strip().lower()
            if cls == "saveimage" or "saveimage" in cls:
                save_node_ids.append(str(node_id))

    for node_id in save_node_ids:
        out = outputs.get(node_id)
        imgs = out.get("images") if isinstance(out, dict) else None
        if imgs:
            image_desc = imgs[0]
            break

    # Fallback: any available image output.
    if not image_desc:
        for _, out in outputs.items():
            imgs = out.get("images") if isinstance(out, dict) else None
            if imgs:
                image_desc = imgs[0]
                break

    if not image_desc:
        raise RuntimeError("ComfyUI returned no output images")

    filename = image_desc.get("filename")
    subfolder = image_desc.get("subfolder", "")
    img_type = image_desc.get("type", "output")
    if not filename:
        raise RuntimeError("output image descriptor missing filename")

    view_url = f"{comfy_url}/view"
    _, out_bytes = _http_get_bytes(
        view_url,
        params={"filename": filename, "subfolder": subfolder, "type": img_type},
        timeout=40,
    )

    out_b64 = base64.b64encode(out_bytes).decode("ascii")
    out_data_url = f"data:image/png;base64,{out_b64}"
    meta = {
        "comfy_url": comfy_url,
        "prompt_id": prompt_id,
        "mask_uploaded": bool(uploaded_mask_name),
        "normal_uploaded": bool(uploaded_normal_name),
        "silhouette_uploaded": bool(uploaded_silhouette_name),
        "opacity_mask_uploaded": bool(uploaded_opacity_name),
        "workflow_fallback_used": bool(workflow_fallback_used),
        "depth_invert": bool(depth_invert),
        "output": {
            "filename": filename,
            "subfolder": subfolder,
            "type": img_type,
        },
    }
    return out_data_url, meta


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
