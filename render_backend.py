import base64
import copy
import importlib
import json
import os
import time
import uuid
from io import BytesIO
from urllib import parse as urlparse

import numpy as np

from comfy_client import http_get_bytes, http_get_json, http_post_json, http_post_multipart

DEFAULT_COMFY_WORKFLOW_FILENAME = "comfyui-sdxl-depth-multi-workflow.json"


def validate_comfy_url(comfy_url, allow_remote_comfy=False):
    parsed = urlparse.urlparse(comfy_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("invalid comfy_url: expected absolute http(s) URL")
    if allow_remote_comfy:
        return comfy_url

    host = parsed.hostname.lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("remote comfy_url is disabled; use localhost/127.0.0.1 or start server with --allow-remote-comfy")
    return comfy_url


def check_comfy_health(comfy_url, allow_remote_comfy=False):
    comfy_url = validate_comfy_url(comfy_url, allow_remote_comfy=allow_remote_comfy)
    _, payload = http_get_json(f"{comfy_url}/system_stats", timeout=10)
    devices = payload.get("devices") if isinstance(payload, dict) else None
    n_devices = len(devices) if isinstance(devices, list) else 0
    return f"ComfyUI reachable at {comfy_url} ({n_devices} device(s))"


def render(backend, params, base_dir, logger, allow_remote_comfy=False):
    """Render image using selected backend; returns (image_data_url, meta)."""
    if backend in {"comfy", "comfyui", "local"}:
        return _render_with_comfy_local(
            comfy_url=validate_comfy_url(
                str(params.get("comfy_url") or "http://127.0.0.1:8188").rstrip("/"),
                allow_remote_comfy=allow_remote_comfy,
            ),
            prompt=params["prompt"],
            depth_data_url=params["depth_data_url"],
            building_mask_data_url=params.get("building_mask_data_url"),
            normal_data_url=params.get("normal_data_url"),
            silhouette_data_url=params.get("silhouette_data_url"),
            opacity_mask_data_url=params.get("opacity_mask_data_url"),
            segmentation_data_url=params.get("segmentation_data_url"),
            depth_invert=bool(params.get("depth_invert", False)),
            workflow_override=params.get("workflow_override"),
            base_dir=base_dir,
            logger=logger,
        )

    if backend != "openai":
        raise ValueError(f"unsupported backend: {backend}. supported: 'openai', 'comfy'")

    return _render_with_openai(
        prompt=params["prompt"],
        depth_data_url=params["depth_data_url"],
        model=str(params.get("openai_model") or "gpt-image-1"),
        quality=str(params.get("openai_quality") or "medium"),
        width=int(params.get("openai_width") or params.get("width") or 1024),
        height=int(params.get("openai_height") or params.get("height") or 1024),
    )


def _decode_data_url_png(data_url):
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise ValueError("depth_data_url must be a PNG data URL")
    raw = data_url[len(prefix):]
    return base64.b64decode(raw)


def _load_comfy_workflow(workflow_override, base_dir):
    if isinstance(workflow_override, dict) and workflow_override:
        return copy.deepcopy(workflow_override)

    wf_path = base_dir / DEFAULT_COMFY_WORKFLOW_FILENAME
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
    uploaded_segmentation_filename=None,
):
    """Best-effort workflow injection for ComfyUI workflow payloads."""
    if not isinstance(workflow, dict):
        raise ValueError("workflow must be a dict")

    if isinstance(workflow.get("prompt"), dict):
        workflow_nodes = workflow["prompt"]
    elif isinstance(workflow.get("workflow"), dict):
        workflow_nodes = workflow["workflow"]
    else:
        workflow_nodes = workflow

    if not isinstance(workflow_nodes, dict):
        raise ValueError("workflow prompt must be a dict keyed by node id")

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

    if not prompt_set:
        for _, node in ordered_nodes:
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs") or {}
            if "text" in inputs and isinstance(inputs["text"], str):
                inputs["text"] = prompt
                prompt_set = True
                break

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

    _assign_to_loaders(
        uploaded_opacity_filename,
        lambda _cls, title, key_blob, value_blob, _inputs: (
            "opacity" in title
            or "alpha" in title
            or "material" in title
            or "id mask" in title
            or "opacity" in key_blob
            or "alpha" in key_blob
            or "material" in key_blob
            or "opacity" in value_blob
            or "alpha" in value_blob
            or "material" in value_blob
        ),
        max_assign=1,
    )
    _assign_by_keys(uploaded_opacity_filename, ("opacity", "alpha", "material_mask", "id_mask"))

    segmentation_assigned = _assign_to_loaders(
        uploaded_segmentation_filename,
        lambda _cls, title, key_blob, value_blob, _inputs: (
            "segmentation" in title
            or "ade20k" in title
            or "semantic" in title
            or "segment" in title
            or "opacity" in title
            or "segmentation" in key_blob
            or "ade20k" in key_blob
            or "semantic" in key_blob
            or "segment" in key_blob
            or "opacity" in key_blob
            or "segmentation" in value_blob
            or "ade20k" in value_blob
            or "semantic" in value_blob
            or "segment" in value_blob
            or "opacity" in value_blob
        ),
        max_assign=1,
    )
    _assign_by_keys(
        uploaded_segmentation_filename,
        ("segmentation", "semantic", "segment_map", "ade20k", "seg_map", "semantic_map", "opacity", "opacity_mask"),
    )

    if (not segmentation_assigned) and uploaded_opacity_filename:
        _assign_to_loaders(
            uploaded_opacity_filename,
            lambda _cls, title, key_blob, value_blob, _inputs: (
                "segmentation" in title
                or "ade20k" in title
                or "semantic" in title
                or "segment" in title
                or "segmentation" in key_blob
                or "ade20k" in key_blob
                or "semantic" in key_blob
                or "segment" in key_blob
                or "segmentation" in value_blob
                or "ade20k" in value_blob
                or "semantic" in value_blob
                or "segment" in value_blob
            ),
            max_assign=1,
        )

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


def _pick_openai_image_size(width, height):
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
        _, img_bytes = http_get_bytes(url, timeout=90)
        out_b64 = base64.b64encode(img_bytes).decode("ascii")
        return f"data:image/png;base64,{out_b64}"
    raise RuntimeError("OpenAI image response missing b64_json/url")


def _render_with_openai(
    prompt,
    depth_data_url,
    model="gpt-image-1",
    quality="medium",
    width=1024,
    height=1024,
):
    api_key = str(os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("missing OpenAI API key (set OPENAI_API_KEY in the server environment)")

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
        _, payload = http_post_multipart(
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
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": clean_quality,
        }
        _, resp = http_post_json(
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
    segmentation_data_url=None,
    depth_invert=False,
    workflow_override=None,
    base_dir=None,
    logger=None,
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
    segmentation_png_bytes = None
    if segmentation_data_url and isinstance(segmentation_data_url, str):
        try:
            segmentation_png_bytes = _decode_data_url_png(segmentation_data_url)
        except Exception:
            segmentation_png_bytes = None

    upload_url = f"{comfy_url}/upload/image"

    def _upload_png(prefix, content):
        file_name = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
        _, upj = http_post_multipart(
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
    uploaded_segmentation_name = _upload_png("segmentation", segmentation_png_bytes) if segmentation_png_bytes else None

    workflow_fallback_used = False
    wf = _load_comfy_workflow(workflow_override, base_dir=base_dir)
    try:
        wf = _inject_prompt_and_depth(
            wf,
            prompt,
            uploaded_name,
            uploaded_mask_filename=uploaded_mask_name,
            uploaded_normal_filename=uploaded_normal_name,
            uploaded_silhouette_filename=uploaded_silhouette_name,
            uploaded_opacity_filename=uploaded_opacity_name,
            uploaded_segmentation_filename=uploaded_segmentation_name,
        )
    except ValueError as exc:
        msg = str(exc).lower()
        if workflow_override and ("non-generative" in msg or "depth passthrough" in msg):
            if logger is not None:
                logger.warning(
                    "workflow_override rejected as non-generative; falling back to repo default workflow"
                )
            workflow_fallback_used = True
            wf = _load_comfy_workflow(None, base_dir=base_dir)
            wf = _inject_prompt_and_depth(
                wf,
                prompt,
                uploaded_name,
                uploaded_mask_filename=uploaded_mask_name,
                uploaded_normal_filename=uploaded_normal_name,
                uploaded_silhouette_filename=uploaded_silhouette_name,
                uploaded_opacity_filename=uploaded_opacity_name,
                uploaded_segmentation_filename=uploaded_segmentation_name,
            )
        else:
            raise

    if isinstance(wf.get("prompt"), dict):
        wf_nodes = wf.get("prompt") or {}
    elif isinstance(wf.get("workflow"), dict):
        wf_nodes = wf.get("workflow") or {}
    else:
        wf_nodes = wf

    submit_url = f"{comfy_url}/prompt"
    client_id = uuid.uuid4().hex
    _, sbj = http_post_json(
        submit_url,
        payload={"prompt": wf, "client_id": client_id},
        timeout=30,
    )
    prompt_id = sbj.get("prompt_id")
    if not prompt_id:
        raise RuntimeError("ComfyUI did not return prompt_id")

    history_url = f"{comfy_url}/history/{prompt_id}"
    deadline = time.time() + 180
    history_obj = None
    while time.time() < deadline:
        _, hj = http_get_json(history_url, timeout=20)
        if isinstance(hj, dict) and hj.get(prompt_id):
            history_obj = hj[prompt_id]
            break
        time.sleep(1.0)

    if history_obj is None:
        raise TimeoutError("ComfyUI generation timed out waiting for history")

    outputs = history_obj.get("outputs") or {}
    image_desc = None

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
    _, out_bytes = http_get_bytes(
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
