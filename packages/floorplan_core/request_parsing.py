class RequestValidationError(ValueError):
    pass


def parse_boolish(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def ensure_boundary(boundary, min_points=3):
    if not boundary or len(boundary) < min_points:
        raise RequestValidationError(f"boundary must have at least {min_points} vertices")
    return boundary


def parse_retrieve_request(body):
    boundary = ensure_boundary(body.get("boundary"), min_points=3)
    return {
        "boundary": boundary,
        "entrance_point": body.get("entrance_point"),
        "requirements": body.get("requirements"),
        "topn": int(body.get("topn", 5)),
        "shape_w": float(body.get("shape_w", 1.0)),
        "ent_w": float(body.get("ent_w", 2.0)),  # paper default (make_dist_histogram.py): shape/entrance/room = 1/2/1
        "room_w": float(body.get("room_w", 1.0)),
        # target boundary edge types [{edge_index, type}] — used by the topology
        # re-rank (Step 1) to arc-align each candidate's daylight demand.
        "typed_edges": body.get("typed_edges") or body.get("boundary_typed_edges"),
    }


def parse_fit_request(body):
    fp = body.get("fp")
    boundary = body.get("boundary")
    if not fp:
        raise RequestValidationError("missing or invalid fp/boundary")
    ensure_boundary(boundary, min_points=3)
    bdry_xy = [[v[0], v[1]] for v in boundary]
    return {"fp": fp, "boundary": bdry_xy}


def parse_render_request(body):
    backend = str(body.get("backend", "openai")).strip().lower()
    prompt = str(body.get("prompt", "")).strip()
    depth_data_url = body.get("depth_data_url")
    if not prompt:
        raise RequestValidationError("missing prompt")
    if not depth_data_url or not isinstance(depth_data_url, str):
        raise RequestValidationError("missing depth_data_url")

    return {
        "backend": backend,
        "params": {
            "prompt": prompt,
            "depth_data_url": depth_data_url,
            "preview_data_url": body.get("preview_data_url"),
            "building_mask_data_url": body.get("building_mask_data_url"),
            "normal_data_url": body.get("normal_data_url"),
            "silhouette_data_url": body.get("silhouette_data_url"),
            "opacity_mask_data_url": body.get("opacity_mask_data_url"),
            "segmentation_data_url": body.get("segmentation_data_url"),
            "depth_invert": parse_boolish(body.get("depth_invert", False)),
            "comfy_url": body.get("comfy_url"),
            "workflow_override": body.get("workflow_override"),
            "openai_model": body.get("openai_model"),
            "openai_quality": body.get("openai_quality"),
            "openai_width": body.get("openai_width"),
            "openai_height": body.get("openai_height"),
            "openai_api_key": body.get("openai_api_key"),
            "width": body.get("width"),
            "height": body.get("height"),
        },
    }


def parse_fit_step1_request(body):
    fp = body.get("fp")
    if not fp or not isinstance(fp, dict):
        raise RequestValidationError("missing or invalid fp")
    boundary = body.get("boundary", [])
    if len(boundary) < 3:
        raise RequestValidationError("boundary must have at least 3 vertices")
    if not fp.get("rooms"):
        raise RequestValidationError("fp.rooms is empty")
    entrance_point = body.get("entrance_point")
    suggested_positions = body.get("suggested_positions")  # {id: {x, y}} — optional
    return {
        "fp": fp,
        "boundary": [[v[0], v[1]] for v in boundary],
        "entrance_point": entrance_point,
        "suggested_positions": suggested_positions,
    }


def parse_comfy_health_request(body):
    return str(body.get("comfy_url") or "http://127.0.0.1:8188").rstrip("/")
