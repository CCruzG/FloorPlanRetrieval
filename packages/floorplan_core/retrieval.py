import copy
from pathlib import Path

import numpy as np

from .geo_ent_num_retrieval import (
    shape_desc,
    entrance_desc_from_point,
    retrieve as geo_retrieve,
    N_FREQS,
    ENT_DIM,
    ROOM_DIM,
)


SHAPE_DIM = N_FREQS
TOTAL_DIM = SHAPE_DIM + ENT_DIM + ROOM_DIM


class RetrievalError(Exception):
    pass


def _req_to_room_desc(requirements):
    """Build an 11-dim room-count descriptor from user requirement hints."""
    if not requirements:
        requirements = {}

    bedrooms = int(requirements.get("bedrooms", 2))
    bathrooms = int(requirements.get("bathrooms", 1))

    living = 1
    master = 1 if bedrooms > 0 else 0
    kitchen = 1
    dining = 1
    child = max(0, bedrooms - 1)
    total = living + master + kitchen + dining + child + bathrooms

    return np.array(
        [
            min(living / 2, 1.0),
            min(master / 1, 1.0),
            min(kitchen / 1, 1.0),
            min(bathrooms / 3, 1.0),
            min(dining / 1, 1.0),
            min(child / 2, 1.0),
            0.0,
            0.0,
            0.0,
            0.0,
            min(total / 15, 1.0),
        ],
        dtype=np.float32,
    )


def run_retrieval(index_store, boundary, entrance_point=None, requirements=None,
                  topn=5, shape_w=1.0, ent_w=2.0, room_w=1.0, logger=None):
    """Run retrieval for one query and return response payload dict."""
    bdry_segs = [[v[0], v[1]] for v in boundary]

    # Normalize winding so descriptor output is draw-order invariant.
    signed_area = sum(
        bdry_segs[i][0] * bdry_segs[(i + 1) % len(bdry_segs)][1]
        - bdry_segs[(i + 1) % len(bdry_segs)][0] * bdry_segs[i][1]
        for i in range(len(bdry_segs))
    )
    if signed_area > 0:
        bdry_segs = bdry_segs[::-1]

    try:
        q_shape = shape_desc(bdry_segs)
        if entrance_point:
            q_ent = entrance_desc_from_point(entrance_point, bdry_segs)
        else:
            q_ent = np.zeros(ENT_DIM, dtype=np.float32)
            ent_w = 0.0
        q_room = _req_to_room_desc(requirements)
    except Exception as exc:
        raise RetrievalError(f"descriptor error: {exc}") from exc

    try:
        if index_store.shapes is not None:
            idx, dists, _, _, _, best_angles, best_mirror = geo_retrieve(
                q_shape,
                q_ent,
                q_room,
                index_store.shapes,
                index_store.ents,
                index_store.rooms,
                shape_w,
                ent_w,
                room_w,
                topn,
            )
        else:
            q_desc = np.concatenate([q_shape, q_ent, q_room]).astype(np.float32)
            flat_dists = np.linalg.norm(index_store.descs - q_desc[np.newaxis], axis=1)
            idx = np.argsort(flat_dists)[:topn]
            dists = flat_dists[idx]
            best_angles = np.zeros(len(idx))
            best_mirror = np.zeros(len(idx), dtype=bool)
    except Exception as exc:
        raise RetrievalError(f"retrieval error: {exc}") from exc

    candidates = []
    cand_idx = []
    for i in idx:
        raw_name = index_store.names[i]
        stem = Path(raw_name).stem
        plan = index_store.plan_cache.get(stem) or index_store.plan_cache.get(raw_name)
        if plan is None:
            if logger is not None:
                logger.warning("Plan not found in cache for db entry '%s'", raw_name)
            continue
        candidates.append(plan)
        cand_idx.append(int(i))

    if not candidates:
        return {
            "status": "ok",
            "query_id": "query",
            "count": 0,
            "plans": [],
        }

    result_list = []
    for rank, (plan, db_i) in enumerate(zip(candidates, cand_idx)):
        p = copy.deepcopy(plan)
        if not isinstance(p.get("retrieval"), dict):
            p["retrieval"] = {}
        p["retrieval"]["dist"] = float(dists[rank])
        p["retrieval"]["source_id"] = index_store.names[db_i]
        p["retrieval"]["rot"] = float(best_angles[rank])
        p["retrieval"]["mirror"] = bool(best_mirror[rank])
        result_list.append(p)

    result_list.sort(key=lambda p: (p.get("retrieval") or {}).get("dist", float("inf")))

    return {
        "status": "ok",
        "query_id": "query",
        "count": len(result_list),
        "plans": result_list,
    }
