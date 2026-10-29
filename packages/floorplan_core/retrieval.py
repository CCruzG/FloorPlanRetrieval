import copy
from pathlib import Path

import numpy as np

from .geo_ent_num_retrieval import (
    shape_desc_complex,
    entrance_desc_from_point,
    retrieve as geo_retrieve,
    N_FREQS,
    ENT_DIM,
    ROOM_DIM,
)
from .topo_signature import (
    topological_fitness, default_edge_types, filter_corner_adjacencies,
)
from .topo_sensors import run_sensors


SHAPE_DIM = N_FREQS
TOTAL_DIM = SHAPE_DIM + ENT_DIM + ROOM_DIM


class RetrievalError(Exception):
    pass


def _req_to_room_desc(requirements):
    """Build an 11-dim room-count descriptor from user requirement hints.

    Dimension order matches geo_ent_num_retrieval.ROOM_TYPES: [Living,
    Master, Kitchen, Bathroom, Dining, Child, Study, Second, Balcony,
    Storage, total]. Bedrooms fill Master then Second then Child, in that
    priority order — RPlan's own label taxonomy treats "Second" (a second
    primary-ish bedroom) as distinct from "Child". Study/Balcony/Storage
    come from explicit UI toggles, default false.

    2026-08-14 fix: previously Second/Study/Balcony/Storage were hardcoded
    to zero here (no UI input existed for them), which meant hand-drawn
    queries could never match real plans on those dimensions. That's not
    a rare case — 95% of real plans have a nonzero Second, 84% a nonzero
    Balcony, 99.5% at least one of the four — so nearly every query
    carried a ~1.0 room_dist floor against nearly every real plan,
    regardless of how good the actual shape/entrance match was."""
    if not requirements:
        requirements = {}

    bedrooms = int(requirements.get("bedrooms", 2))
    bathrooms = int(requirements.get("bathrooms", 1))

    living = 1
    master = 1 if bedrooms > 0 else 0
    second = 1 if bedrooms > 1 else 0
    child = max(0, bedrooms - 2)
    kitchen = 1
    dining = 1
    study = 1 if requirements.get("study") else 0
    balcony = 1 if requirements.get("balcony") else 0
    storage = 1 if requirements.get("storage") else 0
    total = living + master + second + child + kitchen + dining + bathrooms + study + balcony + storage

    return np.array(
        [
            min(living / 2, 1.0),
            min(master / 1, 1.0),
            min(kitchen / 1, 1.0),
            min(bathrooms / 3, 1.0),
            min(dining / 1, 1.0),
            min(child / 2, 1.0),
            min(study / 1, 1.0),
            min(second / 2, 1.0),
            min(balcony / 2, 1.0),
            min(storage / 2, 1.0),
            min(total / 15, 1.0),
        ],
        dtype=np.float32,
    )


def run_retrieval(index_store, boundary, entrance_point=None, requirements=None,
                  topn=5, shape_w=1.0, ent_w=2.0, room_w=1.0, logger=None,
                  topo_rerank=True, pool_n=60, typed_edges=None):
    """Run retrieval for one query and return response payload dict.

    topo_rerank (Step 1, topology → generate → select): when True, shape
    retrieval pulls a larger POOL (pool_n) and the pool is re-ranked by
    topological fitness (topo_signature.topological_fitness) — retrieval by
    topology, not geometry. The discriminator is a perimeter-ring alignment of
    each candidate's daylight demand against the TARGET boundary's typed edges,
    so `typed_edges` (the target's [{edge_index, type}] list; defaulted when
    absent) is passed through. Each returned plan carries its topo breakdown at
    retrieval.topo. Shape distance only gates the pool and breaks ties. Set False
    for the original shape-only ranking."""
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
        q_shape = shape_desc_complex(bdry_segs)
        if entrance_point:
            q_ent = entrance_desc_from_point(entrance_point, bdry_segs)
        else:
            q_ent = np.zeros(ENT_DIM, dtype=np.float32)
            ent_w = 0.0
        q_room = _req_to_room_desc(requirements)
    except Exception as exc:
        raise RetrievalError(f"descriptor error: {exc}") from exc

    # Topology re-rank pulls a larger pool by shape, then re-orders it by
    # topological fitness; without it we fetch exactly topn by shape rank.
    n_fetch = max(topn, pool_n) if topo_rerank else topn

    try:
        if index_store.shapes is not None:
            # Canonical retrieve() (rotation-coupled "Option B" shape matching)
            # returns an 8-tuple — raw_angles (continuous, pre-snap) sits
            # between best_angles and use_mirror; unused here, display only.
            # shape/ent/room dists are the paper's three weighted component
            # distances (combined = their sum) — surfaced per-result so the
            # app can show what actually drove the match, not just the total.
            idx, dists, shape_dists, ent_dists, room_dists, best_angles, _raw_angles, best_mirror = geo_retrieve(
                q_shape,
                q_ent,
                q_room,
                index_store.shapes,
                index_store.ents,
                index_store.rooms,
                shape_w,
                ent_w,
                room_w,
                n_fetch,
            )
        else:
            # Legacy flat-.bin fallback (real-valued, magnitude-only shape
            # descriptor) — not exercised by the canonical .npz index path,
            # which always populates index_store.shapes. Kept for any
            # remaining old-format index still on disk.
            q_desc = np.concatenate([q_shape.real, q_ent, q_room]).astype(np.float32)
            flat_dists = np.linalg.norm(index_store.descs - q_desc[np.newaxis], axis=1)
            idx = np.argsort(flat_dists)[:n_fetch]
            dists = flat_dists[idx]
            best_angles = np.zeros(len(idx))
            best_mirror = np.zeros(len(idx), dtype=bool)
            # Legacy path doesn't decompose distance into components — None,
            # not NaN, so the response stays valid JSON (json.dumps allows
            # NaN; JSON.parse in the browser does not).
            shape_dists = ent_dists = room_dists = [None] * len(idx)
    except Exception as exc:
        raise RetrievalError(f"retrieval error: {exc}") from exc

    # Build one result dict per pooled candidate, keyed to its geo rank so the
    # per-component distance arrays stay aligned even when a plan is missing
    # from the cache (the skipped-plan case the old rank counter mishandled).
    result_list = []
    for rank, i in enumerate(idx):
        raw_name = index_store.names[int(i)]
        stem = Path(raw_name).stem
        plan = index_store.plan_cache.get(stem) or index_store.plan_cache.get(raw_name)
        if plan is None:
            if logger is not None:
                logger.warning("Plan not found in cache for db entry '%s'", raw_name)
            continue
        p = copy.deepcopy(plan)
        if not isinstance(p.get("retrieval"), dict):
            p["retrieval"] = {}
        p["retrieval"]["dist"] = float(dists[rank])
        p["retrieval"]["source_id"] = raw_name
        p["retrieval"]["rot"] = float(best_angles[rank])
        p["retrieval"]["mirror"] = bool(best_mirror[rank])
        # The paper's three weighted component distances (dist = their sum) —
        # what actually drove the match, not just the total.
        p["retrieval"]["shape_dist"] = None if shape_dists[rank] is None else float(shape_dists[rank])
        p["retrieval"]["ent_dist"] = None if ent_dists[rank] is None else float(ent_dists[rank])
        p["retrieval"]["room_dist"] = None if room_dists[rank] is None else float(room_dists[rank])
        # Drop corner-only room↔room edges: a point contact is not an adjacency, so
        # it must not enter the drawn graph, generation, or reachability.
        filter_corner_adjacencies(p)
        result_list.append(p)

    if not result_list:
        return {
            "status": "ok",
            "query_id": "query",
            "count": 0,
            "plans": [],
        }

    if topo_rerank:
        # Retrieval by topology: score each pooled candidate against the TARGET
        # boundary (perimeter-ring alignment) and re-order by fitness, shape
        # distance breaking ties. The pool was gathered by shape; the ranking
        # that leaves it is topological.
        # Resolve the target's per-edge types once — supplied {edge_index,type}
        # list wins, else default (entrance edge → corridor, rest external).
        n_edges = len(boundary)
        resolved_types = default_edge_types(boundary, entrance_point)
        for e in typed_edges or []:
            i = e.get("edge_index")
            if i is not None and 0 <= i < n_edges and e.get("type"):
                resolved_types[i] = e["type"]
        for p in result_list:
            topo = topological_fitness(
                p,
                target_boundary=boundary,
                target_edge_types=resolved_types,
                entrance_point=entrance_point,
                requirements=requirements,
            )
            p["retrieval"]["topo"] = topo
            # Additive: the room<->boundary-edge structure topo's daylight_fit
            # ring pass already computes, materialised instead of thrown away.
            p["retrieval"]["room_edges"] = topo["room_edges"]
            # target-frame node positions (all inside the boundary) — the graph
            # view draws these instead of re-deriving placement client-side.
            p["retrieval"]["node_pos"] = topo.get("node_pos", {})
            p["retrieval"]["flags"] = run_sensors(
                p, p["retrieval"]["room_edges"], requirements, target_boundary=boundary)
        result_list.sort(
            key=lambda p: (-p["retrieval"]["topo"]["score"], p["retrieval"]["dist"])
        )
    else:
        result_list.sort(key=lambda p: (p.get("retrieval") or {}).get("dist", float("inf")))

    result_list = result_list[:topn]

    return {
        "status": "ok",
        "query_id": "query",
        "count": len(result_list),
        "plans": result_list,
    }
