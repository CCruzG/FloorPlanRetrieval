from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class IndexStore:
    descs: np.ndarray
    shapes: np.ndarray | None
    ents: np.ndarray | None
    rooms: np.ndarray | None
    names: list[str]
    dim: int
    plan_cache: dict[str, dict]


def build_index_store(
    *,
    bin_path,
    names_path,
    data_dir,
    total_dim,
    shape_dim,
    ent_dim,
    logger,
):
    """Load descriptor arrays and plan cache into one IndexStore instance."""
    descs, names, dim, shapes, ents, rooms = load_index(
        bin_path,
        names_path,
        total_dim=total_dim,
        shape_dim=shape_dim,
        ent_dim=ent_dim,
    )
    plan_cache = load_plans(data_dir, logger=logger)
    return IndexStore(
        descs=descs,
        shapes=shapes,
        ents=ents,
        rooms=rooms,
        names=names,
        dim=dim,
        plan_cache=plan_cache,
    )


def load_index(bin_path, names_path, total_dim, shape_dim, ent_dim):
    """Load descriptor matrix and names list.

    Returns (descs, names, dim, shapes, ents, rooms). shapes/ents/rooms are
    None when the index is not the expected split format.
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

    if dim == total_dim:
        shapes = descs[:, :shape_dim]
        ents = descs[:, shape_dim:shape_dim + ent_dim]
        rooms = descs[:, shape_dim + ent_dim:]
    else:
        shapes = ents = rooms = None

    return descs, names, dim, shapes, ents, rooms


def load_plans(data_dir, logger):
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


def ensure_manifest(manifest_path, data_dir, logger):
    """Return manifest list; generate and save if file is absent."""
    mp = Path(manifest_path)
    if mp.exists():
        with open(mp) as fh:
            return json.load(fh)
    logger.warning(
        "manifest.json not found at '%s' - generating from %s/", manifest_path, data_dir
    )
    names = sorted(p.name for p in Path(data_dir).glob("*.json"))
    with open(mp, "w") as fh:
        json.dump(names, fh, indent=2)
    return names
