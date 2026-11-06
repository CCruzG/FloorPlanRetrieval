"""Floor plate library (2026-11-06): drawn floor plates saved to a repo JSON file.

Domain logic only, no Flask. The server (apps/api/server.py) maps these functions to
HTTP and turns PlateValidationError into 400 and PlateNotFoundError into 404.

File: data/floor_plates/library.json, overridden by env FLOOR_PLATE_LIBRARY (read on
every call, so tests can point it at a temp file).

    {"schema_version": 2, "units": "m", "frame": "canvas, y down", "plates": [...]}

Plate fields: id, name, boundary_m [[x, y], ...], main_entrance_m [x, y] | null,
edge_types (one value per edge, or []), north_deg (clockwise from canvas up),
street_edges (sorted edge indices, street frontage; edge i joins vertex i and
vertex i+1), notes, saved. Schema 1 files held road_edge (index | null); they are
converted on read and rewritten on the next save.
area_m2 is computed on every response and never stored.
"""
import datetime
import json
import math
import os
import re
import tempfile
import threading
from pathlib import Path

from shapely.geometry import LinearRing, Polygon

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "data" / "floor_plates" / "library.json"
ENV_VAR = "FLOOR_PLATE_LIBRARY"
SCHEMA_VERSION = 2
EDGE_TYPES = ("external", "shared_wall", "facing_corridor")

_LOCK = threading.RLock()


class PlateValidationError(ValueError):
    """Input is not a valid plate (HTTP 400)."""


class PlateNotFoundError(KeyError):
    """No plate with this id (HTTP 404)."""


class LibraryFileError(RuntimeError):
    """The library file exists but cannot be read (HTTP 500)."""


def library_path():
    env = os.environ.get(ENV_VAR)
    return Path(env) if env else DEFAULT_PATH


# ── validation ───────────────────────────────────────────────────────────────

def _num(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PlateValidationError(f"{what} must be a finite number")
    return float(value)


def _point(value, what):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise PlateValidationError(f"{what} must be [x, y]")
    return [_num(value[0], what), _num(value[1], what)]


def _validate(data):
    """Return the cleaned plate fields (no id, no saved). Raise PlateValidationError."""
    if not isinstance(data, dict):
        raise PlateValidationError("body must be a JSON object")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise PlateValidationError("name must be a non-empty string")

    raw = data.get("boundary_m")
    if not isinstance(raw, (list, tuple)):
        raise PlateValidationError("boundary_m must be a list of [x, y]")
    if len(raw) < 3:
        raise PlateValidationError("boundary_m needs at least 3 vertices")
    boundary = [_point(p, f"boundary_m[{i}]") for i, p in enumerate(raw)]
    n = len(boundary)
    for i in range(n):
        if boundary[i] == boundary[(i + 1) % n]:
            raise PlateValidationError(f"boundary_m has a zero-length edge at edge {i}")
    if not LinearRing(boundary).is_simple:
        raise PlateValidationError("boundary_m is self-intersecting")
    if Polygon(boundary).area <= 0:
        raise PlateValidationError("boundary_m has zero area")

    entrance = data.get("main_entrance_m")
    entrance = None if entrance is None else _point(entrance, "main_entrance_m")

    edge_types = data.get("edge_types")
    edge_types = [] if edge_types is None else edge_types
    if not isinstance(edge_types, (list, tuple)):
        raise PlateValidationError("edge_types must be a list")
    if edge_types:
        if len(edge_types) != n:
            raise PlateValidationError(
                f"edge_types has {len(edge_types)} values, boundary has {n} edges")
        for i, t in enumerate(edge_types):
            if t is not None and t not in EDGE_TYPES:
                raise PlateValidationError(f"edge_types[{i}] must be one of {list(EDGE_TYPES)} or null")
    edge_types = list(edge_types)

    north = data.get("north_deg", 0)
    north = 0.0 if north is None else _num(north, "north_deg")

    street = data.get("street_edges")
    street = [] if street is None else street
    if not isinstance(street, (list, tuple)):
        raise PlateValidationError("street_edges must be a list of edge indices")
    for e in street:
        if isinstance(e, bool) or not isinstance(e, int):
            raise PlateValidationError("street_edges must hold integer edge indices")
        if not 0 <= e < n:
            raise PlateValidationError(f"street edge {e} is out of range for {n} edges")
    street = sorted(set(street))

    notes = data.get("notes", "")
    notes = "" if notes is None else notes
    if not isinstance(notes, str):
        raise PlateValidationError("notes must be a string")

    return {
        "name": name.strip(),
        "boundary_m": boundary,
        "main_entrance_m": entrance,
        "edge_types": edge_types,
        "north_deg": north,
        "street_edges": street,
        "notes": notes,
    }


def area_m2(plate):
    return round(Polygon(plate["boundary_m"]).area, 4)


def _with_area(plate):
    return {**plate, "area_m2": area_m2(plate)}


# ── file ─────────────────────────────────────────────────────────────────────

def _empty():
    return {"schema_version": SCHEMA_VERSION, "units": "m", "frame": "canvas, y down", "plates": []}


def _read():
    path = library_path()
    if not path.exists():
        return _empty()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LibraryFileError(f"cannot read {path}: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("plates"), list):
        raise LibraryFileError(f"{path} is not a floor plate library")
    return _migrate(doc)


def _migrate(doc):
    """Schema 1 -> 2: road_edge (index | null) becomes street_edges (list)."""
    for plate in doc["plates"]:
        if "road_edge" in plate:
            road = plate.pop("road_edge")
            plate.setdefault("street_edges", [] if road is None else [road])
    doc["schema_version"] = SCHEMA_VERSION
    return doc


def _write(doc):
    """Atomic: temp file in the same directory, fsync, then rename."""
    path = library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _slug(name):
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "plate"


def _unique_id(name, taken):
    base = _slug(name)
    if base not in taken:
        return base
    k = 2
    while f"{base}-{k}" in taken:
        k += 1
    return f"{base}-{k}"


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ── operations ───────────────────────────────────────────────────────────────

def list_plates():
    with _LOCK:
        doc = _read()
        return {**{k: v for k, v in doc.items() if k != "plates"},
                "plates": [_with_area(p) for p in doc["plates"]]}


def get_plate(plate_id):
    with _LOCK:
        for p in _read()["plates"]:
            if p.get("id") == plate_id:
                return _with_area(p)
    raise PlateNotFoundError(plate_id)


def create_plate(data):
    fields = _validate(data)
    with _LOCK:
        doc = _read()
        plate = {"id": _unique_id(fields["name"], {p.get("id") for p in doc["plates"]}),
                 **fields, "saved": _now()}
        doc["plates"].append(plate)
        _write(doc)
        return _with_area(plate)


def update_plate(plate_id, data):
    """Replace a plate. The id is kept; any id in the body is ignored."""
    fields = _validate(data)
    with _LOCK:
        doc = _read()
        for i, p in enumerate(doc["plates"]):
            if p.get("id") == plate_id:
                plate = {"id": plate_id, **fields, "saved": _now()}
                doc["plates"][i] = plate
                _write(doc)
                return _with_area(plate)
    raise PlateNotFoundError(plate_id)


def delete_plate(plate_id):
    with _LOCK:
        doc = _read()
        kept = [p for p in doc["plates"] if p.get("id") != plate_id]
        if len(kept) == len(doc["plates"]):
            raise PlateNotFoundError(plate_id)
        doc["plates"] = kept
        _write(doc)
