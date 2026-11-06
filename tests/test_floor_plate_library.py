"""Floor plate library tests. Every test points FLOOR_PLATE_LIBRARY at a temp file."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packages.floor_plate_design import library as lib  # noqa: E402

RECT = [[0, 0], [40, 0], [40, 20], [0, 20]]


@pytest.fixture(autouse=True)
def tmp_lib(tmp_path, monkeypatch):
    path = tmp_path / "fp" / "library.json"
    monkeypatch.setenv(lib.ENV_VAR, str(path))
    return path


def plate(**kw):
    d = {"name": "Slab A", "boundary_m": RECT, "main_entrance_m": [20, 0],
         "edge_types": ["external", "shared_wall", "external", "facing_corridor"],
         "north_deg": 30, "street_edges": [2, 3], "notes": "test"}
    d.update(kw)
    return d


def test_default_path_is_repo_file(monkeypatch):
    monkeypatch.delenv(lib.ENV_VAR)
    assert lib.library_path() == lib.REPO_ROOT / "data" / "floor_plates" / "library.json"


def test_empty_library(tmp_lib):
    out = lib.list_plates()
    assert out["plates"] == []
    assert not tmp_lib.exists()


def test_round_trip(tmp_lib):
    made = lib.create_plate(plate())
    assert made["id"] == "slab-a"
    assert made["area_m2"] == 800
    got = lib.get_plate("slab-a")
    for k in ("name", "boundary_m", "main_entrance_m", "edge_types", "north_deg", "street_edges", "notes"):
        assert got[k] == plate()[k] or got[k] == [list(map(float, p)) for p in RECT]
    assert got["boundary_m"] == [[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]]
    assert got["saved"]
    listed = lib.list_plates()
    assert [p["id"] for p in listed["plates"]] == ["slab-a"]
    # file layout; area is not stored
    raw = json.loads(tmp_lib.read_text())
    assert raw["schema_version"] == 2 and raw["units"] == "m" and raw["frame"] == "canvas, y down"
    assert "area_m2" not in raw["plates"][0]
    assert listed["plates"][0]["area_m2"] == 800


def test_defaults():
    p = lib.create_plate({"name": "x", "boundary_m": RECT})
    assert p["north_deg"] == 0 and p["street_edges"] == []
    assert p["main_entrance_m"] is None and p["edge_types"] == [] and p["notes"] == ""


@pytest.mark.parametrize("patch,needle", [
    ({"boundary_m": [[0, 0], [1, 1]]}, "at least 3"),
    ({"boundary_m": [[0, 0], [1, 0], [float("nan"), 1]]}, "finite"),
    ({"boundary_m": [[0, 0], [1, 0], [float("inf"), 1]]}, "finite"),
    ({"boundary_m": [[0, 0], [1, 0], ["a", 1]]}, "finite"),
    ({"boundary_m": [[0, 0], [10, 10], [10, 0], [0, 10]]}, "self-intersecting"),
    ({"boundary_m": [[0, 0], [1, 1], [2, 2]]}, "self-intersecting"),
    ({"boundary_m": [[0, 0], [0, 0], [5, 5], [0, 5]]}, "zero-length"),
    ({"street_edges": [4]}, "out of range"),
    ({"street_edges": [-1]}, "out of range"),
    ({"street_edges": [1.5]}, "integer"),
    ({"edge_types": ["external"]}, "edge_types has 1"),
    ({"edge_types": ["external", "x", "external", "external"]}, "one of"),
    ({"north_deg": float("nan")}, "finite"),
    ({"main_entrance_m": [1]}, "[x, y]"),
    ({"name": "  "}, "name"),
])
def test_validation_rejects(patch, needle, tmp_lib):
    with pytest.raises(lib.PlateValidationError, match=needle.replace("[", r"\[").replace("]", r"\]")):
        lib.create_plate(plate(**patch))
    assert not tmp_lib.exists()  # nothing written


def test_validation_on_update_leaves_file(tmp_lib):
    lib.create_plate(plate())
    before = tmp_lib.read_text()
    with pytest.raises(lib.PlateValidationError):
        lib.update_plate("slab-a", plate(street_edges=[9]))
    assert tmp_lib.read_text() == before


def test_unique_ids():
    ids = [lib.create_plate(plate())["id"] for _ in range(3)]
    assert ids == ["slab-a", "slab-a-2", "slab-a-3"]
    assert lib.create_plate(plate(name="!!!"))["id"] == "plate"


def test_update_keeps_id_and_replaces():
    lib.create_plate(plate())
    out = lib.update_plate("slab-a", plate(name="Renamed", boundary_m=[[0, 0], [10, 0], [10, 5], [0, 5]],
                                           edge_types=[], street_edges=[], north_deg=90))
    assert out["id"] == "slab-a" and out["name"] == "Renamed" and out["area_m2"] == 50
    got = lib.get_plate("slab-a")
    assert got["street_edges"] == [] and got["north_deg"] == 90 and got["edge_types"] == []
    assert len(lib.list_plates()["plates"]) == 1


def test_unknown_id():
    lib.create_plate(plate())
    with pytest.raises(lib.PlateNotFoundError):
        lib.get_plate("nope")
    with pytest.raises(lib.PlateNotFoundError):
        lib.update_plate("nope", plate())
    with pytest.raises(lib.PlateNotFoundError):
        lib.delete_plate("nope")


def test_delete():
    lib.create_plate(plate())
    lib.create_plate(plate(name="B"))
    lib.delete_plate("slab-a")
    assert [p["id"] for p in lib.list_plates()["plates"]] == ["b"]


def test_corrupt_file(tmp_lib):
    tmp_lib.parent.mkdir(parents=True)
    tmp_lib.write_text("{not json")
    with pytest.raises(lib.LibraryFileError):
        lib.list_plates()


def test_atomic_write_failure_keeps_old_file(tmp_lib, monkeypatch):
    lib.create_plate(plate())
    before = tmp_lib.read_text()

    def boom(src, dst):
        raise OSError("disk gone")

    monkeypatch.setattr(lib.os, "replace", boom)
    with pytest.raises(OSError):
        lib.create_plate(plate(name="B"))
    monkeypatch.undo()
    assert tmp_lib.read_text() == before
    assert [f.name for f in tmp_lib.parent.iterdir()] == ["library.json"]  # no temp file left


def test_write_goes_through_temp_then_rename(tmp_lib, monkeypatch):
    seen = []
    real = os.replace

    def spy(src, dst):
        seen.append((Path(src).name, Path(dst).name, Path(src).exists()))
        return real(src, dst)

    monkeypatch.setattr(lib.os, "replace", spy)
    lib.create_plate(plate())
    assert len(seen) == 1
    src, dst, existed = seen[0]
    assert dst == "library.json" and src != dst and src.endswith(".tmp") and existed


def test_http_layer(tmp_lib):
    """Status codes of the Flask routes (the module only; no index load)."""
    from apps.api import server
    c = server.app.test_client()
    assert c.get("/floor_plates").get_json()["plates"] == []
    r = c.post("/floor_plates", json=plate())
    assert r.status_code == 201 and r.get_json()["id"] == "slab-a"
    assert c.post("/floor_plates", json=plate(street_edges=[7])).status_code == 400
    assert "error" in c.post("/floor_plates", json=plate(street_edges=[7])).get_json()
    assert c.post("/floor_plates", data="x").status_code == 400
    r = c.put("/floor_plates/slab-a", json=plate(name="N"))
    assert r.status_code == 200 and r.get_json()["id"] == "slab-a"
    assert c.put("/floor_plates/zzz", json=plate()).status_code == 404
    assert c.put("/floor_plates/slab-a", json=plate(boundary_m=[[0, 0]])).status_code == 400
    assert c.delete("/floor_plates/zzz").status_code == 404
    assert c.delete("/floor_plates/slab-a").status_code == 200
    assert c.get("/floor_plates").get_json()["plates"] == []


def test_street_edges_sorted_and_unique(tmp_lib):
    made = lib.create_plate(plate(street_edges=[3, 0, 3]))
    assert made["street_edges"] == [0, 3]


def test_schema_1_file_is_migrated(tmp_lib):
    old = plate(id="old-a", saved="2026-11-06T00:00:00+11:00")
    del old["street_edges"]
    none = {**old, "id": "old-b", "road_edge": None}
    one = {**old, "road_edge": 2}
    tmp_lib.parent.mkdir(parents=True)
    tmp_lib.write_text(json.dumps({"schema_version": 1, "units": "m", "frame": "canvas, y down",
                                   "plates": [one, none]}))
    assert lib.get_plate("old-a")["street_edges"] == [2]
    assert lib.get_plate("old-b")["street_edges"] == []
    # the next save rewrites the file in the new form, for every plate
    lib.update_plate("old-b", plate())
    doc = json.loads(tmp_lib.read_text())
    assert doc["schema_version"] == 2
    assert not any("road_edge" in p for p in doc["plates"])
    assert doc["plates"][0]["street_edges"] == [2]
