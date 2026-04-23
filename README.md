# Floor Plan Retrieval Tool

A browser-based tool for retrieving similar floor plans by drawing a boundary polygon and specifying room requirements. Results are returned from a pre-built index of 2000 plans and rendered with room colours, door/window markers, and an optional room adjacency graph.

---

## Requirements

- Python 3.10+
- A Unix-like environment (Linux / macOS) or WSL on Windows

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd 02_Retrieval_v02

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install flask numpy matplotlib shapely
```

---

## Running the server

```bash
source .venv/bin/activate
python server.py
```

Then open **http://127.0.0.1:5000** in your browser.

To expose the server on the local network (e.g. for a colleague on the same Wi-Fi):

```bash
python server.py --host 0.0.0.0
```

Then open **http://<your-machine-ip>:5000** from another device.

### Server options

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address (`0.0.0.0` for network access) |
| `--port` | `5000` | Port number |
| `--data_dir` | `test_data` | Directory containing floor plan JSON files |
| `--index_bin` | `browser_files/index.bin` | Pre-built descriptor index (float32 binary) |
| `--index_names` | `browser_files/index_names.json` | Index name list |
| `--debug` | off | Enable Flask debug mode |

---

## Using the browser tool

### Drawing a boundary

1. **Click** on the canvas to place boundary vertices.
2. Hold **Shift** while clicking to lock the edge to horizontal or vertical.
3. Hover near a placed vertex to see **projection lines** — the cursor snaps to them automatically.
4. **Double-click**, press **Enter**, or click the first vertex (shown in red) to close the boundary.
5. Press **Esc** to clear and start over.

### Requirements & parameters

- **Bedrooms / bathrooms** — set the desired room counts using the sliders.
- **Results** — number of alternatives to retrieve (1–20).
- **Shape weight** — how much the boundary shape influences retrieval (0 = semantic only).
- **Semantic weight** — how much room composition influences retrieval (0 = shape only).

### Retrieving

Click **retrieve** once the boundary is closed. Results appear in the right panel as thumbnails. Click a thumbnail to display the full plan on the canvas alongside your drawn boundary.

### Room graph

Check **show room graph** to overlay a graph of room adjacencies on the displayed plan. Solid edges indicate door connections; dashed edges indicate wall adjacency.

---

## Boundary library

Boundaries can be saved, reused, and shared for batch studies.

- **Save boundary…** — saves the current closed boundary with a name into browser `localStorage`.
- Click a name in the library list to **load** it back onto the canvas.
- **✎** to rename, **✕** to delete.
- **Export library JSON** — downloads all saved boundaries as `boundary_library.json`.
- **Import library JSON** — merges boundaries from a JSON file into the local library.

The exported format is:
```json
[
  {
    "name": "rectangle-large",
    "vertices": [{"x": 0.2, "y": 0.2}, ...],
    "saved": "2026-04-23T10:00:00.000Z"
  }
]
```

---

## Rebuilding the index

If you have a different dataset, rebuild the index with:

```bash
source .venv/bin/activate

# Build index from a directory of JSON floor plans
python sem_geo_retrieval.py \
  --query test_data/0.json \
  --data_dir test_data \
  --build_index browser_files/index.pkl

# Export to browser-readable format
python export_index.py \
  --index browser_files/index.pkl \
  --out_dir browser_files
```

### Retrieval weights

| Flag | Default | Description |
|---|---|---|
| `--shape_w` | `1.0` | Weight for boundary shape descriptor |
| `--sem_w` | `0.5` | Weight for room semantic descriptor |
| `--augment` | off | Store 8 rotation/mirror variants per plan |
| `--topn` | `5` | Number of results to return |

---

## Floor plan JSON schema

Plans use schema version **2.1.0** with normalised `[0, 1]` coordinates:

```
boundary   — list of [x, y, direction, is_door_vertex]
rooms      — list of {polygon, type (0–8), source_type (0–17), ...}
edges      — list of {room_indices, connection: 0=wall | 1=door}
doors      — list of {id, bbox: [x1,y1,x2,y2], orientation, room_refs}
windows    — list of {id, bbox: [x1,y1,x2,y2], orientation, room_refs}
```

---

## Project structure

```
retrieval_tool.html    Browser UI (single file, no build step)
server.py              Flask server — serves the UI and runs retrieval
sem_geo_retrieval.py   Descriptor computation and index building
export_index.py        Converts .pkl index to browser-readable .bin + .json
adapt.py               Geometric adaptation (not used in current retrieval flow)
browser_files/         Pre-built index (index.bin, index_names.json)
test_data/             Sample floor plan JSON files
```
