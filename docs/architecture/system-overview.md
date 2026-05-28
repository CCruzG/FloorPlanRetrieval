# System Overview — Floor Plan Retrieval Tool

This document describes how the retrieval system works end-to-end: how floor plans are encoded as descriptors, how a user query is handled, and how the distance between the query and each plan in the dataset is computed.

---

## 1. High-level architecture

```
Browser (retrieval_tool.html)
  │
  │  POST /retrieve  {boundary, entrance_point, requirements, weights}
  ▼
Flask server (server.py)
  ├── Compute query descriptors   ──► geo_ent_num_retrieval.py
  ├── Distance search over index  ──► geo_ent_num_retrieval.retrieve()
  ├── Collect & return top-N plans
  │
  │  POST /adapt  {source_plan, target}
  ▼
  └── Proportional centroid fitting  ──► fit_plan.py
```

At startup the server loads two files built offline by `build_index_ent.py`:

| File | Contents |
|---|---|
| `data/index/browser/index.bin` | Float32 binary matrix, shape **(N × 63)** — one 63-dim descriptor per plan |
| `data/index/browser/index_names.json` | Ordered list of plan names, length N |

All N floor plan JSON files from the dataset are also loaded into memory (`plan_cache`) so they can be returned without disk I/O after retrieval.

---

## 2. Floor plan encoding — the 63-dim descriptor

Each floor plan is encoded as a concatenation of three independent sub-descriptors:

```
[ shape (48) | entrance (4) | rooms (11) ]  =  63 floats
```

### 2.1 Shape descriptor (dims 0–47, 48 values)

Encodes the **outline shape** of the boundary polygon as a Fourier signature of the centroid-distance signal.

**Algorithm:**

1. **Normalise winding** — vertices are forced into clockwise order so the signal is consistent regardless of how the polygon was drawn.
2. **Uniform perimeter sampling** — 256 points are sampled at equal arc-length intervals around the boundary.
3. **Centroid-distance signal** — for each sample point $p_i$, compute the Euclidean distance to the perimeter centroid $\bar{p}$:

$$r_i = \|p_i - \bar{p}\|$$

4. **Mean-normalise** — divide all $r_i$ by their mean $\bar{r}$, making the signal scale-invariant.
5. **DFT magnitude spectrum** — take the real FFT of the normalised signal and retain the magnitudes of Fourier coefficients $k = 1 \ldots 48$ (DC component discarded):

$$\text{shape} = \left[ |F_1|, |F_2|, \ldots, |F_{48}| \right]$$

6. **L2-normalise** — the 48-dim vector is divided by its L2 norm.

**Properties:** Invariant to rigid rotation (only magnitudes used), translation (centroid-relative distances), and scale (mean-normalised). The descriptor captures the overall silhouette shape — elongated rectangles, L-shapes, irregular polygons, etc. all produce distinct signatures.

### 2.2 Entrance descriptor (dims 48–51, 4 values)

Encodes **where the front door is** relative to the building shape, and **which face** of the building it is on.

The 4-dim vector is $[\,\text{rel}\_x,\; \text{rel}\_y,\; \cos\theta,\; \sin\theta\,]$:

| Component | Meaning |
|---|---|
| $\text{rel}\_x,\; \text{rel}\_y$ | 2-D vector from the boundary centroid to the closest point on the boundary to the door, **normalised by the mean centroid-to-boundary radius** $\bar{r}$ |
| $\cos\theta,\; \sin\theta$ | Outward unit normal of the boundary edge that the door snaps to — encodes which face (front, back, left, right) the entrance is on |

**Algorithm:**

1. Find the door centre from the plan's `entrance` bounding box (or `door_rects` in legacy plans).
2. Project the door point onto each boundary edge (clamped nearest-point).
3. Select the edge whose nearest point is closest to the door.
4. Compute $\text{rel}\_x = (\text{snap}\_x - \bar{p}\_x) / \bar{r}$, and similarly for $y$.
5. Compute the outward edge normal; flip it if it points inward.

### 2.3 Room-count descriptor (dims 52–62, 11 values)

Encodes the **room composition** of the plan as a histogram of normalised per-type room counts:

| Dim | Room type | Normalised by |
|---|---|---|
| 0 | Living room | 2 |
| 1 | Master bedroom | 1 |
| 2 | Kitchen | 1 |
| 3 | Bathroom | 3 |
| 4 | Dining room | 1 |
| 5 | Child bedroom | 2 |
| 6 | Study | 1 |
| 7 | Second bedroom | 2 |
| 8 | Balcony | 2 |
| 9 | Storage | 2 |
| 10 | Total room count | 15 |

Each value is clamped to $[0, 1]$. This descriptor is fully rotation- and translation-invariant.

---

## 3. Query descriptor computation

When the user draws a boundary and clicks **Retrieve**, the browser POSTs to `/retrieve` with:

```json
{
  "boundary":       [[x, y, dir, is_door], ...],
  "entrance_point": [x, y],
  "requirements":   {"bedrooms": 2, "bathrooms": 1},
  "topn":           5,
  "shape_w":        1.0,
  "ent_w":          2.0,
  "room_w":         1.0
}
```

The server computes three query sub-descriptors using the same functions as the index builder:

| Sub-descriptor | Function | Input |
|---|---|---|
| Shape | `shape_desc(bdry_segs)` | User-drawn boundary vertices |
| Entrance | `entrance_desc_from_point(entrance_point, bdry_segs)` | Door point + boundary |
| Rooms | `_req_to_room_desc(requirements)` | Bedroom / bathroom sliders |

If no entrance is provided, the entrance descriptor is zeroed and its weight is forced to 0.

---

## 4. Distance computation and ranking

### 4.1 Sub-descriptor distances

For each of the N plans in the index, three partial distances are computed:

$$d_\text{shape}^{(i)} = w_\text{shape} \cdot \| q_\text{shape} - \mathbf{s}^{(i)} \|_2$$

$$d_\text{room}^{(i)} = w_\text{room} \cdot \| q_\text{room} - \mathbf{r}^{(i)} \|_2$$

The entrance distance is **rotation-aware** and requires special handling (see §4.2).

### 4.2 Rotation- and reflection-aware entrance distance

A candidate plan may need to be rotated or mirrored to align its entrance direction with the query. The system evaluates **8 orientations** (4 rotations × 2 mirror states) and selects the best one analytically.

**Rotation model.** A 2-D rotation by angle $\alpha$ acts simultaneously on the position components $(r_x, r_y)$ and the normal components $(c_x, c_y)$ of the entrance descriptor:

$$R(\alpha) \cdot [r_x, r_y, c_x, c_y] = [r_x \cos\alpha - r_y \sin\alpha,\ r_x \sin\alpha + r_y \cos\alpha,\ c_x \cos\alpha - c_y \sin\alpha,\ c_x \sin\alpha + c_y \cos\alpha]$$

**Closed-form minimum over all rotations.** Expanding $\|q - R(\alpha) \cdot d\|^2$:

$$\|q - R(\alpha) \cdot d\|^2 = \|q\|^2 + \|d\|^2 - 2(A \cos\alpha + B \sin\alpha)$$

where:

$$A = q_{r_x} d_{r_x} + q_{r_y} d_{r_y} + q_{c_x} d_{c_x} + q_{c_y} d_{c_y}$$
$$B = q_{r_y} d_{r_x} - q_{r_x} d_{r_y} + q_{c_y} d_{c_x} - q_{c_x} d_{c_y}$$

The minimum over all $\alpha$ is:

$$\min_\alpha \|q - R(\alpha) \cdot d\|^2 = \|q\|^2 + \|d\|^2 - 2\sqrt{A^2 + B^2}$$

achieved at $\alpha^* = \arctan2(B, A)$.

**Mirror states.** Mirroring flips the $x$-components of the entrance descriptor: $[-r_x, r_y, -c_x, c_y]$. The same closed-form minimum is computed for the mirrored descriptor using $A_m, B_m$, and the mirror state with the lower distance is chosen.

**Snapping.** The optimal continuous rotation $\alpha^*$ is snapped to the nearest multiple of $\pi/2$ (i.e. $0°, 90°, 180°, 270°$) to enforce axis-aligned orientations. The entrance distance is then recomputed at the snapped angle.

$$d_\text{ent}^{(i)} = w_\text{ent} \cdot \| q_\text{ent} - R(\hat{\alpha}^{(i)}) \cdot d_\text{ent}^{(i)} \|_2$$

### 4.3 Combined distance and ranking

The final distance for plan $i$ is:

$$D^{(i)} = d_\text{shape}^{(i)} + d_\text{ent}^{(i)} + d_\text{room}^{(i)}$$

Plans are ranked by $D^{(i)}$ in ascending order and the top-$N$ results are returned. The best rotation angle and mirror state are also returned so the browser can render each plan at the correct orientation.

---

## 5. Index building

The index is built offline by `build_index_ent.py`:

```bash
python tools/build_index_ent.py --data_dir data/fixtures/plans/test_data --out_dir data/index/browser
```

For each valid plan JSON file:
1. Compute `shape_desc`, `entrance_desc`, `room_count_desc`.
2. Concatenate into a 63-dim row.
3. Stack all rows into a `(N, 63)` float32 matrix and write to `index.bin`.
4. Write the ordered list of plan names to `index_names.json`.

Plans with no boundary or no front door are skipped.

---

## 6. Geometric fitting (`/adapt`)

After retrieval the browser can send a plan to `/adapt`, which calls `fit_plan.py` to map the retrieved plan's rooms into the user-drawn boundary:

1. **Analyse** — extract the source boundary bounding box and room bounding boxes.
2. **Map centroids** — each room's centroid is expressed as a normalised position within the source bounding box and placed at the equivalent position within the target bounding box. Room dimensions are scaled by the same x/y scale factors.
3. **Assemble output** — replace boundary, entrance, and room polygons/bboxes in the plan JSON; clear the common area (living room) polygon since it is implicitly defined by the boundary.

The common area (largest room with `source_type = 0`) is excluded from the centroid mapping because it should fill whatever space the other rooms leave.
