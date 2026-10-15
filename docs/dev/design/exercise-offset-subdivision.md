---
title: Exercise — Inward Offset Floor Plate Subdivision
project: project-44
created: 2026-05-28
status: ready-for-coding-agent
context: floor-plate-harness-design.md
---

# Exercise: Inward Offset Floor Plate Subdivision

## Purpose

This is a self-contained development exercise for the coding agent. It implements the first parametric subdivision mechanism for the floor plate harness, within the existing app's developer mode.

The exercise produces three things:
- A geometry algorithm (`offset_polygon`) that drives all tools in this exercise
- Three harness tools (planning, execution, check)
- A dev mode UI panel for interactively testing the tools

This exercise does not connect to the head architect or the retrieval pipeline. It is a tool-layer exercise, testable entirely by hand in dev mode.

---

## Core Concept

Offsetting all edges of a floor plate boundary polygon inward by a uniform depth `d` produces three zones simultaneously:

1. **Inner polygon** — the enclosed corridor or core area
2. **Apartment strips** — the annular region between the boundary and the inner polygon, one strip per face
3. **Corner bays** — naturally oversized zones at each corner of the boundary, produced by the intersection geometry of adjacent offset edges

The shape and viability of the inner polygon characterises the building typology. The offset depth determines which configuration emerges.

---

## Configuration Types

For a given boundary and depth `d`, the offset produces one of four configurations:

| Configuration | Condition | Description |
|---|---|---|
| `double_loaded` | Inner polygon is valid and both primary dimensions are viable | Two apartment rows facing each other across an enclosed corridor. Typical for rectangular boundaries with sufficient width. |
| `single_loaded` | Inner polygon collapses along one axis before the other | One apartment row with the corridor pressed against the opposite façade. Occurs in narrow boundaries where the offset from opposing short sides overlaps before reaching depth `d`. |
| `point_block` | Inner polygon is valid and roughly equidimensional (aspect ratio < 2:1) | Apartment units arranged around a central core. Typical for compact or roughly square boundaries. |
| `degenerate` | Inner polygon has zero or negative area | The boundary is too narrow or too small for the requested depth. No viable subdivision possible at this depth. |

The planning tool detects which configuration a given depth produces and reports it before any execution.

---

## Algorithm: `offset_polygon(boundary, depth)`

Input: a polygon as an ordered array of `[x, y]` vertices (assumed anticlockwise in screen coordinates, i.e. y-down), and a scalar depth `d` in metres.

Steps:

1. **Compute inward normals.** For each edge (vertex `i` to vertex `i+1`), compute the edge direction vector, then rotate 90° toward the interior. In screen coordinates (y-down), the interior normal for an anticlockwise polygon is obtained by rotating the edge direction clockwise: `normal = [dy, -dx]` (normalised). Verify winding order from signed area; if polygon is clockwise, flip all normals.

2. **Offset each edge.** Translate each edge inward along its normal by `d`. This produces a set of offset edge lines (not yet connected into a polygon).

3. **Intersect adjacent offset edges.** For each pair of adjacent offset edges, compute their intersection point. This is the corresponding vertex of the inner polygon.

4. **Detect degeneracy.** If any pair of adjacent offset edges is parallel (no intersection) or if the intersection is outside the original boundary, the inner polygon is degenerate at this vertex. Record which vertices are degenerate.

5. **Classify configuration.** After computing all intersections:
   - If no degeneracies: the inner polygon is valid. Compute its aspect ratio. If < 2:1, classify as `point_block`; otherwise `double_loaded`.
   - If degeneracies exist along one pair of opposing sides only: classify as `single_loaded`. The degenerate axis identifies which dimension collapsed.
   - If all vertices are degenerate: classify as `degenerate`.

6. **Return.** The inner polygon vertices (or degenerate indicator), the configuration type, the apartment strips as individual polygons (one per boundary face, clipped to the space between the boundary edge and the corresponding offset edge), and the corner bay polygons.

**Edge case: near-degenerate inner polygon.** An inner polygon dimension below a minimum corridor width (1.2m default) should be classified as degenerate even if geometrically valid. Add this check after step 4.

**Edge case: non-convex boundary.** Inward offsets on a non-convex polygon can produce self-intersecting offset edges. For this exercise, restrict input to convex polygons only. Add a convexity check at the start and return an error if the boundary is non-convex. Extend to non-convex boundaries in a future exercise.

---

## Tool Set

### Planning Tool

**`propose_offset_depth(boundary, depth_range, min_corridor_width?) → OffsetProposal`**

Parameters:
- `boundary`: polygon vertices in metres
- `depth_range`: `{ min: number, max: number }` — acceptable apartment depth range in metres (e.g. `{ min: 5.0, max: 9.0 }`)
- `min_corridor_width`: minimum viable inner polygon dimension in metres (default 1.2)

Algorithm: Sample depths at regular intervals across `depth_range` (at minimum: `min`, midpoint, `max`, plus the depth at which the configuration changes). For each depth, run `offset_polygon` and record the configuration type.

Returns:
```json
{
  "candidates": [
    {
      "depth": 6.0,
      "configuration": "double_loaded",
      "inner_polygon": [[x, y], ...],
      "inner_polygon_dimensions": { "width": 2.0, "length": 8.0, "aspect_ratio": 4.0 },
      "apartment_strips": [
        { "face_index": 0, "polygon": [[x,y],...], "depth": 6.0, "length": 20.0, "area": 120.0 },
        { "face_index": 1, "polygon": [[x,y],...], "depth": 6.0, "length": 14.0, "area": 84.0 },
        { "face_index": 2, "polygon": [[x,y],...], "depth": 6.0, "length": 20.0, "area": 120.0 },
        { "face_index": 3, "polygon": [[x,y],...], "depth": 6.0, "length": 14.0, "area": 84.0 }
      ],
      "corner_bays": [
        { "corner_index": 0, "polygon": [[x,y],...], "area": 36.0 }
      ],
      "viable": true,
      "notes": []
    },
    {
      "depth": 8.0,
      "configuration": "single_loaded",
      "inner_polygon": null,
      "collapsed_axis": "short",
      "apartment_strips": [
        { "face_index": 0, "polygon": [[x,y],...], "depth": 8.0, "length": 20.0, "area": 160.0, "viable": true },
        { "face_index": 2, "polygon": [[x,y],...], "depth": 8.0, "length": 20.0, "area": 160.0, "viable": false, "reason": "offset collision" }
      ],
      "viable": true,
      "notes": ["Single-loaded configuration. Corridor zone is a strip against the short façades."]
    }
  ],
  "critical_depth": 7.0,
  "critical_depth_note": "Below 7.0m: double_loaded. At or above 7.0m: single_loaded."
}
```

The `critical_depth` is the depth at which the configuration changes. This is the most actionable output for the agent: it tells the agent which side of the threshold each candidate depth is on.

---

### Execution Tool

**`apply_offset_split(floor_plate_state, depth, configuration) → FloorPlateState`**

Parameters:
- `depth`: the chosen depth in metres (must come from a `propose_offset_depth` call)
- `configuration`: the expected configuration type (checked against actual result; if mismatch, returns error)

Runs `offset_polygon` with the chosen depth, then writes to `FloorPlateState`:
- Sets `floor_plate.corridor_zone` to the inner polygon (or corridor strip for single-loaded)
- Sets `floor_plate.apartment_strips[]` to the face strips
- Sets `floor_plate.corner_bays[]` to the corner bay polygons
- Sets `floor_plate.configuration` to the configuration type string

Does not yet create unit polygons — that is Stage 2 (bay subdivision). Does not assign typed edges — that happens after unit subdivision.

Triggers check tool immediately after writing.

---

### Check Tool (harness-run)

**`check_offset_split(floor_plate_state) → ComplianceReport`**

Run automatically after `apply_offset_split`. Checks:

| Requirement | Rule |
|---|---|
| Inner polygon viability | For `double_loaded` and `point_block`: inner polygon min dimension ≥ `min_corridor_width` |
| Apartment strip depth | Each viable strip: depth within `depth_range` |
| Area coverage | Sum of apartment strips + corner bays + corridor zone = total boundary area (within floating point tolerance of 0.01 m²) |
| Convexity | Each apartment strip and corner bay is a convex polygon (required for Stage 2 bay subdivision) |
| Minimum strip length | Each viable strip long dimension ≥ 4.5m (minimum single-unit bay width) |

Returns structured report with pass/fail per requirement and actual values.

---

## Dev Mode UI

When `?dev=true` is active and a floor plate boundary exists, add an **Offset Subdivision** section to the dev panel.

**Controls:**
- Depth range inputs: `min` and `max` in metres (defaults: 5.0 and 9.0)
- Min corridor width input (default: 1.2m)
- **Run Planning Tool** button — calls `propose_offset_depth` and populates the results section
- Candidate list — one row per candidate depth returned, showing: depth, configuration type, viability, notes. Clicking a row previews that candidate on the canvas without committing.
- **Apply** button (enabled only when a candidate is selected) — calls `apply_offset_split` with the selected depth

**Canvas overlay (preview mode — before Apply):**
- Inner polygon: dashed outline, no fill
- Apartment strips: light fill, face index label
- Corner bays: slightly different fill colour, labelled "corner"
- Configuration type shown as text overlay in centre of boundary

**Canvas overlay (after Apply):**
- Inner polygon: solid outline, labelled "Corridor" with dimensions
- Apartment strips: filled, labelled with depth × length in metres
- Corner bays: filled, labelled with area in m²
- Compliance report displayed in the compliance log section

**Depth slider (optional enhancement):** A slider over the `depth_range` that live-previews the offset on the canvas as it moves, without calling the planning tool. At the `critical_depth` position, the canvas switches configuration type visually. This is useful for intuiting the geometry before running the full planning tool.

---

## Success Criteria

The exercise is complete when:

1. `propose_offset_depth` returns correct geometry and configuration types for:
   - A 14×20m rectangle at depth 5m (expect `double_loaded`)
   - A 14×20m rectangle at depth 8m (expect `single_loaded`, collapsed on short axis)
   - A 10×10m square at depth 3m (expect `point_block`)
   - A 6×20m rectangle at depth 4m (expect `single_loaded` or `degenerate` — check which)

2. `apply_offset_split` writes the correct zones to `FloorPlateState` and the canvas renders them without error.

3. `check_offset_split` correctly identifies:
   - A passing configuration (all requirements met)
   - A corridor too narrow (inner polygon dimension below minimum)
   - An incomplete area coverage (floating point tolerance test)

4. The dev panel allows selecting between candidates and previewing each on the canvas before committing.

5. All geometry is in metres. No pixel or normalised coordinates appear in tool inputs or outputs.

---

## Out of Scope for This Exercise

- Bay subdivision (splitting apartment strips into unit polygons) — next exercise
- Typed edge assignment — after bay subdivision
- Non-convex boundary polygons — future exercise
- Corridor zone connection to cores/lifts — harness Stage 2
- Head architect integration — harness Stage 7
