---
title: Floor Plate Harness — Design Specification
project: project-44
created: 2026-05-27
status: working-draft
depends-on: floor-plate-pipeline-concept.md, FloorPlanRetrieval_refactor-plan.md
---

# Floor Plate Harness — Design Specification

## Purpose

This document specifies the architecture of the multi-agent harness for floor plate generation, to be implemented after the app refactor (FloorPlanRetrieval_refactor-plan.md). It is written for the coding agent that will build the harness, and should be read alongside `floor-plate-pipeline-concept.md`, which captures the conceptual architecture and the research framing.

---

## Pipeline Overview

The user draws a floor plate boundary and selects a target unit mix. The harness executes through three stages separated by two user gates. Between the gates, everything is automatic.

```
User input: boundary polygon + unit mix target
         │
         ▼
┌─────────────────────────────┐
│  STAGE 1: TYPOLOGY          │
│  Head Architect proposes    │
│  2–3 typology options with  │
│  schematic previews         │
└────────────┬────────────────┘
             │
     ┌───────▼────────┐
     │  USER GATE 1   │
     │  Select        │
     │  typology      │
     └───────┬────────┘
             │
┌────────────▼────────────────┐
│  STAGE 2: CORE +            │
│  CIRCULATION PLACEMENT      │
│  Parametric placement of    │
│  vertical cores + corridor  │
└────────────┬────────────────┘
             │
┌────────────▼────────────────┐
│  STAGE 3: UNIT SUBDIVISION  │
│  Assign unit types to bays, │
│  draw party walls, assign   │
│  typed edges                │
└────────────┬────────────────┘
             │
     ┌───────▼────────┐
     │  USER GATE 2   │
     │  Approve floor │
     │  plate layout  │
     └───────┬────────┘
             │
┌────────────▼────────────────┐
│  RETRIEVAL + FITTING +      │
│  ASSESSMENT (parallel,      │
│  one agent per unit)        │
└────────────┬────────────────┘
             │
┌────────────▼────────────────┐
│  COORDINATION LAYER         │
│  Cross-unit checks          │
└────────────┬────────────────┘
             │
             ▼
         Output
```

---

## Head Architect Role

The head architect is the orchestrating agent. It does not execute geometry operations directly — it calls tools and acts on the results.

Responsibilities:
- Receive the floor plate input and run feasibility checks for each typology candidate
- Generate the Gate 1 preview data for each viable typology (speculative execution)
- After Gate 1, direct Stage 2 and Stage 3 tool execution
- Surface unresolved constraint violations before Gate 2 so the user can act on them
- After Gate 2, dispatch retrieval agents (one per unit polygon) with their requirements
- Receive assessment results and pass them to the coordination layer
- Report final output or flag blocking conflicts

The head architect receives proposals from specialist agents and redistributes information among them. It does not bypass the user gates.

---

## Candidate State Management

Gate 1 requires the head architect to show previews for 2–3 typology options *before* the user commits. This means Stages 2 and 3 must run speculatively for each candidate.

**Implementation pattern:**

Each candidate typology produces an independent `FloorPlateState` object (see schema below). The head architect runs Stages 2–3 for each candidate in parallel using a copy of the input boundary. Results are held in a candidate array. When the user selects a typology at Gate 1, the corresponding candidate state becomes the working state. The other candidates are discarded.

The harness must not commit writes to the active floor plate state until Gate 1 is resolved.

---

## Three-Tier Tool Framework

All tools in this harness follow the three-tier model developed in the harness playground:

**Tier 1 — Check tools (assessment)**
Read-only. Called by the harness, not by the agent. Run after every execution tool call and at each stage transition. Return a structured compliance report: list of passing requirements, list of violations with specific values, list of blocked items (violations that cannot be resolved with available tools). The agent reads this report as input to its next decision.

**Tier 2 — Planning tools**
Read-only. Called by the agent before any execution tool call. Return the information the agent needs to choose tool parameters correctly: candidate positions, expected outcomes, constraint margins. Encode domain conventions — the agent should not need to know about coordinate systems, unit conversions, or typical span ranges.

**Tier 3 — Execution tools**
Modify the floor plate state. Every execution tool must be preceded by the appropriate planning tool call. Execution tools return the updated state and trigger a harness-run check immediately after.

The agent also has access to `report_blocked` — a legitimate output path when the harness is under-tooled or a constraint cannot be resolved. Frame this as a useful result, not a failure.

---

## Stage 1: Typology Selection

### Requirements

These requirements are checked by the harness against the raw boundary polygon before typology candidates are proposed. A typology fails these checks and is excluded from the preview.

| Requirement | Check | Typical threshold |
|---|---|---|
| Minimum clear depth for typology | Measure perpendicular distance across boundary at narrowest point | Double-loaded: ≥14m; Single-loaded: ≥8m; Point block: aspect ratio < 1.8:1 |
| Sufficient GFA | `boundary_area × efficiency_factor ≥ sum(min_unit_areas)` | Efficiency factor ~0.75–0.80 |
| Minimum external edge length | At least one edge must be classifiable as external | Detect from boundary orientation and adjacency context |
| Unit mix achievable | Can the boundary accommodate the requested number of each unit type at minimum sizes? | Per-type minimums in room_types.json |

### Tool Catalog — Stage 1

**Planning tools:**

`analyse_boundary(boundary_polygon) → BoundaryAnalysis`
Returns: area, dimensions (min/max span, aspect ratio), orientation (principal axis angle), perimeter length, edge count, convexity score. This is the first tool called for every new floor plate.

`evaluate_typology_fit(boundary, typology, unit_mix) → TypologyFitReport`
Returns: feasibility (pass/fail), estimated unit count achievable per type, estimated efficiency, gross corridor area, estimated external facade per unit, constraint violations. Called once per typology candidate.

`propose_typology_options(boundary, unit_mix) → TypologyOptionList`
Returns: up to 3 typology candidates that passed feasibility checks, each with a schematic description: corridor axis, core count and positions (approximate), unit bay dimensions, typed edge distribution. This is what populates Gate 1 previews.

**Check tools (harness-run):**

`check_typology_viability(boundary, typology_option) → ComplianceReport`
Confirms the selected typology against all Stage 1 requirements. Run after Gate 1 before Stage 2 begins.

**Execution tools:** none at this stage. Stage 1 is analysis only.

---

## Stage 2: Core and Circulation Placement

### Requirements

| Requirement | Rule |
|---|---|
| Maximum travel distance to core | ≤30m measured along corridor from any unit entrance |
| Core footprint | Minimum 20m² (stair + lift + lobby); must be wholly within boundary |
| Structural grid alignment | Core edges align to structural grid (column spacing 6–9m, configurable) |
| Corridor minimum width | 1.2m clear; 1.5m if primary egress |
| Corridor connectivity | Corridor must connect all cores and provide access to all unit bays |
| Boundary containment | Cores and corridor must be wholly within boundary with ≥0.3m clearance |

### Tool Catalog — Stage 2

**Planning tools:**

`compute_structural_grid(boundary, typology, span_range) → StructuralGrid`
Returns: grid line positions (x and y), column positions, recommended bay widths along corridor axis. `span_range` defaults to [6.0, 9.0] metres.

`propose_core_positions(boundary, typology, structural_grid, unit_mix) → CorePositionList`
Returns: candidate core positions (as polygons), travel distances from each position to boundary extremes, coverage zones. Multiple candidates if more than one core is needed.

`plan_corridor(boundary, typology, cores, structural_grid) → CorridorPlan`
Returns: corridor polygon, width at each segment, connectivity graph, travel distances, any pinch points (width < minimum).

**Execution tools:**

`place_core(floor_plate_state, core_polygon) → FloorPlateState`
Adds a core polygon to the state. Triggers check immediately after.

`place_corridor(floor_plate_state, corridor_polygon) → FloorPlateState`
Adds the corridor polygon to the state. Triggers check immediately after.

`adjust_corridor_width(floor_plate_state, segment_id, delta_metres) → FloorPlateState`
Widens or narrows a corridor segment. Triggers check immediately after.

**Check tools (harness-run):**

`check_circulation(floor_plate_state) → ComplianceReport`
Checks all Stage 2 requirements. Run after each execution tool call and before Stage 3 begins.

---

## Stage 3: Unit Subdivision

### Requirements

| Requirement | Rule |
|---|---|
| Minimum gross area by unit type | 1-bed: 45m²; 2-bed: 65m²; 3-bed: 85m² (configurable) |
| Minimum frontage | Width of unit along corridor-facing edge. 1-bed: 4.5m; 2-bed: 5.5m; 3-bed: 6.5m |
| Minimum unit depth | 6.0m from corridor-facing edge to external wall |
| External edge access | Every unit must have at least one external edge |
| No residual areas | All floor plate area outside cores and corridor must be assigned to a unit |
| Unit mix compliance | Subdivision must match the approved unit mix (±1 unit per type) |
| Party wall alignment | Party walls should align with structural grid columns where possible |

### Tool Catalog — Stage 3

**Planning tools:**

`compute_unit_bays(floor_plate_state, structural_grid, unit_mix) → BayAssignmentPlan`
Returns: bay widths along corridor, recommended unit type per bay, total area per bay, shortfalls (bays too narrow for requested type), surplus area. This is the key planning call before any subdivision execution.

`evaluate_party_wall_position(floor_plate_state, position, axis) → WallPositionReport`
Returns: unit areas on each side, frontage on each side, distance to nearest structural grid line, typed edge classification for each resulting unit.

**Execution tools:**

`draw_party_wall(floor_plate_state, wall_line, unit_type_left, unit_type_right) → FloorPlateState`
Adds a party wall to the state and assigns unit types to the resulting polygons. Triggers check immediately after.

`assign_typed_edges(floor_plate_state, unit_id) → FloorPlateState`
Classifies each edge of a unit polygon as `external`, `shared_wall`, or `facing_corridor` based on adjacency with boundary, other units, and corridor. Should be called after all party walls are placed.

`reassign_unit_type(floor_plate_state, unit_id, new_type) → FloorPlateState`
Changes the type label on an existing unit polygon. Does not change geometry. Triggers check.

**Check tools (harness-run):**

`check_subdivision(floor_plate_state) → ComplianceReport`
Checks all Stage 3 requirements for all units. Run after each execution tool call and before Gate 2.

`check_typed_edges(floor_plate_state) → ComplianceReport`
Verifies that typed edge assignments are internally consistent: no unit has an entrance possibility on a shared_wall edge; every unit has at least one external edge; every unit has exactly one facing_corridor edge (for slab typologies).

---

## User Gate 2 — Subdivision Approval

Before Gate 2, the harness presents:

- The floor plate rendered with all unit polygons, labelled by type and area
- Typed edges colour-coded: external (blue), shared_wall (grey), facing_corridor (yellow)
- The compliance report from `check_subdivision` and `check_typed_edges`
- Any unresolved violations or blocked items, clearly listed

The user may:
- **Approve** — retrieval begins for all units in parallel
- **Request adjustment** — specify a change (e.g. "swap the corner bay to a 1-bed", "widen unit 3 by half a metre"). The head architect executes the adjustment using execution tools and re-runs checks before presenting the updated layout.
- **Reject and restart** — return to Gate 1 to select a different typology

---

## Retrieval, Fitting, and Assessment

After Gate 2, the head architect dispatches one retrieval agent per unit polygon. Each agent receives:

- The unit polygon (boundary geometry)
- Typed edge classifications
- Unit type and target room requirements (from room_types.json)
- The retrieval requirements (room count, areas, entrance position)

Each agent runs the existing retrieval → fitting → assessment pipeline (as described in floor-plate-pipeline-concept.md). This phase is parallel and largely independent.

**Assessment requirements per unit:**

| Check | Rule |
|---|---|
| Entrance on corridor edge | The plan's entrance must be placed on the `facing_corridor` edge of the unit polygon |
| Room containment | All rooms must be within the unit polygon boundary |
| No openings on shared walls | Doors and windows must not be placed on `shared_wall` edges |
| Windows on external edges only | Window placements restricted to `external` edges |
| Room size compliance | All rooms meet minimum area requirements for the unit type |
| Accessibility | All rooms reachable from entrance without passing through another room |

---

## Coordination Layer

After all units complete assessment, the coordination layer runs cross-unit checks.

| Check | Rule |
|---|---|
| Wet area stacking | Bathrooms and kitchens should align vertically across adjacent units (shared plumbing walls) |
| Structural grid consistency | Structural walls across units align with the grid from Stage 2 |
| Corridor access confirmed | Every unit entrance is accessible from the corridor without crossing another unit |
| Shared wall conflicts | No openings on either side of a shared wall |

Violations are flagged visually and reported per unit. The harness does not auto-resolve coordination conflicts — they are surfaced for manual resolution in a subsequent cycle.

---

## Floor Plate Schema

The floor plate schema extends the existing unit schema. The existing boundary + rooms representation is preserved; the floor plate adds a wrapping structure.

```json
{
  "floor_plate": {
    "id": "fp-001",
    "boundary": [[x, y], ...],
    "typology": "double_loaded_slab",
    "structural_grid": {
      "x_lines": [0.0, 6.5, 13.0, 19.5],
      "y_lines": [0.0, 14.2],
      "column_positions": [[x, y], ...]
    },
    "cores": [
      { "id": "core-0", "polygon": [[x, y], ...], "type": "stair_lift" }
    ],
    "corridors": [
      { "id": "corr-0", "polygon": [[x, y], ...], "width_min": 1.4 }
    ],
    "units": [
      {
        "id": "unit-0",
        "type": "2bed",
        "polygon": [[x, y], ...],
        "typed_edges": [
          { "edge_index": 0, "type": "facing_corridor" },
          { "edge_index": 1, "type": "shared_wall" },
          { "edge_index": 2, "type": "external" },
          { "edge_index": 3, "type": "shared_wall" }
        ],
        "retrieved_plan": { ... },
        "assessment": { "valid": true, "violations": [] }
      }
    ],
    "unresolved_regions": [],
    "coordination_report": {
      "wet_stacking": "pass",
      "structural_alignment": "pass",
      "shared_wall_conflicts": []
    }
  }
}
```

`typed_edges` references edges by index in the unit `polygon` array (edge i connects vertex i to vertex i+1).

---

## Notes for the Coding Agent

These are drawn directly from the harness playground lessons and apply with full force here.

**On tool descriptions:**
Specify failure modes, not just capabilities. For every tool, ask: what would a well-reasoning model wrongly attempt to use this for? Foreclose that interpretation in the description. `draw_party_wall` does not move existing walls — if the model wants to adjust a wall position, it must call `evaluate_party_wall_position` on the new location and then create a new wall. Make this explicit.

**On the one-tool-per-response constraint:**
State it prominently at the top of the head architect system prompt. A model that can simulate tool results will do so whenever multi-step execution feels natural. The floor plate pipeline has many sequential steps and is particularly vulnerable to narrative completion (the model calls one tool then fabricates the rest of the workflow).

**On the report_blocked escape hatch:**
Frame it as a useful output, not a fallback. If the corridor is too narrow to satisfy travel distance requirements with the available boundary geometry, that is a meaningful design result — not a failure. The model should be able to report this cleanly without constructing motivated arguments for why an inadequate tool call might work.

**On units and scale:**
All tool interfaces should accept and return values in metres. Do not expose normalised or pixel coordinates to the model. The conversion between the internal canvas coordinate system and metres belongs inside the tool, not in the model's reasoning. (This is L7 from the harness playground — the model has no intuition for what normalised units mean in real space.)

**On state injection:**
After every tool call, the harness should recompute and inject the full current floor plate summary: which stages are complete, which units have been placed, which requirements are passing, which are violated. Do not rely on the model to track state across tool calls.

**On model selection:**
Use a model with native structured tool use support. The floor plate pipeline has a long tool catalog and a multi-stage workflow — a model that produces tool calls in inconsistent formats will generate accumulating parser complexity that becomes the dominant source of bugs. (L10 from the harness playground.)

**On the geometry coordinate system:**
Document the coordinate convention in every geometry function. The existing app uses screen coordinates (y-down). The floor plate schema should declare its convention explicitly. Normal direction computation must detect winding order from signed area and flip accordingly. (L6 from the harness playground.)

**On `getActiveFPState()` and the retrieval results array:**
The existing `getActiveFPState()` returns a live reference into `results[]`. Callers that store the `fp` field and later mutate it are directly mutating the results array — this is intentional for the drag editor. When the floor plate harness references a retrieved plan, be explicit about whether it holds a copy or a reference, and document this in JSDoc. Do not change this behaviour without understanding all callers.

---

## Developer Mode

The app is a prototype, not a production tool. During harness development, it serves as the visual sandbox for testing the tool layer. A developer mode makes this explicit and keeps the test scaffolding out of the normal user experience.

### Activation

A URL parameter activates developer mode: `?dev=true`. No UI changes in normal use. The parameter can be stripped by the coding agent when the harness moves toward production without touching any logic. A keyboard shortcut (e.g. `Ctrl+Shift+D`) can toggle it at runtime as a convenience.

### Developer Panel

When dev mode is active, a floating sidebar renders alongside the existing canvas. It has three sections:

**Tool call panel.** A list of all harness tools, grouped by stage. Each tool has editable parameter inputs (pre-populated with sensible defaults from the current `FloorPlateState`) and a Run button. Tool output is displayed inline — structured JSON with pass/fail fields colour-coded. This is the primary interface for testing tools in isolation before the head architect is connected.

**State inspector.** The full `FloorPlateState` as a collapsible JSON tree, live-updated after every tool call. Shows which stages are complete, how many units are placed, and the current compliance status of each. Includes a Copy button for pasting state into test fixtures.

**Compliance log.** The output of the most recently run check tool, displayed as a requirement-by-requirement table: requirement name, expected value, actual value, pass/fail. Violations highlighted in red. Blocked items (cannot be resolved with available tools) in amber. This is what the head architect reads as input — seeing it directly makes prompt design much easier.

### Canvas Floor Plate Overlay

When dev mode is active and a `FloorPlateState` exists, the canvas renders the floor plate overlay on top of (or instead of) the single-plan view. This is a second rendering path in `canvas-renderer.js` that reads from `FloorPlateState` rather than `results[]`.

The overlay renders:
- Structural grid lines — faint dashed lines at computed column positions
- Core polygons — hatched fill, labelled "Core"
- Corridor polygon — semi-transparent fill, distinct colour
- Unit polygons — outlined, labelled with unit type and area in m²
- Typed edges — colour-coded on each unit polygon: external (blue), shared_wall (grey), facing_corridor (yellow)
- Unresolved regions — filled red with low opacity

The overlay is togglable within dev mode (keyboard shortcut or checkbox in the panel) so the developer can switch between the floor plate view and the existing plan retrieval view during a session.

### Implementation Notes

`canvas-renderer.js` must be designed with this second rendering path in mind from the start of Phase 6. Even if the floor plate overlay is a stub during the refactor, the module structure should accommodate it — a separate `renderFloorPlate(floorPlateState)` function, not an extension of the existing `render()` function.

`state.js` should include a `floorPlateState` field from the start, initialised to `null`. The canvas renderer checks for its presence before deciding which rendering path to use.

The dev panel itself is conditionally rendered in `main.js` when `?dev=true` is detected. It has no dependencies on the head architect or the retrieval pipeline — it talks directly to the tool functions.

---

## Development Sequence

The rendering and inspection layer must exist before any tool is worth testing, because the visual feedback *is* the test. Debugging a party wall placement error through LLM reasoning traces is much harder than seeing the wall draw wrong on the canvas. The sequence reflects this: infrastructure first, tools second, reasoning layer last.

1. **FloorPlateState schema** — define the data structure and add a `floorPlateState: null` slot to `state.js`
2. **Canvas floor plate overlay** — `renderFloorPlate()` in `canvas-renderer.js`; typed edge colour coding, unit labels, grid lines, cores, corridor
3. **Developer mode panel** — URL param activation, tool call panel (stubs), state inspector, compliance log; wired into `main.js`
4. **Stage 1 tools** — `analyse_boundary`, `evaluate_typology_fit`, `propose_typology_options`; immediately testable in dev panel; Gate 1 UI (typology preview selector)
5. **Stage 2 tools** — `compute_structural_grid`, `propose_core_positions`, `plan_corridor`, `place_core`, `place_corridor`, `adjust_corridor_width`; check tools; visual verification in canvas overlay
6. **Stage 3 tools** — `compute_unit_bays`, `evaluate_party_wall_position`, `draw_party_wall`, `assign_typed_edges`, `reassign_unit_type`; check tools; Gate 2 UI
7. **Head architect prompt and harness loop** — system prompt, tool routing, state injection after every call, `report_blocked` handling; connects to the tool layer already tested in steps 4–6
8. **Retrieval agent dispatch** — parallel dispatch per unit using existing retrieval pipeline
9. **Assessment and coordination layer** — per-unit checks, cross-unit checks, conflict visualisation

Steps 1–6 are entirely testable without a live model. The developer drives each tool call manually through the dev panel and verifies the result visually. Step 7 is the first step that requires the head architect — by that point, the tool layer should have no geometry bugs and the system prompt can be written against known-correct tool behaviour.
