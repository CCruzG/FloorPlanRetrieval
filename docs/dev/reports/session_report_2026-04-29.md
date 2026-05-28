# Development Session Report — 29 April 2026

## Topic: Interactive Plan Editing Tools

---

### Context

The retrieval tool allows users to draw a boundary, retrieve similar floor plans, and manually adjust room layouts to fit the target site. Today's session added the interactive editing layer that makes this workflow practical.

---

### Features Implemented

#### 1. Canvas visibility palette
A floating toggle bar was added at the bottom of the canvas with four checkboxes: plan boundary, windows, internal doors, and common area (living/dining). All draw functions were extended with an `opts` parameter so each layer can be switched on/off without restructuring the rendering pipeline.

#### 2. Room selection and move
Clicking inside any non-common-area room polygon selects it (amber highlight) and starts a drag. The room polygon vertices are translated by the mouse delta, converted from canvas pixels to normalised plan coordinates via the same transform used for rendering (rotation, mirror, offset all respected). Common-area rooms (types 0 and 4) are intentionally excluded from interaction.

#### 3. Edge-based resize
Hovering within 8 px of a horizontal or vertical room edge shows an `ns-resize` or `ew-resize` cursor and highlights the edge in amber. Dragging that edge moves only the vertices sharing its coordinate value, resizing the room independently along one axis. Diagonal edges are skipped since floor plans in this dataset are axis-aligned.

#### 4. Entrance drag
The entrance bbox (`[x1, y1, x2, y2]`) can be grabbed and moved as a unit. A dedicated hit-test with 6 px padding handles the thin entrance rectangle. The entrance takes drag priority over room bodies.

#### 5. Displacement arrows
On mouse-up after any successful move (room or entrance), if the centroid actually shifted, a red dashed arrow is recorded from the original centroid position to the new one. Arrows accumulate across multiple edits and are drawn on top of the plan at all times, giving a clear visual record of what was moved and by how much. Arrows are cleared when a new result is selected or a new retrieval is run.

---

### Bug Fixed Mid-session

Room dragging was silently broken because `hitTestRooms` returned `mode:'body'` for interior hits while the drag handler only acted on `mode==='move'`. The fix normalises the mode to `'move'` at drag-start time.

---

### Scope Deliberately Excluded

- Overlap detection between rooms after manual edits
- Snapping moved rooms to adjacent room edges
- Undo / redo
