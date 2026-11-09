// Task 1 (docs/pipeline-architecture.md, "Work method — gated tasks"):
// wall-thickness generation + 2D architectural drawing.
//
// Contract: takes the canonical schema.py fixture shape
// { schema_version, label, boundary: {polygon, typed_edges, entrance_point, ...},
//   plan: {rooms, edges, ...} } and draws it per architectural convention —
// wall poché with resolved corners, door openings cut into the wall body.
//
// Wall thickness is computed here, at draw time, from edge.type — never
// stored in the schema.py data. See pipeline-architecture.md "Two-space
// architecture" for why.
//
// Reuses the project's existing ClipperLib offset/boolean pattern
// (see offset-subdivision.js) rather than re-deriving polygon geometry —
// centerline segments are thickened via ClipperOffset (open paths, square
// ends), door gaps are cut via boolean difference, and the whole wall body
// is a single unioned polygon set so corners/junctions resolve automatically.

import './vendor/clipper.js';

const ClipperLib = globalThis.ClipperLib;
if (!ClipperLib) {
  throw new Error('ClipperLib failed to load from local vendor bundle.');
}

const CLIPPER_SCALE = 1e3; // fixture coords are already mm; 1e3 keeps clipper ints well inside safe range

const DEFAULT_THICKNESS_MM = {
  external: 200,
  shared_wall: 150,
  facing_corridor: 200,
  internal: 100,
};

const DEFAULT_DOOR_WIDTH_MM = 900;

function toPath(poly) {
  return poly.map(([x, y]) => ({
    X: Math.round(x * CLIPPER_SCALE),
    Y: Math.round(y * CLIPPER_SCALE),
  }));
}

function fromPath(path) {
  return path.map(({ X, Y }) => [X / CLIPPER_SCALE, Y / CLIPPER_SCALE]);
}

function cleanPaths(paths) {
  const cleaned = ClipperLib.Clipper.CleanPolygons(paths || [], 1);
  return cleaned.filter((path) => Math.abs(ClipperLib.Clipper.Area(path)) > 1e-6 * CLIPPER_SCALE * CLIPPER_SCALE);
}

function unionPaths(paths) {
  if (!paths?.length) return [];
  const clipper = new ClipperLib.Clipper();
  clipper.AddPaths(paths, ClipperLib.PolyType.ptSubject, true);
  const solution = new ClipperLib.Paths();
  clipper.Execute(
    ClipperLib.ClipType.ctUnion,
    solution,
    ClipperLib.PolyFillType.pftNonZero,
    ClipperLib.PolyFillType.pftNonZero,
  );
  return cleanPaths(solution);
}

function differencePaths(subjectPaths, clipPaths) {
  if (!subjectPaths?.length) return [];
  if (!clipPaths?.length) return cleanPaths(subjectPaths);
  const clipper = new ClipperLib.Clipper();
  clipper.AddPaths(subjectPaths, ClipperLib.PolyType.ptSubject, true);
  clipper.AddPaths(clipPaths, ClipperLib.PolyType.ptClip, true);
  const solution = new ClipperLib.Paths();
  clipper.Execute(
    ClipperLib.ClipType.ctDifference,
    solution,
    ClipperLib.PolyFillType.pftNonZero,
    ClipperLib.PolyFillType.pftNonZero,
  );
  return cleanPaths(solution);
}

function offsetOpenSegment(a, b, halfThickness) {
  const co = new ClipperLib.ClipperOffset(2, 2);
  co.AddPaths([toPath([a, b])], ClipperLib.JoinType.jtSquare, ClipperLib.EndType.etOpenSquare);
  const out = new ClipperLib.Paths();
  co.Execute(out, halfThickness * CLIPPER_SCALE);
  return cleanPaths(out);
}

function offsetClosedPolygon(poly, delta) {
  const co = new ClipperLib.ClipperOffset(2, 2);
  co.AddPaths([toPath(poly)], ClipperLib.JoinType.jtMiter, ClipperLib.EndType.etClosedPolygon);
  const out = new ClipperLib.Paths();
  co.Execute(out, delta * CLIPPER_SCALE);
  return cleanPaths(out);
}

function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function midpoint(a, b) {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}

/** A rectangle spanning `width` centred on `centre`, oriented along the a->b direction, extended past both faces of the wall so it cuts a clean through-opening. */
function doorCutRect(a, b, centre, width, wallThickness) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  // perpendicular, generously oversized so it fully punches through the wall body
  const px = -uy * wallThickness * 2;
  const py = ux * wallThickness * 2;
  const half = width / 2;
  const p1 = [centre[0] - ux * half, centre[1] - uy * half];
  const p2 = [centre[0] + ux * half, centre[1] + uy * half];
  return [
    [p1[0] + px, p1[1] + py],
    [p2[0] + px, p2[1] + py],
    [p2[0] - px, p2[1] - py],
    [p1[0] - px, p1[1] - py],
  ];
}

/** The door leaf: a thin rectangle of `width` along a->b, `leaf` thick, centred on `centre`. */
function doorLeafRect(a, b, centre, width, leaf) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const hw = width / 2;
  const ht = leaf / 2;
  const nx = -uy;
  const ny = ux;
  return [
    [centre[0] - ux * hw + nx * ht, centre[1] - uy * hw + ny * ht],
    [centre[0] + ux * hw + nx * ht, centre[1] + uy * hw + ny * ht],
    [centre[0] + ux * hw - nx * ht, centre[1] + uy * hw - ny * ht],
    [centre[0] - ux * hw - nx * ht, centre[1] - uy * hw - ny * ht],
  ];
}

/**
 * Build the wall poché as a set of Clipper paths (mm, un-scaled), plus the
 * list of door openings (for drawing swing arcs separately).
 */
export function computeWallGeometry(fixture, options = {}) {
  const thickness = { ...DEFAULT_THICKNESS_MM, ...(options.thicknessMm || {}) };
  const doorWidth = options.doorWidthMm ?? DEFAULT_DOOR_WIDTH_MM;
  const { boundary, plan } = fixture;

  const wallPaths = [];
  const doorCuts = [];
  const doors = [];

  // --- Boundary. Default: one closed band, uniform thickness (v1). With
  // options.perEdgeBoundary, one band per typed edge, thickness by edge type
  // (external / shared_wall / facing_corridor); the union resolves the corners.
  if (options.perEdgeBoundary) {
    for (const te of boundary.typed_edges) {
      const half = (thickness[te.type] ?? thickness.external) / 2;
      wallPaths.push(...offsetOpenSegment(te.start, te.end, half));
    }
  } else {
    const boundaryHalf = thickness.external / 2;
    const outer = offsetClosedPolygon(boundary.polygon, boundaryHalf);
    const inner = offsetClosedPolygon(boundary.polygon, -boundaryHalf);
    wallPaths.push(...differencePaths(outer, inner));
  }

  // Explicit doors (plan.doors, from the Generator: centre, width, wall endpoints, kind)
  // replace the legacy rules below (entrance_point, edge.has_door at the midpoint).
  const explicit = Array.isArray(plan.doors);
  const leafMm = options.doorLeafMm ?? 40;
  if (explicit) {
    for (const d of plan.doors) {
      const [a, b] = d.wall;
      const t = d.kind === 'entrance' ? (thickness[d.edgeType] ?? thickness.external) : thickness.internal;
      // cut: along the wall by the door width, across it by ~0.7 x thickness each side
      doorCuts.push(toPath(doorCutRect(a, b, d.centre, d.width, t * 0.35)));
      doors.push({
        centre: d.centre, edge: { start: a, end: b }, kind: d.kind, width: d.width,
        leaf: doorLeafRect(a, b, d.centre, d.width, leafMm),
      });
    }
  } else {
    // Front door opening at the boundary's entrance point.
    const entranceEdge = boundary.typed_edges[boundary.entrance_edge_index];
    if (entranceEdge) {
      const cut = doorCutRect(
        entranceEdge.start, entranceEdge.end, boundary.entrance_point,
        doorWidth, thickness.external,
      );
      doorCuts.push(toPath(cut));
      doors.push({ centre: boundary.entrance_point, edge: entranceEdge, kind: 'entrance' });
    }
  }

  // --- Internal partition walls: one offset rectangle per room-to-room edge.
  for (const edge of Object.values(plan.edges)) {
    if (!edge.room_b) continue; // boundary-touching edge — covered by the boundary band
    if (dist(edge.start, edge.end) < 1) continue; // degenerate (zero-length) — see schema.py note

    const half = (edge.thickness_mm ?? thickness.internal) / 2;
    wallPaths.push(...offsetOpenSegment(edge.start, edge.end, half));

    if (edge.has_door && !explicit) {
      const centre = midpoint(edge.start, edge.end);
      const cut = doorCutRect(edge.start, edge.end, centre, doorWidth, thickness.internal);
      doorCuts.push(toPath(cut));
      doors.push({ centre, edge, kind: 'internal' });
    }
  }

  const merged = unionPaths(wallPaths);
  const withOpenings = differencePaths(merged, doorCuts);

  return {
    wallPolygons: withOpenings.map(fromPath),
    doors,
  };
}

/**
 * Fills the wall poché and (when `doorFill` is given) the door leaves of a
 * computeWallGeometry result. `map([x, y])` takes fixture mm to ctx coordinates,
 * so a caller with its own transform (the Unit and Floor plate canvases) reuses
 * the same geometry. Door leaves use `doorFill`; walls use `wallFill`.
 */
export function drawWallGeometry(ctx, geo, map = (p) => p, wallFill = '#2b2b2b', doorFill = '#b45309') {
  ctx.save();
  ctx.setLineDash([]);
  ctx.fillStyle = wallFill;
  // One path for all rings: a ring that is a hole (a room walled in on every side) must stay
  // empty, which separate fills would not do.
  ctx.beginPath();
  for (const poly of geo.wallPolygons) {
    poly.forEach((p, i) => { const [x, y] = map(p); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.closePath();
  }
  ctx.fill('evenodd');
  if (doorFill) {
    ctx.fillStyle = doorFill;
    for (const d of geo.doors) {
      if (!d.leaf) continue;
      ctx.beginPath();
      d.leaf.forEach((p, i) => { const [x, y] = map(p); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.closePath();
      ctx.fill();
    }
  }
  ctx.restore();
}

function drawDoorSwing(ctx, door, scale) {
  const { centre, edge } = door;
  const dx = edge.end[0] - edge.start[0];
  const dy = edge.end[1] - edge.start[1];
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const nx = -uy;
  const ny = ux;
  const doorWidth = DEFAULT_DOOR_WIDTH_MM;
  const hingeX = centre[0] - ux * (doorWidth / 2);
  const hingeY = centre[1] - uy * (doorWidth / 2);
  const leafX = hingeX + nx * doorWidth;
  const leafY = hingeY + ny * doorWidth;

  ctx.save();
  ctx.strokeStyle = '#888888';
  ctx.lineWidth = Math.max(1, 1 * scale);
  ctx.setLineDash([]);
  // leaf (open position)
  ctx.beginPath();
  ctx.moveTo(hingeX, hingeY);
  ctx.lineTo(leafX, leafY);
  ctx.stroke();
  // swing arc
  const startAngle = Math.atan2(uy, ux);
  const endAngle = Math.atan2(ny, nx);
  ctx.beginPath();
  ctx.arc(hingeX, hingeY, doorWidth, Math.min(startAngle, endAngle), Math.max(startAngle, endAngle));
  ctx.stroke();
  ctx.restore();
}

/**
 * Draws the fixture into a 2D canvas context. Caller owns the ctx transform
 * (translate/scale to fit the canvas) — this function draws in fixture mm
 * coordinates directly (y already increases downward, matching screen space
 * — see render.py's note on this).
 *
 * options.showWalls / showDoors / showOutlines / showLabels each default to
 * true — the drawing panel's toggles pass explicit booleans, but any other
 * caller (e.g. a future golden-fixture snapshot test) can omit them entirely
 * and get the full drawing.
 */
export function renderPlanDrawing(ctx, fixture, options = {}) {
  const { plan } = fixture;
  const scale = options.scale ?? 1; // px per mm, only used to keep stroke widths visually sane
  const showWalls = options.showWalls ?? true;
  const showDoors = options.showDoors ?? true;
  const showOutlines = options.showOutlines ?? true;
  const showLabels = options.showLabels ?? true;

  // Room outlines (thin — the wall poché below carries the visual weight)
  if (showOutlines) {
    ctx.save();
    ctx.strokeStyle = '#bbbbbb';
    ctx.lineWidth = Math.max(0.5, 0.5 / scale);
    for (const room of Object.values(plan.rooms)) {
      if (room.polygon.length < 3) continue;
      ctx.beginPath();
      room.polygon.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
      ctx.closePath();
      ctx.stroke();
    }
    ctx.restore();
  }

  // Wall poché + door swings share the same computed geometry, so compute
  // it once whenever either is requested.
  let doors = [];
  if (showWalls || showDoors) {
    const geo = computeWallGeometry(fixture, options);
    doors = geo.doors;
    if (showWalls) {
      drawWallGeometry(ctx, geo, (p) => p, '#2b2b2b', null);
    }
  }

  if (showDoors) {
    for (const door of doors) {
      if (door.leaf) {
        ctx.save();
        ctx.fillStyle = '#b45309';
        ctx.beginPath();
        door.leaf.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      } else drawDoorSwing(ctx, door, scale);
    }
  }

  if (showLabels) {
    ctx.save();
    ctx.fillStyle = '#333333';
    ctx.font = `${Math.max(120, 150)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const [rid, room] of Object.entries(plan.rooms)) {
      if (room.polygon.length < 3) continue;
      const cx = room.polygon.reduce((s, p) => s + p[0], 0) / room.polygon.length;
      const cy = room.polygon.reduce((s, p) => s + p[1], 0) / room.polygon.length;
      ctx.fillText(`${rid} — ${room.type}`, cx, cy);
    }
    ctx.restore();
  }
}
