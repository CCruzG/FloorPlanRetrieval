// Adapter: a generated plan -> the fixture shape plan-drawing-renderer.js takes.
// Wiring only. The plan (rooms, edges, doors, frame) is read, never changed.
//
//   plan  : {rooms:[{id, polygon}], edges:[{u, v, connection}], doors:[...], frame:{boundary, edge_types}}
//           in plan units (app units; one unit = METRES_PER_UNIT metres)
//   out   : {boundary:{polygon, typed_edges:[{start, end, type}]}, plan:{rooms, edges, doors}}  in mm
//
// Walls: the boundary edges (thickness by type: external 200, shared_wall 150,
// facing_corridor 150) and one internal wall (100) on every stretch two rooms share,
// except between rooms joined by an open edge (connection 2). Doors come from plan.doors.

import { METRES_PER_UNIT } from './constants.js';
import { computeWallGeometry } from './plan-drawing-renderer.js';

export const MM_PER_APP_UNIT = METRES_PER_UNIT * 1000;
export const WALL_THICKNESS_MM = { external: 200, shared_wall: 150, facing_corridor: 150, internal: 100 };
const COLLINEAR_MM = 1;   // two edges closer than this to one line count as collinear
const MIN_OVERLAP_MM = 20; // a shared stretch shorter than this is not a wall

function overlap(a0, a1, b0, b1) {
  // collinear overlap of segments a0-a1 and b0-b1 (mm); returns [p, q] or null
  const dx = a1[0] - a0[0];
  const dy = a1[1] - a0[1];
  const len = Math.hypot(dx, dy);
  if (len < MIN_OVERLAP_MM) return null;
  const ux = dx / len;
  const uy = dy / len;
  const off = (p) => (p[0] - a0[0]) * -uy + (p[1] - a0[1]) * ux;
  if (Math.abs(off(b0)) > COLLINEAR_MM || Math.abs(off(b1)) > COLLINEAR_MM) return null;
  const along = (p) => (p[0] - a0[0]) * ux + (p[1] - a0[1]) * uy;
  const t0 = Math.max(0, Math.min(along(b0), along(b1)));
  const t1 = Math.min(len, Math.max(along(b0), along(b1)));
  if (t1 - t0 < MIN_OVERLAP_MM) return null;
  return [[a0[0] + ux * t0, a0[1] + uy * t0], [a0[0] + ux * t1, a0[1] + uy * t1]];
}

/**
 * @param plan  generated plan in plan units
 * @param opts  {boundary, edgeTypes, mmPerUnit}: fall back to plan.frame
 */
export function planToDrawingFixture(plan, opts = {}) {
  const frame = plan.frame || {};
  const k = opts.mmPerUnit ?? frame.mm_per_unit ?? MM_PER_APP_UNIT;
  const bUnits = opts.boundary || frame.boundary || [];
  const types = opts.edgeTypes || frame.edge_types || [];
  const polygon = bUnits.map(([x, y]) => [x * k, y * k]);
  const typedEdges = polygon.map((p, i) => ({
    start: p, end: polygon[(i + 1) % polygon.length], type: types[i] || 'external',
  }));

  const rooms = {};
  for (const r of plan.rooms || []) {
    if ((r.polygon || []).length < 3) continue;
    rooms[String(r.id)] = { polygon: r.polygon.map(([x, y]) => [x * k, y * k]), type: r.type };
  }
  const ids = Object.keys(rooms);
  const openPairs = new Set();
  for (const e of plan.edges || []) {
    if (e.connection === 2) openPairs.add([String(e.u), String(e.v)].sort().join('|'));
  }

  const edges = {};
  let n = 0;
  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      if (openPairs.has([ids[i], ids[j]].sort().join('|'))) continue;
      const pa = rooms[ids[i]].polygon;
      const pb = rooms[ids[j]].polygon;
      for (let a = 0; a < pa.length; a++) {
        for (let b = 0; b < pb.length; b++) {
          const seg = overlap(pa[a], pa[(a + 1) % pa.length], pb[b], pb[(b + 1) % pb.length]);
          if (seg) edges[`w${n++}`] = { room_a: ids[i], room_b: ids[j], start: seg[0], end: seg[1] };
        }
      }
    }
  }

  const doors = (plan.doors || []).map((d) => {
    const wall = d.wall.map(([x, y]) => [x * k, y * k]);
    let edgeType = 'external';
    if (d.kind === 'entrance') {
      const c = [d.centre[0] * k, d.centre[1] * k];
      let best = Infinity;
      for (const te of typedEdges) {
        const dx = te.end[0] - te.start[0];
        const dy = te.end[1] - te.start[1];
        const l2 = dx * dx + dy * dy || 1;
        const t = Math.max(0, Math.min(1, ((c[0] - te.start[0]) * dx + (c[1] - te.start[1]) * dy) / l2));
        const dd = Math.hypot(c[0] - te.start[0] - t * dx, c[1] - te.start[1] - t * dy);
        if (dd < best) { best = dd; edgeType = te.type; }
      }
    }
    return {
      u: d.u, v: d.v, kind: d.kind, wall, edgeType,
      centre: [d.centre[0] * k, d.centre[1] * k], width: d.width_mm,
    };
  });

  return {
    schema_version: 'generated-plan',
    boundary: { polygon, typed_edges: typedEdges },
    plan: { rooms, edges, doors },
  };
}

/** Wall poché + door leaves for a generated plan, in mm (plan units x mm per unit). */
export function wallGeometryForPlan(plan, opts = {}) {
  return computeWallGeometry(planToDrawingFixture(plan, opts), {
    perEdgeBoundary: true, thicknessMm: WALL_THICKNESS_MM,
  });
}
