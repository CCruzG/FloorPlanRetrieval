import './vendor/clipper.js';
import { METRES_PER_UNIT } from './constants.js';

const ClipperLib = globalThis.ClipperLib;
if (!ClipperLib) {
  throw new Error('ClipperLib failed to load from local vendor bundle.');
}

const CLIPPER_SCALE = 1e6;
const MIN_PATH_AREA = 1e-10;

function polygonArea(poly) {
  if (!Array.isArray(poly) || poly.length < 3) return 0;
  let sum = 0;
  for (let i = 0; i < poly.length; i += 1) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum) * 0.5;
}

function polygonCentroid(poly) {
  if (!Array.isArray(poly) || poly.length < 3) return [0, 0];
  let cx = 0;
  let cy = 0;
  let areaTwice = 0;
  for (let i = 0; i < poly.length; i += 1) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    const cross = x1 * y2 - x2 * y1;
    areaTwice += cross;
    cx += (x1 + x2) * cross;
    cy += (y1 + y2) * cross;
  }
  if (Math.abs(areaTwice) < 1e-12) {
    const avgX = poly.reduce((sum, [x]) => sum + x, 0) / poly.length;
    const avgY = poly.reduce((sum, [, y]) => sum + y, 0) / poly.length;
    return [avgX, avgY];
  }
  const scale = 1 / (3 * areaTwice);
  return [cx * scale, cy * scale];
}

function intersectRayWithSegment(origin, direction, a, b) {
  const [ox, oy] = origin;
  const [dx, dy] = direction;
  const [ax, ay] = a;
  const [bx, by] = b;
  const sx = bx - ax;
  const sy = by - ay;
  const det = dx * sy - dy * sx;
  if (Math.abs(det) < 1e-12) return null;
  const qpx = ax - ox;
  const qpy = ay - oy;
  const t = (qpx * sy - qpy * sx) / det;
  const u = (qpx * dy - qpy * dx) / det;
  if (t <= 1e-9 || u < -1e-9 || u > 1 + 1e-9) return null;
  return {
    t,
    point: [ox + t * dx, oy + t * dy],
  };
}

function projectFromInnerEdgeMidpoints(innerPolygon, boundaryPolygon) {
  if (!Array.isArray(innerPolygon) || innerPolygon.length < 3) return [];
  if (!Array.isArray(boundaryPolygon) || boundaryPolygon.length < 3) return [];

  const lines = [];
  for (let i = 0; i < innerPolygon.length; i += 1) {
    const a = innerPolygon[i];
    const b = innerPolygon[(i + 1) % innerPolygon.length];
    const mx = (a[0] + b[0]) * 0.5;
    const my = (a[1] + b[1]) * 0.5;
    const ex = b[0] - a[0];
    const ey = b[1] - a[1];
    const elen = Math.hypot(ex, ey);
    if (elen < 1e-12) continue;

    const n1 = [ey / elen, -ex / elen];
    const n2 = [-ey / elen, ex / elen];
    const eps = Math.min(1e-4, 0.1 / METRES_PER_UNIT);
    const p1 = [mx + n1[0] * eps, my + n1[1] * eps];
    const p2 = [mx + n2[0] * eps, my + n2[1] * eps];
    const outward = pointInPoly(p1, innerPolygon) && !pointInPoly(p2, innerPolygon) ? n2 : n1;

    let bestHit = null;
    let bestEdgeIndex = -1;
    for (let j = 0; j < boundaryPolygon.length; j += 1) {
      const c = boundaryPolygon[j];
      const d = boundaryPolygon[(j + 1) % boundaryPolygon.length];
      const hit = intersectRayWithSegment([mx, my], outward, c, d);
      if (!hit) continue;
      if (!bestHit || hit.t < bestHit.t) {
        bestHit = hit;
        bestEdgeIndex = j;
      }
    }
    if (!bestHit) continue;

    lines.push({
      source_edge_index: i,
      target_edge_index: bestEdgeIndex,
      line: [[mx, my], bestHit.point],
      length_m: Math.hypot(bestHit.point[0] - mx, bestHit.point[1] - my) * METRES_PER_UNIT,
    });
  }
  return lines;
}

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
  return cleaned.filter((path) => Math.abs(ClipperLib.Clipper.Area(path)) > MIN_PATH_AREA * CLIPPER_SCALE * CLIPPER_SCALE);
}

function executeClip(subjectPaths, clipPaths, clipType) {
  const clipper = new ClipperLib.Clipper();
  clipper.AddPaths(subjectPaths || [], ClipperLib.PolyType.ptSubject, true);
  clipper.AddPaths(clipPaths || [], ClipperLib.PolyType.ptClip, true);
  const solution = new ClipperLib.Paths();
  clipper.Execute(
    clipType,
    solution,
    ClipperLib.PolyFillType.pftNonZero,
    ClipperLib.PolyFillType.pftNonZero,
  );
  return cleanPaths(solution);
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

function offsetPaths(paths, delta) {
  const co = new ClipperLib.ClipperOffset(2, 2);
  co.AddPaths(paths, ClipperLib.JoinType.jtMiter, ClipperLib.EndType.etClosedPolygon);
  const out = new ClipperLib.Paths();
  co.Execute(out, delta);
  return cleanPaths(out);
}

function pointInPoly(pt, poly) {
  if (!Array.isArray(poly) || poly.length < 3) return false;
  const [x, y] = pt;
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    const intersects = (yi > y) !== (yj > y)
      && x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-12) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

function edgeBandPolygon(a, b, depthUnits, boundaryPoly) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  if (len < 1e-12) return null;

  const n1 = [dy / len, -dx / len];
  const n2 = [-dy / len, dx / len];
  const mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];

  const eps = Math.min(depthUnits * 0.2, 1e-4);
  const test1 = [mid[0] + n1[0] * eps, mid[1] + n1[1] * eps];
  const test2 = [mid[0] + n2[0] * eps, mid[1] + n2[1] * eps];
  const useN = pointInPoly(test1, boundaryPoly) || !pointInPoly(test2, boundaryPoly) ? n1 : n2;

  const a2 = [a[0] + useN[0] * depthUnits, a[1] + useN[1] * depthUnits];
  const b2 = [b[0] + useN[0] * depthUnits, b[1] + useN[1] * depthUnits];
  return [a, b, b2, a2];
}

function pathAreaM2(path) {
  const poly = fromPath(path);
  return polygonArea(poly) * METRES_PER_UNIT * METRES_PER_UNIT;
}

function bboxDimensionsM(poly) {
  if (!poly?.length) return { width_m: 0, length_m: 0 };
  const xs = poly.map(([x]) => x);
  const ys = poly.map(([, y]) => y);
  const width = (Math.max(...xs) - Math.min(...xs)) * METRES_PER_UNIT;
  const length = (Math.max(...ys) - Math.min(...ys)) * METRES_PER_UNIT;
  return {
    width_m: Math.min(width, length),
    length_m: Math.max(width, length),
  };
}

function normalizeDepthRange(depthRange = {}) {
  const minRaw = parseFloat(depthRange?.min);
  const maxRaw = parseFloat(depthRange?.max);
  const min = Number.isFinite(minRaw) ? minRaw : 5;
  const max = Number.isFinite(maxRaw) ? maxRaw : 9;
  return min <= max ? { min, max } : { min: max, max: min };
}

function classifyExecutionCase(boundaryPolygon, sample) {
  const dims = bboxDimensionsM(boundaryPolygon);
  const aspect = dims.width_m > 0 && dims.length_m > 0
    ? Math.max(dims.width_m, dims.length_m) / Math.max(Math.min(dims.width_m, dims.length_m), 1e-9)
    : 1;
  if (sample?.diagnostics?.inner_region_count > 1) return 'multi_region';
  if (!sample?.inner_polygons?.length) return 'degenerate';
  if (aspect < 1.8) return 'compact_point_block';
  if (aspect >= 2.2) return 'slab';
  return 'single_or_transition';
}

function buildOffsetSample(boundaryPolygon, depthM, minCorridorWidthM) {
  const result = _generate_offset_regions_single(boundaryPolygon, depthM, minCorridorWidthM);
  const centerSource = result?.primary_inner_polygon?.polygon?.length
    ? result.primary_inner_polygon.polygon
    : boundaryPolygon;
  const centerPoint = polygonCentroid(centerSource);
  const middleRegionDimensions = result?.primary_inner_polygon?.polygon?.length
    ? {
      ...bboxDimensionsM(result.primary_inner_polygon.polygon),
      area_m2: result.primary_inner_polygon.area_m2 ?? null,
    }
    : null;
  const projected_unit_boundaries = result?.primary_inner_polygon?.polygon?.length
    ? projectFromInnerEdgeMidpoints(result.primary_inner_polygon.polygon, boundaryPolygon)
    : [];
  return {
    ...result,
    center_point: centerPoint,
    middle_region_dimensions: middleRegionDimensions,
    projected_unit_boundaries,
    execution_case: classifyExecutionCase(boundaryPolygon, result),
  };
}

export function generate_offset_regions(boundaryPolygon = [], depthRange = { min: 5, max: 9 }, minCorridorWidthM = 1.2) {
  if (!Array.isArray(boundaryPolygon) || boundaryPolygon.length < 3) {
    return { status: 'error', error: 'Boundary must have at least 3 vertices.' };
  }

  const range = normalizeDepthRange(depthRange);
  const sampleDepths = range.min === range.max
    ? [range.min]
    : [range.min, (range.min + range.max) / 2, range.max];
  const samples = sampleDepths.map((depthM) => buildOffsetSample(boundaryPolygon, depthM, minCorridorWidthM));
  const activeSample = samples.find((sample) => sample?.status === 'ok' && sample?.primary_inner_polygon)
    || samples[Math.floor(samples.length / 2)]
    || samples[0];

  const caseNotes = samples.map((sample) => {
    const caseName = sample?.execution_case || 'unknown';
    const noteParts = [caseName, `d=${sample?.depth_m?.toFixed?.(2) ?? 'n/a'}m`];
    if (sample?.diagnostics?.topology_notes?.length) noteParts.push(sample.diagnostics.topology_notes.join('; '));
    return noteParts.join(' | ');
  });

  const boundaryAreaM2 = polygonArea(boundaryPolygon) * METRES_PER_UNIT * METRES_PER_UNIT;
  const activeDiagnostics = activeSample?.diagnostics || {};
  const activeMiddleRegionDimensions = activeSample?.middle_region_dimensions || null;

  return {
    status: 'ok',
    depth_range_m: range,
    min_corridor_width_m: minCorridorWidthM,
    samples,
    active_sample: activeSample,
    diagnostics: {
      boundary_area_m2: boundaryAreaM2,
      sample_count: samples.length,
      case_notes: caseNotes,
      active_case: activeSample?.execution_case || null,
      active_depth_m: activeSample?.depth_m ?? null,
      active_inner_region_count: activeDiagnostics.inner_region_count ?? 0,
      active_closure_error_m2: activeDiagnostics.closure_error_m2 ?? null,
      active_projected_line_count: activeSample?.projected_unit_boundaries?.length ?? 0,
      active_middle_region_dimensions: activeMiddleRegionDimensions,
    },
  };
}

function generate_offset_regions_single(boundaryPolygon = [], depthM = 6, minCorridorWidthM = 1.2) {
  const boundaryPath = toPath(boundaryPolygon);
  const boundaryPaths = [boundaryPath];
  const depthUnits = depthM / METRES_PER_UNIT;
  const deltaInt = Math.round(depthUnits * CLIPPER_SCALE);
  if (deltaInt <= 0) {
    return { status: 'error', error: 'Depth must be positive.' };
  }

  const innerPaths = offsetPaths(boundaryPaths, -deltaInt);
  const topologyNotes = [];
  if (!innerPaths.length) {
    topologyNotes.push('Inset collapsed: no inner regions at this depth.');
  } else if (innerPaths.length > 1) {
    topologyNotes.push(`Inset split into ${innerPaths.length} inner regions.`);
  }

  const annulusPaths = innerPaths.length
    ? executeClip(boundaryPaths, innerPaths, ClipperLib.ClipType.ctDifference)
    : boundaryPaths;

  const strips = [];
  const stripPaths = [];
  for (let i = 0; i < boundaryPolygon.length; i += 1) {
    const a = boundaryPolygon[i];
    const b = boundaryPolygon[(i + 1) % boundaryPolygon.length];
    const band = edgeBandPolygon(a, b, depthUnits, boundaryPolygon);
    if (!band) continue;

    const hitPaths = executeClip([toPath(band)], annulusPaths, ClipperLib.ClipType.ctIntersection);
    if (!hitPaths.length) {
      strips.push({
        face_index: i,
        polygons: [],
        depth_m: depthM,
        length_m: Math.hypot(b[0] - a[0], b[1] - a[1]) * METRES_PER_UNIT,
        area_m2: 0,
      });
      continue;
    }

    stripPaths.push(...hitPaths);
    strips.push({
      face_index: i,
      polygons: hitPaths.map(fromPath),
      depth_m: depthM,
      length_m: Math.hypot(b[0] - a[0], b[1] - a[1]) * METRES_PER_UNIT,
      area_m2: hitPaths.reduce((acc, p) => acc + pathAreaM2(p), 0),
    });
  }

  const stripUnionPaths = unionPaths(stripPaths);
  const cornerPaths = stripUnionPaths.length
    ? executeClip(annulusPaths, stripUnionPaths, ClipperLib.ClipType.ctDifference)
    : annulusPaths;

  const innerRegions = innerPaths.map((path) => {
    const polygon = fromPath(path);
    const areaM2 = polygonArea(polygon) * METRES_PER_UNIT * METRES_PER_UNIT;
    const dims = bboxDimensionsM(polygon);
    return {
      polygon,
      area_m2: areaM2,
      ...dims,
      below_corridor_min: dims.width_m > 0 && dims.width_m < minCorridorWidthM,
    };
  });

  const primaryInner = innerRegions
    .slice()
    .sort((a, b) => b.area_m2 - a.area_m2)[0] || null;

  const cornerBays = cornerPaths.map((path, idx) => ({
    corner_index: idx,
    polygon: fromPath(path),
    area_m2: pathAreaM2(path),
  }));

  const boundaryAreaM2 = polygonArea(boundaryPolygon) * METRES_PER_UNIT * METRES_PER_UNIT;
  const innerAreaM2 = innerRegions.reduce((acc, region) => acc + region.area_m2, 0);
  const stripsAreaM2 = strips.reduce((acc, strip) => acc + strip.area_m2, 0);
  const cornerAreaM2 = cornerBays.reduce((acc, bay) => acc + bay.area_m2, 0);
  const closureErrorM2 = boundaryAreaM2 - (innerAreaM2 + stripsAreaM2 + cornerAreaM2);

  if (innerRegions.some((region) => region.below_corridor_min)) {
    topologyNotes.push('One or more inner regions are below minimum corridor width.');
  }

  return {
    status: 'ok',
    depth_m: depthM,
    min_corridor_width_m: minCorridorWidthM,
    inner_polygons: innerRegions,
    primary_inner_polygon: primaryInner,
    apartment_strips: strips,
    corner_bays: cornerBays,
    diagnostics: {
      boundary_area_m2: boundaryAreaM2,
      inner_area_m2: innerAreaM2,
      strips_area_m2: stripsAreaM2,
      corner_area_m2: cornerAreaM2,
      closure_error_m2: closureErrorM2,
      inner_region_count: innerRegions.length,
      topology_notes: topologyNotes,
    },
  };
}

// Internal single-depth engine reused by depth-range sampling.
function _generate_offset_regions_single(boundaryPolygon = [], depthM = 6, minCorridorWidthM = 1.2) {
  return generate_offset_regions_single(boundaryPolygon, depthM, minCorridorWidthM);
}
