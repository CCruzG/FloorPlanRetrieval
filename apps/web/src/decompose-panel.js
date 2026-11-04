// Decompose (RPlan) panel — loads a raw RPlan plan and shows its rectangle
// decomposition (POST /decompose) on a private overlay canvas.
//
// Isolation: the main canvas state (canvas-renderer / main.js) is not touched.
// The view lives on #decompose-canvas, stacked over #canvas-wrap, and is shown
// only while a plan is loaded ("decompose mode"). "close view" hides it and the
// normal retrieve / generate / swap canvas is visible again.

const SCALE_M = 18.0; // normalised x 18 = metres (RPlan schema 2.1.0)

const state = {
  plan: null,
  planId: null,
  result: null,
  active: false, // decompose-mode flag
  T: null, // last fit transform (screen = o + s * plan), for click -> plan units
  measure: { a: null, b: null }, // plan-unit points
  fit: null, // fit-to-canvas transform before zoom/pan
  view: { zoom: 1, dx: 0, dy: 0 }, // wheel zoom + drag pan, folded into T
  drag: null,
};

const $ = (id) => document.getElementById(id);

async function fetchJson(url, opts) {
  const res = await fetch(url, opts);
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON body */ }
  if (!res.ok) throw new Error((data && data.error) || `HTTP ${res.status}`);
  return data;
}

function setReadout(text) {
  const el = $('dc-readout');
  if (el) el.textContent = text;
}

function setActive(on) {
  state.active = on;
  const cv = $('decompose-canvas');
  if (cv) cv.style.display = on ? 'block' : 'none';
  const closeBtn = $('dc-close');
  if (closeBtn) closeBtn.style.display = on ? 'block' : 'none';
  if (on) draw();
}

// ── drawing ─────────────────────────────────────────────────────────────────

function boundaryPoints(plan) {
  return (plan.boundary || []).map((p) => [p[0], p[1]]);
}

function fitTransform(plan, w, h) {
  const pts = boundaryPoints(plan);
  for (const r of plan.rooms || []) for (const p of r.polygon || []) pts.push(p);
  if (!pts.length) return { s: 1, ox: 0, oy: 0 };
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) {
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  const pad = 40;
  const s = Math.min((w - 2 * pad) / Math.max(x1 - x0, 1e-6), (h - 2 * pad) / Math.max(y1 - y0, 1e-6));
  return {
    s,
    ox: (w - s * (x1 - x0)) / 2 - s * x0,
    oy: (h - s * (y1 - y0)) / 2 - s * y0,
  };
}

function pathPoly(ctx, poly, T) {
  ctx.beginPath();
  poly.forEach(([x, y], i) => {
    const px = T.ox + T.s * x, py = T.oy + T.s * y;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.closePath();
}

function centroid(poly) {
  let sx = 0, sy = 0;
  for (const [x, y] of poly) { sx += x; sy += y; }
  return [sx / poly.length, sy / poly.length];
}

function roomHue(index) {
  return (index * 47) % 360;
}

function hatch(ctx, poly, T, color) {
  ctx.save();
  pathPoly(ctx, poly, T);
  ctx.clip();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of poly) {
    const px = T.ox + T.s * x, py = T.oy + T.s * y;
    x0 = Math.min(x0, px); x1 = Math.max(x1, px);
    y0 = Math.min(y0, py); y1 = Math.max(y1, py);
  }
  ctx.beginPath();
  for (let k = x0 - (y1 - y0); k < x1; k += 5) {
    ctx.moveTo(k, y1);
    ctx.lineTo(k + (y1 - y0), y0);
  }
  ctx.stroke();
  ctx.restore();
}

function drawSegments(ctx, segs, T, color, dash, width) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.lineCap = dash.length && dash[0] <= 2 ? 'round' : 'butt';
  for (const [[ax, ay], [bx, by]] of segs) {
    ctx.beginPath();
    ctx.moveTo(T.ox + T.s * ax, T.oy + T.s * ay);
    ctx.lineTo(T.ox + T.s * bx, T.oy + T.s * by);
    ctx.stroke();
  }
  ctx.restore();
}

// Openings: interior doors (plan.doors[].bbox) and the entrance (plan.entrance
// = [x0,y0,x1,y1]). RPlan boxes are ~1 px thick, so draw at a minimum screen
// thickness to stay visible.
const DOOR_COLOR = '#16a34a';
const ENTRANCE_COLOR = '#7c3aed';

function drawBox(ctx, bbox, T, color, minPx) {
  if (!bbox || bbox.length < 4) return;
  let x0 = T.ox + T.s * Math.min(bbox[0], bbox[2]);
  let x1 = T.ox + T.s * Math.max(bbox[0], bbox[2]);
  let y0 = T.oy + T.s * Math.min(bbox[1], bbox[3]);
  let y1 = T.oy + T.s * Math.max(bbox[1], bbox[3]);
  if (x1 - x0 < minPx) { const c = (x0 + x1) / 2; x0 = c - minPx / 2; x1 = c + minPx / 2; }
  if (y1 - y0 < minPx) { const c = (y0 + y1) / 2; y0 = c - minPx / 2; y1 = c + minPx / 2; }
  ctx.save();
  ctx.setLineDash([]);
  ctx.fillStyle = color;
  ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1;
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  ctx.restore();
}

function drawOpenings(ctx, plan, T) {
  for (const d of plan.doors || []) drawBox(ctx, d.bbox, T, DOOR_COLOR, 5);
  if (plan.entrance && plan.entrance.length >= 4) drawBox(ctx, plan.entrance, T, ENTRANCE_COLOR, 7);
}

function roomName(plan, id) {
  const r = (plan.rooms || []).find((x) => x.id === id);
  return r ? String(r.label ?? r.type ?? id) : String(id);
}

function connectionLines(plan) {
  const kinds = { 1: 'door', 2: 'open' };
  return (plan.edges || [])
    .filter((e) => kinds[e.connection])
    .map((e) => `  ${kinds[e.connection]}: ${roomName(plan, e.u)} – ${roomName(plan, e.v)}`);
}

const AMBER = '#f59e0b';
const ORANGE = '#f97316';

// role lookup: "<room id>#<part index>" -> {role, label}
function roleMap() {
  const m = new Map();
  const lab = state.result && state.result.labels;
  if (!lab) return m;
  for (const r of lab.rooms || []) for (const p of r.parts || []) m.set(`${r.id}#${p.index}`, { ...p, roomLabel: r.label });
  return m;
}

function partText(info) {
  if (info.role === 'absorbed') return '';
  if (info.role === 'primary') {
    const base = info.roomLabel === 'LivingRoom' ? 'living' : String(info.label ?? info.roomLabel ?? '');
    return info.pass_through ? `${base} · pass-through` : base;
  }
  if (info.role === 'whole') return String(info.label ?? info.roomLabel ?? '');
  if (info.role === 'zone' && info.dining_candidate) return 'dining?';
  return info.role; // circulation | zone | jog | alcove | storage | nook | residual
}

const CIRC_FILL = 'rgba(251,191,36,0.8)';
const STORAGE_FILL = 'rgba(203,213,225,0.9)';
const NOOK_FILL = 'rgba(45,212,191,0.7)';
const BLOCK_OPEN_COLOR = '#db2777';
const BLOCK_TOUCH_COLOR = '#94a3b8';

// fill colour of a part; an absorbed part takes its absorber's fill
function partFill(info, hue, roles, roomId) {
  let role = info ? info.role : null;
  if (role === 'absorbed' && info.absorbed_into !== null && info.absorbed_into !== undefined) {
    const ab = roles.get(`${roomId}#${info.absorbed_into}`);
    if (ab) role = ab.role;
  }
  if (role === 'circulation') return CIRC_FILL;
  if (role === 'storage') return STORAGE_FILL;
  if (role === 'nook') return NOOK_FILL;
  if (role === 'residual') return '#ffffff';
  return `hsla(${hue},70%,78%,0.75)`;
}

// intra-room block edges: "open" = short thick bar across the shared edge,
// "touch" = thin grey tick
function drawBlockEdges(ctx, T) {
  const lab = state.result && state.result.labels;
  if (!lab) return;
  ctx.save();
  ctx.setLineDash([]);
  for (const room of lab.rooms || []) {
    for (const e of room.block_edges || []) {
      if (!e.mid || !e.seg) continue;
      const [[ax, ay], [bx, by]] = e.seg;
      const dx = bx - ax, dy = by - ay;
      const horizontal = Math.abs(dx) >= Math.abs(dy);
      const half = e.kind === 'open' ? 9 : 5;
      const px = T.ox + T.s * e.mid[0], py = T.oy + T.s * e.mid[1];
      ctx.strokeStyle = e.kind === 'open' ? BLOCK_OPEN_COLOR : BLOCK_TOUCH_COLOR;
      ctx.lineWidth = e.kind === 'open' ? 4 : 1;
      ctx.lineCap = 'butt';
      ctx.beginPath();
      if (horizontal) { ctx.moveTo(px, py - half); ctx.lineTo(px, py + half); }
      else { ctx.moveTo(px - half, py); ctx.lineTo(px + half, py); }
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawMeasure(ctx) {
  const T = state.T;
  const { a, b } = state.measure;
  if (!T || !a) return;
  const ax = T.ox + T.s * a[0], ay = T.oy + T.s * a[1];
  ctx.save();
  ctx.setLineDash([]);
  ctx.fillStyle = '#0ea5e9';
  ctx.strokeStyle = '#0ea5e9';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(ax, ay, 4, 0, Math.PI * 2); ctx.fill();
  if (b) {
    const bx = T.ox + T.s * b[0], by = T.oy + T.s * b[1];
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    ctx.beginPath(); ctx.arc(bx, by, 4, 0, Math.PI * 2); ctx.fill();
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]) * SCALE_M;
    const txt = `${len.toFixed(2)} m`;
    ctx.font = 'bold 12px ui-monospace, monospace';
    const tw = ctx.measureText(txt).width;
    const mx = (ax + bx) / 2, my = (ay + by) / 2 - 12;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(mx - tw / 2 - 4, my - 9, tw + 8, 18);
    ctx.strokeStyle = '#0ea5e9'; ctx.lineWidth = 1;
    ctx.strokeRect(mx - tw / 2 - 4, my - 9, tw + 8, 18);
    ctx.fillStyle = '#0369a1';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(txt, mx, my);
  }
  ctx.restore();
}

function draw() {
  const cv = $('decompose-canvas');
  if (!cv || !state.active) return;
  const rect = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(rect.width, 50), h = Math.max(rect.height, 50);
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const plan = state.plan;
  if (!plan) return;
  const F = fitTransform(plan, w, h);
  state.fit = F;
  const V = state.view;
  const T = { s: F.s * V.zoom, ox: F.ox * V.zoom + V.dx, oy: F.oy * V.zoom + V.dy };
  state.T = T;
  const roles = roleMap();
  const hasLabels = roles.size > 0;

  // source rooms: outline + label
  ctx.font = '11px ui-monospace, monospace';
  ctx.textAlign = 'center';
  (plan.rooms || []).forEach((r) => {
    if (!r.polygon || r.polygon.length < 3) return;
    pathPoly(ctx, r.polygon, T);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 1;
    ctx.stroke();
  });

  // decomposition overlay
  const res = state.result;
  if (res) {
    res.rooms.forEach((room, i) => {
      const hue = roomHue(i);
      (room.parts || []).forEach((part, pi) => {
        const info = roles.get(`${room.id}#${pi}`);
        const role = info ? info.role : null;
        pathPoly(ctx, part.polygon, T);
        ctx.fillStyle = partFill(info, hue, roles, room.id);
        ctx.fill();
        ctx.strokeStyle = role === 'residual' ? ORANGE : `hsl(${hue},40%,45%)`;
        ctx.lineWidth = role === 'residual' ? 2 : 1;
        ctx.setLineDash(role === 'zone' ? [6, 3] : (role === 'jog' || role === 'alcove') ? [1, 3] : role === 'absorbed' ? [1, 2] : []);
        if (role === 'zone' || role === 'jog' || role === 'alcove') ctx.lineWidth = 2;
        ctx.stroke();
        ctx.setLineDash([]);
        if (role === 'storage') hatch(ctx, part.polygon, T, '#64748b');
        else if (part.sliver && role !== 'residual' && role !== 'absorbed') {
          hatch(ctx, part.polygon, T, '#f97316');
          pathPoly(ctx, part.polygon, T);
          ctx.strokeStyle = '#f97316';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      });
    });
    // circulation regions: thick amber outline
    if (res.labels) {
      for (const lr of res.labels.rooms || []) {
        for (const reg of lr.regions || []) {
          pathPoly(ctx, reg.polygon, T);
          ctx.strokeStyle = '#d97706';
          ctx.lineWidth = 4;
          ctx.setLineDash([]);
          ctx.stroke();
        }
      }
    }
    res.rooms.forEach((room) => {
      drawSegments(ctx, room.cuts || [], T, '#dc2626', [6, 4], 2);
      drawSegments(ctx, room.reversed_cuts || [], T, '#2563eb', [1, 4], 2.5);
    });
  }

  // boundary in black, on top
  const b = boundaryPoints(plan);
  if (b.length > 2) {
    pathPoly(ctx, b, T);
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 3;
    ctx.setLineDash([]);
    ctx.stroke();
  }

  // doors + entrance, over the boundary
  drawOpenings(ctx, plan, T);
  drawBlockEdges(ctx, T);

  // labels
  ctx.fillStyle = '#111';
  ctx.textBaseline = 'middle';
  if (hasLabels) {
    state.result.rooms.forEach((room) => {
      (room.parts || []).forEach((part, pi) => {
        const info = roles.get(`${room.id}#${pi}`);
        if (!info) return;
        const [cx, cy] = centroid(part.polygon);
        ctx.fillText(partText(info), T.ox + T.s * cx, T.oy + T.s * cy);
      });
    });
  } else {
    (plan.rooms || []).forEach((r) => {
      if (!r.polygon || r.polygon.length < 3) return;
      const [cx, cy] = centroid(r.polygon);
      ctx.fillText(String(r.label ?? r.type ?? r.id), T.ox + T.s * cx, T.oy + T.s * cy);
    });
  }

  drawMeasure(ctx);

  // legend
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = '#334155';
  const title = `plan ${state.planId ?? plan.id ?? '?'}`;
  ctx.fillText(title, 12, 20);
  {
    let x = 12;
    const y = h - 14;
    const item = (swatch, label) => {
      swatch(x, y);
      ctx.fillStyle = '#334155';
      ctx.fillText(label, x + 30, y + 4);
      x += 30 + ctx.measureText(label).width + 16;
    };
    const line = (color, dash, width) => (px, py) => {
      ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px + 24, py); ctx.stroke(); ctx.restore();
    };
    item((px, py) => { ctx.fillStyle = DOOR_COLOR; ctx.fillRect(px, py - 3, 24, 6); }, 'door');
    item((px, py) => { ctx.fillStyle = ENTRANCE_COLOR; ctx.fillRect(px, py - 4, 24, 8); }, 'entrance');
    if (res) {
      item((px, py) => { ctx.fillStyle = 'hsla(200,70%,78%,0.75)'; ctx.fillRect(px, py - 6, 24, 12); }, 'part (colour per room)');
      item(line('#dc2626', [6, 4], 2), 'cut kept');
      item(line('#2563eb', [1, 4], 2.5), 'cut reversed');
      item((px, py) => { ctx.strokeStyle = '#f97316'; ctx.lineWidth = 2; ctx.strokeRect(px, py - 6, 24, 12); }, 'sliver');
      if (res.labels) {
        const dashed = (dash, lw) => (px, py) => {
          ctx.fillStyle = 'hsla(200,70%,78%,0.75)'; ctx.fillRect(px, py - 6, 24, 12);
          ctx.save(); ctx.strokeStyle = 'hsl(200,40%,45%)'; ctx.lineWidth = lw; ctx.setLineDash(dash);
          ctx.strokeRect(px, py - 6, 24, 12); ctx.restore();
        };
        item((px, py) => { ctx.fillStyle = CIRC_FILL; ctx.fillRect(px, py - 6, 24, 12); }, 'circulation');
        item(dashed([6, 3], 2), 'zone');
        item(dashed([1, 3], 2), 'jog / alcove');
        item((px, py) => { ctx.fillStyle = NOOK_FILL; ctx.fillRect(px, py - 6, 24, 12); }, 'nook');
        item(dashed([1, 2], 1), 'absorbed');
        item((px, py) => { ctx.strokeStyle = BLOCK_OPEN_COLOR; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(px + 12, py - 7); ctx.lineTo(px + 12, py + 7); ctx.stroke(); }, 'open edge');
        item((px, py) => { ctx.strokeStyle = BLOCK_TOUCH_COLOR; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(px + 12, py - 5); ctx.lineTo(px + 12, py + 5); ctx.stroke(); }, 'touch');
        item((px, py) => {
          ctx.fillStyle = STORAGE_FILL; ctx.fillRect(px, py - 6, 24, 12);
          ctx.save(); ctx.strokeStyle = '#64748b'; ctx.beginPath(); ctx.moveTo(px, py + 6); ctx.lineTo(px + 12, py - 6); ctx.moveTo(px + 12, py + 6); ctx.lineTo(px + 24, py - 6); ctx.stroke(); ctx.restore();
        }, 'storage');
        item((px, py) => { ctx.fillStyle = '#fff'; ctx.fillRect(px, py - 6, 24, 12); ctx.strokeStyle = ORANGE; ctx.lineWidth = 2; ctx.strokeRect(px, py - 6, 24, 12); }, 'residual');
        item((px, py) => { ctx.strokeStyle = '#d97706'; ctx.lineWidth = 4; ctx.strokeRect(px, py - 6, 24, 12); }, 'circ. region');
      }
    }
  }
}

// ── readout ─────────────────────────────────────────────────────────────────

function fmtLens(arr) {
  return (arr || []).map((v) => Number(v).toFixed(2)).join(', ');
}

function renderReadout() {
  const res = state.result;
  if (!res) return;
  const s = res.summary || {};
  const lines = [
    `plan ${res.plan_id ?? state.planId}  sliver<=${res.sliver_max_m} m  merge>=${res.merge_min_contact_m} m`,
    `rooms ${s.n_rooms}  split ${s.n_split}  parts ${s.n_parts}  slivers ${s.n_slivers}  reversed ${s.n_reversed}`,
    `status: ${Object.entries(s.status_counts || {}).map(([k, v]) => `${k} ${v}`).join(', ')}`,
    '',
  ];
  for (const r of res.rooms) {
    let l = `${r.label ?? r.type ?? r.id} [${r.status}] parts ${(r.parts || []).length}`;
    if ((r.cut_lengths_m || []).length) l += `  cuts ${fmtLens(r.cut_lengths_m)} m`;
    if ((r.reversed_cut_lengths_m || []).length) l += `  rev ${fmtLens(r.reversed_cut_lengths_m)} m`;
    lines.push(l);
  }
  const lab = res.labels;
  if (lab) {
    const sm = lab.summary || {};
    lines.push('', `labels: circ ${sm.n_circulation} (regions ${sm.n_regions})  zone ${sm.n_zone}  jog ${sm.n_jog}  alcove ${sm.n_alcove}  storage ${sm.n_storage}  nook ${sm.n_nook}  absorbed ${sm.n_absorbed}  residual ${sm.n_residual}`,
      `pass-through ${sm.n_pass_through}  isolated circ ${sm.n_isolated_circulation}  doors unresolved ${sm.n_door_unresolved}  open unresolved ${sm.n_open_unresolved}`);
    const nm = (node) => {
      if (node === 'exterior') return node;
      const [rid, rest] = String(node).split('/');
      const r = lab.rooms.find((x) => x.id === rid);
      return `${r ? r.label : rid}${rest ? '/' + rest : ''}`;
    };
    for (const lr of lab.rooms) {
      const dr = res.rooms.find((x) => x.id === lr.id);
      const split = lr.parts.length > 1;
      let head = `${lr.label ?? lr.id} (${lr.id})`;
      if (lr.pass_through) head += '  PASS-THROUGH';
      if ((lr.isolated_circulation || []).length) head += `  isolated: ${lr.isolated_circulation.join(',')}`;
      if (lr.kitchen_openable !== null && lr.kitchen_openable !== undefined) head += `  kitchen openable: ${lr.kitchen_openable ? 'yes' : 'no'}`;
      lines.push(head);
      if (!split) continue;
      for (const p of lr.parts) {
        const sz = `${Number(p.short_m).toFixed(2)}x${Number(p.long_m).toFixed(2)} m`;
        const abs = p.absorbed_into !== null && p.absorbed_into !== undefined ? ` -> p${p.absorbed_into}` : '';
        const ext = p.ext_m >= 0.01 ? `  ext ${Number(p.ext_m).toFixed(2)}` : '';
        lines.push(`  p${p.index} ${p.role}${abs}  ${p.node}  ${sz}${ext}  doors ${p.door_ids.length ? p.door_ids.join(',') : '-'}`);
      }
      for (const e of lr.block_edges || []) {
        lines.push(`  edge ${e.kind}: ${e.a.split('/')[1] ?? e.a} | ${e.b.split('/')[1] ?? e.b}  ${Number(e.shared_m).toFixed(2)} m`);
      }
    }
    const doorsC = lab.connections.filter((c) => c.kind !== 'open');
    const openC = lab.connections.filter((c) => c.kind === 'open');
    lines.push('', 'connections (doors):');
    for (const c of doorsC) lines.push(`  ${c.door_id}: ${nm(c.a)} – ${nm(c.b)}`);
    lines.push('connections (open):');
    for (const c of openC) lines.push(`  ${nm(c.a)} – ${nm(c.b)}`);
  } else {
    const conns = connectionLines(state.plan || {});
    if (conns.length) lines.push('', 'connections:', ...conns);
  }
  setReadout(lines.join('\n'));
}

// ── actions ─────────────────────────────────────────────────────────────────

function acceptPlan(plan, idHint) {
  state.plan = plan;
  const pid = Number.isFinite(Number(plan.id)) && plan.id !== '' ? Number(plan.id) : idHint;
  state.planId = pid ?? null;
  state.result = null;
  resetView();
  if (state.planId !== null && $('dc-plan-id')) $('dc-plan-id').value = state.planId;
  $('dc-run').disabled = false;
  const conns = connectionLines(plan);
  setReadout([`plan ${state.planId ?? '?'} loaded: ${(plan.rooms || []).length} rooms, ${(plan.doors || []).length} doors`,
    ...(conns.length ? ['', 'connections:', ...conns] : [])].join('\n'));
  setActive(true);
}

async function loadPlan() {
  const id = parseInt($('dc-plan-id').value, 10);
  if (!Number.isInteger(id) || id < 0) { setReadout('plan id must be a non-negative integer'); return; }
  setReadout('loading...');
  try {
    acceptPlan(await fetchJson(`/rplan/${id}`), id);
  } catch (e) {
    setReadout(`load failed: ${e.message}`);
  }
}

async function loadRandom() {
  setReadout('loading...');
  try {
    acceptPlan(await fetchJson('/rplan/random'), null);
  } catch (e) {
    setReadout(`load failed: ${e.message}`);
  }
}

async function runDecompose() {
  if (!state.plan) return;
  const body = { plan: state.plan };
  const sl = parseFloat($('dc-sliver').value);
  const mg = parseFloat($('dc-merge').value);
  if (Number.isFinite(sl)) body.sliver_max_m = sl;
  if (Number.isFinite(mg)) body.merge_min_contact_m = mg;
  const dcf = parseFloat($('dc-door-frac').value);
  if (Number.isFinite(dcf)) body.door_contact_frac = dcf;
  const ids = { 'dc-absorb-nodoor': 'absorb_nodoor_max_m', 'dc-absorb-door': 'absorb_door_max_m',
    'dc-circ-max': 'circ_max_short_m', 'dc-storage-max': 'storage_max_short_m',
    'dc-nook-min': 'nook_min_short_m', 'dc-ext-min': 'ext_min_m', 'dc-connect-min': 'connect_min_m',
    'dc-absorb-frac': 'absorb_contact_frac' };
  for (const [id, key] of Object.entries(ids)) {
    const v = parseFloat($(id).value);
    if (Number.isFinite(v)) body[key] = v;
  }
  setReadout('decomposing...');
  try {
    state.result = await fetchJson('/decompose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (state.result.plan_id == null) state.result.plan_id = state.planId;
    renderReadout();
    draw();
  } catch (e) {
    setReadout(`decompose failed: ${e.message}`);
  }
}

// ── measure tool ────────────────────────────────────────────────────────────
// Reuses #btn-measure-mode: main.js toggles its 'active' class; we read that.
// The decompose canvas sits over the main canvas, so clicks reach us only here.

function measureOn() {
  const b = $('btn-measure-mode');
  return !!(state.active && b && b.classList.contains('active'));
}

function snapPoint(px, py) {
  const T = state.T;
  const polys = [];
  if (state.result) state.result.rooms.forEach((r) => (r.parts || []).forEach((p) => polys.push(p.polygon)));
  else (state.plan.rooms || []).forEach((r) => polys.push(r.polygon || []));
  let best = null, bd = 8;
  for (const poly of polys) {
    for (const [x, y] of poly) {
      const d = Math.hypot(T.ox + T.s * x - px, T.oy + T.s * y - py);
      if (d <= bd) { bd = d; best = [x, y]; }
    }
  }
  return best || [(px - T.ox) / T.s, (py - T.oy) / T.s];
}

// ── zoom / pan ──────────────────────────────────────────────────────────────
// Wheel zooms about the cursor; left-drag pans (when measure is off);
// double-click resets. The view is folded into state.T, so measure + snap work.

function resetView() {
  state.view = { zoom: 1, dx: 0, dy: 0 };
}

function onWheel(e) {
  if (!state.active || !state.fit || !state.T) return;
  e.preventDefault();
  const rect = e.currentTarget.getBoundingClientRect();
  const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
  const V = state.view, F = state.fit, T = state.T;
  const nz = Math.min(20, Math.max(0.5, V.zoom * Math.exp(-e.deltaY * 0.0015)));
  const px = (cx - T.ox) / T.s, py = (cy - T.oy) / T.s; // plan point under cursor stays put
  V.dx = cx - nz * (F.ox + F.s * px);
  V.dy = cy - nz * (F.oy + F.s * py);
  V.zoom = nz;
  draw();
}

function onPointerDown(e) {
  if (!state.active || measureOn() || e.button !== 0) return;
  state.drag = { x: e.clientX, y: e.clientY, dx: state.view.dx, dy: state.view.dy, moved: false };
}

function onPointerMove(e) {
  const d = state.drag;
  if (!d) return;
  const mx = e.clientX - d.x, my = e.clientY - d.y;
  if (!d.moved && Math.hypot(mx, my) < 3) return;
  d.moved = true;
  state.view.dx = d.dx + mx;
  state.view.dy = d.dy + my;
  draw();
}

function onPointerUp() {
  state.drag = null;
}

function onCanvasClick(e) {
  if (!measureOn() || !state.T) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const pt = snapPoint(e.clientX - rect.left, e.clientY - rect.top);
  const m = state.measure;
  if (!m.a || m.b) { m.a = pt; m.b = null; } else m.b = pt;
  draw();
}

function clearMeasure() {
  if (!state.measure.a) return;
  state.measure = { a: null, b: null };
  if (state.active) draw();
}

export function initDecomposePanel() {
  const on = (id, ev, fn) => { const el = $(id); if (el) el.addEventListener(ev, fn); };
  on('dc-load', 'click', loadPlan);
  on('dc-random', 'click', loadRandom);
  on('dc-run', 'click', runDecompose);
  on('dc-close', 'click', () => setActive(false));
  on('dc-plan-id', 'keydown', (e) => { if (e.key === 'Enter') loadPlan(); });
  window.addEventListener('resize', () => { if (state.active) draw(); });
  on('decompose-canvas', 'click', onCanvasClick);
  const dcv = $('decompose-canvas');
  if (dcv) {
    dcv.addEventListener('wheel', onWheel, { passive: false });
    dcv.addEventListener('mousedown', onPointerDown);
    dcv.addEventListener('dblclick', () => { if (!measureOn()) { resetView(); draw(); } });
  }
  window.addEventListener('mousemove', onPointerMove);
  window.addEventListener('mouseup', onPointerUp);
  window.addEventListener('keydown', (e) => { if (e.key === 'Escape') clearMeasure(); });
  const mb = $('btn-measure-mode');
  if (mb && typeof MutationObserver !== 'undefined') {
    new MutationObserver(() => { if (!mb.classList.contains('active')) clearMeasure(); })
      .observe(mb, { attributes: true, attributeFilter: ['class'] });
  }
}
