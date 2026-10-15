// Task 1 (docs/pipeline-architecture.md, "Work method — gated tasks") —
// human-examination harness, built into the real app rather than a
// side page, so any schema.py fixture can be loaded and checked here
// as they're produced by future pipeline stages (retrieval, decoder,
// floor-plate composer), not just the three test cases.
//
// Deliberately isolated from the rest of main.js: this module owns its
// own DOM (the #pd-modal overlay), its own file loading (via the app's
// existing setupDropZone helper), and calls into plan-drawing-renderer.js
// for the actual drawing. Nothing here touches state.js or the app's own
// dict/JSON plan format — the whole point of the two-space contract is
// that this module only ever needs to know the schema.py shape.

import { setupDropZone } from './retrieval.js';
import { renderPlanDrawing } from './plan-drawing-renderer.js';

let currentFixture = null;

function normaliseFixture(data) {
  if (data && data.boundary && data.plan) return data;
  throw new Error('expected a {boundary, plan} fixture (schema.py contract) — got something else');
}

function renderViolations(fixture) {
  const el = document.getElementById('pd-violations');
  el.innerHTML = '';
  const violations = fixture.violations || [];
  if (violations.length === 0) {
    el.innerHTML = '<div class="pd-violation">none</div>';
    return;
  }
  for (const v of violations) {
    const row = document.createElement('div');
    row.className = 'pd-violation' + (v.severity === 'blocking' ? ' blocking' : '');
    row.innerHTML = `<div class="pd-v-head">${v.severity} · ${v.type} · ${v.room_id}</div><div>${v.detail}</div>`;
    el.appendChild(row);
  }
}

function renderStats(fixture) {
  document.getElementById('pd-modal-label').textContent = fixture.label || '(unlabelled)';
  document.getElementById('pd-stat-schema').textContent = fixture.schema_version || '—';
  document.getElementById('pd-stat-rooms').textContent = Object.keys(fixture.plan.rooms).length;
  document.getElementById('pd-stat-edges').textContent = Object.keys(fixture.plan.edges).length;
}

function draw() {
  if (!currentFixture) return;
  const canvas = document.getElementById('pd-canvas');
  const ctx = canvas.getContext('2d');

  const boundary = currentFixture.boundary.polygon;
  const xs = boundary.map((p) => p[0]);
  const ys = boundary.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const marginMm = 600;
  const w = maxX - minX + marginMm * 2;
  const h = maxY - minY + marginMm * 2;
  const scale = Math.min(canvas.width / w, canvas.height / h);

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.translate(-((minX - marginMm) * scale), -((minY - marginMm) * scale));
  ctx.scale(scale, scale);

  renderPlanDrawing(ctx, currentFixture, {
    scale,
    showWalls: document.getElementById('pd-toggle-walls').checked,
    showDoors: document.getElementById('pd-toggle-doors').checked,
    showOutlines: document.getElementById('pd-toggle-outlines').checked,
    showLabels: document.getElementById('pd-toggle-labels').checked,
  });

  ctx.restore();
}

function openModal() {
  document.getElementById('pd-modal').classList.add('show');
  draw();
}

function closeModal() {
  document.getElementById('pd-modal').classList.remove('show');
}

function loadFixture(data, filename) {
  try {
    currentFixture = normaliseFixture(data);
  } catch (err) {
    alert(`could not load ${filename}: ${err.message}`);
    return;
  }
  renderStats(currentFixture);
  renderViolations(currentFixture);
  openModal();
}

// Entry point for fixtures produced live (e.g. /fit_schema), not just files
// dropped by hand — same {boundary, plan} contract either way.
export function loadFixtureData(data, label = '(live fit)') {
  loadFixture(data, label);
}

export function initPlanDrawingPanel() {
  setupDropZone('dz-plan-drawing', 'fi-plan-drawing', 'lbl-plan-drawing', (files, lbl) => {
    const f = files[0];
    if (!f) return;
    lbl.textContent = f.name;
    f.text().then((txt) => {
      let data;
      try {
        data = JSON.parse(txt);
      } catch (err) {
        alert(`${f.name} is not valid JSON: ${err.message}`);
        return;
      }
      loadFixture(data, f.name);
    });
  });

  document.getElementById('pd-modal-close').addEventListener('click', closeModal);
  document.getElementById('pd-modal').addEventListener('click', (e) => {
    if (e.target.id === 'pd-modal') closeModal(); // click on backdrop
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.getElementById('pd-modal').classList.contains('show')) closeModal();
  });

  for (const id of ['pd-toggle-walls', 'pd-toggle-doors', 'pd-toggle-outlines', 'pd-toggle-labels']) {
    document.getElementById(id).addEventListener('change', draw);
  }
}
