// E2E for the floor plate library (2026-11-06). Real assertions; exit 1 on any failure.
// Starts its own backend (Flask app only, no retrieval index) and its own Vite, on ports
// that are neither the dev ports (5002, 5173) nor the gate's (5012, 5183). Stops only what it started.
// The library file is a temp file (FLOOR_PLATE_LIBRARY); data/floor_plates/library.json is never touched.
//
// Env: API_PORT (5022), WEB_PORT (5193), SHOT (png path), PLAYWRIGHT_PATH (optional).
// Exit: 0 pass, 1 assertion failed, 3 Playwright not found.
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const REPO = path.resolve(__dirname, '..', '..');
const API_PORT = process.env.API_PORT || '5022';
const WEB_PORT = process.env.WEB_PORT || '5193';
const APP = `http://127.0.0.1:${WEB_PORT}`;
const API = `http://127.0.0.1:${API_PORT}`;
const SHOT = process.env.SHOT || path.join(os.tmpdir(), 'fp_library.png');
const PY = process.env.PYTHON || path.join(REPO, '.venv', 'bin', 'python');

function findPlaywright() {
  const cands = [];
  if (process.env.PLAYWRIGHT_PATH) cands.push(process.env.PLAYWRIGHT_PATH);
  const npx = path.join(os.homedir(), '.npm', '_npx');
  if (fs.existsSync(npx)) for (const d of fs.readdirSync(npx)) cands.push(path.join(npx, d, 'node_modules', 'playwright'));
  for (const c of cands) { try { return require(c); } catch (e) { /* next */ } }
  return null;
}
const pw = findPlaywright();
if (!pw) { console.error('E2E FAIL: Playwright not found. Set PLAYWRIGHT_PATH or run `npx playwright --version` once.'); process.exit(3); }

const results = [];
function check(name, ok, detail) {
  results.push({ name, ok: !!ok });
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail !== undefined && !ok ? ' :: ' + String(detail).slice(0, 300) : ''}`);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function waitHttp(url, secs) {
  const end = Date.now() + secs * 1000;
  while (Date.now() < end) {
    try { const r = await fetch(url); if (r.status < 500) return true; } catch (e) { /* retry */ }
    await sleep(500);
  }
  return false;
}
async function portFree(port) {
  try { await fetch(`http://127.0.0.1:${port}/`); return false; } catch (e) { return true; }
}
const apiJson = async (method, route, body) => {
  const r = await fetch(API + route, { method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
  return { status: r.status, json: await r.json() };
};

// Least-squares A = k*M + t (one scale, translation) over both axes. Returns k and the worst residual in metres.
function fitSimilarity(M, A) {
  const n = M.length;
  const mean = (arr, i) => arr.reduce((s, p) => s + p[i], 0) / n;
  const mx = mean(M, 0), my = mean(M, 1), ax = mean(A, 0), ay = mean(A, 1);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (M[i][0] - mx) * (A[i][0] - ax) + (M[i][1] - my) * (A[i][1] - ay);
    den += (M[i][0] - mx) ** 2 + (M[i][1] - my) ** 2;
  }
  const k = num / den;
  let worst = 0;
  for (let i = 0; i < n; i++) {
    const ex = A[i][0] - (k * (M[i][0] - mx) + ax), ey = A[i][1] - (k * (M[i][1] - my) + ay);
    worst = Math.max(worst, Math.hypot(ex, ey) / k);
  }
  return { k, worst };
}

(async () => {
  for (const p of [API_PORT, WEB_PORT]) {
    if (!(await portFree(p))) { console.error(`E2E FAIL: port ${p} is in use`); process.exit(1); }
  }
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fp_lib_e2e_'));
  const libFile = path.join(tmpDir, 'library.json');
  const env = { ...process.env, PYTHONHASHSEED: '0', FLOOR_PLATE_LIBRARY: libFile, GATE_API_PORT: API_PORT };
  const procs = [];
  const start = (cmd, args, opts) => { const p = spawn(cmd, args, { ...opts, detached: true, stdio: 'ignore' }); procs.push(p); return p; };
  let browser = null;
  let code = 1;
  try {
    start(PY, ['-c', `import sys; sys.path.insert(0, '.'); from apps.api import server; server.app.run(host='127.0.0.1', port=${API_PORT}, debug=False, threaded=True)`], { cwd: REPO, env });
    start('./node_modules/.bin/vite', ['--port', WEB_PORT, '--strictPort', '--host', '127.0.0.1'], { cwd: path.join(REPO, 'apps/web'), env });
    if (!(await waitHttp(`${API}/floor_plates`, 120))) throw new Error('backend did not start');
    if (!(await waitHttp(`${APP}/`, 60))) throw new Error('vite did not start');

    browser = await pw.chromium.launch({ headless: true });
    const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
    const dialogs = [];
    const pageErrors = [];
    page.on('dialog', async (d) => { dialogs.push(d.message()); await d.dismiss(); });
    page.on('pageerror', (e) => pageErrors.push(String(e.message).slice(0, 200)));
    await page.addInitScript(() => {
      window.__arcs = [];
      const P = CanvasRenderingContext2D.prototype;
      const oa = P.arc;
      P.arc = function (...a) {
        if (this.canvas && this.canvas.id === 'main-canvas') {
          const m = this.getTransform();
          window.__arcs.push([m.a * a[0] + m.c * a[1] + m.e, m.b * a[0] + m.d * a[1] + m.f, Math.round(a[2] * Math.abs(m.a) * 100) / 100]);
        }
        return oa.apply(this, a);
      };
      const oc = P.clearRect;
      P.clearRect = function (...a) {
        if (this.canvas && this.canvas.id === 'main-canvas' && a[0] === 0 && a[1] === 0) window.__arcs = [];
        return oc.apply(this, a);
      };
    });

    const vertexArcs = async (n) => {
      const arcs = await page.evaluate(() => window.__arcs);
      const count = {};
      for (const a of arcs) count[a[2]] = (count[a[2]] || 0) + 1;
      const modal = Object.keys(count).sort((x, y) => count[y] - count[x])[0];
      return arcs.filter((a) => String(a[2]) === modal).slice(0, n).map((a) => [a[0], a[1]]);
    };
    const canvasBox = async () => page.locator('#main-canvas').boundingBox();

    const openFpTab = async () => {
      await page.click('#tab-fp');
      await page.waitForSelector('#fpl-section', { state: 'visible', timeout: 5000 });
    };

    // ── 1. section visibility ───────────────────────────────────────────────
    await page.goto(APP);
    await sleep(1500);
    check('section hidden in the Unit tab', !(await page.isVisible('#fpl-section')));
    await openFpTab();
    check('section visible in the Floor plate tab', await page.isVisible('#fpl-section'));
    await page.click('#tab-2d');
    check('section hidden again in the Unit tab', !(await page.isVisible('#fpl-section')));
    await openFpTab();
    // The dev panel (left) and the readouts HUD (right) sit over the canvas and take mouse events.
    await page.click('#hud-head');
    await sleep(200);

    // ── 2. zoom out, calibrate metres -> page pixels ───────────────────────
    const box = await canvasBox();
    const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
    await page.mouse.move(cx, cy);
    await page.click('#btn-draw-boundary');
    let ppm = 0, origin = null;
    const calibrate = async () => {
      const read = async (x, y) => {
        await page.mouse.move(x, y);
        await sleep(60);
        const t = await page.textContent('#coord-disp');
        const m = /(-?[\d.]+) m, (-?[\d.]+) m/.exec(t || '');
        return m ? [parseFloat(m[1]), parseFloat(m[2])] : null;
      };
      const p0 = [box.x + 100, box.y + 100], p1 = [box.x + 100 + 600, box.y + 100 + 300];
      const m0 = await read(...p0), m1 = await read(...p1);
      if (!m0 || !m1) return false;
      ppm = (600 / (m1[0] - m0[0]) + 300 / (m1[1] - m0[1])) / 2;
      origin = { px: p0, m: m0 };
      return true;
    };
    for (let i = 0; i < 30; i++) {
      await page.mouse.move(cx, cy);
      await page.mouse.wheel(0, 120);
      await sleep(40);
      if (i >= 4 && (i % 2 === 0)) { await calibrate(); if (ppm > 0 && ppm * 62 < box.width * 0.85 && ppm * 32 < box.height * 0.85) break; }
    }
    await calibrate();
    check('calibration: px per metre found', ppm > 1 && ppm < 200, ppm);
    const toPx = (mx, my) => [origin.px[0] + (mx - origin.m[0]) * ppm, origin.px[1] + (my - origin.m[1]) * ppm];

    // Draw a w x h m rectangle centred on the canvas (the corner panels stay clear).
    const drawRect = async (w, h) => {
      const mc = [origin.m[0] + (cx - origin.px[0]) / ppm, origin.m[1] + (cy - origin.px[1]) / ppm];
      const x0 = mc[0] - w / 2, y0 = mc[1] - h / 2;
      const v = [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]].map(([a, b]) => toPx(a, b));
      for (const [x, y] of v) { await page.mouse.click(x, y); await sleep(60); }
      await page.keyboard.press('Enter');
      await sleep(250);
    };
    const sizeText = async () => (await page.textContent('#fpl-size')).trim();
    const parseSize = (t) => { const m = /([\d.]+) × ([\d.]+) m/.exec(t); return m ? [parseFloat(m[1]), parseFloat(m[2])] : [NaN, NaN]; };

    // ── 3. clamp check: a 60 x 30 m plate ───────────────────────────────────
    await drawRect(60, 30);
    let [w, h] = parseSize(await sizeText());
    check('60 x 30 m plate can be drawn after zooming out (no clamp)', Math.abs(w - 60) < 1 && Math.abs(h - 30) < 1, await sizeText());
    await page.click('#btn-draw-boundary'); // clears the plate, re-enters draw mode
    await sleep(200);
    check('size readout clears with the boundary', (await sizeText()) === '—', await sizeText());

    // ── 4. the 40 x 20 m plate ─────────────────────────────────────────────
    await drawRect(40, 20);
    [w, h] = parseSize(await sizeText());
    check('size readout shows about 40 x 20 m', Math.abs(w - 40) < 1 && Math.abs(h - 20) < 1, await sizeText());
    const preArcs = await vertexArcs(4);
    check('four vertex markers found', preArcs.length === 4, preArcs.length);

    await page.fill('#fpl-name', 'E2E slab');
    await page.fill('#fpl-notes', 'e2e note');
    await page.fill('#fpl-north', '30');
    const mid = (i) => { const a = preArcs[i], b = preArcs[(i + 1) % 4]; return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]; };
    // the arcs are canvas-buffer px; headless dpr is 1 so they equal page px offset by the canvas box
    const pagePt = (p) => [box.x + p[0], box.y + p[1]];

    // main entrance on edge 0 (in the Floor plate tab it does not set an edge type)
    await page.click('#btn-entrance-mode');
    await page.mouse.click(...pagePt(mid(0)));
    await sleep(200);
    // street frontage: edge 2 set, clear, set, then edge 3 as well
    await page.click('#fpl-btn-street');
    check('street frontage button is active', await page.evaluate(() => document.getElementById('fpl-btn-street').classList.contains('active')));
    await page.mouse.click(...pagePt(mid(2)));
    await sleep(150);
    check('street edge 2 set', (await page.textContent('#fpl-street-info')).includes('2'), await page.textContent('#fpl-street-info'));
    await page.mouse.click(...pagePt(mid(2)));
    await sleep(150);
    check('second click on the same edge clears it', (await page.textContent('#fpl-street-info')).includes('none'), await page.textContent('#fpl-street-info'));
    await page.mouse.click(...pagePt(mid(2)));
    await sleep(150);
    check('street edge 2 set again', (await page.textContent('#fpl-street-info')).includes('2'));
    await page.mouse.click(...pagePt(mid(3)));
    await sleep(150);
    check('a second street edge is added', (await page.textContent('#fpl-street-info')).includes('edges 2, 3'), await page.textContent('#fpl-street-info'));
    await page.mouse.click(...pagePt([cx - box.x, cy - box.y + 4000]).map((v) => v)); // far from any edge: ignored
    await page.click('#fpl-btn-street'); // leave street mode
    check('update disabled before any plate is loaded', await page.isDisabled('#fpl-btn-update'));

    // ── 5. save new ────────────────────────────────────────────────────────
    await page.click('#fpl-btn-save-new');
    await page.waitForSelector('.fpl-row', { timeout: 5000 });
    const listed = await apiJson('GET', '/floor_plates');
    check('plate saved in the file', listed.json.plates.length === 1 && listed.json.plates[0].id === 'e2e-slab', JSON.stringify(listed.json).slice(0, 200));
    const saved = listed.json.plates[0];
    check('saved north 30, street edges 2 and 3, notes', saved.north_deg === 30 && JSON.stringify(saved.street_edges) === '[2,3]' && saved.notes === 'e2e note',
      JSON.stringify([saved.north_deg, saved.street_edges, saved.notes]));
    check('saved entrance present, and it sets no edge type on a plate', Array.isArray(saved.main_entrance_m) && saved.edge_types.length === 0,
      JSON.stringify([saved.main_entrance_m, saved.edge_types]));
    check('saved area is about 800 m2 and is computed', Math.abs(saved.area_m2 - 800) < 25 && !('area_m2' in JSON.parse(fs.readFileSync(libFile, 'utf8')).plates[0])
      , saved.area_m2);
    check('update enabled after save', !(await page.isDisabled('#fpl-btn-update')));
    const fileDoc = JSON.parse(fs.readFileSync(libFile, 'utf8'));
    check('file header', fileDoc.schema_version === 2 && fileDoc.units === 'm' && fileDoc.frame === 'canvas, y down', JSON.stringify(fileDoc).slice(0, 120));

    const savedM = saved.boundary_m;
    const pre = fitSimilarity(savedM, preArcs);
    check('saved metres match the drawn vertices within 0.01 m', pre.worst < 0.01, pre.worst);
    check('saved metres use the canvas scale (px per m within 1%)', Math.abs(pre.k / ppm - 1) < 0.01, `${pre.k} vs ${ppm}`);

    // ── 6. reload, load, compare ────────────────────────────────────────────
    await page.reload();
    await sleep(1500);
    await openFpTab();
    await page.waitForSelector('.fpl-row', { timeout: 5000 });
    const rowTxt = await page.textContent('.fpl-row');
    check('list row shows name, area and vertex count', /E2E slab/.test(rowTxt) && /\d+ m²/.test(rowTxt) && /4 vertices/.test(rowTxt), rowTxt);
    await page.click('.fpl-row .fpl-load');
    await sleep(500);
    const postArcs = await vertexArcs(4);
    const post = fitSimilarity(savedM, postArcs);
    check('loaded vertices match the saved vertices within 0.01 m', post.worst < 0.01, post.worst);
    check('north bearing restored', (await page.inputValue('#fpl-north')) === '30');
    check('street edges restored', (await page.textContent('#fpl-street-info')).includes('edges 2, 3'), await page.textContent('#fpl-street-info'));
    check('name and notes restored', (await page.inputValue('#fpl-name')) === 'E2E slab' && (await page.inputValue('#fpl-notes')) === 'e2e note');
    check('entrance restored', await page.isVisible('#entrance-info'), '');
    const [lw, lh] = parseSize(await sizeText());
    check('size readout after load', Math.abs(lw - w) < 0.01 && Math.abs(lh - h) < 0.01, await sizeText());
    const inside = postArcs.every(([x, y]) => x > 0 && y > 0 && x < box.width && y < box.height);
    check('plate fitted inside the canvas after load', inside, JSON.stringify(postArcs));
    check('update enabled for a loaded plate', !(await page.isDisabled('#fpl-btn-update')));
    await page.locator('#main-canvas').screenshot({ path: SHOT });

    // ── 7. update, rename, delete ───────────────────────────────────────────
    await page.fill('#fpl-north', '45');
    await page.click('#fpl-btn-update');
    await sleep(500);
    let now = (await apiJson('GET', '/floor_plates')).json.plates;
    check('update replaces the plate and keeps the id', now.length === 1 && now[0].id === 'e2e-slab' && now[0].north_deg === 45, JSON.stringify(now.map((p) => [p.id, p.north_deg])));
    check('update through the proxy keeps street edges', JSON.stringify(now[0].street_edges) === '[2,3]');
    check('button reads update boundary', (await page.textContent('#fpl-btn-update')).trim() === 'update boundary');

    // ── 7b. the workflow that matters: load, move the entrance, update, reload ─
    await page.click('#btn-clear-entrance');
    await sleep(150);
    await page.click('#fpl-btn-update');
    await sleep(400);
    now = (await apiJson('GET', '/floor_plates')).json.plates;
    check('entrance removed by update', now[0].main_entrance_m === null, JSON.stringify(now[0].main_entrance_m));
    const a1 = postArcs[1], a2 = postArcs[2];
    await page.click('#btn-entrance-mode');
    const edge1Mid = pagePt([(a1[0] + a2[0]) / 2, (a1[1] + a2[1]) / 2]);
    await page.mouse.move(...edge1Mid);
    await sleep(100);
    await page.mouse.click(...edge1Mid);
    await sleep(200);
    await page.click('#fpl-btn-update');
    await sleep(400);
    now = (await apiJson('GET', '/floor_plates')).json.plates;
    const addedEnt = now[0].main_entrance_m;
    check('entrance added to a loaded plate by update', Array.isArray(addedEnt), JSON.stringify(addedEnt));
    check('update keeps the outline', JSON.stringify(now[0].boundary_m) === JSON.stringify(savedM));
    await page.reload();
    await sleep(1500);
    await openFpTab();
    await page.waitForSelector('.fpl-row', { timeout: 5000 });
    await page.click('.fpl-row .fpl-load');
    await sleep(400);
    check('added entrance survives a reload', await page.isVisible('#entrance-info'));
    check('street edges survive the reload', (await page.textContent('#fpl-street-info')).includes('edges 2, 3'));
    await page.click('.fpl-row .fpl-rename');
    await page.fill('.fpl-rename-input', 'Renamed slab');
    await page.click('.fpl-rename-ok');
    await sleep(500);
    now = (await apiJson('GET', '/floor_plates')).json.plates;
    check('rename keeps id, changes name', now[0].id === 'e2e-slab' && now[0].name === 'Renamed slab', JSON.stringify(now.map((p) => [p.id, p.name])));
    const viaProxy = await page.evaluate(async () => { const r = await fetch('/floor_plates/nope', { method: 'DELETE' }); return [r.status, (await r.json()).error]; });
    check('/floor_plates/<id> reaches Flask through the Vite proxy (404 JSON)', viaProxy[0] === 404 && /not found/.test(viaProxy[1] || ''), JSON.stringify(viaProxy));
    await page.click('.fpl-row .fpl-delete');
    check('delete asks for inline confirmation', await page.isVisible('.fpl-delete-yes'));
    await page.click('.fpl-delete-yes');
    await sleep(500);
    check('delete removes the plate', (await apiJson('GET', '/floor_plates')).json.plates.length === 0);
    check('update disabled after the loaded plate is deleted', await page.isDisabled('#fpl-btn-update'));

    // ── 8. server-side validation shows inline, no dialogs ──────────────────
    await page.click('#fpl-btn-save-new'); // boundary is still closed and valid; saves again
    await sleep(400);
    await page.click('#btn-draw-boundary');
    await page.click('#fpl-btn-save-new'); // nothing drawn
    await sleep(300);
    check('save with no boundary shows an inline message', /close the boundary/.test(await page.textContent('#fpl-status')), await page.textContent('#fpl-status'));
    check('no alert/confirm/prompt dialogs', dialogs.length === 0, JSON.stringify(dialogs));
    check('no page errors', pageErrors.length === 0, JSON.stringify(pageErrors));
    check('screenshot written', fs.existsSync(SHOT), SHOT);
    code = results.every((r) => r.ok) ? 0 : 1;
  } catch (e) {
    console.error('E2E ERROR', e && e.stack || e);
    code = 1;
  } finally {
    if (browser) await browser.close().catch(() => {});
    for (const p of procs) { try { process.kill(-p.pid, 'SIGTERM'); } catch (e) { /* gone */ } }
    await sleep(500);
    for (const p of procs) { try { process.kill(-p.pid, 'SIGKILL'); } catch (e) { /* gone */ } }
    try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch (e) { /* ignore */ }
  }
  const bad = results.filter((r) => !r.ok);
  console.log(`${results.length - bad.length}/${results.length} checks passed`);
  process.exit(bad.length ? 1 : code);
})();
