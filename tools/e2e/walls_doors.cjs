// E2E for walls and doors on generated plans (2026-11-09): assemble test-plate-001 (mix 1B1b:1, 2B1b:1),
// generate all units, graph off, screenshot the plate; open u1, generate, screenshot the unit with the graph off.
// Asserts: wall fills are drawn; door leaves drawn on the canvas equal the plan's placed doors
// (plate: sum over the composed slots; unit: the /generate response). Windows (2026-11-09): the plan carries
// windows (> 0) and the canvas draws two glazing strokes (#2b7a9b) per window; no console errors. Exit 1 on any failure.
// Own backend (Flask app only) and own Vite on ports that are neither the dev ports (5002, 5173),
// the gate's (5012, 5183) nor compose_plate.cjs's (5025, 5196).
// FLOOR_PLATE_LIBRARY is a temp copy of data/floor_plates/library.json; the real file is never touched.
//
// Env: API_PORT (5026), WEB_PORT (5197), SHOT_DIR (default os.tmpdir()), PLATE (test-plate-001), PLAYWRIGHT_PATH (optional).
// Exit: 0 pass, 1 assertion failed, 3 Playwright not found.
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const REPO = path.resolve(__dirname, '..', '..');
const API_PORT = process.env.API_PORT || '5026';
const WEB_PORT = process.env.WEB_PORT || '5197';
const APP = `http://127.0.0.1:${WEB_PORT}`;
const API = `http://127.0.0.1:${API_PORT}`;
const SHOT_DIR = process.env.SHOT_DIR || os.tmpdir();
const PLATE = process.env.PLATE || 'test-plate-001';
const PY = process.env.PYTHON || path.join(REPO, '.venv', 'bin', 'python');
const REAL_LIB = path.join(REPO, 'data', 'floor_plates', 'library.json');
const WALL_FILL = '#2b2b2b';
const DOOR_FILL = '#b45309';
const GLAZE = '#2b7a9b';   // window glazing lines: two strokes per window

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

(async () => {
  for (const p of [API_PORT, WEB_PORT]) {
    if (!(await portFree(p))) { console.error(`E2E FAIL: port ${p} is in use`); process.exit(1); }
  }
  fs.mkdirSync(SHOT_DIR, { recursive: true });
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fp_walls_e2e_'));
  const libFile = path.join(tmpDir, 'library.json');
  fs.copyFileSync(REAL_LIB, libFile);
  const env = { ...process.env, PYTHONHASHSEED: '0', FLOOR_PLATE_LIBRARY: libFile, GATE_API_PORT: API_PORT };
  const procs = [];
  const start = (cmd, args, opts) => { const p = spawn(cmd, args, { ...opts, detached: true, stdio: 'ignore' }); procs.push(p); return p; };
  let browser = null;
  let code = 1;
  try {
    start(PY, ['tools/e2e/run_backend.py', '--port', API_PORT], { cwd: REPO, env });
    start('./node_modules/.bin/vite', ['--port', WEB_PORT, '--strictPort', '--host', '127.0.0.1'], { cwd: path.join(REPO, 'apps/web'), env });
    if (!(await waitHttp(`${API}/floor_plates`, 240))) throw new Error('backend did not start');
    if (!(await waitHttp(`${APP}/`, 60))) throw new Error('vite did not start');

    browser = await pw.chromium.launch({ headless: true });
    const ctx = await browser.newContext({ viewport: { width: 1500, height: 950 } });
    const page = await ctx.newPage();
    const dialogs = [];
    const pageErrors = [];
    const consoleErrors = [];
    page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(String(m.text()).slice(0, 200)); });
    page.on('dialog', async (d) => { dialogs.push(d.message()); await d.dismiss(); });
    page.on('pageerror', (e) => pageErrors.push(String(e.message).slice(0, 200)));
    let asmRes = null;
    let cmpRes = null;
    let genRes = null;
    page.on('response', async (r) => {
      try {
        if (r.url().endsWith('/assemble_floor_plate')) asmRes = await r.json();
        if (r.url().endsWith('/compose_plate')) cmpRes = await r.json();
        if (r.url().endsWith('/generate')) genRes = await r.json();
      } catch (e) { /* ignore */ }
    });
    // Count fills by fillStyle on the main canvas since its last full clear (one frame).
    await page.addInitScript(({ W, D, G }) => {
      window.__fills = { wall: 0, door: 0, glaze: 0 };
      const st = CanvasRenderingContext2D.prototype.stroke;
      CanvasRenderingContext2D.prototype.stroke = function (...a) {
        if (this.canvas && this.canvas.id === 'main-canvas' && this.strokeStyle === G) window.__fills.glaze++;
        return st.apply(this, a);
      };
      const P = CanvasRenderingContext2D.prototype;
      const f = P.fill;
      P.fill = function (...a) {
        if (this.canvas && this.canvas.id === 'main-canvas') {
          if (this.fillStyle === W) window.__fills.wall++;
          else if (this.fillStyle === D) window.__fills.door++;
        }
        return f.apply(this, a);
      };
      const cr = P.clearRect;
      P.clearRect = function (...a) {
        if (this.canvas && this.canvas.id === 'main-canvas' && a[0] === 0 && a[1] === 0) window.__fills = { wall: 0, door: 0, glaze: 0 };
        return cr.apply(this, a);
      };
    }, { W: WALL_FILL, D: DOOR_FILL, G: GLAZE });
    const fills = () => page.evaluate(() => ({ ...window.__fills }));
    const setGraph = (on) => page.evaluate((v) => {
      const c = document.getElementById('chk-graph');
      if (c.checked !== v) { c.checked = v; c.dispatchEvent(new Event('change', { bubbles: true })); }
    }, on);

    await page.goto(APP);
    await sleep(1500);
    await page.click('#tab-fp');
    await page.waitForSelector('#fpl-section', { state: 'visible', timeout: 5000 });
    await page.click('#hud-head');
    await sleep(200);
    await page.waitForSelector('.fpl-row', { timeout: 5000 });
    await page.click(`.fpl-row[data-id="${PLATE}"] .fpl-load`);
    await sleep(400);
    for (const [t, v] of [['1B1b', '1'], ['2B1b', '1'], ['2B2b', '0'], ['3B2b', '0']]) await page.fill(`#fpl-mix-${t}`, v);
    asmRes = null;
    await page.click('#fpl-btn-assemble');
    for (let i = 0; i < 100 && !asmRes; i++) await sleep(100);
    await sleep(500);
    check('assemble ok', asmRes && asmRes.status === 'ok', asmRes && asmRes.reason);
    const units = asmRes ? asmRes.units : [];

    await setGraph(false);
    cmpRes = null;
    await page.click('#fpl-btn-compose');
    for (let i = 0; i < 1800 && !cmpRes; i++) await sleep(100);
    await sleep(800);
    check('compose ok', cmpRes && cmpRes.status === 'ok', JSON.stringify(cmpRes).slice(0, 200));
    const slots = cmpRes && cmpRes.record ? cmpRes.record.slots : {};

    // Plate: walls drawn; door leaves = placed doors of all generated slots.
    const perUnit = {};
    let expectedPlate = 0;
    for (const u of units) {
      const s = slots[u.id];
      const pl = s && s.plan;
      const doors = pl ? pl.doors || [] : [];
      const un = pl && pl.report ? pl.report.doors_unplaced || [] : [];
      perUnit[u.id] = { type: u.type, status: s && s.status, placed: doors.length,
                        internal: doors.filter((d) => d.kind === 'internal').length,
                        entrance: doors.filter((d) => d.kind === 'entrance').length,
                        unplaced: un.map((x) => ({ u: x.u, v: x.v, mm: x.segment_mm, reason: x.reason })) };
      if (s && s.status !== 'failed') expectedPlate += doors.length;
    }
    await page.evaluate(() => document.getElementById('main-canvas').dispatchEvent(new Event('mouseleave')));
    await sleep(300);
    const fp = await fills();
    console.log('plate fills', JSON.stringify(fp), 'expected doors', expectedPlate);
    check('plate: wall fills drawn', fp.wall > 0, JSON.stringify(fp));
    let expectedWin = 0;
    for (const u of units) {
      const s = slots[u.id];
      if (s && s.status !== 'failed' && s.plan) expectedWin += (s.plan.windows || []).length;
    }
    check('plate: composed units carry windows', expectedWin > 0, expectedWin);
    check('plate: glazing lines on canvas equal 2 x composed windows', fp.glaze === 2 * expectedWin, `${fp.glaze} vs ${2 * expectedWin}`);
    check('plate: door count on canvas equals composed doors', fp.door === expectedPlate && expectedPlate > 0, `${fp.door} vs ${expectedPlate}`);
    await page.screenshot({ path: path.join(SHOT_DIR, `walls_${PLATE}_plate.png`) });

    // Unit u1: opened from the slot, then generated again; graph off.
    await page.click('#fpl-unit-buttons button[data-unit="u1"]');
    await sleep(900);
    await setGraph(false);
    await sleep(300);
    const f0 = await fills();
    const u1doors = (slots.u1.plan.doors || []).length;
    check('unit (slot plan): wall fills drawn', f0.wall > 0, JSON.stringify(f0));
    check('unit (slot plan): door leaves equal plan doors', f0.door === u1doors, `${f0.door} vs ${u1doors}`);
    genRes = null;
    await page.evaluate(() => document.getElementById('btn-generate').click());
    for (let i = 0; i < 1200 && !genRes; i++) await sleep(100);
    await sleep(900);
    check('generate ok', genRes && genRes.status === 'ok', JSON.stringify(genRes).slice(0, 200));
    const f1 = await fills();
    const gdoors = genRes ? (genRes.doors || []).length : -1;
    console.log('unit fills', JSON.stringify(f1), 'plan doors', gdoors);
    const gwin = genRes ? (genRes.windows || []).length : -1;
    check('unit (generated): window count above 0', gwin > 0, gwin);
    check('unit (generated): glazing lines equal 2 x windows', f1.glaze === 2 * gwin, `${f1.glaze} vs ${2 * gwin}`);
    check('unit (generated): wall fills drawn', f1.wall > 0, JSON.stringify(f1));
    check('unit (generated): door leaves equal plan doors', f1.door === gdoors && gdoors > 0, `${f1.door} vs ${gdoors}`);
    const graphOff = await page.evaluate(() => !document.getElementById('chk-graph').checked);
    check('graph toggle off', graphOff);
    await page.screenshot({ path: path.join(SHOT_DIR, `walls_${PLATE}_unit_u1.png`) });
    // draw walls off: leaves are gone, walls gone
    await page.evaluate(() => { const c = document.getElementById('chk-draw-walls'); c.checked = false; c.dispatchEvent(new Event('change', { bubbles: true })); });
    await sleep(300);
    const f2 = await fills();
    check('draw walls off: no walls, no doors', f2.wall === 0 && f2.door === 0, JSON.stringify(f2));
    await page.evaluate(() => { const c = document.getElementById('chk-draw-walls'); c.checked = true; c.dispatchEvent(new Event('change', { bubbles: true })); });
    await sleep(300);
    const f3 = await fills();
    check('draw walls on again', f3.wall > 0 && f3.door === gdoors, JSON.stringify(f3));

    check('no page errors', pageErrors.length === 0, JSON.stringify(pageErrors));
    check('no console errors', consoleErrors.length === 0, JSON.stringify(consoleErrors));
    check('no dialogs', dialogs.length === 0, JSON.stringify(dialogs));
    console.log('DOORS', JSON.stringify(perUnit));
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
