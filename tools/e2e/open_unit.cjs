// E2E for opening an assembled unit in the Unit tab (2026-11-09). Copy of plate_assembler.cjs setup. Real assertions; exit 1 on any failure.
// Starts its own backend (Flask app only) and its own Vite on ports that are neither the dev
// ports (5002, 5173), the gate's (5012, 5183) nor floor_plate_library.cjs's (5022, 5193).
// FLOOR_PLATE_LIBRARY is a temp copy of data/floor_plates/library.json; the real file is never touched.
//
// Env: API_PORT (5023), WEB_PORT (5194), SHOT_DIR (screenshots, default os.tmpdir()), PLAYWRIGHT_PATH (optional).
// Per plate: open the Floor plate tab, hide dev panel and readouts, load the plate, click "assemble plate",
// assert the overlay draws units (type + area labels counted on the canvas), save assembled2_<plate-id>.png.
// Exit: 0 pass, 1 assertion failed, 3 Playwright not found.
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const REPO = path.resolve(__dirname, '..', '..');
const API_PORT = process.env.API_PORT || '5024';
const WEB_PORT = process.env.WEB_PORT || '5195';
const APP = `http://127.0.0.1:${WEB_PORT}`;
const API = `http://127.0.0.1:${API_PORT}`;
const SHOT_DIR = process.env.SHOT_DIR || os.tmpdir();
const PY = process.env.PYTHON || path.join(REPO, '.venv', 'bin', 'python');
const REAL_LIB = path.join(REPO, 'data', 'floor_plates', 'library.json');

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
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fp_asm_e2e_'));
  const libFile = path.join(tmpDir, 'library.json');
  fs.copyFileSync(REAL_LIB, libFile);
  const plates = JSON.parse(fs.readFileSync(libFile, 'utf8')).plates;
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
    check('library temp copy holds the three plates', plates.length === 3, plates.length);

    browser = await pw.chromium.launch({ headless: true });
    const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
    const dialogs = [];
    const pageErrors = [];
    page.on('dialog', async (d) => { dialogs.push(d.message()); await d.dismiss(); });
    page.on('pageerror', (e) => pageErrors.push(String(e.message).slice(0, 200)));
    let lastResponse = null;
    let lastRequest = null;
    page.on('request', (r) => {
      if (r.url().endsWith('/assemble_floor_plate')) { try { lastRequest = JSON.parse(r.postData()); } catch (e) { lastRequest = null; } }
    });
    page.on('response', async (r) => {
      if (r.url().endsWith('/assemble_floor_plate')) { try { lastResponse = await r.json(); } catch (e) { lastResponse = null; } }
    });
    // Record the text the main canvas draws in the current frame (reset on a full clearRect).
    await page.addInitScript(() => {
      window.__texts = [];
      const P = CanvasRenderingContext2D.prototype;
      const ft = P.fillText;
      P.fillText = function (...a) {
        if (this.canvas && this.canvas.id === 'main-canvas') window.__texts.push(String(a[0]));
        return ft.apply(this, a);
      };
      const cr = P.clearRect;
      P.clearRect = function (...a) {
        if (this.canvas && this.canvas.id === 'main-canvas' && a[0] === 0 && a[1] === 0) window.__texts = [];
        return cr.apply(this, a);
      };
    });

    await page.goto(APP);
    await sleep(1500);
    await page.click('#tab-fp');
    await page.waitForSelector('#fpl-section', { state: 'visible', timeout: 5000 });
    await page.click('#hud-head');
    await sleep(200);
    await page.waitForSelector('.fpl-row', { timeout: 5000 });
    const p = plates.find((x) => x.id === 'test-plate-001') || plates[0];
    await page.click(`.fpl-row[data-id="${p.id}"] .fpl-load`);
    await sleep(400);
    for (const [t, v] of [['1B1b', '1'], ['2B1b', '1'], ['2B2b', '0'], ['3B2b', '0']]) await page.fill(`#fpl-mix-${t}`, v);
    lastResponse = null;
    await page.click('#fpl-btn-assemble');
    for (let i = 0; i < 100 && !lastResponse; i++) await sleep(100);
    await sleep(600);
    check('assemble ok', lastResponse && lastResponse.status === 'ok', lastResponse && lastResponse.reason);
    const units = lastResponse ? lastResponse.units : [];
    const btns = await page.$$eval('#fpl-unit-buttons button', (bs) => bs.map((b) => b.textContent));
    check('one open-button per unit', btns.length === units.length && units.length > 0, JSON.stringify(btns));
    console.log('buttons:', JSON.stringify(btns));
    const u = units.find((x) => x.id === 'u4') || units[units.length - 1];
    let retrieveReqs = 0;
    page.on('request', (r) => { if (r.url().endsWith('/retrieve')) retrieveReqs++; });
    let retrieveReq = null;
    page.on('request', (r) => { if (r.url().endsWith('/retrieve')) { try { retrieveReq = JSON.parse(r.postData()); } catch (e) { /* */ } } });
    await page.click(`#fpl-unit-buttons button[data-unit="${u.id}"]`);
    await sleep(600);
    const tab = await page.evaluate(() => document.getElementById('tab-2d').classList.contains('active'));
    check('Unit tab active after open', tab);
    check('retrieval not run by open', retrieveReqs === 0, retrieveReqs);
    const hint = await page.textContent('#canvas-hint');
    console.log('hint:', hint);
    console.log('run-btn:', JSON.stringify(await page.evaluate(() => { const b = document.getElementById('run-btn'); return b && { dis: b.disabled, vis: b.offsetParent !== null, txt: b.textContent }; })));
    await page.click('#run-btn');
    await page.waitForSelector('.result-card', { timeout: 60000 });
    await sleep(800);
    check('retrieve ran', retrieveReq !== null);
    if (retrieveReq) {
      const k = 18;
      const bnd = retrieveReq.boundary.map((q) => [q[0] * k, q[1] * k]);
      const same = bnd.length === u.polygon.length && bnd.every((q, i) => Math.abs(q[0] - u.polygon[i][0]) < 1e-3 && Math.abs(q[1] - u.polygon[i][1]) < 1e-3);
      check('retrieve boundary equals unit polygon', same, JSON.stringify([bnd.slice(0, 2), u.polygon.slice(0, 2)]));
      const ent = retrieveReq.entrance_point;
      check('retrieve entrance equals unit entrance_point', ent && Math.abs(ent[0] * k - u.entrance_point[0]) < 1e-3 && Math.abs(ent[1] * k - u.entrance_point[1]) < 1e-3, JSON.stringify(ent));
      const te = JSON.stringify((retrieveReq.typed_edges || []).map((e) => [e.edge_index, e.type]).sort((a, b) => a[0] - b[0]));
      const ue = JSON.stringify(u.typed_edges.map((e) => [e.edge_index, e.type]).sort((a, b) => a[0] - b[0]));
      console.log('typed_edges sent:', te, '\nunit typed_edges:', ue);
    }
    await page.evaluate(() => { const c = document.getElementById('chk-source-grid'); if (c && !c.checked) c.click(); });
    const nCards = await page.$$eval('.result-card', (c) => c.length);
    let sawPrivate = false;
    for (let i = 0; i < nCards; i++) {
      await page.locator('.result-card').nth(i).click();
      await sleep(500);
      const gp = page.waitForResponse((r) => r.url().endsWith('/generate'), { timeout: 120000 });
      await page.evaluate(() => document.getElementById('btn-generate').click());
      await gp;
      await sleep(800);
      const sens = await page.evaluate(() => document.getElementById('stage1-sensors').innerText);
      const line = (sens.match(/hard reachability[^✓✗]*/) || ['(none)'])[0];
      console.log(`card ${i}: ${line}`);
      if (/private room/.test(line)) sawPrivate = true;
      check(`card ${i}: reachability line in stage 1 panel`, /reachability/.test(sens));
    }
    console.log('any card failed through a private room:', sawPrivate);
    check('no page errors', pageErrors.length === 0, JSON.stringify(pageErrors));
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
