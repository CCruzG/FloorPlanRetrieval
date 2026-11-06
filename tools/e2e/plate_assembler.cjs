// E2E for the plate assembler (2026-11-06). Real assertions; exit 1 on any failure.
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
const API_PORT = process.env.API_PORT || '5023';
const WEB_PORT = process.env.WEB_PORT || '5194';
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
    start(PY, ['-c', `import sys; sys.path.insert(0, '.'); from apps.api import server; server.app.run(host='127.0.0.1', port=${API_PORT}, debug=False, threaded=True)`], { cwd: REPO, env });
    start('./node_modules/.bin/vite', ['--port', WEB_PORT, '--strictPort', '--host', '127.0.0.1'], { cwd: path.join(REPO, 'apps/web'), env });
    if (!(await waitHttp(`${API}/floor_plates`, 120))) throw new Error('backend did not start');
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
    check('dev panel is hidden when the Floor plate tab opens', !(await page.isVisible('#fp-dev-panel')));
    await page.click('#fp-dev-toggle');
    check('the dev panel toggle shows the panel', await page.isVisible('#fp-dev-panel'));
    await page.click('#fp-dev-hide');
    check('the panel hide button still hides it', !(await page.isVisible('#fp-dev-panel')));
    check('assemble block is in the Floor plate section', await page.isVisible('#fpl-btn-assemble'));
    const defaults = await page.evaluate(() => ['1B1b', '2B1b', '2B2b', '3B2b'].map((t) => document.getElementById('fpl-mix-' + t).value).join(','));
    check('mix weights default to 1,1,0,0', defaults === '1,1,0,0', defaults);
    await page.click('#hud-head');
    await sleep(200);
    await page.waitForSelector('.fpl-row', { timeout: 5000 });

    for (const p of plates) {
      lastResponse = null;
      await page.click(`.fpl-row[data-id="${p.id}"] .fpl-load`);
      await sleep(400);
      check(`${p.id}: no overlay units before assembling`, (await page.evaluate(() => window.__texts.filter((t) => /m2$/.test(t)).length)) === 0);
      await page.click('#fpl-btn-assemble');
      for (let i = 0; i < 100 && !lastResponse; i++) await sleep(100);
      await sleep(600);
      check(`${p.id}: route answered`, !!lastResponse, 'no response');
      const nUnits = lastResponse ? lastResponse.units.length : -1;
      check(`${p.id}: status ok`, lastResponse && lastResponse.status === 'ok', lastResponse && lastResponse.reason);
      const status = (await page.textContent('#fpl-assemble-result')) || '';
      check(`${p.id}: result text shows status, units, commons, walk, residual`,
        /status: ok/.test(status) && new RegExp(`${nUnits} units`).test(status) && /commons [\d.]+%/.test(status) && /farthest walk [\d.]+ m/.test(status) && /residual [\d.]+ m²/.test(status), status);
      const texts = await page.evaluate(() => window.__texts);
      const areaLabels = texts.filter((t) => /^[\d.]+ m2$/.test(t));
      const typeLabels = texts.filter((t) => /\((1B1b|2B1b|2B2b|3B2b)\)$/.test(t));
      check(`${p.id}: overlay draws ${nUnits} unit area labels`, nUnits > 0 && areaLabels.length === nUnits, `${areaLabels.length} vs ${nUnits}`);
      check(`${p.id}: overlay draws a type label per unit`, typeLabels.length === nUnits, `${typeLabels.length} vs ${nUnits}`);
      check(`${p.id}: overlay draws the core block as stair+lift and lobby`, texts.includes('stair+lift') && texts.includes('lobby'), JSON.stringify(texts.slice(0, 8)));
      check(`${p.id}: north bearing sent from the plate`, lastRequest && lastRequest.north_deg === p.north_deg, JSON.stringify(lastRequest && lastRequest.north_deg));
      const shot = path.join(SHOT_DIR, `assembled2_${p.id}.png`);
      await page.locator('#main-canvas').screenshot({ path: shot });
      check(`${p.id}: screenshot written`, fs.existsSync(shot), shot);
    }
    // The mix in Camilo's sketch of test-plate-001: 2B1b, 3B2b, 2B1b = weights 0,2,0,1.
    {
      const p = plates[0];
      await page.click(`.fpl-row[data-id="${p.id}"] .fpl-load`);
      await sleep(400);
      for (const [t, v] of [['1B1b', '0'], ['2B1b', '2'], ['2B2b', '0'], ['3B2b', '1']]) await page.fill(`#fpl-mix-${t}`, v);
      lastResponse = null;
      await page.click('#fpl-btn-assemble');
      for (let i = 0; i < 100 && !lastResponse; i++) await sleep(100);
      await sleep(600);
      check(`${p.id}: user mix 0,2,0,1 answered`, !!lastResponse && lastResponse.status === 'ok', lastResponse && lastResponse.reason);
      const shot = path.join(SHOT_DIR, `assembled2_${p.id}_user.png`);
      await page.locator('#main-canvas').screenshot({ path: shot });
      check(`${p.id}: user-mix screenshot written`, fs.existsSync(shot), shot);
    }
    check('no alert/confirm/prompt dialogs', dialogs.length === 0, JSON.stringify(dialogs));
    check('no page errors', pageErrors.length === 0, JSON.stringify(pageErrors));
    check('real library file unchanged by the run', fs.readFileSync(REAL_LIB, 'utf8').includes('test-plate-001'));
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
