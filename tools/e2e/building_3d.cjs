// E2E for the composed building (2026-11-09, G1 part 2b): load a library plate, assemble, generate all units,
// "build 3D" (4 storeys), check the massing response and the 3D scene (prisms, windows, core, roof, ground),
// rebuild with a max height (fewer storeys), run a depth capture, check the scene source rule.
// Real assertions; exit 1 on any failure.
// Starts its own backend (Flask app only) and its own Vite on ports that are neither the dev
// ports (5002, 5173), the gate's (5012, 5183) nor floor_plate_library.cjs's (5022, 5193), compose_plate.cjs's (5025, 5196) nor walls_doors.cjs's (5026, 5197).
// FLOOR_PLATE_LIBRARY is a temp copy of data/floor_plates/library.json; the real file is never touched.
//
// Env: API_PORT (5027), WEB_PORT (5198), SHOT_DIR (screenshots, default os.tmpdir()), PLAYWRIGHT_PATH (optional).
// Per plate (test-plate-001, -002, -002-2; mix 1B1b:1, 2B1b:1): assemble, "generate all units", check one badge per unit
// and room labels in the main-canvas text, screenshot compose_<id>_before.png; open u1, choose rank 2, generate, back to plate,
// check u1 badge reads edited, screenshot compose_<id>_after.png; export JSON; click a unit on the canvas.
// Exit: 0 pass, 1 assertion failed, 3 Playwright not found.
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const REPO = path.resolve(__dirname, '..', '..');
const API_PORT = process.env.API_PORT || '5027';
const WEB_PORT = process.env.WEB_PORT || '5198';
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



const PLATES = (process.env.PLATES || 'test-plate-001').split(',');

(async () => {
  for (const p of [API_PORT, WEB_PORT]) {
    if (!(await portFree(p))) { console.error(`E2E FAIL: port ${p} is in use`); process.exit(1); }
  }
  fs.mkdirSync(SHOT_DIR, { recursive: true });
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fp_b3d_e2e_'));
  const libFile = path.join(tmpDir, 'library.json');
  fs.copyFileSync(REAL_LIB, libFile);
  const env = { ...process.env, PYTHONHASHSEED: '0', FLOOR_PLATE_LIBRARY: libFile, GATE_API_PORT: API_PORT };
  const procs = [];
  const start = (cmd, args, opts) => { const p = spawn(cmd, args, { ...opts, detached: true, stdio: 'ignore' }); procs.push(p); return p; };
  let browser = null;
  let code = 1;
  const summary = {};
  try {
    start(PY, ['tools/e2e/run_backend.py', '--port', API_PORT], { cwd: REPO, env });
    start('./node_modules/.bin/vite', ['--port', WEB_PORT, '--strictPort', '--host', '127.0.0.1'], { cwd: path.join(REPO, 'apps/web'), env });
    if (!(await waitHttp(`${API}/floor_plates`, 240))) throw new Error('backend did not start');
    if (!(await waitHttp(`${APP}/`, 60))) throw new Error('vite did not start');

    browser = await pw.chromium.launch({ headless: true, args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
    const ctx = await browser.newContext({ viewport: { width: 1500, height: 950 } });
    const page = await ctx.newPage();
    const dialogs = [];
    const pageErrors = [];
    const consoleErrors = [];
    page.on('dialog', async (d) => { dialogs.push(d.message()); await d.dismiss(); });
    page.on('pageerror', (e) => pageErrors.push(String(e.message).slice(0, 200)));
    page.on('console', (m) => { if (m.type() === 'error' && !/favicon|Failed to load resource/.test(m.text())) consoleErrors.push(m.text().slice(0, 200)); });
    let asmRes = null;
    let cmpRes = null;
    let bldRes = null;
    page.on('response', async (r) => {
      try {
        if (r.url().endsWith('/assemble_floor_plate')) asmRes = await r.json();
        if (r.url().endsWith('/compose_plate')) cmpRes = await r.json();
        if (r.url().endsWith('/compose_plate/massing')) bldRes = await r.json();
      } catch (e) { /* ignore */ }
    });
    const scene = () => page.evaluate(() => {
      const v = window.__viewer3D;
      const out = { prisms: 0, windows: 0, ground: 0, roles: {}, capture: {}, lastBuild: v && v.lastBuild, groundVisible: null, y: [0, 0] };
      if (!v || !v.planGroup) return out;
      let ymin = Infinity, ymax = -Infinity;
      v.planGroup.children.forEach((o) => {
        const pr = o.userData.prismRole;
        const cr = o.userData.captureRole;
        if (cr) out.capture[cr] = (out.capture[cr] || 0) + 1;
        if (pr === 'window') out.windows++;
        else if (pr) { out.prisms++; out.roles[pr] = (out.roles[pr] || 0) + 1; }
        if (cr === 'ground') { out.ground++; out.groundVisible = o.visible; }
        else if (window.THREE) { const b = new window.THREE.Box3().setFromObject(o); ymin = Math.min(ymin, b.min.y); ymax = Math.max(ymax, b.max.y); }
      });
      out.y = [ymin, ymax];
      return out;
    });

    for (const pid of PLATES) {
      console.log(`--- ${pid}`);
      const tag = (n) => `${n} [${pid}]`;
      await page.goto(APP);
      await sleep(1500);
      await page.click('#tab-fp');
      await page.waitForSelector('#fpl-section', { state: 'visible', timeout: 5000 });
      await page.click('#hud-head');
      await sleep(200);
      await page.waitForSelector('.fpl-row', { timeout: 5000 });
      await page.click(`.fpl-row[data-id="${pid}"] .fpl-load`);
      await sleep(400);
      for (const [t, v] of [['1B1b', '1'], ['2B1b', '1'], ['2B2b', '0'], ['3B2b', '0']]) await page.fill(`#fpl-mix-${t}`, v);
      asmRes = null;
      await page.click('#fpl-btn-assemble');
      for (let i = 0; i < 100 && !asmRes; i++) await sleep(100);
      await sleep(500);
      check(tag('assemble ok'), asmRes && asmRes.status === 'ok', asmRes && asmRes.reason);
      check(tag('build controls hidden before generate'), !(await page.isVisible('#fpl-btn-build3d')));
      cmpRes = null;
      await page.click('#fpl-btn-compose');
      for (let i = 0; i < 1800 && !cmpRes; i++) await sleep(100);
      await sleep(800);
      check(tag('compose ok'), cmpRes && cmpRes.status === 'ok', JSON.stringify(cmpRes).slice(0, 200));
      const opts = cmpRes && cmpRes.record ? cmpRes.record.options : {};
      check(tag('main_entrance_m reaches the record options'), Array.isArray(opts.main_entrance_m) && opts.main_entrance_m.length === 2, JSON.stringify(opts));
      check(tag('build controls shown after generate'), await page.isVisible('#fpl-btn-build3d'));
      check(tag('defaults 4 storeys, 3.1 m, max height blank'),
        (await page.inputValue('#fpl-storeys')) === '4' && (await page.inputValue('#fpl-f2f')) === '3.1' && (await page.inputValue('#fpl-maxh')) === '');

      // build 3D, 4 storeys
      bldRes = null;
      await page.click('#fpl-btn-build3d');
      for (let i = 0; i < 300 && !bldRes; i++) await sleep(100);
      await sleep(2500);
      check(tag('massing response has no error'), bldRes && !bldRes.error, JSON.stringify(bldRes).slice(0, 200));
      const b = bldRes || { prisms: [], windows: [], heights: [] };
      check(tag('3D tab active after build'), await page.evaluate(() => document.getElementById('tab-3d').classList.contains('active')));
      check(tag('storeys = 4'), b.storeys === 4 && b.heights.length === 4, b.storeys);
      const typical = b.prisms.filter((p) => p.storey === 1).length;
      const ground = b.prisms.filter((p) => p.storey === 0).length;
      const cores = b.prisms.filter((p) => p.role === 'core').length;
      const roofs = b.prisms.filter((p) => p.role === 'roof').length;
      check(tag('prism count = ground + 3 x typical + core + roof'), b.prisms.length === ground + 3 * typical + cores + roofs && cores === 1 && roofs === 1, JSON.stringify({ n: b.prisms.length, ground, typical, cores, roofs }));
      check(tag('typical storeys 2 and 3 repeat storey 1'), [2, 3].every((i) => b.prisms.filter((p) => p.storey === i).length === typical));
      check(tag('window count above 0'), b.windows.length > 0, b.windows.length);
      check(tag('one entrance_lobby on the ground floor'), b.prisms.filter((p) => p.role === 'entrance_lobby' && p.storey === 0).length === 1);
      const top = Math.max(...b.prisms.filter((p) => p.role === 'core').map((p) => p.z1));
      check(tag('core rises above the roof'), top > b.total_height_m, JSON.stringify([top, b.total_height_m]));
      // informational: non-slab prisms whose outline shares a collinear edge stretch with the core footprint
      {
        const core = b.prisms.find((p) => p.role === 'core');
        const segs = (poly) => poly.map((q, i) => [q, poly[(i + 1) % poly.length]]);
        const shared = (A, B) => { // collinear overlap length > 1 cm between any edge pair
          for (const [a0, a1] of segs(A)) for (const [b0, b1] of segs(B)) {
            const dx = a1[0] - a0[0], dy = a1[1] - a0[1], L = Math.hypot(dx, dy); if (L < 1e-9) continue;
            const ux = dx / L, uy = dy / L;
            const d0 = Math.abs((b0[0] - a0[0]) * uy - (b0[1] - a0[1]) * ux), d1 = Math.abs((b1[0] - a0[0]) * uy - (b1[1] - a0[1]) * ux);
            if (d0 > 1e-3 || d1 > 1e-3) continue;
            const t0 = (b0[0] - a0[0]) * ux + (b0[1] - a0[1]) * uy, t1 = (b1[0] - a0[0]) * ux + (b1[1] - a0[1]) * uy;
            if (Math.min(L, Math.max(t0, t1)) - Math.max(0, Math.min(t0, t1)) > 0.01) return true;
          }
          return false;
        };
        const hits = b.prisms.filter((p) => p !== core && p.storey === 1 && shared(core.polygon, p.polygon)).map((p) => `${p.role}:${p.stem}`);
        console.log('info: storey-1 prisms sharing a face with the core:', JSON.stringify(hits));
      }
      let sc = await scene();
      console.log('scene', JSON.stringify(sc));
      check(tag('scene meshes = massing prisms'), sc.prisms === b.prisms.length, JSON.stringify([sc.prisms, b.prisms.length]));
      const nTransom = b.windows.filter((w) => w.transom_m).length;
      const slabPrisms = b.prisms.filter((p) => p.role === 'slab' || p.role === 'roof').length;
      check(tag('living/dining windows carry transom_m 0.9'), nTransom > 0 && b.windows.filter((w) => w.transom_m).every((w) => w.transom_m === 0.9), nTransom);
      check(tag('scene window meshes = windows + 2 per transom window'), sc.windows === b.windows.length + 2 * nTransom, JSON.stringify([sc.windows, b.windows.length, nTransom]));
      check(tag('captureRole tags: walls, slabs, windows, ground'), sc.capture.walls > 0 && sc.capture.slabs === slabPrisms && sc.capture.windows === b.windows.length + 2 * nTransom && sc.ground === 1, JSON.stringify(sc.capture));
      check(tag('scene height matches total height'), Math.abs(sc.y[1] - (b.total_height_m + 1.2)) < 0.01, JSON.stringify(sc.y));
      const status = await page.textContent('#viewer-3d-status');
      check(tag('3D status names the source'), /source: composed building/.test(status), status);
      await page.screenshot({ path: path.join(SHOT_DIR, `building_${pid}_4storeys.png`) });

      // depth capture
      await page.click('#btn-capture-maps');
      await sleep(1500);
      const cap = await page.evaluate(() => {
        const s = window.__renderCaptureState || {};
        return { depth: typeof s.depthDataUrl === 'string' ? s.depthDataUrl.slice(0, 22) : null, preview: typeof s.previewDataUrl === 'string' ? s.previewDataUrl.slice(0, 22) : null, opacity: typeof s.opacityMaskDataUrl === 'string' };
      });
      check(tag('depth capture returns a data URL'), cap.depth && cap.depth.startsWith('data:image/'), JSON.stringify(cap));
      check(tag('preview and opacity mask captured'), cap.preview && cap.opacity, JSON.stringify(cap));
      check(tag('ground plane restored visible after capture'), (await scene()).groundVisible === true);

      // max height: 8 m gives fewer storeys
      await page.click('#tab-fp');
      await sleep(800);
      await page.fill('#fpl-maxh', '8');
      bldRes = null;
      await page.click('#fpl-btn-build3d');
      for (let i = 0; i < 300 && !bldRes; i++) await sleep(100);
      await sleep(2000);
      const b2 = bldRes || { prisms: [], windows: [] };
      check(tag('max height 8 m reduces storeys below 4'), b2.storeys >= 1 && b2.storeys < 4 && b2.total_height_m <= 8, JSON.stringify([b2.storeys, b2.total_height_m]));
      check(tag('requested stays 4, achieved reported'), b2.storeys_requested === 4 && b2.report && b2.report.storeys_achieved === b2.storeys);
      sc = await scene();
      check(tag('scene follows the new building'), sc.prisms === b2.prisms.length && sc.windows === b2.windows.length + 2 * b2.windows.filter((w) => w.transom_m).length, JSON.stringify([sc.prisms, b2.prisms.length]));
      await page.screenshot({ path: path.join(SHOT_DIR, `building_${pid}_maxh.png`) });

      // source rule: 3D reached from the Unit tab does not show the building
      await page.click('#tab-2d');
      await sleep(500);
      await page.click('#tab-3d');
      await sleep(1500);
      sc = await scene();
      check(tag('3D from the Unit tab keeps the plan behaviour'), sc.lastBuild && sc.lastBuild.source === 'plan', JSON.stringify(sc.lastBuild));
      summary[pid] = { storeys: b.storeys, total: b.total_height_m, prisms: b.prisms.length, windows: b.windows.length, entrance: b.report && b.report.entrance_unit, maxh: [b2.storeys, b2.total_height_m] };
    }
    check('no page errors', pageErrors.length === 0, JSON.stringify(pageErrors));
    check('no console errors', consoleErrors.length === 0, JSON.stringify(consoleErrors));
    check('no dialogs', dialogs.length === 0, JSON.stringify(dialogs));
    console.log('SUMMARY', JSON.stringify(summary));
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
