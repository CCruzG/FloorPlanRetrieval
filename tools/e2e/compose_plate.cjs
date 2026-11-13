// E2E for plate composition (2026-11-09): generate all units, badges and room labels on the Floor plate tab,
// drill into u1, choose rank 2, generate, back to plate (u1 edited), export, click-to-open. Real assertions; exit 1 on any failure.
// Starts its own backend (Flask app only) and its own Vite on ports that are neither the dev
// ports (5002, 5173), the gate's (5012, 5183) nor floor_plate_library.cjs's (5022, 5193).
// FLOOR_PLATE_LIBRARY is a temp copy of data/floor_plates/library.json; the real file is never touched.
//
// Env: API_PORT (5025), WEB_PORT (5196), SHOT_DIR (screenshots, default os.tmpdir()), PLAYWRIGHT_PATH (optional).
// Per plate (test-plate-001, -002, -002-2; mix all 1): assemble, "generate all units", check one badge per unit
// and room labels in the main-canvas text, screenshot compose_<id>_before.png; open u1, choose rank 2, generate, back to plate,
// check u1 badge reads edited, screenshot compose_<id>_after.png; export JSON; click a unit on the canvas.
// Exit: 0 pass, 1 assertion failed, 3 Playwright not found.
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const REPO = path.resolve(__dirname, '..', '..');
const API_PORT = process.env.API_PORT || '5025';
const WEB_PORT = process.env.WEB_PORT || '5196';
const APP = `http://127.0.0.1:${WEB_PORT}`;
const API = `http://127.0.0.1:${API_PORT}`;
const SHOT_DIR = process.env.SHOT_DIR || os.tmpdir();
const PY = process.env.PYTHON || path.join(REPO, '.venv', 'bin', 'python');
const REAL_LIB = path.join(REPO, 'tests', 'fixtures', 'floor_plates_g0.json');  // fixed G0 plates; the live library is user data

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


const PLATES = (process.env.PLATES || 'test-plate-001,test-plate-002,test-plate-002-2').split(',');
const BADGE = /^(u\d+) (\S+) (\d+\/\d+|failed)( edited)?$/;

(async () => {
  for (const p of [API_PORT, WEB_PORT]) {
    if (!(await portFree(p))) { console.error(`E2E FAIL: port ${p} is in use`); process.exit(1); }
  }
  fs.mkdirSync(SHOT_DIR, { recursive: true });
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fp_cmp_e2e_'));
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

    browser = await pw.chromium.launch({ headless: true });
    const ctx = await browser.newContext({ viewport: { width: 1500, height: 950 }, acceptDownloads: true });
    const page = await ctx.newPage();
    const dialogs = [];
    const pageErrors = [];
    page.on('dialog', async (d) => { dialogs.push(d.message()); await d.dismiss(); });
    page.on('pageerror', (e) => pageErrors.push(String(e.message).slice(0, 200)));
    let asmRes = null;
    let cmpRes = null;
    let retrieveReqs = 0;
    let composeBody = null;
    page.on('request', (r) => {
      if (r.url().endsWith('/retrieve')) retrieveReqs++;
      if (r.url().endsWith('/compose_plate') && !composeBody) { try { composeBody = JSON.parse(r.postData() || '{}'); } catch (e) { composeBody = {}; } }
    });
    page.on('response', async (r) => {
      try {
        if (r.url().endsWith('/assemble_floor_plate')) asmRes = await r.json();
        if (r.url().endsWith('/compose_plate')) cmpRes = await r.json();
      } catch (e) { /* ignore */ }
    });
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
    const texts = () => page.evaluate(() => window.__texts.slice());
    const badges = async () => (await texts()).filter((t) => BADGE.test(t));

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
      for (const [t, v] of [['1B1b', '1'], ['2B1b', '1'], ['2B2b', '1'], ['3B2b', '1']]) await page.fill(`#fpl-mix-${t}`, v);
      const bv0 = await page.textContent('#s-bv');
      asmRes = null;
      await page.click('#fpl-btn-assemble');
      for (let i = 0; i < 100 && !asmRes; i++) await sleep(100);
      await sleep(500);
      check(tag('assemble ok'), asmRes && asmRes.status === 'ok', asmRes && asmRes.reason);
      const units = asmRes ? asmRes.units : [];
      check(tag('generate-all button present'), (await page.$('#fpl-btn-compose')) !== null);
      check(tag('export button hidden before generate'), !(await page.isVisible('#fpl-btn-export')));

      // generate all
      cmpRes = null;
      check(tag('balconies checkbox present, off by default'), (await page.$('#fpl-chk-balcony')) !== null && !(await page.isChecked('#fpl-chk-balcony')));
      await page.click('#fpl-btn-compose');
      for (let i = 0; i < 1800 && !cmpRes; i++) await sleep(100);
      await sleep(800);
      check(tag('compose ok'), cmpRes && cmpRes.status === 'ok', JSON.stringify(cmpRes).slice(0, 200));
      const slots = cmpRes && cmpRes.record ? cmpRes.record.slots : {};
      check(tag('one slot per unit, none pending'), Object.keys(slots).length === units.length && Object.values(slots).every((s) => s.status !== 'pending'), JSON.stringify(Object.values(slots).map((s) => s.status)));
      check(tag('request body carries balcony false'), composeBody && composeBody.balcony === false, JSON.stringify(composeBody && composeBody.balcony));
      check(tag('requests balcony false and bedrooms from type'), units.every((u) => slots[u.id] && slots[u.id].requirements.balcony === false && slots[u.id].requirements.bedrooms === Number(u.type[0])));
      let b = await badges();
      console.log('badges:', JSON.stringify(b));
      check(tag('one badge per unit on the Floor plate tab'), b.length === units.length && units.every((u) => b.some((t) => t.startsWith(`${u.id} ${u.type} `))), JSON.stringify(b));
      summary[pid] = { before: b.slice() };
      const tx = await texts();
      const roomTypes = new Set();
      for (const s of Object.values(slots)) for (const r of ((s.plan && s.plan.rooms) || [])) if (typeof r.type === 'string') roomTypes.add(r.type);
      const drawnLabels = [...roomTypes].filter((t) => tx.includes(t));
      console.log('room labels on canvas:', JSON.stringify(drawnLabels), 'of', JSON.stringify([...roomTypes]));
      check(tag('room labels drawn in main-canvas text'), drawnLabels.length >= 3, JSON.stringify([tx.length, drawnLabels]));
      const redExpected = Object.values(slots).filter((s) => s.status === 'failed' || (s.sensors && s.sensors.summary && s.sensors.summary.gates_pass === false)).length;
      console.log('hard-fail units (red badges expected):', redExpected);
      await page.screenshot({ path: path.join(SHOT_DIR, `compose_${pid}_before.png`) });
      check(tag('export button shown after generate'), await page.isVisible('#fpl-btn-export'));
      check(tag('assembler overlay kept (4+ unit buttons still present)'), (await page.$$('#fpl-unit-buttons button')).length === units.length);

      // drill down into u1
      retrieveReqs = 0;
      await page.click('#fpl-unit-buttons button[data-unit="u1"]');
      await sleep(900);
      check(tag('Unit tab active after open'), await page.evaluate(() => document.getElementById('tab-2d').classList.contains('active')));
      check(tag('retrieval not run by open'), retrieveReqs === 0, retrieveReqs);
      const s1 = slots.u1;
      check(tag('bedrooms control set from slot'), (await page.inputValue('#rng-bedrooms')) === String(s1.requirements.bedrooms), await page.inputValue('#rng-bedrooms'));
      const nCards = await page.$$eval('.result-card', (c) => c.length);
      check(tag('slot candidates loaded as cards'), nCards === s1.candidates.length && nCards > 0, nCards);
      const sel0 = await page.$$eval('.result-card', (c) => c.map((x) => x.classList.contains('selected')));
      check(tag('rank 1 card selected'), sel0[0] === true && sel0.filter(Boolean).length === 1, JSON.stringify(sel0));
      const sens0 = await page.evaluate(() => document.getElementById('stage1-sensors').innerText);
      check(tag('generated overlay loaded (sensor panel filled)'), /sensors/.test(sens0), sens0.slice(0, 80));
      const hasGen = await page.evaluate(() => /generated|coherent/.test(document.getElementById('generate-readout').textContent));
      check(tag('generate readout shows the slot plan'), hasGen);
      check(tag('back-to-plate button visible'), await page.isVisible('#btn-back-to-plate'));
      await page.screenshot({ path: path.join(SHOT_DIR, `compose_${pid}_unit_u1.png`) });

      // choose rank 2, generate
      await page.locator('.result-card').nth(1).click();
      await sleep(500);
      const gp = page.waitForResponse((r) => r.url().endsWith('/generate'), { timeout: 120000 });
      await page.evaluate(() => document.getElementById('btn-generate').click());
      await gp;
      await sleep(900);
      check(tag('retrieval still not run'), retrieveReqs === 0, retrieveReqs);

      // back to plate
      await page.click('#btn-back-to-plate');
      await sleep(1200);
      check(tag('Floor plate tab active after back'), await page.evaluate(() => document.getElementById('tab-fp').classList.contains('active')));
      check(tag('plate outline restored as boundary'), (await page.textContent('#s-bv')) === bv0, await page.textContent('#s-bv'));
      b = await badges();
      console.log('badges after:', JSON.stringify(b));
      const u1b = b.find((t) => t.startsWith('u1 '));
      check(tag('u1 badge shows edited'), !!u1b && / edited$/.test(u1b), JSON.stringify(b));
      check(tag('other badges unchanged'), units.filter((u) => u.id !== 'u1').every((u) => b.some((t) => t === summary[pid].before.find((x) => x.startsWith(`${u.id} `)))), JSON.stringify(b));
      check(tag('still one badge per unit'), b.length === units.length, JSON.stringify(b));
      summary[pid].after = b.slice();
      await page.screenshot({ path: path.join(SHOT_DIR, `compose_${pid}_after.png`) });

      // export
      const dl = page.waitForEvent('download', { timeout: 20000 });
      await page.click('#fpl-btn-export');
      const d = await dl;
      const dlPath = path.join(SHOT_DIR, `compose_${pid}_export.json`);
      await d.saveAs(dlPath);
      const ex = JSON.parse(fs.readFileSync(dlPath, 'utf8'));
      check(tag('export: u1 edited with rank 2'), ex.slots.u1.status === 'edited' && ex.slots.u1.chosen.rank === 2, JSON.stringify(ex.slots.u1.chosen));
      const xs = ex.plate.boundary.map((q) => q[0]);
      const ys = ex.plate.boundary.map((q) => q[1]);
      const inBox = Object.values(ex.slots).every((s) => !s.plan || s.plan.rooms.every((r) => r.polygon_m.every(([x, y]) => x >= Math.min(...xs) - 0.5 && x <= Math.max(...xs) + 0.5 && y >= Math.min(...ys) - 0.5 && y <= Math.max(...ys) + 0.5)));
      check(tag('export: room polygons in plate metres (inside plate bbox)'), inBox);
      check(tag('export: other slots keep generated status'), Object.values(ex.slots).filter((s) => s.unit_id !== 'u1').every((s) => s.status === 'generated' || s.status === 'failed'));

      // click a unit on the canvas
      const target = units[units.length - 1];
      const pt = await page.evaluate(({ boundary, poly }) => {
        const c = document.getElementById('main-canvas');
        const r = c.getBoundingClientRect();
        const W = r.width, H = r.height;
        const xs = boundary.map((q) => q[0] * 1000), ys = boundary.map((q) => q[1] * 1000);
        const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
        const sx = Math.max(maxX - minX, 1e-6), sy = Math.max(maxY - minY, 1e-6);
        const scale = Math.min(W * 0.84 / sx, H * 0.84 / sy);
        const xo = (W - scale * sx) / 2 - minX * scale, yo = (H - scale * sy) / 2 - minY * scale;
        const cx = poly.reduce((a, q) => a + q[0], 0) / poly.length * 1000, cy = poly.reduce((a, q) => a + q[1], 0) / poly.length * 1000;
        return { x: r.left + cx * scale + xo, y: r.top + cy * scale + yo };
      }, { boundary: asmRes.boundary, poly: target.polygon });
      retrieveReqs = 0;
      await page.mouse.click(pt.x, pt.y);
      await sleep(900);
      const onUnit = await page.evaluate(() => document.getElementById('tab-2d').classList.contains('active'));
      check(tag(`click on ${target.id} opens the Unit tab`), onUnit);
      const hint = await page.textContent('#canvas-hint');
      check(tag(`opened unit is ${target.id}`), hint.includes(`unit ${target.id}`), hint);
      check(tag('clicked unit loaded its candidates'), (await page.$$eval('.result-card', (c) => c.length)) === slots[target.id].candidates.length);
      await page.click('#btn-back-to-plate');
      await sleep(1500);
      b = await badges();
      check(tag(`${target.id} edited after return`), b.some((t) => t.startsWith(`${target.id} `) && / edited$/.test(t)), JSON.stringify(b));
      summary[pid].final = b.slice();
    }
    check('no page errors', pageErrors.length === 0, JSON.stringify(pageErrors));
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
