// E2E flow for the gate suite (2026-11-04). Real assertions; exit 1 on any failure.
// Based on drive4.cjs / drive5.cjs / mv.py (job bcc8ab15). Run by tools/gate_all.py,
// which starts the backend (API_URL) and Vite (APP_URL) and stops only those.
//
// Env: APP_URL, API_URL, OUT_DIR (screenshots), LIB (boundary_library.json),
//      RESULT (json path), PLAYWRIGHT_PATH (optional; else ~/.npm/_npx/*/node_modules/playwright).
// Exit: 0 pass, 1 assertion failed, 3 Playwright not found.
const fs = require('fs');
const os = require('os');
const path = require('path');

function findPlaywright() {
  const cands = [];
  if (process.env.PLAYWRIGHT_PATH) cands.push(process.env.PLAYWRIGHT_PATH);
  const npx = path.join(os.homedir(), '.npm', '_npx');
  if (fs.existsSync(npx)) {
    for (const d of fs.readdirSync(npx)) cands.push(path.join(npx, d, 'node_modules', 'playwright'));
  }
  for (const c of cands) {
    try { return require(c); } catch (e) { /* try next */ }
  }
  return null;
}

const pw = findPlaywright();
if (!pw) {
  console.error('E2E FAIL: Playwright not found. Set PLAYWRIGHT_PATH or run `npx playwright --version` once to cache it in ~/.npm/_npx.');
  process.exit(3);
}

const APP = process.env.APP_URL || 'http://127.0.0.1:5183';
const API = process.env.API_URL || 'http://127.0.0.1:5012';
const OUT = process.env.OUT_DIR || '/tmp/e2e_shots';
const LIB = process.env.LIB;
const RESULT = process.env.RESULT || path.join(OUT, 'e2e_result.json');
fs.mkdirSync(OUT, { recursive: true });

const results = [];
const shots = [];
function check(name, ok, detail) {
  results.push({ name, ok: !!ok, detail: detail === undefined ? '' : String(detail).slice(0, 400) });
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail !== undefined && !ok ? ' :: ' + String(detail).slice(0, 300) : ''}`);
}

async function apiPost(route, body) {
  const r = await fetch(API + route, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return { status: r.status, json: await r.json() };
}

const hallways = (g) => (g.rooms || []).filter((r) => r.type === 'hallway').map((r) => r.id);
const passedOf = (g) => g.sensors && g.sensors.summary ? g.sensors.summary.passed : null;

(async () => {
  const browser = await pw.chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
  const consoleErrors = [];
  const pageErrors = [];
  const httpErrors = [];
  const retrieves = [];
  const gens = [];
  const genReqs = [];
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text().slice(0, 200)); });
  page.on('pageerror', (e) => pageErrors.push(String(e.message).slice(0, 200)));
  page.on('response', async (r) => {
    const u = r.url();
    if (r.status() >= 400) httpErrors.push(`${r.status()} ${u}`);
    try {
      if (u.endsWith('/retrieve') && r.request().method() === 'POST') retrieves.push(await r.json());
      if (u.endsWith('/generate') && r.request().method() === 'POST') gens.push(await r.json());
    } catch (e) { /* body not JSON */ }
  });
  page.on('request', (rq) => {
    if (rq.url().endsWith('/generate') && rq.method() === 'POST') {
      try { genReqs.push(JSON.parse(rq.postData())); } catch (e) { /* ignore */ }
    }
  });
  // Canvas op log: draw calls on #main-canvas since its last clearRect. Lets us compare
  // two renders without comparing pixels (no debug accessor in product code).
  await page.addInitScript(() => {
    window.__ops = [];
    window.__arcs = [];
    const P = CanvasRenderingContext2D.prototype;
    const names = ['moveTo', 'lineTo', 'arc', 'ellipse', 'rect', 'fillRect', 'strokeRect', 'fillText', 'strokeText',
      'bezierCurveTo', 'quadraticCurveTo', 'stroke', 'fill', 'setLineDash'];
    for (const n of names) {
      const o = P[n];
      P[n] = function (...a) {
        if (n === 'arc' && this.canvas && this.canvas.id === 'main-canvas') {
          const m = this.getTransform();
          window.__arcs.push([m.a * a[0] + m.c * a[1] + m.e, m.b * a[0] + m.d * a[1] + m.f, a[2] * Math.abs(m.a), this.canvas.width]);
        }
        if (this.canvas && this.canvas.id === 'main-canvas') {
          window.__ops.push([n, ...a.map((x) => (typeof x === 'number' ? Math.round(x * 100) / 100 : typeof x === 'string' ? x : null))]);
        }
        return o.apply(this, a);
      };
    }
    const oc = P.clearRect;
    P.clearRect = function (...a) {
      if (this.canvas && this.canvas.id === 'main-canvas' && a[0] === 0 && a[1] === 0) { window.__ops = []; window.__arcs = []; }
      return oc.apply(this, a);
    };
  });

  const clk = (id) => page.evaluate((i) => document.getElementById(i).click(), id);
  const waitGen = () => page.waitForResponse((r) => r.url().endsWith('/generate') && r.request().method() === 'POST', { timeout: 60000 });
  const shot = async (name) => {
    const f = path.join(OUT, name);
    await page.locator('#main-canvas').screenshot({ path: f });
    shots.push(f);
  };
  const frame = () => page.evaluate(() => JSON.stringify(window.__ops));
  const genAction = async (action) => { const [resp] = await Promise.all([waitGen(), action()]); await page.waitForTimeout(400); return resp; };

  try {
    await page.goto(APP);
    await page.waitForTimeout(1200);
    await page.setInputFiles('#fi-import-library', LIB);
    await page.waitForTimeout(500);
    const hit = await page.evaluate(() => {
      const els = [...document.querySelectorAll('#lib-list *')].filter((e) => e.children.length === 0 && /005b/.test(e.textContent));
      if (els[0]) { els[0].click(); return els[0].textContent; }
      return null;
    });
    check('library: boundary 005b found and selected', hit !== null, hit);
    await page.waitForTimeout(400);
    await page.click('#run-btn');
    await page.waitForSelector('.result-card', { timeout: 60000 });
    await page.waitForTimeout(1000);
    await page.locator('.result-card').nth(0).click();
    await page.waitForTimeout(600);
    await shot('01_005b_retrieved.png');

    // b. rank 1 carries the node graph; the entrance opens onto one of its nodes.
    // No plan-specific id here: rank 1 changes when retrieval changes (2026-11-05).
    const plans = (retrieves.at(-1) || {}).plans || [];
    const rank1 = plans[0] || {};
    const ret = rank1.retrieval || {};
    check('b: rank 1 has retrieval.node_graph', !!ret.node_graph && Array.isArray(ret.node_graph.nodes), `plans=${plans.length}`);
    const entryNode = (ret.node_graph || {}).entry_node;
    check('b: entry_node is a node of the graph', ((ret.node_graph || {}).nodes || []).some((n) => n.id === entryNode), entryNode);
    const retrievedRoomEdges = JSON.stringify(ret.room_edges || null);
    check('d: /retrieve returned room_edges for rank 1', !!ret.room_edges && Object.keys(ret.room_edges).length > 0);

    const nodesChecked = await page.evaluate(() => document.getElementById('chk-node-generate').checked);
    check('nodes checkbox is on by default', nodesChecked === true, nodesChecked);

    // frame of the retrieved view, before any generate
    const frameBefore = await frame();
    check('canvas op log records the retrieved view', JSON.parse(frameBefore).length > 20, JSON.parse(frameBefore).length);

    // c. generate, nodes on, placement off
    let resp = await genAction(() => clk('btn-generate'));
    check('c(off): /generate status 200', resp.status() === 200, resp.status());
    const gOff = gens.at(-1);
    check('c(off): no error field', gOff && !gOff.error, gOff && gOff.error);
    check('c(off): has hallway rooms', hallways(gOff).length > 0, JSON.stringify(hallways(gOff)));
    check('c(off): boundary_contacts.entrance.met is true',
      ((gOff.report || {}).boundary_contacts || {}).entrance && gOff.report.boundary_contacts.entrance.met === true,
      JSON.stringify(((gOff.report || {}).boundary_contacts || {}).entrance));
    const baseReq = genReqs.at(-1);
    check('c(off): request was placement off, nodes on', baseReq.core_placement === false && baseReq.expand_nodes === true,
      `${baseReq.core_placement} ${baseReq.expand_nodes}`);
    await shot('02_generate_nodes_off.png');
    const frameGenerated = await frame();

    // c. cpsat
    resp = await genAction(() => page.selectOption('#sel-core-placement', 'cpsat'));
    check('c(cpsat): /generate status 200', resp.status() === 200, resp.status());
    const gCp = gens.at(-1);
    check('c(cpsat): no error field', gCp && !gCp.error, gCp && gCp.error);
    check('c(cpsat): request core_placement is cpsat', genReqs.at(-1).core_placement === 'cpsat', genReqs.at(-1).core_placement);
    check('c(cpsat): has hallway rooms', hallways(gCp).length > 0, JSON.stringify(hallways(gCp)));
    check('c(cpsat): boundary_contacts.entrance.met is true',
      ((gCp.report || {}).boundary_contacts || {}).entrance && gCp.report.boundary_contacts.entrance.met === true,
      JSON.stringify(((gCp.report || {}).boundary_contacts || {}).entrance));
    await shot('03_generate_nodes_cpsat.png');

    // d. reset: the retrieved view must come back exactly (room_edges not overwritten by generated contacts)
    await clk('btn-generate-reset');
    await page.waitForTimeout(700);
    const frameAfter = await frame();
    check('d: generated view differs from retrieved view (control)', frameGenerated !== frameBefore);
    check('d: after reset the retrieved view equals the view before generate (room_edges intact)',
      frameAfter === frameBefore,
      `ops before=${JSON.parse(frameBefore).length} after=${JSON.parse(frameAfter).length}`);
    const again = await apiPost('/retrieve', {
      boundary: baseReq.boundary, entrance_point: baseReq.entrance_point, requirements: baseReq.requirements,
      topn: plans.length || 3, typed_edges: baseReq.typed_edges });
    check('d: /retrieve is repeatable (room_edges identical on a second call)',
      JSON.stringify(((again.json.plans || [])[0] || {}).retrieval.room_edges || null) === retrievedRoomEdges);
    await shot('04_after_reset.png');

    // f. generate + swap (placement set back to off)
    await page.selectOption('#sel-core-placement', 'off');
    await page.waitForTimeout(300);
    const arcs = await page.evaluate(() => {
      const r = document.getElementById('main-canvas').getBoundingClientRect();
      const seen = new Map();
      for (const o of window.__arcs) {
        const k = Math.round(o[0]) + ',' + Math.round(o[1]);
        if (o[2] >= 3 && o[2] <= 40) seen.set(k, [r.x + o[0] * r.width / o[3], r.y + o[1] * r.width / o[3]]);
      }
      return [...seen.values()];
    });
    let swapReady = false;
    for (const [ax, ay] of arcs.slice(0, 16)) {
      await page.mouse.click(ax, ay);
      await page.waitForTimeout(150);
      swapReady = await page.evaluate(() => !document.getElementById('btn-generate-swap').disabled);
      if (swapReady) break;
    }
    check('f: two nodes selected, generate+swap enabled', swapReady, `arcs=${arcs.length}`);
    if (swapReady) {
      const before = genReqs.length;
      resp = await genAction(() => clk('btn-generate-swap'));
      const gSw = gens.at(-1);
      check('f: swap request carried a pair', genReqs.length > before && Array.isArray(genReqs.at(-1).swap) && genReqs.at(-1).swap.length === 2,
        JSON.stringify(genReqs.at(-1).swap));
      check('f: generate+swap status 200 and no error', resp.status() === 200 && gSw && !gSw.error, `${resp.status()} ${gSw && gSw.error}`);
      check('f: swap response has rooms and sensors', (gSw.rooms || []).length > 0 && !!gSw.sensors);
      await shot('05_generate_swap.png');
    }
    await clk('btn-generate-reset');
    await page.waitForTimeout(500);

    // e. nodes off: no hallway rooms
    await page.evaluate(() => document.getElementById('chk-node-generate').click());
    await page.waitForTimeout(300);
    const chk = await page.evaluate(() => document.getElementById('chk-node-generate').checked);
    check('e: node checkbox is now off', chk === false, chk);
    await genAction(() => clk('btn-generate'));
    const gRooms = gens.at(-1);
    check('e: request had expand_nodes false', genReqs.at(-1).expand_nodes === false, genReqs.at(-1).expand_nodes);
    check('e: nodes off returns no hallway rooms', hallways(gRooms).length === 0, JSON.stringify(hallways(gRooms)));
    check('e: nodes off returns rooms without error', (gRooms.rooms || []).length > 0 && !gRooms.error, gRooms.error);
    await shot('06_generate_rooms_mode.png');

    // g. every proposed move, applied through /generate, realises its predicted passed count
    const propose = await apiPost('/moves/propose', baseReq);
    const cands = (propose.json || {}).candidates || [];
    check('g: /moves/propose returns candidates on 005b', propose.status === 200 && cands.length > 0, `${propose.status} n=${cands.length}`);
    const mismatches = [];
    for (const c of cands) {
      const e = c.edit || {};
      const extra = e.move === 'SetPrecedence'
        ? { precedence: [{ winner: e.winner, loser: e.loser }] }
        : { connections: [{ u: e.u, v: e.v, type: e.type }] };
      const r = await apiPost('/generate', { ...baseReq, ...extra });
      const pred = c.predicted && c.predicted.summary ? c.predicted.summary.passed : null;
      const act = passedOf(r.json);
      if (r.status !== 200 || r.json.error || pred === null || pred !== act) {
        mismatches.push(`${e.move} ${e.u || e.winner}-${e.v || e.loser} ${e.type || ''} predicted=${pred} actual=${act}`);
      }
    }
    check(`g: all ${cands.length} candidates realise their predicted passed count`, cands.length > 0 && mismatches.length === 0, mismatches.join(' | '));

    // a. whole flow
    check('a: no console errors across the flow', consoleErrors.length === 0, consoleErrors.join(' | '));
    check('a: no page errors across the flow', pageErrors.length === 0, pageErrors.join(' | '));
    check('a: no HTTP status >= 400 seen by the page', httpErrors.length === 0, httpErrors.join(' | '));
  } catch (err) {
    check('flow completed without an exception', false, err && err.stack ? err.stack.split('\n').slice(0, 3).join(' / ') : err);
  } finally {
    await browser.close();
  }
  const failed = results.filter((r) => !r.ok);
  fs.writeFileSync(RESULT, JSON.stringify({ passed: failed.length === 0, checks: results, shots }, null, 1));
  console.log(`E2E ${failed.length === 0 ? 'PASS' : 'FAIL'}: ${results.length - failed.length}/${results.length} checks`);
  process.exit(failed.length === 0 ? 0 : 1);
})();
