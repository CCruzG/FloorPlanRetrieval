import { ABBR, ROOM_GROUPS } from './constants.js';
import { createAdaptedResults, createFitModeState, createResultSelectionState } from './queryState.js';
import { bdryCenter } from './geometry.js';

const FIT_ORDER=[
  new Set([1,5,7]),
  new Set([3]),
  new Set([2]),
  new Set([6,8,9,10,11]),
  new Set([0,4]),
];

function bdryHalfPlanes(boundaryVerts){
  const bn=boundaryVerts.length;
  const cenB=bdryCenter(boundaryVerts);
  const hp=[];
  for(let i=0;i<bn;i++){
    const a=boundaryVerts[i],b=boundaryVerts[(i+1)%bn];
    const dx=b[0]-a[0],dy=b[1]-a[1],len=Math.hypot(dx,dy);
    if(len<1e-10)continue;
    let nx=dy/len,ny=-dx/len;
    if(nx*(cenB[0]-a[0])+ny*(cenB[1]-a[1])<0){nx=-nx;ny=-ny;}
    hp.push({nx,ny,c:nx*a[0]+ny*a[1]});
  }
  return hp;
}

function edgePush(origPoly,halfPlanes,startTx,startTy){
  let tx=startTx,ty=startTy;
  for(let iter=0;iter<40;iter++){
    let worstViol=0,wnx=0,wny=0;
    for(const {nx,ny,c} of halfPlanes){
      for(const v of origPoly){
        const viol=c-(nx*(v[0]+tx)+ny*(v[1]+ty));
        if(viol>worstViol){worstViol=viol;wnx=nx;wny=ny;}
      }
    }
    if(worstViol<1e-8)break;
    tx+=wnx*worstViol;
    ty+=wny*worstViol;
  }
  return [tx,ty];
}

function polyCentroid2(poly){
  return [
    poly.reduce((s,v)=>s+v[0],0)/poly.length,
    poly.reduce((s,v)=>s+v[1],0)/poly.length,
  ];
}

function poly2Dist(a,b){
  const [ax,ay]=polyCentroid2(a),[bx,by]=polyCentroid2(b);
  return Math.hypot(ax-bx,ay-by);
}

export function fitPlanLocal(fp,boundaryVerts){
  const rooms=JSON.parse(JSON.stringify(fp.rooms||[]));
  const stats={moved:0,skipped:0,no_fit:0};
  const origPolys=rooms.map((r)=>r.polygon||[]);
  const placed=new Array(rooms.length).fill(null);
  const hp=bdryHalfPlanes(boundaryVerts);

  for(const typeSet of FIT_ORDER){
    const indices=[];
    for(let i=0;i<rooms.length;i++){
      const rt=rooms[i].source_type??rooms[i].type??-1;
      if(typeSet.has(rt))indices.push(i);
    }

    for(const idx of indices){
      const room=rooms[idx];
      const origPoly=origPolys[idx];
      if(!origPoly||origPoly.length<3){room.fitted=false;room.fit_reason='no_poly';continue;}

      let startTx=0,startTy=0,bestDist=Infinity;
      for(let j=0;j<rooms.length;j++){
        if(!placed[j]||!origPolys[j])continue;
        const d=poly2Dist(origPoly,origPolys[j]);
        if(d<bestDist){bestDist=d;[startTx,startTy]=placed[j];}
      }

      const [tx,ty]=edgePush(origPoly,hp,startTx,startTy);
      if(Math.abs(tx)>1e-8||Math.abs(ty)>1e-8){
        room.polygon=origPoly.map((v)=>[v[0]+tx,v[1]+ty]);
        room.fitted=true;
        room.fit_offset=[tx,ty];
        stats.moved++;
      }else{
        room.fitted=true;
      }
      placed[idx]=[tx,ty];
    }
  }
  return {fp:{...fp,rooms},stats};
}

export async function fetchWithCache(url,{cacheName,fetchImpl=fetch}={}){
  if(!cacheName||!('caches' in window)){
    const resp=await fetchImpl(url);
    if(!resp.ok)throw new Error(`Fetch failed for ${url} (${resp.status})`);
    return resp;
  }

  const store=await caches.open(cacheName);
  const hit=await store.match(url);
  if(hit)return hit;

  const resp=await fetchImpl(url);
  if(!resp.ok)throw new Error(`Fetch failed for ${url} (${resp.status})`);
  store.put(url,resp.clone());
  return resp;
}

export async function clearNamedCache(cacheName){
  if(!cacheName||!('caches' in window))return;
  await caches.delete(cacheName);
}

export async function postJsonWithTimeout(url,payload,{timeoutMs=15000,fetchImpl=fetch}={}){
  const controller=new AbortController();
  const timeoutId=setTimeout(()=>controller.abort(),timeoutMs);
  try{
    const resp=await fetchImpl(url,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload),
      signal:controller.signal,
    });
    if(!resp.ok)throw new Error(`server returned ${resp.status}`);
    const data=await resp.json();
    return {resp,data};
  }finally{
    clearTimeout(timeoutId);
  }
}

export function setStatusUI(msg,cls='',doc=document){
  const el=doc.getElementById('hdr-status');
  if(!el)return;
  el.textContent=msg;
  el.className=cls;
}

export function checkReadyState({closed,boundary},doc=document){
  // Retrieval runs entirely server-side (canonical retrieval-prototype
  // module + rplan_index_v5.npz — 2026-08-10) — no client-side index
  // needed to enable the button.
  const btn=doc.getElementById('run-btn');
  if(!btn)return;
  btn.disabled=!(closed&&boundary.length>=3);
}

export function tryFinalizeIndex({db,onReady,onError,doc=document}){
  if(!db._raw||!db.names)return false;
  const count=db.names.length;
  db.dim=db._raw.length/count;
  if(!Number.isInteger(db.dim)){
    onError?.('index size mismatch');
    return false;
  }
  db.descs=db._raw;
  const idxEl=doc.getElementById('s-idx');
  if(idxEl)idxEl.textContent=count.toLocaleString()+' · dim '+db.dim;
  const stEl=doc.getElementById('idx-st');
  if(stEl){
    stEl.textContent=count.toLocaleString()+' plans indexed';
    stEl.style.color='var(--green)';
  }
  onReady?.();
  return true;
}

export function setupDropZone(dzId,inpId,lblId,handler,doc=document){
  const dz=doc.getElementById(dzId);
  const inp=doc.getElementById(inpId);
  const lbl=doc.getElementById(lblId);
  if(!dz||!inp||!lbl)return;
  dz.addEventListener('dragover',(e)=>{e.preventDefault();dz.classList.add('over');});
  dz.addEventListener('dragleave',()=>dz.classList.remove('over'));
  dz.addEventListener('drop',(e)=>{e.preventDefault();dz.classList.remove('over');handler(e.dataTransfer.files,lbl);});
  inp.addEventListener('change',(e)=>handler(e.target.files,lbl));
}

export async function autoLoadIndex({db,setStatus,tryFinalise,doc=document,fetchImpl=fetch}){
  setStatus('auto-loading…','warn');

  try{
    const idxResp=await fetchImpl('index.bin');
    if(!idxResp.ok)throw new Error();
    db._raw=new Float32Array(await idxResp.arrayBuffer());
    const binLbl=doc.getElementById('lbl-bin');
    if(binLbl)binLbl.textContent='index.bin ✓';
  }catch(_err){
    setStatus('index.bin not found — load manually','err');
    return false;
  }

  try{
    const namesResp=await fetchImpl('index_names.json');
    if(!namesResp.ok)throw new Error();
    db.names=await namesResp.json();
    const nmsLbl=doc.getElementById('lbl-nms');
    if(nmsLbl)nmsLbl.textContent='index_names.json ✓';
  }catch(_err){
    setStatus('index_names.json not found — load manually','err');
    return false;
  }

  tryFinalise();
  const plsLbl=doc.getElementById('lbl-pls');
  if(plsLbl)plsLbl.textContent='plans load on demand ✓';
  return true;
}

export async function runRetrievalFlow({
  closed,
  boundary,
  entrance,
  req,
  params,
  setStatus,
  applyServerPlans,
  afterRender,
  checkReady,
  typedEdges=[],
  doc=document,
}){
  // Retrieval always runs through the backend now (canonical
  // retrieval-prototype module — 2026-08-10). No client-side fallback:
  // the app previously had a divergent local reimplementation that
  // silently kicked in whenever the server was unreachable, which is
  // exactly the drift this decision was meant to eliminate.
  if(!closed||boundary.length<3)return;
  setStatus('retrieving…','warn');
  const runBtn=doc.getElementById('run-btn');
  if(runBtn)runBtn.disabled=true;
  const body={
    boundary:boundary.map(({x,y})=>[x,y,0,0]),
    entrance_point:entrance?[entrance.x,entrance.y]:null,
    requirements:req,
    topn:params.topn,
    shape_w:params.shape_w,
    ent_w:params.ent_w,
    room_w:params.room_w,
    // Step 1 (topology → generate → select): the target's typed edges drive the
    // topology re-rank's perimeter-ring alignment (sparse; backend defaults rest).
    typed_edges:typedEdges,
  };
  try{
    const {data}=await postJsonWithTimeout('/retrieve',body,{timeoutMs:15000});
    if(data.error)throw new Error(data.error);
    applyServerPlans(data.plans);
    setStatus(`top ${data.count} retrieved`,'ok');
    const resultCountEl=doc.getElementById('result-count');
    if(resultCountEl)resultCountEl.textContent=data.count+' results';
    afterRender?.();
  }catch(err){
    console.error('Retrieval failed:',err);
    setStatus(`retrieval failed — is the backend running? (${err.message})`,'err');
  }finally{
    checkReady?.();
    if(runBtn)runBtn.disabled=!(closed&&boundary.length>=3);
  }
}

export function roomSummary(fp){
  const rooms=fp?.rooms||[];
  const cnt={};
  ROOM_GROUPS.forEach((g)=>{cnt[g.name]=0;});
  rooms.forEach((r)=>{
    const st=r.source_type??r.type;
    ROOM_GROUPS.forEach((g)=>{if(g.set.has(st))cnt[g.name]++;});
  });
  return ROOM_GROUPS.filter((g)=>cnt[g.name]>0).map((g)=>`${ABBR[g.name]}:${cnt[g.name]}`).join('  ')||'—';
}

// Topology sensor badge (passed/total), shared by both card renderers below.
// Backend attaches flags to every retrieved plan; absent/malformed → no badge.
function flagsBadgeHtml(flags){
  const s=flags&&flags.summary;
  if(!s||typeof s.passed!=='number'||typeof s.total!=='number')return'';
  const ok=s.passed===s.total;
  const col=ok?'#16a34a':(s.passed>0?'#d97706':'#dc2626');
  return `<div class="card-dist" style="color:${col};font-weight:600">${ok?'✓':'✗'} ${s.passed}/${s.total} checks</div>`;
}

// Card-thumbnail view mode, shared across both result-list renderers below so
// a user's Graph/Plan choice survives a full re-render (new retrieval, boundary
// edit, etc). Defaults to Graph so topology is what's shown first.
let cardView='graph';

// Paints one result card's thumbnail canvas in the current cardView mode.
// 'graph' = new topology view (drawTopologyGraph); 'plan' = existing geometry
// render (drawFP + drawDoorsWindows + drawGraph, unchanged from before this
// toggle existed).
function drawCardThumb(tctx,r,{drawFP,drawDoorsWindows,drawGraph,drawTopologyGraph},showClipped=true){
  tctx.fillStyle='#f1f5f9';
  tctx.fillRect(0,0,200,200);
  if(!r.fp){
    tctx.fillStyle='#94a3b8';
    tctx.font='10px monospace';
    tctx.textAlign='center';
    tctx.fillText('no plan data',100,100);
    return;
  }
  if(cardView==='graph'&&drawTopologyGraph){
    drawTopologyGraph(tctx,r.fp,r.retrieval,200,200,r.rot||0,[0,0],r.mirror||false);
  }else{
    drawFP(tctx,r.fp,200,200,false,showClipped,r.rot||0,[0,0],r.mirror||false);
    drawDoorsWindows(tctx,r.fp,200,200,r.rot||0,[0,0],r.mirror||false);
    drawGraph(tctx,r.fp,200,200,r.rot||0,[0,0],r.mirror||false);
  }
}

// Toggle button shared by both card-list renderers: flips cardView and
// repaints every thumbnail canvas already on the page in place — it does NOT
// rebuild cards, so selection state / fit cache in the caller is untouched.
function appendCardViewToggle(listEl,cardEntries,drawFns){
  const btn=document.createElement('button');
  btn.className='clear-btn';
  btn.style.cssText='margin-bottom:6px;font-size:10px;padding:3px 8px;flex-shrink:0';
  const label=()=>btn.textContent=cardView==='graph'?'◈ Graph view (show Plan)':'▦ Plan view (show Graph)';
  label();
  btn.onclick=(e)=>{
    e.stopPropagation();
    cardView=cardView==='graph'?'plan':'graph';
    label();
    cardEntries.forEach(({r,tctx,showClipped})=>drawCardThumb(tctx,r,drawFns,showClipped));
  };
  listEl.appendChild(btn);
}

export function renderAdaptedResultsView({plans,listEl,selectResult,drawFP,drawDoorsWindows,drawGraph,drawTopologyGraph,sync3DSceneIfNeeded}){
  const results=createAdaptedResults(plans);
  const fittedCache={};
  const selected=null;
  const displacementArrows=[];
  const drawFns={drawFP,drawDoorsWindows,drawGraph,drawTopologyGraph};

  listEl.innerHTML='';
  if(!results.length){
    listEl.innerHTML='<div id="results-empty">no adapted results</div>';
    return {results,fittedCache,selected,displacementArrows};
  }

  const cardEntries=[];
  appendCardViewToggle(listEl,cardEntries,drawFns);

  results.forEach((r,i)=>{
    const card=document.createElement('div');
    card.className='result-card';
    card.onclick=()=>selectResult(i);

    const tc=document.createElement('canvas');
    tc.className='card-thumb';
    tc.width=200;
    tc.height=200;
    const tctx=tc.getContext('2d');
    cardEntries.push({r,tctx,showClipped:true});
    drawCardThumb(tctx,r,drawFns,true);
    card.appendChild(tc);

    const ret=r.retrieval||{};
    const dist=(r.dist??0).toFixed(4);
    // ret.rot is radians (geo_ent_num_retrieval.retrieve()'s best_angles) —
    // the old dead `ret.orientation` field this replaced was never real, so
    // this unit mismatch had no visible symptom until now.
    const rot=((ret.rot??0)*180/Math.PI).toFixed(0);
    const mirror=ret.mirror?' ↔':'';
    // The paper's three weighted component distances (dist = their sum) —
    // shape (Option B rotation-aware Fourier), entrance (door position +
    // wall normal + radial), room-count. null on the legacy flat-.bin
    // fallback, which doesn't decompose distance into components.
    const fmt=(v)=>v==null?'—':v.toFixed(3);
    const meta=document.createElement('div');
    meta.className='card-meta';
    const srcId=ret.source_id??r.name;
    // Topology re-rank (Step 1): the list order IS this score. Show topo +
    // daylight-fit inline; per-room daylight detail on hover so the card stays
    // compact. Bold-green when the boundary is fully satisfied.
    const topo=ret.topo;
    let topoHtml='';
    if(topo){
      const dayfit=topo.components?.daylight_fit;
      const per=topo.daylight_detail?.per_room||{};
      const perTip=Object.entries(per).map(([k,v])=>`${k}: ${v}`).join('\n')
        +(topo.mirror?'\n(mirror)':'');
      const good=(dayfit??0)>=0.999;
      topoHtml=`<div class="card-dist" title="${perTip}" style="color:${good?'#16a34a':'#0f172a'};font-weight:600">`
        +`topo: ${fmt(topo.score)}&nbsp;&nbsp;dayfit: ${fmt(dayfit)}${topo.mirror?' ↔':''}</div>`;
    }
    meta.innerHTML=`<div class="card-title">#${i+1} ${srcId}</div>`+
      flagsBadgeHtml(ret.flags)+
      topoHtml+
      `<div class="card-dist">dist: ${dist}&nbsp;&nbsp;rot: ${rot}°${mirror}</div>`+
      `<div class="card-dist">shp: ${fmt(ret.shape_dist)}&nbsp;&nbsp;ent: ${fmt(ret.ent_dist)}&nbsp;&nbsp;rm: ${fmt(ret.room_dist)}</div>`+
      `<div class="card-rooms">${r.fp?roomSummary(r.fp):'—'}</div>`;
    card.appendChild(meta);
    listEl.appendChild(card);
  });

  sync3DSceneIfNeeded(true);
  return {results,fittedCache,selected,displacementArrows};
}

export function renderResultsListView({results,selected,listEl,selectResult,drawFP,drawDoorsWindows,drawGraph,drawTopologyGraph}){
  const drawFns={drawFP,drawDoorsWindows,drawGraph,drawTopologyGraph};
  listEl.innerHTML='';
  if(!results.length){
    listEl.innerHTML='<div id="results-empty">run retrieval to see results</div>';
    return;
  }

  const cardEntries=[];
  appendCardViewToggle(listEl,cardEntries,drawFns);

  results.forEach((r,i)=>{
    const card=document.createElement('div');
    card.className='result-card'+(i===selected?' selected':'');
    card.onclick=()=>selectResult(i);

    const tc=document.createElement('canvas');
    tc.className='card-thumb';
    tc.width=200;
    tc.height=200;
    const tctx=tc.getContext('2d');
    cardEntries.push({r,tctx,showClipped:false});
    drawCardThumb(tctx,r,drawFns,false);
    card.appendChild(tc);

    const meta=document.createElement('div');
    meta.className='card-meta';
    meta.innerHTML=`<div class="card-title">#${i+1} ${r.name}</div>`+
      flagsBadgeHtml(r.retrieval&&r.retrieval.flags)+
      `<div class="card-dist">dist: ${r.dist.toFixed(4)}</div><div class="card-rooms">${r.fp?roomSummary(r.fp):'—'}</div>`;
    card.appendChild(meta);
    listEl.appendChild(card);
  });
}

export function selectResultState({currentSelected,nextIndex,fitMode,fittedCache,fetchFitted,highlightCards,updateCursor,render,sync3DSceneIfNeeded}){
  const next=createResultSelectionState(currentSelected,nextIndex);
  highlightCards(next.selected);
  if(fitMode&&next.selected!==null&&!fittedCache[next.selected])fetchFitted(next.selected);
  updateCursor();
  render();
  sync3DSceneIfNeeded(true);
  return next;
}

export function toggleFitModeState({on,selected,fetchFitted,render,sync3DSceneIfNeeded}){
  const next=createFitModeState(on);
  if(!next.fittedCache&&selected!==null)fetchFitted(selected);
  render();
  sync3DSceneIfNeeded(true);
  return next;
}

export async function fetchFittedState({i,results,fittedCache,boundaryInPlanSpace,setStatus,fitPlanLocal,render,fetchImpl=fetch}){
  const r=results[i];
  if(!r||!r.fp)return fittedCache;
  if(fittedCache[i])return fittedCache;

  const bdryInPlan=boundaryInPlanSpace(r);
  setStatus('fitting…','warn');

  const nextCache={...fittedCache};
  try{
    const resp=await fetchImpl('/fit',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({fp:r.fp,boundary:bdryInPlan}),
    });
    if(!resp.ok)throw new Error('server '+resp.status);
    const data=await resp.json();
    if(data.error)throw new Error(data.error);
    nextCache[i]={fp:data.fp,stats:data.stats,method:'server'};
    const s=data.stats||{};
    setStatus(`fit (server): ${s.moved||0} moved, ${s.skipped||0} irregular, ${s.no_fit||0} too large`,'ok');
  }catch(e){
    console.info('Server fit unavailable, using local fitting:',e.message);
    const {fp:fittedFp,stats}=fitPlanLocal(r.fp,bdryInPlan);
    nextCache[i]={fp:fittedFp,stats,method:'local'};
    const s=stats;
    setStatus(`fit (local): ${s.moved||0} moved, ${s.skipped||0} irregular, ${s.no_fit||0} too large`,'ok');
  }
  render();
  return nextCache;
}