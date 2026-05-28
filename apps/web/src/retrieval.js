import { ABBR, ENT_DIM, ROOM_DIM, ROOM_GROUPS, SHAPE_DIM, TOTAL_DIM } from './constants.js';
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

export function checkReadyState({db,closed,boundary},doc=document){
  const btn=doc.getElementById('run-btn');
  if(!btn)return;
  btn.disabled=!(db.descs&&db.names&&closed&&boundary.length>=3);
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
  db,
  setStatus,
  runRetrievalLocal,
  applyServerPlans,
  afterRender,
  checkReady,
  doc=document,
}){
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
    console.warn('Server retrieval failed, falling back to local:',err);
    setStatus('server unavailable — using local retrieval','warn');
    await runRetrievalLocal();
  }finally{
    checkReady?.();
    if(runBtn)runBtn.disabled=!(db.descs&&db.names&&closed&&boundary.length>=3);
  }
}

export async function runLocalRetrievalData({
  db,
  boundary,
  entrance,
  req,
  params,
  cache,
  fetchWithCacheFn,
  shapeDescFn,
  entranceDescFn,
  roomCountDescFn,
  combinedDescFn,
  l2distFn,
}){
  const rawVerts=boundary.map(({x,y})=>[x,y]);
  let sigArea=0;
  for(let i=0;i<rawVerts.length;i++){
    const [x0,y0]=rawVerts[i],[x1,y1]=rawVerts[(i+1)%rawVerts.length];
    sigArea+=x0*y1-x1*y0;
  }
  const bdryArr=sigArea>0?[...rawVerts].reverse():rawVerts;
  const N=db.names.length;

  if(db.dim===TOTAL_DIM){
    const qShape=shapeDescFn(bdryArr,1);
    const qEnt=entrance?entranceDescFn(boundary,entrance):null;
    const qRoom=roomCountDescFn(req);

    const dists=new Float32Array(N);
    const bestAngles=new Float32Array(N);
    const bestMirror=new Uint8Array(N);
    for(let i=0;i<N;i++){
      const off=i*TOTAL_DIM;
      let sd=0;
      for(let j=0;j<SHAPE_DIM;j++){
        const d=qShape[j]-db.descs[off+j];
        sd+=d*d;
      }
      sd=Math.sqrt(sd)*params.shape_w;

      let ed=0;
      if(qEnt&&params.ent_w>0){
        const eo=off+SHAPE_DIM;
        const d_rx=db.descs[eo],d_ry=db.descs[eo+1];
        const d_cx=db.descs[eo+2],d_cy=db.descs[eo+3];
        const qSq=qEnt[0]**2+qEnt[1]**2+qEnt[2]**2+qEnt[3]**2;
        const dSq=d_rx**2+d_ry**2+d_cx**2+d_cy**2;
        const A=qEnt[0]*d_rx+qEnt[1]*d_ry+qEnt[2]*d_cx+qEnt[3]*d_cy;
        const B=qEnt[1]*d_rx-qEnt[0]*d_ry+qEnt[3]*d_cx-qEnt[2]*d_cy;
        const ed_n=Math.sqrt(Math.max(0,qSq+dSq-2*Math.sqrt(A*A+B*B)));
        const Am=-qEnt[0]*d_rx+qEnt[1]*d_ry-qEnt[2]*d_cx+qEnt[3]*d_cy;
        const Bm=-qEnt[1]*d_rx-qEnt[0]*d_ry-qEnt[3]*d_cx-qEnt[2]*d_cy;
        const ed_m=Math.sqrt(Math.max(0,qSq+dSq-2*Math.sqrt(Am*Am+Bm*Bm)));
        if(ed_m<ed_n){
          ed=ed_m*params.ent_w;
          bestMirror[i]=1;
          bestAngles[i]=Math.round(Math.atan2(Bm,Am)/(Math.PI/2))*(Math.PI/2);
        }else{
          ed=ed_n*params.ent_w;
          bestMirror[i]=0;
          bestAngles[i]=Math.round(Math.atan2(B,A)/(Math.PI/2))*(Math.PI/2);
        }
      }

      let rd=0;
      for(let j=0;j<ROOM_DIM;j++){
        const d=qRoom[j]-db.descs[off+SHAPE_DIM+ENT_DIM+j];
        rd+=d*d;
      }
      rd=Math.sqrt(rd)*params.room_w;
      dists[i]=sd+ed+rd;
    }

    const idx=Array.from({length:N},(_,i)=>i).sort((a,b)=>dists[a]-dists[b]).slice(0,params.topn);
    const planData=await Promise.all(idx.map(async (i)=>{
      const name=db.names[i];
      const stem=name.replace(/\.(png|json)$/i,'');
      if(cache[stem])return cache[stem];
      if(cache[name])return cache[name];
      try{
        const pr=await fetchWithCacheFn('test_data/'+name);
        const data=await pr.json();
        cache[stem]=data;
        if(data.id)cache[data.id]=data;
        return data;
      }catch(_err){
        return null;
      }
    }));

    return idx.map((dbIdx,rank)=>({
      name:db.names[dbIdx],
      dist:dists[dbIdx],
      fp:planData[rank],
      adapted:false,
      rot:bestAngles[dbIdx],
      mirror:!!bestMirror[dbIdx],
    }));
  }

  const qfp={id:'__q__',boundary:boundary.map(({x,y})=>[x,y,0,0]),rooms:[],edges:[]};
  const qd=combinedDescFn(qfp,params.shape_w,params.room_w);
  const dim=db.dim;
  const dists=new Float32Array(N);
  for(let i=0;i<N;i++)dists[i]=l2distFn(qd,db.descs,i*dim,dim);

  const idx=Array.from({length:N},(_,i)=>i).sort((a,b)=>dists[a]-dists[b]).slice(0,params.topn);
  const planData=await Promise.all(idx.map(async (i)=>{
    const name=db.names[i];
    const stem=name.replace(/\.(png|json)$/i,'');
    if(cache[stem])return cache[stem];
    try{
      const pr=await fetchWithCacheFn('test_data/'+name);
      const data=await pr.json();
      cache[stem]=data;
      if(data.id)cache[data.id]=data;
      return data;
    }catch(_err){
      return null;
    }
  }));

  return idx.map((dbIdx,rank)=>({
    name:db.names[dbIdx],
    dist:dists[dbIdx],
    fp:planData[rank],
    adapted:false,
    rot:0,
  }));
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

export function renderAdaptedResultsView({plans,listEl,selectResult,drawFP,drawDoorsWindows,drawGraph,sync3DSceneIfNeeded}){
  const results=createAdaptedResults(plans);
  const fittedCache={};
  const selected=null;
  const displacementArrows=[];

  listEl.innerHTML='';
  if(!results.length){
    listEl.innerHTML='<div id="results-empty">no adapted results</div>';
    return {results,fittedCache,selected,displacementArrows};
  }

  results.forEach((r,i)=>{
    const card=document.createElement('div');
    card.className='result-card';
    card.onclick=()=>selectResult(i);

    const tc=document.createElement('canvas');
    tc.className='card-thumb';
    tc.width=200;
    tc.height=200;
    const tctx=tc.getContext('2d');
    tctx.fillStyle='#f1f5f9';
    tctx.fillRect(0,0,200,200);
    if(r.fp)drawFP(tctx,r.fp,200,200,false,true,r.rot||0,[0,0],r.mirror||false);
    if(r.fp)drawDoorsWindows(tctx,r.fp,200,200,r.rot||0,[0,0],r.mirror||false);
    if(r.fp)drawGraph(tctx,r.fp,200,200,r.rot||0,[0,0],r.mirror||false);
    else{
      tctx.fillStyle='#94a3b8';
      tctx.font='10px monospace';
      tctx.textAlign='center';
      tctx.fillText('no plan data',100,100);
    }
    card.appendChild(tc);

    const ret=r.retrieval||{};
    const iou=(ret.iou_score??0).toFixed(3);
    const rot=ret.orientation??0;
    const scale=(ret.scale_factor??1).toFixed(2);
    const dist=(r.dist??0).toFixed(4);
    const meta=document.createElement('div');
    meta.className='card-meta';
    const srcId=ret.source_id??r.name;
    meta.innerHTML=`<div class="card-title">#${i+1} ${srcId}</div>`+
      `<div class="card-dist">dist: ${dist}&nbsp;&nbsp;iou: ${iou}&nbsp;&nbsp;rot: ${rot}°&nbsp;&nbsp;scale: ${scale}</div>`+
      `<div class="card-rooms">${r.fp?roomSummary(r.fp):'—'}</div>`;
    card.appendChild(meta);
    listEl.appendChild(card);
  });

  sync3DSceneIfNeeded(true);
  return {results,fittedCache,selected,displacementArrows};
}

export function renderResultsListView({results,selected,listEl,selectResult,drawFP,drawDoorsWindows,drawGraph}){
  listEl.innerHTML='';
  if(!results.length){
    listEl.innerHTML='<div id="results-empty">run retrieval to see results</div>';
    return;
  }
  results.forEach((r,i)=>{
    const card=document.createElement('div');
    card.className='result-card'+(i===selected?' selected':'');
    card.onclick=()=>selectResult(i);

    const tc=document.createElement('canvas');
    tc.className='card-thumb';
    tc.width=200;
    tc.height=200;
    const tctx=tc.getContext('2d');
    tctx.fillStyle='#f1f5f9';
    tctx.fillRect(0,0,200,200);
    if(r.fp)drawFP(tctx,r.fp,200,200,false,false,r.rot||0,[0,0],r.mirror||false);
    if(r.fp)drawDoorsWindows(tctx,r.fp,200,200,r.rot||0,[0,0],r.mirror||false);
    if(r.fp)drawGraph(tctx,r.fp,200,200,r.rot||0,[0,0],r.mirror||false);
    else{
      tctx.fillStyle='#94a3b8';
      tctx.font='10px monospace';
      tctx.textAlign='center';
      tctx.fillText('no plan data',100,100);
    }
    card.appendChild(tc);

    const meta=document.createElement('div');
    meta.className='card-meta';
    meta.innerHTML=`<div class="card-title">#${i+1} ${r.name}</div><div class="card-dist">dist: ${r.dist.toFixed(4)}</div><div class="card-rooms">${r.fp?roomSummary(r.fp):'—'}</div>`;
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