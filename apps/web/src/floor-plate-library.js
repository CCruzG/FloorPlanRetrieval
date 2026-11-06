// Floor plate library (2026-11-06) — sidebar section in the Floor plate tab.
// Saves the drawn outline + its context (entrance, edge types, north bearing,
// street frontage, notes) to the repo file through /floor_plates (see
// packages/floor_plate_design/library.py). Canvas units are normalised; the file is
// in metres (canvas frame, y down), converted with METRES_PER_UNIT.
import { METRES_PER_UNIT } from './constants.js';
import { createLoadedBoundaryState } from './queryState.js';

const BTN='flex-shrink:0;padding:1px 4px;font-size:9px;font-family:var(--mono);background:transparent;border:1px solid var(--border2);border-radius:3px;color:var(--muted);cursor:pointer;line-height:1.5';
const BTN_ON='flex-shrink:0;padding:1px 5px;font-size:9px;font-family:var(--mono);background:var(--accent);border:none;border-radius:3px;color:#fff;cursor:pointer;line-height:1.5';
const BTN_RED='flex-shrink:0;padding:1px 5px;font-size:9px;font-family:var(--mono);background:var(--red);border:none;border-radius:3px;color:#fff;cursor:pointer;line-height:1.5';
const INP='min-width:0;font-size:10px;font-family:var(--mono);background:var(--bg2);border:1px solid var(--border2);border-radius:3px;color:var(--text);padding:2px 4px';

export function createFloorPlateLibrary({
  doc,
  getState,        // () => {boundary, closed, entrance, edgeTypes, mode, cW, cH}
  applyState,      // applyBoundaryEditorState
  getViewport,     // () => viewport object (mutated in place)
  n2c,             // normalised -> logical canvas px (no viewport)
  setMode,
  render,
  updateEntranceInfo,
  checkReady,
  loadAssembled,   // (assemblerResult) => void: puts the result in floorPlateState
  clearAssembled,  // () => void: drops an assembled result from floorPlateState
}) {
  const $=(id)=>doc.getElementById(id);
  let plates=[];
  let loadedId=null;
  let streetEdges=[];
  let visible=false;
  let fetched=false;

  // ── helpers ───────────────────────────────────────────────────────────────
  const northDeg=()=>{
    const v=parseFloat($('fpl-north')?.value);
    return Number.isFinite(v)?v:0;
  };

  function setStatus(msg,isErr=false){
    const el=$('fpl-status');
    if(!el)return;
    el.textContent=msg||'';
    el.style.color=isErr?'var(--red)':'var(--muted)';
  }

  async function api(method,url,body){
    const resp=await fetch(url,{
      method,
      headers:body?{'Content-Type':'application/json'}:undefined,
      body:body?JSON.stringify(body):undefined,
    });
    let json=null;
    try{json=await resp.json();}catch{/* non-JSON body */}
    if(!resp.ok)throw new Error((json&&json.error)||`HTTP ${resp.status}`);
    return json;
  }

  function currentPayload(){
    const { boundary, closed, entrance, edgeTypes }=getState();
    if(!closed||boundary.length<3)throw new Error('close the boundary first');
    const n=boundary.length;
    const anyType=Array.isArray(edgeTypes)&&edgeTypes.slice(0,n).some(Boolean);
    const k=METRES_PER_UNIT;
    return {
      name:($('fpl-name').value||'').trim()||`plate ${plates.length+1}`,
      boundary_m:boundary.map(({x,y})=>[x*k,y*k]),
      main_entrance_m:entrance?[entrance.x*k,entrance.y*k]:null,
      edge_types:anyType?Array.from({length:n},(_,i)=>edgeTypes[i]||null):[],
      north_deg:northDeg(),
      street_edges:streetEdges.filter((e)=>e<n),
      notes:$('fpl-notes').value||'',
    };
  }

  // ── plate size readout, street info, button state ─────────────────────────
  function sync(){
    if(!visible)return;
    const { boundary }=getState();
    let txt='—';
    if(boundary.length>=2){
      const xs=boundary.map((p)=>p.x*METRES_PER_UNIT),ys=boundary.map((p)=>p.y*METRES_PER_UNIT);
      const w=Math.max(...xs)-Math.min(...xs),h=Math.max(...ys)-Math.min(...ys);
      txt=`${w.toFixed(1)} × ${h.toFixed(1)} m`;
    }
    const size=$('fpl-size');
    if(size&&size.textContent!==txt)size.textContent=txt;
    const street=$('fpl-street-info');
    const edges=overlay().streetEdges;
    const stxt=edges.length?`street frontage: edge${edges.length>1?'s':''} ${edges.join(', ')}`:'street frontage: none';
    if(street&&street.textContent!==stxt)street.textContent=stxt;
    const upd=$('fpl-btn-update');
    if(upd){
      upd.disabled=loadedId===null;
      upd.style.opacity=loadedId===null?'0.4':'1';
      upd.title=loadedId===null?'load a plate first':`replace "${loadedId}"`;
    }
  }

  // What the canvas draws in the Floor plate tab (null elsewhere).
  function overlay(){
    const { boundary, closed, mode }=getState();
    const edges=closed?streetEdges.filter((e)=>e<boundary.length):[];
    return {northDeg:northDeg(),streetEdges:edges,streetMode:mode==='streetEdge'};
  }

  // ── street frontage ─────────────────────────────────────────────────────────────
  function toggleStreetMode(){
    const { closed, mode }=getState();
    if(!closed)return;
    setMode(mode==='streetEdge'?'select':'streetEdge');
    $('canvas-hint').textContent=getState().mode==='streetEdge'
      ?'click boundary edges to mark street frontage · click again to clear'
      :'boundary closed · press retrieve';
    syncModeButton();
  }

  function syncModeButton(){
    $('fpl-btn-street')?.classList.toggle('active',getState().mode==='streetEdge');
  }

  function clickStreetEdge(edge){
    streetEdges=streetEdges.includes(edge)
      ?streetEdges.filter((e)=>e!==edge)
      :[...streetEdges,edge].sort((a,b)=>a-b);
    sync();render();
  }

  // ── viewport fit ──────────────────────────────────────────────────────────
  function fitViewport(){
    const { boundary, cW, cH }=getState();
    if(boundary.length<2||!cW||!cH)return;
    const pts=boundary.map(({x,y})=>n2c(x,y));
    const xs=pts.map((p)=>p[0]),ys=pts.map((p)=>p[1]);
    const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
    const bw=Math.max(maxX-minX,1),bh=Math.max(maxY-minY,1);
    const zoom=Math.max(0.15,Math.min(20,Math.min(cW*0.7/bw,cH*0.7/bh)));
    const vp=getViewport();
    vp.zoom=zoom;
    // logical->screen: (l-c/2)*zoom+c/2+pan ; put the bbox centre on the canvas centre
    vp.panX=-((minX+maxX)/2-cW/2)*zoom;
    vp.panY=-((minY+maxY)/2-cH/2)*zoom;
  }

  // ── load / save ───────────────────────────────────────────────────────────
  function loadPlate(p){
    const k=METRES_PER_UNIT;
    applyState(createLoadedBoundaryState({
      vertices:p.boundary_m.map(([x,y])=>({x:x/k,y:y/k})),
      entrance:p.main_entrance_m?{x:p.main_entrance_m[0]/k,y:p.main_entrance_m[1]/k}:null,
      edgeTypes:(p.edge_types||[]).map((t)=>t||undefined),
    }));
    clearAssembled?.();
    showAssembleResult('');
    loadedId=p.id;
    streetEdges=Array.isArray(p.street_edges)?p.street_edges.slice():[];
    $('fpl-name').value=p.name;
    $('fpl-notes').value=p.notes||'';
    $('fpl-north').value=String(p.north_deg??0);
    // same follow-up as the unit boundary library's load
    $('entrance-controls').style.display='block';
    $('btn-entrance-mode').classList.remove('active');
    if(getState().entrance){updateEntranceInfo();}
    else{
      $('btn-clear-entrance').style.display='none';
      $('entrance-info').style.display='none';
      $('s-ent').textContent='—';
    }
    $('canvas-hint').textContent='floor plate loaded';
    $('s-bv').textContent=getState().boundary.length;
    $('result-count').textContent='';
    syncModeButton();
    fitViewport();
    checkReady();
    renderList();
    sync();
    render();
    setStatus(`loaded "${p.name}"`);
  }

  async function saveNew(){
    try{
      const made=await api('POST','/floor_plates',currentPayload());
      loadedId=made.id;
      await refresh();
      setStatus(`saved "${made.name}" as ${made.id}`);
    }catch(err){setStatus(err.message,true);}
    sync();
  }

  async function updateLoaded(){
    if(loadedId===null)return;
    try{
      const made=await api('PUT',`/floor_plates/${encodeURIComponent(loadedId)}`,currentPayload());
      await refresh();
      setStatus(`updated "${made.name}" (${made.id})`);
    }catch(err){setStatus(err.message,true);}
    sync();
  }

  async function refresh(){
    const data=await api('GET','/floor_plates');
    plates=data.plates||[];
    fetched=true;
    renderList();
  }

  async function renamePlate(p,name){
    const { id, area_m2, saved, ...fields }=p;
    try{
      await api('PUT',`/floor_plates/${encodeURIComponent(id)}`,{...fields,name});
      await refresh();
      if(loadedId===id)$('fpl-name').value=name;
      setStatus(`renamed to "${name}"`);
    }catch(err){setStatus(err.message,true);renderList();}
  }

  async function deletePlate(p){
    try{
      await api('DELETE',`/floor_plates/${encodeURIComponent(p.id)}`);
      if(loadedId===p.id)loadedId=null;
      await refresh();
      setStatus(`deleted "${p.name}"`);
    }catch(err){setStatus(err.message,true);renderList();}
    sync();
  }

  // ── list (inline rename / delete confirmation, no dialogs) ───────────────
  function mkBtn(label,title,cb,cls){
    const b=doc.createElement('button');
    b.textContent=label;b.title=title;b.style.cssText=BTN;
    if(cls)b.className=cls;
    b.onmouseover=()=>{b.style.borderColor='var(--accent)';b.style.color='var(--accent)';};
    b.onmouseout=()=>{b.style.borderColor='var(--border2)';b.style.color='var(--muted)';};
    b.onclick=cb;
    return b;
  }

  function renderList(){
    const el=$('fpl-list');
    if(!el)return;
    el.innerHTML='';
    if(!plates.length){
      el.innerHTML='<div style="font-size:10px;color:var(--muted)">no saved floor plates</div>';
      return;
    }
    plates.forEach((p)=>{
      const row=doc.createElement('div');
      row.className='fpl-row';
      row.dataset.id=p.id;
      row.style.cssText='display:flex;flex-direction:column;gap:1px';
      const top=doc.createElement('div');
      top.style.cssText='display:flex;align-items:center;gap:3px';
      const lbl=doc.createElement('span');
      lbl.className='fpl-row-name';
      lbl.style.cssText='flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:10px;color:'+(p.id===loadedId?'var(--accent)':'var(--text)');
      lbl.textContent=p.name;
      lbl.title=p.id+' · saved '+(p.saved||'').slice(0,16).replace('T',' ');
      top.appendChild(lbl);
      top.appendChild(mkBtn('load','load this plate',()=>loadPlate(p),'fpl-load'));
      top.appendChild(mkBtn('✎','rename',()=>startRename(row,p),'fpl-rename'));
      top.appendChild(mkBtn('✕','delete',()=>startDelete(row,p),'fpl-delete'));
      const meta=doc.createElement('div');
      meta.className='fpl-row-meta';
      meta.style.cssText='font-size:9px;color:var(--muted)';
      meta.textContent=`${Math.round(p.area_m2)} m² · ${p.boundary_m.length} vertices`;
      row.appendChild(top);
      row.appendChild(meta);
      el.appendChild(row);
    });
  }

  function startRename(row,p){
    row.innerHTML='';
    const line=doc.createElement('div');
    line.style.cssText='display:flex;align-items:center;gap:3px';
    const inp=doc.createElement('input');
    inp.type='text';inp.value=p.name;inp.className='fpl-rename-input';
    inp.style.cssText=INP+';flex:1';
    const ok=doc.createElement('button');ok.textContent='✓';ok.style.cssText=BTN_ON;ok.className='fpl-rename-ok';
    const no=doc.createElement('button');no.textContent='✕';no.style.cssText=BTN;
    const commit=()=>{const n=inp.value.trim();if(n&&n!==p.name)renamePlate(p,n);else renderList();};
    ok.onclick=commit;no.onclick=()=>renderList();
    inp.onkeydown=(e)=>{if(e.key==='Enter')commit();if(e.key==='Escape')renderList();};
    line.appendChild(inp);line.appendChild(ok);line.appendChild(no);
    row.appendChild(line);
    inp.focus();inp.select();
  }

  function startDelete(row,p){
    row.innerHTML='';
    const line=doc.createElement('div');
    line.style.cssText='display:flex;align-items:center;gap:3px';
    const msg=doc.createElement('span');
    msg.style.cssText='flex:1;font-size:9px;color:var(--red);white-space:nowrap;overflow:hidden;text-overflow:ellipsis';
    msg.textContent=`Delete "${p.name}"?`;
    const yes=doc.createElement('button');yes.textContent='Yes';yes.style.cssText=BTN_RED;yes.className='fpl-delete-yes';
    const no=doc.createElement('button');no.textContent='No';no.style.cssText=BTN;
    yes.onclick=()=>deletePlate(p);no.onclick=()=>renderList();
    line.appendChild(msg);line.appendChild(yes);line.appendChild(no);
    row.appendChild(line);
  }

  // ── plate assembler (POST /assemble_floor_plate) ──────────────────────────
  const MIX_TYPES=['1B1b','2B1b','2B2b','3B2b'];
  const MIX_DEFAULT={'1B1b':1,'2B1b':1,'2B2b':0,'3B2b':0};

  function buildAssembleBlock(){
    const sec=$('fpl-section');
    if(!sec||$('fpl-assemble'))return;
    const box=doc.createElement('div');
    box.id='fpl-assemble';
    box.style.cssText='margin-top:6px;padding-top:4px;border-top:1px solid var(--border2)';
    const title=doc.createElement('div');
    title.style.cssText='font-size:10px;color:var(--muted);margin-bottom:3px';
    title.textContent='assemble · mix weights';
    box.appendChild(title);
    const row=doc.createElement('div');
    row.style.cssText='display:grid;grid-template-columns:repeat(4,1fr);gap:3px';
    MIX_TYPES.forEach((t)=>{
      const cell=doc.createElement('label');
      cell.style.cssText='display:flex;flex-direction:column;font-size:9px;color:var(--muted);gap:1px';
      cell.textContent=t;
      const inp=doc.createElement('input');
      inp.type='number';inp.min='0';inp.step='1';inp.value=String(MIX_DEFAULT[t]);
      inp.id=`fpl-mix-${t}`;inp.style.cssText=INP+';width:100%;box-sizing:border-box';
      cell.appendChild(inp);row.appendChild(cell);
    });
    box.appendChild(row);
    const btn=doc.createElement('button');
    btn.id='fpl-btn-assemble';btn.className='clear-btn';btn.style.marginTop='4px';
    btn.textContent='assemble plate';
    btn.addEventListener('click',assemble);
    box.appendChild(btn);
    const out=doc.createElement('div');
    out.id='fpl-assemble-result';
    out.style.cssText='font-size:9px;color:var(--muted);margin-top:3px;line-height:1.4;white-space:pre-wrap';
    box.appendChild(out);
    const list=$('fpl-list');
    list?sec.insertBefore(box,$('fpl-status')||list):sec.appendChild(box);
  }

  function showAssembleResult(txt,isErr=false){
    const el=$('fpl-assemble-result');
    if(!el)return;
    el.textContent=txt;
    el.style.color=isErr?'var(--red)':'var(--muted)';
  }

  async function assemble(){
    const btn=$('fpl-btn-assemble');
    try{
      const { boundary, closed }=getState();
      if(!closed||boundary.length<3)throw new Error('close the boundary first');
      const mix={};
      MIX_TYPES.forEach((t)=>{
        const v=parseFloat($(`fpl-mix-${t}`).value);
        mix[t]=Number.isFinite(v)&&v>0?v:0;
      });
      if(!Object.values(mix).some((v)=>v>0))throw new Error('give at least one mix weight above 0');
      const k=METRES_PER_UNIT;
      const n=boundary.length;
      if(btn)btn.disabled=true;
      showAssembleResult('assembling…');
      const res=await api('POST','/assemble_floor_plate',{
        boundary_m:boundary.map(({x,y})=>[x*k,y*k]),
        mix,
        street_edges:streetEdges.filter((e)=>e<n),
        north_deg:northDeg(),
      });
      const m=res.metrics||{};
      if(res.status!=='ok'){
        showAssembleResult(`${res.status}: ${res.reason||''}\ncandidates ${m.candidates_tried??0}, survived ${m.candidates_survived??0}`,true);
        return;
      }
      loadAssembled(res);
      const counts=MIX_TYPES.filter((t)=>(m.achieved_counts||{})[t]).map((t)=>`${m.achieved_counts[t]}×${t}`).join(' ');
      showAssembleResult(
        `status: ok\n${m.units} units: ${counts}\ncommons ${m.commons_pct}% · farthest walk ${m.max_walk_m} m · residual ${m.residual_m2} m²`);
    }catch(err){showAssembleResult(err.message,true);}
    finally{if(btn)btn.disabled=false;}
  }

  // ── visibility / lifecycle ────────────────────────────────────────────────
  function setVisible(on){
    visible=on;
    const sec=$('fpl-section');
    if(sec)sec.style.display=on?'block':'none';
    if(!on&&getState().mode==='streetEdge'){setMode('select');syncModeButton();}
    if(on){
      if(!fetched)refresh().catch((err)=>setStatus(err.message,true));
      sync();
    }
  }

  // Drawing a new boundary: the old plate's context no longer applies.
  function onBoundaryCleared(){
    streetEdges=[];
    loadedId=null;
    clearAssembled?.();
    showAssembleResult('');
    if(visible){renderList();sync();}
  }

  function bind(){
    $('fpl-btn-street')?.addEventListener('click',toggleStreetMode);
    $('fpl-btn-save-new')?.addEventListener('click',saveNew);
    $('fpl-btn-update')?.addEventListener('click',updateLoaded);
    $('fpl-north')?.addEventListener('input',()=>render());
  }
  buildAssembleBlock();
  bind();

  return { setVisible, sync, overlay, clickStreetEdge, onBoundaryCleared, syncModeButton };
}
