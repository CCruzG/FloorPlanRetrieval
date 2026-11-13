// Floor plate library (2026-11-06) — sidebar section in the Floor plate tab.
// Saves the drawn outline + its context (entrance, edge types, north bearing,
// street frontage, notes) to the repo file through /floor_plates (see
// packages/floor_plate_design/library.py). Canvas units are normalised; the file is
// in metres (canvas frame, y down), converted with METRES_PER_UNIT.
import { METRES_PER_UNIT } from './constants.js';
import { wallGeometryForPlan } from './plan-drawing-adapter.js';
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
  showUnitTab,     // () => void: switches to the Unit tab
  showPlateTab,    // () => void: switches to the Floor plate tab
  setComposedView, // (view|null) => void: puts the composed drawing data in floorPlateState and renders
  showBuilding,    // (building|null) => void: sets the composed building and, when given, shows the 3D tab
  plateHit,        // (logicalX, logicalY) => unit id | null (hit-test on the drawn plate)
  prepareUnitTab,  // () => void: clears generated overlay, cards and sensor panels of the previous unit
  enterSlot,       // (slot) => void: sets requirements, result cards, selection and generated overlay from a slot
  captureSlot,     // () => Promise<{requirements,candidates,chosen,plan,sensors,options}|null>
}) {
  const $=(id)=>doc.getElementById(id);
  let plates=[];
  let loadedId=null;
  let streetEdges=[];
  let visible=false;
  let fetched=false;
  // Plate composition (2026-11-09): the record from POST /compose_plate (app units; see
  // packages/floor_plate_design/compose.py), the units of the last assemble, the unit open in
  // the Unit tab, and the plate outline to restore when the Floor plate tab shows again.
  let composed=null;
  let assembledUnits=[];
  let activeUnit=null;
  let plateStash=null;

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
    resetComposition();
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
    $('fpl-unit-buttons')?.remove();
    $('fpl-compose')?.remove();
  }

  // One button per assembled unit. Click: Unit tab, boundary = unit polygon,
  // entrance = unit.entrance_point, edge types = unit.typed_edges, boundary closed.
  // Retrieval is not run. Unit data is metres, canvas frame, y down.
  function openUnit(u){
    const k=METRES_PER_UNIT;
    const types=[];
    (u.typed_edges||[]).forEach((e)=>{types[e.edge_index]=e.type;});
    if(!plateStash){
      const st=getState();
      plateStash={
        boundary:st.boundary.map(({x,y})=>({x,y})),
        entrance:st.entrance?{x:st.entrance.x,y:st.entrance.y}:null,
        edgeTypes:(st.edgeTypes||[]).slice(),
      };
    }
    showUnitTab?.();
    prepareUnitTab?.();
    applyState(createLoadedBoundaryState({
      vertices:u.polygon.map(([x,y])=>({x:x/k,y:y/k})),
      entrance:u.entrance_point?{x:u.entrance_point[0]/k,y:u.entrance_point[1]/k}:null,
      edgeTypes:types,
    }));
    $('entrance-controls').style.display='block';
    $('btn-entrance-mode').classList.remove('active');
    if(getState().entrance){updateEntranceInfo();}
    $('canvas-hint').textContent=`unit ${u.id} loaded · press retrieve`;
    $('s-bv').textContent=getState().boundary.length;
    $('result-count').textContent='';
    fitViewport();
    checkReady();
    render();
    activeUnit=u.id;
    const slot=composed?.slots?.[u.id];
    if(slot){
      enterSlot?.(slot);
      $('canvas-hint').textContent=`unit ${u.id} from the plate · pick a rank, generate, then back to plate`;
    }
    syncReturn();
  }

  // ── plate composition: generate all, drill-down return, export ────────────
  const SLOT_DONE=['generated','failed','edited'];

  function resetComposition(){
    composed=null;assembledUnits=[];activeUnit=null;plateStash=null;
    setComposedView?.(null);
    showBuilding?.(null);
    syncReturn();
  }

  function sensorsRecord(sensors){
    if(!sensors||typeof sensors!=='object')return null;
    const checks=Array.isArray(sensors.checks)?sensors.checks:[];
    return {summary:sensors.summary||{},failing:checks.filter((c)=>c&&!c.ok&&!c.skipped)};
  }

  // Drawing data for the canvas: {unitIdx: {rooms (mm), badge{text,fail}}}, joined to
  // floor_plate.units by id. The record itself stays in this module.
  function viewFromRecord(rec,units){
    if(!rec)return null;
    const k=METRES_PER_UNIT*1000;
    const view={};
    (units||[]).forEach((u,idx)=>{
      const slot=rec.slots?.[u.id];
      if(!slot||!SLOT_DONE.includes(slot.status))return;
      const s=slot.sensors?.summary||{};
      if(slot.status==='failed'||!slot.plan){
        view[idx]={rooms:[],badge:{text:`${u.id} ${u.type} failed`,fail:true}};
        return;
      }
      const rooms=(slot.plan.rooms||[]).map((r)=>({
        polygon:(r.polygon||[]).map(([x,y])=>[x*k,y*k]),
        source_type:r.source_type??r.type,
        label:r.label||(typeof r.type==='string'?r.type:'')||'',
      }));
      const score=(typeof s.passed==='number'&&typeof s.total==='number')?` ${s.passed}/${s.total}`:'';
      let walls=null;
      if(slot.plan.frame){
        try{walls=wallGeometryForPlan(slot.plan);}catch(e){console.warn('wall drawing failed',u.id,e);}
      }
      view[idx]={rooms,walls,badge:{
        text:`${u.id} ${u.type}${score}${slot.status==='edited'?' edited':''}`,
        fail:s.gates_pass===false,
      }};
    });
    return view;
  }

  function pushView(){
    setComposedView?.(composed,viewFromRecord);
  }

  function readMix(){
    const mix={};
    MIX_TYPES.forEach((t)=>{
      const v=parseFloat($(`fpl-mix-${t}`).value);
      mix[t]=Number.isFinite(v)&&v>0?v:0;
    });
    return mix;
  }

  function setComposeStatus(txt,isErr=false){
    const el=$('fpl-compose-status');
    if(!el)return;
    el.textContent=txt;
    el.style.color=isErr?'var(--red)':'var(--muted)';
  }

  async function composeAll(){
    const btn=$('fpl-btn-compose');
    try{
      if(!assembledUnits.length)throw new Error('assemble the plate first');
      const { boundary }=getState();
      const k=METRES_PER_UNIT;
      const n=boundary.length;
      if(btn)btn.disabled=true;
      setComposeStatus('generating all units…');
      const res=await api('POST','/compose_plate',{
        boundary_m:boundary.map(({x,y})=>[x*k,y*k]),
        mix:readMix(),
        street_edges:streetEdges.filter((e)=>e<n),
        north_deg:northDeg(),
        main_entrance_m:getState().entrance?[getState().entrance.x*k,getState().entrance.y*k]:null,
        balcony:!!$('fpl-chk-balcony')?.checked,
      });
      if(res.status!=='ok'||!res.record)throw new Error(`${res.status||'no result'}: ${res.reason||''}`);
      const ids=Object.keys(res.record.slots);
      if(ids.length!==assembledUnits.length||!assembledUnits.every((u)=>ids.includes(u.id))){
        throw new Error('the plate changed since it was assembled: assemble again');
      }
      composed=res.record;
      activeUnit=null;
      showBuilding?.(null);
      pushView();
      const slots=Object.values(composed.slots);
      const failed=slots.filter((x)=>x.status==='failed').map((x)=>x.unit_id);
      const red=slots.filter((x)=>x.status!=='failed'&&x.sensors?.summary?.gates_pass===false).map((x)=>x.unit_id);
      setComposeStatus(
        `${slots.length-failed.length}/${slots.length} units generated`+
        (failed.length?`\nfailed: ${failed.join(', ')}`:'')+
        (red.length?`\nhard sensor fails: ${red.join(', ')}`:''));
      const ex=$('fpl-btn-export');
      if(ex)ex.style.display='';
      const bw=$('fpl-build-wrap');
      if(bw)bw.style.display='flex';
    }catch(err){setComposeStatus(err.message,true);}
    finally{if(btn)btn.disabled=false;}
  }

  // "build 3D": storeys, floor-to-floor and maximum height go in the record's options; the
  // massing route returns the building and the 3D tab shows it (ARCHITECTURE.md, "Plate composition").
  async function buildBuilding(){
    const btn=$('fpl-btn-build3d');
    try{
      if(!composed)throw new Error('generate all units first');
      const storeys=parseInt($('fpl-storeys').value,10);
      const f2f=parseFloat($('fpl-f2f').value);
      const maxRaw=($('fpl-maxh').value||'').trim();
      const maxH=maxRaw===''?null:parseFloat(maxRaw);
      if(!Number.isInteger(storeys)||storeys<1)throw new Error('storeys must be a whole number of 1 or more');
      if(!Number.isFinite(f2f)||f2f<=0.25)throw new Error('floor-to-floor must be a number above 0.25');
      if(maxH!==null&&!(Number.isFinite(maxH)&&maxH>0))throw new Error('max height must be a positive number or blank');
      composed.options={...(composed.options||{}),storeys,floor_to_floor_m:f2f,max_height_m:maxH};
      if(btn)btn.disabled=true;
      setComposeStatus('building…');
      const b=await api('POST','/compose_plate/massing',{
        record:composed,storeys,floor_to_floor_m:f2f,max_height_m:maxH,
        main_entrance_m:composed.options.main_entrance_m||null,
      });
      const r=b.report||{};
      setComposeStatus(
        `${b.storeys}/${b.storeys_requested} storeys · ${b.total_height_m} m · ${b.prisms.length} prisms · ${b.windows.length} windows`+
        `\n${r.entrance||''}`+(r.unit_fallbacks?.length?`\nno plan, shown as blocks: ${r.unit_fallbacks.join(', ')}`:''));
      showBuilding?.(b);
    }catch(err){setComposeStatus(err.message,true);}
    finally{if(btn)btn.disabled=false;}
  }

  async function exportComposed(){
    try{
      if(!composed)throw new Error('generate all units first');
      const data=await api('POST','/compose_plate/export',{record:composed});
      const blob=new Blob([JSON.stringify(data,null,1)],{type:'application/json'});
      const a=doc.createElement('a');
      a.href=URL.createObjectURL(blob);
      a.download='composed-plate.json';
      doc.body.appendChild(a);a.click();a.remove();
      setTimeout(()=>URL.revokeObjectURL(a.href),2000);
      setComposeStatus('exported composed-plate.json');
    }catch(err){setComposeStatus(err.message,true);}
  }

  // Show or hide the "back to plate" block (Unit tab, while a unit from the plate is open).
  function syncReturn(){
    const box=$('unit-return');
    if(!box)return;
    box.style.display=(activeUnit&&!visible)?'block':'none';
    const lbl=$('unit-return-label');
    if(lbl)lbl.textContent=activeUnit?`unit ${activeUnit} opened from the plate`:'';
  }

  // Write the unit's current selection, generated plan, sensors and options into its slot
  // (status edited), then show the plate. With no generated plan and no selection the slot
  // is left as it was.
  async function backToPlate(){
    const btn=$('btn-back-to-plate');
    const slot=activeUnit?composed?.slots?.[activeUnit]:null;
    try{
      if(btn)btn.disabled=true;
      if(slot){
        const work=await captureSlot?.();
        if(work){
          slot.requirements=work.requirements;
          slot.candidates=work.candidates;
          slot.chosen=work.chosen;
          slot.plan=work.plan;
          slot.options=work.options;
          slot.sensors=sensorsRecord(work.sensors);
          slot.status='edited';
          slot.error=null;
        }
      }
      activeUnit=null;
      pushView();
      showPlateTab?.();
      syncReturn();
    }catch(err){
      $('canvas-hint').textContent=`back to plate failed: ${err.message}`;
    }finally{if(btn)btn.disabled=false;}
  }

  // Click on the drawn plate (Floor plate tab): open the unit under the pointer.
  function clickPlate(lx,ly){
    if(!visible||!assembledUnits.length)return false;
    const id=plateHit?.(lx,ly);
    const u=id?assembledUnits.find((x)=>x.id===id):null;
    if(!u)return false;
    openUnit(u);
    return true;
  }

  // The Floor plate tab shows again: put the plate outline back as the boundary.
  function restorePlate(){
    if(!plateStash)return;
    const st=plateStash;
    plateStash=null;
    prepareUnitTab?.();
    applyState(createLoadedBoundaryState({vertices:st.boundary,entrance:st.entrance,edgeTypes:st.edgeTypes}));
    const vp=getViewport();
    vp.zoom=1;vp.panX=0;vp.panY=0;
    if(getState().entrance)updateEntranceInfo();
    $('canvas-hint').textContent='floor plate loaded';
    $('s-bv').textContent=getState().boundary.length;
    $('result-count').textContent='';
    checkReady();
    render();
  }

  function showUnitButtons(units){
    const out=$('fpl-assemble-result');
    if(!out)return;
    const wrap=doc.createElement('div');
    wrap.id='fpl-unit-buttons';
    wrap.style.cssText='display:flex;flex-direction:column;gap:2px;margin-top:4px';
    units.forEach((u)=>{
      const b=doc.createElement('button');
      b.className='clear-btn';
      b.dataset.unit=u.id;
      b.textContent=`open ${u.id} (${u.type}, ${Math.round(u.area)} m²)`;
      b.addEventListener('click',()=>openUnit(u));
      wrap.appendChild(b);
    });
    out.after(wrap);
    const box=doc.createElement('div');
    box.id='fpl-compose';
    box.style.cssText='display:flex;flex-direction:column;gap:3px;margin-top:6px';
    const gen=doc.createElement('button');
    gen.id='fpl-btn-compose';gen.className='clear-btn';gen.style.marginTop='0';
    gen.textContent='generate all units';
    gen.addEventListener('click',composeAll);
    const balLbl=doc.createElement('label');
    balLbl.style.cssText='display:flex;gap:4px;align-items:center;font-size:9px;color:var(--muted)';
    const bal=doc.createElement('input');
    bal.id='fpl-chk-balcony';bal.type='checkbox';bal.checked=false;
    balLbl.appendChild(bal);balLbl.appendChild(doc.createTextNode('balconies'));
    const genRow=doc.createElement('div');
    genRow.style.cssText='display:flex;gap:8px;align-items:center';
    genRow.appendChild(gen);genRow.appendChild(balLbl);
    const ex=doc.createElement('button');
    ex.id='fpl-btn-export';ex.className='clear-btn';ex.style.cssText='margin-top:0;display:none';
    ex.textContent='export composed plate';
    ex.addEventListener('click',exportComposed);
    const st=doc.createElement('div');
    st.id='fpl-compose-status';
    st.style.cssText='font-size:9px;color:var(--muted);line-height:1.4;white-space:pre-wrap';
    const bw=doc.createElement('div');
    bw.id='fpl-build-wrap';
    bw.style.cssText='display:none;flex-direction:column;gap:3px';
    const row=doc.createElement('div');
    row.style.cssText='display:flex;gap:4px;align-items:center;font-size:9px;color:var(--muted)';
    const mk=(id,label,val,ph)=>{
      const l=doc.createElement('label');
      l.style.cssText='display:flex;flex-direction:column;gap:1px;flex:1;min-width:0';
      l.textContent=label;
      const i=doc.createElement('input');
      i.id=id;i.type='number';i.step=id==='fpl-storeys'?'1':'0.1';i.value=val;i.placeholder=ph||'';i.style.cssText=INP;
      l.appendChild(i);
      return l;
    };
    row.appendChild(mk('fpl-storeys','storeys','4'));
    row.appendChild(mk('fpl-f2f','floor-to-floor m','3.1'));
    row.appendChild(mk('fpl-maxh','max height m','','off'));
    const b3=doc.createElement('button');
    b3.id='fpl-btn-build3d';b3.className='clear-btn';b3.style.marginTop='0';
    b3.textContent='build 3D';
    b3.addEventListener('click',buildBuilding);
    bw.appendChild(row);bw.appendChild(b3);
    box.appendChild(genRow);box.appendChild(ex);box.appendChild(bw);box.appendChild(st);
    wrap.after(box);
  }

  async function assemble(){
    const btn=$('fpl-btn-assemble');
    try{
      const { boundary, closed }=getState();
      if(!closed||boundary.length<3)throw new Error('close the boundary first');
      const mix=readMix();
      if(!Object.values(mix).some((v)=>v>0))throw new Error('give at least one mix weight above 0');
      const k=METRES_PER_UNIT;
      const n=boundary.length;
      if(btn)btn.disabled=true;
      resetComposition();
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
      assembledUnits=res.units||[];
      const counts=MIX_TYPES.filter((t)=>(m.achieved_counts||{})[t]).map((t)=>`${m.achieved_counts[t]}×${t}`).join(' ');
      showAssembleResult(
        `status: ok\n${m.units} units: ${counts}\ncommons ${m.commons_pct}% · farthest walk ${m.max_walk_m} m · residual ${m.residual_m2} m²`);
      showUnitButtons(res.units||[]);
    }catch(err){showAssembleResult(err.message,true);}
    finally{if(btn)btn.disabled=false;}
  }

  // ── visibility / lifecycle ────────────────────────────────────────────────
  function setVisible(on){
    visible=on;
    const sec=$('fpl-section');
    if(sec)sec.style.display=on?'block':'none';
    if(!on&&getState().mode==='streetEdge'){setMode('select');syncModeButton();}
    if(on)restorePlate();
    syncReturn();
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
    resetComposition();
    showAssembleResult('');
    if(visible){renderList();sync();}
  }

  function bind(){
    $('fpl-btn-street')?.addEventListener('click',toggleStreetMode);
    $('fpl-btn-save-new')?.addEventListener('click',saveNew);
    $('fpl-btn-update')?.addEventListener('click',updateLoaded);
    $('fpl-north')?.addEventListener('input',()=>render());
    $('btn-back-to-plate')?.addEventListener('click',backToPlate);
  }
  buildAssembleBlock();
  bind();

  return { setVisible, sync, overlay, clickStreetEdge, onBoundaryCleared, syncModeButton, clickPlate };
}
