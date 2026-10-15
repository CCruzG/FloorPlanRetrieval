import { COMMON_AREA_TYPES, METRES_PER_UNIT } from './constants.js';
import {
  createBoundaryClearedState,
  createBoundaryClosedState,
  createLoadedBoundaryState,
} from './queryState.js';

const LIB_KEY='fp_boundary_library';

export function createBoundaryEditor({
  doc,
  getState,
  applyState,
  getActiveFPState,
  planToCanvas,
  canvasToPlan,
  ptInPolyFn,
  ptToSegDistSqFn,
  snapToEdge,
  render,
  renderList,
  checkReady,
  sync3DSceneIfNeeded,
}) {
  function libLoad(){
    try{return JSON.parse(localStorage.getItem(LIB_KEY)||'[]');}catch{return [];}
  }

  function libSave(lib){
    localStorage.setItem(LIB_KEY,JSON.stringify(lib));
  }

  function hitTestRooms(cx,cy){
    const s=getActiveFPState();
    if(!s)return null;
    const { fp, rot, mirror, offset, fpBdry }=s;
    const { selected }=getState();
    if(selected===null)return null;
    const rooms=fp.rooms||[];
    const [planX,planY]=canvasToPlan(cx,cy,fpBdry,rot,offset,mirror);
    const toPx=([px,py])=>planToCanvas(px,py,fpBdry,rot,offset,mirror);
    const EDGE_DSQ=64;
    let bestEdge=null;
    let bestEdgeDsq=Infinity;
    let firstBody=null;
    for(let ri=0;ri<rooms.length;ri++){
      const room=rooms[ri];
      const st=room.source_type??room.type;
      if(COMMON_AREA_TYPES.has(st))continue;
      if(!room.polygon||room.polygon.length<3)continue;
      const poly=room.polygon;
      const n=poly.length;
      const cpoly=poly.map(toPx);
      for(let i=0;i<n;i++){
        const [px1,py1]=poly[i];
        const [px2,py2]=poly[(i+1)%n];
        const adx=Math.abs(px2-px1),ady=Math.abs(py2-py1);
        if(adx>1e-4&&ady>1e-4)continue;
        const dsq=ptToSegDistSqFn([cx,cy],cpoly[i],cpoly[(i+1)%n]);
        if(dsq<EDGE_DSQ&&dsq<bestEdgeDsq){
          bestEdgeDsq=dsq;
          const axis=adx<ady?'v':'h';
          const coordValue=axis==='h'?(py1+py2)/2:(px1+px2)/2;
          bestEdge={roomIdx:ri,mode:'edge',edgeAxis:axis,coordValue};
        }
      }
      if(!firstBody&&ptInPolyFn([planX,planY],poly))firstBody={roomIdx:ri,mode:'body'};
    }
    return bestEdge||firstBody||null;
  }

  function hitTestEntrance(cx,cy){
    const s=getActiveFPState();
    if(!s)return false;
    const { fp, fpBdry, rot, offset, mirror }=s;
    const ent=fp.entrance;
    if(!ent||ent.length!==4)return false;
    const toPx=([px,py])=>planToCanvas(px,py,fpBdry,rot,offset,mirror);
    const corners=[[ent[0],ent[1]],[ent[2],ent[1]],[ent[2],ent[3]],[ent[0],ent[3]]].map(toPx);
    const xs=corners.map((p)=>p[0]);
    const ys=corners.map((p)=>p[1]);
    const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
    const PAD_HIT=6;
    return cx>=minX-PAD_HIT&&cx<=maxX+PAD_HIT&&cy>=minY-PAD_HIT&&cy<=maxY+PAD_HIT;
  }

  function toggleEntranceMode(){
    const { closed, entranceMode }=getState();
    if(!closed)return;
    const nextEntranceMode=!entranceMode;
    applyState({entranceMode:nextEntranceMode,entranceSnap:nextEntranceMode?getState().entranceSnap:null});
    doc.getElementById('btn-entrance-mode').classList.toggle('active',nextEntranceMode);
    if(nextEntranceMode){
      doc.getElementById('canvas-hint').textContent='click on a boundary edge to place entrance';
    }else{
      doc.getElementById('canvas-hint').textContent='boundary closed';
      applyState({entranceSnap:null});
    }
    render();
  }

  function clearEntrance(){
    applyState({entrance:null,entranceSnap:null,entranceMode:false});
    doc.getElementById('btn-entrance-mode').classList.remove('active');
    doc.getElementById('btn-clear-entrance').style.display='none';
    doc.getElementById('entrance-info').style.display='none';
    doc.getElementById('s-ent').textContent='—';
    doc.getElementById('canvas-hint').textContent='boundary closed · press retrieve';
    render();
  }

  function updateEntranceInfo(){
    const { entrance }=getState();
    if(!entrance)return;
    const xm=(entrance.x*METRES_PER_UNIT).toFixed(1);
    const ym=(entrance.y*METRES_PER_UNIT).toFixed(1);
    const txt=`x ${xm} m, y ${ym} m`;
    doc.getElementById('entrance-info').textContent=txt;
    doc.getElementById('entrance-info').style.display='block';
    doc.getElementById('btn-clear-entrance').style.display='block';
    doc.getElementById('s-ent').textContent=txt;
  }

  function closeBoundary(){
    applyState(createBoundaryClosedState());
    const { boundary }=getState();
    doc.getElementById('canvas-hint').textContent='boundary closed · press retrieve';
    doc.getElementById('s-bv').textContent=boundary.length;
    doc.getElementById('entrance-controls').style.display='block';
    checkReady();
    render();
  }

  function clearBoundary(){
    applyState(createBoundaryClearedState());
    doc.getElementById('entrance-controls').style.display='none';
    doc.getElementById('btn-entrance-mode').classList.remove('active');
    doc.getElementById('btn-clear-entrance').style.display='none';
    doc.getElementById('entrance-info').style.display='none';
    doc.getElementById('s-ent').textContent='—';
    renderList();
    doc.getElementById('canvas-hint').textContent='draw a boundary polygon to begin';
    doc.getElementById('s-bv').textContent='—';
    doc.getElementById('result-count').textContent='';
    checkReady();
    render();
    sync3DSceneIfNeeded(true);
  }

  function saveBoundary(){
    const { closed, boundary, entrance, edgeTypes, req }=getState();
    if(!closed||boundary.length<3){alert('Close the boundary first.');return;}
    const defaultName='boundary '+(libLoad().length+1);
    const el=doc.getElementById('lib-list');
    const saveRow=doc.createElement('div');
    saveRow.style.cssText='display:flex;align-items:center;gap:3px;margin-bottom:2px';
    const inp=doc.createElement('input');
    inp.type='text';inp.value=defaultName;inp.placeholder='name…';
    inp.style.cssText='flex:1;min-width:0;font-size:10px;font-family:var(--mono);background:var(--bg2);border:1px solid var(--accent);border-radius:3px;color:var(--text);padding:1px 3px';
    const ok=doc.createElement('button');
    ok.textContent='Save';
    ok.style.cssText='flex-shrink:0;padding:1px 6px;font-size:9px;font-family:var(--mono);background:var(--accent);border:none;border-radius:3px;color:#fff;cursor:pointer;line-height:1.5';
    const cancel=doc.createElement('button');
    cancel.textContent='✕';
    cancel.style.cssText='flex-shrink:0;padding:1px 5px;font-size:9px;font-family:var(--mono);background:transparent;border:1px solid var(--border2);border-radius:3px;color:var(--muted);cursor:pointer;line-height:1.5';
    const commit=()=>{
      const name=inp.value.trim()||defaultName;
      const lib=libLoad();
      lib.push({
        name,
        vertices:boundary.map(({x,y})=>({x,y})),
        entrance:entrance?{x:entrance.x,y:entrance.y}:null,
        edgeTypes:Array.isArray(edgeTypes)?edgeTypes.slice():[],
        saved:new Date().toISOString(),
        requirements:{bedrooms:req.bedrooms,bathrooms:req.bathrooms},
      });
      libSave(lib);
      renderLib();
    };
    ok.onclick=commit;
    cancel.onclick=()=>renderLib();
    inp.onkeydown=(e)=>{if(e.key==='Enter')commit();if(e.key==='Escape')renderLib();};
    saveRow.appendChild(inp);
    saveRow.appendChild(ok);
    saveRow.appendChild(cancel);
    el.insertBefore(saveRow,el.firstChild);
    inp.focus();
    inp.select();
  }

  function loadBoundary(idx){
    const lib=libLoad();
    const entry=lib[idx];
    if(!entry)return;
    applyState(createLoadedBoundaryState(entry));
    renderList();
    doc.getElementById('entrance-controls').style.display='block';
    doc.getElementById('btn-entrance-mode').classList.remove('active');
    if(getState().entrance){updateEntranceInfo();}
    else{
      doc.getElementById('btn-clear-entrance').style.display='none';
      doc.getElementById('entrance-info').style.display='none';
    }
    doc.getElementById('canvas-hint').textContent='boundary loaded · press retrieve';
    doc.getElementById('s-bv').textContent=getState().boundary.length;
    doc.getElementById('result-count').textContent='';
    checkReady();
    render();
  }

  function deleteBoundary(idx){
    const lib=libLoad();
    lib.splice(idx,1);
    libSave(lib);
    renderLib();
  }

  function renameBoundary(idx){
    const el=doc.getElementById('lib-list');
    const row=el.children[idx];
    if(!row)return;
    const lib=libLoad();
    row.innerHTML='';
    const inp=doc.createElement('input');
    inp.type='text';
    inp.value=lib[idx].name;
    inp.style.cssText='flex:1;min-width:0;font-size:10px;font-family:var(--mono);background:var(--bg2);border:1px solid var(--accent);border-radius:3px;color:var(--text);padding:1px 3px';
    const ok=doc.createElement('button');
    ok.textContent='✓';
    ok.style.cssText='flex-shrink:0;padding:1px 5px;font-size:9px;font-family:var(--mono);background:var(--accent);border:none;border-radius:3px;color:#fff;cursor:pointer;line-height:1.5';
    const cancel=doc.createElement('button');
    cancel.textContent='✕';
    cancel.style.cssText='flex-shrink:0;padding:1px 5px;font-size:9px;font-family:var(--mono);background:transparent;border:1px solid var(--border2);border-radius:3px;color:var(--muted);cursor:pointer;line-height:1.5';
    const commit=()=>{
      const name=inp.value.trim();
      if(name){lib[idx].name=name;libSave(lib);}
      renderLib();
    };
    ok.onclick=commit;
    cancel.onclick=()=>renderLib();
    inp.onkeydown=(e)=>{if(e.key==='Enter')commit();if(e.key==='Escape')renderLib();};
    row.appendChild(inp);
    row.appendChild(ok);
    row.appendChild(cancel);
    inp.focus();
    inp.select();
  }

  function renderLib(){
    const lib=libLoad();
    const el=doc.getElementById('lib-list');
    el.innerHTML='';
    if(!lib.length){
      el.innerHTML='<div style="font-size:10px;color:var(--muted)">no saved boundaries</div>';
      return;
    }
    lib.forEach((entry,i)=>{
      const row=doc.createElement('div');
      row.style.cssText='display:flex;align-items:center;gap:3px';

      const lbl=doc.createElement('span');
      lbl.style.cssText='flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:10px;cursor:pointer;color:var(--text)';
      lbl.title=entry.vertices.length+' verts · '+entry.saved.slice(0,10);
      lbl.textContent=entry.name;
      lbl.onclick=()=>loadBoundary(i);
      lbl.onmouseover=()=>{lbl.style.color='var(--accent)';};
      lbl.onmouseout=()=>{lbl.style.color='var(--text)';};

      const mkBtn=(label,title,cb)=>{
        const button=doc.createElement('button');
        button.textContent=label;
        button.title=title;
        button.style.cssText='flex-shrink:0;padding:1px 4px;font-size:9px;font-family:var(--mono);background:transparent;border:1px solid var(--border2);border-radius:3px;color:var(--muted);cursor:pointer;line-height:1.5';
        button.onmouseover=()=>{button.style.borderColor='var(--red)';button.style.color='var(--red)';};
        button.onmouseout=()=>{button.style.borderColor='var(--border2)';button.style.color='var(--muted)';};
        button.onclick=cb;
        return button;
      };

      row.appendChild(lbl);
      row.appendChild(mkBtn('✎','rename',()=>renameBoundary(i)));

      const delBtn=doc.createElement('button');
      delBtn.textContent='✕';
      delBtn.title='delete';
      delBtn.style.cssText='flex-shrink:0;padding:1px 4px;font-size:9px;font-family:var(--mono);background:transparent;border:1px solid var(--border2);border-radius:3px;color:var(--muted);cursor:pointer;line-height:1.5';
      delBtn.onmouseover=()=>{delBtn.style.borderColor='var(--red)';delBtn.style.color='var(--red)';};
      delBtn.onmouseout=()=>{delBtn.style.borderColor='var(--border2)';delBtn.style.color='var(--muted)';};
      delBtn.onclick=()=>{
        row.innerHTML='';
        const msg=doc.createElement('span');
        msg.style.cssText='flex:1;font-size:9px;color:var(--red);white-space:nowrap';
        msg.textContent='Delete "'+entry.name+'"?';
        const yes=doc.createElement('button');
        yes.textContent='Yes';
        yes.style.cssText='flex-shrink:0;padding:1px 5px;font-size:9px;font-family:var(--mono);background:var(--red);border:none;border-radius:3px;color:#fff;cursor:pointer;line-height:1.5';
        yes.onclick=()=>deleteBoundary(i);
        const no=doc.createElement('button');
        no.textContent='No';
        no.style.cssText='flex-shrink:0;padding:1px 5px;font-size:9px;font-family:var(--mono);background:transparent;border:1px solid var(--border2);border-radius:3px;color:var(--muted);cursor:pointer;line-height:1.5';
        no.onclick=()=>renderLib();
        row.appendChild(msg);
        row.appendChild(yes);
        row.appendChild(no);
      };
      row.appendChild(delBtn);
      el.appendChild(row);
    });
  }

  function exportLibrary(){
    const lib=libLoad();
    if(!lib.length){alert('Library is empty.');return;}
    const blob=new Blob([JSON.stringify(lib,null,2)],{type:'application/json'});
    const a=doc.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download='boundary_library.json';
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function importLibrary(inp){
    const file=inp.files[0];
    if(!file)return;
    const reader=new FileReader();
    reader.onload=(e)=>{
      try{
        const data=JSON.parse(e.target.result);
        if(!Array.isArray(data))throw new Error();
        const lib=libLoad();
        data.forEach((entry)=>{
          if(entry.name&&Array.isArray(entry.vertices))lib.push(entry);
        });
        libSave(lib);
        renderLib();
        alert('Imported '+data.length+' boundaries.');
      }catch{
        alert('Invalid library JSON.');
      }
      inp.value='';
    };
    reader.readAsText(file);
  }

  return {
    clearBoundary,
    clearEntrance,
    closeBoundary,
    exportLibrary,
    hitTestEntrance,
    hitTestRooms,
    importLibrary,
    renderLib,
    saveBoundary,
    toggleEntranceMode,
    updateEntranceInfo,
  };
}