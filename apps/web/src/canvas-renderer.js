import { COMMON_AREA_TYPES, METRES_PER_UNIT, RCOLS, ROOM_TYPES, VR } from './constants.js';
import { bdryCenter, rotatePtAround } from './geometry.js';

// id -> abbr lookup for the debug type label drawn on each room in drawFP.
const ROOM_ABBR = Object.fromEntries((ROOM_TYPES.types || []).map((t) => [t.id, t.abbr]));

// Shared boundary-edge-type colour convention — same values as the live
// boundary-edge-type overlay in render() (EDGE_COLOURS, ~line 264).
const EDGE_TYPE_COLOURS = { external: '#22c55e', shared_wall: '#64748b', facing_corridor: '#eab308' };

export function createCanvasRenderer({
  canvas,
  ctx,
  getState,
  getActiveFPState,
  n2c,
  ghost,
  snapToEdge,
  planToCanvas,
}) {
  function updateCursor(){
    const { dragState, hoveredEntrance, hoveredRoom, measureMode, mode }=getState();
    if(measureMode){
      canvas.style.cursor='crosshair';
    }else if(dragState){
      canvas.style.cursor=dragState.mode==='entrance'||dragState.mode==='move'?'grabbing':(dragState.edgeAxis==='h'?'ns-resize':'ew-resize');
    }else if(hoveredEntrance){
      canvas.style.cursor='grab';
    }else if(hoveredRoom){
      canvas.style.cursor=hoveredRoom.mode==='body'?'grab':(hoveredRoom.edgeAxis==='h'?'ns-resize':'ew-resize');
    }else{
      // crosshair only in draw/measure (measure handled above) — select/entrance/edgeType get the default arrow
      canvas.style.cursor=mode==='draw'?'crosshair':'default';
    }
  }

  let lastGraphNodes=null;

  function render(){
    const {
      boundary,
      closed,
      entrance,
      entranceMode,
      entranceSnap,
      edgeTypes,
      edgeTypeMode,
      hoveredEdge,
      mouse,
      shiftDown,
      snapFirst,
      results,
      selected,
      fitMode,
      fittedCache,
      showPlanBdry,
      showWindows,
      showDoors,
      showCommonArea,
      selectedRoom,
      graphSelection,
      displacementArrows,
      cW,
      cH,
      params,
      floorPlateState,
      devMode,
      viewport,
      activeTab,
      fitState,
      arapDebugRooms,
      arapDebugLabel,
      arapDebugDoor,
      arapDebugReport,
      generatedRooms,
      generatedNodes,
      generatedGrid,
      generatedEdges,
      measureMode,
      measureA,
      measureB,
    }=getState();

    const vpZoom=viewport?.zoom??1;
    const vpPanX=viewport?.panX??0;
    const vpPanY=viewport?.panY??0;

    lastGraphNodes=null;

    ctx.clearRect(0,0,cW,cH);
    ctx.fillStyle='#e8eaed';
    ctx.fillRect(0,0,cW,cH);

    // Apply viewport transform for all world-space drawing
    ctx.save();
    ctx.translate(cW/2+vpPanX,cH/2+vpPanY);
    ctx.scale(vpZoom,vpZoom);
    ctx.translate(-cW/2,-cH/2);

    drawScaleGrid(ctx,cW,cH);

    if(devMode?.enabled&&devMode?.showFloorPlateOverlay&&floorPlateState){
      renderFloorPlate(ctx,floorPlateState,cW,cH,devMode);
      ctx.restore();
      drawScaleBar(ctx,cW,cH,vpZoom);
      return;
    }

    if(selected!==null&&results[selected]?.fp){
      let selRot=results[selected].rot||0;
      const selMirror=results[selected].mirror||false;
      const rawPlan=results[selected].fp;
      const isFitTab=activeTab==='fit';
      // In Fit tab: always show the original (un-fitted) plan; fitted cache is irrelevant here.
      const fp=(fitMode&&!isFitTab&&fittedCache[selected]?.fp)||rawPlan;
      const qc=bdryCenter(boundary.map(({x,y})=>[x,y]));
      const pc=bdryCenter(fp.boundary||[]);
      const selOff=[qc[0]-pc[0],qc[1]-pc[1]];
      // Auto-align: orient the retrieved plan so its OWN entrance snaps to the
      // entrance placed on the drawn boundary. retrieval.rot is only a
      // shape-match angle — it does not put the plan's door on the query's
      // door. Rotate the plan (and its graph) so the plan-centre->entrance
      // direction matches the query-centre->entrance direction, snapped to the
      // nearest 90° so rooms stay axis-aligned. Falls back to retrieval.rot when
      // either entrance is missing. Display-only: r.rot itself is untouched, so
      // the fit/adaptation path (which recomputes its own alignment) is unaffected.
      const rpc=bdryCenter(rawPlan.boundary||[]);
      const pEnt=rawPlan.entrance;
      if(entrance&&Array.isArray(pEnt)&&pEnt.length===4&&rpc){
        let emx=(pEnt[0]+pEnt[2])/2, emy=(pEnt[1]+pEnt[3])/2;
        if(selMirror)emy=2*rpc[1]-emy;   // mirror is applied before rotation in t()
        const pDir=Math.atan2(emy-rpc[1],emx-rpc[0]);
        const qDir=Math.atan2(entrance.y-qc[1],entrance.x-qc[0]);
        const half=Math.PI/2;
        selRot=Math.round((qDir-pDir)/half)*half;
      }
      // Once an ARAP debug overlay (align/rectify/etc.) is showing, the raw
      // retrieved-plan preview underneath it is just visual noise — hide it
      // rather than draw both on top of each other.
      const arapActive=!!(arapDebugRooms&&arapDebugRooms.length);
      // Stage-0 generated geometry is its own target-space overlay (drawn below);
      // while it is up, hide the source-space preview + source graph the same way
      // the ARAP debug overlay does, so the two coordinate frames never overlap.
      const genActive=!!(generatedRooms&&generatedRooms.length);
      const planVisible=!arapActive&&!genActive&&(!isFitTab||(fitState?.showOriginal!==false));
      // params.showPlanGeom (default true) is the master switch for the plan's
      // polygonal layer — rooms, plan boundary, doors, windows, entrance — so it
      // can be dropped to read the topology graph alone. The query boundary
      // overlay and the graph are drawn separately and are unaffected.
      const showPlanGeom=params.showPlanGeom!==false;
      // Topology-graph view: shrink the WHOLE retrieved plan (room fills, doors,
      // and the graph nodes) uniformly into the drawn boundary so every node sits
      // on its room AND lands inside the boundary — one transform for both layers,
      // no drift. Contain-scale is about the query-boundary centre (selOff already
      // aligned the plan centre there). Only in the main graph view; the plain
      // preview and the fit tab stay native size. n2c, drawFP's fit-box and the
      // graph's own t() are the same mapping (pad 0.07), so a canvas-space scale
      // about n2c(qc) contains all three consistently.
      const useTargetGraph=!isFitTab&&!genActive&&params.showGraph
        &&results[selected].retrieval?.room_edges&&boundary.length>=3;
      let containScale=null;
      if(useTargetGraph){
        const dcen=(selRot||selMirror)?bdryCenter(rawPlan.boundary||[]):null;
        const toDisp=([x,y])=>{
          if(selMirror&&dcen)y=2*dcen[1]-y;
          if(selRot&&dcen)[x,y]=rotatePtAround([x,y],selRot,dcen);
          return n2c(x+selOff[0],y+selOff[1]);
        };
        const pp=(rawPlan.boundary||[]).map((p)=>toDisp([p[0],p[1]]));
        const tp=boundary.map((b)=>n2c(b.x,b.y));
        if(pp.length&&tp.length){
          const pxs=pp.map((p)=>p[0]),pys=pp.map((p)=>p[1]);
          const txs=tp.map((p)=>p[0]),tys=tp.map((p)=>p[1]);
          const pw=Math.max(...pxs)-Math.min(...pxs),ph=Math.max(...pys)-Math.min(...pys);
          const bw=Math.max(...txs)-Math.min(...txs),bh=Math.max(...tys)-Math.min(...tys);
          if(pw>1e-6&&ph>1e-6){
            containScale={s:Math.min(bw/pw,bh/ph),pivot:n2c(qc[0],qc[1])};
          }
        }
      }
      if(planVisible&&showPlanGeom){
        if(isFitTab)ctx.globalAlpha=0.45;
        if(containScale){
          ctx.save();
          ctx.translate(containScale.pivot[0],containScale.pivot[1]);
          ctx.scale(containScale.s,containScale.s);
          ctx.translate(-containScale.pivot[0],-containScale.pivot[1]);
        }
        drawFP(ctx,fp,cW,cH,false,results[selected].adapted,selRot,selOff,selMirror,
          {showBdry:showPlanBdry,showCommon:showCommonArea,hiliteRoom:selectedRoom,hoverEdge:true});
        drawDoorsWindows(ctx,fp,cW,cH,selRot,selOff,selMirror,{showWin:showWindows,showIntDoors:showDoors});
        if(containScale)ctx.restore();
        if(isFitTab)ctx.globalAlpha=1.0;
      }
      if(!arapActive&&!genActive&&(params.showGraph||(isFitTab&&fitState?.showGraph&&!fitState?.mappedGraph))){
        // Prefer the rich topology graph (nodes + adjacency + room->boundary
        // connectors) — same renderer the retrieval cards use. room_edges is
        // keyed to the ORIGINAL retrieved plan's boundary, so draw rawPlan (not
        // the fitted fp). Fall back to the plain adjacency graph when a result
        // has no room_edges (e.g. the fit tab).
        const selRet=results[selected].retrieval;
        if(drawTopologyGraph&&selRet&&selRet.room_edges)
          lastGraphNodes=drawTopologyGraph(ctx,rawPlan,selRet,cW,cH,selRot,selOff,selMirror,graphSelection,boundary.map(({x,y})=>[x,y]),n2c,containScale)||null;
        else
          drawGraph(ctx,fp,cW,cH,selRot,selOff,selMirror);
      }
      if(!arapActive&&!isFitTab&&fitMode&&fittedCache[selected]?.fp)drawFitHighlights(ctx,fp,cW,cH,selRot,selOff,selMirror);
      if(!arapActive&&displacementArrows.length)drawDisplacementArrows(ctx,displacementArrows,fp.boundary||[],selRot,selOff,selMirror);
      if(!isFitTab||(fitState?.showBoundary!==false))drawBoundaryOverlay(isFitTab);
      if(isFitTab&&fitState?.showGraph&&fitState?.mappedGraph)
        drawFitGraph(ctx,fitState.mappedGraph);

      const ent=fp.entrance;
      if(!arapActive&&!genActive&&showPlanGeom&&ent&&ent.length===4){
        const cen=bdryCenter(fp.boundary||[]);
        let [ex1,ey1]=[ent[0],ent[1]];
        let [ex2,ey2]=[ent[2],ent[3]];
        if(selMirror){ey1=2*cen[1]-ey1;ey2=2*cen[1]-ey2;}
        [ex1,ey1]=rotatePtAround([ex1,ey1],selRot,cen);
        [ex2,ey2]=rotatePtAround([ex2,ey2],selRot,cen);
        ex1+=selOff[0];ey1+=selOff[1];ex2+=selOff[0];ey2+=selOff[1];
        const [px1,py1]=n2c(ex1,ey1),[px2,py2]=n2c(ex2,ey2);
        ctx.beginPath();ctx.rect(px1,py1,px2-px1,py2-py1);
        ctx.fillStyle='#88888866';ctx.fill();
        ctx.strokeStyle='#555';ctx.lineWidth=2;ctx.setLineDash([]);ctx.stroke();
      }
    }else if(boundary.length>=3&&closed){
      ctx.beginPath();
      boundary.forEach(({x,y},i)=>{
        const [cx,cy]=n2c(x,y);
        i===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);
      });
      ctx.closePath();
      ctx.fillStyle='rgba(59,130,246,0.04)';
      ctx.fill();
    }

    // ARAP debug overlay (2026-08-11) — draws real-geometry output from an
    // arap_bridge.py step (e.g. similarity_align) via the SAME drawFP used
    // for the normal retrieval preview, not a separate renderer. rot/mirror/
    // offset are 0 because this geometry has already been placed in absolute
    // query-space by the actual pipeline step, not the canvas-time
    // rot/mirror display trick the raw retrieved-plan preview relies on.
    if(arapDebugRooms&&arapDebugRooms.length){
      // drawFP early-returns if fp.boundary is empty (line ~694) — this call
      // never passed one, so it silently no-op'd on every call since this
      // overlay was first built. rot/mirror are 0 so the boundary's actual
      // values don't affect the transform, it just needs to be non-empty;
      // the query boundary is the semantically correct thing to pass anyway.
      drawFP(ctx,{rooms:arapDebugRooms,boundary:boundary.map(({x,y})=>[x,y])},cW,cH,true,false,0,[0,0],false,{showBdry:false});
      if(arapDebugDoor&&arapDebugDoor.length===4){
        // Same coordinate frame as the rooms above (absolute query-space,
        // no rot/mirror/offset) — matches drawFP's own internal transform
        // exactly since it's also called with rot=0/offset=[0,0]/mirror=false.
        const pad=0.07;
        const sc=Math.min(cW,cH)*(1-2*pad);
        const dox=(cW-sc)/2,doy=(cH-sc)/2;
        const [dx1,dy1,dx2,dy2]=arapDebugDoor;
        const px1=dx1*sc+dox,py1=dy1*sc+doy,px2=dx2*sc+dox,py2=dy2*sc+doy;
        ctx.beginPath();
        ctx.rect(Math.min(px1,px2),Math.min(py1,py2),Math.abs(px2-px1),Math.abs(py2-py1));
        ctx.fillStyle='#88888666';ctx.fill();
        ctx.strokeStyle='#555';ctx.lineWidth=2;ctx.setLineDash([]);ctx.stroke();
      }
      if(arapDebugLabel){
        ctx.fillStyle='#1e293b';
        ctx.font='11px monospace';
        ctx.fillText(arapDebugLabel,8,cH-8);
      }
      // Stage 1 (re-solve loop) — draw the arrangement's own judgement over the
      // rooms: under-min rooms flagged red with their area/min tag, and a status
      // HUD making the console report legible where the layout is. Read-only.
      if(arapDebugReport){
        const rep=arapDebugReport;
        const pad=0.07,sc=Math.min(cW,cH)*(1-2*pad),dox=(cW-sc)/2,doy=(cH-sc)/2;
        const toPx=([x,y])=>[x*sc+dox,y*sc+doy];
        const okById={};
        (rep.rooms||[]).forEach((rr)=>{okById[rr.id]=rr;});
        ctx.save();
        (arapDebugRooms||[]).forEach((room)=>{
          const rr=okById[room.id];
          if(!rr||!room.polygon||room.polygon.length<3)return;
          const underMin=!rr.ok;
          const noLight=rr.daylight_ok===false;
          const noAccess=rr.access_ok===false;
          if(!underMin&&!noLight&&!noAccess)return;
          // Outline red for a dimensional failure, amber for a daylight-only one,
          // violet for an access-only one (most-severe wins the outline).
          ctx.beginPath();
          room.polygon.forEach((p,i)=>{const[px,py]=toPx(p);i===0?ctx.moveTo(px,py):ctx.lineTo(px,py);});
          ctx.closePath();
          ctx.fillStyle=underMin?'rgba(239,68,68,0.16)':noLight?'rgba(245,158,11,0.16)':'rgba(168,85,247,0.16)';ctx.fill();
          ctx.strokeStyle=underMin?'#ef4444':noLight?'#f59e0b':'#a855f7';ctx.lineWidth=2.5;ctx.setLineDash([]);ctx.stroke();
          const cx=room.polygon.reduce((a,p)=>a+p[0],0)/room.polygon.length;
          const cy=room.polygon.reduce((a,p)=>a+p[1],0)/room.polygon.length;
          const[tx,ty]=toPx([cx,cy]);
          const tags=[];
          if(underMin)tags.push([`▲ ${rr.label} ${rr.area_m2}/${rr.min_area_m2}m²`,'rgba(239,68,68,0.92)']);
          if(noLight)tags.push([`☼ ${rr.label} no daylight`,'rgba(245,158,11,0.95)']);
          if(noAccess)tags.push([`↛ ${rr.label} no access`,'rgba(168,85,247,0.95)']);
          ctx.font='9px monospace';ctx.textAlign='center';
          tags.forEach((tg,ti)=>{
            const top=ty-7+ti*16;
            const w=ctx.measureText(tg[0]).width+8;
            ctx.fillStyle=tg[1];ctx.fillRect(tx-w/2,top,w,14);
            ctx.fillStyle='#fff';ctx.fillText(tg[0],tx,top+10);
          });
        });
        ctx.restore();
        // The summary HUD lives in a DOM overlay anchored to the canvas window
        // (updateCanvasHud in main.js), not painted here — so it stays put under
        // pan/zoom. Only the per-room flags above are drawn on the canvas.
      }
    }

    // ── Stage-0 generated overlay (node-driven solver, /generate) ────────────
    // generatedRooms are ALREADY in target (drawn-boundary) space, drawn with the
    // pad-0.07 fit-box (toPx) — the same mapping as n2c and the boundary overlay —
    // so they land inside the drawn boundary with no source->target display trick.
    // Gate on generatedRooms directly: genActive is scoped to the selected-plan
    // branch above and is not in scope here.
    if(generatedRooms&&generatedRooms.length){
      // Draw directly (not via drawFP, which colours by numeric source_type the
      // retrieved plan may not carry): a distinct translucent colour per room +
      // a dark outline so adjacent rooms are separable, the room label at the
      // centroid, and the node dot on top so "node inside its room" is visible.
      const pad=0.07,sc=Math.min(cW,cH)*(1-2*pad),dox=(cW-sc)/2,doy=(cH-sc)/2;
      const toPx=([x,y])=>[x*sc+dox,y*sc+doy];
      const GPAL=['#60a5fa','#f472b6','#34d399','#fbbf24','#a78bfa','#fb7185','#22d3ee','#a3e635','#fca5a5','#c4b5fd'];
      ctx.save();
      // Solver grid UNDER the rooms (report.grid_lines): how step 3 cut the
      // boundary. Dashed slate = cut between node bands (moves with the solve);
      // solid amber = pinned to a boundary vertex (notch-aligned, never moves);
      // hatched = cell outside the boundary (no owner, no area credited).
      const G=generatedGrid;
      if(G&&Array.isArray(G.xs)&&Array.isArray(G.ys)&&G.xs.length>1&&G.ys.length>1){
        ctx.save();ctx.globalAlpha=0.3;   // faint: context under the rooms, not content
        const gx=G.xs,gy=G.ys,x0=gx[0],x1=gx[gx.length-1],y0=gy[0],y1=gy[gy.length-1];
        (G.cells||[]).forEach(([i,j,own])=>{
          if(own!==null&&own!==undefined)return;
          const[ax,ay]=toPx([gx[i],gy[j]]),[bx,by]=toPx([gx[i+1],gy[j+1]]);
          ctx.fillStyle='rgba(100,116,139,0.16)';ctx.fillRect(ax,ay,bx-ax,by-ay);
          ctx.save();ctx.beginPath();ctx.rect(ax,ay,bx-ax,by-ay);ctx.clip();
          ctx.strokeStyle='rgba(100,116,139,0.35)';ctx.lineWidth=1;ctx.setLineDash([]);
          for(let t=-(by-ay);t<(bx-ax);t+=6){ctx.beginPath();ctx.moveTo(ax+t,by);ctx.lineTo(ax+t+(by-ay),ay);ctx.stroke();}
          ctx.restore();
        });
        const pinX=new Set(G.pinned_x||[]),pinY=new Set(G.pinned_y||[]);
        const line=(p,q,pinned)=>{
          const[ax,ay]=toPx(p),[bx,by]=toPx(q);
          ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(bx,by);
          ctx.strokeStyle=pinned?'rgba(217,119,6,0.85)':'rgba(51,65,85,0.55)';
          ctx.lineWidth=pinned?1.5:1;ctx.setLineDash(pinned?[]:[4,3]);ctx.stroke();
        };
        gx.forEach((x,k)=>line([x,y0],[x,y1],pinX.has(k)));
        gy.forEach((y,k)=>line([x0,y],[x1,y],pinY.has(k)));
        ctx.setLineDash([]);
        ctx.restore();
      }
      (generatedRooms||[]).forEach((room,ri)=>{
        if(!room.polygon||room.polygon.length<3)return;
        ctx.beginPath();
        room.polygon.forEach((p,i)=>{const[px,py]=toPx(p);i===0?ctx.moveTo(px,py):ctx.lineTo(px,py);});
        ctx.closePath();
        ctx.fillStyle=GPAL[ri%GPAL.length]+'99';ctx.fill();
        ctx.strokeStyle='#1e293b';ctx.lineWidth=1.5;ctx.setLineDash([]);ctx.stroke();
        const cx=room.polygon.reduce((a,p)=>a+p[0],0)/room.polygon.length;
        const cy=room.polygon.reduce((a,p)=>a+p[1],0)/room.polygon.length;
        const[lx,ly]=toPx([cx,cy]);
        ctx.fillStyle='#0f172a';ctx.font='10px monospace';ctx.textAlign='center';
        ctx.fillText(String(room.type??room.id),lx,ly);
      });
      // redraw the topology graph over the generated geometry: adjacency edges
      // between the (re-placed) nodes, so the graph matches the new layout.
      // Three-valued now (moves.py SetConnection): door (connection 1) solid dark,
      // open (connection 2) solid green — the two "passable" types — wall
      // (connection 0 / absent) faint dashed. generatedEdges is the live plan's
      // own edges (reflects a Stage-2 SetConnection edit); falls back to the
      // retrieved source's edges when nothing has been generated with edits yet.
      const genEdges=generatedEdges||(results[selected]?.fp?.edges)||[];
      genEdges.forEach((e)=>{
        const a=generatedNodes?.[String(e.u)],b=generatedNodes?.[String(e.v)];
        if(!Array.isArray(a)||!Array.isArray(b))return;
        const[ax,ay]=toPx(a),[bx,by]=toPx(b);
        const isOpen=(e.connection===2);
        const isDoor=(e.connection===1);
        ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(bx,by);
        if(isOpen){
          ctx.strokeStyle='rgba(34,197,94,0.75)';ctx.lineWidth=2;ctx.setLineDash([6,3]);
        }else if(isDoor){
          ctx.strokeStyle='rgba(15,23,42,0.6)';ctx.lineWidth=2;ctx.setLineDash([]);
        }else{
          ctx.strokeStyle='rgba(15,23,42,0.22)';ctx.lineWidth=1;ctx.setLineDash([3,3]);
        }
        ctx.stroke();
      });
      ctx.setLineDash([]);
      // room↔boundary connectors (room_edges) over the generated geometry — the same
      // data the sensors read, so the realised frontage of each room is visible here
      // too (the graph view only draws them when the overlay is off).
      const _etc={external:'#22c55e',shared_wall:'#64748b',facing_corridor:'#eab308'};
      const genRE=results[selected]?.retrieval?.room_edges||{};
      Object.entries(genRE).forEach(([rid,conns])=>{
        const node=generatedNodes?.[String(rid)];
        if(!Array.isArray(node)||node.length<2)return;
        const[gnx,gny]=toPx(node);
        (Array.isArray(conns)?conns:[]).forEach((cn)=>{
          if(!Array.isArray(cn?.target_point)||cn.target_point.length<2)return;
          const[tx,ty]=toPx(cn.target_point);
          ctx.beginPath();ctx.moveTo(gnx,gny);ctx.lineTo(tx,ty);
          ctx.strokeStyle=_etc[cn.edge_type]||'rgba(15,23,42,0.5)';
          ctx.lineWidth=1.5;ctx.setLineDash([2,2]);ctx.stroke();
        });
      });
      ctx.setLineDash([]);
      Object.values(generatedNodes||{}).forEach((p)=>{
        if(!Array.isArray(p)||p.length<2)return;
        const[px,py]=toPx(p);
        ctx.beginPath();ctx.arc(px,py,5,0,Math.PI*2);
        ctx.fillStyle='#111827';ctx.fill();
        ctx.lineWidth=2;ctx.strokeStyle='#fff';ctx.stroke();
      });
      ctx.restore();
      ctx.fillStyle='#1e293b';ctx.font='11px monospace';ctx.textAlign='left';
      ctx.fillText('stage 0 · generated from nodes',8,cH-8);
    }

    if(!closed&&boundary.length>=1){
      const g=ghost();
      const gx=g?g.x:mouse.x;
      const gy=g?g.y:mouse.y;
      ctx.save();
      ctx.setLineDash([2,4]);
      ctx.lineWidth=0.75;
      boundary.forEach(({x,y})=>{
        const [vx,vy]=n2c(x,y);
        const nearH=g&&Math.abs(gy-y)<0.004;
        ctx.strokeStyle=nearH?'rgba(251,146,60,0.85)':'rgba(100,116,139,0.35)';
        ctx.beginPath();ctx.moveTo(-99999,vy);ctx.lineTo(99999,vy);ctx.stroke();
        const nearV=g&&Math.abs(gx-x)<0.004;
        ctx.strokeStyle=nearV?'rgba(251,146,60,0.85)':'rgba(100,116,139,0.35)';
        ctx.beginPath();ctx.moveTo(vx,-99999);ctx.lineTo(vx,99999);ctx.stroke();
      });
      ctx.restore();
    }

    const g=ghost();
    if(g&&boundary.length>0&&!closed){
      const last=boundary[boundary.length-1];
      const [lx,ly]=n2c(last.x,last.y),[gx,gy]=n2c(g.x,g.y);
      ctx.beginPath();ctx.moveTo(lx,ly);ctx.lineTo(gx,gy);
      ctx.strokeStyle=snapFirst?'rgba(239,68,68,0.7)':'rgba(59,130,246,0.55)';
      ctx.lineWidth=1.5;ctx.setLineDash([5,4]);ctx.stroke();ctx.setLineDash([]);
      if(!snapFirst){
        ctx.beginPath();ctx.arc(gx,gy,3,0,Math.PI*2);
        ctx.fillStyle=shiftDown?'#f59e0b':'rgba(59,130,246,0.7)';ctx.fill();
      }
      const lenM=Math.hypot(g.x-last.x,g.y-last.y)*METRES_PER_UNIT;
      if(lenM>0.05){
        const mx=(lx+gx)/2,my=(ly+gy)/2-8;
        ctx.save();ctx.font='10px monospace';ctx.fillStyle='#1e40af';
        ctx.textAlign='center';ctx.fillText(lenM.toFixed(1)+' m',mx,my);ctx.restore();
      }
    }

    if(boundary.length>=2){
      // Per-edge type colouring once the boundary is closed and edges are being
      // classified (or any type is set); otherwise the plain single-stroke path.
      const EDGE_COLOURS={external:'#22c55e',shared_wall:'#64748b',facing_corridor:'#eab308'};
      const perEdge=closed&&(edgeTypeMode||(Array.isArray(edgeTypes)&&edgeTypes.some(Boolean)));
      if(perEdge){
        const n=boundary.length;
        for(let i=0;i<n;i++){
          const a=boundary[i],b=boundary[(i+1)%n];
          const [ax,ay]=n2c(a.x,a.y),[bx,by]=n2c(b.x,b.y);
          const t=(edgeTypes&&edgeTypes[i])||null;
          ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(bx,by);
          ctx.strokeStyle=t?EDGE_COLOURS[t]:'#1e40af';
          ctx.lineWidth=(hoveredEdge===i)?5:(t?3.5:2);
          ctx.setLineDash(t?[]:(edgeTypeMode?[4,3]:[]));
          ctx.stroke();
        }
        ctx.setLineDash([]);
      }else{
        ctx.beginPath();
        boundary.forEach(({x,y},i)=>{
          const [cx,cy]=n2c(x,y);
          i===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);
        });
        if(closed)ctx.closePath();
        ctx.strokeStyle='#1e40af';ctx.lineWidth=2;ctx.setLineDash([]);ctx.stroke();
      }

      if(closed){
        ctx.save();ctx.font='9px monospace';ctx.fillStyle='rgba(30,64,175,0.75)';ctx.textAlign='center';
        const n=boundary.length;
        for(let i=0;i<n;i++){
          const a=boundary[i],b=boundary[(i+1)%n];
          const lenM=Math.hypot(b.x-a.x,b.y-a.y)*METRES_PER_UNIT;
          if(lenM<0.1)continue;
          const [ax,ay]=n2c(a.x,a.y),[bx,by]=n2c(b.x,b.y);
          const mx=(ax+bx)/2,my=(ay+by)/2;
          const ex=bx-ax,ey=by-ay,eL=Math.hypot(ex,ey);
          const ox=-ey/eL*10,oy=ex/eL*10;
          ctx.fillText(lenM.toFixed(1)+' m',mx+ox,my+oy);
        }
        ctx.restore();
      }
    }

    boundary.forEach(({x,y},i)=>{
      const [cx,cy]=n2c(x,y);
      if(i===0&&snapFirst&&!closed){
        ctx.beginPath();ctx.arc(cx,cy,VR+5,0,Math.PI*2);
        ctx.strokeStyle='#ef4444';ctx.lineWidth=1.5;ctx.stroke();
      }
      ctx.beginPath();ctx.arc(cx,cy,VR,0,Math.PI*2);
      ctx.fillStyle=i===0?'#ef4444':'#1e40af';
      ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.fill();ctx.stroke();
    });

    const drawEnt=entrance||(entranceMode&&entranceSnap?{x:entranceSnap.px,y:entranceSnap.py}:null);
    if(drawEnt){
      const snap=entranceMode&&entranceSnap&&!entrance?entranceSnap:snapToEdge(drawEnt.x,drawEnt.y);
      if(snap){
        const [ex,ey]=n2c(snap.px,snap.py);
        const isPreview=!entrance;
        const alpha=isPreview?0.55:1.0;
        const arrowLen=22,nx=snap.normX,ny=snap.normY;
        const [ax,ay]=[ex+nx*arrowLen,ey+ny*arrowLen];
        ctx.save();ctx.globalAlpha=alpha;
        ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ax,ay);
        ctx.strokeStyle='#f59e0b';ctx.lineWidth=2.5;ctx.setLineDash([]);ctx.stroke();
        const ang=Math.atan2(ny,nx);
        const hs=7;
        ctx.beginPath();
        ctx.moveTo(ax,ay);
        ctx.lineTo(ax-hs*Math.cos(ang-0.45),ay-hs*Math.sin(ang-0.45));
        ctx.lineTo(ax-hs*Math.cos(ang+0.45),ay-hs*Math.sin(ang+0.45));
        ctx.closePath();ctx.fillStyle='#f59e0b';ctx.fill();
        ctx.beginPath();ctx.arc(ex,ey,5,0,Math.PI*2);
        ctx.fillStyle='#f59e0b';ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.fill();ctx.stroke();
        ctx.restore();
      }
    }

    // Measure tool overlay (2026-10-30) — point A to point B (or to the live
    // cursor before B is placed), length in metres via METRES_PER_UNIT. Drawn
    // in world space (inside the viewport transform) so it pans/zooms with
    // the plan like the boundary edges it measures against.
    if(measureMode&&measureA){
      const end=measureB||mouse;
      const [ax,ay]=n2c(measureA.x,measureA.y),[bx,by]=n2c(end.x,end.y);
      ctx.save();
      ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(bx,by);
      ctx.strokeStyle=measureB?'#dc2626':'rgba(220,38,38,0.55)';
      ctx.lineWidth=1.5;ctx.setLineDash(measureB?[]:[5,4]);ctx.stroke();ctx.setLineDash([]);
      [[ax,ay],[bx,by]].forEach(([px,py])=>{
        ctx.beginPath();ctx.arc(px,py,3.5,0,Math.PI*2);ctx.fillStyle='#dc2626';ctx.fill();
      });
      const lenM=Math.hypot(end.x-measureA.x,end.y-measureA.y)*METRES_PER_UNIT;
      const mx=(ax+bx)/2,my=(ay+by)/2-9;
      ctx.font='11px monospace';ctx.fillStyle='#dc2626';ctx.textAlign='center';
      ctx.fillText(lenM.toFixed(2)+' m',mx,my);
      ctx.restore();
    }

    // Restore ctx from viewport transform, then draw screen-space overlays
    ctx.restore();
    drawScaleBar(ctx,cW,cH,vpZoom);
  }

  function renderFloorPlate(c,state,W,H,devMode={}){
    const floorPlate=state?.floor_plate||state;
    const boundary=floorPlate?.boundary;
    if(!Array.isArray(boundary)||boundary.length<3)return;
    const showElementIds=devMode?.showElementIds!==false;
    const showEdgeIndices=!!devMode?.showEdgeIndices;
    const layers=devMode?.floorPlateLayers||{};
    const showBoundary=layers.boundary!==false;
    const showCores=layers.cores!==false;
    const showCommonAreas=layers.commonAreas!==false;
    const showUnitBoundaries=layers.unitBoundaries!==false;
    const showGrid=layers.grid!==false;

    const pad=0.08;
    const xs=boundary.map(([x])=>x);
    const ys=boundary.map(([,y])=>y);
    const minX=Math.min(...xs),maxX=Math.max(...xs);
    const minY=Math.min(...ys),maxY=Math.max(...ys);
    const spanX=Math.max(maxX-minX,1e-6);
    const spanY=Math.max(maxY-minY,1e-6);
    const usableW=W*(1-2*pad);
    const usableH=H*(1-2*pad);
    const scale=Math.min(usableW/spanX,usableH/spanY);
    const xOffset=(W-scale*spanX)/2-minX*scale;
    const yOffset=(H-scale*spanY)/2-minY*scale;
    const t=([x,y])=>[x*scale+xOffset,y*scale+yOffset];
    const centroid=(poly)=>{
      const n=poly.length;
      if(!n)return[0,0];
      let sx=0,sy=0;
      poly.forEach(([x,y])=>{sx+=x;sy+=y;});
      return[sx/n,sy/n];
    };
    const drawPoly=(poly,{fill=null,stroke='#111',lineWidth=1.5,dash=[]}={})=>{
      if(!Array.isArray(poly)||poly.length<3)return;
      c.beginPath();
      poly.forEach((pt,i)=>{
        const [cx,cy]=t(pt);
        i===0?c.moveTo(cx,cy):c.lineTo(cx,cy);
      });
      c.closePath();
      if(fill){
        c.fillStyle=fill;
        c.fill();
      }
      c.strokeStyle=stroke;
      c.lineWidth=lineWidth;
      c.setLineDash(dash);
      c.stroke();
      c.setLineDash([]);
    };

    // Structural grid first so all program geometry remains legible above it.
    const grid=floorPlate.structural_grid||{};
    if(showGrid){
      c.save();
      c.strokeStyle='rgba(30,64,175,0.28)';
      c.lineWidth=1;
      c.setLineDash([5,5]);
      (grid.x_lines||[]).forEach((x)=>{
        const [x1,y1]=t([x,minY]);
        const [x2,y2]=t([x,maxY]);
        c.beginPath();c.moveTo(x1,y1);c.lineTo(x2,y2);c.stroke();
      });
      (grid.y_lines||[]).forEach((y)=>{
        const [x1,y1]=t([minX,y]);
        const [x2,y2]=t([maxX,y]);
        c.beginPath();c.moveTo(x1,y1);c.lineTo(x2,y2);c.stroke();
      });
      c.restore();
    }

    if(showBoundary){
      drawPoly(boundary,{fill:'rgba(59,130,246,0.05)',stroke:'#1f2937',lineWidth:2});
      const lengths = boundary.map(([x1, y1], index) => {
        const [x2, y2] = boundary[(index + 1) % boundary.length];
        return {
          index,
          len_m: resolveLinearMetres(Math.hypot(x2 - x1, y2 - y1)),
          mid: [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
        };
      });
      lengths.forEach((edge) => {
        const [cx, cy] = t(edge.mid);
        c.fillStyle='rgba(15,23,42,0.9)';
        c.font='9px monospace';
        c.textAlign='center';
        c.fillText(`${edge.len_m.toFixed(1)}m`, cx, cy - 4);
      });
    }

    if(showCommonAreas){
      (floorPlate.corridors||[]).forEach((corridor)=>{
        drawPoly(corridor.polygon,{fill:'rgba(250,204,21,0.28)',stroke:'rgba(202,138,4,0.95)',lineWidth:2});
        if(showElementIds){
          const [cx,cy]=t(centroid(corridor.polygon));
          c.fillStyle='#92400e';
          c.font='9px monospace';
          c.textAlign='center';
          c.fillText(`Corridor ${corridor.id||''}`.trim(),cx,cy-3);
        }
      });
    }

    if(showCores){
      (floorPlate.cores||[]).forEach((core)=>{
        drawPoly(core.polygon,{fill:'rgba(148,163,184,0.36)',stroke:'#475569',lineWidth:2});
        const [cx,cy]=t(centroid(core.polygon));
        if(showElementIds){
          c.fillStyle='#334155';
          c.font='9px monospace';
          c.textAlign='center';
          c.fillText(`Core ${core.id||''}`.trim(),cx,cy+3);
        }
      });
    }

    const EDGE_COLORS={
      external:'#2563eb',
      shared_wall:'#64748b',
      facing_corridor:'#eab308',
    };
    if(showUnitBoundaries){
      (floorPlate.units||[]).forEach((unit,unitIdx)=>{
        const poly=unit.polygon;
        drawPoly(poly,{fill:'rgba(16,185,129,0.12)',stroke:'rgba(5,150,105,0.9)',lineWidth:1.5});

        if(Array.isArray(unit.typed_edges)){
          unit.typed_edges.forEach((typedEdge)=>{
            const edgeIndex=typedEdge.edge_index;
            if(!Array.isArray(poly)||!poly.length||typeof edgeIndex!=='number')return;
            const from=poly[edgeIndex];
            const to=poly[(edgeIndex+1)%poly.length];
            if(!from||!to)return;
            const [x1,y1]=t(from);
            const [x2,y2]=t(to);
            c.beginPath();
            c.moveTo(x1,y1);
            c.lineTo(x2,y2);
            c.strokeStyle=EDGE_COLORS[typedEdge.type]||'#0f172a';
            c.lineWidth=3;
            c.setLineDash([]);
            c.stroke();

            if(showEdgeIndices){
              const mx=(x1+x2)/2;
              const my=(y1+y2)/2;
              c.fillStyle='rgba(15,23,42,0.88)';
              c.font='9px monospace';
              c.textAlign='center';
              c.fillText(`e${edgeIndex}`,mx,my-4);
            }
          });
        }

        const [ux,uy]=t(centroid(poly));
        c.fillStyle='#065f46';
        c.font='10px monospace';
        c.textAlign='center';
        const labelType=unit.type||'unit';
        const unitLabel=showElementIds?`${unit.id||'unit'} (${labelType})`:labelType;
        c.fillText(unitLabel,ux,uy-2);
        const area=resolveUnitAreaM2(unit,poly);
        c.fillStyle='#0f766e';
        c.fillText(`${area.toFixed(2)} m2`,ux,uy+10);

        // Retrieved plan id, when a layout has been retrieved for this unit.
        // Makes it readable at a glance which units drew the SAME source plan
        // — common on a slab of repeated bays, where identical unit geometry
        // legitimately retrieves the same match — and lets any single result
        // be traced back to its RPlan entry.
        const retrievedPlanName=floorPlate.unitPlans?.[unitIdx]?.name;
        if(retrievedPlanName){
          c.fillStyle='#7c3aed';
          c.font='9px monospace';
          c.fillText(`plan ${retrievedPlanName}`,ux,uy+21);
          c.font='10px monospace';
        }

        // Edge dimension labels
        for(let i=0;i<poly.length;i++){
          const from=poly[i];
          const to=poly[(i+1)%poly.length];
          const lenM=resolveLinearMetres(Math.hypot(to[0]-from[0],to[1]-from[1]));
          if(lenM<0.05)continue;
          const label=`${lenM.toFixed(2)}m`;
          const [x1,y1]=t(from);
          const [x2,y2]=t(to);
          const mx=(x1+x2)/2;
          const my=(y1+y2)/2;
          const edgeAngle=Math.atan2(y2-y1,x2-x1);
          // Perpendicular offset inward toward polygon centroid
          const perpAngle=edgeAngle-Math.PI/2;
          const testX=mx+Math.cos(perpAngle)*8;
          const testY=my+Math.sin(perpAngle)*8;
          const sign=((testX-ux)**2+(testY-uy)**2)<((mx-ux)**2+(my-uy)**2)?1:-1;
          const INSET=10;
          c.save();
          c.translate(mx+Math.cos(perpAngle)*INSET*sign, my+Math.sin(perpAngle)*INSET*sign);
          let a=edgeAngle;
          if(a>Math.PI/2||a<=-Math.PI/2)a+=Math.PI;
          c.rotate(a);
          c.font='8px monospace';
          c.textAlign='center';
          c.textBaseline='middle';
          const tw=c.measureText(label).width+4;
          c.fillStyle='rgba(255,255,255,0.75)';
          c.fillRect(-tw/2,-6,tw,9);
          c.fillStyle='rgba(5,150,105,0.95)';
          c.fillText(label,0,0);
          c.restore();
        }

        // Entrance door symbol: amber dot on the corridor-facing edge + door-leaf line into unit.
        if(Array.isArray(unit.entrance_point)){
          const [ex,ey]=t(unit.entrance_point);
          const facingEdge=(unit.typed_edges||[]).find(e=>e.type==='facing_corridor');
          let edgeAngle=0;
          if(facingEdge!=null&&Array.isArray(poly)){
            const ei=facingEdge.edge_index;
            const fp=poly[ei]; const tp=poly[(ei+1)%poly.length];
            if(fp&&tp){
              const[fx1,fy1]=t(fp); const[fx2,fy2]=t(tp);
              edgeAngle=Math.atan2(fy2-fy1,fx2-fx1);
            }
          }
          // Perpendicular direction pointing into the unit (away from corridor)
          const inward=edgeAngle+Math.PI/2;
          // Push inward a tiny bit to make sure it's inside
          const testX=ex+Math.cos(inward)*6; const testY=ey+Math.sin(inward)*6;
          const inwardSign=((testX-ux)**2+(testY-uy)**2)<((ex-ux)**2+(ey-uy)**2)?1:-1;
          const perpIn=inward+(inwardSign<0?Math.PI:0);
          const DOOR_PX=Math.max(5, 900*scale);
          c.save();
          c.translate(ex,ey);
          // Door leaf: line from entrance point into unit
          c.beginPath();
          c.moveTo(0,0);
          c.lineTo(Math.cos(perpIn)*DOOR_PX, Math.sin(perpIn)*DOOR_PX);
          c.strokeStyle='rgba(234,179,8,0.9)';
          c.lineWidth=1.5;
          c.setLineDash([]);
          c.stroke();
          // Door swing arc (quarter circle)
          c.beginPath();
          c.arc(0,0,DOOR_PX,perpIn-Math.PI/2,perpIn,false);
          c.strokeStyle='rgba(234,179,8,0.45)';
          c.lineWidth=1;
          c.stroke();
          // Entrance dot
          c.beginPath();
          c.arc(0,0,3,0,Math.PI*2);
          c.fillStyle='rgba(234,179,8,1)';
          c.fill();
          c.strokeStyle='rgba(255,255,255,0.8)';
          c.lineWidth=1;
          c.stroke();
          c.restore();
        }

        // Render retrieved apartment layout for this unit if available.
        // Keyed by POSITION, not unit.id — the generator does not guarantee
        // unique ids (a double-loaded slab repeats unit-0..n once per zone),
        // so keying by id silently collapsed multiple units onto one entry
        // and left the rest of the floor blank. Both sides walk
        // floorPlate.units in the same order, so the index is a reliable join.
        const unitPlan=floorPlate.unitPlans?.[unitIdx];
        if(unitPlan){
          (unitPlan.rooms||[]).forEach((room)=>{
            const rpoly=room.polygon;
            if(!Array.isArray(rpoly)||rpoly.length<3)return;
            const st=room.source_type??room.type;
            const col=RCOLS[st]??'#ccc';
            c.beginPath();
            rpoly.forEach((pt,i)=>{
              const[rx,ry]=t(pt);
              i===0?c.moveTo(rx,ry):c.lineTo(rx,ry);
            });
            c.closePath();
            c.fillStyle=col+'99';
            c.fill();
            c.strokeStyle='#00000030';
            c.lineWidth=0.75;
            c.setLineDash([]);
            c.stroke();
            // Step 2 viability flags: overlay the room red so a non-viable room
            // (no daylight / no access) is seen, not read as an id. Base fill
            // stays visible underneath so the room type is still legible.
            if(Array.isArray(room.flags)&&room.flags.length){
              c.fillStyle='rgba(220,38,38,0.40)';
              c.fill();
              c.strokeStyle='rgba(185,28,28,0.95)';
              c.lineWidth=2;
              c.setLineDash([5,3]);
              c.stroke();
              c.setLineDash([]);
            }
          });
        }
      });
    }

    if(showCommonAreas){
      (floorPlate.unresolved_regions||[]).forEach((region)=>{
        drawPoly(region,{fill:'rgba(239,68,68,0.22)',stroke:'rgba(185,28,28,0.9)',lineWidth:1.5,dash:[4,3]});
      });
    }

    const showOffsetPreview=layers.offsetPreview!==false;
    const offsetResult=devMode?.offsetSubdivision?.latestRegions;
    const activeOffset=offsetResult?.active_sample || offsetResult;
    if(showOffsetPreview&&activeOffset?.status==='ok'){
      (activeOffset.projected_unit_boundaries||[]).forEach((item)=>{
        const line=item.line;
        if(!Array.isArray(line)||line.length<2)return;
        const [a,b]=line;
        const [x1,y1]=t(a);
        const [x2,y2]=t(b);
        c.beginPath();
        c.moveTo(x1,y1);
        c.lineTo(x2,y2);
        c.strokeStyle='rgba(30,64,175,0.85)';
        c.lineWidth=1.5;
        c.setLineDash([6,4]);
        c.stroke();
        c.setLineDash([]);
      });

      (activeOffset.apartment_strips||[]).forEach((strip)=>{
        (strip.polygons||[]).forEach((poly)=>{
          drawPoly(poly,{fill:'rgba(34,197,94,0.08)',stroke:'rgba(22,163,74,0.45)',lineWidth:1,dash:[3,2]});
        });
      });

      (activeOffset.corner_bays||[]).forEach((bay)=>{
        drawPoly(bay.polygon,{fill:'rgba(249,115,22,0.15)',stroke:'rgba(194,65,12,0.75)',lineWidth:1,dash:[2,2]});
      });

      (activeOffset.inner_polygons||[]).forEach((region)=>{
        drawPoly(region.polygon,{fill:'rgba(59,130,246,0.08)',stroke:'rgba(37,99,235,0.9)',lineWidth:1.5,dash:[6,3]});
      });

      const primary=activeOffset.primary_inner_polygon;
      if(primary?.polygon?.length){
        const [cx,cy]=t(centroid(primary.polygon));
        c.fillStyle='rgba(30,64,175,0.9)';
        c.font='9px monospace';
        c.textAlign='center';
        c.fillText(`offset ${activeOffset.depth_m?.toFixed?.(1)??'n/a'}m`,cx,cy-8);
        c.fillText(`inner ${activeOffset.diagnostics?.inner_region_count||0}`,cx,cy+3);
        c.fillText(`proj ${activeOffset.projected_unit_boundaries?.length||0}`,cx,cy+14);
      }
    }
  }

  function polygonArea(poly){
    if(!Array.isArray(poly)||poly.length<3)return 0;
    let sum=0;
    for(let i=0;i<poly.length;i++){
      const [x1,y1]=poly[i];
      const [x2,y2]=poly[(i+1)%poly.length];
      sum+=x1*y2-x2*y1;
    }
    return Math.abs(sum)*0.5;
  }

  function resolveLinearMetres(value){
    if(!Number.isFinite(value)||value<=0)return 0;
    // Generated floor-plate API geometry is in mm, while editor coordinates are normalized units.
    if(value>1000)return value/1000;
    return value*METRES_PER_UNIT;
  }

  function resolveUnitAreaM2(unit,poly){
    const fromModel=Number.isFinite(unit?.area)?unit.area:null;
    const fromPoly=polygonArea(poly);
    const raw=fromModel??fromPoly;
    // Generated API polygons are in millimetres, so convert mm2 -> m2 when values are large.
    if(raw>100000)return raw/1_000_000;
    return raw;
  }

  function drawScaleGrid(c,W,H){
    const gridM=3;
    const step=gridM/METRES_PER_UNIT;
    c.save();c.strokeStyle='rgba(0,0,0,0.07)';c.lineWidth=0.5;c.setLineDash([]);
    // Draw extra range so grid covers canvas when panned or zoomed out
    for(let x=-5;x<=6;x+=step){
      const [px]=n2c(x,0);
      c.beginPath();c.moveTo(px,-99999);c.lineTo(px,99999);c.stroke();
    }
    for(let y=-5;y<=6;y+=step){
      const [,py]=n2c(0,y);
      c.beginPath();c.moveTo(-99999,py);c.lineTo(99999,py);c.stroke();
    }
    c.restore();
  }

  function drawScaleBar(c,W,H,zoom=1){
    // Base logical scale: how many logical pixels per normalised unit
    const sc=Math.min(W,H)*(1-2*0.07);
    // At current zoom: sc*zoom screen pixels per normalised unit = 18 m
    // Pick a nice round bar length that targets ~80 screen pixels
    const targetPx=80;
    const metresPerScreenPx=METRES_PER_UNIT/(sc*zoom);
    const barMetresExact=targetPx*metresPerScreenPx;
    // Round to a nice number
    const mag=Math.pow(10,Math.floor(Math.log10(Math.max(barMetresExact,0.01))));
    let barM=mag;
    for(const m of[1,2,2.5,5,10]){if(m*mag<=barMetresExact*1.8)barM=m*mag;}
    const barPx=barM*(sc*zoom)/METRES_PER_UNIT;
    const ox=16,oy=H-20;
    const tickH=5;
    c.save();
    c.strokeStyle='rgba(0,0,0,0.55)';c.lineWidth=1.5;c.setLineDash([]);
    c.beginPath();c.moveTo(ox,oy);c.lineTo(ox+barPx,oy);c.stroke();
    c.beginPath();c.moveTo(ox,oy-tickH);c.lineTo(ox,oy+tickH);c.stroke();
    c.beginPath();c.moveTo(ox+barPx,oy-tickH);c.lineTo(ox+barPx,oy+tickH);c.stroke();
    c.fillStyle='rgba(0,0,0,0.6)';c.font='10px monospace';
    c.textAlign='center';
    const label=barM>=1000?`${barM/1000} km`:`${barM} m`;
    c.fillText(label,ox+barPx/2,oy-tickH-3);
    // Zoom level indicator when not at 1:1
    if(Math.abs(zoom-1)>0.01){
      c.textAlign='left';
      c.fillStyle='rgba(0,0,0,0.35)';
      c.fillText(`${zoom.toFixed(2)}×`,ox,oy+tickH+12);
    }
    c.restore();
  }

  function drawBoundaryOverlay(prominent=false){
    const { boundary, closed }=getState();
    if(!boundary.length)return;
    if(boundary.length>=3&&closed){
      ctx.beginPath();
      boundary.forEach(({x,y},i)=>{
        const [cx,cy]=n2c(x,y);
        i===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);
      });
      ctx.closePath();
      // In Fit tab: show a light fill so the target shape reads clearly.
      ctx.fillStyle=prominent?'rgba(59,130,246,0.08)':'rgba(0,0,0,0)';
      ctx.fill();
    }
    if(boundary.length>=2){
      ctx.beginPath();
      boundary.forEach(({x,y},i)=>{
        const [cx,cy]=n2c(x,y);
        i===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);
      });
      if(closed)ctx.closePath();
      ctx.strokeStyle='#3b82f6';
      ctx.lineWidth=prominent?3:2;
      ctx.setLineDash(prominent?[]:[6,4]);
      ctx.stroke();ctx.setLineDash([]);
    }
    boundary.forEach(({x,y},i)=>{
      const [cx,cy]=n2c(x,y);
      ctx.beginPath();ctx.arc(cx,cy,prominent?5:4,0,Math.PI*2);
      ctx.fillStyle=i===0?'rgba(239,68,68,0.8)':'rgba(59,130,246,0.8)';
      ctx.fill();
    });
  }

  function drawFP(c,fp,W,H,dark,showClipped,rot=0,offset=[0,0],mirror=false,opts={}){
    const { hoveredRoom, dragState }=getState();
    const rooms=fp.rooms||[];
    const bdry=fp.boundary||[];
    if(!bdry.length)return;
    const pad=0.07;
    const sc=Math.min(W,H)*(1-2*pad);
    const ox=(W-sc)/2,oy=(H-sc)/2;
    const cen=(rot||mirror)?bdryCenter(bdry):null;
    const showBdry=opts.showBdry??true;
    const showCommon=opts.showCommon??true;
    const hiliteRoom=opts.hiliteRoom??null;
    const t=([x,y])=>{
      // y-flip, not x-flip — matches geo_ent_num_retrieval.retrieve()'s
      // actual mirror convention (conj() = angle negation). See 2026-08-11.
      if(mirror)y=2*cen[1]-y;
      if(rot)[x,y]=rotatePtAround([x,y],rot,cen);
      return [(x+offset[0])*sc+ox,(y+offset[1])*sc+oy];
    };
    rooms.forEach((room,ri)=>{
      const st=room.source_type??room.type;
      if(!showCommon&&COMMON_AREA_TYPES.has(st))return;
      if(!room.polygon||room.polygon.length<3)return;
      const col=RCOLS[st]??'#ddd';
      const pts=room.polygon.map(t);
      c.beginPath();
      pts.forEach(([cx,cy],i)=>{i===0?c.moveTo(cx,cy):c.lineTo(cx,cy);});
      c.closePath();
      c.fillStyle=showClipped&&room.clipped?col+'88':col+(dark?'cc':'bb');
      c.fill();
      if(ri===hiliteRoom){
        c.strokeStyle='#f59e0b';c.lineWidth=2.5;c.setLineDash([]);c.stroke();
      }else if(showClipped&&room.clipped){
        c.strokeStyle='#f59e0b';c.lineWidth=1.5;c.setLineDash([3,2]);c.stroke();c.setLineDash([]);
      }else{
        c.strokeStyle='#00000020';c.lineWidth=0.5;c.stroke();
      }
      if(opts.showTypeLabel??true){
        // DEBUG (2026-08-12): label each room with its raw type code + abbr,
        // drawn at the transformed-polygon centroid, so type/colour identity
        // can be read directly off the canvas instead of inferred visually —
        // for the test_boundary_004 room-labelling-after-align question.
        const cx=pts.reduce((s,[x])=>s+x,0)/pts.length;
        const cy=pts.reduce((s,[,y])=>s+y,0)/pts.length;
        c.save();
        c.font='11px sans-serif';
        c.textAlign='center';
        c.textBaseline='middle';
        const label=`${ROOM_ABBR[st]??'?'} (${st})`;
        c.lineWidth=3;
        c.strokeStyle=dark?'#000000cc':'#ffffffcc';
        c.strokeText(label,cx,cy);
        c.fillStyle=dark?'#fff':'#111';
        c.fillText(label,cx,cy);
        c.restore();
      }
    });
    if(opts.hoverEdge&&hoveredRoom&&!dragState){
      const hr=rooms[hoveredRoom.roomIdx];
      if(hr&&hr.polygon){
        const poly=hr.polygon,n=poly.length;
        const axis=hoveredRoom.edgeAxis,cv=hoveredRoom.coordValue;
        c.strokeStyle='rgba(245,158,11,0.9)';c.lineWidth=2.5;c.setLineDash([]);
        for(let i=0;i<n;i++){
          const [px1,py1]=poly[i],[px2,py2]=poly[(i+1)%n];
          const match=axis==='h'?Math.abs((py1+py2)/2-cv)<1e-4:Math.abs((px1+px2)/2-cv)<1e-4;
          if(match){
            const [ax,ay]=t(poly[i]),[bx,by]=t(poly[(i+1)%n]);
            c.beginPath();c.moveTo(ax,ay);c.lineTo(bx,by);c.stroke();
          }
        }
      }
    }
    if(showBdry){
      c.beginPath();
      bdry.forEach((v,i)=>{const [cx,cy]=t([v[0],v[1]]);i===0?c.moveTo(cx,cy):c.lineTo(cx,cy);});
      c.closePath();
      c.strokeStyle='#111';c.lineWidth=dark?1.5:2;c.setLineDash([]);c.stroke();
    }

    const ent=fp.entrance;
    if(ent&&ent.length===4){
      const [ex1,ey1]=t([ent[0],ent[1]]);
      const [ex2,ey2]=t([ent[2],ent[3]]);
      c.beginPath();c.rect(ex1,ey1,ex2-ex1,ey2-ey1);
      c.fillStyle='#88888866';c.fill();
      c.strokeStyle='#555';c.lineWidth=dark?1.5:2;c.setLineDash([]);c.stroke();
    }
  }

  function drawDoorsWindows(c,fp,W,H,rot=0,offset=[0,0],mirror=false,opts={}){
    const showWin=opts.showWin??true;
    const showIntDoors=opts.showIntDoors??true;
    const pad=0.07;
    const sc=Math.min(W,H)*(1-2*pad),ox=(W-sc)/2,oy=(H-sc)/2;
    const cen=(rot||mirror)?bdryCenter(fp.boundary||[]):null;
    const t=([x,y])=>{
      // y-flip, not x-flip — matches geo_ent_num_retrieval.retrieve()'s
      // actual mirror convention (conj() = angle negation). See 2026-08-11.
      if(mirror)y=2*cen[1]-y;
      if(rot)[x,y]=rotatePtAround([x,y],rot,cen);
      return [(x+offset[0])*sc+ox,(y+offset[1])*sc+oy];
    };
    const tb=([x1,y1,x2,y2])=>{
      const corners=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]].map(t);
      const xs=corners.map((p)=>p[0]),ys=corners.map((p)=>p[1]);
      return [Math.min(...xs),Math.min(...ys),Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys)];
    };
    if(showWin)(fp.windows||[]).forEach((win)=>{
      if(!win.bbox||win.bbox.length!==4)return;
      const [rx,ry,rw,rh]=tb(win.bbox);
      c.beginPath();c.rect(rx,ry,rw,rh);
      c.fillStyle='rgba(103,232,249,0.85)';c.fill();
      c.strokeStyle='rgba(6,182,212,0.9)';c.lineWidth=1;c.setLineDash([]);c.stroke();
    });
    if(showIntDoors)(fp.doors||[]).forEach((door)=>{
      if(!door.bbox||door.bbox.length!==4)return;
      const [rx,ry,rw,rh]=tb(door.bbox);
      c.beginPath();c.rect(rx,ry,rw,rh);
      c.fillStyle='rgba(251,146,60,0.90)';c.fill();
      c.strokeStyle='rgba(194,65,12,0.9)';c.lineWidth=1;c.setLineDash([]);c.stroke();
    });
  }

  function drawGraph(c,fp,W,H,rot=0,offset=[0,0],mirror=false){
    const rooms=fp.rooms||[],edges=fp.edges||[];
    if(!rooms.length)return;
    const pad=0.07;
    const sc=Math.min(W,H)*(1-2*pad),ox=(W-sc)/2,oy=(H-sc)/2;
    const rotCen=(rot||mirror)?bdryCenter(fp.boundary||[]):null;
    const t=([x,y])=>{
      if(mirror)y=2*rotCen[1]-y;
      if(rot)[x,y]=rotatePtAround([x,y],rot,rotCen);
      return [(x+offset[0])*sc+ox,(y+offset[1])*sc+oy];
    };
    const roomCen={};
    rooms.forEach((room)=>{
      if(!room.polygon||room.polygon.length<3)return;
      const mx=room.polygon.reduce((sum,pt)=>sum+pt[0],0)/room.polygon.length;
      const my=room.polygon.reduce((sum,pt)=>sum+pt[1],0)/room.polygon.length;
      roomCen[room.id]=t([mx,my]);
    });
    const dashSeg=Math.max(4,W*0.025);
    edges.forEach((edge)=>{
      const a=roomCen[edge.u],b=roomCen[edge.v];
      if(!a||!b)return;
      c.beginPath();c.moveTo(a[0],a[1]);c.lineTo(b[0],b[1]);
      if(edge.connection===1){
        c.strokeStyle='rgba(15,15,15,0.85)';c.lineWidth=2;c.setLineDash([]);
      }else{
        c.strokeStyle='rgba(15,15,15,0.40)';c.lineWidth=1.5;c.setLineDash([dashSeg,dashSeg]);
      }
      c.stroke();c.setLineDash([]);
    });
    const nr=Math.max(3,W*0.016);
    rooms.forEach((room)=>{
      const pos=roomCen[room.id];
      if(!pos)return;
      const col=RCOLS[room.source_type??room.type]??'#ccc';
      c.beginPath();c.arc(pos[0],pos[1],nr,0,Math.PI*2);
      c.fillStyle=col;c.fill();
      c.strokeStyle='#222';c.lineWidth=1;c.stroke();
    });
  }

  // Topology-only view of a retrieved plan: room nodes at polygon centroids,
  // room<->room adjacency edges, and each room's connection to the boundary
  // (external/shared_wall/facing_corridor, from retrieval.room_edges) drawn
  // to the midpoint of the boundary edge it attaches to. Reuses drawFP/
  // drawGraph's coordinate transform and EDGE_TYPE_COLOURS. All lookups are
  // defensive — a malformed/missing field is skipped, never thrown.
  function drawTopologyGraph(c,fp,retrieval,W,H,rot=0,offset=[0,0],mirror=false,selectedNodeIds=[],targetBoundary=null,targetToCanvas=null,containScale=null){
    const rooms=Array.isArray(fp?.rooms)?fp.rooms:[];
    const edges=Array.isArray(fp?.edges)?fp.edges:[];
    const bdry=Array.isArray(fp?.boundary)?fp.boundary:[];
    const roomEdges=(retrieval&&typeof retrieval.room_edges==='object'&&retrieval.room_edges)||{};
    if(!rooms.length&&!bdry.length)return;
    // Main-canvas mode: draw each room's connection to the INPUT boundary the
    // user drew (target_edge_index, via targetToCanvas=n2c) instead of the
    // retrieved plan's own boundary. Cards pass no target boundary and keep the
    // plan-relative connectors. n2c output and the node positions (t()) are both
    // logical-canvas px, so they compose.
    const useTarget=Array.isArray(targetBoundary)&&targetBoundary.length>=3&&typeof targetToCanvas==='function';

    // Flagged rooms = union of `rooms` across failing sensor checks (backend-
    // attached retrieval.flags). Defensive throughout: missing/malformed
    // flags just yields an empty set, never throws.
    const flaggedRooms=new Set();
    (Array.isArray(retrieval?.flags?.checks)?retrieval.flags.checks:[]).forEach((chk)=>{
      if(chk&&chk.ok===false&&Array.isArray(chk.rooms))chk.rooms.forEach((rid)=>flaggedRooms.add(String(rid)));
    });
    const pad=0.07;
    const sc=Math.min(W,H)*(1-2*pad),ox=(W-sc)/2,oy=(H-sc)/2;
    const cen=(rot||mirror)?bdryCenter(bdry):null;
    const t=([x,y])=>{
      if(mirror)y=2*cen[1]-y;
      if(rot)[x,y]=rotatePtAround([x,y],rot,cen);
      return [(x+offset[0])*sc+ox,(y+offset[1])*sc+oy];
    };

    // Boundary edge -> type, where known (first connector referencing it wins).
    const bdryEdgeType={};
    Object.values(roomEdges).forEach((conns)=>{
      (Array.isArray(conns)?conns:[]).forEach((cn)=>{
        if(cn&&typeof cn.edge_index==='number'&&!(cn.edge_index in bdryEdgeType))
          bdryEdgeType[cn.edge_index]=cn.edge_type;
      });
    });

    // Boundary outline, faint, coloured by type where known — context for the graph.
    // Skipped in main-canvas (useTarget) mode: the drawn input boundary is already
    // rendered by the boundary overlay, so drawing the plan's own boundary too
    // would be misleading noise.
    if(!useTarget&&bdry.length>=2){
      const n=bdry.length;
      for(let i=0;i<n;i++){
        const a=bdry[i],b=bdry[(i+1)%n];
        if(!a||!b)continue;
        const [ax,ay]=t(a),[bx,by]=t(b);
        const ty2=bdryEdgeType[i];
        c.beginPath();c.moveTo(ax,ay);c.lineTo(bx,by);
        c.strokeStyle=ty2?(EDGE_TYPE_COLOURS[ty2]||'rgba(15,23,42,0.35)'):'rgba(15,23,42,0.25)';
        c.lineWidth=ty2?2.5:1;
        c.setLineDash(ty2?[]:[3,3]);
        c.stroke();
      }
      c.setLineDash([]);
    }

    // Room centroids, keyed by stringified id — room.id may be numeric while
    // retrieval.room_edges keys and plan.edges u/v are (or may be) strings.
    // Nodes use the SAME display transform t() as the room fills (drawFP), so a
    // node always sits on its room. In useTarget mode the caller passes a
    // contain-scale (about the boundary centre) that also scales the fill layer,
    // so nodes stay on their rooms while the whole plan shrinks inside the drawn
    // boundary. Boundary connectors below are NOT scaled — they touch the actual
    // target boundary.
    const scalePt=(p)=>containScale?[
      containScale.pivot[0]+containScale.s*(p[0]-containScale.pivot[0]),
      containScale.pivot[1]+containScale.s*(p[1]-containScale.pivot[1])]:p;
    const roomCen={};
    rooms.forEach((room)=>{
      if(!room||!Array.isArray(room.polygon)||room.polygon.length<3)return;
      const mx=room.polygon.reduce((s,p)=>s+p[0],0)/room.polygon.length;
      const my=room.polygon.reduce((s,p)=>s+p[1],0)/room.polygon.length;
      roomCen[String(room.id)]=useTarget?scalePt(t([mx,my])):t([mx,my]);
    });

    // Room <-> boundary connectors — drawn from the backend room_edges, which the
    // sensors ALSO consume, so a connector and a sensor can never disagree about
    // which edge a room fronts. room_edges is now raycast against the TARGET
    // boundary (each entry carries target_point, the exact hit); drawn only in
    // useTarget mode. Cards (no target boundary) show nodes + room-room edges
    // only — there is no plan-frame room<->boundary structure any more.
    let connectorCount=0,badConnectors=0;
    if(useTarget)Object.entries(roomEdges).forEach(([roomId,conns])=>{
      const pos=roomCen[String(roomId)];
      if(!pos)return;
      (Array.isArray(conns)?conns:[]).forEach((cn)=>{
        let mx,my;
        if(Array.isArray(cn?.target_point)&&cn.target_point.length===2){
          [mx,my]=targetToCanvas(cn.target_point[0],cn.target_point[1]);
        }else{
          const idx=cn&&cn.target_edge_index;
          if(typeof idx!=='number'){badConnectors++;return;}
          const a=targetBoundary[idx],b=targetBoundary[(idx+1)%targetBoundary.length];
          if(!a||!b){badConnectors++;return;}
          [mx,my]=targetToCanvas((a[0]+b[0])/2,(a[1]+b[1])/2);
        }
        connectorCount++;
        c.beginPath();c.moveTo(pos[0],pos[1]);c.lineTo(mx,my);
        c.strokeStyle=EDGE_TYPE_COLOURS[cn.edge_type]||'rgba(15,23,42,0.5)';
        c.lineWidth=1.5;c.setLineDash([2,2]);c.stroke();c.setLineDash([]);
      });
    });

    // Room <-> room adjacency edges, weighted by access: a DOOR between two
    // rooms (edge.connection===1) is the real access relationship the sensors
    // reason over, so it reads solid, dark and thick; a plain shared-wall
    // adjacency (no door) is drawn thin, light and dashed as a weaker relation.
    let drawnEdges=0;
    edges.forEach((edge)=>{
      const a=roomCen[String(edge?.u)],b=roomCen[String(edge?.v)];
      if(!a||!b)return;
      drawnEdges++;
      const door=edge?.connection===1;
      c.beginPath();c.moveTo(a[0],a[1]);c.lineTo(b[0],b[1]);
      c.strokeStyle=door?'rgba(15,15,15,0.85)':'rgba(15,15,15,0.28)';
      c.lineWidth=door?2.5:1;
      c.setLineDash(door?[]:[3,3]);
      c.stroke();
      c.setLineDash([]);
    });

    // Room nodes.
    const nr=Math.max(3,W*0.02);
    let drawnNodes=0;
    rooms.forEach((room)=>{
      const pos=roomCen[String(room?.id)];
      if(!pos)return;
      drawnNodes++;
      const st=room.source_type??room.type;
      const col=RCOLS[st]??'#ccc';
      c.beginPath();c.arc(pos[0],pos[1],nr,0,Math.PI*2);
      c.fillStyle=col;c.fill();
      c.strokeStyle='#222';c.lineWidth=1;c.stroke();
      if(flaggedRooms.has(String(room?.id))){
        c.beginPath();c.arc(pos[0],pos[1],nr+2.5,0,Math.PI*2);
        c.strokeStyle='#dc2626';c.lineWidth=2.5;c.stroke();
      }
      if(selectedNodeIds.includes(String(room?.id))){
        c.beginPath();c.arc(pos[0],pos[1],nr+4,0,Math.PI*2);
        c.strokeStyle='#2563eb';c.lineWidth=3;c.stroke();
      }
      c.save();
      c.font=`${Math.max(7,nr)}px sans-serif`;c.textAlign='center';c.textBaseline='middle';
      c.fillStyle='#111';c.fillText(ROOM_ABBR[st]??'?',pos[0],pos[1]);
      c.restore();
    });

    // Entrance marker (optional). Skipped in useTarget mode: it would draw the
    // plan's OWN entrance via t(), out of frame with the target-placed nodes;
    // the drawn boundary's entrance is already shown by the boundary overlay.
    const ent=fp.entrance;
    if(!useTarget&&Array.isArray(ent)&&ent.length===4){
      const [ex1,ey1]=t([ent[0],ent[1]]);
      const [ex2,ey2]=t([ent[2],ent[3]]);
      c.beginPath();c.arc((ex1+ex2)/2,(ey1+ey2)/2,4,0,Math.PI*2);
      c.fillStyle='#f59e0b';c.strokeStyle='#fff';c.lineWidth=1.5;c.fill();c.stroke();
    }

    if(drawnNodes!==rooms.length)
      console.warn(`drawTopologyGraph: drew ${drawnNodes}/${rooms.length} room nodes (missing/invalid polygon)`);
    if(drawnEdges!==edges.length)
      console.warn(`drawTopologyGraph: drew ${drawnEdges}/${edges.length} adjacency edges (unresolved room id)`);
    if(badConnectors)
      console.warn(`drawTopologyGraph: ${badConnectors}/${connectorCount+badConnectors} room_edges connector(s) had an invalid/out-of-range edge_index`);
    return { nodes: roomCen, nr };
  }

  // Draw the fitted layout graph: mapped nodes inside the target boundary
  // with edges showing the adjacency relationships from the retrieved plan.
  // graph = { nodes: [{id, label, sourceType, mappedPos}], edges: [{u,v,connection}] }
  function drawFitGraph(c,graph){
    const{nodes,edges}=graph;
    const nodeMap={};
    nodes.forEach(n=>{nodeMap[n.id]=n;});

    // Edges first (below nodes)
    edges.forEach(e=>{
      const a=nodeMap[e.u],b=nodeMap[e.v];
      if(!a||!b)return;
      const[ax,ay]=n2c(a.mappedPos[0],a.mappedPos[1]);
      const[bx,by]=n2c(b.mappedPos[0],b.mappedPos[1]);
      c.beginPath();c.moveTo(ax,ay);c.lineTo(bx,by);
      if(e.connection===1){
        c.strokeStyle='rgba(15,15,20,0.80)';c.lineWidth=2;c.setLineDash([]);
      }else{
        c.strokeStyle='rgba(15,15,20,0.35)';c.lineWidth=1.5;c.setLineDash([6,4]);
      }
      c.stroke();c.setLineDash([]);
    });

    // Nodes
    const nr=12;
    nodes.forEach(n=>{
      const[cx,cy]=n2c(n.mappedPos[0],n.mappedPos[1]);
      const col=RCOLS[n.sourceType]??'#ccc';
      if(n.isEntrance){
        // Square: signals this node is a boundary anchor, not a free-floating room
        c.beginPath();c.rect(cx-nr,cy-nr,nr*2,nr*2);
        c.fillStyle=col;c.fill();
        c.strokeStyle='#1e293b';c.lineWidth=2.5;c.stroke();
      }else{
        c.beginPath();c.arc(cx,cy,nr,0,Math.PI*2);
        c.fillStyle=col;c.fill();
        c.strokeStyle='#1e293b';c.lineWidth=2;c.stroke();
      }
      // Label below node
      c.font=`bold 9px monospace`;
      c.textAlign='center';
      c.textBaseline='top';
      c.fillStyle='#1e293b';
      const lbl=(n.label||n.id).replace('Room','').replace('room','').trim();
      c.fillText(lbl,cx,cy+nr+3);
    });
    c.textBaseline='alphabetic';
  }

  function drawDisplacementArrows(c,arrows,fpBdry,rot,offset,mirror){
    if(!arrows.length)return;
    c.save();
    arrows.forEach(({from,to})=>{
      const [fx,fy]=planToCanvas(from[0],from[1],fpBdry,rot,offset,mirror);
      const [tx,ty]=planToCanvas(to[0],to[1],fpBdry,rot,offset,mirror);
      const dx=tx-fx,dy=ty-fy,len=Math.hypot(dx,dy);
      if(len<1)return;
      const ux=dx/len,uy=dy/len;
      const HEAD=10,WING=5;
      c.beginPath();c.moveTo(fx,fy);c.lineTo(tx,ty);
      c.strokeStyle='rgba(239,68,68,0.85)';c.lineWidth=2;c.setLineDash([4,3]);c.stroke();c.setLineDash([]);
      c.beginPath();
      c.moveTo(tx,ty);
      c.lineTo(tx-HEAD*ux+WING*uy,ty-HEAD*uy-WING*ux);
      c.lineTo(tx-HEAD*ux-WING*uy,ty-HEAD*uy+WING*ux);
      c.closePath();
      c.fillStyle='rgba(239,68,68,0.85)';c.fill();
      c.beginPath();c.arc(fx,fy,3,0,Math.PI*2);
      c.fillStyle='rgba(239,68,68,0.65)';c.fill();
    });
    c.restore();
  }

  function drawFitHighlights(c,fp,W,H,rot=0,offset=[0,0],mirror=false){
    const rooms=fp.rooms||[];
    if(!rooms.some((room)=>room.fit_offset))return;
    const pad=0.07;
    const sc=Math.min(W,H)*(1-2*pad),ox=(W-sc)/2,oy=(H-sc)/2;
    const cen=(rot||mirror)?bdryCenter(fp.boundary||[]):null;
    const t=([x,y])=>{
      // y-flip, not x-flip — matches geo_ent_num_retrieval.retrieve()'s
      // actual mirror convention (conj() = angle negation). See 2026-08-11.
      if(mirror)y=2*cen[1]-y;
      if(rot)[x,y]=rotatePtAround([x,y],rot,cen);
      return [(x+offset[0])*sc+ox,(y+offset[1])*sc+oy];
    };
    rooms.forEach((room)=>{
      if(!room.fit_offset||!room.polygon||room.polygon.length<3)return;
      c.beginPath();
      room.polygon.forEach((pt,i)=>{const [cx,cy]=t(pt);i===0?c.moveTo(cx,cy):c.lineTo(cx,cy);});
      c.closePath();
      c.strokeStyle='rgba(34,197,94,0.9)';
      c.lineWidth=2;c.setLineDash([4,2]);c.stroke();c.setLineDash([]);
    });
  }

  return {
    drawDoorsWindows,
    drawFP,
    drawGraph,
    drawTopologyGraph,
    renderFloorPlate,
    render,
    updateCursor,
    getGraphNodes: () => lastGraphNodes,
  };
}