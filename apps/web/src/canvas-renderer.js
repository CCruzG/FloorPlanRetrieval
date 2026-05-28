import { COMMON_AREA_TYPES, METRES_PER_UNIT, RCOLS, VR } from './constants.js';
import { bdryCenter, rotatePtAround } from './geometry.js';

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
    const { dragState, hoveredEntrance, hoveredRoom, selected, closed }=getState();
    if(dragState){
      canvas.style.cursor=dragState.mode==='entrance'||dragState.mode==='move'?'grabbing':(dragState.edgeAxis==='h'?'ns-resize':'ew-resize');
    }else if(hoveredEntrance){
      canvas.style.cursor='grab';
    }else if(hoveredRoom){
      canvas.style.cursor=hoveredRoom.mode==='body'?'grab':(hoveredRoom.edgeAxis==='h'?'ns-resize':'ew-resize');
    }else{
      canvas.style.cursor=selected!==null&&closed?'default':'crosshair';
    }
  }

  function render(){
    const {
      boundary,
      closed,
      entrance,
      entranceMode,
      entranceSnap,
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
      displacementArrows,
      cW,
      cH,
      params,
      floorPlateState,
      devMode,
      viewport,
    }=getState();

    const vpZoom=viewport?.zoom??1;
    const vpPanX=viewport?.panX??0;
    const vpPanY=viewport?.panY??0;

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
      const selRot=results[selected].rot||0;
      const selMirror=results[selected].mirror||false;
      const rawPlan=results[selected].fp;
      const fp=(fitMode&&fittedCache[selected]?.fp)||rawPlan;
      const qc=bdryCenter(boundary.map(({x,y})=>[x,y]));
      const pc=bdryCenter(fp.boundary||[]);
      const selOff=[qc[0]-pc[0],qc[1]-pc[1]];
      drawFP(ctx,fp,cW,cH,false,results[selected].adapted,selRot,selOff,selMirror,
        {showBdry:showPlanBdry,showCommon:showCommonArea,hiliteRoom:selectedRoom,hoverEdge:true});
      drawDoorsWindows(ctx,fp,cW,cH,selRot,selOff,selMirror,{showWin:showWindows,showIntDoors:showDoors});
      if(params.showGraph)drawGraph(ctx,fp,cW,cH,selRot,selOff,selMirror);
      if(fitMode&&fittedCache[selected]?.fp)drawFitHighlights(ctx,fp,cW,cH,selRot,selOff,selMirror);
      if(displacementArrows.length)drawDisplacementArrows(ctx,displacementArrows,fp.boundary||[],selRot,selOff,selMirror);
      drawBoundaryOverlay();

      const ent=fp.entrance;
      if(ent&&ent.length===4){
        const cen=bdryCenter(fp.boundary||[]);
        let [ex1,ey1]=[ent[0],ent[1]];
        let [ex2,ey2]=[ent[2],ent[3]];
        if(selMirror){ex1=2*cen[0]-ex1;ex2=2*cen[0]-ex2;}
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
      ctx.beginPath();
      boundary.forEach(({x,y},i)=>{
        const [cx,cy]=n2c(x,y);
        i===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);
      });
      if(closed)ctx.closePath();
      ctx.strokeStyle='#1e40af';ctx.lineWidth=2;ctx.setLineDash([]);ctx.stroke();

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
      (floorPlate.units||[]).forEach((unit)=>{
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
        const area=polygonArea(poly);
        c.fillStyle='#0f766e';
        c.fillText(`${area.toFixed(1)} m2`,ux,uy+10);
      });
    }

    if(showCommonAreas){
      (floorPlate.unresolved_regions||[]).forEach((region)=>{
        drawPoly(region,{fill:'rgba(239,68,68,0.22)',stroke:'rgba(185,28,28,0.9)',lineWidth:1.5,dash:[4,3]});
      });
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

  function drawBoundaryOverlay(){
    const { boundary, closed }=getState();
    if(!boundary.length)return;
    if(boundary.length>=3&&closed){
      ctx.beginPath();
      boundary.forEach(({x,y},i)=>{
        const [cx,cy]=n2c(x,y);
        i===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);
      });
      ctx.closePath();
      ctx.fillStyle='rgba(0,0,0,0)';ctx.fill();
    }
    if(boundary.length>=2){
      ctx.beginPath();
      boundary.forEach(({x,y},i)=>{
        const [cx,cy]=n2c(x,y);
        i===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);
      });
      if(closed)ctx.closePath();
      ctx.strokeStyle='#3b82f6';ctx.lineWidth=2;
      ctx.setLineDash([6,4]);ctx.stroke();ctx.setLineDash([]);
    }
    boundary.forEach(({x,y},i)=>{
      const [cx,cy]=n2c(x,y);
      ctx.beginPath();ctx.arc(cx,cy,4,0,Math.PI*2);
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
      if(mirror)x=2*cen[0]-x;
      if(rot)[x,y]=rotatePtAround([x,y],rot,cen);
      return [(x+offset[0])*sc+ox,(y+offset[1])*sc+oy];
    };
    rooms.forEach((room,ri)=>{
      const st=room.source_type??room.type;
      if(!showCommon&&COMMON_AREA_TYPES.has(st))return;
      if(!room.polygon||room.polygon.length<3)return;
      const col=RCOLS[st]??'#ddd';
      c.beginPath();
      room.polygon.forEach((pt,i)=>{const [cx,cy]=t(pt);i===0?c.moveTo(cx,cy):c.lineTo(cx,cy);});
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
      if(mirror)x=2*cen[0]-x;
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
      if(mirror)x=2*rotCen[0]-x;
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
      if(mirror)x=2*cen[0]-x;
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
    renderFloorPlate,
    render,
    updateCursor,
  };
}