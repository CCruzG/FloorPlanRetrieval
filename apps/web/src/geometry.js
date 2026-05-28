import { PAD, SNAP_PX } from './constants.js';

export function n2c(nx,ny,{cW,cH,pad=PAD}){
  const s=cW*(1-2*pad);
  return [pad*cW+nx*s,pad*cH+ny*s];
}

export function c2n(cx,cy,{cW,cH,pad=PAD}){
  const s=cW*(1-2*pad);
  return [(cx-pad*cW)/s,(cy-pad*cH)/s];
}

export function pdist(a,b,canvasState){
  const [ax,ay]=n2c(a.x,a.y,canvasState);
  const [bx,by]=n2c(b.x,b.y,canvasState);
  return Math.hypot(ax-bx,ay-by);
}

export function rotatePtAround([x,y],angle,[cx,cy]){
  const cos=Math.cos(angle),sin=Math.sin(angle);
  const dx=x-cx,dy=y-cy;
  return [cx+dx*cos-dy*sin,cy+dx*sin+dy*cos];
}

export function bdryCenter(bdry){
  if(!bdry.length)return [0.5,0.5];
  let area=0,cx=0,cy=0;
  const n=bdry.length;
  for(let i=0;i<n;i++){
    const [x0,y0]=bdry[i];
    const [x1,y1]=bdry[(i+1)%n];
    const cross=x0*y1-x1*y0;
    area+=cross;
    cx+=(x0+x1)*cross;
    cy+=(y0+y1)*cross;
  }
  area/=2;
  if(Math.abs(area)<1e-10){
    let sx=0,sy=0;
    bdry.forEach((v)=>{sx+=v[0];sy+=v[1];});
    return [sx/n,sy/n];
  }
  return [cx/(6*area),cy/(6*area)];
}

export function planToCanvas(px,py,fpBdry,rot,offset,mirror,{cW,cH,pad=PAD}){
  const sc=cW*(1-2*pad),ox=pad*cW,oy=pad*cH;
  const cen=(rot||mirror)?bdryCenter(fpBdry):null;
  let x=px,y=py;
  if(mirror)x=2*cen[0]-x;
  if(rot)[x,y]=rotatePtAround([x,y],rot,cen);
  return [(x+offset[0])*sc+ox,(y+offset[1])*sc+oy];
}

export function canvasToPlan(cx,cy,fpBdry,rot,offset,mirror,{cW,cH,pad=PAD}){
  const sc=cW*(1-2*pad),ox=pad*cW,oy=pad*cH;
  const cen=(rot||mirror)?bdryCenter(fpBdry):null;
  let x=(cx-ox)/sc-offset[0];
  let y=(cy-oy)/sc-offset[1];
  if(rot)[x,y]=rotatePtAround([x,y],-rot,cen);
  if(mirror)x=2*cen[0]-x;
  return [x,y];
}

export function ptInPoly([px,py],poly){
  let inside=false;
  const n=poly.length;
  for(let i=0,j=n-1;i<n;j=i++){
    const [xi,yi]=poly[i],[xj,yj]=poly[j];
    if((yi>py)!==(yj>py)&&px<(xj-xi)*(py-yi)/(yj-yi)+xi)inside=!inside;
  }
  return inside;
}

export function ptToSegDistSq([px,py],[ax,ay],[bx,by]){
  const dx=bx-ax,dy=by-ay,lenSq=dx*dx+dy*dy;
  if(lenSq===0)return(px-ax)**2+(py-ay)**2;
  const t=Math.max(0,Math.min(1,((px-ax)*dx+(py-ay)*dy)/lenSq));
  return (px-(ax+t*dx))**2+(py-(ay+t*dy))**2;
}

export function projSnap(nx,ny,{boundary,closed,cW,cH,pad=PAD,snapPx=SNAP_PX}){
  if(!boundary.length||closed)return {x:nx,y:ny};
  const s=cW*(1-2*pad);
  const thresh=snapPx/s;
  let sx=nx,sy=ny,bestX=Infinity,bestY=Infinity;
  boundary.forEach(({x,y})=>{
    const dx=Math.abs(nx-x),dy=Math.abs(ny-y);
    if(dx<thresh&&dx<bestX){bestX=dx;sx=x;}
    if(dy<thresh&&dy<bestY){bestY=dy;sy=y;}
  });
  return {x:sx,y:sy};
}

export function snapToEdge(nx,ny,{boundary,closed,normVerts=null}){
  if(!boundary.length||!closed)return null;
  const verts=normVerts||boundary.map((v)=>[v.x,v.y]);
  const n=verts.length;
  const cen=bdryCenter(verts);
  let best=Infinity,result=null;
  for(let i=0;i<n;i++){
    const [ax,ay]=verts[i],[bx,by]=verts[(i+1)%n];
    const ex=bx-ax,ey=by-ay;
    const segL=Math.hypot(ex,ey);
    if(segL<1e-9)continue;
    const t=Math.max(0,Math.min(1,((nx-ax)*ex+(ny-ay)*ey)/(segL*segL)));
    const px=ax+t*ex,py=ay+t*ey;
    const d=Math.hypot(nx-px,ny-py);
    if(d<best){
      best=d;
      let normX=ey/segL,normY=-ex/segL;
      if(normX*(px-cen[0])+normY*(py-cen[1])<0){normX=-normX;normY=-normY;}
      result={px,py,normX,normY,edge:i,dist:d};
    }
  }
  return result;
}

export function samplePerimeter(verts,n){
  const pts=[...verts,verts[0]];
  const lens=[];
  for(let i=0;i<pts.length-1;i++){
    const dx=pts[i+1][0]-pts[i][0],dy=pts[i+1][1]-pts[i][1];
    lens.push(Math.hypot(dx,dy));
  }
  const cum=[0];
  lens.forEach((l)=>cum.push(cum[cum.length-1]+l));
  const tot=cum[cum.length-1];
  if(tot<1e-9)return Array(n).fill([0,0]);
  const res=[];
  for(let i=0;i<n;i++){
    const pos=(i/n)*tot;
    let idx=cum.findIndex((c)=>c>pos)-1;
    idx=Math.max(0,Math.min(idx,lens.length-1));
    const t=(pos-cum[idx])/(lens[idx]+1e-12);
    res.push([
      pts[idx][0]+t*(pts[idx+1][0]-pts[idx][0]),
      pts[idx][1]+t*(pts[idx+1][1]-pts[idx][1]),
    ]);
  }
  return res;
}

export function polyArea(segs){
  let a=0;
  for(let i=0;i<segs.length;i++){
    const j=(i+1)%segs.length;
    a+=segs[i][0]*segs[j][1]-segs[j][0]*segs[i][1];
  }
  return Math.abs(a)/2;
}

export function ghost({boundary,closed,mouse,shiftDown,snapFirst,cW,cH,pad=PAD,snapPx=SNAP_PX}){
  if(!boundary.length||closed)return null;
  let {x,y}=mouse;
  if(shiftDown){
    const last=boundary[boundary.length-1];
    const dx=Math.abs(x-last.x),dy=Math.abs(y-last.y);
    if(dx>dy)y=last.y;else x=last.x;
  }else{
    const snapped=projSnap(x,y,{boundary,closed,cW,cH,pad,snapPx});
    x=snapped.x;
    y=snapped.y;
  }
  if(snapFirst){x=boundary[0].x;y=boundary[0].y;}
  return {x,y};
}