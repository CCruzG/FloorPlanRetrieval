import { N_FREQS, N_SAMP, ROOM_GROUPS, SCALE } from './constants.js';
import { polyArea, samplePerimeter, snapToEdge } from './geometry.js';

export function entranceDescJS(bdry,ent){
  if(!ent||!bdry.length)return new Float32Array(4);
  let sigArea=0;
  for(let i=0;i<bdry.length;i++){
    const v0=bdry[i],v1=bdry[(i+1)%bdry.length];
    sigArea+=v0.x*v1.y-v1.x*v0.y;
  }
  const rawVerts=bdry.map((v)=>[v.x,v.y]);
  const verts=sigArea>0?[...rawVerts].reverse():rawVerts;
  const pts=samplePerimeter(verts,N_SAMP);
  const cx=pts.reduce((sum,p)=>sum+p[0],0)/pts.length;
  const cy=pts.reduce((sum,p)=>sum+p[1],0)/pts.length;
  const meanRadius=pts.reduce((sum,p)=>sum+Math.hypot(p[0]-cx,p[1]-cy),0)/pts.length;
  if(meanRadius<1e-9)return new Float32Array(4);
  const snap=snapToEdge(ent.x,ent.y,{boundary:bdry,closed:true,normVerts:verts});
  if(!snap)return new Float32Array(4);
  return new Float32Array([(snap.px-cx)/meanRadius,(snap.py-cy)/meanRadius,snap.normX,snap.normY]);
}

export function roomCountDescFromReqs(req){
  const bedrooms=req.bedrooms||0,bathrooms=req.bathrooms||0;
  const living=1,master=bedrooms>0?1:0,kitchen=1,dining=1;
  const child=Math.max(0,bedrooms-1);
  const total=living+master+kitchen+dining+child+bathrooms;
  return new Float32Array([
    Math.min(living/2,1),
    Math.min(master/1,1),
    Math.min(kitchen/1,1),
    Math.min(bathrooms/3,1),
    Math.min(dining/1,1),
    Math.min(child/2,1),
    0,0,0,0,
    Math.min(total/15,1),
  ]);
}

export function shapeDesc(bdry){
  const verts=bdry.map((v)=>[v[0]*SCALE,v[1]*SCALE]);
  if(verts.length<3)return new Float32Array(N_FREQS);
  const pts=samplePerimeter(verts,N_SAMP);
  const cx=pts.reduce((sum,p)=>sum+p[0],0)/pts.length;
  const cy=pts.reduce((sum,p)=>sum+p[1],0)/pts.length;
  const distances=pts.map((p)=>Math.hypot(p[0]-cx,p[1]-cy));
  const meanDistance=distances.reduce((sum,value)=>sum+value,0)/distances.length;
  if(meanDistance<1e-9)return new Float32Array(N_FREQS);
  const normalized=distances.map((value)=>value/meanDistance);
  const magnitudes=new Float32Array(N_FREQS);
  for(let k=0;k<N_FREQS;k++){
    let re=0,im=0;
    for(let t=0;t<N_SAMP;t++){
      const angle=2*Math.PI*(k+1)*t/N_SAMP;
      re+=normalized[t]*Math.cos(angle);
      im-=normalized[t]*Math.sin(angle);
    }
    magnitudes[k]=Math.sqrt(re*re+im*im);
  }
  let l2=0;
  for(let i=0;i<N_FREQS;i++)l2+=magnitudes[i]*magnitudes[i];
  l2=Math.sqrt(l2);
  if(l2>1e-9){
    for(let i=0;i<N_FREQS;i++)magnitudes[i]/=l2;
  }
  return magnitudes;
}

export function semDesc(fp,smw){
  const rooms=fp.rooms||[],bdry=fp.boundary||[];
  const groupCounts={},groupAreas={};
  ROOM_GROUPS.forEach((group)=>{groupCounts[group.name]=0;groupAreas[group.name]=0;});
  rooms.forEach((room)=>{
    const st=room.source_type??room.type;
    const poly=(room.polygon||[]).map(([x,y])=>[x*SCALE,y*SCALE]);
    ROOM_GROUPS.forEach((group)=>{
      if(group.set.has(st)){
        groupCounts[group.name]++;
        groupAreas[group.name]+=polyArea(poly);
      }
    });
  });
  const feat=[];
  ROOM_GROUPS.forEach((group)=>{
    feat.push(Math.min(groupCounts[group.name]/group.max,1));
    feat.push(Math.min(groupAreas[group.name]/(SCALE*SCALE),1));
  });
  const boundaryVerts=bdry.map((v)=>[v[0]*SCALE,v[1]*SCALE]);
  feat.push(Math.min(rooms.length/15,1));
  feat.push(Math.min(boundaryVerts.length>=3?polyArea(boundaryVerts):0,SCALE*SCALE)/(SCALE*SCALE));
  return new Float32Array(feat.map((value)=>value*smw));
}

export function combinedDesc(fp,sw,smw){
  const shape=shapeDesc(fp.boundary||[],sw);
  const semantic=semDesc(fp,smw);
  const out=new Float32Array(shape.length+semantic.length);
  out.set(shape);
  out.set(semantic,shape.length);
  return out;
}

export function l2dist(a,b,off,dim){
  let sum=0;
  for(let i=0;i<dim;i++){
    const delta=a[i]-b[off+i];
    sum+=delta*delta;
  }
  return Math.sqrt(sum);
}