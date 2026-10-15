import {
  BALCONY_FLOOR_THICKNESS,
  BALCONY_OPENING_DOOR_HEIGHT,
  BALCONY_PARAPET_HEIGHT,
  BALCONY_PARAPET_THICKNESS,
  BALCONY_TYPE_ID,
  DOOR_HEIGHT,
  DOOR_THICKNESS,
  OPACITY_MASK_COLORS,
  ROOM_HEIGHT,
  WINDOW_BALCONY_EDGE_TOL,
  WINDOW_HEIGHT,
  WINDOW_SILL_HEIGHT,
  WINDOW_THICKNESS,
  WORLD_SCALE,
} from './constants.js';
import { bdryCenter, rotatePtAround } from './geometry.js';

let threeDepsPromise=null;

export function createViewer3D({
  viewer3D,
  renderCaptureState,
  setViewer3DStatus,
  refreshStructuredPromptPreview,
  updateCapturePanel,
  getActiveFPState,
  getSelectionState,
  getActiveTab,
}) {
  function getCameraPerspectivePrompt(view){
    const normalizedView=(view||'exterior').trim().toLowerCase();
    const fallback=normalizedView==='interior'
      ?'eye-level interior two-point perspective'
      :'eye-level two-point perspective exterior view';

    if(!viewer3D.initialized||!viewer3D.camera)return fallback;

    const camera=viewer3D.camera;
    const target=viewer3D.controls?.target||new THREE.Vector3(0,0,0);
    const dy=camera.position.y-target.y;
    const dx=camera.position.x-target.x;
    const dz=camera.position.z-target.z;
    const horizontal=Math.max(1e-4,Math.hypot(dx,dz));
    const pitchDeg=Math.atan2(dy,horizontal)*(180/Math.PI);

    if(normalizedView==='interior'){
      if(dy<1.8&&Math.abs(pitchDeg)<=18)return 'eye-level interior two-point perspective';
      if(dy<3.8)return 'slightly elevated interior perspective';
      return 'elevated interior overview perspective';
    }

    if(dy<1.8&&Math.abs(pitchDeg)<=18)return 'eye-level two-point perspective exterior view';
    if(dy<4.5&&pitchDeg>=0)return 'slightly elevated exterior perspective view';
    if(dy<9.0&&pitchDeg>=8)return 'elevated exterior view';
    if(dy>=9.0)return 'birds-eye exterior oblique view';
    if(pitchDeg<-8)return 'low camera upward exterior perspective';
    return 'architectural exterior perspective view';
  }

  function captureCanvasImageData(srcCanvas){
    const w=srcCanvas.width,h=srcCanvas.height;
    const out=document.createElement('canvas');
    out.width=w;out.height=h;
    const octx=out.getContext('2d');
    octx.drawImage(srcCanvas,0,0);
    return octx.getImageData(0,0,w,h);
  }

  function imageDataToDataUrl(imgData){
    const out=document.createElement('canvas');
    out.width=imgData.width;
    out.height=imgData.height;
    const octx=out.getContext('2d');
    octx.putImageData(imgData,0,0);
    return out.toDataURL('image/png');
  }

  function getRequestedCaptureSize(){
    const rawWidth=parseInt(document.getElementById('py-width')?.value||'1024',10);
    const rawHeight=parseInt(document.getElementById('py-height')?.value||'1024',10);
    const width=Math.max(256,Math.min(2048,Number.isFinite(rawWidth)?rawWidth:1024));
    const height=Math.max(256,Math.min(2048,Number.isFinite(rawHeight)?rawHeight:1024));
    return {width,height};
  }

  function computeCenteredAspectRect(totalWidth,totalHeight,targetAspect){
    const safeW=Math.max(1,Math.floor(totalWidth));
    const safeH=Math.max(1,Math.floor(totalHeight));
    const safeAspect=Math.max(1e-6,targetAspect||1);
    let width=safeW;
    let height=Math.round(width/safeAspect);
    if(height>safeH){
      height=safeH;
      width=Math.round(height*safeAspect);
    }
    const x=Math.floor((safeW-width)/2);
    const y=Math.floor((safeH-height)/2);
    return {x,y,width,height};
  }

  function cropAndScaleImageData(srcImg,frameRect,targetWidth,targetHeight){
    const srcCanvas=document.createElement('canvas');
    srcCanvas.width=srcImg.width;
    srcCanvas.height=srcImg.height;
    const sctx=srcCanvas.getContext('2d');
    sctx.putImageData(srcImg,0,0);

    const outCanvas=document.createElement('canvas');
    outCanvas.width=targetWidth;
    outCanvas.height=targetHeight;
    const octx=outCanvas.getContext('2d');
    octx.imageSmoothingEnabled=true;
    octx.imageSmoothingQuality='high';
    octx.drawImage(
      srcCanvas,
      frameRect.x,
      frameRect.y,
      frameRect.width,
      frameRect.height,
      0,
      0,
      targetWidth,
      targetHeight,
    );
    return octx.getImageData(0,0,targetWidth,targetHeight);
  }

  function updateCaptureFrameGuide(){
    const mount=document.getElementById('viewer-3d');
    const guide=document.getElementById('capture-frame-guide');
    const frameBox=document.getElementById('capture-frame-box');
    const frameLabel=document.getElementById('capture-frame-label');
    const toggle=document.getElementById('toggle-frame-guide');
    if(!mount||!guide||!frameBox||!frameLabel)return;

    const {width,height}=getRequestedCaptureSize();
    const rect=computeCenteredAspectRect(mount.clientWidth,mount.clientHeight,width/Math.max(1,height));
    guide.style.display=(toggle?.checked??true)?'block':'none';
    frameBox.style.left=`${rect.x}px`;
    frameBox.style.top=`${rect.y}px`;
    frameBox.style.width=`${rect.width}px`;
    frameBox.style.height=`${rect.height}px`;
    frameLabel.textContent=`frame ${width}x${height}`;

    layoutRenderPanelInShadedRegion(mount,rect);
    layoutCapturePanelInShadedRegion(mount,rect);
  }

  function layoutRenderPanelInShadedRegion(mount,frameRect){
    const panel=document.getElementById('render-panel');
    if(!panel||!mount)return;

    const toolbarHeight=36;
    const pad=10;
    const panelWidth=Math.ceil(panel.offsetWidth||360);
    const panelHeight=Math.ceil(panel.offsetHeight||220);
    const viewerW=Math.max(1,Math.floor(mount.clientWidth));
    const viewerH=Math.max(1,Math.floor(mount.clientHeight));

    const leftShade=frameRect.x-pad;
    const rightShade=viewerW-(frameRect.x+frameRect.width)-pad;
    const topShade=frameRect.y-pad;
    const bottomShade=viewerH-(frameRect.y+frameRect.height)-pad;

    panel.style.left='';
    panel.style.right='';

    if(rightShade>=panelWidth){
      panel.style.right=`${pad}px`;
      panel.style.top=`${toolbarHeight+pad}px`;
      return;
    }

    if(leftShade>=panelWidth){
      panel.style.left=`${pad}px`;
      panel.style.top=`${toolbarHeight+pad}px`;
      return;
    }

    if(topShade>=panelHeight){
      panel.style.right=`${pad}px`;
      panel.style.top=`${toolbarHeight+pad}px`;
      return;
    }

    if(bottomShade>=panelHeight){
      panel.style.right=`${pad}px`;
      panel.style.top=`${toolbarHeight+frameRect.y+frameRect.height+pad}px`;
      return;
    }

    panel.style.right=`${pad}px`;
    panel.style.top=`${toolbarHeight+pad}px`;
  }

  function layoutCapturePanelInShadedRegion(mount,frameRect){
    const panel=document.getElementById('capture-panel');
    if(!panel||!mount)return;

    const toolbarHeight=36;
    const pad=10;
    const panelWidth=Math.ceil(panel.offsetWidth||220);
    const panelHeight=Math.ceil(panel.offsetHeight||220);
    const viewerW=Math.max(1,Math.floor(mount.clientWidth));
    const viewerH=Math.max(1,Math.floor(mount.clientHeight));

    const leftShade=frameRect.x-pad;
    const rightShade=viewerW-(frameRect.x+frameRect.width)-pad;
    const topShade=frameRect.y-pad;
    const bottomShade=viewerH-(frameRect.y+frameRect.height)-pad;

    panel.style.left='';
    panel.style.right='';

    if(leftShade>=panelWidth){
      panel.style.left=`${pad}px`;
      panel.style.top=`${toolbarHeight+pad}px`;
      return;
    }

    if(rightShade>=panelWidth){
      panel.style.right=`${pad}px`;
      panel.style.top=`${toolbarHeight+pad}px`;
      return;
    }

    if(topShade>=panelHeight){
      panel.style.left=`${pad}px`;
      panel.style.top=`${toolbarHeight+pad}px`;
      return;
    }

    if(bottomShade>=panelHeight){
      panel.style.left=`${pad}px`;
      panel.style.top=`${toolbarHeight+frameRect.y+frameRect.height+pad}px`;
      return;
    }

    panel.style.left=`${pad}px`;
    panel.style.top=`${toolbarHeight+pad}px`;
  }

  function makeBinaryMaskDataUrl(maskImg){
    const out=new ImageData(new Uint8ClampedArray(maskImg.data),maskImg.width,maskImg.height);
    const d=out.data;
    for(let i=0;i<d.length;i+=4){
      const v=d[i]>20?255:0;
      d[i]=v;d[i+1]=v;d[i+2]=v;d[i+3]=255;
    }
    return imageDataToDataUrl(out);
  }

  function decodePackedDepth01(r,g,b,a){
    return r/255;
  }

  function depth01ToViewZ(depth01,near,far){
    const z=depth01*2-1;
    return (2*near*far)/Math.max(1e-6,(far+near - z*(far-near)));
  }

  function makeDepthDataUrlFromImageData(depthImg,maskImg,near,far){
    const out=new ImageData(new Uint8ClampedArray(depthImg.data),depthImg.width,depthImg.height);
    const src=depthImg.data;
    const d=out.data;
    const m=maskImg?.data||null;

    const vals=[];
    let count=0;
    for(let i=0;i<src.length;i+=4){
      const inside=m?m[i]>20:true;
      if(!inside)continue;
      const depth01=Math.max(0,Math.min(1,decodePackedDepth01(src[i],src[i+1],src[i+2],src[i+3])));
      const viewZ=depth01ToViewZ(depth01,near,far);
      vals.push(viewZ);
      count++;
    }

    if(count===0){
      for(let i=0;i<d.length;i+=4){
        d[i]=255;d[i+1]=255;d[i+2]=255;d[i+3]=255;
      }
      return imageDataToDataUrl(out);
    }

    vals.sort((a,b)=>a-b);
    const lo=vals[Math.max(0,Math.floor(0.02*vals.length))];
    const hi=Math.max(lo+1e-6,vals[Math.max(0,Math.floor(0.98*vals.length))]);
    const gamma=0.75;

    for(let i=0;i<src.length;i+=4){
      const inside=m?m[i]>20:true;
      if(!inside){
        d[i]=255;d[i+1]=255;d[i+2]=255;d[i+3]=255;
        continue;
      }
      const depth01=Math.max(0,Math.min(1,decodePackedDepth01(src[i],src[i+1],src[i+2],src[i+3])));
      const viewZ=depth01ToViewZ(depth01,near,far);
      const nv=Math.max(0,Math.min(1,(viewZ-lo)/(hi-lo)));
      const v=Math.round(255*(1-Math.pow(nv,gamma)));
      d[i]=v;d[i+1]=v;d[i+2]=v;d[i+3]=255;
    }

    const blurred=new Uint8ClampedArray(d);
    const w=out.width,h=out.height;
    const idx=(x,y)=>((y*w+x)*4);
    for(let y=1;y<h-1;y++){
      for(let x=1;x<w-1;x++){
        const center=idx(x,y);
        const inside=m?m[center]>20:true;
        if(!inside)continue;
        let sum=0,samples=0;
        for(let oy=-1;oy<=1;oy++){
          for(let ox=-1;ox<=1;ox++){
            const j=idx(x+ox,y+oy);
            if(m&&m[j]<=20)continue;
            sum+=d[j];
            samples++;
          }
        }
        if(samples>0){
          const v=Math.round(sum/samples);
          blurred[center]=v;blurred[center+1]=v;blurred[center+2]=v;
        }
      }
    }
    out.data.set(blurred);
    return imageDataToDataUrl(out);
  }

  function withCaptureVisibility(fn){
    const touched=[];
    const hide=(obj)=>{
      if(obj&&obj.visible){
        touched.push(obj);
        obj.visible=false;
      }
    };

    hide(viewer3D.grid);
    if(viewer3D.planGroup){
      viewer3D.planGroup.traverse((obj)=>{
        if(obj?.userData?.captureRole==='ground')hide(obj);
      });
    }

    try{return fn();}
    finally{touched.forEach((obj)=>{obj.visible=true;});}
  }

  function captureMaskImageData(scene,camera,renderer){
    const prevOverride=scene.overrideMaterial;
    const prevBg=scene.background;
    const prevClr=renderer.getClearColor(new THREE.Color()).clone();
    const prevAlpha=renderer.getClearAlpha();

    const maskMat=new THREE.MeshBasicMaterial({color:0xffffff,side:THREE.DoubleSide});
    scene.overrideMaterial=maskMat;
    scene.background=new THREE.Color(0x000000);
    renderer.setClearColor(0x000000,1);
    renderer.render(scene,camera);
    const maskImg=captureCanvasImageData(renderer.domElement);

    scene.overrideMaterial=prevOverride||null;
    scene.background=prevBg;
    renderer.setClearColor(prevClr,prevAlpha);
    maskMat.dispose();
    return maskImg;
  }

  function captureNormalImageData(scene,camera,renderer){
    const prevOverride=scene.overrideMaterial;
    const prevBg=scene.background;
    const prevClr=renderer.getClearColor(new THREE.Color()).clone();
    const prevAlpha=renderer.getClearAlpha();

    const normalMat=new THREE.MeshNormalMaterial({side:THREE.DoubleSide});
    scene.overrideMaterial=normalMat;
    scene.background=new THREE.Color(0x000000);
    renderer.setClearColor(0x000000,1);
    renderer.render(scene,camera);
    const normalImg=captureCanvasImageData(renderer.domElement);

    scene.overrideMaterial=prevOverride||null;
    scene.background=prevBg;
    renderer.setClearColor(prevClr,prevAlpha);
    normalMat.dispose();
    return normalImg;
  }

  function makeNormalDataUrlFromImageData(normalImg,maskImg){
    const out=new ImageData(new Uint8ClampedArray(normalImg.data),normalImg.width,normalImg.height);
    const d=out.data;
    const m=maskImg?.data||null;
    for(let i=0;i<d.length;i+=4){
      const inside=m?m[i]>20:true;
      if(!inside){
        d[i]=0;d[i+1]=0;d[i+2]=0;d[i+3]=255;
        continue;
      }
      d[i+3]=255;
    }
    return imageDataToDataUrl(out);
  }

  function captureOpacityMaskImageData(scene,camera,renderer){
    const prevOverride=scene.overrideMaterial;
    const prevBg=scene.background;
    const prevClr=renderer.getClearColor(new THREE.Color()).clone();
    const prevAlpha=renderer.getClearAlpha();
    const touched=[];

    if(viewer3D.planGroup){
      viewer3D.planGroup.traverse((obj)=>{
        if(!obj||!obj.isMesh)return;
        const role=String(obj?.userData?.captureRole||'').toLowerCase();
        touched.push({obj,visible:obj.visible,material:obj.material});
        if(!role||!Object.prototype.hasOwnProperty.call(OPACITY_MASK_COLORS,role)){
          obj.visible=false;
          return;
        }
        obj.visible=true;
        obj.material=new THREE.MeshBasicMaterial({
          color:OPACITY_MASK_COLORS[role],
          side:THREE.DoubleSide,
        });
      });
    }

    scene.background=new THREE.Color(0x000000);
    renderer.setClearColor(0x000000,1);
    scene.overrideMaterial=null;
    renderer.render(scene,camera);
    const maskImg=captureCanvasImageData(renderer.domElement);

    touched.forEach(({obj,visible,material})=>{
      if(obj.material&&obj.material!==material){
        if(Array.isArray(obj.material))obj.material.forEach((m)=>m&&m.dispose&&m.dispose());
        else if(obj.material.dispose)obj.material.dispose();
      }
      obj.material=material;
      obj.visible=visible;
    });

    scene.background=prevBg;
    renderer.setClearColor(prevClr,prevAlpha);
    scene.overrideMaterial=prevOverride||null;

    const out=new ImageData(new Uint8ClampedArray(maskImg.data),maskImg.width,maskImg.height);
    const d=out.data;
    for(let i=0;i<d.length;i+=4)d[i+3]=255;
    return out;
  }

  function loadScriptOnce(src){
    return new Promise((resolve,reject)=>{
      const existing=document.querySelector(`script[src="${src}"]`);
      if(existing){
        const isThreeCore=src.includes('three.min.js')&&window.THREE;
        const isOrbit=src.includes('OrbitControls.js')&&window.THREE&&window.THREE.OrbitControls;
        if(existing.dataset.loaded==='1'||isThreeCore||isOrbit)return resolve();
        existing.addEventListener('load',()=>resolve(),{once:true});
        existing.addEventListener('error',()=>reject(new Error(`failed to load ${src}`)),{once:true});
        return;
      }
      const script=document.createElement('script');
      script.src=src;
      script.async=true;
      script.onload=()=>{script.dataset.loaded='1';resolve();};
      script.onerror=()=>reject(new Error(`failed to load ${src}`));
      document.head.appendChild(script);
    });
  }

  function ensureThreeDeps(){
    if(window.THREE&&window.THREE.OrbitControls)return Promise.resolve();
    if(threeDepsPromise)return threeDepsPromise;
    threeDepsPromise=loadScriptOnce('https://unpkg.com/three@0.128.0/build/three.min.js')
      .then(()=>loadScriptOnce('https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js'))
      .catch((err)=>{
        threeDepsPromise=null;
        throw err;
      });
    return threeDepsPromise;
  }

  function buildPositiveGrid(size=10,step=0.5){
    const verts=[];
    for(let p=0;p<=size+1e-9;p+=step){
      verts.push(0,0,p,size,0,p);
      verts.push(p,0,0,p,0,size);
    }
    const geom=new THREE.BufferGeometry();
    geom.setAttribute('position',new THREE.Float32BufferAttribute(verts,3));
    const mat=new THREE.LineBasicMaterial({color:0xd0d0d0,transparent:true,opacity:0.9});
    const lines=new THREE.LineSegments(geom,mat);
    lines.position.y=-0.002;

    const borderGeom=new THREE.BufferGeometry();
    borderGeom.setAttribute('position',new THREE.Float32BufferAttribute([
      0,0,0, size,0,0,
      size,0,0, size,0,size,
      size,0,size, 0,0,size,
      0,0,size, 0,0,0,
    ],3));
    const border=new THREE.LineSegments(borderGeom,new THREE.LineBasicMaterial({color:0x9ca3af}));
    border.position.y=-0.001;

    const group=new THREE.Group();
    group.add(lines);
    group.add(border);
    return group;
  }

  async function init3DViewer(){
    if(viewer3D.initialized)return;
    setViewer3DStatus('Loading Three.js…');
    await ensureThreeDeps();
    const mount=document.getElementById('viewer-3d');
    const w=Math.max(1,mount.clientWidth),h=Math.max(1,mount.clientHeight);

    const scene=new THREE.Scene();
    scene.background=new THREE.Color(0xf0f0f0);

    const camera=new THREE.PerspectiveCamera(60,w/h,0.01,100);
    camera.position.set(5,12,15);
    camera.lookAt(5,0,5);

    const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
    renderer.setPixelRatio(window.devicePixelRatio||1);
    renderer.setSize(w,h);
    mount.appendChild(renderer.domElement);

    const frameGuide=document.createElement('div');
    frameGuide.id='capture-frame-guide';
    const frameBox=document.createElement('div');
    frameBox.id='capture-frame-box';
    const frameLabel=document.createElement('div');
    frameLabel.id='capture-frame-label';
    frameBox.appendChild(frameLabel);
    frameGuide.appendChild(frameBox);
    mount.appendChild(frameGuide);

    const controls=new THREE.OrbitControls(camera,renderer.domElement);
    controls.enableDamping=true;
    controls.target.set(5,0,5);
    controls.addEventListener('change',()=>{
      if(getActiveTab()==='3d')refreshStructuredPromptPreview();
    });

    const amb=new THREE.AmbientLight(0xffffff,0.6);
    scene.add(amb);
    const dir=new THREE.DirectionalLight(0xffffff,0.8);
    dir.position.set(1,2,1.5);
    scene.add(dir);

    const grid=buildPositiveGrid(10,0.5);
    scene.add(grid);

    const planGroup=new THREE.Group();
    scene.add(planGroup);

    viewer3D.scene=scene;
    viewer3D.camera=camera;
    viewer3D.renderer=renderer;
    viewer3D.controls=controls;
    viewer3D.grid=grid;
    viewer3D.planGroup=planGroup;
    viewer3D.initialized=true;

    if(!viewer3D.captureSizeListenersBound){
      ['py-width','py-height'].forEach((id)=>{
        const input=document.getElementById(id);
        if(!input)return;
        input.addEventListener('input',updateCaptureFrameGuide);
        input.addEventListener('change',updateCaptureFrameGuide);
      });
      const frameToggle=document.getElementById('toggle-frame-guide');
      if(frameToggle){
        frameToggle.addEventListener('input',updateCaptureFrameGuide);
        frameToggle.addEventListener('change',updateCaptureFrameGuide);
      }
      viewer3D.captureSizeListenersBound=true;
    }

    updateCaptureFrameGuide();

    const animate=()=>{
      viewer3D.animId=requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene,camera);
    };
    animate();
    setViewer3DStatus('3D viewer ready.');
  }

  function resize3DViewer(){
    if(!viewer3D.initialized)return;
    const mount=document.getElementById('viewer-3d');
    const w=Math.max(1,mount.clientWidth),h=Math.max(1,mount.clientHeight);
    viewer3D.camera.aspect=w/h;
    viewer3D.camera.updateProjectionMatrix();
    viewer3D.renderer.setPixelRatio(window.devicePixelRatio||1);
    viewer3D.renderer.setSize(w,h);
    updateCaptureFrameGuide();
  }

  function clear3DPlanGroup(){
    if(!viewer3D.planGroup)return;
    viewer3D.planGroup.position.set(0,0,0);
    while(viewer3D.planGroup.children.length){
      const child=viewer3D.planGroup.children.pop();
      if(child.geometry)child.geometry.dispose();
      if(child.material){
        if(Array.isArray(child.material))child.material.forEach((m)=>m&&m.dispose&&m.dispose());
        else if(child.material.dispose)child.material.dispose();
      }
      viewer3D.planGroup.remove(child);
    }
  }

  function getCurrentPlanFingerprint(){
    const s=getActiveFPState();
    const { selected, results, fitMode }=getSelectionState();
    if(!s||selected===null)return null;
    const result=results[selected]||{};
    const fp=s.fp||{};
    return [
      selected,
      fitMode?1:0,
      result.name||fp.id||'plan',
      result.rot||0,
      result.mirror?1:0,
      (fp.rooms||[]).length,
      (fp.boundary||[]).length,
    ].join('|');
  }

  function planVertexToXY(v){
    if(Array.isArray(v))return [v[0],v[1]];
    if(v&&typeof v==='object')return [v.x,v.y];
    return [0,0];
  }

  function planPtToSceneXY(pt,state){
    let [x,y]=planVertexToXY(pt);
    const cen=(state.rot||state.mirror)?bdryCenter(state.fpBdry):null;
    if(state.mirror)y=2*cen[1]-y;
    if(state.rot)[x,y]=rotatePtAround([x,y],state.rot,cen);
    x+=state.offset[0];
    y+=state.offset[1];
    return [x*WORLD_SCALE,y*WORLD_SCALE];
  }

  function shapeFromPlanPolygon(poly,state){
    if(!poly||poly.length<3)return null;
    const shape=new THREE.Shape();
    poly.forEach((pt,i)=>{
      const [x,z]=planPtToSceneXY(pt,state);
      if(i===0)shape.moveTo(x,-z);
      else shape.lineTo(x,-z);
    });
    shape.closePath();
    return shape;
  }

  function roomTypeColor(){
    return 0xb8b8b8;
  }

  function bboxToSceneRect(bbox,state){
    if(!Array.isArray(bbox)||bbox.length!==4)return null;
    const [x1,y1,x2,y2]=bbox;
    const corners=[
      planPtToSceneXY([x1,y1],state),
      planPtToSceneXY([x2,y1],state),
      planPtToSceneXY([x2,y2],state),
      planPtToSceneXY([x1,y2],state),
    ];
    const xs=corners.map((p)=>p[0]);
    const zs=corners.map((p)=>p[1]);
    return {
      minX:Math.min(...xs),
      maxX:Math.max(...xs),
      minZ:Math.min(...zs),
      maxZ:Math.max(...zs),
    };
  }

  function closestPointOnSegment2D(px,pz,ax,az,bx,bz){
    const dx=bx-ax,dz=bz-az;
    const len2=dx*dx+dz*dz;
    if(len2<1e-9)return {x:ax,z:az,t:0,d2:(px-ax)*(px-ax)+(pz-az)*(pz-az)};
    const t=Math.max(0,Math.min(1,((px-ax)*dx+(pz-az)*dz)/len2));
    const x=ax+t*dx,z=az+t*dz;
    const d2=(px-x)*(px-x)+(pz-z)*(pz-z);
    return {x,z,t,d2};
  }

  function closestSegmentOnPolygonScene(poly,state,px,pz){
    if(!Array.isArray(poly)||poly.length<2)return null;
    const pts=poly.map((pt)=>planPtToSceneXY(pt,state));
    let best=null;
    for(let i=0;i<pts.length;i++){
      const [ax,az]=pts[i];
      const [bx,bz]=pts[(i+1)%pts.length];
      const cp=closestPointOnSegment2D(px,pz,ax,az,bx,bz);
      if(!best||cp.d2<best.d2){
        best={x:cp.x,z:cp.z,d2:cp.d2,ax,az,bx,bz};
      }
    }
    return best;
  }

  function closestSegmentOnBoundaryScene(boundary,state,px,pz){
    if(!Array.isArray(boundary)||boundary.length<2)return null;
    const pts=boundary.map((v)=>{
      const [x,y]=planVertexToXY(v);
      return planPtToSceneXY([x,y],state);
    });
    let best=null;
    for(let i=0;i<pts.length;i++){
      const [ax,az]=pts[i];
      const [bx,bz]=pts[(i+1)%pts.length];
      const cp=closestPointOnSegment2D(px,pz,ax,az,bx,bz);
      if(!best||cp.d2<best.d2){
        best={x:cp.x,z:cp.z,d2:cp.d2,ax,az,bx,bz};
      }
    }
    return best;
  }

  function isBalconyRoom(room){
    const st=room?.source_type??room?.type;
    return st===BALCONY_TYPE_ID;
  }

  function resolveWindowRefRooms(win,rooms){
    const refs=Array.isArray(win?.room_refs)?win.room_refs:[];
    if(!refs.length)return [];
    const seen=new Set();
    const out=[];
    refs.forEach((ref)=>{
      const refStr=String(ref);
      let room=rooms.find((r)=>String(r.id)===refStr);
      if(!room){
        const idx=parseInt(refStr,10);
        if(Number.isInteger(idx)&&idx>=0&&idx<rooms.length)room=rooms[idx];
      }
      if(!room)return;
      const key=String(room.id??'');
      if(seen.has(key))return;
      seen.add(key);
      out.push(room);
    });
    return out;
  }

  function isBalconyFacingAnchorSegment(seg,balconyRooms,state){
    if(!seg||!Array.isArray(balconyRooms)||!balconyRooms.length)return false;
    const mx=(seg.ax+seg.bx)/2;
    const mz=(seg.az+seg.bz)/2;
    const sdx=seg.bx-seg.ax;
    const sdz=seg.bz-seg.az;
    const slen=Math.hypot(sdx,sdz);
    if(slen<1e-6)return false;

    for(const room of balconyRooms){
      const bseg=closestSegmentOnPolygonScene(room.polygon,state,mx,mz);
      if(!bseg)continue;
      if(Math.sqrt(bseg.d2)>WINDOW_BALCONY_EDGE_TOL)continue;

      const bdx=bseg.bx-bseg.ax;
      const bdz=bseg.bz-bseg.az;
      const blen=Math.hypot(bdx,bdz);
      if(blen<1e-6)continue;

      const absDot=Math.abs((sdx*bdx+sdz*bdz)/(slen*blen));
      if(absDot>0.985)return true;
    }
    return false;
  }

  function findWindowAnchorSegment(win,fp,state,cx,cz){
    const rooms=(fp.rooms||[]).filter((r)=>Array.isArray(r?.polygon)&&r.polygon.length>=2);
    if(rooms.length){
      const refs=Array.isArray(win?.room_refs)?win.room_refs:[];
      for(const ref of refs){
        const refStr=String(ref);
        const byId=rooms.find((r)=>String(r.id)===refStr);
        if(byId){
          const seg=closestSegmentOnPolygonScene(byId.polygon,state,cx,cz);
          if(seg)return {...seg,room:byId};
        }
        const idx=parseInt(refStr,10);
        if(Number.isInteger(idx)&&idx>=0&&idx<rooms.length){
          const seg=closestSegmentOnPolygonScene(rooms[idx].polygon,state,cx,cz);
          if(seg)return {...seg,room:rooms[idx]};
        }
      }

      let best=null;
      rooms.forEach((room)=>{
        const seg=closestSegmentOnPolygonScene(room.polygon,state,cx,cz);
        if(!seg)return;
        if(!best||seg.d2<best.d2)best={...seg,room};
      });
      if(best)return best;
    }

    const boundarySeg=closestSegmentOnBoundaryScene(fp?.boundary||[],state,cx,cz);
    if(boundarySeg)return {...boundarySeg,room:null};

    return null;
  }

  function addBalconyGeometry(room,state){
    if(!room?.polygon||room.polygon.length<3)return;
    const pts=room.polygon.map((pt)=>planPtToSceneXY(pt,state));

    const floorShape=new THREE.Shape();
    pts.forEach(([x,z],i)=>{
      if(i===0)floorShape.moveTo(x,-z);
      else floorShape.lineTo(x,-z);
    });
    floorShape.closePath();

    const floorGeom=new THREE.ExtrudeGeometry(floorShape,{depth:BALCONY_FLOOR_THICKNESS,bevelEnabled:false,steps:1});
    floorGeom.rotateX(-Math.PI/2);
    const floorMesh=new THREE.Mesh(
      floorGeom,
      new THREE.MeshStandardMaterial({color:0xbdbdbd,roughness:0.9,metalness:0.02,side:THREE.DoubleSide})
    );
    floorMesh.userData.captureRole='slabs';
    floorMesh.castShadow=false;
    floorMesh.receiveShadow=true;
    viewer3D.planGroup.add(floorMesh);

    for(let i=0;i<pts.length;i++){
      const [x1,z1]=pts[i];
      const [x2,z2]=pts[(i+1)%pts.length];
      const dx=x2-x1,dz=z2-z1;
      const len=Math.hypot(dx,dz);
      if(len<1e-6)continue;
      const parapetGeom=new THREE.BoxGeometry(len,BALCONY_PARAPET_HEIGHT,BALCONY_PARAPET_THICKNESS);
      const parapetMesh=new THREE.Mesh(
        parapetGeom,
        new THREE.MeshStandardMaterial({color:0xa8a8a8,roughness:0.9,metalness:0.02})
      );
      parapetMesh.position.set((x1+x2)/2,BALCONY_PARAPET_HEIGHT/2,(z1+z2)/2);
      parapetMesh.rotation.y=Math.atan2(dz,dx);
      parapetMesh.userData.captureRole='parapets';
      parapetMesh.castShadow=false;
      parapetMesh.receiveShadow=true;
      viewer3D.planGroup.add(parapetMesh);
    }
  }

  function addDoorGeometry(fp,state){
    const doors=fp.doors||[];
    doors.forEach((door)=>{
      const rect=bboxToSceneRect(door?.bbox,state);
      if(!rect)return;

      const cx=(rect.minX+rect.maxX)/2;
      const cz=(rect.minZ+rect.maxZ)/2;
      const doorLen=Math.max(DOOR_THICKNESS,Math.max(rect.maxX-rect.minX,rect.maxZ-rect.minZ));

      let targetX=cx,targetZ=cz;
      let segAngle=0;
      const seg=findWindowAnchorSegment(door,fp,state,cx,cz);
      if(seg){
        targetX=seg.x;
        targetZ=seg.z;
        segAngle=Math.atan2(seg.bz-seg.az,seg.bx-seg.ax);
      }

      const geom=new THREE.BoxGeometry(doorLen,DOOR_HEIGHT,DOOR_THICKNESS);
      const mesh=new THREE.Mesh(
        geom,
        new THREE.MeshStandardMaterial({color:0x9a6a34,roughness:0.6,metalness:0.08})
      );
      mesh.position.set(targetX,DOOR_HEIGHT/2,targetZ);
      mesh.rotation.y=segAngle;
      mesh.userData.captureRole='doors';
      mesh.castShadow=false;
      mesh.receiveShadow=false;
      viewer3D.planGroup.add(mesh);
    });
  }

  function addWindowGeometry(fp,state){
    const windows=fp.windows||[];
    const rooms=(fp.rooms||[]).filter((room)=>Array.isArray(room?.polygon)&&room.polygon.length>=2);
    const balconyRooms=rooms.filter((room)=>isBalconyRoom(room));
    windows.forEach((windowDef)=>{
      const refRooms=resolveWindowRefRooms(windowDef,rooms);
      if(refRooms.some((room)=>isBalconyRoom(room)))return;

      const rect=bboxToSceneRect(windowDef?.bbox,state);
      if(!rect)return;

      const cx=(rect.minX+rect.maxX)/2;
      const cz=(rect.minZ+rect.maxZ)/2;
      const winLen=Math.max(WINDOW_THICKNESS,Math.max(rect.maxX-rect.minX,rect.maxZ-rect.minZ));

      let targetX=cx,targetZ=cz;
      let segAngle=0;

      const seg=findWindowAnchorSegment(windowDef,fp,state,cx,cz);
      if(seg){
        targetX=seg.x;
        targetZ=seg.z;
        segAngle=Math.atan2(seg.bz-seg.az,seg.bx-seg.ax);
      }

      const balconyDoorOpening=
        seg&&!isBalconyRoom(seg.room)&&isBalconyFacingAnchorSegment(seg,balconyRooms,state);
      const openingHeight=balconyDoorOpening?BALCONY_OPENING_DOOR_HEIGHT:WINDOW_HEIGHT;
      const sillHeight=balconyDoorOpening?0:WINDOW_SILL_HEIGHT;
      const openingMaterial=balconyDoorOpening
        ?new THREE.MeshStandardMaterial({color:0xa36a2a,roughness:0.6,metalness:0.08})
        :new THREE.MeshStandardMaterial({color:0x8a8a8a,roughness:0.35,metalness:0.15,transparent:true,opacity:0.75});

      const geom=new THREE.BoxGeometry(winLen,openingHeight,WINDOW_THICKNESS);
      const mesh=new THREE.Mesh(geom,openingMaterial);
      mesh.position.set(targetX,sillHeight+openingHeight/2,targetZ);
      mesh.rotation.y=segAngle;
      mesh.userData.captureRole=balconyDoorOpening?'doors':'windows';
      mesh.castShadow=false;
      mesh.receiveShadow=false;
      viewer3D.planGroup.add(mesh);
    });
  }

  function build3DSceneFromActivePlan(){
    const s=getActiveFPState();
    const empty=document.getElementById('viewer-3d-empty');
    clear3DPlanGroup();
    if(!s||!s.fp){
      if(empty)empty.style.display='flex';
      setViewer3DStatus('No active plan selected.');
      return;
    }
    if(empty)empty.style.display='none';

    const fp=s.fp;
    const rooms=fp.rooms||[];
    const boundaryPoly=fp.boundary||[];

    const roomGeomOpts={depth:ROOM_HEIGHT,bevelEnabled:false,steps:1};
    rooms.forEach((room)=>{
      const st=room.source_type??room.type;
      if(st===BALCONY_TYPE_ID){
        addBalconyGeometry(room,s);
        return;
      }

      const shape=shapeFromPlanPolygon(room.polygon,s);
      if(!shape)return;
      const geom=new THREE.ExtrudeGeometry(shape,roomGeomOpts);
      geom.rotateX(-Math.PI/2);
      const mat=new THREE.MeshStandardMaterial({color:roomTypeColor(st),roughness:0.85,metalness:0.05,side:THREE.DoubleSide});
      const mesh=new THREE.Mesh(geom,mat);
      mesh.userData.captureRole='walls';
      mesh.castShadow=false;
      mesh.receiveShadow=true;
      viewer3D.planGroup.add(mesh);
    });

    addDoorGeometry(fp,s);
    addWindowGeometry(fp,s);

    const boundaryShape=shapeFromPlanPolygon(boundaryPoly,s);
    if(boundaryShape){
      const slabGeom=new THREE.ExtrudeGeometry(boundaryShape,{depth:0.1,bevelEnabled:false,steps:1});
      slabGeom.rotateX(-Math.PI/2);
      const slab=new THREE.Mesh(
        slabGeom,
        new THREE.MeshStandardMaterial({color:0xd2d2d2,roughness:0.92,metalness:0.0,side:THREE.DoubleSide})
      );
      slab.userData.captureRole='ground';
      slab.position.y=-0.1;
      viewer3D.planGroup.add(slab);

      const roofGeom=new THREE.ExtrudeGeometry(boundaryShape,{depth:0.15,bevelEnabled:false,steps:1});
      roofGeom.rotateX(-Math.PI/2);
      const roof=new THREE.Mesh(
        roofGeom,
        new THREE.MeshStandardMaterial({color:0x9a9a9a,roughness:0.85,metalness:0.03,side:THREE.DoubleSide})
      );
      roof.userData.captureRole='slabs';
      roof.position.y=ROOM_HEIGHT;
      viewer3D.planGroup.add(roof);
    }

    const bCen=bdryCenter((boundaryPoly||[]).map(planVertexToXY));
    const [cx,cz]=planPtToSceneXY(bCen,s);

    const bounds=new THREE.Box3().setFromObject(viewer3D.planGroup);
    if(Number.isFinite(bounds.min.x)&&Number.isFinite(bounds.max.x)){
      const geomCx=(bounds.min.x+bounds.max.x)/2;
      const geomCz=(bounds.min.z+bounds.max.z)/2;
      viewer3D.planGroup.position.x+=5-geomCx;
      viewer3D.planGroup.position.z+=5-geomCz;
    }

    const finalCenterX=viewer3D.planGroup.position.x+cx;
    const finalCenterZ=viewer3D.planGroup.position.z+cz;
    viewer3D.controls.target.set(finalCenterX,0,finalCenterZ);
    viewer3D.controls.update();
    setViewer3DStatus(`Scene built: ${rooms.length} rooms extruded.`);
  }

  async function sync3DSceneIfNeeded(force=false){
    if(getActiveTab()!=='3d')return;
    try{
      await init3DViewer();
      resize3DViewer();
      const fp=getCurrentPlanFingerprint();
      if(force||fp!==viewer3D.lastFingerprint){
        build3DSceneFromActivePlan();
        viewer3D.lastFingerprint=fp;
      }
    }catch(err){
      console.error('3D viewer init failed:',err);
      setViewer3DStatus('3D initialization failed. Check network/CDN access.');
    }
  }

  function captureDepthAndPreview(){
    if(!viewer3D.initialized||!viewer3D.scene||!viewer3D.camera||!viewer3D.renderer){
      setViewer3DStatus('3D viewer not initialized yet.');
      return;
    }
    const scene=viewer3D.scene;
    const camera=viewer3D.camera;
    const renderer=viewer3D.renderer;
    const {width:targetWidth,height:targetHeight}=getRequestedCaptureSize();

    const prevNear=camera.near;
    const prevFar=camera.far;
    const depthMat=new THREE.MeshDepthMaterial({depthPacking:THREE.BasicDepthPacking});

    let captureNear=camera.near;
    let captureFar=camera.far;
    let captureRectPx=null;

    let captured;
    let previewDataUrl=null;
    try{
      captureRectPx=computeCenteredAspectRect(
        renderer.domElement.width,
        renderer.domElement.height,
        targetWidth/Math.max(1,targetHeight),
      );

      const bounds=new THREE.Box3().setFromObject(viewer3D.planGroup);
      if(Number.isFinite(bounds.min.x)&&Number.isFinite(bounds.max.x)){
        const center=bounds.getCenter(new THREE.Vector3());
        const size=bounds.getSize(new THREE.Vector3());
        const radius=Math.max(size.x,size.y,size.z)*0.6+0.25;
        const dist=camera.position.distanceTo(center);
        const fitNear=Math.max(0.01,dist-radius*2.2);
        const fitFar=Math.max(fitNear+1.0,dist+radius*2.2);
        camera.near=fitNear;
        camera.far=fitFar;
        captureNear=fitNear;
        captureFar=fitFar;
        camera.updateProjectionMatrix();
      }

      captured=withCaptureVisibility(()=>{
        scene.overrideMaterial=depthMat;
        renderer.render(scene,camera);
        const depthImgFull=captureCanvasImageData(renderer.domElement);
        const maskImgFull=captureMaskImageData(scene,camera,renderer);
        const normalImgFull=captureNormalImageData(scene,camera,renderer);
        const opacityImgFull=captureOpacityMaskImageData(scene,camera,renderer);

        const depthImg=cropAndScaleImageData(depthImgFull,captureRectPx,targetWidth,targetHeight);
        const maskImg=cropAndScaleImageData(maskImgFull,captureRectPx,targetWidth,targetHeight);
        const normalImg=cropAndScaleImageData(normalImgFull,captureRectPx,targetWidth,targetHeight);
        const opacityImg=cropAndScaleImageData(opacityImgFull,captureRectPx,targetWidth,targetHeight);

        const depthDataUrl=makeDepthDataUrlFromImageData(depthImg,maskImg,captureNear,captureFar);
        const maskDataUrl=makeBinaryMaskDataUrl(maskImg);
        const normalDataUrl=makeNormalDataUrlFromImageData(normalImg,maskImg);
        const opacityMaskDataUrl=imageDataToDataUrl(opacityImg);
        return {
          depthDataUrl,
          maskDataUrl,
          silhouetteDataUrl:maskDataUrl,
          normalDataUrl,
          opacityMaskDataUrl,
        };
      });
    }finally{
      camera.near=prevNear;
      camera.far=prevFar;
      camera.updateProjectionMatrix();
      scene.overrideMaterial=null;
      renderer.render(scene,camera);
      const previewFull=captureCanvasImageData(renderer.domElement);
      const previewCrop=cropAndScaleImageData(previewFull,captureRectPx,targetWidth,targetHeight);
      previewDataUrl=imageDataToDataUrl(previewCrop);
      depthMat.dispose();
    }

    renderCaptureState.depthDataUrl=captured.depthDataUrl;
    renderCaptureState.maskDataUrl=captured.maskDataUrl;
    renderCaptureState.silhouetteDataUrl=captured.silhouetteDataUrl;
    renderCaptureState.normalDataUrl=captured.normalDataUrl;
    renderCaptureState.opacityMaskDataUrl=captured.opacityMaskDataUrl;
    renderCaptureState.previewDataUrl=previewDataUrl;
    renderCaptureState.capturedAt=new Date().toISOString();
    renderCaptureState.activeView='preview';
    updateCapturePanel();
    updateCaptureFrameGuide();
    refreshStructuredPromptPreview();
    setViewer3DStatus(`Maps captured inside frame ${targetWidth}x${targetHeight} (near ${captureNear.toFixed(3)}, far ${captureFar.toFixed(3)}).`);
  }

  return {
    captureDepthAndPreview,
    getCameraPerspectivePrompt,
    init3DViewer,
    resize3DViewer,
    sync3DSceneIfNeeded,
  };
}