import {
  METRES_PER_UNIT,
  PLAN_CACHE,
  RCOLS,
  SNAP_PX,
} from './constants.js';

const CANVAS_PAD = 0.07;
import {
  bdryCenter,
  c2n as geometryC2N,
  canvasToPlan as geometryCanvasToPlan,
  ghost as geometryGhost,
  n2c as geometryN2C,
  pdist as geometryPDist,
  planToCanvas as geometryPlanToCanvas,
  projSnap as geometryProjSnap,
  ptInPoly,
  ptToSegDistSq,
  rotatePtAround,
  snapToEdge as geometrySnapToEdge,
} from './geometry.js';
import {
  getActivePlanView,
} from './queryState.js';
import { createBoundaryEditor } from './boundary-editor.js';
import { createCanvasRenderer } from './canvas-renderer.js';
import {
  autoLoadIndex,
  checkReadyState,
  clearNamedCache,
  fetchFittedState,
  fetchWithCache as fetchWithCacheApi,
  fitPlanLocal,
  postJsonWithTimeout,
  renderAdaptedResultsView,
  renderResultsListView,
  runRetrievalFlow,
  selectResultState,
  setStatusUI,
  setupDropZone,
  toggleFitModeState,
  tryFinalizeIndex,
} from './retrieval.js';
import { createRenderPipeline } from './render-pipeline.js';
import { createViewer3D } from './viewer-3d.js';
import { initPlanDrawingPanel, loadFixtureData } from './plan-drawing-panel.js';
import {
  analyse_boundary,
  check_typology_viability,
  evaluate_typology_fit,
  evaluate_floor_plate_requirements,
  generate_offset_subdivision_regions,
  get_requirement_check_catalog,
  propose_typology_options,
} from './floor-plate-tools.js';
import { createInitialState, DEFAULT_RENDER_PROMPT_CONFIG } from './state.js';

const initialState=createInitialState();
async function clearPlanCache(){
  await clearNamedCache(PLAN_CACHE);
  document.getElementById('lbl-pls').textContent='cache cleared — reload to re-fetch';
  setStatus('plan cache cleared','warn');
}

async function fetchWithCache(url){
  return fetchWithCacheApi(url,{cacheName:PLAN_CACHE});
}

// state
const {db,cache,viewer3D,renderCaptureState,renderRequestState}=initialState;
const {devMode}=initialState;
let {req,params,boundary,closed,entrance,entranceMode,entranceSnap,edgeTypes,edgeTypeMode,hoveredEdge,mouse,shiftDown,snapFirst,results,selected,fitMode,fittedCache,showPlanBdry,showWindows,showDoors,showCommonArea,selectedRoom,hoveredRoom,hoveredEntrance,dragState,displacementArrows,cW,cH,activeTab,floorPlateState}=initialState;

// ARAP debug overlay (2026-08-11) — room polygons from an arap_bridge.py
// canvas step (see runArapCanvasStep below), drawn directly by
// canvas-renderer.js's render() via drawFP. Cleared by clearArapDebugOverlay.
// Topology-graph node selection (room-swap prerequisite) — up to two String
// room ids, most-recently-selected two kept (oldest dropped past the cap).
let graphSelection=[];

let arapDebugRooms=null;
let arapDebugLabel=null;
// Stage 1 (re-solve loop) — the arrange step's judgement report, rendered as an
// overlay so infeasibility is visible on the canvas, not only in the console.
// Null for every step except `arrange`.
let arapDebugReport=null;
// [x1,y1,x2,y2] — the retrieved plan's own entrance rectangle, carried
// through the same align/solve transform as the rooms (2026-08-11), so it
// actually moves with the plan instead of showing a stale pre-transform
// position.
let arapDebugDoor=null;
// Index of the result align has actually been run for, this session — the
// ARAP-adaptation (solve) step is gated on this matching the current
// selection, so it can't be run without align having been executed first.
let arapAlignedIndex=null;
// Same, for solve — rectify is gated on this the same way solve is gated
// on arapAlignedIndex.
let arapSolvedIndex=null;
// Same, for rectify — fill (fill_boundary_with_living) is gated on this.
let arapRectifiedIndex=null;
// Same, for fill — oracle (surface_defects) and agent (orchestrate_refinement)
// are both gated on this: they need the fully-tessellated fill+consolidate
// output vlm_refine.py was built and validated against.
let arapFilledIndex=null;

devMode.enabled=new URLSearchParams(window.location.search).get('dev')==='true';
devMode.panelVisible=devMode.enabled;

// ── Fit tab state ──────────────────────────────────────────────────────────
const fitState={
  showGraph:false,
  showOriginal:true,
  showBoundary:true,
  currentStep:1,
  fittedPlan:null,   // plan dict returned by the server after fitting
};

function _fitUpdateLog(text){
  const el=document.getElementById('fit-log');
  if(el)el.textContent=text;
}

function _fitSyncRunButton(){
  const btn=document.getElementById('fit-run-btn');
  if(!btn)return;
  const hasResult=results&&results.length>0;
  btn.disabled=!hasResult||!closed;
  btn.textContent=`▶ run step ${fitState.currentStep}`;
}

async function _fitRunStep(){
  if(fitState.currentStep===1)await _fitStep1MapGraph();
}

// ── Fit graph computation ──────────────────────────────────────────────────

function _nearestPtOnSeg([px,py],[ax,ay],[bx,by]){
  const dx=bx-ax,dy=by-ay,len2=dx*dx+dy*dy;
  if(len2<1e-12)return[ax,ay];
  const t=Math.max(0,Math.min(1,((px-ax)*dx+(py-ay)*dy)/len2));
  return[ax+t*dx,ay+t*dy];
}

function _nearestPtOnBoundary(pt,bdryPts){
  let best=null,bestD=Infinity;
  const n=bdryPts.length;
  for(let i=0;i<n;i++){
    const q=_nearestPtOnSeg(pt,bdryPts[i],bdryPts[(i+1)%n]);
    const d=(q[0]-pt[0])**2+(q[1]-pt[1])**2;
    if(d<bestD){bestD=d;best=q;}
  }
  return best??pt;
}

// Map each room centroid from plan space into the target boundary using an
// affine transform (scale around plan centroid → center on boundary centroid).
// The entrance is snapped to the nearest point on the target boundary perimeter.
// Returns { nodes, edges } where all positions are in n-space (same as boundary).
function computeFitGraph(plan, bdryPts, selRot, selMirror){
  const rooms=plan.rooms||[];
  const planEdges=plan.edges||[];
  const fpBdry=plan.boundary||[];

  // Room centroids in plan space, with rotation/mirror applied
  const rotCen=(selRot||selMirror)?bdryCenter(fpBdry):null;
  const applyTransform=([x,y])=>{
    if(selMirror)y=2*rotCen[1]-y;
    if(selRot)[x,y]=rotatePtAround([x,y],selRot,rotCen);
    return[x,y];
  };

  const centroids={};
  rooms.forEach(r=>{
    const poly=r.polygon||[];
    if(poly.length<3)return;
    const mx=poly.reduce((s,p)=>s+p[0],0)/poly.length;
    const my=poly.reduce((s,p)=>s+p[1],0)/poly.length;
    centroids[r.id]=applyTransform([mx,my]);
  });

  if(!Object.keys(centroids).length)return{nodes:[],edges:[]};

  // Map from the transformed plan boundary bbox → target boundary bbox.
  // Using boundary spans (not centroid spans) gives the correct scale,
  // and applying applyTransform to the plan boundary preserves rotation/mirror.
  const tBdry=fpBdry.map(v=>applyTransform([v[0],v[1]]));
  // Fallback to centroid cloud bbox if the plan has no boundary
  const refPts=tBdry.length?tBdry:Object.values(centroids);
  const tBdryXs=refPts.map(p=>p[0]), tBdryYs=refPts.map(p=>p[1]);
  const planMinX=Math.min(...tBdryXs), planMaxX=Math.max(...tBdryXs);
  const planMinY=Math.min(...tBdryYs), planMaxY=Math.max(...tBdryYs);
  const planW=(planMaxX-planMinX)||0.1;
  const planH=(planMaxY-planMinY)||0.1;

  const bdryXs=bdryPts.map(p=>p[0]), bdryYs=bdryPts.map(p=>p[1]);
  const bdryMinX=Math.min(...bdryXs), bdryMinY=Math.min(...bdryYs);
  const bdryW=Math.max(...bdryXs)-bdryMinX;
  const bdryH=Math.max(...bdryYs)-bdryMinY;

  const affinePt=([ox,oy])=>[
    bdryMinX+(ox-planMinX)/planW*bdryW,
    bdryMinY+(oy-planMinY)/planH*bdryH,
  ];

  const nodes=rooms.map(r=>{
    const orig=centroids[r.id];
    if(!orig)return null;
    return{id:r.id,label:r.label||r.id,sourceType:r.source_type,
           originalPos:orig,mappedPos:affinePt(orig),isEntrance:false};
  }).filter(Boolean);

  // Entrance node — must lie on the boundary perimeter (boundary anchor)
  const extraEdges=[];
  const ent=plan.entrance;
  if(ent&&ent.length>=4){
    const entOrigPt=applyTransform([(ent[0]+ent[2])/2,(ent[1]+ent[3])/2]);
    const [ex,ey]=entOrigPt;
    // hintAngle: compass bearing (0=N,90=E,180=S,270=W) from entrance to each room centroid
    nodes.forEach(n=>{
      const [rx,ry]=n.originalPos;
      n.hintAngle=Math.round((Math.atan2(rx-ex,-(ry-ey))*180/Math.PI+360)%360);
    });
    const entAffine=affinePt(entOrigPt);
    const entMapped=_nearestPtOnBoundary(entAffine,bdryPts);
    nodes.push({id:'entrance',label:'Entry',sourceType:10,
                originalPos:entOrigPt,mappedPos:entMapped,isEntrance:true});
    const common=rooms.find(r=>[0,4].includes(r.source_type));
    if(common)extraEdges.push({u:'entrance',v:common.id,connection:1,relation:0});
  }

  return{nodes,edges:[...planEdges,...extraEdges]};
}

async function _fitStep1MapGraph(){
  if(!results||!results.length){_fitUpdateLog('retrieve a plan first.');return;}
  if(!closed||boundary.length<3){_fitUpdateLog('draw and close a boundary first.');return;}

  const r=results[0];
  const fp=r?.fp;
  if(!fp){_fitUpdateLog('no plan loaded.');return;}

  _fitUpdateLog('computing layout graph…');

  const bdryPts=boundary.map(({x,y})=>[x,y]);
  const graph=computeFitGraph(fp,bdryPts,r.rot||0,r.mirror||false);

  // Show the initial affine-mapped graph while the agent runs
  fitState.mappedGraph=graph;
  fitState.showGraph=true;
  const chk=document.getElementById('fit-show-graph');
  if(chk)chk.checked=true;
  render();

  _fitUpdateLog(
    `plan: ${r.name} — ${graph.nodes.length} nodes, ${graph.edges.length} edges\n\n`+
    `Sending to spatial reasoning agent…`
  );

  // Pre-computed affine positions (rotation/mirror already applied in computeFitGraph).
  // Server uses these directly so it doesn't have to re-derive the orientation.
  const suggestedPositions=Object.fromEntries(
    graph.nodes
      .filter(n=>!n.isEntrance)
      .map(n=>[n.id,{x:n.mappedPos[0],y:n.mappedPos[1]}])
  );

  const payload={
    boundary:bdryPts,
    fp,
    entrance_point: entrance ? [entrance.x, entrance.y] : null,
    suggested_positions: suggestedPositions,
  };

  let resp;
  try{
    const res=await fetch('/fit_step1',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload),
    });
    resp=await res.json();
    if(!res.ok)throw new Error(resp.error||`HTTP ${res.status}`);
  }catch(err){
    _fitUpdateLog(`Agent error: ${err.message}`);
    _fitSyncRunButton();
    return;
  }

  // Apply returned positions to graph nodes
  const positions=resp.positions||{};
  graph.nodes.forEach(n=>{
    const p=positions[n.id];
    if(p)n.mappedPos=[p.x,p.y];
  });

  fitState.mappedGraph=graph;
  fitState.currentStep=2;
  _fitUpdateLog(
    `plan: ${r.name}\n`+
    `rooms: ${graph.nodes.length} nodes repositioned by agent.\n\n`+
    `Solid lines = direct connections (doorways).\n`+
    `Dashed lines = shared wall, no door.\n\n`+
    `Next: Step 2 will grow rooms from these nodes.`
  );
  _fitSyncRunButton();
  render();
}

function applyBoundaryEditorState(patch){
  if(Object.prototype.hasOwnProperty.call(patch,'boundary'))boundary=patch.boundary;
  if(Object.prototype.hasOwnProperty.call(patch,'closed'))closed=patch.closed;
  if(Object.prototype.hasOwnProperty.call(patch,'snapFirst'))snapFirst=patch.snapFirst;
  if(Object.prototype.hasOwnProperty.call(patch,'selected'))selected=patch.selected;
  if(Object.prototype.hasOwnProperty.call(patch,'results'))results=patch.results;
  if(Object.prototype.hasOwnProperty.call(patch,'selectedRoom'))selectedRoom=patch.selectedRoom;
  if(Object.prototype.hasOwnProperty.call(patch,'dragState'))dragState=patch.dragState;
  if(Object.prototype.hasOwnProperty.call(patch,'hoveredRoom'))hoveredRoom=patch.hoveredRoom;
  if(Object.prototype.hasOwnProperty.call(patch,'hoveredEntrance'))hoveredEntrance=patch.hoveredEntrance;
  if(Object.prototype.hasOwnProperty.call(patch,'displacementArrows'))displacementArrows=patch.displacementArrows;
  if(Object.prototype.hasOwnProperty.call(patch,'entrance'))entrance=patch.entrance;
  if(Object.prototype.hasOwnProperty.call(patch,'entranceMode'))entranceMode=patch.entranceMode;
  if(Object.prototype.hasOwnProperty.call(patch,'entranceSnap'))entranceSnap=patch.entranceSnap;
  if(Object.prototype.hasOwnProperty.call(patch,'edgeTypes'))edgeTypes=patch.edgeTypes;
  if(Object.prototype.hasOwnProperty.call(patch,'edgeTypeMode'))edgeTypeMode=patch.edgeTypeMode;
  if(Object.prototype.hasOwnProperty.call(patch,'hoveredEdge'))hoveredEdge=patch.hoveredEdge;
}

const canvas=document.getElementById('main-canvas');
const ctx=canvas.getContext('2d');
let viewport={zoom:1.0,panX:0,panY:0};
let panDrag=null;
let syncDevModePanel=()=>{};
let devModeSetupDone=false;

const MIN_UNIT_AREA_M2={
  studio:25,
  bed1:36,
  bed2:50,
  bed3:65,
};

const MIN_UNIT_AREA_M2_BY_API_TYPE={
  '1bed':MIN_UNIT_AREA_M2.bed1,
  '2bed':MIN_UNIT_AREA_M2.bed2,
  '3bed':MIN_UNIT_AREA_M2.bed3,
  studio:MIN_UNIT_AREA_M2.studio,
};

function buildMockFloorPlateState(){
  return buildFloorPlateFixture('sample');
}

function buildFloorPlateFixture(fixtureId='sample'){
  const fixtures={
    sample:{
      floor_plate:{
        id:'fp-dev-sample',
        typology:'double_loaded_slab',
        boundary:[[0.1,0.15],[0.9,0.15],[0.9,0.85],[0.1,0.85]],
        structural_grid:{x_lines:[0.22,0.5,0.78],y_lines:[0.39,0.61],column_positions:[]},
        cores:[{id:'core-0',type:'stair_lift',polygon:[[0.45,0.44],[0.55,0.44],[0.55,0.56],[0.45,0.56]]}],
        corridors:[{id:'corr-0',width_min:1.4,polygon:[[0.16,0.48],[0.84,0.48],[0.84,0.52],[0.16,0.52]]}],
        units:[
          {
            id:'unit-0',
            type:'2bed',
            polygon:[[0.12,0.18],[0.44,0.18],[0.44,0.48],[0.12,0.48]],
            typed_edges:[
              {edge_index:0,type:'external'},
              {edge_index:1,type:'shared_wall'},
              {edge_index:2,type:'facing_corridor'},
              {edge_index:3,type:'external'},
            ],
          },
          {
            id:'unit-1',
            type:'1bed',
            polygon:[[0.56,0.18],[0.88,0.18],[0.88,0.48],[0.56,0.48]],
            typed_edges:[
              {edge_index:0,type:'external'},
              {edge_index:1,type:'external'},
              {edge_index:2,type:'facing_corridor'},
              {edge_index:3,type:'shared_wall'},
            ],
          },
        ],
        unresolved_regions:[],
      },
    },
    compact_slab:{
      floor_plate:{
        id:'fp-dev-compact',
        typology:'double_loaded_slab',
        boundary:[[0.18,0.2],[0.82,0.2],[0.82,0.8],[0.18,0.8]],
        structural_grid:{x_lines:[0.3,0.5,0.7],y_lines:[0.4,0.6],column_positions:[]},
        cores:[{id:'core-0',type:'stair_lift',polygon:[[0.46,0.44],[0.54,0.44],[0.54,0.56],[0.46,0.56]]}],
        corridors:[{id:'corr-0',width_min:1.5,polygon:[[0.23,0.48],[0.77,0.48],[0.77,0.52],[0.23,0.52]]}],
        units:[
          {
            id:'unit-a',type:'2bed',polygon:[[0.2,0.22],[0.45,0.22],[0.45,0.48],[0.2,0.48]],
            typed_edges:[
              {edge_index:0,type:'external'},{edge_index:1,type:'shared_wall'},{edge_index:2,type:'facing_corridor'},{edge_index:3,type:'external'},
            ],
          },
          {
            id:'unit-b',type:'1bed',polygon:[[0.55,0.22],[0.8,0.22],[0.8,0.48],[0.55,0.48]],
            typed_edges:[
              {edge_index:0,type:'external'},{edge_index:1,type:'external'},{edge_index:2,type:'facing_corridor'},{edge_index:3,type:'shared_wall'},
            ],
          },
          {
            id:'unit-c',type:'2bed',polygon:[[0.2,0.52],[0.45,0.52],[0.45,0.78],[0.2,0.78]],
            typed_edges:[
              {edge_index:0,type:'facing_corridor'},{edge_index:1,type:'shared_wall'},{edge_index:2,type:'external'},{edge_index:3,type:'external'},
            ],
          },
          {
            id:'unit-d',type:'1bed',polygon:[[0.55,0.52],[0.8,0.52],[0.8,0.78],[0.55,0.78]],
            typed_edges:[
              {edge_index:0,type:'facing_corridor'},{edge_index:1,type:'external'},{edge_index:2,type:'external'},{edge_index:3,type:'shared_wall'},
            ],
          },
        ],
        unresolved_regions:[],
      },
    },
    elongated_slab:{
      floor_plate:{
        id:'fp-dev-elongated',
        typology:'double_loaded_slab',
        boundary:[[0.06,0.28],[0.94,0.28],[0.94,0.72],[0.06,0.72]],
        structural_grid:{x_lines:[0.18,0.32,0.5,0.68,0.82],y_lines:[0.44,0.56],column_positions:[]},
        cores:[
          {id:'core-0',type:'stair_lift',polygon:[[0.27,0.45],[0.33,0.45],[0.33,0.55],[0.27,0.55]]},
          {id:'core-1',type:'stair_lift',polygon:[[0.67,0.45],[0.73,0.45],[0.73,0.55],[0.67,0.55]]},
        ],
        corridors:[{id:'corr-0',width_min:1.5,polygon:[[0.11,0.48],[0.89,0.48],[0.89,0.52],[0.11,0.52]]}],
        units:[
          {id:'unit-0',type:'1bed',polygon:[[0.08,0.3],[0.24,0.3],[0.24,0.48],[0.08,0.48]],typed_edges:[{edge_index:0,type:'external'},{edge_index:1,type:'shared_wall'},{edge_index:2,type:'facing_corridor'},{edge_index:3,type:'external'}]},
          {id:'unit-1',type:'2bed',polygon:[[0.34,0.3],[0.48,0.3],[0.48,0.48],[0.34,0.48]],typed_edges:[{edge_index:0,type:'external'},{edge_index:1,type:'shared_wall'},{edge_index:2,type:'facing_corridor'},{edge_index:3,type:'shared_wall'}]},
          {id:'unit-2',type:'2bed',polygon:[[0.52,0.3],[0.66,0.3],[0.66,0.48],[0.52,0.48]],typed_edges:[{edge_index:0,type:'external'},{edge_index:1,type:'shared_wall'},{edge_index:2,type:'facing_corridor'},{edge_index:3,type:'shared_wall'}]},
          {id:'unit-3',type:'1bed',polygon:[[0.76,0.3],[0.92,0.3],[0.92,0.48],[0.76,0.48]],typed_edges:[{edge_index:0,type:'external'},{edge_index:1,type:'external'},{edge_index:2,type:'facing_corridor'},{edge_index:3,type:'shared_wall'}]},
        ],
        unresolved_regions:[],
      },
    },
    near_failing_narrow:{
      floor_plate:{
        id:'fp-dev-narrow',
        typology:'single_loaded',
        boundary:[[0.12,0.34],[0.88,0.34],[0.88,0.66],[0.12,0.66]],
        structural_grid:{x_lines:[0.26,0.44,0.62,0.8],y_lines:[0.44,0.56],column_positions:[]},
        cores:[{id:'core-0',type:'stair_lift',polygon:[[0.47,0.42],[0.53,0.42],[0.53,0.58],[0.47,0.58]]}],
        corridors:[{id:'corr-0',width_min:1.2,polygon:[[0.16,0.49],[0.84,0.49],[0.84,0.53],[0.16,0.53]]}],
        units:[
          {id:'unit-0',type:'1bed',polygon:[[0.14,0.36],[0.44,0.36],[0.44,0.49],[0.14,0.49]],typed_edges:[{edge_index:0,type:'external'},{edge_index:1,type:'shared_wall'},{edge_index:2,type:'facing_corridor'},{edge_index:3,type:'external'}]},
          {id:'unit-1',type:'2bed',polygon:[[0.56,0.36],[0.86,0.36],[0.86,0.49],[0.56,0.49]],typed_edges:[{edge_index:0,type:'external'},{edge_index:1,type:'external'},{edge_index:2,type:'facing_corridor'},{edge_index:3,type:'shared_wall'}]},
        ],
        unresolved_regions:[[[0.44,0.36],[0.56,0.36],[0.56,0.49],[0.44,0.49]]],
      },
    },
  };

  return fixtures[fixtureId]||fixtures.sample;
}

function loadFixture(fixtureId){
  floorPlateState=buildFloorPlateFixture(fixtureId);
  devMode.activeFixture=fixtureId;
  devMode.lastToolOutput={
    source:'fixture_loader',
    fixtureId,
    loadedAt:new Date().toISOString(),
  };
  runRequirementsCompliance();
}

function getGeneratedFloorPlateTests(){
  return {
    double_loaded:{
      label:'double-loaded slab',
      boundary:[[0,0],[14000,0],[14000,20000],[0,20000]],
      unit_mix:[
        {type:'2bed',count:4,target_area:70},
        {type:'1bed',count:2,target_area:55},
      ],
      constraints:{
        corridor_width:1500,
        max_unit_depth:8000,
        min_entrance_overlap:1200,
        min_common_area:20,
        max_travel_distance:30000,
        core_type:'stair_lift',
      },
    },
    single_loaded:{
      label:'single-loaded slab',
      boundary:[[0,0],[10000,0],[10000,25000],[0,25000]],
      unit_mix:[
        {type:'1bed',count:3,target_area:55},
        {type:'2bed',count:3,target_area:70},
      ],
      constraints:{
        corridor_width:1500,
        max_unit_depth:8000,
        min_entrance_overlap:1200,
        min_common_area:20,
        max_travel_distance:30000,
        core_type:'stair_lift',
      },
    },
    long_slab:{
      label:'long slab (multi-core)',
      boundary:[[0,0],[14000,0],[14000,80000],[0,80000]],
      unit_mix:[
        {type:'2bed',count:8,target_area:70},
      ],
      constraints:{
        corridor_width:1500,
        max_unit_depth:8000,
        min_entrance_overlap:1200,
        min_common_area:20,
        max_travel_distance:30000,
        core_type:'stair_lift',
      },
    },
    point_block:{
      label:'point block',
      boundary:[[0,0],[18000,0],[18000,20000],[0,20000]],
      unit_mix:[
        {type:'2bed',count:4,target_area:70},
      ],
      constraints:{
        corridor_width:1500,
        max_unit_depth:9000,
        min_entrance_overlap:1200,
        min_common_area:20,
        max_travel_distance:30000,
        core_type:'stair_lift',
      },
    },
    mixed_units:{
      label:'mixed units',
      boundary:[[0,0],[12000,0],[12000,18000],[0,18000]],
      unit_mix:[
        {type:'1bed',count:2,target_area:55},
        {type:'2bed',count:2,target_area:70},
        {type:'3bed',count:1,target_area:90},
      ],
      constraints:{
        corridor_width:1500,
        max_unit_depth:8500,
        min_entrance_overlap:1200,
        min_common_area:20,
        max_travel_distance:30000,
        core_type:'stair_lift',
      },
    },
  };
}

function getDistributionSummaryLines(floorPlate){
  const breakdown=floorPlate?.analysis?.unit_distribution_breakdown;
  if(!breakdown)return [];

  const scenario=breakdown.scenario||'unknown';
  const accommodated=Number.isFinite(breakdown.accommodated_units)
    ? breakdown.accommodated_units
    : 'n/a';
  const splitStrategy=breakdown.split_strategy||'n/a';
  return [
    `distribution: ${scenario}`,
    `accommodated units: ${accommodated} | split strategy: ${splitStrategy}`,
  ];
}

async function runGeneratedFloorPlateTest(testId){
  const outputEl=document.getElementById('fp-dev-generated-tests');
  const tests=getGeneratedFloorPlateTests();
  const test=tests[testId];
  if(!test){
    if(outputEl)outputEl.textContent=`Unknown generated test: ${testId}`;
    return null;
  }

  if(outputEl)outputEl.textContent=`Running ${test.label}...`;
  try{
    const resp=await fetch('/generate_floor_plate',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        boundary:test.boundary,
        unit_mix:test.unit_mix,
        constraints:test.constraints,
      }),
    });
    const payload=await resp.json();
    if(!resp.ok||payload?.status!=='ok'||!payload?.floor_plate){
      const errMsg=payload?.error||`HTTP ${resp.status}`;
      if(outputEl)outputEl.textContent=`${test.label}: FAILED\n${errMsg}`;
      return {id:testId,label:test.label,status:'error',error:errMsg};
    }

    floorPlateState={floor_plate:payload.floor_plate};
    devMode.generatedAlternatives=[{
      id:'single_generated',
      label:'single generated',
      request:{boundary:drawnBoundaryMm,unit_mix:unitMixApi},
      floor_plate:payload.floor_plate,
      quality:{
        score:0,
        placedTotal:(payload.floor_plate.units||[]).length,
        requestedTotal:unitMixApi.reduce((acc,item)=>acc+(item.count||0),0),
        shortfall:0,
        unresolvedAreaM2:unresolvedAreaM2(payload.floor_plate.unresolved_regions||[]),
      },
    }];
    devMode.activeAlternativeIndex=0;
    devMode.activeFixture=null;
    devMode.lastToolOutput={
      source:'generated_pipeline_test',
      testId,
      label:test.label,
      result:{
        typology:payload.floor_plate.typology,
        cores:(payload.floor_plate.cores||[]).length,
        corridors:(payload.floor_plate.corridors||[]).length,
        units:(payload.floor_plate.units||[]).length,
        unresolved:(payload.floor_plate.unresolved_regions||[]).length,
        violations:payload.floor_plate.compliance_snapshot?.violations||[],
      },
      generatedAt:new Date().toISOString(),
    };
    runRequirementsCompliance();
    syncDevModePanel();
    render();

    const violations=payload.floor_plate.compliance_snapshot?.violations||[];
    const typologyType=payload.floor_plate.typology?.type||'unknown';
    const typologySub=payload.floor_plate.typology?.sub_type||'unknown';
    const distributionBreakdown=payload.floor_plate.analysis?.unit_distribution_breakdown||null;
    const distributionLines=getDistributionSummaryLines(payload.floor_plate);
    if(outputEl){
      outputEl.textContent=`${test.label}: OK\n`+
        `typology: ${typologyType}/${typologySub}\n`+
        `${distributionLines.length?`${distributionLines.join('\n')}\n`:''}`+
        `cores: ${(payload.floor_plate.cores||[]).length} | corridors: ${(payload.floor_plate.corridors||[]).length} | units: ${(payload.floor_plate.units||[]).length}\n`+
        `violations: ${violations.length}`;
    }
    return {
      id:testId,
      label:test.label,
      status:'ok',
      typology:`${typologyType}/${typologySub}`,
      cores:(payload.floor_plate.cores||[]).length,
      corridors:(payload.floor_plate.corridors||[]).length,
      units:(payload.floor_plate.units||[]).length,
      violations:violations.length,
      distributionScenario:distributionBreakdown?.scenario||null,
    };
  }catch(error){
    const msg=error?.message||String(error);
    if(outputEl)outputEl.textContent=`${test.label}: FAILED\n${msg}`;
    return {id:testId,label:test.label,status:'error',error:msg};
  }
}

async function runAllGeneratedFloorPlateTests(){
  const outputEl=document.getElementById('fp-dev-generated-tests');
  const testIds=['double_loaded','single_loaded','long_slab','point_block','mixed_units'];
  const results=[];
  for(const testId of testIds){
    const result=await runGeneratedFloorPlateTest(testId);
    if(result)results.push(result);
  }

  if(!outputEl)return;
  const okCount=results.filter((row)=>row.status==='ok').length;
  outputEl.textContent=[
    `run all complete: ${okCount}/${results.length} passed`,
    ...results.map((row)=>row.status==='ok'
      ? `- ${row.label}: ok | ${row.typology} | units=${row.units} | violations=${row.violations}${row.distributionScenario?` | scenario=${row.distributionScenario}`:''}`
      : `- ${row.label}: failed | ${row.error||'unknown error'}`),
  ].join('\n');
}

function getDrawnBoundaryInMillimetres(){
  if(!closed||!Array.isArray(boundary)||boundary.length<3)return null;
  const mmPerUnit=METRES_PER_UNIT*1000;
  return boundary.map(({x,y})=>[x*mmPerUnit,y*mmPerUnit]);
}

function getApiUnitMixFromDevState(){
  const unitMix=getDevUnitMix();
  const specs=[
    {count:unitMix.bed1,type:'1bed',target_area:55},
    {count:unitMix.bed2,type:'2bed',target_area:70},
    {count:unitMix.bed3,type:'3bed',target_area:90},
  ];
  return specs.filter((item)=>item.count>0);
}

function convertPolygonMmToM(polygonMm){
  return (polygonMm||[]).map(([x,y])=>[x/1000,y/1000]);
}

function convertPolygonMToMm(polygonM){
  return (polygonM||[]).map(([x,y])=>[x*1000,y*1000]);
}

function contractCandidateToOverlayState(candidate, boundaryMm){
  const response=candidate?.response||candidate?.best||candidate||{};
  const placedUnits=(response.placed_units||[]).map((unit)=>({
    id:unit.id||unit.type,
    type:unit.type,
    polygon:convertPolygonMToMm(unit.polygon),
    typed_edges:Array.isArray(unit.typed_edges)?unit.typed_edges:[],
    entrance_point:Array.isArray(unit.entrance_point)
      ?[unit.entrance_point[0]*1000,unit.entrance_point[1]*1000]
      :(unit.polygon?.[0]||[0,0]).slice(),
    depth:(Number.isFinite(unit.depth_m)?unit.depth_m:0)*1000,
    area:Number.isFinite(unit.area_m2)?unit.area_m2:0,
  }));
  const residualRegions=(response.residual_zones||[]).map((zone)=>convertPolygonMToMm(zone.polygon));
  const cores=(response.cores||[]).map((core)=>({
    id:core.id||'core-0',
    type:core.type||'stair_lift',
    polygon:convertPolygonMToMm(core.polygon),
    centroid:core.centroid?[core.centroid[0]*1000,core.centroid[1]*1000]:[0,0],
    width:(core.width||0)*1000,
    depth:(core.depth||0)*1000,
  }));
  const corridors=(response.corridors||[]).map((corr)=>({
    id:corr.id||'corr-0',
    width_min:(corr.width||0)*1000,
    polygon:convertPolygonMToMm(corr.polygon),
  }));

  return {
    floor_plate:{
      id:'fp-contract-subdivision',
      boundary:boundaryMm,
      typology:{type:'subdivision',sub_type:'contract',confidence:'high',reasoning:'Subdivision contract result'},
      cores,
      corridors,
      units:placedUnits,
      unresolved_regions:residualRegions,
      analysis:{
        unit_distribution_breakdown:{
          mode:response.mode,
          score:response.score,
          area_check:response.area_check,
          shortfall:response.shortfall,
        },
      },
      generation_log:response.warnings||[],
    },
  };
}

function summarizeContractCandidate(candidate){
  const response=candidate?.response||candidate?.best||candidate||{};
  const placed=response.placed_units||[];
  const residual=response.residual_zones||[];
  return {
    score:Number.isFinite(response.score)?response.score:0,
    placedCount:placed.length,
    residualCount:residual.length,
    shortfall:response.shortfall||{},
    warnings:response.warnings||[],
  };
}

function cloneUnitMixApi(unitMixApi){
  return (unitMixApi||[]).map((item)=>({
    type:item.type,
    count:item.count,
    target_area:item.target_area,
  }));
}

function unitMixApiToContractMap(unitMixApi){
  const mapped={};
  (unitMixApi||[]).forEach((item)=>{
    const type=(item?.type||'').trim();
    const count=Math.max(0,parseInt(item?.count,10)||0);
    if(!type||count<=0)return;
    mapped[type]=(mapped[type]||0)+count;
  });
  return mapped;
}

function sumBoundaryAreaM2FromMm(boundaryMm){
  return polygonAreaRaw(boundaryMm)/1_000_000;
}

function scaleUnitMixTargetArea(unitMixApi,factor){
  return cloneUnitMixApi(unitMixApi).map((item)=>({
    ...item,
    target_area:Math.max(20,Math.round((item.target_area||55)*factor)),
  }));
}

function sortUnitMixByTypePriority(unitMixApi,mode){
  const priority=(type)=>{
    if(mode==='larger_first'){
      if(type==='3bed')return 0;
      if(type==='2bed')return 1;
      if(type==='1bed')return 2;
      return 3;
    }
    if(mode==='smaller_first'){
      if(type==='1bed')return 0;
      if(type==='2bed')return 1;
      if(type==='3bed')return 2;
      return 3;
    }
    return 0;
  };
  return cloneUnitMixApi(unitMixApi).sort((a,b)=>priority(a.type)-priority(b.type));
}

function buildSubdivisionAlternativeRequests(boundaryMm,unitMixApi,baseConstraints={}){
  const defaults={
    corridor_width:1500,
    max_unit_depth:8000,
    min_entrance_overlap:1200,
    min_common_area:20,
    max_travel_distance:30000,
    core_type:'stair_lift',
  };
  const merged={...defaults,...baseConstraints,subdivision_only:true};
  return [
    {
      id:'base',
      label:'base subdivision',
      boundary:boundaryMm,
      unit_mix:cloneUnitMixApi(unitMixApi),
      constraints:{...merged,min_unit_depth:7000},
    },
    {
      id:'compact_depth',
      label:'compact depth bias',
      boundary:boundaryMm,
      unit_mix:cloneUnitMixApi(unitMixApi),
      constraints:{...merged,min_unit_depth:6200},
    },
    {
      id:'larger_first',
      label:'larger units first',
      boundary:boundaryMm,
      unit_mix:sortUnitMixByTypePriority(unitMixApi,'larger_first'),
      constraints:{...merged,min_unit_depth:7000},
    },
    {
      id:'smaller_first',
      label:'smaller units first',
      boundary:boundaryMm,
      unit_mix:sortUnitMixByTypePriority(unitMixApi,'smaller_first'),
      constraints:{...merged,min_unit_depth:7000},
    },
    {
      id:'tight_targets',
      label:'tight target areas',
      boundary:boundaryMm,
      unit_mix:scaleUnitMixTargetArea(unitMixApi,0.9),
      constraints:{...merged,min_unit_depth:6800},
    },
    {
      id:'generous_targets',
      label:'generous target areas',
      boundary:boundaryMm,
      unit_mix:scaleUnitMixTargetArea(unitMixApi,1.1),
      constraints:{...merged,min_unit_depth:7400},
    },
  ];
}

function summarizeRequestedCounts(unitMixApi){
  const counts={};
  (unitMixApi||[]).forEach((item)=>{
    const type=item.type||'unknown';
    counts[type]=(counts[type]||0)+Math.max(0,parseInt(item.count,10)||0);
  });
  return counts;
}

function summarizePlacedCounts(units){
  const counts={};
  (units||[]).forEach((unit)=>{
    const type=unit.type||'unknown';
    counts[type]=(counts[type]||0)+1;
  });
  return counts;
}

function unresolvedAreaM2(unresolvedRegions){
  return (unresolvedRegions||[]).reduce((total,poly)=>total+(polygonAreaRaw(poly)/1_000_000),0);
}

function scoreSubdivisionAlternative(payload,request){
  const floorPlate=payload?.floor_plate||{};
  const units=floorPlate.units||[];
  const requestedByType=summarizeRequestedCounts(request.unit_mix||[]);
  const placedByType=summarizePlacedCounts(units);
  const requestedTotal=Object.values(requestedByType).reduce((a,b)=>a+b,0);
  const placedTotal=units.length;

  let shortfall=0;
  Object.entries(requestedByType).forEach(([type,requested])=>{
    shortfall+=Math.max(0,requested-(placedByType[type]||0));
  });

  const unitsArea=(units||[]).reduce((acc,unit)=>acc+(Number.isFinite(unit.area)?unit.area:0),0);
  const boundaryAreaM2=sumBoundaryAreaM2FromMm(request.boundary||[]);
  const unresolvedM2=unresolvedAreaM2(floorPlate.unresolved_regions);
  const violations=(floorPlate.compliance_snapshot?.violations||[]).length;
  const subdivisionOnlyUsed=!(floorPlate.cores||[]).length&&!(floorPlate.corridors||[]).length;

  const score=
    (placedTotal*100)
    -(shortfall*140)
    -(Math.max(0,boundaryAreaM2-unitsArea)*0.35)
    -(unresolvedM2*1.5)
    -(violations*15)
    +(subdivisionOnlyUsed?30:-80);

  return {
    score,
    placedTotal,
    requestedTotal,
    shortfall,
    requestedByType,
    placedByType,
    unitsAreaM2:unitsArea,
    boundaryAreaM2,
    unresolvedAreaM2:unresolvedM2,
    violations,
    subdivisionOnlyUsed,
  };
}

async function generateSubdivisionAlternatives(boundaryMm,unitMixApi,baseConstraints={}){
  const requests=buildSubdivisionAlternativeRequests(boundaryMm,unitMixApi,baseConstraints);
  const alternatives=[];
  for(const request of requests){
    try{
      const resp=await fetch('/subdivide_floor_plate',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
          boundary:request.boundary.map(([x,y])=>[x/1000,y/1000]),
          unit_mix:unitMixApiToContractMap(request.unit_mix),
          constraints:request.constraints,
          scoring:request.scoring||undefined,
          mode:'harness',
          seed:7,
        }),
      });
      const payload=await resp.json();
      if(!resp.ok||payload?.status!=='ok'||!payload?.response)continue;
      const quality=summarizeContractCandidate(payload.response);
      alternatives.push({
        id:request.id,
        label:request.label,
        request,
        response:payload.response,
        overlayState:contractCandidateToOverlayState(payload.response,request.boundary),
        quality,
      });
    }catch(_err){
      // Keep exploring remaining alternatives.
    }
  }

  alternatives.sort((a,b)=>b.quality.score-a.quality.score);
  return alternatives;
}

function renderFloorPlateAlternativesPanel(alternatives,activeIndex=0){
  const listEl=document.getElementById('results-list');
  const countEl=document.getElementById('result-count');
  if(!listEl)return;

  listEl.innerHTML='';
  if(!Array.isArray(alternatives)||!alternatives.length){
    listEl.innerHTML='<div id="results-empty">no subdivision alternatives yet</div>';
    if(countEl)countEl.textContent='0 results';
    return;
  }

  if(countEl)countEl.textContent=`${alternatives.length} alternatives`;
  alternatives.forEach((alt,index)=>{
    const card=document.createElement('div');
    card.className=`result-card${index===activeIndex?' selected':''}`;
    card.onclick=()=>{
      devMode.activeAlternativeIndex=index;
      floorPlateState=alt.overlayState;
      runRequirementsCompliance();
      renderFloorPlateAlternativesPanel(alternatives,index);
      syncDevModePanel();
      render();
    };

    const tc=document.createElement('canvas');
    tc.className='card-thumb';
    tc.width=200;
    tc.height=200;
    const tctx=tc.getContext('2d');
    tctx.fillStyle='#f1f5f9';
    tctx.fillRect(0,0,200,200);
    renderFloorPlate(tctx,alt.overlayState,200,200,{
      showElementIds:false,
      showEdgeIndices:false,
      floorPlateLayers:{
        boundary:true,
        cores:false,
        commonAreas:false,
        unitBoundaries:true,
        grid:false,
        offsetPreview:false,
      },
      offsetSubdivision:{latestRegions:null},
    });
    card.appendChild(tc);

    const meta=document.createElement('div');
    meta.className='card-meta';
    meta.innerHTML=
      `<div class="card-title">#${index+1} ${alt.label}</div>`+
      `<div class="card-dist">score: ${alt.quality.score.toFixed(1)} | shortfall: ${JSON.stringify(alt.quality.shortfall)}</div>`+
      `<div class="card-rooms">units ${alt.quality.placedCount} | residual ${alt.quality.residualCount}</div>`;
    card.appendChild(meta);
    listEl.appendChild(card);
  });
}

async function runGeneratedAlternativesFromDrawnBoundary(){
  const outputEl=document.getElementById('fp-dev-generated-tests');
  const drawnBoundaryMm=getDrawnBoundaryInMillimetres();
  if(!drawnBoundaryMm){
    if(outputEl)outputEl.textContent='Draw and close a boundary on the canvas first, then run alternatives.';
    return;
  }

  const unitMixApi=getApiUnitMixFromDevState();
  if(!unitMixApi.length){
    if(outputEl)outputEl.textContent='Set at least one unit type count above zero before running alternatives.';
    return;
  }

  if(outputEl)outputEl.textContent='Running subdivision alternatives harness...';
  const alternatives=await generateSubdivisionAlternatives(drawnBoundaryMm,unitMixApi,{
    corridor_width:1500,
    max_unit_depth:8000,
    min_entrance_overlap:1200,
    min_common_area:20,
    max_travel_distance:30000,
    core_type:'stair_lift',
    subdivision_only:true,
  });

  if(!alternatives.length){
    if(outputEl)outputEl.textContent='alternatives harness failed: no valid candidates returned by /subdivide_floor_plate';
    return;
  }

  devMode.generatedAlternatives=alternatives;
  devMode.activeAlternativeIndex=0;
  floorPlateState=alternatives[0].overlayState;
  devMode.lastToolOutput={
    source:'generated_subdivision_alternatives',
    candidateCount:alternatives.length,
    best:alternatives[0]?.quality||null,
    generatedAt:new Date().toISOString(),
  };

  runRequirementsCompliance();
  renderFloorPlateAlternativesPanel(alternatives,0);
  syncDevModePanel();
  render();

  if(outputEl){
    outputEl.textContent=[
      `subdivision alternatives: ${alternatives.length} candidates`,
      ...alternatives.slice(0,5).map((alt,idx)=>
        `${idx+1}. ${alt.label} | score=${alt.quality.score.toFixed(1)} | units=${alt.quality.placedCount} | residual=${alt.quality.residualCount}`),
    ].join('\n');
  }
}

async function runAllSubdivisionAlternativeTests(){
  const outputEl=document.getElementById('fp-dev-generated-tests');
  const tests=getGeneratedFloorPlateTests();
  const testIds=Object.keys(tests);
  const rows=[];

  if(outputEl)outputEl.textContent='Running subdivision harness tests...';
  for(const testId of testIds){
    const test=tests[testId];
    const alternatives=await generateSubdivisionAlternatives(test.boundary,test.unit_mix,test.constraints||{});
    if(!alternatives.length){
      rows.push(`- ${test.label}: failed (no candidates)`);
      continue;
    }
    const best=alternatives[0];
    rows.push(`- ${test.label}: ok | best=${best.label} | score=${best.quality.score.toFixed(1)} | units=${best.quality.placedCount} | residual=${best.quality.residualCount}`);
  }

  if(outputEl){
    outputEl.textContent=['subdivision harness tests complete',...rows].join('\n');
  }
}

function polygonAreaRaw(poly){
  if(!Array.isArray(poly)||poly.length<3)return 0;
  let sum=0;
  for(let i=0;i<poly.length;i++){
    const [x1,y1]=poly[i];
    const [x2,y2]=poly[(i+1)%poly.length];
    sum+=x1*y2-x2*y1;
  }
  return Math.abs(sum)*0.5;
}

function estimateBoundaryAreaM2(boundaryPoly){
  const rawArea=polygonAreaRaw(boundaryPoly);
  if(rawArea<=0)return 0;
  // Drawn boundary is in editor units; generated floor plate boundary is in mm.
  if(rawArea>100000)return rawArea/1_000_000;
  return rawArea*METRES_PER_UNIT*METRES_PER_UNIT;
}

function getCapacityEstimate(){
  const boundaryPoly=getDevBoundaryPolygon();
  const areaM2=estimateBoundaryAreaM2(boundaryPoly);
  const mix=getDevUnitMix();

  const requestedCount=mix.bed1+mix.bed2+mix.bed3;
  const requestedMinAreaM2=
    mix.bed1*MIN_UNIT_AREA_M2.bed1+
    mix.bed2*MIN_UNIT_AREA_M2.bed2+
    mix.bed3*MIN_UNIT_AREA_M2.bed3;

  const maxByType={
    studio:Math.floor(areaM2/MIN_UNIT_AREA_M2.studio),
    bed1:Math.floor(areaM2/MIN_UNIT_AREA_M2.bed1),
    bed2:Math.floor(areaM2/MIN_UNIT_AREA_M2.bed2),
    bed3:Math.floor(areaM2/MIN_UNIT_AREA_M2.bed3),
  };

  const weightedMinArea=requestedCount>0
    ? requestedMinAreaM2/requestedCount
    : MIN_UNIT_AREA_M2.bed1;
  const maxUnitsForRequestedMix=Math.floor(areaM2/Math.max(weightedMinArea,1));

  return {
    areaM2,
    requestedCount,
    requestedMinAreaM2,
    maxByType,
    maxUnitsForRequestedMix,
    fitsRequested:requestedCount===0?false:requestedCount<=maxUnitsForRequestedMix,
  };
}

function updateCapacityLabel(){
  const labelEl=document.getElementById('fp-dev-capacity-label');
  if(!labelEl)return;
  const est=getCapacityEstimate();

  if(!closed||boundary.length<3){
    labelEl.textContent='Capacity check: draw and close a boundary to estimate how many units fit by minimum area.';
    return;
  }

  labelEl.textContent=
    `Capacity check (area-only): ${est.areaM2.toFixed(1)} m2 | requested ${est.requestedCount} units requires ${est.requestedMinAreaM2.toFixed(1)} m2 | `+
    `max for current mix ≈ ${est.maxUnitsForRequestedMix} | `+
    `max by type: studio ${est.maxByType.studio}, 1-bed ${est.maxByType.bed1}, 2-bed ${est.maxByType.bed2}, 3-bed ${est.maxByType.bed3} | `+
    `${est.fitsRequested?'requested mix fits':'requested mix exceeds area capacity'}`;
}

async function runGeneratedFromDrawnBoundary(){
  const outputEl=document.getElementById('fp-dev-generated-tests');
  const drawnBoundaryMm=getDrawnBoundaryInMillimetres();
  if(!drawnBoundaryMm){
    if(outputEl)outputEl.textContent='Draw and close a boundary on the canvas first (Enter, double-click, or click first vertex), then run generation.';
    return;
  }

  const unitMixApi=getApiUnitMixFromDevState();
  if(!unitMixApi.length){
    if(outputEl)outputEl.textContent='Set at least one unit type count above zero before generating from drawn boundary.';
    return;
  }

  const capacity=getCapacityEstimate();
  const capacityWarning=!capacity.fitsRequested
    ?`⚠ area may be tight (${capacity.areaM2.toFixed(1)} m2 available, ${capacity.requestedMinAreaM2.toFixed(1)} m2 required min) — proceeding anyway\n`
    :'';

  if(outputEl)outputEl.textContent=capacityWarning+'Running generation from drawn boundary...';
  try{
    const resp=await fetch('/subdivide_floor_plate',{ 
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        boundary:drawnBoundaryMm.map(([x,y])=>[x/1000,y/1000]),
        unit_mix:unitMixApiToContractMap(unitMixApi),
        constraints:{
          min_depth_m:5.5,
          min_area_m2:35,
          area_tolerance_m2:0.01,
          max_candidates:8,
          corridor_width:1200,
          core_type:'stair_lift',
          unit_specs:{
            '1bed':{target_area_m2:55,min_area_m2:35,min_depth_m:5.5},
            '2bed':{target_area_m2:70,min_area_m2:50,min_depth_m:5.5},
            '3bed':{target_area_m2:90,min_area_m2:65,min_depth_m:5.5},
          },
        },
        scoring:{
          placed_unit_reward:10,
          type_priority:{'1bed':1,'2bed':2,'3bed':3},
          shortfall_penalty:8,
          residual_area_penalty:2,
          residual_fragmentation_penalty:3,
          area_deviation_penalty:1,
          compactness_penalty:0.5,
        },
        mode:'single',
        seed:7,
      }),
    });
    const payload=await resp.json();
    if(!resp.ok||payload?.status!=='ok'||!payload?.response){
      const errMsg=payload?.error||`HTTP ${resp.status}`;
      if(outputEl)outputEl.textContent=`drawn-boundary generation: FAILED\n${errMsg}`;
      return;
    }

    const overlayState=contractCandidateToOverlayState(payload.response,drawnBoundaryMm);
    floorPlateState=overlayState;
    devMode.generatedAlternatives=[{
      id:'single_generated',
      label:'single generated',
      request:{boundary:drawnBoundaryMm,unit_mix:unitMixApi},
      response:payload.response,
      overlayState,
      quality:summarizeContractCandidate(payload.response),
    }];
    devMode.activeAlternativeIndex=0;
    devMode.activeFixture=null;
    devMode.lastToolOutput={
      source:'generated_from_drawn_boundary',
      boundaryVertices:drawnBoundaryMm.length,
      unitMix:unitMixApi,
      result:payload.response,
      generatedAt:new Date().toISOString(),
    };

    runRequirementsCompliance();
    renderFloorPlateAlternativesPanel(devMode.generatedAlternatives,0);
    syncDevModePanel();
    render();

    const areaCheck=payload.response.area_check||{};
    if(outputEl){
      outputEl.textContent=`drawn-boundary generation: OK\n`+
        `mode: ${payload.response.mode}\n`+
        `placed_units: ${(payload.response.placed_units||[]).length} | residual_zones: ${(payload.response.residual_zones||[]).length}\n`+
        `area discrepancy: ${(areaCheck.discrepancy_m2||0).toFixed(3)} m2 | passed: ${!!areaCheck.passed}\n`+
        `score: ${Number.isFinite(payload.response.score)?payload.response.score.toFixed(2):'n/a'}`;
    }
  }catch(error){
    const msg=error?.message||String(error);
    if(outputEl)outputEl.textContent=`drawn-boundary generation: FAILED\n${msg}`;
  }
}

async function retrieveAllFloorPlateUnits(){
  const fp=floorPlateState?.floor_plate;
  const units=fp?.units;
  const logEl=document.getElementById('fp-dev-retrieve-units-log');
  if(!units?.length){
    if(logEl)logEl.textContent='No floor plate units found. Load a floor plate first.';
    return;
  }
  if(logEl)logEl.textContent=`Retrieving plans for ${units.length} units…`;
  const MM_PER_UNIT=METRES_PER_UNIT*1000;
  const unitPlans={};
  let done=0;
  const errors=[];
  await Promise.all(units.map(async(unit,unitIdx)=>{
    const polyMm=unit.polygon;
    if(!Array.isArray(polyMm)||polyMm.length<3){
      errors.push(`${unit.id}: bad polygon`);
      return;
    }
    const boundary_=polyMm.map(([x,y])=>[x/MM_PER_UNIT,y/MM_PER_UNIT,0,0]);
    const entMm=unit.entrance_point;
    const entrance_point=Array.isArray(entMm)?[entMm[0]/MM_PER_UNIT,entMm[1]/MM_PER_UNIT]:null;
    const typeStr=unit.type||'';
    const bedrooms=typeStr.startsWith('3')?3:typeStr.startsWith('2')?2:1;
    const body={
      boundary:boundary_,
      entrance_point,
      requirements:{bedrooms,bathrooms:1},
      topn:3,
      shape_w:params.shape_w,
      ent_w:params.ent_w,
      room_w:params.room_w,
    };
    console.log('[unit-retrieve]',unit.id,'boundary[0]:',boundary_[0],'bedrooms:',bedrooms);
    try{
      const {data}=await postJsonWithTimeout('/retrieve',body,{timeoutMs:15000});
      console.log('[unit-retrieve]',unit.id,'response:',JSON.stringify(data).slice(0,200));
      if(data.error)throw new Error(data.error);
      if(!data.plans?.length){errors.push(`${unit.id}: 0 plans returned`);return;}
      const normBdry=polyMm.map(([x,y])=>[x/MM_PER_UNIT,y/MM_PER_UNIT]);
      // Step 4 — viability rerank (docs/arrangement-refactor-build-plan.md).
      // Instead of committing to retrieval's rank-1 (data.plans[0]) blind, build
      // every top candidate in BOTH mirror states and let the backend adapt and
      // score each by the arrangement viability score, then keep the most viable.
      // The mirror has to be baked into the plan geometry before the Python side
      // sees it — the fitting side only rotates in 90° steps with no reflection
      // of its own (see mirrorPlanY) — and the boundary/typed_edges are NOT
      // mirrored (only the plan), so edge_index stays valid against normBdry.
      // This is the agent's first arrangement-level judgement: the silent stage-1
      // commitment becomes an explicit scored choice made before anything
      // irreversible, so a mirrored or lower-ranked candidate carrying fewer
      // daylight/access flags can win.
      const variants=data.plans.flatMap((p,ci)=>(
        p?.rooms?.length?[
          {fp:p,tag:`c${ci}`,candidate:ci,mirror:false},
          {fp:mirrorPlanY(p),tag:`c${ci}·mir`,candidate:ci,mirror:true},
        ]:[]
      ));
      if(!variants.length){errors.push(`${unit.id}: no usable candidates`);return;}
      const {data:rr}=await postJsonWithTimeout('/rerank_unit',{
        variants,
        boundary:normBdry,
        entrance_point,
        boundary_typed_edges:Array.isArray(unit.typed_edges)?unit.typed_edges:[],
      },{timeoutMs:120000});
      if(rr.error)throw new Error(rr.error);
      // The scored table + which variant won. rank-1 in default orientation is
      // rr.table's `c0` row — comparing it to the winner shows when viability
      // overrode descriptor rank (a c1 or a ·mir winning is the point).
      console.log(`[rerank] unit#${unitIdx} (${unit.id}) winner ${rr.winner_tag} ${rr.table?.[0]?.score}`,
        (rr.table||[]).map((t)=>`${t.tag} ${t.score}`).join(' / '));
      console.log('[unit-retrieve]',unit.id,'boundary edge types:',rr.boundary_typed_edges);
      // Winner's viability flags (no-daylight / no-access), keyed to the room by
      // its index in rr.rooms (order preserved by _rooms_to_canvas).
      const viability=(rr.defects||[]).filter((d)=>d.type==='no-daylight'||d.type==='no-access');
      const flagsByIdx={};
      viability.forEach((d)=>{
        const idx=d.rooms?.[0]?.[0];
        if(idx==null)return;
        (flagsByIdx[idx]=flagsByIdx[idx]||[]).push(d.type);
      });
      const wb=rr.winner?.breakdown||{};
      console.log(`[unit-score] unit#${unitIdx} (${unit.id}) ${rr.winner?.score}`,
        `— program ${wb.program}, daylight/access ${wb.daylight_access}, entrance ${wb.entrance}, flags ${wb.flags}`);
      if(viability.length){
        console.log(`[unit-viability] unit#${unitIdx} (${unit.id})`,
          viability.map((d)=>`${d.rooms?.[0]?.[1]}(${d.rooms?.[0]?.[0]}) ${d.type}`).join(', '));
      }
      const rooms=(rr.rooms||[]).map((room,idx)=>({
        ...room,
        polygon:(room.polygon||[]).map(([rx,ry])=>[rx*MM_PER_UNIT,ry*MM_PER_UNIT]),
        flags:flagsByIdx[idx]||[],
      }));
      // Keyed by position, not unit.id — see canvas-renderer.js's matching
      // lookup. The generator repeats unit ids across zones, so id-keying
      // overwrote entries and reported 8/8 retrieved while drawing far fewer.
      const winName=data.plans[rr.winner?.candidate]?.id
        ||data.plans[rr.winner?.candidate]?.retrieval?.source_id||unit.id;
      unitPlans[unitIdx]={name:winName,rooms};
      done++;
    }catch(e){
      console.warn(`[unit-retrieve] failed for unit ${unit.id}:`,e);
      errors.push(`${unit.id}: ${e.message}`);
    }
  }));
  floorPlateState={
    ...floorPlateState,
    floor_plate:{...floorPlateState.floor_plate,unitPlans},
  };
  const msg=`Retrieved ${done}/${units.length} unit plans.`+(errors.length?`\nErrors:\n${errors.join('\n')}`:' ✓');
  if(logEl)logEl.textContent=msg;
  syncDevModePanel();
  render();
}

async function runRetrieveFitHarness(){
  const logEl=document.getElementById('fp-dev-harness-log');
  if(!closed||boundary.length<3){
    if(logEl)logEl.textContent='Draw and close a boundary on the canvas first.';
    return;
  }
  if(logEl)logEl.textContent='Running retrieval…';
  setStatus('retrieving…','warn');

  const bdryArr=boundary.map(({x,y})=>[x,y,0,0]);
  const entArr=entrance?[entrance.x,entrance.y]:null;
  const body={
    boundary:bdryArr,
    entrance_point:entArr,
    requirements:{bedrooms:req.bedrooms,bathrooms:req.bathrooms,balcony:req.balcony,study:req.study,storage:req.storage},
    topn:5,
    shape_w:params.shape_w,
    ent_w:params.ent_w,
    room_w:params.room_w,
    // Step 1 (topology → generate → select): the target's typed edges drive the
    // topology re-rank's perimeter-ring alignment. Sparse; backend defaults rest.
    typed_edges:boundaryTypedEdgesPayload(),
  };

  try{
    const{data}=await postJsonWithTimeout('/retrieve',body,{timeoutMs:15000});
    if(data.error)throw new Error(data.error);
    if(!data.plans?.length)throw new Error('no plans returned by retrieval');

    // Build results state so boundaryInPlanSpace works correctly
    const newResults=data.plans.map((plan)=>{
      const retrieval=plan.retrieval||{};
      return{
        name:plan.id,
        dist:retrieval.dist??0,
        fp:plan,
        adapted:true,
        retrieval,
        rot:retrieval.rot??0,
        mirror:retrieval.mirror??false,
      };
    });
    results=newResults;
    selected=0;
    fitMode=false;
    fittedCache={};

    const first=newResults[0];
    // Step 1 gate — the list is now ordered by TOPOLOGICAL fitness, not shape.
    // Show each candidate's topo score + daylight-fit + per-room daylight so the
    // ranking is inspectable against CC's eye (and against the shape distance it
    // replaced). A candidate with no topo block means the re-rank was off.
    const topoLines=newResults.map((r,i)=>{
      const t=r.retrieval?.topo;
      if(!t)return `#${i+1} ${r.name}  dist ${r.dist.toFixed(3)} (no topo)`;
      const c=t.components||{};
      const per=t.daylight_detail?.per_room||{};
      const perStr=Object.keys(per).length
        ? '   daylight: '+Object.entries(per).map(([k,v])=>`${k} ${v}`).join(', ')+(t.mirror?'  mirror:Y':'')
        : '';
      return `#${i+1} ${r.name}  topo ${t.score.toFixed(3)}  `
        +`[dayfit ${(c.daylight_fit??0).toFixed(2)} · prog ${(c.program??0).toFixed(2)} · ent ${(c.entrance_access??0).toFixed(2)} · priv ${(c.private_access??0).toFixed(2)}]`
        +`  shape-dist ${r.dist.toFixed(3)}`
        +(perStr?`\n${perStr}`:'');
    }).join('\n');
    if(logEl)logEl.textContent=`Topology re-rank (target edge types applied):\n${topoLines}\n\nSelected #1 → switch to Fit tab.`;
    setStatus(`retrieved ${newResults.length} plan${newResults.length!==1?'s':''}  ·  topo-ranked`,'ok');
    console.log(`[retrieve-topo] winner=${first.name} topo=${first.retrieval?.topo?.score} dayfit=${first.retrieval?.topo?.components?.daylight_fit}`,
      newResults.map(r=>({id:r.name,topo:r.retrieval?.topo?.score,dayfit:r.retrieval?.topo?.components?.daylight_fit,shape_dist:+r.dist.toFixed(3)})));
    _fitUpdateLog(`Retrieved (topo-ranked): ${first.name}\nTopo: ${first.retrieval?.topo?.score??'—'}  daylight-fit: ${first.retrieval?.topo?.components?.daylight_fit??'—'}\nRooms: ${first.fp?.rooms?.length??'?'}\n\nSwitch to the Fit tab to begin.`);
    render();
  }catch(err){
    const msg=err?.message||String(err);
    if(logEl)logEl.textContent=`Retrieval failed:\n${msg}`;
    setStatus('retrieval failed','err');
  }
}

function getDevBoundaryPolygon(){
  if(closed&&boundary.length>=3)return boundary.map(({x,y})=>[x,y]);
  if(floorPlateState?.floor_plate?.boundary?.length>=3)return floorPlateState.floor_plate.boundary;
  if(floorPlateState?.boundary?.length>=3)return floorPlateState.boundary;
  return buildFloorPlateFixture('sample').floor_plate.boundary;
}

function getDevUnitMix(){
  const fromState=devMode.unitMix||{};
  return {
    bed1:Math.max(0,parseInt(fromState.bed1,10)||0),
    bed2:Math.max(0,parseInt(fromState.bed2,10)||0),
    bed3:Math.max(0,parseInt(fromState.bed3,10)||0),
  };
}

function defaultDevRequirements(){
  const catalog=get_requirement_check_catalog();
  return catalog.map((item)=>({
    id:item.id,
    label:item.label,
    unit:item.unit,
    value:item.defaultValue,
    enabled:item.id!=='min_unit_area_m2',
  }));
}

function getRequirementMeta(requirementId){
  const catalog=get_requirement_check_catalog();
  return catalog.find((item)=>item.id===requirementId)||null;
}

function normalizeDevRequirements(requirements){
  const fallback=defaultDevRequirements();
  if(!Array.isArray(requirements)||!requirements.length)return fallback;
  return requirements.map((item,index)=>{
    const base=fallback.find((candidate)=>candidate.id===item?.id)||fallback[index]||fallback[0];
    const valueRaw=parseFloat(item?.value);
    return {
      id:base.id,
      label:base.label,
      unit:base.unit,
      value:Number.isFinite(valueRaw)?Math.max(0,valueRaw):base.value,
      enabled:item?.enabled!==false,
    };
  });
}

function ensureDevModeDefaults(){
  devMode.floorPlateLayers={
    boundary:devMode.floorPlateLayers?.boundary!==false,
    cores:devMode.floorPlateLayers?.cores!==false,
    commonAreas:devMode.floorPlateLayers?.commonAreas!==false,
    unitBoundaries:devMode.floorPlateLayers?.unitBoundaries!==false,
    grid:devMode.floorPlateLayers?.grid!==false,
    offsetPreview:devMode.floorPlateLayers?.offsetPreview!==false,
  };
  devMode.offsetSubdivision={
    depthRange:{
      min:Number.isFinite(parseFloat(devMode.offsetSubdivision?.depthRange?.min))
        ?Math.max(0.1,parseFloat(devMode.offsetSubdivision.depthRange.min))
        :5,
      max:Number.isFinite(parseFloat(devMode.offsetSubdivision?.depthRange?.max))
        ?Math.max(0.1,parseFloat(devMode.offsetSubdivision.depthRange.max))
        :9,
    },
    minCorridorWidthM:Number.isFinite(parseFloat(devMode.offsetSubdivision?.minCorridorWidthM))
      ?Math.max(0.1,parseFloat(devMode.offsetSubdivision.minCorridorWidthM))
      :1.2,
    latestRegions:devMode.offsetSubdivision?.latestRegions||null,
  };
  devMode.requirements=normalizeDevRequirements(devMode.requirements);
}

function runRequirementsCompliance(){
  ensureDevModeDefaults();
  const plate=floorPlateState||{floor_plate:{boundary:getDevBoundaryPolygon()}};
  const report=evaluate_floor_plate_requirements(plate,devMode.requirements);
  devMode.lastComplianceReport=report;
  return report;
}

function runSingleRequirementCompliance(requirementId){
  ensureDevModeDefaults();
  const requirement=devMode.requirements.find((item)=>item.id===requirementId);
  if(!requirement)return null;
  const plate=floorPlateState||{floor_plate:{boundary:getDevBoundaryPolygon()}};
  const single=evaluate_floor_plate_requirements(plate,[requirement]);
  const result=single.requirements?.[0];
  if(!result)return null;

  const existing=Array.isArray(devMode.lastComplianceReport?.requirements)
    ? devMode.lastComplianceReport.requirements
    : [];
  const byId=new Map(existing.map((row)=>[row.id,row]));
  byId.set(requirementId,result);

  const rows=devMode.requirements.map((reqItem)=>{
    const known=byId.get(reqItem.id);
    if(known)return known;
    if(reqItem.enabled===false){
      return {
        id:reqItem.id,
        name:reqItem.label,
        expected:'disabled',
        actual:'disabled',
        status:'disabled',
      };
    }
    return {
      id:reqItem.id,
      name:reqItem.label,
      expected:'not run yet',
      actual:'not run yet',
      status:'blocked',
    };
  });

  const enabledRows=rows.filter((row)=>row.status!=='disabled');
  devMode.lastComplianceReport={
    pass:enabledRows.every((row)=>row.status==='pass'),
    summary:{
      enabled:enabledRows.length,
      pass:enabledRows.filter((row)=>row.status==='pass').length,
      fail:enabledRows.filter((row)=>row.status==='fail').length,
      blocked:enabledRows.filter((row)=>row.status==='blocked').length,
      disabled:rows.length-enabledRows.length,
    },
    requirements:rows,
  };
  return result;
}

function runDevStage1Tool(toolName){
  const boundaryPolygon=getDevBoundaryPolygon();
  const unitMix=getDevUnitMix();
  let output;

  if(toolName==='analyse_boundary'){
    output=analyse_boundary(boundaryPolygon);
  }else if(toolName==='evaluate_typology_fit'){
    const typology=(document.getElementById('fp-dev-typology')?.value||'double_loaded_slab').trim();
    output=evaluate_typology_fit(boundaryPolygon,typology,unitMix);
  }else if(toolName==='propose_typology_options'){
    output=propose_typology_options(boundaryPolygon,unitMix);
  }else if(toolName==='check_typology_viability'){
    const typology=(document.getElementById('fp-dev-typology')?.value||'double_loaded_slab').trim();
    output=check_typology_viability(boundaryPolygon,{typology},unitMix);
  }else if(toolName==='generate_offset_regions'){
    output=generate_offset_subdivision_regions(boundaryPolygon,{
      depth_range:devMode.offsetSubdivision?.depthRange,
      min_corridor_width_m:devMode.offsetSubdivision?.minCorridorWidthM,
    });
    devMode.offsetSubdivision.latestRegions=output;
  }

  devMode.lastToolOutput={
    tool:toolName,
    unitMix,
    boundaryVertices:boundaryPolygon.length,
    result:output,
  };
  runRequirementsCompliance();
  syncDevModePanel();
  render();
}

function setupDevModeUI(){
  if(devModeSetupDone)return;
  devModeSetupDone=true;
  ensureDevModeDefaults();
  if(!devMode.activeDevTab)devMode.activeDevTab='tools';

  const wrap=document.getElementById('canvas-wrap');
  if(!wrap)return;

  const style=document.createElement('style');
  style.textContent=`
    #fp-dev-panel {
      position:absolute;
      top:10px;
      left:10px;
      width:320px;
      max-height:calc(100% - 20px);
      overflow:auto;
      border:1px solid #3a4155;
      border-radius:8px;
      background:rgba(15,17,23,0.94);
      color:#e2e8f0;
      font-family:var(--mono);
      font-size:10px;
      z-index:15;
      display:none;
      padding:8px;
      backdrop-filter:blur(4px);
    }
    #fp-dev-panel.show { display:block; }
    .fp-dev-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
    .fp-dev-title { font-size:10px; letter-spacing:0.08em; text-transform:uppercase; color:#93c5fd; }
    .fp-dev-section { border:1px solid #2a2f3f; border-radius:6px; padding:6px; margin-bottom:8px; background:rgba(30,36,54,0.35); }
    .fp-dev-section-title { font-size:9px; letter-spacing:0.08em; text-transform:uppercase; color:#b6babe; margin-bottom:6px; }
    .fp-dev-tabs { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:4px; margin-bottom:8px; }
    .fp-dev-tab {
      border:1px solid #3a4155;
      border-radius:4px;
      background:#0f172a;
      color:#94a3b8;
      font-family:var(--mono);
      font-size:10px;
      padding:4px 6px;
      cursor:pointer;
    }
    .fp-dev-tab.active { border-color:#2563eb; color:#dbeafe; background:#1e293b; }
    .fp-dev-pane { display:none; }
    .fp-dev-pane.active { display:block; }
    .fp-dev-row { display:flex; gap:6px; align-items:center; margin-bottom:6px; }
    .fp-dev-row:last-child { margin-bottom:0; }
    .fp-dev-row-wrap { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:6px; }
    .fp-dev-grid2 { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
    .fp-dev-btn {
      border:1px solid #3a4155;
      border-radius:4px;
      background:#1e2436;
      color:#e2e8f0;
      font-family:var(--mono);
      font-size:10px;
      padding:4px 6px;
      cursor:pointer;
    }
    .fp-dev-btn:hover { border-color:#3b82f6; }
    .fp-dev-input {
      border:1px solid #3a4155;
      border-radius:4px;
      background:#111827;
      color:#e2e8f0;
      font-family:var(--mono);
      font-size:10px;
      padding:4px;
      width:52px;
    }
    .fp-dev-select {
      border:1px solid #3a4155;
      border-radius:4px;
      background:#111827;
      color:#e2e8f0;
      font-family:var(--mono);
      font-size:10px;
      padding:4px;
      width:100%;
    }
    .fp-dev-pre {
      white-space:pre-wrap;
      word-break:break-word;
      border:1px solid #2a2f3f;
      border-radius:4px;
      background:rgba(15,17,23,0.8);
      color:#cbd5e1;
      padding:6px;
      max-height:180px;
      overflow:auto;
      line-height:1.35;
    }
    .fp-dev-req-row {
      display:grid;
      grid-template-columns:auto 1fr auto auto auto auto;
      gap:6px;
      align-items:center;
      border-bottom:1px solid #2a2f3f;
      padding:5px 0;
    }
    .fp-dev-req-row:last-child { border-bottom:none; }
    .fp-dev-req-label { color:#d1d5db; }
    .fp-dev-req-unit { color:#94a3b8; min-width:20px; text-align:right; }
    .fp-dev-mini-btn {
      border:1px solid #3a4155;
      border-radius:4px;
      background:#0f172a;
      color:#cbd5e1;
      font-family:var(--mono);
      font-size:9px;
      padding:2px 5px;
      cursor:pointer;
    }
    .fp-dev-mini-btn:hover { border-color:#3b82f6; color:#eff6ff; }
    #fp-dev-compliance-table { width:100%; border-collapse:collapse; }
    #fp-dev-compliance-table th, #fp-dev-compliance-table td { border-bottom:1px solid #2a2f3f; padding:3px 2px; text-align:left; }
    #fp-dev-compliance-table th { color:#94a3b8; font-weight:500; }
  `;
  document.head.appendChild(style);

  const panel=document.createElement('aside');
  panel.id='fp-dev-panel';
  panel.innerHTML=`
    <div class="fp-dev-head">
      <div class="fp-dev-title">Floor Plate Dev Mode</div>
      <button id="fp-dev-hide" class="fp-dev-btn" type="button">hide</button>
    </div>

    <nav class="fp-dev-tabs" id="fp-dev-tabs">
      <button type="button" class="fp-dev-tab active" data-tab="tools">tests</button>
      <button type="button" class="fp-dev-tab" data-tab="requirements">requirements</button>
      <button type="button" class="fp-dev-tab" data-tab="compliance">compliance</button>
      <button type="button" class="fp-dev-tab" data-tab="state">state</button>
    </nav>

    <section class="fp-dev-section fp-dev-pane active" data-pane="tools">
      <div class="fp-dev-section-title">Generated Pipeline Tests</div>
      <div class="fp-dev-row" style="margin-top:2px">
        <button id="fp-dev-generate-drawn" class="fp-dev-btn" type="button">generate from drawn boundary</button>
      </div>
      <div class="fp-dev-row">
        <label>1-bed</label><input id="fp-dev-mix-bed1" class="fp-dev-input" type="number" min="0" step="1" value="2">
        <label>2-bed</label><input id="fp-dev-mix-bed2" class="fp-dev-input" type="number" min="0" step="1" value="2">
        <label>3-bed</label><input id="fp-dev-mix-bed3" class="fp-dev-input" type="number" min="0" step="1" value="0">
      </div>
      <div class="fp-dev-pre" id="fp-dev-capacity-label" style="margin-top:2px;max-height:none;color:#bfdbfe">Capacity check: draw and close a boundary to estimate how many units fit by minimum area.</div>
      <div class="fp-dev-row-wrap">
        <button id="fp-dev-test-double" class="fp-dev-btn" type="button">run: double-loaded slab</button>
        <button id="fp-dev-test-single" class="fp-dev-btn" type="button">run: single-loaded slab</button>
        <button id="fp-dev-test-long" class="fp-dev-btn" type="button">run: long slab</button>
        <button id="fp-dev-test-point" class="fp-dev-btn" type="button">run: point block</button>
        <button id="fp-dev-test-mixed" class="fp-dev-btn" type="button">run: mixed units</button>
      </div>
      <div class="fp-dev-row" style="margin-top:6px">
        <button id="fp-dev-test-all" class="fp-dev-btn" type="button">run all generated tests</button>
        <button id="fp-dev-test-alt-all" class="fp-dev-btn" type="button">run all harness tests</button>
      </div>
      <div class="fp-dev-pre" id="fp-dev-generated-tests" style="margin-top:6px;max-height:140px">Run a generated test case to call /generate_floor_plate and load its output into the overlay.</div>

      <div class="fp-dev-section-title" style="margin-top:8px">Unit Retrieval</div>
      <div class="fp-dev-row">
        <button id="fp-dev-retrieve-units" class="fp-dev-btn" type="button">retrieve all units</button>
        <button id="fp-dev-clear-unit-plans" class="fp-dev-btn" type="button">clear</button>
      </div>
      <div class="fp-dev-pre" id="fp-dev-retrieve-units-log" style="margin-top:4px;max-height:80px">Load a floor plate first, then retrieve a matching apartment layout for each unit.</div>

      <div class="fp-dev-section-title" style="margin-top:8px">Retrieve + Fit Harness</div>
      <div class="fp-dev-row">
        <button id="fp-dev-retrieve-fit-harness" class="fp-dev-btn" type="button">retrieve + fit (harness)</button>
      </div>
      <div class="fp-dev-pre" id="fp-dev-harness-log" style="margin-top:4px;max-height:120px">Draw and close a boundary on the canvas, then click to retrieve the best matching plan and fit it using the agent harness (clip + fill residual + G1–G5).</div>

      <div class="fp-dev-section-title" style="margin-top:8px">Overlay Controls</div>
      <div class="fp-dev-row">
        <label><input id="fp-dev-overlay" type="checkbox" checked> show floor plate overlay</label>
      </div>
      <div class="fp-dev-row">
        <label><input id="fp-dev-show-ids" type="checkbox" checked> show element IDs</label>
      </div>
      <div class="fp-dev-row">
        <label><input id="fp-dev-show-edges" type="checkbox"> show edge indices</label>
      </div>
      <div class="fp-dev-section-title" style="margin-top:8px">Layer Isolation</div>
      <div class="fp-dev-grid2">
        <label><input id="fp-dev-layer-boundary" type="checkbox" checked> boundary</label>
        <label><input id="fp-dev-layer-cores" type="checkbox" checked> core(s)</label>
        <label><input id="fp-dev-layer-common" type="checkbox" checked> common area(s)</label>
        <label><input id="fp-dev-layer-units" type="checkbox" checked> unit boundaries</label>
      </div>
      <div class="fp-dev-row" style="margin-top:6px">
        <label><input id="fp-dev-layer-grid" type="checkbox" checked> structural grid</label>
      </div>
      <div class="fp-dev-row">
        <label><input id="fp-dev-layer-offset-preview" type="checkbox" checked> offset preview</label>
      </div>
      <div class="fp-dev-row">
        <button id="fp-dev-seed" class="fp-dev-btn" type="button">seed sample floor plate</button>
        <button id="fp-dev-clear" class="fp-dev-btn" type="button">clear floor plate</button>
      </div>
      <div class="fp-dev-pre" id="fp-dev-tools">
Use this panel to run generated pipeline tests and inspect results directly in the app overlay.
      </div>
    </section>

    <section class="fp-dev-section fp-dev-pane" data-pane="requirements">
      <div class="fp-dev-section-title">Requirement Controls</div>
      <div class="fp-dev-pre" style="margin-bottom:6px;max-height:none">
Each requirement can be toggled on/off and its threshold edited.
All checks are evaluated against the current floor plate state.
      </div>
      <div id="fp-dev-req-list"></div>
      <div class="fp-dev-row" style="margin-top:8px">
        <button id="fp-dev-run-req-checks" class="fp-dev-btn" type="button">run all requirement checks</button>
      </div>
      <div id="fp-dev-req-info" class="fp-dev-pre" style="margin-top:6px;max-height:none;color:#bfdbfe">
Select a requirement info button to view how its check is measured.
      </div>
    </section>

    <section class="fp-dev-section fp-dev-pane" data-pane="compliance">
      <div class="fp-dev-section-title">Compliance Log</div>
      <div id="fp-dev-compliance-summary" class="fp-dev-pre" style="margin-bottom:6px;max-height:none"></div>
      <table id="fp-dev-compliance-table">
        <thead>
          <tr><th>Requirement</th><th>Expected</th><th>Actual</th><th>Status</th></tr>
        </thead>
        <tbody id="fp-dev-compliance-body"></tbody>
      </table>
    </section>

    <section class="fp-dev-section fp-dev-pane" data-pane="state" style="margin-bottom:0">
      <div class="fp-dev-section-title">State Inspector</div>
      <div class="fp-dev-pre" id="fp-dev-state"></div>
    </section>
  `;
  wrap.appendChild(panel);

  const hideBtn=document.getElementById('fp-dev-hide');
  const tabsEl=document.getElementById('fp-dev-tabs');
  const overlayToggle=document.getElementById('fp-dev-overlay');
  const showIdsToggle=document.getElementById('fp-dev-show-ids');
  const showEdgesToggle=document.getElementById('fp-dev-show-edges');
  const layerBoundaryToggle=document.getElementById('fp-dev-layer-boundary');
  const layerCoresToggle=document.getElementById('fp-dev-layer-cores');
  const layerCommonToggle=document.getElementById('fp-dev-layer-common');
  const layerUnitsToggle=document.getElementById('fp-dev-layer-units');
  const layerGridToggle=document.getElementById('fp-dev-layer-grid');
  const layerOffsetPreviewToggle=document.getElementById('fp-dev-layer-offset-preview');
  const seedBtn=document.getElementById('fp-dev-seed');
  const clearBtn=document.getElementById('fp-dev-clear');
  const compactBtn=document.getElementById('fp-dev-fixture-compact');
  const elongatedBtn=document.getElementById('fp-dev-fixture-elongated');
  const narrowBtn=document.getElementById('fp-dev-fixture-narrow');
  const runAnalyseBtn=document.getElementById('fp-dev-run-analyse');
  const runEvaluateBtn=document.getElementById('fp-dev-run-evaluate');
  const runProposeBtn=document.getElementById('fp-dev-run-propose');
  const runCheckBtn=document.getElementById('fp-dev-run-check');
  const runOffsetBtn=document.getElementById('fp-dev-run-offset');
  const runReqChecksBtn=document.getElementById('fp-dev-run-req-checks');
  const retrieveUnitsBtn=document.getElementById('fp-dev-retrieve-units');
  const clearUnitPlansBtn=document.getElementById('fp-dev-clear-unit-plans');
  const retrieveFitHarnessBtn=document.getElementById('fp-dev-retrieve-fit-harness');
  const runTestDoubleBtn=document.getElementById('fp-dev-test-double');
  const runTestSingleBtn=document.getElementById('fp-dev-test-single');
  const runTestLongBtn=document.getElementById('fp-dev-test-long');
  const runTestPointBtn=document.getElementById('fp-dev-test-point');
  const runTestMixedBtn=document.getElementById('fp-dev-test-mixed');
  const runTestAllBtn=document.getElementById('fp-dev-test-all');
  const runTestAltAllBtn=document.getElementById('fp-dev-test-alt-all');
  const runGenerateDrawnBtn=document.getElementById('fp-dev-generate-drawn');
  const offsetDepthMinEl=document.getElementById('fp-dev-offset-depth-min');
  const offsetDepthMaxEl=document.getElementById('fp-dev-offset-depth-max');
  const offsetMinCorridorEl=document.getElementById('fp-dev-offset-min-corridor');
  const offsetOutputEl=document.getElementById('fp-dev-offset-output');
  const reqListEl=document.getElementById('fp-dev-req-list');
  const reqInfoEl=document.getElementById('fp-dev-req-info');
  const mixBed1El=document.getElementById('fp-dev-mix-bed1');
  const mixBed2El=document.getElementById('fp-dev-mix-bed2');
  const mixBed3El=document.getElementById('fp-dev-mix-bed3');

  const setDevPanelTab=(tabId)=>{
    devMode.activeDevTab=tabId;
    document.querySelectorAll('#fp-dev-tabs .fp-dev-tab').forEach((btn)=>{
      btn.classList.toggle('active',btn.dataset.tab===tabId);
    });
    document.querySelectorAll('#fp-dev-panel .fp-dev-pane').forEach((pane)=>{
      pane.classList.toggle('active',pane.dataset.pane===tabId);
    });
  };

  if(overlayToggle)overlayToggle.checked=!!devMode.showFloorPlateOverlay;
  if(showIdsToggle)showIdsToggle.checked=!!devMode.showElementIds;
  if(showEdgesToggle)showEdgesToggle.checked=!!devMode.showEdgeIndices;
  if(layerBoundaryToggle)layerBoundaryToggle.checked=devMode.floorPlateLayers.boundary!==false;
  if(layerCoresToggle)layerCoresToggle.checked=devMode.floorPlateLayers.cores!==false;
  if(layerCommonToggle)layerCommonToggle.checked=devMode.floorPlateLayers.commonAreas!==false;
  if(layerUnitsToggle)layerUnitsToggle.checked=devMode.floorPlateLayers.unitBoundaries!==false;
  if(layerGridToggle)layerGridToggle.checked=devMode.floorPlateLayers.grid!==false;
  if(layerOffsetPreviewToggle)layerOffsetPreviewToggle.checked=devMode.floorPlateLayers.offsetPreview!==false;
  if(mixBed1El)mixBed1El.value=String(devMode.unitMix?.bed1??2);
  if(mixBed2El)mixBed2El.value=String(devMode.unitMix?.bed2??2);
  if(mixBed3El)mixBed3El.value=String(devMode.unitMix?.bed3??0);
  if(offsetDepthMinEl)offsetDepthMinEl.value=String(devMode.offsetSubdivision?.depthRange?.min??5);
  if(offsetDepthMaxEl)offsetDepthMaxEl.value=String(devMode.offsetSubdivision?.depthRange?.max??9);
  if(offsetMinCorridorEl)offsetMinCorridorEl.value=String(devMode.offsetSubdivision?.minCorridorWidthM??1.2);

  tabsEl?.addEventListener('click',(event)=>{
    const target=event.target;
    if(!(target instanceof HTMLElement))return;
    const tabId=target.dataset.tab;
    if(!tabId)return;
    setDevPanelTab(tabId);
  });

  hideBtn?.addEventListener('click',()=>{
    devMode.panelVisible=false;
    syncDevModePanel();
  });
  overlayToggle?.addEventListener('change',(e)=>{
    devMode.showFloorPlateOverlay=!!e.target.checked;
    render();
  });
  showIdsToggle?.addEventListener('change',(e)=>{
    devMode.showElementIds=!!e.target.checked;
    render();
  });
  showEdgesToggle?.addEventListener('change',(e)=>{
    devMode.showEdgeIndices=!!e.target.checked;
    render();
  });
  layerBoundaryToggle?.addEventListener('change',(e)=>{
    devMode.floorPlateLayers.boundary=!!e.target.checked;
    render();
  });
  layerCoresToggle?.addEventListener('change',(e)=>{
    devMode.floorPlateLayers.cores=!!e.target.checked;
    render();
  });
  layerCommonToggle?.addEventListener('change',(e)=>{
    devMode.floorPlateLayers.commonAreas=!!e.target.checked;
    render();
  });
  layerUnitsToggle?.addEventListener('change',(e)=>{
    devMode.floorPlateLayers.unitBoundaries=!!e.target.checked;
    render();
  });
  layerGridToggle?.addEventListener('change',(e)=>{
    devMode.floorPlateLayers.grid=!!e.target.checked;
    render();
  });
  layerOffsetPreviewToggle?.addEventListener('change',(e)=>{
    devMode.floorPlateLayers.offsetPreview=!!e.target.checked;
    render();
  });
  [offsetDepthMinEl,offsetDepthMaxEl,offsetMinCorridorEl].forEach((el)=>{
    el?.addEventListener('input',()=>{
      const minRaw=parseFloat(offsetDepthMinEl?.value||'5');
      const maxRaw=parseFloat(offsetDepthMaxEl?.value||'9');
      const minDepth=Math.max(0.1,Number.isFinite(minRaw)?minRaw:5);
      const maxDepth=Math.max(0.1,Number.isFinite(maxRaw)?maxRaw:9);
      devMode.offsetSubdivision.depthRange=minDepth<=maxDepth
        ?{min:minDepth,max:maxDepth}
        :{min:maxDepth,max:minDepth};
      devMode.offsetSubdivision.minCorridorWidthM=Math.max(0.1,parseFloat(offsetMinCorridorEl?.value||'1.2')||1.2);
      syncDevModePanel();
    });
  });
  [mixBed1El,mixBed2El,mixBed3El].forEach((el)=>{
    el?.addEventListener('input',()=>{
      devMode.unitMix={
        bed1:Math.max(0,parseInt(mixBed1El?.value||'0',10)||0),
        bed2:Math.max(0,parseInt(mixBed2El?.value||'0',10)||0),
        bed3:Math.max(0,parseInt(mixBed3El?.value||'0',10)||0),
      };
      syncDevModePanel();
      updateCapacityLabel();
    });
  });
  runAnalyseBtn?.addEventListener('click',()=>runDevStage1Tool('analyse_boundary'));
  runEvaluateBtn?.addEventListener('click',()=>runDevStage1Tool('evaluate_typology_fit'));
  runProposeBtn?.addEventListener('click',()=>runDevStage1Tool('propose_typology_options'));
  runCheckBtn?.addEventListener('click',()=>runDevStage1Tool('check_typology_viability'));
  runOffsetBtn?.addEventListener('click',()=>runDevStage1Tool('generate_offset_regions'));
  runTestDoubleBtn?.addEventListener('click',()=>runGeneratedFloorPlateTest('double_loaded'));
  runTestSingleBtn?.addEventListener('click',()=>runGeneratedFloorPlateTest('single_loaded'));
  runTestLongBtn?.addEventListener('click',()=>runGeneratedFloorPlateTest('long_slab'));
  runTestPointBtn?.addEventListener('click',()=>runGeneratedFloorPlateTest('point_block'));
  runTestMixedBtn?.addEventListener('click',()=>runGeneratedFloorPlateTest('mixed_units'));
  runTestAllBtn?.addEventListener('click',()=>runAllGeneratedFloorPlateTests());
  runTestAltAllBtn?.addEventListener('click',()=>runAllSubdivisionAlternativeTests());
  runGenerateDrawnBtn?.addEventListener('click',()=>runGeneratedFromDrawnBoundary());
  retrieveUnitsBtn?.addEventListener('click',()=>retrieveAllFloorPlateUnits());
  retrieveFitHarnessBtn?.addEventListener('click',()=>runRetrieveFitHarness());
  clearUnitPlansBtn?.addEventListener('click',()=>{
    if(floorPlateState?.floor_plate){
      floorPlateState={...floorPlateState,floor_plate:{...floorPlateState.floor_plate,unitPlans:{}}};
      const logEl=document.getElementById('fp-dev-retrieve-units-log');
      if(logEl)logEl.textContent='Unit plans cleared.';
      render();
    }
  });
  runReqChecksBtn?.addEventListener('click',()=>{
    runRequirementsCompliance();
    setDevPanelTab('compliance');
    syncDevModePanel();
  });

  reqListEl?.addEventListener('change',(event)=>{
    const target=event.target;
    if(!(target instanceof HTMLInputElement))return;
    const row=target.closest('.fp-dev-req-row');
    const indexRaw=row?.getAttribute('data-index')||'';
    const index=parseInt(indexRaw,10);
    if(!Number.isInteger(index)||!devMode.requirements[index])return;
    if(target.dataset.role==='enabled'){
      devMode.requirements[index].enabled=!!target.checked;
      runRequirementsCompliance();
      syncDevModePanel();
    }
  });
  reqListEl?.addEventListener('click',(event)=>{
    const target=event.target;
    if(!(target instanceof HTMLElement))return;
    const role=target.dataset.role;
    if(role!=='run'&&role!=='info')return;
    const row=target.closest('.fp-dev-req-row');
    const indexRaw=row?.getAttribute('data-index')||'';
    const index=parseInt(indexRaw,10);
    if(!Number.isInteger(index)||!devMode.requirements[index])return;
    const requirement=devMode.requirements[index];

    if(role==='run'){
      runSingleRequirementCompliance(requirement.id);
      setDevPanelTab('compliance');
      syncDevModePanel();
      return;
    }

    const meta=getRequirementMeta(requirement.id);
    devMode.activeRequirementInfo=requirement.id;
    if(reqInfoEl){
      if(meta){
        reqInfoEl.textContent=`${meta.label}\n\nWhat it checks: ${meta.description}\nHow it is measured: ${meta.measurement}`;
      }else{
        reqInfoEl.textContent='No documentation available for this requirement.';
      }
    }
  });
  reqListEl?.addEventListener('input',(event)=>{
    const target=event.target;
    if(!(target instanceof HTMLInputElement))return;
    if(target.dataset.role!=='value')return;
    const row=target.closest('.fp-dev-req-row');
    const indexRaw=row?.getAttribute('data-index')||'';
    const index=parseInt(indexRaw,10);
    if(!Number.isInteger(index)||!devMode.requirements[index])return;
    const parsed=parseFloat(target.value);
    devMode.requirements[index].value=Number.isFinite(parsed)?Math.max(0,parsed):0;
    runRequirementsCompliance();
    syncDevModePanel();
  });

  compactBtn?.addEventListener('click',()=>{
    loadFixture('compact_slab');
    syncDevModePanel();
    render();
  });
  elongatedBtn?.addEventListener('click',()=>{
    loadFixture('elongated_slab');
    syncDevModePanel();
    render();
  });
  narrowBtn?.addEventListener('click',()=>{
    loadFixture('near_failing_narrow');
    syncDevModePanel();
    render();
  });
  seedBtn?.addEventListener('click',()=>{
    loadFixture('sample');
    syncDevModePanel();
    render();
  });
  clearBtn?.addEventListener('click',()=>{
    floorPlateState=null;
    devMode.activeFixture=null;
    devMode.generatedAlternatives=[];
    devMode.activeAlternativeIndex=0;
    if(devMode.offsetSubdivision)devMode.offsetSubdivision.latestRegions=null;
    runRequirementsCompliance();
    devMode.lastToolOutput=null;
    renderFloorPlateAlternativesPanel([],0);
    syncDevModePanel();
    render();
  });

  syncDevModePanel=()=>{
    const stateEl=document.getElementById('fp-dev-state');
    const bodyEl=document.getElementById('fp-dev-compliance-body');
    const summaryEl=document.getElementById('fp-dev-compliance-summary');
    const panelEl=document.getElementById('fp-dev-panel');
    const overlayEl=document.getElementById('fp-dev-overlay');
    if(!stateEl||!bodyEl||!panelEl)return;

    ensureDevModeDefaults();

    panelEl.classList.toggle('show',!!devMode.enabled&&!!devMode.panelVisible);
    if(overlayEl)overlayEl.checked=!!devMode.showFloorPlateOverlay;
    if(showIdsToggle)showIdsToggle.checked=!!devMode.showElementIds;
    if(showEdgesToggle)showEdgesToggle.checked=!!devMode.showEdgeIndices;
    if(layerBoundaryToggle)layerBoundaryToggle.checked=devMode.floorPlateLayers.boundary!==false;
    if(layerCoresToggle)layerCoresToggle.checked=devMode.floorPlateLayers.cores!==false;
    if(layerCommonToggle)layerCommonToggle.checked=devMode.floorPlateLayers.commonAreas!==false;
    if(layerUnitsToggle)layerUnitsToggle.checked=devMode.floorPlateLayers.unitBoundaries!==false;
    if(layerGridToggle)layerGridToggle.checked=devMode.floorPlateLayers.grid!==false;
    if(layerOffsetPreviewToggle)layerOffsetPreviewToggle.checked=devMode.floorPlateLayers.offsetPreview!==false;
    if(mixBed1El)mixBed1El.value=String(devMode.unitMix?.bed1??0);
    if(mixBed2El)mixBed2El.value=String(devMode.unitMix?.bed2??0);
    if(mixBed3El)mixBed3El.value=String(devMode.unitMix?.bed3??0);
    if(offsetDepthMinEl)offsetDepthMinEl.value=String(devMode.offsetSubdivision?.depthRange?.min??5);
    if(offsetDepthMaxEl)offsetDepthMaxEl.value=String(devMode.offsetSubdivision?.depthRange?.max??9);
    if(offsetMinCorridorEl)offsetMinCorridorEl.value=String(devMode.offsetSubdivision?.minCorridorWidthM??1.2);

    if(reqListEl){
      reqListEl.innerHTML=devMode.requirements.map((requirement,index)=>{
        const isBoolean=requirement.unit==='bool';
        const disabled=isBoolean?'disabled':'';
        const step=requirement.unit==='m'?'0.1':'1';
        return `
          <div class="fp-dev-req-row" data-index="${index}">
            <input type="checkbox" data-role="enabled" ${requirement.enabled?'checked':''}>
            <div class="fp-dev-req-label">${requirement.label}</div>
            <input class="fp-dev-input" data-role="value" type="number" min="0" step="${step}" value="${requirement.value}" ${disabled}>
            <div class="fp-dev-req-unit">${requirement.unit}</div>
            <button class="fp-dev-mini-btn" data-role="run" type="button">run</button>
            <button class="fp-dev-mini-btn" data-role="info" type="button">i</button>
          </div>
        `;
      }).join('');
    }

    if(reqInfoEl){
      const meta=getRequirementMeta(devMode.activeRequirementInfo);
      if(meta){
        reqInfoEl.textContent=`${meta.label}\n\nWhat it checks: ${meta.description}\nHow it is measured: ${meta.measurement}`;
      }else{
        reqInfoEl.textContent='Select a requirement info button to view how its check is measured.';
      }
    }

    setDevPanelTab(devMode.activeDevTab||'tools');

    if(offsetOutputEl){
      const offsetResult=devMode.offsetSubdivision?.latestRegions;
      if(!offsetResult){
        offsetOutputEl.textContent='Run offset region generation to preview inner zones, strips, and corner bays.';
      }else if(offsetResult.status!=='ok'){
        offsetOutputEl.textContent=`Offset generation failed: ${offsetResult.error||'unknown error'}`;
      }else{
        const diag=offsetResult.diagnostics||{};
        const notes=(diag.case_notes||diag.topology_notes||[]).join(' | ')||'none';
        const active=offsetResult.active_sample||{};
        const middleDims=diag.active_middle_region_dimensions;
        const middleDimsText=middleDims
          ? `middle=${(middleDims.width_m??0).toFixed(2)}m × ${(middleDims.length_m??0).toFixed(2)}m (${(middleDims.area_m2??0).toFixed(2)} m2)`
          : 'middle=n/a';
        offsetOutputEl.textContent=`range=${offsetResult.depth_range_m?.min?.toFixed?.(2)??'n/a'}m..${offsetResult.depth_range_m?.max?.toFixed?.(2)??'n/a'}m | samples=${diag.sample_count||0} | active=${diag.active_case||'n/a'} @ ${diag.active_depth_m?.toFixed?.(2)??'n/a'}m\ninner=${diag.active_inner_region_count||0} | strips=${active.apartment_strips?.length||0} | cornerBays=${active.corner_bays?.length||0} | projLines=${diag.active_projected_line_count||0}\n${middleDimsText}\narea boundary=${(diag.boundary_area_m2||0).toFixed(2)} m2 | closure error=${(diag.active_closure_error_m2||0).toFixed(3)} m2\nnotes: ${notes}`;
      }
    }

    updateCapacityLabel();

    const summary={
      devModeEnabled:!!devMode.enabled,
      panelVisible:!!devMode.panelVisible,
      overlayEnabled:!!devMode.showFloorPlateOverlay,
      showElementIds:!!devMode.showElementIds,
      showEdgeIndices:!!devMode.showEdgeIndices,
      floorPlateLayers:devMode.floorPlateLayers,
      offsetSubdivision:{
        depthRange:devMode.offsetSubdivision?.depthRange,
        minCorridorWidthM:devMode.offsetSubdivision?.minCorridorWidthM,
        hasLatestRegions:!!devMode.offsetSubdivision?.latestRegions,
      },
      unitMix:getDevUnitMix(),
      requirements:devMode.requirements,
      activeFixture:devMode.activeFixture||null,
      floorPlateLoaded:!!floorPlateState,
      floorPlateId:floorPlateState?.floor_plate?.id||floorPlateState?.id||null,
      unitCount:(floorPlateState?.floor_plate?.units||floorPlateState?.units||[]).length,
      boundaryDecomposition:{
        count:(floorPlateState?.floor_plate?.analysis?.boundary_decomposition_count
          ?? floorPlateState?.analysis?.boundary_decomposition_count
          ?? 0),
        polygons:(floorPlateState?.floor_plate?.analysis?.boundary_decomposition
          ?? floorPlateState?.analysis?.boundary_decomposition
          ?? null),
      },
      stageHints:{
        hasGrid:!!(floorPlateState?.floor_plate?.structural_grid||floorPlateState?.structural_grid),
        hasCore:((floorPlateState?.floor_plate?.cores||floorPlateState?.cores||[]).length>0),
        hasCorridor:((floorPlateState?.floor_plate?.corridors||floorPlateState?.corridors||[]).length>0),
      },
      lastToolOutput:devMode.lastToolOutput||null,
    };

    let stateText=JSON.stringify(summary,null,2);
    if(floorPlateState){
      const serialized=JSON.stringify(floorPlateState,null,2);
      stateText+=`\n\nFloorPlateState\n${serialized.slice(0,4000)}${serialized.length>4000?'\n...truncated':''}`;
    }
    stateEl.textContent=stateText;

    const report=devMode.lastComplianceReport||runRequirementsCompliance();
    const rows=Array.isArray(report?.requirements)
      ? report.requirements
      : [];
    if(summaryEl){
      const stats=report?.summary||{};
      summaryEl.textContent=`enabled: ${stats.enabled??0} | pass: ${stats.pass??0} | fail: ${stats.fail??0} | blocked: ${stats.blocked??0} | disabled: ${stats.disabled??0}`;
    }
    bodyEl.innerHTML='';
    if(!rows.length){
      const row=document.createElement('tr');
      row.innerHTML='<td colspan="4" style="color:#94a3b8">No compliance checks run yet.</td>';
      bodyEl.appendChild(row);
      return;
    }
    rows.forEach((item)=>{
      const tr=document.createElement('tr');
      const status=item.status||'unknown';
      const color=status==='pass'
        ?'#22c55e'
        :(status==='blocked'
          ?'#f59e0b'
          :(status==='disabled'?'#94a3b8':'#ef4444'));
      tr.innerHTML=`
        <td>${item.name||'requirement'}</td>
        <td>${item.expected??'-'}</td>
        <td>${item.actual??'-'}</td>
        <td style="color:${color}">${status}</td>
      `;
      bodyEl.appendChild(tr);
    });
  };

  runRequirementsCompliance();
  syncDevModePanel();
}

// ── coordinate helpers ────────────────────────────────────────────────────
// Base scale: normalised [0,1] → centred logical canvas pixels
function _baseScale(){return Math.min(cW,cH)*(1-2*CANVAS_PAD);}
function _baseOx(s){return(cW-s)/2;}
function _baseOy(s){return(cH-s)/2;}

// Normalised → logical canvas pixels (no viewport; ctx transform handles viewport)
function n2c(nx,ny){
  const s=_baseScale(),ox=_baseOx(s),oy=_baseOy(s);
  return[ox+nx*s,oy+ny*s];
}

// Viewport inverse: CSS canvas pixels → logical canvas pixels
function cssToLogical(cx,cy){
  return[
    (cx-viewport.panX-cW/2)/viewport.zoom+cW/2,
    (cy-viewport.panY-cH/2)/viewport.zoom+cH/2,
  ];
}

// Logical canvas pixels → normalised
function c2n(lx,ly){
  const s=_baseScale(),ox=_baseOx(s),oy=_baseOy(s);
  return[(lx-ox)/s,(ly-oy)/s];
}

// Screen pixel distance between two normalised points
function pdist(a,b){
  const s=_baseScale();
  return Math.hypot((a.x-b.x)*s*viewport.zoom,(a.y-b.y)*s*viewport.zoom);
}

// Mirror is a y-flip about the centroid, matching geo_ent_num_retrieval.
// retrieve()'s actual convention — see 2026-08-11 finding.
function planToCanvas(px,py,fpBdry,rot,offset,mirror){
  const s=_baseScale(),ox=_baseOx(s),oy=_baseOy(s);
  const cen=(rot||mirror)?bdryCenter(fpBdry):null;
  let x=px,y=py;
  if(mirror)y=2*cen[1]-y;
  if(rot)[x,y]=rotatePtAround([x,y],rot,cen);
  return[(x+offset[0])*s+ox,(y+offset[1])*s+oy];
}

function canvasToPlan(lx,ly,fpBdry,rot,offset,mirror){
  const s=_baseScale(),ox=_baseOx(s),oy=_baseOy(s);
  const cen=(rot||mirror)?bdryCenter(fpBdry):null;
  let x=(lx-ox)/s-offset[0];
  let y=(ly-oy)/s-offset[1];
  if(rot)[x,y]=rotatePtAround([x,y],-rot,cen);
  if(mirror)y=2*cen[1]-y;
  return[x,y];
}

// Axis-snap in normalised space; snap threshold is in screen pixels
function projSnap(nx,ny){
  const s=_baseScale();
  const thresh=SNAP_PX/(s*viewport.zoom);
  if(!boundary.length||closed)return{x:nx,y:ny};
  let sx=nx,sy=ny,bestX=Infinity,bestY=Infinity;
  boundary.forEach(({x,y})=>{
    const dx=Math.abs(nx-x),dy=Math.abs(ny-y);
    if(dx<thresh&&dx<bestX){bestX=dx;sx=x;}
    if(dy<thresh&&dy<bestY){bestY=dy;sy=y;}
  });
  return{x:sx,y:sy};
}

function snapToEdge(nx,ny,normVerts){
  return geometrySnapToEdge(nx,ny,{boundary,closed,normVerts});
}

function ghost(){
  return geometryGhost({boundary,closed,mouse,shiftDown,snapFirst,cW,cH});
}

// ── resize ────────────────────────────────────────────────────────────────
function resize(){
  if(activeTab==='3d'){
    resize3DViewer();
    return;
  }
  const wrap=document.getElementById('canvas-wrap');
  if(!wrap||wrap.clientWidth<40||wrap.clientHeight<40)return;
  cW=wrap.clientWidth-8;
  cH=wrap.clientHeight-8;
  const dpr=window.devicePixelRatio||1;
  canvas.width=cW*dpr;canvas.height=cH*dpr;
  canvas.style.width=cW+'px';canvas.style.height=cH+'px';
  ctx.setTransform(1,0,0,1,0,0);
  ctx.scale(dpr,dpr);
  render();
}
window.addEventListener('resize',resize);
setTimeout(resize,0);

function setViewer3DStatus(msg){
  const el=document.getElementById('viewer-3d-status');
  if(el)el.textContent=msg;
}

function toIntInput(id, fallback, minVal, maxVal){
  const raw=(document.getElementById(id)?.value||'').trim();
  let n=parseInt(raw,10);
  if(Number.isNaN(n))n=fallback;
  if(typeof minVal==='number')n=Math.max(minVal,n);
  if(typeof maxVal==='number')n=Math.min(maxVal,n);
  return n;
}

function toFloatInput(id, fallback, minVal, maxVal){
  const raw=(document.getElementById(id)?.value||'').trim();
  let n=parseFloat(raw);
  if(Number.isNaN(n))n=fallback;
  if(typeof minVal==='number')n=Math.max(minVal,n);
  if(typeof maxVal==='number')n=Math.min(maxVal,n);
  return n;
}

const RENDER_CONTEXT_OPTIONS=new Set(['rural','urban','suburban','coastal','forest','hillside','rooftop','desert']);

function resolveRenderContextFields(promptConfig){
  const rawContext=promptConfig?.context;
  const rawSiteContext=typeof promptConfig?.siteContext==='string'?promptConfig.siteContext.trim():'';

  if(Array.isArray(rawContext)){
    const preset=String(rawContext[0]||'').trim();
    const notes=rawContext.slice(1).map((part)=>String(part||'').trim()).filter(Boolean).join(', ');
    return {
      preset:RENDER_CONTEXT_OPTIONS.has(preset)?preset:'rural',
      notes:notes||rawSiteContext,
    };
  }

  const contextText=String(rawContext||'').trim();
  if(RENDER_CONTEXT_OPTIONS.has(contextText)){
    return {
      preset:contextText,
      notes:rawSiteContext,
    };
  }

  return {
    preset:RENDER_CONTEXT_OPTIONS.has(rawSiteContext)?rawSiteContext:'rural',
    notes:contextText||rawSiteContext,
  };
}

function applyRenderConfigToDom(){
  const promptConfig=renderRequestState.promptConfig||DEFAULT_RENDER_PROMPT_CONFIG;
  const contextFields=resolveRenderContextFields(promptConfig);
  const setValue=(id,value)=>{
    const el=document.getElementById(id);
    if(el&&typeof value==='string')el.value=value;
  };
  setValue('render-style',promptConfig.style);
  setValue('render-view',promptConfig.view);
  setValue('render-light',promptConfig.lighting);
  setValue('render-material',promptConfig.material);
  setValue('render-context',contextFields.preset);
  setValue('render-prompt',contextFields.notes);
  setValue('opacity-mat-walls',promptConfig.materialOverrides?.walls||'');
  setValue('opacity-mat-doors',promptConfig.materialOverrides?.doors||'');
  setValue('opacity-mat-windows',promptConfig.materialOverrides?.windows||'');
  setValue('opacity-mat-slabs',promptConfig.materialOverrides?.slabs||'');
  setValue('opacity-mat-parapets',promptConfig.materialOverrides?.parapets||'');
  setValue('render-openai-model',promptConfig.openaiModel);
  setValue('render-openai-quality',promptConfig.openaiQuality);
  setValue('render-comfy-url',promptConfig.comfyUrl);
  const invertEl=document.getElementById('render-comfy-invert');
  if(invertEl)invertEl.checked=!!promptConfig.depthInvert;
  const workflowNameEl=document.getElementById('workflow-name');
  if(workflowNameEl&&!renderRequestState.workflowName){
    workflowNameEl.textContent=`Using repo default: ${renderRequestState.workflowDefaultName||'comfyui-sdxl-depth-multi-workflow.json'}`;
  }
}

async function loadServerConfig(){
  try{
    const resp=await fetch('/config');
    if(!resp.ok)return;
    const payload=await resp.json();
    const config=payload&&payload.config;
    const renderCfg=config&&config.render;
    if(!renderCfg||typeof renderCfg!=='object')return;

    if(typeof renderCfg.default_backend==='string'&&renderCfg.default_backend.trim()){
      renderRequestState.backend=renderCfg.default_backend.trim().toLowerCase();
    }

    const currentCfg=renderRequestState.promptConfig||{...DEFAULT_RENDER_PROMPT_CONFIG};
    renderRequestState.promptConfig={
      ...currentCfg,
      context:typeof renderCfg.context==='string'&&renderCfg.context.trim()
        ? renderCfg.context.trim()
        : currentCfg.context,
      openaiModel:typeof renderCfg.openai_model==='string'&&renderCfg.openai_model.trim()
        ? renderCfg.openai_model.trim()
        : currentCfg.openaiModel,
      openaiQuality:typeof renderCfg.openai_quality==='string'&&renderCfg.openai_quality.trim()
        ? renderCfg.openai_quality.trim()
        : currentCfg.openaiQuality,
      comfyUrl:typeof renderCfg.comfy_url==='string'&&renderCfg.comfy_url.trim()
        ? renderCfg.comfy_url.trim()
        : currentCfg.comfyUrl,
    };
    if(typeof renderCfg.workflow_filename==='string'&&renderCfg.workflow_filename.trim()){
      renderRequestState.workflowDefaultName=renderCfg.workflow_filename.trim();
    }
  }catch(_err){
    // Keep local defaults if /config is unavailable.
  }
}

let checkComfyConnection;
let clearWorkflowOverride;
let downloadLatestCaptureBundle;
let generateRenderMock;
let loadWorkflowOverride;
let refreshStructuredPromptPreview=()=>{};
let setCaptureView;
let togglePanelCollapse;
let updateBackendDependentUI;
let updateCapturePanel=()=>{};
let drawDoorsWindows;
let drawFP;
let drawGraph;
let drawTopologyGraph;
let renderFloorPlate;
let render=()=>{};
let updateCursor=()=>{};
let getGraphNodes=()=>null;

const {
  captureDepthAndPreview,
  getCameraPerspectivePrompt,
  resize3DViewer,
  sync3DSceneIfNeeded,
} = createViewer3D({
  viewer3D,
  renderCaptureState,
  setViewer3DStatus,
  refreshStructuredPromptPreview:()=>refreshStructuredPromptPreview(),
  updateCapturePanel:()=>updateCapturePanel(),
  getActiveFPState,
  getSelectionState:()=>({selected,results,fitMode}),
  getActiveTab:()=>activeTab,
});

({
  checkComfyConnection,
  clearWorkflowOverride,
  downloadLatestCaptureBundle,
  generateRenderMock,
  loadWorkflowOverride,
  refreshStructuredPromptPreview,
  setCaptureView,
  togglePanelCollapse,
  updateBackendDependentUI,
  updateCapturePanel,
} = createRenderPipeline({
  renderCaptureState,
  renderRequestState,
  setViewer3DStatus,
  getCameraPerspectivePrompt,
  toIntInput,
}));

const {
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
} = createBoundaryEditor({
  doc:document,
  getState:()=>({
    boundary,
    closed,
    entrance,
    entranceMode,
    entranceSnap,
    edgeTypes,
    selected,
    results,
    selectedRoom,
    dragState,
    hoveredRoom,
    hoveredEntrance,
    displacementArrows,
    req,
  }),
  applyState:applyBoundaryEditorState,
  getActiveFPState,
  planToCanvas,
  canvasToPlan,
  ptInPolyFn:ptInPoly,
  ptToSegDistSqFn:ptToSegDistSq,
  snapToEdge,
  render:()=>render(),
  renderList,
  checkReady,
  sync3DSceneIfNeeded,
});

function setViewTab(tab,opts={}){
  activeTab=tab;
  const panel2d=document.getElementById('panel-2d');
  const panel3d=document.getElementById('panel-3d');
  const tabFp=document.getElementById('tab-fp');
  const tab2d=document.getElementById('tab-2d');
  const tab3d=document.getElementById('tab-3d');
  const tabFitEl=document.getElementById('tab-fit');
  const altPanel=document.getElementById('alt-panel');
  const fitPaletteEl=document.getElementById('fit-palette');

  const is3d=tab==='3d';
  const isFp=tab==='fp';
  const isFit=tab==='fit';

  panel2d.style.display=is3d?'none':'flex';
  panel3d.style.display=is3d?'flex':'none';
  if(tabFp)tabFp.classList.toggle('active',isFp);
  tab2d.classList.toggle('active',tab==='2d');
  tab3d.classList.toggle('active',is3d);
  if(tabFitEl)tabFitEl.classList.toggle('active',isFit);

  // Swap right-panel content between alternatives list and fit palette
  if(altPanel)altPanel.style.display=isFit?'none':'flex';
  if(fitPaletteEl)fitPaletteEl.style.display=isFit?'flex':'none';

  if(isFp){
    devMode.enabled=true;
    devMode.panelVisible=true;
    if(!devModeSetupDone)setupDevModeUI();
    syncDevModePanel();
    render();
  }else if(tab==='2d'){
    devMode.panelVisible=false;
    syncDevModePanel();
  }else if(isFit){
    devMode.panelVisible=false;
    syncDevModePanel();
    _fitSyncRunButton();
    render();
  }

  const is2d=tab==='2d'||isFp||isFit;

  if(is2d){
    const panel=document.getElementById('capture-panel');
    if(panel)panel.classList.remove('show');
    resize();
  }else{
    document.querySelectorAll('input[name="render-backend"]').forEach(el=>{
      el.checked=el.value===renderRequestState.backend;
      el.onchange=()=>{
        if(el.checked)renderRequestState.backend=el.value;
        updateBackendDependentUI();
        refreshStructuredPromptPreview();
      };
    });
    updateBackendDependentUI();
    updateCapturePanel();
    refreshStructuredPromptPreview();
    sync3DSceneIfNeeded(!!opts.triggerBuild);
  }
}

function onGenerate3DClick(){
  setViewTab('3d',{triggerBuild:true});
}

/**
 * Returns a live view into the currently active result/fitted plan state.
 * Callers that mutate `fp` intentionally mutate the backing results state.
 */
function getActiveFPState(){
  return getActivePlanView({
    selected,
    results,
    fitMode,
    fittedCache,
    boundary,
    centerOfBoundary:bdryCenter,
  });
}

({
  drawDoorsWindows,
  drawFP,
  drawGraph,
  drawTopologyGraph,
  renderFloorPlate,
  render,
  updateCursor,
  getGraphNodes,
} = createCanvasRenderer({
  canvas,
  ctx,
  getState:()=>({
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
    hoveredRoom,
    hoveredEntrance,
    dragState,
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
  }),
  getActiveFPState,
  n2c,
  ghost,
  snapToEdge,
  planToCanvas,
}));

const baseRender=render;
render=()=>{
  baseRender();
  syncDevModePanel();
  updateCanvasHud();
  updateTopoFlagsPanel();
  updateSwapPanel();
  if(activeTab==='fit')_fitSyncRunButton();
};

// Boundary area in m² (shoelace on the normalised polygon × METRES_PER_UNIT²).
function boundaryAreaM2(){
  if(!closed||boundary.length<3)return null;
  let a=0;
  for(let i=0;i<boundary.length;i++){
    const p=boundary[i],q=boundary[(i+1)%boundary.length];
    a+=p.x*q.y-q.x*p.y;
  }
  return Math.abs(a/2)*METRES_PER_UNIT*METRES_PER_UNIT;
}

// The canvas HUD — a DOM overlay anchored to the top-left of the canvas window
// (so it does not pan/zoom with the drawing). Shows boundary area always, plus
// the arrange step's judgement when one has been run. Extend by pushing lines.
function updateCanvasHud(){
  const el=document.getElementById('canvas-hud');
  if(!el)return;
  const lines=[];
  const areaM2=boundaryAreaM2();
  if(areaM2!=null)lines.push(['boundary: '+areaM2.toFixed(1)+' m²','#1e293b']);
  if(graphSelection.length===1)lines.push(['swap: pick a 2nd room','#2563eb']);
  else if(graphSelection.length===2)lines.push(['swap: '+graphSelection.join(' ↔ '),'#2563eb']);
  const rep=arapDebugReport;
  if(rep){
    lines.push([rep.feasible?'✓ feasible':'✗ not feasible',rep.feasible?'#16a34a':'#ef4444']);
    const under=(rep.rooms||[]).filter(r=>!r.ok).map(r=>r.label);
    if(under.length)lines.push(['under-min: '+under.join(', '),'#ef4444']);
    if((rep.no_daylight||[]).length)lines.push(['no-daylight: '+rep.no_daylight.join(', '),'#d97706']);
    if((rep.no_access||[]).length)lines.push(['no-access: '+rep.no_access.join(', '),'#a855f7']);
    if((rep.clipped_out_rooms||[]).length)lines.push(['clipped-out: '+rep.clipped_out_rooms.join(', '),'#ef4444']);
    if(rep.enclosures)lines.push(['enclosures: '+rep.enclosures,'#ef4444']);
    if(rep.clip_splits)lines.push(['clip-splits: '+rep.clip_splits,'#ef4444']);
    if((rep.dropped_adjacencies||[]).length)lines.push(['dropped-adj: '+rep.dropped_adjacencies.join(', '),'#d97706']);
    lines.push([`${rep.mode} · grid ${(rep.grid||[]).join('×')} · rot ${rep.rotation_deg||0}°`,'#64748b']);
  }
  if(!lines.length){el.style.display='none';el.innerHTML='';return;}
  el.style.display='block';
  el.innerHTML=lines.map(l=>`<div style="color:${l[1]}">${l[0]}</div>`).join('');
}

// Sidebar panel listing the selected result's topology sensor checks
// (backend-attached retrieval.flags — pure consumption, degrades to hidden
// when the field is absent/malformed or nothing is selected).
const FLAG_SENSOR_LABELS={reachability:'reachability',access_typed:'access typing',bedroom_daylight:'bedroom daylight',common_daylight:'common daylight',entrance_valid:'entrance'};
function updateTopoFlagsPanel(){
  const section=document.getElementById('topo-flags-section');
  const el=document.getElementById('topo-flags-panel');
  if(!section||!el)return;
  const flags=results?.[selected]?.retrieval?.flags;
  const checks=flags&&Array.isArray(flags.checks)?flags.checks:null;
  if(!checks){section.style.display='none';el.innerHTML='';return;}
  section.style.display='block';
  el.innerHTML=checks.map((chk)=>{
    const ok=!!chk?.ok;
    const label=FLAG_SENSOR_LABELS[chk?.sensor]||chk?.sensor||'?';
    return `<div style="color:${ok?'#16a34a':'#dc2626'};opacity:${ok?0.75:1};margin-bottom:3px">`+
      `${ok?'✓':'✗'} ${label} — ${chk?.detail??''}</div>`;
  }).join('');
}

// ── boundary edge types (inter-tenancy / exterior / hallway) ────────────────
// A freehand boundary carries no edge classification; this lets the user set it
// so the compliance oracle (daylight, access) has real edge types to read
// instead of the entrance-derived default. Backend already respects supplied
// types (boundary_to_schema) — see docs/adaptation-process-model.md.
const EDGE_TYPE_LABELS={external:'exterior',shared_wall:'inter-tenancy',facing_corridor:'hallway'};
const EDGE_TYPE_ORDER=['external','shared_wall','facing_corridor'];
const EDGE_TYPE_COLOURS={external:'#22c55e',shared_wall:'#64748b',facing_corridor:'#eab308'};

function toggleEdgeTypeMode(){
  if(!closed)return;
  edgeTypeMode=!edgeTypeMode;
  if(edgeTypeMode&&entranceMode){        // the two edge modes are mutually exclusive
    entranceMode=false;entranceSnap=null;
    document.getElementById('btn-entrance-mode')?.classList.remove('active');
  }
  hoveredEdge=null;
  closeEdgeTypePicker();
  document.getElementById('btn-edge-type-mode')?.classList.toggle('active',edgeTypeMode);
  document.getElementById('canvas-hint').textContent=
    edgeTypeMode?'click a boundary edge to set its type':'boundary closed · press retrieve';
  render();
}

let _edgeTypePickerEl=null;
function closeEdgeTypePicker(){
  if(_edgeTypePickerEl){_edgeTypePickerEl.remove();_edgeTypePickerEl=null;}
}
function openEdgeTypePicker(edgeIdx,clientX,clientY){
  closeEdgeTypePicker();
  const cur=edgeTypes[edgeIdx]||'external';
  const el=document.createElement('div');
  el.style.cssText='position:fixed;z-index:9999;left:'+(clientX+8)+'px;top:'+(clientY+8)+'px;'+
    'background:var(--bg2,#1e293b);border:1px solid var(--border2,#475569);border-radius:6px;padding:4px;'+
    'box-shadow:0 4px 16px rgba(0,0,0,0.45);font-family:var(--mono,monospace);font-size:11px;min-width:140px';
  EDGE_TYPE_ORDER.forEach((t)=>{
    const b=document.createElement('button');
    b.style.cssText='display:flex;align-items:center;gap:7px;width:100%;text-align:left;padding:4px 7px;'+
      'background:transparent;border:none;border-radius:4px;color:var(--text,#e2e8f0);cursor:pointer;line-height:1.5';
    const dot=document.createElement('span');
    dot.style.cssText='flex-shrink:0;width:10px;height:10px;border-radius:2px;background:'+EDGE_TYPE_COLOURS[t];
    const lbl=document.createElement('span');
    lbl.textContent=(t===cur?'● ':'○ ')+EDGE_TYPE_LABELS[t];
    b.appendChild(dot);b.appendChild(lbl);
    b.onmouseover=()=>{b.style.background='rgba(148,163,184,0.18)';};
    b.onmouseout=()=>{b.style.background='transparent';};
    b.onclick=()=>{edgeTypes[edgeIdx]=t;closeEdgeTypePicker();render();};
    el.appendChild(b);
  });
  document.body.appendChild(el);
  _edgeTypePickerEl=el;
  // Dismiss on any click outside the popover.
  setTimeout(()=>{
    const off=(ev)=>{
      if(_edgeTypePickerEl&&!_edgeTypePickerEl.contains(ev.target)){
        closeEdgeTypePicker();document.removeEventListener('mousedown',off,true);
      }
    };
    document.addEventListener('mousedown',off,true);
  },0);
}

// Sparse payload — only explicitly-set edges; the backend defaults the rest
// (entrance edge → facing_corridor, others → external).
function boundaryTypedEdgesPayload(){
  if(!closed)return [];
  const out=[];
  for(let i=0;i<boundary.length;i++){
    if(edgeTypes[i])out.push({edge_index:i,type:edgeTypes[i]});
  }
  return out;
}

// Swap panel — enable/disable the button + readout prompt, driven off
// graphSelection like the canvas HUD's own "swap: ra ↔ rb" hint (line ~2777).
// Runs every render(), same pattern as updateTopoFlagsPanel.
function updateSwapPanel(){
  const btn=document.getElementById('btn-swap-nodes');
  const resetBtn=document.getElementById('btn-swap-reset');
  const el=document.getElementById('swap-readout');
  if(btn)btn.disabled=!(graphSelection.length===2&&selected!==null&&closed);
  if(resetBtn)resetBtn.style.display=results[selected]?.__preswap?'':'none';
  if(el&&graphSelection.length===2)el.textContent=`ready: swap ${graphSelection[0]} ↔ ${graphSelection[1]}`;
}

// Act counterpart to the Observe topology-checks panel: swap the two selected
// graph nodes' geometry via the backend (spatial validity — /swap_nodes),
// then re-run the topology sensors on the swapped plan (/topo_check) so the
// user sees whether a flag clears. Reversible — resetSwap() below restores
// the stashed pre-swap state.
async function runSwapNodes(){
  const el=document.getElementById('swap-readout');
  if(graphSelection.length!==2||selected===null||!closed)return;
  const [ra,rb]=graphSelection;
  const rawPlan=results[selected].fp;
  if(el)el.textContent='swapping…';
  try{
    const {data:resp}=await postJsonWithTimeout('/swap_nodes',{
      fp:rawPlan,
      boundary:boundary.map(({x,y})=>[x,y]),
      room_a:ra,
      room_b:rb,
      entrance_mid:entrance?[entrance.x,entrance.y]:null,
    },{timeoutMs:15000});
    if(resp.status==='rejected'){
      if(el)el.textContent=`Swap rejected: ${resp.reason}`;
      return;
    }

    const swappedPlan=JSON.parse(JSON.stringify(rawPlan));
    const roomA=(swappedPlan.rooms||[]).find((r)=>String(r.id)===String(ra));
    const roomB=(swappedPlan.rooms||[]).find((r)=>String(r.id)===String(rb));
    if(!roomA||!roomB)throw new Error('swapped room not found in plan');
    [roomA.polygon,roomB.polygon]=[roomB.polygon,roomA.polygon];

    const {data:tc}=await postJsonWithTimeout('/topo_check',{
      fp:swappedPlan,
      boundary:boundary.map(({x,y})=>[x,y]),
      entrance_point:entrance?[entrance.x,entrance.y]:null,
      typed_edges:boundaryTypedEdgesPayload(),
      requirements:req||null,
    },{timeoutMs:15000});
    if(tc.error)throw new Error(tc.error);

    if(!results[selected].__preswap){
      results[selected].__preswap={fp:results[selected].fp,retrieval:results[selected].retrieval};
    }
    results[selected].fp=swappedPlan;
    results[selected].retrieval={...results[selected].retrieval,room_edges:tc.room_edges,flags:{summary:tc.summary,checks:tc.checks}};
    graphSelection=[];
    render();
    renderList();  // card badge (flagsBadgeHtml) is baked HTML — needs a rebuild to reflect the new flags

    const {gained=[],dropped=[]}=resp.adjacency_delta||{};
    const adjMsg=(gained.length||dropped.length)?`adjacency: dropped ${dropped.length}, gained ${gained.length}`:'adjacency preserved';
    if(el)el.textContent=`Swapped ${ra} ↔ ${rb} · ${adjMsg} · checks ${tc.summary?.passed??'?'}/${tc.summary?.total??'?'}`;
  }catch(err){
    if(el)el.textContent=`Swap failed: ${err.message}`;
  }
}

// Restores the pre-swap fp/retrieval stashed by runSwapNodes, so the user can
// swap, inspect, revert, and try another pair.
function resetSwap(){
  const pre=results[selected]?.__preswap;
  if(!pre)return;
  results[selected].fp=pre.fp;
  results[selected].retrieval=pre.retrieval;
  delete results[selected].__preswap;
  graphSelection=[];
  render();
  renderList();
}

// Stage 0 pre-analysis — advisory check that the requested program can fit the
// drawn boundary (area + external frontage) before retrieval.
async function runFeasibilityCheck(){
  const el=document.getElementById('feasibility-result');
  if(!el)return;
  if(!closed||boundary.length<3){el.innerHTML='<span style="color:var(--muted)">draw &amp; close a boundary first</span>';return;}
  if(!entrance){el.innerHTML='<span style="color:var(--muted)">place an entrance first</span>';return;}
  el.innerHTML='<span style="color:var(--muted)">checking…</span>';
  try{
    const {data}=await postJsonWithTimeout('/feasibility_check',{
      boundary:boundary.map(({x,y})=>[x,y]),
      entrance_point:[entrance.x,entrance.y],
      req,
      boundary_typed_edges:boundaryTypedEdgesPayload(),
    },{timeoutMs:10000});
    if(data.error)throw new Error(data.error);
    const mark=(ok)=>ok?'<span style="color:#16a34a">✓</span>':'<span style="color:#ef4444">✗</span>';
    const verdict=data.feasible
      ?'<span style="color:#16a34a;font-weight:600">✓ program can fit</span>'
      :'<span style="color:#ef4444;font-weight:600">✗ program does not fit</span>';
    el.innerHTML=verdict
      +`<div>${mark(data.area.ok)} area: need ${data.area.required_m2} m², have ${data.area.available_m2} m²</div>`
      +`<div>${mark(data.frontage.ok)} frontage: need ${data.frontage.required_m} m, have ${data.frontage.available_m} m</div>`
      +`<div style="color:var(--muted)">program: ${data.program.join(', ')}</div>`;
    console.log('[feasibility]',data);
  }catch(err){
    el.innerHTML='<span style="color:#ef4444">check failed: '+err.message+'</span>';
    console.error('feasibility_check failed:',err);
  }
}

// ── mouse ─────────────────────────────────────────────────────────────────
canvas.addEventListener('mousedown',e=>{
  // Middle button: start pan
  if(e.button===1){
    e.preventDefault();
    panDrag={startX:e.clientX,startY:e.clientY,startPanX:viewport.panX,startPanY:viewport.panY};
    canvas.style.cursor='grab';
    return;
  }
  if(!closed||selected===null||entranceMode||edgeTypeMode)return;
  const r=canvas.getBoundingClientRect();
  const rawX=(e.clientX-r.left)*(cW/r.width),rawY=(e.clientY-r.top)*(cH/r.height);
  const [cx,cy]=cssToLogical(rawX,rawY);
  const s=getActiveFPState();
  // entrance drag takes priority
  if(s&&hitTestEntrance(cx,cy)){
    const[planX,planY]=canvasToPlan(cx,cy,s.fpBdry,s.rot,s.offset,s.mirror);
    const ent=s.fp.entrance;
    dragState={mode:'entrance',startPlanXY:[planX,planY],origEntrance:[...ent],
      origCentroid:[(ent[0]+ent[2])/2,(ent[1]+ent[3])/2]};
    e.preventDefault();updateCursor();render();return;
  }
  const hit=hitTestRooms(cx,cy);
  if(hit){
    selectedRoom=hit.roomIdx;
    const[planX,planY]=canvasToPlan(cx,cy,s.fpBdry,s.rot,s.offset,s.mirror);
    const poly=s.fp.rooms[hit.roomIdx].polygon;
    const origCentroid=[poly.reduce((a,v)=>a+v[0],0)/poly.length,
                        poly.reduce((a,v)=>a+v[1],0)/poly.length];
    dragState={
      ...hit,
      mode: hit.mode==='body' ? 'move' : 'edge',
      startPlanXY:[planX,planY],
      origPoly:poly.map(v=>[...v]),
      origCentroid,
    };
    e.preventDefault();
    updateCursor();render();
  }else{
    selectedRoom=null;render();
  }
});

document.addEventListener('mouseup',e=>{
  if(e.button===1&&panDrag){
    panDrag=null;
    canvas.style.cursor=closed?'default':'crosshair';
    return;
  }
  if(dragState){
    // record displacement arrow if the object actually moved
    const s=getActiveFPState();
    if(s&&dragState.origCentroid){
      let toCentroid;
      if(dragState.mode==='entrance'){
        const ent=s.fp.entrance;
        if(ent)toCentroid=[(ent[0]+ent[2])/2,(ent[1]+ent[3])/2];
      }else if(dragState.mode==='move'&&dragState.roomIdx!=null){
        const poly=s.fp.rooms[dragState.roomIdx]?.polygon;
        if(poly)toCentroid=[poly.reduce((a,v)=>a+v[0],0)/poly.length,
                            poly.reduce((a,v)=>a+v[1],0)/poly.length];
      }
      if(toCentroid){
        const[fx,fy]=dragState.origCentroid,[tx,ty]=toCentroid;
        const dsq=(tx-fx)**2+(ty-fy)**2;
        if(dsq>1e-8)displacementArrows.push({from:[fx,fy],to:[tx,ty]});
      }
    }
    dragState=null;updateCursor();render();
  }
});

canvas.addEventListener('mousemove',e=>{
  // Pan with middle button
  if(panDrag){
    viewport.panX=panDrag.startPanX+(e.clientX-panDrag.startX);
    viewport.panY=panDrag.startPanY+(e.clientY-panDrag.startY);
    render();
    return;
  }
  const r=canvas.getBoundingClientRect();
  const rawX=(e.clientX-r.left)*(cW/r.width),rawY=(e.clientY-r.top)*(cH/r.height);
  const [cx,cy]=cssToLogical(rawX,rawY);
  const[nx,ny]=c2n(cx,cy);
  mouse={x:nx,y:ny};
  shiftDown=e.shiftKey;
  snapFirst=(!closed&&boundary.length>=3&&pdist(mouse,boundary[0])<SNAP_PX);
  document.getElementById('shift-ind').style.display=shiftDown&&!closed?'block':'none';

  // ── drag: move or resize room / entrance ─────────────────────────────
  if(dragState){
    const s=getActiveFPState();
    if(s){
      const[planX,planY]=canvasToPlan(cx,cy,s.fpBdry,s.rot,s.offset,s.mirror);
      const[sx,sy]=dragState.startPlanXY;
      const dx=planX-sx,dy=planY-sy;
      if(dragState.mode==='entrance'){
        const[x1,y1,x2,y2]=dragState.origEntrance;
        s.fp.entrance=[x1+dx,y1+dy,x2+dx,y2+dy];
      }else{
        const room=s.fp.rooms[dragState.roomIdx];
        const orig=dragState.origPoly;
        if(dragState.mode==='move'){
          room.polygon=orig.map(([x,y])=>[x+dx,y+dy]);
        }else{
          const axis=dragState.edgeAxis,cv=dragState.coordValue;
          const EPS=1e-4;
          if(axis==='h'){
            room.polygon=orig.map(([x,y])=>Math.abs(y-cv)<EPS?[x,y+dy]:[x,y]);
          }else{
            room.polygon=orig.map(([x,y])=>Math.abs(x-cv)<EPS?[x+dx,y]:[x,y]);
          }
        }
      }
    }
    render();return;
  }

  // ── edge-type mode: highlight the edge under the cursor ──────────────
  if(edgeTypeMode&&closed){
    const snap=snapToEdge(nx,ny);
    hoveredEdge=snap?snap.edge:null;
    canvas.style.cursor=snap?'pointer':'default';
  }
  // ── hover detection for room / entrance editing ─────────────────────
  else if(selected!==null&&closed&&!entranceMode){
    hoveredEntrance=hitTestEntrance(cx,cy);
    hoveredRoom=hoveredEntrance?null:hitTestRooms(cx,cy);
    updateCursor();
  }else{
    hoveredRoom=null;hoveredEntrance=false;
    canvas.style.cursor=closed?'default':'crosshair';
  }

  if(entranceMode&&closed){
    entranceSnap=snapToEdge(nx,ny);
    const coordText=entranceSnap
      ? `${(entranceSnap.px*METRES_PER_UNIT).toFixed(1)} m, ${(entranceSnap.py*METRES_PER_UNIT).toFixed(1)} m`
      : '';
    document.getElementById('coord-disp').textContent=coordText;
  } else {
    document.getElementById('coord-disp').textContent=closed?'':`${(nx*METRES_PER_UNIT).toFixed(1)} m, ${(ny*METRES_PER_UNIT).toFixed(1)} m`;
  }
  render();
});

canvas.addEventListener('mouseleave',()=>{
  document.getElementById('coord-disp').textContent='';
  entranceSnap=null;
  if(!dragState){hoveredRoom=null;updateCursor();}
  render();
});

canvas.addEventListener('click',e=>{
  const r=canvas.getBoundingClientRect();
  const rawX=(e.clientX-r.left)*(cW/r.width),rawY=(e.clientY-r.top)*(cH/r.height);
  const [cx,cy]=cssToLogical(rawX,rawY);

  // Topology-graph node selection (room-swap prerequisite) — takes priority
  // over every other click behaviour whenever the graph is showing. Hit-test
  // against the positions the renderer actually drew (getGraphNodes), in the
  // same logical-canvas-px space cssToLogical just produced (pre-viewport
  // transform — matches roomCen/n2c, not CSS px or normalised coords).
  if(params.showGraph&&selected!==null&&closed){
    const g=getGraphNodes();
    if(g){
      let hitId=null,hitDist=Infinity;
      for(const [id,pos] of Object.entries(g.nodes)){
        const d=Math.hypot(cx-pos[0],cy-pos[1]);
        if(d<=g.nr+6&&d<hitDist){hitDist=d;hitId=id;}
      }
      if(hitId!==null){
        const idx=graphSelection.indexOf(hitId);
        if(idx!==-1)graphSelection.splice(idx,1);
        else{
          graphSelection.push(hitId);
          if(graphSelection.length>2)graphSelection.shift();
        }
        render();
        return;
      }
    }
  }

  let[nx,ny]=c2n(cx,cy);

  // edge-type assignment mode — click an edge, pick its type from a popover
  if(edgeTypeMode&&closed){
    const snap=snapToEdge(nx,ny);
    if(snap)openEdgeTypePicker(snap.edge,e.clientX,e.clientY);
    return;
  }

  // entrance placement mode
  if(entranceMode&&closed){
    const snap=snapToEdge(nx,ny);
    if(snap){
      entrance={x:snap.px,y:snap.py};
      entranceMode=false;
      entranceSnap=null;
      // The edge the entrance sits on is the hallway-facing one by definition —
      // set it unless the user already classified that edge deliberately.
      if(!edgeTypes[snap.edge])edgeTypes[snap.edge]='facing_corridor';
      document.getElementById('btn-entrance-mode').classList.remove('active');
      document.getElementById('canvas-hint').textContent='entrance placed · press retrieve';
      updateEntranceInfo();
      checkReady();render();
    }
    return;
  }

  if(closed)return;
  if(e.shiftKey&&boundary.length>0){
    const l=boundary[boundary.length-1];
    const dx=Math.abs(nx-l.x),dy=Math.abs(ny-l.y);
    if(dx>dy)ny=l.y;else nx=l.x;
  } else {
    const s=projSnap(nx,ny);nx=s.x;ny=s.y;
  }
  if(snapFirst&&boundary.length>=3){closeBoundary();return;}
  boundary.push({x:nx,y:ny});
  document.getElementById('s-bv').textContent=boundary.length;
  document.getElementById('canvas-hint').textContent=
    boundary.length>=3?'double-click or click first vertex to close · shift = lock H/V':
    'click to add vertices · shift = lock H/V';
  checkReady();render();
});

canvas.addEventListener('dblclick',e=>{
  if(closed||boundary.length<3)return;
  e.preventDefault();closeBoundary();
});

// ── zoom (scroll wheel) ───────────────────────────────────────────────────
canvas.addEventListener('wheel',e=>{
  e.preventDefault();
  const r=canvas.getBoundingClientRect();
  const cx=(e.clientX-r.left)*(cW/r.width);
  const cy=(e.clientY-r.top)*(cH/r.height);
  const factor=e.deltaY<0?1.12:1/1.12;
  const newZoom=Math.max(0.15,Math.min(20,viewport.zoom*factor));
  // Zoom about the cursor: keep the logical point under the cursor fixed
  const logX=(cx-viewport.panX-cW/2)/viewport.zoom+cW/2;
  const logY=(cy-viewport.panY-cH/2)/viewport.zoom+cH/2;
  viewport.panX=cx-(logX-cW/2)*newZoom-cW/2;
  viewport.panY=cy-(logY-cH/2)*newZoom-cH/2;
  viewport.zoom=newZoom;
  render();
},{passive:false});

document.addEventListener('keydown',e=>{
  if((e.ctrlKey||e.metaKey)&&e.shiftKey&&e.key.toLowerCase()==='d'){
    e.preventDefault();
    if(activeTab==='fp'){
      setViewTab('2d');
    }else{
      setViewTab('fp');
    }
    return;
  }
  if(e.target&&e.target.matches&&e.target.matches('input, textarea, select'))return;
  if(e.key==='Escape')clearBoundary();
  if(e.key==='Enter'&&boundary.length>=3&&!closed)closeBoundary();
  if(e.key==='Shift')shiftDown=true;
  if(e.key==='z'||e.key==='Z'){viewport.zoom=1;viewport.panX=0;viewport.panY=0;render();}
});
document.addEventListener('keyup',e=>{
  if(e.key==='Shift'){shiftDown=false;document.getElementById('shift-ind').style.display='none';render();}
});

renderLib();

// ── index loading ─────────────────────────────────────────────────────────
function setupDrop(dzId,inpId,lblId,fn){
  setupDropZone(dzId,inpId,lblId,fn,document);
}

setupDrop('dz-bin','fi-bin','lbl-bin',(files,lbl)=>{
  const f=files[0];if(!f)return;lbl.textContent=f.name;
  f.arrayBuffer().then(buf=>{db._raw=new Float32Array(buf);tryFinalise();});
});
setupDrop('dz-nms','fi-nms','lbl-nms',(files,lbl)=>{
  const f=files[0];if(!f)return;lbl.textContent=f.name;
  f.text().then(txt=>{db.names=JSON.parse(txt);tryFinalise();});
});
setupDrop('dz-pls','fi-pls','lbl-pls',(files,lbl)=>{
  lbl.textContent=files.length+' files';
  let n=0;
  Array.from(files).forEach(f=>{
    f.text().then(txt=>{
      try{
        const d=JSON.parse(txt);
        const stem=f.name.replace(/\.json$/i,'');
        cache[stem]=d;if(d.id)cache[d.id]=d;
        document.getElementById('s-pls').textContent=++n;
      }catch(e){}
    });
  });
});

initPlanDrawingPanel();

function tryFinalise(){
  return tryFinalizeIndex({
    db,
    onReady:()=>{setStatus('index ready','ok');checkReady();},
    onError:(msg)=>setStatus(msg,'err'),
    doc:document,
  });
}

function checkReady(){
  checkReadyState({closed,boundary},document);
}
function setStatus(msg,cls=''){
  setStatusUI(msg,cls,document);
}

// ── retrieval ─────────────────────────────────────────────────────────────
// Runs entirely through the backend (canonical retrieval-prototype module —
// 2026-08-10 decision; see docs/pipeline-architecture.md). No client-side
// fallback — see runRetrievalFlow in retrieval.js for why.
async function runRetrieval(){
  await runRetrievalFlow({
    closed,
    boundary,
    entrance,
    req,
    params,
    setStatus,
    applyServerPlans:renderAdaptedResults,
    afterRender:()=>render(),
    checkReady,
    typedEdges:boundaryTypedEdgesPayload(),
    doc:document,
  });
}

// ── adapted results panel ────────────────────────────────────────────────
function renderAdaptedResults(plans){
  const next=renderAdaptedResultsView({
    plans,
    listEl:document.getElementById('results-list'),
    selectResult,
    drawFP,
    drawDoorsWindows,
    drawGraph,
    drawTopologyGraph,
    sync3DSceneIfNeeded,
  });
  results=next.results;
  fittedCache=next.fittedCache;
  selected=next.selected;
  displacementArrows=next.displacementArrows;
}

function renderList(){
  renderResultsListView({
    results,
    selected,
    listEl:document.getElementById('results-list'),
    selectResult,
    drawFP,
    drawDoorsWindows,
    drawGraph,
    drawTopologyGraph,
  });
}

function selectResult(i){
  // Changing selection invalidates any ARAP overlay/alignment state — it
  // was specific to whichever result was previously selected.
  arapDebugRooms=null;
  arapDebugLabel=null;
  arapDebugDoor=null;
  arapDebugReport=null;
  arapAlignedIndex=null;
  arapSolvedIndex=null;
  arapRectifiedIndex=null;
  arapFilledIndex=null;
  ({selected,selectedRoom,dragState,hoveredRoom,hoveredEntrance,displacementArrows}=selectResultState({
    currentSelected:selected,
    nextIndex:i,
    fitMode,
    fittedCache,
    fetchFitted,
    highlightCards:(activeIndex)=>document.querySelectorAll('.result-card').forEach((c,j)=>c.classList.toggle('selected',j===activeIndex)),
    updateCursor,
    render,
    sync3DSceneIfNeeded,
  }));
}

// Transform query boundary into the retrieved plan's original coordinate space
function boundaryInPlanSpace(r){
  const rot=r.rot||0, mirror=r.mirror||false;
  const qc=bdryCenter(boundary.map(({x,y})=>[x,y]));
  const pc=bdryCenter(r.fp.boundary||[]);
  const ox=qc[0]-pc[0], oy=qc[1]-pc[1];
  return boundary.map(({x,y})=>{
    let px=x-ox, py=y-oy;
    [px,py]=rotatePtAround([px,py],-rot,pc);
    // y-flip, not x-flip — matches geo_ent_num_retrieval.retrieve()'s actual
    // mirror convention. See 2026-08-11 finding.
    if(mirror)py=2*pc[1]-py;
    return[px,py];
  });
}

// Mirror a plan's room polygons + entrance about its own boundary centroid —
// a y-flip (reflection about a horizontal line through the centroid), matching
// geo_ent_num_retrieval.retrieve()'s actual mirror convention (conj() = angle
// negation). Corrected 2026-08-11 from an earlier x-flip that disagreed with
// retrieve() — mirrored-match thumbnails were showing a different plan than
// the one actually scored (entrance ended up on the wrong side). Needed
// because place_initial_layout (fitting-prototype) only rotates in 90° steps
// and has no reflection logic of its own — if retrieval's best alignment for
// this plan required a mirror (r.mirror), that mirror has to be baked into
// the plan's actual geometry before the Python side ever sees it.
function mirrorPlanY(fp){
  const cen=bdryCenter(fp.boundary||[]);
  const my=(y)=>2*cen[1]-y;
  return {
    ...fp,
    boundary:(fp.boundary||[]).map(([x,y])=>[x,my(y)]),
    rooms:(fp.rooms||[]).map((r)=>({
      ...r,
      polygon:(r.polygon||[]).map(([x,y])=>[x,my(y)]),
    })),
    entrance:fp.entrance?[fp.entrance[0],my(fp.entrance[1]),fp.entrance[2],my(fp.entrance[3])]:fp.entrance,
  };
}

// task 1.5 (docs/pipeline-architecture.md) — run the schema.py-native
// adaptation chain (fitting-prototype, via /fit_schema) on the selected
// retrieved plan against the drawn boundary, then open the architectural
// 2D drawing panel on the result. Deterministic only (no phase25 LLM loop).
//
// Deliberately does NOT use boundaryInPlanSpace/rot — that transform exists
// for the app's OWN canvas-draw-time rotation trick (rendering the plan's
// native, unrotated geometry while visually rotating it on the canvas).
// /fit_schema instead deforms the plan's actual geometry via
// place_initial_layout, which computes its own rotation from scratch by
// aligning the plan's entrance face to the boundary's corridor side — so it
// needs the boundary exactly as drawn, not pre-rotated into the plan's
// native frame (feeding it the pre-rotated boundary was the 2026-08-11 bug:
// two independent rotations compounding, boundary and rooms ending up
// consistently rotated/mirrored relative to what was actually drawn).
// Mirror is the one thing still applied ahead of time, onto the plan, since
// place_initial_layout has no reflection step to do it downstream.
async function drawSelectedArchitecturally(){
  if(selected===null||!results[selected]?.fp){
    setStatus('select a retrieved plan first','err');
    return;
  }
  if(!entrance){
    setStatus('place an entrance on the boundary first','err');
    return;
  }
  const r=results[selected];
  const fp=r.mirror?mirrorPlanY(r.fp):r.fp;
  setStatus('fitting + repairing (this can take a minute — local LLM agent loop)…','warn');
  try{
    const {data}=await postJsonWithTimeout('/fit_schema',{
      fp,
      boundary:boundary.map(({x,y})=>[x,y]),
      entrance_point:[entrance.x,entrance.y],
      label:`live_${r.name||selected}`,
    },{timeoutMs:240000});
    if(data.error)throw new Error(data.error);
    loadFixtureData(data.fixture, `live · #${selected+1} ${r.name||''}`);
    setStatus(`drawn (${data.fixture.agent_passes||0} repair pass${data.fixture.agent_passes===1?'':'es'})`,'ok');
  }catch(err){
    console.error('fit_schema failed:',err);
    setStatus(`architectural draw failed (${err.message})`,'err');
  }
}

// DEBUG (2026-08-12): checks whether ARAP's align step preserves room
// identity, independent of how the canvas happens to render it. Fits the
// single best rotation+scale+translation (2D similarity, no reflection —
// complex least-squares: k = sum(dst_c * conj(src_c)) / sum(|src_c|^2))
// mapping the room centroids SENT to /arap_canvas onto the centroids ARAP
// RETURNED, matched by room id. Near-zero RMS residual means the whole
// plan moved as one rigid body — every room kept its own id/type, it just
// landed somewhere else on screen. A large residual, or any id whose type
// changed, would be the real bug the visual comparison can't distinguish
// from a rotation the raw preview (drawn separately using retrieval's own
// r.rot/r.mirror, never sent to the ARAP backend at all) doesn't share.
function _roomCentroid(poly){
  const n=poly.length;
  return [poly.reduce((s,p)=>s+p[0],0)/n, poly.reduce((s,p)=>s+p[1],0)/n];
}
function _fitSimilarity2D(srcPts,dstPts){
  const n=srcPts.length;
  const sMean=[srcPts.reduce((s,p)=>s+p[0],0)/n,srcPts.reduce((s,p)=>s+p[1],0)/n];
  const dMean=[dstPts.reduce((s,p)=>s+p[0],0)/n,dstPts.reduce((s,p)=>s+p[1],0)/n];
  let numRe=0,numIm=0,den=0;
  for(let i=0;i<n;i++){
    const a=srcPts[i][0]-sMean[0],b=srcPts[i][1]-sMean[1];
    const c=dstPts[i][0]-dMean[0],d=dstPts[i][1]-dMean[1];
    numRe+=c*a+d*b;numIm+=d*a-c*b;den+=a*a+b*b;
  }
  const kRe=numRe/den,kIm=numIm/den;
  const scale=Math.hypot(kRe,kIm),angleDeg=Math.atan2(kIm,kRe)*180/Math.PI;
  let sse=0;
  for(let i=0;i<n;i++){
    const a=srcPts[i][0]-sMean[0],b=srcPts[i][1]-sMean[1];
    const px=kRe*a-kIm*b+dMean[0],py=kIm*a+kRe*b+dMean[1];
    const dx=dstPts[i][0]-px,dy=dstPts[i][1]-py;
    sse+=dx*dx+dy*dy;
  }
  return {angleDeg,scale,rms:Math.sqrt(sse/n)};
}
function logArapAlignFitCheck(sentFp,alignedRooms,r){
  const srcById=new Map(sentFp.rooms.map((rm)=>[rm.id,rm]));
  const rows=[];
  const typeMismatches=[];
  for(const room of alignedRooms){
    const src=srcById.get(room.id);
    if(!src)continue;
    // room.type here is arap_bridge.py's canvas output field, which is the
    // RPlan-native source_type (see arap_bridge._source_type_by_id) — NOT
    // fitting-prototype's own internal `type` (bedroom subtypes consolidated,
    // Kitchen/Bathroom/Dining shifted by one integer). Comparing against
    // src.type compared the wrong scheme and fired on nearly every room
    // (2026-08-17) — src.source_type (falling back to src.type only when
    // source_type is genuinely absent, the pre-2026-08-12 shape) is the
    // correct same-scheme comparison.
    const srcType=src.source_type??src.type;
    if(srcType!==room.type)typeMismatches.push({id:room.id,sentSourceType:srcType,alignedType:room.type});
    rows.push({id:room.id,type:room.type,src:_roomCentroid(src.polygon),dst:_roomCentroid(room.polygon)});
  }
  const fit=_fitSimilarity2D(rows.map((p)=>p.src),rows.map((p)=>p.dst));
  console.log(
    `[arap align fit-check] fitted rotation ${fit.angleDeg.toFixed(1)}° · scale ${fit.scale.toFixed(3)} · `+
    `RMS residual ${fit.rms.toFixed(4)} (normalised units, ~0 = clean single rigid transform) · `+
    `retrieval preview's own display rotation was ${((r.rot||0)*180/Math.PI).toFixed(1)}° (mirror=${!!r.mirror}) — `+
    `NOT what was sent to the ARAP backend, see fp above.`,
  );
  console.table(rows.map((p)=>({
    id:p.id,type:p.type,
    sent_x:p.src[0].toFixed(3),sent_y:p.src[1].toFixed(3),
    aligned_x:p.dst[0].toFixed(3),aligned_y:p.dst[1].toFixed(3),
  })));
  if(typeMismatches.length)console.warn('[arap align fit-check] TYPE MISMATCH — real bug:',typeMismatches);
  else console.log('[arap align fit-check] every room id kept its own type across align.');
}

// ARAP debug steps (arap_adapt.py, via /arap_canvas) — draws real pipeline
// output directly on the app's own main canvas via drawFP, the same
// renderer already trusted for the boundary and retrieval preview. Replaces
// an earlier version (2026-08-11) that ran the same steps but rendered
// through a second conversion into a separate schema.py-fixture modal —
// which made it impossible to tell a real fitting bug from a bug in that
// extra rendering path. Starting with "align" (similarity_align) only, per
// request: verify the SiGraDi-paper alignment step in isolation first.
async function runArapCanvasStep(step, stepLabel){
  if(selected===null||!results[selected]?.fp){
    setStatus('select a retrieved plan first','err');
    return;
  }
  if(!entrance){
    setStatus('place an entrance on the boundary first','err');
    return;
  }
  // ARAP adaptation (arap_solve) always needs an aligned mesh as its
  // starting point — gate it on align having actually been run for this
  // selection, rather than silently re-running align underneath every time.
  if(step==='solve'&&arapAlignedIndex!==selected){
    setStatus('run align first — arap adaptation depends on it','err');
    return;
  }
  if(step==='rectify'&&arapSolvedIndex!==selected){
    setStatus('run arap adaptation first — rectify depends on it','err');
    return;
  }
  if(step==='fill'&&arapRectifiedIndex!==selected){
    setStatus('run rectify first — fill living depends on it','err');
    return;
  }
  // Oracle (surface_defects) and agent (orchestrate_refinement) both need
  // the fully-tessellated fill+consolidate output — the geometry
  // vlm_refine.py was built and validated against.
  if((step==='oracle'||step==='agent')&&arapFilledIndex!==selected){
    setStatus('run fill living first — the refinement stage depends on it','err');
    return;
  }
  const r=results[selected];
  const fp=r.mirror?mirrorPlanY(r.fp):r.fp;
  setStatus(`arap ${stepLabel}…`,'warn');
  // The agent step runs a bounded VLM loop (oracle -> guarded fix ->
  // paired-preference judge, up to 8 iterations, each iteration a real
  // Ollama round-trip) — materially slower than the other steps, which are
  // all fast deterministic geometry.
  const timeoutMs=step==='agent'?300000:60000;
  try{
    const {data}=await postJsonWithTimeout('/arap_canvas',{
      step,
      fp,
      boundary:boundary.map(({x,y})=>[x,y]),
      entrance_point:[entrance.x,entrance.y],
      boundary_typed_edges:boundaryTypedEdgesPayload(),
    },{timeoutMs});
    if(data.error)throw new Error(data.error);
    arapDebugRooms=data.rooms;
    arapDebugDoor=data.door||null;
    arapDebugReport=null;   // only the arrange step carries a judgement report
    arapDebugLabel=`arap ${stepLabel} · #${selected+1} ${r.name||''}`;
    if(step==='align')logArapAlignFitCheck(fp,data.rooms,r);
    if(step==='align')arapAlignedIndex=selected;
    if(step==='solve')arapSolvedIndex=selected;
    if(step==='rectify')arapRectifiedIndex=selected;
    if(step==='fill')arapFilledIndex=selected;
    let extraStatus='';
    if(step==='oracle'){
      console.log(`[arap oracle] ${data.defects.length} candidate defect(s):`,data.defects);
      extraStatus=` — ${data.defects.length} defect(s), see console`;
    }
    if(step==='agent'){
      console.log(`[arap agent] ${data.verified.length} action(s) verified, ${data.log.length} candidate(s) considered (query ${data.query_id}):`,data.verified,data.log);
      extraStatus=` — ${data.verified.length}/${data.log.length} action(s) kept, see console`;
    }
    if(step==='arrange'){
      const rep=data.report||{};
      arapDebugReport={...rep,feasible:data.feasible,mode:data.mode};
      const under=(rep.rooms||[]).filter(r=>!r.ok).map(r=>r.label);
      const edgeTypesStr=(data.boundary_typed_edges||[]).map(e=>e.type||e).join(', ');
      console.log(`[arrange] mode=${data.mode} feasible=${data.feasible} grid=${JSON.stringify(rep.grid)} rot=${rep.rotation_deg||0}° edge-types=[${edgeTypesStr}] clipped=${rep.clipped_to_boundary?'yes':'no'} clip-splits=${rep.clip_splits||0} clipped-out=[${(rep.clipped_out_rooms||[]).join(', ')}] absorbed=${rep.cells_absorbed} enclosures=${rep.enclosures} under-min=[${under.join(', ')}] no-daylight=[${(rep.no_daylight||[]).join(', ')}] no-access=[${(rep.no_access||[]).join(', ')}] dropped-adj=${JSON.stringify(rep.dropped_adjacencies||[])}`,rep);
      extraStatus=` — ${data.mode}${data.feasible?'':', NOT feasible'}${under.length?`, ${under.length} under-min`:''}, see console`;
    }
    render();
    setStatus(`arap ${stepLabel} drawn on canvas${extraStatus}`,'ok');
  }catch(err){
    console.error(`arap_canvas (${step}) failed:`,err);
    setStatus(`arap ${stepLabel} failed (${err.message})`,'err');
  }
}

function clearArapDebugOverlay(){
  arapDebugRooms=null;
  arapDebugLabel=null;
  arapDebugDoor=null;
  arapDebugReport=null;
  arapAlignedIndex=null;
  arapSolvedIndex=null;
  arapRectifiedIndex=null;
  arapFilledIndex=null;
  render();
  setStatus('arap overlay cleared','ok');
}

function toggleFitMode(on){
  const nextState=toggleFitModeState({
    on,
    selected,
    fetchFitted,
    render,
    sync3DSceneIfNeeded,
  });
  fitMode=nextState.fitMode;
  fittedCache=nextState.fittedCache||fittedCache;
}

async function fetchFitted(i){
  fittedCache=await fetchFittedState({
    i,
    results,
    fittedCache,
    boundaryInPlanSpace,
    setStatus,
    fitPlanLocal,
    render,
  });
}

// ── auto-load on startup (requires local server) ───────────────────────────
async function autoLoad(){
  await autoLoadIndex({
    db,
    setStatus,
    tryFinalise,
    doc:document,
    fetchImpl:fetch,
  });
}

if(location.protocol!=='file:'){
  window.addEventListener('DOMContentLoaded',autoLoad);
}
window.addEventListener('DOMContentLoaded',async ()=>{
  await loadServerConfig();
  applyRenderConfigToDom();
  const ids=[
    'render-style',
    'render-view',
    'render-light',
    'render-material',
    'render-context',
    'render-prompt',
    'opacity-mat-walls',
    'opacity-mat-doors',
    'opacity-mat-windows',
    'opacity-mat-slabs',
    'opacity-mat-parapets',
  ];
  ids.forEach((id)=>{
    const el=document.getElementById(id);
    if(!el)return;
    el.addEventListener('input',refreshStructuredPromptPreview);
    el.addEventListener('change',refreshStructuredPromptPreview);
  });
  refreshStructuredPromptPreview();
});

// ── legend ────────────────────────────────────────────────────────────────
(function buildLegend(){
  // source_type → label (matches Python SEMANTICS dict, filtered to renderable types)
  const LABELS={
    0:'Living',1:'Master Bed',2:'Kitchen',3:'Bathroom',4:'Dining',
    5:'Bedroom',6:'Study',7:'Bedroom',8:'Guest',9:'Balcony',
    10:'Entrance',11:'Storage',
  };
  const el=document.getElementById('legend');
  if(!el)return;
  // room types
  Object.entries(LABELS).forEach(([st,name])=>{
    const col=RCOLS[+st]??'#ddd';
    const row=document.createElement('div');
    row.className='legend-row';
    row.innerHTML=`<span class="legend-swatch" style="background:${col}"></span>${name}`;
    el.appendChild(row);
  });
  // separator
  const sep=document.createElement('div');
  sep.style.cssText='border-top:1px solid var(--border);margin:5px 0';
  el.appendChild(sep);
  // doors & windows
  [
    ['rgba(251,146,60,0.90)','Door','door'],
    ['rgba(103,232,249,0.85)','Window','window'],
  ].forEach(([col,name,cls])=>{
    const row=document.createElement('div');
    row.className='legend-row';
    row.innerHTML=`<span class="legend-swatch ${cls}" style="background:${col}"></span>${name}`;
    el.appendChild(row);
  });
})();

function wirePhase1UI(){
  setupDevModeUI();
  const on=(id,eventName,handler)=>{
    const el=document.getElementById(id);
    if(el)el.addEventListener(eventName,handler);
  };
  const clickFileTarget=(triggerId,inputId)=>{
    on(triggerId,'click',()=>document.getElementById(inputId)?.click());
  };
  const bindRange=(inputId,labelId,assign,formatter=(value)=>value)=>{
    const input=document.getElementById(inputId);
    const label=document.getElementById(labelId);
    if(!input||!label)return;
    const update=()=>{
      label.textContent=formatter(input.value);
      assign(+input.value);
    };
    input.addEventListener('input',update);
    update();
  };

  clickFileTarget('dz-bin','fi-bin');
  clickFileTarget('dz-nms','fi-nms');
  clickFileTarget('dz-pls','fi-pls');

  on('btn-clear-plan-cache','click',()=>clearPlanCache());
  on('btn-clear-boundary','click',()=>clearBoundary());
  on('btn-save-boundary','click',()=>saveBoundary());
  on('btn-entrance-mode','click',()=>toggleEntranceMode());
  on('btn-clear-entrance','click',()=>clearEntrance());
  on('btn-edge-type-mode','click',()=>toggleEdgeTypeMode());
  on('btn-feasibility-check','click',()=>runFeasibilityCheck());
  on('btn-swap-nodes','click',()=>runSwapNodes());
  on('btn-swap-reset','click',()=>resetSwap());
  on('btn-export-library','click',()=>exportLibrary());
  on('fi-import-library','change',(e)=>importLibrary(e.target));

  bindRange('rng-bedrooms','v-bed',(value)=>{req.bedrooms=value;});
  bindRange('rng-bathrooms','v-bath',(value)=>{req.bathrooms=value;});
  on('chk-balcony','change',(e)=>{req.balcony=e.target.checked;});
  on('chk-study','change',(e)=>{req.study=e.target.checked;});
  on('chk-storage','change',(e)=>{req.storage=e.target.checked;});
  bindRange('rng-topn','v-topn',(value)=>{params.topn=value;});
  bindRange('rng-shape-weight','v-sw',(value)=>{params.shape_w=value;},(value)=>(+value).toFixed(1));
  bindRange('rng-entrance-weight','v-entw',(value)=>{params.ent_w=value;},(value)=>(+value).toFixed(1));
  bindRange('rng-room-weight','v-rmw',(value)=>{params.room_w=value;},(value)=>(+value).toFixed(1));

  on('chk-graph','change',(e)=>{params.showGraph=e.target.checked;render();});
  on('chk-plan-geom','change',(e)=>{params.showPlanGeom=e.target.checked;render();});
  on('run-btn','click',()=>runRetrieval());
  on('btn-draw-architectural','click',()=>drawSelectedArchitecturally());
  on('btn-arap-align-canvas','click',()=>runArapCanvasStep('align','align'));
  on('btn-arap-solve-canvas','click',()=>runArapCanvasStep('solve','arap adaptation'));
  on('btn-arap-rectify-canvas','click',()=>runArapCanvasStep('rectify','rectify'));
  on('btn-arap-fill-canvas','click',()=>runArapCanvasStep('fill','fill living'));
  on('btn-arap-oracle-canvas','click',()=>runArapCanvasStep('oracle','oracle'));
  on('btn-arap-agent-canvas','click',()=>runArapCanvasStep('agent','agent refine'));
  on('btn-arap-arrange-canvas','click',()=>runArapCanvasStep('arrange','arrange'));
  on('btn-arap-clear-canvas','click',()=>clearArapDebugOverlay());
  on('tab-fp','click',()=>setViewTab('fp'));
  on('tab-2d','click',()=>setViewTab('2d'));
  on('tab-fit','click',()=>setViewTab('fit'));
  on('tab-3d','click',()=>setViewTab('3d'));

  // ── Fit palette controls ──────────────────────────────────────────────
  on('fit-show-graph','change',(e)=>{fitState.showGraph=e.target.checked;render();});
  on('fit-show-original','change',(e)=>{fitState.showOriginal=e.target.checked;render();});
  on('fit-show-boundary','change',(e)=>{fitState.showBoundary=e.target.checked;render();});
  on('fit-run-btn','click',()=>_fitRunStep());
  on('btn-generate-3d','click',()=>onGenerate3DClick());
  on('chk-show-bdry','change',(e)=>{showPlanBdry=e.target.checked;render();});
  on('chk-show-win','change',(e)=>{showWindows=e.target.checked;render();});
  on('chk-show-idoors','change',(e)=>{showDoors=e.target.checked;render();});
  on('chk-show-common','change',(e)=>{showCommonArea=e.target.checked;render();});
  on('btn-rebuild-scene','click',()=>sync3DSceneIfNeeded(true));
  on('btn-capture-maps','click',()=>captureDepthAndPreview());
  on('btn-collapse-render','click',()=>togglePanelCollapse('render-panel','render-body','btn-collapse-render'));
  on('btn-check-comfy','click',()=>checkComfyConnection());
  on('workflow-file','change',(e)=>loadWorkflowOverride(e.target));
  on('btn-clear-workflow','click',()=>clearWorkflowOverride());
  on('btn-generate-render','click',()=>generateRenderMock());
  on('btn-collapse-capture','click',()=>togglePanelCollapse('capture-panel','capture-body','btn-collapse-capture'));
  on('capture-view-select','change',(e)=>setCaptureView(e.target.value));
  on('btn-download-capture','click',()=>downloadLatestCaptureBundle());
}

if(document.readyState==='loading'){
  window.addEventListener('DOMContentLoaded',wirePhase1UI);
}else{
  wirePhase1UI();
}
