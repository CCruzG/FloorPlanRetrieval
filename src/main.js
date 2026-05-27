import {
  METRES_PER_UNIT,
  PLAN_CACHE,
  RCOLS,
  SNAP_PX,
} from './constants.js';
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
  combinedDesc,
  entranceDescJS,
  l2dist,
  roomCountDescFromReqs,
  shapeDesc,
} from './descriptors.js';
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
  renderAdaptedResultsView,
  renderResultsListView,
  runRetrievalFlow,
  runLocalRetrievalData,
  selectResultState,
  setStatusUI,
  setupDropZone,
  toggleFitModeState,
  tryFinalizeIndex,
} from './retrieval.js';
import { createRenderPipeline } from './render-pipeline.js';
import { createViewer3D } from './viewer-3d.js';
import {
  analyse_boundary,
  check_typology_viability,
  evaluate_typology_fit,
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
let {req,params,boundary,closed,entrance,entranceMode,entranceSnap,mouse,shiftDown,snapFirst,results,selected,fitMode,fittedCache,showPlanBdry,showWindows,showDoors,showCommonArea,selectedRoom,hoveredRoom,hoveredEntrance,dragState,displacementArrows,cW,cH,activeTab,floorPlateState}=initialState;

devMode.enabled=new URLSearchParams(window.location.search).get('dev')==='true';
devMode.panelVisible=devMode.enabled;

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
}

const canvas=document.getElementById('main-canvas');
const ctx=canvas.getContext('2d');
let syncDevModePanel=()=>{};
let devModeSetupDone=false;

function buildMockFloorPlateState(){
  return buildFloorPlateFixture('sample');
}

function makeMockComplianceReport(fixtureId){
  if(fixtureId==='near_failing_narrow'){
    return {
      requirements:[
        {name:'Minimum clear depth',expected:'>= 14.0 m',actual:'12.9 m',status:'fail'},
        {name:'Corridor minimum width',expected:'>= 1.2 m',actual:'1.2 m',status:'pass'},
        {name:'Travel distance',expected:'<= 30 m',actual:'34.4 m',status:'blocked'},
        {name:'Unit mix compliance',expected:'2x1bed, 2x2bed',actual:'2x1bed, 1x2bed',status:'fail'},
      ],
    };
  }
  if(fixtureId==='elongated_slab'){
    return {
      requirements:[
        {name:'Minimum clear depth',expected:'>= 14.0 m',actual:'15.2 m',status:'pass'},
        {name:'Corridor minimum width',expected:'>= 1.2 m',actual:'1.5 m',status:'pass'},
        {name:'Travel distance',expected:'<= 30 m',actual:'28.3 m',status:'pass'},
        {name:'Unit mix compliance',expected:'3x1bed, 3x2bed',actual:'3x1bed, 3x2bed',status:'pass'},
      ],
    };
  }
  if(fixtureId==='compact_slab'){
    return {
      requirements:[
        {name:'Minimum clear depth',expected:'>= 14.0 m',actual:'16.1 m',status:'pass'},
        {name:'Core footprint',expected:'>= 20 m2',actual:'22.8 m2',status:'pass'},
        {name:'Typed edge consistency',expected:'all units valid',actual:'all units valid',status:'pass'},
        {name:'No residual areas',expected:'none',actual:'none',status:'pass'},
      ],
    };
  }
  return {
    requirements:[
      {name:'Minimum clear depth',expected:'>= 14.0 m',actual:'14.8 m',status:'pass'},
      {name:'Core footprint',expected:'>= 20 m2',actual:'21.6 m2',status:'pass'},
      {name:'Corridor minimum width',expected:'>= 1.2 m',actual:'1.4 m',status:'pass'},
      {name:'Unit mix compliance',expected:'2x1bed, 2x2bed',actual:'2x1bed, 2x2bed',status:'pass'},
    ],
  };
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
  devMode.lastComplianceReport=makeMockComplianceReport(fixtureId);
  devMode.lastToolOutput={
    source:'fixture_loader',
    fixtureId,
    loadedAt:new Date().toISOString(),
  };
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

function updateDevComplianceFromTool(toolResult){
  if(Array.isArray(toolResult?.requirements)){
    devMode.lastComplianceReport={requirements:toolResult.requirements};
    return;
  }
  if(Array.isArray(toolResult?.constraint_violations)){
    const violations=toolResult.constraint_violations;
    devMode.lastComplianceReport={
      requirements:[
        {
          name:'Typology feasibility',
          expected:'no blocking violations',
          actual:violations.length?'violations found':'pass',
          status:violations.length?'fail':'pass',
        },
        ...violations.map((violation)=>({
          name:'Constraint',
          expected:'satisfied',
          actual:violation,
          status:'fail',
        })),
      ],
    };
    return;
  }
  devMode.lastComplianceReport=null;
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
  }

  devMode.lastToolOutput={
    tool:toolName,
    unitMix,
    boundaryVertices:boundaryPolygon.length,
    result:output,
  };
  updateDevComplianceFromTool(output);
  syncDevModePanel();
  render();
}

function setupDevModeUI(){
  if(devModeSetupDone)return;
  devModeSetupDone=true;

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
    .fp-dev-row { display:flex; gap:6px; align-items:center; margin-bottom:6px; }
    .fp-dev-row:last-child { margin-bottom:0; }
    .fp-dev-row-wrap { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:6px; }
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

    <section class="fp-dev-section">
      <div class="fp-dev-section-title">Tool Call Panel</div>
      <div class="fp-dev-row">
        <label><input id="fp-dev-overlay" type="checkbox" checked> show floor plate overlay</label>
      </div>
      <div class="fp-dev-row">
        <label><input id="fp-dev-show-ids" type="checkbox" checked> show element IDs</label>
      </div>
      <div class="fp-dev-row">
        <label><input id="fp-dev-show-edges" type="checkbox"> show edge indices</label>
      </div>
      <div class="fp-dev-row-wrap">
        <button id="fp-dev-fixture-compact" class="fp-dev-btn" type="button">fixture: compact slab</button>
        <button id="fp-dev-fixture-elongated" class="fp-dev-btn" type="button">fixture: elongated slab</button>
        <button id="fp-dev-fixture-narrow" class="fp-dev-btn" type="button">fixture: near-failing narrow</button>
      </div>
      <div class="fp-dev-row">
        <label>1-bed</label><input id="fp-dev-mix-bed1" class="fp-dev-input" type="number" min="0" step="1" value="2">
        <label>2-bed</label><input id="fp-dev-mix-bed2" class="fp-dev-input" type="number" min="0" step="1" value="2">
        <label>3-bed</label><input id="fp-dev-mix-bed3" class="fp-dev-input" type="number" min="0" step="1" value="0">
      </div>
      <div class="fp-dev-row">
        <select id="fp-dev-typology" class="fp-dev-select">
          <option value="double_loaded_slab">double_loaded_slab</option>
          <option value="single_loaded">single_loaded</option>
          <option value="point_block">point_block</option>
        </select>
      </div>
      <div class="fp-dev-row-wrap">
        <button id="fp-dev-run-analyse" class="fp-dev-btn" type="button">run analyse_boundary</button>
        <button id="fp-dev-run-evaluate" class="fp-dev-btn" type="button">run evaluate_typology_fit</button>
        <button id="fp-dev-run-propose" class="fp-dev-btn" type="button">run propose_typology_options</button>
        <button id="fp-dev-run-check" class="fp-dev-btn" type="button">run check_typology_viability</button>
      </div>
      <div class="fp-dev-row">
        <button id="fp-dev-seed" class="fp-dev-btn" type="button">seed sample floor plate</button>
        <button id="fp-dev-clear" class="fp-dev-btn" type="button">clear floor plate</button>
      </div>
      <div class="fp-dev-pre" id="fp-dev-tools">
Run Stage 1 tools from this panel. Inputs use metres in outputs, while geometry
is transformed internally from the normalized editor coordinate system.
      </div>
    </section>

    <section class="fp-dev-section">
      <div class="fp-dev-section-title">State Inspector</div>
      <div class="fp-dev-pre" id="fp-dev-state"></div>
    </section>

    <section class="fp-dev-section" style="margin-bottom:0">
      <div class="fp-dev-section-title">Compliance Log</div>
      <table id="fp-dev-compliance-table">
        <thead>
          <tr><th>Requirement</th><th>Expected</th><th>Actual</th><th>Status</th></tr>
        </thead>
        <tbody id="fp-dev-compliance-body"></tbody>
      </table>
    </section>
  `;
  wrap.appendChild(panel);

  const hideBtn=document.getElementById('fp-dev-hide');
  const overlayToggle=document.getElementById('fp-dev-overlay');
  const showIdsToggle=document.getElementById('fp-dev-show-ids');
  const showEdgesToggle=document.getElementById('fp-dev-show-edges');
  const seedBtn=document.getElementById('fp-dev-seed');
  const clearBtn=document.getElementById('fp-dev-clear');
  const compactBtn=document.getElementById('fp-dev-fixture-compact');
  const elongatedBtn=document.getElementById('fp-dev-fixture-elongated');
  const narrowBtn=document.getElementById('fp-dev-fixture-narrow');
  const runAnalyseBtn=document.getElementById('fp-dev-run-analyse');
  const runEvaluateBtn=document.getElementById('fp-dev-run-evaluate');
  const runProposeBtn=document.getElementById('fp-dev-run-propose');
  const runCheckBtn=document.getElementById('fp-dev-run-check');
  const mixBed1El=document.getElementById('fp-dev-mix-bed1');
  const mixBed2El=document.getElementById('fp-dev-mix-bed2');
  const mixBed3El=document.getElementById('fp-dev-mix-bed3');

  if(overlayToggle)overlayToggle.checked=!!devMode.showFloorPlateOverlay;
  if(showIdsToggle)showIdsToggle.checked=!!devMode.showElementIds;
  if(showEdgesToggle)showEdgesToggle.checked=!!devMode.showEdgeIndices;
  if(mixBed1El)mixBed1El.value=String(devMode.unitMix?.bed1??2);
  if(mixBed2El)mixBed2El.value=String(devMode.unitMix?.bed2??2);
  if(mixBed3El)mixBed3El.value=String(devMode.unitMix?.bed3??0);
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
  [mixBed1El,mixBed2El,mixBed3El].forEach((el)=>{
    el?.addEventListener('input',()=>{
      devMode.unitMix={
        bed1:Math.max(0,parseInt(mixBed1El?.value||'0',10)||0),
        bed2:Math.max(0,parseInt(mixBed2El?.value||'0',10)||0),
        bed3:Math.max(0,parseInt(mixBed3El?.value||'0',10)||0),
      };
      syncDevModePanel();
    });
  });
  runAnalyseBtn?.addEventListener('click',()=>runDevStage1Tool('analyse_boundary'));
  runEvaluateBtn?.addEventListener('click',()=>runDevStage1Tool('evaluate_typology_fit'));
  runProposeBtn?.addEventListener('click',()=>runDevStage1Tool('propose_typology_options'));
  runCheckBtn?.addEventListener('click',()=>runDevStage1Tool('check_typology_viability'));
  compactBtn?.addEventListener('click',()=>{
    loadFixture('compact_slab');
    render();
  });
  elongatedBtn?.addEventListener('click',()=>{
    loadFixture('elongated_slab');
    render();
  });
  narrowBtn?.addEventListener('click',()=>{
    loadFixture('near_failing_narrow');
    render();
  });
  seedBtn?.addEventListener('click',()=>{
    loadFixture('sample');
    render();
  });
  clearBtn?.addEventListener('click',()=>{
    floorPlateState=null;
    devMode.activeFixture=null;
    devMode.lastComplianceReport=null;
    devMode.lastToolOutput=null;
    render();
  });

  syncDevModePanel=()=>{
    const stateEl=document.getElementById('fp-dev-state');
    const bodyEl=document.getElementById('fp-dev-compliance-body');
    const panelEl=document.getElementById('fp-dev-panel');
    const overlayEl=document.getElementById('fp-dev-overlay');
    if(!stateEl||!bodyEl||!panelEl)return;

    panelEl.classList.toggle('show',!!devMode.enabled&&!!devMode.panelVisible);
    if(overlayEl)overlayEl.checked=!!devMode.showFloorPlateOverlay;
    if(showIdsToggle)showIdsToggle.checked=!!devMode.showElementIds;
    if(showEdgesToggle)showEdgesToggle.checked=!!devMode.showEdgeIndices;
    if(mixBed1El)mixBed1El.value=String(devMode.unitMix?.bed1??0);
    if(mixBed2El)mixBed2El.value=String(devMode.unitMix?.bed2??0);
    if(mixBed3El)mixBed3El.value=String(devMode.unitMix?.bed3??0);

    const summary={
      devModeEnabled:!!devMode.enabled,
      panelVisible:!!devMode.panelVisible,
      overlayEnabled:!!devMode.showFloorPlateOverlay,
      showElementIds:!!devMode.showElementIds,
      showEdgeIndices:!!devMode.showEdgeIndices,
      unitMix:getDevUnitMix(),
      activeFixture:devMode.activeFixture||null,
      floorPlateLoaded:!!floorPlateState,
      floorPlateId:floorPlateState?.floor_plate?.id||floorPlateState?.id||null,
      unitCount:(floorPlateState?.floor_plate?.units||floorPlateState?.units||[]).length,
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

    const report=devMode.lastComplianceReport;
    const rows=Array.isArray(report?.requirements)
      ? report.requirements
      : [];
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
      const color=status==='pass'?'#22c55e':(status==='blocked'?'#f59e0b':'#ef4444');
      tr.innerHTML=`
        <td>${item.name||'requirement'}</td>
        <td>${item.expected??'-'}</td>
        <td>${item.actual??'-'}</td>
        <td style="color:${color}">${status}</td>
      `;
      bodyEl.appendChild(tr);
    });
  };

  syncDevModePanel();
}

function n2c(nx,ny){
  return geometryN2C(nx,ny,{cW,cH});
}

function c2n(cx,cy){
  return geometryC2N(cx,cy,{cW,cH});
}

function pdist(a,b){
  return geometryPDist(a,b,{cW,cH});
}

function planToCanvas(px,py,fpBdry,rot,offset,mirror){
  return geometryPlanToCanvas(px,py,fpBdry,rot,offset,mirror,{cW,cH});
}

function canvasToPlan(cx,cy,fpBdry,rot,offset,mirror){
  return geometryCanvasToPlan(cx,cy,fpBdry,rot,offset,mirror,{cW,cH});
}

function projSnap(nx,ny){
  return geometryProjSnap(nx,ny,{boundary,closed,cW,cH});
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
  const S=Math.min(wrap.clientWidth-32,wrap.clientHeight-32);
  cW=S;cH=S;
  const dpr=window.devicePixelRatio||1;
  canvas.width=S*dpr;canvas.height=S*dpr;
  canvas.style.width=S+'px';canvas.style.height=S+'px';
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

function applyRenderConfigToDom(){
  const promptConfig=renderRequestState.promptConfig||DEFAULT_RENDER_PROMPT_CONFIG;
  const setValue=(id,value)=>{
    const el=document.getElementById(id);
    if(el&&typeof value==='string')el.value=value;
  };
  setValue('render-style',promptConfig.style);
  setValue('render-view',promptConfig.view);
  setValue('render-light',promptConfig.lighting);
  setValue('render-material',promptConfig.material);
  setValue('render-prompt',promptConfig.siteContext||'');
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
let renderFloorPlate;
let render=()=>{};
let updateCursor=()=>{};

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
  const tab2d=document.getElementById('tab-2d');
  const tab3d=document.getElementById('tab-3d');

  const is2d=tab==='2d';
  panel2d.style.display=is2d?'flex':'none';
  panel3d.style.display=is2d?'none':'flex';
  tab2d.classList.toggle('active',is2d);
  tab3d.classList.toggle('active',!is2d);

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
  renderFloorPlate,
  render,
  updateCursor,
} = createCanvasRenderer({
  canvas,
  ctx,
  getState:()=>({
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
    hoveredRoom,
    hoveredEntrance,
    dragState,
    displacementArrows,
    cW,
    cH,
    params,
    floorPlateState,
    devMode,
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
};

// ── mouse ─────────────────────────────────────────────────────────────────
canvas.addEventListener('mousedown',e=>{
  if(!closed||selected===null||entranceMode)return;
  const r=canvas.getBoundingClientRect();
  const cx=(e.clientX-r.left)*(cW/r.width),cy=(e.clientY-r.top)*(cH/r.height);
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

document.addEventListener('mouseup',()=>{
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
  const r=canvas.getBoundingClientRect();
  const cx=(e.clientX-r.left)*(cW/r.width),cy=(e.clientY-r.top)*(cH/r.height);
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

  // ── hover detection for room / entrance editing ─────────────────────
  if(selected!==null&&closed&&!entranceMode){
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
  const cx=(e.clientX-r.left)*(cW/r.width),cy=(e.clientY-r.top)*(cH/r.height);
  let[nx,ny]=c2n(cx,cy);

  // entrance placement mode
  if(entranceMode&&closed){
    const snap=snapToEdge(nx,ny);
    if(snap){
      entrance={x:snap.px,y:snap.py};
      entranceMode=false;
      entranceSnap=null;
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

document.addEventListener('keydown',e=>{
  if((e.ctrlKey||e.metaKey)&&e.shiftKey&&e.key.toLowerCase()==='d'){
    e.preventDefault();
    devMode.enabled=!devMode.enabled;
    devMode.panelVisible=devMode.enabled;
    syncDevModePanel();
    render();
    return;
  }
  if(e.target&&e.target.matches&&e.target.matches('input, textarea, select'))return;
  if(e.key==='Escape')clearBoundary();
  if(e.key==='Enter'&&boundary.length>=3&&!closed)closeBoundary();
  if(e.key==='Shift')shiftDown=true;
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

function tryFinalise(){
  return tryFinalizeIndex({
    db,
    onReady:()=>{setStatus('index ready','ok');checkReady();},
    onError:(msg)=>setStatus(msg,'err'),
    doc:document,
  });
}

function checkReady(){
  checkReadyState({db,closed,boundary},document);
}
function setStatus(msg,cls=''){
  setStatusUI(msg,cls,document);
}

// ── retrieval ─────────────────────────────────────────────────────────────
async function runRetrievalLocal(){
  if(!db.descs||!closed||boundary.length<3)return;
  setStatus('retrieving…','warn');
  await new Promise(r=>setTimeout(r,10));
  try{
    results=await runLocalRetrievalData({
      db,
      boundary,
      entrance,
      req,
      params,
      cache,
      fetchWithCacheFn:fetchWithCache,
      shapeDescFn:shapeDesc,
      entranceDescFn:entranceDescJS,
      roomCountDescFn:roomCountDescFromReqs,
      combinedDescFn:combinedDesc,
      l2distFn:l2dist,
    });

    fittedCache={};
    selected=null;
    renderList();
    setStatus(`top ${params.topn} retrieved (local)`,'ok');
    document.getElementById('result-count').textContent=results.length+' results';
    render();
    sync3DSceneIfNeeded(true);
  }catch(e){setStatus('error: '+e.message,'err');console.error(e);}
}

async function runRetrieval(){
  await runRetrievalFlow({
    closed,
    boundary,
    entrance,
    req,
    params,
    db,
    setStatus,
    runRetrievalLocal,
    applyServerPlans:renderAdaptedResults,
    afterRender:()=>render(),
    checkReady,
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
  });
}

function selectResult(i){
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
    if(mirror)px=2*pc[0]-px;
    return[px,py];
  });
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
  on('btn-export-library','click',()=>exportLibrary());
  on('fi-import-library','change',(e)=>importLibrary(e.target));

  bindRange('rng-bedrooms','v-bed',(value)=>{req.bedrooms=value;});
  bindRange('rng-bathrooms','v-bath',(value)=>{req.bathrooms=value;});
  bindRange('rng-topn','v-topn',(value)=>{params.topn=value;});
  bindRange('rng-shape-weight','v-sw',(value)=>{params.shape_w=value;},(value)=>(+value).toFixed(1));
  bindRange('rng-entrance-weight','v-entw',(value)=>{params.ent_w=value;},(value)=>(+value).toFixed(1));
  bindRange('rng-room-weight','v-rmw',(value)=>{params.room_w=value;},(value)=>(+value).toFixed(1));

  on('chk-graph','change',(e)=>{params.showGraph=e.target.checked;render();});
  on('chk-fit','change',(e)=>toggleFitMode(e.target.checked));
  on('run-btn','click',()=>runRetrieval());
  on('tab-2d','click',()=>setViewTab('2d'));
  on('tab-3d','click',()=>setViewTab('3d'));
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
