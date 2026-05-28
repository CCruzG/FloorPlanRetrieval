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
  evaluate_floor_plate_requirements,
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
let viewport={zoom:1.0,panX:0,panY:0};
let panDrag=null;
let syncDevModePanel=()=>{};
let devModeSetupDone=false;

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
      <button type="button" class="fp-dev-tab active" data-tab="tools">tools</button>
      <button type="button" class="fp-dev-tab" data-tab="requirements">requirements</button>
      <button type="button" class="fp-dev-tab" data-tab="compliance">compliance</button>
      <button type="button" class="fp-dev-tab" data-tab="state">state</button>
    </nav>

    <section class="fp-dev-section fp-dev-pane active" data-pane="tools">
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
  const seedBtn=document.getElementById('fp-dev-seed');
  const clearBtn=document.getElementById('fp-dev-clear');
  const compactBtn=document.getElementById('fp-dev-fixture-compact');
  const elongatedBtn=document.getElementById('fp-dev-fixture-elongated');
  const narrowBtn=document.getElementById('fp-dev-fixture-narrow');
  const runAnalyseBtn=document.getElementById('fp-dev-run-analyse');
  const runEvaluateBtn=document.getElementById('fp-dev-run-evaluate');
  const runProposeBtn=document.getElementById('fp-dev-run-propose');
  const runCheckBtn=document.getElementById('fp-dev-run-check');
  const runReqChecksBtn=document.getElementById('fp-dev-run-req-checks');
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
  if(mixBed1El)mixBed1El.value=String(devMode.unitMix?.bed1??2);
  if(mixBed2El)mixBed2El.value=String(devMode.unitMix?.bed2??2);
  if(mixBed3El)mixBed3El.value=String(devMode.unitMix?.bed3??0);

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
    runRequirementsCompliance();
    devMode.lastToolOutput=null;
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
    if(mixBed1El)mixBed1El.value=String(devMode.unitMix?.bed1??0);
    if(mixBed2El)mixBed2El.value=String(devMode.unitMix?.bed2??0);
    if(mixBed3El)mixBed3El.value=String(devMode.unitMix?.bed3??0);

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

    const summary={
      devModeEnabled:!!devMode.enabled,
      panelVisible:!!devMode.panelVisible,
      overlayEnabled:!!devMode.showFloorPlateOverlay,
      showElementIds:!!devMode.showElementIds,
      showEdgeIndices:!!devMode.showEdgeIndices,
      floorPlateLayers:devMode.floorPlateLayers,
      unitMix:getDevUnitMix(),
      requirements:devMode.requirements,
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

function planToCanvas(px,py,fpBdry,rot,offset,mirror){
  const s=_baseScale(),ox=_baseOx(s),oy=_baseOy(s);
  const cen=(rot||mirror)?bdryCenter(fpBdry):null;
  let x=px,y=py;
  if(mirror)x=2*cen[0]-x;
  if(rot)[x,y]=rotatePtAround([x,y],rot,cen);
  return[(x+offset[0])*s+ox,(y+offset[1])*s+oy];
}

function canvasToPlan(lx,ly,fpBdry,rot,offset,mirror){
  const s=_baseScale(),ox=_baseOx(s),oy=_baseOy(s);
  const cen=(rot||mirror)?bdryCenter(fpBdry):null;
  let x=(lx-ox)/s-offset[0];
  let y=(ly-oy)/s-offset[1];
  if(rot)[x,y]=rotatePtAround([x,y],-rot,cen);
  if(mirror)x=2*cen[0]-x;
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
    viewport,
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
  // Middle button: start pan
  if(e.button===1){
    e.preventDefault();
    panDrag={startX:e.clientX,startY:e.clientY,startPanX:viewport.panX,startPanY:viewport.panY};
    canvas.style.cursor='grab';
    return;
  }
  if(!closed||selected===null||entranceMode)return;
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
  const rawX=(e.clientX-r.left)*(cW/r.width),rawY=(e.clientY-r.top)*(cH/r.height);
  const [cx,cy]=cssToLogical(rawX,rawY);
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
