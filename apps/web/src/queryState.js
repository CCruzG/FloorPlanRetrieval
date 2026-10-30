export function getActivePlanView({selected,results,fitMode,fittedCache,boundary,centerOfBoundary}){
  if(selected===null||!results[selected])return null;
  const result=results[selected];
  const rot=result.rot||0;
  const mirror=result.mirror||false;
  const fp=(fitMode&&fittedCache[selected]?.fp)||result.fp;
  if(!fp)return null;
  const queryCenter=centerOfBoundary(boundary.map(({x,y})=>[x,y]));
  const planBoundary=fp.boundary||[];
  const planCenter=centerOfBoundary(planBoundary);
  const offset=[queryCenter[0]-planCenter[0],queryCenter[1]-planCenter[1]];
  return {fp,rot,mirror,offset,fpBdry:planBoundary};
}

export function createBoundaryClosedState(){
  return {
    closed:true,
    snapFirst:false,
    mode:'select',          // finishing a drawn boundary always lands back in select (2026-10-30)
  };
}

export function createBoundaryClearedState(){
  return {
    boundary:[],
    closed:false,
    snapFirst:false,
    selected:null,
    results:[],
    selectedRoom:null,
    dragState:null,
    hoveredRoom:null,
    hoveredEntrance:false,
    displacementArrows:[],
    entrance:null,
    entranceMode:false,
    entranceSnap:null,
    edgeTypes:[],
    edgeTypeMode:false,
    hoveredEdge:null,
    mode:'select',          // clearing never leaves the app in draw mode — re-enter via the button
  };
}

export function createLoadedBoundaryState(entry){
  return {
    boundary:entry.vertices.map(({x,y})=>({x,y})),
    entrance:entry.entrance?{x:entry.entrance.x,y:entry.entrance.y}:null,
    closed:true,
    snapFirst:false,
    selected:null,
    results:[],
    edgeTypes:Array.isArray(entry.edgeTypes)?entry.edgeTypes.slice():[],
    edgeTypeMode:false,
    hoveredEdge:null,
    mode:'select',          // a loaded boundary is already closed — select, not draw (2026-10-30)
  };
}

export function createAdaptedResults(plans){
  return plans.map((plan)=>{
    const retrieval=plan.retrieval||{};
    return {
      name:plan.id,
      dist:retrieval.dist??0,
      fp:plan,
      adapted:true,
      retrieval,
      rot:retrieval.rot??0,
      mirror:retrieval.mirror??false,
    };
  });
}

export function createResultSelectionState(currentSelected,nextSelected){
  return {
    selected:currentSelected===nextSelected?null:nextSelected,
    selectedRoom:null,
    dragState:null,
    hoveredRoom:null,
    hoveredEntrance:false,
    displacementArrows:[],
  };
}

export function createFitModeState(on){
  return {
    fitMode:on,
    fittedCache:on?null:{},
  };
}