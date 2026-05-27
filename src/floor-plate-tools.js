import { METRES_PER_UNIT } from './constants.js';

const MIN_AREA_BY_TYPE_M2={
  bed1:45,
  bed2:65,
  bed3:85,
};

function signedArea(poly){
  let sum=0;
  for(let i=0;i<poly.length;i++){
    const [x1,y1]=poly[i];
    const [x2,y2]=poly[(i+1)%poly.length];
    sum+=x1*y2-x2*y1;
  }
  return 0.5*sum;
}

function polygonAreaUnits(poly){
  return Math.abs(signedArea(poly));
}

function polygonPerimeterUnits(poly){
  let p=0;
  for(let i=0;i<poly.length;i++){
    const [x1,y1]=poly[i];
    const [x2,y2]=poly[(i+1)%poly.length];
    p+=Math.hypot(x2-x1,y2-y1);
  }
  return p;
}

function convexHull(points){
  if(points.length<=3)return points.slice();
  const pts=points
    .slice()
    .sort((a,b)=>a[0]===b[0]?a[1]-b[1]:a[0]-b[0]);
  const cross=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);
  const lower=[];
  pts.forEach((pt)=>{
    while(lower.length>=2&&cross(lower[lower.length-2],lower[lower.length-1],pt)<=0)lower.pop();
    lower.push(pt);
  });
  const upper=[];
  for(let i=pts.length-1;i>=0;i--){
    const pt=pts[i];
    while(upper.length>=2&&cross(upper[upper.length-2],upper[upper.length-1],pt)<=0)upper.pop();
    upper.push(pt);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function toMetres(units){
  return units*METRES_PER_UNIT;
}

function toSquareMetres(unitsArea){
  return unitsArea*METRES_PER_UNIT*METRES_PER_UNIT;
}

function normalizeUnitMix(unitMix={}){
  return {
    bed1:Math.max(0,parseInt(unitMix.bed1,10)||0),
    bed2:Math.max(0,parseInt(unitMix.bed2,10)||0),
    bed3:Math.max(0,parseInt(unitMix.bed3,10)||0),
  };
}

function minAreaForMix(unitMix){
  return Object.entries(normalizeUnitMix(unitMix)).reduce((total,[type,count])=>{
    const minArea=MIN_AREA_BY_TYPE_M2[type]||0;
    return total+count*minArea;
  },0);
}

function requiredDepthMetres(typology){
  if(typology==='single_loaded')return 8;
  if(typology==='double_loaded_slab')return 14;
  return 10;
}

function defaultEfficiency(typology){
  if(typology==='single_loaded')return 0.8;
  if(typology==='point_block')return 0.74;
  return 0.78;
}

export function analyse_boundary(boundary_polygon=[]){
  if(!Array.isArray(boundary_polygon)||boundary_polygon.length<3){
    return { valid:false, error:'Boundary must have at least 3 vertices.' };
  }
  const xs=boundary_polygon.map(([x])=>x);
  const ys=boundary_polygon.map(([,y])=>y);
  const minX=Math.min(...xs);
  const maxX=Math.max(...xs);
  const minY=Math.min(...ys);
  const maxY=Math.max(...ys);
  const spanX=maxX-minX;
  const spanY=maxY-minY;
  const minSpan=Math.min(spanX,spanY);
  const maxSpan=Math.max(spanX,spanY);
  const areaUnits=polygonAreaUnits(boundary_polygon);
  const perimeterUnits=polygonPerimeterUnits(boundary_polygon);
  const hull=convexHull(boundary_polygon);
  const hullAreaUnits=Math.max(polygonAreaUnits(hull),1e-9);

  return {
    valid:true,
    area_m2:toSquareMetres(areaUnits),
    perimeter_m:toMetres(perimeterUnits),
    dimensions:{
      min_span_m:toMetres(minSpan),
      max_span_m:toMetres(maxSpan),
      aspect_ratio:maxSpan/Math.max(minSpan,1e-9),
    },
    orientation:{
      principal_axis_angle_deg:spanX>=spanY?0:90,
      axis_hint:spanX>=spanY?'x_major':'y_major',
    },
    perimeter_length_m:toMetres(perimeterUnits),
    edge_count:boundary_polygon.length,
    convexity_score:Math.min(1,Math.max(0,areaUnits/hullAreaUnits)),
  };
}

export function evaluate_typology_fit(boundary,typology,unit_mix={}){
  const analysis=analyse_boundary(boundary);
  if(!analysis.valid){
    return {
      typology,
      feasible:false,
      violations:[analysis.error],
    };
  }

  const unitMix=normalizeUnitMix(unit_mix);
  const minRequiredArea=minAreaForMix(unitMix);
  const efficiency=defaultEfficiency(typology);
  const effectiveArea=analysis.area_m2*efficiency;
  const minDepth=requiredDepthMetres(typology);
  const minSpan=analysis.dimensions.min_span_m;
  const aspect=analysis.dimensions.aspect_ratio;
  const violations=[];

  if(minSpan<minDepth){
    violations.push(`Minimum clear depth not met (${minSpan.toFixed(1)}m < ${minDepth.toFixed(1)}m).`);
  }
  if(typology==='point_block'&&aspect>=1.8){
    violations.push(`Point block aspect ratio too high (${aspect.toFixed(2)} >= 1.80).`);
  }
  if(effectiveArea<minRequiredArea){
    violations.push(`Insufficient GFA (${effectiveArea.toFixed(1)}m2 < ${minRequiredArea.toFixed(1)}m2).`);
  }
  if(analysis.perimeter_length_m<12){
    violations.push('Boundary external edge length is too short for unit frontage.');
  }

  const avgUnitArea=(unitMix.bed1*50+unitMix.bed2*70+unitMix.bed3*90)/Math.max(1,unitMix.bed1+unitMix.bed2+unitMix.bed3);
  const estimatedTotalUnits=Math.floor(effectiveArea/Math.max(avgUnitArea,1));

  return {
    typology,
    feasible:violations.length===0,
    estimated_efficiency:efficiency,
    estimated_gross_corridor_area_m2:Math.max(0,analysis.area_m2*(1-efficiency)),
    estimated_external_facade_per_unit_m:analysis.perimeter_length_m/Math.max(estimatedTotalUnits,1),
    estimated_unit_count_achievable:{
      bed1:Math.floor(effectiveArea/50),
      bed2:Math.floor(effectiveArea/70),
      bed3:Math.floor(effectiveArea/90),
      total:estimatedTotalUnits,
    },
    constraint_violations:violations,
    analysis,
  };
}

export function propose_typology_options(boundary,unit_mix={}){
  const candidates=['double_loaded_slab','single_loaded','point_block'];
  const reports=candidates.map((typology)=>evaluate_typology_fit(boundary,typology,unit_mix));
  const passing=reports.filter((r)=>r.feasible).slice(0,3);

  return {
    options:passing.map((r)=>{
      const axis=r.analysis?.orientation?.axis_hint==='x_major'?'horizontal':'vertical';
      return {
        typology:r.typology,
        corridor_axis:axis,
        estimated_efficiency:r.estimated_efficiency,
        core_count:r.typology==='point_block'?1:(r.analysis.dimensions.max_span_m>24?2:1),
        schematic:{
          unit_bay_dimensions_m:r.typology==='single_loaded'
            ? {depth:7.5,width_range:[4.5,6.5]}
            : {depth:8.0,width_range:[5.0,7.0]},
          typed_edge_distribution:r.typology==='single_loaded'
            ? {external:2,shared_wall:1,facing_corridor:1}
            : {external:1,shared_wall:2,facing_corridor:1},
        },
        fit_report:r,
      };
    }),
    reports,
  };
}

export function check_typology_viability(boundary,typology_option,unit_mix={}){
  const report=evaluate_typology_fit(boundary,typology_option?.typology||typology_option,unit_mix);
  return {
    pass:!!report.feasible,
    requirements:[
      {
        name:'Minimum clear depth for typology',
        expected:`>= ${requiredDepthMetres(report.typology).toFixed(1)}m`,
        actual:`${report.analysis?.dimensions?.min_span_m?.toFixed?.(1)??'n/a'}m`,
        status:(report.analysis?.dimensions?.min_span_m??0)>=requiredDepthMetres(report.typology)?'pass':'fail',
      },
      {
        name:'Sufficient GFA',
        expected:`>= ${minAreaForMix(unit_mix).toFixed(1)}m2`,
        actual:`${((report.analysis?.area_m2||0)*(report.estimated_efficiency||0)).toFixed(1)}m2`,
        status:report.constraint_violations?.some((v)=>v.includes('Insufficient GFA'))?'fail':'pass',
      },
      {
        name:'Unit mix achievable',
        expected:'requested mix at minimum sizes',
        actual:report.feasible?'achievable':'not achievable',
        status:report.feasible?'pass':'fail',
      },
    ],
    violations:report.constraint_violations||[],
    report,
  };
}
