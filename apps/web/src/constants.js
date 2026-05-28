function loadRoomTypesConfig(){
  const fallback={
    version:'1.0',
    types:[
      {id:0,name:'LivingRoom',abbr:'LR',group:'common'},
      {id:1,name:'MasterRoom',abbr:'MR',group:'bedroom'},
      {id:2,name:'Kitchen',abbr:'K',group:'kitchen'},
      {id:3,name:'Bathroom',abbr:'Ba',group:'bathroom'},
      {id:4,name:'DiningRoom',abbr:'DR',group:'common'},
      {id:5,name:'ChildRoom',abbr:'CR',group:'bedroom'},
      {id:6,name:'StudyRoom',abbr:'SR',group:'misc'},
      {id:7,name:'SecondRoom',abbr:'S2',group:'bedroom'},
      {id:8,name:'GuestRoom',abbr:'GR',group:'bedroom'},
      {id:9,name:'Balcony',abbr:'Bl',group:'balcony'},
      {id:10,name:'Entrance',abbr:'En',group:'misc'},
      {id:11,name:'Storage',abbr:'St',group:'misc'},
      {id:12,name:'WalkIn',abbr:'Wi',group:'misc'},
    ],
    groups:{
      bedroom:[1,5,7,8],
      bathroom:[3],
      kitchen:[2],
      common:[0,4],
      misc:[6,10,11,12],
      balcony:[9],
    },
    fit_order:['bedroom','bathroom','kitchen','misc','common'],
    common_area_ids:[0,4],
    balcony_id:9,
  };
  try{
    if(location.protocol==='file:')return fallback;
    const req=new XMLHttpRequest();
    req.open('GET','/room_types.json',false);
    req.send(null);
    if(req.status>=200&&req.status<300&&req.responseText){
      return JSON.parse(req.responseText);
    }
  }catch(_err){}
  return fallback;
}

export const ROOM_TYPES=loadRoomTypesConfig();
export const ROOM_TYPE_IDS_BY_NAME=Object.fromEntries((ROOM_TYPES.types||[]).map((entry)=>[entry.name,entry.id]));
export const N_SAMP=256;
export const N_FREQS=48;
export const SCALE=256;
export const SNAP_PX=14;
export const VR=5;
export const PAD=0.08;
export const SHAPE_DIM=48;
export const ENT_DIM=4;
export const ROOM_DIM=11;
export const TOTAL_DIM=63;
export const METRES_PER_UNIT=18;
export const ROOM_GROUPS=[
  {name:'bedroom', set:new Set([ROOM_TYPE_IDS_BY_NAME.MasterRoom,ROOM_TYPE_IDS_BY_NAME.ChildRoom,ROOM_TYPE_IDS_BY_NAME.SecondRoom]),max:5},
  {name:'living', set:new Set([ROOM_TYPE_IDS_BY_NAME.LivingRoom]),max:2},
  {name:'kitchen', set:new Set([ROOM_TYPE_IDS_BY_NAME.Kitchen]),max:2},
  {name:'bathroom', set:new Set([ROOM_TYPE_IDS_BY_NAME.Bathroom]),max:4},
  {name:'dining', set:new Set([ROOM_TYPE_IDS_BY_NAME.DiningRoom]),max:2},
  {name:'study', set:new Set([ROOM_TYPE_IDS_BY_NAME.StudyRoom]),max:2},
  {name:'guest', set:new Set([ROOM_TYPE_IDS_BY_NAME.GuestRoom]),max:2},
  {name:'balcony', set:new Set([ROOM_TYPE_IDS_BY_NAME.Balcony]),max:3},
  {name:'entrance', set:new Set([ROOM_TYPE_IDS_BY_NAME.Entrance]),max:2},
  {name:'storage', set:new Set([ROOM_TYPE_IDS_BY_NAME.Storage]),max:3},
];
export const ABBR={bedroom:'Bed',living:'Liv',kitchen:'Kit',bathroom:'Bath',dining:'Din',study:'Stu',guest:'Gst',balcony:'Bal',entrance:'Ent',storage:'Sto'};
export const RCOLS=['#E6B8A2','#B8D4E8','#B8E8B8','#E8E8B8','#D4B8E8','#E8B8D4','#B8E8E8','#E8D4B8','#C8E8C8','#B8C8E8','#E8C8B8','#D8D8D8'];
export const PLAN_CACHE='fp-plans-v1';
export const COMMON_AREA_TYPES=new Set((ROOM_TYPES.common_area_ids||[0,4]));
export const WORLD_SCALE=10;
export const ROOM_HEIGHT=2.8;
export const WINDOW_SILL_HEIGHT=1.0;
export const WINDOW_HEIGHT=1.2;
export const WINDOW_THICKNESS=0.08;
export const WINDOW_BALCONY_EDGE_TOL=0.12;
export const BALCONY_OPENING_DOOR_HEIGHT=2.2;
export const BALCONY_TYPE_ID=ROOM_TYPES.balcony_id??9;
export const BALCONY_FLOOR_THICKNESS=0.08;
export const BALCONY_PARAPET_HEIGHT=1.2;
export const BALCONY_PARAPET_THICKNESS=0.08;
export const DOOR_HEIGHT=2.2;
export const DOOR_THICKNESS=0.12;
export const OPACITY_MASK_COLORS={
  walls:0x787878,
  doors:0x08ff33,
  windows:0xe6e6e6,
  slabs:0x503232,
  parapets:0xffb806,
};
export const SILHOUETTE_CONSTRAINT_PROMPT='strict silhouette lock: keep exact building extents and massing from the silhouette, no added wings, annexes, balconies, canopies, or extra built elements.';