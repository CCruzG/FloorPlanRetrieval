import { SILHOUETTE_CONSTRAINT_PROMPT } from './constants.js';
import { DEFAULT_RENDER_PROMPT_CONFIG } from './state.js';

export function createRenderPipeline({
  renderCaptureState,
  renderRequestState,
  setViewer3DStatus,
  getCameraPerspectivePrompt,
  toIntInput,
}) {
  function getPromptConfig(){
    if(!renderRequestState.promptConfig){
      renderRequestState.promptConfig={
        ...DEFAULT_RENDER_PROMPT_CONFIG,
        context:DEFAULT_RENDER_PROMPT_CONFIG.context,
        materialOverrides:{...DEFAULT_RENDER_PROMPT_CONFIG.materialOverrides},
      };
    }
    const promptConfig=renderRequestState.promptConfig;
    if(typeof promptConfig.context!=='string'||!promptConfig.context.trim()){
      promptConfig.context=promptConfig.siteContext||'';
    }
    if(typeof promptConfig.siteContext!=='string'||!promptConfig.siteContext.trim()){
      promptConfig.siteContext=promptConfig.context||'';
    }
    return promptConfig;
  }

  function getDefaultWorkflowName(){
    return renderRequestState.workflowDefaultName||'comfyui-sdxl-depth-multi-workflow.json';
  }

  function syncRenderStateFromDom(){
    const promptConfig=getPromptConfig();
    promptConfig.style=(document.getElementById('render-style')?.value||promptConfig.style||DEFAULT_RENDER_PROMPT_CONFIG.style).trim()||DEFAULT_RENDER_PROMPT_CONFIG.style;
    promptConfig.view=(document.getElementById('render-view')?.value||promptConfig.view||DEFAULT_RENDER_PROMPT_CONFIG.view).trim()||DEFAULT_RENDER_PROMPT_CONFIG.view;
    promptConfig.lighting=(document.getElementById('render-light')?.value||promptConfig.lighting||DEFAULT_RENDER_PROMPT_CONFIG.lighting).trim()||DEFAULT_RENDER_PROMPT_CONFIG.lighting;
    promptConfig.material=(document.getElementById('render-material')?.value||promptConfig.material||DEFAULT_RENDER_PROMPT_CONFIG.material).trim()||DEFAULT_RENDER_PROMPT_CONFIG.material;
    const contextPreset=(document.getElementById('render-context')?.value||DEFAULT_RENDER_PROMPT_CONFIG.context||'rural').trim()||'rural';
    const contextNotes=(document.getElementById('render-prompt')?.value||'').trim();
    promptConfig.context=contextPreset;
    promptConfig.siteContext=contextNotes;
    promptConfig.materialOverrides={
      walls:(document.getElementById('opacity-mat-walls')?.value||promptConfig.materialOverrides?.walls||'').trim(),
      doors:(document.getElementById('opacity-mat-doors')?.value||promptConfig.materialOverrides?.doors||'').trim(),
      windows:(document.getElementById('opacity-mat-windows')?.value||promptConfig.materialOverrides?.windows||'').trim(),
      slabs:(document.getElementById('opacity-mat-slabs')?.value||promptConfig.materialOverrides?.slabs||'').trim(),
      parapets:(document.getElementById('opacity-mat-parapets')?.value||promptConfig.materialOverrides?.parapets||'').trim(),
    };
    promptConfig.openaiModel=(document.getElementById('render-openai-model')?.value||promptConfig.openaiModel||DEFAULT_RENDER_PROMPT_CONFIG.openaiModel).trim()||DEFAULT_RENDER_PROMPT_CONFIG.openaiModel;
    promptConfig.openaiQuality=(document.getElementById('render-openai-quality')?.value||promptConfig.openaiQuality||DEFAULT_RENDER_PROMPT_CONFIG.openaiQuality).trim()||DEFAULT_RENDER_PROMPT_CONFIG.openaiQuality;
    promptConfig.comfyUrl=(document.getElementById('render-comfy-url')?.value||promptConfig.comfyUrl||DEFAULT_RENDER_PROMPT_CONFIG.comfyUrl).trim()||DEFAULT_RENDER_PROMPT_CONFIG.comfyUrl;
    promptConfig.depthInvert=!!document.getElementById('render-comfy-invert')?.checked;
    return promptConfig;
  }

  function updateRenderReadiness(){
    const btn=document.getElementById('btn-generate-render');
    if(!btn)return;
    const backend=getRenderBackend();
    const hasCommonCapture=!!(renderCaptureState.depthDataUrl&&renderCaptureState.previewDataUrl);
    const hasComfyCapture=!!(
      renderCaptureState.depthDataUrl&&
      renderCaptureState.previewDataUrl&&
      renderCaptureState.normalDataUrl&&
      renderCaptureState.opacityMaskDataUrl&&
      (renderCaptureState.silhouetteDataUrl||renderCaptureState.maskDataUrl)
    );
    const hasCapture=backend==='comfy' ? hasComfyCapture : hasCommonCapture;
    btn.disabled=!hasCapture||!!renderRequestState.isRendering;
  }

  function updateCapturePanel(){
    const panel=document.getElementById('capture-panel');
    const activeImg=document.getElementById('capture-active-img');
    const activeLabel=document.getElementById('capture-active-label');
    const captureSelect=document.getElementById('capture-view-select');
    const opacityLegend=document.getElementById('opacity-legend');
    if(!panel||!activeImg||!activeLabel||!captureSelect||!opacityLegend)return;
    if(renderCaptureState.previewDataUrl&&renderCaptureState.depthDataUrl){
      const view=renderCaptureState.activeView||'preview';
      const mapByView={
        preview:{src:renderCaptureState.previewDataUrl,label:'Preview'},
        depth:{src:renderCaptureState.depthDataUrl,label:'Depth'},
        normal:{src:renderCaptureState.normalDataUrl,label:'Normal'},
        silhouette:{src:renderCaptureState.silhouetteDataUrl||renderCaptureState.maskDataUrl,label:'Silhouette'},
        opacity:{src:renderCaptureState.opacityMaskDataUrl,label:'Opacity Mask'},
      };
      const selected=mapByView[view]||mapByView.preview;
      const fallback=selected.src?selected:mapByView.preview;
      activeImg.src=fallback.src||renderCaptureState.previewDataUrl;
      activeLabel.textContent=fallback.label;
      captureSelect.value=view;
      opacityLegend.classList.toggle('hidden',view!=='opacity');
      panel.classList.add('show');
    }else{
      opacityLegend.classList.add('hidden');
      panel.classList.remove('show');
    }
    updateRenderReadiness();
  }

  function setCaptureView(view){
    const allowed=new Set(['preview','depth','normal','silhouette','opacity']);
    renderCaptureState.activeView=allowed.has(view)?view:'preview';
    updateCapturePanel();
    refreshStructuredPromptPreview();
  }

  function togglePanelCollapse(panelId,bodyId,btnId){
    const panel=document.getElementById(panelId);
    const body=document.getElementById(bodyId);
    const btn=document.getElementById(btnId);
    if(!panel||!body||!btn)return;
    const collapsed=panel.classList.toggle('collapsed');
    body.style.display=collapsed?'none':'';
    btn.textContent=collapsed?'+':'-';
  }

  function getRenderBackend(){
    const selected=document.querySelector('input[name="render-backend"]:checked');
    if(selected&&selected.value)return selected.value;
    return renderRequestState.backend||'comfy';
  }

  function updateBackendDependentUI(){
    const backend=getRenderBackend();
    const openaiRowsVisible=backend==='openai';
    const comfyRowsVisible=backend==='comfy';

    ['openai-row-api-key','openai-row-model','openai-row-quality'].forEach((id)=>{
      const el=document.getElementById(id);
      if(el)el.style.display=openaiRowsVisible?'flex':'none';
    });

    const comfyRow=document.getElementById('comfy-row');
    if(comfyRow)comfyRow.style.display=comfyRowsVisible?'flex':'none';

    const resolutionRow=document.getElementById('python-row-resolution');
    if(resolutionRow)resolutionRow.style.display='flex';

    updateRenderReadiness();
  }

  function getRenderImageSize(){
    const width=toIntInput('py-width',1024,256,2048);
    const height=toIntInput('py-height',1024,256,2048);
    return {width,height};
  }

  function getOpenAIRenderParams(){
    const promptConfig=syncRenderStateFromDom();
    const model=promptConfig.openaiModel;
    const quality=promptConfig.openaiQuality;
    const {width,height}=getRenderImageSize();
    return {model,quality,width,height};
  }

  // function composeRenderPrompt(){
  //   const promptConfig=getPromptConfig();
  //   const style=promptConfig.style;
  //   const view=promptConfig.view;
  //   const lighting=promptConfig.lighting;
  //   const material=promptConfig.material;
  //   const siteContext=promptConfig.siteContext;

  //   const lightingByPreset={
  //     'golden hour':'golden hour sunlight, long soft shadows, warm sky bounce',
  //     overcast:'overcast daylight, diffuse skylight, soft shadow edges',
  //     night:'night exterior, practical architectural lighting, balanced warm-cool contrast',
  //   };
  //   const materialMapByPreset={
  //     timber:{
  //       walls:'timber',
  //       doors:'timber - timber frame',
  //       windows:'glass, metal frame',
  //       slabs:'timber',
  //       parapets:'timber',
  //     },
  //     concrete:{
  //       walls:'board-formed concrete',
  //       doors:'concrete surround, timber frame',
  //       windows:'glass, dark metal frame',
  //       slabs:'concrete',
  //       parapets:'concrete',
  //     },
  //     brick:{
  //       walls:'textured brick',
  //       doors:'brick reveal, timber frame',
  //       windows:'glass, dark metal frame',
  //       slabs:'concrete',
  //       parapets:'brick',
  //     },
  //     glass:{
  //       walls:'glass facade, slim metal frame',
  //       doors:'glass door, metal frame',
  //       windows:'glass, slim metal frame',
  //       slabs:'concrete',
  //       parapets:'glass and metal',
  //     },
  //   };

  //   const siteText=(siteContext||'urban residential context').replace(/\s+/g,' ').trim();
  //   const lightingText=lightingByPreset[lighting]||`${lighting} architectural lighting`;
  //   const perspectiveText=getCameraPerspectivePrompt(view);

  //   const baseMaterialMap=materialMapByPreset[material]||{
  //     walls:material,
  //     doors:`${material} door, ${material} frame`,
  //     windows:'glass, metal frame',
  //     slabs:material,
  //     parapets:material,
  //   };

  //   const materialMap={...baseMaterialMap};
  //   Object.entries(promptConfig.materialOverrides||{}).forEach(([label,value])=>{
  //     if(value)materialMap[label]=value;
  //   });

  //   const materialSection=`materials by element: walls: ${materialMap.walls}; doors: ${materialMap.doors}; windows: ${materialMap.windows}; slabs: ${materialMap.slabs}; parapets: ${materialMap.parapets}`;

  //   const clipCore=[
  //     style==='photorealistic'?'photorealistic architectural render':`${style} architectural render`,
  //     'professional architectural photography',
  //     '8k',
  //     'sharp focus',
  //     'realistic scale',
  //     'modern residential building',
  //     perspectiveText,
  //     siteText,
  //     lightingText,
  //   ];

  //   const parts=[clipCore.join(', '),materialSection];

  //   if(renderCaptureState.silhouetteDataUrl||renderCaptureState.maskDataUrl){
  //     parts.push(SILHOUETTE_CONSTRAINT_PROMPT);
  //   }

  //   return parts.join('\n\n');
  // }

  function composeCLIPPrompt() {
  const promptConfig = getPromptConfig();
  const { style, view, lighting, material, context, siteContext, materialOverrides } = promptConfig;

  // ── Context ──────────────────────────────────────────────────────────────
  const CONTEXT_DESCRIPTIONS = {
    rural:     'open countryside, grass ground plane, trees in background',
    urban:     'urban street, paved footpath, neighbouring buildings',
    suburban:  'suburban residential street, landscaped garden',
    coastal:   'coastal setting, sandy dunes, ocean in background',
    forest:    'forest edge, dense tree canopy, dappled light',
    hillside:  'sloped hillside site, valley views beyond',
    rooftop:   'rooftop setting, city skyline in background',
    desert:    'arid landscape, rocky terrain, sparse vegetation',
  };

  // ── Lighting ─────────────────────────────────────────────────────────────
  const LIGHTING_DESCRIPTIONS = {
    'golden hour': 'golden hour sunlight, long warm shadows, amber sky bounce',
    'overcast':    'overcast sky, soft diffuse daylight, no hard shadows',
    'night':       'night exterior, warm interior light spill, cool-blue ambient',
    'midday':      'bright midday sun, high-contrast shadows, clear blue sky',
    'dusk':        'dusk light, deep blue-purple sky, lights beginning to glow',
  };

  // ── View ─────────────────────────────────────────────────────────────────
  const VIEW_DESCRIPTIONS = {
    exterior: 'eye-level exterior view, two-point perspective',
    aerial:   'elevated aerial view, three-point perspective',
    street:   'street-level pedestrian view',
    garden:   'garden view, low angle looking toward building',
  };

  // ── Material presets (base values per element) ───────────────────────────
  const MATERIAL_PRESETS = {
    timber: {
      walls:    'vertical timber board cladding',
      doors:    'timber door, timber frame',
      windows:  'glass, dark metal frame',
      slabs:    'exposed concrete soffit',
      parapets: 'timber-clad parapet',
    },
    concrete: {
      walls:    'board-formed concrete',
      doors:    'concrete reveal, solid timber door',
      windows:  'glass, dark metal frame',
      slabs:    'exposed concrete fascia',
      parapets: 'concrete parapet wall',
    },
    brick: {
      walls:    'textured brick',
      doors:    'brick reveal, timber door',
      windows:  'glass, dark metal frame',
      slabs:    'concrete soffit',
      parapets: 'brick parapet',
    },
    glass: {
      walls:    'full-height glass facade, slim mullions',
      doors:    'glass pivot door, metal frame',
      windows:  'floor-to-ceiling glass, slim metal frame',
      slabs:    'concrete fascia',
      parapets: 'structural glass balustrade',
    },
  };

  // Merge base preset with any per-element overrides from the UI
  const base = MATERIAL_PRESETS[material] || {
    walls:    material,
    doors:    `${material} door`,
    windows:  'glass, metal frame',
    slabs:    material,
    parapets: `${material} parapet`,
  };
  const mat = { ...base, ...(materialOverrides || {}) };

  // Build a single readable material phrase
  // Each element maps to a natural description; filter empties
  const materialPhrase = [
    mat.walls    && `${mat.walls} on exterior walls`,
    mat.doors    && mat.doors,
    mat.windows  && `${mat.windows} windows`,
    mat.slabs    && `${mat.slabs}`,
    mat.parapets && mat.parapets,
  ].filter(Boolean).join(', ');

  // ── Assemble prompt ───────────────────────────────────────────────────────
  const styleText = style === 'photorealistic'
    ? 'photorealistic architectural render, professional photography, 8k, sharp focus, realistic scale'
    : `${style} architectural render`;

  const viewText     = VIEW_DESCRIPTIONS[view]     || `${view} view`;
  const lightingText = LIGHTING_DESCRIPTIONS[lighting] || `${lighting} lighting`;

  // Context accepts a string or an array of keys
  const contextValue = context || siteContext || 'rural';
  const contextKeys = Array.isArray(contextValue) ? contextValue : [contextValue];
  const contextText = contextKeys
    .map(k => CONTEXT_DESCRIPTIONS[k] || k)
    .filter(Boolean)
    .join(', ');
  const siteNotes = typeof siteContext === 'string' ? siteContext.trim() : '';

  const coreParts = [
    styleText,
    `${viewText} of a modern residential building`,
    contextText,
    siteNotes,
    lightingText,
    materialPhrase,
  ];

  if (renderCaptureState.silhouetteDataUrl || renderCaptureState.maskDataUrl) {
    coreParts.push(
      'exact building outline preserved, no added wings balconies or extensions'
    );
  }

  return coreParts.filter(Boolean).join(', ');
}

  function composeOpenAIPrompt(){
    const promptConfig=getPromptConfig();
    const {style,view,lighting,material,context,siteContext,materialOverrides}=promptConfig;

    const MATERIAL_PRESETS={
      timber:{walls:'vertical timber board cladding',doors:'timber door with timber frame',windows:'glass with dark metal frames',slabs:'exposed concrete soffit',parapets:'timber-clad parapet'},
      concrete:{walls:'board-formed concrete',doors:'solid timber door with concrete reveal',windows:'glass with dark metal frames',slabs:'exposed concrete fascia',parapets:'concrete parapet'},
      brick:{walls:'textured brick',doors:'timber door with brick reveal',windows:'glass with dark metal frames',slabs:'concrete soffit',parapets:'brick parapet'},
      glass:{walls:'full-height glass facade with slim mullions',doors:'glass pivot door with metal frame',windows:'floor-to-ceiling glass with slim metal frames',slabs:'concrete fascia',parapets:'structural glass balustrade'},
    };
    const base=MATERIAL_PRESETS[material]||{walls:material,doors:`${material} door`,windows:'glass with metal frames',slabs:material,parapets:`${material} parapet`};
    const mat={...base,...(materialOverrides||{})};

    const VIEW_SENTENCES={
      exterior:'photographed at eye level from the street with a two-point perspective',
      aerial:'photographed from an elevated aerial vantage with a three-point perspective',
      street:'photographed at street level from a pedestrian viewpoint',
      garden:'photographed from the garden, looking toward the building at a low angle',
    };
    const LIGHTING_SENTENCES={
      'golden hour':'bathed in golden hour sunlight, with long warm shadows and an amber sky bounce',
      'overcast':'lit by soft overcast daylight with no hard shadows and an even sky diffusion',
      'night':'shot at night, with warm interior light spilling through windows against a cool blue ambient sky',
      'midday':'in bright midday sun with crisp high-contrast shadows and a clear blue sky',
      'dusk':'at dusk, with a deep blue-purple sky and the building lights beginning to glow',
    };
    const CONTEXT_SENTENCES={
      rural:'set in open countryside with a grass ground plane and trees in the background',
      urban:'situated on an urban street with paved footpaths and neighbouring buildings visible',
      suburban:'on a suburban residential street with a landscaped front garden',
      coastal:'in a coastal setting with sandy dunes and ocean visible in the background',
      forest:'at the forest edge with dense tree canopy overhead and dappled light',
      hillside:'on a sloped hillside site with valley views beyond',
      rooftop:'on a rooftop with the city skyline visible in the background',
      desert:'in an arid landscape with rocky terrain and sparse desert vegetation',
    };

    const styleOpener=style==='photorealistic'
      ?'A photorealistic exterior architectural photograph'
      :`An architectural ${style} render`;
    const viewSentence=VIEW_SENTENCES[view]||`${view} view`;
    const lightingSentence=LIGHTING_SENTENCES[lighting]||`${lighting} lighting`;

    const contextValue=context||'rural';
    const contextKeys=Array.isArray(contextValue)?contextValue:[contextValue];
    const contextSentence=contextKeys.map(k=>CONTEXT_SENTENCES[k]||k).filter(Boolean).join(', ');

    const ELEMENT_LABELS={walls:'walls',windows:'windows',doors:'doors',slabs:'soffits',parapets:'parapets'};
    const materialParts=Object.entries(ELEMENT_LABELS)
      .map(([key,label])=>{
        const val=(mat[key]||'').trim();
        if(!val)return null;
        // avoid "timber walls walls" if the value already names the element
        const alreadyLabelled=val.toLowerCase().includes(label.toLowerCase());
        return alreadyLabelled?val:`${val} ${label}`;
      })
      .filter(Boolean);
    const materialSentence=`Exterior materials: ${materialParts.join(', ')}.`;

    const notes=typeof siteContext==='string'?siteContext.trim():'';

    const silhouetteNote=(renderCaptureState.silhouetteDataUrl||renderCaptureState.maskDataUrl)
      ?'Preserve the exact building silhouette and massing — do not add or remove wings, balconies, or extensions.'
      :'';

    return [
      `${styleOpener} of a house, ${viewSentence}.`,
      `The house is ${contextSentence}, ${lightingSentence}.`,
      materialSentence,
      notes,
      silhouetteNote,
    ].filter(Boolean).join(' ');
  }

  function composeRenderPrompt(){
    return getRenderBackend()==='openai'?composeOpenAIPrompt():composeCLIPPrompt();
  }

  function refreshStructuredPromptPreview(){
    syncRenderStateFromDom();
    const el=document.getElementById('render-prompt-preview');
    if(!el)return;
    const backend=getRenderBackend();
    const labelEl=document.getElementById('render-prompt-preview-label');
    if(labelEl)labelEl.textContent=backend==='openai'?'Prompt preview (OpenAI)':'Prompt preview (CLIP)';
    el.value=composeRenderPrompt();
    updateRenderReadiness();
  }

  function updateRenderResultUI(dataUrl,meta){
    const box=document.getElementById('render-result');
    const img=document.getElementById('render-result-img');
    const metaEl=document.getElementById('render-result-meta');
    if(!box||!img||!metaEl)return;
    img.src=dataUrl;
    metaEl.textContent=meta;
    box.classList.add('show');
  }

  async function parseApiResponse(resp){
    const text=await resp.text();
    let data=null;
    try{data=text?JSON.parse(text):{};}catch{data=null;}
    return {ok:resp.ok,status:resp.status,data,text};
  }

  async function postJsonWithTimeout(url,body,timeoutMs){
    const controller=new AbortController();
    const timeoutId=setTimeout(()=>controller.abort(new Error(`request timed out after ${Math.round(timeoutMs/1000)}s`)),timeoutMs);
    try{
      return await fetch(url,{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify(body),
        signal:controller.signal,
      });
    }catch(err){
      if(err.name==='AbortError')throw new Error(`request timed out after ${Math.round(timeoutMs/1000)}s`);
      throw err;
    }finally{
      clearTimeout(timeoutId);
    }
  }

  function setComfyStatus(msg,isError=false){
    const el=document.getElementById('comfy-status');
    if(!el)return;
    el.textContent=msg;
    el.style.color=isError?'#b91c1c':'#4b5563';
  }

  function getWorkflowNodeTypes(parsed){
    if(!parsed||typeof parsed!=='object')return [];
    let nodes=parsed;
    if(parsed.prompt&&typeof parsed.prompt==='object'&&!Array.isArray(parsed.prompt)){
      nodes=parsed.prompt;
    }else if(parsed.workflow&&typeof parsed.workflow==='object'&&!Array.isArray(parsed.workflow)){
      nodes=parsed.workflow;
    }
    if(!nodes||typeof nodes!=='object'||Array.isArray(nodes))return [];
    return Object.values(nodes)
      .map((node)=>String((node&&((node.class_type)||(node.type)))||'').trim().toLowerCase())
      .filter(Boolean);
  }

  function workflowLooksPassThrough(nodeTypes){
    const hasSampler=nodeTypes.some((t)=>t.includes('ksampler')||t.includes('sampler'));
    const hasModelish=nodeTypes.some((t)=>
      t.includes('checkpointloader')||
      t.includes('unetloader')||
      t.includes('cliploader')||
      t.includes('dualcliploader')||
      t.includes('diffusion')||
      t.includes('model')
    );
    const hasDecode=nodeTypes.some((t)=>t.includes('decode')||t.includes('vaedecode'));
    return !hasSampler&&!(hasModelish&&hasDecode);
  }

  async function checkComfyConnection(){
    const comfyUrl=syncRenderStateFromDom().comfyUrl;
    const btn=document.getElementById('btn-check-comfy');
    if(btn)btn.disabled=true;
    setComfyStatus('Checking connection...');
    try{
      const resp=await fetch('/comfy_health',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({comfy_url:comfyUrl}),
      });
      const parsed=await parseApiResponse(resp);
      const data=parsed.data;
      if(!parsed.ok){
        const msg=(data&&data.error)?data.error:(parsed.text||`server ${parsed.status}`);
        throw new Error(msg);
      }
      if(!data||data.error)throw new Error((data&&data.error)||'invalid server response');
      setComfyStatus(`Connected: ${data.summary||'ComfyUI reachable.'}`);
    }catch(err){
      setComfyStatus(`Connection failed: ${err.message}`,true);
    }finally{
      if(btn)btn.disabled=false;
    }
  }

  function loadWorkflowOverride(input){
    const file=input?.files?.[0];
    if(!file)return;
    const reader=new FileReader();
    reader.onload=()=>{
      try{
        const parsed=JSON.parse(reader.result);
        const looksLikeUIExport=!!(parsed&&Array.isArray(parsed.nodes));
        if(looksLikeUIExport){
          renderRequestState.workflowOverride=null;
          renderRequestState.workflowName='';
          const el=document.getElementById('workflow-name');
          if(el)el.textContent='UI workflow detected. Export from ComfyUI as API  format JSON.';
          const inEl=document.getElementById('workflow-file');
          if(inEl)inEl.value='';
          setComfyStatus('Workflow rejected: use ComfyUI Save (API Format).',true);
          setViewer3DStatus('Workflow must be API format (dict keyed by node id).');
          return;
        }
        const nodeTypes=getWorkflowNodeTypes(parsed);
        if(workflowLooksPassThrough(nodeTypes)){
          renderRequestState.workflowOverride=null;
          renderRequestState.workflowName='';
          const el=document.getElementById('workflow-name');
          if(el)el.textContent='Workflow rejected: appears to be depth passthrough (no sampler/model graph).';
          const inEl=document.getElementById('workflow-file');
          if(inEl)inEl.value='';
          const preview=nodeTypes.length?nodeTypes.slice(0,8).join(', '):'none';
          setComfyStatus(`Workflow rejected: non-generative graph detected (${preview}).`,true);
          setViewer3DStatus('Upload a full ComfyUI API workflow with sampler + model nodes.');
          return;
        }
        renderRequestState.workflowOverride=parsed;
        renderRequestState.workflowName=file.name;
        const el=document.getElementById('workflow-name');
        if(el)el.textContent=`Using uploaded workflow: ${file.name}`;
        setComfyStatus('Workflow loaded (API format).');
        setViewer3DStatus('Workflow override loaded.');
      }catch(err){
        renderRequestState.workflowOverride=null;
        renderRequestState.workflowName='';
        const el=document.getElementById('workflow-name');
        if(el)el.textContent=`Invalid workflow JSON. Using repo default: ${getDefaultWorkflowName()}`;
        setComfyStatus('Invalid workflow JSON file.',true);
        setViewer3DStatus(`Workflow parse error: ${err.message}`);
      }
    };
    reader.onerror=()=>{
      renderRequestState.workflowOverride=null;
      renderRequestState.workflowName='';
      const el=document.getElementById('workflow-name');
      if(el)el.textContent='Workflow read failed. Using repo default.';
      setComfyStatus('Workflow file could not be read.',true);
      setViewer3DStatus('Workflow read failed.');
    };
    reader.readAsText(file);
  }

  function clearWorkflowOverride(){
    renderRequestState.workflowOverride=null;
    renderRequestState.workflowName='';
    const inEl=document.getElementById('workflow-file');
    if(inEl)inEl.value='';
    const nameEl=document.getElementById('workflow-name');
    if(nameEl)nameEl.textContent=`Using repo default: ${getDefaultWorkflowName()}`;
    setComfyStatus('Using repo default workflow.');
    setViewer3DStatus('Workflow override cleared.');
  }

  async function generateRenderMock(){
    const btn=document.getElementById('btn-generate-render');
    if(!renderCaptureState.depthDataUrl||!renderCaptureState.previewDataUrl){
      setViewer3DStatus('Capture depth first before generating render.');
      return;
    }

    const promptConfig=syncRenderStateFromDom();
    const backend=getRenderBackend();
    const prompt=composeRenderPrompt();
    const {width,height}=getRenderImageSize();
    const openaiParams=getOpenAIRenderParams();
    const comfyUrl=promptConfig.comfyUrl;
    const depthInvert=promptConfig.depthInvert;
    renderRequestState.backend=backend;
    renderRequestState.assembledPrompt=prompt;
    renderRequestState.isRendering=true;
    updateRenderReadiness();

    const backendLabel=backend==='comfy'?'ComfyUI local':'OpenAI cloud';
    setViewer3DStatus(`Generating render (${backendLabel})…`);

    try{
      const requestBody={
        backend,
        prompt,
        width,
        height,
        depth_data_url:renderCaptureState.depthDataUrl,
        building_mask_data_url:renderCaptureState.silhouetteDataUrl||renderCaptureState.maskDataUrl,
        silhouette_data_url:renderCaptureState.silhouetteDataUrl||renderCaptureState.maskDataUrl,
        normal_data_url:renderCaptureState.normalDataUrl,
        opacity_mask_data_url:renderCaptureState.opacityMaskDataUrl,
        segmentation_data_url:renderCaptureState.opacityMaskDataUrl,
      };

      if(backend==='comfy'){
        requestBody.comfy_url=comfyUrl;
        requestBody.depth_invert=depthInvert;
        if(renderRequestState.workflowOverride){
          requestBody.workflow_override=renderRequestState.workflowOverride;
        }
      }else{
        requestBody.openai_model=openaiParams.model;
        requestBody.openai_quality=openaiParams.quality;
        requestBody.openai_width=openaiParams.width;
        requestBody.openai_height=openaiParams.height;
        requestBody.preview_data_url=renderCaptureState.previewDataUrl;
        const keyEl=document.getElementById('render-openai-key');
        const keyVal=(keyEl?.value||'').trim();
        if(keyVal)requestBody.openai_api_key=keyVal;
      }

      const resp=await postJsonWithTimeout('/render',requestBody,200000);
      const parsed=await parseApiResponse(resp);
      const data=parsed.data;
      if(!parsed.ok){
        const msg=(data&&data.error)?data.error:(parsed.text||`server ${parsed.status}`);
        throw new Error(msg);
      }
      if(!data||data.error)throw new Error((data&&data.error)||'invalid server response');

      const resultDataUrl=data.image_data_url;
      const metaObj=data.meta||{};
      const meta=`backend: ${backendLabel} | prompt: ${prompt} | id: ${metaObj.prompt_id||'n/a'}`;
      renderRequestState.lastResultDataUrl=resultDataUrl;
      renderRequestState.lastResultMeta=meta;
      updateRenderResultUI(resultDataUrl,meta);
      setViewer3DStatus('Render generated successfully.');
    }catch(err){
      setViewer3DStatus(`Render failed: ${err.message}`);
    }finally{
      renderRequestState.isRendering=false;
      updateRenderReadiness();
    }
  }

  function downloadDataUrlFile(dataUrl,fileName){
    if(!dataUrl||typeof dataUrl!=='string')return;
    const a=document.createElement('a');
    a.href=dataUrl;
    a.download=fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  function downloadTextFile(text,fileName){
    const blob=new Blob([String(text||'')],{type:'text/plain;charset=utf-8'});
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=url;
    a.download=fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function downloadLatestCaptureBundle(){
    if(!renderCaptureState.depthDataUrl){
      setViewer3DStatus('Capture maps first, then download bundle.');
      return;
    }

    const stamp=(renderCaptureState.capturedAt||new Date().toISOString())
      .replace(/[:]/g,'-')
      .replace(/\..*/,'');
    const base=`capture_${stamp}`;

    syncRenderStateFromDom();
    const prompt=composeRenderPrompt();
    const backend=getRenderBackend();
    const comfyUrl=getPromptConfig().comfyUrl;
    const bundleInfo={
      captured_at:renderCaptureState.capturedAt,
      backend,
      comfy_url:comfyUrl,
      prompt,
      files:{
        preview:'preview.png',
        depth:'depth.png',
        silhouette:'silhouette.png',
        building_mask:'building_mask.png',
        normal:'normal.png',
        opacity_mask:'opacity_mask.png',
      },
    };

    downloadDataUrlFile(renderCaptureState.previewDataUrl,`${base}__preview.png`);
    downloadDataUrlFile(renderCaptureState.depthDataUrl,`${base}__depth.png`);
    downloadDataUrlFile(renderCaptureState.silhouetteDataUrl||renderCaptureState.maskDataUrl,`${base}__silhouette.png`);
    downloadDataUrlFile(renderCaptureState.maskDataUrl,`${base}__building_mask.png`);
    downloadDataUrlFile(renderCaptureState.normalDataUrl,`${base}__normal.png`);
    downloadDataUrlFile(renderCaptureState.opacityMaskDataUrl,`${base}__opacity_mask.png`);
    downloadTextFile(prompt,`${base}__prompt.txt`);
    downloadTextFile(JSON.stringify(bundleInfo,null,2),`${base}__bundle.json`);

    setViewer3DStatus('Capture bundle downloaded.');
  }

  return {
    checkComfyConnection,
    clearWorkflowOverride,
    composeRenderPrompt,
    downloadLatestCaptureBundle,
    generateRenderMock,
    loadWorkflowOverride,
    getRenderImageSize,
    refreshStructuredPromptPreview,
    updateRenderReadiness,
    setCaptureView,
    togglePanelCollapse,
    updateBackendDependentUI,
    updateCapturePanel,
  };
}