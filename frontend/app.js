const $ = id => document.getElementById(id);
const taskLabels = {bethesda:'Bethesda cytology categories',sipakmed:'SIPaKMeD cell morphology',herlev:'Herlev cytology grades',binary:'Custom normal / abnormal'};
const descriptions = {bethesda:'ASCUS, HSIL, LSIL, NILM and SCC image categories.',sipakmed:'Five cell morphology classes; not equivalent to cancer stages.',herlev:'Seven original Herlev labels, preserved independently.',binary:'Normal / abnormal labels from the custom dataset.'};
let latestResult, previewURL, ready=false, taskEvaluation={};
function clearResult(){latestResult=undefined;$('result').hidden=true;$('empty').hidden=false;}
function updateTaskHelp(){
  const task=$('task').value, evidence=taskEvaluation[task];
  $('task-help').textContent=(descriptions[task]||'')+(evidence?` Held-out balanced accuracy: ${(evidence.balanced_accuracy*100).toFixed(1)}% (${evidence.test_count} images). ${evidence.balanced_accuracy<.7?'Limited classification performance. ':''}Patient independence is unverified.`:'');
}
async function initialize(){
  try{
    const response=await fetch('/api/health'); if(!response.ok)throw Error('Service unavailable');
    const health=await response.json(); ready=health.ready;taskEvaluation=health.evaluation||{};
    $('connection').textContent=ready?'● Models available':'○ Models not trained';
    for(const task of health.tasks){const option=document.createElement('option');option.value=task;option.textContent=taskLabels[task];$('task').append(option);}
    $('task').disabled=!ready;$('submit').disabled=!ready;
    if(health.tasks.includes('sipakmed'))$('task').value='sipakmed';
    if(ready)updateTaskHelp();else $('task-help').textContent='Train both model checkpoints to enable analysis.';
    if(health.models.some(m=>m.status==='pilot'))$('connection').textContent='● Pilot models · limited training subset';
    const metricsResponse=await fetch('/api/metrics'); if(!metricsResponse.ok)throw Error('Metrics unavailable');
    const metrics=await metricsResponse.json();
    $('metrics').replaceChildren();
    for(const [name,report] of Object.entries(metrics)){
      const title=document.createElement('p');title.textContent=`${report.metadata.architecture} · ${report.metadata.status} · ${report.metadata.architecture==='equal_weight_ensemble'?'equal-weight probability fusion':'frozen ImageNet features'}`;$('metrics').append(title);
      const table=document.createElement('table');const header=document.createElement('tr');
      for(const text of ['Task','Test images','Balanced accuracy','Macro F1']){const cell=document.createElement('th');cell.textContent=text;header.append(cell);}table.append(header);
      for(const [task,m] of Object.entries(report.metrics)){const row=document.createElement('tr');for(const text of [taskLabels[task],m.test_count,(m.balanced_accuracy*100).toFixed(1)+'%',m.classification_report['macro avg']['f1-score'].toFixed(3)]){const cell=document.createElement('td');cell.textContent=text;row.append(cell);}table.append(row);}$('metrics').append(table);
    }
    if(!Object.keys(metrics).length)$('metrics').textContent='Evaluation reports appear after training.';
  }catch(error){$('connection').textContent='○ Service unavailable';$('error').textContent=error.message;}
}
$('task').addEventListener('change',()=>{clearResult();updateTaskHelp();});
$('image').addEventListener('change',()=>{clearResult();const file=$('image').files[0];$('error').textContent='';$('preview').hidden=true;$('filename').textContent='';if(previewURL){URL.revokeObjectURL(previewURL);previewURL=undefined;}if(!file)return;if(file.size>10*1024*1024){$('error').textContent='Choose an image under 10 MB.';$('image').value='';return;}previewURL=URL.createObjectURL(file);$('preview').src=previewURL;$('preview').hidden=false;$('filename').textContent=file.name;});
$('analysis-form').addEventListener('submit',async event=>{
  event.preventDefault();clearResult();$('error').textContent='';const file=$('image').files[0];if(!file||!ready)return;
  const form=new FormData();form.append('file',file);form.append('task',$('task').value);$('submit').disabled=true;$('task').disabled=true;$('image').disabled=true;$('submit').textContent='Analyzing…';
  try{const response=await fetch('/api/predict',{method:'POST',body:form});const result=await response.json();if(!response.ok)throw Error(result.detail||'Analysis failed');latestResult=result;
    $('empty').hidden=true;$('result').hidden=false;$('label').textContent=result.label;$('score').textContent=`${(result.model_score*100).toFixed(1)}% model score · uncalibrated`;
    $('review').textContent=result.review_suggested?'Review suggested: low score or model disagreement.':'Research result: independent expert review required.';
    $('bars').replaceChildren();for(const [label,score] of Object.entries(result.probabilities)){const row=document.createElement('div');row.className='bar-row';const title=document.createElement('span');title.textContent=label;const value=document.createElement('span');value.textContent=(score*100).toFixed(1)+'%';const track=document.createElement('div');track.className='track';const fill=document.createElement('div');fill.className='fill';fill.style.width=score*100+'%';track.append(fill);row.append(title,value,track);$('bars').append(row);}
    $('heatmap').src=result.gradcam_png;$('explanation').textContent=result.explanation;$('latency').textContent=`Analysis: ${result.latency_ms} ms · ${result.training_status.join(' / ')} checkpoints`;
    $('comparison').replaceChildren();for(const [name,values] of Object.entries(result.individual_models)){const p=document.createElement('p');p.textContent=name+': '+Object.entries(values).map(([label,score])=>`${label} ${(score*100).toFixed(1)}%`).join(' · ');$('comparison').append(p);}
  }catch(error){$('error').textContent=error.message;}finally{$('submit').disabled=!ready;$('task').disabled=!ready;$('image').disabled=false;$('submit').textContent='Analyze image →';}
});
$('download').addEventListener('click',()=>{if(!latestResult)return;const blob=new Blob([JSON.stringify(latestResult,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download='cervisight-research-report.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
initialize();
