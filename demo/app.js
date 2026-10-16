const $=id=>document.getElementById(id);
const params=new URLSearchParams(location.search),zh=params.get('lang')==='zh';
const local=(en,cn)=>zh?cn:en;
document.documentElement.lang=zh?'zh-CN':'en';
const messages={title:'谱分布的集中，<br>与截断质量的权衡。',intro:'通过谱熵正则化，语言模型能否更好地承受低秩截断？在三个模型规模上，查看谱结构、困惑度与不同秩预算之间的关系。',archive:'历史实验记录',local:'无需 GPU',truncTitle:'1. 选择保留秩比例',truncSub:'每个数据点均来自保存的测量记录，困惑度越低越好。',model:'模型',rank:'保留秩',method:'方法',vsDense:'PPL / 原始未截断',vsOwn:'PPL / 自身100%秩',trainingTitle:'2. 查看继续训练过程',trainingSub:'谱分布变得集中，同时未截断模型的质量也可能下降。',storageTitle:'3. 秩不等于文件大小',storageSub:'历史实现将截断结果重建为稠密权重。下列因子参数量来自公式估算，尚非实际存储收益。',matrix:'矩阵',shape:'形状',factorRatio:'因子 / 原权重数',readingTitle:'如何理解这些结果',reading:'有效秩降低不保证语言建模质量改善。相对朴素截断的大幅改善，也可能伴随很高的绝对困惑度。方法间的优劣取决于模型规模与保留秩比例，应结合完整曲线判断。',scopeTitle:'研究范围与原始记录',scope:'页面展示历史结果，继续训练后的权重已不再保留。激活感知曲线来自本仓库的白化实现，未包含完整 SVD-LLM 流程。同样继续训练的 LM-only 基线缺少完整截断曲线，因此尚不能把全部改善单独归因于谱熵项。显存和速度尚未测量。',csv:'查看 72 条原始截断记录 ↗',json:'下载页面数据',footer:'Wenjun Pu · 谱熵研究',repository:'代码、方法与历史结果 ↗'};
if(zh)document.querySelectorAll('[data-i18n]').forEach(el=>{const key=el.dataset.i18n;if(messages[key]){if(key==='title')el.innerHTML=messages[key];else el.textContent=messages[key];}});
$('language').textContent=zh?'English':'中文';
$('report').textContent=local('Research report ↗','研究报告 ↗');
$('report').href='https://github.com/fumitoshi0524/spectral-entropy-llm/blob/main/'+(zh?'README_zh.md':'README.md');
const methods={naive:{en:'Naive SVD',zh:'朴素 SVD',color:'#7d8794'},activation:{en:'Activation-aware',zh:'激活感知',color:'#b67a36'},entropy:{en:'Entropy + SVD',zh:'谱熵 + SVD',color:'#247f7a'},entropy_topk:{en:'Entropy + top-k + SVD',zh:'谱熵 + top-k + SVD',color:'#6860a9'}};
const names=id=>methods[id][zh?'zh':'en'];
const selected=new Set(Object.keys(methods));let data,metric='er';
const ns='http://www.w3.org/2000/svg';
const number=value=>new Intl.NumberFormat(zh?'zh-CN':'en-US',{maximumFractionDigits:2}).format(value);
function el(tag,attrs={},text){const node=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))node.setAttribute(k,v);if(text!==undefined)node.textContent=text;return node;}
function chart(svg,{lines,xmin,xmax,xlabel,ylabel,log=false,marker=null,baseline=null}){
  svg.replaceChildren();const [,,W,H]=svg.getAttribute('viewBox').split(' ').map(Number),pad={l:62,r:18,t:20,b:50};
  const all=lines.flatMap(l=>l.points.map(p=>p.y));if(baseline!==null)all.push(baseline);
  if(!all.length)return;
  let low=Math.min(...all),high=Math.max(...all);
  if(log){low=Math.floor(Math.log10(low));high=Math.ceil(Math.log10(high));}else{const spread=Math.max(1,high-low);low=Math.max(0,low-spread*.1);high+=spread*.12;}
  if(high===low)high=low+1;
  const x=v=>pad.l+(v-xmin)/(xmax-xmin)*(W-pad.l-pad.r),y=v=>H-pad.b-((log?Math.log10(v):v)-low)/(high-low)*(H-pad.t-pad.b);
  const tick=(axis,value,label)=>svg.append(el('text',axis==='y'?{x:pad.l-10,y:value+4,'text-anchor':'end'}:{x:value,y:H-pad.b+23,'text-anchor':'middle'},label));
  if(marker!==null)svg.append(el('rect',{x:x(marker)-8,y:pad.t,width:16,height:H-pad.t-pad.b,class:'selected'}));
  const yticks=log?Array.from({length:high-low+1},(_,i)=>Math.pow(10,low+i)):Array.from({length:5},(_,i)=>low+(high-low)*i/4);
  for(const value of yticks){svg.append(el('line',{x1:pad.l,x2:W-pad.r,y1:y(value),y2:y(value),stroke:'#e8edee'}));tick('y',y(value),log?number(value):number(value));}
  for(let i=0;i<=5;i++){const value=xmin+(xmax-xmin)*i/5;tick('x',x(value),number(value));}
  svg.append(el('text',{x:pad.l,y:H-8,class:'axis-label'},xlabel),el('text',{x:pad.l,y:12,class:'axis-label'},ylabel));
  if(baseline!==null)svg.append(el('line',{x1:pad.l,x2:W-pad.r,y1:y(baseline),y2:y(baseline),stroke:'#37424a','stroke-dasharray':'3 4'}));
  for(const line of lines){
    const points=[...line.points].sort((a,b)=>a.x-b.x);
    svg.append(el('polyline',{points:points.map(p=>`${x(p.x)},${y(p.y)}`).join(' '),fill:'none',stroke:line.color,'stroke-width':2.2}));
    for(const p of points){const circle=el('circle',{cx:x(p.x),cy:y(p.y),r:marker===p.x?5:3.5,fill:line.color,tabindex:0});circle.append(el('title',{},`${line.label} · ${xlabel}: ${number(p.x)} · ${ylabel}: ${number(p.y)}`));svg.append(circle);}
  }
}
for(const [id,m] of Object.entries(methods)){
  const button=document.createElement('button');button.type='button';button.setAttribute('aria-pressed','true');
  const dot=document.createElement('span');dot.className='dot';dot.style.background=m.color;button.append(dot,document.createTextNode(names(id)));
  button.onclick=()=>{if(selected.has(id)){if(selected.size===1)return;selected.delete(id);}else selected.add(id);button.setAttribute('aria-pressed',String(selected.has(id)));render();};$('legend').append(button);
}
function row(parent,values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}parent.append(tr);}
function render(){
  const mobile=window.innerWidth<650;
  $('truncation').setAttribute('viewBox',mobile?'0 0 440 320':'0 0 940 380');
  $('training').setAttribute('viewBox',mobile?'0 0 440 300':'0 0 610 330');
  const scale=$('model').value,m=data.models[scale],fraction=Number($('rank').value)/100;
  $('rank-value').textContent=number(fraction*100)+'%';
  const current=new URL(location.href);current.searchParams.set('model',scale);current.searchParams.set('rank',String(fraction*100));history.replaceState(null,'',current);
  const other=new URL(current);other.searchParams.set('lang',zh?'en':'zh');$('language').href=other;
  const lines=Object.entries(m.curves).filter(([id])=>selected.has(id)).map(([id,c])=>({color:methods[id].color,label:names(id),points:c.points.map(p=>({x:p.fraction*100,y:p.ppl}))}));
  chart($('truncation'),{lines,xmin:50,xmax:100,xlabel:local('Retained rank (%)','保留秩比例 (%)'),ylabel:local('Perplexity · log scale','困惑度 · 对数坐标'),log:true,marker:fraction*100,baseline:m.dense_ppl});
  $('comparison').replaceChildren();row($('comparison'),[local('Original dense reference','原始未截断参考'),number(m.dense_ppl),'1.00×','—']);
  let k;
  for(const id of selected){const c=m.curves[id],p=c.points.find(x=>x.fraction===fraction),own=c.points.find(x=>x.fraction===1);k=p.rank;row($('comparison'),[names(id),number(p.ppl),(p.ppl/m.dense_ppl).toFixed(2)+'×',(p.ppl/own.ppl).toFixed(2)+'×']);}
  $('rank-summary').textContent=local(`Selected: ${scale.toUpperCase()} · k = ${k} of d = ${m.dimension}. Dotted line: original dense perplexity ${number(m.dense_ppl)}.`,`当前：${scale.toUpperCase()} · k = ${k}，隐藏维度 d = ${m.dimension}。虚线为原始未截断 PPL ${number(m.dense_ppl)}。`);
  $('source').replaceChildren(document.createTextNode(local('Source records: ','原始记录：')));
  const paths=[...new Set([...selected].map(id=>m.curves[id].source))];paths.forEach((path,i)=>{if(i)$('source').append(' · ');const a=document.createElement('a');a.href='https://github.com/fumitoshi0524/spectral-entropy-llm/blob/main/'+path;a.textContent=path.replace('results/'+scale+'/','');$('source').append(a);});
  const trainLines=Object.entries(m.trajectories).map(([id,points])=>({label:id==='baseline'?local('LM-only','仅语言建模'):names(id),color:id==='baseline'?'#7d8794':methods[id].color,points:points.map(p=>({x:p.step,y:p[metric]}))}));
  chart($('training'),{lines:trainLines,xmin:0,xmax:10000,xlabel:local('Batch iterations','批次迭代数'),ylabel:metric==='er'?local('Effective rank','有效秩'):'PPL'});
  const first=m.trajectories.entropy[0],last=m.trajectories.entropy.at(-1);
  $('training-summary').textContent=local(`Entropy: ER ${number(first.er)} → ${number(last.er)}; untruncated PPL ${number(first.ppl)} → ${number(last.ppl)}. Gray: LM-only; teal: entropy; purple: entropy + top-k.`,`纯谱熵：ER ${number(first.er)} → ${number(last.er)}；未截断 PPL ${number(first.ppl)} → ${number(last.ppl)}。灰：仅语言建模；绿：谱熵；紫：谱熵 + top-k。`);
  $('factors').replaceChildren();for(const [name,a,b] of [['QKV',3,1],[local('Attention output','注意力输出'),1,1],['FFN up / down',4,1]])row($('factors'),[name,name==='FFN up / down'?'4d × d / d × 4d':(a===1?'d':a+'d')+' × '+(b===1?'d':b+'d'),(k*(a+b)/(a*b*m.dimension)).toFixed(2)+'×']);
  $('shape-note').textContent=local(`All four targets factored at k=${k}: ${(16*k/(12*m.dimension)*100).toFixed(1)}% of their original weight count. This excludes embeddings, biases, and other parameters.`,`四类目标矩阵统一分解为 k=${k}：因子权重数为原目标权重的 ${(16*k/(12*m.dimension)*100).toFixed(1)}%。不包括嵌入、bias 及其他参数。`);
}
for(const id of ['er','ppl'])$(id).onclick=()=>{metric=id;for(const item of ['er','ppl']){$(item).classList.toggle('active',item===id);$(item).setAttribute('aria-pressed',String(item===id));}render();};
$('model').onchange=$('rank').oninput=()=>{if(data)render();};
window.addEventListener('resize',()=>{if(data)render();});
try{
  const response=await fetch('data.json');if(!response.ok)throw new Error('Data download failed');data=await response.json();
  const scale=params.get('model'),rank=params.get('rank');if(data.models[scale])$('model').value=scale;if(['50','60','70','80','90','100'].includes(rank))$('rank').value=rank;
  render();
}catch(error){$('error').textContent=local('The result records could not load. Please reload or open the source table.','结果记录加载失败，请刷新页面或打开原始数据表。');console.error(error);}
