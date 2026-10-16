"""Derive report data and tables from archived JSON; no model dependencies."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def stats(entry):return {k:entry[k] for k in ['step','ppl','er','sr'] if k in entry}
def mean(values):return sum(values)/len(values)

def main():
    data={'models':{},'regularizers':{},'metric':'perplexity','records':'archived measurements','schema_version':1}
    csv_rows=[]
    for scale,d in [('160m',768),('410m',1024),('1.4b',2048)]:
        base=Path('results')/scale
        e2=read(base/'e2/truncation.json')
        e4=read(base/'e4/post_trunc.json')
        top=read(base/'e4_5/post_trunc.json')
        pure=read(base/'e3/gamma10.0/result.json')
        aligned=read(base/'e3_5/gamma10.0_mu1.00_topk0.50/result.json')
        baseline=read(base/'e3/baseline/result.json')
        curves={
            'naive':{'source':str(base/'e2/truncation.json').replace('\\','/'),'points':e2['naive']},
            'activation':{'source':str(base/'e2/truncation.json').replace('\\','/'),'points':e2['svdllm']},
            'entropy':{'source':str(base/'e4/post_trunc.json').replace('\\','/'),'points':e4['results']['gamma10.0']},
            'entropy_topk':{'source':str(base/'e4_5/post_trunc.json').replace('\\','/'),'points':top['results']['gamma10.0_mu1.00_topk0.50']},
        }
        trajectories={'baseline':[stats(x) for x in baseline['log']],'entropy':[stats(x) for x in pure['log']],'entropy_topk':[stats(x) for x in aligned['log']]}
        longitudinal=[]
        for entry in read(base/'e1/longitudinal.json')['results']:
            er=[];sr=[]
            for layer in entry['layers'].values():
                for name in ['qkv_weight','o_weight','ffn_up_weight','ffn_down_weight']:
                    metric=layer.get(name)
                    if metric and metric.get('er') is not None and metric.get('sr') is not None:
                        er.append(metric['er']);sr.append(metric['sr'])
            longitudinal.append({'step':entry['step'],'er':mean(er),'sr':mean(sr),'matrices':len(er)})
        downstream={name:read(base/name/'downstream.json') for name in ['e5','e6']}
        model={'dimension':d,'dense_ppl':e2['dense_ppl'],'curves':curves,'trajectories':trajectories,'longitudinal':longitudinal,'downstream':downstream}
        for name,curve in curves.items():
            for point in curve['points']:
                if not math.isfinite(point['ppl']) or point['ppl']<=0:raise ValueError(f'Invalid PPL: {scale}/{name}')
                k=point['rank']
                csv_rows.append({'scale':scale,'method':name,'rank_fraction':point['fraction'],'rank':k,'ppl':point['ppl'],'dense_ppl':model['dense_ppl'],'target_factor_weight_ratio_estimate':16*d*k/(12*d*d),'source':curve['source']})
        data['models'][scale]=model
    e0=read(Path('results/e0_v2/regularizer_comparison.json'))
    data['regularizers']={'config':e0['config'],'methods':{k:{'ppl_final':v['ppl_final'],'er_final':v['er_final'],'sr_final':v['sr_final'],'log':[stats(x) for x in v['log']]} for k,v in e0['results'].items()}}
    (ROOT/'docs/results-summary.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n',encoding='utf-8')
    with (ROOT/'docs/truncation.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    index=[]
    for file in sorted((ROOT/'results').rglob('*.json')):
        value=json.loads(file.read_text(encoding='utf-8'))
        index.append({'path':str(file.relative_to(ROOT)).replace('\\','/'),'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'keys':list(value)})
    (ROOT/'docs/result-index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    # Include run-local source differences rather than merging conflicting baseline values.
    differences=[]
    for scale,m in data['models'].items():
        a=m['downstream']['e5']['results'];b=m['downstream']['e6']['results']
        for variant in ['dense','naive_trunc_r70']:
            if a[variant]!=b[variant]:differences.append({'scale':scale,'variant':variant,'e5':a[variant],'e6':b[variant]})
    (ROOT/'docs/downstream-source-differences.json').write_text(json.dumps(differences,indent=2)+'\n',encoding='utf-8')
    print('Derived',len(csv_rows),'truncation rows from',len(index),'archived JSON files.')


if __name__=='__main__':main()
