"""Explicit entry points for entropy training and truncation experiments.

New runs write to runs/, never to the archived results/ directory.
"""
import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path

import torch
from src.config import load_config
from src.data import build_dataloader
from src.decompose import truncate_all, restore_all
from src.eval_ppl import evaluate_ppl
from src.metrics import compute_metrics
from src.model import load_model, load_tokenizer
from src.train import set_seed, train_baseline, train_spectral, train_spectral_topk

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True,choices=['e1','e2','train','truncate','downstream'])
    p.add_argument('--config',default='configs/entropy.yaml')
    p.add_argument('--scale',default='160m',choices=['160m','410m','1.4b'])
    p.add_argument('--variant',default='entropy',choices=['baseline','entropy','entropy-topk'])
    p.add_argument('--checkpoint',type=Path,help='Required local checkpoint for truncate/downstream')
    p.add_argument('--output',type=Path,help='New output directory; defaults to runs/metrics/<scale>/<variant>')
    p.add_argument('--device')
    args=p.parse_args();cfg=load_config(args.config)
    if args.device:cfg.device=args.device
    output=args.output or cfg.output_path/args.scale/args.variant
    output=output.resolve();archive=Path('results').resolve()
    if output==archive or archive in output.parents:p.error('Choose an output outside the historical results/ archive.')
    if args.experiment in ['truncate','downstream'] and args.checkpoint is None:p.error('--checkpoint is required; historical tuned weights were not retained.')
    if args.experiment=='train' and ((output/'result.json').exists() or (output/'checkpoint').exists()):
        p.error('This run directory already contains results. Choose a new --output directory.')
    set_seed(cfg.seed)
    if args.experiment in ['e1','e2']:
        if args.experiment=='e1':from experiments.e1_longitudinal import run
        else:from experiments.e2_truncation import run
        run(args.scale,cfg,output/args.experiment);return
    device=cfg.device
    if device.startswith('cuda') and not torch.cuda.is_available():p.error('CUDA unavailable. Choose --device cpu explicitly if desired.')
    dtype=torch.bfloat16 if args.scale=='1.4b' and device.startswith('cuda') else torch.float32
    if args.checkpoint:
        from transformers import AutoModelForCausalLM,AutoTokenizer
        model=AutoModelForCausalLM.from_pretrained(args.checkpoint,torch_dtype=dtype).to(device)
        tok=AutoTokenizer.from_pretrained(args.checkpoint)
    else:model=load_model(args.scale,cfg,device=device,dtype=dtype);tok=load_tokenizer(args.scale,cfg)
    bs=1 if args.scale=='1.4b' else cfg.eval['batch_size']
    ev=cfg.eval
    if args.experiment=='downstream':
        from src.eval_downstream import evaluate_downstream
        write_json(output/'downstream.json',evaluate_downstream(model,tok,['hellaswag','arc_easy','piqa'],0))
        return
    eval_loader=build_dataloader(tok,'validation',ev['eval_samples'],ev['max_seq_length'],bs,cfg.seed)
    ppl_fn=lambda m,loader:evaluate_ppl(m,loader,ev['max_eval_batches'])
    if args.experiment=='train':
        train_loader=build_dataloader(tok,'train',ev['train_samples'],ev['max_seq_length'],bs,cfg.seed)
        c=cfg.e3;iterations=c['iterations'];accum=c['grad_accum']
        if iterations%accum:p.error('iterations must be divisible by grad_accum for the preserved historical loop.')
        common=dict(steps=iterations,lr=float(c['lr'][args.scale]),grad_accum=accum,eval_every=c['eval_every'],weight_decay=c['weight_decay'],seed=cfg.seed,ppl_fn=ppl_fn,use_gc=args.scale=='1.4b')
        if args.variant=='baseline':history=train_baseline(model,train_loader,eval_loader,**common)
        elif args.variant=='entropy':history=train_spectral(model,train_loader,eval_loader,'entropy',c['strength'],svd_interval=c['svd_interval'],**common)
        else:history=train_spectral_topk(model,train_loader,eval_loader,c['strength'],c['mu'],c['topk_fraction'],svd_interval=c['svd_interval'],**common)
        model.save_pretrained(output/'checkpoint');tok.save_pretrained(output/'checkpoint')
        fingerprint=hashlib.sha256()
        for text in train_loader.dataset.texts:fingerprint.update(text.encode('utf-8'));fingerprint.update(b'\0')
        write_json(output/'result.json',{'variant':args.variant,'log':history,'ppl_final':history[-1]['ppl'],'er_final':history[-1]['er'],'sr_final':history[-1]['sr']})
        write_json(output/'run.json',{'config':cfg.__dict__,'iterations':iterations,'optimizer_updates':iterations//accum,'train_text_sha256':fingerprint.hexdigest(),'gradient_accumulation':'sum; historical loss is not divided by grad_accum','versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','huggingface_hub']}})
    elif args.experiment=='truncate':
        rows=[]
        for fraction in cfg.e2['rank_fractions']:
            rank=max(1,int(cfg.model_spec(args.scale)['d_model']*fraction))
            backup=truncate_all(model,rank)
            try:rows.append({'fraction':fraction,'rank':rank,'ppl':ppl_fn(model,eval_loader)})
            finally:restore_all(model,backup)
        write_json(output/'post_trunc.json',{'checkpoint':str(args.checkpoint),'representation':'dense reconstructed truncated weights','results':rows})


if __name__=='__main__':main()
