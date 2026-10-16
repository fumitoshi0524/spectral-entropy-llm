#!/usr/bin/env python3
"""
E0 v2 — Regularizer Selection
===============================
LR = 1e-5 (QTA: ~1/30 of Pythia 3e-4).
Entropy γ=30 anchors calibration (grad-norm matched).
Entropy vs nuclear + baseline = 3 runs.
GaLore-style: SVD every 200 steps → σ_approx = diag(UᵀWV).
"""

import gc, json, math, time, warnings
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers.models.gpt_neox import GPTNeoXForCausalLM

warnings.filterwarnings("ignore")
for lib in ("huggingface_hub","datasets","httpx","httpcore","urllib3"):
    import logging; logging.getLogger(lib).setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("e0")

# ── Config ────────────────────────────────────────

MODEL_NAME = "EleutherAI/pythia-160m-deduped"
REVISION   = "step143000"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL    = 768

TARGET_SUFFIXES = (
    "attention.query_key_value",
    "attention.dense",
    "mlp.dense_h_to_4h",
    "mlp.dense_4h_to_h",
)

LR           = 1e-5          # QTA: 1/30 of Pythia pretrain lr=3e-4
TRAIN_STEPS  = 2000
GRAD_ACCUM   = 4
EVAL_EVERY   = 100
SVD_INTERVAL = 200
SEED         = 42
MAX_EVAL_BATCHES = 40
WD           = 0.01          # AdamW baseline weight decay

TRUNC_FRACTIONS = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

TRAIN_SAMPLES = 4000
EVAL_SAMPLES  = 200
MAX_SEQ_LEN   = 512
BATCH_SIZE    = 2
ENTROPY_ANCHOR = 30.0

OUTPUT_DIR = Path("runs/e0_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────

def _load_c4_raw(split: str, max_samples: int) -> list[str]:
    import gzip
    from huggingface_hub import hf_hub_download
    n_shards = {"validation":8,"train":1024}.get(split,8)
    texts = []
    for i in range(min(n_shards,3)):
        try:
            path = hf_hub_download("allenai/c4",
                f"en/c4-{split}.{i:05d}-of-{n_shards:05d}.json.gz",repo_type="dataset")
        except Exception: continue
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line in f:
                if len(texts)>=max_samples: return texts
                try:
                    t=json.loads(line).get("text","").strip()
                    if t: texts.append(t)
                except Exception: pass
    raise RuntimeError(f"Only {len(texts)} of {max_samples} requested C4 {split} samples were loaded; no corpus substitution is performed.")

class _TextDataset(Dataset):
    def __init__(self,t,tk,ml): self.texts=t;self.tok=tk;self.max_len=ml
    def __len__(self): return len(self.texts)
    def __getitem__(self,i):
        enc=self.tok(self.texts[i],truncation=True,padding="max_length",
                     max_length=self.max_len,return_tensors="pt")
        return {k:v.squeeze(0) for k,v in enc.items()}

def build_dataloader(split,max_samples,tokenizer,batch_size=2):
    texts=_load_c4_raw(split,max_samples)
    g=torch.Generator(); g.manual_seed(SEED)
    return DataLoader(_TextDataset(texts,tokenizer,MAX_SEQ_LEN),batch_size=batch_size,generator=g)

# ── Model ─────────────────────────────────────────

def load_model():
    tok=AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.padding_side="right"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=GPTNeoXForCausalLM.from_pretrained(MODEL_NAME,revision=REVISION,torch_dtype=torch.float32)
    if DEVICE!="cpu": model=model.to(DEVICE)
    model.eval(); return model,tok

def target_modules(model):
    for n,m in model.named_modules():
        if isinstance(m,nn.Linear) and any(n.endswith(s) for s in TARGET_SUFFIXES):
            yield n,m

# ── Metrics ───────────────────────────────────────

@torch.no_grad()
def compute_metrics(model):
    er_vals,sr_vals=[],[]
    for _,m in target_modules(model):
        s=torch.linalg.svdvals(m.weight.data.float())
        s=s[s>1e-12]
        if len(s)>=2:
            p=s/s.sum()
            er_vals.append(float(torch.exp(-(p*torch.log(p+1e-12)).sum())))
        else: er_vals.append(float(len(s)))
        sr_vals.append(float((torch.norm(m.weight,'fro')**2)/(s[0]**2+1e-12)))
    return np.mean(er_vals),np.mean(sr_vals)

@torch.no_grad()
def evaluate_ppl(model,dataloader,max_batches=MAX_EVAL_BATCHES):
    model.eval(); total_nll=0.0; total_scored=0
    for i,batch in enumerate(dataloader):
        if i>=max_batches: break
        ids=batch["input_ids"].to(DEVICE); am=batch["attention_mask"].to(DEVICE)
        labels=ids.clone(); labels[am==0]=-100
        out=model(input_ids=ids,attention_mask=am,labels=labels)
        n=(labels[:,1:]!=-100).sum().item()
        total_nll+=out.loss.item()*n; total_scored+=n
    avg=total_nll/total_scored if total_scored>0 else float("inf")
    return math.exp(avg) if avg<100 else float("inf")

# ── SVD Truncation ────────────────────────────────

def svd_truncate(W,k):
    W32=W.float(); k=min(k,min(W32.shape[0],W32.shape[1]))
    U,S,Vt=torch.linalg.svd(W32,full_matrices=False)
    return (U[:,:k]*S[:k].unsqueeze(0))@Vt[:k,:]

@dataclass
class WeightBackup:
    data:dict
    @classmethod
    def save(cls,model):
        return cls({n:m.weight.data.clone().cpu() for n,m in target_modules(model)})
    def truncate(self,model,rank):
        for n,m in target_modules(model):
            m.weight.data.copy_(svd_truncate(self.data[n].to(DEVICE),rank))
    def restore(self,model):
        for n,m in target_modules(model):
            m.weight.data.copy_(self.data[n].to(DEVICE))

def run_truncation_curve(model,eval_loader):
    backup=WeightBackup.save(model); results={}
    for frac in TRUNC_FRACTIONS:
        k=max(1,int(D_MODEL*frac))
        backup.truncate(model,k); results[frac]=evaluate_ppl(model,eval_loader)
    backup.restore(model); return results

# ── SVD Cache ─────────────────────────────────────

@dataclass
class SVDCache:
    U:torch.Tensor; Vt:torch.Tensor

    @classmethod
    @torch.no_grad()
    def build(cls,W):
        W_fp32 = W.float()
        try:
            U,_,Vt=torch.linalg.svd(W_fp32,full_matrices=False)
        except torch._C._LinAlgError:
            # Fallback: add tiny noise to break degenerate singular values
            eps = 1e-9 * W_fp32.norm()
            U,_,Vt=torch.linalg.svd(W_fp32+eps*torch.randn_like(W_fp32),full_matrices=False)
        return cls(U=U.to(device=W.device),Vt=Vt.to(device=W.device))

    def approx_sigma(self,W):
        return torch.diag((self.U.T@W.float())@self.Vt.T)

# ── Regularizer Losses ────────────────────────────

def nuclear_loss_layer(W,cache):
    return torch.clamp(cache.approx_sigma(W),min=0).sum()/(W.norm()+1e-8)

def entropy_loss_layer(W,cache):
    σ=cache.approx_sigma(W); σ=torch.clamp(σ,min=1e-12)
    if len(σ)<2: return torch.tensor(0.0,device=W.device)
    p=σ/σ.sum(); return -(p*torch.log(p+1e-12)).sum()

# ── Calibration (entropy γ=30 anchors gradient-norm matching) ─

@torch.no_grad()
def _entropy_grad_norm(W_param):
    """‖∂H/∂W‖_F where H = -Σ p_i log p_i, analytical."""
    W=W_param.data.float()
    U,S,Vt=torch.linalg.svd(W,full_matrices=False)
    S=S[S>1e-12]; sum_s=S.sum()
    p=S/sum_s; log_p=torch.log(p+1e-12)
    C=float((S*(1+log_p)).sum()/(sum_s**2+1e-8))
    gs=torch.full_like(S,C)-(1+log_p)/sum_s
    grad_full=(U.to(W_param.device)*gs.unsqueeze(0).to(W_param.device))@Vt.to(W_param.device)
    return float(grad_full.norm())

def calibrate(model):
    ent_grad=0.0; nuc_grad=0.0
    for _,m in target_modules(model):
        Wp=m.weight; W=Wp.data.float()
        U,S,Vt=torch.linalg.svd(W,full_matrices=False); S=S[S>1e-12]
        frob_norm=float(torch.sqrt((S**2).sum()))
        ent_grad+=_entropy_grad_norm(Wp)
        nuc_grad+=math.sqrt(len(S))/frob_norm if frob_norm>0 else 0

    ref_pull = ent_grad * ENTROPY_ANCHOR
    return {
        "entropy": ENTROPY_ANCHOR,
        "nuclear": ref_pull/nuc_grad if nuc_grad>0 else 1.0,
    }

# ── Training Loop ─────────────────────────────────

def train_one(model,train_loader,eval_loader,method,strength,label):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    train_iter=iter(train_loader)

    def _refresh():
        return {n:SVDCache.build(m.weight.data) for n,m in target_modules(model)}

    caches=_refresh()
    log=[]

    def _eval(step):
        model.eval(); ppl=evaluate_ppl(model,eval_loader); er,sr=compute_metrics(model)
        model.train()
        e={"step":step,"ppl":round(ppl,2),"er":round(er,1),"sr":round(sr,1)}
        log.append(e)
        logger.info("  [%s] step=%d PPL=%.2f ER=%.1f SR=%.1f",label,step,ppl,er,sr)

    _eval(0)

    for step in range(1,TRAIN_STEPS+1):
        if step%SVD_INTERVAL==0: caches=_refresh()

        try: batch=next(train_iter)
        except StopIteration: train_iter=iter(train_loader); batch=next(train_iter)

        ids=batch["input_ids"].to(DEVICE); am=batch["attention_mask"].to(DEVICE)
        labels=ids.clone(); labels[am==0]=-100
        out=model(input_ids=ids,attention_mask=am,labels=labels)
        lm_loss=out.loss

        reg_loss=torch.tensor(0.0,device=DEVICE)
        for n,m in target_modules(model):
            c=caches.get(n)
            if c is None: continue
            if method=="nuclear":    reg_loss+=nuclear_loss_layer(m.weight,c)
            elif method=="entropy":  reg_loss+=entropy_loss_layer(m.weight,c)

        total=lm_loss+strength*reg_loss
        total.backward()

        if step%GRAD_ACCUM==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); opt.zero_grad()

        if step%EVAL_EVERY==0: _eval(step)

    del opt; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"log":log,"ppl_final":log[-1]["ppl"],"er_final":log[-1]["er"],"sr_final":log[-1]["sr"]}

# ── Main ──────────────────────────────────────────

def main():
    if (OUTPUT_DIR/"regularizer_comparison.json").exists() or any((OUTPUT_DIR/name/"checkpoint").exists() for name in ["baseline","entropy","nuclear"]):
        raise FileExistsError("Choose a fresh OUTPUT_DIR for a new regularizer-comparison run.")
    torch.manual_seed(SEED); np.random.seed(SEED)
    model,tok=load_model()
    def save_checkpoint(label):
        model.save_pretrained(OUTPUT_DIR/label/"checkpoint")
        tok.save_pretrained(OUTPUT_DIR/label/"checkpoint")
    ev_loader=build_dataloader("validation",EVAL_SAMPLES,tok,BATCH_SIZE)
    tr_loader=build_dataloader("train",TRAIN_SAMPLES,tok,BATCH_SIZE)

    ppl_d=evaluate_ppl(model,ev_loader)
    er_d,sr_d=compute_metrics(model)
    dense_state={k:v.clone().cpu() for k,v in model.state_dict().items()}
    logger.info("Dense: PPL=%.2f ER=%.1f SR=%.1f",ppl_d,er_d,sr_d)

    cal=calibrate(model)
    logger.info("Calibration (grad-norm matched to entropy γ=30):")
    for k,v in cal.items(): logger.info("  %s = %.4f",k,v)

    results={}
    # Baseline — 2000-step LM-only with wd=0.01
    logger.info("=== BASELINE ===")
    model.load_state_dict(dense_state)
    r=train_one(model,tr_loader,ev_loader,"entropy",0.0,"baseline")
    save_checkpoint("baseline")
    r["trunc_curve"]={str(k):v for k,v in run_truncation_curve(model,ev_loader).items()}
    results["baseline"]=r

    # Entropy + Nuclear — same gradient-norm pull as entropy γ=30
    for method in ["entropy","nuclear"]:
        s=cal[method]; label=method
        logger.info("=== %s  strength=%.4f ===",label.upper(),s)
        model.load_state_dict(dense_state)
        r=train_one(model,tr_loader,ev_loader,method,s,label)
        save_checkpoint(label)
        r["method"]=method; r["strength"]=s
        r["trunc_curve"]={str(k):v for k,v in run_truncation_curve(model,ev_loader).items()}
        results[label]=r

    output={
        "config":{
            "model":MODEL_NAME,"d_model":D_MODEL,"lr":LR,"wd":WD,
            "svd_interval":SVD_INTERVAL,
            "dense_ppl":ppl_d,"dense_er":er_d,"dense_sr":sr_d,
            "calibration":cal,
            "note":"Entropy vs Nuclear, grad-norm matched to entropy γ=30.",
        },
        "results":results,
    }
    with open(OUTPUT_DIR/"regularizer_comparison.json","w") as f: json.dump(output,f,indent=2)

    print("\n"+"="*100)
    print("SUMMARY")
    print("="*100)
    print(f"{'Method':<24s} {'Str':>10s} {'PPL':>8s} {'ER':>8s} {'ΔER%':>8s} {'SR':>8s}")
    print("-"*66)
    print(f"{'Dense':<24s} {'-':>10s} {ppl_d:>8.1f} {er_d:>8.1f} {'-':>8s} {sr_d:>8.1f}")
    for label in ["baseline","entropy","nuclear"]:
        r=results[label]; der=(r["er_final"]-er_d)/er_d*100
        s=results[label].get("strength",0)
        print(f"{label:<24s} {s:>10.4f} {r['ppl_final']:>8.1f} {r['er_final']:>8.1f} {der:>+7.1f}% {r['sr_final']:>8.1f}")

if __name__=="__main__":
    import time as _time
    _t0=_time.perf_counter()
    main()
    logger.info("E0 v2 complete in %.1f min.",(_time.perf_counter()-_t0)/60)
