#!/usr/bin/env python3
"""
Cosine k=384→192→96→48, per-matrix ER truncation protects SVD.
Each phase: dense start → cosine train → ER flat → next k.
"""
import json, math, gc, time, logging
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers.models.gpt_neox import GPTNeoXForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("verify_cosine")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "EleutherAI/pythia-160m-deduped"
REV = "step143000"; SEED = 42
LR = 1e-5; LAM = 50; WD = 0.01; GA = 4
MAX_PER_K = 8000; SVD_I = 500; EVAL_I = 500
FLAT_W = 3; FLAT_T = 0.005
K_LIST = [384, 192, 96, 48]
OUT = "results_new/cosine_k_sweep.json"

T = ("attention.query_key_value","attention.dense","mlp.dense_h_to_4h","mlp.dense_4h_to_h")
TM = {"attention.query_key_value":"QKV","attention.dense":"AttnO",
      "mlp.dense_h_to_4h":"FFNup","mlp.dense_4h_to_h":"FFNdn"}

# ── data ──
def _c4(split,n):
    import gzip; from huggingface_hub import hf_hub_download
    ns={"validation":8,"train":1024}.get(split,8); txt=[]
    for i in range(min(ns,3)):
        try: p=hf_hub_download("allenai/c4",f"en/c4-{split}.{i:05d}-of-{ns:05d}.json.gz",repo_type="dataset")
        except: continue
        with gzip.open(p,"rt",encoding="utf-8") as f:
            for ln in f:
                if len(txt)>=n: return txt
                try: t=json.loads(ln).get("text","").strip()
                except: continue
                if t: txt.append(t)
    return txt

class _TD(Dataset):
    def __init__(self,t,tk,ml): self.t=t;self.tk=tk;self.ml=ml
    def __len__(self): return len(self.t)
    def __getitem__(self,i):
        e=self.tk(self.t[i],truncation=True,padding="max_length",max_length=self.ml,return_tensors="pt")
        return {k:v.squeeze(0) for k,v in e.items()}

def loader(split,n,tok,bs=2):
    g=torch.Generator();g.manual_seed(SEED)
    return DataLoader(_TD(_c4(split,n),tok,512),batch_size=bs,generator=g)

def tmods(model):
    for n,m in model.named_modules():
        if isinstance(m,nn.Linear) and any(n.endswith(s) for s in T): yield n,m

@torch.no_grad()
def ppl(model,dl,mb=40):
    model.eval();tn=0.0;ts=0
    for i,b in enumerate(dl):
        if i>=mb: break
        ids=b["input_ids"].to(DEVICE);am=b["attention_mask"].to(DEVICE)
        lb=ids.clone();lb[am==0]=-100
        o=model(input_ids=ids,attention_mask=am,labels=lb)
        n=(lb[:,1:]!=-100).sum().item();tn+=o.loss.item()*n;ts+=n
    return math.exp(tn/ts) if ts>0 else float("inf")

# ── SVD cache ──
class Cache:
    def __init__(self,U,Vt): self.U=U;self.Vt=Vt
    @classmethod
    @torch.no_grad()
    def build(cls,W):
        U,_,Vt=torch.linalg.svd(W.float(),full_matrices=False)
        return cls(U=U.to(W.device),Vt=Vt.to(W.device))
    def sigma(self,W):
        return torch.diag((self.U.T@W.float())@self.Vt.T)

def cos(W,c,k):
    s=c.sigma(W);s2=torch.clamp(s,min=0).pow(2)
    return 1.0-torch.sqrt(s2[:k].sum()/(s2.sum()+1e-8))

@torch.no_grad()
def er_single(W):
    s=torch.linalg.svdvals(W.float());s=s[s>1e-12]
    return float(torch.exp(-(s/s.sum()*torch.log(s/s.sum()+1e-12)).sum())) if len(s)>=2 else float(len(s))

# ── per-matrix ER + truncate (not used in current run) ──
@torch.no_grad()
def trunc_and_report(model):
    er_map={}; backup={}
    for n,m in tmods(model):
        W=m.weight.data.float();U,S,Vt=torch.linalg.svd(W,full_matrices=False)
        s_pos=S[S>1e-12]
        er=float(torch.exp(-(s_pos/s_pos.sum()*torch.log(s_pos/s_pos.sum()+1e-12)).sum())) if len(s_pos)>=2 else float(len(s_pos))
        k_er=max(1,int(er))
        backup[n]=m.weight.data.clone()
        m.weight.data.copy_((U[:,:k_er]*S[:k_er].unsqueeze(0))@Vt[:k_er,:])
        er_map[n]=er
    return backup,er_map

def per_type(er_map):
    pt={t:[] for t in TM.values()}
    for n,v in er_map.items():
        for suf,name in TM.items():
            if n.endswith(suf): pt[name].append(v)
    return {t:{"er_mean":np.mean(pt[t]),"er_std":np.std(pt[t]),"er_min":min(pt[t]),"er_max":max(pt[t])} for t in pt if pt[t]}

# ── main ──
def main():
    torch.manual_seed(SEED);np.random.seed(SEED)
    tok=AutoTokenizer.from_pretrained(MODEL);tok.padding_side="right"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=GPTNeoXForCausalLM.from_pretrained(MODEL,revision=REV).to(DEVICE)
    el=loader("validation",200,tok,2);tl=loader("train",50000,tok,2)
    ds={k:v.clone().cpu() for k,v in model.state_dict().items()}
    results={"lam":LAM,"lr":LR,"phases":{}}
    t0=time.perf_counter()

    for phase_i,k in enumerate(K_LIST):
        label=f"k={k}"
        logger.info("=== Phase %d/4: %s ===",phase_i+1,label)
        model.load_state_dict(ds)
        er_map={n:er_single(mm.weight.data) for n,mm in tmods(model)}
        pv=ppl(model,el);er=np.mean(list(er_map.values()))
        logger.info("  [%s] S=0 (%.0fs) PPL=%.1f ER=%.0f",label,time.perf_counter()-t0,pv,er)
        phase_log=[{"step":0,"ppl":round(pv,2),"er":round(er,1)}]

        model.train();opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
        ti=iter(tl);caches={n:Cache.build(mm.weight.data) for n,mm in tmods(model)}
        flat_count=0

        for step in range(1,MAX_PER_K+1):
            if step%SVD_I==0:
                caches={n:Cache.build(mm.weight.data) for n,mm in tmods(model)}

            try: b=next(ti)
            except: ti=iter(tl);b=next(ti)

            ids=b["input_ids"].to(DEVICE);am=b["attention_mask"].to(DEVICE)
            lb=ids.clone();lb[am==0]=-100
            o=model(input_ids=ids,attention_mask=am,labels=lb)
            reg=torch.tensor(0.0,device=DEVICE)
            for n,mm in tmods(model):
                c=caches.get(n)
                if c is not None: reg=reg+cos(mm.weight,c,k)
            (o.loss+LAM*reg).backward()
            if step%GA==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step();opt.zero_grad()

            if step%EVAL_I==0:
                pv=ppl(model,el);er_map={n:er_single(mm.weight.data) for n,mm in tmods(model)};er=np.mean(list(er_map.values()))
                logger.info("  [%s] S=%d (%.0fs) PPL=%.1f ER=%.0f",label,step,time.perf_counter()-t0,pv,er)
                phase_log.append({"step":step,"ppl":round(pv,2),"er":round(er,1)})
                if pv>500:
                    logger.warning("  [%s] PPL exploded -> stop",label);break

        del opt;gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        results["phases"][str(k)]={"log":phase_log,"k":k}
        if pv>500: break

    with open(OUT,"w") as f:json.dump(results,f,indent=2)
    logger.info("Saved -> %s",OUT)

if __name__=="__main__":
    main()
