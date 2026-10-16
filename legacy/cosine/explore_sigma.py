#!/usr/bin/env python3
"""Extract full σ evolution: per-layer, per-type, all 5 checkpoints."""
import json, torch, sys
import numpy as np
from transformers.models.gpt_neox import GPTNeoXForCausalLM

CKPTS = [0, 10000, 40000, 70000, 100000, 140000]
MODEL = "EleutherAI/pythia-160m-deduped"
OUT = "160m_sigma_vectors.json"

TARGETS = {
    "attention.query_key_value": "QKV",
    "attention.dense": "AttnO",
    "mlp.dense_h_to_4h": "FFNup",
    "mlp.dense_4h_to_h": "FFNdn",
}

WEIGHT_PATH = {
    "attention.query_key_value": "gpt_neox.layers.{L}.attention.query_key_value.weight",
    "attention.dense": "gpt_neox.layers.{L}.attention.dense.weight",
    "mlp.dense_h_to_4h": "gpt_neox.layers.{L}.mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h": "gpt_neox.layers.{L}.mlp.dense_4h_to_h.weight",
}

@torch.no_grad()
def svd_full(W):
    s = torch.linalg.svdvals(W.float())
    return s.cpu().numpy().astype(np.float64)

def er_sr(s):
    s_pos = s[s > 1e-12]
    if len(s_pos) >= 2:
        p = s_pos / s_pos.sum()
        er = float(np.exp(-sum(p * np.log(p + 1e-12))))
    else:
        er = float(len(s_pos))
    sr = float((s**2).sum() / (s[0]**2 + 1e-12))
    return er, sr

out = {"d_model": 768, "num_layers": 12, "head_dim": 64, "num_heads": 12,
       "checkpoints": {}, "steps": CKPTS}

for step in CKPTS:
    rev = "step0" if step == 0 else f"step{step}"
    print(f"Loading {rev}...")
    model = GPTNeoXForCausalLM.from_pretrained(MODEL, revision=rev)
    state = model.state_dict()
    del model

    ckpt_data = {}
    for suffix, short in TARGETS.items():
        for L in range(12):
            path = WEIGHT_PATH[suffix].format(L=L)
            W = state.get(path)
            if W is None: continue
            s = svd_full(W)
            r = len(s)
            er, sr = er_sr(s)
            ckpt_data[f"{short}_L{L}"] = {
                "sigma": s.tolist(),
            "r": r,
                "er": round(er, 0),
                "sr": round(sr, 1),
                "sigma1": round(float(s[0]), 2),
                "sigma_min": round(float(s[-1]), 4),
                "sigma1_over_sigma2": round(float(s[0] / s[1]), 2) if r > 1 else 0,
                "r": r,
                "top5": [round(float(x), 2) for x in s[:5]],
                "bot5": [round(float(x), 4) for x in s[-5:]],
            }
    out["checkpoints"][str(step)] = ckpt_data
    print(f"  done ({len(ckpt_data)} matrices)")

# Printed summary
print("\n" + "="*140)
for L in range(12):
    for short in ["QKV", "AttnO", "FFNup", "FFNdn"]:
        c0 = out["checkpoints"]["0"].get(f"{short}_L{L}")
        if c0 is None: continue

        data_steps = {}
        for st_key in ["0","10000","40000","70000","100000","140000"]:
            ck = out["checkpoints"][st_key].get(f"{short}_L{L}")
            if ck: data_steps[st_key] = ck

        s1_vals  = [f"{data_steps[k]['sigma1']:.2f}" for k in ["0","10000","40000","70000","100000","140000"] if k in data_steps]
        smin_vals = [f"{data_steps[k]['sigma_min']:.4f}" for k in ["0","10000","40000","70000","100000","140000"] if k in data_steps]
        er_vals = [f"{int(data_steps[k]['er'])}" for k in ["0","10000","40000","70000","100000","140000"] if k in data_steps]
        sr_vals = [f"{data_steps[k]['sr']:.1f}" for k in ["0","10000","40000","70000","100000","140000"] if k in data_steps]

        c_last = data_steps["140000"]

        print(f"{short}_L{L:<3d} "
              f"ER=[{'→'.join(er_vals)}]  "
              f"SR=[{'→'.join(sr_vals)}]")
        print(f"           "
              f"σ₁=[{'→'.join(s1_vals)}]  "
              f"σ_min=[{'→'.join(smin_vals)}]  "
              f"top5={c_last['top5']}  bot5={c_last['bot5']}")
    print()

with open(OUT, "w") as f:
    json.dump(out, f)
print(f"Saved → {OUT}")
