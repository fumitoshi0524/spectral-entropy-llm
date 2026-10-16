"""Model loading, weight extraction, SVDWeightBackup."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.models.gpt_neox import GPTNeoXForCausalLM


TARGET_SUFFIXES = (
    "attention.query_key_value",
    "attention.dense",
    "mlp.dense_h_to_4h",
    "mlp.dense_4h_to_h",
)

WEIGHT_KEY_MAP = {
    "attention.query_key_value": "qkv_weight",
    "attention.dense": "o_weight",
    "mlp.dense_h_to_4h": "ffn_up_weight",
    "mlp.dense_4h_to_h": "ffn_down_weight",
}

WEIGHT_LAYER_PATTERNS = {
    "attention.query_key_value": "gpt_neox.layers.{L}.attention.query_key_value.weight",
    "attention.dense": "gpt_neox.layers.{L}.attention.dense.weight",
    "mlp.dense_h_to_4h": "gpt_neox.layers.{L}.mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h": "gpt_neox.layers.{L}.mlp.dense_4h_to_h.weight",
}


def model_name(scale: str, cfg) -> str:
    return cfg.model_spec(scale)["name"]


def dtype_for_scale(scale: str) -> torch.dtype:
    """bfloat16 for 1.4B+ (memory), float32 for smaller."""
    if scale == "1.4b":
        return torch.bfloat16
    return torch.float32


def load_model(scale: str, cfg, revision: str = "step143000",
               device: str = "cuda", dtype: torch.dtype | None = None):
    if dtype is None:
        dtype = dtype_for_scale(scale)
    spec = cfg.model_spec(scale)
    m = GPTNeoXForCausalLM.from_pretrained(
        spec["name"], revision=revision, torch_dtype=dtype)
    if device != "cpu":
        m = m.to(device)
    m.eval()
    return m


def load_tokenizer(scale: str, cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model_spec(scale)["name"])
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def target_modules(model):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and any(n.endswith(s) for s in TARGET_SUFFIXES):
            yield n, m


def count_target_modules(model) -> int:
    return sum(1 for _ in target_modules(model))


def load_layer_weights(model, num_layers: int) -> dict[int, dict[str, torch.Tensor]]:
    state = model.state_dict()
    weights: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx in range(num_layers):
        lw = {}
        for suffix, pattern in WEIGHT_LAYER_PATTERNS.items():
            param_path = pattern.format(L=layer_idx)
            t = state.get(param_path)
            if t is not None:
                lw[WEIGHT_KEY_MAP[suffix]] = t.detach().clone()
        if lw:
            weights[layer_idx] = lw
    return weights


# ── WeightBackup (truncation + restore) ──

@dataclass
class WeightBackup:
    data: dict  # name → original weight (CPU)

    @classmethod
    def save(cls, model):
        return cls({n: m.weight.data.clone().cpu() for n, m in target_modules(model)})

    def restore(self, model):
        for n, m in target_modules(model):
            m.weight.data.copy_(self.data[n].to(m.weight.device))
