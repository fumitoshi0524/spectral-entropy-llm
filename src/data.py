"""C4 / wikitext dataloader."""

import gzip, json, logging
import torch
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def _load_c4_raw(split: str, max_samples: int) -> list[str]:
    n_shards = {"validation": 8, "train": 1024}.get(split, 8)
    texts = []
    for i in range(min(n_shards, 3)):
        fname = f"en/c4-{split}.{i:05d}-of-{n_shards:05d}.json.gz"
        try:
            path = hf_hub_download("allenai/c4", fname, repo_type="dataset")
        except Exception:
            continue
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if len(texts) >= max_samples:
                    return texts
                try:
                    t = json.loads(line).get("text", "").strip()
                    if t:
                        texts.append(t)
                except Exception:
                    pass
    raise RuntimeError(
        f"Only {len(texts)} of {max_samples} requested C4 {split} samples were loaded. "
        "Check dataset access; this entry point does not substitute another corpus."
    )


class _TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def build_dataloader(tokenizer, split: str = "validation",
                     max_samples: int = 200, max_seq_length: int = 512,
                     batch_size: int = 2, seed: int = 42) -> DataLoader:
    texts = _load_c4_raw(split, max_samples)
    ds = _TextDataset(texts, tokenizer, max_seq_length)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, generator=g)
