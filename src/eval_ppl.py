"""Perplexity on C4 validation."""

import math, torch


@torch.no_grad()
def evaluate_ppl(model, dataloader, max_batches=40):
    model.eval()
    total_nll, total_scored = 0.0, 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(model.device)
        am = batch["attention_mask"].to(model.device)
        labels = ids.clone(); labels[am == 0] = -100
        out = model(input_ids=ids, attention_mask=am, labels=labels)
        n = (labels[:, 1:] != -100).sum().item()
        total_nll += out.loss.item() * n
        total_scored += n
    avg = total_nll / total_scored if total_scored > 0 else float("inf")
    return math.exp(avg) if avg < 100 else float("inf")
