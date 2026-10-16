# Reproduction and archive notes

## What is available

The release contains historical numerical records, the available training and truncation implementation, restored entropy entry points, and a static result explorer. The adapted model weights were not retained after the original server was cleared. The `results/` files are preserved records; they are not outputs from a new run performed for this release.

`source-manifest.json` maps original relative file names to released paths and records hashes of the source snapshot. `result-index.json` records the hashes and top-level schemas of the published result files. The restored entry point and documentation were added during release preparation; they are not represented as the exact original experiment launcher.

## Record map

| Directory | Meaning |
|---|---|
| `<scale>/e1/longitudinal.json` | Per-matrix ER/SR at 15 Pythia checkpoints, steps 0–140,000 |
| `<scale>/e2/truncation.json` | Original-model naive and activation-aware truncation curves |
| `<scale>/e3/baseline/result.json` | LM-only adaptation trajectory |
| `<scale>/e3/gamma10.0/result.json` | Pure entropy adaptation, γ = 10 |
| `<scale>/e3_5/gamma10.0_mu1.00_topk0.50/result.json` | Entropy plus original top-k subspace alignment |
| `<scale>/e4/post_trunc.json` | Pure entropy's post-adaptation truncation curve; `baseline` is the original dense model's naive curve |
| `<scale>/e4_5/post_trunc.json` | Aligned entropy's post-adaptation truncation curve |
| `<scale>/e5/downstream.json` | Aligned entropy and its 70%-rank downstream evaluations |
| `<scale>/e6/downstream.json` | Pure entropy and its 70%-rank downstream evaluations |
| `e0_v2/regularizer_comparison.json` | 160M preliminary regularizer selection |
| `160m_sigma_vectors.json` | Additional archived singular-value vectors |

`e4`'s baseline must not be relabelled as the matched LM-only adapted baseline. The saved data do not provide the latter's complete 10,000-iteration truncation curve.

## Lightweight data and figure regeneration

From the repository root:

```bash
python scripts/build_results.py
python scripts/write_reports.py
python scripts/plot_results.py
```

The first two commands use only Python's standard library. The third requires matplotlib. None loads a language model or runs an evaluation. The figure generator uses the same compact `demo/data.json` as the explorer. Raw measurements remain unchanged in `results/`.

## New entropy runs

Install the declared dependencies in a fresh environment. The historical environment was not fully pinned; `requirements.txt` expresses compatible API bounds rather than a historical lockfile. CUDA builds must match the machine.

`configs/entropy.yaml` reconstructs the main recipe from the available manuscript, configuration, code, and scalar records. The original root pipeline had been changed back to the earlier cosine route; the new root entry point exposes entropy explicitly.

```bash
python pipeline.py --experiment train --scale 160m --variant entropy
python pipeline.py --experiment train --scale 160m --variant entropy-topk
python pipeline.py --experiment train --scale 160m --variant baseline
python pipeline.py --experiment truncate --scale 160m --checkpoint runs/metrics/160m/entropy/checkpoint
```

An explicit training command starts a new model run. Each variant writes its checkpoint, tokenizer, result log, configuration, library versions, and training-text fingerprint beneath `runs/metrics/<scale>/<variant>/`. These files are ignored by Git. Choose `--output` for independent seeds/runs; reuse of an existing directory is rejected. Truncation requires an actual local checkpoint and reconstructs dense matrices, matching the evaluation representation used in the archive.

Other entry points:

```bash
python pipeline.py --experiment e1 --scale 160m
python pipeline.py --experiment e2 --scale 160m
python experiments/regularizer_selection.py
python pipeline.py --experiment downstream --scale 160m --checkpoint runs/metrics/160m/entropy/checkpoint
```

The regularizer-selection script retains its standalone historical loop and writes to `runs/e0_v2/`. It has fixed configuration constants at its top. Both maintained entropy data loaders now require the requested C4 samples and fail if they cannot be loaded. The original loader could fall back to another corpus, and the archived records do not include complete dataset fingerprints for the old runs.

### Numerical semantics

- A logged `step` in the main training functions counts a batch iteration. With accumulation of four, 10,000 iterations give 2,500 AdamW updates.
- The preserved loop sums gradients without dividing each batch loss by the accumulation factor. Replacing it with averaged accumulation changes the training recipe.
- The dataloader is not shuffled by the preserved implementation. Padding labels are masked, and perplexity averages loss over scored next-token positions.
- The main regularizer uses full cached weight SVD bases and refreshes them every 1,000 iterations; the initial comparison uses 200. The surrogate diagonal can deviate from the actual singular values as the basis becomes stale.
- The 1.4B path uses BF16 and gradient checkpointing. Evaluation uses at most 40 batches with batch size 1 for 1.4B and 2 for smaller models.

## Activation-aware baseline

The implementation computes centered input-activation covariance, adds diagonal regularization, whitens each weight, truncates it by SVD, and de-whitens it. It retains dense reconstructed tensors and does not perform the full sequential low-rank parameter updates of the SVD-LLM paper. Its records are identified as this repository's activation-aware baseline.

## Deployment estimates

The explorer's factor counts follow `k(m+n) / (mn)`. The displayed combined target ratio assumes all four Pythia target matrix types are factorized at the same rank. It excludes embeddings, normalization, biases, and metadata. No factorized exporter, measured memory reduction, or inference speed-up is claimed by this release.

## Historical cosine route

`legacy/cosine/` retains the earlier source and configuration as a separate route. It is not the default entry point for the entropy study. Historical paper drafts and private repository history are not part of this curated release.
