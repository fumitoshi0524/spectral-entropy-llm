"""Build bilingual reports from archived measurements."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
data=json.loads((root/'docs/results-summary.json').read_text(encoding='utf-8'))
def point(model,name,fraction):return next(x for x in model['curves'][name]['points'] if x['fraction']==fraction)
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
summary=[];adaptation=[];longitudinal=[];regularizers=[];downstream=[]
for scale,m in data['models'].items():
    p=point(m,'entropy',.6);n=point(m,'naive',.6);a=point(m,'activation',.6);t=point(m,'entropy_topk',.6)
    summary.append([scale.upper(),f"{n['ppl']:,.2f}",f"{a['ppl']:,.2f}",f"{p['ppl']:,.2f}",f"{t['ppl']:,.2f}",f"{n['ppl']/p['ppl']:.2f}×"])
    original=m['trajectories']['entropy'][0];e=m['trajectories']['entropy'][-1];b=m['trajectories']['baseline'][-1];t=m['trajectories']['entropy_topk'][-1]
    adaptation.append([scale.upper(),f"{original['ppl']:.2f}",f"{b['ppl']:.2f}",f"{e['ppl']:.2f}",f"{t['ppl']:.2f}",f"{original['er']:.1f} → {e['er']:.1f}",f"{(1-e['er']/original['er'])*100:.1f}%"])
    start,end=m['longitudinal'][0],m['longitudinal'][-1]
    longitudinal.append([scale.upper(),f"{start['er']:.1f} → {end['er']:.1f}",f"{start['sr']:.1f} → {end['sr']:.1f}",f"{(1-end['sr']/start['sr'])*100:.1f}%"])
    e6=m['downstream']['e6']['results'];e5=m['downstream']['e5']['results']
    for label,record in [('Dense',e6['dense']),('Naive SVD, 70%',e6['naive_trunc_r70']),('Entropy + SVD, 70%',e6['e3_gamma10.0_trunc_r70']),('Entropy + top-k + SVD, 70%',e5['e3_5_gamma10.0_mu1.00_topk0.50_trunc_r70'])]:
        downstream.append([scale.upper(),label]+[f"{record[task]['acc,none']*100:.2f}" for task in ['hellaswag','arc_easy','piqa']])
for key,value in data['regularizers']['methods'].items():regularizers.append([key,f"{value['ppl_final']:.2f}",f"{value['er_final']:.1f}",f"{value['sr_final']:.1f}"])

equations=r'''
$$
p_i=\frac{\sigma_i}{\sum_j\sigma_j},\qquad
H(W)=-\sum_i p_i\log p_i,\qquad
\mathrm{ER}(W)=\exp H(W),\qquad
\mathrm{SR}(W)=\frac{\sum_i\sigma_i^2}{\sigma_1^2}.
$$
'''
objective=r'''
$$
\mathcal{L}=\mathcal{L}_{\mathrm{LM}}+
\gamma\sum_{\ell}H_{\mathrm{cached}}(W_\ell)
+\mu\sum_\ell\frac{\|W_\ell-P_\ell W_\ell Q_\ell\|_F^2}{\|W_\ell\|_F^2+\epsilon}.
$$
'''
english=fr'''# Spectral Entropy for Low-Rank Language-Model Compression

[中文报告](README_zh.md) · [Archived data](results/) · [Truncation table](docs/truncation.csv)

## Abstract

This study examines whether concentrating the singular-value spectrum of pretrained language-model weights improves their tolerance to subsequent low-rank truncation. Across Pythia-160M, 410M, and 1.4B, entropy-regularized adaptation reduces mean effective rank by approximately 18–36%. At 60% retained rank, subsequent naive SVD produces 16–61× lower perplexity than naive SVD applied to the original model. These gains come with an increase in untruncated perplexity and do not establish uniform superiority over an activation-aware baseline. The central result is a measurable trade-off between spectral concentration, original-model quality, and truncation robustness.

**The experiment measures numerical rank truncation. Its reconstructed dense weight tensors retain their original shapes; actual checkpoint-size, runtime-memory, and speed reductions are not measured here.**

## 1. Research question

A low stable rank can coexist with a broad singular-value tail. Consequently, energy concentration alone does not establish that a weight matrix can be truncated without a substantial language-modeling penalty. This project connects longitudinal spectral measurements with an intervention: minimize spectral entropy during adaptation, then measure the quality of rank-truncated weights.

The study contains three components: spectral tracking across public Pythia checkpoints; comparison of entropy and normalized nuclear-norm regularization; and cross-scale truncation with optional alignment to the original model's leading subspace. The figures and tables below present the measured cross-scale results.

## 2. Spectral metrics and objective

For singular values of a target weight matrix, the metrics are:
{equations}
ER is the effective number of singular components under this normalized spectrum. SR emphasizes squared spectral energy. Neither quantity is a count of stored model parameters or a direct measure of task-relevant information.

The adaptation objective combines language modeling, a cached spectral-entropy surrogate, and an optional subspace penalty:
{objective}
Here, $P_\ell=U_kU_k^\top$ and $Q_\ell=V_kV_k^\top$ project onto the original dense weight's leading left/right singular subspaces. Pure entropy uses $\mu=0$; the aligned variant uses $\mu=1$ with a 50% rank anchor. The entropy term itself does not specify a target retained rank, while the alignment term does.

Full SVD bases are refreshed periodically. Between refreshes, `diag(U_cached.T @ W @ V_cached)` approximates the spectrum; the implementation clamps this vector before normalization and logarithms. This is a differentiable surrogate in a cached basis, rather than an exact SVD of every updated weight. Gradient flow is handled by ordinary autograd through this projection.

The preliminary comparison uses the **normalized nuclear norm**, $\|W\|_*/\|W\|_F$. This scale-invariant ratio is not itself a convex function. The regularizer strengths are calibrated by initial gradient norms; this controls the initial regularization scale, not the entire subsequent optimization trajectory.

## 3. Experimental setup

The base models are [Pythia deduplicated checkpoints](https://github.com/EleutherAI/pythia), revision `step143000`. Target matrices are QKV, attention output, FFN up, and FFN down in each transformer block. Embeddings, normalization parameters, and biases are excluded from the spectral terms and truncation.

| Setting | Main entropy adaptation |
|---|---|
| Models | Pythia-160M / 410M / 1.4B |
| Corpus protocol | 50,000 C4 training texts; C4 validation perplexity |
| Sequence length | 512 tokens |
| Logged iteration budget | 10,000 batch iterations |
| Gradient accumulation | 4 iterations; 2,500 optimizer updates |
| Optimizer | AdamW, weight decay 0.01, gradient clipping 1.0 |
| Learning rate | 1e-5 / 1e-5 / 5e-5 respectively |
| Entropy strength | γ = 10 |
| SVD basis refresh | Every 1,000 batch iterations |
| Optional alignment | μ = 1, original-model top-k fraction 0.5 |
| Precision | FP32 for 160M/410M; BF16 for 1.4B |
| Evaluation interval | Every 500 batch iterations |

The saved loops accumulate unscaled losses over four batches. Evaluation consumes at most 40 batches, with batch size 2 for 160M/410M and 1 for 1.4B. Comparisons within a model scale therefore share its evaluation protocol; absolute scores across scales do not use identical numbers of evaluated sequences.

The preliminary regularizer comparison uses 160M, 2,000 batch iterations, a 200-iteration SVD refresh, and an entropy calibration anchor of γ = 30.

## 4. Spectral evolution in the original models

![Recorded spectral evolution](img/longitudinal.svg)

The longitudinal archive contains 15 checkpoints from step 0 through step 140,000. The final pretrained models used for intervention are separately loaded at step 143,000. Values below average the four target matrix types across layers.

{table(['Model','ER: initial → last recorded','SR: initial → last recorded','SR reduction'],longitudinal)}

Stable rank falls much more sharply than effective rank. This establishes a divergence between two spectral summaries; it does not by itself prove that the remaining tail can be discarded safely.

## 5. Adaptation changes both rank and quality

{table(['Model','Original PPL','LM-only PPL','Entropy PPL','Entropy + top-k PPL','ER: original → entropy','ER reduction'],adaptation)}

![Effective rank and untruncated perplexity during adaptation](img/adaptation.svg)

Entropy adaptation concentrates the spectrum, but also raises perplexity before any truncation. For 1.4B, perplexity rises from 14.38 to 19.23, while its LM-only adaptation baseline ends at 16.74. A lower ER therefore should not be interpreted as an unconditional improvement in the original model.

The 160M regularizer comparison gives:

{table(['Regularizer','Final PPL','Final ER','Final SR'],regularizers)}

Entropy provides slightly lower perplexity at slightly higher ER than the calibrated normalized nuclear-norm variant. This is a trade-off observed at one calibration setting, rather than a matched-ER dominance result.

## 6. Truncation robustness

![Truncation curves across model scales](img/truncation.svg)

All rows below use **60% retained rank**, $k=\lfloor0.6d\rfloor$, applied to the target matrices. Lower PPL is better. The final column compares pure entropy + SVD with original-model naive SVD.

{table(['Model','Naive SVD PPL','Activation-aware PPL','Entropy + SVD PPL','Entropy + top-k + SVD PPL','Naive / entropy'],summary)}

Relative gains over naive truncation are large, but absolute degradation can remain substantial: 1.4B reaches PPL 394.33 at this budget, compared with 19.23 before truncation. At 60% rank, the activation-aware baseline is stronger for 160M and 410M, while entropy is stronger for 1.4B. At 80% rank, entropy reaches 37.56 versus 84.37 for activation-aware truncation on 410M, and 30.16 versus 84.52 on 1.4B. The ordering depends on both model scale and rank budget.

The activation-aware baseline is this repository's implementation: centered activation covariance, whitening, SVD, and de-whitening. It does not include the full sequential parameter-update procedure of SVD-LLM, so these numbers should not be read as official SVD-LLM results.

### Downstream accuracy at 70% rank

{table(['Model','Variant','HellaSwag (%)','ARC-Easy (%)','PIQA (%)'],downstream)}

These are the archived raw accuracies (`acc,none`), using zero-shot prompts. The entropy variants generally recover some of the naive-truncation loss, but the compressed variants remain below the original dense models. The source records also contain normalized accuracy and standard errors. No significance conclusion is drawn from small differences between regularizer variants.

## 7. Rank budget and deployment storage

The current truncation code reconstructs $U_k\Sigma_kV_k^\top$ and writes it into the original dense `Linear` weight. Actual storage therefore remains dense.

If a future implementation stores two factors instead, an $m\times n$ weight requires $k(m+n)$ values rather than $mn$. For a square layer at 60% rank, factorization requires approximately **1.2×** its original weight count. Across the four Pythia target matrix shapes, factoring every target at the same $k=0.6d$ would require approximately **80% of their original weight values**, before biases, embeddings, other parameters, or metadata. These are algebraic estimates, not measured model-file reductions.

Practical export should skip factorizations that increase storage and then measure the resulting model directly.

## 8. Interpretation and open questions

The records support a relationship between deliberate spectral concentration and improved tolerance to naive truncation. They also show a quality cost before truncation and a nonuniform comparison with activation-aware compression. The main missing comparison is truncation of the matched **10,000-iteration LM-only adapted** baseline: its training trajectory is archived, but an equivalent full truncation curve is not. This limits attribution of all observed improvement specifically to the entropy term.

Additional work would assess seed variability, the effect of cached-basis staleness, measured training overhead, and real factorized inference. The cross-scale differences alone do not establish a mechanism such as “spectral saturation.”

## 9. Code and data

```bash
git clone https://github.com/fumitoshi0524/spectral-entropy-llm.git
cd spectral-entropy-llm
python scripts/build_results.py
```

This rebuilds the report summary, CSV table, and result index from archived JSON, without downloading a model or running an evaluation. Figures and tables in this report present the results directly.

For new experiments, install `requirements.txt`, then use the explicit entropy entry point:

```bash
python pipeline.py --experiment train --scale 160m --variant entropy
python pipeline.py --experiment train --scale 160m --variant entropy-topk
python pipeline.py --experiment train --scale 160m --variant baseline
python pipeline.py --experiment truncate --scale 160m --checkpoint runs/metrics/160m/entropy/checkpoint
```

See [reproduction notes](docs/REPRODUCTION.md) for configuration provenance and historical loop semantics. The original adapted weights were not retained; the release contains their recorded results. New runs save checkpoints under `runs/` and leave the archive untouched.

| Path | Content |
|---|---|
| `src/` | Spectral metrics, cached-basis regularizers, training, dense truncation |
| `pipeline.py`, `configs/entropy.yaml` | Restored entropy experiment entry points and recipe |
| `experiments/regularizer_selection.py` | Preliminary calibrated regularizer comparison |
| `results/` | Archived numerical results |
| `scripts/` | Data extraction, report tables, and figure rendering |
| `legacy/cosine/` | Earlier cosine route, isolated from the entropy entry point |

## 10. Related work

- Roy and Vetterli, [The Effective Rank: A Measure of Effective Dimensionality](https://doi.org/10.5281/zenodo.40328), EUSIPCO 2007, defines the entropy-based effective rank used here.
- Biderman et al., [Pythia](https://proceedings.mlr.press/v202/biderman23a.html), ICML 2023, provides the base models and public training checkpoints.
- Zhao et al., [GaLore](https://proceedings.mlr.press/v235/zhao24s.html), ICML 2024, uses periodic low-rank gradient projection for training-memory efficiency. This project uses periodic **weight** SVD bases to approximate a regularizer; it does not implement GaLore's gradient projection or claim its memory benefits.
- Wang et al., [SVD-LLM](https://arxiv.org/abs/2403.07378), ICLR 2025, combines truncation-aware whitening with sequential low-rank parameter updates. It motivates the activation-aware comparison, subject to the implementation scope described above.
- Dolatabadi et al., [NuMuon](https://arxiv.org/abs/2603.03597), 2026, constrains Muon update directions to improve compressibility. Here, an entropy penalty is added during AdamW adaptation of pretrained dense models; no experimental comparison with NuMuon is claimed.
- Janson et al., [Stabilizing Native Low-Rank LLM Pretraining](https://arxiv.org/abs/2602.12429), ICML 2026, introduces Spectron for stabilizing native factorized training. Its update spectral-norm control addresses a different setting from entropy-based reshaping before post-hoc truncation.
'''

chinese=fr'''# 谱熵正则化与语言模型的低秩截断

[English report](README.md) · [原始结果](results/) · [截断数据表](docs/truncation.csv)

## 摘要

本研究考察：在预训练语言模型的继续训练中集中权重的奇异值谱，能否改善后续低秩截断的质量。在 Pythia-160M、410M 和 1.4B 上，谱熵正则化使平均有效秩降低约 18–36%；保留 60% 秩时，后续朴素 SVD 的困惑度，相比直接截断原始模型降低约 16–61 倍。与此同时，未截断模型的困惑度也有所上升，且该方法并未在所有模型规模与秩预算上优于激活感知基线。主要发现是谱集中、原始模型质量与截断鲁棒性之间可观察的权衡。

**本实验测量数值秩截断后的模型质量。现有实现将低秩近似重建为原尺寸的稠密张量，尚未测量实际模型文件、运行显存或推理速度的缩减。**

## 1. 研究问题

较低的稳定秩可以与较宽的奇异值谱尾部同时存在。因此，仅观察谱能量的集中，尚不能确定截断这些方向是否会造成明显的语言建模质量损失。本项目将训练过程中的谱测量，与谱熵正则化后的截断实验联系起来。

研究包含三个环节：跟踪公开 Pythia 检查点的谱演化；比较谱熵与归一化核范数正则项；在三个模型规模上比较截断曲线，并考察对原模型主子空间的可选约束。下文通过曲线和表格直接呈现不同模型规模与保留秩比例的测量结果。

## 2. 谱指标与目标函数

对于目标矩阵的奇异值，定义：
{equations}
ER 表示当前归一化谱下的有效奇异分量数；SR 更强调平方谱能量。二者均不等于实际存储参数量，也不能直接解释为模型保留了多少任务相关信息。

继续训练的目标由语言建模损失、缓存谱基下的熵近似，以及可选的子空间约束组成：
{objective}
其中，$P_\ell=U_kU_k^\top$、$Q_\ell=V_kV_k^\top$ 分别投影到原始稠密权重的前 k 个左右奇异向量子空间。纯谱熵方法取 $\mu=0$；组合方法取 $\mu=1$，子空间锚点使用 50% 秩。谱熵项本身不预先指定目标保留秩，组合方法中的子空间项则需要设定 k。

实现周期性重算完整 SVD 基，在间隔内以 `diag(U_cached.T @ W @ V_cached)` 近似谱量，并在归一化和取对数前进行下限裁剪。这是缓存基下的可微代理，不能始终等同于当前权重的精确奇异值；梯度通过该投影由 autograd 计算。

预实验中的另一正则项为**归一化核范数** $\|W\|_*/\|W\|_F$。该比值具有尺度不变性，但自身并非凸函数。两种正则项按初始梯度范数校准强度，以控制初始正则作用的尺度；这不保证之后每一步的优化作用完全相同。

## 3. 实验设置

基础模型使用 [Pythia 去重版本](https://github.com/EleutherAI/pythia)，加载 `step143000` 检查点。目标矩阵为每个 Transformer block 的 QKV、注意力输出、FFN up 和 FFN down；嵌入、归一化参数和 bias 不参与谱正则项及截断。

| 设置 | 主谱熵实验 |
|---|---|
| 模型 | Pythia-160M / 410M / 1.4B |
| 语料方案 | 50,000 条 C4 训练文本；C4 validation 困惑度 |
| 序列长度 | 512 tokens |
| 记录的迭代预算 | 10,000 次批次迭代 |
| 梯度累积 | 每 4 次迭代更新一次，共 2,500 次优化器更新 |
| 优化器 | AdamW，weight decay 0.01，梯度裁剪 1.0 |
| 学习率 | 分别为 1e-5 / 1e-5 / 5e-5 |
| 谱熵强度 | γ = 10 |
| SVD 基刷新 | 每 1,000 次批次迭代 |
| 可选子空间约束 | μ = 1，原模型 top-k 比例 0.5 |
| 精度 | 160M/410M 使用 FP32；1.4B 使用 BF16 |
| 测量间隔 | 每 500 次批次迭代 |

保存的训练循环将四个批次的梯度直接累加，未按累积次数缩放损失。PPL 最多使用 40 个批次计算：160M/410M 的 batch size 为 2，1.4B 为 1。因此同一模型规模内沿用对应的评估方案，跨规模的绝对分数并非基于完全相同的评估序列数量。

正则项选择预实验在 160M 上进行，使用 2,000 次批次迭代、200 次迭代的 SVD 刷新间隔，以及 γ = 30 的谱熵梯度校准锚点。

## 4. 原模型的谱演化

![原模型的谱演化记录](img/longitudinal.svg)

纵向记录包含从 step 0 到 step 140,000 的 15 个检查点；用于后续干预实验的最终模型另行加载 step 143,000。下表对每层四类目标矩阵计算指标后取平均。

{table(['模型','ER：初始 → 最后记录','SR：初始 → 最后记录','SR 降幅'],longitudinal)}

稳定秩的下降显著大于有效秩的下降。这说明两种谱汇总指标存在分化，但尚不能仅据此断定谱尾部可以被安全舍弃。

## 5. 继续训练同时改变谱结构和模型质量

{table(['模型','原始 PPL','LM-only PPL','谱熵 PPL','谱熵 + top-k PPL','ER：原始 → 谱熵','ER 降幅'],adaptation)}

![继续训练期间的有效秩与未截断困惑度](img/adaptation.svg)

谱熵正则化使谱分布集中，也提高了截断前的 PPL。例如，1.4B 从 14.38 上升到 19.23，同样继续训练的 LM-only 基线结束于 16.74。因此，ER 降低不能被直接解释为原模型质量改善。

160M 上的正则项比较为：

{table(['正则项','最终 PPL','最终 ER','最终 SR'],regularizers)}

谱熵方法相较校准后的归一化核范数具有稍低的 PPL 和稍高的 ER。这是单个校准配置下的权衡结果，尚非严格匹配 ER 后的全面优势。

## 6. 截断鲁棒性

![不同规模模型的截断曲线](img/truncation.svg)

下表统一使用 **60% 保留秩**，即对目标矩阵取 $k=\lfloor0.6d\rfloor$。PPL 越低越好。最后一列比较纯谱熵后 SVD 与原模型朴素 SVD 的 PPL 比值。

{table(['模型','朴素 SVD PPL','激活感知 PPL','谱熵 + SVD PPL','谱熵 + top-k + SVD PPL','朴素 / 谱熵'],summary)}

相对朴素截断的改善很大，但绝对退化仍可能明显：1.4B 在此预算下的 PPL 为 394.33，而截断前为 19.23。60% 秩时，激活感知基线在 160M、410M 上更好，谱熵在 1.4B 上更好；80% 秩时，410M 的谱熵 PPL 为 37.56，激活感知为 84.37，1.4B 则分别为 30.16 和 84.52。比较结论取决于模型规模与保留秩比例。

这里的激活感知基线是本仓库自己的实现：对中心化激活协方差进行白化、SVD 与反白化，未包括完整 SVD-LLM 的顺序参数更新过程，因此不能将这些记录视为 SVD-LLM 官方结果。

### 70% 秩下的下游准确率

{table(['模型','变体','HellaSwag (%)','ARC-Easy (%)','PIQA (%)'],downstream)}

表中使用保存的 zero-shot 原始准确率 `acc,none`。谱熵变体通常能恢复部分朴素截断造成的损失，但截断后的变体仍低于原始稠密模型。原始记录另含 normalized accuracy 与标准误；不根据变体之间的小幅差距作显著性结论。

## 7. 保留秩与实际部署存储

当前实现将 $U_k\Sigma_kV_k^\top$ 重建后写入原尺寸 `Linear`，因此实际权重仍以稠密形式存储。

若后续真正保存两个低秩因子，$m\times n$ 权重的元素数会从 $mn$ 变为 $k(m+n)$。对方阵保留 60% 秩，因子元素数约为原矩阵的 **1.2 倍**；对 Pythia 四类目标矩阵统一使用 $k=0.6d$，总因子权重数约为这些矩阵原权重数的 **80%**，尚未包括 bias、嵌入、其他参数及元数据。这些是公式估算，不是实际模型文件测量。

后续部署应跳过会增加存储的分解，并对真正导出的模型测量体积与性能。

## 8. 解释与尚未解决的问题

现存结果支持“主动集中谱分布与更好的朴素截断容忍度相关”，也显示截断前的质量代价及对激活感知方法不一致的优劣关系。主要缺少的比较是**同样继续训练 10,000 次迭代的 LM-only 模型的完整截断曲线**：其训练轨迹仍在，但对应曲线未保存，因此不能把全部改善单独归因于谱熵项。

后续还需要研究多随机种子、缓存谱基陈旧程度、实际训练开销和真正因子化推理的影响。仅依据跨规模差异，尚不能确立“谱饱和”等机制性解释。

## 9. 代码与数据使用

```bash
git clone https://github.com/fumitoshi0524/spectral-entropy-llm.git
cd spectral-entropy-llm
python scripts/build_results.py
```

这个命令从原始 JSON 生成报告汇总数据、CSV 和结果索引，无需下载模型或运行评测。研究结果直接呈现在本报告的曲线与表格中。

新实验需要安装 `requirements.txt`，随后通过明确的谱熵入口运行：

```bash
python pipeline.py --experiment train --scale 160m --variant entropy
python pipeline.py --experiment train --scale 160m --variant entropy-topk
python pipeline.py --experiment train --scale 160m --variant baseline
python pipeline.py --experiment truncate --scale 160m --checkpoint runs/metrics/160m/entropy/checkpoint
```

配置来源和历史循环语义见[复现说明](docs/REPRODUCTION.md)。历史继续训练权重已不再保留，本次公开其结果记录；后续运行会将权重保存在 `runs/`，不会覆盖历史数据。

| 路径 | 内容 |
|---|---|
| `src/` | 谱指标、缓存基正则项、训练与稠密截断 |
| `pipeline.py`、`configs/entropy.yaml` | 恢复后的谱熵实验入口与配置 |
| `experiments/regularizer_selection.py` | 校准后的正则项选择预实验 |
| `results/` | 历史数值记录 |
| `scripts/` | 数据提取、报告表格与绘图 |
| `legacy/cosine/` | 与谱熵入口分开的早期余弦路线 |

## 10. 相关工作

- Roy 与 Vetterli，[The Effective Rank: A Measure of Effective Dimensionality](https://doi.org/10.5281/zenodo.40328)，EUSIPCO 2007：定义本研究使用的熵型有效秩。
- Biderman 等，[Pythia](https://proceedings.mlr.press/v202/biderman23a.html)，ICML 2023：提供基础模型和公开训练检查点。
- Zhao 等，[GaLore](https://proceedings.mlr.press/v235/zhao24s.html)，ICML 2024：通过周期性低秩梯度投影降低训练状态占用。本项目周期性缓存的是**权重** SVD 基，用于近似正则项，并未实现 GaLore 的梯度投影或主张获得其显存收益。
- Wang 等，[SVD-LLM](https://arxiv.org/abs/2403.07378)，ICLR 2025：结合截断感知白化与顺序低秩参数更新，为本项目的激活感知对照提供背景；本仓库实现范围如上文所述。
- Dolatabadi 等，[NuMuon](https://arxiv.org/abs/2603.03597)，2026：对 Muon 更新方向施加核范数约束以改善可压缩性。本研究在预训练稠密模型的 AdamW 继续训练中加入谱熵项，未进行与 NuMuon 的实验比较。
- Janson 等，[Stabilizing Native Low-Rank LLM Pretraining](https://arxiv.org/abs/2602.12429)，ICML 2026：提出 Spectron 稳定原生因子化训练；其更新谱范数控制与本研究在事后截断前集中谱分布的设置不同。
'''
(root/'README.md').write_text(english,encoding='utf-8')
(root/'README_zh.md').write_text(chinese,encoding='utf-8')
print('Wrote both reports from shared archived data.')
