"""Render report figures from the report's derived archived data."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
data=json.loads((root/'docs/results-summary.json').read_text(encoding='utf-8'))
colors={'naive':'#7d8794','activation':'#b67a36','entropy':'#247f7a','entropy_topk':'#6860a9','baseline':'#7d8794'}
labels={'naive':'Naive SVD','activation':'Activation-aware','entropy':'Entropy','entropy_topk':'Entropy + top-k','baseline':'LM-only'}
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})
fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
for ax,(scale,m) in zip(axes,data['models'].items()):
    for name,curve in m['curves'].items():
        points=sorted(curve['points'],key=lambda x:x['fraction'])
        ax.plot([x['fraction']*100 for x in points],[x['ppl'] for x in points],'-o',ms=3,color=colors[name],label=labels[name])
    ax.axhline(m['dense_ppl'],color='#333b46',ls=':',lw=1,label='Original dense')
    ax.set(yscale='log',xlabel='Retained rank (%)',ylabel='Perplexity (log scale)',title=scale.upper())
    ax.grid(alpha=.16)
handles,legends=axes[0].get_legend_handles_labels();fig.legend(handles,legends,loc='outside lower center',ncol=5,frameon=False,fontsize=9)
fig.savefig(root/'img/truncation.svg');fig.savefig(root/'img/truncation.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(12,6.2),layout='constrained')
for column,(scale,m) in enumerate(data['models'].items()):
    for name,points in m['trajectories'].items():
        for row,metric in enumerate(['er','ppl']):
            ax=axes[row,column]
            ax.plot([x['step'] for x in points],[x[metric] for x in points],color=colors[name],label=labels[name],lw=1.5)
            ax.grid(alpha=.16);ax.set_xlabel('Batch iterations')
    axes[0,column].set_title(scale.upper());axes[0,column].set_ylabel('Effective rank');axes[1,column].set_ylabel('Untruncated perplexity')
handles,legends=axes[0,0].get_legend_handles_labels();fig.legend(handles,legends,loc='outside lower center',ncol=3,frameon=False)
fig.savefig(root/'img/adaptation.svg');fig.savefig(root/'img/adaptation.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
for (scale,m),color in zip(data['models'].items(),['#247f7a','#b67a36','#6860a9']):
    for ax,metric in zip(axes,['er','sr']):
        ax.plot([x['step']/1000 for x in m['longitudinal']],[100*x[metric]/m['dimension'] for x in m['longitudinal']],color=color,label=scale.upper())
        ax.set_xlabel('Pythia checkpoint step (thousands)');ax.grid(alpha=.16)
axes[0].set_ylabel('Effective rank / hidden dimension (%)');axes[1].set_ylabel('Stable rank / hidden dimension (%)')
axes[1].legend(frameon=False)
fig.savefig(root/'img/longitudinal.svg');fig.savefig(root/'img/longitudinal.png',dpi=160);plt.close(fig)
print('Rendered three figure sets.')
