"""Render static diagnostics from the existing audit JSON files only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
a=json.loads((OUT/'audit_summary.json').read_text(encoding='utf-8'))
p=json.loads((OUT/'runtime/probe.json').read_text(encoding='utf-8'))
b=json.loads((OUT/'runtime/boundary_probe.json').read_text(encoding='utf-8'))

fig,axes=plt.subplots(1,3,figsize=(13,4.5))
rows=a['experiment_metrics']; x=np.arange(len(rows)); colors=['#286caa','#b74b47']+['#718eaa']*4
for ax,key,title in zip(axes,['roc_auc','ap','f1'],['ROC-AUC','PR-AUC (AP)','Raw F1']):
    values=[r[key] for r in rows]; ax.bar(x,values,color=colors)
    ax.set_xticks(x,[r['mode'] for r in rows]); ax.set(title=title,ylim=(0,max(values)*1.2))
    for i,v in enumerate(values): ax.text(i,v+max(values)*.02,f'{v:.4f}',ha='center',fontsize=9)
fig.suptitle('Matched six-mode runs (one seed); E0 to E1 changes several structural factors')
fig.tight_layout(); fig.savefig(OUT/'ablation_metrics.png',dpi=150); plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(14,4.5))
for layer,ax in enumerate(axes):
    e0=p['modes']['E0']['splits']['train']['layers'][layer]
    e1=p['modes']['E1']['splits']['train']['layers'][layer]
    x=np.arange(4)
    ax.bar(x-.18,e0['mean_gate'],width=.36,label='E0'); ax.bar(x+.18,e1['mean_gate'],width=.36,label='E1')
    ax.set_xticks(x,[f'Expert {i}\np={s}' for i,s in enumerate(e0['patch_sizes'])])
    ax.set(title=f'Layer {layer+1}',ylim=(0,1),ylabel='Mean gate weight'); ax.legend()
fig.suptitle('Frozen best checkpoints: gate contribution on 64 normal training windows')
fig.tight_layout(); fig.savefig(OUT/'routing_gate_comparison.png',dpi=150); plt.close(fig)

def position(mode,state):
    return np.mean([b['modes'][mode]['splits'][split][state]['position_MSE_0_to_95'] for split in ['train','val']],axis=0)

fig,axes=plt.subplots(1,2,figsize=(14,5))
for mode in ['E0','E1']:
    axes[0].plot(np.arange(96),position(mode,'original'),label=mode)
axes[0].set_yscale('log'); axes[0].set(xlabel='Position in the 96-point window',ylabel='Mean squared error (log)',title='Original frozen checkpoints, 128 normal windows'); axes[0].legend()
values_original=[position(m,'original')[-1] for m in ['E0','E1']]
values_changed=[position(m,'temporary_switch')[-1] for m in ['E0','E1']]
x=np.arange(2); axes[1].bar(x-.18,values_original,width=.36,label='Original padding'); axes[1].bar(x+.18,values_changed,width=.36,label='Temporarily switched padding')
axes[1].set_xticks(x,['E0: circular -> replicate','E1: replicate -> circular']); axes[1].set(ylabel='Last-position MSE',title='Operator sensitivity only: no retraining, possible OOD')
for i,v in enumerate(values_original): axes[1].text(i-.18,v+.001,f'{v:.5f}',ha='center',fontsize=9)
for i,v in enumerate(values_changed): axes[1].text(i+.18,v+.001,f'{v:.5f}',ha='center',fontsize=9)
axes[1].set_ylim(0,max(values_original+values_changed)*1.25); axes[1].legend()
fig.tight_layout(); fig.savefig(OUT/'boundary_position_diagnostic.png',dpi=150); plt.close(fig)

fig,ax=plt.subplots(figsize=(10,5))
categories=['patch_attention','FFN','graph_bias','other_active']
labels=['Patch attention + projections','FFN','Node/position graph bias','Other differentiable parameters']
colors=['#3478ad','#df9855','#b95153','#89979e']; bottom=np.zeros(2)
for category,label,color in zip(categories,labels,colors):
    values=[]
    for mode in ['E0','E1']:
        c=a['parameter_counts'][mode]
        values.append(c['groups'][category] if category!='other_active' else c['structural_differentiable']-sum(c['groups'][k] for k in categories[:-1]))
    ax.bar(['E0','E1'],values,bottom=bottom,label=label,color=color)
    bottom+=values
ax.set(ylabel='Parameter count',title='Parameter allocation on differentiable paths; inactive Q/K and fixed topology excluded')
ax.legend(loc='upper right'); ax.set_ylim(0,bottom.max()*1.25)
for i,v in enumerate(bottom): ax.text(i,v+6000,f'{int(v):,}',ha='center')
fig.tight_layout(); fig.savefig(OUT/'parameter_allocation.png',dpi=150); plt.close(fig)
print('Rendered four audit figures')
