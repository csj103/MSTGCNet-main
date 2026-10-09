from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).parent; ROOT=OUT.parents[1]
a=json.loads((OUT/'analysis.json').read_text())
names={'Engine full power loss':'发动机完全失效','Elevator stuck at zero':'升降舵卡零','Rudder stuck to left':'方向舵卡左','Rudder stuck to right':'方向舵卡右','Both aileron stuck at zero':'双副翼卡零','Left aileron stuck at zero':'左副翼卡零','Right aileron stuck at zero':'右副翼卡零'}
features=list(a['test'][0]['shifts']); train=pd.read_csv(ROOT/'dataset/ALFA10vars/train.csv'); scale=train[features].std(ddof=0)
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']; plt.rcParams['axes.unicode_minus']=False
matrix=np.array([[r['shifts'][f] for f in features] for r in a['test']])
fig,ax=plt.subplots(figsize=(13,5)); im=ax.imshow(matrix,cmap='RdBu_r',vmin=-3,vmax=3,aspect='auto')
ax.set_xticks(range(10),features,rotation=35,ha='right'); ax.set_yticks(range(7),[names[r['type']] for r in a['test']])
for i in range(7):
 for j in range(10): ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',fontsize=9,color='white' if abs(matrix[i,j])>1.8 else 'black')
ax.set_title('测试架次：故障前后中位数变化 / 训练正常标准差\n相邻等点数比较；航向使用原始角度，跨界影响见报告',fontsize=12)
fig.colorbar(im,ax=ax,label='标准化变化（不是显著性检验）'); fig.tight_layout(); fig.savefig(OUT/'feature_shifts.png',dpi=170); plt.close(fig)
lines=['# ALFA 故障数据差异检查（2026-10-08）','',
'## 方法与覆盖','检查 V12 的 46 个有标签原始 CSV（排除 no_ground_truth）、当前 7 个测试架次，以及 baseline/no_graph 已保存评分。46 个文件特征缺失与无穷值均为 0，时间戳均严格递增。',
'故障起点前后使用相同点数 n=min(故障长度, 紧邻故障前正常段长度,600)，n>=30。幅值指标为中位数之差除以训练正常标准差；波动指标为四分位距 IQR 的后/前比。前后片段不一定等时长。相关性使用 Spearman，描述共同变化而非因果。',
'原始文件名用于沿用现有故障类型映射，不是新增人工标签。左副翼类别有一个 rudder_zero__left_aileron 混合文件，跨架次结论应单独核对。',
'nav_yaw 为环绕角度：原始角度相关性、四分位距和中位数可能受跨界影响。nav_x/y/z 的物理含义及 aspd_error 的单位没有在本次分析中从原始采集文档验证，不将它们直接解释成位置或速度。',
'本分析没有重新拟合模型，也未改变划分；不作独立样本显著性检验，因为时间点自相关且每类测试只有一架次。','',
'![变化热图](feature_shifts.png)','', '## 各测试架次']
fig,axes=plt.subplots(7,2,figsize=(16,23))
for i,r in enumerate(a['test']):
 d=pd.read_csv(ROOT/'alfa_10vars'/r['file']); start,n=r['start'],r['n']; lo=max(0,start-n); hi=r['end']; t=d.time_sec.to_numpy()-r['onset_time']
 z=d.iloc[lo:hi]; before=d.iloc[start-n:start]; after=d.iloc[start:start+n]
 yaw=np.rad2deg(np.unwrap(np.deg2rad(d.nav_yaw.to_numpy())))
 rawshift=float(after.nav_yaw.median()-before.nav_yaw.median()); unwrapshift=float(np.median(yaw[start:start+n])-np.median(yaw[start-n:start]))
 no_yaw=[f for f in features if f!='nav_yaw']; tri=np.triu_indices(9,1)
 def cd(x,y):
  delta=(x[no_yaw].corr(method='spearman')-y[no_yaw].corr(method='spearman')).to_numpy()[tri]
  return float(np.nanmean(abs(delta)))
 fault_change=cd(before,after); controls=[]
 for boundary in range(n,start-n+1,max(1,n//4)):
  if d.label_anomaly.iloc[boundary-n:boundary+n].fillna(0).ne(0).any(): continue
  controls.append(cd(d.iloc[boundary-n:boundary],d.iloc[boundary:boundary+n]))
 control=float(np.median(controls)) if controls else None
 r['corr_no_yaw']=fault_change; r['normal_control_median']=control
 left,right=axes[i]
 for f in r['top']:
  left.plot(t[lo:hi],(z[f]-before[f].median())/scale[f],label=f,lw=1)
 left.axvline(0,color='black',ls='--'); left.axvspan(0,t[hi-1],alpha=.08,color='red'); left.legend(fontsize=8); left.set_title(names[r['type']]+'：变化最大的三个变量'); left.set_ylabel('相对故障前中位数 / 训练标准差')
 for directory in (ROOT/'label_results').glob('*v12*'):
  p=directory/'causal_last'; idx=np.load(p/'test_indices.npy'); tm=pd.read_csv(ROOT/'dataset/ALFA10vars/test_meta.csv'); selected=tm.iloc[idx].source_file.eq(r['file']).to_numpy(); tt=tm.iloc[idx].time_sec.to_numpy()[selected]-r['onset_time']; keep=tt>=t[lo]
  name='无图' if 'no_graph' in directory.name else '原版'; color='#D55E00' if name=='无图' else '#0072B2'
  right.plot(tt[keep],np.load(p/'test_energy.npy')[selected][keep],label=name+'分数',color=color,lw=1)
  right.plot(tt[keep],np.load(p/'threshold.npy')[selected][keep],label=name+'阈值',color=color,ls='--',lw=.8)
 right.set_yscale('symlog',linthresh=.001); right.axvline(0,color='black',ls=':'); right.legend(fontsize=8,ncol=2); right.set_title(names[r['type']]+'：实际评分与阈值')
 for ax in (left,right): ax.set_xlabel('距标签故障起点（秒）'); ax.grid(alpha=.15)
 lines += ['', '### '+names[r['type']],f"源文件：`{r['file']}`。起点行（0 基）：{start}；故障 {r['end']-start} 点，首末时间跨度 {r['duration']:.2f} 秒；比较前后各 {n} 点。",'','|变量|中位数变化/训练标准差|IQR 后/前|','|---|---:|---:|']
 for f in features: lines.append(f"|{f}|{r['shifts'][f]:+.3f}|{r['iqr_ratio'][f]:.3f}|")
 lines += ['', f'航向原值中位数变化 {rawshift:+.2f}°；按时间解缠后 {unwrapshift:+.2f}°（解缠假设相邻真实变化小于 180°）。',f'排除航向的平均绝对相关系数变化：{fault_change:.3f}；同架次正常/正常相邻窗口参照中位数：'+(f'{control:.3f}，共 {len(controls)} 个重叠参照。' if control is not None else '无足够长度参照。'),'', '|模型|正常分数中位数|异常分数中位数|异常阈值中位数|原始召回|首次命中延迟/s|','|---|---:|---:|---:|---:|---:|']
 for name,s in r['scores'].items(): lines.append(f"|{name}|{s['normal_score_median']:.5f}|{s['anomaly_score_median']:.5f}|{s['anomaly_threshold_median']:.5f}|{s['recall']:.2%}|{s['first_delay_seconds']:.3f}|")
 lines += ['正常分数统计使用该测试架次全部有效正常点；异常分数统计使用全部有效异常点，不能将两个中位数之比当作分类指标。首次命中只统计标签起点之后，不表示之前没有误报。']
 print(r['type'],'corr',round(fault_change,3),'normal_control',round(control,3) if control else None,'yaw_raw_unwrapped',round(rawshift,2),round(unwrapshift,2))
fig.tight_layout(); fig.savefig(OUT/'flight_comparisons.png',dpi=130); plt.close(fig)
lines += ['', '## 原始文件中的跨架次支持','|类型|可比较架次|中位绝对变化最大的三个变量|','|---|---:|---|']
for g in a['all_types']:
 top=sorted(g['median_abs_shift'],key=g['median_abs_shift'].get,reverse=True)[:3]
 lines.append(f"|{names.get(g['type'],g['type'])}|{g['flights']}|"+', '.join(f"{f}={g['median_abs_shift'][f]:.2f}" for f in top)+'|')
lines += ['', '## 评分曲线', '![各架次变化及评分](flight_comparisons.png)']
(OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
